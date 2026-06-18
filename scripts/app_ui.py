#!/usr/bin/env python3
"""
Streamlit-Bedien-UI für die Füllzustand-Pipeline.

Start:
    streamlit run scripts/app_ui.py
Dann im Browser http://localhost:8501 öffnen (oder den angezeigten Link).

Bedienung:
- Eingabe per Bild-Upload, Kamera-Schnappschuss oder Live-Kamera (Browser-WebRTC).
- Methode je Stufe umschaltbar (Detektion YOLO/Kontur, Stufe 1/2 klassisch/KI/auto).
- Ergebnis: Original mit Overlay, entzerrte Draufsicht, Statistik und Slot-Tabelle.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.inference_config import ORIGINAL, StateInferenceParams  # noqa: E402
from kastendetektion.overlay import draw_pipeline_overlay  # noqa: E402
from kastendetektion.pipeline import CrateAnalysis, CratePipeline  # noqa: E402
from kastendetektion.stabilize import SlotStateStabilizer  # noqa: E402
from kastendetektion.states import label_for  # noqa: E402

_LIVE_PIPELINES: dict[str, CratePipeline] = {}
_LIVE_SETTINGS: dict[str, dict[str, Any]] = {}
_LIVE_RESULTS: dict[str, dict[str, Any]] = {}


def _live_session_id() -> str:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx()
    return ctx.session_id if ctx is not None else "default"


def _decode_upload(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _build_pipeline(
    *,
    contour_only: bool,
    state_method: str,
    slot_method: str,
    cap_method: str,
    conf: float,
    yolo_w: str,
    state_w: str,
    slot_w: str,
    cap_w: str,
    infer_params: StateInferenceParams,
    slot_occ_thr: float,
) -> CratePipeline:
    return CratePipeline(
        state_method=state_method,
        slot_method=slot_method,
        cap_method=cap_method,
        weights_path=(yolo_w.strip() or None) if not contour_only else None,
        prefer_yolo=not contour_only,
        conf=conf,
        state_weights=state_w.strip() or None,
        slot_weights=slot_w.strip() or None,
        cap_weights=cap_w.strip() or None,
        state_inference_params=infer_params,
        slot_occupied_threshold=slot_occ_thr,
    )


def _pipeline_config_key(
    *,
    contour_only: bool,
    state_method: str,
    slot_method: str,
    cap_method: str,
    conf: float,
    yolo_w: str,
    state_w: str,
    slot_w: str,
    cap_w: str,
    infer_params: StateInferenceParams,
    slot_occ_thr: float,
) -> tuple:
    return (
        contour_only,
        state_method,
        slot_method,
        cap_method,
        conf,
        yolo_w.strip(),
        state_w.strip(),
        slot_w.strip(),
        cap_w.strip(),
        infer_params,
        slot_occ_thr,
    )


def _clear_live_session(session_id: str) -> None:
    _LIVE_PIPELINES.pop(session_id, None)
    _LIVE_SETTINGS.pop(session_id, None)
    _LIVE_RESULTS.pop(session_id, None)


def _ensure_pipeline(pipe_key: tuple, **pipe_kwargs) -> CratePipeline:
    if st.session_state.get("pipeline_key") != pipe_key:
        st.session_state.pipeline = _build_pipeline(**pipe_kwargs)
        st.session_state.pipeline_key = pipe_key
    return st.session_state.pipeline


def _mode_label(pipe: CratePipeline) -> str:
    if pipe.state_clf.active_method == "ml":
        return f"3-Klassen: {pipe.state_clf.active_method} + Slot: {pipe.slot_clf.active_method}"
    return f"Stufe 1/2: {pipe.slot_clf.active_method}/{pipe.cap_clf.active_method}"


def _render_analysis(pipe: CratePipeline, frame: np.ndarray, analysis: CrateAnalysis) -> None:
    overlay = draw_pipeline_overlay(frame, analysis)
    st.success(
        f"Detektion: {analysis.detection.source} (conf {analysis.detection.confidence:.2f}) | "
        f"{_mode_label(pipe)}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Belegt", f"{analysis.occupied_count}/{analysis.total}")
    c2.metric("Voll", analysis.full_count)
    c3.metric("Leer", analysis.empty_count)
    c4.metric("Fehlt", analysis.missing_count)

    left, right = st.columns(2)
    left.image(_bgr_to_rgb(overlay), caption="Overlay", width="stretch")
    right.image(_bgr_to_rgb(analysis.warped), caption="Entzerrte Draufsicht", width="stretch")

    with st.expander("Slot-Details"):
        rows = [
            {
                "Slot": s.index,
                "Zeile": s.row,
                "Spalte": s.col,
                "Zustand": label_for(s.state),
                "Slot-Konf.": round(s.slot_conf, 2),
                "Cap-Konf.": round(s.cap_conf, 2),
            }
            for s in analysis.slots
        ]
        st.dataframe(rows, width="stretch")


class CrateVideoProcessor(VideoProcessorBase):
    """WebRTC-Processor: Pipeline pro Frame, Overlay direkt im Videobild."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._lock = threading.Lock()
        self._last_process_ts = 0.0
        self._last_overlay: np.ndarray | None = None
        self._stabilizer: SlotStateStabilizer | None = None
        self._stabilizer_window = 0

    def _settings(self) -> dict[str, Any]:
        return _LIVE_SETTINGS.get(self.session_id, {})

    def _stabilizer_for(self, window: int) -> SlotStateStabilizer | None:
        if window <= 1:
            self._stabilizer = None
            self._stabilizer_window = 0
            return None
        if self._stabilizer is None or self._stabilizer_window != window:
            self._stabilizer = SlotStateStabilizer(window=window)
            self._stabilizer_window = window
        return self._stabilizer

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        settings = self._settings()
        min_interval = float(settings.get("update_ms", 300)) / 1000.0
        now = time.time()

        with self._lock:
            if self._last_overlay is not None and now - self._last_process_ts < min_interval:
                return av.VideoFrame.from_ndarray(self._last_overlay, format="bgr24")
            self._last_process_ts = now

        pipe = _LIVE_PIPELINES.get(self.session_id)
        if pipe is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        analysis = pipe.analyze(img)
        result = _LIVE_RESULTS.setdefault(self.session_id, {})

        if analysis is None:
            result["analysis"] = None
            result["frame_bgr"] = img
            with self._lock:
                self._last_overlay = img
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        stabilizer = self._stabilizer_for(int(settings.get("stabilize_window", 8)))
        if stabilizer is not None:
            stable = stabilizer.update([s.state for s in analysis.slots])
            for slot, state in zip(analysis.slots, stable):
                slot.state = state

        overlay = draw_pipeline_overlay(img, analysis)
        result["analysis"] = analysis
        result["frame_bgr"] = img
        result["overlay"] = overlay
        result["pipe_mode"] = _mode_label(pipe)
        result["ts"] = now

        with self._lock:
            self._last_overlay = overlay

        return av.VideoFrame.from_ndarray(overlay, format="bgr24")


def _render_live_stats(session_id: str, pipe: CratePipeline) -> None:
    result = _LIVE_RESULTS.get(session_id, {})
    analysis: CrateAnalysis | None = result.get("analysis")

    if analysis is None:
        st.info("Auf **START** klicken, Kamerazugriff erlauben und den Kasten ins Bild halten.")
        return

    st.success(
        f"Live | Detektion: {analysis.detection.source} "
        f"(conf {analysis.detection.confidence:.2f}) | "
        f"{result.get('pipe_mode', _mode_label(pipe))}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Belegt", f"{analysis.occupied_count}/{analysis.total}")
    c2.metric("Voll", analysis.full_count)
    c3.metric("Leer", analysis.empty_count)
    c4.metric("Fehlt", analysis.missing_count)

    overlay = result.get("overlay")
    if isinstance(overlay, np.ndarray):
        left, right = st.columns(2)
        left.image(_bgr_to_rgb(overlay), caption="Letztes Overlay", width="stretch")
        right.image(_bgr_to_rgb(analysis.warped), caption="Entzerrte Draufsicht", width="stretch")

    with st.expander("Slot-Details (live)", expanded=False):
        rows = [
            {
                "Slot": s.index,
                "Zeile": s.row,
                "Spalte": s.col,
                "Zustand": label_for(s.state),
                "Slot-Konf.": round(s.slot_conf, 2),
                "Cap-Konf.": round(s.cap_conf, 2),
            }
            for s in analysis.slots
        ]
        st.dataframe(rows, width="stretch")


@st.fragment(run_every=0.5)
def live_stats_fragment(session_id: str, pipe: CratePipeline) -> None:
    if st.session_state.get("live_mode_active"):
        _render_live_stats(session_id, pipe)


def _render_live_camera(
    pipe: CratePipeline,
    *,
    stabilize_window: int,
    update_ms: int,
) -> None:
    session_id = _live_session_id()
    _LIVE_PIPELINES[session_id] = pipe
    _LIVE_SETTINGS[session_id] = {
        "stabilize_window": stabilize_window,
        "update_ms": update_ms,
    }

    st.markdown(
        "1. Auf **START** klicken und Kamerazugriff erlauben  \n"
        "2. Kasten ins Bild halten — **nada** (rot), **leer** (gelb), **voll** (grün) erscheinen live im Video"
    )

    webrtc_streamer(
        key="crate-live-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: CrateVideoProcessor(session_id),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    live_stats_fragment(session_id, pipe)


def main() -> None:
    st.set_page_config(page_title="Bierkasten-Füllzustand", layout="wide")
    st.title("Bierkasten-Füllzustand – Bedien-UI")
    st.caption("Detektion (YOLO/Kontur) + 3-Klassen-KI (nada/leer/voll) → Overlay & Statistik")

    with st.sidebar:
        st.header("Methoden")
        contour_only = st.checkbox("Nur Kontur statt YOLO (keine Gewichte)", value=False)
        state_method = st.selectbox(
            "Zustand (nada/leer/voll)",
            ["auto", "classical", "ml"],
            index=0,
            help="auto nutzt models/state_cnn.pt (alle 3 Klassen trainiert)",
        )
        slot_method = st.selectbox("Stufe 1 – Slot (Fallback)", ["auto", "classical", "ml"], index=0)
        cap_method = st.selectbox("Stufe 2 – Kronkorken (Fallback)", ["auto", "classical", "ml"], index=0)
        conf = st.slider("YOLO-Konfidenz", 0.05, 0.95, 0.25, 0.05)

        st.header("Inferenz-Tuning (nada/voll)")
        use_original = st.checkbox("Original-Schwellwerte (vor Tuning)", value=False)
        slot_occ_thr = st.slider("Slot belegt ab", 0.40, 0.75, 0.58, 0.01)
        full_min = st.slider("Voll min. Konfidenz", 0.40, 0.80, 0.58, 0.01)
        empty_nada_thr = st.slider("Nada wenn p(empty) ≥", 0.20, 0.60, 0.32, 0.01)
        st.caption("Original: Slot 0.50, Voll 0.50, p(empty) 0.45, Fallback p_bf≥p_be→voll")

        infer_params = (
            ORIGINAL
            if use_original
            else StateInferenceParams(
                p_empty_nada=empty_nada_thr,
                full_min=full_min,
                empty_min=0.45,
                empty_vs_full_margin=0.12,
                p_empty_strong=0.38,
                slot_low_conf=0.68,
                p_empty_soft=0.15,
                default_missing=True,
            )
        )

        st.header("Gewichte (optional)")
        yolo_w = st.text_input("YOLO best.pt", os.environ.get("KASTEN_YOLO_WEIGHTS", ""))
        state_w = st.text_input("State-CNN .pt", os.environ.get("KASTEN_STATE_WEIGHTS", ""))
        slot_w = st.text_input("Slot-CNN .pt", os.environ.get("KASTEN_SLOT_WEIGHTS", ""))
        cap_w = st.text_input("Cap-CNN .pt", os.environ.get("KASTEN_CAP_WEIGHTS", ""))

        st.header("Eingabe")
        source = st.radio("Quelle", ["Bild-Upload", "Kamera-Schnappschuss", "Live-Kamera"], index=0)
        stabilize_window = 8
        update_ms = 300
        if source == "Live-Kamera":
            stabilize_window = st.slider("Stabilisierung (Frames)", 1, 20, 8, 1)
            update_ms = st.slider("Aktualisierung (ms)", 200, 2000, 400, 50)

    pipe_kwargs = dict(
        contour_only=contour_only,
        state_method=state_method,
        slot_method=slot_method,
        cap_method=cap_method,
        conf=conf,
        yolo_w=yolo_w,
        state_w=state_w,
        slot_w=slot_w,
        cap_w=cap_w,
        infer_params=infer_params,
        slot_occ_thr=slot_occ_thr,
    )
    pipe_key = _pipeline_config_key(**pipe_kwargs)
    session_id = _live_session_id()

    if source == "Live-Kamera":
        st.session_state.live_mode_active = True
    else:
        st.session_state.live_mode_active = False
        _clear_live_session(session_id)

    if source == "Live-Kamera":
        st.subheader("Live-Kamera")
        pipe = _ensure_pipeline(pipe_key, **pipe_kwargs)
        _render_live_camera(pipe, stabilize_window=stabilize_window, update_ms=update_ms)
        return

    frame: np.ndarray | None = None
    if source == "Bild-Upload":
        up = st.file_uploader("Bild wählen", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            frame = _decode_upload(up.getvalue())
    else:
        shot = st.camera_input("Kamera-Schnappschuss")
        if shot is not None:
            frame = _decode_upload(shot.getvalue())

    if frame is None:
        st.info("Bitte ein Bild hochladen oder einen Kamera-Schnappschuss aufnehmen.")
        return

    pipe = _ensure_pipeline(pipe_key, **pipe_kwargs)
    analysis = pipe.analyze(frame)

    if analysis is None:
        st.error("Kein Kasten erkannt. Tipp: anderes Bild, näher heran, oder 'Nur Kontur' testen.")
        st.image(_bgr_to_rgb(frame), caption="Eingabe", width="stretch")
        return

    _render_analysis(pipe, frame, analysis)


if __name__ == "__main__":
    main()
