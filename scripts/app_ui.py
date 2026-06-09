#!/usr/bin/env python3
"""
Streamlit-Bedien-UI für die Füllzustand-Pipeline.

Start:
    streamlit run scripts/app_ui.py
Dann im Browser http://localhost:8501 öffnen (oder den angezeigten Link).

Bedienung:
- Eingabe per Bild-Upload oder Kamera-Schnappschuss.
- Methode je Stufe umschaltbar (Detektion YOLO/Kontur, Stufe 1/2 klassisch/KI/auto).
- Ergebnis: Original mit Overlay, entzerrte Draufsicht, Statistik und Slot-Tabelle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.overlay import draw_pipeline_overlay  # noqa: E402
from kastendetektion.pipeline import CratePipeline  # noqa: E402
from kastendetektion.states import label_for  # noqa: E402


def _decode_upload(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(page_title="Bierkasten-Füllzustand", layout="wide")
    st.title("Bierkasten-Füllzustand – Bedien-UI")
    st.caption("Detektion (YOLO/Kontur) + Stufe 1 (Slot) + Stufe 2 (Kronkorken) → Overlay & Statistik")

    with st.sidebar:
        st.header("Methoden")
        contour_only = st.checkbox("Nur Kontur statt YOLO (keine Gewichte)", value=False)
        slot_method = st.selectbox("Stufe 1 – Slot", ["auto", "classical", "ml"], index=1)
        cap_method = st.selectbox("Stufe 2 – Kronkorken", ["auto", "classical", "ml"], index=1)
        conf = st.slider("YOLO-Konfidenz", 0.05, 0.95, 0.25, 0.05)

        st.header("Gewichte (optional)")
        yolo_w = st.text_input("YOLO best.pt", os.environ.get("KASTEN_YOLO_WEIGHTS", ""))
        slot_w = st.text_input("Slot-CNN .pt", os.environ.get("KASTEN_SLOT_WEIGHTS", ""))
        cap_w = st.text_input("Cap-CNN .pt", os.environ.get("KASTEN_CAP_WEIGHTS", ""))

        st.header("Eingabe")
        source = st.radio("Quelle", ["Bild-Upload", "Kamera"], index=0)

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

    pipe = CratePipeline(
        slot_method=slot_method,
        cap_method=cap_method,
        weights_path=(yolo_w.strip() or None) if not contour_only else None,
        prefer_yolo=not contour_only,
        conf=conf,
        slot_weights=slot_w.strip() or None,
        cap_weights=cap_w.strip() or None,
    )
    analysis = pipe.analyze(frame)

    if analysis is None:
        st.error("Kein Kasten erkannt. Tipp: anderes Bild, näher heran, oder 'Nur Kontur' testen.")
        st.image(_bgr_to_rgb(frame), caption="Eingabe", use_container_width=True)
        return

    overlay = draw_pipeline_overlay(frame, analysis)

    st.success(
        f"Detektion: {analysis.detection.source} (conf {analysis.detection.confidence:.2f}) | "
        f"Stufe 1/2: {pipe.slot_clf.active_method}/{pipe.cap_clf.active_method}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Belegt", f"{analysis.occupied_count}/{analysis.total}")
    c2.metric("Voll", analysis.full_count)
    c3.metric("Leer", analysis.empty_count)
    c4.metric("Fehlt", analysis.missing_count)

    left, right = st.columns(2)
    left.image(_bgr_to_rgb(overlay), caption="Overlay", use_container_width=True)
    right.image(_bgr_to_rgb(analysis.warped), caption="Entzerrte Draufsicht", use_container_width=True)

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
        st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
