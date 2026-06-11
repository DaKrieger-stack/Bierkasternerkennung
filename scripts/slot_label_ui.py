#!/usr/bin/env python3
"""
Slot-Labels auf Kastenbildern (wie label_ui.py, aber 20 Slots × voll/leer/nada).

Start im Projektroot:
    streamlit run scripts/slot_label_ui.py

NICHT verwechseln mit label_ui.py (YOLO-Kasten-Box „bierkasten“).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG_DIR = ROOT / "data" / "kasten_dataset" / "images"
DEFAULT_DATASET = ROOT / "data" / "slot_dataset"
DEFAULT_IGNORED = ROOT / "data" / "kasten_dataset" / "_ignored_images"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_DISPLAY = 420
N_SLOTS = 20
COLS = 5
ROWS = 4

LABEL_KEYS = ("voll", "leer", "nada")
KEY_TO_FOLDER = {
    "voll": "bottle_full",
    "leer": "bottle_empty",
    "nada": "empty",
}
FOLDER_TO_KEY = {v: k for k, v in KEY_TO_FOLDER.items()}
CLASS_FOLDERS = set(KEY_TO_FOLDER.values()) | {"ignored"}


@dataclass(frozen=True)
class LabelDef:
    key: str
    folder: str
    color_rgb: tuple[int, int, int]
    description: str


LABELS: tuple[LabelDef, ...] = (
    LabelDef("voll", "bottle_full", (0, 200, 0), "Flasche mit Kronkorken"),
    LabelDef("leer", "bottle_empty", (255, 215, 0), "Flasche ohne Kronkorken"),
    LabelDef("nada", "empty", (255, 0, 0), "Keine Flasche im Slot"),
)
KEY_INDEX = {lb.key: i for i, lb in enumerate(LABELS)}
HOTKEY_FOR_LABEL = {"voll": "1", "leer": "2", "nada": "3"}
KEY_TO_LABEL = {v: k for k, v in HOTKEY_FOR_LABEL.items()}

_SLOT_GRID = components.declare_component(
    "slot_label_grid",
    path=Path(__file__).resolve().parent / "slot_label_grid",
)


def ui_grid_to_slot(row: int, col: int) -> int:
    """Raster rechts horizontal gespiegelt — entspricht der Vorschau."""
    return row * COLS + (COLS - 1 - col)


def mirrored_grid_layout() -> list[dict]:
    return [{"index": ui_grid_to_slot(row, col)} for row in range(ROWS) for col in range(COLS)]


@st.cache_resource(show_spinner=False)
def get_pipeline():
    from kastendetektion.pipeline import CratePipeline

    return CratePipeline(slot_method="classical", cap_method="classical")


def format_label_option(key: str) -> str:
    for lb in LABELS:
        if lb.key == key:
            hk = HOTKEY_FOR_LABEL[key]
            return f"{hk} · {lb.key} — {lb.description}"
    return key


def chdir_root() -> None:
    os.chdir(ROOT)


def folder_signature(folder: Path) -> tuple[int, float]:
    if not folder.is_dir():
        return 0, 0.0
    count = 0
    newest = 0.0
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            count += 1
            newest = max(newest, p.stat().st_mtime)
    return count, newest


@st.cache_data(show_spinner=False)
def list_images_cached(folder_str: str, sig_count: int, sig_mtime: float) -> tuple[str, ...]:
    folder = Path(folder_str)
    if not folder.is_dir():
        return ()
    return tuple(
        str(p.resolve())
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )


def slot_roi_name(stem: str, slot: int) -> str:
    return f"{stem}_slot{slot:02d}.png"


def find_existing_label(stem: str, slot: int, dataset: Path) -> str | None:
    name = slot_roi_name(stem, slot)
    for folder, key in FOLDER_TO_KEY.items():
        if (dataset / folder / name).is_file():
            return key
    return None


def load_labels_for_image(stem: str, dataset: Path, pipeline_guess: list[str] | None) -> list[str]:
    labels: list[str] = []
    for i in range(N_SLOTS):
        existing = find_existing_label(stem, i, dataset)
        if existing:
            labels.append(existing)
        elif pipeline_guess and i < len(pipeline_guess):
            labels.append(pipeline_guess[i])
        else:
            labels.append("nada")
    return labels


def pipeline_guess_labels(analysis) -> list[str]:
    from kastendetektion.states import SlotState

    mapping = {
        SlotState.FULL: "voll",
        SlotState.EMPTY: "leer",
        SlotState.MISSING: "nada",
    }
    return [mapping[s.state] for s in sorted(analysis.slots, key=lambda x: x.index)]


def make_preview(frame_bgr: np.ndarray, slot_labels: list[str], analysis) -> np.ndarray:
    from kastendetektion.states import SlotState, color_for

    state_map = {
        "voll": SlotState.FULL,
        "leer": SlotState.EMPTY,
        "nada": SlotState.MISSING,
    }
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    marker_r = max(16, min(32, int(min(h, w) * 0.028)))
    font_scale = max(0.7, min(1.4, marker_r / 22.0))
    thickness = max(2, int(font_scale * 2))

    if analysis.detection is not None:
        pts = analysis.detection.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

    for slot, key in zip(analysis.slots, slot_labels):
        slot.state = state_map.get(key, SlotState.MISSING)
        cx, cy = int(round(slot.center_orig[0])), int(round(slot.center_orig[1]))
        color = color_for(slot.state)
        label = str(slot.index)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx, ty = cx - tw // 2, cy + th // 2

        cv2.circle(out, (cx, cy), marker_r + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), marker_r, color, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), marker_r, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            out, label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return out


def resize_for_display(img_rgb: np.ndarray, max_side: int = MAX_DISPLAY) -> Image.Image:
    h, w = img_rgb.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return Image.fromarray(img_rgb).resize((nw, nh), Image.Resampling.LANCZOS)


def count_saved_slots(stem: str, dataset: Path) -> int:
    return sum(1 for i in range(N_SLOTS) if find_existing_label(stem, i, dataset) is not None)


def labels_on_disk(stem: str, dataset: Path) -> list[str] | None:
    labels: list[str] = []
    for i in range(N_SLOTS):
        existing = find_existing_label(stem, i, dataset)
        if existing is None:
            return None
        labels.append(existing)
    return labels


def load_image_bundle(img_path: Path) -> dict | None:
    """YOLO + Entzerrung einmal pro Bild (Session-Cache)."""
    mtime_ns = img_path.stat().st_mtime_ns
    cache_key = f"img_bundle::{img_path.resolve()}::{mtime_ns}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    frame_bgr = cv2.imread(str(img_path))
    if frame_bgr is None:
        return None

    with st.spinner("Kasten wird erkannt …"):
        analysis = get_pipeline().analyze(frame_bgr)
    if analysis is None:
        return None

    bundle = {"frame_bgr": frame_bgr, "analysis": analysis}
    st.session_state[cache_key] = bundle
    return bundle


def clear_slot_widget_state(stem: str, idx: int) -> None:
    for slot_i in range(N_SLOTS):
        st.session_state.pop(f"slot_cls_{stem}_{idx}_{slot_i}", None)


def save_single_slot_roi(
    analysis,
    stem: str,
    slot_i: int,
    label_key: str,
    dataset: Path,
) -> None:
    from kastendetektion.warp_grid import estimate_slot_half_size, extract_slot_roi

    for folder in CLASS_FOLDERS:
        p = dataset / folder / slot_roi_name(stem, slot_i)
        if p.is_file():
            p.unlink()

    slot = next(s for s in analysis.slots if s.index == slot_i)
    centers = np.array([s.center_warped for s in analysis.slots], dtype=np.float32)
    half = estimate_slot_half_size(centers, analysis.rows, analysis.cols)
    roi = extract_slot_roi(
        analysis.warped,
        float(slot.center_warped[0]),
        float(slot.center_warped[1]),
        half_size=half,
    )
    if roi is None or roi.size == 0:
        return
    roi = cv2.resize(roi, (64, 64))
    folder = KEY_TO_FOLDER[label_key]
    out_dir = dataset / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / slot_roi_name(stem, slot_i)), roi)


def interactive_slot_grid(labels: list[str], key: str) -> list[str] | None:
    result = _SLOT_GRID(
        labels=labels,
        layout=mirrored_grid_layout(),
        label_colors={lb.key: list(lb.color_rgb) for lb in LABELS},
        key=key,
        default=None,
    )
    if result is None:
        return None
    return list(result)


def apply_label_changes(
    analysis,
    stem: str,
    idx: int,
    buf_key: str,
    labels: list[str],
    new_labels: list[str],
    dataset: Path,
) -> None:
    for slot_i, (old, new) in enumerate(zip(labels, new_labels)):
        if old != new:
            save_single_slot_roi(analysis, stem, slot_i, new, dataset)
    st.session_state[buf_key] = list(new_labels)
    clear_slot_widget_state(stem, idx)


def apply_label_to_slot(
    buf_key: str,
    stem: str,
    idx: int,
    labels: list[str],
    slot_i: int,
    label_key: str,
) -> list[str]:
    updated = list(labels)
    updated[slot_i] = label_key
    st.session_state[buf_key] = updated
    clear_slot_widget_state(stem, idx)
    return updated


def save_slot_rois_from_analysis(
    analysis,
    stem: str,
    slot_labels: list[str],
    dataset: Path,
) -> tuple[int, str | None]:
    from kastendetektion.warp_grid import estimate_slot_half_size, extract_slot_roi

    for folder in CLASS_FOLDERS:
        for i in range(N_SLOTS):
            p = dataset / folder / slot_roi_name(stem, i)
            if p.is_file():
                p.unlink()

    centers = np.array([s.center_warped for s in analysis.slots], dtype=np.float32)
    half = estimate_slot_half_size(centers, analysis.rows, analysis.cols)
    saved = 0
    for slot, key in zip(analysis.slots, slot_labels):
        roi = extract_slot_roi(
            analysis.warped,
            float(slot.center_warped[0]),
            float(slot.center_warped[1]),
            half_size=half,
        )
        if roi is None or roi.size == 0:
            continue
        roi = cv2.resize(roi, (64, 64))
        folder = KEY_TO_FOLDER[key]
        out_dir = dataset / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / slot_roi_name(stem, slot.index)), roi)
        saved += 1
    return saved, None


def delete_image_and_slots(img_path: Path, dataset: Path) -> None:
    stem = img_path.stem
    for folder in CLASS_FOLDERS:
        for i in range(N_SLOTS):
            p = dataset / folder / slot_roi_name(stem, i)
            if p.is_file():
                p.unlink()
    if img_path.is_file():
        img_path.unlink()


@st.fragment
def slot_dropdown_grid(stem: str, idx: int, labels: list[str], buf_key: str) -> list[str]:
    updated: list[str] = list(labels)
    for row in range(ROWS):
        cols = st.columns(COLS)
        for col in range(COLS):
            slot_i = ui_grid_to_slot(row, col)
            with cols[col]:
                choice = st.selectbox(
                    f"**{slot_i}**",
                    options=list(LABEL_KEYS),
                    index=KEY_INDEX.get(updated[slot_i], 2),
                    format_func=format_label_option,
                    key=f"slot_cls_{stem}_{idx}_{slot_i}",
                    label_visibility="visible",
                )
                updated[slot_i] = choice
    st.session_state[buf_key] = updated
    return updated


def main() -> None:
    chdir_root()
    st.set_page_config(page_title="Slot-Labels (voll/leer/nada)", layout="wide")
    st.title("Slot-Labels auf Kastenbildern")
    st.caption(
        "Links Vorschau · rechts Slot-Raster: **Maus über Feld → 1 / 2 / 3** (speichert sofort)."
    )

    st.sidebar.subheader("Klassen")
    for lb in LABELS:
        r, g, b = lb.color_rgb
        st.sidebar.markdown(
            f"<span style='color:rgb({r},{g},{b});font-weight:bold'>● {lb.key}</span> — {lb.description}",
            unsafe_allow_html=True,
        )

    img_dir_str = st.sidebar.text_input(
        "Bildordner",
        value=str(DEFAULT_IMG_DIR.relative_to(ROOT)),
    )
    dataset_str = st.sidebar.text_input(
        "Slot-Dataset (ROIs)",
        value=str(DEFAULT_DATASET.relative_to(ROOT)),
    )
    ignored_str = st.sidebar.text_input(
        "Ignorierte Bilder",
        value=str(DEFAULT_IGNORED.relative_to(ROOT)),
    )

    img_dir = Path(img_dir_str)
    dataset = Path(dataset_str)
    ignored_dir = Path(ignored_str)
    if not img_dir.is_absolute():
        img_dir = (ROOT / img_dir).resolve()
    if not dataset.is_absolute():
        dataset = (ROOT / dataset).resolve()
    if not ignored_dir.is_absolute():
        ignored_dir = (ROOT / ignored_dir).resolve()

    for folder in CLASS_FOLDERS:
        (dataset / folder).mkdir(parents=True, exist_ok=True)
    ignored_dir.mkdir(parents=True, exist_ok=True)

    sig_count, sig_mtime = folder_signature(img_dir)
    images = [Path(p) for p in list_images_cached(str(img_dir), sig_count, sig_mtime)]
    if not images:
        st.warning(f"Keine Bilder unter `{img_dir}`. Fotos dort ablegen (wie bei der Kasten-Label-UI).")
        st.stop()

    if "img_index" not in st.session_state:
        st.session_state["img_index"] = 0

    idx = max(0, min(int(st.session_state["img_index"]), len(images) - 1))
    st.session_state["img_index"] = idx
    img_path = images[idx]
    stem = img_path.stem

    bundle = load_image_bundle(img_path)
    if bundle is None:
        st.error("Kein Kasten erkannt. Nächstes Bild wählen oder anderes Foto verwenden.")
        col_nav1, col_nav2, _, _ = st.columns([1, 1, 2, 4])
        with col_nav1:
            if st.button("◀ Zurück", disabled=idx <= 0):
                st.session_state["img_index"] = idx - 1
                st.rerun()
        with col_nav2:
            if st.button("Weiter ▶", disabled=idx >= len(images) - 1):
                st.session_state["img_index"] = idx + 1
                st.rerun()
        st.stop()

    frame_bgr = bundle["frame_bgr"]
    analysis = bundle["analysis"]

    buf_key = f"slot_labels::{img_path.resolve()}"
    if buf_key not in st.session_state:
        guess = pipeline_guess_labels(analysis)
        st.session_state[buf_key] = load_labels_for_image(stem, dataset, guess)

    labels: list[str] = list(st.session_state[buf_key])
    while len(labels) < N_SLOTS:
        labels.append("nada")
    labels = labels[:N_SLOTS]

    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 2, 4])
    with col_nav1:
        if st.button("◀ Zurück", disabled=idx <= 0):
            st.session_state["img_index"] = idx - 1
            st.rerun()
    with col_nav2:
        if st.button("Weiter ▶", disabled=idx >= len(images) - 1):
            st.session_state["img_index"] = idx + 1
            st.rerun()
    with col_nav3:
        st.markdown(f"**{idx + 1}** / **{len(images)}**")
    with col_nav4:
        if st.button("Vorschläge neu laden"):
            st.session_state.pop(buf_key, None)
            st.rerun()

    st.markdown(f"`{img_path.name}`")
    saved_n = count_saved_slots(stem, dataset)
    disk_labels = labels_on_disk(stem, dataset)
    if saved_n == N_SLOTS:
        if disk_labels == labels:
            st.success(f"**{saved_n}/{N_SLOTS}** Slot-ROIs gespeichert.")
        else:
            st.warning("Änderungen noch nicht gespeichert — **Speichern** klicken.")
    elif saved_n > 0:
        st.info(f"**{saved_n}/{N_SLOTS}** Slots gespeichert — noch unvollständig.")
    else:
        st.caption("Noch nicht gespeichert.")

    col_preview, col_slots = st.columns([1, 1])

    with col_preview:
        st.markdown("##### Vorschau")
        preview = make_preview(frame_bgr, labels, analysis)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        disp = resize_for_display(preview_rgb)
        st.image(disp, caption="Farbe = Klasse · Zahl = Slot-Nr.", width=MAX_DISPLAY)

    with col_slots:
        st.markdown("##### Slots labeln")
        grid_key = f"slot_grid_{stem}_{idx}"
        new_from_grid = interactive_slot_grid(labels, key=grid_key)
        if new_from_grid is not None and new_from_grid != labels:
            apply_label_changes(analysis, stem, idx, buf_key, labels, new_from_grid, dataset)
            st.rerun()
        with st.expander("Dropdowns (manuell)"):
            slot_dropdown_grid(stem, idx, labels, buf_key)

    btn_col1, btn_col2, _, _ = st.columns([1, 1, 2, 2])
    with btn_col1:
        save_clicked = st.button("Slot-Labels speichern", type="primary")
    with btn_col2:
        save_and_next = st.button("Speichern + Weiter")

    if save_clicked or save_and_next:
        final_labels = list(st.session_state.get(buf_key, labels))
        n, err = save_slot_rois_from_analysis(analysis, stem, final_labels, dataset)
        if err:
            st.error(err)
        else:
            st.toast(f"Gespeichert: {n} Slot-ROIs für {img_path.name}", icon="✅")
            if save_and_next and idx < len(images) - 1:
                st.session_state["img_index"] = idx + 1
            st.rerun()

    st.divider()
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("Bild ignorieren", help="Bild verschieben, keine ROIs speichern"):
            delete_image_and_slots(img_path, dataset)
            dest = ignored_dir / img_path.name
            if img_path.is_file():
                shutil.move(str(img_path), str(dest))
            list_images_cached.clear()
            st.toast("Ignoriert", icon="⏭")
            st.rerun()
    with c2:
        if st.button("Bild löschen", help="Quellbild + alle Slot-ROIs löschen"):
            delete_image_and_slots(img_path, dataset)
            if idx >= len(images) - 1:
                st.session_state["img_index"] = max(0, idx - 1)
            list_images_cached.clear()
            st.toast("Gelöscht", icon="🗑")
            st.rerun()

    with st.expander("Workflow"):
        st.markdown(
            """
            1. Kastenfotos in `data/kasten_dataset/images/`.  
            2. Rechts: Feld fahren → **1/2/3** (voll/leer/nada), speichert ROI sofort.  
            3. **Speichern** — ROIs unter `data/slot_dataset/`.  
            4. Danach: `python scripts/train_slot_cnn.py` und `python scripts/train_cap_cnn.py`
            """
        )


if __name__ == "__main__":
    main()
