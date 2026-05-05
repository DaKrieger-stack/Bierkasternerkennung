#!/usr/bin/env python3
"""
Lokale Labeling-UI (Streamlit): Bounding Boxes zeichnen, Klassen pro Box per Dropdown.

Start im Projektroot:
    streamlit run scripts/label_ui.py

Bilder liegen z. B. unter data/kasten_dataset/images/ (nach Dropbox-Download).
Ausgabe: YOLO-Zeilen in data/kasten_dataset/labels/<gleicher_stem>.txt
Klassenliste: labeling/classes.txt (eine Klasse pro Zeile; Reihenfolge = Klassen-ID 0,1,2,…).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from PIL import Image

# streamlit-drawable-canvas nutzt noch image_to_url(image, width, …); aktuelle Streamlit-API:
# image_to_url(image, layout_config=LayoutConfig(width=…), …).
import streamlit.elements.image as _streamlit_image_element
from streamlit.elements.lib import image_utils
from streamlit.elements.lib.layout_utils import LayoutConfig


def _patch_streamlit_image_to_url_for_drawable_canvas() -> None:
    def image_to_url_legacy(image, width, clamp, channels, output_format, image_id):
        layout = LayoutConfig(width=int(width))
        return image_utils.image_to_url(
            image,
            layout,
            clamp,
            channels,
            output_format,
            image_id,
        )

    _streamlit_image_element.image_to_url = image_to_url_legacy  # type: ignore[attr-defined]


_patch_streamlit_image_to_url_for_drawable_canvas()

from streamlit_drawable_canvas import st_canvas

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = SCRIPT_ROOT / "labeling" / "classes.txt"
DEFAULT_IMG_DIR = SCRIPT_ROOT / "data" / "kasten_dataset" / "images"
DEFAULT_LBL_DIR = SCRIPT_ROOT / "data" / "kasten_dataset" / "labels"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_DISPLAY = 960


def chdir_root() -> None:
    os.chdir(SCRIPT_ROOT)


def load_classes(path: Path) -> list[str]:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bierkasten\n", encoding="utf-8")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines or ["bierkasten"]


def save_classes(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def read_yolo_labels(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not txt_path.is_file():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        rows.append((cls, cx, cy, w, h))
    return rows


def yolo_to_canvas_objects(
    rows: list[tuple[int, float, float, float, float]],
    w_img: int,
    h_img: int,
    scale: float,
) -> list[dict]:
    objs: list[dict] = []
    for _cls, cx, cy, bw, bh in rows:
        x_ctr, y_ctr = cx * w_img, cy * h_img
        w_pix, h_pix = bw * w_img, bh * h_img
        left = x_ctr - w_pix / 2
        top = y_ctr - h_pix / 2
        objs.append(
            {
                "type": "rect",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": float(left * scale),
                "top": float(top * scale),
                "width": float(w_pix * scale),
                "height": float(h_pix * scale),
                "fill": "rgba(255, 165, 0, 0.25)",
                "stroke": "#00AA00",
                "strokeWidth": 2,
            }
        )
    return objs


def canvas_objects_to_yolo(
    objects: list[dict],
    classes_per_box: list[int],
    w_img: int,
    h_img: int,
    scale: float,
) -> list[str]:
    lines: list[str] = []
    inv = 1.0 / scale if scale > 0 else 1.0
    for obj, cls_idx in zip(objects, classes_per_box):
        if obj.get("type") != "rect":
            continue
        left = float(obj["left"]) * inv
        top = float(obj["top"]) * inv
        width = float(obj["width"]) * inv
        height = float(obj["height"]) * inv
        if width <= 1 or height <= 1:
            continue
        cx = (left + width / 2.0) / w_img
        cy = (top + height / 2.0) / h_img
        nw = width / w_img
        nh = height / h_img
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 1e-6), 1.0)
        nh = min(max(nh, 1e-6), 1.0)
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def main() -> None:
    chdir_root()
    st.set_page_config(page_title="Kasten labeln", layout="wide")
    st.title("Bounding Boxes labeln")
    st.caption(
        "Rechtecke auf der Bildfläche ziehen. Pro Box die **Klasse** wählen, dann **YOLO-Labels speichern**. "
        "Kurs-Bilder zuerst nach `data/kasten_dataset/images/` legen (z. B. aus Dropbox)."
    )

    classes_path_str = st.sidebar.text_input(
        "Klassen-Datei",
        value=str(DEFAULT_CLASSES.relative_to(SCRIPT_ROOT)),
    )
    classes_path = Path(classes_path_str)
    if not classes_path.is_absolute():
        classes_path = (SCRIPT_ROOT / classes_path).resolve()

    if "classes_editor" not in st.session_state:
        st.session_state["classes_editor"] = "\n".join(load_classes(classes_path))

    st.sidebar.subheader("Klassen für Dropdowns")
    new_body = st.sidebar.text_area(
        "Eine Klasse pro Zeile (Zeile 1 = ID 0, Zeile 2 = ID 1, …)",
        value=st.session_state["classes_editor"],
        height=180,
    )
    if st.sidebar.button("Klassen speichern"):
        raw_lines = [ln.strip() for ln in new_body.splitlines() if ln.strip()]
        if len(raw_lines) < 1:
            st.sidebar.error("Mindestens eine Klasse.")
        else:
            save_classes(classes_path, raw_lines)
            st.session_state["classes_editor"] = new_body
            st.sidebar.success(f"Gespeichert: {classes_path}")
            st.rerun()

    classes = load_classes(classes_path)

    img_dir_str = st.sidebar.text_input(
        "Bildordner",
        value=str(DEFAULT_IMG_DIR.relative_to(SCRIPT_ROOT)),
    )
    lbl_dir_str = st.sidebar.text_input(
        "Labelordner",
        value=str(DEFAULT_LBL_DIR.relative_to(SCRIPT_ROOT)),
    )
    img_dir = Path(img_dir_str)
    lbl_dir = Path(lbl_dir_str)
    if not img_dir.is_absolute():
        img_dir = (SCRIPT_ROOT / img_dir).resolve()
    if not lbl_dir.is_absolute():
        lbl_dir = (SCRIPT_ROOT / lbl_dir).resolve()
    lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(img_dir)
    if not images:
        st.warning(
            f"Keine Bilder unter `{img_dir}`. "
            "Bitte Fotos dort ablegen (z. B. aus dem Kurs-Dropbox) oder den Bildordner anpassen."
        )
        st.stop()

    if "img_index" not in st.session_state:
        st.session_state["img_index"] = 0

    idx = int(st.session_state["img_index"])
    idx = max(0, min(idx, len(images) - 1))
    st.session_state["img_index"] = idx

    img_path = images[idx]
    lbl_path = lbl_dir / f"{img_path.stem}.txt"

    img_key = str(img_path.resolve())
    buf_canvas = f"canvas_json::{img_key}"
    buf_cls = f"box_cls::{img_key}"
    key_slug = hashlib.sha256(img_key.encode("utf-8")).hexdigest()[:16]

    image = Image.open(img_path).convert("RGB")
    w_img, h_img = image.size
    scale = min(1.0, MAX_DISPLAY / max(w_img, h_img))
    disp_w = max(1, int(round(w_img * scale)))
    disp_h = max(1, int(round(h_img * scale)))
    disp_image = image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

    rows = read_yolo_labels(lbl_path)
    initial_objects = yolo_to_canvas_objects(rows, w_img, h_img, scale)
    file_canvas = {"version": "4.4.0", "objects": initial_objects}

    if buf_canvas not in st.session_state:
        st.session_state[buf_canvas] = json.dumps(file_canvas)

    # Nach externem Dateizugriff: „Zurücksetzen“ möglich
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
        st.markdown(f"**{idx + 1}** / **{len(images)}** — `{img_path.name}`")
    with col_nav4:
        if st.button("Dieses Bild aus Datei neu laden"):
            st.session_state.pop(buf_canvas, None)
            st.session_state.pop(buf_cls, None)
            st.rerun()

    try:
        initial_drawing = json.loads(st.session_state[buf_canvas])
    except json.JSONDecodeError:
        st.session_state[buf_canvas] = json.dumps(file_canvas)
        initial_drawing = json.loads(st.session_state[buf_canvas])

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.25)",
        stroke_width=2,
        stroke_color="#00AA00",
        background_image=disp_image,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key=f"canvas_{img_path.stem}_{idx}",
        initial_drawing=initial_drawing,
    )

    if canvas_result.json_data:
        st.session_state[buf_canvas] = canvas_result.json_data

    try:
        data = json.loads(st.session_state[buf_canvas])
    except json.JSONDecodeError:
        st.error("Interner Canvas-JSON-Fehler — „Aus Datei neu laden“ nutzen.")
        st.stop()

    objects = [o for o in data.get("objects", []) if o.get("type") == "rect"]

    if buf_cls not in st.session_state:
        st.session_state[buf_cls] = [rows[i][0] if i < len(rows) else 0 for i in range(len(objects))]
    lst: list[int] = st.session_state[buf_cls]
    while len(lst) < len(objects):
        lst.append(0)
    while len(lst) > len(objects):
        lst.pop()

    st.subheader("Klasse je Bounding Box")
    if not objects:
        st.info("Noch keine Box — mit dem **Rect**-Modus Rechtecke auf dem Bild ziehen.")

    updated: list[int] = []
    n_cols = min(4, max(1, len(objects)))
    cols = st.columns(n_cols) if objects else []
    for i, _obj in enumerate(objects):
        safe_idx = min(max(lst[i], 0), len(classes) - 1)
        with cols[i % len(cols)]:
            choice = st.selectbox(
                f"Box {i + 1}",
                options=list(range(len(classes))),
                format_func=lambda j: classes[j],
                index=int(safe_idx),
                key=f"cls_sb_{key_slug}_{i}",
            )
            updated.append(int(choice))
    st.session_state[buf_cls] = updated

    if st.button("YOLO-Labels speichern", type="primary"):
        lines = canvas_objects_to_yolo(objects, st.session_state[buf_cls], w_img, h_img, scale)
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        st.success(f"Gespeichert: `{lbl_path}` ({len(lines)} Boxen).")

    with st.expander("Workflow"):
        st.markdown(
            """
            1. Bilder nach `data/kasten_dataset/images/` kopieren.  
            2. Klassen unter „Klassen speichern“ pflegen (Dropdown-Einträge).  
            3. Boxen zeichnen, Klassen wählen, **YOLO-Labels speichern**.  
            4. Danach: `python scripts/verify_labels.py` → `python scripts/prepare_dataset.py` → `python scripts/train_yolo.py`.
            """
        )


if __name__ == "__main__":
    main()
