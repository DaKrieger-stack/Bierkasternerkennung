#!/usr/bin/env python3
"""
Extrahiert Trainings-ROIs exakt wie in der Live-Pipeline (YOLO → Warp → Grid).

Manuelle Crops in Bilder_Labels sehen oft anders aus als Live-ROIs — deshalb
erkennt das Modell nada live schlecht, obwohl die Label-Dateien 100 % passen.

Für jedes Label ``{bildstem}_slot{07}.png`` wird das Quellbild gesucht,
der Slot per Pipeline extrahiert und nach ``data/slot_dataset_pipeline/`` geschrieben.
Standalone-Crops (ohne _slotNN) werden zusätzlich kopiert.

    python scripts/sync_pipeline_rois.py
    python scripts/import_bilder_labels.py --clear --copy
    python scripts/sync_pipeline_rois.py --merge-into-slot-dataset
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.detector import detect_crate  # noqa: E402
from kastendetektion.warp_grid import (  # noqa: E402
    estimate_slot_half_size,
    extract_slot_roi,
    grid_slot_centers,
    warp_crate_top_down,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SRC = ROOT / "data" / "Bilder_Labels" / "Bilder_Labels"
FOLDER_MAP = {"voll": "bottle_full", "leer": "bottle_empty", "nada": "empty"}
OUT_DEFAULT = ROOT / "data" / "slot_dataset_pipeline"
OUT_MERGE = ROOT / "data" / "slot_dataset"
ROI_SIZE = 64
SLOT_RE = re.compile(r"_slot(\d+)$")
IMG_DIRS = (
    ROOT / "data" / "kasten_dataset" / "images",
    ROOT / "data" / "frames",
    ROOT / "data" / "Bilder_Labels",
    ROOT / "data" / "Bilder_Labels" / "Bilder_Labels",
)


def _find_source_image(stem: str) -> Path | None:
    for d in IMG_DIRS:
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem == stem:
                return p
    return None


def _extract_pipeline_roi(frame_bgr, slot_index: int) -> tuple[np.ndarray | None, bool]:
    det = detect_crate(frame_bgr)
    if det is None:
        return None, False
    warped, _ = warp_crate_top_down(frame_bgr, det.corners, out_width=500, out_height=400)
    centers = grid_slot_centers(500, 400, rows=4, cols=5)
    if slot_index < 0 or slot_index >= len(centers):
        return None, False
    half = estimate_slot_half_size(centers, 4, 5)
    cx, cy = centers[slot_index]
    roi = extract_slot_roi(warped, float(cx), float(cy), half_size=half)
    if roi is None or roi.size == 0:
        return None, False
    roi = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))
    return roi, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline-ROIs für Training synchronisieren")
    parser.add_argument("--src", type=Path, default=LABEL_SRC)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--merge-into-slot-dataset",
        action="store_true",
        help="Zusätzlich nach data/slot_dataset/ schreiben (ersetzt gleichnamige Dateien)",
    )
    args = parser.parse_args()

    out_root = args.out.resolve()
    for folder in FOLDER_MAP.values():
        (out_root / folder).mkdir(parents=True, exist_ok=True)

    pipeline_ok = pipeline_miss = standalone = 0

    for ui_folder, train_folder in FOLDER_MAP.items():
        src_dir = args.src / ui_folder
        if not src_dir.is_dir():
            continue
        for src_path in sorted(src_dir.rglob("*")):
            if not src_path.is_file() or src_path.suffix.lower() not in IMAGE_EXTS:
                continue
            m = SLOT_RE.search(src_path.stem)
            if m:
                slot_i = int(m.group(1))
                img_stem = src_path.stem[: m.start()]
                img_path = _find_source_image(img_stem)
                if img_path is None:
                    pipeline_miss += 1
                    # Fallback: manuelles Crop behalten
                    img = cv2.imread(str(src_path))
                    if img is None:
                        continue
                    roi = cv2.resize(img, (ROI_SIZE, ROI_SIZE))
                else:
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        pipeline_miss += 1
                        continue
                    roi, ok = _extract_pipeline_roi(frame, slot_i)
                    if not ok or roi is None:
                        pipeline_miss += 1
                        img = cv2.imread(str(src_path))
                        roi = cv2.resize(img, (ROI_SIZE, ROI_SIZE)) if img is not None else None
                    else:
                        pipeline_ok += 1
            else:
                standalone += 1
                img = cv2.imread(str(src_path))
                if img is None:
                    continue
                roi = cv2.resize(img, (ROI_SIZE, ROI_SIZE))

            if roi is None:
                continue
            out_name = src_path.stem + ".png"
            out_path = out_root / train_folder / out_name
            cv2.imwrite(str(out_path), roi)
            if args.merge_into_slot_dataset:
                merge_path = OUT_MERGE / train_folder / out_name
                merge_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(merge_path), roi)

    print(f"Pipeline-ROIs extrahiert: {pipeline_ok}")
    print(f"Fallback (kein Bild/Detektion): {pipeline_miss}")
    print(f"Standalone-Crops: {standalone}")
    print(f"Ziel: {out_root}")
    if args.merge_into_slot_dataset:
        print(f"Merged nach: {OUT_MERGE.resolve()}")


if __name__ == "__main__":
    main()
