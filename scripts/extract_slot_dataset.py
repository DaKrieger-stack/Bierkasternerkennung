#!/usr/bin/env python3
"""
Extrahiert Slot-ROIs aus (gelabelten) Kastenbildern für das CNN-Training von Stufe 1/2.

Ablauf: Bild -> detect_crate -> warp_crate_top_down -> 20 Slot-ROIs -> als PNG speichern.
Die ROIs landen flach in ``data/slot_dataset/unsorted/`` und werden anschließend
manuell in die Klassenordner sortiert:

    data/slot_dataset/
        empty/          # kein Slot belegt   -> SlotState.MISSING
        bottle_empty/   # Flasche ohne Korken -> SlotState.EMPTY
        bottle_full/    # Flasche mit Korken  -> SlotState.FULL

Beispiel:
    python scripts/extract_slot_dataset.py --images data/kasten_dataset/images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.detector import detect_crate  # noqa: E402
from kastendetektion.warp_grid import (  # noqa: E402
    extract_slot_roi,
    grid_slot_centers,
    warp_crate_top_down,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Slot-ROIs für CNN-Training extrahieren")
    parser.add_argument("--images", type=Path, default=ROOT / "data/kasten_dataset/images")
    parser.add_argument("--out", type=Path, default=ROOT / "data/slot_dataset/unsorted")
    parser.add_argument("--weights", default=None, help="YOLO best.pt (optional)")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--out-width", type=int, default=500)
    parser.add_argument("--out-height", type=int, default=400)
    parser.add_argument("--roi", type=int, default=64, help="Kantenlänge des gespeicherten ROI")
    args = parser.parse_args()

    if not args.images.is_dir():
        raise SystemExit(f"Bildordner nicht gefunden: {args.images}")
    args.out.mkdir(parents=True, exist_ok=True)

    half = max(8, int(min(args.out_width / args.cols, args.out_height / args.rows) * 0.4))
    n_imgs = 0
    n_rois = 0
    for img_path in iter_images(args.images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        det = detect_crate(frame, weights_path=args.weights)
        if det is None:
            continue
        warped, _ = warp_crate_top_down(
            frame, det.corners, out_width=args.out_width, out_height=args.out_height
        )
        centers = grid_slot_centers(args.out_width, args.out_height, rows=args.rows, cols=args.cols)
        n_imgs += 1
        for i, (cx, cy) in enumerate(centers):
            roi = extract_slot_roi(warped, float(cx), float(cy), half_size=half)
            if roi is None or roi.size == 0:
                continue
            roi = cv2.resize(roi, (args.roi, args.roi))
            name = f"{img_path.stem}_slot{i:02d}.png"
            cv2.imwrite(str(args.out / name), roi)
            n_rois += 1

    print(f"Fertig: {n_rois} ROIs aus {n_imgs} Kastenbildern -> {args.out}")
    print("Nächster Schritt: ROIs in data/slot_dataset/{empty,bottle_empty,bottle_full}/ einsortieren.")


if __name__ == "__main__":
    main()
