#!/usr/bin/env python3
"""
Erzeugt ``data/slot_dataset/{empty,bottle_empty,bottle_full}/`` automatisch.

Nutzt die Pipeline (klassisch) als Pseudo-Labels, extrahiert je Slot ein 64×64-ROI
aus der entzerrten Draufsicht. Anschließend:

    python scripts/train_slot_cnn.py
    python scripts/train_cap_cnn.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.pipeline import CratePipeline  # noqa: E402
from kastendetektion.states import SlotState  # noqa: E402
from kastendetektion.warp_grid import estimate_slot_half_size, extract_slot_roi  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_STATE_TO_FOLDER = {
    SlotState.MISSING: "empty",
    SlotState.EMPTY: "bottle_empty",
    SlotState.FULL: "bottle_full",
}


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Slot-Dataset per Pipeline bootstrapen")
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=[
            ROOT / "data/kasten_dataset/images",
            ROOT / "data/frames",
        ],
        help="Ein oder mehrere Bildordner (rekursiv)",
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/slot_dataset")
    parser.add_argument("--roi", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0, help="0 = alle Bilder")
    parser.add_argument("--clear", action="store_true", help="Klassenordner vorher leeren")
    args = parser.parse_args()

    out_dirs = {name: args.dataset / name for name in _STATE_TO_FOLDER.values()}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        if args.clear:
            for old in d.glob("*.png"):
                old.unlink()

    pipe = CratePipeline(slot_method="classical", cap_method="classical")
    counts = {name: 0 for name in _STATE_TO_FOLDER.values()}
    n_imgs = 0
    n_skip = 0

    seen: set[Path] = set()
    for img_root in args.images:
        if not img_root.is_dir():
            print(f"Überspringe (kein Ordner): {img_root}")
            continue
        for img_path in iter_images(img_root):
            if img_path in seen:
                continue
            seen.add(img_path)
            if args.max_images and n_imgs >= args.max_images:
                break

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            analysis = pipe.analyze(frame)
            if analysis is None:
                n_skip += 1
                continue

            centers = [s.center_warped for s in analysis.slots]
            import numpy as np

            half = estimate_slot_half_size(np.array(centers, dtype=np.float32), analysis.rows, analysis.cols)
            n_imgs += 1
            for slot in analysis.slots:
                cx, cy = slot.center_warped
                roi = extract_slot_roi(analysis.warped, float(cx), float(cy), half_size=half)
                if roi is None or roi.size == 0:
                    continue
                roi = cv2.resize(roi, (args.roi, args.roi))
                folder = _STATE_TO_FOLDER[slot.state]
                name = f"{img_path.stem}_slot{slot.index:02d}.png"
                cv2.imwrite(str(out_dirs[folder] / name), roi)
                counts[folder] += 1

        if args.max_images and n_imgs >= args.max_images:
            break

    print(f"Bilder verarbeitet: {n_imgs} | ohne Kasten: {n_skip}")
    for folder, n in counts.items():
        print(f"  {folder}: {n}")
    print(f"Dataset: {args.dataset.resolve()}")
    total = sum(counts.values())
    if total < 30:
        raise SystemExit("Zu wenige ROIs — mehr Kastenbilder nötig oder Pipeline prüfen.")


if __name__ == "__main__":
    main()
