#!/usr/bin/env python3
"""
Trainiert den Stufe-1-Klassifikator (Slot belegt/leer).

Erwartet die einsortierten ROIs unter ``data/slot_dataset/`` (siehe extract_slot_dataset.py):
``empty/`` = leer, ``bottle_empty/`` + ``bottle_full/`` = belegt.

Beispiel:
    python scripts/train_slot_cnn.py --epochs 30 --arch smallcnn
Inferenz dann via KASTEN_SLOT_WEIGHTS=models/slot_cnn.pt (oder --slot-weights).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.slot_classifier import SLOT_CLASSES  # noqa: E402
from roi_training import train_roi_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stufe-1 Slot-CNN trainieren")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/slot_dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "models/slot_cnn.pt")
    parser.add_argument("--arch", choices=["smallcnn", "mobilenet_v2"], default="smallcnn")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    # empty -> 0 (= "empty"/nicht belegt), beide Flaschen-Ordner -> 1 (= "occupied")
    folder_to_label = {
        "empty": SLOT_CLASSES.index("empty"),
        "bottle_empty": SLOT_CLASSES.index("occupied"),
        "bottle_full": SLOT_CLASSES.index("occupied"),
    }
    train_roi_classifier(
        dataset_dir=args.dataset,
        folder_to_label=folder_to_label,
        classes=list(SLOT_CLASSES),
        out_path=args.out,
        arch=args.arch,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
