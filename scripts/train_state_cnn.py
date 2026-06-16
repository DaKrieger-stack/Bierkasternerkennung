#!/usr/bin/env python3
"""
Trainiert den 3-Klassen-Slot-Zustand (nada / leer / voll) mit allen Labels.

    python scripts/import_bilder_labels.py --clear --copy
    python scripts/train_state_cnn.py --epochs 40 --arch mobilenet_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.state_classifier import STATE_CLASSES  # noqa: E402
from roi_training import train_roi_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="3-Klassen State-CNN (nada/leer/voll)")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/slot_dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "models/state_cnn.pt")
    parser.add_argument("--arch", choices=["smallcnn", "mobilenet_v2"], default="mobilenet_v2")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    folder_to_label = {name: i for i, name in enumerate(STATE_CLASSES)}
    train_roi_classifier(
        dataset_dir=args.dataset,
        folder_to_label=folder_to_label,
        classes=list(STATE_CLASSES),
        out_path=args.out,
        arch=args.arch,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        augment=True,
        class_weights=True,
        focus_class=STATE_CLASSES.index("empty"),
        min_recall_weight=0.5,
    )


if __name__ == "__main__":
    main()
