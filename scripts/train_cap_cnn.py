#!/usr/bin/env python3
"""
Trainiert den Stufe-2-Klassifikator (Kronkorken vorhanden = voll, sonst leer).

Nutzt nur die belegten Slots: ``bottle_empty/`` vs. ``bottle_full/`` unter
``data/slot_dataset/`` (siehe extract_slot_dataset.py).

Beispiel:
    python scripts/train_cap_cnn.py --epochs 30
Inferenz dann via KASTEN_CAP_WEIGHTS=models/cap_cnn.pt (oder --cap-weights).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.cap_classifier import CAP_CLASSES  # noqa: E402
from roi_training import train_roi_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stufe-2 Kronkorken-CNN trainieren")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/slot_dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "models/cap_cnn.pt")
    parser.add_argument("--arch", choices=["smallcnn", "mobilenet_v2"], default="smallcnn")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    folder_to_label = {
        "bottle_empty": CAP_CLASSES.index("bottle_empty"),
        "bottle_full": CAP_CLASSES.index("bottle_full"),
    }
    train_roi_classifier(
        dataset_dir=args.dataset,
        folder_to_label=folder_to_label,
        classes=list(CAP_CLASSES),
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
