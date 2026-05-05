"""
Teilt flache Bilder/Etiketten in train/ und val/ auf (Standard-YOLO-Ordnerstruktur).

Voraussetzung (z. B. nach LabelImg, Format YOLO):
  data/kasten_dataset/images/*.jpg|png
  data/kasten_dataset/labels/*.txt   (gleicher Dateiname wie Bild)

Erzeugt:
  data/kasten_dataset/images/train|val/
  data/kasten_dataset/labels/train|val/
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXT:
            continue
        label = labels_dir / f"{img.stem}.txt"
        pairs.append((img, label if label.is_file() else None))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/Val-Split für YOLO-Kasten-Dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/kasten_dataset"),
        help="Wurzel mit Unterordnern images/ und labels/ (flach)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Anteil Validierung (0–1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(SCRIPT_ROOT)

    root: Path = args.dataset_root.resolve()
    flat_img = root / "images"
    flat_lbl = root / "labels"

    if not flat_img.is_dir():
        raise SystemExit(f"Bildordner fehlt: {flat_img}")

    pairs = collect_pairs(flat_img, flat_lbl)
    if not pairs:
        raise SystemExit(
            f"Keine Bilder in {flat_img}. Nach Dropbox-Download Bilder hier ablegen "
            "(oder Unterordner verschieben), dann erneut ausführen."
        )

    missing = [p[0].name for p in pairs if p[1] is None]
    if missing:
        print(f"Hinweis: {len(missing)} Bild(er) ohne passende .txt in {flat_lbl} – nur Bilder mit Labels werden verwendet.")
    labeled = [(i, l) for i, l in pairs if l is not None]
    if not labeled:
        raise SystemExit(
            f"Keine Label-Paare gefunden. Bitte mit LabelImg annotieren (YOLO-Format) unter {flat_lbl}."
        )

    if len(labeled) < 2:
        raise SystemExit("Mindestens zwei annotierte Bilder für einen Train/Val-Split nötig.")

    random.seed(args.seed)
    random.shuffle(labeled)

    n_val = max(1, int(len(labeled) * args.val_fraction))
    if n_val >= len(labeled):
        n_val = len(labeled) - 1

    val_pairs = labeled[:n_val]
    train_pairs = labeled[n_val:]
    val_keys = set(val_pairs)

    # Zielstruktur
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def move_pair(img: Path, lbl: Path, split: str) -> None:
        dst_i = root / "images" / split / img.name
        dst_l = root / "labels" / split / lbl.name
        if img.resolve() != dst_i.resolve():
            shutil.move(str(img), str(dst_i))
        if lbl.resolve() != dst_l.resolve():
            shutil.move(str(lbl), str(dst_l))

    for img, lbl in labeled:
        split = "val" if (img, lbl) in val_keys else "train"
        move_pair(img, lbl, split)

    print(f"Dataset: {root}")
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    print("Training starten: python scripts/train_yolo.py")


if __name__ == "__main__":
    main()
