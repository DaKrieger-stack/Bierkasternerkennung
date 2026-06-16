#!/usr/bin/env python3
"""
Importiert manuell gelabelte ROIs aus ``data/Bilder_Labels/`` ins Trainingsformat.

Quelle (UI-Labels):
    data/Bilder_Labels/Bilder_Labels/{voll,leer,nada}/*.png

Ziel (Training):
    data/slot_dataset/{bottle_full,bottle_empty,empty}/*.png

Beispiel:
    python scripts/import_bilder_labels.py --clear
    python scripts/train_slot_cnn.py
    python scripts/train_cap_cnn.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SRC = ROOT / "data" / "Bilder_Labels" / "Bilder_Labels"
DEFAULT_DST = ROOT / "data" / "slot_dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# UI-Klasse -> Trainingsordner
LABEL_TO_FOLDER = {
    "voll": "bottle_full",
    "leer": "bottle_empty",
    "nada": "empty",
}


def resolve_src(path: Path) -> Path:
    path = path.resolve()
    if (path / "voll").is_dir() or (path / "leer").is_dir() or (path / "nada").is_dir():
        return path
    nested = path / "Bilder_Labels"
    if nested.is_dir():
        return nested.resolve()
    raise SystemExit(f"Keine Label-Ordner (voll/leer/nada) unter {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bilder_Labels -> slot_dataset importieren")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Zielordner bottle_full/bottle_empty/empty vorher leeren",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Kopieren statt Hardlink (Standard: Hardlink auf Windows, sonst Kopie)",
    )
    args = parser.parse_args()

    src = resolve_src(args.src)
    dst = args.dst.resolve()
    folders = set(LABEL_TO_FOLDER.values()) | {"ignored"}

    for folder in folders:
        out = dst / folder
        out.mkdir(parents=True, exist_ok=True)
        if args.clear:
            for old in out.glob("*"):
                if old.is_file():
                    old.unlink()

    counts = {name: 0 for name in LABEL_TO_FOLDER}
    skipped = 0
    use_copy = args.copy

    for label, out_folder in LABEL_TO_FOLDER.items():
        in_dir = src / label
        if not in_dir.is_dir():
            print(f"Warnung: fehlt {in_dir}")
            continue
        out_dir = dst / out_folder
        for p in sorted(in_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            target = out_dir / p.name
            if target.exists():
                skipped += 1
                continue
            try:
                if not use_copy:
                    target.hardlink_to(p)
                else:
                    shutil.copy2(p, target)
            except OSError:
                shutil.copy2(p, target)
            counts[label] += 1

    total = sum(counts.values())
    print(f"Quelle:  {src}")
    print(f"Ziel:    {dst}")
    for label, n in counts.items():
        print(f"  {label:5s} -> {LABEL_TO_FOLDER[label]:12s}: {n}")
    print(f"Gesamt: {total} ROIs importiert ({skipped} übersprungen, bereits vorhanden)")
    if total == 0:
        raise SystemExit("Nichts importiert — Pfad prüfen.")


if __name__ == "__main__":
    main()
