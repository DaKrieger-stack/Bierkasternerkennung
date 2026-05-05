#!/usr/bin/env python3
"""Prüft Bilder und YOLO-Labels (eine Klasse ``bierkasten`` = Index 0)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASS = 0


def check_label_file(txt_path: Path) -> list[str]:
    errors: list[str] = []
    text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        errors.append("Datei ist leer (mindestens eine Box nötig)")
        return errors
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Zeile {lineno}: erwartet 5 Felder (class cx cy w h), hat {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            nums = [float(parts[i]) for i in range(1, 5)]
        except ValueError:
            errors.append(f"Zeile {lineno}: keine gültigen Zahlen")
            continue
        if cls != EXPECTED_CLASS:
            errors.append(
                f"Zeile {lineno}: Klassen-ID {cls} (erwartet nur {EXPECTED_CLASS} für „bierkasten“)"
            )
        for name, v in zip(("cx", "cy", "w", "h"), nums):
            if not (0.0 <= v <= 1.0):
                errors.append(f"Zeile {lineno}: {name}={v} liegt nicht in [0, 1] (YOLO-normalisiert)")
        if nums[2] <= 0 or nums[3] <= 0:
            errors.append(f"Zeile {lineno}: Breite/Höhe müssen > 0 sein")
    return errors


def scan_folder(images_dir: Path, labels_dir: Path, label: str) -> tuple[int, int, list[str]]:
    """Returns (ok_images, issues_count, messages)."""
    messages: list[str] = []
    ok = 0
    issues = 0
    if not images_dir.is_dir():
        messages.append(f"[{label}] Bildordner fehlt: {images_dir}")
        return 0, 1, messages

    imgs = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    if not imgs:
        messages.append(f"[{label}] Keine Bilder in {images_dir}")
        return 0, 1, messages

    labels_dir.mkdir(parents=True, exist_ok=True)

    for img in imgs:
        stem = img.stem
        txt = labels_dir / f"{stem}.txt"
        if not txt.is_file():
            messages.append(f"[{label}] Kein Label zu {img.name} ({txt.name} fehlt)")
            issues += 1
            continue
        errs = check_label_file(txt)
        if errs:
            messages.append(f"[{label}] {img.name}:")
            messages.extend(f"    {e}" for e in errs)
            issues += 1
        else:
            ok += 1

    if labels_dir.is_dir():
        image_stems = {p.stem for p in imgs}
        for txt in labels_dir.glob("*.txt"):
            if txt.stem not in image_stems:
                messages.append(f"[{label}] Label ohne passendes Bild: {txt.name}")
                issues += 1

    return ok, issues, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-Labels für bierkasten prüfen")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/kasten_dataset"),
        help="Dataset-Wurzel mit images/ und labels/",
    )
    args = parser.parse_args()
    os.chdir(SCRIPT_ROOT)

    root = args.dataset_root.resolve()
    messages: list[str] = []
    total_ok = 0
    total_issues = 0

    train_img = root / "images" / "train"
    val_img = root / "images" / "val"
    flat_img = root / "images"

    if train_img.is_dir() or val_img.is_dir():
        if train_img.is_dir():
            ok, iss, msg = scan_folder(train_img, root / "labels" / "train", "train")
            total_ok += ok
            total_issues += iss
            messages.extend(msg)
        if val_img.is_dir():
            ok, iss, msg = scan_folder(val_img, root / "labels" / "val", "val")
            total_ok += ok
            total_issues += iss
            messages.extend(msg)
    elif flat_img.is_dir():
        ok, iss, msg = scan_folder(flat_img, root / "labels", "flat")
        total_ok += ok
        total_issues += iss
        messages.extend(msg)
    else:
        raise SystemExit(f"Kein gültiger Dataset-Pfad: {root}")

    print(f"OK (sauber gelabelt): {total_ok}")
    if messages:
        print(f"Hinweise/Fehler: {total_issues}")
        print("\n".join(messages))
        raise SystemExit(1)

    print("Keine offensichtlichen Probleme — du kannst prepare_dataset.py / train_yolo.py ausführen.")


if __name__ == "__main__":
    main()
