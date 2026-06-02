#!/usr/bin/env python3
"""
Erzeugt YOLO-Labels (Klasse 0 „bierkasten“) per klassischer Detektion.

Standard: Rot-Kasten-HSV, bei Misserfolg Kontur-Fallback (``--method auto``).

Beispiel (Projektroot):
    python scripts/auto_label.py
    python scripts/auto_label.py --preview-dir data/kasten_dataset/_auto_preview
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kastendetektion.detector import CrateDetectionResult, detect_crate_contour, draw_detection
from red_crate_detect import draw_result, run_detection

DEFAULT_IMG_DIR = SCRIPT_ROOT / "data" / "kasten_dataset" / "images"
DEFAULT_LBL_DIR = SCRIPT_ROOT / "data" / "kasten_dataset" / "labels"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0
Method = Literal["auto", "red", "contour"]


@dataclass
class DetectionOutcome:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    source: str


def xywh_to_yolo_line(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    *,
    class_id: int = CLASS_ID,
) -> str:
    """Pixel-BBox (x,y,w,h) → eine YOLO-Zeile (wie label_ui.py)."""
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 1e-6), 1.0)
    nh = min(max(nh, 1e-6), 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def detect_red(image_path: Path) -> DetectionOutcome | None:
    result = run_detection(image_path)
    candidate = result.candidate
    if candidate is None:
        return None
    x, y, w, h = candidate.bbox
    return DetectionOutcome(
        x=x,
        y=y,
        w=w,
        h=h,
        confidence=float(candidate.score),
        source="red",
    )


def detect_contour(image_bgr) -> DetectionOutcome | None:
    det = detect_crate_contour(image_bgr)
    if det is None:
        return None
    return DetectionOutcome(
        x=det.x,
        y=det.y,
        w=det.w,
        h=det.h,
        confidence=det.confidence,
        source="contour",
    )


def detect_image(
    image_path: Path,
    image_bgr,
    method: Method,
) -> DetectionOutcome | None:
    if method == "red":
        return detect_red(image_path)
    if method == "contour":
        return detect_contour(image_bgr)
    # auto: red, then contour
    outcome = detect_red(image_path)
    if outcome is not None:
        return outcome
    return detect_contour(image_bgr)


def render_preview(image_bgr, outcome: DetectionOutcome, image_path: Path):
    if outcome.source == "red":
        result = run_detection(image_path)
        return draw_result(
            image_bgr,
            result.candidate,
            result.lines,
            result.circles,
        )
    det = CrateDetectionResult(
        x=outcome.x,
        y=outcome.y,
        w=outcome.w,
        h=outcome.h,
        corners=_corners_from_xywh(outcome.x, outcome.y, outcome.w, outcome.h),
        confidence=outcome.confidence,
        source="contour",
    )
    return draw_detection(image_bgr, det)


def _corners_from_xywh(x: int, y: int, w: int, h: int):
    import numpy as np

    x2, y2 = x + w, y + h
    return np.array([[x, y], [x2, y], [x2, y2], [x, y2]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO-Labels für bierkasten automatisch aus CV-Detektion erzeugen",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMG_DIR,
        help="Ordner mit Trainingsbildern",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LBL_DIR,
        help="Zielordner für YOLO-.txt",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "red", "contour"),
        default="auto",
        help="Detektion: auto (Rot, dann Kontur), nur red oder nur contour",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vorhandene Label-Dateien nicht überschreiben (Standard: an)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Alle Labels neu erzeugen (setzt --no-skip-existing voraus)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Statistik, keine Dateien schreiben",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional: Overlay-Bilder zum Stichproben-Check",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Mindest-Konfidenz/Score (Rot: candidate.score, Kontur: det.confidence)",
    )
    args = parser.parse_args()

    os.chdir(SCRIPT_ROOT)

    skip_existing = args.skip_existing and not args.overwrite
    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"Keine Bilder in {images_dir}")

    method: Method = args.method
    ok = 0
    skipped = 0
    failed: list[str] = []
    low_conf: list[str] = []

    preview_dir = args.preview_dir.resolve() if args.preview_dir else None
    if preview_dir and not args.dry_run:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if skip_existing and label_path.is_file():
            skipped += 1
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            failed.append(image_path.name)
            print(f"  FEHLER (lesen): {image_path.name}")
            continue

        img_h, img_w = image_bgr.shape[:2]
        outcome = detect_image(image_path, image_bgr, method)

        if outcome is None:
            failed.append(image_path.name)
            print(f"  FEHLER (keine Box): {image_path.name}")
            continue

        if outcome.confidence < args.min_confidence:
            low_conf.append(image_path.name)
            failed.append(image_path.name)
            print(
                f"  FEHLER (Konfidenz {outcome.confidence:.3f} < {args.min_confidence}): "
                f"{image_path.name} [{outcome.source}]"
            )
            continue

        line = xywh_to_yolo_line(outcome.x, outcome.y, outcome.w, outcome.h, img_w, img_h)

        if not args.dry_run:
            label_path.write_text(line + "\n", encoding="utf-8")
            if preview_dir is not None:
                vis = render_preview(image_bgr, outcome, image_path)
                cv2.imwrite(str(preview_dir / f"{image_path.stem}_preview.jpg"), vis)

        ok += 1
        print(f"  OK [{outcome.source}] {image_path.name} (conf={outcome.confidence:.3f})")

    print()
    print(f"Methode: {method}")
    print(f"Bilder gesamt: {len(images)}")
    print(f"Labels erzeugt: {ok}")
    print(f"Übersprungen (vorhanden): {skipped}")
    print(f"Fehlgeschlagen: {len(failed)}")
    if args.dry_run:
        print("(Dry-Run — keine Dateien geschrieben)")
    if preview_dir and not args.dry_run:
        print(f"Vorschau: {preview_dir}")

    if failed:
        print("\nFehlgeschlagene Dateien:")
        for name in failed:
            print(f"  - {name}")

    if low_conf:
        print("\nUnter min-confidence (nicht gespeichert):")
        for name in low_conf:
            print(f"  - {name}")

    if failed and not args.dry_run:
        print(
            "\nHinweis: Fehlbilder in der Label-UI nachziehen:\n"
            "  streamlit run scripts/label_ui.py"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
