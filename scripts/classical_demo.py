#!/usr/bin/env python3
"""Klassischer Pfad (Arbeitspaket Schritt 3): Canny + Konturen auf einem Testbild."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.detector import detect_crate_contour, draw_detection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Pfad zum Testbild")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Ausgabe-Bild speichern")
    args = parser.parse_args()

    img_path = args.image.resolve()
    if not img_path.is_file():
        raise SystemExit(f"Datei nicht gefunden: {img_path}")

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"Bild konnte nicht gelesen werden: {img_path}")

    det = detect_crate_contour(frame)
    if det is None:
        print("Keine Kontur gefunden.")
        return

    vis = draw_detection(frame, det, color_contour=(0, 255, 0))
    cv2.imshow("Kontur-Fallback (Arbeitspaket: gruene Box)", vis)
    print("Beliebige Taste zum Schließen.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), vis)
        print(f"Gespeichert: {args.output.resolve()}")


if __name__ == "__main__":
    main()
