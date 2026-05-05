#!/usr/bin/env python3
"""Live-Kamera: cv2.VideoCapture + detect_crate (YOLO mit Fallback)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Projektroot auf PYTHONPATH (Aufruf aus beliebigem CWD)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.detector import detect_crate, draw_detection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Kamera-Stream mit Kastendetektion")
    parser.add_argument("--camera", type=int, default=0, help="Geräteindex für VideoCapture")
    parser.add_argument("--weights", default=None, help="Pfad zu best.pt (optional)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--log-corners", action="store_true", help="Eckpunkte ins Log")
    parser.add_argument("--contour-only", action="store_true", help="Nur Canny/Kontur-Fallback")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {args.camera} lässt sich nicht öffnen.")

    weights = None if args.contour_only else args.weights

    print("q oder ESC beendet.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        det = detect_crate(
            frame,
            weights_path=weights,
            prefer_yolo=not args.contour_only,
            conf=args.conf,
            log_corners=args.log_corners,
        )

        vis = frame
        if det is not None:
            vis = draw_detection(frame, det)

        cv2.imshow("Bierkasten — q zum Beenden", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
