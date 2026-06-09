#!/usr/bin/env python3
"""
End-to-End-Demo der Füllzustand-Pipeline auf einem Bild oder Live-Kamera.

Beispiele:
    python scripts/pipeline_demo.py --image demo_output/synthetic_crate.png --save out.png
    python scripts/pipeline_demo.py --camera 0 --slot-method classical --cap-method classical
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.overlay import draw_pipeline_overlay  # noqa: E402
from kastendetektion.pipeline import CratePipeline  # noqa: E402
from kastendetektion.stabilize import SlotStateStabilizer  # noqa: E402


def _build_pipeline(args: argparse.Namespace) -> CratePipeline:
    return CratePipeline(
        slot_method=args.slot_method,
        cap_method=args.cap_method,
        weights_path=None if args.contour_only else args.weights,
        prefer_yolo=not args.contour_only,
        conf=args.conf,
    )


def run_image(args: argparse.Namespace) -> None:
    img_path = Path(args.image).resolve()
    if not img_path.is_file():
        raise SystemExit(f"Bild nicht gefunden: {img_path}")
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"Bild konnte nicht gelesen werden: {img_path}")

    pipe = _build_pipeline(args)
    analysis = pipe.analyze(frame)
    if analysis is None:
        print("Kein Kasten erkannt.")
        return

    vis = draw_pipeline_overlay(frame, analysis)
    print(
        f"Stufe1/2 [{pipe.slot_clf.active_method}/{pipe.cap_clf.active_method}] -> "
        f"{analysis.occupied_count}/{analysis.total} belegt, {analysis.full_count} voll, "
        f"{analysis.empty_count} leer, {analysis.missing_count} fehlt"
    )

    if args.save:
        out = Path(args.save).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        print(f"Gespeichert: {out}")
    if not args.no_window:
        cv2.imshow("Pipeline-Demo - Taste zum Schliessen", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {args.camera} lässt sich nicht öffnen.")

    pipe = _build_pipeline(args)
    stab = SlotStateStabilizer(window=args.stabilize) if args.stabilize > 1 else None
    print("q oder ESC beendet.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        analysis = pipe.analyze(frame)
        vis = frame
        if analysis is not None:
            if stab is not None:
                stable = stab.update([s.state for s in analysis.slots])
                for slot, st in zip(analysis.slots, stable):
                    slot.state = st
            vis = draw_pipeline_overlay(frame, analysis)
        cv2.imshow("Pipeline-Demo - q zum Beenden", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Füllzustand-Pipeline (Bild oder Kamera)")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", help="Pfad zu einem Bild")
    src.add_argument("--camera", type=int, default=0, help="Kamera-Index (Standard 0)")
    parser.add_argument("--slot-method", choices=["classical", "ml", "auto"], default="auto")
    parser.add_argument("--cap-method", choices=["classical", "ml", "auto"], default="auto")
    parser.add_argument("--weights", default=None, help="Pfad zu YOLO best.pt (optional)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--contour-only", action="store_true", help="Nur klassischer Kasten-Fallback")
    parser.add_argument("--stabilize", type=int, default=1, help="Frame-Fenster fürs Stabilisieren (1 = aus)")
    parser.add_argument("--save", default=None, help="Ergebnisbild speichern (nur --image)")
    parser.add_argument("--no-window", action="store_true", help="Kein Fenster anzeigen (nur speichern)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.image:
        run_image(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
