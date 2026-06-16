#!/usr/bin/env python3
"""Live-Test der Pipeline auf Kastenbilder + Label-ROIs (nada→voll prüfen)."""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.inference_config import ACTIVE, ORIGINAL, StateInferenceParams  # noqa: E402
from kastendetektion.pipeline import CratePipeline  # noqa: E402
from kastendetektion.state_classifier import StateClassifier  # noqa: E402
from kastendetektion.states import SlotState  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def eval_crops(params: StateInferenceParams, slot_thr: float) -> tuple[int, int, int, int]:
    clf = StateClassifier("ml", inference_params=params, slot_occupied_threshold=slot_thr)
    from kastendetektion.slot_classifier import SlotOccupancyClassifier

    slot = SlotOccupancyClassifier("ml", occupied_threshold=slot_thr)
    ok = nada_full = nada_n = total = 0
    for folder, want in [("empty", SlotState.MISSING), ("bottle_empty", SlotState.EMPTY), ("bottle_full", SlotState.FULL)]:
        for p in (ROOT / "data/slot_dataset" / folder).rglob("*"):
            if p.suffix.lower() not in EXTS:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            total += 1
            sp = slot.predict([img])[0]
            st = clf.predict([img], slot_gates=[sp])[0]
            if want == SlotState.MISSING:
                nada_n += 1
                if st[0] == SlotState.FULL:
                    nada_full += 1
            if st[0] == want:
                ok += 1
    return ok, total, nada_full, nada_n


def eval_kasten_images(params: StateInferenceParams, slot_thr: float) -> None:
    pipe = CratePipeline(
        state_method="ml",
        slot_method="ml",
        state_inference_params=params,
        slot_occupied_threshold=slot_thr,
    )
    img_dir = ROOT / "data/kasten_dataset/images"
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in EXTS:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        a = pipe.analyze(frame)
        if a is None:
            print(f"  {img_path.name[:50]} — kein Kasten")
            continue
        print(
            f"  {img_path.name[:52]:52} | voll {a.full_count:2d} leer {a.empty_count:2d} nada {a.missing_count:2d}"
        )


def main() -> None:
    print("=== Label-ROIs ===")
    for name, params, thr in [("ORIGINAL", ORIGINAL, 0.50), ("ACTIVE", ACTIVE, 0.58)]:
        ok, total, nf, nn = eval_crops(params, thr)
        print(f"{name}: {ok}/{total} korrekt | nada->voll {nf}/{nn}")

    print("\n=== Live Kastenbilder (ACTIVE) ===")
    eval_kasten_images(ACTIVE, 0.58)


if __name__ == "__main__":
    main()
