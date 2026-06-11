#!/usr/bin/env python3
"""Diagnose: Kasten-Orientierung vs. 4×5-Gitter (Rotation)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.detector import detect_crate  # noqa: E402
from kastendetektion.overlay import draw_pipeline_overlay  # noqa: E402
from kastendetektion.pipeline import analyze_frame  # noqa: E402
from kastendetektion.warp_grid import canonicalize_corners_for_grid, warp_crate_top_down  # noqa: E402


def edge_lengths(corners: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = corners
    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(bl - br))
    left = float(np.linalg.norm(tl - bl))
    return top, right, bottom, left


def diagnose(frame: np.ndarray, label: str, out_dir: Path) -> None:
    det = detect_crate(frame)
    if det is None:
        print(f"=== {label} === KEINE DETEKTION")
        return

    top, right, bottom, left = edge_lengths(det.corners)
    canon = canonicalize_corners_for_grid(det.corners)
    c_top, c_right, _, _ = edge_lengths(canon)
    long_e = max(top, right, bottom, left)
    short_e = min(top, right, bottom, left)
    ratio = long_e / max(short_e, 1e-6)
    top_is_long = c_top >= c_right - 1.0
    was_rotated = not np.allclose(det.corners, canon)

    analysis = analyze_frame(frame, slot_method="classical", cap_method="classical")

    print(f"=== {label} ===")
    print(f"  Detektion: {det.source}, conf={det.confidence:.2f}, Box {det.w}x{det.h}")
    print(f"  Kanten roh (TL-TR, TR-BR, BR-BL, BL-TL): {top:.0f}, {right:.0f}, {bottom:.0f}, {left:.0f}")
    print(f"  Nach Kanonisierung TL-TR={c_top:.0f}, TR-BR={c_right:.0f}, rotiert={was_rotated}")
    print(f"  Lang/kurz Ratio: {ratio:.2f} (Paulaner erwartet ~1.25 = 5/4)")
    if top_is_long:
        print("  OBEN-Kante (TL-TR) = LANG -> cols=5 horizontal (KORREKT)")
    else:
        print("  OBEN-Kante (TL-TR) = KURZ -> cols=5 auf 4er-Seite (FEHLER!)")
    if analysis:
        print(
            f"  Slots: {analysis.occupied_count}/20 belegt, "
            f"{analysis.missing_count} fehlt, {analysis.full_count} voll"
        )
        cv2.imwrite(str(out_dir / f"overlay_{label}.jpg"), draw_pipeline_overlay(frame, analysis))
    warped, _ = warp_crate_top_down(frame, det.corners)
    cv2.imwrite(str(out_dir / f"warp_{label}.jpg"), warped)


def rotate(frame: np.ndarray, deg: float) -> np.ndarray:
    h, w = frame.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (nw / 2) - w / 2
    m[1, 2] += (nh / 2) - h / 2
    return cv2.warpAffine(frame, m, (nw, nh))


def main() -> None:
    img_path = ROOT / "demo_output" / "diagnose" / "capture.jpg"
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    if not img_path.is_file():
        # Fallback: bekanntes Testbild
        candidates = list(Path(ROOT).rglob("*.png")) + list(Path(ROOT).rglob("*.jpg"))
        img_path = next((p for p in candidates if "image-22c5" in p.name or "capture" in p.name), None)
    if img_path is None or not Path(img_path).is_file():
        raise SystemExit("Kein Bild gefunden. Nutzung: python scripts/diagnose_orientation.py [bild.jpg]")

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"Bild nicht lesbar: {img_path}")

    out_dir = ROOT / "demo_output" / "diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bild: {img_path}\n")
    diagnose(frame, "original", out_dir)
    for deg, name in [(90, "rot90"), (180, "rot180"), (270, "rot270")]:
        diagnose(rotate(frame, deg), name, out_dir)
    print(f"\nOverlays gespeichert unter: {out_dir}")


if __name__ == "__main__":
    main()
