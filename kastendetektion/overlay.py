"""
Farbcodiertes Overlay + Gesamtstatistik für eine ``CrateAnalysis``.

Legende (siehe ``states.py``): grün = voll, gelb = leer, rot = fehlt.
"""

from __future__ import annotations

import cv2
import numpy as np

from kastendetektion.pipeline import CrateAnalysis
from kastendetektion.states import SlotState, color_for, label_for


def draw_pipeline_overlay(
    frame_bgr: np.ndarray,
    analysis: CrateAnalysis,
    *,
    draw_box: bool = True,
    draw_labels: bool = True,
    marker_radius: int = 12,
) -> np.ndarray:
    """Zeichnet Slot-Marker (zurückprojiziert), Kastenrahmen, Statistik und Legende."""
    out = frame_bgr.copy()

    if draw_box:
        det = analysis.detection
        pts = det.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    for slot in analysis.slots:
        cx, cy = int(round(slot.center_orig[0])), int(round(slot.center_orig[1]))
        color = color_for(slot.state)
        cv2.circle(out, (cx, cy), marker_radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), marker_radius, (0, 0, 0), 1, cv2.LINE_AA)
        if draw_labels:
            cv2.putText(
                out, str(slot.index), (cx - 6, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA
            )

    _draw_stats(out, analysis)
    _draw_legend(out)
    return out


def _draw_stats(img: np.ndarray, analysis: CrateAnalysis) -> None:
    text = (
        f"{analysis.occupied_count}/{analysis.total} belegt, "
        f"davon {analysis.full_count} voll"
    )
    src = analysis.detection.source
    text2 = f"Detektion: {src}  |  Slots: {analysis.full_count} voll, {analysis.empty_count} leer, {analysis.missing_count} fehlt"
    _label_box(img, [text, text2], org=(10, 10))


def _draw_legend(img: np.ndarray) -> None:
    h = img.shape[0]
    x = 10
    y = h - 70
    cv2.rectangle(img, (x - 5, y - 20), (x + 175, y + 55), (40, 40, 40), -1)
    for i, state in enumerate((SlotState.FULL, SlotState.EMPTY, SlotState.MISSING)):
        yy = y + i * 22
        cv2.circle(img, (x + 8, yy), 8, color_for(state), -1, cv2.LINE_AA)
        cv2.putText(
            img, label_for(state), (x + 26, yy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )


def _label_box(img: np.ndarray, lines: list[str], org: tuple[int, int]) -> None:
    x, y = org
    pad = 6
    line_h = 22
    width = max((cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines), default=0)
    cv2.rectangle(
        img, (x, y), (x + width + 2 * pad, y + line_h * len(lines) + pad), (40, 40, 40), -1
    )
    for i, t in enumerate(lines):
        cv2.putText(
            img, t, (x + pad, y + line_h * (i + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
