"""
Perspektivische Entzerrung und 4×5-Slot-Gitter (Projektplan Abschnitt 2.2).

Voraussetzung: ``corners`` von ``detect_crate`` in der Reihenfolge TL, TR, BR, BL
(Bildkoordinaten). Das sind bei YOLO achsparallele Ecken — Annäherung an die
sichtbare Kastenfläche; idealerweise später echte Oberflächen-Ecken kalibrieren.
"""

from __future__ import annotations

import cv2
import numpy as np


def warp_crate_top_down(
    image_bgr: np.ndarray,
    corners_tl_tr_br_bl: np.ndarray,
    out_width: int = 500,
    out_height: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Entzerrt die durch die vier Eckpunkte aufgespannte Fläche in eine Norm-Ansicht.

    Returns:
        warped BGR-Bild, 3×3 Homographie ``H`` (wie ``cv2.getPerspectiveTransform``).
    """
    src = np.asarray(corners_tl_tr_br_bl, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [
            [0.0, 0.0],
            [float(out_width - 1), 0.0],
            [float(out_width - 1), float(out_height - 1)],
            [0.0, float(out_height - 1)],
        ],
        dtype=np.float32,
    )
    h_mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, h_mat, (out_width, out_height))
    return warped, h_mat


def grid_slot_centers(
    warped_width: int,
    warped_height: int,
    rows: int = 4,
    cols: int = 5,
) -> np.ndarray:
    """
    Mittelpunkte der Slots im entzerrten Bild (Paulaner 4×5).

    Reihenfolge: Zeile für Zeile von oben nach unten, links nach rechts → 20 Punkte.
    """
    centers = np.zeros((rows * cols, 2), dtype=np.float32)
    cell_w = warped_width / float(cols)
    cell_h = warped_height / float(rows)
    k = 0
    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            centers[k, 0] = cx
            centers[k, 1] = cy
            k += 1
    return centers


def map_points_to_original(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Mappt Punkte aus Entzerr-Koordinaten zurück ins Kamerabild (Homographie-Inverse)."""
    h_inv = np.linalg.inv(H)
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, h_inv)
    return mapped.reshape(-1, 2)


def extract_slot_roi(
    warped_bgr: np.ndarray,
    cx: float,
    cy: float,
    half_size: int = 32,
) -> np.ndarray | None:
    """Quadratischer ROI um einen Slot-Mittelpunkt (für späteres CNN 64×64 o. Ä.)."""
    h, w = warped_bgr.shape[:2]
    x1 = int(round(cx - half_size))
    y1 = int(round(cy - half_size))
    x2 = int(round(cx + half_size))
    y2 = int(round(cy + half_size))
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    return warped_bgr[y1:y2, x1:x2].copy()
