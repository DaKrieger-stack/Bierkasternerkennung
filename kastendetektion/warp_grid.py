"""
Perspektivische Entzerrung und 4×5-Slot-Gitter (Projektplan Abschnitt 2.2).

Voraussetzung: ``corners`` von ``detect_crate`` in der Reihenfolge TL, TR, BR, BL
(Bildkoordinaten). Vor dem Warp wird die Orientierung kanonisiert: die lange
Kastenkante (5 Flaschen) liegt immer horizontal (TL→TR), die kurze (4) vertikal.
"""

from __future__ import annotations

import cv2
import numpy as np


def _edge_lengths(corners: np.ndarray) -> tuple[float, float, float, float]:
    """Kantenlängen für TL→TR, TR→BR, BR→BL, BL→TL."""
    tl, tr, br, bl = corners.reshape(4, 2)
    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(bl - br))
    left = float(np.linalg.norm(tl - bl))
    return top, right, bottom, left


def canonicalize_corners_for_grid(
    corners_tl_tr_br_bl: np.ndarray,
    *,
    long_cols: int = 5,
    short_rows: int = 4,
) -> np.ndarray:
    """
    Dreht die Ecken so, dass die lange Kastenseite (``long_cols`` Flaschen) auf
    TL→TR (horizontal) und die kurze (``short_rows``) auf TL→BL (vertikal) liegt.

    Statische Regel: Kantenlängen vergleichen, bei Bedarf 90° im Uhrzeigersinn rotieren.
    """
    _ = long_cols, short_rows  # Dokumentation; physisches Verhältnis ~5:4
    corners = np.asarray(corners_tl_tr_br_bl, dtype=np.float32).reshape(4, 2).copy()
    top, right, bottom, left = _edge_lengths(corners)

    # Mittlere Länge je Orientierung (robuster als nur top/right bei Trapezen).
    horiz = 0.5 * (top + bottom)
    vert = 0.5 * (right + left)

    if horiz + 1e-3 < vert:
        # Kurze Seite liegt horizontal → 90° CW: TL←BL, TR←TL, BR←TR, BL←BR
        tl, tr, br, bl = corners
        corners = np.stack([bl, tl, tr, br], axis=0)

    return corners.astype(np.float32)


def warp_crate_top_down(
    image_bgr: np.ndarray,
    corners_tl_tr_br_bl: np.ndarray,
    out_width: int = 500,
    out_height: int = 400,
    *,
    canonicalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Entzerrt die durch die vier Eckpunkte aufgespannte Fläche in eine Norm-Ansicht.

    Returns:
        warped BGR-Bild, 3×3 Homographie ``H`` (wie ``cv2.getPerspectiveTransform``).
    """
    src = np.asarray(corners_tl_tr_br_bl, dtype=np.float32).reshape(4, 2)
    if canonicalize:
        src = canonicalize_corners_for_grid(src)
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


# Physische Kalibrierung Paulaner-Maßkiste 4×5 (cm).
# Messung: Flaschenmitte unten-rechts → x=6 cm zum Innenrand rechts,
#           y=4,5 cm zum Innenrand unten (symmetrisch für links/oben).
_INNER_LONG_CM = 38.5  # 5 Flaschen (Innenfläche Raster)
_INNER_SHORT_CM = 29.0  # 4 Flaschen
_CENTER_TO_INNER_EDGE_LONG_CM = 6.0
_CENTER_TO_INNER_EDGE_SHORT_CM = 4.5
_OUTER_LONG_CM = 40.0  # Außenkante ≈ YOLO-/Warp-Rand (5-Flaschen-Seite)
_OUTER_SHORT_CM = 30.0  # Außenkante (4-Flaschen-Seite)


def _slot_center_positions_cm(
    rows: int,
    cols: int,
    *,
    inner_long: float = _INNER_LONG_CM,
    inner_short: float = _INNER_SHORT_CM,
    offset_long: float = _CENTER_TO_INNER_EDGE_LONG_CM,
    offset_short: float = _CENTER_TO_INNER_EDGE_SHORT_CM,
) -> tuple[list[float], list[float]]:
    """Flaschenmittelpunkte in cm, Ursprung = oben-links Innen-Raster."""
    pitch_x = (inner_long - 2.0 * offset_long) / max(cols - 1, 1)
    pitch_y = (inner_short - 2.0 * offset_short) / max(rows - 1, 1)
    xs = [offset_long + c * pitch_x for c in range(cols)]
    ys = [offset_short + r * pitch_y for r in range(rows)]
    return xs, ys


def grid_slot_centers(
    warped_width: int,
    warped_height: int,
    rows: int = 4,
    cols: int = 5,
) -> np.ndarray:
    """
    Mittelpunkte der Slots im entzerrten Bild (Paulaner 4×5).

    Statisch kalibriert aus Innenmaßen + deiner Eck-Messung (6 cm / 4,5 cm).
    Reihenfolge: Zeile für Zeile von oben nach unten, links nach rechts → 20 Punkte.
    """
    rim_x = 0.5 * (_OUTER_LONG_CM - _INNER_LONG_CM)
    rim_y = 0.5 * (_OUTER_SHORT_CM - _INNER_SHORT_CM)
    xs_cm, ys_cm = _slot_center_positions_cm(rows, cols)

    centers = np.zeros((rows * cols, 2), dtype=np.float32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            centers[k, 0] = (rim_x + xs_cm[c]) / _OUTER_LONG_CM * float(warped_width)
            centers[k, 1] = (rim_y + ys_cm[r]) / _OUTER_SHORT_CM * float(warped_height)
            k += 1
    return centers


def estimate_slot_half_size(centers: np.ndarray, rows: int, cols: int) -> int:
    """ROI-Halbbreite aus dem Abstand benachbarter Slot-Mittelpunkte."""
    if len(centers) < 2:
        return 32
    xs = centers[:, 0].reshape(rows, cols)
    ys = centers[:, 1].reshape(rows, cols)
    dx = float(np.median(np.abs(np.diff(xs, axis=1)))) if cols > 1 else 50.0
    dy = float(np.median(np.abs(np.diff(ys, axis=0)))) if rows > 1 else 50.0
    return max(8, int(min(dx, dy) * 0.40))


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
