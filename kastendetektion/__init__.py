"""Bierkasten-Erkennung: YOLOv8 + klassischer Kanten-Fallback (Arbeitspaket Kastendetektion)."""

from kastendetektion.detector import CrateDetectionResult, detect_crate
from kastendetektion.warp_grid import (
    extract_slot_roi,
    grid_slot_centers,
    map_points_to_original,
    warp_crate_top_down,
)

__all__ = [
    "CrateDetectionResult",
    "detect_crate",
    "warp_crate_top_down",
    "grid_slot_centers",
    "map_points_to_original",
    "extract_slot_roi",
]
