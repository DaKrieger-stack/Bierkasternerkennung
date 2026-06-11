"""Bierkasten-Erkennung: YOLOv8 + klassischer Kanten-Fallback plus Füllzustand-Pipeline."""

from kastendetektion.cap_classifier import CapClassifier, classify_cap_classical
from kastendetektion.detector import (
    CrateDetectionResult,
    detect_crate,
    detect_crate_contour,
    draw_detection,
)
from kastendetektion.overlay import draw_pipeline_overlay
from kastendetektion.pipeline import CrateAnalysis, CratePipeline, SlotResult, analyze_frame
from kastendetektion.slot_classifier import (
    SlotOccupancyClassifier,
    classify_slot_classical,
)
from kastendetektion.stabilize import SlotStateStabilizer
from kastendetektion.states import SlotState, color_for, label_for
from kastendetektion.warp_grid import (
    canonicalize_corners_for_grid,
    extract_slot_roi,
    grid_slot_centers,
    map_points_to_original,
    warp_crate_top_down,
)

__all__ = [
    # Detektion
    "CrateDetectionResult",
    "detect_crate",
    "detect_crate_contour",
    "draw_detection",
    # Warp / Grid
    "canonicalize_corners_for_grid",
    "warp_crate_top_down",
    "grid_slot_centers",
    "map_points_to_original",
    "extract_slot_roi",
    # Klassifikatoren
    "SlotOccupancyClassifier",
    "classify_slot_classical",
    "CapClassifier",
    "classify_cap_classical",
    # Pipeline / Zustände
    "CratePipeline",
    "CrateAnalysis",
    "SlotResult",
    "analyze_frame",
    "SlotState",
    "color_for",
    "label_for",
    "draw_pipeline_overlay",
    "SlotStateStabilizer",
]
