"""
Zentrale Definition der Slot-Zustände und ihrer Farbcodierung.

Eine einzige Quelle der Wahrheit für Pipeline, Overlay und Stabilisierung.
Legende laut Präsentation (Zwischenstand):

- FULL    = Flasche mit Kronkorken  -> grün
- EMPTY   = Flasche ohne Kronkorken -> gelb
- MISSING = kein Slot belegt        -> rot
"""

from __future__ import annotations

from enum import Enum


class SlotState(Enum):
    """Zustand eines einzelnen Slots im 4×5-Raster."""

    MISSING = "missing"  # kein Slot belegt
    EMPTY = "empty"      # Flasche ohne Kronkorken
    FULL = "full"        # Flasche mit Kronkorken


# BGR-Farben (OpenCV) je Zustand.
STATE_COLORS_BGR: dict[SlotState, tuple[int, int, int]] = {
    SlotState.FULL: (0, 200, 0),      # grün
    SlotState.EMPTY: (0, 215, 255),   # gelb
    SlotState.MISSING: (0, 0, 255),   # rot
}

# Kurzlabels fürs Overlay.
STATE_LABELS: dict[SlotState, str] = {
    SlotState.FULL: "voll",
    SlotState.EMPTY: "leer",
    SlotState.MISSING: "fehlt",
}


def color_for(state: SlotState) -> tuple[int, int, int]:
    """BGR-Farbe für einen Zustand (Default Weiß bei unbekanntem Zustand)."""
    return STATE_COLORS_BGR.get(state, (255, 255, 255))


def label_for(state: SlotState) -> str:
    """Kurzlabel für einen Zustand."""
    return STATE_LABELS.get(state, "?")
