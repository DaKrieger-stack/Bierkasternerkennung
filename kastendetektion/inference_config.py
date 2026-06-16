"""
Inferenz-Schwellwerte für State- und Slot-Klassifikator.

ORIGINAL: Stand vor dem Tuning (2026-06-16).
Aktiv:   verschärfte Werte — leere Slots sollen nicht mehr als „voll“ enden.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StateInferenceParams:
    """Entscheidungsregeln in ``StateClassifier._state_from_probs`` (mit slot_gate)."""

    p_empty_nada: float = 0.45
    full_min: float = 0.50
    empty_min: float = 0.40
    empty_vs_full_margin: float = 0.10
    p_empty_strong: float = 0.50
    slot_low_conf: float = 0.78
    p_empty_soft: float = 0.25
    # Unsicher → nada statt voll (kein p_bf >= p_be-Fallback mehr).
    default_missing: bool = True


# Ursprüngliche Werte (inkl. Fallback: p_bf >= p_be → voll).
ORIGINAL = StateInferenceParams(
    p_empty_nada=0.45,
    full_min=0.50,
    empty_min=0.40,
    empty_vs_full_margin=0.0,
    p_empty_strong=0.50,
    slot_low_conf=0.78,
    p_empty_soft=0.25,
    default_missing=False,
)

# Getunt: höhere voll-Schwelle, nada bevorzugt bei Unsicherheit.
ACTIVE = StateInferenceParams(
    p_empty_nada=0.32,
    full_min=0.58,
    empty_min=0.45,
    empty_vs_full_margin=0.12,
    p_empty_strong=0.38,
    slot_low_conf=0.68,
    p_empty_soft=0.15,
    default_missing=True,
)

SLOT_OCCUPIED_THRESHOLD_ORIGINAL = 0.50
SLOT_OCCUPIED_THRESHOLD_ACTIVE = float(os.environ.get("KASTEN_SLOT_OCC_THRESHOLD", "0.58"))


def use_original() -> bool:
    return os.environ.get("KASTEN_INFERENCE_ORIGINAL", "").strip().lower() in ("1", "true", "yes")


def get_state_params() -> StateInferenceParams:
    return ORIGINAL if use_original() else ACTIVE


def get_slot_occupied_threshold() -> float:
    if use_original():
        return SLOT_OCCUPIED_THRESHOLD_ORIGINAL
    return SLOT_OCCUPIED_THRESHOLD_ACTIVE
