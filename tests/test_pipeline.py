"""Tests für Grid, Pipeline, Klassifikatoren und Stabilisierung (klassischer Pfad)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.cap_classifier import classify_cap_classical
from kastendetektion.pipeline import analyze_frame
from kastendetektion.slot_classifier import classify_slot_classical
from kastendetektion.stabilize import SlotStateStabilizer
from kastendetektion.states import SlotState
from kastendetektion.warp_grid import grid_slot_centers

SYNTH = ROOT / "demo_output" / "synthetic_crate.png"


def test_grid_has_20_points():
    centers = grid_slot_centers(500, 400, rows=4, cols=5)
    assert centers.shape == (20, 2)
    assert (centers[:, 0] > 0).all() and (centers[:, 0] < 500).all()
    assert (centers[:, 1] > 0).all() and (centers[:, 1] < 400).all()


@pytest.mark.skipif(not SYNTH.is_file(), reason="synthetisches Testbild fehlt")
def test_analyze_frame_returns_20_slots():
    import cv2

    frame = cv2.imread(str(SYNTH))
    assert frame is not None
    analysis = analyze_frame(frame, slot_method="classical", cap_method="classical")
    assert analysis is not None
    assert analysis.total == 20
    assert len(analysis.slots) == 20
    for s in analysis.slots:
        assert isinstance(s.state, SlotState)
    # Zählungen sind konsistent.
    assert analysis.full_count + analysis.empty_count + analysis.missing_count == 20
    assert analysis.occupied_count == analysis.full_count + analysis.empty_count


def test_slot_classifier_returns_bool_conf():
    # Leerer (uniformer) ROI -> nicht belegt; texturierter ROI -> belegt.
    flat = np.full((64, 64, 3), 50, np.uint8)
    occ_flat, conf_flat = classify_slot_classical(flat)
    assert isinstance(occ_flat, bool) and 0.0 <= conf_flat <= 1.0
    assert occ_flat is False

    noisy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    occ_noisy, conf_noisy = classify_slot_classical(noisy)
    assert isinstance(occ_noisy, bool) and 0.0 <= conf_noisy <= 1.0
    assert occ_noisy is True


def test_cap_classifier_returns_bool_conf():
    roi = np.full((64, 64, 3), 50, np.uint8)
    has_cap, conf = classify_cap_classical(roi)
    assert isinstance(has_cap, bool)
    assert 0.0 <= conf <= 1.0


def test_stabilizer_majority_vote():
    stab = SlotStateStabilizer(window=5)
    seq = [
        [SlotState.FULL, SlotState.MISSING],
        [SlotState.FULL, SlotState.MISSING],
        [SlotState.EMPTY, SlotState.MISSING],  # Ausreißer in Slot 0
        [SlotState.FULL, SlotState.EMPTY],     # Ausreißer in Slot 1
        [SlotState.FULL, SlotState.MISSING],
    ]
    out = None
    for frame_states in seq:
        out = stab.update(frame_states)
    assert out == [SlotState.FULL, SlotState.MISSING]
