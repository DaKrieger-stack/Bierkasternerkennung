"""
3-Klassen-Klassifikator: nada / leer / voll in einem Schritt.

Training auf ``empty`` / ``bottle_empty`` / ``bottle_full``.
Inferenz: Slot-CNN entscheidet nada; State-CNN leer/voll auf belegten Slots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from kastendetektion.inference_config import StateInferenceParams, get_slot_occupied_threshold, get_state_params
from kastendetektion.slot_classifier import _resolve_weights
from kastendetektion.states import SlotState

logger = logging.getLogger(__name__)

Method = Literal["classical", "ml", "auto"]

STATE_CLASSES = ("empty", "bottle_empty", "bottle_full")
_CLASS_TO_STATE: dict[str, SlotState] = {
    "empty": SlotState.MISSING,
    "bottle_empty": SlotState.EMPTY,
    "bottle_full": SlotState.FULL,
}


class StateClassifier:
    """ML-Klassifikator voll/leer/nada pro Slot-ROI."""

    def __init__(
        self,
        method: Method = "auto",
        *,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        inference_params: StateInferenceParams | None = None,
        slot_occupied_threshold: float | None = None,
    ) -> None:
        self.method: Method = method
        self._model = None
        self._classes: list[str] = []
        self._input_size = 64
        self._device = device
        self.active_method: Method = "classical"
        self.inference_params = inference_params or get_state_params()
        self.slot_occupied_threshold = (
            slot_occupied_threshold
            if slot_occupied_threshold is not None
            else get_slot_occupied_threshold()
        )

        weights = _resolve_weights(
            weights_path,
            "KASTEN_STATE_WEIGHTS",
            ("models/state_cnn.pt",),
        )
        want_ml = method in ("ml", "auto")
        if want_ml and weights is not None:
            try:
                from kastendetektion.ml_models import load_checkpoint

                self._model, self._classes, self._input_size = load_checkpoint(str(weights), device=device)
                self.active_method = "ml"
            except Exception:
                logger.exception("State-ML-Modell konnte nicht geladen werden.")
                self._model = None
        elif method == "ml" and weights is None:
            logger.warning("method='ml' gewählt, aber keine State-Gewichte gefunden.")

    def _class_indices(self) -> dict[str, int]:
        return {
            name: self._classes.index(name) if name in self._classes else STATE_CLASSES.index(name)
            for name in STATE_CLASSES
        }

    def _predict_probs(self, rois_bgr: list[np.ndarray]) -> list[np.ndarray | None]:
        if self._model is None:
            return [None] * len(rois_bgr)

        import torch

        valid_idx: list[int] = []
        batch: list[np.ndarray] = []
        for i, roi in enumerate(rois_bgr):
            if roi is None or roi.size == 0:
                continue
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self._input_size, self._input_size))
            batch.append(rgb.astype(np.float32) / 255.0)
            valid_idx.append(i)

        out: list[np.ndarray | None] = [None] * len(rois_bgr)
        if batch:
            arr = np.stack(batch).transpose(0, 3, 1, 2)
            with torch.no_grad():
                logits = self._model(torch.from_numpy(arr).to(self._device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            for k, i in enumerate(valid_idx):
                out[i] = probs[k]
        return out

    def _apply_slot_gate(self, occupied: bool, slot_conf: float) -> tuple[bool, float]:
        thr = self.slot_occupied_threshold
        if thr is not None and occupied and slot_conf < thr:
            return False, 1.0 - slot_conf
        return occupied, slot_conf

    def _state_from_probs(
        self,
        probs: np.ndarray,
        *,
        slot_gate: tuple[bool, float] | None = None,
    ) -> tuple[SlotState, float]:
        p = self.inference_params
        idx = self._class_indices()
        p_empty = float(probs[idx["empty"]])
        p_be = float(probs[idx["bottle_empty"]])
        p_bf = float(probs[idx["bottle_full"]])

        if slot_gate is not None:
            occupied, slot_conf = slot_gate
            occupied, slot_conf = self._apply_slot_gate(occupied, slot_conf)
            if not occupied:
                return SlotState.MISSING, max(slot_conf, p_empty)
            if p_empty >= p.p_empty_nada and p_empty >= p_be and p_empty >= p_bf:
                return SlotState.MISSING, p_empty
            if (
                p_bf >= p.full_min
                and p_bf > p_be
                and p_bf > p_empty + p.empty_vs_full_margin
            ):
                return SlotState.FULL, p_bf
            if p_be >= p.empty_min and p_be > p_bf and p_be > p_empty + 0.08:
                return SlotState.EMPTY, p_be
            if p_empty >= p.p_empty_strong and p_empty >= p_be and p_empty >= p_bf:
                return SlotState.MISSING, p_empty
            if slot_conf < p.slot_low_conf and p_empty >= p.p_empty_soft:
                return SlotState.MISSING, p_empty
            if not p.default_missing:
                # Ursprüngliches Verhalten: unsicher → voll wenn p_bf >= p_be.
                if p_bf >= p_be:
                    return SlotState.FULL, p_bf
                return SlotState.EMPTY, p_be
            return SlotState.MISSING, max(p_empty, 1.0 - max(p_be, p_bf))

        cls_idx = int(probs.argmax())
        cls_name = self._classes[cls_idx] if cls_idx < len(self._classes) else STATE_CLASSES[0]
        conf = float(probs[cls_idx])
        return _CLASS_TO_STATE.get(cls_name, SlotState.MISSING), conf

    def predict(
        self,
        rois_bgr: list[np.ndarray],
        *,
        slot_gates: list[tuple[bool, float]] | None = None,
    ) -> list[tuple[SlotState, float]]:
        probs_list = self._predict_probs(rois_bgr)
        out: list[tuple[SlotState, float]] = []
        for i, probs in enumerate(probs_list):
            if probs is None:
                out.append((SlotState.MISSING, 0.0))
                continue
            gate = slot_gates[i] if slot_gates is not None and i < len(slot_gates) else None
            out.append(self._state_from_probs(probs, slot_gate=gate))
        return out
