"""
Stufe 1 — Slot-Belegung: Ist an einem Slot eine Flasche vorhanden oder ist er leer?

Zwei Wege, vergleichbar umschaltbar:

- ``classical``: OpenCV-Heuristik (Kantendichte + Hough-Kreis der Flaschenöffnung).
- ``ml``: kleines CNN/MobileNetV2 (Gewichte via ``KASTEN_SLOT_WEIGHTS`` oder ``weights_path``).
- ``auto``: ML, wenn Gewichte gefunden werden, sonst klassisch.

Eingabe je Slot ist ein quadratischer ROI aus der entzerrten Draufsicht
(``extract_slot_roi``). Rückgabe: ``(occupied: bool, confidence: float)``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from kastendetektion.inference_config import get_slot_occupied_threshold

logger = logging.getLogger(__name__)

Method = Literal["classical", "ml", "auto"]

# Reihenfolge der Modellklassen; positive Klasse = "occupied".
SLOT_CLASSES = ("empty", "occupied")
_POSITIVE_CLASS = "occupied"


def classify_slot_classical(
    roi_bgr: np.ndarray,
    *,
    edge_density_thresh: float = 0.06,
    canny_low: int = 50,
    canny_high: int = 150,
) -> tuple[bool, float]:
    """
    Klassische Slot-Belegung: belegte Slots zeigen eine Flaschenöffnung/Kronkorken
    mit deutlich mehr Kanten und oft einem zentralen Kreis als ein leerer Slot
    (gleichmäßiger Kastenboden).

    Returns ``(occupied, confidence)``.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    h, w = gray.shape[:2]
    min_dim = min(h, w)
    circle_found = False
    if min_dim >= 16:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim,
            param1=120,
            param2=18,
            minRadius=int(min_dim * 0.15),
            maxRadius=int(min_dim * 0.55),
        )
        circle_found = circles is not None

    # Konfidenz aus Kantendichte relativ zum Schwellwert (auf [0,1] geklemmt).
    conf = min(1.0, edge_density / max(1e-6, edge_density_thresh * 2.0))
    occupied = edge_density >= edge_density_thresh or circle_found
    if circle_found:
        conf = max(conf, 0.6)
    return bool(occupied), float(conf)


def _resolve_weights(explicit: str | Path | None, env_var: str, defaults: tuple[str, ...]) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    env = os.environ.get(env_var, "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    root = Path(__file__).resolve().parents[1]
    for rel in defaults:
        p = root / rel
        if p.is_file():
            return p
    return None


class _TorchRoiClassifier:
    """Dünner Inferenz-Wrapper um ein ROI-Klassifikationsmodell (lazy torch)."""

    def __init__(
        self,
        weights: Path,
        positive_class: str,
        device: str = "cpu",
        *,
        pos_threshold: float = 0.5,
    ) -> None:
        from kastendetektion.ml_models import load_checkpoint

        self.model, self.classes, self.input_size = load_checkpoint(str(weights), device=device)
        self.device = device
        self.pos_threshold = pos_threshold
        if positive_class in self.classes:
            self._pos_idx = self.classes.index(positive_class)
        else:
            # Fallback: zweite Klasse gilt als "positiv".
            self._pos_idx = min(1, len(self.classes) - 1)

    def predict(self, rois_bgr: list[np.ndarray]) -> list[tuple[bool, float]]:
        import torch

        out: list[tuple[bool, float]] = []
        valid_idx: list[int] = []
        batch: list[np.ndarray] = []
        for i, roi in enumerate(rois_bgr):
            if roi is None or roi.size == 0:
                continue
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.input_size, self.input_size))
            batch.append(rgb.astype(np.float32) / 255.0)
            valid_idx.append(i)

        results: dict[int, tuple[bool, float]] = {}
        if batch:
            arr = np.stack(batch).transpose(0, 3, 1, 2)  # NCHW
            with torch.no_grad():
                logits = self.model(torch.from_numpy(arr).to(self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            for k, i in enumerate(valid_idx):
                p_pos = float(probs[k, self._pos_idx])
                results[i] = (p_pos >= self.pos_threshold, p_pos)

        for i in range(len(rois_bgr)):
            out.append(results.get(i, (False, 0.0)))
        return out


class SlotOccupancyClassifier:
    """Einheitliche Stufe-1-Schnittstelle (klassisch / ML / auto)."""

    def __init__(
        self,
        method: Method = "auto",
        *,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        occupied_threshold: float | None = None,
    ) -> None:
        self.method: Method = method
        self._ml: _TorchRoiClassifier | None = None
        self.active_method: Method = "classical"
        occ_thr = occupied_threshold if occupied_threshold is not None else get_slot_occupied_threshold()

        weights = _resolve_weights(
            weights_path,
            "KASTEN_SLOT_WEIGHTS",
            ("runs/classify/slot/weights/best.pt", "models/slot_cnn.pt"),
        )
        want_ml = method in ("ml", "auto")
        if want_ml and weights is not None:
            try:
                self._ml = _TorchRoiClassifier(
                    weights, _POSITIVE_CLASS, device=device, pos_threshold=occ_thr
                )
                self.active_method = "ml"
            except Exception:
                logger.exception("Slot-ML-Modell konnte nicht geladen werden, nutze klassisch.")
                self._ml = None
        elif method == "ml" and weights is None:
            logger.warning("method='ml' gewählt, aber keine Slot-Gewichte gefunden -> klassisch.")

    def predict(self, rois_bgr: list[np.ndarray]) -> list[tuple[bool, float]]:
        """Pro ROI ``(occupied, confidence)``."""
        if self._ml is not None:
            return self._ml.predict(rois_bgr)
        return [classify_slot_classical(roi) if roi is not None else (False, 0.0) for roi in rois_bgr]
