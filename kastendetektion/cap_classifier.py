"""
Stufe 2 — Kronkorken: Hat eine vorhandene Flasche einen Kronkorken (= voll) oder nicht (= leer)?

Wird nur auf Slots angewendet, die Stufe 1 als belegt erkannt hat.

- ``classical``: HSV/Helligkeit + Kreisform + Specular-Highlight-Heuristik.
  Ein Kronkorken ist eine farbige/metallische Kreisfläche mit Glanzlichtern;
  eine offene Flasche zeigt eher ein dunkles Loch (Flaschenhals).
- ``ml``: kleines CNN/MobileNetV2 (Gewichte via ``KASTEN_CAP_WEIGHTS`` oder ``weights_path``).
- ``auto``: ML, wenn Gewichte vorhanden, sonst klassisch.

Rückgabe je ROI: ``(has_cap: bool, confidence: float)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from kastendetektion.slot_classifier import _TorchRoiClassifier, _resolve_weights

logger = logging.getLogger(__name__)

Method = Literal["classical", "ml", "auto"]

CAP_CLASSES = ("bottle_empty", "bottle_full")
_POSITIVE_CLASS = "bottle_full"


def classify_cap_classical(
    roi_bgr: np.ndarray,
    *,
    sat_thresh: float = 60.0,
    specular_val_thresh: int = 220,
    specular_ratio_thresh: float = 0.01,
) -> tuple[bool, float]:
    """
    Klassische Kronkorken-Erkennung im zentralen Bereich des Slot-ROIs.

    Heuristik (kombiniert):
    - mittlere Sättigung im Zentrum (farbiger Kronkorken vs. dunkles Loch),
    - Anteil heller Specular-Pixel (Glanzlicht auf Metall),
    - vorhandener zentraler Kreis (Hough).

    Returns ``(has_cap, confidence)``.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False, 0.0

    h, w = roi_bgr.shape[:2]
    # Zentralregion (Kronkorken sitzt mittig im Slot).
    cy0, cy1 = int(h * 0.2), int(h * 0.8)
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    center = roi_bgr[cy0:cy1, cx0:cx1]
    if center.size == 0:
        center = roi_bgr

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    val = hsv[:, :, 2]
    specular_ratio = float(np.count_nonzero(val >= specular_val_thresh)) / float(val.size)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    min_dim = min(h, w)
    circle_found = False
    if min_dim >= 16:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim,
            param1=120,
            param2=20,
            minRadius=int(min_dim * 0.2),
            maxRadius=int(min_dim * 0.5),
        )
        circle_found = circles is not None

    # Einzelne Indizien zu einem Score [0,1] verrechnen.
    sat_score = min(1.0, mean_sat / max(1e-6, sat_thresh * 2.0))
    spec_score = min(1.0, specular_ratio / max(1e-6, specular_ratio_thresh * 2.0))
    circle_score = 0.5 if circle_found else 0.0
    conf = float(np.clip(0.5 * sat_score + 0.3 * spec_score + circle_score * 0.4, 0.0, 1.0))

    has_cap = (mean_sat >= sat_thresh or specular_ratio >= specular_ratio_thresh) and circle_found
    if not circle_found:
        # Ohne Kreis nur bei sehr deutlicher Farb-/Glanz-Evidenz.
        has_cap = mean_sat >= sat_thresh * 1.5
    return bool(has_cap), conf


class CapClassifier:
    """Einheitliche Stufe-2-Schnittstelle (klassisch / ML / auto)."""

    def __init__(
        self,
        method: Method = "auto",
        *,
        weights_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.method: Method = method
        self._ml: _TorchRoiClassifier | None = None
        self.active_method: Method = "classical"

        weights = _resolve_weights(
            weights_path,
            "KASTEN_CAP_WEIGHTS",
            ("runs/classify/cap/weights/best.pt", "models/cap_cnn.pt"),
        )
        want_ml = method in ("ml", "auto")
        if want_ml and weights is not None:
            try:
                # Höhere Schwelle: „voll“ nur bei sicherer Evidenz → weniger leere als voll.
                self._ml = _TorchRoiClassifier(
                    weights, _POSITIVE_CLASS, device=device, pos_threshold=0.58
                )
                self.active_method = "ml"
            except Exception:
                logger.exception("Cap-ML-Modell konnte nicht geladen werden, nutze klassisch.")
                self._ml = None
        elif method == "ml" and weights is None:
            logger.warning("method='ml' gewählt, aber keine Cap-Gewichte gefunden -> klassisch.")

    def predict(self, rois_bgr: list[np.ndarray]) -> list[tuple[bool, float]]:
        """Pro ROI ``(has_cap, confidence)``."""
        if self._ml is not None:
            return self._ml.predict(rois_bgr)
        return [classify_cap_classical(roi) if roi is not None else (False, 0.0) for roi in rois_bgr]
