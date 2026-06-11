"""
End-to-End-Pipeline: Frame -> Kastendetektion -> Entzerrung -> 4×5-Grid ->
Stufe 1 (Slot belegt?) -> Stufe 2 (Kronkorken?) -> Zustände je Slot + Statistik.

Nutzt die bestehenden Bausteine aus ``detector.py`` und ``warp_grid.py`` und
kombiniert sie mit den Klassifikatoren aus ``slot_classifier.py`` / ``cap_classifier.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kastendetektion.cap_classifier import CapClassifier
from kastendetektion.cap_classifier import Method as CapMethod
from kastendetektion.detector import CrateDetectionResult, detect_crate
from kastendetektion.slot_classifier import Method as SlotMethod
from kastendetektion.slot_classifier import SlotOccupancyClassifier
from kastendetektion.states import SlotState
from kastendetektion.warp_grid import (
    estimate_slot_half_size,
    extract_slot_roi,
    grid_slot_centers,
    map_points_to_original,
    warp_crate_top_down,
)


@dataclass
class SlotResult:
    """Ergebnis für einen einzelnen Slot."""

    index: int
    row: int
    col: int
    center_warped: tuple[float, float]
    center_orig: tuple[float, float]
    state: SlotState
    slot_conf: float = 0.0
    cap_conf: float = 0.0


@dataclass
class CrateAnalysis:
    """Gesamtergebnis eines Frames."""

    detection: CrateDetectionResult
    warped: np.ndarray
    homography: np.ndarray
    slots: list[SlotResult] = field(default_factory=list)
    rows: int = 4
    cols: int = 5

    @property
    def occupied_count(self) -> int:
        return sum(1 for s in self.slots if s.state is not SlotState.MISSING)

    @property
    def full_count(self) -> int:
        return sum(1 for s in self.slots if s.state is SlotState.FULL)

    @property
    def empty_count(self) -> int:
        return sum(1 for s in self.slots if s.state is SlotState.EMPTY)

    @property
    def missing_count(self) -> int:
        return sum(1 for s in self.slots if s.state is SlotState.MISSING)

    @property
    def total(self) -> int:
        return len(self.slots)


class CratePipeline:
    """
    Hält Detektor-Konfiguration und (einmal geladene) Klassifikatoren,
    damit Video/Streams Modelle nicht pro Frame neu laden.
    """

    def __init__(
        self,
        *,
        slot_method: SlotMethod = "auto",
        cap_method: CapMethod = "auto",
        rows: int = 4,
        cols: int = 5,
        out_width: int = 500,
        out_height: int = 400,
        weights_path: str | Path | None = None,
        prefer_yolo: bool = True,
        conf: float = 0.25,
        slot_weights: str | Path | None = None,
        cap_weights: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.out_width = out_width
        self.out_height = out_height
        self.weights_path = weights_path
        self.prefer_yolo = prefer_yolo
        self.conf = conf
        self.slot_clf = SlotOccupancyClassifier(slot_method, weights_path=slot_weights, device=device)
        self.cap_clf = CapClassifier(cap_method, weights_path=cap_weights, device=device)

    def analyze(self, frame_bgr: np.ndarray) -> CrateAnalysis | None:
        det = detect_crate(
            frame_bgr,
            weights_path=self.weights_path,
            prefer_yolo=self.prefer_yolo,
            conf=self.conf,
        )
        if det is None:
            return None

        warped, h_mat = warp_crate_top_down(
            frame_bgr, det.corners, out_width=self.out_width, out_height=self.out_height
        )
        centers = grid_slot_centers(
            self.out_width,
            self.out_height,
            rows=self.rows,
            cols=self.cols,
        )
        half = estimate_slot_half_size(centers, self.rows, self.cols)

        rois: list[np.ndarray | None] = [
            extract_slot_roi(warped, float(cx), float(cy), half_size=half) for cx, cy in centers
        ]

        # Stufe 1: Belegung je Slot.
        slot_preds = self.slot_clf.predict([r if r is not None else np.zeros((1, 1, 3), np.uint8) for r in rois])

        # Stufe 2: Kronkorken nur für belegte Slots (für leere/None: neutral).
        cap_input: list[np.ndarray] = []
        cap_map: list[int] = []  # slot-index je cap_input-Eintrag
        for i, (roi, (occ, _)) in enumerate(zip(rois, slot_preds)):
            if occ and roi is not None:
                cap_input.append(roi)
                cap_map.append(i)
        cap_preds_raw = self.cap_clf.predict(cap_input) if cap_input else []
        cap_by_slot: dict[int, tuple[bool, float]] = {cap_map[k]: cap_preds_raw[k] for k in range(len(cap_map))}

        # Rückprojektion aller Mittelpunkte ins Originalbild.
        centers_orig = map_points_to_original(h_mat, centers)

        slots: list[SlotResult] = []
        for i, (cx, cy) in enumerate(centers):
            row, col = divmod(i, self.cols)
            occupied, slot_conf = slot_preds[i]
            if not occupied:
                state = SlotState.MISSING
                cap_conf = 0.0
            else:
                has_cap, cap_conf = cap_by_slot.get(i, (False, 0.0))
                state = SlotState.FULL if has_cap else SlotState.EMPTY
            ox, oy = centers_orig[i]
            slots.append(
                SlotResult(
                    index=i,
                    row=row,
                    col=col,
                    center_warped=(float(cx), float(cy)),
                    center_orig=(float(ox), float(oy)),
                    state=state,
                    slot_conf=float(slot_conf),
                    cap_conf=float(cap_conf),
                )
            )

        return CrateAnalysis(
            detection=det,
            warped=warped,
            homography=h_mat,
            slots=slots,
            rows=self.rows,
            cols=self.cols,
        )


def analyze_frame(
    frame_bgr: np.ndarray,
    *,
    slot_method: SlotMethod = "auto",
    cap_method: CapMethod = "auto",
    weights_path: str | Path | None = None,
    prefer_yolo: bool = True,
    conf: float = 0.25,
    rows: int = 4,
    cols: int = 5,
) -> CrateAnalysis | None:
    """Bequeme Einzelaufruf-Variante (baut die Pipeline intern; für Streams ``CratePipeline`` nutzen)."""
    pipe = CratePipeline(
        slot_method=slot_method,
        cap_method=cap_method,
        rows=rows,
        cols=cols,
        weights_path=weights_path,
        prefer_yolo=prefer_yolo,
        conf=conf,
    )
    return pipe.analyze(frame_bgr)
