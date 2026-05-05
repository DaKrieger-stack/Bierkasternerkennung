"""
Schnittstelle für das nächste Modul (Warp Perspective / Grid Mapping):

    detect_crate(frame) -> CrateDetectionResult | None

- Primär: Ultralytics YOLOv8 (nach Fine-Tuning auf Klasse ``bierkasten``).
- Fallback: Canny + Konturen, größtes plausibles Viereck (minAreaRect).

``corners`` Reihenfolge: oben-links, oben-rechts, unten-rechts, unten-links (float32, Shape (4, 2)).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ein geladenes YOLO-Modell pro Gewichtsdatei (wichtig für Video/Streams).
_YOLO_MODELS: dict[str, Any] = {}

Source = Literal["yolo", "contour"]


@dataclass
class CrateDetectionResult:
    """Achsparallele Bounding Box + vier Eckpunkte (wie gefordert für Grid Mapping)."""

    x: int
    y: int
    w: int
    h: int
    corners: np.ndarray  # (4, 2) float32 — TL, TR, BR, BL
    confidence: float
    source: Source
    orientation_deg: float | None = None  # OpenCV minAreaRect-Winkel; bei source=="contour" gesetzt


def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Sortiert vier Punkte in TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def _aabb_from_corners(corners: np.ndarray) -> tuple[int, int, int, int]:
    xs = corners[:, 0]
    ys = corners[:, 1]
    x1, y1 = int(np.floor(xs.min())), int(np.floor(ys.min()))
    x2, y2 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def detect_crate_contour(
    frame_bgr: np.ndarray,
    *,
    canny_low: int = 40,
    canny_high: int = 120,
    min_area_ratio: float = 0.03,
    max_area_ratio: float = 0.92,
) -> CrateDetectionResult | None:
    """Klassischer Fallback: Canny + findContours, größtes plausibles Rechteck (minAreaRect)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    frame_area = float(h * w)
    min_area = min_area_ratio * frame_area
    max_area = max_area_ratio * frame_area

    best: tuple[float, np.ndarray, float] | None = None  # (quad_area, box_points, minAreaRect_angle)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        angle = float(rect[2])
        box = cv2.boxPoints(rect)
        box = box.astype(np.float32)
        qw = cv2.contourArea(box)
        if qw <= 0:
            continue
        if best is None or qw > best[0]:
            best = (qw, box, angle)

    if best is None:
        return None

    corners = _order_corners_tl_tr_br_bl(best[1])
    x, y, bw, bh = _aabb_from_corners(corners)

    # Clamp to image
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))

    return CrateDetectionResult(
        x=x,
        y=y,
        w=bw,
        h=bh,
        corners=corners,
        confidence=float(min(1.0, best[0] / max_area)),
        source="contour",
        orientation_deg=best[2],
    )


def _resolve_weights(explicit: str | Path | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None

    env = os.environ.get("KASTEN_YOLO_WEIGHTS", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p

    for rel in (
        "runs/detect/kasten/weights/best.pt",
        "runs/detect/train/weights/best.pt",
    ):
        p = PROJECT_ROOT / rel
        if p.is_file():
            return p
    return None


def _get_yolo(weights: Path) -> Any:
    key = str(weights.resolve())
    if key not in _YOLO_MODELS:
        from ultralytics import YOLO

        _YOLO_MODELS[key] = YOLO(key)
    return _YOLO_MODELS[key]


def _detect_yolo(
    frame_bgr: np.ndarray,
    weights: Path,
    conf: float,
    iou: float,
) -> CrateDetectionResult | None:
    model = _get_yolo(weights)
    results = model.predict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    if not results:
        return None
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None

    # höchste Konfidenz
    boxes = r0.boxes
    idx = int(boxes.conf.argmax().item())
    xyxy = boxes.xyxy[idx].cpu().numpy()
    score = float(boxes.conf[idx].item())

    x1, y1, x2, y2 = xyxy.tolist()
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    xi, yi, bw, bh = _aabb_from_corners(corners)

    return CrateDetectionResult(
        x=xi,
        y=yi,
        w=bw,
        h=bh,
        corners=corners,
        confidence=score,
        source="yolo",
    )


def detect_crate(
    frame_bgr: np.ndarray,
    *,
    weights_path: str | Path | None = None,
    prefer_yolo: bool = True,
    conf: float = 0.25,
    iou: float = 0.45,
    log_corners: bool = False,
) -> CrateDetectionResult | None:
    """
    Erkennt einen Bierkasten im BGR-Bild.

    Args:
        frame_bgr: OpenCV-BGR ``numpy.ndarray``.
        weights_path: Pfad zu ``best.pt`` / eigene Gewichte. Übersteuert Umgebungsvariable.
        prefer_yolo: Zuerst YOLO versuchen (wenn Gewichte gefunden werden).
        conf / iou: YOLO-Schwellen.
        log_corners: Eckpunkte über ``logging`` ausgeben (für Abnahme „4 Punkte werden geloggt“).

    Returns:
        ``CrateDetectionResult`` oder ``None``.
    """
    weights = _resolve_weights(weights_path)

    if prefer_yolo and weights is not None:
        try:
            det = _detect_yolo(frame_bgr, weights, conf=conf, iou=iou)
            if det is not None:
                if log_corners:
                    logger.info(
                        "detect_crate [yolo] xywh=(%s,%s,%s,%s) corners=%s conf=%.3f",
                        det.x,
                        det.y,
                        det.w,
                        det.h,
                        det.corners.tolist(),
                        det.confidence,
                    )
                return det
        except Exception:
            logger.exception("YOLO-Inferenz fehlgeschlagen, Fallback auf Konturen.")

    det = detect_crate_contour(frame_bgr)
    if det is not None and log_corners:
        logger.info(
            "detect_crate [contour] xywh=(%s,%s,%s,%s) corners=%s conf=%.3f",
            det.x,
            det.y,
            det.w,
            det.h,
            det.corners.tolist(),
            det.confidence,
        )
    return det


def draw_detection(frame_bgr: np.ndarray, det: CrateDetectionResult, *, color_yolo=None, color_contour=None) -> np.ndarray:
    """Zeichnet Bounding Box und Eckpunkte (YOLO: Grün laut Aufgabenstellung; Kontur: Orange)."""
    out = frame_bgr.copy()
    color_yolo = color_yolo or (0, 255, 0)
    color_contour = color_contour or (0, 165, 255)
    color = color_yolo if det.source == "yolo" else color_contour

    cv2.rectangle(out, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
    pts = det.corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, color, 2)
    for i, (px, py) in enumerate(det.corners.astype(int)):
        cv2.circle(out, (int(px), int(py)), 4, color, -1)
        cv2.putText(out, str(i), (int(px) + 4, int(py) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    label = f"{det.source} {det.confidence:.2f}"
    cv2.putText(out, label, (det.x, max(15, det.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
