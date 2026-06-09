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


def _segment_red_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """HSV-Rotsegmentierung (Paulaner-Kasten) mit Otsu-Verfeinerung auf der Sättigung."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 60, 40], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([170, 60, 40], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red_1, upper_red_1),
        cv2.inRange(hsv, lower_red_2, upper_red_2),
    )
    saturation = hsv[:, :, 1]
    masked_sat = cv2.bitwise_and(saturation, saturation, mask=red_mask)
    _, otsu = cv2.threshold(masked_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_and(red_mask, otsu)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)


def _score_rect_contour(
    contour: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[float, np.ndarray, float] | None:
    """Bewertet eine Kontur als Kasten-Rechteck; gibt (score, box_points, angle) zurück."""
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    img_h, img_w = image_shape[:2]
    frame_area = float(img_h * img_w)
    if area < min_area_ratio * frame_area or area > max_area_ratio * frame_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    if circularity > 0.92:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    if w < 20 or h < 20:
        return None

    aspect = max(w, h) / max(1.0, min(w, h))
    if aspect > 4.5:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    rect_area = max(rect[1][0] * rect[1][1], 1.0)
    rectangularity = float(area / rect_area)
    if rectangularity < 0.55:
        return None

    # Randkonturen (Boden, Wand) abwerten.
    margin = 4
    touches_border = (
        x <= margin
        or y <= margin
        or x + w >= img_w - margin
        or y + h >= img_h - margin
    )
    border_penalty = 0.55 if touches_border else 1.0

    score = area * rectangularity * border_penalty * (1.0 - min(abs(circularity - 0.65), 0.65))
    return float(score), box, float(rect[2])


def _best_contour_from_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.92,
) -> tuple[float, np.ndarray, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, np.ndarray, float] | None = None
    for cnt in contours:
        scored = _score_rect_contour(
            cnt,
            image_shape,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        if scored is None:
            continue
        if best is None or scored[0] > best[0]:
            best = scored
    return best


def _result_from_box(
    box_points: np.ndarray,
    angle: float,
    score: float,
    image_shape: tuple[int, int, int],
    *,
    max_area_ratio: float = 0.92,
) -> CrateDetectionResult:
    corners = _order_corners_tl_tr_br_bl(box_points)
    img_h, img_w = image_shape[:2]
    corners[:, 0] = np.clip(corners[:, 0], 0.0, float(img_w - 1))
    corners[:, 1] = np.clip(corners[:, 1], 0.0, float(img_h - 1))
    x, y, bw, bh = _aabb_from_corners(corners)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    bw = max(1, min(bw, img_w - x))
    bh = max(1, min(bh, img_h - y))
    frame_area = float(img_h * img_w)
    confidence = float(min(1.0, score / max(frame_area * max_area_ratio, 1.0)))
    return CrateDetectionResult(
        x=x,
        y=y,
        w=bw,
        h=bh,
        corners=corners,
        confidence=confidence,
        source="contour",
        orientation_deg=angle,
    )


def _detect_red_hsv(
    frame_bgr: np.ndarray,
    *,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.92,
) -> CrateDetectionResult | None:
    """Rot-HSV: zuverlässiger als reines Canny bei Paulaner-Kästen."""
    mask = _segment_red_mask(frame_bgr)
    best = _best_contour_from_mask(
        mask,
        frame_bgr.shape,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    if best is None:
        return None
    score, box, angle = best
    return _result_from_box(box, angle, score, frame_bgr.shape, max_area_ratio=max_area_ratio)


def _detect_canny_contour(
    frame_bgr: np.ndarray,
    *,
    canny_low: int = 30,
    canny_high: int = 100,
    min_area_ratio: float = 0.03,
    max_area_ratio: float = 0.92,
) -> CrateDetectionResult | None:
    """Canny-Fallback mit Rechteck-Scoring statt nur größter Fläche."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    # Adaptive Schwellwert-Kanten ergänzen (robuster bei ungleichmäßigem Licht).
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    combined = cv2.bitwise_or(edges, adapt)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
    )

    best = _best_contour_from_mask(
        combined,
        frame_bgr.shape,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    if best is None:
        return None
    score, box, angle = best
    return _result_from_box(box, angle, score, frame_bgr.shape, max_area_ratio=max_area_ratio)


def _refine_corners_red_hsv(frame_bgr: np.ndarray, det: CrateDetectionResult) -> CrateDetectionResult:
    """
    Verfeinert YOLO-/Kontur-Ecken über Rot-Maske im Detektions-ROI.
    Liefert gedrehte minAreaRect-Ecken statt achsparalleler YOLO-Box.
    """
    img_h, img_w = frame_bgr.shape[:2]
    pad_x = max(8, int(det.w * 0.05))
    pad_y = max(8, int(det.h * 0.05))
    x0 = max(0, det.x - pad_x)
    y0 = max(0, det.y - pad_y)
    x1 = min(img_w, det.x + det.w + pad_x)
    y1 = min(img_h, det.y + det.h + pad_y)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return det

    refined = _detect_red_hsv(roi, min_area_ratio=0.15, max_area_ratio=0.98)
    if refined is None or refined.confidence < 0.15:
        return det

    corners = refined.corners.copy()
    corners[:, 0] += x0
    corners[:, 1] += y0
    corners[:, 0] = np.clip(corners[:, 0], 0.0, float(img_w - 1))
    corners[:, 1] = np.clip(corners[:, 1], 0.0, float(img_h - 1))

    x, y, bw, bh = _aabb_from_corners(corners)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    bw = max(1, min(bw, img_w - x))
    bh = max(1, min(bh, img_h - y))

    # Nur übernehmen, wenn die Verfeinerung zur ursprünglichen Detektion passt.
    det_cx = det.x + det.w / 2.0
    det_cy = det.y + det.h / 2.0
    ref_cx = x + bw / 2.0
    ref_cy = y + bh / 2.0
    center_shift = np.hypot(ref_cx - det_cx, ref_cy - det_cy)
    if center_shift > min(det.w, det.h) * 0.25:
        return det

    overlap_w = min(x + bw, det.x + det.w) - max(x, det.x)
    overlap_h = min(y + bh, det.y + det.h) - max(y, det.y)
    if overlap_w <= 0 or overlap_h <= 0:
        return det
    overlap_area = overlap_w * overlap_h
    if overlap_area < det.w * det.h * 0.45:
        return det

    return CrateDetectionResult(
        x=x,
        y=y,
        w=bw,
        h=bh,
        corners=corners,
        confidence=max(det.confidence, refined.confidence),
        source=det.source,
        orientation_deg=refined.orientation_deg,
    )


def detect_crate_contour(
    frame_bgr: np.ndarray,
    *,
    canny_low: int = 30,
    canny_high: int = 100,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.92,
) -> CrateDetectionResult | None:
    """Klassischer Fallback: zuerst Rot-HSV, dann verbessertes Canny."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    det = _detect_red_hsv(frame_bgr, min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio)
    if det is not None:
        return det

    return _detect_canny_contour(
        frame_bgr,
        canny_low=canny_low,
        canny_high=canny_high,
        min_area_ratio=max(min_area_ratio, 0.03),
        max_area_ratio=max_area_ratio,
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
        "runs/detect/runs/detect/kasten-retrain/weights/best.pt",
        "runs/detect/runs/detect/kasten-2/weights/best.pt",
        "runs/detect/runs/detect/kasten/weights/best.pt",
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
                det = _refine_corners_red_hsv(frame_bgr, det)
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
    if det is not None:
        det = _refine_corners_red_hsv(frame_bgr, det)
        if log_corners:
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
