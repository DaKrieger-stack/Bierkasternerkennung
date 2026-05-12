#!/usr/bin/env python3
"""Detektiert einen roten Bierkasten in einem Bild mit einem klassischen OpenCV-Workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WINDOW_NAME = "Roter Bierkasten - Detektion"


@dataclass
class DetectionCandidate:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    box_points: np.ndarray
    area: float
    perimeter: float
    circularity: float
    hu_moments: np.ndarray
    score: float


@dataclass
class DetectionResult:
    image_path: Path
    refined_mask: np.ndarray
    lines: list[np.ndarray]
    circles: np.ndarray | None
    candidate: DetectionCandidate | None
    visualization: np.ndarray


@dataclass
class ViewerState:
    zoom: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    dragging: bool = False
    drag_start_x: int = 0
    drag_start_y: int = 0
    drag_offset_x: int = 0
    drag_offset_y: int = 0


def segment_red_hsv(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segmentiert rote Pixel im HSV-Farbraum und verfeinert sie mit Otsu auf der Sättigung."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 60, 40], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([170, 60, 40], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask_red_1, mask_red_2)

    saturation = hsv[:, :, 1]
    masked_saturation = cv2.bitwise_and(saturation, saturation, mask=red_mask)
    _, otsu_mask = cv2.threshold(
        masked_saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    combined_mask = cv2.bitwise_and(red_mask, otsu_mask)
    return hsv, red_mask, combined_mask


def refine_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Schließt Lücken und entfernt kleine Störungen."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return opened


def detect_lines(binary_mask: np.ndarray) -> list[np.ndarray]:
    """Findet gerade Außenkanten über HoughLinesP."""
    lines = cv2.HoughLinesP(
        binary_mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=60,
        minLineLength=max(30, binary_mask.shape[1] // 8),
        maxLineGap=20,
    )
    if lines is None:
        return []
    return [line[0] for line in lines]


def detect_circles(image_bgr: np.ndarray, binary_mask: np.ndarray) -> np.ndarray | None:
    """Optionale Kreissuche, z. B. für sichtbare Flaschenöffnungen bei Draufsicht."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    masked_gray = cv2.bitwise_and(gray, gray, mask=binary_mask)

    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=18,
        minRadius=4,
        maxRadius=30,
    )
    if circles is None:
        return None
    return np.round(circles[0]).astype(int)


def contour_hu_signature(contour: np.ndarray) -> np.ndarray:
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    return np.sign(hu) * np.log10(np.abs(hu) + 1e-12)


def score_contour(contour: np.ndarray, image_shape: tuple[int, int, int]) -> DetectionCandidate | None:
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    image_area = float(image_shape[0] * image_shape[1])
    if area < image_area * 0.01:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(approx) < 4:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.int32)
    rect_area = max(rect[1][0] * rect[1][1], 1.0)
    rectangularity = float(area / rect_area)
    aspect_ratio = max(w, h) / max(1.0, min(w, h))
    hu = contour_hu_signature(contour)

    # Rechtecke sind typischerweise nicht stark zirkulär; die Kompaktheit bleibt dennoch hoch.
    if circularity > 0.9:
        return None
    if rectangularity < 0.55:
        return None
    if aspect_ratio > 4.5:
        return None

    score = area * rectangularity * (1.0 - min(abs(circularity - 0.65), 0.65))
    return DetectionCandidate(
        contour=contour,
        bbox=(x, y, w, h),
        box_points=box,
        area=float(area),
        perimeter=float(perimeter),
        circularity=circularity,
        hu_moments=hu,
        score=float(score),
    )


def find_best_candidate(binary_mask: np.ndarray, image_bgr: np.ndarray) -> DetectionCandidate | None:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [score_contour(contour, image_bgr.shape) for contour in contours]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)


def draw_result(
    image_bgr: np.ndarray,
    candidate: DetectionCandidate | None,
    lines: list[np.ndarray],
    circles: np.ndarray | None,
    *,
    show_lines: bool = False,
    show_circles: bool = False,
) -> np.ndarray:
    output = image_bgr.copy()

    if show_lines:
        for x1, y1, x2, y2 in lines:
            cv2.line(output, (x1, y1), (x2, y2), (255, 255, 0), 2)

    if show_circles and circles is not None:
        for x, y, radius in circles:
            cv2.circle(output, (x, y), radius, (255, 0, 255), 2)
            cv2.circle(output, (x, y), 2, (255, 0, 255), 3)

    if candidate is not None:
        x, y, w, h = candidate.bbox
        cv2.drawContours(output, [candidate.box_points], 0, (0, 255, 0), 3)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.drawContours(output, [candidate.contour], -1, (0, 128, 255), 2)

        label = (
            f"Roter Bierkasten | circ={candidate.circularity:.3f} "
            f"| score={candidate.score:.1f}"
        )
        cv2.putText(
            output,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def print_metrics(candidate: DetectionCandidate | None, lines: list[np.ndarray], circles: np.ndarray | None) -> None:
    print(f"Hough-Linien: {len(lines)}")
    print(f"Hough-Kreise: {0 if circles is None else len(circles)}")
    if candidate is None:
        print("Kein roter Bierkasten gefunden.")
        return

    x, y, w, h = candidate.bbox
    hu_str = ", ".join(f"{value:.3f}" for value in candidate.hu_moments)
    print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
    print(f"Flaeche: {candidate.area:.1f}")
    print(f"Umfang: {candidate.perimeter:.1f}")
    print(f"Zirkularitaet: {candidate.circularity:.4f}")
    print(f"Hu-Momente (log): [{hu_str}]")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def run_detection(
    image_path: Path,
    *,
    show_lines: bool = False,
    show_circles: bool = False,
) -> DetectionResult:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Bild konnte nicht gelesen werden: {image_path}")

    _, _, combined_mask = segment_red_hsv(image_bgr)
    refined_mask = refine_mask(combined_mask)
    lines = detect_lines(refined_mask)
    circles = detect_circles(image_bgr, refined_mask)
    candidate = find_best_candidate(refined_mask, image_bgr)
    vis = draw_result(
        image_bgr,
        candidate,
        lines,
        circles,
        show_lines=show_lines,
        show_circles=show_circles,
    )
    return DetectionResult(
        image_path=image_path,
        refined_mask=refined_mask,
        lines=lines,
        circles=circles,
        candidate=candidate,
        visualization=vis,
    )


def clamp_view(state: ViewerState, image_shape: tuple[int, int, int], window_size: tuple[int, int]) -> None:
    img_h, img_w = image_shape[:2]
    win_w, win_h = window_size
    scaled_w = max(1, int(round(img_w * state.zoom)))
    scaled_h = max(1, int(round(img_h * state.zoom)))
    max_x = max(0, scaled_w - win_w)
    max_y = max(0, scaled_h - win_h)
    state.offset_x = min(max(0, state.offset_x), max_x)
    state.offset_y = min(max(0, state.offset_y), max_y)


def render_view(image_bgr: np.ndarray, state: ViewerState, window_size: tuple[int, int]) -> np.ndarray:
    img_h, img_w = image_bgr.shape[:2]
    win_w, win_h = window_size
    scaled_w = max(1, int(round(img_w * state.zoom)))
    scaled_h = max(1, int(round(img_h * state.zoom)))
    interpolation = cv2.INTER_LINEAR if state.zoom >= 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(image_bgr, (scaled_w, scaled_h), interpolation=interpolation)
    clamp_view(state, image_bgr.shape, window_size)

    x0 = state.offset_x
    y0 = state.offset_y
    x1 = min(scaled_w, x0 + win_w)
    y1 = min(scaled_h, y0 + win_h)
    crop = scaled[y0:y1, x0:x1]

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    canvas[: crop.shape[0], : crop.shape[1]] = crop
    return canvas


def create_mouse_callback(state: ViewerState, image_shape: tuple[int, int, int], window_size: tuple[int, int]):
    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging = True
            state.drag_start_x = x
            state.drag_start_y = y
            state.drag_offset_x = state.offset_x
            state.drag_offset_y = state.offset_y
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.offset_x = state.drag_offset_x - (x - state.drag_start_x)
            state.offset_y = state.drag_offset_y - (y - state.drag_start_y)
            clamp_view(state, image_shape, window_size)
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1.25 if flags > 0 else 0.8
            state.zoom = min(max(0.2, state.zoom * delta), 8.0)
            clamp_view(state, image_shape, window_size)

    return on_mouse


def show_interactive(results: list[DetectionResult]) -> None:
    if not results:
        return

    window_size = (1280, 900)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *window_size)

    index = 0
    state = ViewerState()
    cv2.setMouseCallback(
        WINDOW_NAME,
        create_mouse_callback(state, results[index].visualization.shape, window_size),
    )

    while True:
        result = results[index]
        image = result.visualization.copy()
        help_text = (
            f"{index + 1}/{len(results)}  {result.image_path.name}  "
            "[n/p] Bild  [WASD/Pfeile] scrollen  [+/-] Zoom  [r] Reset  [q] Ende"
        )
        cv2.putText(image, help_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        view = render_view(image, state, window_size)
        cv2.imshow(WINDOW_NAME, view)

        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        if key in (ord("q"), 27):
            break
        if key in (ord("r"), ord("R")):
            state = ViewerState()
            cv2.setMouseCallback(
                WINDOW_NAME,
                create_mouse_callback(state, results[index].visualization.shape, window_size),
            )
            continue
        if key in (ord("+"), ord("=")):
            state.zoom = min(8.0, state.zoom * 1.25)
            clamp_view(state, image.shape, window_size)
            continue
        if key in (ord("-"), ord("_")):
            state.zoom = max(0.2, state.zoom * 0.8)
            clamp_view(state, image.shape, window_size)
            continue
        if key in (ord("n"), ord("N"), 2555904):
            index = (index + 1) % len(results)
            state = ViewerState()
            cv2.setMouseCallback(
                WINDOW_NAME,
                create_mouse_callback(state, results[index].visualization.shape, window_size),
            )
            continue
        if key in (ord("p"), ord("P"), 2424832):
            index = (index - 1) % len(results)
            state = ViewerState()
            cv2.setMouseCallback(
                WINDOW_NAME,
                create_mouse_callback(state, results[index].visualization.shape, window_size),
            )
            continue
        if key in (ord("a"), ord("A"), 2424832):
            state.offset_x -= 80
        elif key in (ord("d"), ord("D"), 2555904):
            state.offset_x += 80
        elif key in (ord("w"), ord("W"), 2490368):
            state.offset_y -= 80
        elif key in (ord("s"), ord("S"), 2621440):
            state.offset_y += 80
        clamp_view(state, image.shape, window_size)

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detektiert einen roten Bierkasten in einem Bild.")
    parser.add_argument("image", type=Path, help="Pfad zu einem Bild oder einem Ordner mit Bildern")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optionaler Pfad fuer das visualisierte Ausgabebild",
    )
    parser.add_argument(
        "--save-mask",
        type=Path,
        default=None,
        help="Optionaler Pfad zum Speichern der bereinigten Binarmaske",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Kein cv2.imshow; nur Ausgaben auf der Konsole bzw. in Dateien",
    )
    parser.add_argument(
        "--show-lines",
        action="store_true",
        help="Zeigt zusaetzlich die gelben Hough-Linien an",
    )
    parser.add_argument(
        "--show-circles",
        action="store_true",
        help="Zeigt zusaetzlich die lila Hough-Kreise an",
    )
    return parser.parse_args()


def resolve_output_path(base: Path, image_path: Path, suffix: str, *, multiple: bool) -> Path:
    if base.suffix and not multiple:
        return base
    return base / f"{image_path.stem}{suffix}"


def main() -> None:
    args = parse_args()
    input_path = args.image.resolve()
    image_paths = collect_images(input_path)
    if not image_paths:
        raise SystemExit(f"Keine Bilder gefunden unter: {input_path}")

    results: list[DetectionResult] = []
    multiple = len(image_paths) > 1
    for image_path in image_paths:
        print(f"\n=== {image_path.name} ===")
        result = run_detection(
            image_path,
            show_lines=args.show_lines,
            show_circles=args.show_circles,
        )
        print_metrics(result.candidate, result.lines, result.circles)
        results.append(result)

        if args.save_mask is not None:
            mask_path = resolve_output_path(args.save_mask.resolve(), image_path, "_mask.png", multiple=multiple)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), result.refined_mask)
            print(f"Maske gespeichert: {mask_path}")

        if args.output is not None:
            output_path = resolve_output_path(args.output.resolve(), image_path, "_detected.png", multiple=multiple)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result.visualization)
            print(f"Ergebnis gespeichert: {output_path}")

    if not args.no_display:
        print("Fenstersteuerung: n/p = Bildwechsel, WASD/Pfeile = scrollen, +/- = Zoom, r = Reset, q = Ende.")
        show_interactive(results)


if __name__ == "__main__":
    main()
