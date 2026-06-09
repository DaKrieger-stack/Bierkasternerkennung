#!/usr/bin/env python3
"""Analysiert HSV-Werte in einem Bild oder ganzen Ordner für die Rot-Erkennung."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = SCRIPT_ROOT / "data" / "kasten_dataset" / "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class RunningStats:
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    sum: float = 0.0
    sum_sq: float = 0.0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        flat = values.astype(np.float64).ravel()
        current_min = float(flat.min())
        current_max = float(flat.max())
        self.minimum = current_min if self.minimum is None else min(self.minimum, current_min)
        self.maximum = current_max if self.maximum is None else max(self.maximum, current_max)
        self.count += flat.size
        self.sum += float(flat.sum())
        self.sum_sq += float(np.square(flat).sum())

    def mean(self) -> float:
        return self.sum / self.count if self.count else float("nan")

    def std(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.mean()
        variance = max(self.sum_sq / self.count - mean * mean, 0.0)
        return float(np.sqrt(variance))


def list_images(path: Path, recursive: bool = True) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Pfad nicht gefunden: {path}")

    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted(
        candidate
        for candidate in iterator
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_hsv(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Bild konnte nicht gelesen werden: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def crop_center_roi(
    hsv: np.ndarray,
    *,
    y_start_factor: float = 0.25,
    y_end_factor: float = 0.75,
    x_start_factor: float = 0.15,
    x_end_factor: float = 0.85,
) -> np.ndarray:
    height, width = hsv.shape[:2]
    y_start, y_end = int(height * y_start_factor), int(height * y_end_factor)
    x_start, x_end = int(width * x_start_factor), int(width * x_end_factor)
    return hsv[y_start:y_end, x_start:x_end]


def red_mask(hsv: np.ndarray) -> np.ndarray:
    lower_red_1 = np.array([0, 80, 50], dtype=np.uint8)
    upper_red_1 = np.array([15, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([165, 80, 50], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    return cv2.bitwise_or(mask_red_1, mask_red_2)


def print_channel_stats(prefix: str, data: np.ndarray) -> None:
    print(f"{prefix}: min={int(data.min())}, max={int(data.max())}, "
          f"mean={data.mean():.1f}, std={data.std():.1f}")


def percentile(values: np.ndarray, p: float) -> int:
    return int(np.clip(np.round(np.percentile(values, p)), 0, 255))


def analyse_images(image_paths: list[Path], *, full_image: bool) -> None:
    if not image_paths:
        print("Keine Bilder gefunden.")
        return

    roi_h = RunningStats()
    roi_s = RunningStats()
    roi_v = RunningStats()
    red_pixels_all: list[np.ndarray] = []
    processed = 0
    with_red = 0

    for image_path in image_paths:
        try:
            hsv = load_hsv(image_path)
        except ValueError as exc:
            print(exc)
            continue

        processed += 1
        roi = hsv if full_image else crop_center_roi(hsv)

        roi_h.update(roi[:, :, 0])
        roi_s.update(roi[:, :, 1])
        roi_v.update(roi[:, :, 2])

        mask = red_mask(roi)
        red_pixels = roi[mask > 0]
        if red_pixels.size == 0:
            continue

        with_red += 1
        red_pixels_all.append(red_pixels)

    print(f"Verarbeitete Bilder: {processed}/{len(image_paths)}")
    print(f"Bilder mit roten Pixeln: {with_red}")
    print()
    print("--- HSV-Statistiken für den ausgewählten Bereich ---")
    print(
        f"Hue (H): min={int(roi_h.minimum or 0)}, max={int(roi_h.maximum or 0)}, "
        f"mean={roi_h.mean():.1f}, std={roi_h.std():.1f}"
    )
    print(
        f"Sat (S): min={int(roi_s.minimum or 0)}, max={int(roi_s.maximum or 0)}, "
        f"mean={roi_s.mean():.1f}, std={roi_s.std():.1f}"
    )
    print(
        f"Val (V): min={int(roi_v.minimum or 0)}, max={int(roi_v.maximum or 0)}, "
        f"mean={roi_v.mean():.1f}, std={roi_v.std():.1f}"
    )

    if not red_pixels_all:
        print()
        print("Keine roten Pixel mit den aktuellen Kriterien gefunden.")
        return

    red_pixels = np.concatenate(red_pixels_all, axis=0)
    low_red = red_pixels[red_pixels[:, 0] <= 15]
    high_red = red_pixels[red_pixels[:, 0] >= 165]

    print()
    print(f"--- Rot-Pixel gesamt ({len(red_pixels)} Pixel) ---")
    print_channel_stats("Hue (H)", red_pixels[:, 0])
    print_channel_stats("Sat (S)", red_pixels[:, 1])
    print_channel_stats("Val (V)", red_pixels[:, 2])

    if len(low_red) > 0:
        print()
        print(f"--- Niedrige Rot-Hue-Gruppe ({len(low_red)} Pixel) ---")
        print_channel_stats("Hue (H)", low_red[:, 0])
    if len(high_red) > 0:
        print()
        print(f"--- Hohe Rot-Hue-Gruppe ({len(high_red)} Pixel) ---")
        print_channel_stats("Hue (H)", high_red[:, 0])

    hue_low_max = percentile(low_red[:, 0], 95) if len(low_red) > 0 else 15
    hue_high_min = percentile(high_red[:, 0], 5) if len(high_red) > 0 else 165
    sat_min = max(20, percentile(red_pixels[:, 1], 10))
    val_min = max(20, percentile(red_pixels[:, 2], 10))

    print()
    print("--- Empfohlene Startwerte für `scripts/red_crate_detect.py` ---")
    print(f"lower_red_1 = [0, {sat_min}, {val_min}]")
    print(f"upper_red_1 = [{hue_low_max}, 255, 255]")
    print(f"lower_red_2 = [{hue_high_min}, {sat_min}, {val_min}]")
    print("upper_red_2 = [179, 255, 255]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analysiert HSV-Werte in einem Bild oder einem ganzen Ordner."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Bild oder Ordner mit Bildern (Standard: data/kasten_dataset/images)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Ordner nicht rekursiv durchsuchen.",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Den kompletten Bildinhalt statt des mittleren ROI auswerten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.path)
    image_paths = list_images(input_path, recursive=not args.no_recursive)
    analyse_images(image_paths, full_image=args.full_image)


if __name__ == "__main__":
    main()
