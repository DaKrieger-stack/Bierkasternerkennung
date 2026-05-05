#!/usr/bin/env python3
"""YOLOv8 Fine-Tuning für Klasse ``bierkasten`` (Ultralytics)."""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

import torch
from ultralytics import YOLO

SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def default_workers() -> int:
    """Unter Windows oft stabilere DataLoader mit workers=0."""
    return 0 if platform.system() == "Windows" else min(8, (os.cpu_count() or 8))


def pick_batch(user_batch: int | None, device: str) -> int:
    """
    Standard-Batch: GPU → AutoBatch (-1), CPU → klein halten (RAM).
    Explizit gesetztes ``user_batch`` hat Vorrang.
    """
    if user_batch is not None:
        return user_batch
    d = device.strip().lower()
    if d == "cpu" or d.endswith(":cpu"):
        return 4
    return -1 if torch.cuda.is_available() else 4


def resolve_device(user_device: str) -> str:
    if user_device.strip():
        return user_device.strip()
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert YOLOv8 auf data/kasten_dataset (siehe kasten.yaml)")
    parser.add_argument("--model", default="yolov8n.pt", help="Basismodell oder eigene .pt")
    parser.add_argument("--data", default="kasten.yaml", type=Path, help="Dataset-YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, help="Eingangsauflösung (416 schneller, 640 genauer)")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch-Größe; weglassen = GPU: AutoBatch (-1), CPU: 4",
    )
    parser.add_argument(
        "--device",
        default="",
        help="z.B. 0, cuda:0 oder cpu (leer = CUDA wenn verfügbar, sonst cpu)",
    )
    parser.add_argument("--workers", type=int, default=None, help="DataLoader-Workers (Standard: 0 unter Windows, sonst bis 8)")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early-Stopping-Geduld (Epochs ohne Verbesserung)",
    )
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="kasten")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Datensatz cachen (bei wenigen Bildern oft schneller, braucht mehr RAM/Disk je nach Modus)",
    )
    args = parser.parse_args()

    os.chdir(SCRIPT_ROOT)

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"kasten.yaml nicht gefunden: {data_path}")

    device = resolve_device(args.device)
    workers = default_workers() if args.workers is None else args.workers
    batch = pick_batch(args.batch, device)

    model = YOLO(args.model)
    train_kw = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )
    if args.cache:
        train_kw["cache"] = True

    print(f"Device: {device} | batch: {batch} | workers: {workers} | imgsz: {args.imgsz} | patience: {args.patience}")

    model.train(**train_kw)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Fertig. Gewichte (lokal): {best.resolve()}")
    print("Für Inferenz: Umgebungsvariable KASTEN_YOLO_WEIGHTS setzen oder Pfad an detect_crate weitergeben.")
    print("Hinweis: Ultralytics kann den Ordnernamen bei Kollision erhöhen (z.B. kasten2) — dann best.pt dort suchen.")


if __name__ == "__main__":
    main()
