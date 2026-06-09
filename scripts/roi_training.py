#!/usr/bin/env python3
"""
Gemeinsame Trainingsroutine für die ROI-Klassifikatoren (Stufe 1 Slot, Stufe 2 Kronkorken).

Wird von ``train_slot_cnn.py`` und ``train_cap_cnn.py`` genutzt. Liest Bilder aus
Klassenordnern, mappt Ordnernamen auf Zielklassen, trainiert ein kleines CNN bzw.
MobileNetV2 und speichert einen Checkpoint im Format aus ``kastendetektion/ml_models.py``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kastendetektion.ml_models import build_model, save_checkpoint  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_samples(dataset_dir: Path, folder_to_label: dict[str, int]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for folder, label in folder_to_label.items():
        sub = dataset_dir / folder
        if not sub.is_dir():
            continue
        for p in sorted(sub.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                samples.append((p, label))
    return samples


def _to_tensor_batch(paths: list[Path], input_size: int):
    import torch

    arrs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            img = np.zeros((input_size, input_size, 3), np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (input_size, input_size)).astype(np.float32) / 255.0
        arrs.append(rgb)
    arr = np.stack(arrs).transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


def train_roi_classifier(
    *,
    dataset_dir: Path,
    folder_to_label: dict[str, int],
    classes: list[str],
    out_path: Path,
    arch: str = "smallcnn",
    input_size: int = 64,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "",
    seed: int = 42,
) -> None:
    import torch
    import torch.nn as nn

    dev = device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    samples = _load_samples(dataset_dir, folder_to_label)
    if not samples:
        raise SystemExit(
            f"Keine Bilder in {dataset_dir} für Ordner {list(folder_to_label)} gefunden. "
            "Zuerst scripts/extract_slot_dataset.py ausführen und einsortieren."
        )
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_split))
    val = samples[:n_val]
    train = samples[n_val:] or samples

    counts = {c: 0 for c in range(len(classes))}
    for _, y in samples:
        counts[y] = counts.get(y, 0) + 1
    print(f"Samples: {len(samples)} (train {len(train)}, val {len(val)}) | Klassen {classes} -> {counts}")

    model = build_model(arch, num_classes=len(classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def batches(data):
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size]
            x = _to_tensor_batch([p for p, _ in chunk], input_size).to(dev)
            y = torch.tensor([lbl for _, lbl in chunk], dtype=torch.long, device=dev)
            yield x, y

    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train)
        total_loss = 0.0
        for x, y in batches(train):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(y)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in batches(val):
                pred = model(x).argmax(1)
                correct += int((pred == y).sum())
        acc = correct / max(1, len(val))
        print(f"Epoch {epoch:3d}/{epochs} | loss {total_loss / len(train):.4f} | val_acc {acc:.3f}")

        if acc >= best_acc:
            best_acc = acc
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(str(out_path), model, arch=arch, classes=classes, input_size=input_size)

    print(f"Bestes val_acc {best_acc:.3f} | Checkpoint: {out_path.resolve()}")
