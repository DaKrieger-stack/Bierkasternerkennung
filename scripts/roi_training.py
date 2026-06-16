#!/usr/bin/env python3
"""
Gemeinsame Trainingsroutine für ROI-Klassifikatoren.

Verbesserungen gegenüber der Baseline:
- Daten-Augmentation (Flip, Helligkeit, leichte Rotation)
- Klassen-Gewichte + WeightedRandomSampler (hilft „leer“)
- Stratifizierter Train/Val-Split
- Checkpoint nach Macro-F1 (nicht nur Gesamt-Accuracy)
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


def _stratified_split(
    samples: list[tuple[Path, int]],
    val_split: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for item in samples:
        by_label.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for items in by_label.values():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_split)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train or samples, val or samples


def _augment_bgr(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    if random.random() < 0.5:
        out = cv2.flip(out, 1)
    if random.random() < 0.4:
        angle = random.uniform(-12.0, 12.0)
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    alpha = random.uniform(0.82, 1.18)
    beta = random.randint(-18, 18)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if random.random() < 0.25:
        noise = np.random.normal(0, 6, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def _load_rgb_tensor(path: Path, input_size: int, *, augment: bool) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        img = np.zeros((input_size, input_size, 3), np.uint8)
    if augment:
        img = _augment_bgr(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size)).astype(np.float32) / 255.0
    return rgb


def _to_tensor_batch(paths: list[Path], input_size: int, *, augment: bool = False):
    import torch

    arrs = [_load_rgb_tensor(p, input_size, augment=augment) for p in paths]
    arr = np.stack(arrs).transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


def _macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    f1s: list[float] = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append(2 * prec * rec / max(1e-6, prec + rec))
    return float(sum(f1s) / max(1, num_classes))


def _class_recall(y_true: list[int], y_pred: list[int], cls: int) -> float:
    hits = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    total = sum(1 for t in y_true if t == cls)
    return hits / max(1, total)


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
    augment: bool = True,
    class_weights: bool = True,
    focus_class: int | None = None,
    focus_classes: list[int] | None = None,
    min_recall_weight: float = 0.35,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import WeightedRandomSampler

    dev = device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    samples = _load_samples(dataset_dir, folder_to_label)
    if not samples:
        raise SystemExit(
            f"Keine Bilder in {dataset_dir} für Ordner {list(folder_to_label)} gefunden."
        )

    train, val = _stratified_split(samples, val_split, seed)

    counts = {c: 0 for c in range(len(classes))}
    for _, y in samples:
        counts[y] = counts.get(y, 0) + 1
    print(f"Samples: {len(samples)} (train {len(train)}, val {len(val)}) | Klassen {classes} -> {counts}")

    model = build_model(arch, num_classes=len(classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if class_weights:
        total = sum(counts.values())
        w = torch.tensor(
            [total / max(1, counts[c] * len(classes)) for c in range(len(classes))],
            dtype=torch.float32,
            device=dev,
        )
        loss_fn: nn.Module = nn.CrossEntropyLoss(weight=w)
        print(f"Klassen-Gewichte: {[round(float(x), 3) for x in w.tolist()]}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    train_weights = [1.0 / max(1, counts[y]) for _, y in train]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train), replacement=True)

    def run_epoch(data: list[tuple[Path, int]], *, train_mode: bool) -> tuple[list[int], list[int], float]:
        if train_mode:
            model.train()
            order = list(range(len(data)))
            random.shuffle(order)
        else:
            model.eval()
            order = list(range(len(data)))

        y_true: list[int] = []
        y_pred: list[int] = []
        total_loss = 0.0
        n_seen = 0

        if train_mode:
            indices = list(sampler)
            chunks = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
        else:
            chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]

        for chunk in chunks:
            batch = [data[i] for i in chunk]
            x = _to_tensor_batch([p for p, _ in batch], input_size, augment=train_mode and augment).to(dev)
            y = torch.tensor([lbl for _, lbl in batch], dtype=torch.long, device=dev)
            if train_mode:
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach()) * len(y)
            else:
                with torch.no_grad():
                    out = model(x)
                    loss = loss_fn(out, y)
                    total_loss += float(loss.detach()) * len(y)
            pred = out.argmax(1).detach().cpu().tolist()
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(pred)
            n_seen += len(y)
        return y_true, y_pred, total_loss / max(1, n_seen)

    best_score = -1.0
    for epoch in range(1, epochs + 1):
        _, _, train_loss = run_epoch(train, train_mode=True)
        y_true, y_pred, _ = run_epoch(val, train_mode=False)
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))
        macro = _macro_f1(y_true, y_pred, len(classes))
        recalls = {classes[c]: _class_recall(y_true, y_pred, c) for c in range(len(classes))}
        min_rec = min(recalls.values()) if recalls else 0.0
        if focus_classes:
            focus_rec = sum(recalls[classes[c]] for c in focus_classes) / len(focus_classes)
        elif focus_class is not None:
            focus_rec = recalls[classes[focus_class]]
        else:
            focus_rec = macro
        score = (1.0 - min_recall_weight) * macro + min_recall_weight * min_rec
        if focus_classes or focus_class is not None:
            score = 0.55 * score + 0.45 * focus_rec
        print(
            f"Epoch {epoch:3d}/{epochs} | loss {train_loss:.4f} | val_acc {acc:.3f} | "
            f"macro_f1 {macro:.3f} | recall {recalls}"
        )
        if score >= best_score:
            best_score = score
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(str(out_path), model, arch=arch, classes=classes, input_size=input_size)

    print(f"Bester Score {best_score:.3f} | Checkpoint: {out_path.resolve()}")
