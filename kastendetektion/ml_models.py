"""
Gemeinsame Modell-Definitionen für die ML-Wege von Stufe 1 (Slot) und Stufe 2 (Kronkorken).

Wird sowohl vom Training (``scripts/train_slot_cnn.py`` / ``train_cap_cnn.py``) als auch
von der Inferenz (``slot_classifier.py`` / ``cap_classifier.py``) genutzt, damit Architektur
und Checkpoint-Format an einer Stelle definiert sind.

Checkpoint-Format (``torch.save``)::

    {
        "arch": "smallcnn" | "mobilenet_v2",
        "input_size": 64,
        "classes": ["...", ...],   # Reihenfolge = Modell-Index
        "state_dict": <state_dict>,
    }

Torch wird nur bei Bedarf importiert, damit der klassische Pfad ohne PyTorch läuft.
"""

from __future__ import annotations

from typing import Any

INPUT_SIZE_DEFAULT = 64


def build_model(arch: str, num_classes: int) -> Any:
    """Erzeugt ein untrainiertes Modell der angegebenen Architektur."""
    import torch.nn as nn  # lazy import

    arch = arch.lower()
    if arch == "smallcnn":
        return _SmallCNN(num_classes)
    if arch == "mobilenet_v2":
        from torchvision import models

        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unbekannte Architektur: {arch!r} (erlaubt: smallcnn, mobilenet_v2)")


def _SmallCNN(num_classes: int) -> Any:
    """Kompaktes CNN für kleine ROIs (wenig Daten, CPU-tauglich)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
    )


def save_checkpoint(
    path: str,
    model: Any,
    arch: str,
    classes: list[str],
    input_size: int = INPUT_SIZE_DEFAULT,
) -> None:
    """Speichert ein Modell im einheitlichen Checkpoint-Format."""
    import torch

    torch.save(
        {
            "arch": arch,
            "input_size": int(input_size),
            "classes": list(classes),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu") -> tuple[Any, list[str], int]:
    """
    Lädt einen Checkpoint und gibt ``(model_eval, classes, input_size)`` zurück.

    Das Modell ist im ``eval``-Modus und auf ``device`` verschoben.
    """
    import torch

    ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("arch", "smallcnn")
    classes = list(ckpt.get("classes", []))
    input_size = int(ckpt.get("input_size", INPUT_SIZE_DEFAULT))
    model = build_model(arch, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, classes, input_size
