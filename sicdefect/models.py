"""Model construction, shared by training, export and inference.

Kept separate from ``train_cnn.py`` so that exporting or re-loading a model
does not pull in the training script and its argument parser.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision

MODELS = ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"]


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """A torchvision backbone with its classifier head resized to ``num_classes``."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
    weights = "DEFAULT" if pretrained else None
    m = getattr(torchvision.models, name)(weights=weights)
    if name.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name.startswith("efficientnet"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m
