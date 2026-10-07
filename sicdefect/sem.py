"""SEM image classification: loading, preprocessing and the model bundle.

Expected folder layout (one sub-folder per class, any names you like):

    SEM-artifacts/
        dirt/       img001.tif  img002.png ...
        scratch/    ...
        nanowire/   ...
        clean/      ...   (optional, but recommended: "nothing wrong" examples)

Supported files: .tif/.tiff (8- or 16-bit), .png, .jpg, .bmp.

Preprocessing, identical at training and prediction time:
  1. read as grayscale (16-bit is contrast-stretched to 8-bit, 0.5-99.5 percentile)
  2. optionally crop the SEM info bar (the strip with kV / mag / scale bar)
  3. resize to img_size x img_size
  4. scale to 0-1, repeat to 3 channels, ImageNet mean/std normalisation
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODELS = ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny"]


# --------------------------------------------------------------------------- data
def scan_folder(root: str | Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """Return (files, labels, class_names) from a one-folder-per-class layout."""
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(classes) < 2:
        raise SystemExit(f"{root} needs at least 2 class sub-folders, found {classes}")
    files, labels = [], []
    for i, c in enumerate(classes):
        fs = sorted(f for f in (root / c).rglob("*") if f.suffix.lower() in IMG_EXTS)
        files += fs
        labels += [i] * len(fs)
    return files, np.array(labels, dtype=np.int64), classes


def list_images(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)


def load_gray(path: str | Path, crop_bottom: float = 0.0) -> np.ndarray:
    """Read any SEM image as 8-bit grayscale, with the bottom info bar removed."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, (0.5, 99.5))
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    if crop_bottom > 0:
        img = img[: max(1, int(round(img.shape[0] * (1 - crop_bottom))))]
    return img


def preprocess(img: np.ndarray, img_size: int) -> torch.Tensor:
    """8-bit grayscale -> normalised 3x(img_size)^2 float tensor."""
    import cv2

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = np.repeat((img.astype(np.float32) / 255.0)[None], 3, axis=0)
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    return torch.from_numpy(x)


def augment(img: np.ndarray) -> np.ndarray:
    """SEM-safe augmentation.

    Rotations/flips: a particle or scratch has no preferred orientation on the die.
    Brightness/contrast/gamma: detector, kV and working distance change the grey levels.
    Mild noise: SEM shot noise varies with scan speed and dwell time.
    A random crop-and-zoom mimics small magnification changes.
    No colour jitter (SEM is grayscale) and no heavy warps (they change the defect shape).
    """
    img = np.rot90(img, random.randint(0, 3))
    if random.random() < 0.5:
        img = np.fliplr(img)
    h, w = img.shape
    if random.random() < 0.5:
        s = random.uniform(0.8, 1.0)
        ch, cw = int(h * s), int(w * s)
        y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
        img = img[y0:y0 + ch, x0:x0 + cw]
    x = img.astype(np.float32) / 255.0
    x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)
    x = x ** random.uniform(0.8, 1.25)
    if random.random() < 0.3:
        x = np.clip(x + np.random.normal(0, random.uniform(0.01, 0.04), x.shape), 0, 1)
    return np.ascontiguousarray((x * 255).astype(np.uint8))


class SEMDataset(Dataset):
    def __init__(self, files, labels, img_size=224, crop_bottom=0.0, train=False, cache=True):
        self.files, self.labels = list(files), np.asarray(labels)
        self.img_size, self.crop_bottom, self.train = img_size, crop_bottom, train
        # SEM images are often 1-4 MP; cache a downsized copy so epochs stay fast
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.files)

    def image(self, i) -> np.ndarray:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        import cv2

        img = load_gray(self.files[i], self.crop_bottom)
        side = int(self.img_size * 1.5)  # keep headroom for the random crop
        if min(img.shape) > side:
            f = side / min(img.shape)
            img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        if self._cache is not None:
            self._cache[i] = img
        return img

    def __getitem__(self, i):
        img = self.image(i)
        if self.train:
            img = augment(img)
        return preprocess(img, self.img_size), int(self.labels[i])


def stratified_split(labels: np.ndarray, val: float, test: float, seed: int) -> np.ndarray:
    """Per-class random split so every class appears in train/val/test."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(labels), dtype=object)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_te = max(1, round(len(idx) * test)) if len(idx) >= 3 else 0
        n_va = max(1, round(len(idx) * val)) if len(idx) >= 3 else 0
        split[idx[:n_te]] = "test"
        split[idx[n_te:n_te + n_va]] = "val"
        split[idx[n_te + n_va:]] = "train"
    return split


# --------------------------------------------------------------------------- model
def build(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    m = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch.startswith(("efficientnet", "convnext")):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch {arch}; choose from {MODELS}")
    return m


def save_bundle(path, model, arch, classes, img_size, crop_bottom, extra=None):
    """Everything needed to use the model later, in one file."""
    torch.save({"model": model.state_dict(), "arch": arch, "classes": list(classes),
                "img_size": img_size, "crop_bottom": crop_bottom, "task": "sem",
                **(extra or {})}, path)


def load_bundle(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def metrics(y_true, y_pred, classes) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    labels = list(range(len(classes)))
    out = {"macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
           "balanced_acc": balanced_accuracy_score(y_true, y_pred),
           "accuracy": accuracy_score(y_true, y_pred)}
    for c, f in zip(classes, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)):
        out[f"f1_{c}"] = f
    return {k: float(v) for k, v in out.items()}
