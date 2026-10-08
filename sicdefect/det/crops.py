"""Defect crops for the CNN stage.

The CNN sees one candidate region at a time: the box plus 50% context on each
side, resized to a fixed square. It predicts one of the defect classes or
"background" (not a defect), so in the two-stage pipeline it can both re-label
and reject YOLO boxes.

Background examples come from two sources:
  random  boxes of realistic size that don't touch any labelled defect
  hard    the detector's own false alarms on train/val images (``mine_hard_negatives``):
          the cases the CNN actually has to reject in the pipeline
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .boxes import iou_matrix
from .data import ground_truth, load_manifest
from .preprocess import make_input, read_gray
from .viz import crop_with_context

CROP = 64
PAD = 0.5


def model_input(ds: Path, split: str, img_id: str, mode: str) -> np.ndarray:
    """The preprocessed (HxWx3) image for one dataset entry."""
    view = ds / "views" / mode / "images" / split / f"{img_id}.png"
    if view.exists():
        return cv2.imread(str(view))
    img = read_gray(ds / "images" / split / f"{img_id}.png")
    ref_p = ds / "references" / split / f"{img_id}.png"
    return make_input(img, read_gray(ref_p) if ref_p.exists() else None, mode)


def cut(img: np.ndarray, box, size: int = CROP, pad: float = PAD) -> np.ndarray:
    c = crop_with_context(img, box, pad=pad)
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def random_background(gt_boxes, w, h, sizes, n, rng, max_iou=0.0, tries=200):
    out = []
    gt = np.array([b for _, b in gt_boxes]).reshape(-1, 4)
    for _ in range(tries):
        if len(out) >= n:
            break
        bw, bh = sizes[rng.integers(len(sizes))]
        x1, y1 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        b = (x1, y1, x1 + bw, y1 + bh)
        if not len(gt) or iou_matrix([b], gt).max() <= max_iou:
            out.append(b)
    return out


def build_crops(ds: str | Path, mode: str, splits=("train", "val", "test"), bg_per_image: int = 3,
                hard: dict | None = None, seed: int = 0):
    """Return {split: (X uint8 [N,64,64,3], y int64 [N], meta list)}; y == n_classes means background."""
    ds = Path(ds)
    from .data import load_classes

    n_cls = len(load_classes(ds))
    rng = np.random.default_rng(seed)
    rows = load_manifest(ds)
    out = {}
    for s in splits:
        gt = ground_truth(ds, s)
        sizes = [(b[2] - b[0], b[3] - b[1]) for v in gt.values() for _, b in v] or [(32, 32)]
        X, Y, meta = [], [], []
        for r in [r for r in rows if r["split"] == s]:
            img = model_input(ds, s, r["id"], mode)
            h, w = img.shape[:2]
            for c, b in gt[r["id"]]:
                X.append(cut(img, b)), Y.append(c), meta.append((r["id"], b, "defect"))
            for b in random_background(gt[r["id"]], w, h, sizes, bg_per_image, rng):
                X.append(cut(img, b)), Y.append(n_cls), meta.append((r["id"], b, "random_bg"))
            for b in (hard or {}).get(r["id"], []):
                X.append(cut(img, b)), Y.append(n_cls), meta.append((r["id"], b, "hard_bg"))
        out[s] = (np.stack(X), np.array(Y, dtype=np.int64), meta)
    return out


def mine_hard_negatives(ds: str | Path, mode: str, det_weights: str, splits=("train", "val"),
                        conf: float = 0.05, max_iou: float = 0.1, imgsz: int = 640, device="cpu") -> dict:
    """Detector boxes that hit no labelled defect -> {image_id: [box, ...]}."""
    from ultralytics import YOLO

    ds = Path(ds)
    model = YOLO(det_weights)
    out = {}
    for s in splits:
        gt = ground_truth(ds, s)
        for img_id, g in gt.items():
            img = model_input(ds, s, img_id, mode)
            r = model.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            if not len(boxes):
                continue
            gb = np.array([b for _, b in g]).reshape(-1, 4)
            keep = boxes if not len(gb) else boxes[iou_matrix(boxes, gb).max(1) <= max_iou]
            if len(keep):
                out[img_id] = [tuple(map(float, b)) for b in keep]
    return out


MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.25, 0.25, 0.25], np.float32)


def to_tensor(crops: np.ndarray) -> torch.Tensor:
    """uint8 [N,H,W,3] (BGR) -> float [N,3,H,W] normalised."""
    x = crops.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


class CropDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:  # dihedral: defects have no preferred orientation on the board
            x = np.rot90(x, np.random.randint(4))
            if np.random.rand() < 0.5:
                x = np.fliplr(x)
            x = np.ascontiguousarray(x)
        return to_tensor(x[None])[0], int(self.y[i])
