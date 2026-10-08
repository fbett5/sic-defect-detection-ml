"""Box utilities. Boxes are (x1, y1, x2, y2) in pixels."""
from __future__ import annotations

import numpy as np


def iou_matrix(a, b) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(iou: np.ndarray, scores: np.ndarray, thr: float,
                 allowed: np.ndarray | None = None) -> dict[int, int]:
    """COCO-style matching: predictions in descending score order each take the
    highest-IoU still-unmatched ground truth with IoU >= thr. Returns {pred: gt}.

    allowed: optional (N,M) bool mask (e.g. same class only).
    """
    out, taken = {}, set()
    for p in np.argsort(-np.asarray(scores), kind="stable"):
        best, best_iou = -1, thr
        for g in range(iou.shape[1]):
            if g in taken or (allowed is not None and not allowed[p, g]):
                continue
            if iou[p, g] >= best_iou:
                best, best_iou = g, iou[p, g]
        if best >= 0:
            out[int(p)] = best
            taken.add(best)
    return out
