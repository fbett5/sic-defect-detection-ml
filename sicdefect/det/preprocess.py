"""Turn a raw inspection image (and optionally a defect-free reference) into model input.

This is the only place preprocessing is defined. Dataset views for training and
the inference pipeline both call ``make_input``, so they can never drift apart.

Modes
-----
gray     the tested image alone, grayscale repeated to 3 channels.
refdiff  3 channels = [tested, reference, |tested - reference|].
         The model sees what *changed* versus a known-good image, the same idea as
         die-to-die / die-to-database comparison in wafer inspection or a golden-board
         comparison in PCB AOI. Needs an aligned reference image.
clahe    tested image with CLAHE local contrast equalisation, for SEM / optical images
         with uneven illumination or charging. No reference needed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODES = ("gray", "refdiff", "clahe")


def read_gray(path: str | Path) -> np.ndarray:
    """Any image (8/16-bit, gray/colour) -> 8-bit grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, (0.5, 99.5))
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return img


def make_input(img: np.ndarray, ref: np.ndarray | None = None, mode: str = "gray") -> np.ndarray:
    """Return an HxWx3 uint8 BGR image ready for YOLO."""
    if mode == "gray":
        return cv2.merge([img, img, img])
    if mode == "clahe":
        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        return cv2.merge([eq, eq, eq])
    if mode == "refdiff":
        if ref is None:
            raise ValueError("mode 'refdiff' needs a reference (defect-free) image")
        if ref.shape != img.shape:
            ref = cv2.resize(ref, (img.shape[1], img.shape[0]))
        diff = cv2.absdiff(img, ref)
        return cv2.merge([img, ref, diff])
    raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
