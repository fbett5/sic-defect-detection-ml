"""Drawing helpers: boxes, TP/FP/FN overlays, galleries, PR curves."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Colour-blind-safe categorical palette (Okabe-Ito + extras), BGR for OpenCV
_PALETTE_RGB = [(0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167), (86, 180, 233),
                (213, 94, 0), (240, 228, 66), (120, 94, 240), (100, 100, 100), (0, 0, 0)]
OUTCOME_BGR = {"TP": (60, 170, 30), "FP": (40, 40, 230), "FN": (0, 165, 255), "GT": (60, 170, 30),
               "MISCLS": (200, 60, 200)}


def class_colors(n: int, rgb01: bool = False):
    out = []
    for i in range(n):
        r, g, b = _PALETTE_RGB[i % len(_PALETTE_RGB)]
        out.append((r / 255, g / 255, b / 255) if rgb01 else (b, g, r))
    return out


def _label(img, text, x, y, color, scale=0.45):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y0 = max(int(y) - th - 4, 0)
    cv2.rectangle(img, (int(x), y0), (int(x) + tw + 4, y0 + th + 4), color, -1)
    cv2.putText(img, text, (int(x) + 2, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)


def draw_boxes(img, items, classes, color=None, thickness=2, prefix=""):
    """items: [(xyxy, class_id, confidence or None)]"""
    img = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cols = class_colors(len(classes))
    for b, c, conf in items:
        x1, y1, x2, y2 = map(int, b)
        col = color or cols[c]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thickness)
        txt = prefix + classes[c] + (f" {conf:.2f}" if conf is not None else "")
        _label(img, txt, x1, y1, col)
    return img


def draw_outcomes(img, matched, classes):
    """Overlay evaluation outcomes for one image.

    matched: dicts from evaluate.match_image(): {"kind": TP|FP|FN|MISCLS, "box", "cls", "conf", "gt_cls"}
    TP green, FP red, FN (missed) orange dashed, MISCLS (right place, wrong class) purple.
    """
    img = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for m in matched:
        x1, y1, x2, y2 = map(int, m["box"])
        col = OUTCOME_BGR[m["kind"]]
        if m["kind"] == "FN":
            for i in range(x1, x2, 8):
                cv2.line(img, (i, y1), (min(i + 4, x2), y1), col, 2)
                cv2.line(img, (i, y2), (min(i + 4, x2), y2), col, 2)
            for j in range(y1, y2, 8):
                cv2.line(img, (x1, j), (x1, min(j + 4, y2)), col, 2)
                cv2.line(img, (x2, j), (x2, min(j + 4, y2)), col, 2)
            _label(img, f"MISSED {classes[m['cls']]}", x1, y1, col)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            if m["kind"] == "MISCLS":
                txt = f"{classes[m['cls']]}? (is {classes[m['gt_cls']]}) {m['conf']:.2f}"
            else:
                txt = f"{m['kind']} {classes[m['cls']]} {m['conf']:.2f}"
            _label(img, txt, x1, y1, col)
    return img


def legend_strip(width: int) -> np.ndarray:
    strip = np.full((26, width, 3), 255, np.uint8)
    x = 6
    for k, t in (("TP", "correct"), ("FP", "false alarm"), ("FN", "missed"), ("MISCLS", "wrong class")):
        cv2.rectangle(strip, (x, 6), (x + 14, 20), OUTCOME_BGR[k], -1)
        cv2.putText(strip, t, (x + 18, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        x += 30 + 8 * len(t)
    return strip


def grid(tiles: list[np.ndarray], cols: int = 4, tile: int = 320, captions: list[str] | None = None) -> np.ndarray:
    out = []
    for i, t in enumerate(tiles):
        t = cv2.resize(t, (tile, tile))
        if captions:
            cap = np.full((22, tile, 3), 255, np.uint8)
            cv2.putText(cap, captions[i][:48], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            t = np.vstack([t, cap])
        out.append(t)
    if not out:
        return np.full((40, tile, 3), 255, np.uint8)
    while len(out) % cols:
        out.append(np.full_like(out[0], 255))
    return np.vstack([np.hstack(out[i:i + cols]) for i in range(0, len(out), cols)])


def crop_with_context(img, box, pad: float = 0.5, min_side: int = 24):
    """Crop around a box with context padding (fraction of box size), clipped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, min_side), max(y2 - y1, min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = bw * (1 + pad) / 2, bh * (1 + pad) / 2
    a, b = int(max(0, cx - hw)), int(max(0, cy - hh))
    c, d = int(min(w, cx + hw)), int(min(h, cy + hh))
    return img[b:d, a:c]


def save(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
