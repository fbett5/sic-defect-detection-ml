"""Make a small fake SEM dataset to test train_sem.py / predict_sem.py end to end.

Not a substitute for real data: the "artifacts" are simple drawn shapes on a
noisy textured background, with an SEM-style info bar at the bottom.

  python tools/make_synthetic_sem.py --out data/synthetic_sem --n 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def background(rng, h=480, w=640):
    base = rng.normal(110, 8, (h // 8, w // 8)).astype(np.float32)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    img += rng.normal(0, rng.uniform(6, 14), (h, w))  # shot noise
    return img


def dirt(img, rng):
    for _ in range(rng.integers(1, 4)):
        c = (int(rng.integers(60, img.shape[1] - 60)), int(rng.integers(60, img.shape[0] - 90)))
        pts = (np.array(c) + rng.normal(0, rng.uniform(10, 35), (9, 2))).astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(img, [hull], float(rng.uniform(180, 240)))   # charging particle: bright
        cv2.polylines(img, [hull], True, 250.0, 2)


def scratch(img, rng):
    h, w = img.shape
    p0 = np.array([rng.integers(0, w), rng.integers(0, h - 60)])
    ang = rng.uniform(0, np.pi)
    p1 = (p0 + 900 * np.array([np.cos(ang), np.sin(ang)])).astype(int)
    p0 = (p0 - 900 * np.array([np.cos(ang), np.sin(ang)])).astype(int)
    t = int(rng.integers(2, 6))
    cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 45.0, t)  # groove: dark
    cv2.line(img, tuple(int(v) + 2 for v in p0), tuple(int(v) + 2 for v in p1), 200.0, 1)  # lit edge


def nanowire(img, rng):
    h, w = img.shape
    for _ in range(rng.integers(3, 9)):
        x, y = rng.uniform(0, w), rng.uniform(0, h - 60)
        ang = rng.uniform(0, 2 * np.pi)
        pts = []
        for _ in range(rng.integers(20, 50)):
            ang += rng.normal(0, 0.15)
            x, y = x + 6 * np.cos(ang), y + 6 * np.sin(ang)
            pts.append((int(x), int(y)))
        cv2.polylines(img, [np.array(pts, np.int32)], False, float(rng.uniform(200, 250)), 1)


def info_bar(img, rng):
    h, w = img.shape
    img[h - 40:] = 0
    cv2.putText(img, f"{rng.integers(2, 20)}.0kV  x{rng.integers(5, 50)}k  SE", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
    cv2.line(img, (w - 140, h - 18), (w - 40, h - 18), 255, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_sem")
    ap.add_argument("--n", type=int, default=60, help="images per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    makers = {"clean": None, "dirt": dirt, "scratch": scratch, "nanowire": nanowire}
    for cls, fn in makers.items():
        d = Path(args.out) / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            img = background(rng)
            if fn:
                fn(img, rng)
            img = np.clip(img, 0, 255).astype(np.uint8)
            info_bar(img, rng)
            if i % 3 == 0:  # some 16-bit TIFFs, like many SEM exports
                cv2.imwrite(str(d / f"{cls}_{i:03d}.tif"), img.astype(np.uint16) * 257)
            else:
                cv2.imwrite(str(d / f"{cls}_{i:03d}.png"), img)
    print(f"Wrote {args.n * len(makers)} images to {args.out}")


if __name__ == "__main__":
    main()
