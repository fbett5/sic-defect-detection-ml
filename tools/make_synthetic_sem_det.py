"""Synthetic SEM-like *detection* set: patterned die + defects + boxes + a reference image.

Each sample is a 640x640 grayscale "SEM" image of a metal line/space pattern with
pads, plus a reference image of the same location on a good die (same pattern,
new noise, up to 1 px registration offset: like die-to-die inspection). Defects:

  particle   bright charging particle (dirt)
  scratch    dark groove with a lit edge, a finite segment
  nanowire   thin bright curly wire(s) lying across the pattern
  bridge     metal connecting two neighbouring lines (a short)
  open       a gap cut through a line

Written as a YOLO export with references, ready for:
  python tools/make_synthetic_sem_det.py --out data/raw/synthetic_sem_det --n 300
  python prepare_detection.py --dataset yolo --src data/raw/synthetic_sem_det --out data/det/synthetic_sem \\
      --views gray refdiff

It exists to show the pipeline on SEM-style images and to test it end to end. It is not
a replacement for real labelled SEM data: real defects are far more varied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["particle", "scratch", "nanowire", "bridge", "open"]
S = 640


def pattern(rng):
    """Metal lines (bright) on dielectric (dark) with a few pads; returns mask + line geometry."""
    m = np.zeros((S, S), np.uint8)
    horizontal = rng.random() < 0.5
    pitch = int(rng.integers(36, 64))
    width = int(pitch * rng.uniform(0.35, 0.55))
    off = int(rng.integers(0, pitch))
    lines = []
    for p in range(off, S, pitch):
        a, b = p, min(p + width, S)
        if horizontal:
            m[a:b] = 1
        else:
            m[:, a:b] = 1
        lines.append((a, b))
    for _ in range(rng.integers(1, 4)):  # pads / vias
        cx, cy, r = int(rng.integers(60, S - 60)), int(rng.integers(60, S - 60)), int(rng.integers(14, 26))
        cv2.rectangle(m, (cx - r, cy - r), (cx + r, cy + r), 1, -1)
    return m, lines, horizontal, pitch


def render(mask, rng, shift=(0, 0)):
    """SEM look: bright metal, edge brightening, texture, shot noise."""
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    mk = cv2.warpAffine(mask.astype(np.float32), M, (S, S), flags=cv2.INTER_NEAREST)
    edges = cv2.morphologyEx(mk, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    tex = cv2.resize(rng.normal(0, 6, (S // 16, S // 16)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)
    img = 70 + 80 * mk + 60 * edges + tex
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img + rng.normal(0, rng.uniform(5, 10), (S, S))


def add_particle(img, mask, rng):
    c = np.array([rng.integers(40, S - 40), rng.integers(40, S - 40)])
    pts = (c + rng.normal(0, rng.uniform(5, 14), (9, 2))).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillPoly(img, [hull], float(rng.uniform(200, 245)))
    cv2.polylines(img, [hull], True, 252.0, 2)
    x, y, w, h = cv2.boundingRect(hull)
    return (x - 2, y - 2, x + w + 2, y + h + 2)


def add_scratch(img, mask, rng):
    p0 = np.array([rng.integers(40, S - 40), rng.integers(40, S - 40)], float)
    ang = rng.uniform(0, np.pi)
    ln = rng.uniform(80, 260)
    p1 = np.clip(p0 + ln * np.array([np.cos(ang), np.sin(ang)]), 5, S - 5)
    t = int(rng.integers(2, 5))
    a, b = tuple(p0.astype(int)), tuple(p1.astype(int))
    cv2.line(img, a, b, 35.0, t)
    cv2.line(img, (a[0] + 2, a[1] + 2), (b[0] + 2, b[1] + 2), 210.0, 1)
    return (min(a[0], b[0]) - t - 3, min(a[1], b[1]) - t - 3, max(a[0], b[0]) + t + 3, max(a[1], b[1]) + t + 3)


def add_nanowire(img, mask, rng):
    x, y = rng.uniform(60, S - 60), rng.uniform(60, S - 60)
    ang = rng.uniform(0, 2 * np.pi)
    pts = []
    for _ in range(rng.integers(12, 30)):
        ang += rng.normal(0, 0.2)
        x, y = np.clip(x + 5 * np.cos(ang), 2, S - 3), np.clip(y + 5 * np.sin(ang), 2, S - 3)
        pts.append((int(x), int(y)))
    arr = np.array(pts, np.int32)
    cv2.polylines(img, [arr], False, float(rng.uniform(215, 250)), 1)
    x1, y1 = arr.min(0)
    x2, y2 = arr.max(0)
    return (x1 - 3, y1 - 3, x2 + 3, y2 + 3)


def add_bridge(img, mask, rng, lines, horizontal, pitch):
    if len(lines) < 2:
        return None
    k = int(rng.integers(0, len(lines) - 1))
    a, b = lines[k][1], lines[k + 1][0]
    if b - a < 3:
        return None
    pos = int(rng.integers(30, S - 30))
    w = int(rng.integers(4, 12))
    if horizontal:
        x1, y1, x2, y2 = pos, a - 2, pos + w, b + 2
    else:
        x1, y1, x2, y2 = a - 2, pos, b + 2, pos + w
    cv2.rectangle(img, (x1, y1), (x2, y2), 160.0, -1)
    return (x1 - 4, y1 - 4, x2 + 4, y2 + 4)


def add_open(img, mask, rng, lines, horizontal, pitch):
    a, b = lines[int(rng.integers(0, len(lines)))]
    if b - a < 4:
        return None
    pos = int(rng.integers(30, S - 30))
    g = int(rng.integers(4, 10))
    if horizontal:
        x1, y1, x2, y2 = pos, a - 1, pos + g, b + 1
    else:
        x1, y1, x2, y2 = a - 1, pos, b + 1, pos + g
    cv2.rectangle(img, (x1, y1), (x2, y2), 72.0, -1)
    return (x1 - 4, y1 - 4, x2 + 4, y2 + 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/synthetic_sem_det")
    ap.add_argument("--n", type=int, default=300, help="number of images")
    ap.add_argument("--clean-share", type=float, default=0.1, help="share of images without defects")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for d in ("images", "labels", "references"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    counts = np.zeros(len(CLASSES), int)
    for i in range(args.n):
        mask, lines, horiz, pitch = pattern(rng)
        img = render(mask, rng)
        ref = render(mask, rng, shift=(int(rng.integers(-1, 2)), int(rng.integers(-1, 2))))
        boxes = []
        n_def = 0 if rng.random() < args.clean_share else int(rng.integers(1, 5))
        for _ in range(n_def):
            c = int(rng.integers(0, len(CLASSES)))
            fn = [add_particle, add_scratch, add_nanowire, add_bridge, add_open][c]
            b = fn(img, mask, rng, lines, horiz, pitch) if c >= 3 else fn(img, mask, rng)
            if b is None:
                continue
            x1, y1, x2, y2 = (max(0, b[0]), max(0, b[1]), min(S, b[2]), min(S, b[3]))
            boxes.append((c, (x1, y1, x2, y2)))
            counts[c] += 1
        sample = f"die{i // 4:03d}_{i % 4}"  # 4 shots per "die": they stay in one split
        cv2.imwrite(str(out / "images" / f"{sample}.png"), np.clip(img, 0, 255).astype(np.uint8))
        cv2.imwrite(str(out / "references" / f"{sample}.png"), np.clip(ref, 0, 255).astype(np.uint8))
        lab = [f"{c} {(x1 + x2) / 2 / S:.6f} {(y1 + y2) / 2 / S:.6f} {(x2 - x1) / S:.6f} {(y2 - y1) / S:.6f}"
               for c, (x1, y1, x2, y2) in boxes]
        (out / "labels" / f"{sample}.txt").write_text("\n".join(lab) + ("\n" if lab else ""))
    print(f"Wrote {args.n} images to {out}: " + ", ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))


if __name__ == "__main__":
    main()
