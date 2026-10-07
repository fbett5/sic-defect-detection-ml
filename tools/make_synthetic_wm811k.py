"""Write a small fake LSWMD.pkl with the same schema as the real one.

Use it to check the whole pipeline runs before downloading the real data:
  python tools/make_synthetic_wm811k.py --out data/raw/LSWMD_synthetic.pkl
  python prepare_wm811k.py --pkl data/raw/LSWMD_synthetic.pkl --out data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"]


def wafer(pattern: str, size: int, rng) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    r = np.hypot(yy - c, xx - c) / (size / 2)
    on = r <= 1.0
    fail = rng.random((size, size)) < 0.03
    ang = np.arctan2(yy - c, xx - c)
    a0 = rng.uniform(-np.pi, np.pi)
    if pattern == "Center":
        fail |= r < 0.35
    elif pattern == "Donut":
        fail |= (r > 0.35) & (r < 0.6)
    elif pattern == "Edge-Ring":
        fail |= r > 0.85
    elif pattern == "Edge-Loc":
        fail |= (r > 0.75) & (np.abs(np.angle(np.exp(1j * (ang - a0)))) < 0.5)
    elif pattern == "Loc":
        cy, cx = rng.uniform(0.3, 0.7, 2) * size
        fail |= np.hypot(yy - cy, xx - cx) < size * 0.12
    elif pattern == "Random":
        fail |= rng.random((size, size)) < 0.25
    elif pattern == "Scratch":
        d = np.abs((yy - c) * np.cos(a0) - (xx - c) * np.sin(a0))
        fail |= (d < 1.2) & (r < 0.8)
    elif pattern == "Near-full":
        fail |= rng.random((size, size)) < 0.9
    m = np.where(on, np.where(fail, 2, 1), 0).astype(np.uint8)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/LSWMD_synthetic.pkl")
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # roughly the real imbalance, but enough rare examples to learn from
    probs = np.array([0.5, 0.07, 0.05, 0.08, 0.08, 0.07, 0.05, 0.05, 0.05])
    rows = []
    for i in range(args.n):
        lot = f"lot{i // 25}"
        if rng.random() < 0.2:  # unlabeled wafer, as in the real file
            ft, size = np.array([]), int(rng.integers(26, 60))
            m = wafer("none", size, rng)
        else:
            cls = CLASSES[rng.choice(len(CLASSES), p=probs / probs.sum())]
            size = int(rng.integers(26, 60))
            m = wafer(cls, size, rng)
            ft = np.array([[cls]], dtype=object)
        rows.append({"waferMap": m, "dieSize": float(m.size), "lotName": lot,
                     "waferIndex": float(i % 25 + 1),
                     "trianTestLabel": np.array([["Training"]], dtype=object) if ft.size else np.array([]),
                     "failureType": ft})
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out)
    print(f"Wrote {len(df)} wafers ({(df.failureType.map(len) > 0).sum()} labeled) to {args.out}")


if __name__ == "__main__":
    main()
