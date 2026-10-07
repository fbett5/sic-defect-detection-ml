"""Step 1: turn LSWMD.pkl into fixed-size arrays + lot-grouped splits.

Writes:
  data/processed/wm811k_<size>.npz   maps, labels, lots, split  (used by train_cnn.py)
  data/processed/yolo_<size>/{train,val,test}/<class>/*.png     (used by train_yolo.py)
  data/processed/split_summary.csv   class counts per split
  data/processed/samples.png         a grid of example wafers per class (EDA)

Both training scripts read the SAME split, so CNN vs YOLO comparisons are fair.

Usage:
  python prepare_wm811k.py                 # uses the path in sicdefect/paths.py
  python prepare_wm811k.py --pkl "path/to/LSWMD.pkl"   # or a folder containing it
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sicdefect.paths import resolve_pkl, wm811k_default
from sicdefect.wm811k import CLASSES, load_wm811k, lot_grouped_split, resize_map, to_png


def save_sample_grid(maps, labels, path, per_class=6, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.3, len(CLASSES) * 1.3))
    for r, c in enumerate(CLASSES):
        idx = np.flatnonzero(labels == r)
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False) if len(idx) else []
        for j in range(per_class):
            ax = axes[r, j]
            ax.axis("off")
            if j < len(pick):
                ax.imshow(maps[pick[j]], cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
        axes[r, 0].set_title(c, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=wm811k_default(),
                    help="LSWMD.pkl or the folder containing it (default: sicdefect/paths.py)")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=64, help="resize wafer maps to size x size")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-yolo", action="store_true", help="skip writing PNG folders for YOLO")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {resolve_pkl(args.pkl)} ...")
    df = load_wm811k(args.pkl)
    print(f"Labeled wafers: {len(df):,} across {df['lot'].nunique():,} lots")

    print(f"Resizing to {args.size}x{args.size} (nearest neighbour) ...")
    maps = np.stack([resize_map(m, args.size) for m in df["waferMap"].values])
    labels = df["y"].to_numpy()
    lots = df["lot"].to_numpy()

    split = lot_grouped_split(lots, args.val_frac, args.test_frac, args.seed)
    print("OK: no lot appears in more than one split")

    npz = out / f"wm811k_{args.size}.npz"
    np.savez_compressed(npz, maps=maps, y=labels, lot=lots, split=split, classes=np.array(CLASSES))
    print(f"Wrote {npz}")

    summary = (
        pd.DataFrame({"label": df["label"], "split": split})
        .value_counts().unstack(fill_value=0).reindex(CLASSES).fillna(0).astype(int)
    )
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(out / "split_summary.csv")
    print(summary.to_string())

    save_sample_grid(maps, labels, out / "samples.png")

    if not args.no_yolo:
        ydir = out / f"yolo_{args.size}"
        print(f"Writing PNG folders for YOLO to {ydir} ...")
        for s in ("train", "val", "test"):
            for c in CLASSES:
                (ydir / s / c).mkdir(parents=True, exist_ok=True)
        for i, (m, y, s) in enumerate(zip(maps, labels, split)):
            cv2.imwrite(str(ydir / s / CLASSES[y] / f"{i:06d}.png"), to_png(m))
        print("Done.")


if __name__ == "__main__":
    main()
