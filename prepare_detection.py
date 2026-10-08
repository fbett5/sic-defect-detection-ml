"""Step 1 of the detection track: build a curated, documented detection dataset.

Examples:
  # DeepPCB (public, PCB AOI, 6 defect types, defect-free reference per image)
  git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/raw/DeepPCB
  python prepare_detection.py --dataset deeppcb --src data/raw/DeepPCB --out data/det/deeppcb

  # MVTec AD: defect masks become boxes (download MVTec AD first, see README)
  python prepare_detection.py --dataset mvtec --src data/raw/MVTecAD --out data/det/mvtec --categories grid tile carpet

  # Your own labelled images (CVAT / Label Studio / Roboflow "YOLO" export)
  python prepare_detection.py --dataset yolo --src my_sem_export --out data/det/my_sem

Writes images/labels/manifest in a fixed layout, baseline statistics, plots, a
DATASET_CARD.md, and the model-input views for the preprocessing modes you ask for.
"""
from __future__ import annotations

import argparse
import json

from sicdefect.det.data import (build_view, convert_deeppcb, convert_mvtec, convert_yolo, dataset_stats,
                                write_card)
from sicdefect.det.preprocess import MODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["deeppcb", "mvtec", "yolo"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", nargs="*", help="MVTec categories (default: all)")
    ap.add_argument("--views", nargs="*", default=None, choices=MODES,
                    help="preprocessing views to build (default: gray, plus refdiff when references exist)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "deeppcb":
        ds = convert_deeppcb(args.src, args.out, seed=args.seed)
    elif args.dataset == "mvtec":
        ds = convert_mvtec(args.src, args.out, categories=args.categories, seed=args.seed)
    else:
        ds = convert_yolo(args.src, args.out, seed=args.seed)

    st = dataset_stats(ds)
    card = write_card(ds, st)
    views = args.views or (["gray", "refdiff"] if (ds / "references").exists() else ["gray"])
    for v in views:
        print("view", v, "->", build_view(ds, v))
    print(json.dumps({s: {k: st["splits"][s][k] for k in ("images", "boxes")} for s in st["splits"]}, indent=1))
    print(f"Dataset card: {card}")


if __name__ == "__main__":
    main()
