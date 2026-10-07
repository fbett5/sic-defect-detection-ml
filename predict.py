"""Classify wafer maps with a trained CNN checkpoint and list likely root causes.

Examples:
  # a few test-split wafers from the processed file
  python predict.py --ckpt outputs/resnet18-weighted_sampler/best.pt --npz data/processed/wm811k_64.npz --n 10

  # a single wafer map saved as .npy (values 0/1/2, any size)
  python predict.py --ckpt outputs/resnet18-weighted_sampler/best.pt --npy my_wafer.npy
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from sicdefect.rootcause import likely_causes
from sicdefect.utils import get_device
from sicdefect.wm811k import CLASSES, encode, resize_map
from train_cnn import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", help="processed file; predicts on test-split wafers")
    ap.add_argument("--npy", help="single wafer map (.npy, values 0/1/2)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False).to(device).eval()
    model.load_state_dict(ck["model"])

    if args.npy:
        maps, truth = resize_map(np.load(args.npy), args.size)[None], [None]
    elif args.npz:
        d = np.load(args.npz, allow_pickle=True)
        idx = np.flatnonzero(d["split"] == "test")[: args.n]
        maps, truth = d["maps"][idx], [CLASSES[i] for i in d["y"][idx]]
    else:
        raise SystemExit("Pass --npz or --npy")

    x = torch.from_numpy(encode(maps)).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1).cpu().numpy()

    for p, t in zip(probs, truth):
        k = int(p.argmax())
        pred = ck["classes"][k]
        head = f"pred={pred} ({p[k]:.2f})" + (f"  true={t}" if t else "")
        print(head)
        for c in likely_causes(pred):
            print(f"    - {c}")


if __name__ == "__main__":
    main()
