"""Classify new SEM images with a model trained by train_sem.py.

Examples:
  python predict_sem.py --model outputs/sem-resnet18/model.pt --input new_images/
  python predict_sem.py --model model.pt --input img_0042.tif --heatmaps
  python predict_sem.py --model model.pt --input lot_A/ --threshold 0.7 --out lot_A_results

Writes <out>/predictions.csv (file, predicted class, confidence, top-3, needs_review)
and <out>/gallery.png. With --heatmaps it also saves a Grad-CAM overlay per image
showing which region drove the decision, so you can check the model is looking
at the particle / scratch and not at the scale bar or a charging edge.

Images below --threshold confidence are marked needs_review=yes: send those to a
human. Anything that isn't one of the trained classes (a new artifact type) will
usually land there too, but not always, so spot-check results on new tools/products.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from sicdefect.sem import list_images, load_bundle, load_gray, preprocess
from sicdefect.utils import get_device


class GradCAM:
    """Grad-CAM on the last convolutional block."""

    def __init__(self, model, arch):
        layer = model.layer4 if arch.startswith("resnet") else model.features[-1]
        self.acts = self.grads = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))
        self.model = model

    def __call__(self, x, cls):
        self.model.zero_grad()
        out = self.model(x)
        out[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1))[0].detach().cpu().numpy()
        return cam / (cam.max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.pt from train_sem.py")
    ap.add_argument("--input", required=True, help="an image file or a folder of images")
    ap.add_argument("--threshold", type=float, default=0.7, help="below this confidence -> needs_review")
    ap.add_argument("--heatmaps", action="store_true", help="save Grad-CAM overlays")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import cv2

    device = get_device(args.device)
    model, ck = load_bundle(args.model, device)
    classes, size, crop = ck["classes"], ck["img_size"], ck["crop_bottom"]
    files = list_images(args.input)
    if not files:
        raise SystemExit(f"No images found in {args.input}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model, ck["arch"]) if args.heatmaps else None
    if cam:
        (out / "heatmaps").mkdir(exist_ok=True)

    rows, thumbs = [], []
    for f in files:
        img = load_gray(f, crop)
        x = preprocess(img, size).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0].cpu().numpy()
        top = np.argsort(p)[::-1][:3]
        k = int(top[0])
        review = p[k] < args.threshold
        rows.append({"file": str(f), "prediction": classes[k], "confidence": round(float(p[k]), 4),
                     "top3": "; ".join(f"{classes[i]} {p[i]:.2f}" for i in top),
                     "needs_review": "yes" if review else "no"})
        print(f"{f.name:40s} {classes[k]:15s} {p[k]:.2f}{'   <- review' if review else ''}")

        small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        if cam:
            heat = cam(x.clone().requires_grad_(True), k)
            heat = cv2.resize(heat, (size, size))
            color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), 0.6, color, 0.4, 0)
            cv2.imwrite(str(out / "heatmaps" / f"{f.stem}_{classes[k]}.png"),
                        np.hstack([cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), overlay]))
        thumbs.append((small, rows[-1]))

    with open(out / "predictions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gallery(thumbs[:48], out / "gallery.png")

    n_rev = sum(r["needs_review"] == "yes" for r in rows)
    counts = {c: sum(r["prediction"] == c for r in rows) for c in classes}
    print(f"\n{len(rows)} images: {counts}  ({n_rev} flagged for review)")
    print(f"Results: {out / 'predictions.csv'}   Gallery: {out / 'gallery.png'}")


def gallery(items, path, cols=6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, r) in zip(axes.flat, items):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{Path(r['file']).name[:22]}\n{r['prediction']} {r['confidence']:.2f}", fontsize=7,
                     color="red" if r["needs_review"] == "yes" else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
