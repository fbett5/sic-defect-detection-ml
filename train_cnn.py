"""Step 2: train a CNN classifier on WM-811K and log everything to MLflow.

Examples:
  python train_cnn.py                                    # resnet18, weighted sampler
  python train_cnn.py --model efficientnet_b3 --imbalance focal --pretrained
  python train_cnn.py --model resnet50 --imbalance class_weights --epochs 30
  python train_cnn.py --limit 2000 --epochs 1            # quick smoke test

Imbalance strategies (run each as an ablation and compare in MLflow):
  none            plain cross-entropy
  weighted_sampler  oversample rare classes in each batch
  class_weights   inverse-frequency weighted cross-entropy
  focal           focal loss (gamma=2) + class weights

Model selection uses val macro-F1. The test split is evaluated once, with the
best checkpoint, at the end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from sicdefect.losses import FocalLoss
from sicdefect.metrics import classification_metrics, plot_confusion, report
from sicdefect.utils import get_device, seed_everything, setup_mlflow
from sicdefect.wm811k import CLASSES, WaferDataset, class_weights, load_processed, make_weighted_sampler

MODELS = ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"]


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    m = getattr(torchvision.models, name)(weights=weights)
    if name.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name.startswith("efficientnet"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def subsample(idx: np.ndarray, labels: np.ndarray, limit: int, rng) -> np.ndarray:
    """Random subsample for quick runs (class mix preserved on average)."""
    if limit <= 0 or len(idx) <= limit:
        return idx
    return np.sort(rng.choice(idx, size=limit, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/wm811k_64.npz")
    ap.add_argument("--model", default="resnet18", choices=MODELS)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet weights (downloads once)")
    ap.add_argument("--imbalance", default="weighted_sampler",
                    choices=["none", "weighted_sampler", "class_weights", "focal"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5, help="early stop after N epochs without val macro-F1 gain")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="cap train set size (0 = all), for quick tests")
    ap.add_argument("--experiment", default="wm811k-cnn")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    rng = np.random.default_rng(args.seed)

    d = load_processed(args.data)
    maps, y, split = d["maps"], d["y"], d["split"]
    tr = subsample(np.flatnonzero(split == "train"), y, args.limit, rng)
    va = subsample(np.flatnonzero(split == "val"), y, args.limit // 4 if args.limit else 0, rng)
    te = np.flatnonzero(split == "test")
    if args.limit:
        te = subsample(te, y, args.limit // 4, rng)
    print(f"device={device}  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    train_ds = WaferDataset(maps[tr], y[tr], augment=True)
    val_ds = WaferDataset(maps[va], y[va])
    test_ds = WaferDataset(maps[te], y[te])

    pin = device.type == "cuda"
    if args.imbalance == "weighted_sampler":
        train_loader = DataLoader(train_ds, args.batch_size, sampler=make_weighted_sampler(y[tr]),
                                  num_workers=args.workers, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, args.batch_size * 2, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, args.batch_size * 2, num_workers=args.workers, pin_memory=pin)

    cw = class_weights(y[tr]).to(device)
    if args.imbalance == "class_weights":
        criterion = nn.CrossEntropyLoss(weight=cw)
    elif args.imbalance == "focal":
        criterion = FocalLoss(gamma=2.0, weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    model = build_model(args.model, len(CLASSES), args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_name = args.run_name or f"{args.model}-{args.imbalance}"
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best.pt"

    mlflow = setup_mlflow(args.experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**vars(args), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                           "device": str(device)})
        best_f1, bad_epochs = -1.0, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, total, n = time.time(), 0.0, 0
            for x, yb in train_loader:
                x, yb = x.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(x), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(yb)
                n += len(yb)
            sched.step()

            yv, pv = predict(model, val_loader, device)
            m = classification_metrics(yv, pv)
            mlflow.log_metrics({"train_loss": total / max(n, 1), "lr": sched.get_last_lr()[0],
                                **{f"val_{k}": v for k, v in m.items()}}, step=epoch)
            print(f"epoch {epoch:3d}  loss {total / max(n, 1):.4f}  val macro-F1 {m['macro_f1']:.4f}  "
                  f"bal-acc {m['balanced_acc']:.4f}  ({time.time() - t0:.0f}s)")

            if m["macro_f1"] > best_f1:
                best_f1, bad_epochs = m["macro_f1"], 0
                torch.save({"model": model.state_dict(), "arch": args.model, "classes": CLASSES,
                            "epoch": epoch, "val_macro_f1": best_f1}, ckpt)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping (no val gain for {args.patience} epochs)")
                    break

        # Final test evaluation with the best checkpoint
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        yt, pt = predict(model, test_loader, device)
        tm = classification_metrics(yt, pt)
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})
        mlflow.log_metric("best_val_macro_f1", best_f1)

        rep = report(yt, pt)
        (out_dir / "test_report.txt").write_text(rep)
        (out_dir / "test_metrics.json").write_text(json.dumps(tm, indent=2))
        plot_confusion(yt, pt, out_dir / "test_confusion.png", title=f"{run_name} — test")
        for f in ("test_report.txt", "test_metrics.json", "test_confusion.png", "best.pt"):
            mlflow.log_artifact(str(out_dir / f))

        print("\nTEST\n" + rep)
        print(f"test macro-F1 {tm['macro_f1']:.4f}  balanced acc {tm['balanced_acc']:.4f}  "
              f"defect recall {tm['defect_recall']:.4f}")
        print(f"Checkpoint + report: {out_dir}")


if __name__ == "__main__":
    main()
