"""Fine-tune an ImageNet CNN on your own SEM images (dirt, scratch, nanowire, ...).

Folder layout: one sub-folder per class (see sicdefect/sem.py).

Examples:
  python train_sem.py --data "/content/drive/MyDrive/SEM-artifacts"
  python train_sem.py --data SEM-artifacts --model efficientnet_b0 --epochs 40 --crop-bottom 0.07
  python train_sem.py --data data/synthetic_sem --epochs 2         # quick smoke test

Training recipe (works with a few dozen to a few hundred images per class):
  - start from ImageNet weights (not from the WM-811K wafer-map model: 64 px
    pass/fail grids teach nothing useful about SEM texture, edges and charging)
  - first --freeze-epochs train only the new last layer, then unfreeze everything
    with a 10x smaller learning rate for the pretrained backbone
  - weighted sampling so rare classes are seen as often as common ones
  - SEM-safe augmentation (rotate/flip, brightness/contrast/gamma, noise, mild zoom)

Output, in outputs/<run-name>/:
  model.pt       the bundle to keep: weights + class names + preprocessing settings
                 (use with predict_sem.py on new images)
  model_ts.pt    TorchScript copy that loads with plain PyTorch, no repo code needed
  test_report.txt, test_metrics.json, test_confusion.png, split.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

from sicdefect.live import write_live
from sicdefect.metrics import plot_confusion, report
from sicdefect.sem import (MODELS, SEMDataset, build, metrics, save_bundle, scan_folder,
                           stratified_split)
from sicdefect.utils import get_device, seed_everything, setup_mlflow


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, total = [], [], 0.0
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        total += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return ys, ps, total / max(len(ys), 1)


def head_params(model, arch):
    return list((model.fc if arch.startswith("resnet") else model.classifier).parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--model", default="resnet18", choices=MODELS)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                    help="start from ImageNet weights (default; --no-pretrained only for offline tests)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop-bottom", type=float, default=0.0,
                    help="fraction of image height to cut off the bottom (SEM info bar), e.g. 0.07")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="train only the last layer first")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="last layer; backbone gets lr/10")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--experiment", default="sem-artifacts")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    files, labels, classes = scan_folder(args.data)
    split = stratified_split(labels, args.val, args.test, args.seed)
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    counts = {c: int((labels == i).sum()) for i, c in enumerate(classes)}
    print(f"device={device}  classes={counts}")
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}")
    small = [c for c, n in counts.items() if n < 20]
    if small:
        print(f"WARNING: very few images for {small}; results for these classes will be unreliable")

    run_name = args.run_name or f"sem-{args.model}"
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "split"])
        for fp, y, s in zip(files, labels, split):
            w.writerow([str(fp), classes[y], s])

    kw = dict(img_size=args.img_size, crop_bottom=args.crop_bottom)
    train_ds = SEMDataset([files[i] for i in tr], labels[tr], train=True, **kw)
    val_ds = SEMDataset([files[i] for i in va], labels[va], **kw)
    test_ds = SEMDataset([files[i] for i in te], labels[te], **kw)
    cnt = np.bincount(labels[tr], minlength=len(classes)).astype(float)
    sampler = WeightedRandomSampler(torch.as_tensor((1 / np.maximum(cnt, 1))[labels[tr]]),
                                    num_samples=len(tr), replacement=True)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=pin, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=args.workers, pin_memory=pin)

    model = build(args.model, len(classes), pretrained=args.pretrained).to(device)
    head = head_params(model, args.model)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr / 10}, {"params": head, "lr": args.lr}],
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=pin)

    ckpt = out_dir / "model.pt"
    live_path = out_dir / "live.json"
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(va), size=min(12, len(va)), replace=False) if len(va) else np.array([], int)
    sample_imgs = [cv_thumb(val_ds.image(i)) for i in sample_idx]
    live = {"run": run_name, "classes": classes,
            "history": {k: [] for k in ("epoch", "train_loss", "val_loss", "val_macro_f1", "val_balanced_acc")}}
    write_live(live_path, live)

    mlflow = setup_mlflow(args.experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**vars(args), "classes": ",".join(classes), "n_train": len(tr),
                           "n_val": len(va), "n_test": len(te), "device": str(device)})
        best_f1, best_loss, bad = -1.0, float("inf"), 0
        for epoch in range(1, args.epochs + 1):
            frozen = epoch <= args.freeze_epochs
            for p in backbone:
                p.requires_grad_(not frozen)
            model.train()
            t0, total, n, last = time.time(), 0.0, 0, 0.0
            for b, (x, yb) in enumerate(train_loader, 1):
                x, yb = x.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=pin):
                    loss = criterion(model(x), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * len(yb)
                n += len(yb)
                if time.time() - last > 3 or b == len(train_loader):
                    live["progress"] = {"epoch": epoch, "epochs": args.epochs, "batch": b,
                                        "batches": len(train_loader), "loss": total / n,
                                        "phase": "training last layer only" if frozen else "training all layers"}
                    write_live(live_path, live)
                    last = time.time()
            sched.step()

            live["progress"]["phase"] = "validating"
            write_live(live_path, live)
            yv, pv, val_loss = predict(model, val_loader, device)
            m = metrics(yv, pv, classes)
            h = live["history"]
            for k, v in (("epoch", epoch), ("train_loss", total / max(n, 1)), ("val_loss", val_loss),
                         ("val_macro_f1", m["macro_f1"]), ("val_balanced_acc", m["balanced_acc"])):
                h[k].append(float(v))
            live["val_confusion"] = confusion_matrix(yv, pv, labels=range(len(classes))).tolist()
            live["val_epoch"] = epoch
            live["samples"] = {"maps": sample_imgs, "cmap": "gray", "vmax": 255,
                               "true": [classes[i] for i in yv[sample_idx]],
                               "pred": [classes[i] for i in pv[sample_idx]]}
            write_live(live_path, live)
            mlflow.log_metrics({"train_loss": total / max(n, 1), "val_loss": val_loss,
                                **{f"val_{k}": v for k, v in m.items()}}, step=epoch)
            print(f"epoch {epoch:3d}  loss {total / max(n, 1):.4f}  val loss {val_loss:.4f}  "
                  f"val macro-F1 {m['macro_f1']:.4f}  ({time.time() - t0:.0f}s){'  [head only]' if frozen else ''}")

            if m["macro_f1"] > best_f1 or (m["macro_f1"] == best_f1 and val_loss < best_loss):
                best_f1, best_loss, bad = m["macro_f1"], val_loss, 0
                save_bundle(ckpt, model, args.model, classes, args.img_size, args.crop_bottom,
                            {"epoch": epoch, "val_macro_f1": best_f1})
            elif not frozen:
                bad += 1
                if bad >= args.patience:
                    print(f"Early stopping (no val gain for {args.patience} epochs)")
                    break

        live["progress"]["phase"] = "testing best checkpoint"
        write_live(live_path, live)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        yt, pt, _ = predict(model, test_loader, device)
        tm = metrics(yt, pt, classes)
        rep = report(yt, pt, classes=classes)
        (out_dir / "test_report.txt").write_text(rep)
        (out_dir / "test_metrics.json").write_text(json.dumps(tm, indent=2))
        plot_confusion(yt, pt, out_dir / "test_confusion.png", classes=classes, title=f"{run_name} - test")
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})
        mlflow.log_metric("best_val_macro_f1", best_f1)

        # Portable copy: loads with torch.jit.load(), no repo code needed
        try:
            model.eval().cpu()
            ts = torch.jit.trace(model, torch.zeros(1, 3, args.img_size, args.img_size))
            ts.save(str(out_dir / "model_ts.pt"))
        except Exception as e:  # noqa: BLE001
            print(f"(TorchScript export skipped: {e})")
        for f in ("model.pt", "test_report.txt", "test_metrics.json", "test_confusion.png", "split.csv"):
            mlflow.log_artifact(str(out_dir / f))

        print("\nTEST\n" + rep)
        print(f"test macro-F1 {tm['macro_f1']:.4f}  balanced acc {tm['balanced_acc']:.4f}")
        print(f"Checkpoint + report: {out_dir}")
        print(f"Use it:  python predict_sem.py --model {ckpt} --input <image or folder>")


def cv_thumb(img: np.ndarray, side: int = 96) -> list:
    import cv2

    return cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA).tolist()


if __name__ == "__main__":
    main()
