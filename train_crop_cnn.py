"""Track 3, step 3: CNN that classifies defect regions (the second stage).

Trained on crops around labelled defects plus "background" crops, so in the
pipeline it can re-label a YOLO box or reject it as a false alarm.

Examples:
  python train_crop_cnn.py --config configs/cnn_deeppcb_refdiff.yaml
  # add the detector's own false alarms as hard negatives (recommended once YOLO is trained):
  python train_crop_cnn.py --config configs/cnn_deeppcb_refdiff.yaml \
      --det-weights outputs/det/yolov8n-refdiff/weights/best.pt

Output: outputs/cnn/<run_name>/
  model.pt           weights + class list (+ "background") + view, crop size, padding
  test_report.txt    classification on ground-truth test crops (upper bound: perfect boxes)
  test_confusion.png, test_metrics.json, config_used.yaml, live.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

from sicdefect.det.crops import CROP, PAD, CropDataset, build_crops, mine_hard_negatives
from sicdefect.det.data import load_classes
from sicdefect.live import write_live
from sicdefect.metrics import plot_confusion, report
from sicdefect.sem import build, metrics
from sicdefect.utils import ROOT, get_device, seed_everything, setup_mlflow


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def make_model(arch, n, pretrained):
    if pretrained:
        try:
            return build(arch, n, pretrained=True)
        except Exception as e:  # noqa: BLE001  (offline: no ImageNet download)
            print(f"WARNING: ImageNet weights unavailable ({type(e).__name__}); training from scratch")
    return build(arch, n, pretrained=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--det-weights", default=None, help="YOLO weights to mine hard negatives from")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="outputs/cnn")
    ap.add_argument("--experiment", default="det-cnn")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    t = cfg["train"]
    seed_everything(t.get("seed", 42))
    device = get_device(args.device)

    ds = ROOT / cfg["dataset"]
    classes = load_classes(ds)
    names = classes + ["background"]
    run = cfg["run_name"] + ("-hardneg" if args.det_weights else "")
    out = ROOT / args.out / run
    out.mkdir(parents=True, exist_ok=True)

    hard = None
    if args.det_weights:
        print("Mining hard negatives (detector false alarms on train/val) ...")
        hard = mine_hard_negatives(ds, cfg["view"], args.det_weights, conf=cfg.get("hard_conf", 0.05),
                                   device=0 if device.type == "cuda" else "cpu")
        print(f"  {sum(map(len, hard.values()))} hard negatives from {len(hard)} images")
    crops = build_crops(ds, cfg["view"], bg_per_image=cfg.get("bg_per_image", 3), hard=hard,
                        seed=t.get("seed", 42))
    (Xtr, ytr, _), (Xva, yva, mva), (Xte, yte, _) = crops["train"], crops["val"], crops["test"]
    for s, (_, y, _) in crops.items():
        print(f"{s}: " + ", ".join(f"{names[i]}={int((y == i).sum())}" for i in range(len(names))))
    cfg_used = {**cfg, "run_name": run, "det_weights": args.det_weights, "crop": CROP, "pad": PAD}
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg_used, sort_keys=False))

    cnt = np.bincount(ytr, minlength=len(names)).astype(float)
    sampler = WeightedRandomSampler(torch.as_tensor((1 / np.maximum(cnt, 1))[ytr]), len(ytr), replacement=True)
    tl = DataLoader(CropDataset(Xtr, ytr, augment=True), t["batch"], sampler=sampler, num_workers=args.workers)
    vl = DataLoader(CropDataset(Xva, yva), 256, num_workers=args.workers)
    el = DataLoader(CropDataset(Xte, yte), 256, num_workers=args.workers)

    model = make_model(cfg["model"], len(names), t.get("pretrained", True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, t["lr"], total_steps=t["epochs"] * len(tl))
    crit = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.05))

    live_path = out / "live.json"
    sidx = np.random.default_rng(0).choice(len(yva), size=min(12, len(yva)), replace=False)
    live = {"run": run, "classes": names, "history": {k: [] for k in
            ("epoch", "train_loss", "val_macro_f1", "val_balanced_acc")}}
    best, ckpt = -1.0, out / "model.pt"
    mlflow = setup_mlflow(args.experiment)
    with mlflow.start_run(run_name=run):
        mlflow.log_params({"view": cfg["view"], "model": cfg["model"], "hard_negatives": bool(hard), **t})
        for ep in range(1, t["epochs"] + 1):
            model.train()
            t0, tot, n, last = time.time(), 0.0, 0, 0.0
            for b, (x, y) in enumerate(tl, 1):
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(y)
                n += len(y)
                if time.time() - last > 3 or b == len(tl):
                    live["progress"] = {"epoch": ep, "epochs": t["epochs"], "batch": b, "batches": len(tl),
                                        "loss": tot / n, "phase": "training"}
                    write_live(live_path, live)
                    last = time.time()
            yv, pv = predict(model, vl, device)
            m = metrics(yv, pv, names)
            h = live["history"]
            for k, v in (("epoch", ep), ("train_loss", tot / n), ("val_macro_f1", m["macro_f1"]),
                         ("val_balanced_acc", m["balanced_acc"])):
                h[k].append(float(v))
            live.update(val_epoch=ep, val_confusion=confusion_matrix(yv, pv, labels=range(len(names))).tolist(),
                        samples={"maps": [x[..., ::-1].tolist() for x in Xva[sidx]], "cmap": None, "vmax": 255,
                                 "true": [names[i] for i in yv[sidx]], "pred": [names[i] for i in pv[sidx]]})
            write_live(live_path, live)
            mlflow.log_metrics({"train_loss": tot / n, **{f"val_{k}": v for k, v in m.items()}}, step=ep)
            print(f"epoch {ep:3d}  loss {tot / n:.4f}  val macro-F1 {m['macro_f1']:.4f}  ({time.time() - t0:.0f}s)")
            if m["macro_f1"] > best:
                best = m["macro_f1"]
                torch.save({"model": model.state_dict(), "arch": cfg["model"], "classes": names,
                            "n_defect_classes": len(classes), "view": cfg["view"], "crop": CROP, "pad": PAD,
                            "epoch": ep, "val_macro_f1": best, "task": "det-crop"}, ckpt)

        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        yt, pt = predict(model, el, device)
        tm = metrics(yt, pt, names)
        (out / "test_report.txt").write_text(report(yt, pt, classes=names))
        (out / "test_metrics.json").write_text(json.dumps(tm, indent=2))
        plot_confusion(yt, pt, out / "test_confusion.png", classes=names,
                       title=f"{run}: test crops (ground-truth boxes)")
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})
        for f in ("model.pt", "test_report.txt", "test_metrics.json", "test_confusion.png", "config_used.yaml"):
            mlflow.log_artifact(str(out / f))
    print("\nTEST (ground-truth crops)\n" + (out / "test_report.txt").read_text())
    print(f"Checkpoint + report: {out}")


if __name__ == "__main__":
    main()
