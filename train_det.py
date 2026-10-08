"""Track 3, step 2: train a YOLO defect detector from a YAML config.

Examples:
  python train_det.py --config configs/det_deeppcb_refdiff.yaml
  python train_det.py --config configs/det_deeppcb_gray.yaml --set epochs=30 batch=8
  python train_det.py --config configs/det_deeppcb_refdiff.yaml --set epochs=1 fraction=0.05   # smoke test

Output: outputs/det/<run_name>/
  weights/best.pt        the detector (Ultralytics format)
  config_used.yaml       exact config incl. overrides, so the run can be repeated
  results.csv            per-epoch losses and val mAP (Ultralytics), also drives the live dashboard
  train_summary.json     final val metrics, training time, environment

The test split is NOT used here; evaluate_det.py scores it once at the end.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import yaml

from sicdefect.det.data import build_view
from sicdefect.utils import ROOT, get_device, setup_mlflow


def parse_set(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in parse_set(overrides).items():
        if k in ("run_name", "dataset", "view", "model"):
            cfg[k] = v
        else:
            cfg.setdefault("train", {})[k] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. epochs=30 view=gray")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="outputs/det")
    ap.add_argument("--experiment", default="det-yolo")
    args = ap.parse_args()

    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})
    cfg = load_config(args.config, args.set)
    ds = (ROOT / cfg["dataset"]).resolve() if not Path(cfg["dataset"]).is_absolute() else Path(cfg["dataset"])
    data_yaml = ds / "views" / cfg["view"] / "data.yaml"
    if not data_yaml.exists():
        print(f"Building {cfg['view']} view of {ds} ...")
        data_yaml = build_view(ds, cfg["view"])

    dev = get_device(args.device)
    device = 0 if dev.type == "cuda" else "cpu"
    out = (ROOT / args.out).resolve()
    run = cfg["run_name"]
    (out / run).mkdir(parents=True, exist_ok=True)
    (out / run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"device={dev}  data={data_yaml}\nconfig: {json.dumps(cfg['train'])}")

    t0 = time.time()
    model = YOLO(cfg["model"])
    model.train(data=str(data_yaml), project=str(out), name=run, exist_ok=True, device=device,
                workers=args.workers, plots=True, verbose=False, **cfg["train"])
    train_s = time.time() - t0

    best = out / run / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(data_yaml), split="val", device=device, plots=False, verbose=False,
                            imgsz=cfg["train"].get("imgsz", 640), batch=8, project=str(out / run), name="val",
                            exist_ok=True)
    summary = {
        "run": run, "config": cfg, "weights": str(best.relative_to(ROOT)), "train_seconds": round(train_s),
        "val": {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp),
                "recall": float(m.box.mr)},
        "env": {"device": str(dev), "python": platform.python_version(),
                "torch": __import__("torch").__version__, "ultralytics": __import__("ultralytics").__version__},
    }
    (out / run / "train_summary.json").write_text(json.dumps(summary, indent=2))

    mlflow = setup_mlflow(args.experiment)
    with mlflow.start_run(run_name=run):
        mlflow.log_params({"view": cfg["view"], "model": cfg["model"], **cfg["train"]})
        mlflow.log_metrics({f"val_{k}": v for k, v in summary["val"].items()})
        mlflow.log_metric("train_seconds", train_s)
        for f in ("config_used.yaml", "train_summary.json", "results.csv"):
            if (out / run / f).exists():
                mlflow.log_artifact(str(out / run / f))
    print(json.dumps(summary["val"], indent=1))
    print(f"Weights: {best}")


if __name__ == "__main__":
    main()
