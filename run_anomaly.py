"""Step 4: unsupervised anomaly detection + localisation on MVTec AD (anomalib 2.x).

Trains on defect-free images only, then scores and localises defects on the
test set. This is the proxy for SiC micrographs, where labeled defects are
scarce. The five texture categories are the closest visual match to SiC
surfaces, so they're the default.

Metrics: image AUROC (is the image defective?), pixel AUROC and PRO / AUPRO
(how well is the defect localised?).

Examples:
  python run_anomaly.py                                      # PatchCore on the 5 textures
  python run_anomaly.py --model padim --categories carpet tile
  python run_anomaly.py --categories all
  python run_anomaly.py --root /path/to/MVTecAD              # if you already downloaded it

The dataset (~5 GB) downloads automatically to --root on first use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sicdefect.utils import setup_mlflow

TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECTS = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill",
           "screw", "toothbrush", "transistor", "zipper"]


def build_model(name: str, pretrained: bool, coreset: float):
    from anomalib.metrics import AUPRO, AUROC, Evaluator
    from anomalib.models import Padim, Patchcore

    evaluator = Evaluator(test_metrics=[
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
    ])
    if name == "patchcore":
        return Patchcore(pre_trained=pretrained, coreset_sampling_ratio=coreset, evaluator=evaluator)
    if name == "padim":
        return Padim(pre_trained=pretrained, evaluator=evaluator)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="patchcore", choices=["patchcore", "padim"])
    ap.add_argument("--categories", nargs="+", default=TEXTURES,
                    help="category names, or 'textures' / 'objects' / 'all'")
    ap.add_argument("--root", default="data/MVTecAD")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--coreset", type=float, default=0.1, help="PatchCore memory-bank sampling ratio")
    ap.add_argument("--no-pretrained", action="store_true", help="random backbone (testing only)")
    ap.add_argument("--accelerator", default="auto", help="auto / gpu / cpu")
    ap.add_argument("--experiment", default="mvtec-anomaly")
    ap.add_argument("--out", default="outputs/anomalib")
    args = ap.parse_args()

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    cats = args.categories
    if cats == ["textures"]:
        cats = TEXTURES
    elif cats == ["objects"]:
        cats = OBJECTS
    elif cats == ["all"]:
        cats = TEXTURES + OBJECTS

    mlflow = setup_mlflow(args.experiment)
    results = {}
    for cat in cats:
        print(f"\n=== {args.model} / {cat} ===")
        dm = MVTecAD(root=args.root, category=cat, train_batch_size=args.batch_size,
                     eval_batch_size=args.batch_size, num_workers=args.workers)
        model = build_model(args.model, not args.no_pretrained, args.coreset)
        engine = Engine(default_root_dir=str(Path(args.out) / args.model / cat),
                        accelerator=args.accelerator, logger=False)
        engine.fit(model=model, datamodule=dm)
        res = engine.test(model=model, datamodule=dm)[0]
        res = {k: float(v) for k, v in res.items()}
        results[cat] = res
        with mlflow.start_run(run_name=f"{args.model}-{cat}"):
            mlflow.log_params({"model": args.model, "category": cat, "coreset": args.coreset,
                               "texture": cat in TEXTURES})
            mlflow.log_metrics(res)

    print("\n" + "-" * 64)
    keys = ["image_AUROC", "pixel_AUROC", "pixel_AUPRO"]
    print(f"{'category':<12}" + "".join(f"{k:>16}" for k in keys))
    for cat, r in results.items():
        print(f"{cat:<12}" + "".join(f"{r.get(k, float('nan')):>16.4f}" for k in keys))
    out = Path(args.out) / f"{args.model}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {out}")


if __name__ == "__main__":
    main()
