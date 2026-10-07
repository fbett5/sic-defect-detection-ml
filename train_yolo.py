"""Step 3: YOLOv8 classification on the SAME lot-grouped split, for a fair comparison.

Requires prepare_wm811k.py to have written data/processed/yolo_<size>/.

Examples:
  python train_yolo.py                          # yolov8n-cls, 30 epochs
  python train_yolo.py --model yolov8s-cls.pt --epochs 50
  python train_yolo.py --epochs 1 --fraction 0.05   # quick smoke test

Test metrics are computed with the same functions as train_cnn.py and logged
to the same MLflow store, so the runs line up side by side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sicdefect.metrics import classification_metrics, plot_confusion, report
from sicdefect.utils import get_device, setup_mlflow
from sicdefect.wm811k import CLASS_TO_IDX, CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/yolo_64")
    ap.add_argument("--model", default="yolov8n-cls.pt", help="yolov8{n,s,m}-cls.pt or yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiment", default="wm811k-yolo")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})  # we log our own, comparable run below

    data = Path(args.data).resolve()
    if not (data / "train").exists():
        raise SystemExit(f"{data}/train not found — run prepare_wm811k.py first.")

    dev = get_device(args.device)
    device = 0 if dev.type == "cuda" else dev.type
    run_name = args.run_name or Path(args.model).stem
    out_dir = Path(args.out).resolve()

    model = YOLO(args.model)
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        fraction=args.fraction, patience=args.patience, device=device, workers=args.workers,
        seed=args.seed, project=str(out_dir / "yolo"), name=run_name, exist_ok=True,
        # Dihedral-only augmentation: keep wafer maps physically valid.
        fliplr=0.5, flipud=0.5, degrees=0.0, scale=0.0, translate=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, auto_augment=None,
        plots=True, verbose=False,
    )

    best = out_dir / "yolo" / run_name / "weights" / "best.pt"
    model = YOLO(str(best))

    # Evaluate on the held-out test split with our metrics
    y_true, y_pred = [], []
    for cls in CLASSES:
        files = sorted((data / "test" / cls).glob("*.png"))
        for i in range(0, len(files), 256):
            batch = [str(f) for f in files[i:i + 256]]
            for r in model.predict(batch, imgsz=args.imgsz, device=device, verbose=False):
                y_pred.append(CLASS_TO_IDX[r.names[int(r.probs.top1)]])
                y_true.append(CLASS_TO_IDX[cls])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tm = classification_metrics(y_true, y_pred)

    res_dir = out_dir / f"yolo-{run_name}"
    res_dir.mkdir(parents=True, exist_ok=True)
    rep = report(y_true, y_pred)
    (res_dir / "test_report.txt").write_text(rep)
    (res_dir / "test_metrics.json").write_text(json.dumps(tm, indent=2))
    plot_confusion(y_true, y_pred, res_dir / "test_confusion.png", title=f"{run_name} — test")

    mlflow = setup_mlflow(args.experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**vars(args), "n_test": len(y_true)})
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})
        for f in ("test_report.txt", "test_metrics.json", "test_confusion.png"):
            mlflow.log_artifact(str(res_dir / f))
        mlflow.log_artifact(str(best))

    print("\nTEST\n" + rep)
    print(f"test macro-F1 {tm['macro_f1']:.4f}  balanced acc {tm['balanced_acc']:.4f}  "
          f"defect recall {tm['defect_recall']:.4f}")
    print(f"Weights: {best}\nReport:  {res_dir}")


if __name__ == "__main__":
    main()
