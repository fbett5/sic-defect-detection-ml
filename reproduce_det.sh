#!/usr/bin/env bash
# Reproduce the detection track end to end (Track 3).
#   bash reproduce_det.sh            full run (GPU recommended: ~30 min on a T4, many hours on CPU)
#   QUICK=1 bash reproduce_det.sh    2-epoch smoke run to check the plumbing (~20 min on CPU)
set -euo pipefail
cd "$(dirname "$0")"

EP_DET=""; EP_CNN=""
if [[ "${QUICK:-0}" == "1" ]]; then EP_DET="epochs=2"; EP_CNN="--quick"; fi

# 1. Dataset: download DeepPCB and curate it (images, labels, splits, stats, dataset card, views)
[[ -d data/raw/DeepPCB ]] || git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/raw/DeepPCB
[[ -f data/det/deeppcb/manifest.csv ]] || python prepare_detection.py --dataset deeppcb --src data/raw/DeepPCB --out data/det/deeppcb

# 2. YOLO detectors: main (reference-difference input) and the gray-only ablation
python train_det.py --config configs/det_deeppcb_refdiff.yaml ${EP_DET:+--set $EP_DET}
python train_det.py --config configs/det_deeppcb_gray.yaml ${EP_DET:+--set $EP_DET}

# 3. CNN second stage, with the detector's own false alarms as hard negatives
CNN_CFG=configs/cnn_deeppcb_refdiff.yaml
if [[ -n "$EP_CNN" ]]; then sed 's/epochs: .*/epochs: 1/' $CNN_CFG > /tmp/cnn_quick.yaml; CNN_CFG=/tmp/cnn_quick.yaml; fi
python train_crop_cnn.py --config $CNN_CFG --det-weights outputs/det/yolov8n-refdiff/weights/best.pt

# 4. Benchmark on the test split + automated report
python evaluate_det.py --det outputs/det/yolov8n-refdiff/weights/best.pt \
  --cnn outputs/cnn/cnn-resnet18-refdiff-hardneg/model.pt --view refdiff \
  --compare outputs/det/yolov8n-gray/weights/best.pt:gray --name deeppcb

# 5. Engineer workflow demo: 20 test images -> review page + batch report
python run_pipeline.py --det outputs/det/yolov8n-refdiff/weights/best.pt \
  --cnn outputs/cnn/cnn-resnet18-refdiff-hardneg/model.pt --view refdiff \
  --from-dataset data/det/deeppcb --n 20 --metrics outputs/eval/deeppcb/metrics.json --out outputs/runs/demo

echo
echo "Report:      outputs/eval/deeppcb/report.html"
echo "Review page: outputs/runs/demo/review/index.html  (export decisions, then: python ingest_review.py --decisions <file> --out outputs/runs/demo/reviewed)"
