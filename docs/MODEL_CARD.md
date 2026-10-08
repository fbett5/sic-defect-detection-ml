# Model card: DeepPCB defect detector (YOLO) + region classifier (CNN)

| | detector | classifier |
|---|---|---|
| file | `outputs/det/yolov8n-refdiff/weights/best.pt` | `outputs/cnn/cnn-resnet18-refdiff-hardneg/model.pt` |
| architecture | YOLOv8n (3.0 M parameters), COCO-pretrained start | ResNet-18, ImageNet-pretrained start, 7 outputs |
| input | 640×640, 3 channels = [tested, reference, \|difference\|] (`refdiff`) | 64×64 crop of a box + 50% context, same 3 channels |
| output | boxes, class (6), confidence | probabilities over 6 defect classes + background |
| config | `configs/det_deeppcb_refdiff.yaml` (+ `config_used.yaml` in the run folder) | `configs/cnn_deeppcb_refdiff.yaml` |
| training data | DeepPCB train split (848 images, 5,817 boxes) | crops of train defects + random background + detector false alarms on train/val |
| selection | best val mAP | best val macro-F1 |
| ablation | `yolov8n-gray` (tested image only) | – |

Classes: open, short, mousebite, spur, copper (spurious copper), pin-hole.

**Fusion and decision.** score = √(YOLO confidence × CNN probability of its class). The operating
threshold is tuned on val and stored in `outputs/eval/deeppcb/metrics.json`. Boxes are
pre-accepted when score ≥ 0.70 and the models agree; everything else at or above the threshold
goes to engineer review.

**Performance.** See `outputs/eval/deeppcb/report.html`, and the summary in
[TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md#3-results) once the GPU run is done.

**Intended use.** An engineer-assist tool: it proposes defect locations and classes for review.
It is not for autonomous disposition.

**Not suitable for:**
- images unlike the training data (other layers, magnifications, modalities) without retraining
- defect types that are not in the class list
- deciding root cause on its own

**Using it.** `run_pipeline.py --det … --cnn … --view refdiff --input <images> --refs <references>`.
It needs an aligned defect-free reference per image. For images without one, train with
`view: gray`.

**Reproduce.** `bash reproduce_det.sh`, or Colab section 11. Seeds are fixed (42). Environment
versions are recorded in `outputs/det/<run>/train_summary.json`.
