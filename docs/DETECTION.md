# Track 3: defect detection, localisation and engineer review

```
semiconductor / inspection image (+ known-good reference)
        │  preprocess            sicdefect/det/preprocess.py   gray | refdiff | clahe
        ▼
   YOLO detector             train_det.py                     boxes + class + confidence
        │  crop each box (+50% context)
        ▼
   CNN classifier            train_crop_cnn.py                defect class or "background"
        │  fuse: score = sqrt(YOLO conf × CNN prob)
        ▼
   engineer review           run_pipeline.py → review/index.html   accept / reject / relabel / add missed
        │                    ingest_review.py                       → corrected labels (retrain)
        ▼
   evaluation + report       evaluate_det.py → report.html, metrics.json, cases.csv
```

Results, failure analysis and limitations: [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md).
Trained-model details: [MODEL_CARD.md](MODEL_CARD.md). Dataset documentation:
[dataset-cards/deeppcb.md](dataset-cards/deeppcb.md).

Everything runs with `bash reproduce_det.sh`, or step by step below, or in Colab
(`colab_runner.ipynb`, section 11).

---

## 1. Dataset

### Why DeepPCB

The goal is semiconductor failure analysis, so the first choice would be a public SEM or
optical defect dataset with box annotations. None was found that is both public and
box-labelled:

| candidate | what it is | why not the main dataset |
|---|---|---|
| WM-811K, MixedWM38 | wafer **maps** (die pass/fail grids) | classification only, no images of defects, no boxes (used in Track 1) |
| imec SEM resist-defect sets (bridge, line collapse, gap, …) | real SEM images with boxes | not publicly released |
| MVTec AD | industrial textures/objects with pixel masks | not semiconductor; masks convertible to boxes (`prepare_detection.py --dataset mvtec`) |
| NEU-DET, GC10-DET | steel surface defects with boxes | texture defects, no reference image, further from FA |
| **DeepPCB** | 1,500 PCB inspection image pairs, 640×640, 6 defect types, boxes, **defect-free template per image**, MIT licence | **chosen** |

DeepPCB ([Tang et al. 2019, arXiv:1902.06197](https://arxiv.org/abs/1902.06197),
[github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)) is the closest
public analogue to semiconductor inspection:

- **Images come with a known-good reference.** Each tested image has an aligned defect-free
  template, the same principle as die-to-die / die-to-database comparison in wafer
  inspection and golden-sample comparison in FA.
- **The defects are interconnect defects.** Open, short, mousebite, spur, spurious copper and
  pin-hole are the PCB equivalents of line opens, bridges, line-edge notches, protrusions,
  extra-pattern particles and voids in metal layers seen under SEM/optical inspection.
- **The images are grayscale with small defects.** About 35 px boxes in 640 px images, scanned
  at about 48 px/mm, which is the same "small object in a large field of view" problem.

### How the methodology transfers to semiconductor FA

| here (DeepPCB) | in semiconductor FA |
|---|---|
| tested image + template | SEM/optical image of the suspect die + same location on a good die (or CAD/layout render) |
| `refdiff` input (tested, reference, \|difference\|) | die-to-die difference image; needs registration first |
| open / short / mousebite / spur / copper / pin-hole | open / bridge / notch / protrusion / particle / void (or your own classes: dirt, scratch, nanowire, residue, …) |
| `prepare_detection.py --dataset yolo` | your boxes exported from CVAT / Label Studio / Roboflow in YOLO format |
| `gray` / `clahe` views | SEM images without a reference; CLAHE for charging / uneven contrast |

Nothing in the pipeline is PCB-specific: the converters, preprocessing modes, training,
evaluation, review and report all work on any dataset in the canonical layout below.

### Curated layout (`prepare_detection.py`)

```
data/det/<name>/
    images/<split>/<id>.png        tested images, 8-bit grayscale
    references/<split>/<id>.png    defect-free reference (when available)
    labels/<split>/<id>.txt        YOLO boxes: class cx cy w h (normalised)
    manifest.csv                   id, split, group, size, number of boxes, source file
    classes.json                   class names in id order + source/licence
    stats.json, stats_*.png        baseline statistics
    DATASET_CARD.md                generated documentation
    views/<mode>/                  model-input images per preprocessing mode + data.yaml
```

Curation decisions for DeepPCB:

- **Split.** The official 1,000 / 500 trainval / test split is kept, so results compare with
  the paper. Validation (15%) is carved from trainval, stratified by board group.
- **Clean controls.** All DeepPCB images contain defects, so on its own the test set can't
  show how often a good board raises an alarm. 100 defect-free templates from test-split
  boards are added to test (id suffix `_clean`, no boxes).
- **Box cleaning.** Coordinates are clipped to the image, corners ordered, and boxes smaller
  than 2 px dropped.
- **Grayscale.** Images are stored as 8-bit grayscale (16-bit TIFFs are contrast-stretched) by
  `read_gray`, which is the same function the pipeline uses at inference time.
- **Fixed seed.** The split is random but seeded (`--seed 42`), and the manifest records it.

---

## 2. Preprocessing (`sicdefect/det/preprocess.py`)

One function, `make_input(image, reference, mode)`, is used for the training views *and* at
inference, so the two cannot drift apart.

| mode | channels | when |
|---|---|---|
| `gray` | tested ×3 | no reference available |
| `refdiff` | tested, reference, \|tested − reference\| | aligned known-good image available (main model) |
| `clahe` | CLAHE-equalised tested ×3 | SEM/optical with uneven illumination or charging |

---

## 3. YOLO detector (`train_det.py`)

- **Model.** YOLOv8n, starting from COCO-pretrained weights (`yolov8n.pt`). `yolov8s.pt` and
  `yolo11n.pt` are drop-in replacements.
- **Config.** `configs/det_deeppcb_refdiff.yaml` (main) and `configs/det_deeppcb_gray.yaml`
  (ablation) hold every hyperparameter. `--set key=value` overrides one, and the exact config
  used is saved as `outputs/det/<run>/config_used.yaml`.
- **Resolution.** Training runs at the native 640 px. Defects are about 35 px, and
  downscaling would shrink them toward the detector's smallest stride.
- **Augmentation.** Flips (vertical and horizontal), mosaic (off for the last epochs),
  translate 0.1 and scale 0.25. **There is no colour or HSV jitter**: inspection images are
  grayscale, and in `refdiff` each channel means something (tested / reference / difference),
  so colour jitter would corrupt the signal. There are also no arbitrary rotations, which
  would make axis-aligned boxes too loose.
- **Optimiser.** AdamW, lr 0.002 with cosine decay and 2 warm-up epochs, early stopping
  (patience 15). The best checkpoint is chosen by val mAP.
- **Test split.** The test split is never used for training or model selection.

## 4. CNN second stage (`train_crop_cnn.py`)

- **Inputs.** Crops around each box with 50% context on every side, resized to 64×64, from
  the same view as the detector.
- **Classes.** The 6 defect classes plus **background**, so the CNN can *reject* a YOLO box,
  not only relabel it.
- **Background examples.** Random non-defect regions, plus **hard negatives**: the
  detector's own false alarms on train/val (`--det-weights`). These are exactly the cases the
  CNN has to reject in the pipeline.
- **Training.** ResNet-18 (ImageNet start when downloadable), weighted sampling across
  classes, dihedral augmentation, label smoothing 0.05, OneCycle LR. Config:
  `configs/cnn_deeppcb_refdiff.yaml`.
- **Standalone test.** The CNN is tested on ground-truth test crops (perfect boxes), which
  gives the upper bound of the second stage on its own.

### Fusion (`sicdefect/det/pipeline.py`)

| approach | class | score |
|---|---|---|
| YOLO only | YOLO class | YOLO confidence |
| YOLO + CNN | CNN's best defect class | √(YOLO confidence × CNN probability of that class) |

A box the CNN calls background gets a low score and drops below the operating threshold.
YOLO's NMS is per class, so one defect can carry two boxes with different YOLO classes.
After the CNN relabels them, duplicates (same class, IoU > 0.6) are merged.

---

## 5. Evaluation (`evaluate_det.py`, `sicdefect/det/evaluate.py`)

The **operating threshold** for each approach is the one with the best F1 on **val**. It is
then frozen and test is scored once.

| metric | meaning |
|---|---|
| mAP@0.5, mAP@0.5:0.95 | COCO-style, 101-point interpolation, per class then averaged. Cross-checked against Ultralytics `val()` (agrees within ~0.01) |
| AP50 any-defect | class ignored: "did we put a box on the defect at all?" Separates localisation from classification |
| precision, recall, F1 | at the operating threshold, IoU ≥ 0.5, micro (all boxes) and per class |
| confusion matrix | includes a background row (false alarms) and a missed column |
| wrong-class rate | found in the right place, but with the wrong class |
| false alarms / image, miss rate | what an engineer would experience per image |
| good boards alarmed | share of defect-free boards with at least one alarm |
| recall by size | localisation recall for small / medium / large boxes (area tertiles) |
| inference time | ms per image for preprocess, detect and classify |

Matching at the operating point is class-agnostic and greedy by confidence. A matched box
with the wrong class counts as `MISCLS`: an FP for the predicted class and an FN for the
true class in the per-class numbers.

The script also compares the two approaches **box by box**: false alarms the CNN removed or
added, real defects it lost or gained, and classes it fixed or broke.

---

## 6. Engineer review (human in the loop)

```bash
python run_pipeline.py --det <yolo best.pt> --cnn <cnn model.pt> --view refdiff \
    --input new_images/ --refs golden_images/ --metrics outputs/eval/deeppcb/metrics.json \
    --out outputs/runs/<batch>
```

- **Status per box.** Each box gets one of three statuses:
  - **pre-accepted:** score ≥ 0.70 and YOLO and CNN agree.
  - **needs review:** the score is at or above the val-tuned threshold, or the models disagree.
  - **hidden:** a low-confidence box, which can be shown with a toggle.
- **Review page** (`review/index.html`, opens in any browser and needs no server):
  - It shows the image, reference and difference views, plus both models' opinions for every box.
  - Keyboard shortcuts: **A** accept, **R** reject, **1–9** relabel, **D** draw a missed defect, **N/P** next/previous image, **Enter** accept all.
  - A "models disagree" flag marks boxes where YOLO and the CNN chose different classes.
  - A filter shows only the images that need review.
  - Progress is kept in the browser until you click **Export decisions**.
- **Batch report** (`batch_report.html`): counts, a per-class table, and images with
  overlays, highest-risk first.
- **Ingest** (`python ingest_review.py --decisions decisions_<batch>.json --out <dir>`):
  - It writes the confirmed labels as a ready YOLO dataset, for the active-learning loop:
    `prepare_detection.py --dataset yolo --src <dir>`.
  - It also writes `review_summary.json`: accepted, relabelled, rejected and added counts,
    plus model precision and recall as judged by the engineer.
  - Only images marked reviewed are written. An unreviewed image would otherwise become a
    wrong "no defects" label.

The model never closes a case on its own. Even pre-accepted boxes stay visible and can be
rejected, and everything the engineer changes is logged (`review_log.csv`).

---

## 7. Reports and visual outputs

`outputs/eval/<name>/`

| file | what |
|---|---|
| `report.html` | summary cards, dataset, benchmark table, ablation, CNN effect, PR curves, threshold trade-off, timing, per-class table, confusion matrices, failure galleries, individual cases, limitations |
| `metrics.json` | every number in the report |
| `cases.csv` | one row per test image: correct, wrong class, false alarms, missed, details |
| `failures/*.png` | highest-confidence false alarms, missed defects, wrong classes, false alarms on good boards, what the CNN removed or lost |
| `cases/*.jpg` | per-image overlays (green correct, red false alarm, orange dashed missed, purple wrong class) |
| `predictions_*.json` | raw pipeline output (both models' opinions per box), for re-analysis |

---

## 8. Use it on your own SEM / optical images

1. **Label boxes** in CVAT, Label Studio or Roboflow and export in YOLO format
   (`images/`, `labels/`, `classes.txt`). Name files `<sample>_<n>.png` so shots of the same
   sample stay in one split.
2. **Curate:**
   `python prepare_detection.py --dataset yolo --src my_export --out data/det/my_sem --views gray clahe`.
   If you have reference images of the same location on a good die, put them in
   `references/` and use `refdiff`.
3. **Configure:** copy a config, set `dataset:` and `view:`, and train both stages:
   `train_det.py`, then `train_crop_cnn.py --det-weights …`.
4. **Evaluate and review:** `evaluate_det.py`, then `run_pipeline.py` on new images.

How much data: a few hundred boxes per class is a workable start for YOLO fine-tuning,
and more for classes that look alike. Keep at least 20 defect-free images in test to
measure false alarms.
