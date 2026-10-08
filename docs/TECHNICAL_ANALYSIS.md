# Technical analysis: automated defect detection and localisation

Status: **pipeline complete; final benchmark numbers pending the GPU run** (Colab, section 11).
`evaluate_det.py` writes them to `outputs/eval/deeppcb/results.md` and `report.html`; paste
`results.md` into section 3 below. Everything else here holds regardless of the final numbers, and
every number quoted now is labelled with where it came from.

## 1. What was built

| requirement | where |
|---|---|
| curated, benchmarked dataset | `prepare_detection.py`, [dataset-cards/deeppcb.md](dataset-cards/deeppcb.md) (classes, sizes, distribution, splits, box statistics, provenance, limits) |
| YOLO detection + localisation | `train_det.py`, `configs/det_*.yaml` (preprocessing, augmentation, hyperparameters, model all in config) |
| CNN classification of detected regions | `train_crop_cnn.py`, `configs/cnn_*.yaml`; fused with YOLO in `sicdefect/det/pipeline.py` |
| quantitative benchmarking | `evaluate_det.py`: mAP50, mAP50-95, P/R/F1 per class, confusion with background, false alarms and misses, false alarms on defect-free boards, recall by size, ms per stage |
| human-in-the-loop workflow | `run_pipeline.py` → `review/index.html` → `ingest_review.py` |
| visual outputs + automated reports | `report.html` (benchmark), `batch_report.html` (per batch), TP/FP/FN overlays, failure galleries, `cases.csv` |
| reproducible repo | `reproduce_det.sh`, configs, fixed seeds, `config_used.yaml` per run, tests, Colab notebook |

## 2. Method choices and why

**Dataset.** No public semiconductor SEM/optical dataset with box labels was found. The SEM
defect sets used in published work, such as imec's resist-wafer images, are proprietary, and
WM-811K is wafer maps only. DeepPCB was chosen because it is the closest open analogue: grayscale
inspection images, small interconnect defects (opens, shorts, notches, protrusions, extra material,
voids), and a defect-free reference for every image, which is the die-to-die principle. A synthetic
SEM-style set (`tools/make_synthetic_sem_det.py`) runs the identical pipeline on patterned-metal
images with particle, scratch, nanowire, bridge and open defects, to show the transfer path to real
SEM data.

**Reference-difference input (`refdiff`).** The detector sees [tested, reference, |difference|]
instead of the tested image alone. On a patterned layout, most of the image is legitimate
structure. The difference channel removes it and leaves only what changed, which is what an FA
engineer does by comparing against a good die. `configs/det_deeppcb_gray.yaml` is the ablation
without a reference, so the benefit is measured rather than assumed.

**Two stages.** YOLO is good at *where*. Its class decision uses features shared across the whole
image, and it must make one call per box at one threshold. The CNN sees only the region at a fixed
scale with context, and has a **background** class trained on the detector's own false alarms
(hard negatives). That lets it do two separate jobs: relabel a box, or reject it. The fused score
√(YOLO × CNN) needs both models to agree for a box to rank high. The evaluation measures the CNN's
effect box by box (false alarms removed/added, defects lost/gained, classes fixed/broken) instead
of reporting only the total.

**Thresholds tuned on val, frozen for test.** Choosing the threshold on test would inflate every
operating-point number.

**False alarms on good boards.** DeepPCB images all contain defects, so 100 defect-free templates
were added to test. Without them, the number an FA lab cares about, "how often does a good sample
get flagged", cannot be measured.

## 3. Results

_Paste `outputs/eval/deeppcb/results.md` and `outputs/eval/synthetic_sem/results.md` here after
the GPU run._

Evidence so far (local CPU run, stopped early to move training to the A100; **validation split,
YOLO only, refdiff**):

| epoch (of 25) | val mAP50 | val mAP50-95 |
|---|---|---|
| 1 | 0.323 | 0.190 |
| 3 | 0.874 | 0.576 |
| 9 | 0.958 | 0.671 |
| 13 | 0.976 | 0.683 |
| 17 | 0.977 | 0.503 |

Two observations are already clear from the log:

1. **Finding the defects is the easy part.** Val mAP50 passed 0.95 by epoch 9.
2. **Box tightness is the weak part.** From epoch 7 on, mAP50 stayed between 0.93 and 0.98,
   while mAP50-95 swung between 0.50 and 0.68 from one epoch to the next. The defects are about 35 px, so a 3-4 px border error drops IoU
   below 0.75. Where exactly a "mousebite" or "spur" ends is also partly a labelling convention.
   For FA this matters less than it looks: the engineer needs the location, not a pixel-exact box.

On ground-truth crops (perfect boxes), the CNN alone reached 0.954 macro-F1 after **one** epoch
trained from scratch on CPU (test crops, 6 classes + background). Its weakest class was
**mousebite** (precision 0.82), which absorbed **open** crops (open recall 0.84). That is the
pair to watch after full training.

## 4. What the system can and cannot reliably detect

Confirm the specific numbers against the final report. The mechanisms below are structural.

**Reliable:**
- Defects that add or remove material where the reference has none: short, spurious copper,
  pin-hole, and on SEM a particle, bridge or open. These give a strong, compact signal in the
  difference channel.
- Telling whether something is there at all. Class-agnostic AP ("AP50 any-defect" in the report)
  is the number to quote for "the model finds the defect", separately from naming it.

**Less reliable:**
- **Look-alike edge defects (open vs mousebite, spur vs copper).** All of them sit on a line edge
  and differ in shape and extent, not in kind. They are where class confusion concentrates, and
  the confusion matrix shows it directly.
- **The smallest defects.** The evaluation reports recall per box-size tertile, and the lowest
  tertile is where misses accumulate. At 640 px input, a 15-20 px defect lands on a few cells of
  the stride-8 feature map.
- **Exact box extent** (see mAP50-95 above).
- **Anything without a reference.** The `gray` ablation measures how much is lost when only the
  tested image is available.

**Out of scope, not detectable by design:**
- **Defect types that are not in training.** A new particle morphology, a crack, or charging
  artefacts get either a low score (and land in review) or the nearest known class. Nothing in a
  closed-set detector guarantees the first. The "models disagree" and low-score review states are
  the safety net, not a guarantee.
- **Subsurface or electrical-only failures.** This is image-based analysis, so it can't catch a
  failure with no visible signature. That remains the job of fault isolation (thermal, OBIRCH,
  EMMI, CT).

## 5. False positives and false negatives

The report's failure galleries are built from the test outcomes. Each has a recognisable cause:

| failure | typical cause | mitigation |
|---|---|---|
| false alarm at a pad or line edge | registration error between tested and reference images: the difference channel lights up along edges (visible as thin rings in the refdiff view) | register images before differencing; erode thin difference structures; CNN background class trained on these hard negatives |
| false alarm on a good board | as above, plus binarisation noise | measured explicitly with the clean controls; raise the threshold (see the trade-off figure) |
| missed small defect | too few pixels at stride 8 | higher input resolution, tiling, or a P2 (stride 4) detection head |
| wrong class at an edge | open/mousebite and spur/copper share appearance | the CNN stage; more context in the crop; merge the classes if the FA disposition is the same |
| CNN rejects a real defect | crop resembles a hard negative (edge-like defect) | the review queue shows boxes YOLO is confident about even when the CNN disagrees; "models disagree" flag |

Operationally, a miss is worse than a false alarm in FA: a false alarm costs an engineer a few
seconds in review, while a miss can close a case on the wrong root cause. So the review design
shows everything above the val-tuned threshold and pre-accepts only high-score boxes where both
models agree. Nothing is auto-closed.

## 6. Dataset limitations

- **The images are PCB, not silicon.** They are binarised, so there is no grey-level texture,
  SEM charging, edge brightening, or detector noise. Absolute numbers will not transfer; the
  method will.
- **Some defects are synthetic** (added by the dataset authors), so they are probably more
  regular than real ones.
- **The references are near-perfectly aligned.** Real die-to-die comparison has registration
  error, and that is the main false-alarm source to expect on SEM.
- **The clean controls are templates**, so they are cleaner than a real good board. The
  good-board false-alarm rate is optimistic.
- **The synthetic SEM set tests the plumbing, not real-world accuracy.** Its defects are drawn
  shapes.

## 7. Improving the system

1. **Real labelled SEM/optical data from your lab.** Use `prepare_detection.py --dataset yolo`,
   then the review loop: every reviewed batch becomes training data via `ingest_review.py`. Start
   with the classes the lab dispositions differently, and keep at least 20 good images in test.
2. **Registration before differencing** (phase correlation or ECC). This is the biggest expected
   gain once real reference images are used.
3. **Small-defect recall**: tiling or a higher input resolution, or a stride-4 head; compare the
   size-tertile recall before and after.
4. **Open-set handling**: an "unknown" route, triggered by low CNN confidence plus a high
   background probability, or by distance to the training features, so a new defect type is sent
   to an engineer instead of being forced into a known class.
5. **Calibration**: temperature-scale both models on val, so the 0.70 pre-accept threshold means
   ~70% probability.
6. **Larger detector** (`yolov8s`/`yolo11s`) once data volume justifies it. Change one line in the
   config.
7. **Link to Track 1**: wafer-map pattern classification decides *which dies* to image, and Track
   3 then localises the defect on those dies. That is the full FA flow from electrical test to
   physical defect.
