# Leakage-Controlled Wafer-Map Defect Pattern Recognition as a Proxy Pipeline for Silicon-Carbide Power-Device Inspection

**Festus Bett** · Sofia University · festusbett12@gmail.com

> This is the readable Markdown version of `main.tex`. Both are generated from the
> same content; tables and figures come from `paper/make_results.py`.

---

## Abstract

Silicon-carbide (SiC) power devices fail through crystallographic and
process-induced defects that are expensive to find and, once missed, expensive to
ship. Machine-learned inspection is an attractive response, but there is no large
public SiC defect corpus to develop it on. We therefore build a **proxy**
pipeline: the methodology, evaluation protocol and deployment tooling are
developed against two public industrial-inspection datasets — WM-811K
electrical-test wafer maps for supervised pattern recognition, and MVTec AD for
unsupervised anomaly detection — so that only the imaging front-end, not the
method, has to be replaced when SiC data becomes available.

The contribution is threefold:

1. An evaluation protocol that controls the leakage specific to this domain.
   Wafers are grouped by lot before splitting, because wafers from one lot share
   process history and a random split leaks that history across the train/test
   boundary. The split is *asserted* lot-disjoint at load time rather than assumed.
2. An explicit treatment of the class imbalance that dominates wafer-map data.
   Roughly 85% of labelled WM-811K wafers carry no defect pattern, so accuracy is
   uninformative; macro-F1, balanced accuracy and defect recall are reported
   instead, with four imbalance strategies exposed as a controlled ablation.
3. A deployment artifact: a self-describing model bundle carrying the network as
   ONNX and TorchScript together with the exact preprocessing contract it was
   trained under, verified at export time against the source model, and scoreable
   on new wafer maps with nothing but NumPy and an ONNX runtime.

> ⚠️ **The quantitative results reported here come from a synthetic dataset used to
> validate that the pipeline executes end to end.** The synthetic patterns are far
> cleaner than real wafer maps and the near-perfect scores they produce say
> nothing about real-world performance. [Protocol for the real runs](#protocol-for-the-real-wm-811k-runs)
> gives the exact commands for the real WM-811K experiments, which need GPU time
> and the Kaggle distribution of the dataset.

---

## 1. Introduction

Silicon carbide has displaced silicon in high-voltage power conversion because its
wider bandgap and higher thermal conductivity permit smaller, faster, hotter-running
devices. The penalty is material quality. SiC epitaxy carries defect populations
that silicon largely does not — basal-plane dislocations, micropipes, threading
screw dislocations, triangular "carrot" defects — and a subset are *killer* defects
that destroy the blocking capability of any device built on top of them. Because a
power device occupies a large die area, one killer defect removes a disproportionate
share of the wafer's value; because the devices are safety-critical, an escape is
far more costly than an unnecessary inspection.

That makes defect inspection an obvious target for machine learning and an awkward
one. There is no public SiC inspection corpus comparable in size to the datasets
that drove general computer vision, and the proprietary ones live inside fabs. Work
that waits for such a corpus does not start.

We split the problem in two. The **imaging front-end** — what a photoluminescence,
optical or X-ray topography image of a SiC wafer looks like — is genuinely
SiC-specific and cannot be substituted. The **inspection method** — how to split
industrial data without leaking, how to train against extreme imbalance, which
metrics to believe, how to ship a model a fab engineer can run — is not. All of it
can be developed on public industrial data, and most of the engineering mistakes
available in this problem can be made and corrected there at no cost.

Two public datasets carry the work:

- **WM-811K** — ~811k wafer maps from real production, ~173k of them labelled with
  one of nine classes (eight spatial defect patterns plus `none`). A wafer map
  records the pass/fail outcome of electrical probing per die, so the "image" is a
  low-resolution spatial failure signature, not a photograph. *Supervised track.*
- **MVTec AD** — high-resolution images of industrial objects and textures with
  pixel-level anomaly masks, trained without defective examples. *Unsupervised
  track*, which matters because a real fab cannot enumerate its defect classes in
  advance and needs a detector that can say "this is unlike anything normal".

## 2. Related work

**Wafer-map pattern recognition.** Wu et al. (2015) introduced WM-811K with
hand-crafted rotation-invariant features and similarity ranking; it is now the
standard benchmark. Wang et al. (2020) extended the problem to *mixed-type*
defects, where several patterns co-occur on one wafer, releasing MixedWM38 and
showing deformable convolutions help when a defect's spatial support is irregular.
Our supervised track follows the single-label WM-811K formulation; the multi-label
extension is future work.

**Backbones and imbalance.** We use standard ImageNet backbones — ResNets (He et
al., 2016) and EfficientNets (Tan & Le, 2019) — rather than a bespoke architecture,
because the interesting difficulty here is the data, not the function class. For
imbalance we include focal loss (Lin et al., 2017) alongside resampling and loss
reweighting, treating the choice as an empirical question.

**Unsupervised industrial anomaly detection.** MVTec AD (Bergmann et al., 2019)
established the benchmark; PaDiM (Defard et al., 2021) models patch-embedding
distributions and PatchCore (Roth et al., 2022) retrieves from a coreset memory
bank of normal patches. Both are attractive for SiC because they need only
defect-free examples for training.

**Leakage.** Kapoor & Narayanan (2023) survey how data leakage has produced
irreproducible results across machine-learning-based science, with grouped data
split at the wrong granularity among the recurring causes. Our lot-grouping
discipline is a direct response, and we make it an assertion rather than a convention.

## 3. Data and evaluation protocol

### 3.1 WM-811K

Each record holds a 2-D wafer map with entries `0` (off-wafer), `1` (die passed)
and `2` (die failed), a lot name, and for the labelled subset one of nine classes:
`none`, `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random`, `Scratch`,
`Near-full`. Unlabelled wafers are dropped. Maps vary in physical size, so they are
resized to a fixed 64×64 grid.

Two properties drive every later decision: the classes are severely imbalanced
(`none` ≈ 85% of labelled wafers), and the records are **grouped** — many wafers
belong to the same lot.

### 3.2 Why the split must be grouped by lot

Wafers processed in one lot pass through the same tools, in the same order, within
the same time window. They share process excursions: if a chamber drifted, every
wafer in that lot carries the signature. A random split places near-duplicate
wafers on both sides of the train/test boundary, and the resulting score measures
partly the model's ability to recognise *lots it has already seen* rather than
defect patterns in general.

We split by lot with `GroupShuffleSplit`, holding out 15% of lots for validation
and 15% for test. The guarantee is enforced, not documented: `assert_no_lot_leakage`
recomputes the lot sets of all three splits and raises if any lot appears in two of
them, and it runs both when the split is created and every time the processed file
is loaded. A leak becomes an immediate, loud failure rather than an inflated number.

What this does **not** control: lot grouping removes within-lot leakage, but it does
not make the split chronological, so a model may still be validated on a time period
it was trained across. See [Threats to validity](#7-threats-to-validity).

### 3.3 Preprocessing

Maps are resized with **nearest-neighbour interpolation and nothing else**. This is
a correctness requirement, not a preference: bilinear or bicubic resampling produces
intermediate values between `0`, `1` and `2` that correspond to no physical die
state and that the model never sees during training. Values are then divided by 2,
mapping `{0,1,2}` → `{0, 0.5, 1}`, and the single channel is repeated three times to
match the input geometry ImageNet backbones expect. No ImageNet mean/std
normalisation is applied, because the input is a ternary occupancy map, not a photograph.

### 3.4 Augmentation

Augmentation is restricted to the dihedral group: the four 90° rotations and
horizontal flips. Wafers are round and have no canonical orientation, so these map a
valid wafer map to another valid wafer map. Scaling, shearing, perspective warps and
colour jitter do not — they produce maps that could not arise on a real wafer, and
several would break the `{0,1,2}` value domain. The augmentation is deliberately
weak because the geometry of the data, not a transformation budget, decides what is
admissible.

### 3.5 Metrics

Accuracy is **not** a headline number. With `none` at ~85% prevalence, a model that
predicts `none` unconditionally scores ~85% accuracy while detecting nothing. We report:

- **macro-F1** — unweighted mean of per-class F1, giving a rare pattern the same
  weight as the dominant one;
- **balanced accuracy** — mean per-class recall;
- **defect recall / precision** — on the collapsed binary question "does this wafer
  carry any defect pattern?", the first question a fab asks and the one with direct
  cost consequences;
- **per-class F1** — because an aggregate can hide a class the model has abandoned.

Model selection uses validation macro-F1. The test split is evaluated once, at the
end, with the best checkpoint.

### 3.6 Imbalance strategies

Four strategies are compared under an otherwise identical configuration:

| strategy | what it does |
|---|---|
| `none` | plain cross-entropy |
| `weighted_sampler` | inverse-frequency oversampling of rare classes per batch |
| `class_weights` | inverse-frequency weighted cross-entropy |
| `focal` | focal loss (γ=2) combined with class weights |

Treating this as an ablation rather than picking one reflects that the right answer
depends on the imbalance ratio and the model, and that the comparison is cheap
relative to how informative it is.

## 4. Deployment: the model bundle

**A training checkpoint is not a deliverable.** The checkpoint written by the
training script holds a state dictionary and an architecture name, so reconstructing
a usable model requires this repository, a compatible torchvision version, and
knowledge of the preprocessing contract that lives in the training code. Handing
that to a colleague, or to a fab's inspection system, does not work.

The export step produces a self-describing **bundle**:

| file | purpose |
|---|---|
| `model.onnx` | the network as a framework-independent graph, **weights inlined in the single file** so nothing can be left behind in transit. Scoring needs only NumPy and an ONNX runtime — no PyTorch, no torchvision, none of this repo's code, not necessarily Python. |
| `model.ts` | a TorchScript copy for callers already running PyTorch |
| `bundle.json` | class list and `none` index, input geometry, the preprocessing contract stated explicitly (nearest interpolation, divisor 2.0, three repeated channels, NCHW, no ImageNet normalisation), plus provenance: checkpoint SHA-256, recorded test metrics, library versions, export timestamp |
| `model_card.md` | what the model is, how it scored, where it should not be trusted |

**Export-time verification.** An export that silently diverges from its source model
is worse than no export, so the step verifies rather than asserts. Both the
TorchScript and ONNX graphs are run on a fixed pseudo-random probe batch and compared
against the source PyTorch model; a maximum absolute logit deviation above 1e-4
aborts the export, and the measured deviations are recorded in the bundle. In our
runs TorchScript reproduces the source model **exactly** and ONNX agrees to within
**4e-6**, with identical `argmax` decisions.

**Inference.** The companion module accepts a bundle directory or zip and scores
whatever form new data arrives in: a single `.npy` map, a stack, a processed `.npz`
(optionally filtered to one split), PNG/JPEG images decoded back through the
`{0,127,255}` convention, or a whole directory. Maps of any size are resized to the
bundle's input geometry. Output is a table with the predicted pattern, its
confidence, the collapsed defect probability `1 − p(none)`, and the full per-class
distribution. Predictions can be accompanied by candidate process root causes from a
rule table — a hypothesis generator for the engineer, explicitly **not** a verdict.

Because preprocessing at inference must be the preprocessing the weights were fitted
under, the resize uses OpenCV when installed, matching the training-time call
exactly. A NumPy fallback keeps the dependency-free path available, and its one
divergence — destination indices whose source coordinate is an exact integer, where
OpenCV's floating-point `floor` can take the previous column — is documented rather
than hidden. In the normal flow the question does not arise, because the maps have
already been resized once during preparation and pass through untouched.

## 5. Experiments

### 5.1 Pipeline validation on synthetic data

Before any download, a generator writes a small dataset with the same schema as the
real distribution — including unlabelled rows, variable map sizes, lot structure and
a comparable class imbalance — by drawing the eight defect patterns as explicit
geometric priors (a central disc, an annulus, an edge ring, an angular edge sector, a
local blob, elevated random failure, a linear scratch, near-total failure) on a
circular wafer support with a background failure rate. The full pipeline then runs on
it: preparation, lot-grouped splitting, CNN training, export and inference.

The purpose is to establish that every stage executes and the pieces fit together,
which it does: the run below completed, produced a checkpoint, exported cleanly with
verified ONNX agreement, and scored new maps through the inference path.

> ⚠️ **These numbers must not be read as performance.** The synthetic patterns are
> generated from the clean geometric templates the model is then asked to recognise,
> with no label noise, no ambiguous or overlapping patterns and no measurement
> artefacts. Near-perfect scores are the expected outcome of a correctly wired
> pipeline on such data and carry no information about real wafer maps. They are
> reported only to document what was actually run.

### 5.2 Protocol for the real WM-811K runs

The real experiments need the Kaggle distribution of WM-811K (`LSWMD.pkl`) and GPU
time; the pipeline runs them unchanged:

```bash
python prepare_wm811k.py --pkl data/raw/LSWMD.pkl --size 64

# four-way imbalance ablation on a fixed backbone
for s in none weighted_sampler class_weights focal; do
  python train_cnn.py --model resnet18 --imbalance $s
done

# carry the best strategy to larger pretrained backbones
python train_cnn.py --model resnet50        --imbalance <best> --pretrained
python train_cnn.py --model efficientnet_b3 --imbalance <best> --pretrained

# YOLO baselines on the identical split
python train_yolo.py --model yolov8n-cls.pt --epochs 30

python paper/make_results.py --tag wm811k     # regenerates the tables below
```

Selection is on validation macro-F1 and the test split is touched once per run.
Every run logs to MLflow. Because the tables are regenerated from run outputs by a
single script, completing these runs fills them in without any manual transcription.

## 6. Results

Regenerated by `paper/make_results.py` from `outputs/*/test_metrics.json`:

<!-- BEGIN:results -->
| run | macro-F1 | balanced acc. | accuracy | defect recall | defect precision |
|---|---|---|---|---|---|
| `resnet18-synthetic` | 0.992 | 0.991 | 0.995 | 1.000 | 1.000 |
<!-- END:results -->

Per-class F1:

<!-- BEGIN:per-class -->
| run | none | Center | Donut | Edge-Loc | Edge-Ring | Loc | Random | Scratch | Near-full |
|---|---|---|---|---|---|---|---|---|---|
| `resnet18-synthetic` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.971 | 1.000 | 0.957 | 1.000 |
<!-- END:per-class -->

Export verification for this run, read from `bundle.json`:

| check | value |
|---|---|
| max abs. logit difference, TorchScript vs source | 0.0 |
| max abs. logit difference, ONNX vs source | 1.9e-06 |
| ONNX opset | 17 |

Figures (in `paper/figures/`):

- `training_curves.png` — training and validation trajectories. Validation macro-F1
  saturating near 1.0 reflects the separability of the synthetic patterns, not model quality.
- `class_balance.png` — class counts per lot-grouped split (log scale). The real
  WM-811K distribution is considerably more skewed.
- `samples.png` — example wafer maps per class.
- `confusion_resnet18-synthetic.png` — row-normalised test confusion matrix.

## 7. Threats to validity

**The reported numbers are not real-data performance.** The dominant limitation,
restated deliberately: the quantitative results come from synthetic data generated to
validate the pipeline. No claim about real WM-811K or SiC performance is supported by them.

**Lot grouping is necessary, not sufficient.** It removes within-lot leakage but does
not make the evaluation chronological, so a model can still be validated on a time
window it was trained across, and slow tool drift remains uncontrolled. A
chronological, lot-disjoint protocol would be strictly stronger and is the natural
next hardening step.

**Label noise in WM-811K.** Labels are human-assigned and the dataset is known to
contain ambiguous and contested cases; mixed-type wafers in particular are forced
into a single class by the formulation we adopt. This bounds achievable macro-F1 by
an unknown amount.

**The proxy gap is real.** A wafer map is a per-die pass/fail lattice from electrical
probing. A SiC inspection image is a photoluminescence, optical or X-ray topography
measurement of crystal structure. They differ in modality, resolution and defect
physics. What transfers is the protocol, the metric discipline and the deployment
machinery; the trained **weights** should not be expected to transfer at all, and the
model card says so.

**Closed-set classification.** The classifier assigns every wafer to one of nine
known classes and cannot abstain. A novel excursion signature will be forced into the
nearest class, possibly with high confidence. This is the gap the unsupervised
anomaly-detection track exists to fill.

**Single seed.** Runs here use one seed. Differences between imbalance strategies
should not be interpreted before they are shown to exceed seed-to-seed variation.

## 8. Reproducibility

Every stage runs from the command line with pinned seeds, and the dataset loader
re-verifies the lot-disjointness invariant on each load. The synthetic validation
path requires no downloads and runs on CPU in a few minutes. Experiment metadata,
parameters and metrics are logged to MLflow; model artifacts are exported with their
checkpoint SHA-256 and library versions recorded in the bundle. The tables and
figures here are produced by `paper/make_results.py` directly from run outputs, so a
stale number is a regenerable artefact rather than a transcription error.

## 9. Conclusion and future work

We have described a defect-inspection pipeline for SiC power devices built
deliberately on public proxy data, so the parts of the problem that are not
SiC-specific — splitting, imbalance, metric choice, and shipping a model somebody
else can run — can be solved now, with the SiC-specific imaging front-end substituted
later. The evaluation protocol enforces lot-disjoint splits as a runtime invariant;
the reporting avoids the accuracy trap an 85%-majority dataset sets; and the
deployment artifact separates a trained model from the code that produced it,
verified against its source at export time.

The immediate next step is to complete the real WM-811K runs and populate the results
table with numbers that mean something. Beyond that:

- a chronological, lot-disjoint split, to control drift as well as grouping;
- the multi-label mixed-type formulation of Wang et al. (2020), closer to how defects
  actually co-occur;
- the MVTec AD anomaly-detection track, to give the system an open-set response;
- seed repetitions with confidence intervals before any strategy is declared better;
- transfer to real SiC photoluminescence or X-ray topography imagery, where the proxy
  assumption is finally tested rather than relied upon.

## References

1. M.-J. Wu, J.-S. R. Jang, J.-L. Chen. "Wafer Map Failure Pattern Recognition and
   Similarity Ranking for Large-Scale Data Sets." *IEEE Trans. Semiconductor
   Manufacturing* 28(1):1–12, 2015.
2. J. Wang, C. Xu, Z. Yang, J. Zhang, X. Li. "Deformable Convolutional Networks for
   Efficient Mixed-Type Wafer Defect Pattern Recognition." *IEEE Trans. Semiconductor
   Manufacturing* 33(4):587–596, 2020.
3. K. He, X. Zhang, S. Ren, J. Sun. "Deep Residual Learning for Image Recognition."
   *CVPR*, 2016.
4. M. Tan, Q. V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks." *ICML*, 2019.
5. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. "Focal Loss for Dense Object
   Detection." *ICCV*, 2017.
6. P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. "MVTec AD — A Comprehensive
   Real-World Dataset for Unsupervised Anomaly Detection." *CVPR*, 2019.
7. K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler. "Towards Total
   Recall in Industrial Anomaly Detection." *CVPR*, 2022.
8. T. Defard, A. Setkov, A. Loesch, R. Audigier. "PaDiM: A Patch Distribution Modeling
   Framework for Anomaly Detection and Localization." *ICPR Workshops*, 2021.
9. S. Kapoor, A. Narayanan. "Leakage and the Reproducibility Crisis in
   Machine-Learning-Based Science." *Patterns* 4(9), 2023.
10. T. Kimoto, J. A. Cooper. *Fundamentals of Silicon Carbide Technology.* Wiley, 2014.
11. T. Kimoto. "Material Science and Device Physics in SiC Technology for High-Voltage
    Power Devices." *Jpn. J. Appl. Phys.* 54(4):040103, 2015.
12. A. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning
    Library." *NeurIPS*, 2019.
13. F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *JMLR*
    12:2825–2830, 2011.
