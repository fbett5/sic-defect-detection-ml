# SiC Defect Detection ML

Computer-vision models for semiconductor defect detection, built as a proxy
pipeline for SiC power-device inspection:

| Track | Dataset | Task | Models | Headline metrics |
|---|---|---|---|---|
| 1 | WM-811K wafer maps | 9-class pattern classification | ResNet / EfficientNet, YOLOv8-cls | macro-F1, balanced accuracy |
| 2 | MVTec AD | Unsupervised anomaly detection + localisation | PatchCore, PaDiM | image AUROC, pixel AUROC, PRO |

All runs log to MLflow so you can compare them in one dashboard.

---

## 1. What you need

- **Python 3.11** (3.10–3.12 fine). Check with `python --version`.
- **RAM:** 16 GB recommended. Loading the full WM-811K pickle takes several GB.
- **GPU:** strongly recommended (NVIDIA + CUDA). A CPU works for the smoke test
  and for small runs, but a full ResNet-18 epoch on 120k wafers takes 20+ min
  on a laptop CPU vs. about a minute on a GPU. No GPU? Google Colab's free T4 works.
- **Disk:** ~3 GB for WM-811K, ~5 GB for MVTec AD.
- A **Kaggle account** (free) to download WM-811K.

## Run on Google Colab (free GPU)

Open `colab_runner.ipynb` in Colab: File > Open notebook > GitHub tab, search
`fbett5/sic-defect-detection-ml` (tick "Include private repos" if the repo is private).
Set Runtime > Change runtime type > **T4 GPU** and run the cells in order.

Training cells show a live dashboard while they run (progress, loss and score curves,
confusion matrix, sample predictions, GPU use), via `sicdefect.live.watch()`.

Data, MLflow history and checkpoints are saved to `MyDrive/WafferMap-811/` in
Google Drive. Put `LSWMD.pkl` there, or add `KAGGLE_USERNAME` / `KAGGLE_KEY` as
Colab secrets and the notebook downloads it. If the repo is private, also add a
`GITHUB_TOKEN` secret. The first cell of the notebook explains each one.

## Windows quick start (Festus's machine)

The data path is already set in `sicdefect/paths.py`:
`C:\Users\festu\Downloads\Sofia University\ML Projects\WafferMap-811`
(the scripts look inside that folder for `LSWMD.pkl`, including subfolders,
so extract the Kaggle zip there first).

In **PowerShell**:

```powershell
cd "C:\Users\festu\Downloads\Sofia University\ML Projects"
git clone https://github.com/fbett5/sic-defect-detection-ml.git
cd sic-defect-detection-ml

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

nvidia-smi        # prints a GPU table? -> NVIDIA GPU present
# GPU:    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# No GPU: pip install torch torchvision
pip install -r requirements.txt

pytest -q tests                       # sanity check
python prepare_wm811k.py              # reads your WafferMap-811 folder
python train_cnn.py --limit 5000 --epochs 2   # quick real-data check
python train_cnn.py                   # full run
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://127.0.0.1:5000
```

Data moved? Edit `WM811K_DEFAULT` in `sicdefect/paths.py`, set
`$env:WM811K_PATH = "D:\new\place"`, or pass `--pkl "D:\new\place"`.

## 2. Setup (one time)

```bash
git clone https://github.com/fbett5/sic-defect-detection-ml.git
cd sic-defect-detection-ml

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# NVIDIA GPU: install the CUDA build of PyTorch first (pick your CUDA version at pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Check it works (takes ~10 s, no data needed):

```bash
pytest -q tests
```

## 3. Smoke test the whole pipeline on fake data (~2 min, CPU is fine)

Do this before downloading anything, so you know every script runs.

```bash
python tools/make_synthetic_wm811k.py
python prepare_wm811k.py --pkl data/raw/LSWMD_synthetic.pkl --out data/synthetic
python train_cnn.py --data data/synthetic/wm811k_64.npz --epochs 3 --experiment smoke
python train_yolo.py --data data/synthetic/yolo_64 --epochs 2 --experiment smoke

# export the result and score new data with it (Step 6)
python -m sicdefect.export --ckpt outputs/resnet18-weighted_sampler/best.pt
python -m sicdefect.infer --bundle exported/resnet18-weighted_sampler \
                          --input data/synthetic/wm811k_64.npz --split test --n 5
```

The synthetic patterns are easy, so near-perfect scores here only mean the
plumbing works.

## 4. Get the real WM-811K data

Download **LSWMD.pkl** from Kaggle ("WM-811K wafer map"), either in the browser
or with the Kaggle CLI:

```bash
pip install kaggle        # then put your API token in ~/.kaggle/kaggle.json
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw --unzip
```

You should end up with `data/raw/LSWMD.pkl` (or anywhere else; see `sicdefect/paths.py`).

## 5. Run order

### Step 1 — prepare data (once)
```bash
python prepare_wm811k.py                                   # path from sicdefect/paths.py
python prepare_wm811k.py --pkl data/raw/LSWMD.pkl --size 64   # or give it explicitly
```
Keeps the ~173k labeled wafers, resizes to 64×64 (nearest-neighbour, so die
values stay 0/1/2), and splits **by lot** into train/val/test. It refuses to
continue if any lot lands in two splits. Check `data/processed/split_summary.csv`
and `data/processed/samples.png` before training.

### Step 2 — CNN baselines + imbalance ablation
```bash
python train_cnn.py --model resnet18 --imbalance weighted_sampler
python train_cnn.py --model resnet18 --imbalance focal
python train_cnn.py --model resnet18 --imbalance class_weights
python train_cnn.py --model resnet18 --imbalance none
python train_cnn.py --model efficientnet_b3 --imbalance <best one> --pretrained
python train_cnn.py --model resnet50        --imbalance <best one> --pretrained
```
Each run saves `outputs/<run>/best.pt`, a test report and a confusion matrix.

### Step 3 — YOLO on the identical split
```bash
python train_yolo.py --model yolov8n-cls.pt --epochs 30
python train_yolo.py --model yolov8s-cls.pt --epochs 30
```

### Step 4 — anomaly detection on MVTec AD
anomalib pins its own library versions, so give it its own environment:
```bash
python -m venv .venv-anomaly && source .venv-anomaly/bin/activate
pip install -r requirements-anomaly.txt
python run_anomaly.py --model patchcore                 # 5 texture categories
python run_anomaly.py --model padim
python run_anomaly.py --model patchcore --categories all
```
MVTec AD downloads automatically into `data/MVTecAD` on first run.

### Step 5 — predictions with root-cause hints
```bash
python predict.py --ckpt outputs/resnet18-weighted_sampler/best.pt \
                  --npz data/processed/wm811k_64.npz --n 10
```
Prints the predicted pattern plus candidate process causes from
`sicdefect/rootcause.py`. Edit that table as you learn what applies to your process.

### Step 6 — export a portable model, and run it on new data

`outputs/<run>/best.pt` is a training checkpoint, not something you can hand to
anyone: re-loading it needs this repo, a matching torchvision, and the
preprocessing rules that live in the training code. Export turns it into a
self-contained bundle instead.

```bash
python -m sicdefect.export --ckpt outputs/resnet18-weighted_sampler/best.pt --zip
```

That writes `exported/resnet18-weighted_sampler/`:

| file | what it is |
|---|---|
| `model.onnx` | the network as a portable graph, weights inlined in the one file. Scoring it needs only `numpy` + `onnxruntime` — no PyTorch, no torchvision, none of this repo, not even Python. |
| `model.ts` | TorchScript copy, if you already run PyTorch |
| `bundle.json` | class list, input size, the **exact** preprocessing contract, plus the checkpoint's SHA-256, its test metrics and library versions |
| `model_card.md` | what it scored and where not to trust it |

Export verifies itself: both graphs are run against the source PyTorch model on a
fixed probe batch and the export **fails** if they disagree by more than 1e-4. The
measured difference is recorded in `bundle.json` (TorchScript matches exactly; ONNX
to ~2e-6).

Then point it at new wafer maps — a `.npy`, a stack, an `.npz`, PNGs, or a whole
directory, at any map size:

```bash
python -m sicdefect.infer --bundle exported/resnet18-weighted_sampler \
                          --input new_wafers/ --out scored.csv --root-causes
```

You get per-wafer predicted pattern, confidence, `defect_probability`
(= 1 − p(none)) and the full per-class distribution. `--backend onnx|torch` picks
the runtime; `--split test` filters a processed `.npz`.

## 6. View results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000. Sort runs by `test_macro_f1` (WM-811K) or
`pixel_AUPRO` (MVTec).

**Judge WM-811K on macro-F1 and balanced accuracy, not accuracy.** About 85%
of wafers are `none`, so predicting `none` every time scores ~85% accuracy.
Choose models on **val** metrics; look at **test** only for the final comparison.

## 7. Layout

```
sicdefect/            library code
  wm811k.py           loading, lot-grouped split, dataset, dihedral augmentation
  models.py           backbone construction (shared by training, export, inference)
  metrics.py          macro-F1, balanced acc, defect recall, confusion plot
  losses.py           focal loss
  rootcause.py        pattern -> likely process causes
  utils.py            device selection, seeding, MLflow setup
  export.py           Step 6 — checkpoint -> portable ONNX/TorchScript bundle
  infer.py            Step 6 — apply a bundle to new .npy/.npz/PNG/directory data
prepare_wm811k.py     Step 1
train_cnn.py          Step 2
train_yolo.py         Step 3
run_anomaly.py        Step 4
predict.py            Step 5
tools/make_synthetic_wm811k.py   fake data for smoke tests
tests/                pytest unit tests
paper/                the write-up (main.tex + paper.md) and its results harness
extras/               optional side track, not part of the pipeline — see its README
```

## The paper

`paper/` holds the write-up of this project — the lot-grouped evaluation protocol,
the imbalance handling, the metric argument, and the deployment artifact — as
`main.tex` (for submission) and `paper.md` (readable).

Every number in it is generated from run outputs, never typed by hand:

```bash
python paper/make_results.py --tag wm811k
```

That reads each `outputs/<run>/test_metrics.json` and rewrites the tables and
figures. **The results currently committed come from the synthetic smoke-test run**,
so they only demonstrate that the pipeline executes — the paper says so in the
abstract, the results section and the threats-to-validity section. Replace them with
real WM-811K numbers before circulating it. See `paper/README.md`.

## 8. Troubleshooting

- **`LSWMD.pkl` won't load** (`No module named 'pandas.indexes'` or similar):
  the file was saved with a 2018 pandas. The loader has a compatibility shim; if
  that still fails, make a Python 3.10 env, `pip install "pandas<2"`, load it and
  re-save with `df.to_pickle("data/raw/LSWMD_v2.pkl")`, then use that file.
- **CUDA out of memory:** lower `--batch-size` (CNN) or `--batch` (YOLO).
- **DataLoader errors on Windows/macOS:** add `--workers 0`.
- **Want a quick run on real data:** `python train_cnn.py --limit 5000 --epochs 2`.
