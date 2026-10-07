"""WM-811K wafer map loading, lot-grouped splitting and PyTorch dataset.

The Kaggle file ``LSWMD.pkl`` is a pandas DataFrame with columns:
    waferMap        2-D uint8 array, 0 = off-wafer, 1 = pass die, 2 = fail die
    dieSize         float
    lotName         str, e.g. "lot1"
    waferIndex      float
    trianTestLabel  array([['Training']]) / array([['Test']]) or empty
    failureType     array([['Center']]) ... array([['none']]) or empty (unlabeled)

Only ~173k of the 811k wafers are labeled; unlabeled rows are dropped.

Splits are grouped by lot so wafers from one lot never appear in two splits.
Wafers within a lot share process history, so a random split would leak and
inflate test scores.
"""
from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler

CLASSES = [
    "none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Random", "Scratch", "Near-full",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEFECT_CLASSES = CLASSES[1:]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
class _CompatUnpickler(pickle.Unpickler):
    """Load the 2018-era pickle under modern pandas.

    LSWMD.pkl references modules that newer pandas renamed or removed
    (``pandas.indexes``, ``Int64Index``). This remaps them.
    """

    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        if name in ("Int64Index", "UInt64Index", "Float64Index"):
            return pd.Index
        if module == "pandas.core.indexes.numeric":
            module = "pandas.core.indexes.base"
        return super().find_class(module, name)


def _read_lswmd(path: Path) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except Exception as first_err:  # noqa: BLE001
        try:
            with open(path, "rb") as f:
                return _CompatUnpickler(io.BufferedReader(f), encoding="latin1").load()
        except Exception as second_err:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load {path}.\n"
                f"  pandas.read_pickle: {first_err!r}\n"
                f"  compat loader:      {second_err!r}\n"
                "Workaround: create a Python 3.10 env with `pip install 'pandas<2'`, "
                "load the file there and re-save it with df.to_pickle('LSWMD_v2.pkl')."
            ) from second_err


def _unwrap(v) -> str | None:
    """failureType / trianTestLabel come as nested arrays, sometimes empty."""
    arr = np.asarray(v, dtype=object).ravel()
    if arr.size == 0:
        return None
    s = str(arr[0]).strip()
    return s or None


def load_wm811k(pkl_path: str | Path) -> pd.DataFrame:
    """Load LSWMD.pkl and return labeled wafers only.

    Returns a DataFrame with columns: waferMap, lot, label (str), y (int).
    """
    path = Path(pkl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'WM-811K wafer map' from Kaggle "
            "(file LSWMD.pkl) and put it in data/raw/."
        )
    df = _read_lswmd(path)
    if "lotName" not in df.columns or "waferMap" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")

    out = pd.DataFrame({
        "waferMap": df["waferMap"].values,
        "lot": df["lotName"].astype(str).values,
        "label": [_unwrap(v) for v in df["failureType"].values],
    })
    out = out[out["label"].isin(CLASSES)].reset_index(drop=True)
    out["y"] = out["label"].map(CLASS_TO_IDX).astype(np.int64)
    return out


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def resize_map(wafer: np.ndarray, size: int) -> np.ndarray:
    """Resize with NEAREST interpolation so values stay in {0, 1, 2}."""
    wafer = np.asarray(wafer, dtype=np.uint8)
    return cv2.resize(wafer, (size, size), interpolation=cv2.INTER_NEAREST)


def lot_grouped_split(
    lots: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Return an array of 'train' / 'val' / 'test' with no lot in two splits."""
    n = len(lots)
    idx = np.arange(n)
    split = np.empty(n, dtype=object)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trval, test = next(gss.split(idx, groups=lots))
    split[test] = "test"

    rel_val = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr, va = next(gss2.split(trval, groups=lots[trval]))
    split[trval[tr]] = "train"
    split[trval[va]] = "val"

    assert_no_lot_leakage(lots, split)
    return split


def assert_no_lot_leakage(lots: np.ndarray, split: np.ndarray) -> None:
    sets = {s: set(lots[split == s]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sets[a] & sets[b]
        if shared:
            raise AssertionError(
                f"Lot leakage between {a} and {b}: {len(shared)} lots, e.g. {sorted(shared)[:5]}"
            )


def load_processed(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load the arrays written by prepare_wm811k.py."""
    d = np.load(npz_path, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    assert_no_lot_leakage(data["lot"], data["split"])
    return data


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def encode(maps: np.ndarray) -> np.ndarray:
    """{0,1,2} -> {0.0, 0.5, 1.0} float32."""
    return maps.astype(np.float32) / 2.0


class WaferDataset(Dataset):
    """Wafer maps as 3-channel tensors (grayscale repeated) for ImageNet backbones.

    Augmentation is dihedral only (90-degree rotations + flips). Wafers are
    round, so these keep the map physically valid; scaling, shearing or color
    jitter would create maps that can't exist on a real wafer.
    """

    def __init__(self, maps: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.x = encode(maps)
        self.y = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        m = self.x[i]
        if self.augment:
            k = np.random.randint(4)
            m = np.rot90(m, k)
            if np.random.rand() < 0.5:
                m = np.fliplr(m)
            m = np.ascontiguousarray(m)
        t = torch.from_numpy(m).unsqueeze(0).repeat(3, 1, 1)
        return t, self.y[i]


def make_weighted_sampler(labels: np.ndarray, num_classes: int = len(CLASSES)) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = (1.0 / counts)[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)


def class_weights(labels: np.ndarray, num_classes: int = len(CLASSES)) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def to_png(m: np.ndarray) -> np.ndarray:
    """{0,1,2} -> {0,127,255} uint8 for saving as an image."""
    return (np.asarray(m, dtype=np.uint16) * 127).clip(0, 255).astype(np.uint8)


if __name__ == "__main__":  # quick inspection: python -m sicdefect.wm811k data/raw/LSWMD.pkl
    df = load_wm811k(sys.argv[1])
    print(df["label"].value_counts())
    print("lots:", df["lot"].nunique())
