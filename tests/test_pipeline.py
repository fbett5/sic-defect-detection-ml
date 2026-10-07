import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sicdefect.losses import FocalLoss  # noqa: E402
from sicdefect.metrics import classification_metrics  # noqa: E402
from sicdefect.wm811k import (  # noqa: E402
    CLASSES, WaferDataset, assert_no_lot_leakage, load_wm811k, lot_grouped_split, resize_map,
)


def test_resize_keeps_discrete_values():
    m = np.random.default_rng(0).integers(0, 3, (37, 41)).astype(np.uint8)
    r = resize_map(m, 64)
    assert r.shape == (64, 64)
    assert set(np.unique(r)) <= {0, 1, 2}


def test_lot_split_has_no_leakage():
    lots = np.repeat([f"lot{i}" for i in range(60)], 20)
    split = lot_grouped_split(lots, 0.15, 0.15, seed=1)
    assert set(split) == {"train", "val", "test"}
    assert_no_lot_leakage(lots, split)


def test_leakage_is_detected():
    lots = np.array(["a", "a", "b"])
    with pytest.raises(AssertionError):
        assert_no_lot_leakage(lots, np.array(["train", "test", "val"]))


def test_dataset_shapes_and_augment():
    maps = np.random.default_rng(0).integers(0, 3, (4, 64, 64)).astype(np.uint8)
    ds = WaferDataset(maps, np.arange(4) % len(CLASSES), augment=True)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert float(x.max()) <= 1.0


def test_metrics_penalise_majority_guessing():
    y = np.array([0] * 90 + list(range(1, 9)) + [1, 2])
    always_none = np.zeros_like(y)
    m = classification_metrics(y, always_none)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["macro_f1"] < 0.15
    assert m["defect_recall"] == 0.0


def test_focal_loss_runs():
    loss = FocalLoss(2.0)(torch.randn(8, 9, requires_grad=True), torch.randint(0, 9, (8,)))
    loss.backward()
    assert loss.item() > 0


def test_loads_synthetic_pickle(tmp_path):
    pkl = tmp_path / "LSWMD.pkl"
    subprocess.run([sys.executable, str(ROOT / "tools/make_synthetic_wm811k.py"),
                    "--out", str(pkl), "--n", "200"], check=True, cwd=ROOT)
    df = load_wm811k(pkl)
    assert 0 < len(df) < 200  # unlabeled rows dropped
    assert set(df["label"]) <= set(CLASSES)


def test_sem_loading_crop_and_split(tmp_path):
    import cv2

    from sicdefect.sem import SEMDataset, load_gray, scan_folder, stratified_split

    for c in ("dirt", "scratch"):
        (tmp_path / c).mkdir()
        for i in range(6):
            img = np.full((100, 80), 100, np.uint16) * 257  # 16-bit, like many SEM TIFFs
            img[90:] = 0  # info bar
            cv2.imwrite(str(tmp_path / c / f"{i}.tif"), img)
    files, labels, classes = scan_folder(tmp_path)
    assert classes == ["dirt", "scratch"] and len(files) == 12
    g = load_gray(files[0], crop_bottom=0.1)
    assert g.dtype == np.uint8 and g.shape == (90, 80)
    split = stratified_split(labels, 0.2, 0.2, seed=0)
    for c in (0, 1):
        assert set(split[labels == c]) == {"train", "val", "test"}
    x, y = SEMDataset(files, labels, img_size=64, crop_bottom=0.1, train=True)[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
