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


# ------------------------------------------------------------------ detection track
def test_iou_and_matching():
    from sicdefect.det.boxes import greedy_match, iou_matrix

    m = iou_matrix([(0, 0, 10, 10), (20, 20, 30, 30)], [(0, 0, 10, 10), (5, 0, 15, 10)])
    assert m[0, 0] == pytest.approx(1.0) and m[0, 1] == pytest.approx(1 / 3) and m[1].max() == 0
    # higher-score prediction takes the GT first
    assert greedy_match(iou_matrix([(0, 0, 10, 10), (0, 0, 10, 10)], [(0, 0, 10, 10)]), np.array([0.2, 0.9]), 0.5) == {1: 0}


def test_detection_metrics():
    from sicdefect.det.evaluate import average_precision, operating_point

    gt = {"a": [(0, (0, 0, 10, 10)), (1, (20, 20, 30, 30))], "b": [(0, (5, 5, 15, 15))], "clean": []}
    perfect = {k: [(c, b, 0.9) for c, b in v] for k, v in gt.items()}
    ap = average_precision(gt, perfect, 2)
    assert ap["mAP50"] == pytest.approx(1.0) and ap["mAP50_95"] == pytest.approx(1.0)
    preds = {"a": [(0, (0, 0, 10, 10), 0.9), (0, (20, 20, 30, 30), 0.8)],   # TP + wrong class
             "b": [], "clean": [(1, (1, 1, 5, 5), 0.7), (1, (40, 40, 50, 50), 0.1)]}  # FP, below threshold
    op = operating_point(gt, preds, ["x", "y"], 0.5, clean_ids={"clean"})
    assert op["counts"] == {"gt_defects": 3, "correct": 1, "wrong_class": 1, "false_alarms": 1, "missed": 1,
                            "images": 3}
    assert op["clean_images"]["false_alarm_image_rate"] == 1.0
    assert op["confusion"][1][0] == 1 and op["confusion"][2][1] == 1 and op["confusion"][0][2] == 1


def test_preprocess_modes_and_mask_boxes():
    from sicdefect.det.data import mask_to_boxes, read_labels, write_labels
    from sicdefect.det.preprocess import make_input

    img = np.full((40, 40), 200, np.uint8)
    ref = img.copy()
    img[10:15, 10:15] = 0
    x = make_input(img, ref, "refdiff")
    assert x.shape == (40, 40, 3) and x[12, 12, 2] == 200 and x[0, 0, 2] == 0
    assert make_input(img, None, "gray").shape == (40, 40, 3)
    with pytest.raises(ValueError):
        make_input(img, None, "refdiff")
    m = np.zeros((40, 40), np.uint8)
    m[5:10, 5:12] = 1
    m[30:35, 30:35] = 1
    assert sorted(mask_to_boxes(m, min_area=4)) == [(5, 5, 12, 10), (30, 30, 35, 35)]


def test_label_roundtrip(tmp_path):
    from sicdefect.det.data import read_labels, write_labels

    write_labels(tmp_path / "a.txt", [(2, (10, 20, 50, 60))], 100, 200)
    (c, b), = read_labels(tmp_path / "a.txt", 100, 200)
    assert c == 2 and np.allclose(b, (10, 20, 50, 60))


def test_crop_with_context_clips():
    from sicdefect.det.crops import cut

    img = np.zeros((100, 100, 3), np.uint8)
    assert cut(img, (90, 90, 100, 100)).shape == (64, 64, 3)
