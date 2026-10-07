"""An exported bundle must score new data correctly, on its own.

These tests train a tiny model from scratch so they never depend on a
checkpoint being present, then export it and check that every consumer path
agrees with the source PyTorch model.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sicdefect.export import export, load_checkpoint
from sicdefect.infer import (
    Bundle,
    decode_png,
    load_bundle,
    nearest_resize,
    prepare,
    read_input,
    score,
    softmax,
)
from sicdefect.models import build_model
from sicdefect.wm811k import CLASSES, resize_map, to_png

SIZE = 32


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A real (untrained) checkpoint in the format train_cnn.py writes."""
    d = tmp_path_factory.mktemp("ckpt")
    torch.manual_seed(0)
    model = build_model("resnet18", len(CLASSES), pretrained=False)
    path = d / "best.pt"
    torch.save(
        {"model": model.state_dict(), "arch": "resnet18", "classes": CLASSES,
         "epoch": 3, "val_macro_f1": 0.5},
        path,
    )
    (d / "test_metrics.json").write_text(
        json.dumps({"macro_f1": 0.5, "balanced_acc": 0.5, "accuracy": 0.9,
                    "defect_recall": 0.4, "f1_none": 0.95})
    )
    return path


@pytest.fixture(scope="module")
def bundle_dir(checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("bundles") / "tiny"
    return export(checkpoint, out, input_size=SIZE, make_zip=True)


@pytest.fixture(scope="module")
def maps():
    rng = np.random.default_rng(1)
    return rng.integers(0, 3, (6, SIZE, SIZE)).astype(np.uint8)


def test_checkpoint_validation_rejects_junk(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"model": {}}, bad)
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(bad)


def test_bundle_contains_everything_needed(bundle_dir):
    names = {p.name for p in bundle_dir.iterdir()}
    assert {"bundle.json", "model.onnx", "model.ts", "model_card.md"} <= names
    assert not any(n.endswith(".onnx.data") for n in names), "weights must be inlined"


def test_bundle_spec_records_the_preprocessing_contract(bundle_dir):
    spec = json.loads((bundle_dir / "bundle.json").read_text())
    assert spec["format_version"] == 1
    assert spec["classes"] == CLASSES
    assert spec["input_size"] == SIZE
    assert spec["preprocessing"]["resize_interpolation"] == "nearest"
    assert spec["preprocessing"]["scale_divisor"] == 2.0
    assert spec["none_index"] == CLASSES.index("none")
    assert spec["training"]["checkpoint_sha256"]


def test_export_verifies_itself(bundle_dir):
    spec = json.loads((bundle_dir / "bundle.json").read_text())
    assert spec["export"]["max_abs_logit_diff_torchscript"] < 1e-4
    if spec["export"]["max_abs_logit_diff_onnx"] is not None:
        assert spec["export"]["max_abs_logit_diff_onnx"] < 1e-4


def test_exported_logits_match_the_source_model(bundle_dir, checkpoint, maps):
    ck = load_checkpoint(checkpoint)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    b = load_bundle(bundle_dir, backend="onnx")
    x = prepare(list(maps), b)
    with torch.no_grad():
        reference = model(torch.from_numpy(x)).numpy()
    assert np.abs(b.logits(x) - reference).max() < 1e-4


def test_onnx_and_torchscript_backends_agree(bundle_dir, maps):
    bo = load_bundle(bundle_dir, backend="onnx")
    bt = load_bundle(bundle_dir, backend="torch")
    x = prepare(list(maps), bo)
    lo, lt = bo.logits(x), bt.logits(x)
    assert np.abs(lo - lt).max() < 1e-4
    assert (lo.argmax(1) == lt.argmax(1)).all()


def test_zip_bundle_is_equivalent(bundle_dir, maps):
    z = Path(str(bundle_dir) + ".zip")
    assert zipfile.is_zipfile(z)
    bz = load_bundle(z, backend="onnx")
    bd = load_bundle(bundle_dir, backend="onnx")
    x = prepare(list(maps), bd)
    np.testing.assert_allclose(bz.logits(x), bd.logits(x), atol=0)


def test_rejects_an_unknown_bundle_version(bundle_dir, tmp_path):
    spec = json.loads((bundle_dir / "bundle.json").read_text())
    d = tmp_path / "future"
    d.mkdir()
    (d / "bundle.json").write_text(json.dumps({**spec, "format_version": 99}))
    with pytest.raises(ValueError, match="format_version"):
        load_bundle(d)


def test_rejects_a_directory_that_is_not_a_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path)


# ---------------------------------------------------------------- preprocessing
def test_resize_matches_the_training_time_resize():
    rng = np.random.default_rng(0)
    for _ in range(200):
        h, w = rng.integers(8, 120, 2)
        m = rng.integers(0, 3, (h, w)).astype(np.uint8)
        size = int(rng.choice([32, 64]))
        np.testing.assert_array_equal(nearest_resize(m, size), resize_map(m, size))


def test_resize_keeps_the_value_domain():
    rng = np.random.default_rng(2)
    m = rng.integers(0, 3, (37, 41)).astype(np.uint8)
    r = nearest_resize(m, 64)
    assert r.shape == (64, 64)
    assert set(np.unique(r)) <= {0, 1, 2}


def test_resize_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        nearest_resize(np.zeros((2, 4, 4)), 8)


def test_png_round_trip_is_lossless():
    rng = np.random.default_rng(3)
    m = rng.integers(0, 3, (48, 48)).astype(np.uint8)
    np.testing.assert_array_equal(decode_png(to_png(m)), m)


def test_prepare_applies_the_contract(bundle_dir, maps):
    b = load_bundle(bundle_dir, backend="onnx")
    x = prepare(list(maps), b)
    assert x.shape == (len(maps), 3, SIZE, SIZE)
    assert x.dtype == np.float32
    assert set(np.unique(x)) <= {0.0, 0.5, 1.0}
    # the three channels are copies of one another
    np.testing.assert_array_equal(x[:, 0], x[:, 1])
    np.testing.assert_array_equal(x[:, 0], x[:, 2])


def test_prepare_rejects_out_of_domain_values(bundle_dir):
    b = load_bundle(bundle_dir, backend="onnx")
    with pytest.raises(ValueError, match="only 0/1/2"):
        prepare([np.full((SIZE, SIZE), 7, dtype=np.uint8)], b)


def test_softmax_is_a_distribution():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1e6, 0.0, -1e6]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


# ---------------------------------------------------------------- input readers
def test_reads_a_single_npy(tmp_path, maps):
    p = tmp_path / "one.npy"
    np.save(p, maps[0])
    m, ids = read_input(p)
    assert len(m) == 1 and ids == ["one.npy"]


def test_reads_a_stacked_npy(tmp_path, maps):
    p = tmp_path / "many.npy"
    np.save(p, maps)
    m, ids = read_input(p)
    assert len(m) == len(maps) and ids[0].endswith("[0]")


def test_reads_an_npz_and_filters_by_split(tmp_path, maps):
    p = tmp_path / "proc.npz"
    split = np.array(["train"] * 4 + ["test"] * 2)
    np.savez_compressed(p, maps=maps, y=np.zeros(6, int), split=split)
    assert len(read_input(p)[0]) == 6
    assert len(read_input(p, split="test")[0]) == 2
    with pytest.raises(ValueError, match="no rows with split"):
        read_input(p, split="val")


def test_npz_without_maps_is_rejected(tmp_path):
    p = tmp_path / "wrong.npz"
    np.savez(p, something=np.zeros(3))
    with pytest.raises(ValueError, match="no 'maps' array"):
        read_input(p)


def test_reads_a_directory_of_pngs(tmp_path, maps):
    import cv2

    d = tmp_path / "pngs"
    d.mkdir()
    for i, m in enumerate(maps):
        cv2.imwrite(str(d / f"w{i:02d}.png"), to_png(m))
    got, ids = read_input(d)
    assert len(got) == len(maps)
    np.testing.assert_array_equal(got[0], maps[0])


def test_empty_directory_is_reported(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(ValueError, match="no .npy"):
        read_input(d)


def test_missing_input_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_input(tmp_path / "nope.npy")


# ---------------------------------------------------------------- scoring
def test_score_frame_shape_and_probabilities(bundle_dir, tmp_path, maps):
    p = tmp_path / "batch.npy"
    np.save(p, maps)
    b = load_bundle(bundle_dir, backend="onnx")
    df = score(b, p)
    assert len(df) == len(maps)
    assert df["predicted"].isin(CLASSES).all()
    prob_cols = [f"p_{c}" for c in CLASSES]
    np.testing.assert_allclose(df[prob_cols].to_numpy().sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(
        df["defect_probability"].to_numpy(), 1.0 - df["p_none"].to_numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        df["confidence"].to_numpy(), df[prob_cols].to_numpy().max(axis=1), atol=1e-6
    )


def test_scoring_is_invariant_to_batch_size(bundle_dir, tmp_path, maps):
    p = tmp_path / "batch.npy"
    np.save(p, maps)
    b = load_bundle(bundle_dir, backend="onnx")
    a = score(b, p, batch_size=1)["p_none"].to_numpy()
    c = score(b, p, batch_size=256)["p_none"].to_numpy()
    np.testing.assert_allclose(a, c, atol=1e-6)


def test_png_and_npy_inputs_score_identically(bundle_dir, tmp_path, maps):
    import cv2

    npy = tmp_path / "x.npy"
    np.save(npy, maps[0])
    png_dir = tmp_path / "p"
    png_dir.mkdir()
    cv2.imwrite(str(png_dir / "x.png"), to_png(maps[0]))
    b = load_bundle(bundle_dir, backend="onnx")
    np.testing.assert_allclose(
        score(b, npy)["p_none"].to_numpy(), score(b, png_dir)["p_none"].to_numpy(), atol=1e-6
    )


def test_oversized_maps_are_resized_and_scored(bundle_dir, tmp_path):
    rng = np.random.default_rng(5)
    big = rng.integers(0, 3, (113, 97)).astype(np.uint8)
    p = tmp_path / "big.npy"
    np.save(p, big)
    b = load_bundle(bundle_dir, backend="onnx")
    assert len(score(b, p)) == 1


def test_cli_writes_a_csv(bundle_dir, tmp_path, maps):
    import pandas as pd

    src = tmp_path / "in.npy"
    np.save(src, maps)
    out = tmp_path / "scored.csv"
    r = subprocess.run(
        [sys.executable, "-m", "sicdefect.infer", "--bundle", str(bundle_dir),
         "--input", str(src), "--out", str(out), "--backend", "onnx", "--root-causes"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert len(df) == len(maps)
    assert {"id", "predicted", "confidence"} <= set(df.columns)
    assert "predicted pattern counts" in r.stderr
