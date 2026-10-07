"""The saved artifact must score new data on its own, in any accepted layout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sicdd.data import load_secom, temporal_split
from sicdd.predict import load_artifact, predict, read_features

ARTIFACT = Path("models/sicdd_defect_detector.joblib")
pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="run `make train` to build the artifact first"
)


@pytest.fixture(scope="module")
def bundle():
    return load_artifact(ARTIFACT)


@pytest.fixture(scope="module")
def held_out(tmp_path_factory):
    """The last 20 runs of the record, written out in three layouts."""
    ds = load_secom("data/raw")
    _, test_idx = temporal_split(len(ds), test_frac=0.2)
    idx = test_idx[-20:]
    X, y = ds.X.iloc[idx].reset_index(drop=True), ds.y[idx]
    d = tmp_path_factory.mktemp("newdata")

    named = d / "named.csv"
    X.to_csv(named, index=False)

    kaggle = d / "kaggle_style.csv"
    k = X.copy()
    k.columns = [str(i) for i in range(X.shape[1])]
    k.insert(0, "Time", ds.timestamps.iloc[idx].astype(str).to_numpy())
    k["Pass/Fail"] = np.where(y == 1, 1, -1)
    k.to_csv(kaggle, index=False)

    original = d / "runs.data"
    X.to_csv(original, sep=" ", header=False, index=False, na_rep="NaN")
    return {"named": named, "kaggle": kaggle, "original": original, "X": X, "y": y}


def test_artifact_is_self_describing(bundle):
    assert bundle["format_version"] == 1
    assert len(bundle["expected_features"]) == 590
    assert np.isfinite(bundle["threshold"])
    assert bundle["metadata"]["source_digests"]
    assert "scikit-learn" in bundle["metadata"]["versions"]


def test_scores_new_runs(bundle, held_out):
    out = predict(bundle, held_out["named"])
    assert len(out) == 20
    assert out["defect_score"].between(0, 1).all()
    assert set(out["predicted_defect"].unique()) <= {0, 1}


def test_all_three_layouts_give_identical_scores(bundle, held_out):
    a = predict(bundle, held_out["named"])["defect_score"].to_numpy()
    b = predict(bundle, held_out["kaggle"])["defect_score"].to_numpy()
    c = predict(bundle, held_out["original"])["defect_score"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12)
    np.testing.assert_allclose(a, c, rtol=1e-12)


def test_timestamps_are_carried_through(bundle, held_out):
    out = predict(bundle, held_out["kaggle"])
    assert "timestamp" in out.columns
    assert out["timestamp"].notna().all()


def test_scoring_is_deterministic(bundle, held_out):
    a = predict(bundle, held_out["named"])["defect_score"].to_numpy()
    b = predict(bundle, held_out["named"])["defect_score"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_reloading_from_disk_reproduces_scores(held_out):
    a = predict(load_artifact(ARTIFACT), held_out["named"])["defect_score"].to_numpy()
    b = predict(load_artifact(ARTIFACT), held_out["named"])["defect_score"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_matches_the_recorded_test_predictions(bundle):
    """Re-scoring the test window must reproduce the published predictions."""
    recorded = pd.read_csv("results/predictions_test.csv")
    ds = load_secom("data/raw")
    _, test_idx = temporal_split(len(ds), test_frac=0.2)
    fresh = bundle["pipeline"].predict_proba(ds.X.iloc[test_idx])[:, 1]
    np.testing.assert_allclose(recorded["defect_score"].to_numpy(), fresh, rtol=1e-10)


def test_threshold_override_changes_only_the_decision(bundle, held_out):
    base = predict(bundle, held_out["named"])
    forced = predict(bundle, held_out["named"], threshold=-np.inf)
    np.testing.assert_allclose(
        base["defect_score"].to_numpy(), forced["defect_score"].to_numpy()
    )
    assert forced["predicted_defect"].all()


def test_stored_policies_are_selectable(bundle, held_out):
    out = predict(bundle, held_out["named"], policy="budget_20pct")
    assert out.attrs["threshold"] == bundle["thresholds"]["budget_20pct"]
    with pytest.raises(ValueError, match="unknown policy"):
        predict(bundle, held_out["named"], policy="nope")


def test_a_looser_budget_never_flags_fewer_runs(bundle, held_out):
    counts = {
        b: int(predict(bundle, held_out["named"], policy=f"budget_{b}pct")["predicted_defect"].sum())
        for b in (5, 10, 20)
    }
    assert counts[5] <= counts[10] <= counts[20]


def test_wrong_column_count_is_a_clear_error(bundle, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame(np.zeros((3, 7))).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="expects"):
        predict(bundle, bad)


def test_missing_values_in_new_data_are_handled(bundle, held_out):
    X = held_out["X"].copy()
    X.iloc[:, :50] = np.nan  # a sensor bank goes offline
    p = held_out["named"].parent / "gappy.csv"
    X.to_csv(p, index=False)
    out = predict(bundle, p)
    assert out["defect_score"].notna().all()


def test_non_numeric_cells_are_coerced(bundle, held_out):
    X = held_out["X"].astype(object).copy()
    X.iloc[0, 0] = "n/a"
    p = held_out["named"].parent / "dirty.csv"
    X.to_csv(p, index=False)
    assert predict(bundle, p)["defect_score"].notna().all()


def test_rejects_a_foreign_pickle(tmp_path):
    p = tmp_path / "not_ours.joblib"
    joblib.dump({"hello": "world"}, p)
    with pytest.raises(ValueError, match="does not look like a sicdd artifact"):
        load_artifact(p)


def test_rejects_an_unsupported_format_version(tmp_path, bundle):
    p = tmp_path / "future.joblib"
    joblib.dump({**bundle, "format_version": 99}, p)
    with pytest.raises(ValueError, match="format_version"):
        load_artifact(p)


def test_every_model_bundle_is_usable(held_out):
    """All five models ship as portable bundles, not just the deployed one."""
    bundles = sorted(Path("models/bundles").glob("*.joblib"))
    assert len(bundles) == 5
    for b in bundles:
        out = predict(load_artifact(b), held_out["named"])
        assert len(out) == 20 and out["defect_score"].notna().all()


def test_cli_round_trip(tmp_path, held_out):
    out = tmp_path / "scored.csv"
    r = subprocess.run(
        [sys.executable, "-m", "sicdd.predict", "--model", str(ARTIFACT),
         "--input", str(held_out["named"]), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert len(df) == 20
    assert {"run_index", "defect_score", "predicted_defect"} <= set(df.columns)
    assert "flagged" in r.stderr


def test_cli_describe():
    r = subprocess.run(
        [sys.executable, "-m", "sicdd.predict", "--model", str(ARTIFACT), "--describe",
         "--input", "unused"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "model_name" in r.stdout and "thresholds" in r.stdout
