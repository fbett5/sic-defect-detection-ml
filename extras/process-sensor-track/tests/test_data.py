"""The dataset is the one experiment nothing else can compensate for."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sicdd.data import CHECKSUMS, N_RUNS, N_SENSORS, load_secom, temporal_split


@pytest.fixture(scope="module")
def ds():
    return load_secom("data/raw")


def test_shape_and_schema(ds):
    assert ds.X.shape == (N_RUNS, N_SENSORS)
    assert list(ds.X.columns)[:2] == ["sensor_000", "sensor_001"]
    assert list(ds.X.columns)[-1] == f"sensor_{N_SENSORS - 1:03d}"
    assert ds.X.dtypes.eq("float64").all()


def test_labels_are_binary_with_known_prevalence(ds):
    assert set(np.unique(ds.y)) == {0, 1}
    assert int(ds.y.sum()) == 104  # the canonical SECOM failure count
    assert ds.fail_rate == pytest.approx(104 / N_RUNS)


def test_checksums_pin_the_source(ds):
    assert ds.source_digests == CHECKSUMS


def test_tampered_copy_is_rejected(tmp_path):
    for name in ("secom.data", "secom_labels.data"):
        (tmp_path / name).write_bytes(open(f"data/raw/{name}", "rb").read())
    with open(tmp_path / "secom.data", "a") as fh:
        fh.write("0 " * N_SENSORS + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_secom(tmp_path)


def test_missing_files_are_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_secom(tmp_path)


def test_runs_are_time_ordered(ds):
    assert ds.timestamps.is_monotonic_increasing
    assert str(ds.timestamps.iloc[0]) == "2008-07-19 11:55:00"
    assert str(ds.timestamps.iloc[-1]) == "2008-10-17 06:07:00"


def test_timestamps_are_day_first(ds):
    """Guards against the day/month swap in the circulating CSV re-export.

    A month-first reading of the same file yields dates in January, which would
    both break the ordering and silently reshuffle the train/test split.
    """
    assert ds.timestamps.dt.month.min() == 7
    assert ds.timestamps.dt.month.max() == 10


def test_known_degeneracies_are_present(ds):
    """These are the properties preprocessing exists to handle."""
    assert int((ds.X.nunique(dropna=True) <= 1).sum()) == 116
    assert int((ds.X.isna().mean() > 0.5).sum()) == 28
    assert ds.X.isna().to_numpy().mean() == pytest.approx(0.0454, abs=5e-4)


def test_temporal_split_is_disjoint_and_ordered(ds):
    train_idx, test_idx = temporal_split(len(ds), test_frac=0.2)
    assert len(train_idx) + len(test_idx) == len(ds)
    assert not set(train_idx) & set(test_idx)
    assert ds.timestamps.iloc[test_idx].min() >= ds.timestamps.iloc[train_idx].max()


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_temporal_split_rejects_degenerate_fractions(frac):
    with pytest.raises(ValueError):
        temporal_split(100, test_frac=frac)
