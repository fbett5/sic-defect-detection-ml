"""Preprocessing must learn only from the rows it is fitted on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sicdd.preprocess import MissingRateFilter, build_preprocessor, surviving_sensors


def test_removes_nan_and_constants_and_standardises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=200) * 50,
            "b": np.full(200, 7.0),            # constant
            "c": rng.normal(size=200),
            "d": np.where(np.arange(200) < 150, np.nan, 1.0),  # 75% missing
        }
    )
    Z = build_preprocessor(max_missing_rate=0.5).fit_transform(X)
    assert not np.isnan(Z).any()
    assert Z.shape[1] == 2  # 'b' constant, 'd' too sparse
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-10)


def test_statistics_come_only_from_fit_rows():
    """Transforming later rows must not re-centre on their own statistics."""
    fit = pd.DataFrame({"a": [1.0, 1.0, 1.0, 3.0], "b": [0.0, 1.0, 0.0, 1.0]})
    later = pd.DataFrame({"a": [100.0, 200.0], "b": [0.5, 0.5]})
    prep = build_preprocessor().fit(fit)
    Z = prep.transform(later)
    assert Z[:, 0].min() > 10  # scaled by the fit rows' spread, so far off-centre
    assert not np.allclose(Z.mean(axis=0), 0, atol=1e-6)


def test_imputation_uses_fit_median_not_transform_median():
    fit = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    prep = build_preprocessor().fit(fit)
    median_a = 2.0
    out = prep.transform(pd.DataFrame({"a": [np.nan], "b": [2.5]}))
    scaler = prep.named_steps["scale"]
    recovered = out[0, 0] * np.sqrt(scaler.var_[0]) + scaler.mean_[0]
    assert recovered == pytest.approx(median_a)


def test_missing_rate_filter_reports_and_validates():
    X = pd.DataFrame({"keep": [1.0, 2.0, 3.0], "drop": [np.nan, np.nan, 1.0]})
    f = MissingRateFilter(max_rate=0.5).fit(X)
    assert list(f.get_feature_names_out(["keep", "drop"])) == ["keep"]
    with pytest.raises(ValueError, match="expected 2 columns"):
        f.transform(pd.DataFrame({"only": [1.0]}))


def test_missing_rate_filter_rejects_impossible_rate():
    with pytest.raises(ValueError):
        MissingRateFilter(max_rate=1.5).fit(pd.DataFrame({"a": [1.0]}))


def test_filter_raises_when_everything_would_be_dropped():
    X = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="removed every column"):
        MissingRateFilter(max_rate=0.1).fit(X)


def test_surviving_sensors_matches_output_width():
    rng = np.random.default_rng(1)
    names = [f"s{i}" for i in range(6)]
    X = pd.DataFrame(rng.normal(size=(50, 6)), columns=names)
    X["s3"] = 1.0
    X["s5"] = np.nan
    prep = build_preprocessor().fit(X)
    kept = surviving_sensors(prep, names)
    assert len(kept) == prep.transform(X).shape[1]
    assert "s3" not in kept and "s5" not in kept
