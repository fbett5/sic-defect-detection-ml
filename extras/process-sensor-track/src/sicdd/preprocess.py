"""Preprocessing for wide, sparse, partly degenerate process-sensor matrices.

The raw matrix has three properties that have to be handled before any model
sees it, and all of them must be learned from training data only:

1. 4.5% of cells are missing, concentrated in 32 sensors that are missing for
   more than a fifth of runs (28 for more than half).
2. 116 sensors are constant -- a sensor logged at a fixed set point carries no
   information and makes scaling ill-conditioned.
3. Sensor scales differ by several orders of magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MissingRateFilter(BaseEstimator, TransformerMixin):
    """Drop columns whose missing rate on the training split exceeds ``max_rate``.

    Imputing a sensor that is absent for most runs invents the majority of its
    column, so such sensors are removed rather than filled.
    """

    def __init__(self, max_rate: float = 0.5):
        self.max_rate = max_rate

    def fit(self, X, y=None):
        X = self._frame(X)
        if not 0.0 <= self.max_rate <= 1.0:
            raise ValueError(f"max_rate must be in [0, 1], got {self.max_rate}")
        rates = X.isna().mean(axis=0).to_numpy()
        self.support_ = rates <= self.max_rate
        if not self.support_.any():
            raise ValueError("MissingRateFilter removed every column")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._frame(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} columns, got {X.shape[1]}"
            )
        return X.loc[:, self.support_].to_numpy(dtype="float64")

    def get_feature_names_out(self, input_features=None):
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )
        return names[self.support_]

    @staticmethod
    def _frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))


def build_preprocessor(max_missing_rate: float = 0.5) -> Pipeline:
    """Missing-rate filter -> median imputation -> constant-column drop -> scaling.

    Every step is a fitted transformer, so when this pipeline is cross-validated
    or saved as part of a model artifact, the statistics it applies to new data
    are exactly the ones learned on the training runs.
    """
    return Pipeline(
        [
            ("missing_filter", MissingRateFilter(max_rate=max_missing_rate)),
            ("impute", SimpleImputer(strategy="median")),
            ("drop_constant", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
        ]
    )


def surviving_sensors(fitted_preprocessor: Pipeline, names: list[str]) -> list[str]:
    """Names of the sensors a fitted preprocessor keeps, in output order."""
    kept = fitted_preprocessor.named_steps["missing_filter"].get_feature_names_out(names)
    var_support = fitted_preprocessor.named_steps["drop_constant"].get_support()
    return [str(n) for n in np.asarray(kept)[var_support]]
