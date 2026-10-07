"""Loading and provenance verification for the SECOM process dataset.

The dataset records 590 in-line sensor measurements for 1,567 production runs
of a semiconductor fabrication process, each labelled pass (-1) or fail (+1)
by end-of-line testing, with a timestamp.

Provenance note
---------------
We read the *original* UCI distribution files (``secom.data`` /
``secom_labels.data``), not the widely circulated Kaggle CSV re-export.  The
CSV re-export parses the DD/MM/YYYY timestamps ambiguously and corrupts 582 of
them (e.g. ``01/08/2008`` becomes 8 January instead of 1 August), which
destroys the run ordering that our temporal evaluation protocol depends on.
The feature matrices of the two copies are otherwise bit-identical.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: SHA-256 of the authoritative UCI files, verified at load time.
CHECKSUMS = {
    "secom.data": "4503bdb42e63687d1c7586bdc1796df9a7cb5ae9cc9c0443d84957a446de3aaf",
    "secom_labels.data": "2fb2e8b5dfe77ba96ecf0960ff00f2ba9311795383a8342d4588a7cb37c816ef",
}

N_RUNS = 1567
N_SENSORS = 590
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


def feature_names(n: int = N_SENSORS) -> list[str]:
    """Stable, explicit sensor column names (``sensor_000`` ... ``sensor_589``)."""
    return [f"sensor_{i:03d}" for i in range(n)]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Dataset:
    """A loaded, verified copy of the process dataset."""

    X: pd.DataFrame
    y: np.ndarray  # int8, 1 = fail (defect), 0 = pass
    timestamps: pd.Series
    source_digests: dict[str, str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def fail_rate(self) -> float:
        return float(self.y.mean())


def load_secom(data_dir: str | Path = "data/raw", verify: bool = True) -> Dataset:
    """Load the dataset, sorted by run time.

    Parameters
    ----------
    data_dir
        Directory holding ``secom.data`` and ``secom_labels.data``.
    verify
        If true, fail loudly when a file's SHA-256 does not match
        :data:`CHECKSUMS`.  Turn this off only when deliberately substituting
        your own data in the same format.
    """
    data_dir = Path(data_dir)
    x_path, y_path = data_dir / "secom.data", data_dir / "secom_labels.data"
    for p in (x_path, y_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. See data/README.md for how to obtain the dataset."
            )

    digests = {p.name: sha256(p) for p in (x_path, y_path)}
    if verify:
        for name, digest in digests.items():
            expected = CHECKSUMS[name]
            if digest != expected:
                raise ValueError(
                    f"checksum mismatch for {name}: expected {expected}, got {digest}. "
                    "Pass verify=False to load a modified copy deliberately."
                )

    X = pd.read_csv(x_path, sep=r"\s+", header=None, na_values=["NaN"], dtype="float64")
    if X.shape != (N_RUNS, N_SENSORS):
        raise ValueError(f"expected feature matrix {(N_RUNS, N_SENSORS)}, got {X.shape}")
    X.columns = feature_names(X.shape[1])

    # The label file is: <label> "<DD/MM/YYYY HH:MM:SS>"; quoting keeps the
    # timestamp in a single whitespace-separated field.
    labels = pd.read_csv(
        y_path, sep=r"\s+", header=None, names=["label", "stamp"], quotechar='"'
    )
    if len(labels) != len(X):
        raise ValueError(f"label/feature length mismatch: {len(labels)} vs {len(X)}")

    y = (labels["label"].to_numpy() == 1).astype(np.int8)  # 1 = fail
    ts = pd.to_datetime(labels["stamp"], format=TIMESTAMP_FORMAT)
    if ts.isna().any():
        raise ValueError("unparseable timestamps in label file")

    order = np.argsort(ts.to_numpy(), kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = y[order]
    ts = ts.iloc[order].reset_index(drop=True).rename("timestamp")
    return Dataset(X=X, y=y, timestamps=ts, source_digests=digests)


def temporal_split(n: int, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Split run indices chronologically: earliest ``1 - test_frac`` for training.

    Assumes the rows are already time-ordered (:func:`load_secom` guarantees it).
    The test set therefore contains only runs that happened *after* every
    training run, which is the deployment situation a fab actually faces.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    cut = int(round(n * (1.0 - test_frac)))
    if cut <= 0 or cut >= n:
        raise ValueError(f"test_frac={test_frac} leaves an empty split for n={n}")
    return np.arange(cut), np.arange(cut, n)
