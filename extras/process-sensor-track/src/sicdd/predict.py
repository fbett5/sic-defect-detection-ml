"""Apply a saved detector to new process data.

The saved artifact is self-contained: it carries its own preprocessing
(missing-rate filter, median imputation statistics, constant-column mask and
scaling), its decision threshold, the exact feature schema it expects, and the
provenance of the runs it was trained on.  Scoring new data therefore needs
nothing from this repository's training code beyond the class definitions.

Accepted inputs
---------------
* the original whitespace-separated SECOM layout (590 numeric columns, no header);
* a CSV with one column per sensor, named either ``sensor_000``-style or
  ``0``-style; a leading timestamp column and a trailing label column are
  dropped automatically, so the Kaggle-style CSV works unchanged.

Example
-------
::

    sicdd-predict --model models/sicdd_defect_detector.joblib \\
                  --input new_runs.csv --output scored.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LABEL_COLUMN_HINTS = {"pass/fail", "pass_fail", "label", "y", "target", "fail"}
TIME_COLUMN_HINTS = {"time", "timestamp", "date", "datetime"}


def load_artifact(path: str | Path) -> dict:
    """Load a saved bundle and check it is one of ours."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            f"{path} does not look like a sicdd artifact "
            "(expected a dict with a 'pipeline' key)"
        )
    if bundle.get("format_version") != 1:
        raise ValueError(
            f"unsupported artifact format_version {bundle.get('format_version')!r}; "
            "this build reads version 1"
        )
    return bundle


def read_features(path: str | Path, expected: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    """Read new runs and align them to the schema the model was trained on.

    Returns the feature frame plus any timestamp column found, which is carried
    through to the output for traceability.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        # Original SECOM layout: whitespace separated, no header, "NaN" literals.
        df = pd.read_csv(path, sep=r"\s+", header=None, na_values=["NaN"])
        df.columns = [f"sensor_{i:03d}" for i in range(df.shape[1])]

    timestamps = None
    for col in list(df.columns):
        if str(col).strip().lower() in TIME_COLUMN_HINTS:
            timestamps = pd.to_datetime(df[col], errors="coerce").rename("timestamp")
            df = df.drop(columns=[col])
            break
    for col in list(df.columns):
        if str(col).strip().lower() in LABEL_COLUMN_HINTS:
            df = df.drop(columns=[col])
            break

    # Either the names already match, or the column count does and we map by
    # position -- anything else is a schema error the caller must resolve.
    if list(df.columns) == list(expected):
        pass
    elif set(expected).issubset(df.columns):
        df = df.loc[:, expected]
    elif df.shape[1] == len(expected):
        df.columns = expected
    else:
        raise ValueError(
            f"input has {df.shape[1]} feature columns but the model expects "
            f"{len(expected)}. Supply the {len(expected)} sensor columns, named "
            f"{expected[0]}..{expected[-1]} or in the original column order."
        )

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        df[non_numeric] = df[non_numeric].apply(pd.to_numeric, errors="coerce")
    return df.astype("float64"), timestamps


def predict(
    artifact: str | Path | dict,
    input_path: str | Path,
    threshold: float | None = None,
    policy: str | None = None,
) -> pd.DataFrame:
    """Score new runs, returning one row per run.

    ``threshold`` overrides the artifact's default; ``policy`` selects one of
    the alternative operating points stored in the artifact (for example
    ``budget_5pct`` or ``min_expected_cost``).
    """
    bundle = artifact if isinstance(artifact, dict) else load_artifact(artifact)
    X, timestamps = read_features(input_path, bundle["expected_features"])

    thr = bundle["threshold"]
    thr_source = f"artifact default ({bundle.get('policy', 'unspecified policy')})"
    if policy is not None:
        available = bundle.get("thresholds", {})
        if policy not in available:
            raise ValueError(
                f"unknown policy {policy!r}; available: {sorted(available)}"
            )
        thr, thr_source = available[policy], f"policy {policy}"
    if threshold is not None:
        thr, thr_source = threshold, "caller-supplied --threshold"

    score = bundle["pipeline"].predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            "run_index": np.arange(len(X)),
            "defect_score": score,
            "predicted_defect": (score >= thr).astype(int),
        }
    )
    if timestamps is not None:
        out.insert(1, "timestamp", timestamps.to_numpy())
    out.attrs["threshold"] = float(thr)
    out.attrs["threshold_source"] = thr_source
    out.attrs["model_name"] = bundle.get("model_name", "unknown")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--model", default="models/sicdd_defect_detector.joblib")
    p.add_argument("--input", required=True, help="CSV or original-layout data file")
    p.add_argument("--output", default=None, help="write scored runs here (default: stdout)")
    p.add_argument("--threshold", type=float, default=None, help="override the threshold")
    p.add_argument(
        "--policy",
        default=None,
        help="use a stored operating point instead of the default "
        "(e.g. budget_5pct, budget_20pct, min_expected_cost)",
    )
    p.add_argument("--describe", action="store_true", help="print artifact metadata and exit")
    a = p.parse_args(argv)

    bundle = load_artifact(a.model)
    if a.describe:
        print(
            json.dumps(
                {
                    "model_name": bundle.get("model_name"),
                    "policy": bundle.get("policy"),
                    "threshold": bundle.get("threshold"),
                    "thresholds": bundle.get("thresholds"),
                    "n_expected_features": len(bundle["expected_features"]),
                    "n_sensors_used": len(bundle.get("sensors_used", [])),
                    "metadata": bundle.get("metadata"),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    out = predict(bundle, a.input, threshold=a.threshold, policy=a.policy)
    flagged = int(out["predicted_defect"].sum())
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.output, index=False)
        print(
            f"scored {len(out)} runs with {out.attrs['model_name']} "
            f"at threshold {out.attrs['threshold']:.6g} ({out.attrs['threshold_source']}); "
            f"flagged {flagged} ({flagged / len(out):.1%}) -> {a.output}",
            file=sys.stderr,
        )
    else:
        out.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
