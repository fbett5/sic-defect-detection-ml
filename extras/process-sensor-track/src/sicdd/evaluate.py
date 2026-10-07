"""Metrics for a rare-defect detector, and how to pick its decision threshold.

Two choices drive everything here.

*Average precision, not accuracy.*  With a 6.6% failure rate, a model that
calls every run good scores 93.4% accuracy and is worthless.  Average precision
(area under the precision-recall curve) has the prevalence as its no-skill
floor, so it states how much better than guessing a model actually is.

*A threshold chosen by cost, not 0.5.*  Shipping a defective lot costs far more
than inspecting a good one.  We therefore pick the threshold that minimises
expected cost under a stated cost ratio, on held-out predictions, never on the
test set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Cost of missing a defective run relative to a false alarm.  A placeholder
#: for a real fab's scrap-versus-inspection economics; every cost-sensitive
#: number in the paper is reported against this ratio and recomputed easily.
COST_FALSE_NEGATIVE = 10.0
COST_FALSE_POSITIVE = 1.0


@dataclass
class ThresholdFreeMetrics:
    """Metrics that rank runs without committing to a cut-off."""

    average_precision: float
    roc_auc: float
    brier: float
    prevalence: float
    ap_lift_over_prevalence: float
    n: int
    n_positive: int

    def as_dict(self) -> dict:
        return asdict(self)


def threshold_free_metrics(y_true, y_score) -> ThresholdFreeMetrics:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    prevalence = float(y_true.mean())
    both_classes = 0 < y_true.sum() < len(y_true)
    ap = float(average_precision_score(y_true, y_score)) if both_classes else float("nan")
    return ThresholdFreeMetrics(
        average_precision=ap,
        roc_auc=float(roc_auc_score(y_true, y_score)) if both_classes else float("nan"),
        brier=float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
        prevalence=prevalence,
        ap_lift_over_prevalence=float(ap / prevalence) if prevalence > 0 else float("nan"),
        n=int(len(y_true)),
        n_positive=int(y_true.sum()),
    )


def expected_cost(
    y_true,
    y_score,
    threshold: float,
    c_fn: float = COST_FALSE_NEGATIVE,
    c_fp: float = COST_FALSE_POSITIVE,
) -> float:
    """Mean cost per run of acting on ``y_score >= threshold``."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_score, dtype=float).ravel() >= threshold).astype(int)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float((c_fn * fn + c_fp * fp) / len(y_true))


def pick_threshold_min_cost(
    y_true,
    y_score,
    c_fn: float = COST_FALSE_NEGATIVE,
    c_fp: float = COST_FALSE_POSITIVE,
) -> tuple[float, float]:
    """Threshold minimising expected cost on *these* predictions.

    Candidates are the midpoints between consecutive distinct scores, plus the
    two degenerate extremes (flag nothing / flag everything), so the search
    covers every achievable confusion matrix.  Returns ``(threshold, cost)``;
    ties go to the higher threshold, i.e. the fewer false alarms.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    uniq = np.unique(y_score)
    if len(uniq) == 1:
        cands = np.array([uniq[0], np.nextafter(uniq[0], np.inf)])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        cands = np.concatenate(
            [[np.nextafter(uniq[0], -np.inf)], mids, [np.nextafter(uniq[-1], np.inf)]]
        )
    costs = np.array([expected_cost(y_true, y_score, t, c_fn, c_fp) for t in cands])
    best = np.flatnonzero(costs == costs.min())[-1]
    return float(cands[best]), float(costs[best])


def operating_point_metrics(
    y_true,
    y_score,
    threshold: float,
    c_fn: float = COST_FALSE_NEGATIVE,
    c_fp: float = COST_FALSE_POSITIVE,
) -> dict:
    """Confusion-matrix metrics at one threshold, plus its cost and workload."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    flagged = int(tp + fp)
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "flagged_rate": float(flagged / len(y_true)),
        "expected_cost": expected_cost(y_true, y_score, threshold, c_fn, c_fp),
        "cost_vs_flag_nothing": float(
            expected_cost(y_true, y_score, threshold, c_fn, c_fp)
            / (c_fn * y_true.mean())
        )
        if y_true.sum()
        else float("nan"),
    }


def recall_at_inspection_budget(y_true, y_score, budget: float = 0.10) -> dict:
    """Recall when only the top ``budget`` fraction of runs can be inspected.

    This is the question an engineer actually asks: if I can re-inspect one run
    in ten, how many of the defective ones do I catch?
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    k = max(1, int(round(len(y_true) * budget)))
    top = np.argsort(-y_score, kind="stable")[:k]
    caught = int(y_true[top].sum())
    total = int(y_true.sum())
    return {
        "budget": float(budget),
        "n_inspected": k,
        "caught": caught,
        "recall": float(caught / total) if total else float("nan"),
        "precision": float(caught / k),
        "lift": float((caught / k) / y_true.mean()) if y_true.mean() > 0 else float("nan"),
    }


def bootstrap_ci(
    y_true,
    y_score,
    metric: str = "average_precision",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20251007,
) -> dict:
    """Percentile bootstrap interval for a ranking metric.

    Resamples runs with replacement, skipping draws that lose a class.  With
    only ~17 defective runs in the test window these intervals are wide, which
    is the honest message rather than a defect of the method.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    fn = {"average_precision": average_precision_score, "roc_auc": roc_auc_score}[metric]
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if 0 < yt.sum() < n:
            vals.append(fn(yt, y_score[idx]))
    if not vals:
        return {"metric": metric, "point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    vals = np.asarray(vals)
    return {
        "metric": metric,
        "point": float(fn(y_true, y_score)),
        "lo": float(np.quantile(vals, alpha / 2)),
        "hi": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot": int(len(vals)),
    }


def threshold_for_flag_rate(reference_scores, flag_rate: float) -> float:
    """Threshold that flags ``flag_rate`` of ``reference_scores``.

    Operating points are transferred between models as *rates*, not as raw
    score cut-offs.  A model fitted on one forward-chaining fold and the same
    model refitted on every training run emit different score scales, so a
    threshold read off the former flags nothing when applied to the latter.
    A rate is invariant to that shift, and a fab's inspection capacity is a
    rate in the first place.

    ``flag_rate=0`` returns ``+inf`` (flag nothing) and ``flag_rate=1`` returns
    ``-inf`` (flag everything), so both degenerate policies stay expressible.
    """
    if not 0.0 <= flag_rate <= 1.0:
        raise ValueError(f"flag_rate must be in [0, 1], got {flag_rate}")
    if flag_rate == 0.0:
        return float("inf")
    if flag_rate == 1.0:
        return float("-inf")
    scores = np.asarray(reference_scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("reference_scores is empty")
    return float(np.quantile(scores, 1.0 - flag_rate))


def cost_sensitivity(
    y_true,
    y_score,
    threshold: float,
    ratios: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0),
    c_fp: float = COST_FALSE_POSITIVE,
) -> list[dict]:
    """Cost of acting at ``threshold`` versus flagging nothing, across cost ratios.

    The 10:1 ratio used throughout is a stand-in for a real fab's economics, so
    every cost claim is reported against a sweep.  ``break_even`` marks where
    acting stops being cheaper than ignoring the model.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    out = []
    for ratio in ratios:
        c_fn = ratio * c_fp
        acted = expected_cost(y_true, y_score, threshold, c_fn=c_fn, c_fp=c_fp)
        nothing = float(c_fn * y_true.mean())
        out.append(
            {
                "cost_ratio_fn_to_fp": float(ratio),
                "cost_acting": acted,
                "cost_flag_nothing": nothing,
                "savings_fraction": float((nothing - acted) / nothing) if nothing else float("nan"),
                "worthwhile": bool(acted < nothing),
            }
        )
    return out
