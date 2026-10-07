"""Experiment runner: model selection, honest evaluation, and a saved artifact.

Protocol
--------
1. Runs are ordered by time and the final ``--test-frac`` of them is locked
   away.  Nothing -- no imputation statistic, no hyperparameter, no threshold --
   is fitted or chosen using those runs.
2. Hyperparameters are selected on the training runs by forward-chaining
   cross-validation (:class:`~sklearn.model_selection.TimeSeriesSplit`), which
   only ever validates on runs later than the ones it trained on.
3. The same selected configuration is *also* scored under random stratified
   k-fold cross-validation.  The gap between the two quantifies how much a
   random split flatters a model on a process that drifts.
4. Operating points are fixed without touching the test runs, under two
   policies (see :mod:`sicdd.evaluate`): minimum expected cost, and a fixed
   inspection budget.  Thresholds are transferred as *flag rates* mapped onto
   the refitted model's own score distribution, because models trained on
   partial folds emit systematically different score scales.
5. The winning pipeline is refitted on all training runs and written out as a
   single artifact carrying its own preprocessing, thresholds and provenance.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from . import __version__
from .data import load_secom, temporal_split
from .evaluate import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    bootstrap_ci,
    cost_sensitivity,
    operating_point_metrics,
    pick_threshold_min_cost,
    recall_at_inspection_budget,
    threshold_free_metrics,
    threshold_for_flag_rate,
)
from .models import DEFAULT_SEED, model_zoo
from .preprocess import build_preprocessor, surviving_sensors

ARTIFACT_NAME = "sicdd_defect_detector.joblib"

#: Inspection capacities to report.  These are *a priori* operating choices --
#: a fab can re-inspect a fixed fraction of runs -- not quantities tuned on the
#: test window.
INSPECTION_BUDGETS = (0.05, 0.10, 0.20)

#: The budget the saved artifact uses by default.
DEFAULT_BUDGET = 0.10


def _scores(estimator, X) -> np.ndarray:
    """Positive-class scores, whatever the estimator exposes."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def _forward_chaining_oof(estimator, X, y, cv) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold scores from a forward-chaining splitter.

    ``cross_val_predict`` cannot be used: ``TimeSeriesSplit`` is not a
    partition -- the earliest block is never validated -- so the folds are
    walked explicitly and a mask records which runs received a prediction.
    """
    oof = np.full(len(y), np.nan)
    for fit_idx, val_idx in cv.split(X):
        fold = clone(estimator).fit(X.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = _scores(fold, X.iloc[val_idx])
    return oof, ~np.isnan(oof)



def _make_bundle(
    name: str,
    pipeline,
    threshold: float,
    thresholds: dict,
    deploy_budget: float,
    ds,
    train_idx,
    test_idx,
    y_train,
    y_test,
    split_boundary,
    model_metrics: dict,
    seed: int,
    max_missing_rate: float,
    c_fn: float,
    c_fp: float,
) -> dict:
    """Package a fitted pipeline into a self-contained, portable artifact.

    The bundle carries everything needed to score new runs without this
    repository's training code: the fitted preprocessing, the threshold, the
    feature schema, and the provenance of the runs it learned from.
    """
    # Pin inference to a single worker.  A forest predicting with n_jobs=-1
    # reduces per-tree votes in whatever order threads finish, so two loads of
    # the same artifact can disagree in the last bit.  Scoring is cheap; exact
    # reproducibility of a shipped model is worth more than the parallelism.
    clf = pipeline.named_steps.get("clf")
    if clf is not None and "n_jobs" in clf.get_params():
        clf.set_params(n_jobs=1)

    return {
        "format_version": 1,
        "pipeline": pipeline,
        "threshold": float(threshold),
        "policy": f"inspection budget: flag the highest-scoring {deploy_budget:.0%} of runs",
        "thresholds": thresholds,
        "model_name": name,
        "expected_features": list(ds.X.columns),
        "sensors_used": surviving_sensors(pipeline.named_steps["prep"], list(ds.X.columns)),
        "metadata": {
            "sicdd_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": "UCI SECOM (in-line semiconductor process measurements)",
            "source_digests": ds.source_digests,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_window_end": str(split_boundary),
            "train_fail_rate": float(y_train.mean()),
            "test_fail_rate": float(y_test.mean()),
            "deploy_budget": deploy_budget,
            "cost_false_negative": c_fn,
            "cost_false_positive": c_fp,
            "seed": seed,
            "max_missing_rate": max_missing_rate,
            "selection_metric": "average_precision (TimeSeriesSplit CV on training runs)",
            "best_params": model_metrics["best_params"],
            "test_metrics": model_metrics["test_threshold_free"],
            "test_at_deployed_threshold": model_metrics["test_at_budget_thresholds"].get(
                f"budget_{int(deploy_budget * 100)}pct"
            ),
            "caveat": "Held-out performance on this dataset is weak and model "
            "ranking is not statistically identifiable; see results/metrics.json "
            "and the paper before relying on any single model.",
            "versions": _library_versions(),
        },
    }


def _library_versions() -> dict:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass
    return versions


def run(
    data_dir: str = "data/raw",
    out_dir: str = "results",
    model_dir: str = "models",
    test_frac: float = 0.2,
    cv_folds: int = 5,
    seed: int = DEFAULT_SEED,
    max_missing_rate: float = 0.5,
    c_fn: float = COST_FALSE_NEGATIVE,
    c_fp: float = COST_FALSE_POSITIVE,
    n_boot: int = 2000,
    deploy_budget: float = DEFAULT_BUDGET,
    deploy_model: str | None = None,
    only: list[str] | None = None,
) -> dict:
    out_dir, model_dir = Path(out_dir), Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ds = load_secom(data_dir)
    train_idx, test_idx = temporal_split(len(ds), test_frac=test_frac)
    X_train, y_train = ds.X.iloc[train_idx], ds.y[train_idx]
    X_test, y_test = ds.X.iloc[test_idx], ds.y[test_idx]

    split_boundary = ds.timestamps.iloc[train_idx[-1]]
    if ds.timestamps.iloc[test_idx[0]] < split_boundary:
        raise AssertionError("temporal split is not chronologically ordered")

    print(
        f"runs: {len(ds)} (train {len(train_idx)}, test {len(test_idx)}) | "
        f"fail rate train {y_train.mean():.3%} test {y_test.mean():.3%}\n"
        f"train window ends {split_boundary} | costs c_fn={c_fn} c_fp={c_fp}\n"
    )

    temporal_cv = TimeSeriesSplit(n_splits=cv_folds)
    random_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    zoo = model_zoo(seed=seed)
    if only:
        missing = sorted(set(only) - set(zoo))
        if missing:
            raise SystemExit(f"unknown model(s): {missing}. Available: {sorted(zoo)}")
        zoo = {k: v for k, v in zoo.items() if k in only}

    rows, per_model, cv_frames = [], {}, []
    for name, (estimator, grid) in zoo.items():
        print(f"--- {name}")
        pipe = Pipeline(
            [
                ("prep", build_preprocessor(max_missing_rate=max_missing_rate)),
                ("clf", estimator),
            ]
        )
        search = GridSearchCV(
            pipe,
            grid,
            scoring="average_precision",
            cv=temporal_cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        cv_frames.append(pd.DataFrame(search.cv_results_).assign(model=name))

        # The same configuration, scored under a random split, to measure optimism.
        random_cv_ap = cross_val_score(
            clone(best), X_train, y_train, scoring="average_precision", cv=random_cv, n_jobs=1
        )

        # Held-out training-run predictions, used only to fix operating points.
        oof, oof_mask = _forward_chaining_oof(best, X_train, y_train, temporal_cv)
        train_scores = _scores(best, X_train)  # refitted model's own score scale
        test_score = _scores(best, X_test)

        # --- Policy A: minimum expected cost, chosen on held-out runs. ---
        if y_train[oof_mask].sum() > 0:
            thr_oof, cost_oof = pick_threshold_min_cost(
                y_train[oof_mask], oof[oof_mask], c_fn=c_fn, c_fp=c_fp
            )
            flag_rate_oof = float((oof[oof_mask] >= thr_oof).mean())
        else:
            thr_oof, cost_oof, flag_rate_oof = float("inf"), float("nan"), 0.0
        # Transfer the *rate*, not the raw threshold: partial-fold models emit a
        # different score scale from the refitted model.
        thr_cost = threshold_for_flag_rate(train_scores, flag_rate_oof)
        cost_policy_abstains = flag_rate_oof == 0.0

        # --- Policy B: fixed inspection budget (stated a priori). ---
        thr_budget = {b: threshold_for_flag_rate(train_scores, b) for b in INSPECTION_BUDGETS}

        tf = threshold_free_metrics(y_test, test_score)
        op_cost = operating_point_metrics(y_test, test_score, thr_cost, c_fn=c_fn, c_fp=c_fp)
        op_budget = {
            f"budget_{int(b*100)}pct": operating_point_metrics(
                y_test, test_score, thr_budget[b], c_fn=c_fn, c_fp=c_fp
            )
            for b in INSPECTION_BUDGETS
        }
        op_half = operating_point_metrics(y_test, test_score, 0.5, c_fn=c_fn, c_fp=c_fp)
        budgets = {
            f"recall_at_{int(b*100)}pct_budget": recall_at_inspection_budget(y_test, test_score, b)
            for b in INSPECTION_BUDGETS
        }
        ci_ap = bootstrap_ci(y_test, test_score, "average_precision", n_boot=n_boot, seed=seed)
        ci_auc = bootstrap_ci(y_test, test_score, "roc_auc", n_boot=n_boot, seed=seed)
        deployed_thr = thr_budget[deploy_budget] if deploy_budget in thr_budget else (
            threshold_for_flag_rate(train_scores, deploy_budget)
        )

        per_model[name] = {
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "cv_temporal_ap_mean": float(search.best_score_),
            "cv_temporal_ap_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "cv_random_ap_mean": float(random_cv_ap.mean()),
            "cv_random_ap_std": float(random_cv_ap.std()),
            "cv_optimism_gap": float(random_cv_ap.mean() - search.best_score_),
            "n_oof_scored": int(oof_mask.sum()),
            "cost_policy": {
                "threshold_on_oof": float(thr_oof),
                "oof_expected_cost": float(cost_oof),
                "oof_flag_rate": flag_rate_oof,
                "abstains": bool(cost_policy_abstains),
                "deployed_threshold": float(thr_cost),
            },
            "budget_policy_thresholds": {
                f"budget_{int(b*100)}pct": float(thr_budget[b]) for b in INSPECTION_BUDGETS
            },
            "deployed_threshold": float(deployed_thr),
            "test_threshold_free": tf.as_dict(),
            "test_at_cost_threshold": op_cost,
            "test_at_budget_thresholds": op_budget,
            "test_at_threshold_0.5": op_half,
            "test_inspection_budgets": budgets,
            "test_ap_ci95": ci_ap,
            "test_roc_auc_ci95": ci_auc,
            "test_cost_sensitivity": cost_sensitivity(
                y_test, test_score, deployed_thr, c_fp=c_fp
            ),
        }
        key = f"budget_{int(deploy_budget*100)}pct"
        dep = op_budget.get(
            key, operating_point_metrics(y_test, test_score, deployed_thr, c_fn=c_fn, c_fp=c_fp)
        )
        # Budget metrics for the leaderboard come from the rank-based helper, not
        # from the threshold.  A constant-score model (the prior baseline) has
        # every run tied at one value, so any threshold either flags all runs or
        # none; taking the top-k by rank keeps the budget honest under ties.
        rank_key = f"recall_at_{int(deploy_budget*100)}pct_budget"
        ranked_budget = budgets.get(
            rank_key, recall_at_inspection_budget(y_test, test_score, deploy_budget)
        )
        rows.append(
            {
                "model": name,
                "cv_temporal_ap": search.best_score_,
                "cv_random_ap": random_cv_ap.mean(),
                "optimism_gap": random_cv_ap.mean() - search.best_score_,
                "test_ap": tf.average_precision,
                "test_ap_lo": ci_ap["lo"],
                "test_ap_hi": ci_ap["hi"],
                "test_roc_auc": tf.roc_auc,
                "test_ap_lift": tf.ap_lift_over_prevalence,
                "recall_at_budget": ranked_budget["recall"],
                "precision_at_budget": ranked_budget["precision"],
                "lift_at_budget": ranked_budget["lift"],
                "n_inspected_at_budget": ranked_budget["n_inspected"],
                "flagged_rate_at_threshold": dep["flagged_rate"],
                "cost_at_budget": dep["expected_cost"],
                "cost_vs_flag_nothing": dep["cost_vs_flag_nothing"],
                "cost_policy_abstains": cost_policy_abstains,
            }
        )
        joblib.dump(best, model_dir / f"{name}.joblib")
        print(
            f"    temporal CV AP {search.best_score_:.4f} | random CV AP {random_cv_ap.mean():.4f}"
            f" | test AP {tf.average_precision:.4f} | test ROC-AUC {tf.roc_auc:.4f}"
            f" | recall@{int(deploy_budget*100)}% {ranked_budget['recall']:.3f}"
        )

    leaderboard = (
        pd.DataFrame(rows).sort_values("cv_temporal_ap", ascending=False).reset_index(drop=True)
    )

    # Select the deployed model on cross-validated *training* performance, never
    # on the test runs; the prior baseline cannot rank runs and is excluded.
    ranked = [r for r in rows if r["model"] != "prior_baseline"]
    if not ranked:
        raise SystemExit("no ranking model was trained; nothing to deploy")
    protocol_choice = max(ranked, key=lambda r: r["cv_temporal_ap"])["model"]
    best_name = deploy_model or protocol_choice
    if best_name not in per_model:
        raise SystemExit(f"--deploy-model {best_name!r} was not trained; got {sorted(per_model)}")
    if deploy_model and deploy_model != protocol_choice:
        print(
            f"note: deploying {deploy_model} by explicit request; the selection "
            f"protocol chose {protocol_choice}"
        )
    best_pipeline = joblib.load(model_dir / f"{best_name}.joblib")
    kept = surviving_sensors(best_pipeline.named_steps["prep"], list(ds.X.columns))

    test_score = _scores(best_pipeline, X_test)
    deployed_thr = per_model[best_name]["deployed_threshold"]
    pd.DataFrame(
        {
            "timestamp": ds.timestamps.iloc[test_idx].to_numpy(),
            "y_true": y_test,
            "defect_score": test_score,
            "predicted_defect": (test_score >= deployed_thr).astype(int),
        }
    ).to_csv(out_dir / "predictions_test.csv", index=False)

    # Every model is written out as a complete, portable bundle, not just the
    # protocol-selected one.  Held-out performance here is weak and the models
    # are not statistically separable, so pinning the deliverable to a single
    # learner would overstate what this dataset supports.
    bundle_dir = model_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for name in per_model:
        pipe = joblib.load(model_dir / f"{name}.joblib")
        thresholds = {
            **per_model[name]["budget_policy_thresholds"],
            "min_expected_cost": per_model[name]["cost_policy"]["deployed_threshold"],
        }
        bundles[name] = _make_bundle(
            name=name,
            pipeline=pipe,
            threshold=per_model[name]["deployed_threshold"],
            thresholds=thresholds,
            deploy_budget=deploy_budget,
            ds=ds,
            train_idx=train_idx,
            test_idx=test_idx,
            y_train=y_train,
            y_test=y_test,
            split_boundary=split_boundary,
            model_metrics=per_model[name],
            seed=seed,
            max_missing_rate=max_missing_rate,
            c_fn=c_fn,
            c_fp=c_fp,
        )
        joblib.dump(bundles[name], bundle_dir / f"{name}.joblib")

    artifact = model_dir / ARTIFACT_NAME
    joblib.dump(bundles[best_name], artifact)

    results = {
        "protocol": {
            "split": "chronological",
            "test_frac": test_frac,
            "cv": f"TimeSeriesSplit(n_splits={cv_folds}) for selection; "
            f"StratifiedKFold(n_splits={cv_folds}, shuffle=True) for the optimism comparison",
            "selection_metric": "average_precision",
            "operating_point_policies": {
                "min_expected_cost": f"cost-optimal flag rate on forward-chaining out-of-fold "
                f"predictions (c_fn={c_fn}, c_fp={c_fp}), mapped onto the refitted "
                f"model's training score distribution",
                "inspection_budget": f"flag the highest-scoring fraction of runs; "
                f"budgets {list(INSPECTION_BUDGETS)}, deployed {deploy_budget}",
            },
            "seed": seed,
        },
        "dataset": {
            "n_runs": int(len(ds)),
            "n_sensors": int(ds.X.shape[1]),
            "n_sensors_after_preprocessing": len(kept),
            "fail_rate_overall": float(ds.y.mean()),
            "fail_rate_train": float(y_train.mean()),
            "fail_rate_test": float(y_test.mean()),
            "missing_cell_fraction": float(ds.X.isna().to_numpy().mean()),
            "constant_sensors": int((ds.X.nunique(dropna=True) <= 1).sum()),
            "sensors_over_50pct_missing": int((ds.X.isna().mean() > 0.5).sum()),
            "time_span": [str(ds.timestamps.iloc[0]), str(ds.timestamps.iloc[-1])],
            "source_digests": ds.source_digests,
        },
        "deployed_model": best_name,
        "protocol_selected_model": protocol_choice,
        "deployed_threshold": float(deployed_thr),
        "models": per_model,
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=False))
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    pd.concat(cv_frames, ignore_index=True).to_csv(out_dir / "cv_results.csv", index=False)

    cols = [
        "model", "cv_temporal_ap", "cv_random_ap", "optimism_gap", "test_ap",
        "test_roc_auc", "recall_at_budget", "precision_at_budget", "lift_at_budget",
        "cost_vs_flag_nothing", "cost_policy_abstains",
    ]
    print(f"\n{leaderboard[cols].to_string(index=False)}")
    print(f"\ndeployed: {best_name} (threshold {deployed_thr:.4f}) -> {artifact}")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-missing-rate", type=float, default=0.5)
    p.add_argument("--cost-fn", type=float, default=COST_FALSE_NEGATIVE)
    p.add_argument("--cost-fp", type=float, default=COST_FALSE_POSITIVE)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--deploy-budget", type=float, default=DEFAULT_BUDGET)
    p.add_argument("--deploy-model", default=None,
                   help="deploy this model instead of the protocol-selected one")
    p.add_argument("--only", nargs="*", default=None, help="restrict to these model names")
    a = p.parse_args(argv)
    run(
        data_dir=a.data_dir,
        out_dir=a.out_dir,
        model_dir=a.model_dir,
        test_frac=a.test_frac,
        cv_folds=a.cv_folds,
        seed=a.seed,
        max_missing_rate=a.max_missing_rate,
        c_fn=a.cost_fn,
        c_fp=a.cost_fp,
        n_boot=a.n_boot,
        deploy_budget=a.deploy_budget,
        deploy_model=a.deploy_model,
        only=a.only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
