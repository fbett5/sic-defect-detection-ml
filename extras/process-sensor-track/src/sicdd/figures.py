"""Figures for the paper, regenerated from saved results and models.

Every figure is produced from ``results/metrics.json``, the saved per-model
artifacts and the dataset itself, so nothing in the paper is drawn by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

from .data import load_secom, temporal_split  # noqa: E402
from .evaluate import expected_cost  # noqa: E402
from .train import INSPECTION_BUDGETS  # noqa: E402

# A small, colour-blind-safe qualitative set; the deployed model is drawn heavier.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _scores(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def fig_drift(ds, train_idx, out: Path) -> Path:
    """Monthly failure rate, with the train/test boundary marked."""
    df = pd.DataFrame({"ts": ds.timestamps, "y": ds.y}).set_index("ts")
    monthly = df["y"].resample("ME").agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.bar(
        monthly.index, monthly["mean"] * 100, width=20, color=PALETTE[0], alpha=0.85,
        label="monthly failure rate",
    )
    ax.axhline(
        ds.y.mean() * 100, color=PALETTE[3], ls="--", lw=1.2,
        label=f"overall {ds.y.mean():.1%}",
    )
    boundary = ds.timestamps.iloc[train_idx[-1]]
    ax.axvline(boundary, color="black", lw=1.2, ls=":", label="train/test boundary")
    for ts, row in monthly.iterrows():
        ax.annotate(f"n={int(row['size'])}", (ts, row["mean"] * 100),
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("failing runs (%)")
    ax.set_title("The failure rate drifts by a factor of eight across the record")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_missingness(ds, out: Path) -> Path:
    """Distribution of per-sensor missing rates and constant sensors."""
    rates = ds.X.isna().mean().to_numpy()
    nuniq = ds.X.nunique(dropna=True).to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    a1.hist(rates * 100, bins=40, color=PALETTE[0])
    a1.axvline(50, color=PALETTE[3], ls="--", lw=1.2, label="drop threshold (50%)")
    a1.set_xlabel("missing values per sensor (%)")
    a1.set_ylabel("sensors")
    a1.set_yscale("log")
    a1.set_title(f"{(rates > 0.5).sum()} sensors exceed 50% missing")
    a1.legend(fontsize=8)
    a2.hist(np.clip(nuniq, 0, 60), bins=40, color=PALETTE[2])
    a2.set_xlabel("distinct values per sensor (clipped at 60)")
    a2.set_title(f"{(nuniq <= 1).sum()} sensors are constant")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_curves(y_test, score_by_model, deployed, out_pr: Path, out_roc: Path) -> list[Path]:
    """Precision-recall and ROC curves on the held-out window."""
    prevalence = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    for i, (name, s) in enumerate(score_by_model.items()):
        if len(np.unique(s)) <= 1:
            continue
        pr, rc, _ = precision_recall_curve(y_test, s)
        is_dep = name == deployed
        ax.step(rc, pr, where="post", color=PALETTE[i % len(PALETTE)],
                lw=2.2 if is_dep else 1.1, alpha=1.0 if is_dep else 0.75,
                label=f"{name}{' (deployed)' if is_dep else ''}")
    ax.axhline(prevalence, color="black", ls="--", lw=1.0, label=f"no skill ({prevalence:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall, held-out window")
    ax.legend(fontsize=7, loc="upper right")
    fig.savefig(out_pr, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    for i, (name, s) in enumerate(score_by_model.items()):
        if len(np.unique(s)) <= 1:
            continue
        fpr, tpr, _ = roc_curve(y_test, s)
        is_dep = name == deployed
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2.2 if is_dep else 1.1,
                alpha=1.0 if is_dep else 0.75,
                label=f"{name}{' (deployed)' if is_dep else ''}")
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1.0, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC, held-out window")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(out_roc, bbox_inches="tight")
    plt.close(fig)
    return [out_pr, out_roc]


def fig_optimism(metrics: dict, out: Path) -> Path:
    """Random-split versus forward-chaining cross-validated average precision."""
    names, temporal, random_ = [], [], []
    for name, m in metrics["models"].items():
        names.append(name)
        temporal.append(m["cv_temporal_ap_mean"])
        random_.append(m["cv_random_ap_mean"])
    order = np.argsort(temporal)
    names = [names[i] for i in order]
    temporal = np.array(temporal)[order]
    random_ = np.array(random_)[order]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.barh(y - 0.2, random_, height=0.38, color=PALETTE[1], label="random 5-fold CV")
    ax.barh(y + 0.2, temporal, height=0.38, color=PALETTE[0], label="forward-chaining CV")
    for i, (r, t) in enumerate(zip(random_, temporal)):
        ax.annotate(f"+{r - t:+.3f}".replace("++", "+"), (max(r, t) + 0.004, i),
                    va="center", fontsize=7.5)
    ax.set_yticks(y, names)
    ax.set_xlabel("average precision (training runs)")
    ax.set_title("Random splits overstate performance on a drifting process")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cost(y_test, score, metrics: dict, deployed: str, out: Path) -> Path:
    """Expected cost against threshold, with the policies marked."""
    c_fn = metrics["models"][deployed]["test_cost_sensitivity"][2]["cost_ratio_fn_to_fp"]
    grid = np.unique(np.concatenate([[0.0], np.quantile(score, np.linspace(0, 1, 200)), [1.0]]))
    costs = [expected_cost(y_test, score, t, c_fn=c_fn, c_fp=1.0) for t in grid]
    nothing = c_fn * float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(grid, costs, color=PALETTE[0], lw=1.6, label="cost of acting at threshold")
    ax.axhline(nothing, color=PALETTE[3], ls="--", lw=1.2, label="flag nothing")
    thresholds = metrics["models"][deployed]["budget_policy_thresholds"]
    for i, b in enumerate(INSPECTION_BUDGETS):
        t = thresholds[f"budget_{int(b*100)}pct"]
        ax.axvline(t, color=PALETTE[2 + i % 3], lw=1.0, ls=":",
                   label=f"{int(b*100)}% inspection budget")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel(f"expected cost per run ({c_fn:g}:1)")
    ax.set_title(f"Cost landscape, {deployed}")
    ax.legend(fontsize=7.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_confusion(y_test, score, threshold: float, deployed: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (score >= threshold).astype(int),
        display_labels=["pass", "fail"], cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"{deployed} at deployed threshold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_importance(pipeline, X_test, y_test, names, out: Path, n_top: int = 15,
                   seed: int = 20251007) -> Path:
    """Permutation importance of individual sensors on the held-out window."""
    r = permutation_importance(
        pipeline, X_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=seed, n_jobs=1,
    )
    order = np.argsort(r.importances_mean)[-n_top:]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.barh(np.arange(len(order)), r.importances_mean[order],
            xerr=r.importances_std[order], color=PALETTE[0], alpha=0.9,
            error_kw={"lw": 0.8, "capsize": 2})
    ax.set_yticks(np.arange(len(order)), [names[i] for i in order], fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("drop in average precision when the sensor is shuffled")
    ax.set_title(f"Top {n_top} sensors by permutation importance")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default="results/metrics.json")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--also-copy-to", default="paper/figures")
    p.add_argument("--skip-importance", action="store_true")
    a = p.parse_args(argv)

    metrics = json.loads(Path(a.results).read_text())
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_secom(a.data_dir)
    train_idx, test_idx = temporal_split(len(ds), test_frac=metrics["protocol"]["test_frac"])
    X_test, y_test = ds.X.iloc[test_idx], ds.y[test_idx]
    deployed = metrics["deployed_model"]

    score_by_model = {}
    for name in metrics["models"]:
        path = Path(a.model_dir) / f"{name}.joblib"
        if path.exists():
            score_by_model[name] = _scores(joblib.load(path), X_test)

    made = [
        fig_drift(ds, train_idx, out_dir / "fig1_drift.png"),
        fig_missingness(ds, out_dir / "fig2_missingness.png"),
        *fig_curves(y_test, score_by_model, deployed,
                    out_dir / "fig4_pr_curves.png", out_dir / "fig5_roc_curves.png"),
        fig_optimism(metrics, out_dir / "fig3_optimism.png"),
        fig_cost(y_test, score_by_model[deployed], metrics, deployed,
                 out_dir / "fig6_cost.png"),
        fig_confusion(y_test, score_by_model[deployed], metrics["deployed_threshold"],
                      deployed, out_dir / "fig7_confusion.png"),
    ]
    if not a.skip_importance:
        made.append(
            fig_importance(
                joblib.load(Path(a.model_dir) / f"{deployed}.joblib"),
                X_test, y_test, list(ds.X.columns), out_dir / "fig8_importance.png",
            )
        )

    if a.also_copy_to:
        dest = Path(a.also_copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        for f in made:
            dest.joinpath(f.name).write_bytes(f.read_bytes())

    for f in made:
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
