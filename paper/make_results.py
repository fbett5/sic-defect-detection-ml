"""Build the paper's tables and figures from whatever runs exist on disk.

Nothing in the paper is typed by hand.  Every number comes from a
``outputs/<run>/test_metrics.json`` written by ``train_cnn.py``, so re-running
an experiment and re-running this script keeps the manuscript honest.

Usage::

    python paper/make_results.py                      # all runs under outputs/
    python paper/make_results.py --outputs outputs --tag synthetic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicdefect.wm811k import CLASSES  # noqa: E402

HEADLINE = ["macro_f1", "balanced_acc", "accuracy", "defect_recall", "defect_precision"]
PRETTY = {
    "macro_f1": "macro-F1",
    "balanced_acc": "balanced acc.",
    "accuracy": "accuracy",
    "defect_recall": "defect recall",
    "defect_precision": "defect precision",
}
plt.rcParams.update(
    {"figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
     "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
     "axes.spines.right": False, "legend.frameon": False}
)


def collect(outputs: Path) -> pd.DataFrame:
    rows = []
    for metrics_file in sorted(outputs.glob("*/test_metrics.json")):
        run = metrics_file.parent.name
        m = json.loads(metrics_file.read_text())
        ck = metrics_file.parent / "best.pt"
        row = {"run": run, **m}
        live = metrics_file.parent / "live.json"
        if live.exists():
            hist = json.loads(live.read_text()).get("history", {})
            row["epochs_trained"] = len(hist.get("epoch", []))
            if hist.get("val_macro_f1"):
                row["best_val_macro_f1"] = max(hist["val_macro_f1"])
        row["has_checkpoint"] = ck.exists()
        rows.append(row)
    return pd.DataFrame(rows)


def latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> Path:
    cols = [c for c in HEADLINE if c in df.columns]
    body = "\n".join(
        r"\texttt{" + r["run"].replace("_", r"\_") + "} & "
        + " & ".join(f"{r[c]:.3f}" for c in cols)
        + r" \\"
        for _, r in df.iterrows()
    )
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n"
        "run & " + " & ".join(PRETTY[c] for c in cols) + r" \\" + "\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    return path


def markdown_table(df: pd.DataFrame, path: Path) -> Path:
    cols = [c for c in HEADLINE if c in df.columns]
    lines = ["| run | " + " | ".join(PRETTY[c] for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for _, r in df.iterrows():
        lines.append(f"| `{r['run']}` | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path


def per_class_table(df: pd.DataFrame, path: Path) -> Path:
    cols = [f"f1_{c}" for c in CLASSES if f"f1_{c}" in df.columns]
    lines = ["| run | " + " | ".join(c[3:] for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for _, r in df.iterrows():
        lines.append(f"| `{r['run']}` | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path


def fig_training_curves(outputs: Path, path: Path) -> Path | None:
    runs = sorted(outputs.glob("*/live.json"))
    if not runs:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    drew = False
    for i, live in enumerate(runs):
        h = json.loads(live.read_text()).get("history", {})
        if not h.get("epoch"):
            continue
        drew = True
        name = live.parent.name
        a1.plot(h["epoch"], h["train_loss"], lw=1.4, label=f"{name} train")
        if h.get("val_loss"):
            a1.plot(h["epoch"], h["val_loss"], lw=1.4, ls="--", label=f"{name} val")
        a2.plot(h["epoch"], h["val_macro_f1"], lw=1.6, label=f"{name}")
    if not drew:
        plt.close(fig)
        return None
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("Loss"); a1.legend(fontsize=7)
    a2.set_xlabel("epoch"); a2.set_ylabel("validation macro-F1")
    a2.set_title("Validation macro-F1"); a2.set_ylim(0, 1.02); a2.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_class_balance(processed: Path, path: Path) -> Path | None:
    summary = processed / "split_summary.csv"
    if not summary.exists():
        return None
    df = pd.read_csv(summary, index_col=0).drop(index="TOTAL", errors="ignore")
    splits = [c for c in ("train", "val", "test") if c in df.columns]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    bottom = np.zeros(len(df))
    for s in splits:
        ax.bar(df.index, df[s], bottom=bottom, label=s)
        bottom += df[s].to_numpy()
    ax.set_yscale("log")
    ax.set_ylabel("wafers (log scale)")
    ax.set_title("Class counts per lot-grouped split")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--outputs", default=str(root / "outputs"))
    ap.add_argument("--processed", default=str(root / "data" / "synthetic"))
    ap.add_argument("--out", default=str(root / "paper"))
    ap.add_argument("--tag", default="results", help="label used in filenames")
    a = ap.parse_args(argv)

    outputs, out = Path(a.outputs), Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = collect(outputs)
    if df.empty:
        print(f"no runs with test_metrics.json under {outputs}; nothing to build")
        return 0

    made = [
        df.to_csv(out / "tables" / f"{a.tag}.csv", index=False) or out / "tables" / f"{a.tag}.csv",
        latex_table(df, out / "tables" / f"{a.tag}.tex",
                    "Test-split results, regenerated by \\texttt{paper/make\\_results.py}.",
                    f"tab:{a.tag}"),
        markdown_table(df, out / "tables" / f"{a.tag}.md"),
        per_class_table(df, out / "tables" / f"{a.tag}_per_class.md"),
    ]
    for f in (
        fig_training_curves(outputs, out / "figures" / "training_curves.png"),
        fig_class_balance(Path(a.processed), out / "figures" / "class_balance.png"),
    ):
        if f:
            made.append(f)

    # Carry over artefacts the training script already produced.
    for run_dir in sorted(outputs.glob("*/")):
        cm = run_dir / "test_confusion.png"
        if cm.exists():
            (out / "figures" / f"confusion_{run_dir.name}.png").write_bytes(cm.read_bytes())
            made.append(out / "figures" / f"confusion_{run_dir.name}.png")
    samples = Path(a.processed) / "samples.png"
    if samples.exists():
        (out / "figures" / "samples.png").write_bytes(samples.read_bytes())
        made.append(out / "figures" / "samples.png")

    print(f"{len(df)} run(s) found: {', '.join(df['run'])}")
    for f in made:
        print(f"  wrote {Path(f).relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
