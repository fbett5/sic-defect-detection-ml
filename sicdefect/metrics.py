"""Metrics for imbalanced wafer-map classification.

Plain accuracy is misleading here: ~85% of labeled wafers are 'none', so a
model that always predicts 'none' scores ~85%. Use macro-F1 and balanced
accuracy as the headline numbers.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .wm811k import CLASSES, CLASS_TO_IDX


def classification_metrics(y_true, y_pred, classes=CLASSES) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(classes)))
    out = {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    # Binary view: is the wafer defective at all? This is the question a fab asks first.
    none = CLASS_TO_IDX["none"]
    t_def = y_true != none
    p_def = y_pred != none
    out["defect_recall"] = recall_score(t_def, p_def, zero_division=0)
    out["defect_precision"] = precision_score(t_def, p_def, zero_division=0)

    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for c, f in zip(classes, per_class):
        out[f"f1_{c}"] = float(f)
    return {k: float(v) for k, v in out.items()}


def report(y_true, y_pred, classes=CLASSES) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(classes))), target_names=classes, zero_division=0, digits=3
    )


def plot_confusion(y_true, y_pred, path, classes=CLASSES, title="Confusion matrix (row-normalised)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
