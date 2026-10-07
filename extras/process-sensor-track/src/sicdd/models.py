"""Model zoo: a prior-rate baseline, three classical learners, and a small MLP.

Every entry is an (estimator, parameter grid) pair that plugs into the same
preprocessing pipeline, so the comparison isolates the learner rather than the
feature handling.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

DEFAULT_SEED = 20251007


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """A small class-weighted MLP trained by backpropagation, on CPU.

    Deliberately modest: with roughly 1,250 training runs and 444 inputs, a
    large network would memorise the training split long before it generalised.
    Training uses a class-weighted binary cross-entropy so the 6.6% failing runs
    are not drowned out, and stops early on the validation average precision of
    a chronologically held-out tail of the training split.

    Parameters
    ----------
    hidden : sizes of the hidden layers.
    dropout : dropout probability applied after each hidden layer.
    lr, weight_decay : AdamW optimiser settings.
    max_epochs, patience : early-stopping budget.
    val_frac : fraction of the training split (its chronological tail) held out
        to select the stopping epoch.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        max_epochs: int = 300,
        patience: int = 30,
        batch_size: int = 64,
        val_frac: float = 0.2,
        class_weight: str | None = "balanced",
        random_state: int = DEFAULT_SEED,
        verbose: bool = False,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

    def _build(self, n_features: int):
        import torch.nn as nn

        layers: list[nn.Module] = []
        prev = n_features
        for width in self.hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        from sklearn.metrics import average_precision_score

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32).ravel()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # Chronological validation tail: rows arrive time-ordered, so holding out
        # the tail mirrors how the model will be used.
        n_val = max(1, int(round(len(X) * self.val_frac)))
        if n_val >= len(X):
            raise ValueError("val_frac leaves no training rows")
        Xtr, ytr = X[:-n_val], y[:-n_val]
        Xva, yva = X[-n_val:], y[-n_val:]

        pos = float(ytr.sum())
        if self.class_weight == "balanced" and pos > 0:
            pos_weight = torch.tensor([(len(ytr) - pos) / pos], dtype=torch.float32)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)

        model = self._build(X.shape[1])
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        Xtr_t = torch.from_numpy(Xtr)
        ytr_t = torch.from_numpy(ytr).unsqueeze(1)
        Xva_t = torch.from_numpy(Xva)

        gen = torch.Generator().manual_seed(self.random_state)
        best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(Xtr_t), generator=gen)
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_scores = torch.sigmoid(model(Xva_t)).numpy().ravel()
            # Average precision is undefined without both classes present.
            score = (
                average_precision_score(yva, val_scores)
                if 0 < yva.sum() < len(yva)
                else -np.inf
            )
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.best_epoch_ = best_epoch
        self.best_val_ap_ = float(best_score) if np.isfinite(best_score) else None
        return self

    def predict_proba(self, X):
        import torch

        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.from_numpy(X))).numpy().ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def model_zoo(seed: int = DEFAULT_SEED) -> dict[str, tuple[BaseEstimator, dict]]:
    """Estimators and the (small) grids searched over for each.

    Grids are kept short on purpose: with 104 failing runs in total, a large
    search would select on noise.
    """
    return {
        "prior_baseline": (DummyClassifier(strategy="prior"), {}),
        "logistic_l2": (
            LogisticRegression(
                class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed
            ),
            {"clf__C": [0.003, 0.01, 0.03, 0.1, 1.0]},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            ),
            {"clf__max_features": ["sqrt", 0.1], "clf__max_depth": [None, 8]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=seed,
            ),
            {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [7, 31]},
        ),
        "torch_mlp": (
            TorchMLPClassifier(random_state=seed),
            {"clf__hidden": [(64, 32), (32,)], "clf__weight_decay": [1e-3, 1e-2]},
        ),
    }
