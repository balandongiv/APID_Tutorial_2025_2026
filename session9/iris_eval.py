"""session 9 evaluation helpers for experiment runs."""

from __future__ import annotations

import numpy as np


def train_test_split(X, y, test_size=0.2, seed=1):
    """NumPy-only train/test split."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    test_n = int(n * test_size)
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy_score(y_true, y_pred):
    """Return scalar accuracy."""
    return float(np.mean(y_true == y_pred))
