"""Evaluation metrics beyond sklearn defaults: calibration (ECE) and helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE. A user-facing probability must be calibrated:
    '0.9 manipulation score' should be wrong about 1 time in 10."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(conf - acc)
    return float(ece)


def classification_report_dict(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1_clickbait": float(f1_score(y_true, y_pred, pos_label=1)),
        "ece": expected_calibration_error(np.asarray(y_true), np.asarray(y_prob)),
        "positive_rate": float(np.mean(y_pred)),
        "n": float(len(y_true)),
    }
