"""Probability calibration fitted on the validation split.

We calibrate on val (never train, never test) and report ECE before/after on
test. Isotonic for larger val sets, Platt/sigmoid for small ones.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    """Maps raw model probabilities -> calibrated probabilities."""

    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "sigmoid"}:
            raise ValueError(f"unknown calibration method: {method}")
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> Calibrator:
        probs_val = np.asarray(probs_val, dtype=float)
        y_val = np.asarray(y_val, dtype=int)
        # isotonic needs enough points to be stable; fall back to sigmoid
        if self.method == "isotonic" and len(y_val) >= 200:
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(probs_val, y_val)
        else:
            self.method = "sigmoid"
            self._platt = LogisticRegression(C=1e6)
            self._platt.fit(probs_val.reshape(-1, 1), y_val)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if self._iso is not None:
            return np.asarray(self._iso.transform(probs))
        assert self._platt is not None, "Calibrator not fitted"
        return np.asarray(self._platt.predict_proba(probs.reshape(-1, 1))[:, 1])
