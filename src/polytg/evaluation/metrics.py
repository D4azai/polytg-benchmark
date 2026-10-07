"""Regression metrics used uniformly across model families."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if truth.size == 0 or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Metrics require non-empty finite arrays")
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if truth.size >= 2 else None,
    }
