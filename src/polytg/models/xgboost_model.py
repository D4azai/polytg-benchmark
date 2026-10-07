"""XGBoost construction with deterministic defaults."""

from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor


def build_xgboost(parameters: dict[str, Any], seed: int) -> XGBRegressor:
    defaults: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "objective": "reg:squarederror",
        "n_jobs": 1,
        "tree_method": "hist",
    }
    defaults.update(parameters)
    defaults["random_state"] = seed
    return XGBRegressor(**defaults)

