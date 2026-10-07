"""Random Forest construction."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor


def build_random_forest(parameters: dict[str, Any], seed: int) -> RandomForestRegressor:
    defaults: dict[str, Any] = {"n_estimators": 500, "n_jobs": -1}
    defaults.update(parameters)
    defaults["random_state"] = seed
    return RandomForestRegressor(**defaults)

