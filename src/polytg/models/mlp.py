"""Compact scikit-learn MLP baseline for fixed molecular features."""

from __future__ import annotations

from typing import Any

from sklearn.neural_network import MLPRegressor


def build_mlp(parameters: dict[str, Any], seed: int) -> MLPRegressor:
    defaults: dict[str, Any] = {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 30,
    }
    defaults.update(parameters)
    defaults["random_state"] = seed
    return MLPRegressor(**defaults)
