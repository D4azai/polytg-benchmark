"""Regularized linear regression factories."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import ElasticNet, Ridge


def build_ridge(parameters: dict[str, Any], seed: int) -> Ridge:
    defaults: dict[str, Any] = {"alpha": 1.0}
    defaults.update(parameters)
    defaults.setdefault("random_state", seed)
    return Ridge(**defaults)


def build_elastic_net(parameters: dict[str, Any], seed: int) -> ElasticNet:
    defaults: dict[str, Any] = {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 10000}
    defaults.update(parameters)
    defaults["random_state"] = seed
    return ElasticNet(**defaults)

