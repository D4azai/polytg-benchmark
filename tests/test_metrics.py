import numpy as np
import pytest

from polytg.evaluation.metrics import regression_metrics


def test_regression_metrics_known_values() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["r2"] == pytest.approx(0.5)


def test_metrics_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_r2_is_none_for_single_sample() -> None:
    metrics = regression_metrics(np.array([1.0]), np.array([1.5]))
    assert metrics["r2"] is None
