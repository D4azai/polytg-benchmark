import numpy as np
import pandas as pd
import pytest

from polytg.evaluation.statistics import paired_bootstrap_mae_difference, summarize_results


def test_repeated_result_summary_has_sample_statistics() -> None:
    results = pd.DataFrame(
        {
            "representation": ["ecfp", "ecfp", "ecfp"],
            "model": ["ridge", "ridge", "ridge"],
            "split_strategy": ["random", "random", "random"],
            "mae": [1.0, 2.0, 3.0],
            "rmse": [2.0, 3.0, 4.0],
            "r2": [0.7, 0.6, 0.5],
            "training_time_seconds": [1.0, 1.5, 2.0],
        }
    )
    summary = summarize_results(results)
    assert summary.loc[0, "runs"] == 3
    assert summary.loc[0, "mae_mean"] == pytest.approx(2.0)
    assert summary.loc[0, "mae_std"] == pytest.approx(1.0)
    assert summary.loc[0, "mae_ci_low"] < 2.0 < summary.loc[0, "mae_ci_high"]


def test_paired_bootstrap_is_deterministic_and_directional() -> None:
    actual = np.arange(20, dtype=float)
    prediction_a = actual + 1.0
    prediction_b = actual + 3.0
    first = paired_bootstrap_mae_difference(
        actual, prediction_a, prediction_b, n_bootstrap=500, seed=11
    )
    second = paired_bootstrap_mae_difference(
        actual, prediction_a, prediction_b, n_bootstrap=500, seed=11
    )
    assert first == second
    assert first["mae_difference_a_minus_b"] == pytest.approx(-2.0)
    assert first["ci_high"] < 0.0
