"""Repeated-run summaries and paired prediction-level comparisons."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GROUP_COLUMNS = ["representation", "model", "split_strategy"]


def summarize_results(results: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    required = set(GROUP_COLUMNS + ["mae", "rmse", "r2", "training_time_seconds"])
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results table is missing column(s): {', '.join(sorted(missing))}")
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        row = dict(zip(GROUP_COLUMNS, keys, strict=True))
        row["runs"] = len(group)
        for metric in ("mae", "rmse", "r2", "training_time_seconds"):
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            if len(values) > 1:
                sem = stats.sem(values)
                half_width = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1) * sem)
                row[f"{metric}_ci_low"] = float(values.mean() - half_width)
                row[f"{metric}_ci_high"] = float(values.mean() + half_width)
            else:
                row[f"{metric}_ci_low"] = np.nan
                row[f"{metric}_ci_high"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_mae_difference(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    prediction_a = np.asarray(prediction_a, dtype=float)
    prediction_b = np.asarray(prediction_b, dtype=float)
    if actual.shape != prediction_a.shape or actual.shape != prediction_b.shape:
        raise ValueError("Paired bootstrap arrays must have identical shapes")
    if actual.size < 2 or n_bootstrap <= 0:
        raise ValueError("Paired bootstrap requires at least two samples and positive iterations")
    per_sample_difference = np.abs(actual - prediction_a) - np.abs(actual - prediction_b)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, actual.size, size=actual.size)
        bootstrap[index] = per_sample_difference[sampled].mean()
    alpha = 1.0 - confidence
    probability_nonpositive = float(np.mean(bootstrap <= 0.0))
    probability_nonnegative = float(np.mean(bootstrap >= 0.0))
    return {
        "n_test": int(actual.size),
        "n_bootstrap": n_bootstrap,
        "mae_difference_a_minus_b": float(per_sample_difference.mean()),
        "ci_low": float(np.quantile(bootstrap, alpha / 2)),
        "ci_high": float(np.quantile(bootstrap, 1 - alpha / 2)),
        "two_sided_tail_probability": min(
            1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)
        ),
    }


def _prediction_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def pairwise_comparisons(
    results: pd.DataFrame,
    project_root: str | Path,
    *,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare runs only when they share a persisted split and seed."""
    project_root = Path(project_root)
    rows: list[dict[str, object]] = []
    grouping = ["split_id", "split_strategy", "seed"]
    for keys, matched in results.groupby(grouping, dropna=False, sort=True):
        for (_, run_a), (_, run_b) in itertools.combinations(matched.iterrows(), 2):
            label_a = f"{run_a['representation']}::{run_a['model']}"
            label_b = f"{run_b['representation']}::{run_b['model']}"
            if label_a == label_b:
                continue
            predictions_a = pd.read_csv(
                _prediction_path(str(run_a["prediction_file"]), project_root)
            )
            predictions_b = pd.read_csv(
                _prediction_path(str(run_b["prediction_file"]), project_root)
            )
            test_a = predictions_a.loc[
                predictions_a["partition"] == "test", ["row_index", "actual", "predicted"]
            ]
            test_b = predictions_b.loc[
                predictions_b["partition"] == "test", ["row_index", "actual", "predicted"]
            ]
            paired = test_a.merge(
                test_b,
                on="row_index",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            if len(paired) != len(test_a) or len(paired) != len(test_b):
                raise ValueError(f"Test rows are not paired for {label_a} and {label_b}")
            if not np.allclose(paired["actual_a"], paired["actual_b"]):
                raise ValueError(f"Actual targets disagree for {label_a} and {label_b}")
            comparison = paired_bootstrap_mae_difference(
                paired["actual_a"].to_numpy(),
                paired["predicted_a"].to_numpy(),
                paired["predicted_b"].to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "split_id": keys[0],
                    "split_strategy": keys[1],
                    "seed": keys[2],
                    "model_a": label_a,
                    "model_b": label_b,
                    **comparison,
                }
            )
    return pd.DataFrame(rows)
