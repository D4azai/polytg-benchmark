"""Publication-friendly diagnostic plots for completed regression runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def save_prediction_plots(
    predictions: pd.DataFrame, output_dir: str | Path, stem: str
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = predictions.loc[predictions["partition"] == "test"]
    if test.empty:
        raise ValueError("Prediction plots require test-set predictions")
    actual = test["actual"].to_numpy(dtype=float)
    predicted = test["predicted"].to_numpy(dtype=float)
    residual = actual - predicted
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.scatter(actual, predicted, alpha=0.75, edgecolor="none")
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    axis.set(xlabel="Experimental Tg", ylabel="Predicted Tg", title="Predicted vs. experimental Tg")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / f"{stem}_predicted_vs_actual.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = min(30, max(5, int(np.sqrt(len(residual)))))
    axes[0].hist(residual, bins=bins, color="#4472C4", alpha=0.85)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Residual (experimental − predicted)",
        ylabel="Count",
        title="Residual distribution",
    )
    axes[1].scatter(predicted, residual, alpha=0.75, edgecolor="none")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Predicted Tg", ylabel="Residual", title="Residuals vs. predictions")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / f"{stem}_residuals.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths.append(path)
    return paths


def save_learning_curve(history: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(history["epoch"], history["train_loss"], label="Training loss")
    axis.plot(history["epoch"], history["validation_mae"], label="Validation MAE")
    axis.set(xlabel="Epoch", ylabel="Metric value", title="Neural-network learning curve")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def save_benchmark_comparison_plots(
    summary: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    if summary.empty:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = (
        summary["representation"].astype(str)
        + " / "
        + summary["model"].astype(str)
        + " / "
        + summary["split_strategy"].astype(str)
    )
    positions = np.arange(len(summary))
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(max(8, len(summary) * 0.55), 5.5))
    axis.bar(positions, summary["r2_mean"], color="#4472C4")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(ylabel="Mean test R²", title="Model comparison across representations and splits")
    axis.set_xticks(positions, labels, rotation=55, ha="right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = output_dir / "model_comparison_r2.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(2, 1, figsize=(max(8, len(summary) * 0.55), 9), sharex=True)
    for axis, metric, title in (
        (axes[0], "mae", "Mean absolute error"),
        (axes[1], "rmse", "Root mean squared error"),
    ):
        values = summary[f"{metric}_mean"].to_numpy(dtype=float)
        standard_deviation = summary[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
        axis.bar(positions, values, yerr=standard_deviation, capsize=3, color="#70AD47")
        axis.set(ylabel=f"{metric.upper()} (mean ± SD)", title=title)
        axis.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(positions, labels, rotation=55, ha="right")
    figure.tight_layout()
    path = output_dir / "error_metrics_comparison.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    paths.append(path)
    return paths
