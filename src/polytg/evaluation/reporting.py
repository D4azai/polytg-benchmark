"""Structured run tracking and result aggregation."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost

RESULT_COLUMNS = [
    "experiment_id", "timestamp_utc", "representation", "model", "split_strategy",
    "seed", "mae", "rmse", "r2", "validation_mae", "training_time_seconds",
    "inference_time_seconds", "split_id", "config_hash", "git_commit",
    "prediction_file", "model_file", "n_trainable_parameters", "checkpoint_path",
    "epochs_trained",
]


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": rdkit.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    for package in ("torch", "torch-geometric", "transformers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def append_benchmark_row(row: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=RESULT_COLUMNS)
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated = updated.drop_duplicates(subset=["experiment_id"], keep="last")
    updated.to_csv(path, index=False)
    return path
