"""Unified classical-model training for 2D molecular representations."""

from __future__ import annotations

import logging
import time
import uuid
from copy import deepcopy
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polytg.data.preparation import prepare_from_config
from polytg.evaluation.metrics import regression_metrics
from polytg.evaluation.plots import save_prediction_plots
from polytg.evaluation.reporting import (
    append_benchmark_row,
    git_commit,
    software_versions,
    utc_timestamp,
    write_json,
)
from polytg.models.linear import build_elastic_net, build_ridge
from polytg.models.mlp import build_mlp
from polytg.models.random_forest import build_random_forest
from polytg.models.xgboost_model import build_xgboost
from polytg.representations.fingerprints import compute_morgan_fingerprints
from polytg.representations.language_model import (
    DEFAULT_CHEMBERTA_MODEL,
    DEFAULT_CHEMBERTA_REVISION,
    compute_or_load_embeddings,
)
from polytg.representations.rdkit_descriptors import DEFAULT_DESCRIPTOR_NAMES, compute_descriptors
from polytg.training.seeds import set_global_seed
from polytg.utils.config import config_hash
from polytg.utils.paths import PROJECT_ROOT, display_path, resolve_project_path

LOGGER = logging.getLogger(__name__)


def _build_model(name: str, parameters: dict[str, Any], seed: int) -> object:
    if name == "random_forest":
        return build_random_forest(parameters, seed)
    if name == "xgboost":
        return build_xgboost(parameters, seed)
    if name == "ridge":
        return build_ridge(parameters, seed)
    if name == "elastic_net":
        return build_elastic_net(parameters, seed)
    if name == "mlp":
        return build_mlp(parameters, seed)
    raise ValueError(f"Unsupported classical model: {name}")


def _compute_representation(
    config: dict[str, Any], smiles: list[str]
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    representation = config["representation"]
    name = representation.get("name")
    if name == "rdkit_descriptors":
        requested = representation.get("descriptors", list(DEFAULT_DESCRIPTOR_NAMES))
        features, failures = compute_descriptors(smiles, requested)
        metadata = {"descriptor_names": list(requested)}
        return features, failures, metadata
    if name == "ecfp":
        parameters = {
            "radius": int(representation.get("radius", 2)),
            "n_bits": int(representation.get("n_bits", 2048)),
            "include_chirality": bool(representation.get("include_chirality", False)),
            "use_bond_types": bool(representation.get("use_bond_types", True)),
        }
        features, failures = compute_morgan_fingerprints(smiles, **parameters)
        return features, failures, parameters
    if name == "chembert":
        parameters = {
            "model_name": representation.get("model_name", DEFAULT_CHEMBERTA_MODEL),
            "revision": str(
                representation.get("revision", DEFAULT_CHEMBERTA_REVISION)
            ),
            "pooling": representation.get("pooling", "mean"),
            "max_length": int(representation.get("max_length", 256)),
            "batch_size": int(representation.get("batch_size", 32)),
            "device": representation.get("device", "auto"),
            "cache_dir": resolve_project_path(
                representation.get("cache_dir", "data/processed/embeddings")
            ),
        }
        features, metadata = compute_or_load_embeddings(smiles, **parameters)
        metadata["cache_dir"] = display_path(parameters["cache_dir"])
        return features, [], metadata
    raise ValueError(
        "Unsupported classical representation. Choose rdkit_descriptors, ecfp, or chembert."
    )


def _feature_names(pipeline: Pipeline, original_names: list[str]) -> list[str]:
    selected = pipeline.named_steps["variance"].get_support()
    return [name for name, keep in zip(original_names, selected, strict=True) if keep]


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare data, train one model, and persist a complete run record."""
    if config["representation"].get("name") == "molecular_graph":
        from polytg.training.graph_trainer import run_graph_experiment

        return run_graph_experiment(config)
    seed = int(config.get("training", {}).get("seed", config["split"].get("seed", 42)))
    set_global_seed(seed)
    frame, split, split_path = prepare_from_config(config)
    data_config = config["data"]
    smiles_column = data_config["smiles_column"]
    target_column = data_config["target_column"]

    smiles = frame[smiles_column].astype(str).tolist()
    features, failures, representation_metadata = _compute_representation(config, smiles)
    all_nan_rows = features.isna().all(axis=1)
    if all_nan_rows.any():
        bad = np.flatnonzero(all_nan_rows.to_numpy()).tolist()
        raise ValueError(f"Representation generation failed for cleaned row(s): {bad}")
    if failures:
        raise ValueError(f"Representation generation failed for cleaned row(s): {failures}")

    model_config = config["model"]
    model_name = model_config["name"]
    estimator = _build_model(model_name, model_config.get("parameters", {}), seed)
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]
    if model_name in {"ridge", "elastic_net", "mlp"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    train_indices = split["train"]
    y = frame[target_column].to_numpy(dtype=float)
    start = time.perf_counter()
    pipeline.fit(features.iloc[train_indices], y[train_indices])
    training_seconds = time.perf_counter() - start

    prediction_frames: list[pd.DataFrame] = []
    metrics_by_partition: dict[str, dict[str, float | None]] = {}
    inference_seconds = 0.0
    for partition in ("train", "validation", "test"):
        indices = split[partition]
        prediction_start = time.perf_counter()
        predicted = pipeline.predict(features.iloc[indices])
        inference_seconds += time.perf_counter() - prediction_start
        metrics_by_partition[partition] = regression_metrics(y[indices], predicted)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": indices,
                    "source_row": frame.iloc[indices]["source_row"].to_numpy(),
                    "smiles": frame.iloc[indices][smiles_column].to_numpy(),
                    "canonical_smiles": frame.iloc[indices]["canonical_smiles"].to_numpy(),
                    "partition": partition,
                    "actual": y[indices],
                    "predicted": predicted,
                    "residual": y[indices] - predicted,
                }
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)

    timestamp = utc_timestamp()
    digest = config_hash(config)
    representation_name = config["representation"]["name"]
    experiment_id = (
        f"{config['experiment']['name']}_{representation_name}_{model_name}_"
        f"{config['split']['strategy']}_"
        f"s{seed}_{digest}_{uuid.uuid4().hex[:6]}"
    )
    raw_root = resolve_project_path(config.get("outputs", {}).get("raw_dir", "results/raw"))
    run_dir = raw_root / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.csv"
    model_path = run_dir / "model.joblib"
    feature_names_path = run_dir / "feature_names.json"
    config_path = run_dir / "config.yaml"
    predictions.to_csv(predictions_path, index=False)
    joblib.dump(pipeline, model_path)
    input_names = list(features.columns)
    selected_names = _feature_names(pipeline, input_names)
    write_json(
        {
            "representation": representation_name,
            "representation_parameters": representation_metadata,
            "input_features": input_names,
            "selected_features": selected_names,
            "representation_failure_rows": failures,
        },
        feature_names_path,
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    fitted_model = pipeline.named_steps["model"]
    importance_path: str | None = None
    if hasattr(fitted_model, "feature_importances_"):
        importance = pd.DataFrame(
            {"feature": selected_names, "importance": fitted_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance_file = run_dir / "feature_importance.csv"
        importance.to_csv(importance_file, index=False)
        importance_path = display_path(importance_file)
    elif hasattr(fitted_model, "coef_"):
        coefficients = np.asarray(fitted_model.coef_).reshape(-1)
        importance = pd.DataFrame(
            {
                "feature": selected_names,
                "coefficient": coefficients,
                "absolute_coefficient": np.abs(coefficients),
            }
        ).sort_values("absolute_coefficient", ascending=False)
        importance_file = run_dir / "feature_coefficients.csv"
        importance.to_csv(importance_file, index=False)
        importance_path = display_path(importance_file)

    figure_paths = save_prediction_plots(
        predictions,
        resolve_project_path(config.get("outputs", {}).get("figures_dir", "results/figures")),
        experiment_id,
    )
    test_metrics = metrics_by_partition["test"]
    record = {
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp,
        "experiment_name": config["experiment"]["name"],
        "representation": representation_name,
        "model": model_name,
        "split_strategy": config["split"]["strategy"],
        "seed": seed,
        "metrics": metrics_by_partition,
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "r2": test_metrics["r2"],
        "validation_mae": metrics_by_partition["validation"]["mae"],
        "training_time_seconds": training_seconds,
        "inference_time_seconds": inference_seconds,
        "n_trainable_parameters": None,
        "split_id": split_path.stem,
        "split_path": display_path(split_path),
        "config_hash": digest,
        "git_commit": git_commit(PROJECT_ROOT),
        "software_versions": software_versions(),
        "prediction_file": display_path(predictions_path),
        "model_file": display_path(model_path),
        "feature_names_file": display_path(feature_names_path),
        "feature_importance_file": importance_path,
        "figure_files": [display_path(path) for path in figure_paths],
    }
    write_json(record, run_dir / "run.json")
    benchmark_path = resolve_project_path(
        config.get("outputs", {}).get("benchmark_file", "results/benchmark_results.csv")
    )
    flat_row = {key: record.get(key) for key in (
        "experiment_id", "timestamp_utc", "representation", "model", "split_strategy",
        "seed", "mae", "rmse", "r2", "validation_mae", "training_time_seconds",
        "inference_time_seconds", "split_id", "config_hash", "git_commit",
        "prediction_file", "model_file", "n_trainable_parameters", "checkpoint_path",
        "epochs_trained",
    )}
    append_benchmark_row(flat_row, benchmark_path)
    logged_r2 = test_metrics["r2"] if test_metrics["r2"] is not None else float("nan")
    LOGGER.info(
        "Completed %s: test MAE=%.4f RMSE=%.4f R2=%.4f",
        experiment_id,
        test_metrics["mae"],
        test_metrics["rmse"],
        logged_r2,
    )
    return record


def with_overrides(
    config: dict[str, Any], *, model: str | None = None, seed: int | None = None,
    split_strategy: str | None = None, representation: str | None = None
) -> dict[str, Any]:
    updated = deepcopy(config)
    if model is not None:
        parameter_sets = updated["model"].get("parameter_sets", {})
        updated["model"]["name"] = model
        updated["model"]["parameters"] = parameter_sets.get(model, {})
    if seed is not None:
        updated.setdefault("training", {})["seed"] = seed
        updated["split"]["seed"] = seed
    if split_strategy is not None:
        updated["split"]["strategy"] = split_strategy
    if representation is not None:
        parameter_sets = updated["representation"].get("parameter_sets", {})
        shared = {"name": representation, "parameter_sets": parameter_sets}
        shared.update(parameter_sets.get(representation, {}))
        updated["representation"] = shared
    return updated
