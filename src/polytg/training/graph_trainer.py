"""Leakage-safe training and evaluation for molecular graph regressors."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from polytg.data.preparation import prepare_from_config
from polytg.evaluation.metrics import regression_metrics
from polytg.evaluation.plots import save_learning_curve, save_prediction_plots
from polytg.evaluation.reporting import (
    append_benchmark_row,
    git_commit,
    software_versions,
    utc_timestamp,
    write_json,
)
from polytg.models.gcn import GCNRegressor
from polytg.models.gin import GINRegressor
from polytg.representations.graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, molecules_to_graphs
from polytg.training.callbacks import EarlyStopping
from polytg.training.seeds import set_global_seed
from polytg.utils.config import config_hash
from polytg.utils.paths import PROJECT_ROOT, display_path, resolve_project_path

LOGGER = logging.getLogger(__name__)


def _set_torch_seed(seed: int, deterministic: bool) -> None:
    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda was requested but CUDA is unavailable")
    return torch.device(name)


def _model(config: dict[str, Any]) -> nn.Module:
    name = config["model"]["name"]
    parameters = config["model"].get("parameters", {})
    model_type = GINRegressor if name == "gin" else GCNRegressor if name == "gcn" else None
    if model_type is None:
        raise ValueError("Graph experiments require model.name=gin or model.name=gcn")
    return model_type(node_dim=NODE_FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM, **parameters)


def _loader(graphs, indices: list[int], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        [graphs[index] for index in indices],
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


@torch.no_grad()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices: list[int] = []
    actual: list[float] = []
    predicted: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        predicted.extend((output * target_scale + target_mean).cpu().tolist())
        actual.extend(batch.y.view(-1).cpu().tolist())
        indices.extend(batch.sample_id.view(-1).cpu().tolist())
    return np.asarray(indices), np.asarray(actual), np.asarray(predicted)


def run_graph_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if config["representation"].get("name") != "molecular_graph":
        raise ValueError("Graph training requires representation.name=molecular_graph")
    training = config.get("training", {})
    seed = int(training.get("seed", config["split"].get("seed", 42)))
    deterministic = bool(training.get("deterministic", True))
    _set_torch_seed(seed, deterministic)
    device = _device(training.get("device", "auto"))
    frame, split, split_path = prepare_from_config(config)
    smiles_column = config["data"]["smiles_column"]
    target_column = config["data"]["target_column"]
    targets = frame[target_column].to_numpy(dtype=float)
    graphs, failures = molecules_to_graphs(frame[smiles_column].astype(str).tolist(), targets)
    if failures or len(graphs) != len(frame):
        raise ValueError(f"Graph conversion failed for cleaned row(s): {failures}")

    batch_size = int(training.get("batch_size", 32))
    loaders = {
        partition: _loader(graphs, indices, batch_size, partition == "train", seed)
        for partition, indices in split.items()
    }
    train_targets = targets[split["train"]]
    target_mean = float(train_targets.mean())
    target_scale = float(train_targets.std(ddof=0))
    if target_scale == 0.0:
        target_scale = 1.0

    digest = config_hash(config)
    model_name = config["model"]["name"]
    experiment_id = (
        f"{config['experiment']['name']}_molecular_graph_{model_name}_"
        f"{config['split']['strategy']}_s{seed}_{digest}_{uuid.uuid4().hex[:6]}"
    )
    raw_dir = resolve_project_path(
        config.get("outputs", {}).get("raw_dir", "results/raw")
    )
    run_dir = raw_dir / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "checkpoint.pt"
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    model = _model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 10)),
    )
    stopping = EarlyStopping(
        patience=int(training.get("early_stopping_patience", 30)),
        min_delta=float(training.get("min_delta", 0.0)),
    )
    loss_function = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    start = time.perf_counter()
    for epoch in range(1, int(training.get("max_epochs", 300)) + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            normalized_target = (batch.y.view(-1) - target_mean) / target_scale
            loss = loss_function(output, normalized_target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
            total_examples += batch.num_graphs
        _, validation_actual, validation_predicted = _predict(
            model, loaders["validation"], device, target_mean, target_scale
        )
        validation_mae = float(np.mean(np.abs(validation_actual - validation_predicted)))
        train_loss = total_loss / total_examples
        scheduler.step(validation_mae)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_mae": validation_mae}
        )
        improved = validation_mae < stopping.best_value - stopping.min_delta
        should_stop = stopping.update(validation_mae)
        if improved:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state_dict": best_state,
                    "target_mean": target_mean,
                    "target_scale": target_scale,
                    "node_feature_dim": NODE_FEATURE_DIM,
                    "edge_feature_dim": EDGE_FEATURE_DIM,
                    "config_hash": digest,
                    "epoch": epoch,
                    "validation_mae": validation_mae,
                },
                checkpoint_path,
            )
        if should_stop:
            break
    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Graph training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)

    prediction_frames: list[pd.DataFrame] = []
    metrics_by_partition: dict[str, dict[str, float | None]] = {}
    inference_seconds = 0.0
    for partition, loader in loaders.items():
        prediction_start = time.perf_counter()
        indices, actual, predicted = _predict(model, loader, device, target_mean, target_scale)
        inference_seconds += time.perf_counter() - prediction_start
        metrics_by_partition[partition] = regression_metrics(actual, predicted)
        integer_indices = indices.astype(int)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": integer_indices,
                    "source_row": frame.iloc[integer_indices]["source_row"].to_numpy(),
                    "smiles": frame.iloc[integer_indices][smiles_column].to_numpy(),
                    "canonical_smiles": frame.iloc[integer_indices]["canonical_smiles"].to_numpy(),
                    "partition": partition,
                    "actual": actual,
                    "predicted": predicted,
                    "residual": actual - predicted,
                }
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history_frame = pd.DataFrame(history)

    predictions_path = run_dir / "predictions.csv"
    history_path = run_dir / "training_history.csv"
    predictions.to_csv(predictions_path, index=False)
    history_frame.to_csv(history_path, index=False)
    figures_dir = resolve_project_path(
        config.get("outputs", {}).get("figures_dir", "results/figures")
    )
    figure_paths = save_prediction_plots(predictions, figures_dir, experiment_id)
    figure_paths.append(
        save_learning_curve(history_frame, figures_dir / f"{experiment_id}_learning_curve.png")
    )
    test_metrics = metrics_by_partition["test"]
    versions = software_versions()
    versions.update(
        {
            "torch": torch.__version__,
            "torch_geometric": __import__("torch_geometric").__version__,
        }
    )
    record = {
        "experiment_id": experiment_id,
        "timestamp_utc": utc_timestamp(),
        "experiment_name": config["experiment"]["name"],
        "representation": "molecular_graph",
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
        "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs_trained": len(history),
        "best_epoch": int(history_frame["validation_mae"].idxmin() + 1),
        "device": str(device),
        "split_id": split_path.stem,
        "split_path": display_path(split_path),
        "config_hash": digest,
        "git_commit": git_commit(PROJECT_ROOT),
        "software_versions": versions,
        "prediction_file": display_path(predictions_path),
        "model_file": display_path(checkpoint_path),
        "checkpoint_path": display_path(checkpoint_path),
        "training_history_file": display_path(history_path),
        "figure_files": [display_path(path) for path in figure_paths],
    }
    write_json(record, run_dir / "run.json")
    benchmark_path = resolve_project_path(
        config.get("outputs", {}).get("benchmark_file", "results/benchmark_results.csv")
    )
    keys = (
        "experiment_id", "timestamp_utc", "representation", "model", "split_strategy",
        "seed", "mae", "rmse", "r2", "validation_mae", "training_time_seconds",
        "inference_time_seconds", "split_id", "config_hash", "git_commit",
        "prediction_file", "model_file", "n_trainable_parameters", "checkpoint_path",
        "epochs_trained",
    )
    append_benchmark_row({key: record.get(key) for key in keys}, benchmark_path)
    LOGGER.info("Completed graph experiment %s on %s", experiment_id, device)
    return record
