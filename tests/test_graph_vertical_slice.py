from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from polytg.training.trainer import run_experiment  # noqa: E402


def test_gin_vertical_slice_persists_checkpoint_and_predictions(tmp_path: Path) -> None:
    smiles = [
        "CCO", "CCN", "CCC", "CCCC", "CCCl", "CCBr", "CC(=O)O", "CC#N",
        "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "c1ccncc1", "C1CCCCC1",
        "C1CCNCC1", "COC", "CCS", "CCF", "CNC", "CO", "CN",
    ]
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {"smiles": smiles, "tg": [100.0 + 3 * index for index in range(len(smiles))]}
    ).to_csv(raw_path, index=False)
    config = {
        "experiment": {"name": "test_graph_slice"},
        "data": {
            "input_path": str(raw_path),
            "processed_path": str(tmp_path / "clean.csv"),
            "smiles_column": "smiles",
            "target_column": "tg",
            "aggregate_repeats": "none",
        },
        "split": {
            "strategy": "random", "seed": 7, "train_fraction": 0.7,
            "validation_fraction": 0.15, "test_fraction": 0.15,
        },
        "representation": {"name": "molecular_graph"},
        "model": {
            "name": "gin",
            "parameters": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0},
        },
        "training": {
            "seed": 7,
            "deterministic": True,
            "device": "cpu",
            "batch_size": 8,
            "max_epochs": 3,
            "learning_rate": 0.001,
            "early_stopping_patience": 2,
            "lr_patience": 1,
        },
        "outputs": {
            "raw_dir": str(tmp_path / "runs"),
            "figures_dir": str(tmp_path / "figures"),
            "benchmark_file": str(tmp_path / "benchmark.csv"),
        },
    }
    record = run_experiment(config)
    run_dir = tmp_path / "runs" / record["experiment_id"]
    assert record["n_trainable_parameters"] > 0
    assert 1 <= record["epochs_trained"] <= 3
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "training_history.csv").is_file()
    assert (run_dir / "predictions.csv").is_file()
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == len(smiles)
    assert set(predictions["partition"]) == {"train", "validation", "test"}
