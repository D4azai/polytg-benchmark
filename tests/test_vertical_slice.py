from pathlib import Path

import pandas as pd

from polytg.training.trainer import run_experiment


def test_random_forest_vertical_slice_persists_artifacts(tmp_path: Path) -> None:
    smiles = [
        "CCO", "CCN", "CCC", "CCCC", "CCCl", "CCBr", "CC(=O)O", "CC#N",
        "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCNCC1",
        "COC", "CCS", "CCF", "CNC", "CO", "CN",
    ]
    raw_path = tmp_path / "raw.csv"
    data = {"smiles": smiles, "tg": [100.0 + 3 * i for i in range(len(smiles))]}
    pd.DataFrame(data).to_csv(raw_path, index=False)
    config = {
        "experiment": {"name": "test_slice"},
        "data": {
            "input_path": str(raw_path), "processed_path": str(tmp_path / "clean.csv"),
            "smiles_column": "smiles", "target_column": "tg", "aggregate_repeats": "none",
        },
        "split": {
            "strategy": "random", "seed": 7, "train_fraction": 0.7,
            "validation_fraction": 0.15, "test_fraction": 0.15,
        },
        "representation": {"name": "rdkit_descriptors"},
        "model": {"name": "random_forest", "parameters": {"n_estimators": 5, "n_jobs": 1}},
        "training": {"seed": 7},
        "outputs": {
            "raw_dir": str(tmp_path / "runs"), "figures_dir": str(tmp_path / "figures"),
            "benchmark_file": str(tmp_path / "benchmark.csv"),
        },
    }
    record = run_experiment(config)
    assert set(record["metrics"]) == {"train", "validation", "test"}
    run_dir = tmp_path / "runs" / record["experiment_id"]
    assert (run_dir / "model.joblib").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert (run_dir / "run.json").is_file()
    assert (tmp_path / "benchmark.csv").is_file()
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == len(smiles)
    assert set(predictions["partition"]) == {"train", "validation", "test"}
