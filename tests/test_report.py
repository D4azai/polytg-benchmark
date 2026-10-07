from pathlib import Path

import pandas as pd

from scripts.generate_report import generate_report


def test_report_is_derived_from_saved_results_and_predictions(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    base = pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "source_row": [0, 1, 2],
            "smiles": ["CCO", "CCN", "CCC"],
            "partition": ["test", "test", "test"],
            "actual": [100.0, 110.0, 120.0],
        }
    )
    first = base.assign(predicted=[101.0, 109.0, 121.0], residual=[-1.0, 1.0, -1.0])
    second = base.assign(predicted=[104.0, 106.0, 124.0], residual=[-4.0, 4.0, -4.0])
    first.to_csv(prediction_dir / "first.csv", index=False)
    second.to_csv(prediction_dir / "second.csv", index=False)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_path = results_dir / "benchmark_results.csv"
    common = {
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "split_strategy": "random",
        "seed": 42,
        "rmse": 2.0,
        "r2": 0.8,
        "training_time_seconds": 1.0,
        "split_id": "shared",
        "config_hash": "hash",
    }
    pd.DataFrame(
        [
            {
                **common,
                "experiment_id": "first",
                "representation": "rdkit_descriptors",
                "model": "ridge",
                "mae": 1.0,
                "prediction_file": str(prediction_dir / "first.csv"),
            },
            {
                **common,
                "experiment_id": "second",
                "representation": "ecfp",
                "model": "ridge",
                "mae": 4.0,
                "prediction_file": str(prediction_dir / "second.csv"),
            },
        ]
    ).to_csv(results_path, index=False)
    report_path = results_dir / "REPORT.md"
    tables = results_dir / "tables"
    figures = results_dir / "figures"
    generate_report(results_path, report_path, tables, figures, n_bootstrap=100)
    report = report_path.read_text(encoding="utf-8")
    assert "Aggregate test performance" in report
    assert "no completed matched summary for molecular_graph" in report
    assert (tables / "benchmark_summary.csv").is_file()
    assert (tables / "pairwise_comparisons.csv").is_file()
    assert (figures / "model_comparison_r2.png").is_file()
