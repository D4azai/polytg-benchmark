"""Run the controlled molecular-representation benchmark suite."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from generate_report import generate_report

from polytg.training.trainer import run_experiment, with_overrides
from polytg.utils.config import load_config
from polytg.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def run_matrix(base: dict) -> int:
    matrix = base.get("benchmark", {})
    models = matrix.get("models", [base["model"]["name"]])
    representations = matrix.get("representations", [base["representation"]["name"]])
    seeds = matrix.get("seeds", [base.get("training", {}).get("seed", 42)])
    strategies = matrix.get("split_strategies", [base["split"]["strategy"]])
    total = len(models) * len(representations) * len(seeds) * len(strategies)
    LOGGER.info("Running %d controlled experiment(s)", total)
    for strategy in strategies:
        for seed in seeds:
            for representation in representations:
                for model in models:
                    config = with_overrides(
                        base,
                        model=model,
                        seed=int(seed),
                        split_strategy=strategy,
                        representation=representation,
                    )
                    run_experiment(config)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    base = load_config(args.config)
    run_matrix(base)
    for additional_path in base.get("benchmark", {}).get("additional_configs", []):
        LOGGER.info("Continuing benchmark suite with %s", additional_path)
        run_matrix(load_config(additional_path))
    outputs = base.get("outputs", {})
    benchmark_path = Path(outputs.get("benchmark_file", "results/benchmark_results.csv"))
    generate_report(
        results_path=benchmark_path,
        output_path=Path("results/REPORT.md"),
        tables_dir=Path("results/tables"),
        figures_dir=Path(outputs.get("figures_dir", "results/figures")),
        n_bootstrap=10_000,
    )
    LOGGER.info("Generated results/REPORT.md from completed experiment artifacts")


if __name__ == "__main__":
    main()
