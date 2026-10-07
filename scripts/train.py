"""Train one configured PolyTg-Bench experiment."""

from __future__ import annotations

import argparse
import json

from polytg.training.trainer import run_experiment, with_overrides
from polytg.utils.config import load_config
from polytg.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        choices=("random_forest", "xgboost", "ridge", "elastic_net", "mlp", "gin", "gcn"),
    )
    parser.add_argument(
        "--representation",
        choices=("rdkit_descriptors", "ecfp", "chembert", "molecular_graph"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-strategy", choices=("random", "scaffold"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = with_overrides(
        load_config(args.config),
        model=args.model,
        seed=args.seed,
        split_strategy=args.split_strategy,
        representation=args.representation,
    )
    record = run_experiment(config)
    print(json.dumps(record["metrics"], indent=2))
    print(f"Run: {record['experiment_id']}")


if __name__ == "__main__":
    main()
