"""Recompute metrics from a saved prediction file without retraining."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from polytg.evaluation.metrics import regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)
    required = {"partition", "actual", "predicted"}
    missing = required - set(predictions.columns)
    if missing:
        parser.error(f"Prediction file is missing column(s): {', '.join(sorted(missing))}")
    output = {
        partition: regression_metrics(group["actual"].to_numpy(), group["predicted"].to_numpy())
        for partition, group in predictions.groupby("partition", sort=False)
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

