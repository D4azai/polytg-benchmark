"""Prepare and split a PolyTg-Bench dataset."""

from __future__ import annotations

import argparse

from polytg.data.preparation import prepare_from_config
from polytg.utils.config import load_config
from polytg.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    frame, split, split_path = prepare_from_config(load_config(args.config))
    sizes = ", ".join(f"{key}={len(values)}" for key, values in split.items())
    print(f"Prepared {len(frame)} rows; split sizes: {sizes}")
    print(f"Split manifest: {split_path}")


if __name__ == "__main__":
    main()
