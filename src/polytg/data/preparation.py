"""End-to-end data preparation with persisted audit artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from polytg.data.cleaning import clean_dataset
from polytg.data.loader import load_csv, save_processed
from polytg.data.splitting import load_split, make_split, save_split
from polytg.utils.paths import resolve_project_path

LOGGER = logging.getLogger(__name__)


def prepare_from_config(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, list[int]], Path]:
    data = config["data"]
    split_config = config["split"]
    smiles_column = data["smiles_column"]
    target_column = data["target_column"]
    raw_path = resolve_project_path(data["input_path"])
    processed_path = resolve_project_path(data.get("processed_path", "data/processed/cleaned.csv"))
    report_path = processed_path.with_name(f"{processed_path.stem}_cleaning_report.json")

    raw = load_csv(raw_path, smiles_column, target_column)
    cleaned, report = clean_dataset(
        raw,
        smiles_column=smiles_column,
        target_column=target_column,
        aggregation=data.get("aggregate_repeats", "none"),
    )
    if cleaned.empty:
        raise ValueError("No valid rows remain after cleaning")
    save_processed(cleaned, processed_path)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    fractions = {
        "train": float(split_config.get("train_fraction", 0.70)),
        "validation": float(split_config.get("validation_fraction", 0.15)),
        "test": float(split_config.get("test_fraction", 0.15)),
    }
    strategy = split_config.get("strategy", "random")
    seed = int(split_config.get("seed", 42))
    split_name = f"{processed_path.stem}_{strategy}_seed{seed}.json"
    split_path = processed_path.parent / "splits" / split_name

    if split_path.exists():
        try:
            split = load_split(
                split_path,
                cleaned,
                smiles_column,
                target_column,
                expected_strategy=strategy,
                expected_seed=seed,
                expected_fractions=fractions,
            )
            LOGGER.info("Reusing verified split manifest: %s", split_path)
            return cleaned, split, split_path
        except ValueError:
            LOGGER.warning("Existing split is stale; regenerating %s", split_path)

    split = make_split(
        cleaned,
        smiles_column,
        strategy=strategy,
        train_fraction=fractions["train"],
        validation_fraction=fractions["validation"],
        test_fraction=fractions["test"],
        seed=seed,
    )
    save_split(
        split,
        split_path,
        frame=cleaned,
        smiles_column=smiles_column,
        target_column=target_column,
        strategy=strategy,
        seed=seed,
        fractions=fractions,
    )
    LOGGER.info("Prepared %d rows and saved split manifest to %s", len(cleaned), split_path)
    return cleaned, split, split_path
