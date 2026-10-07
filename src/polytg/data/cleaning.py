"""Auditable cleaning for molecular regression datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from polytg.data.validation import canonicalize_smiles

Aggregation = Literal["none", "mean", "median"]


@dataclass(frozen=True)
class CleaningReport:
    input_rows: int
    missing_target_rows: int
    non_numeric_target_rows: int
    invalid_smiles_rows: int
    exact_duplicate_rows_removed: int
    repeated_structure_count: int
    conflicting_structure_count: int
    aggregated_rows_removed: int
    output_rows: int
    invalid_smiles_examples: list[str]
    conflicting_structures: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def clean_dataset(
    frame: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    aggregation: Aggregation = "none",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a dataset without mutating the source frame."""
    if aggregation not in {"none", "mean", "median"}:
        raise ValueError("aggregation must be one of: none, mean, median")
    reserved = {"source_row", "canonical_smiles"}.intersection(frame.columns)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"Input uses reserved provenance column(s): {names}")

    work = frame.copy(deep=True)
    work["source_row"] = np.arange(len(work), dtype=int)
    missing_target = int(work[target_column].isna().sum())
    work = work.loc[work[target_column].notna()].copy()
    numeric_target = pd.to_numeric(work[target_column], errors="coerce")
    non_numeric_target = int(numeric_target.isna().sum())
    work[target_column] = numeric_target
    work = work.loc[work[target_column].notna()].copy()

    work["canonical_smiles"] = work[smiles_column].map(canonicalize_smiles)
    invalid_mask = work["canonical_smiles"].isna()
    invalid_examples = (
        work.loc[invalid_mask, smiles_column].astype(str).drop_duplicates().head(20).tolist()
    )
    invalid_count = int(invalid_mask.sum())
    work = work.loc[~invalid_mask].copy()

    before_exact = len(work)
    comparison_columns = [column for column in frame.columns if column in work.columns]
    work = work.drop_duplicates(subset=comparison_columns, keep="first")
    exact_removed = before_exact - len(work)

    grouped = work.groupby("canonical_smiles", sort=True, dropna=False)[target_column]
    counts = grouped.size()
    repeated = counts[counts > 1]
    conflicts: list[dict[str, object]] = []
    for structure in repeated.index:
        values = sorted(float(value) for value in grouped.get_group(structure).unique())
        if len(values) > 1:
            conflicts.append(
                {
                    "canonical_smiles": structure,
                    "n_measurements": int(counts.loc[structure]),
                    "target_values": values,
                    "target_range": float(max(values) - min(values)),
                }
            )

    aggregated_removed = 0
    if aggregation != "none":
        before_aggregation = len(work)
        reducer = "mean" if aggregation == "mean" else "median"
        targets = grouped.agg(reducer).rename(target_column)
        first_rows = work.sort_values("source_row").drop_duplicates("canonical_smiles")
        first_rows = first_rows.drop(columns=[target_column]).set_index("canonical_smiles")
        work = first_rows.join(targets, how="inner").reset_index()
        aggregated_removed = before_aggregation - len(work)

    work = work.sort_values("source_row").reset_index(drop=True)
    report = CleaningReport(
        input_rows=len(frame),
        missing_target_rows=missing_target,
        non_numeric_target_rows=non_numeric_target,
        invalid_smiles_rows=invalid_count,
        exact_duplicate_rows_removed=exact_removed,
        repeated_structure_count=len(repeated),
        conflicting_structure_count=len(conflicts),
        aggregated_rows_removed=aggregated_removed,
        output_rows=len(work),
        invalid_smiles_examples=invalid_examples,
        conflicting_structures=conflicts,
    )
    return work, report
