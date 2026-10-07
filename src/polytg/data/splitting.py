"""Leakage-aware and reproducible dataset splitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from polytg.data.validation import parse_smiles

SplitStrategy = Literal["random", "scaffold"]


def dataset_fingerprint(frame: pd.DataFrame, smiles_column: str, target_column: str) -> str:
    payload = frame[[smiles_column, target_column]].to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_fractions(train: float, validation: float, test: float) -> None:
    fractions = np.asarray([train, validation, test], dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must be positive and sum to 1.0")


def _structure_groups(frame: pd.DataFrame, smiles_column: str) -> list[list[int]]:
    key_column = "canonical_smiles" if "canonical_smiles" in frame.columns else smiles_column
    grouped_indices = frame.groupby(key_column, sort=True).indices.values()
    return [list(map(int, values)) for values in grouped_indices]


def _scaffold_key(smiles: str) -> str:
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES reached split generation: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    return scaffold or f"ACYCLIC::{smiles}"


def _assign_groups(
    groups: list[list[int]], n_rows: int, fractions: tuple[float, float, float], seed: int
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    decorated = [(group, float(rng.random())) for group in groups]
    decorated.sort(key=lambda item: (-len(item[0]), item[1]))
    names = ("train", "validation", "test")
    targets = np.asarray(fractions) * n_rows
    assigned = {name: [] for name in names}
    sizes = np.zeros(3, dtype=int)
    for group, _ in decorated:
        deficits = targets - sizes
        candidates = np.flatnonzero(deficits == deficits.max())
        destination = int(rng.choice(candidates))
        assigned[names[destination]].extend(group)
        sizes[destination] += len(group)
    for values in assigned.values():
        values.sort()
    return assigned


def make_split(
    frame: pd.DataFrame,
    smiles_column: str,
    strategy: SplitStrategy = "random",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    if len(frame) < 3:
        raise ValueError("At least three cleaned rows are required to create a split")
    if strategy == "random":
        groups = _structure_groups(frame, smiles_column)
    elif strategy == "scaffold":
        scaffold_groups: dict[str, list[int]] = {}
        for index, smiles in enumerate(frame[smiles_column].astype(str)):
            scaffold_groups.setdefault(_scaffold_key(smiles), []).append(index)
        groups = list(scaffold_groups.values())
    else:
        raise ValueError("strategy must be 'random' or 'scaffold'")
    split = _assign_groups(
        groups, len(frame), (train_fraction, validation_fraction, test_fraction), seed
    )
    if any(len(split[name]) == 0 for name in split):
        raise ValueError(
            "The grouped split produced an empty partition. Add more distinct structures/scaffolds "
            "or adjust the fractions."
        )
    return split


def save_split(
    split: dict[str, list[int]],
    path: str | Path,
    *,
    frame: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    strategy: str,
    seed: int,
    fractions: dict[str, float],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset_sha256": dataset_fingerprint(frame, smiles_column, target_column),
        "n_rows": len(frame),
        "strategy": strategy,
        "seed": seed,
        "fractions": fractions,
        "indices": split,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_split(
    path: str | Path,
    frame: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    *,
    expected_strategy: str | None = None,
    expected_seed: int | None = None,
    expected_fractions: dict[str, float] | None = None,
) -> dict[str, list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = dataset_fingerprint(frame, smiles_column, target_column)
    if payload.get("dataset_sha256") != actual or payload.get("n_rows") != len(frame):
        raise ValueError("Split manifest does not match the cleaned dataset")
    if expected_strategy is not None and payload.get("strategy") != expected_strategy:
        raise ValueError("Split manifest strategy does not match the configuration")
    if expected_seed is not None and payload.get("seed") != expected_seed:
        raise ValueError("Split manifest seed does not match the configuration")
    if expected_fractions is not None:
        stored_fractions = payload.get("fractions", {})
        if any(
            key not in stored_fractions
            or not np.isclose(float(stored_fractions[key]), float(expected_fractions[key]))
            for key in expected_fractions
        ):
            raise ValueError("Split manifest fractions do not match the configuration")
    indices = payload["indices"]
    flattened = [index for name in ("train", "validation", "test") for index in indices[name]]
    if sorted(flattened) != list(range(len(frame))) or len(set(flattened)) != len(frame):
        raise ValueError("Split manifest indices are incomplete or overlapping")
    return {name: list(map(int, indices[name])) for name in ("train", "validation", "test")}
