"""Configurable Morgan/ECFP fingerprint generation."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from polytg.data.validation import parse_smiles

LOGGER = logging.getLogger(__name__)


def compute_morgan_fingerprints(
    smiles: Sequence[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    include_chirality: bool = False,
    use_bond_types: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """Return binary Morgan fingerprints and failed input row positions."""
    if radius < 0:
        raise ValueError("Morgan fingerprint radius must be non-negative")
    if n_bits <= 0:
        raise ValueError("Morgan fingerprint n_bits must be positive")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
        useBondTypes=use_bond_types,
    )
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    failures: list[int] = []
    for index, value in enumerate(smiles):
        mol = parse_smiles(value)
        if mol is None:
            failures.append(index)
            continue
        fingerprint = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    if failures:
        LOGGER.warning("Fingerprint generation failed for %d molecule(s)", len(failures))
    columns = [f"ECFP_{index}" for index in range(n_bits)]
    return pd.DataFrame(matrix, columns=columns), failures

