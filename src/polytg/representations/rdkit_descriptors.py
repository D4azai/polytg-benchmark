"""Stable, interpretable RDKit 2D descriptor extraction."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors

from polytg.data.validation import parse_smiles

LOGGER = logging.getLogger(__name__)

DEFAULT_DESCRIPTOR_NAMES = (
    "MolWt", "HeavyAtomMolWt", "ExactMolWt", "MolLogP", "MolMR", "TPSA",
    "NumValenceElectrons", "NumRadicalElectrons", "FractionCSP3", "HeavyAtomCount",
    "NHOHCount", "NOCount", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "NumAromaticHeterocycles", "NumAromaticCarbocycles",
    "NumAliphaticHeterocycles", "NumAliphaticCarbocycles", "NumSaturatedHeterocycles",
    "NumSaturatedCarbocycles", "BalabanJ", "BertzCT", "HallKierAlpha", "Kappa1",
    "Kappa2", "Kappa3", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "LabuteASA",
    "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex", "qed",
)


def descriptor_functions(names: Sequence[str]) -> list[tuple[str, object]]:
    available = dict(Descriptors._descList)
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError(f"Installed RDKit lacks required descriptor(s): {', '.join(missing)}")
    return [(name, available[name]) for name in names]


def compute_descriptors(
    smiles: Sequence[str], names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES
) -> tuple[pd.DataFrame, list[int]]:
    """Return descriptors and row indices that failed extraction."""
    functions = descriptor_functions(names)
    rows: list[list[float]] = []
    failures: list[int] = []
    for index, value in enumerate(smiles):
        mol = parse_smiles(value)
        if mol is None:
            failures.append(index)
            rows.append([np.nan] * len(functions))
            continue
        values: list[float] = []
        failed = False
        for _, function in functions:
            try:
                result = float(function(mol))
            except (ArithmeticError, RuntimeError, ValueError, TypeError):
                result = np.nan
                failed = True
            values.append(result if np.isfinite(result) else np.nan)
        if failed:
            failures.append(index)
        rows.append(values)
    if failures:
        LOGGER.warning("Descriptor extraction had failures for %d molecule(s)", len(failures))
    return pd.DataFrame(rows, columns=[name for name, _ in functions]), failures

