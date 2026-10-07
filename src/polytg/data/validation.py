"""Chemical structure validation helpers."""

from __future__ import annotations

from rdkit import Chem


def parse_smiles(smiles: object) -> Chem.Mol | None:
    """Parse a SMILES string, returning ``None`` for missing or invalid values."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def is_valid_smiles(smiles: object) -> bool:
    return parse_smiles(smiles) is not None


def canonicalize_smiles(smiles: object) -> str | None:
    mol = parse_smiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None

