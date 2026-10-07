"""Tabular dataset I/O with schema checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, smiles_column: str, target_column: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the source CSV under data/raw/ "
            "or update data.input_path."
        )
    frame = pd.read_csv(path)
    missing = [name for name in (smiles_column, target_column) if name not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")
    return frame


def save_processed(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
