"""Repository-relative path handling."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def display_path(path: str | Path) -> str:
    """Prefer repository-relative artifact paths while supporting temporary output roots."""
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)
