"""Frozen Hugging Face SMILES embeddings with verified disk caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)
DEFAULT_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
DEFAULT_CHEMBERTA_REVISION = "761d6a18cf99db371e0b43baf3e2d21b3e865a20"


def embedding_cache_key(
    smiles: list[str],
    model_name: str,
    revision: str,
    pooling: str,
    max_length: int,
    device: str = "cpu",
) -> str:
    payload = json.dumps(
        {
            "smiles": smiles,
            "model_name": model_name,
            "revision": revision,
            "pooling": pooling,
            "max_length": max_length,
            "device": device,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pool_hidden_state(
    hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str
) -> torch.Tensor:
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    raise ValueError("pooling must be 'mean' or 'cls'")


def compute_or_load_embeddings(
    smiles: list[str],
    *,
    cache_dir: str | Path,
    model_name: str,
    revision: str = DEFAULT_CHEMBERTA_REVISION,
    pooling: str = "mean",
    max_length: int = 256,
    batch_size: int = 32,
    device: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute frozen encoder embeddings, or load an exactly matching cache."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Language-model dependencies are missing. Install with `pip install -e .[language]`."
        ) from error
    if not smiles:
        raise ValueError("At least one SMILES string is required for embedding extraction")
    if max_length <= 0 or batch_size <= 0:
        raise ValueError("max_length and batch_size must be positive")
    if pooling not in {"mean", "cls"}:
        raise ValueError("pooling must be 'mean' or 'cls'")
    selected_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for embeddings but is unavailable")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = embedding_cache_key(
        smiles, model_name, revision, pooling, max_length, str(selected_device)
    )
    array_path = cache_dir / f"{key}.npz"
    metadata_path = cache_dir / f"{key}.json"
    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(array_path, allow_pickle=False) as payload:
            embeddings = payload["embeddings"]
        if metadata.get("cache_key") != key or embeddings.shape[0] != len(smiles):
            raise ValueError(f"Embedding cache is inconsistent: {array_path}")
        metadata["resolved_revision"] = (
            metadata.get("resolved_revision") or metadata.get("requested_revision")
        )
        metadata["cache_hit"] = True
        columns = [f"LM_{index}" for index in range(embeddings.shape[1])]
        return pd.DataFrame(embeddings, columns=columns), metadata

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Language-model dependencies are missing. Install with `pip install -e .[language]`."
        ) from error

    extraction_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(selected_device)
    model.eval()
    lengths = [len(tokenizer.encode(value, add_special_tokens=True)) for value in smiles]
    truncated_count = sum(length > max_length for length in lengths)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            tokens = tokenizer(
                smiles[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(selected_device) for name, tensor in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            pooled = _pool_hidden_state(hidden, tokens["attention_mask"], pooling)
            batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    embeddings = np.concatenate(batches, axis=0)
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    metadata = {
        "cache_key": key,
        "cache_hit": False,
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "pooling": pooling,
        "max_length": max_length,
        "batch_size": batch_size,
        "embedding_dimension": int(embeddings.shape[1]),
        "n_smiles": len(smiles),
        "truncated_smiles_count": truncated_count,
        "device": str(selected_device),
        "extraction_time_seconds": time.perf_counter() - extraction_start,
    }
    np.savez_compressed(array_path, embeddings=embeddings)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    columns = [f"LM_{index}" for index in range(embeddings.shape[1])]
    LOGGER.info("Cached %d embeddings at %s", len(smiles), array_path)
    return pd.DataFrame(embeddings, columns=columns), metadata
