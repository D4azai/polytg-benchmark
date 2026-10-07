import numpy as np
import pytest

from polytg.representations.language_model import _pool_hidden_state, embedding_cache_key


def test_embedding_cache_key_changes_with_scientific_inputs() -> None:
    base = embedding_cache_key(["CCO"], "model", "revision", "mean", 128)
    assert base == embedding_cache_key(["CCO"], "model", "revision", "mean", 128)
    assert base != embedding_cache_key(["CCN"], "model", "revision", "mean", 128)
    assert base != embedding_cache_key(["CCO"], "model", "revision", "cls", 128)


def test_masked_mean_pooling_ignores_padding() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = _pool_hidden_state(hidden, mask, "mean")
    assert np.allclose(pooled.numpy(), [[2.0, 4.0]])
