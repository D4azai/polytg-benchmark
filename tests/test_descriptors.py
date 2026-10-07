import numpy as np

from polytg.representations.rdkit_descriptors import (
    DEFAULT_DESCRIPTOR_NAMES,
    compute_descriptors,
)


def test_descriptor_generation_is_numeric_and_named() -> None:
    features, failures = compute_descriptors(["CCO", "c1ccccc1"])
    assert features.shape == (2, len(DEFAULT_DESCRIPTOR_NAMES))
    assert list(features.columns) == list(DEFAULT_DESCRIPTOR_NAMES)
    assert np.isfinite(features.to_numpy()).all()
    assert failures == []


def test_invalid_smiles_is_reported() -> None:
    features, failures = compute_descriptors(["invalid"])
    assert failures == [0]
    assert features.isna().all(axis=None)

