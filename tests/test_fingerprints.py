import numpy as np
import pytest

from polytg.representations.fingerprints import compute_morgan_fingerprints


def test_morgan_fingerprint_shape_type_and_determinism() -> None:
    first, failures = compute_morgan_fingerprints(["CCO", "c1ccccc1"], radius=2, n_bits=128)
    second, _ = compute_morgan_fingerprints(["CCO", "c1ccccc1"], radius=2, n_bits=128)
    assert first.shape == (2, 128)
    assert failures == []
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    assert set(np.unique(first.to_numpy())).issubset({0, 1})
    assert list(first.columns) == [f"ECFP_{index}" for index in range(128)]


def test_morgan_fingerprint_reports_invalid_smiles() -> None:
    features, failures = compute_morgan_fingerprints(["invalid"], n_bits=32)
    assert failures == [0]
    assert features.to_numpy().sum() == 0


@pytest.mark.parametrize("kwargs", [{"radius": -1}, {"n_bits": 0}])
def test_morgan_parameters_are_validated(kwargs) -> None:
    with pytest.raises(ValueError):
        compute_morgan_fingerprints(["CCO"], **kwargs)
