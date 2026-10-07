from polytg.data.validation import canonicalize_smiles, is_valid_smiles


def test_smiles_validation_and_canonicalization() -> None:
    assert is_valid_smiles("CCO")
    assert not is_valid_smiles("not-a-smiles")
    assert not is_valid_smiles(None)
    assert canonicalize_smiles("C(C)O") == canonicalize_smiles("CCO")

