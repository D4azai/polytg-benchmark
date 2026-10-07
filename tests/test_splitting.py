import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from polytg.data.splitting import load_split, make_split, save_split


def sample_frame() -> pd.DataFrame:
    smiles = [
        "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "c1ccncc1", "Cc1ccncc1",
        "C1CCCCC1", "CC1CCCCC1", "C1CCNCC1", "CCO", "CCN", "CCCC", "CCCl",
    ]
    return pd.DataFrame({"smiles": smiles, "canonical_smiles": smiles, "tg": range(len(smiles))})


def test_random_split_is_reproducible_and_complete() -> None:
    frame = sample_frame()
    first = make_split(frame, "smiles", strategy="random", seed=17)
    second = make_split(frame, "smiles", strategy="random", seed=17)
    assert first == second
    flattened = [index for values in first.values() for index in values]
    assert sorted(flattened) == list(range(len(frame)))
    assert len(flattened) == len(set(flattened))


def test_scaffolds_do_not_cross_partitions() -> None:
    frame = sample_frame()
    split = make_split(frame, "smiles", strategy="scaffold", seed=42)
    scaffold_to_partition: dict[str, str] = {}
    for partition, indices in split.items():
        for index in indices:
            smiles = frame.iloc[index]["smiles"]
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles))
            if scaffold:
                assert scaffold_to_partition.setdefault(scaffold, partition) == partition


def test_duplicate_structures_stay_together_in_random_split() -> None:
    frame = sample_frame()
    duplicate = frame.iloc[[0]].copy()
    frame = pd.concat([frame, duplicate], ignore_index=True)
    split = make_split(frame, "smiles", strategy="random", seed=42)
    containing = [
        name for name, indices in split.items() if 0 in indices or len(frame) - 1 in indices
    ]
    assert len(set(containing)) == 1
    assert {0, len(frame) - 1}.issubset(split[containing[0]])


def test_saved_split_rejects_changed_fractions(tmp_path) -> None:
    frame = sample_frame()
    split = make_split(frame, "smiles", strategy="random", seed=42)
    path = tmp_path / "split.json"
    fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    save_split(
        split,
        path,
        frame=frame,
        smiles_column="smiles",
        target_column="tg",
        strategy="random",
        seed=42,
        fractions=fractions,
    )
    changed = {"train": 0.60, "validation": 0.20, "test": 0.20}
    with pytest.raises(ValueError, match="fractions"):
        load_split(
            path,
            frame,
            "smiles",
            "tg",
            expected_strategy="random",
            expected_seed=42,
            expected_fractions=changed,
        )
