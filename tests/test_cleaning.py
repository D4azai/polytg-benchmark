import pandas as pd

from polytg.data.cleaning import clean_dataset


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": ["CCO", "CCO", "C(C)O", "CCN", "bad", "CCC", "CCCl"],
            "tg": [100.0, 100.0, 120.0, None, 90.0, "unknown", 130.0],
            "source": ["a", "a", "b", "c", "d", "e", "f"],
        }
    )


def test_cleaning_reports_failures_duplicates_and_conflicts() -> None:
    raw = source_frame()
    cleaned, report = clean_dataset(raw, "smiles", "tg", aggregation="none")
    assert len(raw) == 7  # source is not mutated
    assert report.missing_target_rows == 1
    assert report.non_numeric_target_rows == 1
    assert report.invalid_smiles_rows == 1
    assert report.exact_duplicate_rows_removed == 1
    assert report.repeated_structure_count == 1
    assert report.conflicting_structure_count == 1
    assert len(cleaned) == 3
    assert cleaned["source_row"].tolist() == [0, 2, 6]


def test_mean_aggregation_uses_canonical_structure() -> None:
    cleaned, report = clean_dataset(source_frame(), "smiles", "tg", aggregation="mean")
    ethanol = cleaned.loc[cleaned["canonical_smiles"] == "CCO", "tg"]
    assert ethanol.item() == 110.0
    assert report.aggregated_rows_removed == 1

