"""Generate tables, statistical comparisons, figures, and a factual Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from polytg.evaluation.plots import save_benchmark_comparison_plots
from polytg.evaluation.statistics import pairwise_comparisons, summarize_results


def _latest_unique_runs(results: pd.DataFrame) -> pd.DataFrame:
    identity = ["representation", "model", "split_strategy", "seed", "config_hash"]
    available = [column for column in identity if column in results.columns]
    if "timestamp_utc" in results.columns:
        results = results.sort_values("timestamp_utc")
    return results.drop_duplicates(subset=available, keep="last").reset_index(drop=True)


def _representation_answer(summary: pd.DataFrame, first: str, second: str) -> str:
    available = set(summary["representation"])
    if first not in available or second not in available:
        missing = sorted({first, second} - available)
        return f"Unavailable: no completed matched summary for {', '.join(missing)}."
    statements: list[str] = []
    for split_strategy in sorted(summary["split_strategy"].unique()):
        subset = summary.loc[summary["split_strategy"] == split_strategy]
        first_rows = subset.loc[subset["representation"] == first]
        second_rows = subset.loc[subset["representation"] == second]
        if first_rows.empty or second_rows.empty:
            continue
        first_best = first_rows.loc[first_rows["mae_mean"].idxmin()]
        second_best = second_rows.loc[second_rows["mae_mean"].idxmin()]
        statements.append(
            f"For the {split_strategy} split, the lowest observed mean MAE was "
            f"{first_best['mae_mean']:.4f} for {first}/{first_best['model']} and "
            f"{second_best['mae_mean']:.4f} for {second}/{second_best['model']}."
        )
    return " ".join(statements) or "No split contains completed runs for both representations."


def _degradation_table(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["representation", "model"],
        columns="split_strategy",
        values="mae_mean",
        aggfunc="first",
    ).reset_index()
    if {"random", "scaffold"}.issubset(pivot.columns):
        pivot["scaffold_minus_random_mae"] = pivot["scaffold"] - pivot["random"]
        return pivot.sort_values("scaffold_minus_random_mae", ascending=False)
    return pd.DataFrame()


def _largest_errors(results: pd.DataFrame, project_root: Path, limit: int = 10) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, run in results.iterrows():
        path = Path(str(run["prediction_file"]))
        path = path if path.is_absolute() else project_root / path
        predictions = pd.read_csv(path)
        test = predictions.loc[predictions["partition"] == "test"].copy()
        test["absolute_error"] = test["residual"].abs()
        test["experiment_id"] = run["experiment_id"]
        test["representation"] = run["representation"]
        test["model"] = run["model"]
        frames.append(test)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    columns = [
        "experiment_id", "representation", "model", "smiles", "actual", "predicted",
        "absolute_error",
    ]
    return combined.nlargest(limit, "absolute_error")[columns]


def generate_report(
    results_path: Path,
    output_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    n_bootstrap: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        output_path.write_text(
            "# PolyTg-Bench Results\n\n"
            "No completed experiments are available. No conclusions can be drawn.\n",
            encoding="utf-8",
        )
        return
    results = _latest_unique_runs(pd.read_csv(results_path))
    if results.empty:
        output_path.write_text(
            "# PolyTg-Bench Results\n\n"
            "No completed experiments are available. No conclusions can be drawn.\n",
            encoding="utf-8",
        )
        return
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)
    summary.to_csv(tables_dir / "benchmark_summary.csv", index=False)
    project_root = results_path.resolve().parents[1]
    comparisons = pairwise_comparisons(
        results, project_root, n_bootstrap=n_bootstrap
    )
    comparisons.to_csv(tables_dir / "pairwise_comparisons.csv", index=False)
    degradation = _degradation_table(summary)
    degradation.to_csv(tables_dir / "scaffold_degradation.csv", index=False)
    largest_errors = _largest_errors(results, project_root)
    largest_errors.to_csv(tables_dir / "largest_errors.csv", index=False)
    save_benchmark_comparison_plots(summary, figures_dir)

    graph_answer = _representation_answer(summary, "molecular_graph", "rdkit_descriptors")
    fingerprint_answer = _representation_answer(summary, "ecfp", "molecular_graph")
    language_answer = _representation_answer(summary, "chembert", "rdkit_descriptors")
    degradation_text = (
        degradation.to_markdown(index=False, floatfmt=".4f")
        if not degradation.empty
        else "Both random and scaffold results are required to quantify degradation."
    )
    errors_text = (
        largest_errors.to_markdown(index=False, floatfmt=".4f")
        if not largest_errors.empty
        else "No saved test predictions were available."
    )
    comparison_text = (
        comparisons.to_markdown(index=False, floatfmt=".4f")
        if not comparisons.empty
        else "No two completed runs shared the same split ID and seed."
    )
    text = f"""# PolyTg-Bench Results

This report was generated from `{results_path.as_posix()}` using {len(results)} unique completed
run(s). All metrics below come from saved test predictions. Mean, sample standard deviation, and 95%
Student-t
confidence intervals summarize repeated seeds; intervals are undefined for a single run.

## Aggregate test performance

{summary.to_markdown(index=False, floatfmt=".4f")}

## Research questions

### Do graph representations outperform RDKit descriptors?

{graph_answer} These values are descriptive; use the paired table below before interpreting a
difference.

### Do fingerprints remain competitive with graph models?

{fingerprint_answer}

### Do fixed chemical-language-model embeddings provide an advantage?

{language_answer}

### Which models degrade most under scaffold splitting?

{degradation_text}

### Are richer representations worth their computational cost?

Compare `training_time_seconds_mean` with MAE/RMSE in the aggregate table. Embedding extraction time
is
amortized through the verified cache and is recorded in cache metadata; it is not part of downstream
regressor training time. No cost-benefit claim is made automatically.

## Paired bootstrap comparisons

Each row compares per-molecule absolute errors on identical saved test rows. The reported difference
is
MAE(A) minus MAE(B), so negative favors A. Intervals containing zero do not establish a consistent
difference. Tail probabilities are exploratory and are not corrected for multiple comparisons.

{comparison_text}

## Largest recorded test errors

{errors_text}

## Limitations

SMILES may omit chain length, end groups, tacticity, copolymer composition, morphology,
molecular-weight
distribution, and measurement conditions. Scaffold splitting is an imperfect extrapolation proxy,
especially for small or mostly acyclic collections. Repeated literature Tg values can reflect real
condition dependence. Small repeated-seed samples produce uncertain confidence intervals, while many
pairwise comparisons increase false-positive risk. Results apply only to the cited dataset, cleaning
rules, target units, pretrained-model revision, and stored splits represented by these artifacts.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_results.csv")
    parser.add_argument("--output", default="results/REPORT.md")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    generate_report(
        Path(args.results),
        Path(args.output),
        Path(args.tables_dir),
        Path(args.figures_dir),
        args.bootstrap_iterations,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
