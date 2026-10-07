"""Convenience entry point for the PolyTg-Bench command-line tools."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyTg-Bench research benchmark")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("prepare", "train", "benchmark", "evaluate", "report"),
        help="Run the corresponding script from the scripts/ directory.",
    )
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    script = "run_benchmark" if args.command == "benchmark" else args.command
    parser.error(f"Use `python scripts/{script}.py --help` to see the reproducible CLI options.")


if __name__ == "__main__":
    main()
