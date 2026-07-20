#!/usr/bin/env python3
"""
Convenience wrapper that runs the entire ConsistencyBench pipeline end-to-end
by calling scripts 01 through 06 in sequence. This is what the notebook does
interactively, cell by cell; running it as one process is mostly useful for a
`--quick-test` smoke run on CI or a fresh machine, since a full run is many
hours of API calls and is normally executed stage-by-stage (with the
resumable checkpoints doing the heavy lifting across sessions).

Usage:
    export OPENROUTER_API_KEY=sk-...
    python scripts/run_full_pipeline.py --quick-test          # behavioral only
    python scripts/run_full_pipeline.py --quick-test --with-interpretability  # + Part B (needs GPU)
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def _run(script: str, argv: list[str] | None = None) -> None:
    print(f"\n{'#' * 70}\n# Running {script}\n{'#' * 70}")
    old_argv = sys.argv
    sys.argv = [script] + (argv or [])
    try:
        runpy.run_path(str(SCRIPTS_DIR / script), run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--with-interpretability", action="store_true",
                         help="Also run Part B (requires a local GPU for reasonable speed).")
    parser.add_argument("--skip-intervention", action="store_true")
    parser.add_argument("--skip-hints", action="store_true")
    args = parser.parse_args()

    probegen_args = ["--quick-test"] if args.quick_test else []
    _run("01_run_probegen.py", probegen_args)
    _run("02_run_experiments.py")
    _run("03_run_scoring.py")

    analysis_args = []
    if args.skip_intervention:
        analysis_args.append("--skip-intervention")
    if args.skip_hints:
        analysis_args.append("--skip-hints")
    _run("04_run_analysis.py", analysis_args)

    if args.with_interpretability:
        _run("05_run_interpretability.py")

    _run("06_generate_deliverables.py")
    print("\nFull pipeline complete.")


if __name__ == "__main__":
    main()
