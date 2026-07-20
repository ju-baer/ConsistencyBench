#!/usr/bin/env python3
"""
Stage 3 (Part A): Score every raw (prompt_a, prompt_b, response_a, response_b)
result with the RBS + LJS ensemble, then run cross-judge validation and
inter-annotator agreement as scoring-quality checks.

Usage:
    export OPENROUTER_API_KEY=sk-...
    python scripts/03_run_scoring.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from consistencybench import config  # noqa: E402
from consistencybench.agreement import compute_iaa  # noqa: E402
from consistencybench.client import load_checkpoint  # noqa: E402
from consistencybench.scoring import cross_judge_validation, score_all  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    raw_results = load_checkpoint("cb_raw_responses.json")
    if not raw_results:
        raise SystemExit("No raw responses found. Run scripts/02_run_experiments.py first.")

    scored_results = score_all(raw_results)

    print("\nCross-judge robustness (re-scoring LJS probes with an alternative judge)...")
    cross_judge_validation(scored_results)

    print("\nInter-annotator agreement (RBS vs LJS)...")
    compute_iaa(scored_results)

    print(f"\nDone. {len(scored_results):,} scored results saved to "
          f"{config.BASE_DIR / 'checkpoints' / 'cb_scored.json'}")


if __name__ == "__main__":
    main()
