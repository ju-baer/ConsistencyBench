#!/usr/bin/env python3
"""
Stage 4 (Part A, continued): Consistency Profiles, intervention ladder,
calibration (CCS), hint sensitivity, and scaling analysis. Everything in this
script only needs `cb_scored.json` (and, for intervention/hints, more API
calls) -- no local GPU is required.

Usage:
    export OPENROUTER_API_KEY=sk-...
    python scripts/04_run_analysis.py [--skip-intervention] [--skip-hints]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from consistencybench import config  # noqa: E402
from consistencybench.calibration import compute_ccs_for_all_models  # noqa: E402
from consistencybench.client import load_checkpoint  # noqa: E402
from consistencybench.hint_sensitivity import run_hint_sensitivity  # noqa: E402
from consistencybench.intervention import run_full_intervention  # noqa: E402
from consistencybench.profiles import build_results_dataframe, compute_all_profiles, profile_distance_matrix
from consistencybench.scaling import run_scaling_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-intervention", action="store_true")
    parser.add_argument("--skip-hints", action="store_true")
    args = parser.parse_args()

    config.ensure_dirs()
    scored_results = load_checkpoint("cb_scored.json")
    all_probes = load_checkpoint("cb_probes.json")
    if not scored_results or not all_probes:
        raise SystemExit("Missing checkpoints. Run scripts 01-03 first.")

    df = build_results_dataframe(scored_results)

    print("=== Consistency Profiles ===")
    cp_vectors = compute_all_profiles(df)
    dist = profile_distance_matrix(cp_vectors)
    profiles_dir = config.BASE_DIR / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    dist.to_csv(profiles_dir / "cp_distance_matrix.csv")
    with open(profiles_dir / "cp_vectors.json", "w") as f:
        json.dump({m: {"label": config.MODEL_LABELS.get(m, m), "cp_vector": list(v),
                        "families": config.TRANSFORMATION_FAMILIES}
                   for m, v in cp_vectors.items()}, f, indent=2)

    print("\n=== Scaling Analysis ===")
    run_scaling_analysis(df)

    print("\n=== Consistency-Calibration Score (CCS) ===")
    df_cal, ccs_scores, bin_data_all = compute_ccs_for_all_models(scored_results)

    if not args.skip_intervention:
        print("\n=== Intervention: Baseline -> CR -> SC -> FTSC ===")
        run_full_intervention(all_probes)
    else:
        print("\nSkipping intervention (--skip-intervention).")

    if not args.skip_hints:
        print("\n=== Hint Sensitivity ===")
        run_hint_sensitivity(scored_results)
    else:
        print("\nSkipping hint sensitivity (--skip-hints).")

    print("\nDone. See data/profiles/, data/results/, data/intervention/, data/hints/.")


if __name__ == "__main__":
    main()
