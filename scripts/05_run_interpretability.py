#!/usr/bin/env python3
"""
Stage 5 (Part B): Mechanistic interpretability extension on a local
open-weight model (Qwen2.5-1.5B-Instruct by default). Requires a GPU for
reasonable runtime (a free-tier Colab T4 is sufficient); will run on CPU but
slowly.

Chains: activation extraction -> shortcut probing (per-layer linear probes) ->
causal tests (activation patching + diff-in-means steering + specificity check).

Usage:
    export OPENROUTER_API_KEY=sk-...   # still needed: LJS scoring uses the API judge
    python scripts/05_run_interpretability.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from consistencybench import config  # noqa: E402
from consistencybench.client import load_checkpoint  # noqa: E402
from consistencybench.interpretability.activations import (  # noqa: E402
    build_activation_matrix,
    extract_activations,
    score_local_model,
)
from consistencybench.interpretability.causal_tests import (  # noqa: E402
    get_target_module,
    run_activation_patching,
    run_feature_steering,
    run_specificity_check,
)
from consistencybench.interpretability.shortcut_probe import run_shortcut_probing  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    all_probes = load_checkpoint("cb_probes.json")
    if not all_probes:
        raise SystemExit("No probe dataset found. Run scripts/01_run_probegen.py first.")

    print("=== Activation extraction ===")
    interp_backend, interp_probes, local_dir = extract_activations(all_probes)
    interp_raw = load_checkpoint("cb_interp_raw.json")
    interp_scored = score_local_model(interp_raw)
    n_layers = interp_backend.n_layers
    X_by_layer, y, fam_arr, valid_ids = build_activation_matrix(interp_scored, n_layers, local_dir)

    print("\n=== Shortcut probing ===")
    probe_out = run_shortcut_probing(X_by_layer, y, fam_arr, n_layers)
    best_layer = probe_out["best_layer"]

    print("\n=== Causal tests ===")
    target_module = get_target_module(interp_backend, best_layer)
    patching_results = run_activation_patching(
        interp_backend, target_module, best_layer, interp_probes, interp_raw,
        fam_arr, y, valid_ids, local_dir=str(local_dir),
    )
    df_dose, direction_tensor = run_feature_steering(
        interp_backend, target_module, best_layer, X_by_layer, y, valid_ids, interp_probes, interp_raw,
    )
    run_specificity_check(interp_backend, target_module, direction_tensor)

    print("\nDone. See data/interpretability/ for all outputs:")
    print("  layer_probe_results.csv, family_probe_results.csv, pca_pseudo_features.csv,")
    print("  probe_directions.npz, activation_patching_results.csv, steering_dose_response.csv,")
    print("  activation_matrix.npz")
    print(f"\nBest probe layer: {best_layer}/{n_layers}  (chance={probe_out['chance_rate']:.3f})")
    print(f"Activation patching flip rate: "
          f"{sum(r['flipped_toward_source'] for r in patching_results)}/{len(patching_results)} families")
    print(f"Steering: IR {df_dose.iloc[0]['ir_pct']:.1f}% -> {df_dose['ir_pct'].min():.1f}% at max strength")


if __name__ == "__main__":
    main()
