#!/usr/bin/env python3
"""
Stage 6 (Part C): Generate all figures, LaTeX tables, qualitative examples,
the HuggingFace export bundle, and a final experiment summary JSON. Runs
purely on saved checkpoints -- no API or GPU calls.

Usage:
    python scripts/06_generate_deliverables.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from consistencybench import config  # noqa: E402
from consistencybench.analysis import figures, hf_export, qualitative  # noqa: E402
from consistencybench.client import load_checkpoint  # noqa: E402
from consistencybench.profiles import build_results_dataframe, compute_all_profiles  # noqa: E402
from consistencybench.statistics import generate_latex_tables, run_statistical_tests  # noqa: E402


def _load_json(path) -> dict | None:
    return json.load(open(path)) if os.path.exists(path) else None


def main() -> None:
    config.ensure_dirs()

    all_probes = load_checkpoint("cb_probes.json")
    scored_results = load_checkpoint("cb_scored.json")
    if not all_probes or not scored_results:
        raise SystemExit("Missing checkpoints. Run scripts 01-03 first.")

    df = build_results_dataframe(scored_results)
    cp_vectors = compute_all_profiles(df)

    # Optional artifacts from later stages -- everything degrades gracefully if absent.
    iv_results = load_checkpoint("cb_intervention.json") or []
    df_iv = pd.DataFrame([
        {"model": r["model"], "condition": r["condition"], "family": r["family"],
         "ir": float(not r["consistent"]) * 100}
        for r in iv_results if r.get("consistent") is not None
    ]) if iv_results else pd.DataFrame()

    ccs_path = config.BASE_DIR / "results" / "ccs_scores.json"
    ccs_scores = _load_json(ccs_path) or {}

    hint_csv = config.BASE_DIR / "hints" / "hint_sensitivity_full.csv"
    df_hint = pd.read_csv(hint_csv) if hint_csv.exists() else None

    layer_csv = config.BASE_DIR / "interpretability" / "layer_probe_results.csv"
    df_layers = pd.read_csv(layer_csv) if layer_csv.exists() else None
    best_layer = int(df_layers.loc[df_layers["test_acc"].idxmax(), "layer"]) if df_layers is not None else None
    chance_rate = None
    interp_matrix = config.BASE_DIR / "interpretability" / "activation_matrix.npz"
    if interp_matrix.exists():
        y = np.load(interp_matrix)["y"]
        chance_rate = float(max(y.mean(), 1 - y.mean()))

    dose_csv = config.BASE_DIR / "interpretability" / "steering_dose_response.csv"
    df_dose = pd.read_csv(dose_csv) if dose_csv.exists() else None

    patching_csv = config.BASE_DIR / "interpretability" / "activation_patching_results.csv"
    patching_results = pd.read_csv(patching_csv).to_dict("records") if patching_csv.exists() else None

    print("=== Figures ===")
    figures.generate_all_figures(
        df, cp_vectors, df_iv=df_iv if len(df_iv) else None,
        intervention_models=list(df_iv["model"].unique()) if len(df_iv) else None,
        ccs_scores=ccs_scores, df_layers=df_layers, best_layer=best_layer,
        chance_rate=chance_rate, df_dose=df_dose,
    )

    print("\n=== Statistical tests + LaTeX tables ===")
    stats = run_statistical_tests(df, figures.MMLU)
    interp_summary = None
    if best_layer is not None:
        best_acc = float(df_layers["test_acc"].max())
        interp_summary = {"model_label": config.INTERP_MODEL_LABEL, "best_layer": best_layer,
                           "n_layers": int(df_layers["layer"].max()), "best_acc": best_acc,
                           "chance_rate": chance_rate}
    generate_latex_tables(
        df, cp_vectors, stats, tables_dir="tables", df_iv=df_iv if len(df_iv) else None,
        intervention_models=list(df_iv["model"].unique()) if len(df_iv) else None,
        df_hint=df_hint, interp_summary=interp_summary,
    )

    print("\n=== Qualitative error analysis ===")
    qualitative.collect_qualitative_examples(scored_results)

    print("\n=== HuggingFace export ===")
    hf_export.export_all(
        all_probes, scored_results, cp_vectors, ccs_scores, df_hint=df_hint,
        df_layers=df_layers, patching_results=patching_results, df_dose=df_dose,
        best_layer=best_layer, n_layers=int(df_layers["layer"].max()) if df_layers is not None else None,
        best_acc=float(df_layers["test_acc"].max()) if df_layers is not None else None,
    )

    print("\n=== Final summary ===")
    meta = {
        "n_probes": len(all_probes), "n_models": len(config.MODELS),
        "transformation_families": config.TRANSFORMATION_FAMILIES,
        "domains": config.DOMAINS, "difficulties": config.DIFFICULTIES,
        "total_scored_pairs": len(df), "overall_ir_pct": round(df["ir"].mean(), 1),
        "per_model_ir": df.groupby("model")["ir"].mean().round(1).to_dict(),
        "interpretability_run": best_layer is not None,
        "best_probe_layer": best_layer,
    }
    with open(config.BASE_DIR / "experiment_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"\nAll deliverables written under {config.BASE_DIR}")


if __name__ == "__main__":
    main()
