"""
HuggingFace Export: 4 ready-to-upload dataset bundles + auto-generated dataset
cards (READMEs with YAML frontmatter).

  consistencybench-probes           4,500 model-agnostic probe pairs (prompts only)
  consistencybench-results          Full scored (probe, model) pairs + leaderboard CSV
  consistencybench-interpretability Layer-probing, patching, steering, hint-sensitivity bundle

Direct, environment-independent port of Notebook Cell 22.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .. import config


def export_probes(all_probes: list[dict], out_dir: str | Path) -> pd.DataFrame:
    df_p = pd.DataFrame([{
        "probe_id": p["probe_id"], "transformation_family": p["family"],
        "domain": p["domain"], "difficulty": p["difficulty"],
        "semantic_distance_delta": p.get("delta", 0.0),
        "prompt_a": p["prompt_a"], "prompt_b": p["prompt_b"],
        "logical_constraint": p.get("logical_constraint", ""),
        "expected_inconsistency": p.get("expected_inconsistency", ""),
        "scoring_hint": p.get("scoring_hint", "ljs"),
        "difficulty_rationale": p.get("difficulty_rationale", ""),
    } for p in all_probes])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "consistencybench_probes.csv"
    df_p.to_csv(path, index=False, encoding="utf-8")
    print(f"Probes CSV: {len(df_p):,} rows -> {path}")
    return df_p


def export_results(scored_results: list[dict], out_dir: str | Path) -> pd.DataFrame:
    df_r = pd.DataFrame([{
        "probe_id": r.get("probe_id"), "transformation_family": r.get("family"),
        "domain": r.get("domain"), "difficulty": r.get("difficulty"), "delta": r.get("delta", 0.0),
        "model": r.get("model"), "model_id": r.get("model_id", ""),
        "prompt_a": r.get("prompt_a", ""), "prompt_b": r.get("prompt_b", ""),
        "response_a": r.get("response_a", ""), "response_b": r.get("response_b", ""),
        "consistent": r.get("consistent"),
        "inconsistent": (not r.get("consistent")) if r.get("consistent") is not None else None,
        "score_method": r.get("score_method", ""), "score_reason": r.get("score_reason", ""),
        "ljs_confidence": r.get("ljs_confidence", ""),
        "ans_a_summary": r.get("ans_a_summary", ""), "ans_b_summary": r.get("ans_b_summary", ""),
    } for r in scored_results if r.get("consistent") is not None])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "consistencybench_results.csv"
    df_r.to_csv(path, index=False, encoding="utf-8")
    print(f"Results CSV: {len(df_r):,} rows -> {path}")
    return df_r


def export_leaderboard(
    df_r: pd.DataFrame, cp_vectors: dict, ccs_scores: dict,
    df_hint: pd.DataFrame | None, out_dir: str | Path,
) -> pd.DataFrame:
    hint_by_model_map = {}
    if df_hint is not None and len(df_hint) > 0:
        hb = df_hint.groupby("model").agg(hint_ir=("hint_induced_inconsistency", "mean")).round(3)
        hint_by_model_map = hb["hint_ir"].to_dict()

    lb_rows = []
    for m in df_r["model"].unique():
        mdf = df_r[df_r["model"] == m]
        row = {
            "model": config.MODEL_LABELS.get(m, m),
            "model_id": mdf["model_id"].iloc[0] if len(mdf) > 0 else "",
            "n_probes": len(mdf), "overall_ir": round(mdf["inconsistent"].mean() * 100, 1),
            "paradigm": next((g for g, ms in config.MODEL_GROUPS.items() if m in ms), "Other"),
            "ccs": round(ccs_scores.get(m, float("nan")), 4) if m in ccs_scores else None,
            "hint_induced_ir_pct": round(hint_by_model_map.get(m, float("nan")) * 100, 1)
            if m in hint_by_model_map else None,
        }
        for f in config.TRANSFORMATION_FAMILIES:
            sub = mdf[mdf["transformation_family"] == f]["inconsistent"]
            row[f"ir_{f}"] = round(sub.mean() * 100, 1) if len(sub) > 0 else None
        for d in config.DIFFICULTIES:
            sub = mdf[mdf["difficulty"] == d]["inconsistent"]
            row[f"ir_{d}"] = round(sub.mean() * 100, 1) if len(sub) > 0 else None
        for dom in config.DOMAINS:
            sub = mdf[mdf["domain"] == dom]["inconsistent"]
            row[f"ir_{dom}"] = round(sub.mean() * 100, 1) if len(sub) > 0 else None
        if m in cp_vectors:
            for fi, f in enumerate(config.TRANSFORMATION_FAMILIES):
                row[f"cp_{f}"] = round(float(cp_vectors[m][fi]), 1)
        lb_rows.append(row)

    df_lb = pd.DataFrame(lb_rows).sort_values("overall_ir")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "consistencybench_leaderboard.csv"
    df_lb.to_csv(path, index=False, encoding="utf-8")
    print(f"Leaderboard: {len(df_lb)} models -> {path}")
    return df_lb


def export_interpretability_bundle(
    out_dir: str | Path,
    df_layers: pd.DataFrame | None = None,
    patching_results: list[dict] | None = None,
    df_dose: pd.DataFrame | None = None,
    df_hint: pd.DataFrame | None = None,
    best_layer: int | None = None,
    n_layers: int | None = None,
    best_acc: float | None = None,
) -> None:
    interp_dir = Path(out_dir) / "interpretability"
    interp_dir.mkdir(parents=True, exist_ok=True)
    if df_layers is not None:
        df_layers.to_csv(interp_dir / "layer_probe_results.csv", index=False)
    if patching_results is not None:
        pd.DataFrame(patching_results).to_csv(interp_dir / "activation_patching.csv", index=False)
    if df_dose is not None:
        df_dose.to_csv(interp_dir / "steering_dose_response.csv", index=False)
    if df_hint is not None:
        df_hint.to_csv(interp_dir / "hint_sensitivity.csv", index=False)
    meta = {
        "interpretability_model": config.INTERP_MODEL_ID,
        "n_layers": n_layers, "best_probe_layer": best_layer, "best_probe_accuracy": best_acc,
        "note": "This model is a dedicated white-box interpretability testbed, distinct from "
                "the black-box API leaderboard.",
    }
    with open(interp_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Interpretability bundle saved to {interp_dir}")


README_PROBES = """---
language: [en]
license: cc-by-4.0
task_categories: [text-classification, question-answering]
tags: [llm-evaluation, logical-consistency, transformation-families, benchmark]
size_categories: [1K<n<10K]
pretty_name: "ConsistencyBench - Probe Pairs"
configs:
- config_name: default
  data_files:
  - split: train
    path: consistencybench_probes.csv
---

# ConsistencyBench - Probe Pairs

**A Constraint-Preserving Framework for Evaluating Logical Consistency as a First-Class
Property of Language Models**

4,500 constraint-driven probe pairs testing whether LLM responses are invariant under
truth-preserving logical transformations. Model-agnostic (prompts only) - see
`consistencybench-results` for model responses, scores, and the leaderboard.

## Five Transformation Families

| Family | Logical Basis | Formal Constraint | Coverage |
|--------|--------------|-------------------|----------|
| Composition | Relation composition (transitivity) | A=>B AND B=>C => A=>C | 18.4% |
| Reversal | Symmetric relation reversal | rel(X,Y) <=> rel(Y,X) | 14.2% |
| Complement | Truth complement (negation) | NOT(assert(P) AND assert(NOT-P)) | 28.6% |
| Ordering | Asymmetric temporal ordering | before(A,B) <=> after(B,A) | 10.8% |
| Equivalence | Semantic equivalence preservation | equiv(pA,pB) => compat(rA,rB) | 11.4% |

## Schema

`probe_id, transformation_family, domain, difficulty, semantic_distance_delta,
prompt_a, prompt_b, logical_constraint, expected_inconsistency, scoring_hint,
difficulty_rationale`

Difficulty is defined via semantic distance `delta = 0.4*d_lex + 0.4*d_sem + 0.2*d_syn`,
not an arbitrary label. Easy: delta<0.35, Medium: 0.35-0.65, Hard: delta>=0.65.

## Related Datasets

- `consistencybench-results` - scored (probe, model) pairs + leaderboard
- `consistencybench-interpretability` - white-box mechanistic analysis (layer probing,
  activation patching, feature steering) on a local open-weight model

## Citation

See the [GitHub repository](https://github.com/YOUR_USERNAME/ConsistencyBench) for the
citation entry.
"""

README_RESULTS = """---
language: [en]
license: cc-by-4.0
tags: [llm-evaluation, logical-consistency, leaderboard, benchmark]
size_categories: [10K<n<100K]
pretty_name: "ConsistencyBench - Results & Leaderboard"
configs:
- config_name: results
  data_files: [{split: train, path: consistencybench_results.csv}]
- config_name: leaderboard
  data_files: [{split: train, path: consistencybench_leaderboard.csv}]
---

# ConsistencyBench - Results & Leaderboard

Full model responses + consistency scores for every evaluated model across the
ConsistencyBench probe set. Leaderboard includes IR, CCS (calibration), and
hint-induced inconsistency rate as three independent reliability dimensions.

## Submitting Your Model

1. Evaluate on `consistencybench-probes` at temperature=0 with the standard system prompt
   (see `src/consistencybench/harness.py::SYSTEM_PROMPT` in the GitHub repo)
2. Score with `src/consistencybench/scoring.py`
3. Open a PR or issue with your results CSV

## Key Findings (illustrative — regenerate with your own run for current numbers)

| Finding | Result |
|---------|--------|
| All evaluated models show non-trivial IR | see `overall_ir` column |
| IR is largely orthogonal to accuracy | see Kendall tau in `tab_stats.tex` |
| Reasoning training helps selectively | compare `ir_*` columns across ablation pairs |
| Ethics domain is consistently hardest | see `ir_ethics` vs `ir_science` |
| FTSC intervention reduces IR on hard probes | see the intervention CSVs |
| Some models are swayed by unsupported hints | see `hint_induced_ir_pct` column |
"""

README_INTERPRETABILITY = """---
language: [en]
license: cc-by-4.0
tags: [interpretability, activation-patching, feature-steering, mechanistic]
pretty_name: "ConsistencyBench - Interpretability Extension"
---

# ConsistencyBench - Interpretability Extension

White-box mechanistic analysis of logical inconsistency using a local open-weight model
(full activation access) as a dedicated interpretability testbed -- distinct from the
black-box API leaderboard.

## Contents

- `layer_probe_results.csv` - per-layer logistic-regression probe accuracy for decoding
  "will this response be inconsistent?" directly from residual-stream activations
- `activation_patching.csv` - literal patching results: copying a consistent run's
  activation into an inconsistent run and checking whether the output flips
- `steering_dose_response.csv` - IR at increasing diff-in-means steering strength (a
  causal dose-response curve, not just a correlational probe)
- `hint_sensitivity.csv` - flip rate and hint-induced inconsistency under misleading
  epistemic pressure
- `metadata.json` - best probe layer, probe accuracy, model config

## Method Summary

1. Train a linear probe at every layer to decode inconsistency from the residual stream
   at the final prompt token (Alain & Bengio, 2017 methodology)
2. Take the best layer's probe direction (and the diff-in-means direction) as the
   "shortcut direction" -- in place of a pretrained SAE, which does not exist for this
   checkpoint
3. Run two causal tests: literal activation patching between matched consistent/
   inconsistent pairs, and a steering dose-response sweep with a specificity check on
   unrelated control questions

## Citation

Same as `consistencybench-probes` and `consistencybench-results`.
"""


def write_dataset_cards(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README_probes.md").write_text(README_PROBES, encoding="utf-8")
    (out_dir / "README_results.md").write_text(README_RESULTS, encoding="utf-8")
    (out_dir / "README_interpretability.md").write_text(README_INTERPRETABILITY, encoding="utf-8")
    print(f"Dataset cards written to {out_dir}")


def export_all(
    all_probes: list[dict],
    scored_results: list[dict],
    cp_vectors: dict,
    ccs_scores: dict,
    df_hint: pd.DataFrame | None = None,
    df_layers: pd.DataFrame | None = None,
    patching_results: list[dict] | None = None,
    df_dose: pd.DataFrame | None = None,
    best_layer: int | None = None,
    n_layers: int | None = None,
    best_acc: float | None = None,
    out_dir: str | Path = config.BASE_DIR / "hf_export",
) -> None:
    out_dir = Path(out_dir)
    export_probes(all_probes, out_dir)
    df_r = export_results(scored_results, out_dir)
    export_leaderboard(df_r, cp_vectors, ccs_scores, df_hint, out_dir)
    export_interpretability_bundle(
        out_dir, df_layers=df_layers, patching_results=patching_results, df_dose=df_dose,
        df_hint=df_hint, best_layer=best_layer, n_layers=n_layers, best_acc=best_acc,
    )
    write_dataset_cards(out_dir)

    print(f"\nSaved to {out_dir}:")
    for fname in ["consistencybench_probes.csv", "consistencybench_results.csv",
                  "consistencybench_leaderboard.csv", "README_probes.md",
                  "README_results.md", "README_interpretability.md"]:
        p = out_dir / fname
        size = os.path.getsize(p) / 1024 if p.exists() else 0
        print(f"  {fname}: {size:.1f} KB")

    print("\nHuggingFace upload commands:")
    print(f"  huggingface-cli upload YOUR_USERNAME/consistencybench-probes "
          f"{out_dir}/consistencybench_probes.csv --repo-type dataset")
    print(f"  huggingface-cli upload YOUR_USERNAME/consistencybench-results "
          f"{out_dir}/consistencybench_results.csv {out_dir}/consistencybench_leaderboard.csv "
          f"--repo-type dataset")
    print(f"  huggingface-cli upload YOUR_USERNAME/consistencybench-interpretability "
          f"{out_dir}/interpretability --repo-type dataset")
