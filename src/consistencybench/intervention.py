"""
Intervention: Baseline -> CR -> SC -> FTSC.

Four conditions run on hard probes across representative models:

  baseline — independent queries, no consistency instruction
  CR       — Consistency Reminder in the system prompt (generic)
  SC       — Self-Check: prompt B is prefixed with the model's own answer to A
  FTSC     — Family-Targeted Self-Check: SC, but the system additionally names
             the specific transformation family and its formal consistency rule

FTSC tests whether telling a model *which kind* of constraint applies helps
more than a generic consistency reminder.

Direct, environment-independent port of Notebook Cell 11.
"""
from __future__ import annotations

import random
import time

import pandas as pd
from tqdm.auto import tqdm

from . import config
from .client import call_model, load_checkpoint, save_checkpoint
from .harness import APIBackend
from .scoring import score_result
from .theory import FAMILY_FORMAL_SPEC

INTERVENTION_MODELS = ["gpt4o", "claude_opus", "deepseek_r1", "llama4"]

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the following question directly "
    "and concisely. When the question has a clear yes or no answer, begin your "
    "response with 'Yes' or 'No' followed by a brief explanation of 1-3 sentences. "
    "Do not hedge unnecessarily."
)

CR_SYSTEM = (
    SYSTEM_PROMPT +
    " Before answering, note that you may be asked related questions. "
    "Ensure your answers are logically consistent with each other."
)


def query_safe(model_id: str, prompt: str, system: str = SYSTEM_PROMPT) -> dict:
    return APIBackend(model_id).query(prompt, system=system)


def get_ftsc_prompt(probe: dict, response_a: str) -> str:
    """Family-Targeted Self-Check: names the transformation family and its rule."""
    family = probe.get("family", "")
    spec = FAMILY_FORMAL_SPEC.get(family, {})
    family_hint = spec.get("consistency_rule", "ensure logical consistency")
    return (f"In this conversation, the following logical constraint applies: "
            f"{family_hint}. "
            f"You previously responded: \"{response_a[:200]}\". "
            f"Ensure your answer to the following is logically consistent with that "
            f"prior response given the constraint above.\n\n{probe['prompt_b']}")


def run_intervention_conditions(
    probe: dict, model_name: str, model_id: str,
    done_set: set, iv_results: list, judge: str = config.GENERATOR_MODEL,
) -> None:
    base = {
        "probe_id": probe["probe_id"], "model": model_name, "family": probe["family"],
        "difficulty": "hard", "prompt_a": probe["prompt_a"], "prompt_b": probe["prompt_b"],
        "logical_constraint": probe.get("logical_constraint", ""),
        "expected_inconsistency": probe.get("expected_inconsistency", ""),
        "scoring_hint": probe.get("scoring_hint", config.FAMILY_SCORING.get(probe["family"], "ljs")),
    }
    if (probe["probe_id"], model_name, "baseline") not in done_set:
        ra = query_safe(model_id, probe["prompt_a"]); time.sleep(0.4)
        rb = query_safe(model_id, probe["prompt_b"]); time.sleep(0.4)
        r = {**base, "condition": "baseline",
             "response_a": ra["response"], "response_b": rb["response"],
             "error_a": ra["error"], "error_b": rb["error"]}
        iv_results.append(score_result(r, judge=judge))

    if (probe["probe_id"], model_name, "cr") not in done_set:
        ra = query_safe(model_id, probe["prompt_a"]); time.sleep(0.4)
        try:
            rb_txt = call_model(model_id, probe["prompt_b"], system=CR_SYSTEM)
        except Exception:
            rb_txt = None
        r = {**base, "condition": "cr", "response_a": ra["response"], "response_b": rb_txt,
             "error_a": ra["error"], "error_b": None}
        iv_results.append(score_result(r, judge=judge)); time.sleep(0.4)

    if (probe["probe_id"], model_name, "sc") not in done_set:
        ra = query_safe(model_id, probe["prompt_a"]); time.sleep(0.4)
        if ra["response"]:
            sc_b = (f"You previously answered: \"{ra['response'][:200]}\". "
                    f"Ensure your answer is logically consistent with that.\n\n{probe['prompt_b']}")
            rb = query_safe(model_id, sc_b)
        else:
            rb = {"response": None, "latency_s": None, "error": "no response A"}
        r = {**base, "condition": "sc", "response_a": ra["response"], "response_b": rb["response"],
             "error_a": ra["error"], "error_b": rb["error"]}
        iv_results.append(score_result(r, judge=judge)); time.sleep(0.4)

    if (probe["probe_id"], model_name, "ftsc") not in done_set:
        ra = query_safe(model_id, probe["prompt_a"]); time.sleep(0.4)
        if ra["response"]:
            rb = query_safe(model_id, get_ftsc_prompt(probe, ra["response"]))
        else:
            rb = {"response": None, "latency_s": None, "error": "no response A"}
        r = {**base, "condition": "ftsc", "response_a": ra["response"], "response_b": rb["response"],
             "error_a": ra["error"], "error_b": rb["error"]}
        iv_results.append(score_result(r, judge=judge)); time.sleep(0.4)


def run_full_intervention(
    all_probes: list[dict], n_probes: int = 300, seed: int = 42,
) -> list[dict]:
    random.seed(seed)
    hard_probes = [p for p in all_probes if p.get("difficulty") == "hard"]
    iv_probes = random.sample(hard_probes, min(n_probes, len(hard_probes)))

    iv_results = load_checkpoint("cb_intervention.json") or []
    done_iv = {(r["probe_id"], r["model"], r["condition"]) for r in iv_results}
    print(f"Intervention: {len(iv_probes)} hard probes x {len(INTERVENTION_MODELS)} models x 4 conditions")

    for probe in tqdm(iv_probes, desc="Intervention"):
        for mn in INTERVENTION_MODELS:
            run_intervention_conditions(probe, mn, config.MODELS[mn], done_iv, iv_results)

    save_checkpoint(iv_results, "cb_intervention.json")

    df_iv_plot = pd.DataFrame([
        {"model": r["model"], "condition": r["condition"], "family": r["family"],
         "ir": float(not r["consistent"]) * 100}
        for r in iv_results if r.get("consistent") is not None
    ])
    if len(df_iv_plot) > 0:
        summary = df_iv_plot.groupby(["model", "condition"])["ir"].mean().unstack()
        conds = ["baseline", "cr", "sc", "ftsc"]
        for c in conds:
            if c in summary.columns and "baseline" in summary.columns:
                summary[f"delta_{c}"] = (summary[c] - summary["baseline"]) / summary["baseline"] * 100
        print("\nIntervention Results (IR% on hard probes):")
        print(summary[[c for c in conds if c in summary.columns]].round(1).to_string())
        if "delta_ftsc" in summary.columns:
            print(f"\nAvg FTSC relative reduction: {summary['delta_ftsc'].mean():.1f}%")
        out_dir = config.BASE_DIR / "intervention"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "iv_summary.csv")
        df_iv_plot.to_csv(out_dir / "iv_full.csv", index=False)

    return iv_results
