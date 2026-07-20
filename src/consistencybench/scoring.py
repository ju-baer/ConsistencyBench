"""
Consistency Scoring (RBS + LJS Ensemble + Cross-Judge Validation).

Rule-Based Scoring (RBS) handles Reversal, Ordering, and Complement
(deterministic yes/no extraction). LLM-as-Judge (LJS) handles Composition and
Equivalence (requires actual reasoning about content). Cross-judge validation
re-scores a sample with an alternative judge model to confirm scoring isn't an
artifact of the primary judge's preferences.

Direct, environment-independent port of Notebook Cell 9.
"""
from __future__ import annotations

import re
import time

import numpy as np
from tqdm.auto import tqdm

from . import config
from .client import call_model, safe_json

YES_RE = re.compile(r"^\s*(yes|true|correct|affirmative|indeed|certainly)\b", re.I)
NO_RE = re.compile(r"^\s*(no|false|incorrect|negative|not\b|never|cannot)\b", re.I)


def extract_yn(text: str | None) -> str | None:
    if not text:
        return None
    first = re.split(r"[.!?\n]", text.strip())[0]
    if YES_RE.match(first):
        return "yes"
    if NO_RE.match(first):
        return "no"
    return None


def rule_based_score(result: dict) -> dict | None:
    rule = config.RBS_RULES.get(result.get("family", ""))
    if rule is None:
        return None
    ya = extract_yn(result.get("response_a", "") or "")
    yb = extract_yn(result.get("response_b", "") or "")
    if ya is None or yb is None:
        return None
    consistent = (ya == yb) if rule == "same" else (ya != yb)
    return {"consistent": consistent, "method": "rbs",
            "reason": f"A={ya}, B={yb}; rule={rule}", "ans_a": ya, "ans_b": yb}


LJS_PROMPT = """You are a logic expert. Determine if an AI model gave LOGICALLY CONSISTENT answers to two related prompts.

Transformation family: {family}
Logical constraint: {logical_constraint}
Expected inconsistency: {expected_inconsistency}

Prompt A: {prompt_a}
Response A: {response_a}

Prompt B: {prompt_b}
Response B: {response_b}

Evaluate ONLY logical consistency. Ignore factual accuracy or quality.

Respond ONLY with valid JSON (no markdown):
{{"consistent": true or false, "confidence": "high|medium|low",
  "ans_a_summary": "5-word max", "ans_b_summary": "5-word max",
  "reason": "one sentence: why consistent or inconsistent"}}"""


def ljs_score(result: dict, judge: str | None = None) -> dict:
    j = judge or config.GENERATOR_MODEL
    prompt = LJS_PROMPT.format(
        family=result.get("family", ""),
        logical_constraint=result.get("logical_constraint", ""),
        expected_inconsistency=result.get("expected_inconsistency", ""),
        prompt_a=result.get("prompt_a", ""),
        response_a=(result.get("response_a", "") or "")[:500],
        prompt_b=result.get("prompt_b", ""),
        response_b=(result.get("response_b", "") or "")[:500],
    )
    try:
        raw = call_model(j, prompt, max_tokens=300, temperature=0.0)
        parsed = safe_json(raw)
        parsed = parsed[0] if isinstance(parsed, list) else parsed
        parsed["method"] = "ljs"
        return parsed
    except Exception as e:
        return {"consistent": None, "method": "ljs", "confidence": "low",
                "reason": f"Judge error: {e}", "ans_a_summary": "", "ans_b_summary": ""}


def score_result(result: dict, judge: str | None = None) -> dict:
    if result.get("error_a") or result.get("error_b"):
        return {**result, "consistent": None, "score_method": "skipped",
                "score_reason": "API error", "ljs_confidence": "n/a",
                "ans_a_summary": "", "ans_b_summary": ""}
    if not result.get("response_a") or not result.get("response_b"):
        return {**result, "consistent": None, "score_method": "skipped",
                "score_reason": "missing response", "ljs_confidence": "n/a",
                "ans_a_summary": "", "ans_b_summary": ""}
    rbs = rule_based_score(result)
    if rbs is not None:
        return {**result, "consistent": rbs["consistent"], "score_method": "rbs",
                "score_reason": rbs["reason"], "ljs_confidence": "n/a",
                "ans_a_summary": rbs.get("ans_a", ""), "ans_b_summary": rbs.get("ans_b", "")}
    if judge is None and result.get("scoring_hint", "ljs") in ("rbs_same", "rbs_opposite"):
        return {**result, "consistent": None, "score_method": "skipped",
                "score_reason": "requires LJS scoring but no judge provided",
                "ljs_confidence": "n/a", "ans_a_summary": "", "ans_b_summary": ""}
    ljs = ljs_score(result, judge)
    return {**result, "consistent": ljs.get("consistent"), "score_method": "ljs",
            "score_reason": ljs.get("reason", ""), "ljs_confidence": ljs.get("confidence", ""),
            "ans_a_summary": ljs.get("ans_a_summary", ""), "ans_b_summary": ljs.get("ans_b_summary", "")}


def score_all(raw_results: list[dict], judge: str = config.GENERATOR_MODEL) -> list[dict]:
    """Score every raw result with the RBS+LJS ensemble, checkpointing along the way."""
    from .client import load_checkpoint, save_checkpoint

    scored_results = load_checkpoint("cb_scored.json") or []
    scored_ids = {(r["probe_id"], r["model"]) for r in scored_results}
    to_score = [r for r in raw_results if (r["probe_id"], r["model"]) not in scored_ids]
    ljs_n = sum(1 for r in to_score if r.get("family") in ("composition", "equivalence"))
    print(f"To score: {len(to_score):,} | RBS: {len(to_score) - ljs_n:,} | LJS: {ljs_n:,}")

    for i, result in enumerate(tqdm(to_score, desc="Scoring")):
        scored = score_result(result, judge=judge)
        scored_results.append(scored)
        if scored.get("score_method") == "ljs":
            time.sleep(0.4)
        if (i + 1) % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(scored_results, "cb_scored.json")

    save_checkpoint(scored_results, "cb_scored.json")
    scoreable = [r for r in scored_results if r.get("consistent") is not None]
    n_incons = sum(1 for r in scoreable if not r["consistent"])
    print(f"\nScored: {len(scoreable):,} | Inconsistent: {n_incons:,} "
          f"({100 * n_incons / max(len(scoreable), 1):.1f}%)")
    return scored_results


def cross_judge_validation(
    scored_results: list[dict], alt_judge: str = "openai/gpt-4o", sample_size: int = 50,
) -> dict | None:
    """Re-score a sample of LJS-scored results with an alternative judge to confirm
    scoring isn't an artifact of the primary judge's preferences. Returns
    agreement / kappa / alpha statistics."""
    import krippendorff
    from sklearn.metrics import cohen_kappa_score

    ljs_scored = [r for r in scored_results if r.get("score_method") == "ljs"
                  and r.get("consistent") is not None]
    sample_cj = ljs_scored[:min(400, len(ljs_scored))][:sample_size]

    primary_labels, alt_labels = [], []
    for result in tqdm(sample_cj, desc="Cross-judge"):
        alt = ljs_score(result, judge=alt_judge)
        if alt.get("consistent") is not None:
            primary_labels.append(1 if result["consistent"] else 0)
            alt_labels.append(1 if alt["consistent"] else 0)
        time.sleep(0.3)

    if len(primary_labels) < 10:
        print("Not enough overlapping samples for cross-judge validation.")
        return None

    kappa = cohen_kappa_score(primary_labels, alt_labels)
    alpha = krippendorff.alpha(np.array([primary_labels, alt_labels]), level_of_measurement="nominal")
    agree = float(np.mean(np.array(primary_labels) == np.array(alt_labels)))
    stats = {"n": len(primary_labels), "agreement": agree, "kappa": float(kappa), "alpha": float(alpha)}
    print(f"  Agreement: {agree:.3f} ({agree * 100:.1f}%) | Kappa: {kappa:.3f} | Alpha: {alpha:.3f}")
    return stats
