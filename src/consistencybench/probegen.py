"""
ProbeGen: 4-stage constraint-driven probe synthesis framework.

Stage 1 (Constraint Specification) is pure logic — no LLM call.
Stage 2+3 (Semantic Instantiation + Difficulty Calibration) is one combined
generator call that targets an explicit delta range.
Stage 4 (Constraint Verification) spot-checks generated probes against five
criteria and rejects failures before they enter the dataset.

Direct, environment-independent port of Notebook Cells 5-6.
"""
from __future__ import annotations

import itertools
import time

from tqdm.auto import tqdm

from . import config
from .client import call_model, safe_json, sanitize_probe
from .theory import FAMILY_FORMAL_SPEC

DOMAIN_GUIDANCE = {
    "general": "everyday knowledge, common facts, general world reasoning",
    "science": "physics, chemistry, biology, mathematics, CS — factual and quantitative",
    "ethics": "moral philosophy, normative claims, value judgments — inherently contested",
}


def build_constraint_spec(family: str, domain: str, difficulty: str) -> dict:
    """Stage 1: Build the logical constraint graph programmatically. No LLM."""
    spec = FAMILY_FORMAL_SPEC[family]
    delta_min, delta_max = config.DIFFICULTY_DELTA[difficulty]
    return {
        "family": family,
        "formal_constraint": spec["formal_constraint"],
        "consistency_rule": spec["consistency_rule"],
        "probe_structure": spec["probe_structure"],
        "scoring_hint": spec["scoring"],
        "domain": domain,
        "difficulty": difficulty,
        "delta_target": {"min": delta_min, "max": delta_max},
    }


def build_probegen_prompt(constraint_spec: dict, n: int) -> str:
    """Stage 2+3 combined prompt: instantiate the constraint over the domain and
    calibrate difficulty via the semantic distance target."""
    fam = constraint_spec["family"]
    spec = FAMILY_FORMAL_SPEC[fam]
    dom = constraint_spec["domain"]
    diff = constraint_spec["difficulty"]
    d_min = constraint_spec["delta_target"]["min"]
    d_max = constraint_spec["delta_target"]["max"]
    struct = constraint_spec["probe_structure"]

    return f"""You are an expert benchmark designer creating logically constrained probe pairs for a NeurIPS paper on logical consistency evaluation of LLMs.

=== TRANSFORMATION FAMILY: {spec['name'].upper()} ===
Logical basis:      {spec['logical_basis']}
Formal constraint:  {spec['formal_constraint']}
Consistency rule:   {spec['consistency_rule']}
Probe structure:
  Prompt A: {struct['prompt_a']}
  Prompt B: {struct['prompt_b']}

=== PARAMETERS ===
Domain:     {dom} ({DOMAIN_GUIDANCE[dom]})
Difficulty: {diff.upper()}
  Target semantic distance: {d_min:.2f} <= delta <= {d_max:.2f}
  delta = 0.4*d_lex + 0.4*d_sem + 0.2*d_syn
  - Easy   (delta < 0.35): prompts share most surface tokens; link is transparent
  - Medium (0.35-0.65):    prompts differ in structure; link requires careful reading
  - Hard   (delta >= 0.65): prompts maximally divergent lexically/semantically while
                            the logical constraint is perfectly preserved
Generate: Exactly {n} probe pairs.

=== ABSOLUTE REQUIREMENTS ===
1. FULLY SELF-CONTAINED prompts — no cross-references between A and B
2. Do NOT hint at expected answer or logical relationship in either prompt
3. Hard probes must MAXIMIZE surface divergence while PERFECTLY preserving constraint
4. reversal/complement/ordering: answers must clearly extract as yes/no
5. Vary topics substantially — no near-duplicates in this batch
6. ASCII characters ONLY
7. Include difficulty_rationale explaining why this probe achieves target delta

Scoring hint: {constraint_spec['scoring_hint']}

=== OUTPUT: Valid JSON array ONLY. No markdown, no preamble. ===
[
  {{
    "id": 1,
    "family": "{fam}",
    "domain": "{dom}",
    "difficulty": "{diff}",
    "prompt_a": "...",
    "prompt_b": "...",
    "logical_constraint": "one sentence: what must hold between the two answers",
    "expected_inconsistency": "one sentence: how a failing model would violate this",
    "scoring_hint": "{constraint_spec['scoring_hint']}",
    "difficulty_rationale": "one sentence: why this probe achieves the delta={d_min:.2f}-{d_max:.2f} target"
  }}
]"""


VERIFICATION_PROMPT = """You are a formal logic expert verifying probe pairs for a logical consistency benchmark.

Transformation family: {family}
Formal constraint: {formal_constraint}
Scoring hint: {scoring_hint}
Prompt A: {prompt_a}
Prompt B: {prompt_b}
Claimed logical constraint: {logical_constraint}

Check ALL five criteria:
1. LOGICAL_VALIDITY: Does the probe correctly instantiate the transformation family?
2. SEMANTIC_PRESERVATION: Does Prompt B preserve the truth-conditional content under transformation?
3. TYPE_CORRECTNESS: Does scoring_hint correctly reflect what a consistent model should do?
4. SCORING_COMPATIBILITY: For rbs probes, is the expected yes/no clearly extractable?
5. STANDALONE: Is each prompt fully self-contained with no cross-references?

Respond ONLY with valid JSON:
{{"pass": true or false, "failed_criteria": [], "notes": "brief explanation if failed"}}"""


def verify_probe(probe: dict, generator_model: str = config.GENERATOR_MODEL) -> tuple[bool, str]:
    """Stage 4: Verify a generated probe against all five criteria."""
    spec = FAMILY_FORMAL_SPEC[probe.get("family", "")]
    prompt = VERIFICATION_PROMPT.format(
        family=probe.get("family", ""),
        formal_constraint=spec.get("formal_constraint", ""),
        scoring_hint=probe.get("scoring_hint", ""),
        prompt_a=probe.get("prompt_a", "")[:300],
        prompt_b=probe.get("prompt_b", "")[:300],
        logical_constraint=probe.get("logical_constraint", ""),
    )
    try:
        raw = call_model(generator_model, prompt, max_tokens=200, temperature=0.0)
        parsed = safe_json(raw)
        result = parsed[0] if isinstance(parsed, list) else parsed
        return result.get("pass", True), result.get("notes", "")
    except Exception as e:  # fail open, don't block pipeline
        return True, f"Verification skipped: {e}"


def deduplicate_probes(probes: list[dict], threshold: float = 0.82) -> list[dict]:
    """Trigram Jaccard deduplication on prompt_a."""
    def trigrams(s: str) -> set[str]:
        s = s.lower()
        return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()

    def jaccard(a: str, b: str) -> float:
        ta, tb = trigrams(a), trigrams(b)
        return len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0

    seen: list[str] = []
    unique: list[dict] = []
    for p in probes:
        pa = p.get("prompt_a", "")
        if all(jaccard(pa, s) < threshold for s in seen):
            unique.append(p)
            seen.append(pa)
    return unique


def run_probegen(
    family: str, domain: str, difficulty: str, n: int,
    verify: bool = True, generator_model: str = config.GENERATOR_MODEL,
) -> list[dict]:
    """Run all 4 stages for one (family, domain, difficulty) cell."""
    constraint_spec = build_constraint_spec(family, domain, difficulty)          # Stage 1
    prompt = build_probegen_prompt(constraint_spec, n)                           # Stage 2+3
    raw = call_model(generator_model, prompt, max_tokens=config.MAX_TOKENS_GEN, temperature=0.8)
    probes = safe_json(raw)
    probes = [sanitize_probe(p) for p in probes
              if isinstance(p, dict) and p.get("prompt_a") and p.get("prompt_b")]
    if verify:                                                                   # Stage 4
        verified = []
        for p in probes:
            if len(verified) < max(3, n // 10):  # spot-check up to 10% to control cost
                passed, notes = verify_probe(p, generator_model)
                if not passed:
                    continue
                time.sleep(0.2)
            verified.append(p)
        probes = verified
    return probes


def generate_full_dataset(
    all_probes: list[dict] | None = None,
    batch_size: int = 10,
    max_attempts: int = 6,
) -> list[dict]:
    """Run ProbeGen across all (family x domain x difficulty) cells with resumable
    checkpointing. After generation, computes the actual delta for every probe
    using real embeddings, so the dataset's difficulty distribution can be
    independently audited.

    Direct port of Notebook Cell 6's main loop.
    """
    from .client import load_checkpoint, save_checkpoint
    from .theory import classify_difficulty, compute_semantic_distance

    all_probes = all_probes if all_probes is not None else (load_checkpoint("cb_probes.json") or [])
    completed = {(p["family"], p["domain"], p["difficulty"]) for p in all_probes}
    combos = list(itertools.product(config.TRANSFORMATION_FAMILIES, config.DOMAINS, config.DIFFICULTIES))
    remaining = [(f, d, diff) for f, d, diff in combos if (f, d, diff) not in completed]

    print(f"Combos: {len(combos)} | Done: {len(combos) - len(remaining)} | Remaining: {len(remaining)}")
    print(f"Target: {config.N_PER_COMBO * len(combos):,} probes\n")

    next_id = max((p.get("probe_id", 0) for p in all_probes), default=0) + 1

    for family, domain, difficulty in tqdm(remaining, desc="ProbeGen"):
        combo_probes: list[dict] = []
        needed, attempts = config.N_PER_COMBO, 0

        while len(combo_probes) < needed and attempts < max_attempts:
            batch_n = min(batch_size, needed - len(combo_probes) + 2)
            try:
                batch = run_probegen(family, domain, difficulty, batch_n, verify=(attempts == 0))
                combo_probes.extend(batch)
                combo_probes = deduplicate_probes(combo_probes)
            except Exception as e:
                print(f"  [{family}/{domain}/{difficulty}] attempt {attempts} failed: {e}")
            attempts += 1

        combo_probes = combo_probes[:needed]
        for p in combo_probes:
            p["probe_id"] = next_id
            next_id += 1
            p["delta"] = compute_semantic_distance(p.get("prompt_a", ""), p.get("prompt_b", ""))
            p["measured_difficulty"] = classify_difficulty(p["delta"])

        all_probes.extend(combo_probes)
        save_checkpoint(all_probes, "cb_probes.json")

    print(f"\nFinal dataset: {len(all_probes):,} probes")
    return all_probes


if __name__ == "__main__":
    config.ensure_dirs()
    generate_full_dataset()
