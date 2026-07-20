"""
Qualitative Error Analysis: one worked failure example per transformation
family, prioritizing high-confidence LJS verdicts and spreading examples
across different models so no single model's errors dominate the writeup.

Direct, environment-independent port of Notebook Cell 21.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config


def collect_qualitative_examples(
    scored_results: list[dict],
    out_dir: str | Path = config.BASE_DIR / "tables",
) -> list[dict]:
    examples: list[dict] = []
    used_models: set[str] = set()

    for family in config.TRANSFORMATION_FAMILIES:
        candidates = sorted(
            [r for r in scored_results if r.get("family") == family
             and r.get("consistent") is False and r.get("ljs_confidence") == "high"
             and r.get("response_a") and r.get("response_b")],
            key=lambda x: len(x.get("score_reason", "")),
        )
        if not candidates:
            candidates = [r for r in scored_results
                          if r.get("family") == family and r.get("consistent") is False
                          and r.get("response_a") and r.get("response_b")]
        ex = next((c for c in candidates if c.get("model") not in used_models), None)
        if ex is None and candidates:
            ex = candidates[0]
        if not ex:
            continue

        used_models.add(ex.get("model", ""))
        entry = {
            "family": family, "family_label": config.FAMILY_LABELS.get(family, family),
            "model": ex.get("model", ""), "model_label": config.MODEL_LABELS.get(ex.get("model", ""), ""),
            "domain": ex.get("domain", ""), "difficulty": ex.get("difficulty", ""),
            "prompt_a": ex.get("prompt_a", ""), "prompt_b": ex.get("prompt_b", ""),
            "response_a": (ex.get("response_a", "") or "")[:250],
            "response_b": (ex.get("response_b", "") or "")[:250],
            "logical_constraint": ex.get("logical_constraint", ""),
            "reason": ex.get("score_reason", ""),
        }
        examples.append(entry)

        print(f"\n[{config.FAMILY_LABELS.get(family, family).upper()} / {ex.get('domain', '')} / "
              f"{ex.get('difficulty', '')}]")
        print(f"  Model:  {config.MODEL_LABELS.get(ex.get('model', ''), ex.get('model', ''))}")
        print(f"  Constraint: {ex.get('logical_constraint', '')}")
        print(f"  A: {ex.get('prompt_a', '')}")
        print(f"  -> {(ex.get('response_a', '') or '')[:120]}...")
        print(f"  B: {ex.get('prompt_b', '')}")
        print(f"  -> {(ex.get('response_b', '') or '')[:120]}...")
        print(f"  Violation: {ex.get('score_reason', '')}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "qualitative_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(examples)} examples.")
    return examples


if __name__ == "__main__":
    from ..client import load_checkpoint

    config.ensure_dirs()
    scored = load_checkpoint("cb_scored.json")
    if not scored:
        raise SystemExit("No cb_scored.json checkpoint found — run scoring first.")
    collect_qualitative_examples(scored)
