"""
Inter-Annotator Agreement (Krippendorff alpha + Cohen kappa).

RBS (Rule-Based Scoring) and LJS (LLM-as-Judge) are two independent scoring
methods used for different transformation families. This module asks a
narrower validity question than cross-judge validation in `scoring.py`: on
probes that RBS *can* score deterministically (Reversal, Ordering,
Complement), does an LLM judge shown the same prompt/response pair reach the
same verdict? High agreement here is evidence that RBS is not silently
disagreeing with what a careful reader would conclude; it is also evidence
that LJS -- the method relied on for Composition and Equivalence, where no
rule-based extraction is possible -- is trustworthy on the subset where it
can be checked against ground truth.

Direct, environment-independent port of Notebook Cell 12.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from . import config
from .scoring import ljs_score


def compute_iaa(
    scored_results: list[dict],
    judge: str = config.GENERATOR_MODEL,
    sample_size: int = 50,
    out_dir: str | Path = config.BASE_DIR / "scoring",
) -> dict | None:
    """Re-score a sample of RBS-scored results with an LLM judge and compute
    agreement, Cohen's kappa, and Krippendorff's alpha between the two
    scoring methods, overall and broken down per transformation family.
    """
    import krippendorff
    from sklearn.metrics import cohen_kappa_score

    print("Computing inter-annotator agreement on RBS/LJS overlap set...")
    rbs_scored = [r for r in scored_results
                  if r.get("score_method") == "rbs" and r.get("consistent") is not None]

    rbs_labels: list[int] = []
    ljs_labels: list[int] = []
    per_family = {f: {"r": [], "l": []} for f in config.TRANSFORMATION_FAMILIES}

    for result in tqdm(rbs_scored[:sample_size], desc="IAA re-score"):
        ljs = ljs_score(result, judge=judge)
        if ljs.get("consistent") is not None:
            rv = 1 if result["consistent"] else 0
            lv = 1 if ljs["consistent"] else 0
            rbs_labels.append(rv)
            ljs_labels.append(lv)
            f = result.get("family", "")
            if f in per_family:
                per_family[f]["r"].append(rv)
                per_family[f]["l"].append(lv)
        time.sleep(0.3)

    if len(rbs_labels) < 10:
        print("Not enough overlapping samples for IAA.")
        return None

    kappa = cohen_kappa_score(rbs_labels, ljs_labels)
    alpha = krippendorff.alpha(np.array([rbs_labels, ljs_labels]), level_of_measurement="nominal")
    agree = float(np.mean(np.array(rbs_labels) == np.array(ljs_labels)))

    print(f"\nIAA (RBS vs LJS, n={len(rbs_labels)})")
    print(f"  Agreement:          {agree:.3f} ({agree * 100:.1f}%)")
    print(f"  Cohen kappa:        {kappa:.3f}")
    print(f"  Krippendorff alpha: {alpha:.3f}")
    print("  Per-family rates:")
    per_family_rates = {}
    for f, v in per_family.items():
        if len(v["r"]) >= 3:
            fa = float(np.mean(np.array(v["r"]) == np.array(v["l"])))
            per_family_rates[f] = fa
            print(f"    {config.FAMILY_LABELS.get(f, f):15s}: {fa * 100:.1f}%")

    stats = {
        "n": len(rbs_labels), "agreement": agree, "kappa": float(kappa),
        "alpha": float(alpha), "per_family_agreement": per_family_rates,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "iaa_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    from .client import load_checkpoint

    config.ensure_dirs()
    scored = load_checkpoint("cb_scored.json")
    if not scored:
        raise SystemExit("No cb_scored.json checkpoint found — run scoring first.")
    compute_iaa(scored)
