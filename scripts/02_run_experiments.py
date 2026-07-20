#!/usr/bin/env python3
"""
Stage 2 (Part A): Run every evaluated API model through the EvaluationHarness
on the full probe set.

Usage:
    export OPENROUTER_API_KEY=sk-...
    python scripts/02_run_experiments.py [--models gpt4o claude_opus ...]

Resumable per-model: each model gets its own harness checkpoint
(data/checkpoints/harness_<model>_<run_id>.json), so an interrupted run can
be restarted and will only query the remaining (probe, model) pairs.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from consistencybench import config  # noqa: E402
from consistencybench.client import load_checkpoint, save_checkpoint  # noqa: E402
from consistencybench.harness import APIBackend, EvaluationHarness  # noqa: E402

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the following question directly "
    "and concisely. When the question has a clear yes or no answer, begin your "
    "response with 'Yes' or 'No' followed by a brief explanation of 1-3 sentences. "
    "Do not hedge unnecessarily."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                         help="Subset of config.MODELS keys to run (default: all 16).")
    args = parser.parse_args()

    config.ensure_dirs()
    all_probes = load_checkpoint("cb_probes.json")
    if not all_probes:
        raise SystemExit("No probe dataset found. Run scripts/01_run_probegen.py first.")

    model_names = args.models or list(config.MODELS.keys())
    raw_results = load_checkpoint("cb_raw_responses.json") or []
    done_pairs = {(r["probe_id"], r["model"]) for r in raw_results}

    for model_name in model_names:
        model_id = config.MODELS[model_name]
        remaining = [p for p in all_probes if (p["probe_id"], model_name) not in done_pairs]
        if not remaining:
            print(f"[{model_name}] already complete, skipping.")
            continue

        backend = APIBackend(model_id)
        harness = EvaluationHarness(
            backend=backend, system_prompt=SYSTEM_PROMPT,
            checkpoint_dir=config.BASE_DIR / "checkpoints", seed=42,
        )
        model_results = harness.run(remaining, delay=0.5, checkpoint_every=config.CHECKPOINT_EVERY,
                                     tag=model_name)
        for r in model_results:
            r["model"] = model_name  # normalize back to short name
        raw_results.extend(model_results)
        save_checkpoint(raw_results, "cb_raw_responses.json")
        print(f"[{model_name}] done.\n")

    save_checkpoint(raw_results, "cb_raw_responses.json")
    print(f"\nAll experiments complete: {len(raw_results):,} results.")


if __name__ == "__main__":
    main()
