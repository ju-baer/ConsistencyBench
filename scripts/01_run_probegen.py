#!/usr/bin/env python3
"""
Stage 1: Generate the full ConsistencyBench probe dataset with ProbeGen.

Usage:
    export OPENROUTER_API_KEY=sk-...
    python scripts/01_run_probegen.py [--quick-test]

Resumable: re-running this script picks up from data/checkpoints/cb_probes.json
and only generates the (family, domain, difficulty) combinations not yet done.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from consistencybench import config  # noqa: E402
from consistencybench.probegen import generate_full_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-test", action="store_true",
                         help="Generate 5 probes/combo instead of 100 (for a fast smoke test).")
    args = parser.parse_args()

    if args.quick_test:
        os.environ["CONSISTENCYBENCH_QUICK_TEST"] = "1"
        import importlib
        importlib.reload(config)

    config.ensure_dirs()
    print(config.summarize())
    probes = generate_full_dataset()
    print(f"\nDone. {len(probes):,} probes saved to {config.BASE_DIR / 'checkpoints' / 'cb_probes.json'}")


if __name__ == "__main__":
    main()
