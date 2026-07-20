"""
Scaling Analysis: IR Stability vs. Dataset Size.

Answers a practical reproducibility question: how many probes does a new
model actually need to be evaluated on before its overall Inconsistency Rate
estimate stabilizes? Bootstraps sub-samples of increasing size for every
model and reports the smallest sample size at which the bootstrapped mean IR
converges to within +/-1.5 percentage points of the full-dataset IR. This
number is what a practitioner should use as a minimum evaluation budget when
adapting ConsistencyBench to a new model on a limited compute/API budget.

Direct, environment-independent port of Notebook Cell 13.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def run_scaling_analysis(
    df: pd.DataFrame,
    n_boot: int = 30,
    convergence_threshold_pp: float = 1.5,
    out_dir: str | Path = config.BASE_DIR / "results",
) -> pd.DataFrame:
    """Bootstrap IR estimates at increasing sample sizes for every model and
    report the convergence point (smallest n within `convergence_threshold_pp`
    of the full-dataset IR).
    """
    models = [m for m in config.MODELS if m in df["model"].unique()]
    max_per = df.groupby("model").size().min() if len(df) > 0 else 100
    candidate_sizes = [100, 200, 300, 500, 600, 800, 1000, 1500, 2000, 3000, int(max_per)]
    sample_sizes = [n for n in candidate_sizes if n <= max_per]

    rows = []
    for model in models:
        mdf = df[df["model"] == model]
        full_ir = mdf["ir"].mean()
        for n in sample_sizes:
            if n > len(mdf):
                continue
            boot = [mdf.sample(n=n, replace=False)["ir"].mean() for _ in range(n_boot)]
            rows.append({
                "model": model, "n": n,
                "mean": np.mean(boot), "ci95": 1.96 * np.std(boot),
                "full_ir": full_ir, "delta": abs(np.mean(boot) - full_ir),
            })

    df_scaling = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_scaling.to_csv(out_dir / "scaling_analysis.csv", index=False)

    print(f"Convergence at +/-{convergence_threshold_pp}% of full-dataset IR:")
    for model in models[:6]:
        conv = df_scaling[(df_scaling["model"] == model) & (df_scaling["delta"] <= convergence_threshold_pp)]
        if len(conv) > 0:
            print(f"  {config.MODEL_LABELS.get(model, model):22s}: {conv['n'].min()} probes")

    return df_scaling


if __name__ == "__main__":
    from .client import load_checkpoint
    from .profiles import build_results_dataframe

    config.ensure_dirs()
    scored = load_checkpoint("cb_scored.json")
    if not scored:
        raise SystemExit("No cb_scored.json checkpoint found — run scoring first.")
    run_scaling_analysis(build_results_dataframe(scored))
