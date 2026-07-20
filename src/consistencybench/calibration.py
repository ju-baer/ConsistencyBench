"""
Consistency-Calibration Score (CCS).

CCS is Expected Calibration Error (ECE) adapted for consistency: does a
model's expressed confidence predict whether it will actually be
inconsistent? A model can have high Inconsistency Rate (IR) but still be
well-calibrated (its own low-confidence answers are the ones that fail) —
that's a meaningfully safer deployment profile than the same IR with no such
signal.

Direct, environment-independent port of Notebook Cell 14.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

LJS_CONF_MAP = {"high": 0.90, "medium": 0.65, "low": 0.40}
HEDGE = ["it depends", "generally", "typically", "usually", "in some", "could be",
         "might", "may ", "often", "sometimes", "context", "however", "although"]


def rbs_confidence(result: dict) -> float:
    for field in ("response_a", "response_b"):
        text = (result.get(field) or "").strip().lower()
        if not text:
            return 0.50
        if re.match(r"^(yes|no)[.,!]?\s", text):
            continue
        if any(h in text[:120] for h in HEDGE):
            return 0.55
    return 0.80


def extract_confidence(result: dict) -> float | None:
    consistent = result.get("consistent")
    if consistent is None:
        return None
    method = result.get("score_method", "")
    if method == "ljs":
        base = LJS_CONF_MAP.get(result.get("ljs_confidence", "medium"), 0.65)
        return base if not consistent else 1.0 - base
    if method == "rbs":
        conf = rbs_confidence(result)
        return conf if not consistent else 1.0 - conf
    return None


def compute_ccs(p_pred: list[float], y_true: list[float], n_bins: int = 10) -> tuple[float, list[dict]]:
    """Weighted-binned calibration error between predicted P(inconsistent) and
    observed inconsistency rate — the consistency analogue of ECE."""
    p_pred_arr, y_true_arr = np.array(p_pred, dtype=float), np.array(y_true, dtype=float)
    if len(p_pred_arr) == 0:
        return 0.0, []
    bins = np.linspace(0, 1, n_bins + 1)
    ccs, bin_data = 0.0, []
    for i in range(n_bins):
        mask = (p_pred_arr >= bins[i]) & ((p_pred_arr <= bins[i + 1]) if i == n_bins - 1 else (p_pred_arr < bins[i + 1]))
        n_in = int(mask.sum())
        if n_in == 0:
            bin_data.append({"bin_lower": float(bins[i]), "bin_upper": float(bins[i + 1]), "n": 0})
            continue
        mp = float(p_pred_arr[mask].mean())
        ma = float(y_true_arr[mask].mean())
        err = abs(mp - ma)
        wt = n_in / len(p_pred_arr)
        ccs += wt * err
        bin_data.append({"bin_lower": float(bins[i]), "bin_upper": float(bins[i + 1]),
                          "n": n_in, "mean_pred": mp, "mean_actual": ma, "error": err, "weight": wt})
    return ccs, bin_data


def compute_ccs_for_all_models(scored_results: list[dict]) -> tuple[pd.DataFrame, dict, dict]:
    cal_rows = []
    for r in scored_results:
        p = extract_confidence(r)
        if p is None:
            continue
        cal_rows.append({"model": r["model"], "family": r["family"], "domain": r["domain"],
                          "difficulty": r["difficulty"], "score_method": r.get("score_method", ""),
                          "inconsistent": int(not r["consistent"]), "p_inconsistent": p,
                          "ljs_confidence": r.get("ljs_confidence", "")})

    df_cal = pd.DataFrame(cal_rows) if cal_rows else pd.DataFrame()
    ccs_scores, bin_data_all = {}, {}
    if len(df_cal) > 0:
        print(f"{'Model':<22s} {'N':>6s} {'IR%':>7s} {'CCS':>8s}")
        print("-" * 48)
        for model in config.MODELS:
            mdf = df_cal[df_cal["model"] == model]
            if len(mdf) < 50:
                continue
            ccs, bins = compute_ccs(mdf["p_inconsistent"].values, mdf["inconsistent"].values)
            ccs_scores[model] = ccs
            bin_data_all[model] = bins
            ir = mdf["inconsistent"].mean() * 100
            print(f"{config.MODEL_LABELS.get(model, model):<22s} {len(mdf):>6d} {ir:>6.1f}% {ccs:>8.4f}")
    return df_cal, ccs_scores, bin_data_all
