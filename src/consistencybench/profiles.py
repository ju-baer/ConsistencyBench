"""
Consistency Profiles (5D CP Vectors per Model).

A scalar Inconsistency Rate (IR) collapses a rich failure structure into one
number. Two models with identical overall IR can fail on completely different
transformation families. This module computes the 5-dimensional Consistency
Profile for every model, the pairwise profile-distance matrix, and the
cross-model rank-correlation check that motivates treating family difficulty
as a property of the task rather than of any individual model.

Direct, environment-independent port of Notebook Cell 10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from . import config


def build_results_dataframe(scored_results: list[dict]) -> pd.DataFrame:
    """Turn the flat list of scored results into the tidy per-probe dataframe
    used by every downstream analysis stage (profiles, scaling, calibration,
    figures, statistics)."""
    df = pd.DataFrame([
        {"model": r["model"], "family": r["family"], "domain": r["domain"],
         "difficulty": r["difficulty"], "delta": r.get("delta", 0.0),
         "consistent": r["consistent"], "ir": float(not r["consistent"]) * 100,
         "method": r.get("score_method", "")}
        for r in scored_results if r.get("consistent") is not None
    ])
    assert len(df) > 0, "No scored results yet — run scoring first."
    df["model_label"] = df["model"].map(config.MODEL_LABELS).fillna(df["model"])
    return df


def compute_cp(model: str, df: pd.DataFrame) -> np.ndarray:
    """5D Consistency Profile vector: mean IR per transformation family."""
    return np.array([
        df[(df["model"] == model) & (df["family"] == f)]["ir"].mean()
        if len(df[(df["model"] == model) & (df["family"] == f)]) > 0 else 0.0
        for f in config.TRANSFORMATION_FAMILIES
    ])


def compute_all_profiles(df: pd.DataFrame) -> dict[str, np.ndarray]:
    models = [m for m in config.MODELS if m in df["model"].unique()]
    return {m: compute_cp(m, df) for m in models}


def profile_distance_matrix(cp_vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Euclidean distance between model Consistency Profiles."""
    models = list(cp_vectors.keys())
    mat = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1 in models:
        for m2 in models:
            mat.loc[m1, m2] = euclidean(cp_vectors[m1], cp_vectors[m2])
    return mat


def family_rank_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation of per-family IR across models, to test whether
    family difficulty is a property of the task (high correlation) rather than
    of any individual model."""
    pivot = df.groupby(["model", "family"])["ir"].mean().unstack()
    return pivot.corr(method="spearman")
