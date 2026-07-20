"""
Part B, Stage 2 — Shortcut Probing: Where Does Inconsistency Become Linearly
Decodable?

If a model's inconsistency were purely random noise, no direction in
activation space should predict it. If instead the model is relying on a
**shortcut** — a surface-level heuristic that works most of the time but
silently breaks on hard probes — we'd expect that shortcut to leave a linear
trace in the residual stream: a direction along which "this response is about
to be inconsistent" is decodable well before the model finishes generating.

A logistic-regression probe is trained at every layer to predict the binary
inconsistency label directly from that layer's residual-stream activation at
the final prompt token — the standard linear-probing methodology from the
interpretability literature (Alain & Bengio, 2017), used here as a
lightweight, always-available substitute for a pretrained Sparse Autoencoder
(SAE), which does not currently exist for this exact checkpoint. The **probe
weight vector itself** doubles as an interpretable "shortcut direction" and is
reused directly as the intervention vector in `causal_tests.py`, following the
same diff-in-means / linear-direction methodology used in activation-steering
work (Turner et al., 2023) when a trained SAE isn't available.

A family-conditioned breakdown is also reported: is inconsistency equally
decodable for every transformation family, or are some families' failures
more "shortcut-like" (early, linearly obvious) than others (late, only
decodable near the output layer, suggesting inconsistency emerges from
accumulated computation rather than a single shortcut feature)?

Direct, environment-independent port of Notebook Cell 17.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

from .. import config

N_FOLDS = 5


def train_layer_probes(
    X_by_layer: dict[int, np.ndarray], y: np.ndarray, n_layers: int,
) -> tuple[pd.DataFrame, dict, int, float]:
    """Train a logistic-regression probe at every layer via k-fold CV. Returns
    (df_layers, probe_weight_vectors, best_layer, chance_rate)."""
    layer_results = []
    probe_weight_vectors: dict[int, dict] = {}  # layer -> trained probe's weight vector

    print("Training per-layer linear probes to decode inconsistency from activations...\n")
    print(f"{'Layer':>6s} {'Test Acc':>10s} {'AUROC':>8s} {'vs. chance':>12s}")
    print("-" * 42)

    chance_rate = max(y.mean(), 1 - y.mean())  # majority-class baseline

    for l in range(n_layers + 1):
        X_l = X_by_layer[l]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_l)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        cv_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")

        y_proba = cross_val_predict(clf, X_scaled, y, cv=skf, method="predict_proba")[:, 1]
        auroc = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5

        # Fit on full data to extract the probe direction for causal_tests.py
        clf.fit(X_scaled, y)
        probe_weight_vectors[l] = {
            "weight": clf.coef_[0].copy(), "scaler_mean": scaler.mean_.copy(),
            "scaler_scale": scaler.scale_.copy(),
        }

        mean_acc = cv_scores.mean()
        layer_results.append({"layer": l, "test_acc": mean_acc, "test_acc_std": cv_scores.std(),
                               "auroc": auroc, "above_chance": mean_acc - chance_rate})
        marker = " <-- best" if mean_acc == max(r["test_acc"] for r in layer_results) else ""
        print(f"{l:>6d} {mean_acc:>10.3f} {auroc:>8.3f} {mean_acc - chance_rate:>+11.3f}{marker}")

    df_layers = pd.DataFrame(layer_results)
    best_layer = int(df_layers.loc[df_layers["test_acc"].idxmax(), "layer"])
    best_acc = df_layers["test_acc"].max()
    print(f"\nChance baseline (majority class): {chance_rate:.3f}")
    print(f"Best layer: {best_layer} (test accuracy {best_acc:.3f}, "
          f"{best_acc - chance_rate:+.3f} above chance)")

    out_dir = config.BASE_DIR / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_layers.to_csv(out_dir / "layer_probe_results.csv", index=False)

    return df_layers, probe_weight_vectors, best_layer, chance_rate


def family_conditioned_decodability(
    X_by_layer: dict[int, np.ndarray], y: np.ndarray, fam_arr: np.ndarray, best_layer: int,
) -> pd.DataFrame:
    """Is inconsistency equally decodable for every transformation family at
    the best layer?"""
    print("\nFamily-conditioned decodability at the best layer:")
    X_best = X_by_layer[best_layer]
    family_probe_results = []
    for fam in config.TRANSFORMATION_FAMILIES:
        mask = fam_arr == fam
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            continue
        scaler_f = StandardScaler()
        X_f = scaler_f.fit_transform(X_best[mask])
        y_f = y[mask]
        n_splits = min(N_FOLDS, int(y_f.sum()), int((1 - y_f).sum()) + 1) if y_f.sum() > 1 else 2
        skf_f = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            acc_f = cross_val_score(
                LogisticRegression(max_iter=2000, class_weight="balanced"),
                X_f, y_f, cv=skf_f, scoring="accuracy",
            ).mean()
        except Exception:
            acc_f = float("nan")
        family_probe_results.append({"family": fam, "n": int(mask.sum()),
                                      "inconsistent_rate": float(y_f.mean()), "probe_acc": acc_f})
        print(f"  {config.FAMILY_LABELS[fam]:15s} n={mask.sum():3d}  IR={y_f.mean() * 100:5.1f}%  "
              f"probe_acc={acc_f:.3f}")
    df_fam = pd.DataFrame(family_probe_results)
    out_dir = config.BASE_DIR / "interpretability"
    df_fam.to_csv(out_dir / "family_probe_results.csv", index=False)
    return df_fam


def pca_pseudo_features(X_by_layer: dict[int, np.ndarray], y: np.ndarray, best_layer: int) -> pd.DataFrame:
    """PCA-derived "pseudo-SAE" interpretable directions (documented proxy).

    No pretrained SAE exists for this checkpoint. As a lightweight,
    always-available substitute, the top principal components of the best
    layer's activations are extracted and checked for which components
    separate consistent from inconsistent examples — analogous in spirit to
    inspecting individual SAE feature directions, but using an orthogonal
    basis computed on the fly rather than a learned overcomplete dictionary.
    """
    print(f"\nPCA-based pseudo-feature analysis at layer {best_layer}...")
    X_best = X_by_layer[best_layer]
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_best))

    component_separations = []
    for comp_idx in range(20):
        comp_vals = X_pca[:, comp_idx]
        mean_incons = comp_vals[y == 1].mean()
        mean_consis = comp_vals[y == 0].mean()
        pooled_std = comp_vals.std()
        sep = abs(mean_incons - mean_consis) / (pooled_std + 1e-8)
        component_separations.append({"component": comp_idx, "separation_d": sep,
                                       "variance_explained": float(pca.explained_variance_ratio_[comp_idx])})
    df_pca = pd.DataFrame(component_separations).sort_values("separation_d", ascending=False)
    print("Top 5 PCA components by consistent/inconsistent separation (Cohen's d):")
    print(df_pca.head(5).to_string(index=False))
    out_dir = config.BASE_DIR / "interpretability"
    df_pca.to_csv(out_dir / "pca_pseudo_features.csv", index=False)
    return df_pca


def save_probe_directions(probe_weight_vectors: dict, best_layer: int) -> None:
    out_dir = config.BASE_DIR / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "probe_directions.npz",
        best_layer=best_layer,
        **{f"weight_layer_{l}": v["weight"] for l, v in probe_weight_vectors.items()},
        **{f"mean_layer_{l}": v["scaler_mean"] for l, v in probe_weight_vectors.items()},
        **{f"scale_layer_{l}": v["scaler_scale"] for l, v in probe_weight_vectors.items()},
    )
    print(f"\nSaved layer probe results, family breakdown, PCA features, and probe "
          f"directions to {out_dir}/")


def run_shortcut_probing(
    X_by_layer: dict[int, np.ndarray], y: np.ndarray, fam_arr: np.ndarray, n_layers: int,
):
    """Full Cell-17 pipeline: layer probes -> family breakdown -> PCA -> save."""
    df_layers, probe_weight_vectors, best_layer, chance_rate = train_layer_probes(X_by_layer, y, n_layers)
    df_fam = family_conditioned_decodability(X_by_layer, y, fam_arr, best_layer)
    df_pca = pca_pseudo_features(X_by_layer, y, best_layer)
    save_probe_directions(probe_weight_vectors, best_layer)
    return {
        "df_layers": df_layers, "df_family": df_fam, "df_pca": df_pca,
        "probe_weight_vectors": probe_weight_vectors, "best_layer": best_layer,
        "chance_rate": chance_rate,
    }
