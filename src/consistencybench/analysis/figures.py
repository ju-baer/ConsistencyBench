"""
11 Publication-Quality Figures (9 behavioral + 2 interpretability).

Every figure is saved as both PDF (for LaTeX inclusion) and PNG (for the
README / HTML rendering), to `figures/` and to `config.BASE_DIR/figures/`.

  Fig 1  Evaluation Space           MMLU accuracy vs. consistency score quadrant plot
  Fig 2  ProbeGen Framework         4-stage pipeline diagram
  Fig 3  Model x Family Heatmap     IR (%) for every (model, family) cell
  Fig 4  Consistency Profiles       5D radar plots, 8 representative models
  Fig 5  Orthogonality Scatter      MMLU vs IR, with Kendall tau annotation
  Fig 6  Difficulty Curves          IR vs. difficulty with 95% CI bands
  Fig 7  Intervention Results       Baseline/CR/SC/FTSC grouped bar chart
  Fig 8  CCS vs IR Scatter          calibration vs. inconsistency, per model
  Fig 9  Domain Breakdown           IR by domain, grouped bar chart
  Fig 10 Layer Probe Accuracy       (interpretability) decodability by layer
  Fig 11 Steering Dose-Response     (interpretability) IR vs. steering strength

Direct, environment-independent port of Notebook Cells 19 (Cells 38-39 in the
underlying .ipynb JSON).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp

from .. import config

# Reference MMLU accuracies (static, published figures at time of writing) used
# only for the orthogonality analysis (Figs 1 & 5) — NOT part of ConsistencyBench's
# own measurements. Replace with current numbers if reproducing later.
MMLU = {
    "gpt4_1": 90.1, "gpt4o": 88.7, "claude_opus": 88.1, "claude_sonnet": 85.4,
    "grok3": 87.5, "o4_mini": 89.4, "o3_mini": 88.2, "deepseek_r1": 86.1,
    "gemini_flash": 82.3, "gpt4o_mini": 82.0, "llama4": 83.1, "llama33_70b": 83.7,
    "deepseek_v3": 84.0, "qwen3_235b": 85.2, "mistral": 82.6, "phi4": 78.9,
}

MODEL_COLORS = {
    "gpt4_1": "#1a56db", "gpt4o": "#2563EB", "claude_opus": "#D97706",
    "claude_sonnet": "#f59e0b", "grok3": "#7C3AED", "o4_mini": "#9333EA",
    "o3_mini": "#a855f7", "deepseek_r1": "#ec4899", "gemini_flash": "#059669",
    "gpt4o_mini": "#10b981", "llama4": "#DC2626", "llama33_70b": "#ef4444",
    "deepseek_v3": "#0891B2", "qwen3_235b": "#65A30D", "mistral": "#6B7280",
    "phi4": "#374151",
}
GROUP_COLORS = {"Frontier Dense": "#2563EB", "Reasoning": "#9333EA",
                 "Efficient": "#059669", "Open Large": "#DC2626", "Open Small": "#374151"}
REASONING_MODELS = {"o4_mini", "o3_mini", "deepseek_r1"}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
        "figure.dpi": 150,
    })


def _savefig(name: str, out_dir: Path, drive_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        p = out_dir / f"{name}.{fmt}"
        plt.savefig(p, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        shutil.copy(p, drive_dir / f"{name}.{fmt}")
    plt.close()
    print(f"Saved: {name}")


def fig1_evaluation_space(df: pd.DataFrame, models: list[str], out_dir: Path, drive_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between([0, 50], [50, 50], [100, 100], color="green", alpha=0.05)
    ax.fill_between([50, 100], [50, 50], [100, 100], color="blue", alpha=0.05)
    ax.fill_between([0, 50], [0, 0], [50, 50], color="gray", alpha=0.05)
    ax.fill_between([50, 100], [0, 0], [50, 50], color="red", alpha=0.08)
    ax.axvline(75, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.axhline(70, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.text(38, 90, "Coherent but wrong", ha="center", fontsize=9, color="#555")
    ax.text(87, 90, "RELIABLE \u2713", ha="center", fontsize=10, color="#1a56db", fontweight="bold")
    ax.text(38, 30, "Poor", ha="center", fontsize=9, color="#999")
    ax.text(87, 30, "Knowledgeable\nbut unreliable", ha="center", fontsize=9, color="#DC2626")
    for model in models:
        mmlu_v = MMLU.get(model)
        if mmlu_v is None:
            continue
        ir_v = df[df["model"] == model]["ir"].mean()
        ax.scatter(mmlu_v, 100 - ir_v, s=110, color=MODEL_COLORS.get(model, "gray"),
                   zorder=5, edgecolors="white", linewidths=1.2)
        ax.annotate(config.MODEL_LABELS.get(model, model), (mmlu_v, 100 - ir_v),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5)
    ax.set_xlabel("MMLU Accuracy (%)", fontsize=11)
    ax.set_ylabel("Consistency Score (100 - IR%)", fontsize=11)
    ax.set_title("Figure 1: The LLM Evaluation Space\n"
                 "Standard benchmarks only see the horizontal axis. ConsistencyBench adds the vertical.",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig("fig1_evaluation_space", out_dir, drive_dir)


def fig2_probegen_framework(out_dir: Path, drive_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    stages = [
        ("Stage 1\nConstraint\nSpecification", "No LLM\nLogic graph only", "#e8f2ff"),
        ("Stage 2\nSemantic\nInstantiation", "Generator LLM\n(Gemini 2.5 Pro)", "#fff3e0"),
        ("Stage 3\nDifficulty\nCalibration", "Maximize delta\npreserve constraint", "#f3e5f5"),
        ("Stage 4\nConstraint\nVerification", "5 criteria\n4.2% rejected", "#e8f5e9"),
    ]
    for i, (title, sub, color) in enumerate(stages):
        x = 0.12 + i * 0.23
        rect = mpatches.FancyBboxPatch((x, 0.15), 0.19, 0.7, boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="#aaa", linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.095, 0.65, title, ha="center", va="center", fontsize=9,
                fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.095, 0.30, sub, ha="center", va="center", fontsize=7.5,
                color="#555", transform=ax.transAxes)
        if i < 3:
            ax.annotate("", xy=(x + 0.21, 0.5), xytext=(x + 0.195, 0.5), xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#555", lw=2))
    ax.annotate("", xy=(0.12, 0.15), xytext=(0.7, 0.15), xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5, connectionstyle="arc3,rad=-0.3"),
        annotation_clip=False)
    ax.text(0.41, 0.04, "reject (4.2%)", ha="center", fontsize=8, color="#DC2626", transform=ax.transAxes)
    ax.text(0.93, 0.5, "Verified\nProbe Pair", ha="center", va="center", fontsize=9,
            fontweight="bold", transform=ax.transAxes)
    ax.set_title("Figure 2: ProbeGen 4-Stage Constraint-Driven Probe Synthesis Framework",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    _savefig("fig2_probegen_framework", out_dir, drive_dir)


def fig3_heatmap(df: pd.DataFrame, out_dir: Path, drive_dir: Path) -> None:
    pivot = df.groupby(["model", "family"])["ir"].mean().unstack()
    pivot.index = [config.MODEL_LABELS.get(m, m) for m in pivot.index]
    pivot.columns = [config.FAMILY_LABELS.get(c, c) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, linecolor="white",
                vmin=0, vmax=65, cbar_kws={"label": "IR (%)", "shrink": 0.8},
                annot_kws={"size": 9, "weight": "bold"}, ax=ax)
    ax.set_xlabel("Transformation Family", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Figure 3: Inconsistency Rate (%) by Model and Transformation Family\n"
                 "Complement and Equivalence are universally hardest across all evaluated models",
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    _savefig("fig3_heatmap", out_dir, drive_dir)


def fig4_consistency_profiles(
    cp_vectors: dict, df: pd.DataFrame, models: list[str], out_dir: Path, drive_dir: Path,
) -> None:
    fig = plt.figure(figsize=(14, 7))
    fam_labels = [config.FAMILY_LABELS[f] for f in config.TRANSFORMATION_FAMILIES]
    angles = np.linspace(0, 2 * np.pi, len(config.TRANSFORMATION_FAMILIES), endpoint=False).tolist()
    angles += angles[:1]
    preferred = ["claude_opus", "o4_mini", "gpt4_1", "deepseek_r1", "phi4", "llama4", "gemini_flash", "deepseek_v3"]
    rep_models = [m for m in preferred if m in models][:8]
    for idx, model in enumerate(rep_models):
        ax = fig.add_subplot(2, 4, idx + 1, polar=True)
        if model not in cp_vectors:
            continue
        vals = list(cp_vectors[model])
        vals += vals[:1]
        c = MODEL_COLORS.get(model, "gray")
        ax.plot(angles, vals, color=c, linewidth=2)
        ax.fill(angles, vals, color=c, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label[:4] for label in fam_labels], fontsize=7)
        ax.set_ylim(0, 65)
        ax.set_yticks([20, 40, 60])
        ax.set_yticklabels(["20", "40", "60"], fontsize=6)
        overall_ir = df[df["model"] == model]["ir"].mean() if model in df["model"].values else 0
        ax.set_title(f"{config.MODEL_LABELS.get(model, model)}\nIR={overall_ir:.1f}%",
                     fontsize=8, fontweight="bold", pad=8)
    fig.suptitle("Figure 4: Consistency Profiles \u2014 5D Fingerprints per Model\n"
                 "Axes: Comp/Rev/Compl/Ord/Equiv. Outward = more inconsistent.",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig("fig4_consistency_profiles", out_dir, drive_dir)


def fig5_orthogonality(df: pd.DataFrame, models: list[str], out_dir: Path, drive_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in models:
        mmlu_v = MMLU.get(m)
        if not mmlu_v:
            continue
        ir_v = df[df["model"] == m]["ir"].mean()
        ax.scatter(mmlu_v, ir_v, s=110, color=MODEL_COLORS.get(m, "gray"),
                   marker="^" if m in REASONING_MODELS else "o",
                   zorder=5, edgecolors="white", linewidths=1.2)
        ax.annotate(config.MODEL_LABELS.get(m, m), (mmlu_v, ir_v),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5)
    xv = [MMLU[m] for m in models if m in MMLU]
    yv = [df[df["model"] == m]["ir"].mean() for m in models if m in MMLU]
    if len(xv) > 3:
        tau, p_tau = sp.kendalltau(xv, yv)
        ax.text(0.05, 0.95, f"Kendall tau={tau:.2f}, p={p_tau:.2f}", transform=ax.transAxes,
                fontsize=10, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.scatter([], [], marker="o", color="gray", s=80, label="Standard model")
    ax.scatter([], [], marker="^", color="#9333EA", s=80, label="Reasoning model")
    ax.set_xlabel("MMLU Accuracy (%)", fontsize=11)
    ax.set_ylabel("Overall Inconsistency Rate (%)", fontsize=11)
    ax.set_title("Figure 5: IR is Orthogonal to Accuracy", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    _savefig("fig5_orthogonality", out_dir, drive_dir)


def fig6_difficulty_curves(df: pd.DataFrame, models: list[str], out_dir: Path, drive_dir: Path) -> None:
    diffs = [d for d in config.DIFFICULTIES if d in df["difficulty"].unique()]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(diffs))
    for m in models:
        mdf = df[df["model"] == m]
        means = [mdf[mdf["difficulty"] == d]["ir"].mean() for d in diffs]
        cis = [1.96 * sp.sem(mdf[mdf["difficulty"] == d]["ir"].values)
               if len(mdf[mdf["difficulty"] == d]) > 1 else 0 for d in diffs]
        c = MODEL_COLORS.get(m, "gray")
        ax.plot(x, means, marker="o", markersize=6, lw=2, color=c, label=config.MODEL_LABELS.get(m, m))
        ax.fill_between(x, [a - b for a, b in zip(means, cis)], [a + b for a, b in zip(means, cis)],
                         alpha=0.07, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in diffs])
    ax.set_ylabel("IR (%)")
    ax.set_xlabel("Probe Difficulty (by semantic distance delta)")
    ax.set_title("Figure 6: Hard Probes (delta>=0.65) Elicit 2-3x Higher IR", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=4)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    _savefig("fig6_difficulty", out_dir, drive_dir)


def fig7_intervention(df_iv: pd.DataFrame, intervention_models: list[str], out_dir: Path, drive_dir: Path) -> None:
    if len(df_iv) == 0:
        return
    cond_order = ["baseline", "cr", "sc", "ftsc"]
    cond_labels = {"baseline": "Baseline", "cr": "+CR", "sc": "+SC", "ftsc": "+FTSC"}
    cond_colors = {"baseline": "#aaa", "cr": "#f59e0b", "sc": "#2563EB", "ftsc": "#059669"}
    summary_iv = df_iv.groupby(["model", "condition"])["ir"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(intervention_models))
    w = 0.2
    for i, cond in enumerate(cond_order):
        vals = [summary_iv[(summary_iv["model"] == m) & (summary_iv["condition"] == cond)]["ir"].values
                for m in intervention_models]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        ax.bar(x + i * w, vals, w, label=cond_labels[cond], color=cond_colors[cond], alpha=0.85)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([config.MODEL_LABELS.get(m, m) for m in intervention_models])
    ax.set_ylabel("IR% on Hard Probes")
    ax.legend(frameon=False)
    ax.set_title("Figure 7: FTSC Reduces Inconsistency Beyond Generic Reminders", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    _savefig("fig7_intervention", out_dir, drive_dir)


def fig8_ccs_vs_ir(df: pd.DataFrame, ccs_scores: dict, models: list[str], out_dir: Path, drive_dir: Path) -> None:
    if not ccs_scores:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model in models:
        ccs_v = ccs_scores.get(model)
        if ccs_v is None:
            continue
        ir_v = df[df["model"] == model]["ir"].mean()
        ax.scatter(ir_v, ccs_v, s=120, color=MODEL_COLORS.get(model, "gray"), zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(config.MODEL_LABELS.get(model, model), (ir_v, ccs_v),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.set_xlabel("Inconsistency Rate \u2014 IR (%)", fontsize=11)
    ax.set_ylabel("CCS (lower = better calibrated)", fontsize=11)
    ax.set_title("Figure 8: CCS vs IR \u2014 Two Independent Reliability Dimensions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig("fig8_ccs_vs_ir", out_dir, drive_dir)


def fig9_domain_breakdown(df: pd.DataFrame, models: list[str], out_dir: Path, drive_dir: Path) -> None:
    domains = [d for d in config.DOMAINS if d in df["domain"].unique()]
    dom_colors = {"general": "#2563EB", "science": "#059669", "ethics": "#D97706"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    w = 0.25
    offsets = [-w, 0, w][:len(domains)]
    for i, dom in enumerate(domains):
        vals = [df[(df["model"] == m) & (df["domain"] == dom)]["ir"].mean() for m in models]
        ax.bar(x + offsets[i], vals, w, label=dom.capitalize(), color=dom_colors.get(dom, "gray"),
               alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([config.MODEL_LABELS.get(m, m) for m in models], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("IR (%)")
    ax.legend(title="Domain", frameon=False)
    ax.set_title("Figure 9: Ethics Domain Elicits Highest IR Across Models", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    _savefig("fig9_domain", out_dir, drive_dir)


def fig10_layer_probe_accuracy(
    df_layers: pd.DataFrame, best_layer: int, chance_rate: float,
    model_label: str, out_dir: Path, drive_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_layers["layer"], df_layers["test_acc"], marker="o", color="#9333EA", lw=2,
            label="Probe test accuracy")
    ax.axhline(chance_rate, color="gray", ls="--", lw=1.2, label=f"Chance ({chance_rate:.2f})")
    ax.axvline(best_layer, color="#DC2626", ls=":", lw=1.5, label=f"Best layer ({best_layer})")
    ax.fill_between(df_layers["layer"],
                     df_layers["test_acc"] - df_layers["test_acc_std"],
                     df_layers["test_acc"] + df_layers["test_acc_std"], alpha=0.15, color="#9333EA")
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Cross-validated probe accuracy")
    ax.set_title(f"Figure 10: Inconsistency is Linearly Decodable at Layer {best_layer}\n({model_label})",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    _savefig("fig10_layer_probe_accuracy", out_dir, drive_dir)


def fig11_steering_dose_response(
    df_dose: pd.DataFrame, best_layer: int, model_label: str, out_dir: Path, drive_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_dose["alpha"], df_dose["ir_pct"], marker="o", color="#059669", lw=2, markersize=8)
    for _, row in df_dose.iterrows():
        ax.annotate(f"{row['ir_pct']:.0f}%", (row["alpha"], row["ir_pct"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Steering strength (alpha, multiples of unit diff-in-means direction)")
    ax.set_ylabel("IR (%) on originally-inconsistent held-out probes")
    ax.set_title(f"Figure 11: Causal Dose-Response of Diff-in-Means Steering\nLayer {best_layer}, {model_label}",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    _savefig("fig11_steering_dose_response", out_dir, drive_dir)


def generate_all_figures(
    df: pd.DataFrame,
    cp_vectors: dict,
    df_iv: pd.DataFrame | None = None,
    intervention_models: list[str] | None = None,
    ccs_scores: dict | None = None,
    df_layers: pd.DataFrame | None = None,
    best_layer: int | None = None,
    chance_rate: float | None = None,
    df_dose: pd.DataFrame | None = None,
    interp_model_label: str = config.INTERP_MODEL_LABEL,
    out_dir: str | Path = "figures",
    drive_dir: str | Path = config.BASE_DIR / "figures",
) -> None:
    """Generate all 11 figures. Figures whose required inputs are None/empty
    are skipped with a printed notice rather than raising, so this can be run
    at any point in the pipeline (e.g. behavioral-only, before Part B)."""
    _setup_style()
    out_dir, drive_dir = Path(out_dir), Path(drive_dir)
    models = [m for m in config.MODELS if m in df["model"].unique()]

    fig1_evaluation_space(df, models, out_dir, drive_dir)
    fig2_probegen_framework(out_dir, drive_dir)
    fig3_heatmap(df, out_dir, drive_dir)
    fig4_consistency_profiles(cp_vectors, df, models, out_dir, drive_dir)
    fig5_orthogonality(df, models, out_dir, drive_dir)
    fig6_difficulty_curves(df, models, out_dir, drive_dir)

    if df_iv is not None and intervention_models:
        fig7_intervention(df_iv, intervention_models, out_dir, drive_dir)
    else:
        print("Skipping Fig 7 (intervention): no intervention results provided.")

    if ccs_scores:
        fig8_ccs_vs_ir(df, ccs_scores, models, out_dir, drive_dir)
    else:
        print("Skipping Fig 8 (CCS): no calibration scores provided.")

    fig9_domain_breakdown(df, models, out_dir, drive_dir)

    if df_layers is not None and best_layer is not None and chance_rate is not None:
        fig10_layer_probe_accuracy(df_layers, best_layer, chance_rate, interp_model_label, out_dir, drive_dir)
    else:
        print("Skipping Fig 10 (layer probing): Part B not run.")

    if df_dose is not None and best_layer is not None:
        fig11_steering_dose_response(df_dose, best_layer, interp_model_label, out_dir, drive_dir)
    else:
        print("Skipping Fig 11 (steering): Part B not run.")

    print("\nAll available figures saved.")
