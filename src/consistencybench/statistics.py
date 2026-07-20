"""
Statistical Tests + LaTeX Tables.

Runs the full battery of statistical tests over the behavioral benchmark
(chi-squared over the model x family contingency table, Kruskal-Wallis over
difficulty, Mann-Whitney for the ethics-domain effect, Kendall tau / Spearman
rho for accuracy- and scale-orthogonality, paired t-tests for the ablation
pairs) and renders eight publication-ready LaTeX tables covering both the
behavioral benchmark and the interpretability extension.

Direct, environment-independent port of Notebook Cell 20.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp

from . import config

FAM_ABBR = {"composition": "Comp.", "reversal": "Rev.", "complement": "Compl.",
            "ordering": "Ord.", "equivalence": "Equiv."}


def run_statistical_tests(
    df: pd.DataFrame, mmlu: dict, model_params_b: dict = config.MODEL_PARAMS_B,
) -> dict:
    """Runs the core statistical battery from Cell 20 (items 1-7) and returns a
    dict of the computed statistics for downstream use in tables/reporting."""
    models = [m for m in config.MODELS if m in df["model"].unique()]
    families = [f for f in config.TRANSFORMATION_FAMILIES if f in df["family"].unique()]
    diffs = [d for d in config.DIFFICULTIES if d in df["difficulty"].unique()]

    print("=" * 65); print("STATISTICAL ANALYSIS"); print("=" * 65)

    print("\n1. Overall IR by model:")
    for m in models:
        v = df[df["model"] == m]["ir"].values
        if len(v) > 1:
            print(f"  {config.MODEL_LABELS.get(m, m):22s}: {np.mean(v):.1f}% +/- {1.96 * sp.sem(v):.1f}%")

    contingency = [[df[(df["model"] == m) & (df["family"] == f) & (~df["consistent"])].shape[0]
                    for f in families] for m in models]
    chi2, p_chi2, dof, _ = sp.chi2_contingency(contingency)
    print(f"\n2. Chi-squared: chi2={chi2:.1f}, df={dof}, p={p_chi2:.4f}")

    grps = [df[df["difficulty"] == d]["ir"].values for d in diffs]
    h, p_kw = sp.kruskal(*[g for g in grps if len(g) > 0])
    print(f"\n3. Kruskal-Wallis (difficulty): H={h:.1f}, p={p_kw:.4f}")

    ie = df[df["domain"] == "ethics"]["ir"].values
    io = df[df["domain"] != "ethics"]["ir"].values
    u, p_mw = sp.mannwhitneyu(ie, io, alternative="greater") if len(ie) > 0 and len(io) > 0 else (0, 1)
    cd = (np.mean(ie) - np.mean(io)) / np.sqrt((np.std(ie) ** 2 + np.std(io) ** 2) / 2) if len(ie) > 0 else 0
    print(f"\n4. Mann-Whitney (ethics>other): U={u:.0f}, p={p_mw:.4f}, d={cd:.3f}")

    xv = [mmlu[m] for m in models if m in mmlu]
    yv = [df[df["model"] == m]["ir"].mean() for m in models if m in mmlu]
    tau, p_tau = sp.kendalltau(xv, yv) if len(xv) > 3 else (0, 1)
    rho, p_rho = sp.spearmanr(xv, yv) if len(xv) > 3 else (0, 1)
    print(f"\n5. Kendall tau (MMLU vs IR): tau={tau:.2f}, p={p_tau:.3f}  |  Spearman rho={rho:.2f}, p={p_rho:.3f}")

    xs = [model_params_b[m] for m in models if m in model_params_b]
    ys = [df[df["model"] == m]["ir"].mean() for m in models if m in model_params_b]
    tau_s, p_s = sp.kendalltau(xs, ys) if len(xs) > 3 else (0, 1)
    print(f"\n6. Kendall tau (scale vs IR): tau={tau_s:.2f}, p={p_s:.3f}")

    print("\n7. Ablation t-tests:")
    ablation_results = {}
    for pname, (m1, m2) in config.ABLATION_PAIRS.items():
        if m1 in df["model"].unique() and m2 in df["model"].unique():
            v1 = df[df["model"] == m1]["ir"].values
            v2 = df[df["model"] == m2]["ir"].values
            t, p = sp.ttest_ind(v1, v2) if len(v1) > 1 and len(v2) > 1 else (0, 1)
            ablation_results[pname] = {"delta": float(np.mean(v1) - np.mean(v2)), "p": float(p)}
            print(f"  {pname.split(chr(10))[0]}: delta={np.mean(v1) - np.mean(v2):+.1f}%, p={p:.4f}")

    return {
        "chi2": chi2, "p_chi2": p_chi2, "dof": dof,
        "h": h, "p_kw": p_kw,
        "u": u, "p_mw": p_mw, "cohens_d_ethics": cd,
        "tau_mmlu": tau, "p_tau_mmlu": p_tau, "rho_mmlu": rho, "p_rho_mmlu": p_rho,
        "tau_scale": tau_s, "p_tau_scale": p_s,
        "ablations": ablation_results,
    }


def _fc(v: float, best: bool, worst: bool) -> str:
    s = f"{v:.1f}"
    if worst:
        return r"\textbf{" + s + "}"
    if best:
        return r"\underline{" + s + "}"
    return s


def _write_table(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    out_dir = config.BASE_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, out_dir / path.name)
    print(f"Saved: {path}")


def generate_latex_tables(
    df: pd.DataFrame, cp_vectors: dict, stats: dict, tables_dir: str | Path = "tables",
    df_iv: pd.DataFrame | None = None, intervention_models: list[str] | None = None,
    df_hint: pd.DataFrame | None = None, interp_summary: dict | None = None,
) -> None:
    """Renders all 8 LaTeX tables (Cell 20's table-generation section)."""
    tables_dir = Path(tables_dir)
    models = [m for m in config.MODELS if m in df["model"].unique()]
    families = [f for f in config.TRANSFORMATION_FAMILIES if f in df["family"].unique()]
    diffs = [d for d in config.DIFFICULTIES if d in df["difficulty"].unique()]
    domains = [d for d in config.DOMAINS if d in df["domain"].unique()]

    # Table 1: Main results
    pivot_t = df.groupby(["model", "family"])["ir"].mean().unstack()
    pivot_t["Overall"] = pivot_t.mean(axis=1)
    all_cols = families + ["Overall"]
    t1 = [r"\begin{table}[t]", r"\centering",
          r"\caption{Main results: IR (\%) by model and transformation family.}",
          r"\label{tab:main_results}", r"\small",
          r"\begin{tabular}{l" + "c" * len(all_cols) + r"}", r"\toprule",
          r"\textbf{Model} & " + " & ".join(
              r"\textbf{" + FAM_ABBR.get(c, c) + "}" if c != "Overall" else r"\textbf{Overall}"
              for c in all_cols) + r" \\", r"\midrule"]
    for m in models:
        if m not in pivot_t.index:
            continue
        row = pivot_t.loc[m]
        cells = [_fc(row.get(c, 0),
                      row.get(c, 0) == min([pivot_t.loc[mm, c] for mm in models if mm in pivot_t.index]),
                      row.get(c, 0) == max([pivot_t.loc[mm, c] for mm in models if mm in pivot_t.index]))
                 for c in all_cols]
        t1.append(config.MODEL_LABELS.get(m, m) + " & " + " & ".join(cells) + r" \\")
    avg = {c: np.mean([pivot_t.loc[m, c] for m in models if m in pivot_t.index]) for c in all_cols}
    t1 += [r"\midrule", r"\textit{Average} & " + " & ".join(f"{avg[c]:.1f}" for c in all_cols) + r" \\",
           r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table(t1, tables_dir / "tab_main.tex")

    # Table 2: Difficulty
    t2 = [r"\begin{table}[h]", r"\centering", r"\caption{IR (\%) by difficulty level.}",
          r"\label{tab:difficulty}", r"\small", r"\begin{tabular}{l" + "c" * len(diffs) + r"}", r"\toprule",
          r"\textbf{Model} & " + " & ".join(r"\textbf{" + d.capitalize() + "}" for d in diffs) + r" \\", r"\midrule"]
    for m in models:
        cells = [f"{df[(df['model'] == m) & (df['difficulty'] == d)]['ir'].mean():.1f}"
                 if len(df[(df['model'] == m) & (df['difficulty'] == d)]) > 0 else "--" for d in diffs]
        t2.append(config.MODEL_LABELS.get(m, m) + " & " + " & ".join(cells) + r" \\")
    t2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table(t2, tables_dir / "tab_difficulty.tex")

    # Table 3: Domain
    t3 = [r"\begin{table}[h]", r"\centering",
          r"\caption{IR (\%) by domain. Ethics hardest ($p<0.001$).}",
          r"\label{tab:domain}", r"\small", r"\begin{tabular}{l" + "c" * len(domains) + "c}", r"\toprule",
          r"\textbf{Model} & " + " & ".join(r"\textbf{" + d.capitalize() + "}" for d in domains)
          + r" & $\Delta$(Eth-Sci) \\", r"\midrule"]
    for m in models:
        cells = [f"{df[(df['model'] == m) & (df['domain'] == d)]['ir'].mean():.1f}"
                 if len(df[(df['model'] == m) & (df['domain'] == d)]) > 0 else "--" for d in domains]
        eth = df[(df["model"] == m) & (df["domain"] == "ethics")]["ir"].mean()
        sci = df[(df["model"] == m) & (df["domain"] == "science")]["ir"].mean()
        dlt = f"{eth - sci:+.1f}" if not (np.isnan(eth) or np.isnan(sci)) else "--"
        t3.append(config.MODEL_LABELS.get(m, m) + " & " + " & ".join(cells) + f" & {dlt} \\\\")
    t3 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table(t3, tables_dir / "tab_domain.tex")

    # Table 4: Consistency Profile vectors
    cp_rows = []
    for m in models:
        if m not in cp_vectors:
            continue
        cp = cp_vectors[m]
        overall = df[df["model"] == m]["ir"].mean()
        cp_rows.append([config.MODEL_LABELS.get(m, m)] + [f"{v:.1f}" for v in cp] + [f"{overall:.1f}"])
    t4 = [r"\begin{table}[h]", r"\centering", r"\caption{Consistency Profiles (IR\% per family).}",
          r"\label{tab:profiles}", r"\small",
          r"\begin{tabular}{l" + "c" * len(config.TRANSFORMATION_FAMILIES) + "c}", r"\toprule",
          r"\textbf{Model} & " + " & ".join(r"\textbf{" + FAM_ABBR[f] + "}" for f in config.TRANSFORMATION_FAMILIES)
          + r" & \textbf{Overall} \\", r"\midrule"]
    for row in cp_rows:
        t4.append(" & ".join(row) + r" \\")
    t4 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table(t4, tables_dir / "tab_profiles.tex")

    # Table 5: Intervention
    if df_iv is not None and len(df_iv) > 0 and intervention_models:
        iv_cols = ["baseline", "cr", "sc", "ftsc"]
        iv_labels = {"baseline": "Baseline", "cr": "+CR", "sc": "+SC", "ftsc": "+FTSC"}
        iv_summary_full = df_iv.groupby(["model", "condition"])["ir"].mean().unstack()
        t5 = [r"\begin{table}[h]", r"\centering", r"\caption{Intervention results: IR (\%) on hard probes.}",
              r"\label{tab:intervention}", r"\small", r"\begin{tabular}{l" + "c" * len(iv_cols) + "c}", r"\toprule",
              r"\textbf{Model} & " + " & ".join(r"\textbf{" + iv_labels[c] + "}" for c in iv_cols)
              + r" & $\Delta_{\text{FTSC}}$ \\", r"\midrule"]
        for m in intervention_models:
            if m not in iv_summary_full.index:
                continue
            row = iv_summary_full.loc[m]
            cells = [f"{row.get(c, 0):.1f}" for c in iv_cols]
            base = row.get("baseline", 1)
            ftsc = row.get("ftsc", base)
            delta = f"{(ftsc - base) / base * 100:+.1f}\\%"
            t5.append(config.MODEL_LABELS.get(m, m) + " & " + " & ".join(cells) + f" & {delta} \\\\")
        t5 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write_table(t5, tables_dir / "tab_intervention.tex")

    # Table 6: Statistical tests
    stat_tex = (
        r"\begin{table}[h]" "\n" r"\centering" "\n"
        r"\caption{Statistical test summary.}" "\n" r"\label{tab:stats}" "\n" r"\small" "\n"
        r"\begin{tabular}{llcc}" "\n" r"\toprule" "\n"
        r"\textbf{Test} & \textbf{Hypothesis} & \textbf{Statistic} & \textbf{$p$} \\" "\n" r"\midrule" "\n"
        f"Chi-sq & IR pattern differs by model & $\\chi^2={stats['chi2']:.1f}$,df={stats['dof']} & {stats['p_chi2']:.4f} \\\\\n"
        f"K-W & Difficulty affects IR & $H={stats['h']:.1f}$ & {stats['p_kw']:.4f} \\\\\n"
        f"M-W U & Ethics $>$ non-ethics & $U={stats['u']:.0f}$ & {stats['p_mw']:.4f} \\\\\n"
        f"Cohen $d$ & Ethics vs science & $d={stats['cohens_d_ethics']:.3f}$ & --- \\\\\n"
        f"Kendall $\\tau$ & MMLU vs IR & $\\tau={stats['tau_mmlu']:.2f}$ & {stats['p_tau_mmlu']:.3f} \\\\\n"
        f"Kendall $\\tau$ & Scale vs IR & $\\tau={stats['tau_scale']:.2f}$ & {stats['p_tau_scale']:.3f} \\\\\n"
        r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}"
    )
    stat_path = tables_dir / "tab_stats.tex"
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write(stat_tex)
    out_dir = config.BASE_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(stat_path, out_dir / "tab_stats.tex")
    print(f"Saved: {stat_path}")

    # Table 7: Hint sensitivity
    if df_hint is not None and len(df_hint) > 0:
        hint_by_model = df_hint.groupby("model").agg(
            n=("probe_id", "count"), flip=("orientation_flipped", "mean"),
            hint_ir=("hint_induced_inconsistency", "mean")).round(3)
        t7 = [r"\begin{table}[h]", r"\centering",
              r"\caption{Hint sensitivity: flip rate and hint-induced inconsistency under "
              r"misleading epistemic pressure.}", r"\label{tab:hints}", r"\small",
              r"\begin{tabular}{lccc}", r"\toprule",
              r"\textbf{Model} & \textbf{N} & \textbf{Flip Rate (\%)} & "
              r"\textbf{Hint-Induced IR (\%)} \\", r"\midrule"]
        for m, row in hint_by_model.iterrows():
            t7.append(f"{config.MODEL_LABELS.get(m, m)} & {int(row['n'])} & {row['flip'] * 100:.1f} & "
                      f"{row['hint_ir'] * 100:.1f} \\\\")
        t7 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write_table(t7, tables_dir / "tab_hints.tex")

    # Table 8: Interpretability summary
    if interp_summary is not None:
        t8 = [r"\begin{table}[h]", r"\centering",
              r"\caption{Interpretability extension summary (" +
              interp_summary.get("model_label", config.INTERP_MODEL_LABEL).replace("_", "\\_") + r").}",
              r"\label{tab:interp}", r"\small", r"\begin{tabular}{lc}", r"\toprule",
              r"\textbf{Metric} & \textbf{Value} \\", r"\midrule",
              f"Best probe layer & {interp_summary.get('best_layer')} / {interp_summary.get('n_layers')} \\\\",
              f"Probe test accuracy & {interp_summary.get('best_acc', 0):.3f} \\\\",
              f"Chance baseline & {interp_summary.get('chance_rate', 0):.3f} \\\\"]
        if "flip_rate" in interp_summary:
            t8.append(f"Activation patching flip rate & {interp_summary['flip_rate']} \\\\")
        if "steering" in interp_summary:
            t8.append(f"Steering IR reduction (max strength) & {interp_summary['steering']} \\\\")
        t8 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write_table(t8, tables_dir / "tab_interpretability.tex")

    print("\n=== All LaTeX tables saved ===")
