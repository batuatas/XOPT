#!/usr/bin/env python3
"""
build_figures_v4.py

Builds Figures 1–12 from scenario_results_v4.csv and scenario_results_v4_akif.csv.
Saves each figure as both .png and .pdf in workspace_v4/reports/figures/.

Design rules:
  Background: #FFFFFF, ink: #1A1A1A, blue: #1F6FBF, gold: #C8780A, grey: #888888
  No rainbow palettes. Label axes with units. Show n= counts.
  16:9 aspect ratio, 150 DPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE   = Path(__file__).resolve().parent.parent.parent
REPORTS     = WORKSPACE / "reports"
FIGURES     = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------
BG      = "#FFFFFF"
INK     = "#1A1A1A"
BLUE    = "#1F6FBF"
GOLD    = "#C8780A"
GREY    = "#888888"
RED     = "#C0392B"
GREEN   = "#1E7A4E"

REGIME_COLORS = {
    "reflation_risk_on":       BLUE,
    "higher_for_longer":       GOLD,
    "soft_landing":            GREEN,
    "mixed_mid_cycle":         GREY,
    "risk_off_stress":         RED,
    "high_stress_defensive":   "#8B0000",
    "disinflationary_slowdown":"#6A5ACD",
}

def regime_color(label):
    return REGIME_COLORS.get(label, "#444444")

FIGSIZE = (16, 9)
DPI     = 150

def savefig(name: str):
    for ext in ("png", "pdf"):
        path = FIGURES / f"{name}.{ext}"
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {name}")

def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY)
    if title:  ax.set_title(title, color=INK, fontsize=10)
    if xlabel: ax.set_xlabel(xlabel, color=INK)
    if ylabel: ax.set_ylabel(ylabel, color=INK)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    print("Loading data...")
    results    = pd.read_csv(REPORTS / "scenario_results_v4.csv")
    results_ak = pd.read_csv(REPORTS / "scenario_results_v4_akif.csv")
    results_all = pd.concat([results, results_ak], ignore_index=True)
    fm = pd.read_parquet(WORKSPACE / "data_refs" / "feature_master_monthly.parquet")
    fm["month_end"] = pd.to_datetime(fm["month_end"])
    anchor_truth = pd.read_csv(REPORTS / "scenario_anchor_truth_v4.csv")
    print(f"  results: {len(results)} rows | akif: {len(results_ak)} rows | combined: {len(results_all)} rows")
    return results, results_ak, results_all, fm, anchor_truth


# ---------------------------------------------------------------------------
# Figure 1 — G-function landscape
# ---------------------------------------------------------------------------
def fig1_g_landscape(results_all: pd.DataFrame):
    print("Figure 1: G-function landscape...")
    questions = sorted(results_all["question_id"].unique())
    n_q = len(questions)
    n_anchors = results_all["anchor_date"].nunique()

    fig, axes = plt.subplots(n_q, 1, figsize=(16, max(9, n_q * 2.5)), facecolor=BG)
    if n_q == 1:
        axes = [axes]

    for ax, qid in zip(axes, questions):
        grp = results_all[results_all["question_id"] == qid]
        for anchor, agrp in grp.groupby("anchor_date"):
            vals = agrp["G_value"].values
            if len(vals) < 5:
                continue
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 300)
            ax.plot(xs, kde(xs), label=str(anchor), alpha=0.8)
            # G(m0) reference: tau_effective * 5 = G(m0)
            if "tau_effective" in agrp.columns:
                tau = float(agrp["tau_effective"].iloc[0])
                g_m0 = tau * 5.0
                ax.axvline(g_m0, linestyle="--", alpha=0.5, linewidth=1)

        apply_style(ax, title=f"{qid}  (n={len(grp)})", xlabel="G(m) value", ylabel="density")
        ax.legend(fontsize=8, framealpha=0.5)

    fig.suptitle("Figure 1 — G-function landscape by question and anchor", color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig01_g_landscape")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Prior vs scenario macro distributions
# ---------------------------------------------------------------------------
def fig2_prior_vs_scenario(results_all: pd.DataFrame, fm: pd.DataFrame):
    print("Figure 2: Prior vs scenario macro distributions...")
    macro_vars = ["infl_US", "short_rate_US", "vix", "ig_oas"]
    questions  = sorted(results_all["question_id"].unique())
    n_q = len(questions)
    n_v = len(macro_vars)

    # Historical prior: feature_master 2006–2021, macro rows (sleeve_id == first sleeve or deduplicated)
    fm_macro = fm.drop_duplicates(subset=["month_end"]).copy()
    fm_prior = fm_macro[
        (fm_macro["month_end"] >= "2006-01-01") &
        (fm_macro["month_end"] <= "2021-12-31")
    ]

    fig, axes = plt.subplots(n_v, n_q, figsize=(max(16, n_q * 4), n_v * 3.5), facecolor=BG)
    if n_q == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, var in enumerate(macro_vars):
        if var not in fm_prior.columns:
            continue
        prior_vals = fm_prior[var].dropna().values
        for col_idx, qid in enumerate(questions):
            ax = axes[row_idx, col_idx]
            grp = results_all[results_all["question_id"] == qid]
            if var not in grp.columns:
                ax.set_visible(False)
                continue

            scen_vals = grp[var].dropna().values
            anchors   = sorted(grp["anchor_date"].unique())

            # Prior KDE
            if len(prior_vals) > 5:
                kde_prior = gaussian_kde(prior_vals)
                xmin = min(prior_vals.min(), scen_vals.min()) if len(scen_vals) else prior_vals.min()
                xmax = max(prior_vals.max(), scen_vals.max()) if len(scen_vals) else prior_vals.max()
                xs = np.linspace(xmin, xmax, 300)
                ax.plot(xs, kde_prior(xs), color=GREY, linewidth=1.5, label="prior (2006–2021)", zorder=2)

            # Scenario KDE (all anchors pooled)
            if len(scen_vals) > 5:
                kde_scen = gaussian_kde(scen_vals)
                xmin = prior_vals.min() if len(prior_vals) else scen_vals.min()
                xmax = prior_vals.max() if len(prior_vals) else scen_vals.max()
                xs = np.linspace(xmin, xmax, 300)
                ax.plot(xs, kde_scen(xs), color=BLUE, linewidth=1.5, label=f"scenario (n={len(scen_vals)})", zorder=3)

            # Anchor m0 lines
            for anchor in anchors:
                a_grp = grp[grp["anchor_date"] == anchor]
                m0_val = a_grp[var].median()
                ax.axvline(m0_val, color=GOLD, linewidth=0.8, linestyle=":", alpha=0.7)

            title = f"{var}\n{qid.replace('_',' ')}" if row_idx == 0 else var
            apply_style(ax, title=title if col_idx == 0 else (qid.replace("_"," ") if row_idx == 0 else ""),
                        xlabel=var if row_idx == n_v - 1 else "", ylabel="density" if col_idx == 0 else "")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, framealpha=0.5)

    fig.suptitle("Figure 2 — Prior vs scenario macro distributions (gold dashes = anchor m0)", color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig02_prior_vs_scenario")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — MALA diagnostics
# ---------------------------------------------------------------------------
def fig3_mala_diagnostics(results_all: pd.DataFrame):
    print("Figure 3: MALA diagnostics...")
    diag_cols = ["mean_acceptance_rate", "ess_min", "ess_median", "tau_effective", "G_value"]
    for c in diag_cols:
        if c not in results_all.columns:
            print(f"  WARNING: missing column {c}")
            return

    questions = sorted(results_all["question_id"].unique())
    anchors   = sorted(results_all["anchor_date"].unique())

    # Build per-question×anchor diagnostics (take first row per group — these are replicated)
    diag_rows = []
    for (qid, anc), grp in results_all.groupby(["question_id", "anchor_date"]):
        diag_rows.append({
            "question_id":         qid,
            "anchor_date":         anc,
            "mean_acceptance_rate": float(grp["mean_acceptance_rate"].iloc[0]),
            "ess_min":              float(grp["ess_min"].iloc[0]),
            "ess_median":           float(grp["ess_median"].iloc[0]),
            "tau_effective":        float(grp["tau_effective"].iloc[0]),
            "G_m0":                 float(grp["tau_effective"].iloc[0]) * 5.0,
            "n":                    len(grp),
        })
    diag = pd.DataFrame(diag_rows)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, facecolor=BG)

    # Panel 1: Acceptance rate bar
    ax = axes[0]
    ax.axhspan(0.20, 0.50, color=BLUE, alpha=0.08, label="target zone 20–50%")
    for i, (qid, qgrp) in enumerate(diag.groupby("question_id")):
        xs = [i + 0.2 * j for j in range(len(qgrp))]
        ax.bar(xs, qgrp["mean_acceptance_rate"].values, width=0.15,
               color=BLUE, alpha=0.7, label=qid if i == 0 else "")
    ax.axhline(0.20, color=GREY, linewidth=0.8, linestyle="--")
    ax.axhline(0.50, color=GREY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(diag["question_id"].unique())))
    ax.set_xticklabels([q.replace("_", "\n") for q in sorted(diag["question_id"].unique())],
                       fontsize=7)
    apply_style(ax, title="Acceptance Rate", xlabel="question", ylabel="acceptance rate")

    # Panel 2: ESS dot plot
    ax = axes[1]
    for i, (qid, qgrp) in enumerate(diag.groupby("question_id")):
        xs = [i] * len(qgrp)
        ax.scatter(xs, qgrp["ess_min"].values,   color=BLUE,  alpha=0.8, s=50,  label="ESS min"    if i == 0 else "")
        ax.scatter(xs, qgrp["ess_median"].values, color=GOLD,  alpha=0.8, s=50,  marker="^", label="ESS median" if i == 0 else "")
    ax.axhline(20, color=GREY, linewidth=0.8, linestyle="--", label="min threshold=20")
    ax.set_xticks(range(len(diag["question_id"].unique())))
    ax.set_xticklabels([q.replace("_", "\n") for q in sorted(diag["question_id"].unique())], fontsize=7)
    ax.legend(fontsize=8)
    apply_style(ax, title="ESS min / median", xlabel="question", ylabel="effective sample size")

    # Panel 3: tau_effective vs G(m0)
    ax = axes[2]
    for qid, qgrp in diag.groupby("question_id"):
        ax.scatter(qgrp["G_m0"], qgrp["tau_effective"],
                   label=qid.replace("_", " "), alpha=0.8, s=60)
    ax.set_xlabel("G(m0)", color=INK)
    ax.set_ylabel("tau_effective", color=INK)
    apply_style(ax, title="tau_effective vs G(m0)")
    ax.legend(fontsize=7, framealpha=0.5)

    fig.suptitle("Figure 3 — MALA diagnostics per question × anchor", color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig03_mala_diagnostics")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4 — Regime distribution per question × anchor
# ---------------------------------------------------------------------------
def fig4_regime_distribution(results_all: pd.DataFrame):
    print("Figure 4: Regime distribution per question × anchor...")
    questions = sorted(results_all["question_id"].unique())
    anchors   = sorted(results_all["anchor_date"].unique())
    all_regimes = sorted(results_all["regime_label"].unique())

    fig, axes = plt.subplots(1, len(questions), figsize=FIGSIZE, facecolor=BG)
    if len(questions) == 1:
        axes = [axes]

    for ax, qid in zip(axes, questions):
        grp = results_all[results_all["question_id"] == qid]
        bottoms = np.zeros(len(anchors))
        for regime in all_regimes:
            shares = []
            for anchor in anchors:
                sub = grp[grp["anchor_date"] == anchor]
                share = (sub["regime_label"] == regime).mean() if len(sub) else 0.0
                shares.append(share)
            if any(s > 0 for s in shares):
                ax.bar(range(len(anchors)), shares, bottom=bottoms,
                       color=regime_color(regime), label=regime, alpha=0.9)
                bottoms += np.array(shares)

        ax.set_xticks(range(len(anchors)))
        ax.set_xticklabels([str(a)[:10] for a in anchors], rotation=30, fontsize=8)
        apply_style(ax, title=qid.replace("_", "\n"), xlabel="anchor", ylabel="share" if ax == axes[0] else "")

    # Unified legend
    handles = [mpatches.Patch(color=regime_color(r), label=r) for r in all_regimes
               if r in results_all["regime_label"].values]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, framealpha=0.5)
    fig.suptitle("Figure 4 — Regime share by question × anchor", color=INK, fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    savefig("fig04_regime_distribution")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5 — Macro state pairplot
# ---------------------------------------------------------------------------
def fig5_macro_pairplot(results_all: pd.DataFrame, anchor_truth: pd.DataFrame):
    print("Figure 5: Macro pairplot...")
    pairs = [
        ("infl_US", "short_rate_US"),
        ("vix", "ig_oas"),
        ("us_real10y", "unemp_US"),
        ("long_rate_US", "term_slope_US"),
    ]
    anchors = sorted(results_all["anchor_date"].unique())

    n_anchors = len(anchors)
    n_pairs   = len(pairs)

    fig, axes = plt.subplots(n_pairs, n_anchors, figsize=(max(16, n_anchors*4), n_pairs * 3.5),
                             facecolor=BG)
    if n_anchors == 1:
        axes = axes.reshape(-1, 1)
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for row_i, (xvar, yvar) in enumerate(pairs):
        for col_j, anchor in enumerate(anchors):
            ax = axes[row_i, col_j]
            sub = results_all[results_all["anchor_date"] == anchor]
            if xvar not in sub.columns or yvar not in sub.columns:
                ax.set_visible(False)
                continue

            for regime, rgrp in sub.groupby("regime_label"):
                ax.scatter(rgrp[xvar], rgrp[yvar],
                           color=regime_color(regime), alpha=0.3, s=12, rasterized=True)

            # Anchor m0 point
            if not anchor_truth.empty:
                at = anchor_truth[anchor_truth["anchor_date"].astype(str).str[:10] == str(anchor)[:10]]
                if not at.empty:
                    m0x = at[xvar].values[0] if xvar in at.columns else None
                    m0y = at[yvar].values[0] if yvar in at.columns else None
                    if m0x is not None and m0y is not None:
                        ax.scatter([m0x], [m0y], color=INK, s=120, marker="*", zorder=5,
                                   label="m0")

            title = f"{str(anchor)[:10]}" if row_i == 0 else ""
            ylabel = f"{yvar}" if col_j == 0 else ""
            apply_style(ax, title=title, xlabel=xvar if row_i == n_pairs-1 else "", ylabel=ylabel)

    fig.suptitle("Figure 5 — Macro pairplot colored by regime (★ = anchor m0)", color=INK, fontsize=12)
    handles = [mpatches.Patch(color=regime_color(r), label=r)
               for r in sorted(results_all["regime_label"].unique())]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, framealpha=0.5)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    savefig("fig05_macro_pairplot")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6 — Portfolio weight response by regime (Q1)
# ---------------------------------------------------------------------------
def fig6_q1_weights(results_all: pd.DataFrame):
    print("Figure 6: Q1 portfolio weight response by regime...")
    q1 = results_all[results_all["question_id"] == "Q1_gold_favorable"].copy()
    if q1.empty:
        print("  No Q1 data — skipping")
        return

    weight_cols = ["w_ALT_GLD", "w_EQ_US", "w_CR_US_HY"]
    regimes     = sorted(q1["regime_label"].unique())

    fig, axes = plt.subplots(1, len(weight_cols), figsize=FIGSIZE, facecolor=BG)

    for ax, wcol in zip(axes, weight_cols):
        if wcol not in q1.columns:
            ax.set_visible(False)
            continue
        data  = [q1[q1["regime_label"] == r][wcol].values for r in regimes]
        vp = ax.violinplot(data, positions=range(len(regimes)), showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(BLUE)
            body.set_alpha(0.5)
        vp["cmedians"].set_color(GOLD)

        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=7)

        # Anchor m0 reference (median across all Q1 samples as proxy)
        m0_val = q1[wcol].median()
        ax.axhline(m0_val, color=RED, linewidth=1, linestyle="--", label=f"median={m0_val:.3f}")
        ax.legend(fontsize=8)
        apply_style(ax, title=wcol.replace("w_", "w "), xlabel="regime", ylabel="weight")

    fig.suptitle("Figure 6 — Portfolio weights by regime (Q1: gold-favorable)", color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig06_q1_weights")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 7 — Excess vs total return by anchor
# ---------------------------------------------------------------------------
def fig7_excess_vs_total(results_all: pd.DataFrame):
    print("Figure 7: Excess vs total return by anchor...")
    anchors = sorted(results_all["anchor_date"].unique())
    n_a = len(anchors)
    ncols = min(n_a, 4)
    nrows = (n_a + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE, facecolor=BG)
    axes = np.array(axes).flatten()

    for i, anchor in enumerate(anchors):
        ax = axes[i]
        sub = results_all[results_all["anchor_date"] == anchor]
        if "pred_return_excess" not in sub.columns or "pred_return_total" not in sub.columns:
            ax.set_visible(False)
            continue

        for regime, rgrp in sub.groupby("regime_label"):
            ax.scatter(rgrp["pred_return_excess"] * 100, rgrp["pred_return_total"] * 100,
                       color=regime_color(regime), alpha=0.3, s=12, rasterized=True, label=regime)

        rf_med = sub["rf_rate"].median() * 100 if "rf_rate" in sub.columns else 0.0
        # 45-degree line + rf offset
        xlim = ax.get_xlim() if ax.get_xlim() != (0, 1) else \
               (sub["pred_return_excess"].min() * 100 - 0.5, sub["pred_return_excess"].max() * 100 + 0.5)
        xs = np.linspace(sub["pred_return_excess"].min() * 100, sub["pred_return_excess"].max() * 100, 100)
        ax.plot(xs, xs + rf_med, color=GREY, linewidth=1, linestyle="--",
                label=f"y=x+rf (rf={rf_med:.2f}%)")

        apply_style(ax, title=str(anchor)[:10], xlabel="excess return (%)", ylabel="total return (%)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [mpatches.Patch(color=regime_color(r), label=r)
               for r in sorted(results_all["regime_label"].unique())]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, framealpha=0.5)
    fig.suptitle("Figure 7 — Excess vs total return by anchor (dashed = y=x+rf)", color=INK, fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    savefig("fig07_excess_vs_total")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 8 — Regime transition heatmap
# ---------------------------------------------------------------------------
def fig8_transition_heatmap(results_all: pd.DataFrame):
    print("Figure 8: Regime transition heatmap...")
    q_groups = {
        "Q1-Q3 (original)": results_all[results_all["question_id"].str.startswith("Q")
                                        & results_all["question_id"].isin(
                                            ["Q1_gold_favorable", "Q2_ew_deviation", "Q3_house_view_7pct_total"])],
        "Q4-Q8 (Akif)": results_all[results_all["question_id"].str.startswith("Q")
                                    & ~results_all["question_id"].isin(
                                        ["Q1_gold_favorable", "Q2_ew_deviation", "Q3_house_view_7pct_total"])],
    }

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=BG)

    for ax, (group_label, df) in zip(axes, q_groups.items()):
        if df.empty or "anchor_regime" not in df.columns:
            ax.set_visible(False)
            continue

        pivot = df.groupby(["anchor_regime", "regime_label"]).size().unstack(fill_value=0)
        if pivot.empty:
            ax.set_visible(False)
            continue

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if val > 0:
                    ax.text(c, r, str(val), ha="center", va="center",
                            fontsize=6, color=INK if val < pivot.values.max() * 0.6 else "white")
        plt.colorbar(im, ax=ax, shrink=0.6)
        apply_style(ax, title=f"{group_label}", xlabel="scenario regime", ylabel="anchor regime")

    fig.suptitle("Figure 8 — Regime transition heatmap (anchor → scenario)", color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig08_transition_heatmap")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 9 — Q4 diversification landscape
# ---------------------------------------------------------------------------
def fig9_q4_diversification(results_all: pd.DataFrame):
    print("Figure 9: Q4 diversification landscape...")
    q4 = results_all[results_all["question_id"] == "Q4_max_diversification"].copy()
    if q4.empty:
        print("  No Q4 data — skipping")
        return

    if "portfolio_entropy" not in q4.columns:
        print("  Missing portfolio_entropy — skipping")
        return

    regimes = sorted(q4["regime_label"].unique())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=BG)

    # Panel 1: violin by regime
    ax = axes[0]
    data = [q4[q4["regime_label"] == r]["portfolio_entropy"].values for r in regimes]
    vp = ax.violinplot(data, positions=range(len(regimes)), showmedians=True)
    for body in vp["bodies"]:
        body.set_facecolor(BLUE)
        body.set_alpha(0.5)
    vp["cmedians"].set_color(GOLD)
    ax.axhline(np.log(14), color=RED, linewidth=1.2, linestyle="--",
               label=f"log(14)={np.log(14):.3f} (equal weight)")
    ax.axhline(2.0, color=GREY, linewidth=1.0, linestyle=":",
               label="2.0 (practical threshold)")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=7)
    ax.legend(fontsize=8)
    apply_style(ax, title="Portfolio entropy by regime (Q4)", xlabel="regime", ylabel="entropy (nats)")

    # Panel 2: by anchor
    ax = axes[1]
    for anchor, agrp in q4.groupby("anchor_date"):
        vals = agrp["portfolio_entropy"].values
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, kde(xs), label=f"{str(anchor)[:10]} (n={len(vals)})")
    ax.axvline(np.log(14), color=RED, linewidth=1.2, linestyle="--", label=f"log(14)={np.log(14):.3f}")
    ax.axvline(2.0, color=GREY, linewidth=1.0, linestyle=":", label="threshold=2.0")
    ax.legend(fontsize=8)
    apply_style(ax, title="Portfolio entropy KDE by anchor (Q4)", xlabel="entropy (nats)", ylabel="density")

    fig.suptitle("Figure 9 — Q4: Diversification landscape", color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig09_q4_diversification")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 10 — Q5 risk sensitivity map
# ---------------------------------------------------------------------------
def fig10_q5_risk(results_all: pd.DataFrame):
    print("Figure 10: Q5 risk sensitivity map...")
    q5 = results_all[results_all["question_id"] == "Q5_max_risk"].copy()
    if q5.empty:
        print("  No Q5 data — skipping")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG)

    for regime, rgrp in q5.groupby("regime_label"):
        ax.scatter(rgrp["portfolio_risk"] * 100, rgrp["pred_return_total"] * 100,
                   color=regime_color(regime), alpha=0.4, s=18, label=regime, rasterized=True)

    # Approximate efficient frontier (upper envelope via rolling percentile)
    if len(q5) > 20:
        risk_sorted = q5.sort_values("portfolio_risk")
        risk_vals = risk_sorted["portfolio_risk"].values * 100
        ret_vals  = risk_sorted["pred_return_total"].values * 100
        # Rolling max in bins
        bins = np.linspace(risk_vals.min(), risk_vals.max(), 20)
        bin_maxret = []
        bin_risk   = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (risk_vals >= b0) & (risk_vals < b1)
            if mask.sum() > 0:
                bin_maxret.append(ret_vals[mask].max())
                bin_risk.append((b0 + b1) / 2)
        if len(bin_risk) > 2:
            ax.plot(bin_risk, bin_maxret, color=INK, linewidth=1.5,
                    linestyle="--", label="frontier (upper envelope)", zorder=5)

    ax.legend(fontsize=8, framealpha=0.5)
    apply_style(ax, title="Q5 — Risk sensitivity map: portfolio risk vs total return",
                xlabel="portfolio risk (% ann.)", ylabel="predicted total return (% ann.)")
    fig.suptitle("Figure 10 — Q5: Risk/return tradeoff across macro states", color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig10_q5_risk")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 11 — Q6 60/40 proximity
# ---------------------------------------------------------------------------
def fig11_q6_sixty_forty(results_all: pd.DataFrame):
    print("Figure 11: Q6 60/40 proximity...")
    q6 = results_all[results_all["question_id"] == "Q6_sixty_forty"].copy()
    if q6.empty:
        print("  No Q6 data — skipping")
        return

    eq_sleeves   = ["w_EQ_US", "w_EQ_EZ", "w_EQ_JP", "w_EQ_CN", "w_EQ_EM"]
    ficr_sleeves = ["w_FI_UST", "w_FI_EU_GOVT", "w_CR_US_IG", "w_CR_EU_IG", "w_CR_US_HY"]

    eq_cols   = [c for c in eq_sleeves   if c in q6.columns]
    ficr_cols = [c for c in ficr_sleeves if c in q6.columns]

    anchors = sorted(q6["anchor_date"].unique())

    fig, axes = plt.subplots(1, len(anchors), figsize=FIGSIZE, facecolor=BG, sharey=True)
    if len(anchors) == 1:
        axes = [axes]

    eq_colors   = [BLUE, "#4A9EDF", "#7BBFE0", "#A8D4EC", "#C8E6F5"]
    ficr_colors = [GOLD, "#D99A2E", "#DEAF5D", "#E3C480", "#EDD9A3"]

    for ax, anchor in zip(axes, anchors):
        sub = q6[q6["anchor_date"] == anchor]
        if sub.empty:
            ax.set_visible(False)
            continue

        mean_eq   = [sub[c].mean() for c in eq_cols]
        mean_ficr = [sub[c].mean() for c in ficr_cols]

        # Stacked bar for EQ
        bottom = 0.0
        for j, (col, color) in enumerate(zip(eq_cols, eq_colors)):
            val = sub[col].mean()
            ax.bar([0], [val], bottom=[bottom], color=color, alpha=0.9,
                   label=col.replace("w_","") if anchor == anchors[0] else "")
            bottom += val

        # Stacked bar for FI/CR
        bottom = 0.0
        for j, (col, color) in enumerate(zip(ficr_cols, ficr_colors)):
            val = sub[col].mean()
            ax.bar([1], [val], bottom=[bottom], color=color, alpha=0.9,
                   label=col.replace("w_","") if anchor == anchors[0] else "")
            bottom += val

        ax.axhline(0.60, color=BLUE, linewidth=1.2, linestyle="--", label="60%" if anchor == anchors[0] else "")
        ax.axhline(0.40, color=GOLD, linewidth=1.2, linestyle="--", label="40%" if anchor == anchors[0] else "")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["EQ (5)", "FI+CR (5)"], fontsize=9)
        apply_style(ax, title=str(anchor)[:10], ylabel="mean weight" if anchor == anchors[0] else "")

    if anchors:
        axes[0].legend(fontsize=7, loc="upper right", framealpha=0.5)

    fig.suptitle("Figure 11 — Q6: Mean EQ and FI/CR allocation by anchor (ref lines: 60% / 40%)",
                 color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig11_q6_sixty_forty")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 12 — Q7+Q8: Sharpe and equity tilt
# ---------------------------------------------------------------------------
def fig12_q7_q8(results_all: pd.DataFrame):
    print("Figure 12: Q7 Sharpe + Q8 equity tilt...")
    q7 = results_all[results_all["question_id"] == "Q7_max_sharpe_total"].copy()
    q8 = results_all[results_all["question_id"] == "Q8_max_equity_tilt"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=BG)

    # Left: Q7 Sharpe by regime
    if not q7.empty and "sharpe_pred_total" in q7.columns:
        regimes_q7 = sorted(q7["regime_label"].unique())
        data = [q7[q7["regime_label"] == r]["sharpe_pred_total"].values for r in regimes_q7]
        vp = ax1.violinplot(data, positions=range(len(regimes_q7)), showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(BLUE)
            body.set_alpha(0.5)
        vp["cmedians"].set_color(GOLD)
        ax1.axhline(0.6, color=RED, linewidth=1.2, linestyle="--", label="Sharpe=0.6")
        ax1.axhline(0.0, color=GREY, linewidth=0.8, linestyle=":", label="Sharpe=0")
        ax1.set_xticks(range(len(regimes_q7)))
        ax1.set_xticklabels([r.replace("_", "\n") for r in regimes_q7], fontsize=7)
        ax1.legend(fontsize=8)
        apply_style(ax1, title="Q7 — Sharpe (total) by regime", xlabel="regime",
                    ylabel="sharpe_pred_total")
    else:
        ax1.text(0.5, 0.5, "No Q7 data", ha="center", va="center", transform=ax1.transAxes)
        apply_style(ax1, title="Q7 — No data")

    # Right: Q8 w_EQ_US by regime
    if not q8.empty and "w_EQ_US" in q8.columns:
        regimes_q8 = sorted(q8["regime_label"].unique())
        data = [q8[q8["regime_label"] == r]["w_EQ_US"].values for r in regimes_q8]
        vp = ax2.violinplot(data, positions=range(len(regimes_q8)), showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(GOLD)
            body.set_alpha(0.5)
        vp["cmedians"].set_color(BLUE)
        ax2.set_xticks(range(len(regimes_q8)))
        ax2.set_xticklabels([r.replace("_", "\n") for r in regimes_q8], fontsize=7)
        apply_style(ax2, title="Q8 — w_EQ_US by regime", xlabel="regime", ylabel="w_EQ_US")
    else:
        ax2.text(0.5, 0.5, "No Q8 data", ha="center", va="center", transform=ax2.transAxes)
        apply_style(ax2, title="Q8 — No data")

    fig.suptitle("Figure 12 — Q7: risk-adjusted return | Q8: US equity tilt by regime",
                 color=INK, fontsize=12)
    plt.tight_layout()
    savefig("fig12_q7_q8_sharpe_equity")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Building Figures 1–12")
    print("=" * 60)

    results, results_ak, results_all, fm, anchor_truth = load_data()

    fig1_g_landscape(results_all)
    fig2_prior_vs_scenario(results_all, fm)
    fig3_mala_diagnostics(results_all)
    fig4_regime_distribution(results_all)
    fig5_macro_pairplot(results_all, anchor_truth)
    fig6_q1_weights(results_all)
    fig7_excess_vs_total(results_all)
    fig8_transition_heatmap(results_all)
    fig9_q4_diversification(results_all)
    fig10_q5_risk(results_all)
    fig11_q6_sixty_forty(results_all)
    fig12_q7_q8(results_all)

    print("\nAll figures saved to:", FIGURES)


if __name__ == "__main__":
    main()
