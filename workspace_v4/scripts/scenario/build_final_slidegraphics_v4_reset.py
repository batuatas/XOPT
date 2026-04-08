"""
build_final_slidegraphics_v4_reset.py
HARD RESET — 6-figure editorial conference package.
Run from workspace_v4/:
    python scripts/scenario/build_final_slidegraphics_v4_reset.py
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA    = ROOT / "data_refs"
OUTDIR  = REPORTS / "final_slidegraphics_v4_reset"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM — minimal, restrained, editorial
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor":    "#FFFFFF",
    "axes.facecolor":      "#FFFFFF",
    "axes.grid":           False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    True,
    "axes.spines.bottom":  True,
    "axes.edgecolor":      "#CCCCCC",
    "axes.linewidth":      0.7,
    "axes.titlesize":      12,
    "axes.titlepad":       8,
    "axes.labelsize":      10,
    "axes.labelpad":       5,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "xtick.color":         "#555555",
    "ytick.color":         "#555555",
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "legend.fontsize":     8.5,
    "legend.frameon":      False,
})

# Restrained palette
C = {
    "ink":       "#1A1A1A",   # near-black for text and primary lines
    "charcoal":  "#3A3A3A",   # secondary text
    "mid":       "#888888",   # muted / helper lines
    "light":     "#CCCCCC",   # very muted / grid
    "bg":        "#FFFFFF",
    "blue":      "#1F6FBF",   # one strong blue — benchmark / scenario
    "blue_soft": "#A8C8E8",   # muted blue fill
    "red":       "#B03A2E",   # stress / negative / gap
    "gold":      "#D4820A",   # gold — used only for ALT_GLD
    "gold_soft": "#F0D5A0",   # gold fill / area
}

# Sleeve color function — only two accents, rest muted grey family
def sleeve_color(s):
    if s in ("ALT_GLD",):           return C["gold"]
    if s in ("FI_UST","FI_EU_GOVT"):return C["blue"]
    if s in ("CR_US_HY","CR_US_IG","CR_EU_IG"): return "#6B84A3"
    if s.startswith("EQ"):          return "#4A7A4A"
    return C["mid"]

SLEEVE_SHORT = {
    "EQ_US":"EQ US","EQ_EZ":"EQ EZ","EQ_JP":"EQ JP","EQ_CN":"EQ CN","EQ_EM":"EQ EM",
    "FI_UST":"Treasuries","FI_EU_GOVT":"EU Govt",
    "CR_US_IG":"US IG","CR_EU_IG":"EU IG","CR_US_HY":"US HY",
    "RE_US":"RE","LISTED_RE":"L-RE","LISTED_INFRA":"Infra",
    "ALT_GLD":"Gold",
}

SLEEVES_14 = [
    "EQ_US","EQ_EZ","EQ_JP","EQ_CN","EQ_EM",
    "FI_UST","FI_EU_GOVT",
    "CR_US_IG","CR_EU_IG","CR_US_HY",
    "RE_US","LISTED_RE","LISTED_INFRA","ALT_GLD",
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def save_fig(fig, name):
    png = OUTDIR / f"{name}.png"
    pdf = OUTDIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  {png.name}")
    plt.close(fig)


def spine_off(ax, sides=("top","right","left","bottom")):
    for s in sides:
        ax.spines[s].set_visible(False)


def tag(ax, text, x=0.0, y=1.06, size=10):
    """Panel tag — left-aligned, bold."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=size, fontweight="bold", color=C["ink"],
            va="bottom", ha="left")


def rule(fig, y, x0=0.06, x1=0.94, color=C["light"], lw=0.8):
    """Horizontal rule across figure in figure coordinates."""
    line = mlines.Line2D([x0, x1], [y, y], transform=fig.transFigure,
                         color=color, lw=lw, zorder=10)
    fig.add_artist(line)


def pct(x, _):
    return f"{x:.0f}%"

def pct1(x, _):
    return f"{x:.1f}%"


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def load_data():
    wp = pd.read_csv(REPORTS / "portfolio/v4_allocator_refinement_wealth_paths.csv")
    wp["month_end"] = pd.to_datetime(wp["month_end"])

    alloc = pd.read_csv(REPORTS / "hero_allocation_all_anchors_v4.csv")
    alloc["anchor_date"] = pd.to_datetime(alloc["anchor_date"])

    scenario = pd.read_csv(REPORTS / "scenario_results_v4.csv")
    at       = pd.read_csv(REPORTS / "scenario_anchor_truth_v4.csv")
    at["anchor_date"] = pd.to_datetime(at["anchor_date"])

    pred_val = pd.read_parquet(DATA / "predictions_validation_v4_benchmark.parquet")

    fm = pd.read_parquet(DATA / "feature_master_monthly.parquet")
    fm["month_end"] = pd.to_datetime(fm["month_end"])
    fm_ts = fm.groupby("month_end").first().sort_index()

    return dict(wp=wp, alloc=alloc, scenario=scenario, at=at,
                pred_val=pred_val, fm_ts=fm_ts)


def chain_wealth(df):
    """Build continuous wealth path from anchor-period wealth indexes."""
    result, cum = [], 1.0
    for anc in sorted(df["anchor_month_end"].unique()):
        sub = df[df["anchor_month_end"]==anc].sort_values("month_end").reset_index(drop=True).copy()
        sub["cum_wealth"] = sub["wealth_index"] / sub["wealth_index"].iloc[0] * cum
        cum = sub["cum_wealth"].iloc[-1]
        result.append(sub)
    return pd.concat(result).sort_values("month_end").reset_index(drop=True)


# ---------------------------------------------------------------------------
# FIG 01 — Big Picture
# ---------------------------------------------------------------------------

def fig01(d):
    """Pipeline in one glance. Four steps, one sentence each, no decoration."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    # Title
    fig.text(0.5, 0.90, "From Data to Scenario",
             ha="center", va="top",
             fontsize=22, fontweight="bold", color=C["ink"])
    fig.text(0.5, 0.84, "A 5-year long-horizon allocation framework with macro-state scenario explanation",
             ha="center", va="top",
             fontsize=11, color=C["mid"])

    rule(fig, 0.82)

    # Four steps as clean text columns — no colored boxes
    steps = [
        ("01",
         "Macro Features",
         "303 monthly indicators\n(rates, spreads, inflation,\nmomentum) × 14 sleeves"),
        ("02",
         "AI Prediction",
         "Elastic Net regression\npredicts 5-year annualized\nexcess return per sleeve"),
        ("03",
         "Robust Allocation",
         "Robust MVO converts\npredictions to long-only\nweights with risk control"),
        ("04",
         "Scenario Search",
         "MALA explores plausible\nmacro neighborhoods to\nexplain benchmark behavior"),
    ]

    xs = [0.13, 0.37, 0.62, 0.87]
    for (num, title, body), x in zip(steps, xs):
        # Step number — large, light
        fig.text(x, 0.73, num, ha="center", va="top",
                 fontsize=38, color=C["light"],
                 fontweight="bold")
        # Title
        fig.text(x, 0.66, title, ha="center", va="top",
                 fontsize=13, color=C["ink"], fontweight="bold")
        # Body
        fig.text(x, 0.60, body, ha="center", va="top",
                 fontsize=9.5, color=C["charcoal"],
                 multialignment="center", linespacing=1.6)

    # Connecting arrows — minimal, horizontal at step number center line
    ax_arrow = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
    ax_arrow.axis("off")
    for i in range(len(xs) - 1):
        x0 = xs[i] + 0.075
        x1 = xs[i+1] - 0.075
        ax_arrow.annotate("", xy=(x1, 0.695), xytext=(x0, 0.695),
                     xycoords="axes fraction",
                     textcoords="axes fraction",
                     arrowprops=dict(
                         arrowstyle="-|>",
                         color=C["light"],
                         lw=1.5,
                         mutation_scale=12,
                     ))

    rule(fig, 0.38)

    # Bottom key facts — 6 numbers, clean and minimal
    facts = [
        ("303",    "input features"),
        ("14",     "investable sleeves"),
        ("5Y",     "prediction horizon"),
        ("2368",   "training observations"),
        ("0.76",   "prediction correlation"),
        ("+ 37%",  "vs equal weight  (2015–2025)"),
    ]
    xs_facts = [0.08 + i * 0.155 for i in range(6)]
    for (num, lab), x in zip(facts, xs_facts):
        fig.text(x, 0.33, num, ha="left", va="top",
                 fontsize=24, fontweight="bold", color=C["blue"])
        fig.text(x, 0.25, lab, ha="left", va="top",
                 fontsize=9, color=C["mid"])

    save_fig(fig, "finalgfx_01_big_picture_v4")


# ---------------------------------------------------------------------------
# FIG 02 — Data & Target Design
# ---------------------------------------------------------------------------

def fig02(d):
    """Universe cards + feature groups + 5Y target. Three columns, no decoration."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.93, "Data & Model Design",
             ha="center", va="top",
             fontsize=22, fontweight="bold", color=C["ink"])

    rule(fig, 0.89)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- Column A: Investable Universe ----
    ax.text(0.10, 0.85, "Investable Universe", ha="center", va="top",
            fontsize=13, fontweight="bold", color=C["ink"])
    ax.text(0.10, 0.80, "14 sleeves / 5 groups", ha="center", va="top",
            fontsize=9, color=C["mid"])

    groups = [
        ("Equity",        5, "#4A7A4A"),
        ("Fixed Income",  2, C["blue"]),
        ("Credit",        3, "#6B84A3"),
        ("Real Asset",    3, C["mid"]),
        ("Alternative",   1, C["gold"]),
    ]
    y = 0.73
    for grp, n, col in groups:
        # Thin color bar on left + group name + count
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.017), 0.004, 0.034,
            boxstyle="square,pad=0",
            facecolor=col, edgecolor="none",
            transform=ax.transAxes, clip_on=False))
        ax.text(0.032, y, grp, ha="left", va="center",
                fontsize=9.5, color=C["ink"],
                transform=ax.transAxes)
        ax.text(0.17, y, str(n), ha="right", va="center",
                fontsize=9.5, color=C["mid"],
                transform=ax.transAxes)
        y -= 0.075

    # ---- Column B: Feature Groups ----
    ax.text(0.50, 0.85, "Feature Space", ha="center", va="top",
            fontsize=13, fontweight="bold", color=C["ink"])
    ax.text(0.50, 0.80, "303 selected features", ha="center", va="top",
            fontsize=9, color=C["mid"])

    feat_groups = [
        ("Macro",         "Inflation · Rates · Unemployment · FCI"),
        ("Cross-market",  "IG OAS · VIX · Oil · USD"),
        ("Momentum",      "12–1m return · Volatility · Drawdown"),
        ("Interactions",  "Macro × Momentum  ·  Rate × Spread"),
        ("Lagged",        "1m, 3m, 6m, 12m transforms"),
    ]
    y = 0.73
    for title, detail in feat_groups:
        ax.text(0.355, y, title + "  —",
                ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=C["ink"],
                transform=ax.transAxes)
        ax.text(0.460, y, detail,
                ha="left", va="center", fontsize=9,
                color=C["charcoal"], transform=ax.transAxes)
        y -= 0.075

    # ---- Column C: Target ----
    ax.text(0.86, 0.85, "Target Variable", ha="center", va="top",
            fontsize=13, fontweight="bold", color=C["ink"])
    ax.text(0.86, 0.80, "per sleeve, per month", ha="center", va="top",
            fontsize=9, color=C["mid"])

    target_items = [
        ("Horizon",   "60-month forward window"),
        ("Measure",   "Annualized excess return"),
        ("vs",        "Risk-free rate"),
        ("Model",     "Separate Elastic Net per sleeve"),
        ("Training",  "Expanding window to Feb 2021"),
    ]
    y = 0.73
    for k, v in target_items:
        ax.text(0.72, y, k, ha="left", va="center",
                fontsize=9, color=C["mid"], transform=ax.transAxes)
        ax.text(0.795, y, v, ha="left", va="center",
                fontsize=9.5, color=C["ink"], transform=ax.transAxes)
        y -= 0.075

    # Two vertical dividers
    for xd in [0.235, 0.655]:
        rule(fig, y_dummy := 0, x0=xd, x1=xd, color=C["light"])
        line = mlines.Line2D([xd, xd], [0.22, 0.88],
                             transform=fig.transFigure,
                             color=C["light"], lw=0.8)
        fig.add_artist(line)

    rule(fig, 0.22)
    fig.text(0.5, 0.17,
             "Elastic Net  ·  l1_ratio = 0.5  ·  alpha = 0.005  ·  "
             "Validation correlation = 0.76  ·  Test correlation = 0.63",
             ha="center", va="top", fontsize=9.5, color=C["mid"])

    save_fig(fig, "finalgfx_02_data_target_design_v4")


# ---------------------------------------------------------------------------
# FIG 03 — Prediction to Allocation
# ---------------------------------------------------------------------------

def fig03(d):
    """
    Left half: prediction scatter (validation).
    Right half: allocation formula + key params.
    Two clean panels. No boxes everywhere.
    """
    pred_val = d["pred_val"]
    bm = pred_val[(pred_val["feature_set_name"]=="core_plus_interactions") &
                  (pred_val["horizon_mode"]=="separate_60") &
                  (pred_val["horizon_months"]==60)].dropna(subset=["y_true","y_pred"])

    corr = bm[["y_true","y_pred"]].corr().iloc[0,1]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.94, "Prediction  →  Allocation",
             ha="center", va="top",
             fontsize=22, fontweight="bold", color=C["ink"])

    rule(fig, 0.90)

    gs = gridspec.GridSpec(1, 2, left=0.07, right=0.97,
                           top=0.86, bottom=0.10, wspace=0.15)
    ax_L = fig.add_subplot(gs[0])
    ax_R = fig.add_subplot(gs[1])

    # ---- Left: prediction scatter ----
    spine_off(ax_L, ("top","right"))
    ax_L.set_facecolor("white")

    # Pool all sleeves; one color per group
    group_map = {
        "EQ": ("#4A7A4A", "Equity"),
        "FI": (C["blue"],  "Fixed Income"),
        "CR": ("#6B84A3",  "Credit"),
        "RE": (C["mid"],   "Real Asset"),
        "AL": (C["gold"],  "Gold"),
    }
    plotted = set()
    for s in SLEEVES_14:
        grp = s[:2]
        col, lab = group_map.get(grp, (C["mid"], grp))
        sub = bm[bm["sleeve_id"]==s]
        if len(sub) < 3:
            continue
        lbl = lab if lab not in plotted else None
        ax_L.scatter(sub["y_true"]*100, sub["y_pred"]*100,
                     s=22, color=col, alpha=0.55,
                     linewidths=0, label=lbl, zorder=3)
        plotted.add(lab)

    # 45-degree line
    lo, hi = -18, 25
    ax_L.plot([lo, hi], [lo, hi], color=C["light"], lw=1.0, zorder=1)
    ax_L.axhline(0, color=C["light"], lw=0.6, zorder=1)
    ax_L.axvline(0, color=C["light"], lw=0.6, zorder=1)

    ax_L.set_xlim(lo, hi); ax_L.set_ylim(lo, hi)
    ax_L.set_xlabel("Realized 5Y excess return (%)", fontsize=10, color=C["charcoal"])
    ax_L.set_ylabel("Predicted 5Y excess return (%)", fontsize=10, color=C["charcoal"])
    ax_L.xaxis.set_major_formatter(ticker.FuncFormatter(pct))
    ax_L.yaxis.set_major_formatter(ticker.FuncFormatter(pct))

    ax_L.text(0.05, 0.96,
              f"Validation correlation  {corr:.2f}",
              transform=ax_L.transAxes,
              fontsize=10, color=C["ink"], fontweight="bold", va="top")
    ax_L.text(0.05, 0.90,
              "Out-of-sample  ·  14 sleeves  ·  Validation period",
              transform=ax_L.transAxes,
              fontsize=8.5, color=C["mid"], va="top")

    handles = [mpatches.Patch(color=v[0], label=v[1]) for v in group_map.values()]
    ax_L.legend(handles=handles, fontsize=8, frameon=False,
                loc="lower right", handlelength=0.8)

    tag(ax_L, "Prediction Evidence")

    # ---- Right: allocation method ----
    spine_off(ax_R, ("top","right","left","bottom"))
    ax_R.set_facecolor("white")
    ax_R.set_xlim(0, 1); ax_R.set_ylim(0, 1)
    ax_R.set_xticks([]); ax_R.set_yticks([])
    tag(ax_R, "Robust Allocation")

    # Formula — clean, monospace-ish, no box
    ax_R.text(0.06, 0.87,
              "maximize",
              ha="left", va="top",
              fontsize=10, color=C["mid"],
              transform=ax_R.transAxes)
    ax_R.text(0.06, 0.79,
              "w'μ  −  (λ/2) · w'Ωw  −  κ · ‖w − w_eq‖₁",
              ha="left", va="top",
              fontsize=14, color=C["ink"],
              fontweight="bold",
              transform=ax_R.transAxes,
              family="monospace")
    ax_R.text(0.06, 0.72,
              "subject to  w ≥ 0,   Σw = 1",
              ha="left", va="top",
              fontsize=10, color=C["mid"],
              transform=ax_R.transAxes)

    # Thin rule under formula
    rule(fig, 0.0, x0=0.0, x1=0.0)  # dummy; draw manually
    ax_R.axhline(0.65, color=C["light"], lw=0.6, xmin=0.04, xmax=0.96)

    # Parameter table — clean, no boxes
    rows = [
        ("w'μ",              "expected return term"),
        ("λ = 8.0",          "risk-aversion parameter"),
        ("Ω = Identity",     "equal sleeve risk scaling"),
        ("κ = 0.10",         "equal-weight deviation penalty"),
        ("Long-only",        "no short selling"),
        ("Annual rebalance", "one refit per December"),
    ]
    y = 0.60
    for param, desc in rows:
        ax_R.text(0.06, y, param, ha="left", va="top",
                  fontsize=9.5, color=C["blue"], fontweight="bold",
                  transform=ax_R.transAxes)
        ax_R.text(0.36, y, desc, ha="left", va="top",
                  fontsize=9.5, color=C["charcoal"],
                  transform=ax_R.transAxes)
        # thin row rule
        ax_R.axhline(y - 0.048, color=C["light"], lw=0.4, xmin=0.04, xmax=0.96)
        y -= 0.075

    save_fig(fig, "finalgfx_03_prediction_to_allocation_v4")


# ---------------------------------------------------------------------------
# FIG 04 — Hero Benchmark
# ---------------------------------------------------------------------------

def fig04(d):
    """Top: wealth path. Bottom: allocation stack. The empirical centrepiece."""
    wp    = d["wp"]
    alloc = d["alloc"]

    bm_raw = wp[(wp["config_label"]=="lam8_kap0.1_identity") &
                (wp["strategy_label"]=="model")]
    ew_raw = wp[(wp["config_label"]=="lam8_kap0.1_identity") &
                (wp["strategy_label"]=="equal_weight")]

    bm_c = chain_wealth(bm_raw)
    ew_c = chain_wealth(ew_raw)

    bm_final = round(bm_c["cum_wealth"].iloc[-1], 2)
    ew_final = round(ew_c["cum_wealth"].iloc[-1], 2)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    fig.text(0.07, 0.95, "Benchmark Performance",
             ha="left", va="top",
             fontsize=22, fontweight="bold", color=C["ink"])
    fig.text(0.07, 0.89,
             "Elastic Net + Robust MVO  ·  Annual rebalance  ·  2015 – 2025  ·  λ=8.0, κ=0.10",
             ha="left", va="top",
             fontsize=10, color=C["mid"])

    rule(fig, 0.87)

    gs = gridspec.GridSpec(2, 1,
                           left=0.07, right=0.88,
                           top=0.84, bottom=0.06,
                           hspace=0.06,
                           height_ratios=[1.6, 1.0])
    ax_t = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_t)

    # ---- Wealth path ----
    spine_off(ax_t, ("top","right"))
    ax_t.set_facecolor("white")

    dates = bm_c["month_end"]
    ax_t.plot(dates, ew_c["cum_wealth"],
              color=C["light"], lw=1.5, zorder=2, label="Equal weight")
    ax_t.fill_between(dates, 1.0, bm_c["cum_wealth"],
                      color=C["blue"], alpha=0.08, zorder=1)
    ax_t.plot(dates, bm_c["cum_wealth"],
              color=C["blue"], lw=2.2, zorder=3, label="Benchmark")
    ax_t.axhline(1.0, color=C["light"], lw=0.6, ls="--")

    # Vertical rebalance lines
    for anc in sorted(bm_c["anchor_month_end"].unique()):
        ax_t.axvline(pd.Timestamp(anc), color=C["light"],
                     lw=0.7, zorder=1, ls=":")

    # End labels — right margin
    for val, col, lbl in [(bm_final, C["blue"], f"Benchmark  {bm_final:.2f}×"),
                          (ew_final, C["mid"],  f"Equal wt.  {ew_final:.2f}×")]:
        ax_t.text(dates.iloc[-1] + pd.DateOffset(months=3), val,
                  lbl, va="center", fontsize=9, color=col)

    ax_t.set_ylabel("Wealth index  (Jan 2015 = 1.0)", fontsize=9.5, color=C["charcoal"])
    ax_t.set_ylim(0.7, None)
    ax_t.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
    ax_t.legend(fontsize=9, frameon=False, loc="upper left")
    ax_t.tick_params(labelbottom=False)
    tag(ax_t, "Wealth Path")

    # ---- Allocation stacked bar ----
    spine_off(ax_b, ("top","right"))
    ax_b.set_facecolor("white")

    alloc_s = alloc.sort_values("anchor_date").reset_index(drop=True)

    # Group into 6 display buckets
    def build_buckets(row):
        return {
            "Gold":       row["w_ALT_GLD"],
            "Treasuries": row["w_FI_UST"],
            "Credit":     row["w_CR_US_HY"] + row["w_CR_US_IG"] + row.get("w_CR_EU_IG",0),
            "Equity":     row["w_EQ_US"] + row["w_EQ_EZ"] + row["w_EQ_JP"] +
                          row["w_EQ_CN"] + row["w_EQ_EM"],
            "Real":       row["w_RE_US"] + row["w_LISTED_RE"] + row["w_LISTED_INFRA"],
            "Other FI":   row["w_FI_EU_GOVT"],
        }

    bucket_colors = {
        "Gold":       C["gold"],
        "Treasuries": C["blue"],
        "Credit":     "#6B84A3",
        "Equity":     "#4A7A4A",
        "Real":       C["mid"],
        "Other FI":   "#A8C8E8",
    }

    x = np.arange(len(alloc_s))
    bots = np.zeros(len(alloc_s))
    buckets_list = ["Equity","Treasuries","Credit","Gold","Real","Other FI"]
    for bkt in buckets_list:
        vals = np.array([build_buckets(row)[bkt]*100 for _,row in alloc_s.iterrows()])
        ax_b.bar(x, vals, 0.55, bottom=bots,
                 color=bucket_colors[bkt], label=bkt,
                 edgecolor="white", linewidth=0.5, zorder=3)
        for xi, (h, bot) in enumerate(zip(vals, bots)):
            if h > 5.0:
                ax_b.text(x[xi], bot + h/2, f"{h:.0f}",
                          ha="center", va="center",
                          fontsize=7, color="white", fontweight="bold")
        bots += vals

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([d.strftime("%Y") for d in alloc_s["anchor_date"]],
                         fontsize=9)
    ax_b.set_ylabel("Allocation (%)", fontsize=9.5, color=C["charcoal"])
    ax_b.set_ylim(0, 105)
    ax_b.yaxis.set_major_formatter(ticker.FuncFormatter(pct))
    ax_b.legend(ncol=6, fontsize=8, frameon=False,
                loc="upper left", handlelength=0.8,
                handleheight=0.8, columnspacing=0.6)
    tag(ax_b, "Annual Allocation")

    save_fig(fig, "finalgfx_04_hero_benchmark_v4")


# ---------------------------------------------------------------------------
# FIG 05 — Gold Activation
# ---------------------------------------------------------------------------

def fig05(d):
    """
    Left: two-bar chart (Gold weight 2021 vs 2022, anchor vs scenario).
    Right: three macro shift panels stacked.
    No 2x2 grid. No colored boxes.
    """
    scenario = d["scenario"]
    at       = d["at"]

    q1 = scenario[scenario["question_id"]=="Q1_gold_favorable"].copy()

    # Verified anchor values
    ANCHORS   = ["2021-12-31","2022-12-31","2023-12-31","2024-12-31"]
    LABELS    = ["Dec 2021","Dec 2022","Dec 2023","Dec 2024"]
    at_idx    = at.set_index(at["anchor_date"].dt.strftime("%Y-%m-%d"))

    gld_anchor  = [at_idx.loc[a,"w_ALT_GLD"]*100 for a in ANCHORS]
    gld_scen    = [q1[q1["anchor_date"]==a]["w_ALT_GLD"].mean()*100 for a in ANCHORS]

    # Anchor macro state (verified from feature master)
    real10y = [-1.04, 1.58, 1.72, 2.24]
    ig_oas  = [0.98,  1.38, 1.04, 0.82]
    short   = [0.05,  4.15, 5.27, 4.42]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    fig.text(0.07, 0.95, "Gold Activation",
             ha="left", va="top",
             fontsize=22, fontweight="bold", color=C["ink"])
    fig.text(0.07, 0.89,
             "Scenario Q1: what macro regime does the model favor gold?",
             ha="left", va="top",
             fontsize=10, color=C["mid"])

    rule(fig, 0.87)

    # Left panel: Gold weight
    ax_gld = fig.add_axes([0.07, 0.10, 0.38, 0.72])
    spine_off(ax_gld, ("top","right"))
    ax_gld.set_facecolor("white")

    x = np.arange(4)
    bw = 0.35
    bars_a = ax_gld.bar(x - bw/2, gld_anchor, bw,
                        color=C["ink"], alpha=0.25, label="Anchor")
    bars_s = ax_gld.bar(x + bw/2, gld_scen, bw,
                        color=C["gold"], alpha=0.90, label="Scenario mean")

    # Value labels
    for b in list(bars_a) + list(bars_s):
        h = b.get_height()
        if h > 1:
            ax_gld.text(b.get_x() + b.get_width()/2, h + 0.4,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=8, color=C["charcoal"])

    # 3x annotation
    ax_gld.annotate("",
                    xy=(0.5 + bw/2, gld_scen[1] + 0.5),
                    xytext=(0.5 - bw/2, gld_anchor[0] + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=C["gold"],
                                   lw=1.5, mutation_scale=10,
                                   connectionstyle="arc3,rad=-0.35"))
    ax_gld.text(0.65, 24, "~3x", fontsize=11,
                color=C["gold"], fontweight="bold")

    ax_gld.set_xticks(x)
    ax_gld.set_xticklabels(LABELS, fontsize=9)
    ax_gld.set_ylabel("Gold allocation (%)", fontsize=10, color=C["charcoal"])
    ax_gld.set_ylim(0, 32)
    ax_gld.yaxis.set_major_formatter(ticker.FuncFormatter(pct))
    ax_gld.legend(fontsize=9, frameon=False, loc="upper left")
    tag(ax_gld, "Gold Weight  (Anchor vs Scenario)")

    # Thin divider
    rule(fig, 0.0, x0=0.485, x1=0.485)
    vline = mlines.Line2D([0.488, 0.488], [0.08, 0.88],
                          transform=fig.transFigure,
                          color=C["light"], lw=0.8)
    fig.add_artist(vline)

    # Right: three macro panels stacked
    gs_r = gridspec.GridSpec(3, 1, left=0.54, right=0.97,
                             top=0.82, bottom=0.10, hspace=0.52)

    # Panel R1: Real yield
    ax_r1 = fig.add_subplot(gs_r[0])
    spine_off(ax_r1, ("top","right"))
    bar_cols = [C["red"] if v < 0 else C["blue"] for v in real10y]
    ax_r1.bar(x, real10y, 0.55, color=bar_cols, alpha=0.85)
    ax_r1.axhline(0, color=C["ink"], lw=1.2)
    for xi, v in enumerate(real10y):
        va = "bottom" if v >= 0 else "top"
        off = 0.06 if v >= 0 else -0.10
        ax_r1.text(xi, v + off, f"{v:+.2f}%",
                   ha="center", va=va, fontsize=8,
                   color="white" if abs(v) > 0.6 else C["ink"],
                   fontweight="bold")
    ax_r1.set_xticks(x); ax_r1.set_xticklabels([])
    ax_r1.set_ylim(-2.2, 3.5)
    ax_r1.set_ylabel("%", fontsize=8, color=C["mid"])
    ax_r1.text(-0.12, 1.05, "US Real 10Y",
               transform=ax_r1.transAxes,
               fontsize=9, fontweight="bold", color=C["ink"])
    # Zero threshold label
    ax_r1.text(3.48, 0.15, "Gold activation\nthreshold",
               ha="right", va="bottom", fontsize=7.5,
               color=C["blue"], style="italic")

    # Panel R2: Short rate
    ax_r2 = fig.add_subplot(gs_r[1])
    spine_off(ax_r2, ("top","right"))
    ax_r2.bar(x, short, 0.55, color=C["charcoal"], alpha=0.5)
    for xi, v in enumerate(short):
        ax_r2.text(xi, v + 0.06, f"{v:.2f}%",
                   ha="center", va="bottom", fontsize=8, color=C["charcoal"])
    ax_r2.set_xticks(x); ax_r2.set_xticklabels([])
    ax_r2.set_ylim(0, 7)
    ax_r2.set_ylabel("%", fontsize=8, color=C["mid"])
    ax_r2.text(-0.12, 1.05, "Short Rate US",
               transform=ax_r2.transAxes,
               fontsize=9, fontweight="bold", color=C["ink"])

    # Panel R3: IG OAS
    ax_r3 = fig.add_subplot(gs_r[2])
    spine_off(ax_r3, ("top","right"))
    ax_r3.bar(x, ig_oas, 0.55, color=C["charcoal"], alpha=0.5)
    for xi, v in enumerate(ig_oas):
        ax_r3.text(xi, v + 0.015, f"{v:.2f}%",
                   ha="center", va="bottom", fontsize=8, color=C["charcoal"])
    ax_r3.set_xticks(x)
    ax_r3.set_xticklabels(LABELS, fontsize=9)
    ax_r3.set_ylim(0, 2.0)
    ax_r3.set_ylabel("%", fontsize=8, color=C["mid"])
    ax_r3.text(-0.12, 1.05, "IG Credit Spread",
               transform=ax_r3.transAxes,
               fontsize=9, fontweight="bold", color=C["ink"])

    # Takeaway at bottom
    rule(fig, 0.085)
    fig.text(0.5, 0.060,
             "Gold crosses from 8% to 22%+ when real yields turn positive."
             "  The model learned this as a threshold — not a gradual relationship.",
             ha="center", va="top",
             fontsize=9.5, color=C["charcoal"], style="italic")

    save_fig(fig, "finalgfx_05_gold_activation_v4")


# ---------------------------------------------------------------------------
# FIG 06 — Allocation Implication (house-view gap — stronger story)
# ---------------------------------------------------------------------------

def fig06(d):
    """
    Left: return range bar chart with 5% house-view line.
    Right: clean gap table + best-case macro state.
    """
    scenario = d["scenario"]
    at       = d["at"]

    q3 = scenario[scenario["question_id"]=="Q3_house_view_5pct"].copy()
    q3["ret_pct"] = q3["pred_return"] * 100

    ANCHORS  = ["2021-12-31","2022-12-31","2023-12-31","2024-12-31"]
    LABELS   = ["Dec 2021","Dec 2022","Dec 2023","Dec 2024"]
    ANCH_RET = [2.01, 3.30, 2.82, 2.24]
    HV       = 5.0

    scen_mean = [q3[q3["anchor_date"]==a]["ret_pct"].mean() for a in ANCHORS]
    scen_max  = [q3[q3["anchor_date"]==a]["ret_pct"].max() for a in ANCHORS]
    scen_min  = [q3[q3["anchor_date"]==a]["ret_pct"].min() for a in ANCHORS]
    best      = q3.loc[q3["pred_return"].idxmax()]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    fig.text(0.07, 0.95, "The Return Ceiling",
             ha="left", va="top",
             fontsize=22, fontweight="bold", color=C["ink"])
    fig.text(0.07, 0.89,
             "Scenario Q3: how far can the benchmark predicted return be lifted?",
             ha="left", va="top",
             fontsize=10, color=C["mid"])

    rule(fig, 0.87)

    ax_L = fig.add_axes([0.07, 0.10, 0.48, 0.72])
    spine_off(ax_L, ("top","right"))
    ax_L.set_facecolor("white")

    x = np.arange(4)
    bw = 0.28

    # House-view line
    ax_L.axhline(HV, color=C["red"], lw=1.5, ls="--", zorder=4)
    ax_L.text(3.6, HV + 0.08, "5%\ntarget",
              ha="center", va="bottom", fontsize=8.5,
              color=C["red"], fontweight="bold")

    # Scenario range band
    for i in range(4):
        ax_L.fill_between([x[i]-bw*1.3, x[i]+bw*1.3],
                          scen_min[i], scen_max[i],
                          color=C["blue"], alpha=0.12, zorder=1)

    # Bars
    ax_L.bar(x - bw/2, ANCH_RET, bw,
             color=C["ink"], alpha=0.30, label="Anchor", zorder=3)
    ax_L.bar(x + bw/2, scen_mean, bw,
             color=C["blue"], alpha=0.75, label="Scenario mean", zorder=3)
    ax_L.scatter(x + bw/2, scen_max, s=60,
                 color="white", edgecolors=C["blue"],
                 lw=1.8, zorder=5, label="Scenario max")

    # Gap arrows + labels
    for i in range(4):
        gap = HV - scen_max[i]
        mx = x[i] + bw/2
        ax_L.annotate("",
                      xy=(mx, HV - 0.05),
                      xytext=(mx, scen_max[i] + 0.05),
                      arrowprops=dict(arrowstyle="<->",
                                      color=C["red"], lw=1.0,
                                      mutation_scale=7))
        ax_L.text(mx + 0.20, (HV + scen_max[i]) / 2,
                  f"−{gap:.2f}pp",
                  ha="left", va="center", fontsize=8,
                  color=C["red"])

    ax_L.set_xticks(x); ax_L.set_xticklabels(LABELS, fontsize=9.5)
    ax_L.set_ylabel("Predicted 5Y ann. excess return (%)", fontsize=9.5, color=C["charcoal"])
    ax_L.set_ylim(0, 6.8)
    ax_L.yaxis.set_major_formatter(ticker.FuncFormatter(pct1))
    ax_L.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)
    tag(ax_L, "Return Range vs House-View Target")

    # Divider
    vline = mlines.Line2D([0.576, 0.576], [0.08, 0.88],
                          transform=fig.transFigure,
                          color=C["light"], lw=0.8)
    fig.add_artist(vline)

    # Right panel — best scenario + gap table
    ax_R = fig.add_axes([0.600, 0.10, 0.37, 0.72])
    spine_off(ax_R, ("top","right","left","bottom"))
    ax_R.set_facecolor("white")
    ax_R.set_xlim(0, 1); ax_R.set_ylim(0, 1)
    ax_R.set_xticks([]); ax_R.set_yticks([])
    tag(ax_R, "Best-Case Achievable Scenario")

    ax_R.text(0.0, 0.86,
              f"Dec 2022  ·  Higher for Longer  ·  pred_return = {best['pred_return']*100:.2f}%",
              ha="left", va="top", fontsize=9.5, color=C["charcoal"],
              transform=ax_R.transAxes)

    # Macro state of best scenario — clean rows
    macro_rows = [
        ("Short rate US",  f"{best['short_rate_US']:.2f}%"),
        ("US Real 10Y",    f"{best['us_real10y']:.2f}%"),
        ("IG OAS",         f"{best['ig_oas']:.2f}%"),
        ("VIX",            f"{best['vix']:.1f}"),
        ("Inflation US",   f"{best['infl_US']:.2f}%"),
    ]
    y = 0.77
    for lbl, val in macro_rows:
        ax_R.axhline(y + 0.010, color=C["light"], lw=0.5, xmin=0.0, xmax=0.95)
        ax_R.text(0.0, y - 0.008, lbl, ha="left", va="top",
                  fontsize=9, color=C["mid"], transform=ax_R.transAxes)
        ax_R.text(0.90, y - 0.008, val, ha="right", va="top",
                  fontsize=9.5, fontweight="bold", color=C["ink"],
                  transform=ax_R.transAxes)
        y -= 0.075
    ax_R.axhline(y + 0.010, color=C["light"], lw=0.5, xmin=0.0, xmax=0.95)

    # Gap by anchor
    ax_R.text(0.0, y - 0.02, "Gap to 5% target", ha="left", va="top",
              fontsize=10, fontweight="bold", color=C["red"],
              transform=ax_R.transAxes)
    y -= 0.085
    gap_rows = [
        ("Dec 2021", "best 2.00%", "−3.00pp"),
        ("Dec 2022", "best 3.08%", "−1.92pp"),
        ("Dec 2023", "best 2.82%", "−2.18pp"),
        ("Dec 2024", "best 1.88%", "−3.12pp"),
    ]
    for lbl, best_s, gap_s in gap_rows:
        ax_R.axhline(y + 0.010, color=C["light"], lw=0.5, xmin=0.0, xmax=0.95)
        ax_R.text(0.0,  y - 0.008, lbl,   ha="left",  va="top", fontsize=8.5,
                  color=C["charcoal"], transform=ax_R.transAxes)
        ax_R.text(0.50, y - 0.008, best_s, ha="center", va="top", fontsize=8.5,
                  color=C["mid"], transform=ax_R.transAxes)
        ax_R.text(0.95, y - 0.008, gap_s, ha="right",  va="top", fontsize=8.5,
                  fontweight="bold", color=C["red"], transform=ax_R.transAxes)
        y -= 0.065
    ax_R.axhline(y + 0.010, color=C["light"], lw=0.5, xmin=0.0, xmax=0.95)

    # Takeaway
    rule(fig, 0.085)
    fig.text(0.5, 0.060,
             "No locally plausible macro state closes the gap to 5%."
             "  The ceiling reflects post-GFC return compression — a model signal, not a failure.",
             ha="center", va="top",
             fontsize=9.5, color=C["charcoal"], style="italic")

    save_fig(fig, "finalgfx_06_allocation_implication_v4")


# ---------------------------------------------------------------------------
# INDEX
# ---------------------------------------------------------------------------

def write_index():
    p = REPORTS / "final_slidegraphics_v4_reset_index.md"
    text = """# Final Slidegraphics — v4 Reset Package

Output: `reports/final_slidegraphics_v4_reset/`  |  16:9  |  300 dpi PNG + vector PDF
Style: Editorial / institutional finance  |  6 figures  |  one main point per figure

---

## finalgfx_01_big_picture_v4
**Title:** From Data to Scenario
**Purpose:** Orient the audience in one slide — four pipeline steps, six key statistics.
**Takeaway:** "The framework goes from 303 macro features through AI prediction and robust allocation to scenario-based explanation. The benchmark compounds 37% ahead of equal weight over the decade."

---

## finalgfx_02_data_target_design_v4
**Title:** Data & Model Design
**Purpose:** Show the investable universe, feature space, and 5Y return target in one clean three-column layout.
**Takeaway:** "Fourteen sleeves, 303 features, one elastic net per sleeve predicting 5-year annualized excess return."

---

## finalgfx_03_prediction_to_allocation_v4
**Title:** Prediction → Allocation
**Purpose:** Out-of-sample prediction scatter (validation corr=0.76) alongside the allocation objective and parameter table.
**Takeaway:** "The model predicts with meaningful correlation; robust MVO converts those predictions to stable, risk-controlled weights."

---

## finalgfx_04_hero_benchmark_v4  *** CENTREPIECE ***
**Title:** Benchmark Performance
**Purpose:** Wealth path (benchmark vs equal weight, 2015–2025) with annual allocation composition below.
**Takeaway:** "The benchmark reaches 1.77x vs 1.57x for equal weight. The 2022 rebalance shows a visible regime shift: gold and bonds replace equity."

---

## finalgfx_05_gold_activation_v4  *** SCENARIO HERO ***
**Title:** Gold Activation
**Purpose:** Gold weight (anchor vs scenario) across four dates alongside three macro driver panels (real yield, short rate, IG OAS).
**Takeaway:** "Gold triples from 8% to 22%+ when real yields cross zero. The model learned this as a threshold."

---

## finalgfx_06_allocation_implication_v4
**Title:** The Return Ceiling
**Purpose:** Scenario return range vs 5% house-view target across all four anchors, with best-case macro state and gap table.
**Takeaway:** "The benchmark's ceiling is 3.08%, reached at the 2022 anchor. The house-view gap of 2–3pp reflects post-GFC return compression — a model signal."

---

## Notes
- Figures 01–03: methodology / context
- Figure 04: empirical centrepiece (hero)
- Figures 05–06: scenario results
- All figures use one palette: ink / charcoal / single blue / single red / gold only for Gold sleeve
- No rainbow colors. No decorative boxes. No infographic clutter.
"""
    p.write_text(text)
    print(f"  {p.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("\nLoading data...")
    data = load_data()
    print("  OK\n")

    print("Building 6 figures...")
    print("\n[01] Big Picture")
    fig01(data)
    print("\n[02] Data & Target Design")
    fig02(data)
    print("\n[03] Prediction to Allocation")
    fig03(data)
    print("\n[04] Hero Benchmark")
    fig04(data)
    print("\n[05] Gold Activation")
    fig05(data)
    print("\n[06] Return Ceiling")
    fig06(data)

    print("\nWriting index...")
    write_index()

    files = sorted(OUTDIR.glob("*.png"))
    print(f"\nDone. {len(files)*2} files in {OUTDIR}\n")


if __name__ == "__main__":
    main()
