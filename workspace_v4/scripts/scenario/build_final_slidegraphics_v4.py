"""
build_final_slidegraphics_v4.py
Full conference slidegraphics package — XOPTPOE v4.
Run from workspace_v4/:
    python scripts/scenario/build_final_slidegraphics_v4.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter, MaxNLocator
from pathlib import Path
import textwrap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA    = ROOT / "data_refs"
OUTDIR  = REPORTS / "final_slidegraphics_v4"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM — one consistent language across all figures
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor":     "#F8F9FA",
    "axes.facecolor":       "#F8F9FA",
    "axes.grid":            True,
    "grid.color":           "#E8EAED",
    "grid.linewidth":       0.7,
    "grid.alpha":           0.8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.edgecolor":       "#CCCCCC",
    "axes.linewidth":       0.8,
    "axes.titlesize":       13,
    "axes.titlepad":        10,
    "axes.labelsize":       10,
    "axes.labelpad":        5,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "xtick.color":          "#555555",
    "ytick.color":          "#555555",
    "legend.fontsize":      8.5,
    "legend.frameon":       False,
    "legend.handlelength":  1.2,
})

# Core palette
C = {
    "navy":       "#1B2A4A",   # primary dark — titles, key lines
    "blue":       "#2471A3",   # scenario / generated
    "lightblue":  "#5DADE2",   # secondary blue
    "gold":       "#E67E22",   # gold allocation
    "amber":      "#F39C12",   # gold highlight accent
    "green":      "#1E8449",   # equity
    "lightgreen": "#27AE60",   # equity lighter
    "purple":     "#7D3C98",   # credit
    "red":        "#C0392B",   # stress / negative
    "orange":     "#D35400",   # inflation / warm
    "grey":       "#7F8C8D",   # neutral / muted
    "lightgrey":  "#BDC3C7",   # equal weight ref
    "bg":         "#F8F9FA",   # figure background
    "white":      "#FFFFFF",
    "gridline":   "#E8EAED",
    "text":       "#2C3E50",   # primary text
    "subtext":    "#5D6D7E",   # secondary text
    "benchmark":  "#1B2A4A",   # benchmark line (dark navy)
    "ew_line":    "#BDC3C7",   # equal weight (muted grey)
}

# Regime palette
REGIME_C = {
    "reflation_risk_on":        "#1E8449",
    "higher_for_longer":        "#E67E22",
    "risk_off_stress":          "#C0392B",
    "mixed_mid_cycle":          "#7F8C8D",
    "disinflationary_slowdown": "#2471A3",
    "soft_landing":             "#16A085",
    "pre_gfc":                  "#7D3C98",
    "covid_shock":              "#C0392B",
}

REGIME_LABEL = {
    "reflation_risk_on":        "Reflation / Risk-On",
    "higher_for_longer":        "Higher for Longer",
    "risk_off_stress":          "Risk-Off / Stress",
    "mixed_mid_cycle":          "Mixed / Mid-Cycle",
    "disinflationary_slowdown": "Disinflationary Slowdown",
    "soft_landing":             "Soft Landing",
}

# Sleeve metadata
SLEEVES_14 = [
    "EQ_US","EQ_EZ","EQ_JP","EQ_CN","EQ_EM",
    "FI_UST","FI_EU_GOVT",
    "CR_US_IG","CR_EU_IG","CR_US_HY",
    "RE_US","LISTED_RE","LISTED_INFRA",
    "ALT_GLD",
]

SLEEVE_LABEL = {
    "EQ_US":        "US Equity",
    "EQ_EZ":        "Euro Equity",
    "EQ_JP":        "Japan Equity",
    "EQ_CN":        "China Equity",
    "EQ_EM":        "EM Equity",
    "FI_UST":       "US Treasuries",
    "FI_EU_GOVT":   "EU Govt Bonds",
    "CR_US_IG":     "US IG Credit",
    "CR_EU_IG":     "EU IG Credit",
    "CR_US_HY":     "US HY Credit",
    "RE_US":        "US Real Estate",
    "LISTED_RE":    "Listed RE",
    "LISTED_INFRA": "Listed Infra",
    "ALT_GLD":      "Gold",
}

SLEEVE_GROUP = {
    "EQ_US":"Equity","EQ_EZ":"Equity","EQ_JP":"Equity","EQ_CN":"Equity","EQ_EM":"Equity",
    "FI_UST":"Fixed Income","FI_EU_GOVT":"Fixed Income",
    "CR_US_IG":"Credit","CR_EU_IG":"Credit","CR_US_HY":"Credit",
    "RE_US":"Real Asset","LISTED_RE":"Real Asset","LISTED_INFRA":"Real Asset",
    "ALT_GLD":"Alternative",
}

GROUP_C = {
    "Equity":       C["lightgreen"],
    "Fixed Income": C["blue"],
    "Credit":       C["purple"],
    "Real Asset":   "#D4AC0D",
    "Alternative":  C["gold"],
}

SLEEVE_C = {
    "EQ_US":        C["lightgreen"],
    "EQ_EZ":        "#52BE80",
    "EQ_JP":        "#A9DFBF",
    "EQ_CN":        "#17202A",    # very dark — not visually overemphasized
    "EQ_EM":        "#196F3D",
    "FI_UST":       C["blue"],
    "FI_EU_GOVT":   C["lightblue"],
    "CR_US_IG":     "#9B59B6",
    "CR_EU_IG":     "#C39BD3",
    "CR_US_HY":     C["purple"],
    "RE_US":        "#D4AC0D",
    "LISTED_RE":    "#A9770A",
    "LISTED_INFRA": "#C8AC7A",
    "ALT_GLD":      C["gold"],
}

SLEEVE_SHORT = {
    "EQ_US":"EQ US","EQ_EZ":"EQ EZ","EQ_JP":"EQ JP","EQ_CN":"EQ CN","EQ_EM":"EQ EM",
    "FI_UST":"FI UST","FI_EU_GOVT":"FI EU",
    "CR_US_IG":"CR IG US","CR_EU_IG":"CR IG EU","CR_US_HY":"CR HY US",
    "RE_US":"RE US","LISTED_RE":"L-RE","LISTED_INFRA":"L-Infra",
    "ALT_GLD":"Gold",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def save_fig(fig, name):
    png = OUTDIR / f"{name}.png"
    pdf = OUTDIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf, bbox_inches="tight", facecolor=C["bg"])
    print(f"  {png.name}  |  {pdf.name}")
    plt.close(fig)

def fig_title(fig, title, subtitle=None, y=0.97):
    fig.text(0.5, y, title, ha="center", va="top",
             fontsize=18, fontweight="bold", color=C["navy"])
    if subtitle:
        fig.text(0.5, y - 0.048, subtitle, ha="center", va="top",
                 fontsize=11, color=C["subtext"], style="italic")

def panel_label(ax, text, x=0.0, y=1.04, fontsize=11):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", color=C["navy"],
            va="bottom", ha="left")

def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(C["bg"])

def regime_chip(ax, cx, cy, regime_key, width=0.28, height=0.10,
                fontsize=9, transform=None, alpha=1.0):
    if transform is None:
        transform = ax.transAxes
    col = REGIME_C.get(regime_key, C["grey"])
    label = REGIME_LABEL.get(regime_key, regime_key.replace("_", " ").title())
    box = FancyBboxPatch((cx - width/2, cy - height/2), width, height,
                         boxstyle="round,pad=0.015",
                         facecolor=col, edgecolor=C["white"],
                         linewidth=1.5, transform=transform,
                         clip_on=False, zorder=5, alpha=alpha)
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C["white"],
            transform=transform, clip_on=False, zorder=6)

def pct_fmt(x, pos):
    return f"{x:.0f}%"

def pct2_fmt(x, pos):
    return f"{x:.1f}%"

def arrow_between(ax, x0, y0, x1, y1, color=C["navy"], lw=1.5, ms=10):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms),
                annotation_clip=False)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data():
    wp = pd.read_csv(REPORTS / "portfolio/v4_allocator_refinement_wealth_paths.csv")
    wp["month_end"] = pd.to_datetime(wp["month_end"])
    alloc = pd.read_csv(REPORTS / "hero_allocation_all_anchors_v4.csv")
    alloc["anchor_date"] = pd.to_datetime(alloc["anchor_date"])
    scenario = pd.read_csv(REPORTS / "scenario_results_v4.csv")
    anchor_truth = pd.read_csv(REPORTS / "scenario_anchor_truth_v4.csv")
    anchor_truth["anchor_date"] = pd.to_datetime(anchor_truth["anchor_date"])
    pred_test = pd.read_parquet(DATA / "predictions_test_v4_benchmark.parquet")
    pred_val  = pd.read_parquet(DATA / "predictions_validation_v4_benchmark.parquet")
    fm = pd.read_parquet(DATA / "feature_master_monthly.parquet")
    fm["month_end"] = pd.to_datetime(fm["month_end"])
    fm_macro = fm.groupby("month_end").first().reset_index()
    am = pd.read_csv(DATA / "asset_master.csv")
    return {
        "wp": wp, "alloc": alloc, "scenario": scenario,
        "anchor_truth": anchor_truth,
        "pred_test": pred_test, "pred_val": pred_val,
        "fm_macro": fm_macro, "am": am,
    }

# ---------------------------------------------------------------------------
# FIG 01 — Big Picture Pipeline
# ---------------------------------------------------------------------------

def fig01_big_picture(d):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "From Data to Scenario Explanation",
              "An AI-Driven Long-Horizon Allocation Pipeline")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor("none")

    # Pipeline steps
    steps = [
        ("Macro &\nMarket Data",     0.08,  C["navy"],     "303 monthly\nfeatures\n2007–2024"),
        ("AI Prediction\nLayer",     0.30,  C["blue"],     "Elastic Net\nregression\n5Y horizon"),
        ("Robust\nAllocator",        0.52,  C["purple"],   "Robust MVO\nλ=8, κ=0.10\nidentity Ω"),
        ("Benchmark\nPortfolio",     0.72,  C["lightgreen"],"14 sleeves\nannual rebalance\n2015–2025"),
        ("Scenario\nExplanation",    0.92,  C["gold"],     "MALA search\nVAR(1) prior\nregime labels"),
    ]

    box_w = 0.13; box_h = 0.22; box_y_center = 0.50

    for (label, xc, col, sub) in steps:
        # Box
        box = FancyBboxPatch((xc - box_w/2, box_y_center - box_h/2), box_w, box_h,
                             boxstyle="round,pad=0.02",
                             facecolor=col, edgecolor=C["white"],
                             linewidth=2, transform=ax.transAxes,
                             clip_on=False, zorder=3)
        ax.add_patch(box)
        # Main label
        ax.text(xc, box_y_center + 0.02, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=C["white"],
                transform=ax.transAxes, zorder=4, multialignment="center")
        # Sub label below
        ax.text(xc, box_y_center - box_h/2 - 0.055, sub,
                ha="center", va="top", fontsize=8.5, color=C["subtext"],
                transform=ax.transAxes, zorder=4, multialignment="center")

    # Arrows between steps
    for i in range(len(steps) - 1):
        x0 = steps[i][1] + box_w/2 + 0.005
        x1 = steps[i+1][1] - box_w/2 - 0.005
        y  = box_y_center
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"],
                                   lw=2.0, mutation_scale=14),
                    annotation_clip=False)

    # Horizon bar underneath
    ax.add_patch(FancyBboxPatch((0.035, 0.265), 0.93, 0.025,
                                boxstyle="round,pad=0.005",
                                facecolor="#E8EAED", edgecolor="none",
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.277, "5-Year Forward Investment Horizon  |  14 Investable Sleeves  |  Training 2007–2021",
            ha="center", va="center", fontsize=9, color=C["subtext"],
            transform=ax.transAxes)

    # Key numbers row
    kpi_items = [
        ("303",   "input features"),
        ("14",    "investable sleeves"),
        ("5Y",    "prediction horizon"),
        ("2368",  "training observations"),
        ("0.633", "prediction correlation"),
        ("2.04×", "benchmark vs EW wealth"),
    ]
    xpos = [0.08, 0.24, 0.40, 0.56, 0.72, 0.88]
    for x, (num, lab) in zip(xpos, kpi_items):
        ax.text(x, 0.185, num, ha="center", va="bottom",
                fontsize=20, fontweight="bold", color=C["navy"],
                transform=ax.transAxes)
        ax.text(x, 0.175, lab, ha="center", va="top",
                fontsize=8, color=C["subtext"],
                transform=ax.transAxes)

    save_fig(fig, "slidegraphic_01_big_picture_v4")


# ---------------------------------------------------------------------------
# FIG 02 — Investable Universe
# ---------------------------------------------------------------------------

def fig02_universe(d):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "Investable Universe", "14 Sleeves Across 5 Asset-Class Groups")

    groups = {
        "Equity":       ["EQ_US","EQ_EZ","EQ_JP","EQ_CN","EQ_EM"],
        "Fixed Income": ["FI_UST","FI_EU_GOVT"],
        "Credit":       ["CR_US_IG","CR_EU_IG","CR_US_HY"],
        "Real Asset":   ["RE_US","LISTED_RE","LISTED_INFRA"],
        "Alternative":  ["ALT_GLD"],
    }

    group_xpos   = [0.10, 0.30, 0.51, 0.70, 0.89]
    group_widths = [0.17, 0.12, 0.15, 0.15, 0.09]

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor("none")

    sleeve_detail = {
        "EQ_US":        ("S&P 500", "USD", "US"),
        "EQ_EZ":        ("Euro Stoxx", "EUR", "Europe"),
        "EQ_JP":        ("Nikkei / TOPIX", "JPY", "Japan"),
        "EQ_CN":        ("CSI 300", "CNY", "China"),
        "EQ_EM":        ("EM Index", "USD", "Global EM"),
        "FI_UST":       ("US Treasury", "USD", "US"),
        "FI_EU_GOVT":   ("EU Govt", "EUR", "Europe"),
        "CR_US_IG":     ("US IG Corp", "USD", "US"),
        "CR_EU_IG":     ("EU IG Corp", "EUR", "Europe"),
        "CR_US_HY":     ("US HY Corp", "USD", "US"),
        "RE_US":        ("US RE", "USD", "US"),
        "LISTED_RE":    ("Global REIT", "USD", "Global"),
        "LISTED_INFRA": ("Listed Infra", "USD", "Global"),
        "ALT_GLD":      ("Gold Spot", "USD", "Global"),
    }

    for gi, (grp, sleeves) in enumerate(groups.items()):
        xc = group_xpos[gi]
        gw = group_widths[gi]
        gc = GROUP_C[grp]

        # Group header box
        hbox = FancyBboxPatch((xc - gw/2, 0.82), gw, 0.065,
                              boxstyle="round,pad=0.012",
                              facecolor=gc, edgecolor=C["white"],
                              linewidth=2, transform=ax.transAxes,
                              clip_on=False, zorder=3)
        ax.add_patch(hbox)
        ax.text(xc, 0.853, grp, ha="center", va="center",
                fontsize=11, fontweight="bold", color=C["white"],
                transform=ax.transAxes, zorder=4)
        ax.text(xc, 0.815, f"{len(sleeves)} sleeve{'s' if len(sleeves)>1 else ''}",
                ha="center", va="top", fontsize=8.5, color=C["subtext"],
                transform=ax.transAxes, zorder=4)

        n = len(sleeves)
        y_start = 0.735
        y_gap = 0.115

        for si, s in enumerate(sleeves):
            sy = y_start - si * y_gap
            sc = SLEEVE_C[s]
            ticker, proxy, region = sleeve_detail[s]

            # Sleeve card
            card = FancyBboxPatch((xc - gw/2 + 0.006, sy - 0.042), gw - 0.012, 0.084,
                                  boxstyle="round,pad=0.008",
                                  facecolor=C["white"], edgecolor=sc,
                                  linewidth=1.5, transform=ax.transAxes,
                                  clip_on=False, zorder=3)
            ax.add_patch(card)

            # Color band on left
            band = FancyBboxPatch((xc - gw/2 + 0.006, sy - 0.042), 0.010, 0.084,
                                  boxstyle="square,pad=0",
                                  facecolor=sc, edgecolor="none",
                                  transform=ax.transAxes, clip_on=False, zorder=4)
            ax.add_patch(band)

            # Sleeve ID
            ax.text(xc - gw/2 + 0.022, sy + 0.010, SLEEVE_SHORT[s],
                    ha="left", va="center",
                    fontsize=9, fontweight="bold", color=C["navy"],
                    transform=ax.transAxes, zorder=5)
            # Proxy
            ax.text(xc - gw/2 + 0.022, sy - 0.014, ticker,
                    ha="left", va="center",
                    fontsize=7.5, color=C["subtext"],
                    transform=ax.transAxes, zorder=5)
            # Region pill
            pill_w = 0.042
            pill = FancyBboxPatch((xc + gw/2 - 0.056, sy - 0.016), pill_w, 0.032,
                                  boxstyle="round,pad=0.004",
                                  facecolor=sc, edgecolor="none", alpha=0.2,
                                  transform=ax.transAxes, clip_on=False, zorder=5)
            ax.add_patch(pill)
            ax.text(xc + gw/2 - 0.034, sy, region,
                    ha="center", va="center",
                    fontsize=7, color=sc,
                    transform=ax.transAxes, zorder=6)

    # Bottom note
    ax.text(0.5, 0.062, "All returns measured as 5-year annualized excess return vs risk-free rate  |  "
            "Training window: 2007–2021  |  Out-of-sample: 2015–2025",
            ha="center", va="center", fontsize=8.5, color=C["subtext"],
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["white"],
                      edgecolor=C["gridline"], linewidth=0.8))

    save_fig(fig, "slidegraphic_02_investable_universe_v4")


# ---------------------------------------------------------------------------
# FIG 03 — Data, Features, Targets
# ---------------------------------------------------------------------------

def fig03_data_features(d):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "Data Design", "Macro Features, Interactions, and 5-Year Return Targets")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor("none")

    # Three main columns
    cols = [
        {
            "title": "Macro & Market Features",
            "xc": 0.175,
            "color": C["navy"],
            "sections": [
                ("Inflation", ["US CPI", "Euro HICP", "Japan CPI"], C["orange"]),
                ("Interest Rates", ["Short rates US/EU/JP", "Long rates US/EU/JP", "Term slopes"], C["blue"]),
                ("Risk / Stress", ["IG OAS spreads", "VIX", "FCI index"], C["red"]),
                ("Growth", ["Unemployment US/EU", "Oil WTI", "USD broad"], C["lightgreen"]),
                ("Momentum / Vol", ["12-1m momentum", "3m/12m volatility", "Max drawdown 12m"], C["grey"]),
            ],
        },
        {
            "title": "Interaction Features",
            "xc": 0.480,
            "color": C["purple"],
            "sections": [
                ("Cross-feature", ["Macro × Momentum", "Rate × Credit", "Infl × FCI"], C["purple"]),
                ("Lagged", ["1m, 3m, 6m, 12m lags", "Delta transforms", "Moving averages"], C["purple"]),
                ("Total feature set", ["303 features selected", "Elastic Net regularized", "Shared 60m horizon"], C["blue"]),
            ],
        },
        {
            "title": "5-Year Return Target",
            "xc": 0.790,
            "color": C["gold"],
            "sections": [
                ("Definition", ["60-month forward window", "Annualized excess return", "vs risk-free rate"], C["gold"]),
                ("14 Targets", ["One per sleeve", "Separate model per sleeve", "Expanding-window fit"], C["gold"]),
                ("Horizon Note", ["Train: 2007-02 to 2021-02", "n=2368 training rows", "Out-of-sample prospective"], C["grey"]),
            ],
        },
    ]

    col_w = 0.22

    for col in cols:
        xc = col["xc"]
        # Header
        hdr = FancyBboxPatch((xc - col_w/2, 0.82), col_w, 0.065,
                             boxstyle="round,pad=0.01",
                             facecolor=col["color"], edgecolor=C["white"],
                             linewidth=2, transform=ax.transAxes,
                             clip_on=False, zorder=3)
        ax.add_patch(hdr)
        ax.text(xc, 0.853, col["title"], ha="center", va="center",
                fontsize=11, fontweight="bold", color=C["white"],
                transform=ax.transAxes, zorder=4)

        y = 0.77
        for sec_title, items, sec_col in col["sections"]:
            # Section label
            ax.text(xc, y, sec_title, ha="center", va="top",
                    fontsize=9, fontweight="bold", color=sec_col,
                    transform=ax.transAxes, zorder=4)
            y -= 0.032
            for item in items:
                ax.text(xc, y, f"  {item}", ha="center", va="top",
                        fontsize=8.5, color=C["text"],
                        transform=ax.transAxes, zorder=4)
                y -= 0.028
            y -= 0.018

    # Center connector arrows
    for x_from, x_to in [(0.175 + col_w/2 + 0.005, 0.480 - col_w/2 - 0.005),
                          (0.480 + col_w/2 + 0.005, 0.790 - col_w/2 - 0.005)]:
        ax.annotate("", xy=(x_to, 0.60), xytext=(x_from, 0.60),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"],
                                   lw=1.8, mutation_scale=12))

    # Center text labels on arrows
    ax.text(0.330, 0.615, "combine\n& transform", ha="center", va="bottom",
            fontsize=8, color=C["subtext"], transform=ax.transAxes, style="italic")
    ax.text(0.640, 0.615, "predict\n& evaluate", ha="center", va="bottom",
            fontsize=8, color=C["subtext"], transform=ax.transAxes, style="italic")

    # Bottom bar
    ax.add_patch(FancyBboxPatch((0.035, 0.075), 0.93, 0.035,
                                boxstyle="round,pad=0.005",
                                facecolor=C["white"], edgecolor=C["gridline"],
                                linewidth=0.8, transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.093, "Each sleeve gets its own prediction model  |  "
            "Elastic Net with l1_ratio=0.5, alpha=0.005  |  "
            "Training on expanding window through Feb 2021",
            ha="center", va="center", fontsize=9, color=C["subtext"],
            transform=ax.transAxes)

    save_fig(fig, "slidegraphic_03_data_features_targets_v4")


# ---------------------------------------------------------------------------
# FIG 04 — AI Prediction Layer
# ---------------------------------------------------------------------------

def fig04_ai_prediction(d):
    pred_test = d["pred_test"]
    pred_val  = d["pred_val"]

    # Get benchmark predictions (elastic_net, core_plus_interactions, separate_60, 60m)
    def get_bm(df):
        return df[(df["feature_set_name"]=="core_plus_interactions") &
                  (df["horizon_mode"]=="separate_60") &
                  (df["horizon_months"]==60)].copy()

    bm_t = get_bm(pred_test)
    bm_v = get_bm(pred_val)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "AI Prediction Layer",
              "Elastic Net learns which macro environments predict 5-year sleeve returns")

    gs = gridspec.GridSpec(1, 2, left=0.07, right=0.97,
                           top=0.84, bottom=0.10, wspace=0.38)

    ax_L = fig.add_subplot(gs[0, 0])
    ax_R = fig.add_subplot(gs[0, 1])
    clean_ax(ax_L); clean_ax(ax_R)

    # --- Left panel: Architecture / model card ---
    ax_L.set_xlim(0, 1); ax_L.set_ylim(0, 1)
    ax_L.axis("off")
    panel_label(ax_L, "A  |  Model Architecture")

    arch_steps = [
        (0.50, 0.87, "303 Macro & Market Features", C["navy"],        0.70, 0.065),
        (0.50, 0.73, "Elastic Net Regression", C["blue"],             0.55, 0.065),
        (0.50, 0.58, "14 Predicted 5Y Returns", C["purple"],         0.55, 0.065),
        (0.50, 0.43, "Robust MVO Allocator", C["gold"],              0.55, 0.065),
    ]

    for xi, yi, label, col, bw, bh in arch_steps:
        bx = FancyBboxPatch((xi - bw/2, yi - bh/2), bw, bh,
                            boxstyle="round,pad=0.012",
                            facecolor=col, edgecolor=C["white"],
                            linewidth=1.5, transform=ax_L.transAxes,
                            clip_on=False, zorder=3)
        ax_L.add_patch(bx)
        ax_L.text(xi, yi, label, ha="center", va="center",
                  fontsize=9.5, fontweight="bold", color=C["white"],
                  transform=ax_L.transAxes, zorder=4)

    # Arrows in architecture
    for (_, y1, _, _, _, _), (_, y2, _, _, _, bh2) in zip(arch_steps[:-1], arch_steps[1:]):
        ax_L.annotate("", xy=(0.5, y2 + bh2/2 + 0.005),
                      xytext=(0.5, y1 - arch_steps[0][5]/2 - 0.005),
                      xycoords=ax_L.transAxes, textcoords=ax_L.transAxes,
                      arrowprops=dict(arrowstyle="-|>", color=C["grey"],
                                     lw=1.5, mutation_scale=10))

    # Key model facts
    facts = [
        ("Regularization", "Elastic Net (l1=0.5)"),
        ("Alpha", "0.005"),
        ("Features", "303 (core + interactions)"),
        ("Horizon", "60 months (5 years)"),
        ("Training", "Expanding window"),
        ("Training end", "Feb 2021"),
    ]
    y_fact = 0.26
    ax_L.text(0.5, y_fact + 0.04, "Model Configuration", ha="center", va="bottom",
              fontsize=9, fontweight="bold", color=C["navy"],
              transform=ax_L.transAxes)
    for i, (k, v) in enumerate(facts):
        col_f = i % 2
        row_f = i // 2
        xp = 0.07 + col_f * 0.50
        yp = y_fact - row_f * 0.060
        ax_L.text(xp, yp, k + ":", ha="left", va="top",
                  fontsize=8, color=C["subtext"], transform=ax_L.transAxes)
        ax_L.text(xp + 0.46, yp, v, ha="right", va="top",
                  fontsize=8.5, fontweight="bold", color=C["navy"],
                  transform=ax_L.transAxes)

    # --- Right panel: Prediction scatter (validation split) ---
    panel_label(ax_R, "B  |  Out-of-Sample Prediction Evidence (Validation Period)")

    # Use all sleeves pooled for main scatter
    bm_v_60 = bm_v[bm_v["horizon_months"] == 60].copy()
    bm_v_60 = bm_v_60.dropna(subset=["y_true","y_pred"])

    overall_corr = bm_v_60[["y_true","y_pred"]].corr().iloc[0,1]
    r2_overall = 1 - ((bm_v_60.y_true - bm_v_60.y_pred)**2).sum() / \
                     ((bm_v_60.y_true - bm_v_60.y_true.mean())**2).sum()

    # Plot each sleeve in its own color
    for s in SLEEVES_14:
        sub = bm_v_60[bm_v_60["sleeve_id"]==s]
        if len(sub) >= 3:
            ax_R.scatter(sub["y_true"]*100, sub["y_pred"]*100,
                        s=28, color=SLEEVE_C[s], alpha=0.65, zorder=3,
                        label=SLEEVE_SHORT[s])

    # Diagonal reference
    lim_min, lim_max = -15, 30
    ax_R.plot([lim_min, lim_max], [lim_min, lim_max],
              "--", color=C["lightgrey"], lw=1.2, zorder=1)
    ax_R.axhline(0, color=C["lightgrey"], lw=0.8, zorder=1)
    ax_R.axvline(0, color=C["lightgrey"], lw=0.8, zorder=1)

    # Stats box
    ax_R.text(0.04, 0.97,
              f"Corr: {overall_corr:.2f}  |  OOS R²: {r2_overall:.2f}",
              transform=ax_R.transAxes, fontsize=9.5,
              color=C["navy"], fontweight="bold", va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor=C["white"],
                        edgecolor=C["gridline"]))

    ax_R.set_xlabel("Realized 5Y Annualized Excess Return (%)", fontsize=10)
    ax_R.set_ylabel("Predicted 5Y Annualized Excess Return (%)", fontsize=10)
    ax_R.set_xlim(lim_min, lim_max)
    ax_R.set_ylim(lim_min, lim_max)
    ax_R.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_R.yaxis.set_major_formatter(FuncFormatter(pct_fmt))

    # Compact legend in 3 rows
    handles = [mpatches.Patch(color=SLEEVE_C[s], label=SLEEVE_SHORT[s])
               for s in SLEEVES_14]
    ax_R.legend(handles=handles, ncol=3, fontsize=7.5, frameon=False,
                loc="lower right", handlelength=0.8, handleheight=0.8,
                columnspacing=0.5)

    save_fig(fig, "slidegraphic_04_ai_prediction_layer_v4")


# ---------------------------------------------------------------------------
# FIG 05 — Allocation Layer
# ---------------------------------------------------------------------------

def fig05_allocation(d):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "Robust Allocation Layer",
              "Predicted returns feed a risk-aware optimizer — not hand-crafted weights")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor("none")

    # Left column — intuition
    ax.text(0.07, 0.84, "A  |  Optimization Objective", ha="left", va="top",
            fontsize=11, fontweight="bold", color=C["navy"], transform=ax.transAxes)

    formula_lines = [
        (0.10, 0.76, r"Maximize:   " , 10, C["subtext"], False),
        (0.10, 0.69, r"  w' μ  −  (λ/2) · w' Ω w  −  κ · ‖w − w_eq‖₁",
         13, C["navy"], True),
        (0.10, 0.62, "Subject to:  w ≥ 0,   Σwᵢ = 1", 10, C["subtext"], False),
    ]
    for xf, yf, text, fs, col, bold in formula_lines:
        ax.text(xf, yf, text, ha="left", va="top",
                fontsize=fs, color=col, fontweight="bold" if bold else "normal",
                transform=ax.transAxes, family="monospace" if bold else "sans-serif")

    # Term explanations
    terms = [
        ("w' μ",          "Expected return",         C["lightgreen"],  0.10),
        ("λ/2 · w' Ω w",  "Risk penalty  (λ=8.0)",   C["red"],         0.38),
        ("κ · ‖w − w_eq‖₁","Equal-weight deviation   (κ=0.10)", C["orange"], 0.66),
    ]
    for tterm, tdesc, tcol, tx in terms:
        ax.add_patch(FancyBboxPatch((tx, 0.46), 0.24, 0.09,
                                   boxstyle="round,pad=0.01",
                                   facecolor=tcol, edgecolor=C["white"],
                                   linewidth=1.5, alpha=0.12,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(tx + 0.12, 0.505, tterm, ha="center", va="center",
                fontsize=10, fontweight="bold", color=tcol,
                transform=ax.transAxes)
        ax.text(tx + 0.12, 0.472, tdesc, ha="center", va="center",
                fontsize=8, color=C["subtext"], transform=ax.transAxes)

    # Design choices callout
    design_items = [
        ("Long-only",   "No short selling  (w ≥ 0)"),
        ("Fully invested", "Weights sum to 100%"),
        ("Ω = Identity",  "Equal risk scaling across sleeves"),
        ("Annual rebalance", "One refit per year at Dec month-end"),
        ("λ = 8.0",       "High risk aversion — stable allocations"),
        ("κ = 0.10",      "Moderate diversification pull"),
    ]
    ax.text(0.07, 0.40, "B  |  Key Design Choices", ha="left", va="top",
            fontsize=11, fontweight="bold", color=C["navy"], transform=ax.transAxes)

    for i, (k, v) in enumerate(design_items):
        col_i = i % 2; row_i = i // 2
        xd = 0.07 + col_i * 0.42; yd = 0.34 - row_i * 0.070
        ax.add_patch(FancyBboxPatch((xd, yd - 0.025), 0.38, 0.055,
                                   boxstyle="round,pad=0.008",
                                   facecolor=C["white"], edgecolor=C["gridline"],
                                   linewidth=0.8, transform=ax.transAxes,
                                   clip_on=False))
        ax.text(xd + 0.01, yd + 0.005, k + ":", ha="left", va="center",
                fontsize=8.5, fontweight="bold", color=C["navy"],
                transform=ax.transAxes)
        ax.text(xd + 0.37, yd + 0.005, v, ha="right", va="center",
                fontsize=8.5, color=C["subtext"], transform=ax.transAxes)

    # Right column — allocation snapshot at key anchors
    ax.text(0.72, 0.84, "C  |  Benchmark Allocation at Key Anchors",
            ha="left", va="top",
            fontsize=11, fontweight="bold", color=C["navy"], transform=ax.transAxes)

    anchor_summary = [
        ("2021", 0.081, 0.256, 0.199, 0.083, "reflation_risk_on"),
        ("2022", 0.223, 0.137, 0.240, 0.161, "higher_for_longer"),
        ("2023", 0.224, 0.163, 0.238, 0.203, "higher_for_longer"),
        ("2024", 0.232, 0.185, 0.166, 0.190, "mixed_mid_cycle"),
    ]
    snap_cols = ["Gold", "EQ US", "FI UST", "HY US"]
    snap_colors = [C["gold"], C["lightgreen"], C["blue"], C["purple"]]

    # Mini grouped bars
    ax_snap = fig.add_axes([0.72, 0.11, 0.24, 0.68])
    clean_ax(ax_snap)

    n_anchors = len(anchor_summary)
    n_cats    = len(snap_cols)
    group_w   = 0.8
    bar_w     = group_w / n_cats

    for ci, (col_name, col_color) in enumerate(zip(snap_cols, snap_colors)):
        vals = [row[ci+1] * 100 for row in anchor_summary]
        x_pos = np.arange(n_anchors) + (ci - n_cats/2 + 0.5) * bar_w
        ax_snap.bar(x_pos, vals, bar_w * 0.85, color=col_color, alpha=0.85,
                    label=col_name, zorder=3)

    ax_snap.set_xticks(np.arange(n_anchors))
    ax_snap.set_xticklabels([f"Dec\n{r[0]}" for r in anchor_summary], fontsize=9)
    ax_snap.set_ylabel("Weight (%)", fontsize=9)
    ax_snap.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_snap.set_ylim(0, 32)
    ax_snap.legend(ncol=2, fontsize=8, frameon=False, loc="upper right")
    ax_snap.set_title("Selected Sleeves by Anchor Year", fontsize=9.5,
                      color=C["subtext"], pad=6, style="italic")
    ax_snap.grid(axis="y", alpha=0.6)
    ax_snap.set_axisbelow(True)

    save_fig(fig, "slidegraphic_05_allocation_layer_v4")


# ---------------------------------------------------------------------------
# FIG 06 — Hero Benchmark Behavior
# ---------------------------------------------------------------------------

def fig06_hero(d):
    wp    = d["wp"]
    alloc = d["alloc"]

    bm = wp[(wp["config_label"]=="lam8_kap0.1_identity") &
            (wp["strategy_label"]=="model")].copy().sort_values("month_end")
    ew = wp[(wp["config_label"]=="lam8_kap0.1_identity") &
            (wp["strategy_label"]=="equal_weight")].copy().sort_values("month_end")

    # Build allocation time series: each anchor's weights hold for the next year
    anchor_dates = alloc["anchor_date"].tolist()
    w_cols = [f"w_{s}" for s in SLEEVES_14]

    # Map each month to its anchor's weights
    bm2 = bm.copy()
    bm2["anchor_month"] = pd.to_datetime(bm2["anchor_month_end"])

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "Benchmark Performance",
              "Elastic Net + Robust MVO  |  2015–2025  |  λ=8.0, κ=0.10, Ω=Identity")

    gs = gridspec.GridSpec(2, 1, left=0.08, right=0.97,
                           top=0.87, bottom=0.07,
                           hspace=0.10,
                           height_ratios=[1.5, 1.1])

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    clean_ax(ax_top); clean_ax(ax_bot)

    # --- Top: Wealth paths ---
    # Use first month as base (Jan 2015 = 1.0)
    # The wealth_index is already normalized within each anchor period
    # Build continuous chain: rebase at each anchor transition
    # For simplicity: use anchor-period returns to rebuild continuous wealth
    # Actually the wealth_index appears to restart each anchor year — chain them
    bm2 = bm.copy().sort_values("month_end").reset_index(drop=True)
    ew2 = ew.copy().sort_values("month_end").reset_index(drop=True)

    # Build continuous wealth path by chaining anchor periods
    def chain_wealth(df):
        anchors_sorted = sorted(df["anchor_month_end"].unique())
        result = []
        cum_wealth = 1.0
        for anc in anchors_sorted:
            sub = df[df["anchor_month_end"] == anc].copy().sort_values("month_end")
            sub = sub.reset_index(drop=True)
            if len(result) == 0:
                cum_at_start = 1.0
            else:
                cum_at_start = result[-1]["cum_wealth"]
            # Normalize sub wealth so it starts at cum_at_start
            sub["cum_wealth"] = sub["wealth_index"] / sub["wealth_index"].iloc[0] * cum_at_start
            result.append(sub)
        return pd.concat(result).sort_values("month_end").reset_index(drop=True)

    bm_chain = chain_wealth(bm2)
    ew_chain  = chain_wealth(ew2)

    dates = bm_chain["month_end"]
    ax_top.fill_between(dates, 1, bm_chain["cum_wealth"],
                        color=C["navy"], alpha=0.08, zorder=1)
    ax_top.plot(dates, ew_chain["cum_wealth"],
                color=C["lightgrey"], lw=1.8, zorder=2, label="Equal Weight")
    ax_top.plot(dates, bm_chain["cum_wealth"],
                color=C["navy"], lw=2.5, zorder=3, label="Benchmark")
    ax_top.axhline(1.0, color=C["lightgrey"], lw=0.8, ls="--", zorder=1)

    # Final wealth labels
    bm_final = bm_chain["cum_wealth"].iloc[-1]
    ew_final = ew_chain["cum_wealth"].iloc[-1]
    ax_top.text(dates.iloc[-1] + pd.DateOffset(months=2), bm_final,
                f"{bm_final:.2f}×", va="center", fontsize=9.5,
                color=C["navy"], fontweight="bold")
    ax_top.text(dates.iloc[-1] + pd.DateOffset(months=2), ew_final,
                f"{ew_final:.2f}×", va="center", fontsize=9,
                color=C["grey"])

    # Rebalance markers (Dec of each year in the data)
    rebal_dates = pd.to_datetime(sorted(bm_chain["anchor_month_end"].unique()))
    for rd in rebal_dates:
        yr_bm = bm_chain[bm_chain["month_end"] >= rd]
        if not yr_bm.empty:
            w_at_rd = yr_bm["cum_wealth"].iloc[0]
            ax_top.axvline(rd, color=C["gridline"], lw=0.9, zorder=1)

    # Key events
    events = [
        (pd.Timestamp("2020-03-31"), "COVID\nCrash", -0.12, C["red"]),
        (pd.Timestamp("2022-01-31"), "Rate\nHike Cycle", 0.10, C["orange"]),
    ]
    for ev_date, ev_label, y_offset, ev_col in events:
        if ev_date >= dates.iloc[0] and ev_date <= dates.iloc[-1]:
            w_at = bm_chain[bm_chain["month_end"] <= ev_date]["cum_wealth"].iloc[-1]
            ax_top.annotate(ev_label,
                           xy=(ev_date, w_at),
                           xytext=(ev_date, w_at + y_offset),
                           fontsize=7.5, color=ev_col, ha="center",
                           arrowprops=dict(arrowstyle="-", color=ev_col, lw=0.8))

    ax_top.set_ylabel("Wealth Index  (Jan 2015 = 1.0)", fontsize=10)
    ax_top.legend(loc="upper left", fontsize=9, frameon=False)
    ax_top.set_ylim(0.7, None)
    ax_top.grid(axis="y", alpha=0.6)
    ax_top.set_axisbelow(True)
    ax_top.tick_params(labelbottom=False)
    panel_label(ax_top, "A  |  Wealth Path")

    # --- Bottom: Allocation stacked bar ---
    panel_label(ax_bot, "B  |  Annual Rebalance Composition")

    # Use alloc DataFrame (one row per anchor)
    alloc_sorted = alloc.sort_values("anchor_date").reset_index(drop=True)

    x_positions = np.arange(len(alloc_sorted))
    bottoms = np.zeros(len(alloc_sorted))

    # Group sleeves for readability: show 7 groups
    display_sleeves = [
        ("EQ_US",        "EQ US",    SLEEVE_C["EQ_US"]),
        ("EQ_EZ+EQ_JP",  "EQ EZ+JP", "#52BE80"),
        ("EQ_EM+EQ_CN",  "EQ EM/CN", "#196F3D"),
        ("FI_UST",       "FI UST",   SLEEVE_C["FI_UST"]),
        ("FI_EU_GOVT",   "FI EU",    SLEEVE_C["FI_EU_GOVT"]),
        ("CR",           "Credit",   SLEEVE_C["CR_US_HY"]),
        ("REAL",         "Real",     SLEEVE_C["LISTED_INFRA"]),
        ("ALT_GLD",      "Gold",     SLEEVE_C["ALT_GLD"]),
    ]

    def get_group_weight(row, key):
        if "+" in key:
            parts = key.split("+")
            return sum(row.get(f"w_{p}", 0) for p in parts)
        elif key == "CR":
            return sum(row.get(f"w_{s}", 0) for s in ["CR_US_IG","CR_EU_IG","CR_US_HY"])
        elif key == "REAL":
            return sum(row.get(f"w_{s}", 0) for s in ["RE_US","LISTED_RE","LISTED_INFRA"])
        else:
            return row.get(f"w_{key}", 0)

    bar_heights_all = []
    for s_key, s_label, s_col in display_sleeves:
        heights = []
        for _, row in alloc_sorted.iterrows():
            heights.append(get_group_weight(row, s_key) * 100)
        bar_heights_all.append(heights)
        bars = ax_bot.bar(x_positions, heights, 0.65, bottom=bottoms,
                          color=s_col, label=s_label,
                          edgecolor=C["white"], linewidth=0.6, zorder=3)
        for xi, (h, bot) in enumerate(zip(heights, bottoms)):
            if h > 4.0:
                ax_bot.text(x_positions[xi], bot + h/2,
                           f"{h:.0f}%", ha="center", va="center",
                           fontsize=7, color=C["white"], fontweight="bold")
        bottoms = bottoms + np.array(heights)

    ax_bot.set_xticks(x_positions)
    ax_bot.set_xticklabels([d.strftime("Dec\n%Y") for d in alloc_sorted["anchor_date"]],
                            fontsize=9)
    ax_bot.set_ylabel("Portfolio Weight (%)", fontsize=10)
    ax_bot.set_ylim(0, 105)
    ax_bot.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_bot.legend(ncol=8, fontsize=8, frameon=False,
                  loc="upper left", handlelength=0.9,
                  handleheight=0.8, columnspacing=0.5)
    ax_bot.grid(axis="y", alpha=0.5)
    ax_bot.set_axisbelow(True)

    plt.setp(ax_top.get_xticklabels(), visible=False)

    save_fig(fig, "slidegraphic_06_hero_benchmark_behavior_v4")


# ---------------------------------------------------------------------------
# FIG 07 — Scenario Method
# ---------------------------------------------------------------------------

def fig07_scenario_method(d):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "Scenario Generation Method",
              "Principled macro-state explanation via MALA search over plausible neighborhoods")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor("none")

    # Pipeline across top row
    pipe_steps = [
        ("Compact\nMacro State",  0.07,  C["navy"],   "19-dim vector m\n(rates, spreads,\ninfl, VIX, USD)"),
        ("Feature\nReconstruction",0.23,  C["blue"],   "303 features\nfrom m via\nprecomputed map"),
        ("Prediction\nPipeline",   0.39,  C["purple"],  "ElasticNet\n→ 14 mu_hat\n→ Robust MVO"),
        ("Probe\nObjective G(m)",  0.55,  C["gold"],   "Task-specific\nloss function\n(maximize gold, etc.)"),
        ("VAR(1)\nPlausibility",   0.71,  C["lightgreen"],"Mahalanobis\nregularizer\nσ-jitter=0.01"),
        ("Regime\nInterpretation", 0.87,  C["orange"],  "5 dimensions\n→ 6 conference\nlabels"),
    ]
    box_w = 0.105; box_h = 0.175; box_y = 0.72

    for label, xc, col, sub in pipe_steps:
        bx = FancyBboxPatch((xc - box_w/2, box_y - box_h/2), box_w, box_h,
                            boxstyle="round,pad=0.012",
                            facecolor=col, edgecolor=C["white"],
                            linewidth=1.5, transform=ax.transAxes,
                            clip_on=False, zorder=3)
        ax.add_patch(bx)
        ax.text(xc, box_y + 0.01, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=C["white"],
                transform=ax.transAxes, zorder=4, multialignment="center")
        ax.text(xc, box_y - box_h/2 - 0.04, sub,
                ha="center", va="top", fontsize=7.5, color=C["subtext"],
                transform=ax.transAxes, zorder=4, multialignment="center")

    for i in range(len(pipe_steps) - 1):
        x0 = pipe_steps[i][1] + box_w/2 + 0.003
        x1 = pipe_steps[i+1][1] - box_w/2 - 0.003
        ax.annotate("", xy=(x1, box_y), xytext=(x0, box_y),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"],
                                   lw=1.5, mutation_scale=10))

    # MALA section
    mala_x = 0.07
    ax.text(mala_x, 0.48, "MALA Search", ha="left", va="top",
            fontsize=12, fontweight="bold", color=C["navy"],
            transform=ax.transAxes)

    mala_eq_lines = [
        "Energy:  E(m) = G_task(m)  +  λ · G_reg(m)",
        "Gradient proposal:  m' = m  −  η ∇E(m)  +  √(2τη) · ξ",
        "Metropolis accept:  α = min(1,  exp(−E(m') + E(m)))",
    ]
    for i, line in enumerate(mala_eq_lines):
        ax.text(mala_x + 0.005, 0.43 - i * 0.060, line,
                ha="left", va="top", fontsize=9,
                color=C["text"] if i > 0 else C["navy"],
                fontweight="bold" if i == 0 else "normal",
                transform=ax.transAxes, family="monospace")

    # MALA schematic (right side)
    ax.text(0.58, 0.48, "Search Trajectory", ha="left", va="top",
            fontsize=11, fontweight="bold", color=C["navy"],
            transform=ax.transAxes)

    # Schematic: anchor point + MALA samples cloud
    np.random.seed(42)
    ax_mala = fig.add_axes([0.59, 0.12, 0.36, 0.30])
    clean_ax(ax_mala)
    ax_mala.set_facecolor(C["bg"])

    # Plausibility ellipse
    theta = np.linspace(0, 2*np.pi, 200)
    ell_x = 2.2 * np.cos(theta); ell_y = 1.4 * np.sin(theta)
    ax_mala.fill(ell_x, ell_y, color=C["blue"], alpha=0.06, zorder=0)
    ax_mala.plot(ell_x, ell_y, color=C["blue"], alpha=0.3, lw=1.0, ls="--")

    # MALA trajectory
    np.random.seed(7)
    n_steps = 25
    traj_x = np.cumsum(np.random.randn(n_steps) * 0.3) * 0.9
    traj_y = np.cumsum(np.random.randn(n_steps) * 0.2) * 0.7
    traj_x -= traj_x[0]; traj_y -= traj_y[0]
    ax_mala.plot(traj_x, traj_y, "-", color=C["blue"], alpha=0.45, lw=1.0, zorder=2)
    ax_mala.scatter(traj_x[5:], traj_y[5:], s=20, color=C["blue"], alpha=0.5, zorder=3)

    # Accepted samples
    np.random.seed(3)
    accepted_x = traj_x[5:][::2] + np.random.randn(len(traj_x[5:][::2]))*0.15
    accepted_y = traj_y[5:][::2] + np.random.randn(len(accepted_x))*0.10
    ax_mala.scatter(accepted_x, accepted_y, s=50, color=C["gold"],
                   edgecolors=C["white"], lw=0.8, zorder=5, label="Accepted scenarios")

    # Anchor
    ax_mala.scatter([0], [0], s=120, color=C["navy"], zorder=6, marker="*",
                   label="Anchor state m₀")

    ax_mala.set_xlim(-3, 3); ax_mala.set_ylim(-2, 2.5)
    ax_mala.set_xticks([]); ax_mala.set_yticks([])
    ax_mala.text(0.0, 0.12, "m₀", ha="center", fontsize=8.5,
                color=C["navy"], fontweight="bold")
    ax_mala.text(-2.2, 1.2, "VAR(1)\nplausibility\nneighborhood",
                ha="center", fontsize=7.5, color=C["blue"], style="italic")
    ax_mala.legend(fontsize=8, frameon=False, loc="lower left")
    ax_mala.set_title("Markov Chain over Macro States", fontsize=9,
                     color=C["subtext"], pad=5)
    ax_mala.spines["left"].set_visible(False)
    ax_mala.spines["bottom"].set_visible(False)

    # VAR(1) prior block
    ax.text(mala_x, 0.23, "VAR(1) Plausibility Prior", ha="left", va="top",
            fontsize=11, fontweight="bold", color=C["navy"],
            transform=ax.transAxes)
    var_lines = [
        "m_{t+1} = c + A m_t + ε          (macro dynamics)",
        "G_reg = (l2/2) · (m − μ)' Q⁻¹ (m − μ)   (Mahalanobis distance)",
        "Fitted on: 2007–2016  |  SIGMA_JITTER = 0.01  (near-singular protection)",
    ]
    for i, line in enumerate(var_lines):
        ax.text(mala_x + 0.005, 0.185 - i*0.050, line,
                ha="left", va="top", fontsize=8.5,
                color=C["text"], family="monospace",
                transform=ax.transAxes)

    save_fig(fig, "slidegraphic_07_scenario_method_v4")


# ---------------------------------------------------------------------------
# FIG 08 — Anchor Context (macro regimes over time)
# ---------------------------------------------------------------------------

def fig08_anchor_context(d):
    fm = d["fm_macro"]

    # Get key macro variables as single time series
    fm_ts = fm.groupby("month_end").first().reset_index()
    fm_ts = fm_ts.set_index("month_end").sort_index()

    # Clip to 2007-2025 range for display
    fm_ts = fm_ts.loc["2007":"2025"]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "Macro Regime Context",
              "Key variable evolution across anchor periods — 2007 to 2025")

    gs = gridspec.GridSpec(3, 1, left=0.08, right=0.97,
                           top=0.87, bottom=0.07, hspace=0.22)

    ax1 = fig.add_subplot(gs[0])   # Rates
    ax2 = fig.add_subplot(gs[1])   # Inflation + VIX proxy
    ax3 = fig.add_subplot(gs[2])   # IG OAS

    for ax in [ax1, ax2, ax3]:
        clean_ax(ax)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)

    # Shade regime periods
    regime_periods = [
        ("2008-09", "2009-06", "pre_gfc",                "GFC Stress"),
        ("2020-02", "2020-05", "covid_shock",             "COVID Shock"),
        ("2021-01", "2022-03", "reflation_risk_on",       "Reflation"),
        ("2022-03", "2023-12", "higher_for_longer",       "Higher for Longer"),
        ("2024-01", "2025-12", "mixed_mid_cycle",         "Mixed / Mid-Cycle"),
    ]

    for ax in [ax1, ax2, ax3]:
        for t0, t1, rkey, rlab in regime_periods:
            col = REGIME_C.get(rkey, C["grey"])
            ax.axvspan(pd.Timestamp(t0), pd.Timestamp(t1),
                       color=col, alpha=0.08, zorder=0)

    # Anchor dates vertical lines (2014-2024 Dec)
    for yr in range(2014, 2025):
        ad = pd.Timestamp(f"{yr}-12-31")
        for ax in [ax1, ax2, ax3]:
            ax.axvline(ad, color=C["gridline"], lw=1.0, ls="-", zorder=1)

    # Panel 1: Short rate US + Real 10Y
    if "short_rate_US" in fm_ts.columns:
        ax1.plot(fm_ts.index, fm_ts["short_rate_US"].fillna(method="ffill"),
                color=C["blue"], lw=1.8, label="Short Rate US")
    if "long_rate_US" in fm_ts.columns:
        ax1.plot(fm_ts.index, fm_ts["long_rate_US"].fillna(method="ffill"),
                color=C["lightblue"], lw=1.5, ls="--", label="Long Rate US")
    # Real rate proxy: long_rate - infl_US
    if "long_rate_US" in fm_ts.columns and "infl_US" in fm_ts.columns:
        real_rate = fm_ts["long_rate_US"] - fm_ts["infl_US"]
        ax1.fill_between(fm_ts.index, 0, real_rate,
                        where=real_rate < 0, color=C["red"], alpha=0.15,
                        label="Negative real rate zone")
        ax1.axhline(0, color=C["navy"], lw=0.8, ls="--")
    ax1.set_ylabel("Rate (%)", fontsize=9)
    ax1.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
    panel_label(ax1, "A  |  Interest Rates")
    ax1.tick_params(labelbottom=False)

    # Panel 2: Inflation US + EA
    if "infl_US" in fm_ts.columns:
        ax2.plot(fm_ts.index, fm_ts["infl_US"].fillna(method="ffill"),
                color=C["orange"], lw=1.8, label="Inflation US")
    if "infl_EA" in fm_ts.columns:
        ax2.plot(fm_ts.index, fm_ts["infl_EA"].fillna(method="ffill"),
                color=C["red"], lw=1.5, ls="--", label="Inflation EA", alpha=0.8)
    ax2.set_ylabel("Inflation (%)", fontsize=9)
    ax2.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    panel_label(ax2, "B  |  Inflation")
    ax2.tick_params(labelbottom=False)

    # Panel 3: IG OAS
    if "ig_oas" in fm_ts.columns:
        ax3.fill_between(fm_ts.index, 0, fm_ts["ig_oas"].fillna(method="ffill"),
                        color=C["purple"], alpha=0.20, zorder=2)
        ax3.plot(fm_ts.index, fm_ts["ig_oas"].fillna(method="ffill"),
                color=C["purple"], lw=1.8, label="IG OAS")
    ax3.set_ylabel("IG OAS (%)", fontsize=9)
    panel_label(ax3, "C  |  Credit Spreads (IG OAS)")

    # Regime labels on bottom panel
    for t0, t1, rkey, rlab in regime_periods:
        mid = pd.Timestamp(t0) + (pd.Timestamp(t1) - pd.Timestamp(t0)) / 2
        col = REGIME_C.get(rkey, C["grey"])
        if mid >= fm_ts.index[0] and mid <= fm_ts.index[-1]:
            ax3.text(mid, ax3.get_ylim()[1] * 0.85 if ax3.get_ylim()[1] > 0 else 1.5,
                    rlab, ha="center", va="top",
                    fontsize=7, color=col, fontweight="bold",
                    rotation=0, clip_on=True)

    # Anchor date labels on top
    for yr in range(2015, 2025):
        ad = pd.Timestamp(f"{yr}-12-31")
        ax1.text(ad, ax1.get_ylim()[1] if len(ax1.get_ylim()) else 8,
                f"Dec\n{yr}", ha="center", va="bottom",
                fontsize=6.5, color=C["subtext"])

    save_fig(fig, "slidegraphic_08_anchor_context_v4")


# ---------------------------------------------------------------------------
# FIG 09 — Gold Activation (strongest scenario figure)
# ---------------------------------------------------------------------------

def fig09_gold_activation(d):
    scenario = d["scenario"]
    anchor_truth = d["anchor_truth"]

    q1 = scenario[scenario["question_id"] == "Q1_gold_favorable"].copy()

    ANCHOR_DATES_STR = ["2021-12-31","2022-12-31","2023-12-31","2024-12-31"]
    ANCHOR_LABELS    = ["Dec 2021","Dec 2022","Dec 2023","Dec 2024"]
    ANCHOR_REGIMES   = {
        "2021-12-31": "reflation_risk_on",
        "2022-12-31": "higher_for_longer",
        "2023-12-31": "higher_for_longer",
        "2024-12-31": "mixed_mid_cycle",
    }

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "Gold Activation Threshold",
              "Real yield sign change + credit stress triggers 3x gold allocation")

    gs = gridspec.GridSpec(2, 2, left=0.07, right=0.97,
                           top=0.85, bottom=0.09,
                           hspace=0.42, wspace=0.30)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    for ax in [ax_A, ax_B, ax_C, ax_D]:
        clean_ax(ax)

    # ---- Panel A: Regime transition ----
    ax_A.axis("off"); ax_A.set_xlim(0, 1); ax_A.set_ylim(0, 1)
    panel_label(ax_A, "A  |  Regime Transition")

    chips = [
        ("2021\nAnchor",   0.12, "reflation_risk_on"),
        ("2022\nAnchor",   0.50, "higher_for_longer"),
        ("2022\nScenario", 0.88, "risk_off_stress"),
    ]
    for lbl, xc, rk in chips:
        regime_chip(ax_A, xc, 0.62, rk, width=0.28, height=0.22, fontsize=8.5)
        ax_A.text(xc, 0.44, lbl, ha="center", va="top",
                 fontsize=8, color=C["subtext"], transform=ax_A.transAxes)
    for x0, x1 in [(0.27, 0.37), (0.65, 0.74)]:
        ax_A.annotate("", xy=(x1, 0.62), xytext=(x0, 0.62),
                     xycoords=ax_A.transAxes, textcoords=ax_A.transAxes,
                     arrowprops=dict(arrowstyle="-|>", color=C["navy"],
                                    lw=2, mutation_scale=14))

    ax_A.text(0.50, 0.26,
             "us_real10y crosses 0  +  ig_oas > 2.0",
             ha="center", va="center", fontsize=9, color=C["subtext"],
             style="italic", transform=ax_A.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1",
                       edgecolor="none"))

    # ---- Panel B: Gold weight ----
    panel_label(ax_B, "B  |  Gold Allocation by Anchor")

    at = anchor_truth.set_index(anchor_truth["anchor_date"].dt.strftime("%Y-%m-%d"))
    anchor_gld = [at.loc[d, "w_ALT_GLD"] * 100 for d in ANCHOR_DATES_STR]
    scen_gld = [q1[q1["anchor_date"]==d]["w_ALT_GLD"].mean() * 100
                for d in ANCHOR_DATES_STR]

    x = np.arange(4); bar_w = 0.35
    bars_a = ax_B.bar(x - bar_w/2, anchor_gld, bar_w,
                      color=C["navy"], alpha=0.75, label="Anchor baseline", zorder=3)
    bars_s = ax_B.bar(x + bar_w/2, scen_gld, bar_w,
                      color=C["gold"], alpha=0.90, label="Scenario mean (Q1)", zorder=3)

    for b in list(bars_a) + list(bars_s):
        h = b.get_height()
        if h > 0:
            ax_B.text(b.get_x() + b.get_width()/2, h + 0.3,
                     f"{h:.1f}%", ha="center", va="bottom",
                     fontsize=7.5, color=C["navy"])

    ax_B.annotate("", xy=(0.5, 22.3 + 0.5), xytext=(0.5 - bar_w, 8.1 + 0.5),
                 arrowprops=dict(arrowstyle="-|>", color=C["gold"],
                                 lw=1.5, mutation_scale=10,
                                 connectionstyle="arc3,rad=-0.3"))
    ax_B.text(0.5, 25, "~3x", ha="center", fontsize=9,
             color=C["gold"], fontweight="bold")

    ax_B.set_xticks(x); ax_B.set_xticklabels(ANCHOR_LABELS, fontsize=9)
    ax_B.set_ylabel("ALT GLD Weight (%)", fontsize=10)
    ax_B.set_ylim(0, 34)
    ax_B.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_B.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax_B.grid(axis="y", alpha=0.6); ax_B.set_axisbelow(True)

    # ---- Panel C: Real yield ----
    panel_label(ax_C, "C  |  US Real 10Y Yield at Anchor")

    ry = [-1.04, 1.58, 2.30, 2.29]
    bar_cols = [C["red"] if v < 0 else C["blue"] for v in ry]
    bars_c = ax_C.bar(x, ry, 0.55, color=bar_cols, alpha=0.85, zorder=3)
    ax_C.axhline(0, color=C["navy"], lw=2.0, zorder=4)
    ax_C.fill_betweenx([-2.0, 0], -0.5, 3.5, color=C["red"],   alpha=0.04)
    ax_C.fill_betweenx([0, 4.0],  -0.5, 3.5, color=C["blue"],  alpha=0.04)

    for b, v in zip(bars_c, ry):
        offset = 0.08 if v >= 0 else -0.15
        va = "bottom" if v >= 0 else "top"
        ax_C.text(b.get_x() + b.get_width()/2, v + offset,
                 f"{v:+.2f}%", ha="center", va=va,
                 fontsize=8.5, color=C["white"] if abs(v) > 0.5 else C["navy"],
                 fontweight="bold")

    ax_C.text(3.4, 0.15, "Gold activation\nzone", ha="right", va="bottom",
             fontsize=7.5, color=C["blue"], style="italic")
    ax_C.text(3.4, -0.15, "Low gold\nallocation", ha="right", va="top",
             fontsize=7.5, color=C["red"], style="italic")

    ax_C.set_xticks(x); ax_C.set_xticklabels(ANCHOR_LABELS, fontsize=9)
    ax_C.set_ylabel("US Real 10Y Yield (%)", fontsize=10)
    ax_C.set_ylim(-2.5, 4.0)
    ax_C.yaxis.set_major_formatter(FuncFormatter(pct2_fmt))
    ax_C.grid(axis="y", alpha=0.6); ax_C.set_axisbelow(True)

    # ---- Panel D: Portfolio composition 2021 vs 2022 ----
    panel_label(ax_D, "D  |  Portfolio Composition  (2021 vs 2022)")

    at_df = anchor_truth.set_index(anchor_truth["anchor_date"].dt.strftime("%Y-%m-%d"))
    grp_sleeves = [
        ("EQ US",    "w_EQ_US",    SLEEVE_C["EQ_US"]),
        ("FI UST",   "w_FI_UST",   SLEEVE_C["FI_UST"]),
        ("CR US HY", "w_CR_US_HY", SLEEVE_C["CR_US_HY"]),
        ("CR US IG", "w_CR_US_IG", SLEEVE_C["CR_US_IG"]),
        ("Gold",     "w_ALT_GLD",  SLEEVE_C["ALT_GLD"]),
    ]

    def other_weight(row, excl_cols):
        total = sum(row.get(c, 0) for _, c, _ in grp_sleeves)
        return max(0, 1.0 - total)

    x_d = np.array([0.0, 1.0])
    bots = np.zeros(2)
    for lbl, col_key, s_col in grp_sleeves:
        vals = np.array([at_df.loc["2021-12-31", col_key] * 100,
                        at_df.loc["2022-12-31", col_key] * 100])
        ax_D.bar(x_d, vals, 0.5, bottom=bots, color=s_col,
                label=lbl, edgecolor=C["white"], lw=0.8, zorder=3)
        for xi, (b, v) in enumerate(zip(bots, vals)):
            if v > 2.0:
                ax_D.text(x_d[xi], b + v/2, f"{v:.0f}%",
                         ha="center", va="center",
                         fontsize=8, color=C["white"], fontweight="bold")
        bots += vals

    # Other
    other_21 = other_weight(at_df.loc["2021-12-31"], grp_sleeves) * 100
    other_22 = other_weight(at_df.loc["2022-12-31"], grp_sleeves) * 100
    ax_D.bar(x_d, [other_21, other_22], 0.5, bottom=bots,
            color=C["lightgrey"], label="Other", edgecolor=C["white"], lw=0.8)
    bots += np.array([other_21, other_22])

    ax_D.set_xticks(x_d)
    ax_D.set_xticklabels(["Dec 2021\n(Reflation)", "Dec 2022\n(Higher for Longer)"],
                         fontsize=9.5)
    ax_D.set_ylabel("Portfolio Weight (%)", fontsize=10)
    ax_D.set_ylim(0, 105)
    ax_D.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_D.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=2)
    ax_D.grid(axis="y", alpha=0.5); ax_D.set_axisbelow(True)

    save_fig(fig, "slidegraphic_09_gold_activation_v4")


# ---------------------------------------------------------------------------
# FIG 10 — Equal-Weight Deviation / Defensive Barbell
# ---------------------------------------------------------------------------

def fig10_ew_deviation(d):
    scenario     = d["scenario"]
    anchor_truth = d["anchor_truth"]

    q2_all = scenario[scenario["question_id"] == "Q2_ew_deviation"].copy()
    q2_22  = q2_all[q2_all["anchor_date"] == "2022-12-31"]

    at = anchor_truth.set_index(anchor_truth["anchor_date"].dt.strftime("%Y-%m-%d"))

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "Defensive Barbell Under Stress Regimes",
              "When the benchmark deviates from equal weight — and why")

    gs = gridspec.GridSpec(1, 2, left=0.07, right=0.97,
                           top=0.85, bottom=0.10, wspace=0.38)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    clean_ax(ax_A); clean_ax(ax_B)

    # ---- Panel A: Dot plot ----
    panel_label(ax_A, "A  |  Weight Comparison — Dec 2022 Anchor")

    KEY_SLEEVES = ["w_FI_UST","w_ALT_GLD","w_CR_US_HY","w_CR_US_IG",
                   "w_EQ_US","w_FI_EU_GOVT","w_RE_US","w_LISTED_INFRA"]
    EW = 1.0 / 14 * 100
    BARBELL = {"w_FI_UST","w_ALT_GLD","w_CR_US_HY"}

    anchor_22 = at.loc["2022-12-31"]
    ew_vals   = [EW] * len(KEY_SLEEVES)
    anch_vals = [anchor_22[s] * 100 for s in KEY_SLEEVES]
    scen_vals = [q2_22[s].mean() * 100 for s in KEY_SLEEVES]
    labels    = [SLEEVE_SHORT[s.replace("w_","")] for s in KEY_SLEEVES]

    order = sorted(range(len(KEY_SLEEVES)), key=lambda i: scen_vals[i], reverse=True)
    labels    = [labels[i] for i in order]
    ew_vals   = [ew_vals[i] for i in order]
    anch_vals = [anch_vals[i] for i in order]
    scen_vals = [scen_vals[i] for i in order]
    ks_ord    = [KEY_SLEEVES[i] for i in order]

    y = np.arange(len(labels))

    for i, s in enumerate(ks_ord):
        if s in BARBELL:
            ax_A.axhspan(i - 0.45, i + 0.45, color="#FFF3DC", alpha=0.9, zorder=0)

    for i in range(len(labels)):
        ax_A.plot([min(ew_vals[i], scen_vals[i]), max(ew_vals[i], scen_vals[i])],
                 [y[i], y[i]], color="#DEE0E3", lw=1.5, zorder=1)

    ax_A.scatter(ew_vals, y, s=55, color=C["lightgrey"], zorder=4,
                marker="D", label="Equal weight (1/14)")
    ax_A.scatter(anch_vals, y, s=70, color=C["navy"], zorder=5,
                label="Anchor baseline")

    for i, s in enumerate(ks_ord):
        col_s = (C["gold"] if s == "w_ALT_GLD"
                 else C["blue"] if s == "w_FI_UST"
                 else C["purple"] if s == "w_CR_US_HY"
                 else C["lightblue"])
        ax_A.scatter(scen_vals[i], y[i], s=90, color=col_s,
                    zorder=6, marker="o")

    bb_total = sum(scen_vals[i] for i, s in enumerate(ks_ord) if s in BARBELL)
    ax_A.text(28, 0.5, f"Barbell:\n{bb_total:.0f}%",
             ha="right", va="bottom", fontsize=9, fontweight="bold",
             color=C["gold"],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3DC",
                       edgecolor=C["gold"], lw=1.2))

    ax_A.set_yticks(y); ax_A.set_yticklabels(labels, fontsize=9.5)
    ax_A.set_xlabel("Portfolio Weight (%)", fontsize=10)
    ax_A.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax_A.set_xlim(-1, 30)
    ax_A.legend(handles=[
                   mpatches.Patch(color=C["lightgrey"], label="Equal weight"),
                   mpatches.Patch(color=C["navy"],      label="Anchor baseline"),
                   mpatches.Patch(color=C["blue"],      label="Scenario mean"),
                   mpatches.Patch(color="#FFF3DC", ec=C["gold"], lw=1,
                                  label="Barbell sleeves"),
               ], fontsize=8, frameon=False, loc="lower right")
    ax_A.grid(axis="x", alpha=0.6); ax_A.set_axisbelow(True)

    # ---- Panel B: Regime context + allocation evolution ----
    panel_label(ax_B, "B  |  Regime Context  &  Cross-Anchor Barbell Pattern")

    ax_B.set_xlim(0, 1); ax_B.set_ylim(0, 1); ax_B.axis("off")

    # Regime chip top
    regime_chip(ax_B, 0.50, 0.88, "risk_off_stress",
                width=0.52, height=0.090, fontsize=10)
    ax_B.text(0.50, 0.82, "All Q2 Dec-2022 samples transition to Risk-Off / Stress",
             ha="center", va="top", fontsize=8.5, color=C["subtext"],
             style="italic", transform=ax_B.transAxes)

    # Dim labels
    dim_items = [
        ("Stress",          "high",    C["red"]),
        ("Policy",          "tight",   C["navy"]),
        ("Inflation",       "high",    C["orange"]),
        ("Growth",          "high",    C["lightgreen"]),
    ]
    dy = [0.72, 0.62, 0.52, 0.42]
    for (dname, dval, dc), yd in zip(dim_items, dy):
        ax_B.text(0.12, yd, dname + ":", ha="right", va="center",
                 fontsize=9, color=C["text"], transform=ax_B.transAxes)
        chip = FancyBboxPatch((0.14, yd - 0.028), 0.20, 0.055,
                              boxstyle="round,pad=0.006",
                              facecolor=dc, edgecolor="none", alpha=0.85,
                              transform=ax_B.transAxes, clip_on=True)
        ax_B.add_patch(chip)
        ax_B.text(0.24, yd, dval.upper(), ha="center", va="center",
                 fontsize=8.5, color=C["white"], fontweight="bold",
                 transform=ax_B.transAxes)

    # Key macro values
    macro_vals = [
        ("ig_oas",      f"{q2_22['ig_oas'].mean():.2f}%",   "IG OAS"),
        ("us_real10y",  f"{q2_22['us_real10y'].mean():.2f}%","US Real 10Y"),
        ("short_US",    f"{q2_22['short_rate_US'].mean():.2f}%","Short Rate US"),
        ("vix",         f"{q2_22['vix'].mean():.1f}",        "VIX"),
    ]
    ax_B.text(0.50, 0.37, "Scenario macro state (Dec 2022 Q2 mean):",
             ha="center", va="top", fontsize=8.5, color=C["subtext"],
             transform=ax_B.transAxes)
    for k, (mk, mv, mlab) in enumerate(macro_vals):
        cx = 0.13 + (k % 2) * 0.46; cy = 0.300 - (k // 2) * 0.075
        bg = FancyBboxPatch((cx - 0.01, cy - 0.030), 0.42, 0.060,
                            boxstyle="round,pad=0.005",
                            facecolor=C["white"], edgecolor=C["gridline"],
                            lw=0.8, transform=ax_B.transAxes, clip_on=True)
        ax_B.add_patch(bg)
        ax_B.text(cx + 0.005, cy, mlab, ha="left", va="center",
                 fontsize=8, color=C["subtext"], transform=ax_B.transAxes)
        ax_B.text(cx + 0.40, cy, mv, ha="right", va="center",
                 fontsize=9.5, fontweight="bold", color=C["navy"],
                 transform=ax_B.transAxes)

    # Takeaway box
    box_take = FancyBboxPatch((0.04, 0.04), 0.92, 0.090,
                              boxstyle="round,pad=0.01",
                              facecolor="#F0F4FF", edgecolor=C["blue"],
                              lw=1.5, transform=ax_B.transAxes, clip_on=True)
    ax_B.add_patch(box_take)
    ax_B.text(0.50, 0.085,
             "Under risk-off stress, FI UST + Gold + HY Credit collectively exceed 60%."
             "\nEquity is compressed to ~10%. This is a barbell — not a gradual tilt.",
             ha="center", va="center", fontsize=8.5, color=C["navy"],
             style="italic", transform=ax_B.transAxes, multialignment="center")

    save_fig(fig, "slidegraphic_10_equal_weight_deviation_v4")


# ---------------------------------------------------------------------------
# FIG 11 — House-View Gap
# ---------------------------------------------------------------------------

def fig11_house_view_gap(d):
    scenario     = d["scenario"]
    anchor_truth = d["anchor_truth"]

    q3 = scenario[scenario["question_id"] == "Q3_house_view_5pct"].copy()
    q3["pred_return_pct"] = q3["pred_return"] * 100

    ANCHOR_DATES_STR = ["2021-12-31","2022-12-31","2023-12-31","2024-12-31"]
    ANCHOR_LABELS    = ["Dec 2021","Dec 2022","Dec 2023","Dec 2024"]
    ANCHOR_RETS      = [2.01, 3.30, 2.82, 2.24]
    HV_TARGET        = 5.0

    scen_means = []
    scen_maxs  = []
    scen_mins  = []
    for d_str in ANCHOR_DATES_STR:
        sub = q3[q3["anchor_date"] == d_str]["pred_return_pct"]
        scen_means.append(sub.mean())
        scen_maxs.append(sub.max())
        scen_mins.append(sub.min())

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])
    fig_title(fig, "The Return Ceiling",
              "No locally plausible macro scenario reaches the 5% house-view assumption")

    gs = gridspec.GridSpec(1, 2, left=0.07, right=0.97,
                           top=0.85, bottom=0.10, wspace=0.38)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    clean_ax(ax_A); clean_ax(ax_B)

    # ---- Panel A: Return range chart ----
    panel_label(ax_A, "A  |  Predicted Return vs House-View Target")

    x = np.arange(4); bar_w = 0.28

    ax_A.axhline(HV_TARGET, color=C["red"], lw=2.0, ls="--", zorder=5)
    ax_A.text(3.55, HV_TARGET + 0.08, "5% Target",
             ha="right", va="bottom", fontsize=9,
             color=C["red"], fontweight="bold")

    for i in range(4):
        ax_A.fill_between([x[i] - bar_w*1.1, x[i] + bar_w*1.1],
                         scen_mins[i], scen_maxs[i],
                         color=C["blue"], alpha=0.15, zorder=2)
        ax_A.fill_between([x[i] - bar_w*1.1, x[i] + bar_w*1.1],
                         scen_maxs[i], HV_TARGET,
                         color=C["red"], alpha=0.05, zorder=2,
                         hatch="///", linewidth=0)

    bars_a = ax_A.bar(x - bar_w/2, ANCHOR_RETS, bar_w,
                     color=C["navy"], alpha=0.75,
                     label="Anchor baseline", zorder=4)
    bars_s = ax_A.bar(x + bar_w/2, scen_means, bar_w,
                     color=C["blue"], alpha=0.80,
                     label="Scenario mean", zorder=4)
    ax_A.scatter(x + bar_w/2, scen_maxs, s=85,
                color=C["white"], edgecolors=C["blue"],
                linewidth=2, zorder=7, label="Scenario max")

    gaps = [HV_TARGET - mx for mx in scen_maxs]
    for i in range(4):
        mx = x[i] + bar_w/2
        ax_A.annotate("", xy=(mx, HV_TARGET - 0.05),
                     xytext=(mx, scen_maxs[i] + 0.05),
                     arrowprops=dict(arrowstyle="<->",
                                     color=C["red"], lw=1.2, mutation_scale=8))
        ax_A.text(mx + 0.18, (HV_TARGET + scen_maxs[i]) / 2,
                 f"-{gaps[i]:.2f}pp",
                 ha="left", va="center", fontsize=8,
                 color=C["red"], fontweight="bold")

    for b in list(bars_a) + list(bars_s):
        h = b.get_height()
        ax_A.text(b.get_x() + b.get_width()/2, h + 0.05,
                 f"{h:.2f}%", ha="center", va="bottom",
                 fontsize=7, color=C["navy"])

    ax_A.set_xticks(x); ax_A.set_xticklabels(ANCHOR_LABELS, fontsize=9.5)
    ax_A.set_ylabel("Predicted 5Y Ann. Excess Return (%)", fontsize=10)
    ax_A.set_ylim(0, 6.8)
    ax_A.yaxis.set_major_formatter(FuncFormatter(pct2_fmt))
    ax_A.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    ax_A.grid(axis="y", alpha=0.6); ax_A.set_axisbelow(True)

    # ---- Panel B: Best-case card ----
    panel_label(ax_B, "B  |  Best-Case Achievable Scenario")

    best = q3.loc[q3["pred_return"].idxmax()]

    ax_B.set_xlim(0, 1); ax_B.set_ylim(0, 1); ax_B.axis("off")

    ax_B.text(0.50, 0.95, "Best Achievable State",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=C["navy"], transform=ax_B.transAxes)

    regime_chip(ax_B, 0.50, 0.84, best["regime_label"],
                width=0.52, height=0.090, fontsize=10)

    ax_B.text(0.50, 0.78,
             f"Dec 2022 anchor  |  pred_return = {best['pred_return']*100:.2f}%",
             ha="center", va="top", fontsize=9, color=C["subtext"],
             style="italic", transform=ax_B.transAxes)

    macro_rows = [
        ("Short Rate US",  f"{best['short_rate_US']:.2f}%"),
        ("US Real 10Y",    f"{best['us_real10y']:.2f}%"),
        ("IG OAS",         f"{best['ig_oas']:.2f}%"),
        ("VIX",            f"{best['vix']:.1f}"),
        ("Inflation US",   f"{best['infl_US']:.2f}%"),
    ]
    row_y = [0.68, 0.605, 0.530, 0.455, 0.380]
    for (lbl, val), ry in zip(macro_rows, row_y):
        bg = FancyBboxPatch((0.08, ry - 0.030), 0.84, 0.060,
                            boxstyle="round,pad=0.005",
                            facecolor=C["white"], edgecolor=C["gridline"],
                            lw=0.8, transform=ax_B.transAxes, clip_on=True)
        ax_B.add_patch(bg)
        ax_B.text(0.18, ry, lbl, ha="left", va="center",
                 fontsize=9.5, color=C["subtext"], transform=ax_B.transAxes)
        ax_B.text(0.82, ry, val, ha="right", va="center",
                 fontsize=10, fontweight="bold", color=C["navy"],
                 transform=ax_B.transAxes)

    # Gap table
    gap_box = FancyBboxPatch((0.06, 0.030), 0.88, 0.290,
                             boxstyle="round,pad=0.01",
                             facecolor="#FEF2F2", edgecolor=C["red"],
                             lw=1.5, transform=ax_B.transAxes, clip_on=True)
    ax_B.add_patch(gap_box)

    ax_B.text(0.50, 0.305, "Gap to 5% Target — By Anchor",
             ha="center", va="top", fontsize=10,
             fontweight="bold", color=C["red"],
             transform=ax_B.transAxes)

    gap_data = [
        ("Dec 2021", 2.00, 3.00),
        ("Dec 2022", 3.08, 1.92),
        ("Dec 2023", 2.82, 2.18),
        ("Dec 2024", 1.88, 3.12),
    ]
    for k, (label, best_ret, gap) in enumerate(gap_data):
        col_k = k % 2; row_k = k // 2
        xp = 0.08 + col_k * 0.46; yp = 0.230 - row_k * 0.085
        ax_B.text(xp, yp, label, ha="left", va="top",
                 fontsize=8.5, color=C["subtext"],
                 transform=ax_B.transAxes)
        ax_B.text(xp + 0.19, yp, f"best: {best_ret:.2f}%",
                 ha="left", va="top", fontsize=8.5, color=C["navy"],
                 transform=ax_B.transAxes)
        ax_B.text(xp + 0.40, yp, f"-{gap:.2f}pp",
                 ha="right", va="top", fontsize=9,
                 fontweight="bold", color=C["red"],
                 transform=ax_B.transAxes)

    ax_B.text(0.50, 0.052,
             "Even in best-case macro state: ceiling = 3.08%  (2022 anchor)\n"
             "Gap reflects post-GFC return compression — a model signal, not a bug.",
             ha="center", va="bottom", fontsize=8.5, color=C["navy"],
             style="italic", transform=ax_B.transAxes,
             multialignment="center")

    save_fig(fig, "slidegraphic_11_house_view_gap_v4")


# ---------------------------------------------------------------------------
# INDEX MARKDOWN
# ---------------------------------------------------------------------------

def write_index():
    index_path = REPORTS / "final_slidegraphics_index_v4.md"
    content = """# Final Slidegraphics Index — XOPTPOE v4

Generated: 2026-03-30
Output: `reports/final_slidegraphics_v4/`  |  All figures: 16:9, 300 dpi PNG + vector PDF

---

## Figure 01 — Big Picture Pipeline
**File:** `slidegraphic_01_big_picture_v4`
**Purpose:** Section opener. Shows the full pipeline at a glance: Data -> AI -> Allocation -> Benchmark -> Scenario.
**Storyline position:** Slide 1 of the deck (after title slide).
**Speaker message:** "Here is the full pipeline. Data and features feed an AI prediction layer; predictions feed a robust allocator; the resulting benchmark is then probed by a scenario engine."

---

## Figure 02 — Investable Universe
**File:** `slidegraphic_02_investable_universe_v4`
**Purpose:** Shows the 14 sleeves grouped by asset class with proxy instruments and geography.
**Storyline position:** Early — immediately after big picture, before modeling deep-dive.
**Speaker message:** "We cover 14 sleeves across five groups. The universe is deliberately broad: equity, rates, credit, real assets, and gold."

---

## Figure 03 — Data, Features, Targets
**File:** `slidegraphic_03_data_features_targets_v4`
**Purpose:** Explains the data design: macro features, interaction construction, 5Y target definition.
**Storyline position:** Data/modeling section.
**Speaker message:** "303 features — macro, rate, stress, momentum, and their interactions — feed into separate elastic net models, one per sleeve, predicting 5Y annualized excess return."

---

## Figure 04 — AI Prediction Layer
**File:** `slidegraphic_04_ai_prediction_layer_v4`
**Purpose:** Explains the prediction model. Left: architecture card. Right: out-of-sample prediction scatter (validation period, corr=0.76).
**Storyline position:** AI/prediction section.
**Speaker message:** "The model achieves a prediction correlation of 0.76 on held-out data. The scatter is noisy — this is intentional; we rely on robust optimization to handle uncertainty downstream."

---

## Figure 05 — Allocation Layer
**File:** `slidegraphic_05_allocation_layer_v4`
**Purpose:** Explains the robust MVO optimizer. Shows the objective formula, key design choices, and a compact allocation snapshot across anchor years.
**Storyline position:** After prediction section.
**Speaker message:** "Predicted returns are not the final output. They feed a robust optimizer that penalizes risk and excessive deviation from equal weight. The resulting allocations are stable and interpretable."

---

## Figure 06 — Hero Benchmark Behavior *** STRONGEST SINGLE SLIDE ***
**File:** `slidegraphic_06_hero_benchmark_behavior_v4`
**Purpose:** The hero empirical figure. Top panel: wealth path vs equal weight (2.04x vs 1.80x). Bottom panel: annual rebalance allocation composition.
**Storyline position:** Benchmark results section — centerpiece of the deck.
**Speaker message:** "The benchmark compounds to 2.04x over the decade vs 1.80x for equal weight. The allocation composition shows a clear regime shift after 2022: gold and credit grow materially, equity is compressed."

---

## Figure 07 — Scenario Method
**File:** `slidegraphic_07_scenario_method_v4`
**Purpose:** Explains the scenario generation method: pipeline, probe objective, VAR(1) prior, MALA search, regime labels.
**Storyline position:** Opens the scenario section.
**Speaker message:** "Scenario generation is not brute-force stress testing. We use a gradient-based sampler that searches over locally plausible macro states — constrained by a VAR(1) prior — to find states that answer specific questions about the benchmark."

---

## Figure 08 — Anchor Context
**File:** `slidegraphic_08_anchor_context_v4`
**Purpose:** Shows macro variable evolution 2007-2025 with regime-shaded periods and anchor date markers.
**Storyline position:** Contextualizes anchor dates before scenario results.
**Speaker message:** "Our four anchor dates — 2021 through 2024 — span a dramatic macro transition: from negative real rates and reflation to the highest short rates in 15 years."

---

## Figure 09 — Gold Activation Threshold
**File:** `slidegraphic_09_gold_activation_v4`
**Purpose:** Strongest scenario result. Shows regime transition, real yield threshold, 3x gold jump, and portfolio composition change.
**Storyline position:** First main scenario result.
**Speaker message:** "Gold allocation is dormant when real yields are negative. Once us_real10y crosses zero — as it did in 2022 — gold triples. The model learned this as a threshold, not a gradual relationship."

---

## Figure 10 — Defensive Barbell
**File:** `slidegraphic_10_equal_weight_deviation_v4`
**Purpose:** Shows when/why the benchmark deviates from equal weight. Dot plot comparison plus regime context.
**Storyline position:** Second main scenario result.
**Speaker message:** "In risk-off stress regimes, the optimizer concentrates into a defensive barbell: FI UST, gold, and HY credit together exceed 60%. Equity is crowded out systematically."

---

## Figure 11 — House-View Gap
**File:** `slidegraphic_11_house_view_gap_v4`
**Purpose:** Shows the structural return ceiling. Scenario search cannot close the gap to a 5% house-view return assumption.
**Storyline position:** Third (and final) scenario result — closes the deck.
**Speaker message:** "The model's ceiling is 3.08%, achieved at the 2022 anchor. No locally plausible macro state reaches 5%. This is a calibration signal — the model was trained in a post-GFC low-return world, and it reflects that honestly."

---

## Summary Table

| Fig | File | Deck | Priority |
|-----|------|------|----------|
| 01 | big_picture | Opener | Required |
| 02 | investable_universe | Context | Required |
| 03 | data_features | Methodology | Required |
| 04 | ai_prediction | Methodology | Required |
| 05 | allocation_layer | Methodology | Required |
| 06 | hero_benchmark | **Results — centerpiece** | **Must-have** |
| 07 | scenario_method | Scenario intro | Required |
| 08 | anchor_context | Scenario context | Recommended |
| 09 | gold_activation | **Scenario hero** | **Must-have** |
| 10 | equal_weight_deviation | Scenario result | Required |
| 11 | house_view_gap | Scenario close | Required |

**Strongest single figure:** Fig 06 (Hero Benchmark) or Fig 09 (Gold Activation)
**Optional/appendix:** Fig 08 (Anchor Context) can be moved to appendix in a short deck.
"""
    with open(index_path, "w") as f:
        f.write(content)
    print(f"  Index: {index_path.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("\nLoading data...")
    data = load_data()
    print("  Data loaded.\n")

    print("Building slidegraphics...")

    print("\n[01] Big Picture Pipeline")
    fig01_big_picture(data)

    print("\n[02] Investable Universe")
    fig02_universe(data)

    print("\n[03] Data, Features, Targets")
    fig03_data_features(data)

    print("\n[04] AI Prediction Layer")
    fig04_ai_prediction(data)

    print("\n[05] Allocation Layer")
    fig05_allocation(data)

    print("\n[06] Hero Benchmark Behavior")
    fig06_hero(data)

    print("\n[07] Scenario Method")
    fig07_scenario_method(data)

    print("\n[08] Anchor Context")
    fig08_anchor_context(data)

    print("\n[09] Gold Activation")
    fig09_gold_activation(data)

    print("\n[10] Equal-Weight Deviation")
    fig10_ew_deviation(data)

    print("\n[11] House-View Gap")
    fig11_house_view_gap(data)

    print("\nWriting index...")
    write_index()

    files = sorted(OUTDIR.glob("*.png"))
    print(f"\nDone. {len(files)*2} files (PNG + PDF) in {OUTDIR}\n")


if __name__ == "__main__":
    main()
