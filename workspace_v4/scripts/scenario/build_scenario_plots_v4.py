"""
build_scenario_plots_v4.py
Conference-grade matplotlib figures for XOPTPOE v4 scenario analysis.
Run from workspace_v4/:
    python scripts/scenario/build_scenario_plots_v4.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # workspace_v4/
REPORTS = ROOT / "reports"
OUTDIR = REPORTS / "scenario_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.color": "#ECF0F1",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10,
    "axes.labelpad": 6,
})

COLORS = {
    "anchor":     "#2C3E50",
    "scenario":   "#2980B9",
    "positive":   "#2980B9",
    "negative":   "#C0392B",
    "gold":       "#F39C12",
    "defensive":  "#7F8C8D",
    "equity":     "#27AE60",
    "credit":     "#8E44AD",
    "stress":     "#E74C3C",
    "hv_target":  "#E74C3C",
    "bg":         "#FAFAFA",
    "grid":       "#ECF0F1",
    "fi":         "#2980B9",
}

REGIME_COLORS = {
    "reflation_risk_on":        "#27AE60",
    "higher_for_longer":        "#E67E22",
    "risk_off_stress":          "#C0392B",
    "mixed_mid_cycle":          "#7F8C8D",
    "disinflationary_slowdown": "#2980B9",
    "soft_landing":             "#1ABC9C",
}

REGIME_LABELS = {
    "reflation_risk_on":        "Reflation / Risk-On",
    "higher_for_longer":        "Higher for Longer",
    "risk_off_stress":          "Risk-Off / Stress",
    "mixed_mid_cycle":          "Mixed / Mid-Cycle",
    "disinflationary_slowdown": "Disinflationary Slowdown",
    "soft_landing":             "Soft Landing",
}

SLEEVE_COLORS = {
    "w_EQ_US":       COLORS["equity"],
    "w_EQ_EZ":       "#52BE80",
    "w_EQ_JP":       "#A9DFBF",
    "w_EQ_CN":       "#1E8449",
    "w_EQ_EM":       "#196F3D",
    "w_FI_UST":      COLORS["fi"],
    "w_FI_EU_GOVT":  "#5DADE2",
    "w_CR_US_IG":    "#9B59B6",
    "w_CR_EU_IG":    "#C39BD3",
    "w_CR_US_HY":    COLORS["credit"],
    "w_RE_US":       "#BDC3C7",
    "w_LISTED_RE":   "#AAB7B8",
    "w_LISTED_INFRA":"#99A3A4",
    "w_ALT_GLD":     COLORS["gold"],
}

SLEEVE_SHORT = {
    "w_EQ_US":       "EQ US",
    "w_EQ_EZ":       "EQ EZ",
    "w_EQ_JP":       "EQ JP",
    "w_EQ_CN":       "EQ CN",
    "w_EQ_EM":       "EQ EM",
    "w_FI_UST":      "FI UST",
    "w_FI_EU_GOVT":  "FI EU",
    "w_CR_US_IG":    "CR US IG",
    "w_CR_EU_IG":    "CR EU IG",
    "w_CR_US_HY":    "CR US HY",
    "w_RE_US":       "RE US",
    "w_LISTED_RE":   "Listed RE",
    "w_LISTED_INFRA":"Listed Infra",
    "w_ALT_GLD":     "ALT GLD",
}

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
anchor = pd.read_csv(REPORTS / "scenario_anchor_truth_v4.csv")
results = pd.read_csv(REPORTS / "scenario_results_v4.csv")

# Anchor regime labels (hardcoded from verified key facts)
ANCHOR_REGIMES = {
    "2021-12-31": "reflation_risk_on",
    "2022-12-31": "higher_for_longer",
    "2023-12-31": "higher_for_longer",
    "2024-12-31": "mixed_mid_cycle",
}
ANCHOR_DATES = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]
ANCHOR_LABELS = ["Dec 2021", "Dec 2022", "Dec 2023", "Dec 2024"]

# Convert pred_return to percent in results
results["pred_return_pct"] = results["pred_return"] * 100

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def save_figure(fig, name):
    """Save figure as PNG and PDF to OUTDIR."""
    png_path = OUTDIR / f"{name}.png"
    pdf_path = OUTDIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=COLORS["bg"])
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"  Saved: {png_path.name}  |  {pdf_path.name}")
    return png_path, pdf_path


def regime_chip(ax, x, y, label, regime_key, width=0.16, height=0.055,
                fontsize=9, transform=None):
    """Draw a rounded-rectangle regime chip centered at (x,y) in axes coords."""
    if transform is None:
        transform = ax.transAxes
    color = REGIME_COLORS.get(regime_key, "#999999")
    text = REGIME_LABELS.get(regime_key, label)
    # Background box
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor="white", linewidth=1.5,
        transform=transform, clip_on=False, zorder=5,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white",
            transform=transform, clip_on=False, zorder=6)


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(COLORS["bg"])


def supra_title(fig, title, subtitle=None):
    """Add a supra-title + optional subtitle to a figure."""
    y = 0.97 if subtitle else 0.97
    fig.text(0.5, y, title, ha="center", va="top",
             fontsize=17, fontweight="bold", color=COLORS["anchor"])
    if subtitle:
        fig.text(0.5, y - 0.044, subtitle, ha="center", va="top",
                 fontsize=11, color="#5D6D7E", style="italic")


# ---------------------------------------------------------------------------
# FIGURE 1: Story Overview
# ---------------------------------------------------------------------------

def build_fig1():
    """3-panel story overview — section opener."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["bg"])

    supra_title(fig, "Three Portfolio Questions", "XOPTPOE v4 — Scenario Analysis Overview")

    stories = [
        {
            "id": "Q1",
            "title": "Gold Activation Threshold",
            "anchor_regime": "reflation_risk_on",
            "scenario_regime": "risk_off_stress",
            "bullets": [
                "Gold weight: 8.1%  ->  22.3%+ across anchors",
                "Trigger: real yield crosses zero + ig_oas > 2.0",
                "us_real10y: -1.04%  ->  +1.58%  (2021->2022)",
                "EQ US compresses from 25.6%  to  13.7%  (2021->2022)",
            ],
            "takeaway": "A sign change in real yields is the threshold — not magnitude.",
            "color_accent": COLORS["gold"],
        },
        {
            "id": "Q2",
            "title": "Defensive Barbell Displacement",
            "anchor_regime": "higher_for_longer",
            "scenario_regime": "risk_off_stress",
            "bullets": [
                "FI UST + ALT GLD + CR US HY ≈ 60%+ of portfolio",
                "EQ US compressed to ~10% in stress scenarios",
                "Equal-weight baseline (1/14 ≈ 7.1%) far exceeded",
                "Barbell pattern stable across 2022–2023 anchors",
            ],
            "takeaway": "Stress regimes produce a barbell — not a gradual tilt.",
            "color_accent": COLORS["defensive"],
        },
        {
            "id": "Q3",
            "title": "The Return Ceiling",
            "anchor_regime": "higher_for_longer",
            "scenario_regime": "higher_for_longer",
            "bullets": [
                "Max achievable pred_return: 3.08% (2022 anchor)",
                "House-view target: 5.00%",
                "Gap: −1.92pp to −3.12pp across all anchors",
                "Best-case macro state cannot close the gap",
            ],
            "takeaway": "No locally plausible macro scenario reaches the 5% target.",
            "color_accent": COLORS["hv_target"],
        },
    ]

    # 3 columns with generous margins
    left_starts = [0.04, 0.37, 0.70]
    col_width = 0.28

    for i, story in enumerate(stories):
        xl = left_starts[i]
        xr = xl + col_width

        # Column background
        bg = FancyBboxPatch(
            (xl, 0.08), col_width, 0.80,
            boxstyle="round,pad=0.01",
            facecolor="white", edgecolor=story["color_accent"],
            linewidth=2.0,
            transform=fig.transFigure, clip_on=False, zorder=1,
        )
        fig.add_artist(bg)

        xcenter = xl + col_width / 2

        # Story ID badge
        badge = FancyBboxPatch(
            (xl + 0.005, 0.84), 0.036, 0.032,
            boxstyle="round,pad=0.005",
            facecolor=story["color_accent"], edgecolor="none",
            transform=fig.transFigure, clip_on=False, zorder=3,
        )
        fig.add_artist(badge)
        fig.text(xl + 0.023, 0.856, story["id"],
                 ha="center", va="center", fontsize=9, fontweight="bold",
                 color="white", transform=fig.transFigure, zorder=4)

        # Story title
        fig.text(xl + 0.050, 0.856, story["title"],
                 ha="left", va="center", fontsize=12, fontweight="bold",
                 color=COLORS["anchor"], transform=fig.transFigure, zorder=4)

        # Regime transition chips — drawn in figure coordinates
        # We'll use a hidden axes spanning the column
        ax_col = fig.add_axes([xl + 0.005, 0.68, col_width - 0.01, 0.12])
        ax_col.set_xlim(0, 1)
        ax_col.set_ylim(0, 1)
        ax_col.axis("off")
        ax_col.set_facecolor("none")
        ax_col.patch.set_alpha(0)

        # Anchor chip
        regime_chip(ax_col, 0.22, 0.55, "", story["anchor_regime"],
                    width=0.38, height=0.48, fontsize=8)
        # Arrow
        ax_col.annotate("", xy=(0.63, 0.55), xytext=(0.44, 0.55),
                        arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                        lw=1.5, mutation_scale=12),
                        annotation_clip=False)
        # Scenario chip
        regime_chip(ax_col, 0.80, 0.55, "", story["scenario_regime"],
                    width=0.34, height=0.48, fontsize=8)

        # Label above chips
        ax_col.text(0.22, 0.95, "Anchor", ha="center", va="top",
                    fontsize=7.5, color="#7F8C8D")
        ax_col.text(0.80, 0.95, "Scenario", ha="center", va="top",
                    fontsize=7.5, color="#7F8C8D")

        # Divider line
        fig.add_artist(mpl.lines.Line2D(
            [xl + 0.01, xr - 0.01], [0.67, 0.67],
            transform=fig.transFigure, color=COLORS["grid"],
            linewidth=1.0, zorder=3,
        ))

        # Bullets
        bullet_y_start = 0.635
        for j, bullet in enumerate(story["bullets"]):
            fig.text(xl + 0.016, bullet_y_start - j * 0.055,
                     f"•  {bullet}",
                     ha="left", va="top", fontsize=9,
                     color="#2C3E50", transform=fig.transFigure, zorder=4,
                     wrap=True)

        # Divider before takeaway
        fig.add_artist(mpl.lines.Line2D(
            [xl + 0.01, xr - 0.01], [0.155, 0.155],
            transform=fig.transFigure, color=story["color_accent"],
            linewidth=0.8, alpha=0.5, zorder=3,
        ))

        # Takeaway
        fig.text(xcenter, 0.125, f"{story['takeaway']}",
                 ha="center", va="center", fontsize=8.5,
                 color="#5D6D7E", style="italic",
                 transform=fig.transFigure, zorder=4,
                 wrap=False)

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# FIGURE 2: Gold Activation
# ---------------------------------------------------------------------------

def build_fig2():
    """Gold Activation Threshold — headline figure."""
    q1 = results[results["question_id"] == "Q1_gold_favorable"].copy()

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    supra_title(fig, "Gold Activation Threshold",
                "Real yield sign change + credit stress triggers 3× gold allocation")

    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97,
                          top=0.86, bottom=0.09,
                          hspace=0.42, wspace=0.32)

    ax_A = fig.add_subplot(gs[0, 0])   # Regime transition
    ax_B = fig.add_subplot(gs[0, 1])   # Gold allocation bars
    ax_C = fig.add_subplot(gs[1, 0])   # Real yield shift
    ax_D = fig.add_subplot(gs[1, 1])   # Portfolio stacked bars

    for ax in [ax_A, ax_B, ax_C, ax_D]:
        clean_ax(ax)

    # ---- Panel A: Regime transition diagram --------------------------------
    ax_A.set_xlim(0, 1)
    ax_A.set_ylim(0, 1)
    ax_A.axis("off")
    ax_A.set_title("A  |  Regime Transition", loc="left", fontsize=11,
                   fontweight="bold", color=COLORS["anchor"], pad=8)

    # Three chips across
    positions = [
        ("2021\nAnchor", 0.12, 0.62, "reflation_risk_on"),
        ("2022\nAnchor", 0.50, 0.62, "higher_for_longer"),
        ("2022\nScenario", 0.88, 0.62, "risk_off_stress"),
    ]
    chip_w, chip_h = 0.28, 0.22
    for label, x, y, regime in positions:
        regime_chip(ax_A, x, y, "", regime,
                    width=chip_w, height=chip_h, fontsize=8.5)
        ax_A.text(x, y - 0.17, label, ha="center", va="top",
                  fontsize=8, color="#5D6D7E")

    for x_start, x_end in [(0.27, 0.37), (0.65, 0.74)]:
        ax_A.annotate("", xy=(x_end, 0.62), xytext=(x_start, 0.62),
                      arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                      lw=2.0, mutation_scale=14),
                      annotation_clip=False)

    # Macro driver note
    ax_A.text(0.50, 0.24,
              "us_real10y crosses zero\n+ ig_oas > 2.0",
              ha="center", va="center", fontsize=9, color="#5D6D7E",
              style="italic",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECF0F1",
                        edgecolor="none"))

    # ---- Panel B: Gold allocation bars ------------------------------------
    ax_B.set_title("B  |  Gold Allocation", loc="left", fontsize=11,
                   fontweight="bold", color=COLORS["anchor"], pad=8)

    anchor_gld = [8.1, 22.3, 22.4, 23.2]
    scenario_gld = q1.groupby("anchor_date")["w_ALT_GLD"].mean().reindex(ANCHOR_DATES).values * 100

    x = np.arange(len(ANCHOR_DATES))
    bar_w = 0.35

    bars_a = ax_B.bar(x - bar_w / 2, anchor_gld, bar_w,
                      color=COLORS["anchor"], alpha=0.75, label="Anchor baseline",
                      zorder=3)
    bars_s = ax_B.bar(x + bar_w / 2, scenario_gld, bar_w,
                      color=COLORS["gold"], alpha=0.90, label="Scenario mean (Q1)",
                      zorder=3)

    ax_B.set_xticks(x)
    ax_B.set_xticklabels(ANCHOR_LABELS, fontsize=9)
    ax_B.set_ylabel("ALT GLD Weight (%)", fontsize=10)
    ax_B.set_ylim(0, 32)
    ax_B.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f%%"))
    ax_B.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax_B.grid(axis="y", alpha=0.6)
    ax_B.set_axisbelow(True)

    # Value labels on bars
    for bar in list(bars_a) + list(bars_s):
        h = bar.get_height()
        ax_B.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                  f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5,
                  color="#2C3E50")

    # Annotate 3× jump arrow
    ax_B.annotate("", xy=(0.5, 22.3 + 0.4), xytext=(0.5 - bar_w, 8.1 + 0.4),
                  arrowprops=dict(arrowstyle="-|>", color=COLORS["gold"],
                                  lw=1.5, mutation_scale=10,
                                  connectionstyle="arc3,rad=-0.3"))
    ax_B.text(0.5, 26, "3× jump", ha="center", fontsize=8,
              color=COLORS["gold"], fontweight="bold")

    # ---- Panel C: Real yield shift ----------------------------------------
    ax_C.set_title("C  |  Real Yield Shift", loc="left", fontsize=11,
                   fontweight="bold", color=COLORS["anchor"], pad=8)

    ry_anchor = [-1.04, 1.58, 2.30, 2.29]
    x_c = np.arange(len(ANCHOR_DATES))

    bar_colors = [COLORS["negative"] if v < 0 else COLORS["positive"] for v in ry_anchor]
    bars_c = ax_C.bar(x_c, ry_anchor, 0.55, color=bar_colors, alpha=0.85, zorder=3)

    ax_C.axhline(0, color=COLORS["anchor"], linewidth=2.0, zorder=4, label="Zero threshold")
    ax_C.fill_betweenx([-2.5, 0], -0.5, 3.5, color=COLORS["negative"], alpha=0.04, zorder=1)
    ax_C.fill_betweenx([0, 4.0], -0.5, 3.5, color=COLORS["positive"], alpha=0.04, zorder=1)

    ax_C.set_xticks(x_c)
    ax_C.set_xticklabels(ANCHOR_LABELS, fontsize=9)
    ax_C.set_ylabel("US Real 10Y Yield (%)", fontsize=10)
    ax_C.set_ylim(-2.5, 4.0)
    ax_C.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f%%"))
    ax_C.grid(axis="y", alpha=0.6)
    ax_C.set_axisbelow(True)

    # Labels
    for bar, val in zip(bars_c, ry_anchor):
        offset = 0.08 if val >= 0 else -0.12
        va = "bottom" if val >= 0 else "top"
        ax_C.text(bar.get_x() + bar.get_width() / 2, val + offset,
                  f"{val:+.2f}%", ha="center", va=va, fontsize=8,
                  color="white" if abs(val) > 0.6 else COLORS["anchor"],
                  fontweight="bold")

    ax_C.text(3.45, 0.12, "Gold\nactivation\nzone",
              ha="right", va="bottom", fontsize=7.5, color=COLORS["positive"],
              style="italic")
    ax_C.text(3.45, -0.12, "Low gold\nallocation",
              ha="right", va="top", fontsize=7.5, color=COLORS["negative"],
              style="italic")

    # ---- Panel D: Portfolio stacked bars 2021 vs 2022 --------------------
    ax_D.set_title("D  |  Portfolio Composition  (2021 vs 2022 anchor)",
                   loc="left", fontsize=11, fontweight="bold",
                   color=COLORS["anchor"], pad=8)

    # Use 6 sleeve groups: EQ_US, FI_UST, ALT_GLD, CR_US_HY, CR_US_IG, Other
    anchor_row_2021 = anchor[anchor["anchor_date"] == "2021-12-31"].iloc[0]
    anchor_row_2022 = anchor[anchor["anchor_date"] == "2022-12-31"].iloc[0]

    def get_grouped(row):
        d = {
            "EQ US":    row["w_EQ_US"],
            "FI UST":   row["w_FI_UST"],
            "ALT GLD":  row["w_ALT_GLD"],
            "CR US HY": row["w_CR_US_HY"],
            "CR US IG": row["w_CR_US_IG"],
        }
        total = sum(d.values())
        d["Other"] = max(0, 1.0 - total)
        return d

    grp_2021 = get_grouped(anchor_row_2021)
    grp_2022 = get_grouped(anchor_row_2022)
    sleeves = list(grp_2021.keys())

    sleeve_colors_map = {
        "EQ US":    COLORS["equity"],
        "FI UST":   COLORS["fi"],
        "ALT GLD":  COLORS["gold"],
        "CR US HY": COLORS["credit"],
        "CR US IG": "#9B59B6",
        "Other":    "#BDC3C7",
    }

    x_d = np.array([0.0, 1.0])
    bottoms = np.zeros(2)

    for sleeve in sleeves:
        vals = np.array([grp_2021[sleeve] * 100, grp_2022[sleeve] * 100])
        ax_D.bar(x_d, vals, 0.5, bottom=bottoms,
                 color=sleeve_colors_map[sleeve], label=sleeve,
                 edgecolor="white", linewidth=0.8, zorder=3)
        for xi, (bot, val) in enumerate(zip(bottoms, vals)):
            if val > 2.0:
                ax_D.text(x_d[xi], bot + val / 2,
                          f"{val:.0f}%", ha="center", va="center",
                          fontsize=8, color="white", fontweight="bold")
        bottoms += vals

    ax_D.set_xticks(x_d)
    ax_D.set_xticklabels(["Dec 2021\n(Reflation)", "Dec 2022\n(Higher for Longer)"],
                          fontsize=9.5)
    ax_D.set_ylabel("Portfolio Weight (%)", fontsize=10)
    ax_D.set_ylim(0, 105)
    ax_D.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f%%"))
    ax_D.legend(loc="upper right", fontsize=8, frameon=False,
                ncol=2, handlelength=1.0, handleheight=0.8)
    ax_D.grid(axis="y", alpha=0.5)
    ax_D.set_axisbelow(True)

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# FIGURE 3: Equal-Weight Deviation
# ---------------------------------------------------------------------------

def build_fig3():
    """Defensive Barbell Under Stress Regimes."""
    q2 = results[results["question_id"] == "Q2_ew_deviation"].copy()
    q2_2022 = q2[q2["anchor_date"] == "2022-12-31"]

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    supra_title(fig, "Defensive Barbell Under Stress Regimes",
                "FI UST + ALT GLD + CR US HY (~60%) displace equity when stress rises")

    gs = fig.add_gridspec(1, 2, left=0.07, right=0.97,
                          top=0.85, bottom=0.10,
                          hspace=0.0, wspace=0.35)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    clean_ax(ax_A)
    clean_ax(ax_B)

    # ---- Panel A: Dot plot weight comparison -------------------------------
    ax_A.set_title("A  |  Weight Comparison — Dec 2022 Anchor",
                   loc="left", fontsize=11, fontweight="bold",
                   color=COLORS["anchor"], pad=8)

    # Key sleeves only (sorted by Q2 scenario mean descending)
    key_sleeves = ["w_FI_UST", "w_ALT_GLD", "w_CR_US_HY", "w_CR_US_IG",
                   "w_EQ_US", "w_FI_EU_GOVT", "w_RE_US", "w_LISTED_INFRA"]

    anchor_row_2022 = anchor[anchor["anchor_date"] == "2022-12-31"].iloc[0]
    ew_weight = 1.0 / 14 * 100  # equal weight

    ew_vals = [ew_weight] * len(key_sleeves)
    anchor_vals = [anchor_row_2022[s] * 100 for s in key_sleeves]
    scenario_vals = [q2_2022[s].mean() * 100 for s in key_sleeves]
    labels = [SLEEVE_SHORT[s] for s in key_sleeves]

    # Sort by scenario descending
    order = sorted(range(len(key_sleeves)),
                   key=lambda i: scenario_vals[i], reverse=True)
    labels = [labels[i] for i in order]
    ew_vals = [ew_vals[i] for i in order]
    anchor_vals = [anchor_vals[i] for i in order]
    scenario_vals = [scenario_vals[i] for i in order]
    key_sleeves_ord = [key_sleeves[i] for i in order]

    y = np.arange(len(labels))

    # Highlight barbell sleeves
    barbell = {"w_FI_UST", "w_ALT_GLD", "w_CR_US_HY"}
    for i, s in enumerate(key_sleeves_ord):
        if s in barbell:
            ax_A.axhspan(i - 0.45, i + 0.45, color="#FFF9E6", alpha=0.8, zorder=0)

    # Connecting lines
    for i in range(len(labels)):
        ax_A.plot([ew_vals[i], scenario_vals[i]], [y[i], y[i]],
                  color="#DEE0E3", linewidth=1.2, zorder=1)
        ax_A.plot([anchor_vals[i], scenario_vals[i]], [y[i], y[i]],
                  color="#DEE0E3", linewidth=1.2, zorder=1)

    # EW dots
    ax_A.scatter(ew_vals, y, s=55, color="#BDC3C7", zorder=4,
                 label="Equal weight (1/14)", marker="D")
    # Anchor dots
    ax_A.scatter(anchor_vals, y, s=70, color=COLORS["anchor"], zorder=5,
                 label="Anchor baseline")
    # Scenario dots
    sc_colors = [COLORS["gold"] if s == "w_ALT_GLD"
                 else COLORS["fi"] if s == "w_FI_UST"
                 else COLORS["credit"] if s == "w_CR_US_HY"
                 else COLORS["scenario"]
                 for s in key_sleeves_ord]
    for i in range(len(labels)):
        ax_A.scatter(scenario_vals[i], y[i], s=90,
                     color=sc_colors[i], zorder=6, marker="o")

    # Legend proxy for scenario
    proxy_scen = mpatches.Patch(color=COLORS["scenario"], label="Scenario mean (Q2)")
    proxy_barbell = mpatches.Patch(color="#FFF9E6", edgecolor="#F39C12",
                                   linewidth=1, label="Barbell sleeves")

    ax_A.set_yticks(y)
    ax_A.set_yticklabels(labels, fontsize=9.5)
    ax_A.set_xlabel("Portfolio Weight (%)", fontsize=10)
    ax_A.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f%%"))
    ax_A.set_xlim(-1, 30)
    ax_A.legend(handles=[
                    mpatches.Patch(color="#BDC3C7", label="Equal weight (1/14)"),
                    mpatches.Patch(color=COLORS["anchor"], label="Anchor baseline"),
                    proxy_scen, proxy_barbell],
                fontsize=8.5, frameon=False, loc="lower right")
    ax_A.grid(axis="x", alpha=0.6)
    ax_A.set_axisbelow(True)

    # Barbell total annotation
    bb_total = sum(scenario_vals[i] for i, s in enumerate(key_sleeves_ord)
                   if s in barbell)
    ax_A.text(28.5, 0.5, f"Barbell\ntotal:\n{bb_total:.0f}%",
              ha="right", va="bottom", fontsize=9, fontweight="bold",
              color=COLORS["gold"],
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF9E6",
                        edgecolor=COLORS["gold"], linewidth=1.2))

    # ---- Panel B: Regime context card --------------------------------------
    ax_B.set_title("B  |  Regime Context & Macro Drivers",
                   loc="left", fontsize=11, fontweight="bold",
                   color=COLORS["anchor"], pad=8)
    ax_B.set_xlim(0, 1)
    ax_B.set_ylim(0, 1)
    ax_B.axis("off")

    # Regime chip at top
    regime_chip(ax_B, 0.5, 0.87, "", "risk_off_stress",
                width=0.55, height=0.09, fontsize=10)
    ax_B.text(0.5, 0.81, "All Q2 Dec-2022 samples -> Risk-Off / Stress",
              ha="center", va="top", fontsize=8.5, color="#5D6D7E", style="italic")

    # Regime dimension labels (categorical — display as chips/labels)
    dim_items = [
        ("Stress",           q2_2022["dim_stress"].mode()[0],    COLORS["stress"]),
        ("Policy",           q2_2022["dim_policy"].mode()[0],    COLORS["anchor"]),
        ("Financial Cond.",  q2_2022["dim_fin_cond"].mode()[0],  COLORS["fi"]),
        ("Growth",           q2_2022["dim_growth"].mode()[0],    COLORS["equity"]),
    ]

    y_d = np.array([0.66, 0.53, 0.40, 0.27])

    for j, (dim_name, dim_val, dc) in enumerate(dim_items):
        ax_B.text(0.13, y_d[j], dim_name,
                  ha="right", va="center", fontsize=9, color="#2C3E50")
        chip = FancyBboxPatch(
            (0.15, y_d[j] - 0.025), 0.30, 0.05,
            boxstyle="round,pad=0.005",
            facecolor=dc, edgecolor="none", alpha=0.8,
            transform=ax_B.transAxes, clip_on=True,
        )
        ax_B.add_patch(chip)
        ax_B.text(0.30, y_d[j], dim_val.replace("_", " ").upper(),
                  ha="center", va="center",
                  fontsize=8.5, color="white", fontweight="bold")

    ax_B.text(0.5, 0.205, "Key macro state (Q2 Dec-2022 scenario mean):",
              ha="center", va="top", fontsize=8.5, color="#5D6D7E")
    macro_lines = [
        ("ig_oas",         f"{q2_2022['ig_oas'].mean():.2f}",    "IG OAS (%)"),
        ("us_real10y",     f"{q2_2022['us_real10y'].mean():.2f}", "US Real 10Y (%)"),
        ("vix",            f"{q2_2022['vix'].mean():.1f}",        "VIX"),
        ("short_rate_US",  f"{q2_2022['short_rate_US'].mean():.2f}", "Short Rate US (%)"),
    ]
    for k, (col, val, label) in enumerate(macro_lines):
        xpos = 0.12 + (k % 2) * 0.50
        ypos = 0.135 - (k // 2) * 0.065
        ax_B.text(xpos, ypos, label, ha="left", va="top",
                  fontsize=8, color="#5D6D7E")
        ax_B.text(xpos, ypos - 0.030, val, ha="left", va="top",
                  fontsize=11, fontweight="bold", color=COLORS["anchor"])

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# FIGURE 4: House-View Gap
# ---------------------------------------------------------------------------

def build_fig4():
    """The Return Ceiling — house-view gap story."""
    q3 = results[results["question_id"] == "Q3_house_view_5pct"].copy()

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    supra_title(fig, "The Return Ceiling",
                "No locally plausible macro state closes the house-view gap")

    gs = fig.add_gridspec(1, 2, left=0.07, right=0.97,
                          top=0.85, bottom=0.10,
                          hspace=0.0, wspace=0.35)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    clean_ax(ax_A)
    clean_ax(ax_B)

    # ---- Panel A: Return ranges per anchor --------------------------------
    ax_A.set_title("A  |  Predicted Return vs House-View Target",
                   loc="left", fontsize=11, fontweight="bold",
                   color=COLORS["anchor"], pad=8)

    HV_TARGET = 5.0

    anchor_rets = [2.01, 3.30, 2.82, 2.24]
    scen_means = []
    scen_mins = []
    scen_maxs = []
    for d in ANCHOR_DATES:
        sub = q3[q3["anchor_date"] == d]["pred_return_pct"]
        scen_means.append(sub.mean())
        scen_mins.append(sub.min())
        scen_maxs.append(sub.max())

    gaps = [HV_TARGET - mx for mx in scen_maxs]
    x = np.arange(len(ANCHOR_DATES))
    bar_w = 0.28

    # House-view target line
    ax_A.axhline(HV_TARGET, color=COLORS["hv_target"], linewidth=2.0,
                 linestyle="--", zorder=5, label="House-view target (5%)")
    ax_A.text(len(ANCHOR_DATES) - 0.5, HV_TARGET + 0.1, "5% Target",
              ha="right", va="bottom", fontsize=9,
              color=COLORS["hv_target"], fontweight="bold")

    # Scenario range (shaded band per anchor)
    for i, d in enumerate(ANCHOR_DATES):
        ax_A.fill_between([x[i] - bar_w * 1.1, x[i] + bar_w * 1.1],
                          scen_mins[i], scen_maxs[i],
                          color=COLORS["scenario"], alpha=0.18, zorder=2)
        # Gap hatching
        ax_A.fill_between([x[i] - bar_w * 1.1, x[i] + bar_w * 1.1],
                          scen_maxs[i], HV_TARGET,
                          color=COLORS["hv_target"], alpha=0.06, zorder=2,
                          hatch="///", linewidth=0)

    # Anchor baseline bars
    bars_a = ax_A.bar(x - bar_w / 2, anchor_rets, bar_w,
                      color=COLORS["anchor"], alpha=0.75,
                      label="Anchor baseline", zorder=4)

    # Scenario mean bars
    bars_s = ax_A.bar(x + bar_w / 2, scen_means, bar_w,
                      color=COLORS["scenario"], alpha=0.80,
                      label="Scenario mean", zorder=4)

    # Scenario max markers
    ax_A.scatter(x + bar_w / 2, scen_maxs, s=80, color="white",
                 edgecolors=COLORS["scenario"], linewidth=2,
                 zorder=7, label="Scenario max")

    # Gap annotations
    for i in range(len(ANCHOR_DATES)):
        mid_x = x[i] + bar_w / 2
        ax_A.annotate("",
                      xy=(mid_x, HV_TARGET - 0.05),
                      xytext=(mid_x, scen_maxs[i] + 0.05),
                      arrowprops=dict(arrowstyle="<->",
                                      color=COLORS["hv_target"],
                                      lw=1.2, mutation_scale=8))
        ax_A.text(mid_x + 0.18, (HV_TARGET + scen_maxs[i]) / 2,
                  f"−{gaps[i]:.2f}pp",
                  ha="left", va="center", fontsize=8,
                  color=COLORS["hv_target"], fontweight="bold")

    ax_A.set_xticks(x)
    ax_A.set_xticklabels(ANCHOR_LABELS, fontsize=9.5)
    ax_A.set_ylabel("Predicted Return (%)", fontsize=10)
    ax_A.set_ylim(0, 6.5)
    ax_A.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f%%"))
    ax_A.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    ax_A.grid(axis="y", alpha=0.6)
    ax_A.set_axisbelow(True)

    # Value labels
    for bar in list(bars_a) + list(bars_s):
        h = bar.get_height()
        ax_A.text(bar.get_x() + bar.get_width() / 2, h + 0.07,
                  f"{h:.2f}%", ha="center", va="bottom", fontsize=7.5,
                  color="#2C3E50")

    # ---- Panel B: Why the ceiling is low -----------------------------------
    ax_B.set_title("B  |  Best-Case Scenario — Why the Ceiling Is Low",
                   loc="left", fontsize=11, fontweight="bold",
                   color=COLORS["anchor"], pad=8)
    ax_B.set_xlim(0, 1)
    ax_B.set_ylim(0, 1)
    ax_B.axis("off")

    # Best achievable result
    best_row = q3.loc[q3["pred_return"].idxmax()]

    ax_B.text(0.5, 0.94, "Best Achievable Scenario", ha="center", va="top",
              fontsize=13, fontweight="bold", color=COLORS["anchor"])

    # Regime chip
    regime_chip(ax_B, 0.5, 0.83, "", best_row["regime_label"],
                width=0.55, height=0.09, fontsize=10)

    ax_B.text(0.5, 0.77, f"Dec 2022 anchor  |  pred_return = {best_row['pred_return']*100:.2f}%",
              ha="center", va="top", fontsize=9, color="#5D6D7E", style="italic")

    # Key macro drivers table
    macro_items = [
        ("Short Rate US",   f"{best_row['short_rate_US']:.2f}%"),
        ("US Real 10Y",     f"{best_row['us_real10y']:.2f}%"),
        ("IG OAS",          f"{best_row['ig_oas']:.2f}%"),
        ("VIX",             f"{best_row['vix']:.1f}"),
        ("Inflation US",    f"{best_row['infl_US']:.2f}%"),
    ]

    row_y = [0.680, 0.600, 0.520, 0.440, 0.360]
    for (label, val), ry in zip(macro_items, row_y):
        # Row background
        bg = FancyBboxPatch(
            (0.08, ry - 0.03), 0.84, 0.060,
            boxstyle="round,pad=0.005",
            facecolor="white", edgecolor="#ECF0F1", linewidth=0.8,
            transform=ax_B.transAxes, clip_on=True,
        )
        ax_B.add_patch(bg)
        ax_B.text(0.18, ry, label, ha="left", va="center",
                  fontsize=9.5, color="#5D6D7E")
        ax_B.text(0.82, ry, val, ha="right", va="center",
                  fontsize=10, fontweight="bold", color=COLORS["anchor"])

    # Gap summary box
    gap_box = FancyBboxPatch(
        (0.06, 0.03), 0.88, 0.26,
        boxstyle="round,pad=0.01",
        facecolor="#FEF9F9", edgecolor=COLORS["hv_target"],
        linewidth=1.5, transform=ax_B.transAxes, clip_on=True,
    )
    ax_B.add_patch(gap_box)

    ax_B.text(0.5, 0.275, "Gap Summary", ha="center", va="top",
              fontsize=10, fontweight="bold", color=COLORS["hv_target"])

    gap_data = [
        ("Dec 2021", "−3.00pp"),
        ("Dec 2022", "−1.92pp"),
        ("Dec 2023", "−2.18pp"),
        ("Dec 2024", "−3.12pp"),
    ]
    for k, (date, gap) in enumerate(gap_data):
        col = k % 2
        row_k = k // 2
        xp = 0.16 + col * 0.50
        yp = 0.215 - row_k * 0.085
        ax_B.text(xp, yp, date, ha="left", va="top",
                  fontsize=8.5, color="#5D6D7E")
        ax_B.text(xp + 0.32, yp, gap, ha="right", va="top",
                  fontsize=10, fontweight="bold", color=COLORS["hv_target"])

    ax_B.text(0.5, 0.048, "Even with ideal macro state, ceiling = 3.08%",
              ha="center", va="bottom", fontsize=9,
              color=COLORS["anchor"], style="italic",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF9E6",
                        edgecolor="none"))

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    created_files = []

    print("\nBuilding Figure 1: Story Overview…")
    fig1 = build_fig1()
    p1, p2 = save_figure(fig1, "scenario_story_overview_v4")
    created_files += [p1, p2]

    print("\nBuilding Figure 2: Gold Activation…")
    fig2 = build_fig2()
    p1, p2 = save_figure(fig2, "scenario_gold_activation_v4")
    created_files += [p1, p2]

    print("\nBuilding Figure 3: Equal-Weight Deviation…")
    fig3 = build_fig3()
    p1, p2 = save_figure(fig3, "scenario_equal_weight_deviation_v4")
    created_files += [p1, p2]

    print("\nBuilding Figure 4: House-View Gap…")
    fig4 = build_fig4()
    p1, p2 = save_figure(fig4, "scenario_house_view_gap_v4")
    created_files += [p1, p2]

    # ---- Write index markdown ----------------------------------------------
    index_md = REPORTS / "scenario_plot_index_v4.md"
    with open(index_md, "w") as f:
        f.write("# Scenario Plot Index — XOPTPOE v4\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("| Figure | Title | Files |\n")
        f.write("|--------|-------|-------|\n")
        entries = [
            ("Fig 1", "Story Overview",
             "scenario_story_overview_v4.png / .pdf"),
            ("Fig 2", "Gold Activation Threshold",
             "scenario_gold_activation_v4.png / .pdf"),
            ("Fig 3", "Defensive Barbell / EW Deviation",
             "scenario_equal_weight_deviation_v4.png / .pdf"),
            ("Fig 4", "The Return Ceiling / House-View Gap",
             "scenario_house_view_gap_v4.png / .pdf"),
        ]
        for fig_id, title, files in entries:
            f.write(f"| {fig_id} | {title} | {files} |\n")
        f.write("\n## Notes\n")
        f.write("- All figures 16:9 (300 dpi PNG + vector PDF)\n")
        f.write("- Output directory: `reports/scenario_plots/`\n")
        f.write("- Data sources: `reports/scenario_anchor_truth_v4.csv`, "
                "`reports/scenario_results_v4.csv`\n")

    print(f"\n  Index written: {index_md.name}")
    print(f"\nDone. {len(created_files)} files saved to {OUTDIR}\n")


if __name__ == "__main__":
    main()
