"""
build_scenario_explainer_graphics_v4_final.py

Scenario Explainer Slidegraphics — v4 Final Package
Six main-stage + up to 3 appendix figures.
Editorial / institutional finance style. 16:9. 300 dpi.

Run from workspace_v4/:
    python scripts/scenario/build_scenario_explainer_graphics_v4_final.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUTDIR  = REPORTS / "scenario_explainer_graphics_v4_final"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor":   "#FFFFFF",
    "axes.facecolor":     "#FFFFFF",
    "axes.grid":          False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.edgecolor":     "#CCCCCC",
    "axes.linewidth":     0.6,
    "axes.titlesize":     11,
    "axes.titlepad":      6,
    "axes.labelsize":     9,
    "axes.labelpad":      4,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "legend.fontsize":    8,
    "legend.frameon":     False,
    "text.color":         "#1A1A1A",
})

C = {
    "ink":       "#1A1A1A",
    "charcoal":  "#3A3A3A",
    "mid":       "#888888",
    "light":     "#CCCCCC",
    "vlight":    "#EBEBEB",
    "bg":        "#FFFFFF",
    "blue":      "#1F6FBF",
    "blue_soft": "#A8C8E8",
    "blue_dim":  "#D0E4F5",
    "red":       "#B03A2E",
    "gold":      "#C8780A",
    "gold_soft": "#EDD59A",
    "anchor":    "#555555",
    "scenario":  "#1F6FBF",
}

FIG_W, FIG_H = 16, 9   # inches 16:9
DPI          = 150      # 150 for generation speed; 300 for final export

ANCHOR_LABELS = ["Dec 2021", "Dec 2022", "Dec 2023", "Dec 2024"]
ANCHOR_DATES  = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]

# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def _save(fig, stem: str):
    png = OUTDIR / f"{stem}.png"
    pdf = OUTDIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    fig.savefig(pdf,           bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved: {png.name}")

def _title(fig, title: str, subtitle: str = ""):
    fig.text(0.5, 0.97, title,
             ha="center", va="top",
             fontsize=17, fontweight="bold", color=C["ink"],
             transform=fig.transFigure)
    if subtitle:
        fig.text(0.5, 0.925, subtitle,
                 ha="center", va="top",
                 fontsize=9.5, color=C["mid"],
                 transform=fig.transFigure)

def _tag(ax, text, x=0.0, y=1.06, fontsize=8, color=None):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, color=color or C["mid"],
            va="bottom", ha="left")

def _hline(ax, y, color=C["light"], lw=0.7, **kw):
    ax.axhline(y, color=color, linewidth=lw, **kw)

def _spine_style(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if left:   ax.spines["left"].set_color(C["light"])
    if bottom: ax.spines["bottom"].set_color(C["light"])

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
def load_data():
    truth  = pd.read_csv(REPORTS / "scenario_anchor_truth_v4.csv")
    truth["anchor_date"] = pd.to_datetime(truth["anchor_date"])

    results = pd.read_csv(REPORTS / "scenario_reset_results_v4.csv")
    selected= pd.read_csv(REPORTS / "scenario_reset_selected_cases_v4.csv")
    manifest= pd.read_csv(REPORTS / "scenario_reset_question_manifest_v4.csv")
    shifts  = pd.read_csv(REPORTS / "scenario_reset_state_shift_summary_v4.csv")
    port    = pd.read_csv(REPORTS / "scenario_reset_portfolio_response_summary_v4.csv")
    regime  = pd.read_csv(REPORTS / "scenario_reset_regime_summary_v4.csv")

    return dict(truth=truth, results=results, selected=selected,
                manifest=manifest, shifts=shifts, port=port, regime=regime)

# ---------------------------------------------------------------------------
# FIG 01 — OVERVIEW: 3 questions, what we learn
# ---------------------------------------------------------------------------
def fig01_overview(d):
    """Clean opener. Three surviving questions + one key quantified finding each."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    _title(fig,
           "Three Questions About the Benchmark",
           "What can a macro-state scenario engine reveal about this portfolio?")

    # Layout: top rule, then 3 columns each with question + key data panel
    gs = gridspec.GridSpec(2, 3,
                           left=0.05, right=0.97,
                           top=0.86, bottom=0.08,
                           hspace=0.55, wspace=0.14)

    Q_TITLES = [
        "Q1 — Gold Activation Threshold",
        "Q3 — Return With Discipline",
        "Q4 — Return Ceiling",
    ]
    Q_SUBTITLES = [
        "What regime shift makes gold material?",
        "Which regime lifts return without excessive concentration?",
        "How far can plausible scenarios lift predicted return?",
    ]
    Q_FINDINGS = [
        "Gold triples from 8% → 22%+ when\nus_real10y crosses zero (Dec 2022).",
        "Only the Higher-for-Longer regime\napproaches 3.2% disciplined return.",
        "No scenario reaches 5%.\nCeiling is 3.3% (Dec 2022 anchor).",
    ]
    Q_TAG_COLORS = [C["gold"], C["blue"], C["charcoal"]]

    truth = d["truth"]
    manifest = d["manifest"]

    for col, (qt, qs, qf, qc) in enumerate(zip(Q_TITLES, Q_SUBTITLES, Q_FINDINGS, Q_TAG_COLORS)):
        # ---- top: question text ----
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.set_xlim(0, 1); ax_top.set_ylim(0, 1)
        ax_top.axis("off")
        # Colored rule at top
        ax_top.axhline(0.97, xmin=0.02, xmax=0.98, color=qc, linewidth=2.5)
        ax_top.text(0.5, 0.82, qt,
                    ha="center", va="top", fontsize=10.5,
                    fontweight="bold", color=C["ink"],
                    transform=ax_top.transAxes)
        ax_top.text(0.5, 0.54, qs,
                    ha="center", va="top", fontsize=8.5,
                    color=C["mid"], style="italic",
                    transform=ax_top.transAxes)
        ax_top.text(0.5, 0.20, qf,
                    ha="center", va="top", fontsize=9,
                    color=C["charcoal"],
                    transform=ax_top.transAxes,
                    multialignment="center")

        # ---- bottom: small data panel ----
        ax_bot = fig.add_subplot(gs[1, col])
        _spine_style(ax_bot)

        if col == 0:
            # Gold weight across anchors — anchor bars
            gw = [float(truth[truth["anchor_date"] == pd.Timestamp(d_)]["w_ALT_GLD"].values[0]) * 100
                  for d_ in ANCHOR_DATES]
            bars = ax_bot.bar(range(4), gw,
                              color=[C["gold"] if g > 15 else C["gold_soft"] for g in gw],
                              width=0.55, edgecolor="none")
            ax_bot.set_xticks(range(4)); ax_bot.set_xticklabels(ANCHOR_LABELS, fontsize=7.5)
            ax_bot.set_ylabel("ALT_GLD weight (%)", fontsize=8)
            ax_bot.set_ylim(0, 30)
            _hline(ax_bot, 0)
            # Label bars
            for i, v in enumerate(gw):
                ax_bot.text(i, v + 0.5, f"{v:.0f}%",
                            ha="center", va="bottom", fontsize=8, color=C["ink"])
            # Threshold annotation
            ax_bot.axhline(15, color=C["gold"], linewidth=0.8, linestyle="--", alpha=0.5)
            ax_bot.text(3.55, 15.5, "material\nthreshold", fontsize=7, color=C["gold"], va="bottom", ha="right")

        elif col == 1:
            # Mean pred return by anchor for Q3
            q3 = d["manifest"][d["manifest"]["question_id"] == "Q3_return_discipline"]
            ret_vals = []
            anchors_found = []
            for ad in ANCHOR_DATES:
                row = q3[q3["anchor_date"] == ad]
                if not row.empty:
                    ret_vals.append(float(row["mean_pred_return"].values[0]) * 100)
                    anchors_found.append(ad)
            if ret_vals:
                bars = ax_bot.bar(range(len(ret_vals)), ret_vals,
                                  color=[C["blue"] if v > 2.5 else C["blue_soft"] for v in ret_vals],
                                  width=0.55, edgecolor="none")
                ax_bot.set_xticks(range(len(ret_vals)))
                ax_bot.set_xticklabels([ANCHOR_LABELS[ANCHOR_DATES.index(a)] for a in anchors_found], fontsize=7.5)
                ax_bot.set_ylabel("Mean pred. return (%)", fontsize=8)
                ax_bot.set_ylim(0, 4.5)
                _hline(ax_bot, 0)
                for i, v in enumerate(ret_vals):
                    ax_bot.text(i, v + 0.05, f"{v:.1f}%",
                                ha="center", va="bottom", fontsize=8, color=C["ink"])
                ax_bot.axhline(4.0, color=C["blue"], linewidth=0.8, linestyle="--", alpha=0.4)
                ax_bot.text(3.55, 4.05, "4% target", fontsize=7, color=C["blue"], va="bottom", ha="right")

        else:
            # Q4 — return ceiling: max pred return per anchor
            q4 = d["results"][d["results"]["question_id"] == "Q4_return_ceiling"]
            maxrets = []
            for ad in ANCHOR_DATES:
                sub = q4[q4["anchor_date"] == ad]
                if not sub.empty:
                    maxrets.append(float(sub["pred_return"].max()) * 100)
                else:
                    maxrets.append(0.0)

            ax_bot.bar(range(4), maxrets,
                       color=C["charcoal"],
                       width=0.55, edgecolor="none", alpha=0.75)
            ax_bot.axhline(5.0, color=C["red"], linewidth=1.0, linestyle="--")
            ax_bot.text(3.55, 5.1, "5% target", fontsize=7, color=C["red"], va="bottom", ha="right")
            ax_bot.set_xticks(range(4)); ax_bot.set_xticklabels(ANCHOR_LABELS, fontsize=7.5)
            ax_bot.set_ylabel("Best achievable pred. return (%)", fontsize=8)
            ax_bot.set_ylim(0, 6)
            _hline(ax_bot, 0)
            for i, v in enumerate(maxrets):
                ax_bot.text(i, v + 0.05, f"{v:.1f}%",
                            ha="center", va="bottom", fontsize=8, color=C["ink"])

    # Footer
    fig.text(0.5, 0.02,
             "Q2 (Equal-weight departure) omitted — probe not executable with available return data.",
             ha="center", fontsize=8, color=C["mid"], style="italic")

    _save(fig, "scenario_explainer_01_overview_v4")


# ---------------------------------------------------------------------------
# FIG 02 — SEARCH METHOD (reset pipeline)
# ---------------------------------------------------------------------------
def fig02_search_method(d):
    """Explain the reset search pipeline as a clean horizontal flow."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    _title(fig,
           "Scenario Search: How We Find Plausible Macro Regimes",
           "Historical Analog Selection  ·  Latin Hypercube Sampling  ·  Bounded Gradient Refinement  ·  Diverse Selection")

    ax = fig.add_axes([0.03, 0.08, 0.94, 0.80])
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis("off")

    # --- 5 pipeline stages as clean rectangles ---
    stages = [
        ("01\nAnchor\nMacro State",
         "m₀ at anchor date\n19-dim macro vector\n(rates, inflation, spreads,\nFX, oil, VIX)",
         0.6, C["charcoal"]),
        ("02\nCandidate\nGeneration",
         "Historical analogs:\n20 matching months\n+\nLatin Hypercube:\n60 diverse states\nin plausibility box",
         2.4, C["blue"]),
        ("03\nPlausibility\nFilter",
         "VAR(1) prior:\nmₜ₊₁ = c + A·mₜ + ε\nKeep states within\n90th pct Mahalanobis\ndistance",
         4.4, C["blue"]),
        ("04\nGradient\nRefinement",
         "15-step gradient\ndescent (no noise)\nPure task objective\nBox-constrained\nto prior space",
         6.4, C["blue"]),
        ("05\nRanked\nSelection",
         "Composite score:\n50% objective\n30% plausibility\n20% diversity\nMin 2 regimes",
         8.4, C["charcoal"]),
    ]

    box_w, box_h = 1.45, 4.2
    for (top_text, body_text, cx, color) in stages:
        bx = cx - box_w / 2
        by = 0.8
        # Shadow feel — light rect behind
        rect_bg = mpatches.FancyBboxPatch(
            (bx + 0.05, by - 0.05), box_w, box_h,
            boxstyle="round,pad=0.08", linewidth=0,
            facecolor=C["vlight"], zorder=1)
        ax.add_patch(rect_bg)
        # Main rect
        rect = mpatches.FancyBboxPatch(
            (bx, by), box_w, box_h,
            boxstyle="round,pad=0.08", linewidth=1.0,
            edgecolor=color, facecolor=C["bg"], zorder=2)
        ax.add_patch(rect)
        # Stage number + title (top of box)
        lines = top_text.split("\n")
        ax.text(cx, by + box_h - 0.18, lines[0],
                ha="center", va="top",
                fontsize=8, color=color, fontweight="bold", zorder=3)
        ax.text(cx, by + box_h - 0.52, "\n".join(lines[1:]),
                ha="center", va="top",
                fontsize=10, color=C["ink"], fontweight="bold", zorder=3,
                multialignment="center")
        # Divider line
        ax.plot([bx + 0.12, bx + box_w - 0.12],
                [by + box_h - 1.08, by + box_h - 1.08],
                color=C["light"], linewidth=0.7, zorder=3)
        # Body text
        ax.text(cx, by + box_h - 1.22, body_text,
                ha="center", va="top",
                fontsize=8, color=C["charcoal"], zorder=3,
                multialignment="center", linespacing=1.45)

    # Arrows between stages
    arrow_xs = [1.35, 3.35, 5.35, 7.35]
    for ax_x in arrow_xs:
        ax.annotate("",
            xy=(ax_x + 0.28, 3.0), xytext=(ax_x, 3.0),
            arrowprops=dict(arrowstyle="-|>",
                            color=C["mid"],
                            lw=1.2,
                            mutation_scale=12),
            zorder=4)

    # --- Below pipeline: what each stage produces ---
    outputs = [
        ("m₀: anchor state", 0.6),
        ("80 diverse\ncandidates", 2.4),
        ("~50 plausible\ncandidates", 4.4),
        ("15 refined\ncandidates", 6.4),
        ("5 selected\nscenarios", 8.4),
    ]
    for text, cx in outputs:
        ax.text(cx, 0.55, text,
                ha="center", va="top", fontsize=7.5, color=C["mid"],
                multialignment="center")

    # --- Side labels for the two key properties ---
    # Plausibility maintained throughout
    ax.annotate("",
        xy=(8.85, 0.55), xytext=(1.75, 0.55),
        arrowprops=dict(arrowstyle="-",
                        color=C["blue_soft"], lw=1.5,
                        connectionstyle="arc3,rad=0.0"),
        annotation_clip=False)
    ax.text(5.3, 0.28, "Plausibility maintained throughout (VAR(1) box constraint + Mahalanobis filter)",
            ha="center", va="top", fontsize=7.5, color=C["blue"], style="italic")

    _save(fig, "scenario_explainer_02_search_method_v4")


# ---------------------------------------------------------------------------
# FIG 03 — QUESTIONS AND OBJECTIVES
# ---------------------------------------------------------------------------
def fig03_questions_objectives(d):
    """Show 3 surviving questions, their G-function, what is rewarded vs penalised."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    _title(fig,
           "What Each Question Asks — and How the Search Optimises It",
           "Three objectives, each encoded as a scalar loss G(m) over the 19-dimensional macro state")

    gs = gridspec.GridSpec(1, 3,
                           left=0.04, right=0.97,
                           top=0.84, bottom=0.06,
                           wspace=0.08)

    questions = [
        {
            "id": "Q1",
            "title": "Gold Activation\nThreshold",
            "color": C["gold"],
            "objective": "G(m)  =  −w_gold(m)  +  reg(m)",
            "maximises": "Gold portfolio weight\nw_ALT_GLD(m)",
            "penalises": "Distance from VAR(1)\none-step prediction\n(Mahalanobis)",
            "answer": "Gold maximised in:\n→ low us_real10y\n→ elevated ig_oas / vix\n→ high inflation",
            "result": "Best case: 10.4%\n(2021 anchor)\n22–24% at later anchors\n(already activated)",
        },
        {
            "id": "Q3",
            "title": "Return With\nDiscipline",
            "color": C["blue"],
            "objective": "G(m)  =  (pred_ret(m) − 4%)²  +  reg(m)",
            "maximises": "Portfolio predicted return\nclosest to 4% target",
            "penalises": "Deviation from 4% target\n(quadratic loss)\n+ plausibility penalty",
            "answer": "Best regime:\n→ Higher-for-longer\n→ High credit spreads\n→ Elevated real yields",
            "result": "Best achievable:\n~3.2% at Dec 2022\nEntropy 1.86 (diversified)\nNot excessively concentrated",
        },
        {
            "id": "Q4",
            "title": "Return Ceiling /\nHouse-View Gap",
            "color": C["charcoal"],
            "objective": "G(m)  =  (pred_ret(m) − 5%)²  +  reg(m)",
            "maximises": "Portfolio predicted return\nclosest to 5% target",
            "penalises": "Deviation from 5% target\n(quadratic loss)\n+ plausibility penalty",
            "answer": "No plausible macro\nstate reaches 5%:\n→ λ=8.0 limits concentration\n→ EN is mean-reverting",
            "result": "Ceiling: 3.32%\n(Dec 2022 anchor)\nStructural gap: −1.68pp\nThis is a design feature",
        },
    ]

    for col, q in enumerate(questions):
        ax = fig.add_subplot(gs[col])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        # Top color bar
        ax.add_patch(mpatches.Rectangle((0, 0.965), 1, 0.035,
                                         facecolor=q["color"], edgecolor="none",
                                         transform=ax.transAxes, clip_on=False))
        # Title
        ax.text(0.5, 0.93, f"  {q['id']}  ·  {q['title']}",
                ha="center", va="top", fontsize=11, fontweight="bold",
                color=C["ink"], multialignment="center")

        # Objective
        ax.text(0.5, 0.815, "Objective",
                ha="center", va="top", fontsize=8, color=C["mid"],
                fontweight="bold", transform=ax.transAxes)
        obj_box = mpatches.FancyBboxPatch((0.04, 0.70), 0.92, 0.09,
                                           boxstyle="round,pad=0.02",
                                           facecolor=C["vlight"], edgecolor="none",
                                           transform=ax.transAxes)
        ax.add_patch(obj_box)
        ax.text(0.5, 0.745, q["objective"],
                ha="center", va="center", fontsize=9,
                color=C["ink"], family="monospace", transform=ax.transAxes)

        # Maximises / Penalises
        for i, (label, text, ystart) in enumerate([
            ("Minimising G rewards:", q["maximises"], 0.62),
            ("Plausibility penalises:", q["penalises"], 0.43),
        ]):
            ax.text(0.07, ystart, label,
                    ha="left", va="top", fontsize=8, color=C["mid"],
                    fontweight="bold", transform=ax.transAxes)
            ax.text(0.07, ystart - 0.055, text,
                    ha="left", va="top", fontsize=8.5, color=C["charcoal"],
                    transform=ax.transAxes, multialignment="left")

        # Divider
        ax.axline((0.05, 0.275), (0.95, 0.275), color=C["light"], linewidth=0.8,
                  transform=ax.transAxes)

        # Result
        ax.text(0.07, 0.255, "Search finds:",
                ha="left", va="top", fontsize=8, color=C["mid"],
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.07, 0.20, q["answer"],
                ha="left", va="top", fontsize=8.5, color=C["charcoal"],
                transform=ax.transAxes, multialignment="left")

        ax.text(0.07, 0.03, q["result"],
                ha="left", va="bottom", fontsize=8.5, color=q["color"],
                fontweight="bold",
                transform=ax.transAxes, multialignment="left")

        # Outer border
        for spine_name in ["top", "right", "left", "bottom"]:
            ax.spines[spine_name].set_visible(False)
        rect_outer = mpatches.FancyBboxPatch((0.01, 0.01), 0.98, 0.985,
                                              boxstyle="round,pad=0.01",
                                              linewidth=0.8, edgecolor=C["light"],
                                              facecolor="none",
                                              transform=ax.transAxes)
        ax.add_patch(rect_outer)

    _save(fig, "scenario_explainer_03_question_and_objective_v4")


# ---------------------------------------------------------------------------
# FIG 04 — SEARCH ACHIEVEMENT: did we hit the objective?
# ---------------------------------------------------------------------------
def fig04_search_achievement(d):
    """Show what the search achieved per question: anchor level, candidate spread, selected."""
    results = d["results"]
    truth   = d["truth"]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    _title(fig,
           "Did the Search Achieve Its Objective?",
           "Anchor baseline  ·  candidate distribution  ·  selected scenarios  (predicted return %)")

    gs = gridspec.GridSpec(1, 3,
                           left=0.07, right=0.97,
                           top=0.84, bottom=0.08,
                           wspace=0.32)

    qs = [
        ("Q1_gold_threshold",   "Q1 — Gold Weight", "w_ALT_GLD",
         "Gold weight (%)", [0, 32], C["gold"]),
        ("Q3_return_discipline", "Q3 — Predicted Return (target 4%)", "pred_return",
         "Predicted return (%)", [0, 5.5], C["blue"]),
        ("Q4_return_ceiling",    "Q4 — Predicted Return (target 5%)", "pred_return",
         "Predicted return (%)", [0, 5.5], C["charcoal"]),
    ]

    for col, (qid, qname, metric, ylabel, ylim, color) in enumerate(qs):
        ax = fig.add_subplot(gs[col])
        _spine_style(ax)

        sub = results[results["question_id"] == qid].copy()

        scale = 100.0 if metric in ("pred_return", "w_ALT_GLD") else 1.0
        sub["_val"] = sub[metric] * scale

        # For each anchor: show strip of all candidates + selected mean
        selected = d["selected"][d["selected"]["question_id"] == qid].copy()
        selected["_val"] = selected[metric] * scale if metric in selected.columns else np.nan

        x_positions = np.arange(4)
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(sub))

        # Anchor truth values
        anchor_vals = []
        for ad in ANCHOR_DATES:
            trow = truth[truth["anchor_date"] == pd.Timestamp(ad)]
            if not trow.empty:
                if metric == "pred_return":
                    av = float(trow["pred_return_pct"].values[0])
                elif metric == "w_ALT_GLD":
                    av = float(trow["w_ALT_GLD"].values[0]) * 100
                else:
                    av = float(trow[metric].values[0]) * 100
            else:
                av = np.nan
            anchor_vals.append(av)

        # Draw anchor values as horizontal ticks
        for i, av in enumerate(anchor_vals):
            if not np.isnan(av):
                ax.plot([i - 0.25, i + 0.25], [av, av],
                        color=C["anchor"], linewidth=2.0, zorder=4, solid_capstyle="round")

        # Draw candidate strip for each anchor
        for i, ad in enumerate(ANCHOR_DATES):
            sub_a = sub[sub["anchor_date"] == ad]["_val"].dropna()
            if len(sub_a) == 0:
                continue
            jit_a = np.random.default_rng(i * 7 + col).uniform(-0.18, 0.18, len(sub_a))
            ax.scatter(i + jit_a, sub_a.values,
                       color=color, alpha=0.35, s=16, zorder=3, linewidths=0)

            # Selected mean
            sel_a = selected[selected["anchor_date"] == ad]["_val"].dropna()
            if len(sel_a) > 0:
                sm = sel_a.mean()
                ax.scatter([i], [sm], color=color, s=55, zorder=5,
                           edgecolors=C["bg"], linewidths=1.2)

        # Target line
        if qid == "Q3_return_discipline":
            ax.axhline(4.0, color=color, linewidth=0.9, linestyle="--", alpha=0.5)
            ax.text(3.6, 4.1, "4%\ntarget", fontsize=7, color=color, va="bottom", ha="right")
        elif qid == "Q4_return_ceiling":
            ax.axhline(5.0, color=C["red"], linewidth=0.9, linestyle="--")
            ax.text(3.6, 5.1, "5%\ntarget", fontsize=7, color=C["red"], va="bottom", ha="right")

        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(ylim)
        ax.set_xticks(range(4)); ax.set_xticklabels(ANCHOR_LABELS, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(qname, fontsize=10, fontweight="bold", color=C["ink"], pad=8)

        # Mini legend
        l1 = mlines.Line2D([], [], color=C["anchor"], linewidth=2, label="Anchor baseline")
        l2 = mlines.Line2D([], [], marker="o", color="w", markersize=5,
                           markerfacecolor=color, alpha=0.5, label="All candidates")
        l3 = mlines.Line2D([], [], marker="o", color="w", markersize=7,
                           markerfacecolor=color, markeredgecolor=C["bg"],
                           markeredgewidth=1, label="Selected mean")
        ax.legend(handles=[l1, l2, l3], fontsize=7.5, loc="upper left",
                  handlelength=1.2, handletextpad=0.5)

    _save(fig, "scenario_explainer_04_search_achievement_v4")


# ---------------------------------------------------------------------------
# FIG 05 — PRIOR vs SELECTED MACRO SHIFTS
# ---------------------------------------------------------------------------
def fig05_prior_vs_selected(d):
    """
    Show the plausibility box vs where the selected scenarios land —
    for 4 key macro variables, at the 2022 anchor (richest regime).
    """
    from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS, box_constraints, state_scales
    from xoptpoe_v4_scenario.var1_prior import VAR1Prior
    import pandas as _pd

    feature_master = _pd.read_parquet(ROOT / "data_refs" / "feature_master_monthly.parquet")
    feature_master["month_end"] = _pd.to_datetime(feature_master["month_end"])
    prior = VAR1Prior.fit_from_feature_master(
        feature_master, MACRO_STATE_COLS,
        train_end=_pd.Timestamp("2016-02-29")
    )

    anchor_date = _pd.Timestamp("2022-12-31")
    fm_row = feature_master[feature_master["month_end"] == anchor_date]
    m0 = np.array([float(fm_row.iloc[0][col]) for col in MACRO_STATE_COLS])

    mu_pred = prior.predict_next(m0)
    sigma_pred = np.sqrt(np.maximum(np.diag(prior.Q), 0.0))

    # Selected scenarios at 2022
    selected = d["selected"].copy()
    sel_2022 = selected[selected["anchor_date"] == "2022-12-31"]

    # 4 key macro variables
    KEY_VARS = ["us_real10y", "ig_oas", "vix", "infl_US"]
    KEY_LABELS = ["US Real 10Y Yield (%)", "IG Credit Spread (OAS)", "VIX", "US Inflation (%)"]
    KEY_INDICES = [MACRO_STATE_COLS.index(v) for v in KEY_VARS]

    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.12, wspace=0.38)
    _title(fig,
           "Plausibility Box vs Selected Scenario Distribution",
           "Dec 2022 anchor  ·  VAR(1) one-step prediction interval (±3σ) vs where selected scenarios land")

    for col, (var, label, idx) in enumerate(zip(KEY_VARS, KEY_LABELS, KEY_INDICES)):
        ax = axes[col]
        _spine_style(ax, left=False, bottom=True)
        ax.set_yticks([])

        m0_val    = m0[idx]
        mu_val    = mu_pred[idx]
        sig_val   = sigma_pred[idx]
        box_lo    = mu_val - 3 * sig_val
        box_hi    = mu_val + 3 * sig_val

        # Get selected values
        if var in sel_2022.columns:
            sel_vals = sel_2022[var].dropna().values
        else:
            sel_vals = np.array([])

        # Y axis range
        all_vals = np.concatenate([[m0_val, mu_val, box_lo, box_hi], sel_vals])
        pad = (box_hi - box_lo) * 0.25
        ylo = min(all_vals) - pad
        yhi = max(all_vals) + pad

        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([])

        # ±3σ plausibility band
        band = mpatches.FancyBboxPatch(
            (0.15, box_lo), 0.7, box_hi - box_lo,
            boxstyle="square,pad=0",
            facecolor=C["blue_dim"], edgecolor=C["blue_soft"],
            linewidth=0.8, alpha=0.6)
        ax.add_patch(band)
        ax.text(0.5, box_hi + pad * 0.12, "±3σ\nplausible\nspace",
                ha="center", va="bottom", fontsize=7, color=C["blue"], multialignment="center")

        # VAR(1) prediction mean
        ax.plot([0.15, 0.85], [mu_val, mu_val],
                color=C["blue"], linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(0.87, mu_val, "VAR(1)\npred.", fontsize=6.5, color=C["blue"],
                va="center", ha="left")

        # Anchor value
        ax.scatter([0.5], [m0_val], color=C["anchor"], s=80, zorder=5,
                   marker="D", edgecolors=C["bg"], linewidths=1)
        ax.text(0.5, m0_val - pad * 0.18, "anchor\nm₀",
                ha="center", va="top", fontsize=7, color=C["anchor"],
                multialignment="center")

        # Selected scenarios
        if len(sel_vals) > 0:
            jitter = np.random.default_rng(col + 42).uniform(0.35, 0.65, len(sel_vals))
            ax.scatter(jitter, sel_vals, color=C["blue"], s=45, zorder=6,
                       edgecolors=C["bg"], linewidths=0.8, alpha=0.85)
            sel_mean = sel_vals.mean()
            ax.plot([0.25, 0.75], [sel_mean, sel_mean],
                    color=C["blue"], linewidth=2.0, zorder=7, solid_capstyle="round")
            ax.text(0.5, sel_mean + pad * 0.1, f"{sel_mean:.2f}",
                    ha="center", va="bottom", fontsize=8, color=C["blue"], fontweight="bold")

        ax.set_title(label, fontsize=9.5, fontweight="bold", color=C["ink"], pad=8)
        ax.spines["bottom"].set_color(C["light"])
        ax.set_xlabel(var, fontsize=7.5, color=C["mid"])

        # Anchor value annotation
        ax.text(-0.35, m0_val, f"{m0_val:.2f}", ha="center", va="center",
                fontsize=7.5, color=C["anchor"])

    # Legend
    l1 = mlines.Line2D([], [], marker="D", color="w", markersize=6,
                       markerfacecolor=C["anchor"], label="Anchor m₀")
    l2 = mlines.Line2D([], [], marker="o", color="w", markersize=6,
                       markerfacecolor=C["blue"], label="Selected scenarios")
    l3 = mpatches.Patch(facecolor=C["blue_dim"], edgecolor=C["blue_soft"],
                         label="VAR(1) ±3σ box")
    fig.legend(handles=[l1, l2, l3], loc="lower center",
               ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.02))

    _save(fig, "scenario_explainer_05_prior_vs_selected_macros_v4")


# ---------------------------------------------------------------------------
# FIG 05 (fallback — no pipeline import)
# ---------------------------------------------------------------------------
def fig05_prior_vs_selected_simple(d):
    """
    Fallback version using pre-computed shift data instead of pipeline import.
    Shows anchor value, scenario distribution, and relative shift magnitude.
    """
    shifts = d["shifts"].copy()
    results = d["results"].copy()

    # Use Q1 at 2022 for the richest regime story
    ANCHOR = "2022-12-31"
    KEY_VARS  = ["us_real10y", "ig_oas", "vix", "infl_US"]
    KEY_LABELS = ["US Real 10Y\nYield (%)", "IG Credit\nSpread (OAS)", "VIX", "US Inflation (%)"]
    KEY_ANCHOR = {"us_real10y": 1.76, "ig_oas": 1.38, "vix": 21.67, "infl_US": 7.12}
    # Historical ±3σ boxes (approximate from feature_master statistics)
    KEY_BOXES  = {
        "us_real10y": (-2.0, 4.5),
        "ig_oas":     (0.5,  3.2),
        "vix":        (8.0,  48.0),
        "infl_US":    (-0.5, 11.0),
    }

    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14, wspace=0.42)
    _title(fig,
           "Plausibility Space vs Selected Scenario Values",
           "Dec 2022 anchor  ·  Historical range  ·  Where Q1 scenarios land  ·  Key macro drivers")

    sub = results[(results["question_id"] == "Q1_gold_threshold") &
                  (results["anchor_date"] == ANCHOR)].copy()

    for col, (var, label) in enumerate(zip(KEY_VARS, KEY_LABELS)):
        ax = axes[col]
        _spine_style(ax, left=False, bottom=False)
        ax.set_xticks([]); ax.set_yticks([])

        anch_val = KEY_ANCHOR[var]
        box_lo, box_hi = KEY_BOXES[var]
        pad = (box_hi - box_lo) * 0.15
        ylo = box_lo - pad
        yhi = box_hi + pad * 1.5

        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0, 1)

        # Historical range band
        band = mpatches.Rectangle((0.2, box_lo), 0.6, box_hi - box_lo,
                                   facecolor=C["blue_dim"], edgecolor=C["blue_soft"],
                                   linewidth=0.7, alpha=0.55)
        ax.add_patch(band)
        ax.text(0.5, box_hi + pad * 0.3, "Historical\nrange",
                ha="center", va="bottom", fontsize=7, color=C["blue"],
                multialignment="center")

        # Anchor value — diamond
        ax.scatter([0.5], [anch_val], color=C["anchor"], s=90,
                   marker="D", zorder=5, edgecolors=C["bg"], linewidths=1.2)
        ax.text(0.5, anch_val, f"  {anch_val:.2f}",
                ha="left", va="center", fontsize=8, color=C["anchor"])

        # Scenario distribution
        if var in sub.columns:
            vals = sub[var].dropna().values
            if len(vals) > 0:
                jitter = np.random.default_rng(col * 13).uniform(0.3, 0.7, len(vals))
                ax.scatter(jitter, vals, color=C["gold"] if var == "ig_oas" else C["blue"],
                           s=30, zorder=6, alpha=0.7, edgecolors=C["bg"], linewidths=0.5)
                smean = vals.mean()
                ax.plot([0.22, 0.78], [smean, smean],
                        color=C["gold"] if var == "ig_oas" else C["blue"],
                        linewidth=2.0, zorder=7)
                ax.text(0.5, smean + pad * 0.2,
                        f"Scen. mean\n{smean:.2f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color=C["gold"] if var == "ig_oas" else C["blue"],
                        fontweight="bold", multialignment="center")

        ax.set_title(label, fontsize=9.5, fontweight="bold",
                     color=C["ink"], pad=8, multialignment="center")

        # Horizontal axis line
        ax.axhline(ylo + pad * 0.3, color=C["light"], linewidth=0.6)

    # Legend
    l1 = mlines.Line2D([], [], marker="D", color="w", markersize=6,
                       markerfacecolor=C["anchor"], label="Anchor value")
    l2 = mlines.Line2D([], [], marker="o", color="w", markersize=6,
                       markerfacecolor=C["blue"], label="Selected scenarios (Q1, Dec 2022)")
    l3 = mpatches.Patch(facecolor=C["blue_dim"], edgecolor=C["blue_soft"],
                         label="Historical range")
    fig.legend(handles=[l1, l2, l3], loc="lower center",
               ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.01))

    _save(fig, "scenario_explainer_05_prior_vs_selected_macros_v4")


# ---------------------------------------------------------------------------
# FIG 06 — REGIME TO PORTFOLIO (Gold activation hero)
# ---------------------------------------------------------------------------
def fig06_regime_to_portfolio(d):
    """
    The result hero figure.
    Left: gold weight across anchors (anchor baseline)
    Centre: 3 macro drivers that caused the transition
    Right: portfolio composition at 2021 vs 2022 (selected scenario vs anchor)
    """
    truth   = d["truth"]
    results = d["results"]
    selected= d["selected"]

    SLEEVES = ["EQ_US","EQ_EZ","EQ_JP","EQ_CN","EQ_EM",
               "FI_UST","FI_EU_GOVT",
               "CR_US_IG","CR_EU_IG","CR_US_HY",
               "RE_US","LISTED_RE","LISTED_INFRA","ALT_GLD"]
    SLEEVE_GROUPS = {
        "EQ_US":      ("Equity",   "#4A7A4A"),
        "EQ_EZ":      ("Equity",   "#4A7A4A"),
        "EQ_JP":      ("Equity",   "#4A7A4A"),
        "EQ_CN":      ("Equity",   "#4A7A4A"),
        "EQ_EM":      ("Equity",   "#4A7A4A"),
        "FI_UST":     ("Treasuries","#1F6FBF"),
        "FI_EU_GOVT": ("Treasuries","#1F6FBF"),
        "CR_US_IG":   ("Credit",   "#6B84A3"),
        "CR_EU_IG":   ("Credit",   "#6B84A3"),
        "CR_US_HY":   ("Credit",   "#8BA0C0"),
        "RE_US":      ("Real Assets","#A08050"),
        "LISTED_RE":  ("Real Assets","#A08050"),
        "LISTED_INFRA":("Real Assets","#A08050"),
        "ALT_GLD":    ("Gold",     C["gold"]),
    }

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    _title(fig,
           "Gold Activation — From Regime Shift to Portfolio Consequence",
           "The 2021→2022 macro transition triggered a structural rebalancing: gold tripled, equities halved")

    gs = gridspec.GridSpec(1, 3,
                           left=0.05, right=0.97,
                           top=0.84, bottom=0.08,
                           wspace=0.28)

    # ---- PANEL 1: Gold weight timeline ----
    ax1 = fig.add_subplot(gs[0])
    _spine_style(ax1)

    gw_anch = [float(truth[truth["anchor_date"] == pd.Timestamp(ad)]["w_ALT_GLD"].values[0]) * 100
               for ad in ANCHOR_DATES]

    colors_bar = [C["gold_soft"] if g < 15 else C["gold"] for g in gw_anch]
    bars = ax1.bar(range(4), gw_anch, color=colors_bar, width=0.55, edgecolor="none")
    for i, v in enumerate(gw_anch):
        ax1.text(i, v + 0.5, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=C["gold"] if v >= 15 else C["ink"])

    # Regime labels below bars
    regime_labels = ["Reflation /\nRisk-On", "Higher\nFor Longer", "Higher\nFor Longer", "Mixed\nMid-Cycle"]
    for i, rl in enumerate(regime_labels):
        ax1.text(i, -2.8, rl, ha="center", va="top", fontsize=7.5,
                 color=C["mid"], multialignment="center")

    # Arrow annotation from 2021 to 2022
    ax1.annotate("",
        xy=(1, gw_anch[1] + 0.5), xytext=(0, gw_anch[0] + 0.5),
        arrowprops=dict(arrowstyle="-|>", color=C["gold"],
                        lw=1.5, mutation_scale=14,
                        connectionstyle="arc3,rad=-0.35"),
        zorder=5)
    ax1.text(0.5, (gw_anch[0] + gw_anch[1]) / 2 + 5,
             f"+{gw_anch[1]-gw_anch[0]:.0f}pp",
             ha="center", va="center", fontsize=10,
             color=C["gold"], fontweight="bold")

    ax1.axhline(15, color=C["gold"], linewidth=0.7, linestyle="--", alpha=0.5)
    ax1.text(-0.45, 15.3, "material\n(15%)", fontsize=7, color=C["gold"], va="bottom")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(ANCHOR_LABELS, fontsize=8.5)
    ax1.set_ylabel("Gold allocation (%)", fontsize=9)
    ax1.set_ylim(-5, 34)
    ax1.set_xlim(-0.6, 3.6)
    ax1.set_title("Gold Weight Across Anchors", fontsize=10, fontweight="bold",
                  color=C["ink"], pad=8)
    _hline(ax1, 0, color=C["light"])

    # ---- PANEL 2: 3 macro drivers ----
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_title("Macro Drivers of the Transition", fontsize=10, fontweight="bold",
                  color=C["ink"], pad=8)

    drivers = [
        ("US Real 10Y Yield", "us_real10y", "-1.04%", "+1.76%", "+2.80pp",
         "Negative→positive real yields\nfundamentally reprice gold", C["blue"]),
        ("Short Rate US", "short_rate_US", "0.05%", "4.15%", "+4.10pp",
         "ZIRP ended — cash now competes\nbut gold held due to inflation", C["charcoal"]),
        ("IG Credit Spread", "ig_oas", "0.98", "1.38", "+0.40",
         "Credit stress confirms\ndefensive regime shift", C["mid"]),
    ]

    y_starts = [0.82, 0.52, 0.22]
    for (dname, dvar, v21, v22, shift, desc, dcolor), y0 in zip(drivers, y_starts):
        # Driver name
        ax2.text(0.04, y0, dname, ha="left", va="top", fontsize=9,
                 fontweight="bold", color=C["ink"])
        # Values
        ax2.text(0.04, y0 - 0.065, f"Dec 2021: {v21}",
                 ha="left", va="top", fontsize=8.5, color=C["anchor"])
        ax2.text(0.04, y0 - 0.115, f"Dec 2022: {v22}",
                 ha="left", va="top", fontsize=8.5, color=dcolor)
        # Shift badge
        badge = mpatches.FancyBboxPatch((0.68, y0 - 0.105), 0.30, 0.075,
                                         boxstyle="round,pad=0.01",
                                         facecolor=C["vlight"],
                                         edgecolor=dcolor, linewidth=0.8)
        ax2.add_patch(badge)
        ax2.text(0.83, y0 - 0.065, shift,
                 ha="center", va="center", fontsize=9,
                 fontweight="bold", color=dcolor)
        # Description
        ax2.text(0.04, y0 - 0.175, desc,
                 ha="left", va="top", fontsize=7.5, color=C["mid"],
                 multialignment="left")
        # Divider
        if y0 > 0.3:
            ax2.axline((0.02, y0 - 0.24), (0.98, y0 - 0.24),
                       color=C["light"], linewidth=0.6,
                       transform=ax2.transAxes)

    # ---- PANEL 3: Portfolio composition 2021 vs 2022 ----
    ax3 = fig.add_subplot(gs[2])
    _spine_style(ax3, left=False, bottom=True)
    ax3.set_title("Portfolio Composition", fontsize=10, fontweight="bold",
                  color=C["ink"], pad=8)

    w21_row = truth[truth["anchor_date"] == pd.Timestamp("2021-12-31")].iloc[0]
    w22_row = truth[truth["anchor_date"] == pd.Timestamp("2022-12-31")].iloc[0]

    w21 = {s: float(w21_row.get(f"w_{s}", 0)) * 100 for s in SLEEVES}
    w22 = {s: float(w22_row.get(f"w_{s}", 0)) * 100 for s in SLEEVES}

    # Group into 5 categories
    groups = [
        ("Equity",      [s for s in SLEEVES if s.startswith("EQ")],     "#4A7A4A"),
        ("Treasuries",  ["FI_UST", "FI_EU_GOVT"],                       C["blue"]),
        ("Credit",      ["CR_US_IG", "CR_EU_IG", "CR_US_HY"],           "#6B84A3"),
        ("Real Assets", ["RE_US", "LISTED_RE", "LISTED_INFRA"],         "#A08050"),
        ("Gold",        ["ALT_GLD"],                                     C["gold"]),
    ]

    g21 = {g: sum(w21.get(s, 0) for s in slvs) for g, slvs, _ in groups}
    g22 = {g: sum(w22.get(s, 0) for s in slvs) for g, slvs, _ in groups}

    group_names = [g for g, _, _ in groups]
    colors_g    = [c for _, _, c in groups]
    vals21 = [g21[g] for g in group_names]
    vals22 = [g22[g] for g in group_names]

    y = np.arange(len(group_names))
    bar_h = 0.32

    for i, (gname, v21, v22, gc) in enumerate(zip(group_names, vals21, vals22, colors_g)):
        ax3.barh(i + bar_h / 2 + 0.04, v21, bar_h,
                 color=gc, alpha=0.4, edgecolor="none")
        ax3.barh(i - bar_h / 2 - 0.04, v22, bar_h,
                 color=gc, alpha=0.9, edgecolor="none")
        # Labels
        if v21 >= 1.0:
            ax3.text(v21 + 0.4, i + bar_h / 2 + 0.04,
                     f"{v21:.0f}%", va="center", fontsize=8, color=gc, alpha=0.6)
        if v22 >= 1.0:
            ax3.text(v22 + 0.4, i - bar_h / 2 - 0.04,
                     f"{v22:.0f}%", va="center", fontsize=8.5,
                     color=gc, fontweight="bold")

    ax3.set_yticks(y)
    ax3.set_yticklabels(group_names, fontsize=9)
    ax3.set_xlabel("Portfolio weight (%)", fontsize=8.5)
    ax3.set_xlim(0, 50)
    ax3.set_ylim(-0.6, len(group_names) - 0.4)
    _spine_style(ax3, left=False, bottom=True)
    ax3.yaxis.set_visible(True)
    ax3.spines["left"].set_visible(False)

    # Legend
    p1 = mpatches.Patch(facecolor=C["mid"], alpha=0.4, label="Dec 2021")
    p2 = mpatches.Patch(facecolor=C["mid"], alpha=0.9, label="Dec 2022")
    ax3.legend(handles=[p1, p2], fontsize=8, loc="lower right")

    _save(fig, "scenario_explainer_06_regime_to_portfolio_v4")


# ---------------------------------------------------------------------------
# APPENDIX FIG 07 — Real world vs model world (cross-anchor Gold)
# ---------------------------------------------------------------------------
def fig07_real_vs_model(d):
    """Appendix: four anchor macro snapshots side by side — shows what actually changed."""
    truth   = d["truth"]

    MACRO_ROWS = [
        ("US Real 10Y Yield (%)", "us_real10y_approx",
         [-1.04, 1.76, 2.12, 1.89],   C["blue"]),
        ("Short Rate US (%)",     "short_rate_US_approx",
         [0.05, 4.15, 5.27, 4.42],    C["charcoal"]),
        ("IG OAS",                "ig_oas_approx",
         [0.98, 1.38, 1.04, 0.82],    C["mid"]),
        ("US Inflation (%)",      "infl_US_approx",
         [6.90, 7.12, 3.13, 2.72],    C["red"]),
    ]
    GW = [8.1, 22.3, 22.4, 23.2]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    _title(fig,
           "What Actually Changed Between Anchor Dates",
           "Four macro snapshots  ·  The 2021→2022 shift is visible in every dimension  ·  Appendix")

    gs = gridspec.GridSpec(2, 1,
                           left=0.08, right=0.95,
                           top=0.84, bottom=0.08,
                           hspace=0.50)

    # Top: macro heat strip
    ax_heat = fig.add_subplot(gs[0])
    ax_heat.set_xlim(-0.5, 3.5)
    ax_heat.set_ylim(0, len(MACRO_ROWS) + 0.5)
    ax_heat.axis("off")

    for row_i, (label, _, vals, color) in enumerate(reversed(MACRO_ROWS)):
        y_pos = row_i + 0.5
        vmin, vmax = min(vals), max(vals)
        rng = max(vmax - vmin, 0.01)
        for col_i, (v, al) in enumerate(zip(vals, ANCHOR_LABELS)):
            norm_v = (v - vmin) / rng
            alpha_v = 0.2 + 0.7 * norm_v
            rect = mpatches.Rectangle((col_i - 0.4, y_pos - 0.35), 0.8, 0.7,
                                       facecolor=color, alpha=alpha_v,
                                       edgecolor="none")
            ax_heat.add_patch(rect)
            ax_heat.text(col_i, y_pos, f"{v:.2f}",
                         ha="center", va="center", fontsize=9,
                         color="#FFFFFF" if alpha_v > 0.55 else C["charcoal"],
                         fontweight="bold" if col_i == 1 else "normal")
        ax_heat.text(-0.52, y_pos, label,
                     ha="right", va="center", fontsize=8.5, color=C["charcoal"])

    for col_i, al in enumerate(ANCHOR_LABELS):
        ax_heat.text(col_i, len(MACRO_ROWS) + 0.2, al,
                     ha="center", va="bottom", fontsize=8.5,
                     color=C["ink"], fontweight="bold")

    ax_heat.set_title("Macro Conditions at Each Anchor Date",
                      fontsize=10, fontweight="bold", color=C["ink"], pad=6)

    # Bottom: gold weight response
    ax_gw = fig.add_subplot(gs[1])
    _spine_style(ax_gw)
    bar_colors = [C["gold_soft"] if g < 15 else C["gold"] for g in GW]
    ax_gw.bar(range(4), GW, color=bar_colors, width=0.55, edgecolor="none")
    ax_gw.axhline(15, color=C["gold"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax_gw.set_xticks(range(4)); ax_gw.set_xticklabels(ANCHOR_LABELS, fontsize=9)
    ax_gw.set_ylabel("Gold weight (%)", fontsize=9)
    ax_gw.set_ylim(0, 30)
    for i, v in enumerate(GW):
        ax_gw.text(i, v + 0.4, f"{v:.0f}%", ha="center", va="bottom",
                   fontsize=9.5, fontweight="bold",
                   color=C["gold"] if v >= 15 else C["ink"])
    ax_gw.set_title("Benchmark Gold Allocation Response",
                    fontsize=10, fontweight="bold", color=C["ink"], pad=6)

    fig.text(0.5, 0.02,
             "Appendix — Real-world macro context for anchor dates. "
             "The 2022 column shows simultaneous increases in real yields, short rates, and inflation — "
             "triggering the gold activation.",
             ha="center", fontsize=7.5, color=C["mid"], style="italic")

    _save(fig, "scenario_explainer_07_real_world_vs_model_world_v4")


# ---------------------------------------------------------------------------
# APPENDIX FIG 08 — Allocation shift detail
# ---------------------------------------------------------------------------
def fig08_allocation_shift_detail(d):
    """Appendix: detailed sleeve-level portfolio shift 2021→2022 and return ceiling."""
    truth = d["truth"]

    SLEEVES   = ["EQ_US","EQ_EZ","EQ_JP","EQ_CN","EQ_EM",
                 "FI_UST","FI_EU_GOVT","CR_US_IG","CR_EU_IG","CR_US_HY",
                 "RE_US","LISTED_RE","LISTED_INFRA","ALT_GLD"]
    SLEEVE_LABELS = ["EQ US","EQ EZ","EQ JP","EQ CN","EQ EM",
                     "UST","EU Govt","US IG","EU IG","US HY",
                     "RE","L.RE","Infra","Gold"]
    SLEEVE_COLORS = [
        "#4A7A4A","#4A7A4A","#4A7A4A","#4A7A4A","#4A7A4A",
        C["blue"], C["blue"],
        "#6B84A3","#6B84A3","#8BA0C0",
        "#A08050","#A08050","#A08050",
        C["gold"],
    ]

    w21_row = truth[truth["anchor_date"] == pd.Timestamp("2021-12-31")].iloc[0]
    w22_row = truth[truth["anchor_date"] == pd.Timestamp("2022-12-31")].iloc[0]
    w23_row = truth[truth["anchor_date"] == pd.Timestamp("2023-12-31")].iloc[0]
    w24_row = truth[truth["anchor_date"] == pd.Timestamp("2024-12-31")].iloc[0]

    def get_w(row):
        return np.array([float(row.get(f"w_{s}", 0)) * 100 for s in SLEEVES])

    w21 = get_w(w21_row); w22 = get_w(w22_row)
    w23 = get_w(w23_row); w24 = get_w(w24_row)
    delta = w22 - w21

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.08, wspace=0.38)
    _title(fig,
           "Portfolio Shift Detail — Dec 2021 → Dec 2022",
           "Appendix  ·  Sleeve-level weight changes  ·  Return ceiling across anchors")

    # Panel 1: sleeve-level change
    y = np.arange(len(SLEEVES))
    bar_colors_delta = [C["gold"] if d_ > 5 else (C["blue"] if d_ > 1 else
                        (C["red"] if d_ < -2 else C["mid"])) for d_ in delta]
    ax1.barh(y, delta, color=bar_colors_delta, edgecolor="none", height=0.55)
    ax1.axvline(0, color=C["ink"], linewidth=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(SLEEVE_LABELS, fontsize=8.5)
    ax1.set_xlabel("Weight change (pp)", fontsize=9)
    ax1.set_title("Weight Change 2021→2022 (pp)", fontsize=10, fontweight="bold",
                  color=C["ink"], pad=8)
    _spine_style(ax1, left=False, bottom=True)
    ax1.spines["left"].set_visible(False)
    ax1.axvline(-5, color=C["light"], linewidth=0.5, linestyle=":")
    ax1.axvline(5, color=C["light"], linewidth=0.5, linestyle=":")

    # Value labels on significant bars
    for i, v in enumerate(delta):
        if abs(v) >= 1.5:
            ha = "left" if v > 0 else "right"
            ax1.text(v + (0.3 if v > 0 else -0.3), i, f"{v:+.1f}pp",
                     va="center", ha=ha, fontsize=7.5, color=bar_colors_delta[i])

    # Panel 2: return ceiling scatter
    results_q4 = d["results"][d["results"]["question_id"] == "Q4_return_ceiling"].copy()
    _spine_style(ax2)
    ax2.set_title("Return Ceiling — All Q4 Scenarios", fontsize=10, fontweight="bold",
                  color=C["ink"], pad=8)

    for i, (ad, al) in enumerate(zip(ANCHOR_DATES, ANCHOR_LABELS)):
        sub = results_q4[results_q4["anchor_date"] == ad]["pred_return"].dropna() * 100
        if len(sub) == 0: continue
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(sub))
        ax2.scatter(i + jit, sub.values, color=C["charcoal"], s=20,
                    alpha=0.5, edgecolors="none")
        # Truth pred return
        trow = truth[truth["anchor_date"] == pd.Timestamp(ad)]
        if not trow.empty:
            tv = float(trow["pred_return_pct"].values[0])
            ax2.plot([i - 0.25, i + 0.25], [tv, tv],
                     color=C["anchor"], linewidth=2.2, solid_capstyle="round")
        # Max
        mx = sub.max()
        ax2.scatter([i], [mx], color=C["blue"], s=55, zorder=5,
                    edgecolors=C["bg"], linewidths=1.2)
        ax2.text(i, mx + 0.06, f"{mx:.1f}%",
                 ha="center", va="bottom", fontsize=8, color=C["blue"], fontweight="bold")

    ax2.axhline(5.0, color=C["red"], linewidth=0.9, linestyle="--")
    ax2.text(3.5, 5.08, "5% target", fontsize=7.5, color=C["red"],
             va="bottom", ha="right")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(ANCHOR_LABELS, fontsize=8.5)
    ax2.set_ylabel("Predicted return (%)", fontsize=9)
    ax2.set_ylim(0, 5.8)

    l1 = mlines.Line2D([], [], color=C["anchor"], lw=2, label="Anchor baseline")
    l2 = mlines.Line2D([], [], marker="o", color="w", markersize=6,
                       markerfacecolor=C["charcoal"], alpha=0.5, label="All Q4 candidates")
    l3 = mlines.Line2D([], [], marker="o", color="w", markersize=7,
                       markerfacecolor=C["blue"], markeredgecolor=C["bg"],
                       markeredgewidth=1, label="Best achievable")
    ax2.legend(handles=[l1, l2, l3], fontsize=8, loc="upper right")

    fig.text(0.5, 0.02,
             "Appendix — Sleeve-level detail for the 2021→2022 rebalancing and Q4 return ceiling evidence.",
             ha="center", fontsize=7.5, color=C["mid"], style="italic")

    _save(fig, "scenario_explainer_08_allocation_shift_detail_v4")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    d = load_data()

    print("Building figures...")

    print("\n[1/8] Overview (3 questions)")
    fig01_overview(d)

    print("[2/8] Search method (reset pipeline)")
    fig02_search_method(d)

    print("[3/8] Questions and objectives")
    fig03_questions_objectives(d)

    print("[4/8] Search achievement")
    fig04_search_achievement(d)

    print("[5/8] Prior vs selected macros")
    try:
        fig05_prior_vs_selected(d)
    except Exception as e:
        print(f"  Pipeline import failed ({e}), using fallback version")
        fig05_prior_vs_selected_simple(d)

    print("[6/8] Regime to portfolio (Gold hero)")
    fig06_regime_to_portfolio(d)

    print("[7/8] Appendix — Real world vs model world")
    fig07_real_vs_model(d)

    print("[8/8] Appendix — Allocation shift detail")
    fig08_allocation_shift_detail(d)

    print(f"\nAll figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
