#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data_refs"
OUTDIR = REPORTS / "scenario_flagship_graphics_v4"
OUTDIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = REPORTS / "scenario_flagship_graphics_index_v4.md"

for p in [str(ROOT / "src"), str(ROOT.parent / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS
from xoptpoe_v4_scenario_reset.regime_v2 import build_regime_thresholds, classify_regime_v2, score_dimensions


FIGSIZE = (14.6, 8.2)
C = {
    "bg": "#ffffff",
    "ink": "#16181b",
    "charcoal": "#3c434a",
    "mid": "#7c8791",
    "light": "#d7dde2",
    "grid": "#eceff2",
    "blue": "#1f69b3",
    "blue_soft": "#dbe9f5",
    "blue_faint": "#edf4fa",
    "red": "#ad4637",
    "red_soft": "#ecd6d1",
    "gold": "#bc7f19",
    "gold_soft": "#efe0bd",
    "neutral_fill": "#f5f6f7",
}

QUESTION_META = {
    "Q1_gold_threshold": {
        "short": "Q1",
        "title": "Gold activation threshold",
        "formula": r"$G(m) = -\,w_{\mathrm{GLD}}(m)\ +\ \Pi(m)$",
        "reward": "higher gold weight",
        "penalty": "fragile or implausible macro state",
        "takeaway": "Gold matters only when the state moves toward stress\nand positive real yields.",
    },
    "Q3_return_discipline": {
        "short": "Q3",
        "title": "Return with discipline",
        "formula": r"$G(m) = (r(m)-4\%)^2\ +\ \Pi(m)$",
        "reward": "return near 4% with diversification",
        "penalty": "excessive concentration",
        "takeaway": "Best disciplined-return states cluster in the 2022\nhigher-for-longer regime.",
    },
    "Q4_return_ceiling": {
        "short": "Q4",
        "title": "Return ceiling",
        "formula": r"$G(m) = (r(m)-5\%)^2\ +\ \Pi(m)$",
        "reward": "higher return under constraints",
        "penalty": "implausible macro mix",
        "takeaway": "The benchmark cannot plausibly reach 5%; the ceiling stays\nnear 3.3%.",
    },
}

REGIME_COLORS = {
    "reflation_risk_on": C["blue"],
    "higher_for_longer": C["charcoal"],
    "risk_off_stress": C["red"],
    "high_stress_defensive": C["red"],
    "mixed_mid_cycle": C["mid"],
    "soft_landing": "#2d8568",
    "disinflationary_slowdown": "#5c7f96",
}

DIM_LABELS = {
    "dim_growth": "Growth",
    "dim_inflation": "Inflation",
    "dim_policy": "Policy",
    "dim_stress": "Stress",
    "dim_fin_cond": "Fin. cond.",
}

SLEEVE_LABELS = {
    "w_EQ_US": "US Eq",
    "w_EQ_EZ": "Euro Eq",
    "w_EQ_JP": "Japan Eq",
    "w_EQ_CN": "China Eq",
    "w_EQ_EM": "EM Eq",
    "w_FI_UST": "UST",
    "w_FI_EU_GOVT": "Euro Gov",
    "w_CR_US_IG": "US IG",
    "w_CR_EU_IG": "Euro IG",
    "w_CR_US_HY": "US HY",
    "w_RE_US": "US RE",
    "w_LISTED_RE": "Listed RE",
    "w_LISTED_INFRA": "Infra",
    "w_ALT_GLD": "Gold",
}

SLEEVE_COLORS = {
    "w_EQ_US": "#315b85",
    "w_EQ_EZ": "#577ca0",
    "w_EQ_JP": "#7f9fbe",
    "w_EQ_CN": "#3f6f97",
    "w_EQ_EM": "#a8bfd5",
    "w_FI_UST": "#6a9a83",
    "w_FI_EU_GOVT": "#88b49d",
    "w_CR_US_IG": "#4d8a84",
    "w_CR_EU_IG": "#72a8a1",
    "w_CR_US_HY": "#2f6f6e",
    "w_RE_US": "#9c7757",
    "w_LISTED_RE": "#b89167",
    "w_LISTED_INFRA": "#cfb07d",
    "w_ALT_GLD": C["gold"],
}

PLOT_META = [
    (
        "scenario_flagship_01_search_funnel_v4",
        "The reset search narrows a broad candidate set to a small selected scenario set",
        "The method is selective on purpose: many plausible candidates enter, only a few regime-distinct states survive.",
    ),
    (
        "scenario_flagship_02_question_to_objective_v4",
        "The surviving questions define what the search rewards and what it refuses to force",
        "Only three questions survive the main-stage package, and each has a clear optimization target and a clear limit.",
    ),
    (
        "scenario_flagship_03_plausibility_vs_macro_shift_v4",
        "Selected macro shifts are large enough to matter but still sit inside a plausible historical state space",
        "The search does not invent impossible states; it moves toward macro zones that have real historical analogs.",
    ),
    (
        "scenario_flagship_04_regime_transition_v4",
        "The strongest scenario story is a transition from a reflationary anchor to a more defensive high-stress regime",
        "The model changes portfolios because the macro regime changes, not because the optimizer is unstable.",
    ),
    (
        "scenario_flagship_05_portfolio_consequence_v4",
        "The strongest regime shift produces a clear portfolio consequence: more gold and duration, less equity concentration",
        "The scenario logic matters because it maps into a recognizably different strategic allocation.",
    ),
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": C["bg"],
            "axes.facecolor": C["bg"],
            "savefig.facecolor": C["bg"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.color": C["ink"],
            "axes.edgecolor": C["light"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": C["grid"],
            "grid.linewidth": 0.8,
            "savefig.dpi": 240,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTDIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_data() -> dict[str, pd.DataFrame]:
    feature_master = pd.read_parquet(DATA / "feature_master_monthly.parquet")
    feature_master["month_end"] = pd.to_datetime(feature_master["month_end"])
    feature_unique = feature_master.drop_duplicates("month_end").sort_values("month_end").reset_index(drop=True)

    return {
        "feature_master": feature_unique,
        "truth": pd.read_csv(REPORTS / "scenario_anchor_truth_v4.csv"),
        "results": pd.read_csv(REPORTS / "scenario_reset_results_v4.csv"),
        "selected": pd.read_csv(REPORTS / "scenario_reset_selected_cases_v4.csv"),
        "manifest": pd.read_csv(REPORTS / "scenario_reset_question_manifest_v4.csv"),
        "shift": pd.read_csv(REPORTS / "scenario_reset_state_shift_summary_v4.csv"),
        "port": pd.read_csv(REPORTS / "scenario_reset_portfolio_response_summary_v4.csv"),
        "regime": pd.read_csv(REPORTS / "scenario_reset_regime_summary_v4.csv"),
        "transition": pd.read_csv(REPORTS / "scenario_reset_regime_transition_summary_v4.csv"),
    }


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.045, 0.952, title, fontsize=22, fontweight="bold", color=C["ink"], ha="left", va="top")
    fig.text(0.045, 0.905, subtitle, fontsize=11.2, color=C["mid"], ha="left", va="top")


def draw_stage(ax, x0: float, x1: float, h0: float, h1: float, color: str, label: str, big: str, detail: str) -> None:
    yc = 0.52
    poly = Polygon(
        [(x0, yc - h0 / 2), (x0, yc + h0 / 2), (x1, yc + h1 / 2), (x1, yc - h1 / 2)],
        closed=True,
        facecolor=color,
        edgecolor="none",
        alpha=1.0,
    )
    ax.add_patch(poly)
    width = x1 - x0
    xm = 0.5 * (x0 + x1)
    label_fs = 12.5
    big_fs = 28
    detail_fs = 10.0
    label_y = yc + 0.11
    big_y = yc + 0.01
    detail_y = yc - 0.11
    if width < 0.19 and x0 > 0.70:
        xm = x0 + 0.34 * width
        label_fs = 10.8
        big_fs = 24
        detail_fs = 9.0
        label_y = yc + 0.095
        big_y = yc + 0.005
        detail_y = yc - 0.095
    ax.text(xm, label_y, label, ha="center", va="center", fontsize=label_fs, color="white", fontweight="bold")
    ax.text(xm, big_y, big, ha="center", va="center", fontsize=big_fs, color="white", fontweight="bold")
    ax.text(xm, detail_y, detail, ha="center", va="center", fontsize=detail_fs, color="white", linespacing=1.15)


def fig01_search_funnel(d: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title_block(
        fig,
        "The reset search is wide at the start and selective at the end",
        "Per question and anchor: real macro candidates enter first, only a small regime-distinct set survives ranking.",
    )

    stages = [
        (0.05, 0.18, 0.76, 0.70, C["charcoal"], "Anchor state", "1", "fixed benchmark state"),
        (0.18, 0.39, 0.70, 0.54, "#4f5b67", "Broad generation", "80", "20 analogs\n+ 60 LHS draws"),
        (0.39, 0.59, 0.54, 0.39, C["blue"], "Plausibility", "40–60", "inside the prior\nenvelope"),
        (0.59, 0.77, 0.39, 0.27, "#18548d", "Refinement", "15", "bounded GD"),
        (0.77, 0.95, 0.27, 0.17, C["ink"], "Top 5", "5", "final states"),
    ]
    for args in stages:
        draw_stage(ax, *args)

    ax.text(0.05, 0.16, "Reset method logic", fontsize=11.0, color=C["mid"], ha="left")
    ax.text(
        0.05,
        0.095,
        "Analogs and Latin Hypercube draws create regime diversity first.\nOnly then do plausibility filtering, refinement, and ranking narrow the set.",
        fontsize=12.1,
        color=C["charcoal"],
        ha="left",
        linespacing=1.25,
    )
    ax.text(0.95, 0.06, "Counts from the reset search plan: 20 analogs, 60 LHS draws, 15 refined states, 5 selected states.", fontsize=10.0, color=C["mid"], ha="right")
    save(fig, "scenario_flagship_01_search_funnel_v4")


def fig02_question_to_objective(d: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title_block(
        fig,
        "Three questions survived the reset",
        "Q2 is demoted because its selected states collapse to identical mixed-mid-cycle solutions; the surviving questions have clearer objectives and sharper limits.",
    )

    rows = [
        ("Q1_gold_threshold", 0.77),
        ("Q3_return_discipline", 0.50),
        ("Q4_return_ceiling", 0.23),
    ]
    for _, y in rows:
        ax.plot([0.04, 0.96], [y - 0.11, y - 0.11], color=C["light"], lw=0.9)

    for qid, y in rows:
        meta = QUESTION_META[qid]
        ax.text(0.06, y + 0.05, meta["short"], fontsize=12, color=C["mid"], fontweight="bold", ha="left")
        ax.text(0.11, y + 0.05, meta["title"], fontsize=16, color=C["ink"], fontweight="bold", ha="left")
        ax.text(0.11, y - 0.01, meta["formula"], fontsize=15, color=C["charcoal"], ha="left", va="center")

        ax.text(0.52, y + 0.045, "Rewards", fontsize=11.2, color=C["mid"], ha="left")
        ax.text(0.52, y + 0.005, meta["reward"], fontsize=11.8, color=C["ink"], ha="left")
        ax.text(0.52, y - 0.055, "Penalizes", fontsize=11.2, color=C["mid"], ha="left")
        ax.text(0.52, y - 0.095, meta["penalty"], fontsize=11.4, color=C["ink"], ha="left")

        ax.text(0.78, y + 0.045, "What the search teaches", fontsize=11.2, color=C["mid"], ha="left")
        ax.text(0.78, y - 0.01, meta["takeaway"], fontsize=10.7, color=C["charcoal"], ha="left", va="center", linespacing=1.18)

    ax.text(0.06, 0.07, "Demoted from the main package", fontsize=10.8, color=C["mid"], ha="left")
    ax.text(0.24, 0.07, "Q2 equal-weight departure", fontsize=12.0, color=C["ink"], fontweight="bold", ha="left")
    ax.text(0.46, 0.07, "Selected states collapse to one identical mixed-mid-cycle solution,\nso the question is not credible as a flagship result.", fontsize=10.3, color=C["charcoal"], ha="left", linespacing=1.15)
    save(fig, "scenario_flagship_02_question_to_objective_v4")


def _hist_density(values: np.ndarray, bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist


def fig03_plausibility_vs_macro_shift(d: dict[str, pd.DataFrame]) -> None:
    feature = d["feature_master"]
    selected = d["selected"]
    shift = d["shift"]

    q = "Q1_gold_threshold"
    anchor = "2021-12-31"
    chosen = selected[(selected["question_id"] == q) & (selected["anchor_date"] == anchor) & (selected["selection_rank"] <= 5)].copy()
    target_anchor = feature.loc[feature["month_end"].eq(pd.Timestamp("2022-12-31"))].iloc[0]
    variables = [
        ("us_real10y", "US real 10Y yield", "%"),
        ("ig_oas", "IG spread", "%"),
        ("vix", "VIX", ""),
        ("infl_US", "US inflation", "%"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=FIGSIZE, sharex=False)
    title_block(
        fig,
        "Selected states move meaningfully without leaving the plausible macro space",
        "Gold-threshold search from the 2021 anchor: selected scenarios shift toward the 2022 stress-and-real-yield zone without becoming arbitrary outliers.",
    )

    for ax, (var, label, unit) in zip(axes, variables):
        hist_x, hist_y = _hist_density(feature[var].dropna().to_numpy(dtype=float))
        hist_y = hist_y / hist_y.max()
        ax.fill_between(hist_x, 0, hist_y, color=C["neutral_fill"], alpha=1.0, linewidth=0)
        ax.plot(hist_x, hist_y, color=C["light"], lw=1.0)

        anchor_val = shift[(shift["question_id"] == q) & (shift["anchor_date"] == anchor) & (shift["variable"] == var)]["anchor_value"].iloc[0]
        sel_vals = chosen[var].to_numpy(dtype=float)
        sel_mean = float(sel_vals.mean())
        target_val = float(target_anchor[var])

        ax.axvline(anchor_val, color=C["charcoal"], lw=1.8, zorder=4)
        ax.axvline(target_val, color=C["gold"], lw=2.2, linestyle=(0, (5, 4)), zorder=4)
        ax.scatter(sel_vals, np.repeat(0.18, len(sel_vals)) + np.linspace(-0.03, 0.03, len(sel_vals)), color=C["blue"], s=42, zorder=5, edgecolor="white", linewidth=0.5)
        ax.plot([sel_mean, sel_mean], [0.0, 0.92], color=C["blue"], lw=3.0, zorder=5)

        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=78, ha="left", va="center", color=C["ink"], fontsize=11.5)
        ax.grid(axis="x", color=C["grid"], linewidth=0.8)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", colors=C["mid"])

        ax.text(anchor_val, 1.02, "anchor", color=C["charcoal"], fontsize=10.0, ha="center", va="bottom")
        ax.text(sel_mean, 1.02, "selected mean", color=C["blue"], fontsize=10.0, ha="center", va="bottom")
        ax.text(target_val, 1.02, "2022 anchor", color=C["gold"], fontsize=10.0, ha="center", va="bottom")

        if unit:
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    handles = [
        Line2D([0], [0], color=C["charcoal"], lw=1.8, label="2021 anchor"),
        Line2D([0], [0], color=C["blue"], lw=3.0, label="selected scenario mean"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=C["blue"], markeredgecolor="white", markersize=7, linestyle="None", label="selected scenarios"),
        Line2D([0], [0], color=C["gold"], lw=2.2, linestyle=(0, (5, 4)), label="2022 anchor reference"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.84, bottom=0.14, hspace=0.36)
    save(fig, "scenario_flagship_03_plausibility_vs_macro_shift_v4")


def _macro_vector_from_feature_row(row: pd.Series) -> np.ndarray:
    return np.array([float(row[c]) for c in MACRO_STATE_COLS], dtype=float)


def fig04_regime_transition(d: dict[str, pd.DataFrame]) -> None:
    feature = d["feature_master"]
    selected = d["selected"]
    transitions = d["transition"]

    q = "Q1_gold_threshold"
    anchor_date = pd.Timestamp("2021-12-31")
    scenario_row = selected[
        (selected["question_id"] == q)
        & (selected["anchor_date"] == "2021-12-31")
        & (selected["candidate_idx"] == 0)
    ].iloc[0]

    thresholds = build_regime_thresholds(feature)
    anchor_row = feature.loc[feature["month_end"].eq(anchor_date)].iloc[0]
    anchor_m = _macro_vector_from_feature_row(anchor_row)
    scenario_m = np.array([float(scenario_row[c]) for c in MACRO_STATE_COLS], dtype=float)
    anchor_regime, anchor_dims = classify_regime_v2(anchor_m, thresholds)
    _, scenario_dims = classify_regime_v2(scenario_m, thresholds)

    dims = list(DIM_LABELS.keys())
    ypos = np.arange(len(dims))[::-1]
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.70, 0.30], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[0, 1])
    ax_note.axis("off")

    title_block(
        fig,
        "The strongest scenario changes the regime, not just the weights",
        "Gold-threshold search from the 2021 anchor: the selected state moves from reflationary risk-on into a high-stress defensive regime.",
    )

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.6, len(dims) - 0.4)
    ax.axvline(0, color=C["light"], lw=1.0)
    for y, dim in zip(ypos, dims):
        ax.plot([-1, 1], [y, y], color=C["grid"], lw=0.8, zorder=1)
        a = anchor_dims[dim]
        s = scenario_dims[dim]
        a_draw, s_draw = a, s
        if abs(a - s) < 0.05:
            a_draw -= 0.035
            s_draw += 0.035
        ax.plot([a_draw, s_draw], [y, y], color=C["mid"], lw=2.2, zorder=2)
        ax.scatter(a_draw, y, s=95, color=C["charcoal"], marker="o", edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter(s_draw, y, s=120, color=C["blue"], marker="o", edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(-1.15, y, DIM_LABELS[dim], ha="left", va="center", fontsize=12.5, color=C["ink"])

    ax.set_yticks([])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["weak", "", "neutral", "", "strong"])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", colors=C["mid"])

    ax.text(0.02, 1.02, "Anchor regime", transform=ax.transAxes, fontsize=10.8, color=C["mid"], ha="left")
    ax.text(0.02, 0.97, anchor_regime.replace("_", " "), transform=ax.transAxes, fontsize=15.5, color=C["charcoal"], ha="left", fontweight="bold")
    ax.text(0.72, 1.02, "Selected scenario regime", transform=ax.transAxes, fontsize=10.8, color=C["mid"], ha="left")
    ax.text(0.72, 0.97, scenario_row["regime_label"].replace("_", " "), transform=ax.transAxes, fontsize=15.5, color=C["blue"], ha="left", fontweight="bold")

    top_shift = transitions[
        (transitions["question_id"] == q)
        & (transitions["anchor_date"] == "2021-12-31")
        & (transitions["candidate_idx"] == 0)
    ].copy()
    top_shift["abs_std"] = top_shift["shift_std"].abs()
    top_shift = top_shift.sort_values("abs_std", ascending=False).head(3)

    ax_note.text(0.00, 0.86, "Three shifts doing the work", fontsize=14.5, fontweight="bold", color=C["ink"], ha="left")
    yy = 0.70
    for _, row in top_shift.iterrows():
        ax_note.text(0.00, yy, row["shift_variable"], fontsize=12.4, color=C["charcoal"], ha="left", fontweight="bold")
        ax_note.text(0.72, yy, f"{row['shift_std']:+.2f}σ", fontsize=13.0, color=C["blue"], ha="right", fontweight="bold")
        yy -= 0.12
    ax_note.text(0.00, 0.26, "Interpretation", fontsize=11.2, color=C["mid"], ha="left")
    ax_note.text(
        0.00,
        0.16,
        "The selected state pushes stress and financial conditions higher while keeping inflation elevated. The benchmark responds by moving toward gold and other defensives.",
        fontsize=12.1,
        color=C["charcoal"],
        ha="left",
        va="center",
        wrap=True,
    )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.12)
    save(fig, "scenario_flagship_04_regime_transition_v4")


def fig05_portfolio_consequence(d: dict[str, pd.DataFrame]) -> None:
    truth = d["truth"].copy()
    truth["anchor_date"] = pd.to_datetime(truth["anchor_date"])
    base = truth.loc[truth["anchor_date"].eq(pd.Timestamp("2021-12-31"))].iloc[0]
    after = truth.loc[truth["anchor_date"].eq(pd.Timestamp("2022-12-31"))].iloc[0]

    sleeve_cols = [c for c in truth.columns if c.startswith("w_")]
    delta = pd.DataFrame(
        {
            "sleeve": sleeve_cols,
            "base": [float(base[c]) for c in sleeve_cols],
            "after": [float(after[c]) for c in sleeve_cols],
        }
    )
    delta["change"] = delta["after"] - delta["base"]
    delta = delta.loc[delta["change"].abs() >= 0.01].sort_values("change")

    y = np.arange(len(delta))
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.74, 0.26], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_side.axis("off")

    title_block(
        fig,
        "The regime shift has a clear allocation consequence",
        "The strongest scenario story is the 2021 to 2022 transition: more gold and duration, less equity concentration, higher expected return.",
    )

    for _, row in delta.iterrows():
        color = SLEEVE_COLORS[row["sleeve"]]
        ax.plot([row["base"] * 100, row["after"] * 100], [y[delta.index.get_loc(row.name)], y[delta.index.get_loc(row.name)]], color=C["light"], lw=2.0, zorder=1)
        ax.scatter(row["base"] * 100, y[delta.index.get_loc(row.name)], s=90, color=C["charcoal"], edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(row["after"] * 100, y[delta.index.get_loc(row.name)], s=120, color=color, edgecolor="white", linewidth=0.6, zorder=4)

    ax.set_yticks(y, [SLEEVE_LABELS[s] for s in delta["sleeve"]])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Portfolio weight")
    ax.grid(axis="x", color=C["grid"], linewidth=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(0, max(delta["after"].max(), delta["base"].max()) * 100 + 8)

    handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor=C["charcoal"], markeredgecolor="white", markersize=8, linestyle="None", label="Dec 2021 baseline"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=C["blue"], markeredgecolor="white", markersize=8, linestyle="None", label="Dec 2022 benchmark"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    ax_side.text(0.00, 0.82, "What changed most", fontsize=14.5, fontweight="bold", color=C["ink"], ha="left")
    ax_side.text(0.00, 0.69, f"Gold  {base['w_ALT_GLD']*100:.1f}% to {after['w_ALT_GLD']*100:.1f}%", fontsize=13.6, color=C["gold"], fontweight="bold", ha="left")
    ax_side.text(0.00, 0.59, f"UST  {base['w_FI_UST']*100:.1f}% to {after['w_FI_UST']*100:.1f}%", fontsize=13.0, color=SLEEVE_COLORS['w_FI_UST'], ha="left")
    ax_side.text(0.00, 0.49, f"US Eq  {base['w_EQ_US']*100:.1f}% to {after['w_EQ_US']*100:.1f}%", fontsize=13.0, color=SLEEVE_COLORS['w_EQ_US'], ha="left")
    ax_side.text(0.00, 0.33, "Expected return", fontsize=11.2, color=C["mid"], ha="left")
    ax_side.text(0.00, 0.25, f"{base['pred_return_pct']:.1f}% to {after['pred_return_pct']:.1f}%", fontsize=20, color=C["charcoal"], fontweight="bold", ha="left")
    ax_side.text(
        0.00,
        0.10,
        "The portfolio response is not random. The macro regime shift that activates gold also reduces US equity concentration and raises the role of defensives.",
        fontsize=12.0,
        color=C["charcoal"],
        ha="left",
        va="center",
        wrap=True,
    )

    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.12)
    save(fig, "scenario_flagship_05_portfolio_consequence_v4")


def write_index() -> None:
    lines = [
        "# Scenario Flagship Graphics Index — v4",
        "",
        "- Package: reset-based scenario flagship graphics",
        "- Benchmark: `best_60_tuned_robust`",
        "- Main method story: broad candidate generation, plausibility filter, targeted refinement, ranking, regime-transition interpretation",
        "",
        "| File | Final title | Purpose | Speaker takeaway | Deck use |",
        "| --- | --- | --- | --- | --- |",
        "| `scenario_flagship_01_search_funnel_v4.png` | The reset search is wide at the start and selective at the end | Explain how the reset search narrows real macro candidates into a small selected set. | We do not hand-pick states; the process starts broad and becomes selective only after plausibility and ranking. | main deck |",
        "| `scenario_flagship_02_question_to_objective_v4.png` | Three questions survived the reset | Show what the surviving questions optimize and why Q2 is demoted. | Only three questions produce credible flagship content, and each has a clear objective and a clear limit. | main deck |",
        "| `scenario_flagship_03_plausibility_vs_macro_shift_v4.png` | Selected states move meaningfully without leaving the plausible macro space | Show that the chosen macro shifts are large enough to matter but still historically plausible. | The selected states are not arbitrary; they move toward a real macro zone that history has already shown. | main deck |",
        "| `scenario_flagship_04_regime_transition_v4.png` | The strongest scenario changes the regime, not just the weights | Explain the transition from the anchor regime to the selected defensive regime. | The model changes the portfolio because the regime changes, not because the optimizer is unstable. | main deck |",
        "| `scenario_flagship_05_portfolio_consequence_v4.png` | The regime shift has a clear allocation consequence | Show how the strongest regime story changes the strategic allocation. | Gold and duration rise, equity concentration falls, and the allocation shift is economically interpretable. | main deck |",
    ]
    INDEX_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    configure_style()
    data = load_data()
    fig01_search_funnel(data)
    fig02_question_to_objective(data)
    fig03_plausibility_vs_macro_shift(data)
    fig04_regime_transition(data)
    fig05_portfolio_consequence(data)
    write_index()


if __name__ == "__main__":
    main()
