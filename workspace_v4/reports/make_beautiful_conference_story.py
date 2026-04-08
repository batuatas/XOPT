#!/usr/bin/env python3
# make_conference_story_slide_v3_clean.py
#
# Key change vs earlier versions:
# --------------------------------
# "Before" allocation is now pulled from the EXACT graphic05 walk-forward
# benchmark path, not rebuilt through the scenario pipeline helper.
#
# That means:
#   - cutoff = anchor - MonthEnd(60)
#   - selected_params come from reports/benchmark/v4_prediction_benchmark_metrics.csv
#   - scores are produced by _fit_best60_model_scores(...)
#   - weights are solved with the locked benchmark optimizer config
#   - covariance comes from monthly_excess_history (exact graphic05 path)
#
# If monthly_excess_history is missing, this script raises a clear error
# instead of silently drawing a misleading "Before" bar.

from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# =========================================================
# Theme — close to graphic05
# =========================================================
BG = "#f2f0ec"
TEXT = "#2f3437"
SUBTEXT = "#65707a"
GRID = "#d7d3cd"

ACCENT = "#6f9a82"
ACCENT_DARK = "#5f8972"
ACCENT_LIGHT = "#9ab7a5"

REGIME_RISK = "#a67958"
REGIME_HFL = "#c8b07c"
REGIME_MID = "#7f96ad"
REGIME_REC = "#89a893"
REGIME_SOFT = "#b9c8a8"
REGIME_SLOW = "#7a8694"
REGIME_CRISIS = "#8d5f4b"

GOLD = "#c7b04a"
UST = "#6a9d79"
US_EQ = "#355f8a"
US_IG = "#4d8f85"
EM_EQ = "#5c85ae"
JP_EQ = "#7a9abb"
EUR_GOV = "#8ead96"
US_RE = "#a5724d"
OTHER = "#c9c5be"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
})


# =========================================================
# Labels
# =========================================================
QUESTION_TEXT = {
    "Q1_more_gold": "What macro conditions make the benchmark want more gold?",
    "Q2_ew_deviation": "When does the benchmark lean hardest away from a neutral allocation?",
    "Q3_house_view_7pct_total": "What macro conditions make the benchmark deliver about 6% total return?",
    "Q3_house_saa_total": "What macro conditions make the benchmark deliver about 7% total return?",
    "Q4_less_gold": "What macro conditions make the benchmark want less gold?",
    "Q5_more_diversified": "What macro condition spreads the portfolio out the most?",
    "Q6_classic_60_40": "What macro conditions push the benchmark toward a classic 60/40 mix?",
    "Q7_stretch_excess": "What macro conditions stretch excess returns toward 5%?",
    "Q8_more_equity": "What macro conditions make the benchmark lean hardest into equities?",
    "Q9_flight_to_safety": "What macro conditions force the benchmark to completely abandon growth for ultimate safety?",
    "Q10_real_asset_rotation": "What macro conditions force a complete rotation into hard assets (Gold + REITs)?",
    "Q11_max_sharpe_total": "What macro conditions deliver the perfect efficient frontier?",
}

G_TEXT = {
    "Q1_more_gold": "G-function: Search for plausible macro states that increase gold weight.",
    "Q2_ew_deviation": "G-function: Search for plausible macro states that strengthen the model's active conviction.",
    "Q3_house_view_7pct_total": "G-function: Search for plausible macro states that move predicted total return toward 6%.",
    "Q3_house_saa_total": "G-function: Search for plausible macro states that move predicted total return toward 7%.",
    "Q4_less_gold": "G-function: Search for plausible macro states that decrease gold weight.",
    "Q5_more_diversified": "G-function: Search for plausible macro states that maximize portfolio entropy.",
    "Q6_classic_60_40": "G-function: Search for plausible macro states that push equity to 60%, fixed income to 40%.",
    "Q7_stretch_excess": "G-function: Search for plausible macro states that lift expected excess return toward 5%.",
    "Q8_more_equity": "G-function: Search for plausible macro states that maximize total equity weight.",
    "Q9_flight_to_safety": "G-function: Search for plausible macro states that maximize Treasuries and Gold.",
    "Q10_real_asset_rotation": "G-function: Search for plausible macro states that maximize Real Estate and Gold.",
    "Q11_max_sharpe_total": "G-function: Search for plausible macro states that maximize the total portfolio Sharpe ratio.",
}

G_EQUATION = {
    "Q1_more_gold":
        r"$G(m) = -\,w_{\mathrm{gold}}(w^*(m))"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q2_ew_deviation":
        r"$G(m) = -[w^*(m)^T r\ -\ \bar{w}^T r]"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q3_house_view_7pct_total":
        r"$G(m) = [p_{\mathrm{total}}(w^*(m),\,m) - 6\%]^2"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q4_less_gold":
        r"$G(m) = +\,w_{\mathrm{gold}}(w^*(m))"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q5_more_diversified":
        r"$G(m) = -\,H(w^*(m))"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q6_classic_60_40":
        r"$G(m) = (w_{\mathrm{eq}} - 60\%)^{2} + (w_{\mathrm{fi}} - 40\%)^{2} + (w_{\mathrm{alt}} - 0\%)^{2}"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q7_stretch_excess":
        r"$G(m) = [p_{\mathrm{ex}}(w^*(m)) - 5\%]^2"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q8_more_equity":
        r"$G(m) = -\,w_{\mathrm{eq\_total}}(w^*(m))"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q9_flight_to_safety":
        r"$G(m) = -\,(w_{\mathrm{ust}} + w_{\mathrm{gold}})(w^*(m))"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q10_real_asset_rotation":
        r"$G(m) = -\,(w_{\mathrm{reits}} + w_{\mathrm{gold}})(w^*(m))"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q11_max_sharpe_total":
        r"$G(m) = -\,S_{\mathrm{total}}(w^*(m))"
        r"\ +\ \lambda\,R(m,\,m_0)$",
    "Q3_house_saa_total":
        r"$G(m) = L(m) + \lambda R(m,m_0)$",
}

REGIME_FRIENDLY = {
    "recession_stress": "Recession / severe stress",
    "high_stress": "High stress",
    "higher_for_longer": "Sticky inflation, tight policy",
    "inflationary_expansion": "Inflationary expansion",
    "soft_landing": "Soft landing",
    "disinflationary_slowdown": "Weak growth, cooling inflation",
    "risk_off_defensive": "Risk-off / defensive",
    "mid_cycle_neutral": "Mid-cycle neutral",
    # Legacy fallbacks
    "risk_off_stress": "Risk-off / stress",
    "mixed_mid_cycle": "Cooling inflation, still restrictive",
    "reflation_risk_on": "Recovery / risk-on",
    "high_stress_defensive": "Acute crisis stress",
}

REGIME_COLORS = {
    "recession_stress": REGIME_CRISIS,
    "high_stress": REGIME_RISK,
    "higher_for_longer": REGIME_HFL,
    "inflationary_expansion": REGIME_REC,
    "soft_landing": REGIME_SOFT,
    "disinflationary_slowdown": REGIME_SLOW,
    "risk_off_defensive": REGIME_RISK,
    "mid_cycle_neutral": REGIME_MID,
    # Legacy fallbacks
    "risk_off_stress": REGIME_RISK,
    "mixed_mid_cycle": REGIME_MID,
    "reflation_risk_on": REGIME_REC,
    "high_stress_defensive": REGIME_CRISIS,
}

VAR_FRIENDLY = {
    "ig_oas": "Credit spreads",
    "vix": "Equity volatility",
    "infl_US": "US inflation",
    "infl_EA": "Euro inflation",
    "infl_JP": "Japan inflation",
    "short_rate_US": "US short rate",
    "short_rate_EA": "Euro short rate",
    "short_rate_JP": "Japan short rate",
    "long_rate_US": "US long yield",
    "long_rate_EA": "Euro long yield",
    "long_rate_JP": "Japan long yield",
    "term_slope_US": "US curve slope",
    "term_slope_EA": "Euro curve slope",
    "term_slope_JP": "Japan curve slope",
    "unemp_US": "US unemployment",
    "unemp_EA": "Euro unemployment",
    "us_real10y": "US real 10Y yield",
    "oil_wti": "Oil",
    "usd_broad": "Broad USD",
}

MACRO_COLS = [
    "infl_US", "infl_EA", "infl_JP",
    "short_rate_US", "short_rate_EA", "short_rate_JP",
    "long_rate_US", "long_rate_EA", "long_rate_JP",
    "term_slope_US", "term_slope_EA", "term_slope_JP",
    "unemp_US", "unemp_EA", "ig_oas", "us_real10y", "vix", "oil_wti", "usd_broad",
]

WEIGHT_COLORS = {
    "w_ALT_GLD": GOLD,
    "w_FI_UST": UST,
    "w_EQ_US": US_EQ,
    "w_CR_US_IG": US_IG,
    "w_EQ_EM": EM_EQ,
    "w_EQ_JP": JP_EQ,
    "w_FI_EU_GOVT": EUR_GOV,
    "w_RE_US": US_RE,
    "Other": OTHER,
}

WEIGHT_LABELS = {
    "w_ALT_GLD": "Gold",
    "w_FI_UST": "UST",
    "w_EQ_US": "US Eq",
    "w_CR_US_IG": "US IG",
    "w_EQ_EM": "EM Eq",
    "w_EQ_JP": "Japan Eq",
    "w_FI_EU_GOVT": "Euro Gov",
    "w_RE_US": "US RE",
    "Other": "Other",
}

ALL_WEIGHT_COLS = [
    "w_ALT_GLD", "w_CR_EU_IG", "w_CR_US_HY", "w_CR_US_IG", "w_EQ_CN", "w_EQ_EM",
    "w_EQ_EZ", "w_EQ_JP", "w_EQ_US", "w_FI_EU_GOVT", "w_FI_UST", "w_LISTED_INFRA",
    "w_LISTED_RE", "w_RE_US",
]

# Dimension -> friendly label for regime dimension chips
DIM_FRIENDLY = {
    "dim_growth": "Growth",
    "dim_inflation": "Inflation",
    "dim_policy": "Policy",
    "dim_stress": "Stress",
}


# =========================================================
# Helpers
# =========================================================
def setup_paths(workspace: Path) -> None:
    ws_src = workspace / "src"
    repo_src = workspace.parent.parent / "src"
    if str(ws_src) not in sys.path:
        sys.path.insert(0, str(ws_src))
    if str(repo_src) not in sys.path:
        sys.path.append(str(repo_src))


def chip(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=12, weight="bold"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        linewidth=0.8,
        facecolor=fc,
        edgecolor=ec or fc,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fs, color=tc, fontweight=weight,
        transform=ax.transAxes
    )


# Abbreviate dimension values for chip display
_DIM_ABBREV = {
    "neutral/strong": "strong",
    "neutral/low": "low",
    "dynamic": "dynamic",
    "high": "high",
    "moderate": "moderate",
    "low": "low",
    "weak": "weak",
    "tight": "tight",
    "easy": "easy",
    "neutral": "neutral",
}


def dim_chip(ax, x, y, label, value, fc, w=0.11, h=0.14):
    """Draw a compact dimension chip: label on top, value below."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=ax.transAxes,
        linewidth=0.6,
        facecolor=fc,
        edgecolor="#00000018",
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h * 0.70, label,
        ha="center", va="center",
        fontsize=9, color="#ffffff", fontweight="bold",
        transform=ax.transAxes
    )
    display_val = _DIM_ABBREV.get(value, value)
    ax.text(
        x + w / 2, y + h * 0.30, display_val,
        ha="center", va="center",
        fontsize=8.5, color="#ffffffdd",
        transform=ax.transAxes
    )


def friendly_regime(x: str) -> str:
    return REGIME_FRIENDLY.get(x, x)


def top_moves_from_results(grp: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    parts = []
    for k in [1, 2, 3]:
        vcol = f"top_shift_{k}_var"
        scol = f"top_shift_{k}_val"
        if vcol in grp.columns and scol in grp.columns:
            tmp = grp[[vcol, scol]].copy()
            tmp.columns = ["var", "val"]
            tmp = tmp[tmp["var"].astype(str) != ""]
            parts.append(tmp)

    if not parts:
        return pd.DataFrame(columns=["var", "score", "mean_val", "count"])

    z = pd.concat(parts, ignore_index=True)
    out = (
        z.groupby("var")
        .agg(
            score=("val", lambda s: float(np.mean(np.abs(s)) * len(s))),
            mean_val=("val", "mean"),
            count=("val", "size"),
        )
        .sort_values("score", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return out


def monthly_macro_panel(fm: pd.DataFrame) -> pd.DataFrame:
    keep = ["month_end"] + [c for c in MACRO_COLS if c in fm.columns]
    out = (
        fm[keep]
        .drop_duplicates(subset=["month_end"])
        .sort_values("month_end")
        .reset_index(drop=True)
    )
    return out


def prior_stats(monthly_macro: pd.DataFrame, train_end: pd.Timestamp):
    hist = monthly_macro.loc[monthly_macro["month_end"].le(train_end), MACRO_COLS].copy()
    mu = hist.mean()
    sd = hist.std(ddof=1).replace(0, np.nan).fillna(1.0)
    return hist, mu, sd


def anchor_row(monthly_macro: pd.DataFrame, anchor: pd.Timestamp) -> pd.Series:
    row = monthly_macro.loc[monthly_macro["month_end"].eq(anchor)]
    if row.empty:
        raise ValueError(f"No macro row for {anchor.date()}")
    return row.iloc[0]


def build_excess_returns(workspace: Path, fm: pd.DataFrame) -> pd.DataFrame:
    try:
        from xoptpoe_v4_models.data import load_modeling_inputs
        inputs = load_modeling_inputs(workspace, feature_set_name="core_baseline")
        hist = inputs.monthly_excess_history
        if hist is not None:
            out = hist.copy()
            out.index = pd.to_datetime(out.index)
            return out.sort_index()
    except Exception:
        pass
    pivot = fm.pivot_table(index="month_end", columns="sleeve_id", values="ret_1m_lag", aggfunc="first")
    return pivot.sort_index()


def compress_weights(w: dict[str, float], keep_cols: list[str]) -> dict[str, float]:
    out = {}
    other = 0.0
    for c in ALL_WEIGHT_COLS:
        val = float(w.get(c, 0.0))
        if c in keep_cols:
            out[c] = val
        else:
            other += val
    out["Other"] = other
    return out


def stacked_bar(ax, y, weights: dict[str, float], title=None):
    left = 0.0
    order = [
        k for k in [
            "w_ALT_GLD", "w_FI_UST", "w_EQ_US", "w_CR_US_IG",
            "w_EQ_EM", "w_EQ_JP", "w_FI_EU_GOVT", "w_RE_US", "Other"
        ] if k in weights
    ]

    # Pass 1: draw all bar segments
    segments = []
    for c in order:
        val = float(weights[c])
        if val <= 1e-6:
            continue
        ax.barh(
            [y], [val], left=left, height=0.28,
            color=WEIGHT_COLORS[c], edgecolor=BG, linewidth=1.2,
            zorder=2,
        )
        segments.append((c, val, left))
        left += val

    # Pass 2: draw text labels on top of all bars (high zorder)
    for c, val, seg_left in segments:
        if val >= 0.12:
            txt_color = "white" if c not in ["Other", "w_ALT_GLD"] else TEXT
            ax.text(
                seg_left + val / 2, y, WEIGHT_LABELS[c],
                ha="center", va="center",
                fontsize=7.5, color=txt_color, clip_on=False,
                zorder=5,
            )

    if title:
        ax.text(-0.03, y, title, ha="right", va="center", fontsize=12, color=SUBTEXT)


def macro_driver_panel(ax, vars_top, grp, arow, hist, mu, sd):
    """Macro driver z-score panel — redesigned for clarity and no overlap."""
    BAR_MIN, BAR_MAX = -2.35, 2.35
    LEFT_X = -3.80
    RIGHT_X = 3.55

    n_vars = len(vars_top)
    y_positions = np.arange(n_vars)[::-1] * 1.15  # extra spacing between rows

    for y, var in zip(y_positions, vars_top):
        prior_z = ((hist[var] - mu[var]) / sd[var]).dropna()
        gen_z = ((grp[var] - mu[var]) / sd[var]).dropna()

        if len(prior_z) == 0 or len(gen_z) == 0:
            continue

        a_z = float((arow[var] - mu[var]) / sd[var])
        g_z = float(gen_z.mean())

        p10, p90 = np.percentile(prior_z, [10, 90])
        g10, g90 = np.percentile(gen_z, [10, 90])

        ax.hlines(
            y,
            np.clip(p10, BAR_MIN, BAR_MAX),
            np.clip(p90, BAR_MIN, BAR_MAX),
            color=SUBTEXT,
            alpha=0.25,
            linewidth=13,
            capstyle="round",
            zorder=1,
        )

        ax.hlines(
            y,
            np.clip(g10, BAR_MIN, BAR_MAX),
            np.clip(g90, BAR_MIN, BAR_MAX),
            color=ACCENT_DARK,
            linewidth=4,
            capstyle="round",
            zorder=2,
        )

        if g90 > BAR_MAX:
            ax.annotate(
                "",
                xy=(BAR_MAX + 0.10, y),
                xytext=(BAR_MAX - 0.15, y),
                arrowprops=dict(arrowstyle="->", color=ACCENT_DARK, lw=1.5),
                clip_on=False,
            )
        if g10 < BAR_MIN:
            ax.annotate(
                "",
                xy=(BAR_MIN - 0.10, y),
                xytext=(BAR_MIN + 0.15, y),
                arrowprops=dict(arrowstyle="->", color=ACCENT_DARK, lw=1.5),
                clip_on=False,
            )

        ax.plot(np.clip(a_z, BAR_MIN, BAR_MAX), y, "o", 
                color="white", markeredgecolor=SUBTEXT, markeredgewidth=1.5, markersize=8, zorder=5)
        ax.plot(np.clip(g_z, BAR_MIN, BAR_MAX), y, "D", 
                color=ACCENT_DARK, markersize=8, zorder=6)

        label = VAR_FRIENDLY.get(var, var)
        direction = "higher" if g_z > a_z else "lower"

        ax.text(
            LEFT_X, y, label,
            ha="left", va="center",
            fontsize=11.5, color=TEXT,
            clip_on=False,
        )

        ax.text(
            RIGHT_X, y, direction,
            ha="right", va="center",
            fontsize=10.5, color=SUBTEXT,
            bbox=dict(facecolor=BG, edgecolor="none", pad=0.15),
            clip_on=False,
        )

    ax.axvline(0, color=GRID, linewidth=1.0)

    y_min = -0.80
    y_max = max(y_positions) + 0.55 if len(y_positions) > 0 else 1.0
    ax.set_xlim(-4.0, 3.7)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([])
    ax.set_xticks([])

    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)

    # Legend row — well separated at bottom
    y0 = y_min + 0.15
    ax.hlines(y0, -3.30, -2.70, color="#b8bec3", linewidth=9, capstyle="round")
    ax.text(-2.55, y0, "history range", va="center", fontsize=10, color=SUBTEXT)

    ax.hlines(y0, -1.20, -0.60, color=ACCENT, linewidth=5, capstyle="round")
    ax.text(-0.45, y0, "generated range", va="center", fontsize=10, color=SUBTEXT)

    ax.plot(0.90, y0, "o", color=TEXT, markersize=6)
    ax.text(1.05, y0, "anchor", va="center", fontsize=10, color=SUBTEXT)

    ax.plot(1.90, y0, "D", color=ACCENT_DARK, markersize=5)
    ax.text(2.05, y0, "gen. mean", va="center", fontsize=10, color=SUBTEXT)


# =========================================================
# EXACT graphic05 anchor reconstruction
# =========================================================
def load_exact_graphic05_anchor_object(
    *,
    workspace: Path,
    anchor: pd.Timestamp,
    lambda_risk: float = 8.0,
    kappa: float = 0.10,
    omega_type: str = "identity",
) -> dict:
    """
    Return the exact benchmark anchor object used by the walk-forward graphic05 path.

    This mirrors the benchmark construction:
      - cutoff = anchor - MonthEnd(60)
      - selected benchmark predictor params from v4_prediction_benchmark_metrics.csv
      - _fit_best60_model_scores(...)
      - covariance from monthly_excess_history
      - optimizer solve with locked config

    Returns
    -------
    dict with:
      weights_full: {"w_ALT_GLD": ..., ...}
      pred_return_excess: float
      rf_rate: float
      pred_return_total: float
      top_sleeve: str
    """
    from xoptpoe_v4_plots.io import BEST_60_EXPERIMENT, _fit_best60_model_scores
    from xoptpoe_v4_models.data import SLEEVE_ORDER, load_modeling_inputs
    from xoptpoe_v4_models.optim_layers import (
        OptimizerConfig,
        RiskConfig,
        RobustOptimizerCache,
        build_sigma_map,
    )

    # Local direct file reads to avoid fragile package dependencies.
    full_panel = pd.read_parquet(workspace / "data_refs" / "modeling_panel_hstack.parquet")
    full_panel["month_end"] = pd.to_datetime(full_panel["month_end"])
    full_panel = full_panel.loc[full_panel["sleeve_id"].isin(SLEEVE_ORDER)].copy()

    feature_manifest = pd.read_csv(workspace / "data_refs" / "feature_set_manifest.csv")
    feature_columns = feature_manifest.loc[
        feature_manifest["include_core_plus_interactions"].eq(1),
        "feature_name"
    ].tolist()

    pred_metrics = pd.read_csv(workspace / "reports" / "benchmark" / "v4_prediction_benchmark_metrics.csv")
    row = pred_metrics.loc[pred_metrics["experiment_name"].eq(BEST_60_EXPERIMENT)]
    if row.empty:
        raise RuntimeError(f"Could not find {BEST_60_EXPERIMENT} in v4_prediction_benchmark_metrics.csv")
    params = ast.literal_eval(str(row["selected_params"].iloc[0]))

    cutoff = pd.Timestamp(anchor) - pd.offsets.MonthEnd(60)

    train_pool = full_panel.loc[
        full_panel["horizon_months"].eq(60)
        & full_panel["baseline_trainable_flag"].eq(1)
        & full_panel["target_available_flag"].eq(1)
        & full_panel["annualized_excess_forward_return"].notna()
    ].copy()

    score_pool = full_panel.loc[full_panel["horizon_months"].eq(60)].copy()

    train_df = train_pool.loc[train_pool["month_end"].le(cutoff)].copy()
    score_df = score_pool.loc[score_pool["month_end"].eq(anchor)].copy()

    if score_df.empty:
        raise RuntimeError(f"No score rows found for anchor={anchor.date()}")

    ordered = score_df.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
    if ordered["sleeve_id"].isna().any():
        raise RuntimeError("Anchor score_df could not be aligned to SLEEVE_ORDER")

    ordered = _fit_best60_model_scores(
        train_df=train_df,
        score_df=ordered,
        feature_manifest=feature_manifest,
        feature_columns=feature_columns,
        params=params,
    )

    loaded = load_modeling_inputs(workspace, feature_set_name="core_baseline")
    monthly_excess_history = loaded.monthly_excess_history
    if monthly_excess_history is None:
        raise RuntimeError(
            "monthly_excess_history is missing. "
            "Cannot reproduce graphic05 exactly for the 'Before' portfolio."
        )

    sigma_map = build_sigma_map([anchor], excess_history=monthly_excess_history, risk_config=RiskConfig())
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    cfg = OptimizerConfig(lambda_risk=lambda_risk, kappa=kappa, omega_type=omega_type)

    mu = ordered["y_pred"].to_numpy(dtype=float)
    w = optimizer_cache.solve(anchor, mu, cfg)
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = float(w.sum())
    if s <= 0:
        raise RuntimeError("Exact benchmark optimizer returned non-positive total weight.")
    w = w / s

    # RF should be common across sleeves at the anchor; use first row if available.
    if "short_rate_US" in ordered.columns:
        rf_rate = float(ordered["short_rate_US"].iloc[0]) / 100.0
    else:
        rf_rate = float("nan")

    pred_return_excess = float(np.dot(w, mu))
    pred_return_total = pred_return_excess + rf_rate if np.isfinite(rf_rate) else float("nan")

    top_idx = int(np.argmax(w))
    top_sleeve = SLEEVE_ORDER[top_idx]

    weights_full = {f"w_{sid}": float(val) for sid, val in zip(SLEEVE_ORDER, w)}

    return {
        "weights_full": weights_full,
        "pred_return_excess": pred_return_excess,
        "rf_rate": rf_rate,
        "pred_return_total": pred_return_total,
        "top_sleeve": top_sleeve,
    }


# =========================================================
# Main
# =========================================================
def make_plot(workspace: Path, scenario_csv: Path, anchor_str: str, question_id: str, output: Path):
    setup_paths(workspace)

    df = pd.read_csv(scenario_csv)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    anchor_str = str(pd.Timestamp(anchor_str).date())

    grp = df[(df["anchor_date"] == anchor_str) & (df["question_id"] == question_id)].copy()
    if grp.empty:
        raise ValueError(f"No rows for anchor={anchor_str}, question_id={question_id}")

    anchor = pd.Timestamp(anchor_str)

    from xoptpoe_v4_scenario.state_space import load_state
    from xoptpoe_v4_scenario.pipeline import (
        build_benchmark_aligned_pipeline_at_date, benchmark_train_end, SLEEVES_14,
    )

    data_refs = workspace / "data_refs"
    fm = pd.read_parquet(data_refs / "feature_master_monthly.parquet")
    mp = pd.read_parquet(data_refs / "modeling_panel_hstack.parquet")
    fsm = pd.read_csv(data_refs / "feature_set_manifest.csv")
    for _df in [fm, mp]:
        _df["month_end"] = pd.to_datetime(_df["month_end"])
    feat_cols = fsm[fsm["include_core_plus_interactions"] == 1]["feature_name"].tolist()

    excess_ret = build_excess_returns(workspace, fm)
    train_end = benchmark_train_end(anchor)
    bench_metrics = workspace / "reports" / "benchmark" / "v4_prediction_benchmark_metrics.csv"

    # ── Before: try exact graphic05 path first, fall back to aligned pipeline ──
    before_full = None
    _before_source = "unknown"
    _exact_obj: dict = {}
    try:
        _exact_obj = load_exact_graphic05_anchor_object(workspace=workspace, anchor=anchor)
        before_full = _exact_obj["weights_full"]
        _before_source = "benchmark_exact"
        print(f"  [BEFORE] covariance_source=benchmark_exact  top={_exact_obj['top_sleeve']}")
    except Exception as _e:
        print(f"  [BEFORE] exact path unavailable ({_e}); falling back to aligned pipeline")

    # Build the pipeline using SLEEVES_14 — same sleeve order as the benchmark runner
    pipeline = build_benchmark_aligned_pipeline_at_date(
        anchor_date=anchor,
        feature_master=fm,
        modeling_panel=mp,
        feature_manifest=fsm,
        feature_columns=feat_cols,
        excess_returns_monthly=excess_ret,
        benchmark_metrics_path=bench_metrics,
        experiment_name="elastic_net__core_plus_interactions__separate_60",
        train_end=train_end,
        sleeve_order=SLEEVES_14,
    )

    m0, _ = load_state(anchor, fm, SLEEVES_14, modeling_panel=mp, horizon_months=60)
    ev0 = pipeline.evaluate_at(m0)

    if before_full is None:
        before_full = {
            f"w_{s}": float(ev0["w"][i])
            for i, s in enumerate(SLEEVES_14)
        }
        _before_source = "fallback_ret_1m_lag"
        print(f"  [BEFORE] covariance_source=fallback_ret_1m_lag  top={max(before_full, key=before_full.get)}")
    after_full = {c: float(grp[c].mean()) for c in ALL_WEIGHT_COLS if c in grp.columns}

    keep_cols = ["w_ALT_GLD", "w_FI_UST", "w_FI_EU_GOVT", "w_EQ_US", "w_CR_US_IG", "w_EQ_EM", "w_EQ_JP", "w_RE_US"]
    before = compress_weights(before_full, keep_cols)
    after = compress_weights(after_full, keep_cols)

    if _before_source == "benchmark_exact":
        _before_total  = float(_exact_obj.get("pred_return_total",  float("nan")))
        _before_excess = float(_exact_obj.get("pred_return_excess", float("nan")))
    else:
        _before_total  = float(ev0.get("pred_return_total",  ev0.get("total_return",        float("nan"))))
        _before_excess = float(ev0.get("pred_return",        ev0.get("pred_return_excess",  float("nan"))))

    anchor_regime = str(grp["anchor_regime"].iloc[0])
    regime_shares = grp["regime_label"].value_counts(normalize=True)
    top2 = regime_shares.head(2)
    top1_name = top2.index[0]
    top1_share = float(top2.iloc[0])

    mean_total = float(grp["pred_return_total"].mean())
    mean_excess = float(grp["pred_return_excess"].mean())

    # ── Sharpe ratio (replaces Gold weight) ─────────────────
    _before_sharpe = float(ev0.get("sharpe_pred_total", float("nan")))
    mean_sharpe = float(grp["sharpe_pred_total"].mean()) if "sharpe_pred_total" in grp.columns else float("nan")

    # ── Regime dimensions ────────────────────────────────────
    # Anchor (before) dimensions — from the first row's anchor_regime side
    # We re-derive them from ev0/m0 via the scenario CSV columns
    # The scenario CSV stores per-sample dim_* columns; for the anchor we
    # compute a mode across generated samples' *anchor* regime.
    dim_cols = ["dim_growth", "dim_inflation", "dim_policy", "dim_stress"]

    # Before dims: use the anchor regime dimensions.
    # The scenario runner stores anchor regime info; we can infer from the
    # first row since all rows share the same anchor.
    # However, anchor dim_* are not directly stored. We reconstitute from the
    # generated side's mode as a proxy, or build from m0 via the regime
    # classifier.
    try:
        from xoptpoe_v4_scenario.regime import classify_single_state, compute_regime_thresholds, load_nfci
        nfci_path = workspace / "NFCI (1).csv"
        nfci_df = load_nfci(nfci_path) if nfci_path.exists() else None
        thresholds = compute_regime_thresholds(fm, nfci_df)
        _, _, signals_before = classify_single_state(m0, thresholds)
        stress_high = signals_before.get("stress_high", 0.0) == 1.0
        stress_mod = signals_before.get("stress_moderate", 0.0) == 1.0
        # Derive policy from short rate: tight if above median, easy if below
        sr_val = signals_before.get("short_rate_US", 0.0)
        sr_med = thresholds.get("short_rate_US_p50", 2.0)
        policy_desc = "tight" if sr_val > sr_med else "easy"
        before_dims = {
            "dim_growth": "weak" if signals_before.get("unemp_US", 0.0) > thresholds.get("unemp_US_p75", 6.0) else "neutral/strong",
            "dim_inflation": "high" if signals_before.get("infl_US", 0.0) > thresholds.get("infl_US_p75", 3.0) else "neutral/low",
            "dim_policy": policy_desc,
            "dim_stress": "high" if stress_high else "moderate" if stress_mod else "low",
        }
    except Exception:
        before_dims = {d: "—" for d in dim_cols}

    # After dims: mode across generated samples
    after_dims = {}
    for d in dim_cols:
        if d in grp.columns:
            after_dims[d] = str(grp[d].mode().iloc[0]) if not grp[d].mode().empty else "—"
        else:
            after_dims[d] = "—"

    monthly_macro = monthly_macro_panel(fm)
    hist, mu_macro, sd_macro = prior_stats(monthly_macro, train_end)
    arow = anchor_row(monthly_macro, anchor)

    moves = top_moves_from_results(grp, top_n=6)
    vars_top = moves["var"].tolist() if not moves.empty else [
        "vix", "ig_oas", "infl_US", "short_rate_US", "long_rate_US", "usd_broad"
    ]

    # =====================================================
    # Figure layout — conference-ready
    # =====================================================
    fig = plt.figure(figsize=(16.0, 12.4))
    gs = fig.add_gridspec(
        3, 6,
        height_ratios=[0.72, 1.05, 1.70],
        width_ratios=[1.8, 0.9, 0.9, 0.9, 1.45, 1.45],
        hspace=0.50,
        wspace=0.50,
    )

    # ── Header ────────────────────────────────────────────
    axh = fig.add_subplot(gs[0, :])
    axh.axis("off")

    # CHANGE #4: Remove "Scenario story", make Date/Question/G big & bold
    axh.text(
        0.01, 0.96, anchor_str,
        fontsize=24, fontweight="bold",
        color=TEXT, transform=axh.transAxes, va="top"
    )

    q_wrapped = textwrap.fill(QUESTION_TEXT.get(question_id, question_id), width=55)
    axh.text(
        0.01, 0.72, f"Q:  {q_wrapped}",
        fontsize=16, fontweight="bold", color=SUBTEXT,
        transform=axh.transAxes, va="top"
    )

    eq_str = G_EQUATION.get(question_id, r"$G(m) = L(m) + \lambda R(m,m_0)$")
    axh.text(
        0.01, 0.22, eq_str,
        fontsize=18, fontweight="bold", color=TEXT,
        transform=axh.transAxes, va="top"
    )

    # ── Regime shift section (top-right) ──────────────────
    axh.text(
        0.755, 0.88, "Regime shift",
        fontsize=15, fontweight="bold",
        ha="center", va="top", color=TEXT,
        transform=axh.transAxes
    )

    # 1. Start Block (Left)
    # Center = 0.63
    chip(axh, 0.54, 0.46, 0.18, 0.24, "Start", fc="#c5c9cd", tc=TEXT, fs=13)
    axh.text(
        0.63, 0.44, textwrap.fill(friendly_regime(anchor_regime), 18),
        fontsize=10.5, ha="center", color=TEXT,
        transform=axh.transAxes, va="top", fontweight="bold"
    )

    # 2. Transition Arrow
    axh.text(
        0.755, 0.35, "→",
        fontsize=32, color=ACCENT_DARK,
        ha="center", va="center", transform=axh.transAxes
    )

    # 3. Generator Block (Right)
    chip(axh, 0.79, 0.46, 0.18, 0.24, "Generator", fc=ACCENT, tc="white", fs=13)
    axh.text(
        0.88, 0.44, textwrap.fill(friendly_regime(top1_name), 18),
        fontsize=10.5, ha="center", color=ACCENT_DARK,
        transform=axh.transAxes, va="top", fontweight="bold"
    )

    # Dimension chips: Growth | Inflation | Policy | Stress
    _dim_val_color = {
        "weak": REGIME_SLOW, "neutral/strong": ACCENT, "strong": ACCENT,
        "high": REGIME_RISK, "neutral/low": ACCENT_LIGHT, "low": ACCENT_LIGHT,
        "moderate": REGIME_HFL,
        "dynamic": REGIME_MID, "tight": REGIME_RISK, "easy": ACCENT,
        "neutral": REGIME_MID,
    }

    cw = 0.088
    ch = 0.160

    # Dimension chips: shifted lower
    y0, y1 = 0.14, -0.04
    dk = "dim_growth"
    dim_chip(axh, 0.535, y0, DIM_FRIENDLY[dk], before_dims.get(dk, "—"), _dim_val_color.get(before_dims.get(dk, "—"), REGIME_MID), w=cw, h=ch)
    dk = "dim_inflation"
    dim_chip(axh, 0.635, y0, DIM_FRIENDLY[dk], before_dims.get(dk, "—"), _dim_val_color.get(before_dims.get(dk, "—"), REGIME_MID), w=cw, h=ch)
    dk = "dim_policy"
    dim_chip(axh, 0.535, y1, DIM_FRIENDLY[dk], before_dims.get(dk, "—"), _dim_val_color.get(before_dims.get(dk, "—"), REGIME_MID), w=cw, h=ch)
    dk = "dim_stress"
    dim_chip(axh, 0.635, y1, DIM_FRIENDLY[dk], before_dims.get(dk, "—"), _dim_val_color.get(before_dims.get(dk, "—"), REGIME_MID), w=cw, h=ch)

    dk = "dim_growth"
    dim_chip(axh, 0.785, y0, DIM_FRIENDLY[dk], after_dims.get(dk, "—"), _dim_val_color.get(after_dims.get(dk, "—"), REGIME_MID), w=cw, h=ch)
    dk = "dim_inflation"
    dim_chip(axh, 0.885, y0, DIM_FRIENDLY[dk], after_dims.get(dk, "—"), _dim_val_color.get(after_dims.get(dk, "—"), REGIME_MID), w=cw, h=ch)
    dk = "dim_policy"
    dim_chip(axh, 0.785, y1, DIM_FRIENDLY[dk], after_dims.get(dk, "—"), _dim_val_color.get(after_dims.get(dk, "—"), REGIME_MID), w=cw, h=ch)
    dk = "dim_stress"
    dim_chip(axh, 0.885, y1, DIM_FRIENDLY[dk], after_dims.get(dk, "—"), _dim_val_color.get(after_dims.get(dk, "—"), REGIME_MID), w=cw, h=ch)

    # ── Panel 1: Where scenarios go (compact, 1 column) ──
    ax1 = fig.add_subplot(gs[1, 0:1])
    ax1.set_title("Where the\nscenarios go", fontsize=13, loc="left", pad=5)

    left = 0.0
    regime_positions = []
    for reg, share in top2.items():
        ax1.barh(
            [0], [share], left=left, height=0.40,
            color=REGIME_COLORS.get(reg, REGIME_MID),
            edgecolor=BG, linewidth=1.2
        )
        if share >= 0.08:
             ax1.text(
                left + share / 2, 0, f"{share * 100:.0f}%",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if reg in ["risk_off_stress", "higher_for_longer",
                                          "recession_stress", "high_stress"] else TEXT
            )
        regime_positions.append((left + share / 2, reg))
        left += share

    if left < 1:
        ax1.barh([0], [1 - left], left=left, height=0.40, color=GRID, edgecolor=BG, linewidth=1.2)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.80, 0.38) # Slightly more room at bottom
    ax1.set_yticks([])
    ax1.set_xticks([])

    for s in ["top", "right", "left", "bottom"]:
        ax1.spines[s].set_visible(False)
        
    for xpos, reg in regime_positions:
        ax1.text(
            xpos, -0.28,
            textwrap.fill(friendly_regime(reg), 12),
            ha="center", va="top",
            fontsize=9.5, fontweight="bold",
            color=REGIME_COLORS.get(reg, SUBTEXT)
        )

    # ── Panel 2: outcomes before → after (3 columns) ────
    # CHANGE #2: Replace Gold weight with Sharpe ratio
    ax2 = fig.add_subplot(gs[1, 1:4])
    ax2.set_title("Scenario outcomes", fontsize=15, loc="left", pad=7)
    ax2.axis("off")

    def _fmt(v):
        return "—" if np.isnan(v) else f"{v * 100:.1f}%"

    def _fmt_sharpe(v):
        return "—" if np.isnan(v) else f"{v:.2f}"

    def _delta(b, a, is_pct=True):
        if np.isnan(b) or np.isnan(a):
            return "", SUBTEXT
        d = a - b
        if is_pct:
            s = f"+{d*100:.1f}%" if d >= 0 else f"{d*100:.1f}%"
        else:
            s = f"+{d:.2f}" if d >= 0 else f"{d:.2f}"
        c = ACCENT if d > 0 else REGIME_RISK
        return s, c

    outcome_rows = [
        ("Total return",   _before_total,  mean_total,   True),
        ("Excess return",  _before_excess, mean_excess,  True),
        ("Sharpe ratio",   _before_sharpe, mean_sharpe,  False),
    ]
    row_ys = [0.80, 0.47, 0.14]
    label_dy = 0.20

    ax2.text(0.34, 0.99, "Before", ha="center", va="top", fontsize=11,
             color=SUBTEXT, transform=ax2.transAxes, style="italic")
    ax2.text(0.66, 0.99, "After",  ha="center", va="top", fontsize=11,
             color=SUBTEXT, transform=ax2.transAxes, style="italic")
    ax2.text(0.88, 0.99, "Change", ha="center", va="top", fontsize=11,
             color=SUBTEXT, transform=ax2.transAxes, style="italic")

    for (label, bval, aval, is_pct), ry in zip(outcome_rows, row_ys):
        ax2.text(0.02, ry + label_dy, label, ha="left", va="top", fontsize=12,
                 color=SUBTEXT, transform=ax2.transAxes)
        formatter = _fmt if is_pct else _fmt_sharpe
        ax2.text(0.34, ry, formatter(bval), ha="center", va="center",
                 fontsize=20, fontweight="bold", color=TEXT, transform=ax2.transAxes)
        ax2.text(0.50, ry, "→", ha="center", va="center",
                 fontsize=18, color=SUBTEXT, transform=ax2.transAxes)
        ax2.text(0.66, ry, formatter(aval), ha="center", va="center",
                 fontsize=20, fontweight="bold", color=ACCENT, transform=ax2.transAxes)
        ds, dc = _delta(bval, aval, is_pct=is_pct)
        ax2.text(0.88, ry, ds, ha="center", va="center",
                 fontsize=13, fontweight="bold", color=dc, transform=ax2.transAxes)

    # ── Panel 3: allocation before/after ─────────────────
    # CHANGE #6: Remove weight callout text
    ax3 = fig.add_subplot(gs[1, 4:])
    ax3.set_title("Portfolio shift", fontsize=15, loc="left", pad=7)

    stacked_bar(ax3, 0.70, before, "Before")
    stacked_bar(ax3, 0.20, after, "After")
    # Removed: add_weight_callouts(...)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.05, 1.00)
    ax3.set_yticks([])
    ax3.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax3.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax3.grid(axis="x", color=GRID, linewidth=0.8)

    for s in ["top", "right", "left"]:
        ax3.spines[s].set_visible(False)

    # ── Panel 4: macro drivers ───────────────────────────
    # CHANGE #5: Fixed spacing, no overlap
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_title("Main macro moves: history vs generated", fontsize=15, loc="left", pad=7)
    macro_driver_panel(ax4, vars_top, grp, arow, hist, mu_macro, sd_macro)

    fig.subplots_adjust(
        left=0.05, right=0.96, top=0.96, bottom=0.05,
        hspace=0.50, wspace=0.50,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--scenario_csv", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--question_id", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    make_plot(
        workspace=Path(args.workspace),
        scenario_csv=Path(args.scenario_csv),
        anchor_str=args.anchor,
        question_id=args.question_id,
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()