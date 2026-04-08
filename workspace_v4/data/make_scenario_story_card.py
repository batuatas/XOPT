#!/usr/bin/env python3
"""
make_scenario_story_card.py

Create a single, story-first figure for ONE anchor and ONE question.

What it shows
-------------
- Anchor regime and anchor portfolio snapshot
- Generated regime distribution ("where scenarios go")
- Portfolio shift from anchor -> generated mean
- Prior/history vs generated distributions for the most important macro variables
- Question text and G-function description in plain English

Usage
-----
cd /Users/batuhanatas/Desktop/XOPTPOE/workspace_v4

PYTHONPATH=src python /mnt/data/make_scenario_story_card.py \
  --workspace /Users/batuhanatas/Desktop/XOPTPOE/workspace_v4 \
  --scenario_csv /Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/scenario_results_v4.csv \
  --anchor 2022-12-31 \
  --question_id Q1_gold_favorable \
  --output /Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/story_card_2022_q1.png

Optional:
  --top_k 6          # number of macro panels
  --variables vix ig_oas infl_US short_rate_US us_real10y usd_broad
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MACRO_COLS = [
    "infl_US","infl_EA","infl_JP",
    "short_rate_US","short_rate_EA","short_rate_JP",
    "long_rate_US","long_rate_EA","long_rate_JP",
    "term_slope_US","term_slope_EA","term_slope_JP",
    "unemp_US","unemp_EA","ig_oas","us_real10y","vix","oil_wti","usd_broad",
]

KEY_SLEEVES = [
    "w_ALT_GLD","w_FI_UST","w_EQ_US","w_CR_US_IG","w_EQ_CN","w_EQ_JP"
]

QUESTION_TEXT = {
    "Q1_gold_favorable": "What macro regime increases the model's gold allocation?",
    "Q2_ew_deviation": "When does the benchmark express the strongest active tilt away from neutral?",
    "Q3_house_view_7pct_total": "What macro regime makes the benchmark predict about 7% total return?",
}

G_FUNCTION_TEXT = {
    "Q1_gold_favorable": (
        "G-function idea: search for plausible macro states that make ALT_GLD weight higher."
    ),
    "Q2_ew_deviation": (
        "G-function idea: search for plausible macro states where the benchmark's conviction over equal-weight is strongest."
    ),
    "Q3_house_view_7pct_total": (
        "G-function idea: search for plausible macro states that move predicted total return toward the 7% target."
    ),
}

REGIME_ORDER = [
    "reflation_risk_on",
    "higher_for_longer",
    "mixed_mid_cycle",
    "risk_off_stress",
    "high_stress_defensive",
    "soft_landing",
    "disinflationary_slowdown",
]


def _read_feature_master(workspace: Path) -> pd.DataFrame:
    fp = workspace / "data_refs" / "feature_master_monthly.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}")
    fm = pd.read_parquet(fp)
    fm["month_end"] = pd.to_datetime(fm["month_end"])
    return fm


def _macro_monthly_panel(fm: pd.DataFrame) -> pd.DataFrame:
    keep = ["month_end"] + [c for c in MACRO_COLS if c in fm.columns]
    out = fm[keep].drop_duplicates(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)
    return out


def _anchor_row(monthly_macro: pd.DataFrame, anchor: pd.Timestamp) -> pd.Series:
    row = monthly_macro.loc[monthly_macro["month_end"].eq(anchor)]
    if row.empty:
        raise ValueError(f"No macro row found for anchor {anchor.date()}")
    return row.iloc[0]


def _prior_stats(monthly_macro: pd.DataFrame, train_end: pd.Timestamp):
    hist = monthly_macro.loc[monthly_macro["month_end"].le(train_end), MACRO_COLS].copy()
    mu = hist.mean()
    sd = hist.std(ddof=1).replace(0, np.nan).fillna(1.0)
    return hist, mu, sd


def _choose_variables(grp: pd.DataFrame, anchor_row: pd.Series, hist_mu: pd.Series, hist_sd: pd.Series, top_k: int):
    mean_gen = grp[MACRO_COLS].mean()
    z_shift = ((mean_gen - anchor_row[MACRO_COLS]) / hist_sd).abs().sort_values(ascending=False)
    return z_shift.index[:top_k].tolist(), z_shift


def _fmt_pct(x: float) -> str:
    return f"{100*x:.1f}%"


def make_story_card(workspace: Path, scenario_csv: Path, anchor_str: str, question_id: str,
                    output: Path, top_k: int = 6, variables: list[str] | None = None) -> None:
    df = pd.read_csv(scenario_csv)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    anchor_str = str(pd.Timestamp(anchor_str).date())

    grp = df[(df["anchor_date"] == anchor_str) & (df["question_id"] == question_id)].copy()
    if grp.empty:
        raise ValueError(f"No rows found for anchor={anchor_str}, question_id={question_id}")

    workspace = Path(workspace)
    fm = _read_feature_master(workspace)
    monthly_macro = _macro_monthly_panel(fm)

    anchor = pd.Timestamp(anchor_str)
    train_end = pd.Timestamp(grp["train_end"].iloc[0])
    hist, hist_mu, hist_sd = _prior_stats(monthly_macro, train_end)
    arow = _anchor_row(monthly_macro, anchor)

    # Select variables for bottom row
    if variables:
        chosen_vars = [v for v in variables if v in MACRO_COLS][:top_k]
        z_shift = ((grp[MACRO_COLS].mean() - arow[MACRO_COLS]) / hist_sd).abs()
    else:
        chosen_vars, z_shift = _choose_variables(grp, arow, hist_mu, hist_sd, top_k=top_k)

    # Summary values
    regime_counts = grp["regime_label"].value_counts(normalize=True)
    dominant_regime = regime_counts.index[0]
    dominant_share = regime_counts.iloc[0]

    anchor_regime = str(grp["anchor_regime"].iloc[0]) if "anchor_regime" in grp.columns else "unknown"
    pred_excess_mean = grp["pred_return_excess"].mean()
    pred_total_mean = grp["pred_return_total"].mean()
    gold_mean = grp["w_ALT_GLD"].mean()

    # Anchor portfolio values: take generated rows' anchor regime but actual anchor weights are not stored.
    # We use the generated mean vs implicit anchor shown in the title text from alignment logs.
    # For practical storytelling we compare generated mean across key sleeves.
    key_weight_means = grp[KEY_SLEEVES].mean().sort_values(ascending=False)

    # Figure layout
    n_vars = len(chosen_vars)
    ncols = 3
    nrows_bottom = math.ceil(n_vars / ncols)
    total_rows = 2 + nrows_bottom

    fig = plt.figure(figsize=(16, 4.8 + 3.2 * total_rows))
    gs = fig.add_gridspec(total_rows, 3, height_ratios=[1.0, 1.2] + [1.1] * nrows_bottom)

    # Header text panel
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    txt = (
        f"Anchor: {anchor_str}    |    Train cutoff: {train_end.date()}    |    Samples: {len(grp)}\n"
        f"Question: {QUESTION_TEXT.get(question_id, question_id)}\n"
        f"{G_FUNCTION_TEXT.get(question_id, '')}\n"
        f"Anchor regime: {anchor_regime}    →    Generated dominant regime: {dominant_regime} ({100*dominant_share:.1f}%)\n"
        f"Generated mean outcomes: total={100*pred_total_mean:.2f}% | excess={100*pred_excess_mean:.2f}% | gold={100*gold_mean:.1f}%"
    )
    ax0.text(0.01, 0.95, txt, va="top", fontsize=13)

    # Regime distribution
    ax1 = fig.add_subplot(gs[1, 0])
    regime_shares = grp["regime_label"].value_counts(normalize=True)
    bars = [regime_shares.get(r, 0.0) for r in REGIME_ORDER if regime_shares.get(r, 0.0) > 0 or r == anchor_regime]
    labels = [r for r in REGIME_ORDER if regime_shares.get(r, 0.0) > 0 or r == anchor_regime]
    y = np.arange(len(labels))
    ax1.barh(y, bars)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(0.6, max(bars) * 1.15 if bars else 0.6))
    ax1.set_title("Where the scenarios go")
    for i, val in enumerate(bars):
        ax1.text(val + 0.01, i, f"{100*val:.1f}%", va="center", fontsize=9)
    if anchor_regime in labels:
        idx = labels.index(anchor_regime)
        ax1.text(0.01, idx, " anchor", va="center", ha="left", fontsize=9)

    # Portfolio response
    ax2 = fig.add_subplot(gs[1, 1])
    show_sleeves = key_weight_means.index.tolist()
    vals = key_weight_means.values
    y2 = np.arange(len(show_sleeves))
    ax2.barh(y2, vals)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([s.replace("w_", "") for s in show_sleeves], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_title("Generated mean portfolio")
    for i, val in enumerate(vals):
        ax2.text(val + 0.005, i, f"{100*val:.1f}%", va="center", fontsize=9)
    ax2.set_xlim(0, max(0.4, vals.max() * 1.25 if len(vals) else 0.4))

    # Top macro shifts summary
    ax3 = fig.add_subplot(gs[1, 2])
    shift_vals = [float(((grp[v].mean() - arow[v]) / hist_sd[v])) for v in chosen_vars]
    y3 = np.arange(len(chosen_vars))
    ax3.barh(y3, shift_vals)
    ax3.axvline(0, linewidth=1)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(chosen_vars, fontsize=9)
    ax3.invert_yaxis()
    ax3.set_title("Largest macro moves (z-shift from anchor)")
    for i, val in enumerate(shift_vals):
        ax3.text(val + (0.05 if val >= 0 else -0.05), i, f"{val:.2f}",
                 va="center", ha="left" if val >= 0 else "right", fontsize=9)

    # Distribution panels
    for i, var in enumerate(chosen_vars):
        r = 2 + i // 3
        c = i % 3
        ax = fig.add_subplot(gs[r, c])

        prior_z = ((hist[var] - hist_mu[var]) / hist_sd[var]).dropna().values
        gen_z = ((grp[var] - hist_mu[var]) / hist_sd[var]).dropna().values
        anchor_z = float((arow[var] - hist_mu[var]) / hist_sd[var])
        mean_z = float(np.mean(gen_z)) if len(gen_z) else np.nan

        ax.hist(prior_z, bins=18, density=True, alpha=0.45, label="History")
        ax.hist(gen_z, bins=18, density=True, alpha=0.45, label="Generated")
        ax.axvline(anchor_z, linestyle="--", linewidth=1.5, label="Anchor")
        ax.axvline(mean_z, linewidth=1.8, label="Generated mean")
        ax.set_title(f"{var}  |  shift={mean_z-anchor_z:+.2f}σ", fontsize=10)
        ax.tick_params(labelsize=8)

        if i == 0:
            ax.legend(fontsize=8)

    # Empty panels if needed
    used = len(chosen_vars)
    total_slots = nrows_bottom * 3
    for j in range(used, total_slots):
        r = 2 + j // 3
        c = j % 3
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    fig.suptitle(f"Scenario story card — {anchor_str} | {question_id}", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--scenario_csv", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--question_id", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--variables", nargs="*", default=None)
    args = ap.parse_args()

    make_story_card(
        workspace=Path(args.workspace),
        scenario_csv=Path(args.scenario_csv),
        anchor_str=args.anchor,
        question_id=args.question_id,
        output=Path(args.output),
        top_k=args.top_k,
        variables=args.variables,
    )


if __name__ == "__main__":
    main()
