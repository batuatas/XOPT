#!/usr/bin/env python3
"""
make_conference_story_slide.py

Create a cleaner, narrative-style slide for ONE anchor and ONE question.

What it shows
-------------
1. Start state:
   - anchor date
   - anchor regime (friendly label)
   - question and plain-English G-function

2. Where the generator goes:
   - regime distribution with friendly names
   - anchor regime marked explicitly

3. What changes:
   - generated mean outcomes (total return, excess return, gold weight)
   - generated mean portfolio emphasis for a few key sleeves

4. What drives it:
   - top macro moves summarized from scenario_results top_shift columns
   - human-readable labels, not code labels

5. One-sentence takeaway:
   - auto-written summary for non-technical audiences

Usage
-----
PYTHONPATH=src python make_conference_story_slide.py \
  --scenario_csv /path/to/scenario_results_v4.csv \
  --anchor 2022-12-31 \
  --question_id Q1_gold_favorable \
  --output /path/to/story_slide_2022_q1.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


QUESTION_TEXT = {
    "Q1_gold_favorable": "Question: What macro conditions make the model want more gold?",
    "Q2_ew_deviation": "Question: When does the benchmark lean hardest away from a neutral allocation?",
    "Q3_house_view_7pct_total": "Question: What macro conditions make the benchmark deliver about 7% total return?",
    "Q3_house_saa_total": "Question: What macro conditions make the benchmark deliver about 7% total return?",
}

G_TEXT = {
    "Q1_gold_favorable": "Search target: plausible macro states that increase gold weight.",
    "Q2_ew_deviation": "Search target: plausible macro states that strengthen the model's active conviction.",
    "Q3_house_view_7pct_total": "Search target: plausible macro states that move predicted total return toward 7%.",
    "Q3_house_saa_total": "Search target: plausible macro states that move predicted total return toward 7%.",
}

REGIME_FRIENDLY = {
    # Current regime.py labels
    "recession_stress": "Recession / severe stress",
    "high_stress": "High stress",
    "higher_for_longer": "Sticky inflation, tight policy",
    "inflationary_expansion": "Inflationary expansion",
    "soft_landing": "Soft landing",
    "disinflationary_slowdown": "Weak growth, cooling inflation",
    "risk_off_defensive": "Risk-off / defensive",
    "mid_cycle_neutral": "Mid-cycle neutral",
    # Legacy labels (backward compat with older scenario_results)
    "risk_off_stress": "Risk-off / stress",
    "mixed_mid_cycle": "Cooling inflation, still restrictive",
    "reflation_risk_on": "Recovery / risk-on",
    "high_stress_defensive": "Acute crisis stress",
}

REGIME_DESC = {
    # Current regime.py labels
    "recession_stress": "Recession: NFCI proxy high plus weak growth.",
    "high_stress": "High financial stress without confirmed recession.",
    "higher_for_longer": "Inflation still elevated and policy still restrictive.",
    "inflationary_expansion": "High inflation, low stress, low unemployment.",
    "soft_landing": "Inflation cooling without a clear collapse in growth.",
    "disinflationary_slowdown": "Cooling inflation but weaker activity.",
    "risk_off_defensive": "Defensive conditions: stress, flight to safety, wider spreads.",
    "mid_cycle_neutral": "Growth still okay, inflation cooler, policy not yet easy.",
    # Legacy labels (backward compat with older scenario_results)
    "risk_off_stress": "Defensive conditions: stress, flight to safety, wider spreads.",
    "mixed_mid_cycle": "Growth still okay, inflation cooler, policy not yet easy.",
    "reflation_risk_on": "Recovery-style environment with low stress and risk appetite.",
    "high_stress_defensive": "Severe stress / crisis-like environment.",
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

KEY_SLEEVES = {
    "w_ALT_GLD": "Gold",
    "w_FI_UST": "US Treasuries",
    "w_EQ_US": "US equities",
    "w_CR_US_IG": "US IG credit",
    "w_EQ_CN": "China equities",
    "w_EQ_JP": "Japan equities",
}


def friendly_regime(label: str) -> str:
    return REGIME_FRIENDLY.get(label, label)


def summarize_top_moves(grp: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    pieces = []
    for k in [1, 2, 3]:
        vcol = f"top_shift_{k}_var"
        scol = f"top_shift_{k}_val"
        if vcol in grp.columns and scol in grp.columns:
            tmp = grp[[vcol, scol]].copy()
            tmp.columns = ["var", "val"]
            tmp = tmp[tmp["var"].astype(str) != ""]
            pieces.append(tmp)
    if not pieces:
        return pd.DataFrame(columns=["var", "score", "mean_val", "count"])
    allm = pd.concat(pieces, ignore_index=True)
    out = (
        allm.groupby("var")
        .agg(
            score=("val", lambda x: float(np.mean(np.abs(x)) * len(x))),
            mean_val=("val", "mean"),
            count=("val", "size"),
        )
        .sort_values("score", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return out


def build_takeaway(anchor_regime: str, dominant_regime: str, qid: str, gold_mean: float, total_mean: float) -> str:
    start = friendly_regime(anchor_regime)
    end = friendly_regime(dominant_regime)
    if qid == "Q1_gold_favorable":
        core = f"Starting from a {start.lower()} anchor, the scenarios that make the model want more gold mostly move toward {end.lower()}."
    elif qid == "Q2_ew_deviation":
        core = f"Starting from a {start.lower()} anchor, the strongest active tilts appear mainly in {end.lower()} states."
    else:
        core = f"Starting from a {start.lower()} anchor, the scenarios that support the return target mostly sit in {end.lower()} states."
    tail = f" In those generated scenarios, gold averages {100*gold_mean:.1f}% and predicted total return averages {100*total_mean:.1f}%."
    return core + tail


def make_slide(scenario_csv: Path, anchor: str, question_id: str, output: Path) -> None:
    df = pd.read_csv(scenario_csv)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    anchor = str(pd.Timestamp(anchor).date())

    grp = df[(df["anchor_date"] == anchor) & (df["question_id"] == question_id)].copy()
    if grp.empty:
        raise ValueError(f"No rows found for anchor={anchor}, question_id={question_id}")

    anchor_regime = str(grp["anchor_regime"].iloc[0]) if "anchor_regime" in grp.columns else "unknown"
    regime_shares = grp["regime_label"].value_counts(normalize=True)
    dominant_regime = regime_shares.index[0]
    dominant_share = regime_shares.iloc[0]

    mean_total = float(grp["pred_return_total"].mean())
    mean_excess = float(grp["pred_return_excess"].mean())
    mean_gold = float(grp["w_ALT_GLD"].mean())

    key_weights = (
        grp[list(KEY_SLEEVES.keys())]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    top_moves = summarize_top_moves(grp, top_n=5)

    takeaway = build_takeaway(anchor_regime, dominant_regime, question_id, mean_gold, mean_total)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.4, 1.1])

    # Header
    axh = fig.add_subplot(gs[0, :])
    axh.axis("off")
    header = (
        f"{anchor}  |  {QUESTION_TEXT.get(question_id, question_id)}\n"
        f"{G_TEXT.get(question_id, '')}\n"
        f"Start regime: {friendly_regime(anchor_regime)}  →  Most generated scenarios: {friendly_regime(dominant_regime)} ({100*dominant_share:.1f}%)"
    )
    axh.text(0.01, 0.95, header, va="top", fontsize=18)

    # Regime distribution
    ax1 = fig.add_subplot(gs[1, 0])
    show_regimes = regime_shares.index.tolist()
    y = np.arange(len(show_regimes))
    vals = regime_shares.values
    labels = [friendly_regime(x) for x in show_regimes]
    ax1.barh(y, vals)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.invert_yaxis()
    ax1.set_title("Where the scenarios go", fontsize=14)
    ax1.set_xlim(0, max(0.6, vals.max() * 1.18))
    for i, v in enumerate(vals):
        ax1.text(v + 0.01, i, f"{100*v:.1f}%", va="center", fontsize=11)
    if anchor_regime in show_regimes:
        idx = show_regimes.index(anchor_regime)
        ax1.text(0.01, idx, "start", va="center", ha="left", fontsize=10)

    # Outcomes
    ax2 = fig.add_subplot(gs[1, 1])
    outcome_names = ["Pred. total return", "Pred. excess return", "Gold weight"]
    outcome_vals = [100 * mean_total, 100 * mean_excess, 100 * mean_gold]
    y2 = np.arange(len(outcome_names))
    ax2.barh(y2, outcome_vals)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(outcome_names, fontsize=11)
    ax2.invert_yaxis()
    ax2.set_title("What the generated scenarios imply", fontsize=14)
    for i, v in enumerate(outcome_vals):
        ax2.text(v + max(outcome_vals) * 0.02, i, f"{v:.1f}", va="center", fontsize=11)

    # Portfolio
    ax3 = fig.add_subplot(gs[1, 2])
    ky = key_weights.index.tolist()
    kvals = key_weights.values * 100
    y3 = np.arange(len(ky))
    ax3.barh(y3, kvals)
    ax3.set_yticks(y3)
    ax3.set_yticklabels([KEY_SLEEVES.get(k, k) for k in ky], fontsize=11)
    ax3.invert_yaxis()
    ax3.set_title("Generated mean portfolio emphasis", fontsize=14)
    for i, v in enumerate(kvals):
        ax3.text(v + max(kvals) * 0.02, i, f"{v:.1f}%", va="center", fontsize=11)

    # Macro moves
    ax4 = fig.add_subplot(gs[2, :2])
    if top_moves.empty:
        ax4.axis("off")
        ax4.text(0.01, 0.8, "No macro shift summary available.", fontsize=13)
    else:
        move_labels = [VAR_FRIENDLY.get(v, v) for v in top_moves["var"]]
        move_vals = top_moves["mean_val"].values
        y4 = np.arange(len(move_labels))
        ax4.barh(y4, move_vals)
        ax4.axvline(0, linewidth=1)
        ax4.set_yticks(y4)
        ax4.set_yticklabels(move_labels, fontsize=11)
        ax4.invert_yaxis()
        ax4.set_title("Main macro moves the generator uses", fontsize=14)
        for i, v in enumerate(move_vals):
            sign = "up" if v > 0 else "down"
            ax4.text(v + (0.04 if v >= 0 else -0.04), i, f"{sign} ({v:+.2f})",
                     va="center", ha="left" if v >= 0 else "right", fontsize=11)

    # Takeaway
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis("off")
    wrapped = textwrap.fill(takeaway, width=34)
    ax5.text(0.02, 0.95, "Takeaway", fontsize=15, va="top")
    ax5.text(0.02, 0.82, wrapped, fontsize=13, va="top")

    fig.suptitle("Scenario storyline", fontsize=22, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario_csv", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--question_id", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    make_slide(Path(args.scenario_csv), args.anchor, args.question_id, Path(args.output))


if __name__ == "__main__":
    main()
