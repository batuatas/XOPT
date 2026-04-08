#!/usr/bin/env python3
"""
make_scenario_storyline_plots.py

Story-first plots for XOPTPOE v4 scenario results.

What it creates
---------------
1) regime_storyline_stacked.png
   Stacked shares of generated regimes by anchor/question.

2) portfolio_response_heatmap.png
   Mean generated portfolio weights for key sleeves by anchor/question.

3) macro_shift_heatmap_z.png
   Mean generated macro shifts in standardized units relative to the anchor
   and the historical training distribution.

4) prior_vs_generated/pvg_<anchor>_<question>.png
   One 19-panel page per anchor/question. Each panel overlays:
   - historical training distribution ("prior/history")
   - generated scenario distribution
   - anchor value
   - generated mean

How to run
----------
python make_scenario_storyline_plots.py \
  --workspace /path/to/workspace_v4 \
  --scenario_csv /path/to/workspace_v4/reports/scenario_results_v4.csv \
  --output_dir /path/to/workspace_v4/reports/scenario_storyline

Notes
-----
- If feature_master_monthly.parquet is available under workspace/data_refs/,
  the script makes the full prior-vs-generated and macro-shift plots.
- If feature_master is missing, it still makes the regime and portfolio plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math

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

REGIME_ORDER = [
    # Current regime.py labels
    "recession_stress",
    "high_stress",
    "higher_for_longer",
    "inflationary_expansion",
    "soft_landing",
    "disinflationary_slowdown",
    "risk_off_defensive",
    "mid_cycle_neutral",
    # Legacy labels (backward compat with older scenario_results)
    "reflation_risk_on",
    "mixed_mid_cycle",
    "risk_off_stress",
    "high_stress_defensive",
]

QUESTION_LABELS = {
    "Q1_gold_favorable": "Q1 Gold",
    "Q2_ew_deviation": "Q2 Active tilt",
    "Q3_house_view_7pct_total": "Q3 7% total",
    "Q3_house_saa_total": "Q3 7% total",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_feature_master(workspace: Path) -> pd.DataFrame | None:
    fp = workspace / "data_refs" / "feature_master_monthly.parquet"
    if not fp.exists():
        return None
    fm = pd.read_parquet(fp)
    fm["month_end"] = pd.to_datetime(fm["month_end"])
    return fm


def _macro_monthly_panel(fm: pd.DataFrame) -> pd.DataFrame:
    keep = ["month_end"] + [c for c in MACRO_COLS if c in fm.columns]
    out = fm[keep].drop_duplicates(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)
    return out


def _anchor_row_from_fm(monthly_macro: pd.DataFrame, anchor: pd.Timestamp) -> pd.Series:
    row = monthly_macro.loc[monthly_macro["month_end"].eq(anchor)]
    if row.empty:
        raise ValueError(f"No anchor macro row found for {anchor.date()}")
    return row.iloc[0]


def _prior_stats(monthly_macro: pd.DataFrame, train_end: pd.Timestamp) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    hist = monthly_macro.loc[monthly_macro["month_end"].le(train_end), MACRO_COLS].copy()
    mu = hist.mean()
    sd = hist.std(ddof=1).replace(0, np.nan)
    sd = sd.fillna(1.0)
    return mu, sd, hist


def plot_regime_storyline(df: pd.DataFrame, outpath: Path) -> None:
    order_rows = []
    for anchor in sorted(df["anchor_date"].unique()):
        for q in ["Q1_gold_favorable","Q2_ew_deviation","Q3_house_view_7pct_total"]:
            order_rows.append((anchor, q))
    plot_df = (
        df.groupby(["anchor_date","question_id","regime_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = plot_df.groupby(["anchor_date","question_id"])["count"].transform("sum")
    plot_df["share"] = plot_df["count"] / totals

    xlabels = [f"{a[:4]}\n{QUESTION_LABELS.get(q,q)}" for a, q in order_rows]
    x = np.arange(len(order_rows))
    bottoms = np.zeros(len(order_rows))

    fig, ax = plt.subplots(figsize=(14, 6))
    for regime in REGIME_ORDER:
        vals = []
        for a, q in order_rows:
            row = plot_df[(plot_df["anchor_date"] == a) & (plot_df["question_id"] == q) & (plot_df["regime_label"] == regime)]
            vals.append(float(row["share"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x, vals, bottom=bottoms, label=regime)
        bottoms += np.array(vals)

    ax.set_title("Where the scenario engine goes from each anchor")
    ax.set_ylabel("Share of generated scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_response_heatmap(df: pd.DataFrame, outpath: Path) -> None:
    rows = []
    for (anchor, q), grp in df.groupby(["anchor_date","question_id"]):
        row = {"row_label": f"{anchor[:4]} | {QUESTION_LABELS.get(q,q)}"}
        for c in KEY_SLEEVES:
            row[c] = grp[c].mean()
        rows.append(row)
    mat = pd.DataFrame(rows).set_index("row_label")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(mat))))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_title("Mean generated portfolio weights for key sleeves")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels([c.replace("w_", "") for c in mat.columns], rotation=45, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_macro_shift_heatmap(df: pd.DataFrame, monthly_macro: pd.DataFrame, outpath: Path) -> None:
    rows = []
    for (anchor_str, q), grp in df.groupby(["anchor_date","question_id"]):
        anchor = pd.Timestamp(anchor_str)
        train_end = pd.Timestamp(grp["train_end"].iloc[0])
        mu, sd, _hist = _prior_stats(monthly_macro, train_end)
        arow = _anchor_row_from_fm(monthly_macro, anchor)
        mean_gen = grp[MACRO_COLS].mean()
        z_shift = (mean_gen - arow[MACRO_COLS]) / sd
        row = {"row_label": f"{anchor_str[:4]} | {QUESTION_LABELS.get(q,q)}"}
        row.update(z_shift.to_dict())
        rows.append(row)
    mat = pd.DataFrame(rows).set_index("row_label")[MACRO_COLS]

    fig, ax = plt.subplots(figsize=(16, max(4, 0.45 * len(mat))))
    vmax = np.nanmax(np.abs(mat.values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(mat.values, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Mean macro shift from anchor to generated scenarios (z-score units)")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=75, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_hist_with_lines(ax, prior_z, gen_z, anchor_z, mean_z, title):
    bins = 20
    ax.hist(prior_z, bins=bins, density=True, alpha=0.5, label="History")
    ax.hist(gen_z, bins=bins, density=True, alpha=0.5, label="Generated")
    ax.axvline(anchor_z, linestyle="--", linewidth=1.2, label="Anchor")
    ax.axvline(mean_z, linewidth=1.5, label="Generated mean")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def plot_prior_vs_generated_pages(df: pd.DataFrame, monthly_macro: pd.DataFrame, outdir: Path) -> None:
    _ensure_dir(outdir)
    for (anchor_str, q), grp in df.groupby(["anchor_date","question_id"]):
        anchor = pd.Timestamp(anchor_str)
        train_end = pd.Timestamp(grp["train_end"].iloc[0])
        mu, sd, hist = _prior_stats(monthly_macro, train_end)
        arow = _anchor_row_from_fm(monthly_macro, anchor)

        fig, axes = plt.subplots(5, 4, figsize=(16, 16))
        axes = axes.ravel()

        for i, col in enumerate(MACRO_COLS):
            ax = axes[i]
            prior_z = ((hist[col] - mu[col]) / sd[col]).dropna().values
            gen_z = ((grp[col] - mu[col]) / sd[col]).dropna().values
            anchor_z = float((arow[col] - mu[col]) / sd[col])
            mean_z = float(np.mean(gen_z)) if len(gen_z) else np.nan
            _plot_hist_with_lines(ax, prior_z, gen_z, anchor_z, mean_z, col)

        # last empty panel = legend/story
        if len(axes) > len(MACRO_COLS):
            axes[len(MACRO_COLS)].axis("off")
            txt = (
                f"Anchor: {anchor_str}\n"
                f"Question: {QUESTION_LABELS.get(q,q)}\n"
                f"Train cutoff: {train_end.date()}\n"
                f"Interpretation:\n"
                f"Gray = historical training distribution\n"
                f"Blue = generated scenarios\n"
                f"Dashed line = anchor level\n"
                f"Solid line = generated mean"
            )
            axes[len(MACRO_COLS)].text(0.02, 0.98, txt, va="top", fontsize=10)

        for j in range(len(MACRO_COLS) + 1, len(axes)):
            axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.suptitle(f"Prior vs generated macro distributions — {anchor_str[:4]} | {QUESTION_LABELS.get(q,q)}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.975])

        fname = f"pvg_{anchor_str}_{q}.png".replace(":", "-")
        fig.savefig(outdir / fname, dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_story_summary(df: pd.DataFrame, outpath: Path) -> None:
    rows = []
    for (anchor, q), grp in df.groupby(["anchor_date","question_id"]):
        vc = grp["regime_label"].value_counts(normalize=True)
        rows.append({
            "anchor_date": anchor,
            "question_id": q,
            "question_label": QUESTION_LABELS.get(q, q),
            "mean_total_return": grp["pred_return_total"].mean(),
            "mean_excess_return": grp["pred_return_excess"].mean(),
            "mean_gold_weight": grp["w_ALT_GLD"].mean(),
            "dominant_regime": vc.index[0],
            "dominant_regime_share": vc.iloc[0],
            "second_regime": vc.index[1] if len(vc) > 1 else "",
            "second_regime_share": vc.iloc[1] if len(vc) > 1 else np.nan,
        })
    pd.DataFrame(rows).to_csv(outpath, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=str, required=True, help="Path to workspace_v4")
    ap.add_argument("--scenario_csv", type=str, required=True, help="Path to scenario_results_v4.csv")
    ap.add_argument("--output_dir", type=str, default=None, help="Output folder")
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    scenario_csv = Path(args.scenario_csv).resolve()
    outdir = Path(args.output_dir).resolve() if args.output_dir else workspace / "reports" / "scenario_storyline"
    _ensure_dir(outdir)

    df = pd.read_csv(scenario_csv)
    plot_regime_storyline(df, outdir / "regime_storyline_stacked.png")
    plot_portfolio_response_heatmap(df, outdir / "portfolio_response_heatmap.png")
    write_story_summary(df, outdir / "story_summary.csv")

    fm = _read_feature_master(workspace)
    if fm is None:
        print(f"feature_master_monthly.parquet not found under {workspace / 'data_refs'}")
        print("Created regime + portfolio plots only.")
        return

    monthly_macro = _macro_monthly_panel(fm)
    plot_macro_shift_heatmap(df, monthly_macro, outdir / "macro_shift_heatmap_z.png")
    plot_prior_vs_generated_pages(df, monthly_macro, outdir / "prior_vs_generated")
    print(f"Saved storyline plots to: {outdir}")


if __name__ == "__main__":
    main()
