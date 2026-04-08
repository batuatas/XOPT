"""Case-based conference figures for the locked robust 5Y benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from xoptpoe_v3_plots.style import (
    FOCUS_COLOR,
    NEUTRAL_DARK,
    NEUTRAL_LIGHT,
    NEUTRAL_MID,
    GRID_COLOR,
    apply_conference_style,
    save_figure,
    sleeve_color,
    sleeve_label,
)


CASE_COLORS = {
    "upside": FOCUS_COLOR,
    "breadth": "#6b8f71",
    "adverse": "#b36a3c",
}


def _pct_fmt(x: float, pos: int | None = None) -> str:
    return f"{100 * x:.1f}%"


def _pp_fmt(x: float, pos: int | None = None) -> str:
    return f"{100 * x:+.2f} pp"


def plot_case_benchmark_story(case_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    plot_df = case_df.copy()
    y = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.1), sharey=True)
    specs = [
        ("baseline_predicted_return", "scenario_predicted_return", "Predicted return"),
        ("baseline_max_weight", "scenario_max_weight", "Max sleeve weight"),
        ("baseline_effective_n", "scenario_effective_n", "Effective N"),
    ]
    for ax, (base_col, scen_col, xlabel) in zip(axes, specs, strict=True):
        ax.hlines(y, plot_df[base_col], plot_df[scen_col], color=NEUTRAL_LIGHT, linewidth=2.2, zorder=1)
        ax.scatter(plot_df[base_col], y, s=90, color="white", edgecolor=NEUTRAL_MID, linewidth=1.2, zorder=3)
        for yi, (_, row) in zip(y, plot_df.iterrows(), strict=True):
            ax.scatter(
                row[scen_col],
                yi,
                s=110,
                color=CASE_COLORS[str(row["case_role"])],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        ax.set_yticks(y, plot_df["case_label"])
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xlabel)
        if "return" in base_col or "weight" in base_col:
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    axes[2].tick_params(axis="y", labelleft=False)
    fig.tight_layout(pad=1.4, w_pad=1.8)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)


def plot_case_allocation_change(case_df: pd.DataFrame, weights_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    keep_ids = ["upside_soft_landing", "breadth_return_with_breadth"]
    case_lookup = case_df.set_index("case_id")["case_label"].to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), sharey=False)
    for ax, case_id in zip(axes, keep_ids, strict=True):
        plot_df = weights_df.loc[weights_df["case_id"].eq(case_id)].copy()
        baseline = plot_df.loc[plot_df["case_phase"].eq("baseline")].copy()
        scenario = plot_df.loc[plot_df["case_phase"].eq("scenario")].copy()
        merged = baseline.merge(
            scenario[["sleeve_id", "weight"]],
            on="sleeve_id",
            suffixes=("_baseline", "_scenario"),
            validate="1:1",
        )
        merged["delta_weight"] = merged["weight_scenario"] - merged["weight_baseline"]
        merged = merged.sort_values("delta_weight", ascending=False).reset_index(drop=True)
        y = np.arange(len(merged))
        ax.hlines(y, merged["weight_baseline"], merged["weight_scenario"], color=NEUTRAL_LIGHT, linewidth=2.1, zorder=1)
        ax.scatter(merged["weight_baseline"], y, s=55, color="white", edgecolor=NEUTRAL_MID, linewidth=1.0, zorder=3)
        ax.scatter(
            merged["weight_scenario"],
            y,
            s=70,
            color=[sleeve_color(s) for s in merged["sleeve_id"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.axvline(0.0, color=NEUTRAL_LIGHT, linewidth=0.8)
        ax.set_yticks(y, [sleeve_label(s) for s in merged["sleeve_id"]])
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(case_lookup[case_id])
    fig.tight_layout(pad=1.4, w_pad=2.0)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)


def plot_case_macro_fingerprint(case_df: pd.DataFrame, fingerprint_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    case_order = case_df["case_id"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.8), sharex=False)
    for ax, case_id in zip(axes, case_order, strict=True):
        sub = fingerprint_df.loc[fingerprint_df["case_id"].eq(case_id)].sort_values("shift_rank", ascending=False)
        labels = sub["variable_name"].tolist()
        values = sub["shift_std_units_signed"].tolist()
        colors = [CASE_COLORS[case_df.set_index("case_id").loc[case_id, "case_role"]] if v >= 0 else NEUTRAL_MID for v in values]
        y = np.arange(len(sub))
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
        ax.axvline(0.0, color=NEUTRAL_MID, linewidth=0.9)
        ax.set_yticks(y, labels)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos=None: f"{x:+.2f}"))
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(case_df.set_index("case_id").loc[case_id, "case_label"])
    fig.tight_layout(pad=1.4, w_pad=2.2)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)


def plot_case_comparison(case_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    fig, ax = plt.subplots(figsize=(12.8, 3.8))
    ax.axis("off")
    x_positions = [0.03, 0.35, 0.67]
    for x, (_, row) in zip(x_positions, case_df.iterrows(), strict=True):
        color = CASE_COLORS[str(row["case_role"])]
        rect = plt.Rectangle((x, 0.12), 0.28, 0.74, facecolor="white", edgecolor=NEUTRAL_LIGHT, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.02, 0.78, str(row["case_label"]), fontsize=15, weight="bold", color=color, transform=ax.transAxes)
        ax.text(x + 0.02, 0.66, f"{pd.Timestamp(row['anchor_date']).strftime('%Y-%m-%d')} | {row['regime_label']}", fontsize=10.8, color=NEUTRAL_DARK, transform=ax.transAxes)
        ax.text(
            x + 0.02,
            0.50,
            f"Return {_pp_fmt(float(row['delta_predicted_return']))}\nMax wt {_pp_fmt(float(row['delta_max_weight']))}\nEff N {float(row['delta_effective_n']):+.2f}",
            fontsize=11.3,
            color=NEUTRAL_DARK,
            transform=ax.transAxes,
            va="top",
        )
        ax.text(x + 0.02, 0.18, str(row["short_interpretation"]), fontsize=10.6, color=NEUTRAL_DARK, transform=ax.transAxes, va="bottom", wrap=True)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)

