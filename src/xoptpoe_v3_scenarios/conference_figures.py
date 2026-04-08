"""Conference-facing scenario figures for the active v3 branch."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from xoptpoe_v3_plots.style import (
    CHINA_COLOR,
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


RAW_COLOR = NEUTRAL_DARK
DECONC_COLOR = "#6b8f71"


def _pct_fmt(x: float, pos: int | None = None) -> str:
    return f"{100 * x:.1f}%"


def _bps_fmt(x: float, pos: int | None = None) -> str:
    return f"{10000 * x:.0f}"


def _case_color(case_id: str) -> str:
    if case_id == "robust_return_up":
        return FOCUS_COLOR
    if case_id == "raw_return_up":
        return RAW_COLOR
    if case_id == "raw_deconcentration":
        return DECONC_COLOR
    if case_id.startswith("disagreement"):
        return NEUTRAL_MID
    return CHINA_COLOR


def plot_scenario_story_compact(casebook_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    order = ["robust_return_up", "raw_return_up", "raw_deconcentration"]
    plot_df = casebook_df.set_index("case_id").loc[order].reset_index()
    plot_df["display"] = [
        "Robust 60m\nupside",
        "Raw 120m\nupside",
        "120m\ndeconcentration",
    ]
    y = np.arange(len(plot_df))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    specs = [
        ("baseline_predicted_return", "scenario_predicted_return", "Predicted return"),
        ("baseline_max_weight", "scenario_max_weight", "Max sleeve weight"),
    ]
    for ax, (base_col, scen_col, xlabel) in zip(axes, specs, strict=True):
        ax.hlines(y, plot_df[base_col], plot_df[scen_col], color=NEUTRAL_LIGHT, linewidth=2.0, zorder=1)
        ax.scatter(plot_df[base_col], y, s=90, color="white", edgecolor=NEUTRAL_MID, linewidth=1.3, zorder=3)
        for yi, (_, row) in zip(y, plot_df.iterrows(), strict=True):
            ax.scatter(row[scen_col], yi, s=110, color=_case_color(str(row["case_id"])), edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_yticks(y, plot_df["display"])
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xlabel)
        if "return" in base_col:
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
        else:
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    axes[1].tick_params(axis="y", labelleft=False)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor=NEUTRAL_MID, markeredgewidth=1.2, markersize=8, label="Baseline"),
        plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FOCUS_COLOR, markeredgecolor="white", markeredgewidth=0.8, markersize=8, label="Scenario"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=1.4, w_pad=2.2)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)


def plot_scenario_case_grid(casebook_df: pd.DataFrame, case_shift_df: pd.DataFrame, contrast_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    case_order = ["robust_return_up", "raw_return_up", "raw_deconcentration", "disagreement_case_120"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6))
    axes = axes.flatten()

    for ax, case_id in zip(axes, case_order, strict=True):
        case_row = casebook_df.loc[casebook_df["case_id"].eq(case_id)].iloc[0]
        shifts = case_shift_df.loc[case_shift_df["case_id"].eq(case_id)].sort_values("shift_rank")
        labels = list(reversed(shifts["variable_name"].map(str).tolist()))
        values = list(reversed(shifts["shift_std_units_signed"].astype(float).tolist()))
        colors = [FOCUS_COLOR if v >= 0 else NEUTRAL_MID for v in values]
        ypos = np.arange(len(labels))
        ax.barh(ypos, values, color=colors, edgecolor="white", linewidth=0.7)
        ax.axvline(0.0, color=NEUTRAL_MID, linewidth=0.9)
        ax.set_yticks(ypos, labels)
        ax.xaxis.set_major_formatter(lambda x, pos=None: f"{x:+.2f}")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55)
        ax.grid(axis="y", visible=False)
        ax.set_title(str(case_row["short_case_label"]), fontsize=14, loc="left", pad=8)
        summary_text = (
            f"Return {_pct_fmt(float(case_row['scenario_predicted_return'] - case_row['baseline_predicted_return']))}\n"
            f"Max wt {_pct_fmt(float(case_row['baseline_max_weight']))} -> {_pct_fmt(float(case_row['scenario_max_weight']))}\n"
            f"Eff N {case_row['baseline_effective_n']:.2f} -> {case_row['scenario_effective_n']:.2f}"
        )
        if case_id == "disagreement_case_120":
            contrast_row = contrast_df.loc[contrast_df["anchor_date"].eq(case_row["anchor_date"])].iloc[0]
            summary_text = (
                f"60m max wt {_pct_fmt(float(contrast_row['baseline_60_max_weight']))} -> {_pct_fmt(float(contrast_row['scenario_60_max_weight']))}\n"
                f"120m max wt {_pct_fmt(float(contrast_row['baseline_120_max_weight']))} -> {_pct_fmt(float(contrast_row['scenario_120_max_weight']))}"
            )
        ax.text(
            0.98,
            0.04,
            summary_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.2,
            color=NEUTRAL_DARK,
            bbox={"facecolor": "white", "edgecolor": NEUTRAL_LIGHT, "boxstyle": "round,pad=0.35"},
        )
    fig.tight_layout(pad=1.4, h_pad=1.6, w_pad=1.8)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)


def plot_robust_vs_raw_response(response_cloud_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    plot_df = response_cloud_df.loc[response_cloud_df["candidate_name"].isin(["best_60_predictor", "best_120_predictor"])].copy()
    plot_df["display"] = plot_df["candidate_name"].map({
        "best_60_predictor": "Robust 60m",
        "best_120_predictor": "Raw 120m",
    })
    plot_df["x"] = plot_df["display"].map({"Robust 60m": 0.0, "Raw 120m": 1.0}).astype(float)

    rng = np.random.default_rng(42)
    plot_df["jitter"] = rng.uniform(-0.09, 0.09, size=len(plot_df))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, ycol, ylabel in [
        (axes[0], "portfolio_return_change", "Scenario return change (bps)"),
        (axes[1], "scenario_max_weight", "Scenario max sleeve weight"),
    ]:
        for candidate_name, color in [("best_60_predictor", FOCUS_COLOR), ("best_120_predictor", RAW_COLOR)]:
            sub = plot_df.loc[plot_df["candidate_name"].eq(candidate_name)]
            x = sub["x"] + sub["jitter"]
            y = sub[ycol]
            ax.scatter(x, y, s=55, color=color, alpha=0.85, edgecolor="white", linewidth=0.6)
            ax.scatter(sub["x"].iloc[0], y.median(), s=95, color=color, marker="D", edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_xticks([0, 1], ["Robust 60m", "Raw 120m"])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
        ax.grid(axis="x", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylabel(ylabel)
        if ycol == "portfolio_return_change":
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(_bps_fmt))
        else:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    fig.tight_layout(pad=1.4, w_pad=1.8)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)


def plot_raw_deconcentration_case(casebook_df: pd.DataFrame, weights_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    case_row = casebook_df.loc[casebook_df["case_id"].eq("raw_deconcentration")].iloc[0]
    plot_df = weights_df.loc[weights_df["case_id"].eq("raw_deconcentration")].copy()
    ordered = plot_df.loc[plot_df["case_phase"].eq("baseline")].sort_values("weight", ascending=False)["sleeve_id"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), sharey=True)
    for ax, phase, title in zip(axes, ["baseline", "scenario"], ["Baseline", "Scenario"], strict=True):
        sub = plot_df.loc[plot_df["case_phase"].eq(phase)].set_index("sleeve_id").loc[ordered].reset_index()
        y = np.arange(len(sub))
        ax.barh(y, sub["weight"], color=[sleeve_color(s) for s in sub["sleeve_id"]], edgecolor="white", linewidth=0.7)
        ax.set_yticks(y, [sleeve_label(s) for s in sub["sleeve_id"]])
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(title)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.text(
        0.5,
        0.02,
        f"Predicted return {_pct_fmt(float(case_row['baseline_predicted_return']))} -> {_pct_fmt(float(case_row['scenario_predicted_return']))} | Effective N {case_row['baseline_effective_n']:.2f} -> {case_row['scenario_effective_n']:.2f}",
        ha="center",
        va="bottom",
        fontsize=11.2,
        color=NEUTRAL_DARK,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1], pad=1.4, w_pad=2.2)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)


def plot_china_under_scenarios_compact(china_df: pd.DataFrame, out_base: Path) -> None:
    apply_conference_style()
    plot_df = china_df.copy()
    plot_df["anchor_label"] = pd.to_datetime(plot_df["anchor_date"]).dt.strftime("%Y")
    y = np.arange(len(plot_df))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    axes[0].hlines(y, plot_df["eq_cn_weight_before"], plot_df["eq_cn_weight_after"], color=NEUTRAL_LIGHT, linewidth=2.0)
    axes[0].scatter(plot_df["eq_cn_weight_before"], y, s=55, color="white", edgecolor=NEUTRAL_MID, linewidth=1.0, zorder=3)
    axes[0].scatter(plot_df["eq_cn_weight_after"], y, s=65, color=CHINA_COLOR, edgecolor="white", linewidth=0.7, zorder=4)
    axes[0].axvline(0.05, color=NEUTRAL_MID, linewidth=1.0, linestyle=(0, (4, 4)))
    axes[0].set_yticks(y, plot_df["anchor_label"])
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    axes[0].grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    axes[0].grid(axis="y", visible=False)
    axes[0].set_xlabel("EQ_CN weight")

    axes[1].hlines(y, plot_df["eq_cn_rank_before"], plot_df["eq_cn_rank_after"], color=NEUTRAL_LIGHT, linewidth=2.0)
    axes[1].scatter(plot_df["eq_cn_rank_before"], y, s=55, color="white", edgecolor=NEUTRAL_MID, linewidth=1.0, zorder=3)
    axes[1].scatter(plot_df["eq_cn_rank_after"], y, s=65, color=CHINA_COLOR, edgecolor="white", linewidth=0.7, zorder=4)
    axes[1].set_xlabel("EQ_CN sleeve rank")
    axes[1].grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    axes[1].grid(axis="y", visible=False)
    axes[1].tick_params(axis="y", labelleft=False)

    fig.tight_layout(pad=1.4, w_pad=2.2)
    save_figure(fig, out_dir=out_base.parent, stem=out_base.stem)
