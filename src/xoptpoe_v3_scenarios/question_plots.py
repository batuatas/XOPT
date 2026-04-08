"""Conference-friendly plots for the v3 scenario question layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_question_regime_map(regime_summary: pd.DataFrame, selected_questions: pd.DataFrame, out_base: Path) -> None:
    keep_ids = selected_questions["question_id"].tolist()
    subset = regime_summary.loc[regime_summary["question_id"].isin(keep_ids)].copy()
    if subset.empty:
        return
    palette = {
        "soft landing": "#2c5d8a",
        "risk-on reflation": "#4e8b5f",
        "higher-for-longer tightness": "#b36a3c",
        "high-stress defensive": "#7a4f7d",
        "mixed mid-cycle": "#8d99a6",
        "disinflationary slowdown": "#6a8f9f",
        "stagflation-like": "#aa5a5a",
        "stagflation-like stress": "#8c4d4d",
        "risk-off stress": "#6b7280",
        "risk-on growth": "#457b9d",
    }
    label_map = {
        "q_robust_double_digit": "Robust upside",
        "q_raw_ceiling_upside": "Raw upside",
        "q_raw_deconcentration": "Raw breadth",
        "q_gold_tilt": "Gold tilt",
        "q_robust_raw_disagreement": "60m vs 120m gap",
        "q_em_tilt": "EM tilt",
    }
    short_regime = {
        "risk-on reflation": "Reflation",
        "risk-off stress": "Stress",
        "mixed mid-cycle": "Mid-cycle",
        "soft landing": "Soft landing",
        "higher-for-longer tightness": "Higher-for-longer",
        "high-stress defensive": "Defensive",
        "disinflationary slowdown": "Disinflation",
        "stagflation-like": "Stagflation",
        "stagflation-like stress": "Stagflation",
        "risk-on growth": "Growth",
    }
    plot_df = subset.copy()
    plot_df["display_label"] = plot_df["question_id"].map(label_map).fillna(plot_df["short_label"])
    plot_df["anchor_label"] = pd.to_datetime(plot_df["anchor_date"]).dt.strftime("%Y")
    question_order = [label_map[qid] for qid in keep_ids if qid in label_map]
    anchor_order = sorted(plot_df["anchor_label"].unique())
    y_map = {label: idx for idx, label in enumerate(question_order)}
    x_map = {label: idx for idx, label in enumerate(anchor_order)}
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for _, row in plot_df.iterrows():
        x = x_map[row["anchor_label"]]
        y = y_map[row["display_label"]]
        color = palette.get(str(row["scenario_regime_label"]), "#8d99a6")
        rect = plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, short_regime.get(str(row["scenario_regime_label"]), str(row["scenario_regime_label"])), ha="center", va="center", fontsize=9, color="white")
    ax.set_xlim(-0.5, len(anchor_order) - 0.5)
    ax.set_ylim(-0.5, len(question_order) - 0.5)
    ax.set_xticks(range(len(anchor_order)))
    ax.set_xticklabels(anchor_order)
    ax.set_yticks(range(len(question_order)))
    ax.set_yticklabels(question_order)
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(length=0)
    _save(fig, out_base)


def plot_benchmark_question_response(question_results: pd.DataFrame, out_base: Path) -> None:
    keep = ["q_robust_double_digit", "q_raw_ceiling_upside", "q_raw_deconcentration", "q_robust_raw_disagreement"]
    subset = question_results.loc[question_results["question_id"].isin(keep)].copy()
    if subset.empty:
        return
    agg = subset.groupby(["question_id", "short_label"], as_index=False).agg(
        delta_predicted_return=("delta_predicted_return", "mean"),
        delta_max_weight=("delta_max_weight", "mean"),
    )
    label_map = {
        "q_robust_double_digit": "Robust upside",
        "q_raw_ceiling_upside": "Raw upside",
        "q_raw_deconcentration": "Raw breadth",
        "q_robust_raw_disagreement": "60m vs 120m gap",
    }
    agg["display_label"] = agg["question_id"].map(label_map).fillna(agg["short_label"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
    y = np.arange(len(agg))
    axes[0].barh(y, agg["delta_predicted_return"] * 100, color="#2c5d8a", alpha=0.9)
    axes[1].barh(y, agg["delta_max_weight"] * 100, color="#b36a3c", alpha=0.9)
    axes[0].axvline(0.0, color="#c9d2dc", lw=1.0)
    axes[1].axvline(0.0, color="#c9d2dc", lw=1.0)
    axes[0].set_xlabel("Predicted return change (pp)")
    axes[1].set_xlabel("Max-weight change (pp)")
    axes[0].grid(alpha=0.15, axis="x")
    axes[1].grid(alpha=0.15, axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(agg["display_label"])
    _save(fig, out_base)


def plot_allocation_tilt_questions(question_results: pd.DataFrame, out_base: Path) -> None:
    keep = [
        ("q_gold_tilt", "ALT_GLD", "Gold"),
        ("q_us_equity_tilt", "EQ_US", "US Eq"),
        ("q_em_tilt", "EQ_EM", "EM Eq"),
        ("q_china_materiality", "EQ_CN", "China Eq"),
    ]
    records = []
    for qid, sleeve_id, target in keep:
        subset = question_results.loc[question_results["question_id"].eq(qid)].copy()
        if subset.empty:
            continue
        records.append(
            {
                "short_label": subset["short_label"].iloc[0],
                "target_sleeve": target,
                "delta_target_weight": float((subset[f"weight_after_{sleeve_id}"] - subset[f"weight_before_{sleeve_id}"]).mean()),
                "delta_return": float(subset["delta_predicted_return"].mean()),
                "delta_max_weight": float(subset["delta_max_weight"].mean()),
            }
        )
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(plot_df))
    ax.bar(x - 0.22, plot_df["delta_target_weight"] * 100, width=0.22, color="#2c5d8a", label="Target sleeve weight change (pp)")
    ax.bar(x, plot_df["delta_return"] * 100, width=0.22, color="#6a8f9f", label="Predicted return change (pp)")
    ax.bar(x + 0.22, plot_df["delta_max_weight"] * 100, width=0.22, color="#b36a3c", label="Max-weight change (pp)")
    ax.axhline(0.0, color="#c9d2dc", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["target_sleeve"], rotation=0)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    ax.grid(alpha=0.15, axis="y")
    _save(fig, out_base)
