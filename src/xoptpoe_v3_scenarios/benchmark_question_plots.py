"""Conference-friendly plots for the one-benchmark regime question layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_regime_question_response(summary: pd.DataFrame, out_base: Path) -> None:
    plot_df = summary.loc[summary["recommended_for_conference"].eq(1)].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("avg_delta_return", ascending=True)
    y = range(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    axes[0].barh(y, plot_df["avg_delta_return"] * 100, color="#2c5d8a")
    axes[1].barh(y, plot_df["avg_delta_max_weight"] * 100, color="#b36a3c")
    axes[0].axvline(0.0, color="#c9d2dc", lw=1.0)
    axes[1].axvline(0.0, color="#c9d2dc", lw=1.0)
    axes[0].set_xlabel("Avg predicted return change (pp)")
    axes[1].set_xlabel("Avg max-weight change (pp)")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(plot_df["short_label"])
    axes[0].grid(alpha=0.15, axis="x")
    axes[1].grid(alpha=0.15, axis="x")
    _save(fig, out_base)


def plot_regime_map(results: pd.DataFrame, selected_questions: pd.DataFrame, out_base: Path) -> None:
    keep_ids = selected_questions["question_id"].tolist()
    plot_df = results.loc[results["question_id"].isin(keep_ids), ["anchor_date", "question_id", "short_label", "scenario_regime_label", "nfci_bucket", "recession_overlay"]].copy()
    if plot_df.empty:
        return
    label_map = {qid: label for qid, label in selected_questions.set_index("question_id")["short_label"].items()}
    short_regime = {
        "soft landing": "Soft landing",
        "risk-on reflation": "Reflation",
        "high-stress defensive": "Defensive",
        "higher-for-longer tightness": "Higher-for-longer",
        "disinflationary slowdown": "Slowdown",
        "recessionary stress": "Recession stress",
        "risk-on growth": "Growth",
        "mixed mid-cycle": "Mid-cycle",
    }
    color_map = {
        "soft landing": "#4e8b5f",
        "risk-on reflation": "#3d7a4f",
        "high-stress defensive": "#7a4f7d",
        "higher-for-longer tightness": "#b36a3c",
        "disinflationary slowdown": "#6a8f9f",
        "recessionary stress": "#5f6370",
        "risk-on growth": "#457b9d",
        "mixed mid-cycle": "#96a4b3",
    }
    plot_df["question_label"] = plot_df["question_id"].map(label_map).fillna(plot_df["short_label"])
    plot_df["anchor_label"] = pd.to_datetime(plot_df["anchor_date"]).dt.strftime("%Y")
    question_order = list(selected_questions["short_label"])
    anchor_order = sorted(plot_df["anchor_label"].unique())
    y_map = {label: idx for idx, label in enumerate(question_order)}
    x_map = {label: idx for idx, label in enumerate(anchor_order)}
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for _, row in plot_df.iterrows():
        x = x_map[row["anchor_label"]]
        y = y_map[row["question_label"]]
        color = color_map.get(str(row["scenario_regime_label"]), "#96a4b3")
        rect = plt.Rectangle((x - 0.45, y - 0.42), 0.9, 0.84, facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        text = f"{short_regime.get(str(row['scenario_regime_label']), str(row['scenario_regime_label']))}\n{row['nfci_bucket']}"
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color="white")
    ax.set_xlim(-0.5, len(anchor_order) - 0.5)
    ax.set_ylim(-0.5, len(question_order) - 0.5)
    ax.set_xticks(range(len(anchor_order)))
    ax.set_xticklabels(anchor_order)
    ax.set_yticks(range(len(question_order)))
    ax.set_yticklabels(question_order)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.grid(False)
    _save(fig, out_base)


def plot_allocation_tilt_summary(summary: pd.DataFrame, out_base: Path) -> None:
    keep = summary.loc[
        summary["question_family"].eq("allocation-tilt question")
        & summary["recommended_for_conference"].eq(1)
    ].copy()
    if keep.empty:
        return
    metric_map = {
        "q_gold_tilt": ("Gold", "avg_delta_gold_weight"),
        "q_us_equity_tilt": ("US Eq", "avg_delta_eq_us_weight"),
        "q_em_equity_tilt": ("EM Eq", "avg_delta_eq_em_weight"),
    }
    rows = []
    for _, row in keep.iterrows():
        label, weight_col = metric_map.get(str(row["question_id"]), (str(row["short_label"]), "avg_delta_eq_us_weight"))
        rows.append(
            {
                "label": label,
                "delta_weight": float(row[weight_col]),
                "delta_return": float(row["avg_delta_return"]),
                "delta_max_weight": float(row["avg_delta_max_weight"]),
            }
        )
    plot_df = pd.DataFrame(rows)
    x = range(len(plot_df))
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.bar([i - 0.22 for i in x], plot_df["delta_weight"] * 100, width=0.22, color="#2c5d8a", label="Target sleeve weight change (pp)")
    ax.bar(x, plot_df["delta_return"] * 100, width=0.22, color="#6a8f9f", label="Predicted return change (pp)")
    ax.bar([i + 0.22 for i in x], plot_df["delta_max_weight"] * 100, width=0.22, color="#b36a3c", label="Max-weight change (pp)")
    ax.axhline(0.0, color="#c9d2dc", lw=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["label"])
    ax.grid(alpha=0.15, axis="y")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_base)


def plot_house_view_ladder(results: pd.DataFrame, out_base: Path) -> None:
    keep = results.loc[results["question_id"].isin(["q_house_view_6", "q_house_view_7", "q_target_10"])].copy()
    if keep.empty:
        return
    agg = keep.groupby(["question_id", "short_label"], as_index=False).agg(
        baseline_return=("baseline_predicted_return", "mean"),
        scenario_return=("scenario_predicted_return", "mean"),
        scenario_max_weight=("scenario_max_weight", "mean"),
    )
    label_order = ["6% house view", "7% house view", "10% long-run return"]
    agg["short_label"] = pd.Categorical(agg["short_label"], categories=label_order, ordered=True)
    agg = agg.sort_values("short_label")
    y = range(len(agg))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), sharey=True)
    axes[0].hlines(y, agg["baseline_return"] * 100, agg["scenario_return"] * 100, color="#d5dde5", lw=2.2)
    axes[0].scatter(agg["baseline_return"] * 100, y, s=120, facecolors="white", edgecolors="#96a4b3", linewidth=1.5)
    axes[0].scatter(agg["scenario_return"] * 100, y, s=120, color="#2c5d8a")
    axes[0].set_xlabel("Predicted return (%)")
    axes[0].grid(alpha=0.15, axis="x")
    axes[1].barh(y, agg["scenario_max_weight"] * 100, color="#b36a3c")
    axes[1].set_xlabel("Scenario max weight (%)")
    axes[1].grid(alpha=0.15, axis="x")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(agg["short_label"].astype(str))
    _save(fig, out_base)
