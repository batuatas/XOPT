"""Plotting helpers for v3 scenario experiments."""

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


def plot_state_shift_heatmap(representative_state_df: pd.DataFrame, out_base: Path) -> None:
    top_cases = representative_state_df["case_id"].drop_duplicates().tolist()[:8]
    subset = representative_state_df.loc[representative_state_df["case_id"].isin(top_cases)].copy()
    pivot = subset.pivot(index="variable_name", columns="case_id", values="shift_std_units").fillna(0.0)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdBu_r", aspect="auto", vmin=-2.0, vmax=2.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Shift (std units)")
    _save(fig, out_base)


def plot_portfolio_change(portfolio_response_df: pd.DataFrame, out_base: Path) -> None:
    keep = portfolio_response_df.sort_values("hhi_change").drop_duplicates("probe_id").head(6)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, keep.iterrows(), strict=False):
        vals = pd.Series(
            {
                "pred_return": row["predicted_return_change"],
                "max_weight": row["max_weight_change"],
                "HHI": row["hhi_change"],
                "eff_N": row["effective_n_change"],
                "EQ_CN_w": row["eq_cn_weight_change"],
            }
        )
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
        ax.barh(vals.index, vals.values, color=colors)
        ax.axvline(0.0, color="#999999", lw=0.8)
        ax.set_title(f"{pd.Timestamp(row['anchor_date']).date()} | {row['probe_id']}", fontsize=10)
    for ax in axes[len(keep):]:
        ax.axis("off")
    _save(fig, out_base)


def plot_return_vs_concentration(summary_df: pd.DataFrame, portfolio_df: pd.DataFrame, out_base: Path) -> None:
    merged = summary_df.merge(
        portfolio_df[["anchor_date", "probe_id", "candidate_name", "predicted_return_change", "hhi_change"]],
        on=["anchor_date", "probe_id", "candidate_name"],
        how="inner",
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        merged["hhi_change"],
        merged["predicted_return_change"],
        c=np.where(merged["probe_id"].str.contains("120"), "#1f77b4", "#444444"),
        alpha=0.8,
    )
    del scatter
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.axvline(0.0, color="#999999", lw=0.8)
    ax.set_xlabel("HHI change")
    ax.set_ylabel("Predicted return change")
    _save(fig, out_base)


def plot_china_role(china_df: pd.DataFrame, out_base: Path) -> None:
    subset = china_df.loc[china_df["eq_cn_weight_before"].notna()].copy()
    subset["label"] = subset["anchor_date"].astype(str).str[:10] + " | " + subset["probe_id"]
    keep = subset.sort_values("eq_cn_weight_after", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(keep))
    ax.hlines(y, keep["eq_cn_weight_before"], keep["eq_cn_weight_after"], color="#bbbbbb", lw=2)
    ax.scatter(keep["eq_cn_weight_before"], y, color="#666666", label="before", s=40)
    ax.scatter(keep["eq_cn_weight_after"], y, color="#c0392b", label="after", s=40)
    ax.set_yticks(y)
    ax.set_yticklabels(keep["label"], fontsize=9)
    ax.set_xlabel("EQ_CN portfolio weight")
    ax.legend(frameon=False)
    _save(fig, out_base)
