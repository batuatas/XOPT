"""Small reporting helpers for v3 scenario experiments."""

from __future__ import annotations

import pandas as pd


def strongest_findings_markdown(
    summary_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    china_df: pd.DataFrame,
) -> str:
    """Build a compact markdown findings block from experiment tables."""
    top_up_60 = summary_df.loc[summary_df["probe_id"].eq("probe_60_target_up")].sort_values("scenario_response_best", ascending=False).head(1)
    top_up_120 = summary_df.loc[summary_df["probe_id"].eq("probe_120_target_up")].sort_values("scenario_response_best", ascending=False).head(1)
    deconc = portfolio_df.loc[portfolio_df["probe_id"].eq("probe_120_deconcentration")].sort_values("hhi_change").head(1)
    china_candidates = china_df.loc[china_df["probe_id"].eq("probe_60_china_role")].copy()
    if not china_candidates.empty:
        china_candidates["eq_cn_weight_change"] = (
            china_candidates["eq_cn_weight_after"] - china_candidates["eq_cn_weight_before"]
        )
    china = china_candidates.sort_values("eq_cn_weight_change", ascending=False).head(1)

    lines = ["## Compact Findings"]
    if not top_up_60.empty:
        row = top_up_60.iloc[0]
        lines.append(
            f"- Most favorable 60m return-up case: {pd.Timestamp(row['anchor_date']).date()} with scenario_response_best={row['scenario_response_best']:.4f} vs baseline={row['baseline_response']:.4f}."
        )
    if not top_up_120.empty:
        row = top_up_120.iloc[0]
        lines.append(
            f"- Most favorable 120m return-up case: {pd.Timestamp(row['anchor_date']).date()} with scenario_response_best={row['scenario_response_best']:.4f} vs baseline={row['baseline_response']:.4f}."
        )
    if not deconc.empty:
        row = deconc.iloc[0]
        lines.append(
            f"- Largest 120m deconcentration move: {pd.Timestamp(row['anchor_date']).date()} with hhi_change={row['hhi_change']:.4f} and predicted_return_change={row['predicted_return_change']:.4f}."
        )
    if not china.empty:
        row = china.iloc[0]
        if row["eq_cn_weight_change"] > 0:
            lines.append(
                f"- Largest EQ_CN role increase in the robust benchmark: {pd.Timestamp(row['anchor_date']).date()} with weight {row['eq_cn_weight_before']:.4f} -> {row['eq_cn_weight_after']:.4f}."
            )
        else:
            lines.append(
                f"- EQ_CN remained marginal in the robust benchmark across first-pass scenarios; the least-negative case was {pd.Timestamp(row['anchor_date']).date()} with weight {row['eq_cn_weight_before']:.4f} -> {row['eq_cn_weight_after']:.4f}."
            )
    return "\n".join(lines)
