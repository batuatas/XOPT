"""EDA summaries for first-pass modeling preparation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


TARGET_COL = "excess_ret_fwd_1m"

IMPORTANT_FEATURES: tuple[str, ...] = (
    "ret_1m_lag",
    "mom_12_1",
    "vol_12m",
    "infl_US",
    "infl_EA",
    "infl_JP",
    "short_rate_US",
    "short_rate_EA",
    "short_rate_JP",
    "usd_broad",
    "vix",
    "us_real10y",
    "ig_oas",
    "oil_wti",
)


def infer_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Infer numeric feature columns excluding identifiers/labels."""
    excluded = {
        "sleeve_id",
        "month_end",
        "split",
        "ret_fwd_1m",
        "rf_fwd_1m",
        TARGET_COL,
        "sample_inclusion_flag",
        "target_quality_flag",
        "feature_complete_flag",
        "proxy_flag",
    }
    numeric = list(panel.select_dtypes(include=[np.number]).columns)
    return sorted([col for col in numeric if col not in excluded])


def summarize_target(panel_with_splits: pd.DataFrame) -> pd.DataFrame:
    """Target summary by split and sleeve, with panel-level rows."""
    if TARGET_COL not in panel_with_splits.columns:
        raise ValueError(f"panel_with_splits missing {TARGET_COL}")

    records: list[dict] = []
    split_values = ["ALL", "train", "validation", "test"]

    for split_name in split_values:
        if split_name == "ALL":
            chunk = panel_with_splits.copy()
        else:
            chunk = panel_with_splits.loc[panel_with_splits["split"] == split_name].copy()
        if chunk.empty:
            continue

        for sleeve_id in ["ALL", *sorted(chunk["sleeve_id"].unique())]:
            if sleeve_id == "ALL":
                sleeve_chunk = chunk
            else:
                sleeve_chunk = chunk.loc[chunk["sleeve_id"] == sleeve_id]
            series = sleeve_chunk[TARGET_COL]
            records.append(
                {
                    "split": split_name,
                    "sleeve_id": sleeve_id,
                    "row_count": int(len(sleeve_chunk)),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)),
                    "min": float(series.min()),
                    "p05": float(series.quantile(0.05)),
                    "p50": float(series.quantile(0.50)),
                    "p95": float(series.quantile(0.95)),
                    "max": float(series.max()),
                }
            )

    out = pd.DataFrame(records)
    split_order = ["ALL", "train", "validation", "test"]
    out["split"] = pd.Categorical(out["split"], categories=split_order, ordered=True)
    return out.sort_values(["split", "sleeve_id"]).reset_index(drop=True)


def summarize_features(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute missingness and scale summary for candidate feature columns."""
    records: list[dict] = []
    for col in feature_cols:
        series = panel[col]
        records.append(
            {
                "feature_name": col,
                "dtype": str(series.dtype),
                "row_count": int(series.shape[0]),
                "non_null_count": int(series.notna().sum()),
                "missing_share": float(series.isna().mean()),
                "mean": float(series.mean(skipna=True)),
                "std": float(series.std(skipna=True, ddof=1)),
                "min": float(series.min(skipna=True)),
                "p01": float(series.quantile(0.01)),
                "p50": float(series.quantile(0.50)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max(skipna=True)),
            }
        )
    return pd.DataFrame(records).sort_values("feature_name").reset_index(drop=True)


def correlation_diagnostics(
    panel: pd.DataFrame,
    *,
    feature_subset: list[str],
    pairwise_threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute target-correlation and high-pairwise-correlation diagnostics."""
    cols = [TARGET_COL, *feature_subset]
    cols = [c for c in cols if c in panel.columns]
    corr = panel[cols].corr(numeric_only=True)

    target_corr = (
        corr[TARGET_COL]
        .drop(labels=[TARGET_COL], errors="ignore")
        .rename("corr_with_target")
        .reset_index()
        .rename(columns={"index": "feature_name"})
    )
    target_corr["abs_corr_with_target"] = target_corr["corr_with_target"].abs()
    target_corr = target_corr.sort_values("abs_corr_with_target", ascending=False).reset_index(drop=True)

    pair_rows: list[dict] = []
    usable_features = [c for c in feature_subset if c in corr.columns]
    for a, b in combinations(usable_features, 2):
        value = corr.loc[a, b]
        if pd.isna(value):
            continue
        if abs(value) >= pairwise_threshold:
            pair_rows.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "corr": float(value),
                    "abs_corr": float(abs(value)),
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty:
        pairwise = pairwise.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return target_corr, pairwise
