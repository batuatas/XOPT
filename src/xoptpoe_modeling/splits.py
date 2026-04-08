"""Filtering and reproducible time-series split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SPLIT_ORDER: tuple[str, str, str] = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    """Deterministic split configuration for panel-month blocks."""

    validation_months: int = 24
    test_months: int = 24
    min_train_months: int = 60


def filter_modeling_panel(modeling_panel: pd.DataFrame) -> pd.DataFrame:
    """Apply locked first-pass filtering for model-ready rows."""
    required = {
        "sleeve_id",
        "month_end",
        "sample_inclusion_flag",
        "target_quality_flag",
        "excess_ret_fwd_1m",
    }
    missing = required - set(modeling_panel.columns)
    if missing:
        raise ValueError(f"modeling_panel missing required columns: {sorted(missing)}")

    work = modeling_panel.copy()
    work["month_end"] = pd.to_datetime(work["month_end"])

    dup_cnt = int(work.duplicated(subset=["sleeve_id", "month_end"]).sum())
    if dup_cnt > 0:
        raise ValueError(f"modeling_panel has duplicate (sleeve_id, month_end) keys: {dup_cnt}")

    mask = (
        work["sample_inclusion_flag"].eq(1)
        & work["target_quality_flag"].astype(bool)
        & work["excess_ret_fwd_1m"].notna()
    )
    filtered = work.loc[mask].copy()

    if filtered.empty:
        raise ValueError("Filtering removed all rows; no model-ready sample remains")

    filtered = filtered.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)
    return filtered


def assign_time_splits(filtered_panel: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Assign contiguous month-block train/validation/test labels."""
    if "month_end" not in filtered_panel.columns:
        raise ValueError("filtered_panel must contain month_end")

    work = filtered_panel.copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    months = pd.Index(sorted(work["month_end"].dropna().unique()))
    month_count = len(months)
    if month_count == 0:
        raise ValueError("No months available for split assignment")

    holdout_months = config.validation_months + config.test_months
    if month_count <= holdout_months:
        raise ValueError(
            f"Not enough months ({month_count}) for validation+test holdout ({holdout_months})"
        )

    train_months = month_count - holdout_months
    if train_months < config.min_train_months:
        raise ValueError(
            "Training span too short for default split config: "
            f"train_months={train_months}, min_train_months={config.min_train_months}"
        )

    train_idx_end = train_months
    val_idx_end = train_idx_end + config.validation_months

    train_set = set(months[:train_idx_end])
    val_set = set(months[train_idx_end:val_idx_end])
    test_set = set(months[val_idx_end:])

    split_by_month: dict[pd.Timestamp, str] = {}
    for month in months:
        if month in train_set:
            split_by_month[month] = "train"
        elif month in val_set:
            split_by_month[month] = "validation"
        else:
            split_by_month[month] = "test"

    work["split"] = work["month_end"].map(split_by_month)
    if work["split"].isna().any():
        raise ValueError("Split assignment failed for one or more rows")

    work["split"] = pd.Categorical(work["split"], categories=list(SPLIT_ORDER), ordered=True)
    return work.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)


def split_subsets(panel_with_splits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return split-specific dataframes keyed by split name."""
    if "split" not in panel_with_splits.columns:
        raise ValueError("panel_with_splits must contain split column")

    out: dict[str, pd.DataFrame] = {}
    for split_name in SPLIT_ORDER:
        subset = panel_with_splits.loc[panel_with_splits["split"] == split_name].copy()
        subset = subset.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)
        out[split_name] = subset
    return out


def build_split_manifest(panel_with_splits: pd.DataFrame) -> pd.DataFrame:
    """Summarize split date ranges and row counts."""
    if "split" not in panel_with_splits.columns:
        raise ValueError("panel_with_splits must contain split column")

    records: list[dict] = []
    for split_name in SPLIT_ORDER:
        chunk = panel_with_splits.loc[panel_with_splits["split"] == split_name]
        if chunk.empty:
            continue
        records.append(
            {
                "split": split_name,
                "start_month_end": pd.to_datetime(chunk["month_end"]).min().date().isoformat(),
                "end_month_end": pd.to_datetime(chunk["month_end"]).max().date().isoformat(),
                "month_count": int(chunk["month_end"].nunique()),
                "row_count": int(len(chunk)),
                "sleeve_count": int(chunk["sleeve_id"].nunique()),
            }
        )
    return pd.DataFrame(records)


def build_split_summary(panel_with_splits: pd.DataFrame) -> pd.DataFrame:
    """Build split and split-by-sleeve row/month coverage summary."""
    if "split" not in panel_with_splits.columns:
        raise ValueError("panel_with_splits must contain split column")

    base = (
        panel_with_splits.groupby(["split"], as_index=False, observed=False)
        .agg(
            start_month_end=("month_end", "min"),
            end_month_end=("month_end", "max"),
            month_count=("month_end", "nunique"),
            row_count=("month_end", "size"),
        )
        .assign(sleeve_id="ALL")
    )

    by_sleeve = (
        panel_with_splits.groupby(["split", "sleeve_id"], as_index=False, observed=False)
        .agg(
            start_month_end=("month_end", "min"),
            end_month_end=("month_end", "max"),
            month_count=("month_end", "nunique"),
            row_count=("month_end", "size"),
        )
    )

    summary = pd.concat([base, by_sleeve], ignore_index=True, sort=False)
    summary["split"] = pd.Categorical(summary["split"], categories=list(SPLIT_ORDER), ordered=True)
    summary["start_month_end"] = pd.to_datetime(summary["start_month_end"]).dt.date.astype(str)
    summary["end_month_end"] = pd.to_datetime(summary["end_month_end"]).dt.date.astype(str)
    return summary.sort_values(["split", "sleeve_id"]).reset_index(drop=True)
