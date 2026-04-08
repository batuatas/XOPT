"""Time-respecting split helpers for XOPTPOE v2 long-horizon modeling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")
EXCLUDED_SPLIT_NAME = "excluded_60_only_tail"


@dataclass(frozen=True)
class SplitConfig:
    """Deterministic split configuration for the default first-pass setup."""

    validation_months: int = 24
    test_months: int = 24
    min_train_months: int = 60
    required_horizons: tuple[int, ...] = (60, 120)



def filter_firstpass_panel(modeling_panel: pd.DataFrame, *, horizons: tuple[int, ...] = (60, 120)) -> pd.DataFrame:
    """Apply the locked v2 first-pass filtering rules."""
    required = {
        "sleeve_id",
        "month_end",
        "horizon_months",
        "baseline_trainable_flag",
        "target_available_flag",
        "annualized_excess_forward_return",
    }
    missing = required - set(modeling_panel.columns)
    if missing:
        raise ValueError(f"modeling_panel_hstack missing required columns: {sorted(missing)}")

    work = modeling_panel.copy()
    work["month_end"] = pd.to_datetime(work["month_end"])

    dup_cnt = int(work.duplicated(subset=["sleeve_id", "month_end", "horizon_months"]).sum())
    if dup_cnt > 0:
        raise ValueError(
            f"modeling_panel_hstack has duplicate (sleeve_id, month_end, horizon_months) keys: {dup_cnt}"
        )

    mask = (
        work["baseline_trainable_flag"].eq(1)
        & work["target_available_flag"].eq(1)
        & work["annualized_excess_forward_return"].notna()
        & work["horizon_months"].isin(horizons)
    )
    filtered = work.loc[mask].copy()
    if filtered.empty:
        raise ValueError("Filtering removed all rows; no first-pass v2 sample remains")

    return filtered.sort_values(["month_end", "sleeve_id", "horizon_months"]).reset_index(drop=True)



def identify_common_horizon_months(
    filtered_panel: pd.DataFrame,
    *,
    required_horizons: tuple[int, ...] = (60, 120),
) -> pd.Index:
    """Identify months where all required horizons are present for the full sleeve cross-section."""
    work = filtered_panel.copy()
    work["month_end"] = pd.to_datetime(work["month_end"])

    by_month_horizon = (
        work.groupby(["month_end", "horizon_months"], as_index=False)
        .agg(row_count=("sleeve_id", "size"), sleeve_count=("sleeve_id", "nunique"))
    )
    pivot = by_month_horizon.pivot(index="month_end", columns="horizon_months", values="sleeve_count").fillna(0)
    required_present = pivot.reindex(columns=list(required_horizons), fill_value=0).gt(0).all(axis=1)
    common_months = pd.Index(sorted(pivot.index[required_present]))
    if common_months.empty:
        raise ValueError("No common multi-horizon months found for the default split design")
    return common_months



def assign_default_splits(filtered_panel: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Assign default train/validation/test splits on the common 60m/120m window."""
    work = filtered_panel.copy()
    work["month_end"] = pd.to_datetime(work["month_end"])

    common_months = identify_common_horizon_months(work, required_horizons=config.required_horizons)
    month_count = len(common_months)
    holdout_months = config.validation_months + config.test_months
    if month_count <= holdout_months:
        raise ValueError(
            f"Not enough common-horizon months ({month_count}) for validation+test holdout ({holdout_months})"
        )

    train_months = month_count - holdout_months
    if train_months < config.min_train_months:
        raise ValueError(
            "Training span too short for default v2 split config: "
            f"train_months={train_months}, min_train_months={config.min_train_months}"
        )

    train_set = set(common_months[:train_months])
    val_set = set(common_months[train_months : train_months + config.validation_months])
    test_set = set(common_months[train_months + config.validation_months :])

    def split_name(month_end: pd.Timestamp) -> str:
        if month_end in train_set:
            return "train"
        if month_end in val_set:
            return "validation"
        if month_end in test_set:
            return "test"
        return EXCLUDED_SPLIT_NAME

    work["default_split_eligible_flag"] = work["month_end"].isin(common_months).astype(int)
    work["common_horizon_window_flag"] = work["default_split_eligible_flag"]
    work["default_split"] = work["month_end"].map(split_name)

    if work.loc[work["default_split_eligible_flag"].eq(1), "default_split"].isin(SPLIT_ORDER).all() is False:
        raise ValueError("Split assignment failed on one or more split-eligible rows")

    split_categories = list(SPLIT_ORDER) + [EXCLUDED_SPLIT_NAME]
    work["default_split"] = pd.Categorical(work["default_split"], categories=split_categories, ordered=True)
    return work.sort_values(["month_end", "sleeve_id", "horizon_months"]).reset_index(drop=True)



def split_subsets(panel_with_splits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return train/validation/test tables keyed by split name."""
    out: dict[str, pd.DataFrame] = {}
    for split_name in SPLIT_ORDER:
        chunk = panel_with_splits.loc[panel_with_splits["default_split"].eq(split_name)].copy()
        out[split_name] = chunk.sort_values(["month_end", "sleeve_id", "horizon_months"]).reset_index(drop=True)
    return out



def build_split_manifest(panel_with_splits: pd.DataFrame) -> pd.DataFrame:
    """Build a compact split manifest including excluded 60m-only tail rows."""
    records: list[dict[str, object]] = []
    for split_name in list(SPLIT_ORDER) + [EXCLUDED_SPLIT_NAME]:
        chunk = panel_with_splits.loc[panel_with_splits["default_split"].eq(split_name)]
        if chunk.empty:
            continue
        records.append(
            {
                "split_name": split_name,
                "start_month_end": pd.to_datetime(chunk["month_end"]).min().date().isoformat(),
                "end_month_end": pd.to_datetime(chunk["month_end"]).max().date().isoformat(),
                "month_count": int(chunk["month_end"].nunique()),
                "row_count": int(len(chunk)),
                "sleeve_count": int(chunk["sleeve_id"].nunique()),
                "horizon_count": int(chunk["horizon_months"].nunique()),
                "horizons": ",".join(str(v) for v in sorted(chunk["horizon_months"].unique().tolist())),
                "notes": (
                    "Common-window default split with both 60m and 120m horizons present."
                    if split_name in SPLIT_ORDER
                    else "Filtered first-pass rows outside the default common 60m/120m split window; retained in modeling_panel_firstpass only."
                ),
            }
        )
    return pd.DataFrame(records)



def build_split_summary(panel_with_splits: pd.DataFrame) -> pd.DataFrame:
    """Summarize split coverage overall, by horizon, by sleeve, and by horizon x sleeve."""
    split_panel = panel_with_splits.loc[panel_with_splits["default_split"].isin(SPLIT_ORDER)].copy()
    rows: list[dict[str, object]] = []

    def add_rows(summary_scope: str, group_cols: list[str]) -> None:
        grouped = split_panel.groupby(group_cols, as_index=False, observed=True).agg(
            start_month_end=("month_end", "min"),
            end_month_end=("month_end", "max"),
            month_count=("month_end", "nunique"),
            row_count=("month_end", "size"),
        )
        for row in grouped.to_dict("records"):
            out = {
                "summary_scope": summary_scope,
                "split_name": row.get("default_split", row.get("split_name")),
                "horizon_months": row.get("horizon_months"),
                "sleeve_id": row.get("sleeve_id"),
                "start_month_end": pd.to_datetime(row["start_month_end"]).date().isoformat(),
                "end_month_end": pd.to_datetime(row["end_month_end"]).date().isoformat(),
                "month_count": int(row["month_count"]),
                "row_count": int(row["row_count"]),
            }
            rows.append(out)

    add_rows("overall", ["default_split"])
    add_rows("by_horizon", ["default_split", "horizon_months"])
    add_rows("by_sleeve", ["default_split", "sleeve_id"])
    add_rows("by_horizon_sleeve", ["default_split", "horizon_months", "sleeve_id"])

    out = pd.DataFrame(rows)
    return out.sort_values(["summary_scope", "split_name", "horizon_months", "sleeve_id"], na_position="last").reset_index(drop=True)
