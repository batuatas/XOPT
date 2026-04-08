"""Data loading helpers for XOPTPOE v2 neural modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xoptpoe_v2_modeling.features import DEFAULT_FEATURE_SET, feature_columns_for_set
from workspace_v4.src.xoptpoe_v2_modeling.io import load_csv, load_parquet


TARGET_COL = "annualized_excess_forward_return"
DEFAULT_HORIZONS: tuple[int, ...] = (60, 120)
SLEEVE_ORDER: tuple[str, ...] = (
    "EQ_US",
    "EQ_EZ",
    "EQ_JP",
    "EQ_EM",
    "FI_UST",
    "FI_IG",
    "ALT_GLD",
    "RE_US",
)
SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class ModelPaths:
    """Filesystem layout for the v2 neural modeling stage."""

    project_root: Path
    train_split: Path
    validation_split: Path
    test_split: Path
    feature_manifest: Path
    sleeve_monthly_returns: Path
    tb3ms_monthly: Path
    data_out_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class LoadedModelingInputs:
    """Loaded split tables, feature manifest, and excess-return history."""

    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_manifest: pd.DataFrame
    feature_columns: list[str]
    monthly_excess_history: pd.DataFrame



def default_paths(project_root: Path) -> ModelPaths:
    root = project_root.resolve()
    return ModelPaths(
        project_root=root,
        train_split=root / "data" / "modeling_v2" / "train_split.parquet",
        validation_split=root / "data" / "modeling_v2" / "validation_split.parquet",
        test_split=root / "data" / "modeling_v2" / "test_split.parquet",
        feature_manifest=root / "data" / "modeling_v2" / "feature_set_manifest.csv",
        sleeve_monthly_returns=root / "data" / "intermediate" / "sleeve_monthly_returns.csv",
        tb3ms_monthly=root / "data" / "intermediate" / "tb3ms_monthly.csv",
        data_out_dir=root / "data" / "modeling_v2",
        reports_dir=root / "reports" / "v2_long_horizon",
    )



def _sort_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sleeve_id", "month_end", "horizon_months", TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split frame missing required columns: {sorted(missing)}")

    work = df.copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    dup_cnt = int(work.duplicated(subset=["sleeve_id", "month_end", "horizon_months"]).sum())
    if dup_cnt > 0:
        raise ValueError(f"split frame has duplicate keys: {dup_cnt}")

    work["sleeve_id"] = pd.Categorical(work["sleeve_id"], categories=list(SLEEVE_ORDER), ordered=True)
    work = work.sort_values(["month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)
    if work["sleeve_id"].isna().any():
        raise ValueError("split frame contains sleeves outside the locked v2 sleeve order")
    work["sleeve_id"] = work["sleeve_id"].astype(str)
    return work



def _load_monthly_excess_history(paths: ModelPaths) -> pd.DataFrame:
    returns_df = load_csv(paths.sleeve_monthly_returns, parse_dates=["month_end"])
    tb3ms = load_csv(paths.tb3ms_monthly, parse_dates=["month_end"])

    returns_df["ret_1m_realized"] = pd.to_numeric(returns_df["ret_1m_realized"], errors="coerce")
    tb3ms["tb3ms"] = pd.to_numeric(tb3ms["tb3ms"], errors="coerce")
    tb3ms["rf_1m"] = tb3ms["tb3ms"] / 1200.0

    merged = returns_df.merge(tb3ms[["month_end", "rf_1m"]], on="month_end", how="left", validate="m:1")
    merged["excess_ret_1m"] = merged["ret_1m_realized"] - merged["rf_1m"]

    panel = (
        merged.pivot(index="month_end", columns="sleeve_id", values="excess_ret_1m")
        .sort_index()
        .reindex(columns=list(SLEEVE_ORDER))
    )
    if panel.columns.isna().any():
        raise ValueError("monthly excess history could not be aligned to locked sleeve order")
    return panel.astype(float)



def load_modeling_inputs(
    project_root: Path,
    *,
    feature_set_name: str = DEFAULT_FEATURE_SET,
) -> LoadedModelingInputs:
    """Load the frozen first-pass splits and the selected feature-set columns."""
    paths = default_paths(project_root)
    train_df = _sort_split_frame(load_parquet(paths.train_split))
    validation_df = _sort_split_frame(load_parquet(paths.validation_split))
    test_df = _sort_split_frame(load_parquet(paths.test_split))
    feature_manifest = load_csv(paths.feature_manifest, parse_dates=["first_valid_date", "last_valid_date"])

    feature_columns = feature_columns_for_set(feature_manifest, feature_set_name)
    missing_features = [col for col in feature_columns if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Selected feature set is missing columns from train split: {missing_features[:10]}")

    monthly_excess_history = _load_monthly_excess_history(paths)
    return LoadedModelingInputs(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_manifest=feature_manifest,
        feature_columns=feature_columns,
        monthly_excess_history=monthly_excess_history,
    )



def aggregate_horizon_values(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Average 60m and 120m horizon values into a single monthly sleeve-level signal."""
    required = {"month_end", "sleeve_id", "horizon_months", value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns for aggregation: {sorted(missing)}")

    work = frame[["month_end", "sleeve_id", "horizon_months", value_col]].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    grouped = (
        work.groupby(["month_end", "sleeve_id"], as_index=False)
        .agg(
            horizon_count=("horizon_months", "nunique"),
            value=(value_col, "mean"),
        )
        .sort_values(["month_end", "sleeve_id"])
        .reset_index(drop=True)
    )
    if not grouped["horizon_count"].eq(len(DEFAULT_HORIZONS)).all():
        raise ValueError("Horizon aggregation expected both 60m and 120m rows for every sleeve-month")
    return grouped.drop(columns=["horizon_count"]).rename(columns={"value": value_col})



def build_sleeve_horizon_benchmark(train_df: pd.DataFrame) -> pd.DataFrame:
    """Naive benchmark: train-set mean target by sleeve and horizon."""
    grouped = (
        train_df.groupby(["sleeve_id", "horizon_months"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "benchmark_pred"})
    )
    return grouped
