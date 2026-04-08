"""Data loading helpers for XOPTPOE v4 supervised benchmark modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xoptpoe_data.targets.build_monthly_targets import build_monthly_realized_returns, collapse_target_to_month_end_prices
from xoptpoe_v4_modeling.features import DEFAULT_FEATURE_SET, feature_columns_for_set
from xoptpoe_v4_modeling.io import load_csv, load_parquet


TARGET_COL = "annualized_excess_forward_return"
DEFAULT_HORIZONS: tuple[int, ...] = (60, 120)
DEFAULT_EXCLUDED_SLEEVES: frozenset[str] = frozenset({"CR_EU_HY"})


def _module_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_v4_sleeve_order(excluded_sleeves: frozenset[str] = DEFAULT_EXCLUDED_SLEEVES) -> tuple[str, ...]:
    asset_master_path = _module_project_root() / "data" / "final_v4_expanded_universe" / "asset_master.csv"
    asset_master = pd.read_csv(asset_master_path)
    sleeves = tuple(s for s in asset_master["sleeve_id"].astype(str).tolist() if s not in excluded_sleeves)
    if "CR_EU_HY" in sleeves:
        raise ValueError("Default v4 benchmark roster must exclude CR_EU_HY")
    return sleeves


SLEEVE_ORDER: tuple[str, ...] = _load_v4_sleeve_order()


@dataclass(frozen=True)
class ModelPaths:
    project_root: Path
    train_split: Path
    validation_split: Path
    test_split: Path
    feature_manifest: Path
    asset_master: Path
    target_raw_direct: Path
    euro_synth_month_end: Path
    tb3ms_monthly: Path
    data_out_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class LoadedModelingInputs:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_manifest: pd.DataFrame
    feature_columns: list[str]
    monthly_excess_history: pd.DataFrame | None = None


def default_paths(project_root: Path) -> ModelPaths:
    root = project_root.resolve()
    return ModelPaths(
        project_root=root,
        train_split=root / "data" / "modeling_v4" / "train_split.parquet",
        validation_split=root / "data" / "modeling_v4" / "validation_split.parquet",
        test_split=root / "data" / "modeling_v4" / "test_split.parquet",
        feature_manifest=root / "data" / "modeling_v4" / "feature_set_manifest.csv",
        asset_master=root / "data" / "final_v4_expanded_universe" / "asset_master.csv",
        target_raw_direct=root / "data" / "final_v4_expanded_universe" / "target_raw_direct.csv",
        euro_synth_month_end=root / "data" / "final_v4_expanded_universe" / "euro_fixed_income_month_end_usd_synth.csv",
        tb3ms_monthly=root / "data" / "intermediate" / "tb3ms_monthly.csv",
        data_out_dir=root / "data" / "modeling_v4",
        reports_dir=root / "reports",
    )


def _sort_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sleeve_id", "month_end", "horizon_months", TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split frame missing required columns: {sorted(missing)}")

    work = df.copy()
    work = work.loc[~work["sleeve_id"].isin(DEFAULT_EXCLUDED_SLEEVES)].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    dup_cnt = int(work.duplicated(subset=["sleeve_id", "month_end", "horizon_months"]).sum())
    if dup_cnt > 0:
        raise ValueError(f"split frame has duplicate keys: {dup_cnt}")

    work["sleeve_id"] = pd.Categorical(work["sleeve_id"], categories=list(SLEEVE_ORDER), ordered=True)
    work = work.sort_values(["month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)
    if work["sleeve_id"].isna().any():
        raise ValueError("split frame contains sleeves outside the active v4 benchmark sleeve order")
    work["sleeve_id"] = work["sleeve_id"].astype(str)
    return work


def load_modeling_inputs(
    project_root: Path,
    *,
    feature_set_name: str = DEFAULT_FEATURE_SET,
) -> LoadedModelingInputs:
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


def build_sleeve_horizon_benchmark(train_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        train_df.groupby(["sleeve_id", "horizon_months"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "benchmark_pred"})
    )
    return grouped


def _load_monthly_excess_history(paths: ModelPaths) -> pd.DataFrame:
    direct_raw = load_csv(paths.target_raw_direct, parse_dates=["trade_date"])
    euro_synth = load_csv(paths.euro_synth_month_end, parse_dates=["month_end", "trade_date"])
    tb3ms = load_csv(paths.tb3ms_monthly, parse_dates=["month_end"])

    direct_month_end = collapse_target_to_month_end_prices(direct_raw)
    euro_month_end = euro_synth[["sleeve_id", "ticker", "month_end", "trade_date", "adj_close", "close"]].copy()
    prices = pd.concat([direct_month_end, euro_month_end], ignore_index=True)
    prices = prices.loc[~prices["sleeve_id"].isin(DEFAULT_EXCLUDED_SLEEVES)].copy()
    prices["month_end"] = pd.to_datetime(prices["month_end"])
    prices = prices.sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)

    monthly_returns = build_monthly_realized_returns(prices)
    tb3ms["tb3ms"] = pd.to_numeric(tb3ms["tb3ms"], errors="coerce")
    tb3ms["rf_1m"] = tb3ms["tb3ms"] / 1200.0

    merged = monthly_returns.merge(tb3ms[["month_end", "rf_1m"]], on="month_end", how="left", validate="m:1")
    merged["excess_ret_1m"] = merged["ret_1m_realized"] - merged["rf_1m"]
    panel = (
        merged.pivot(index="month_end", columns="sleeve_id", values="excess_ret_1m")
        .sort_index()
        .reindex(columns=list(SLEEVE_ORDER))
    )
    if panel.columns.isna().any():
        raise ValueError("monthly excess history could not be aligned to the active v4 sleeve order")
    return panel.astype(float)


def aggregate_horizon_values(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    required = {"month_end", "sleeve_id", "horizon_months", value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns for aggregation: {sorted(missing)}")

    work = frame[["month_end", "sleeve_id", "horizon_months", value_col]].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    grouped = (
        work.groupby(["month_end", "sleeve_id"], as_index=False)
        .agg(horizon_count=("horizon_months", "nunique"), value=(value_col, "mean"))
        .sort_values(["month_end", "sleeve_id"])
        .reset_index(drop=True)
    )
    if not grouped["horizon_count"].eq(len(DEFAULT_HORIZONS)).all():
        raise ValueError("Horizon aggregation expected both 60m and 120m rows for every sleeve-month")
    return grouped.drop(columns=["horizon_count"]).rename(columns={"value": value_col})
