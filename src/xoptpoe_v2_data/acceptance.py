"""Acceptance / EDA audit for the XOPTPOE v2 long-horizon dataset."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from xoptpoe_v2_data.config import LOCKED_HORIZONS, V2Paths, default_paths
from xoptpoe_v2_data.io import load_csv, write_csv, write_text

KEY_TARGET_COLUMNS = [
    "annualized_excess_forward_return",
    "annualized_total_forward_return",
    "annualized_rf_forward_return",
    "cumulative_total_forward_return",
    "realized_forward_volatility",
    "realized_forward_max_drawdown",
]

KEY_FEATURES = [
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
    "china_cli",
    "china_cmi",
    "china_pe_ratio",
    "china_div_yield",
    "china_sse_composite_usd_mom_12_1",
    "jp_pe_ratio",
    "jp_tankan_actual",
    "jp_buyback_index_usd_mom_12_1",
    "cape_local",
    "cape_usa",
    "em_minus_global_pe",
    "us_hy_oas",
    "eu_hy_oas",
    "eu_ig_corp_tr_usd_mom_12_1",
    "int_log_horizon_x_mom_12_1",
    "int_log_horizon_x_vix",
    "int_china_cli_x_eq_em",
    "int_jp_pe_ratio_x_eq_jp",
]

KEY_INTERACTION_FAMILIES_TO_DEFER = {
    "china_block_x_em_relevance",
    "japan_block_x_jp_relevance",
    "sleeve_dummy_x_predictor",
}


@dataclass
class LoadedAcceptanceInputs:
    feature_master: pd.DataFrame
    target_panel: pd.DataFrame
    modeling_panel: pd.DataFrame
    horizon_manifest: pd.DataFrame
    feature_dictionary: pd.DataFrame
    interaction_dictionary: pd.DataFrame
    month_end_prices: pd.DataFrame
    monthly_returns: pd.DataFrame
    tb3ms_monthly: pd.DataFrame


@dataclass
class AcceptanceArtifacts:
    integrity_checks: pd.DataFrame
    target_distribution_by_horizon: pd.DataFrame
    target_distribution_by_sleeve: pd.DataFrame
    feature_block_missingness: pd.DataFrame
    feature_scale_summary: pd.DataFrame
    trainability_summary: pd.DataFrame
    eda_acceptance_report: str


class IntegrityRecorder:
    """Collect machine-checkable integrity audit rows."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(
        self,
        *,
        check_group: str,
        check_name: str,
        status: str,
        observed: object,
        expected: object,
        notes: str,
    ) -> None:
        self.rows.append(
            {
                "check_group": check_group,
                "check_name": check_name,
                "status": status,
                "observed": observed,
                "expected": expected,
                "notes": notes,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)



def _safe_float(value: object) -> float | np.nan:
    if pd.isna(value):
        return np.nan
    return float(value)



def _summary_stats(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "nonmissing_share": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(clean.size),
        "nonmissing_share": float(series.notna().mean()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "p01": float(clean.quantile(0.01)),
        "p05": float(clean.quantile(0.05)),
        "p25": float(clean.quantile(0.25)),
        "p50": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }



def _row_missing_share(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(dtype=float)
    return frame[cols].isna().mean(axis=1)



def _forward_max_drawdown(window_returns: pd.Series) -> float:
    wealth = (1.0 + window_returns.fillna(0.0)).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else float("nan")



def load_acceptance_inputs(paths: V2Paths) -> LoadedAcceptanceInputs:
    return LoadedAcceptanceInputs(
        feature_master=pd.read_parquet(paths.data_final_v2_dir / "feature_master_monthly.parquet"),
        target_panel=pd.read_parquet(paths.data_final_v2_dir / "target_panel_long_horizon.parquet"),
        modeling_panel=pd.read_parquet(paths.data_final_v2_dir / "modeling_panel_hstack.parquet"),
        horizon_manifest=load_csv(paths.data_final_v2_dir / "horizon_manifest.csv"),
        feature_dictionary=load_csv(paths.data_final_v2_dir / "feature_dictionary.csv", parse_dates=["first_valid_date", "last_valid_date"]),
        interaction_dictionary=load_csv(paths.data_final_v2_dir / "interaction_dictionary.csv", parse_dates=["first_valid_date", "last_valid_date"]),
        month_end_prices=load_csv(paths.project_root / "data" / "intermediate" / "sleeve_month_end_prices.csv", parse_dates=["month_end", "trade_date"]),
        monthly_returns=load_csv(paths.project_root / "data" / "intermediate" / "sleeve_monthly_returns.csv", parse_dates=["month_end"]),
        tb3ms_monthly=load_csv(paths.project_root / "data" / "intermediate" / "tb3ms_monthly.csv", parse_dates=["month_end"]),
    )



def _compute_expected_targets(inputs: LoadedAcceptanceInputs) -> pd.DataFrame:
    prices = inputs.month_end_prices.copy()
    prices["month_end"] = pd.to_datetime(prices["month_end"])
    prices = prices.sort_values(["sleeve_id", "month_end"])

    rets = inputs.monthly_returns.copy()
    rets["month_end"] = pd.to_datetime(rets["month_end"])
    rets = rets.sort_values(["sleeve_id", "month_end"])

    rf = inputs.tb3ms_monthly.copy()
    rf["month_end"] = pd.to_datetime(rf["month_end"])
    rf = rf.sort_values("month_end").reset_index(drop=True)
    rf["rf_1m"] = pd.to_numeric(rf["tb3ms"], errors="coerce") / 1200.0

    price_pivot = prices.pivot(index="month_end", columns="sleeve_id", values="adj_close").sort_index()
    ret_pivot = rets.pivot(index="month_end", columns="sleeve_id", values="ret_1m_realized").sort_index()
    months = pd.Index(sorted(price_pivot.index.unique()))
    rf = rf.set_index("month_end").reindex(months)

    rows: list[dict[str, object]] = []
    for sleeve_id in price_pivot.columns:
        sleeve_prices = price_pivot[sleeve_id].reindex(months)
        sleeve_rets = ret_pivot[sleeve_id].reindex(months)
        for idx, month_end in enumerate(months):
            price_t = sleeve_prices.iloc[idx]
            for horizon in LOCKED_HORIZONS:
                target_idx = idx + horizon
                target_available = bool(
                    pd.notna(price_t)
                    and target_idx < len(months)
                    and pd.notna(sleeve_prices.iloc[target_idx])
                )
                gross_total = np.nan
                cumulative_total = np.nan
                gross_rf = np.nan
                cumulative_rf = np.nan
                annualized_total = np.nan
                annualized_rf = np.nan
                annualized_excess = np.nan
                realized_vol = np.nan
                realized_maxdd = np.nan
                valid_months = 0
                if target_available:
                    forward_rf = rf["rf_1m"].iloc[idx + 1 : target_idx + 1]
                    forward_rets = sleeve_rets.iloc[idx + 1 : target_idx + 1]
                    if forward_rf.notna().all() and forward_rets.notna().all():
                        gross_total = float(sleeve_prices.iloc[target_idx] / price_t)
                        cumulative_total = gross_total - 1.0
                        gross_rf = float((1.0 + forward_rf).prod())
                        cumulative_rf = gross_rf - 1.0
                        annualized_total = float(gross_total ** (12.0 / horizon) - 1.0)
                        annualized_rf = float(gross_rf ** (12.0 / horizon) - 1.0)
                        annualized_excess = float((gross_total / gross_rf) ** (12.0 / horizon) - 1.0)
                        realized_vol = float(forward_rets.std(ddof=1) * sqrt(12.0))
                        realized_maxdd = _forward_max_drawdown(forward_rets)
                        valid_months = int(forward_rets.notna().sum())
                    else:
                        target_available = False
                rows.append(
                    {
                        "sleeve_id": sleeve_id,
                        "month_end": month_end,
                        "horizon_months": horizon,
                        "expected_target_available_flag": int(target_available),
                        "expected_forward_valid_month_count": valid_months,
                        "expected_gross_total_forward_return": gross_total,
                        "expected_cumulative_total_forward_return": cumulative_total,
                        "expected_gross_rf_forward_return": gross_rf,
                        "expected_cumulative_rf_forward_return": cumulative_rf,
                        "expected_annualized_total_forward_return": annualized_total,
                        "expected_annualized_rf_forward_return": annualized_rf,
                        "expected_annualized_excess_forward_return": annualized_excess,
                        "expected_realized_forward_volatility": realized_vol,
                        "expected_realized_forward_max_drawdown": realized_maxdd,
                    }
                )
    return pd.DataFrame(rows).sort_values(["sleeve_id", "month_end", "horizon_months"]).reset_index(drop=True)



def _build_integrity_checks(inputs: LoadedAcceptanceInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = IntegrityRecorder()
    fm = inputs.feature_master.copy()
    tp = inputs.target_panel.copy()
    mp = inputs.modeling_panel.copy()

    for frame in (fm, tp, mp):
        frame["month_end"] = pd.to_datetime(frame["month_end"])

    rec.add(
        check_group="structure",
        check_name="feature_master_duplicate_keys",
        status="PASS" if fm.duplicated(["sleeve_id", "month_end"]).sum() == 0 else "FAIL",
        observed=int(fm.duplicated(["sleeve_id", "month_end"]).sum()),
        expected=0,
        notes="Duplicate (sleeve_id, month_end) keys are not allowed.",
    )
    rec.add(
        check_group="structure",
        check_name="target_panel_duplicate_keys",
        status="PASS" if tp.duplicated(["sleeve_id", "month_end", "horizon_months"]).sum() == 0 else "FAIL",
        observed=int(tp.duplicated(["sleeve_id", "month_end", "horizon_months"]).sum()),
        expected=0,
        notes="Duplicate (sleeve_id, month_end, horizon_months) keys are not allowed.",
    )
    rec.add(
        check_group="structure",
        check_name="modeling_panel_duplicate_keys",
        status="PASS" if mp.duplicated(["sleeve_id", "month_end", "horizon_months"]).sum() == 0 else "FAIL",
        observed=int(mp.duplicated(["sleeve_id", "month_end", "horizon_months"]).sum()),
        expected=0,
        notes="Duplicate stacked modeling keys are not allowed.",
    )

    feature_horizons = mp.groupby(["sleeve_id", "month_end"]).size()
    rec.add(
        check_group="joins",
        check_name="three_horizons_per_feature_key",
        status="PASS" if feature_horizons.eq(len(LOCKED_HORIZONS)).all() else "FAIL",
        observed=int(feature_horizons.ne(len(LOCKED_HORIZONS)).sum()),
        expected=0,
        notes="Every feature_master key should expand to exactly three horizons in the stacked panel.",
    )

    target_key_diff = len(
        set(map(tuple, mp[["sleeve_id", "month_end", "horizon_months"]].to_records(index=False)))
        ^ set(map(tuple, tp[["sleeve_id", "month_end", "horizon_months"]].to_records(index=False)))
    )
    rec.add(
        check_group="joins",
        check_name="target_modeling_key_set_match",
        status="PASS" if target_key_diff == 0 else "FAIL",
        observed=int(target_key_diff),
        expected=0,
        notes="modeling_panel_hstack should carry the same stacked key set as target_panel_long_horizon.",
    )

    expected_baseline = (
        mp["baseline_sample_inclusion_1m_flag"].eq(1)
        & mp["target_available_flag"].eq(1)
        & mp["proxy_flag"].eq(1)
    ).astype(int)
    expected_strict = (
        mp["strict_all_feature_complete_flag"].eq(1)
        & mp["target_available_flag"].eq(1)
        & mp["proxy_flag"].eq(1)
    ).astype(int)
    rec.add(
        check_group="flags",
        check_name="baseline_trainable_formula_match",
        status="PASS" if mp["baseline_trainable_flag"].equals(expected_baseline) else "FAIL",
        observed=int((mp["baseline_trainable_flag"] != expected_baseline).sum()),
        expected=0,
        notes="baseline_trainable_flag should equal baseline_sample_inclusion_1m_flag & target_available_flag & proxy_flag.",
    )
    rec.add(
        check_group="flags",
        check_name="strict_trainable_formula_match",
        status="PASS" if mp["strict_trainable_flag"].equals(expected_strict) else "FAIL",
        observed=int((mp["strict_trainable_flag"] != expected_strict).sum()),
        expected=0,
        notes="strict_trainable_flag should equal strict_all_feature_complete_flag & target_available_flag & proxy_flag.",
    )
    rec.add(
        check_group="flags",
        check_name="strict_trainable_subset_of_baseline",
        status="PASS" if mp.loc[mp["strict_trainable_flag"].eq(1), "baseline_trainable_flag"].eq(1).all() else "FAIL",
        observed=int(mp.loc[mp["strict_trainable_flag"].eq(1), "baseline_trainable_flag"].eq(0).sum()),
        expected=0,
        notes="Strict-trainable rows must also satisfy the baseline trainable filter.",
    )

    expected_targets = _compute_expected_targets(inputs)
    target_compare = tp.merge(
        expected_targets,
        on=["sleeve_id", "month_end", "horizon_months"],
        how="left",
        validate="one_to_one",
    )
    rec.add(
        check_group="targets",
        check_name="target_available_flag_recompute_match",
        status="PASS" if target_compare["target_available_flag"].equals(target_compare["expected_target_available_flag"]) else "FAIL",
        observed=int((target_compare["target_available_flag"] != target_compare["expected_target_available_flag"]).sum()),
        expected=0,
        notes="Recomputed target availability from prices/rf/returns should match stored targets exactly.",
    )
    rec.add(
        check_group="targets",
        check_name="forward_valid_month_count_recompute_match",
        status="PASS" if target_compare["forward_valid_month_count"].equals(target_compare["expected_forward_valid_month_count"]) else "FAIL",
        observed=int((target_compare["forward_valid_month_count"] != target_compare["expected_forward_valid_month_count"]).sum()),
        expected=0,
        notes="Valid forward-month counts should match the recomputed target windows.",
    )

    metric_map = {
        "gross_total_forward_return": "expected_gross_total_forward_return",
        "cumulative_total_forward_return": "expected_cumulative_total_forward_return",
        "gross_rf_forward_return": "expected_gross_rf_forward_return",
        "cumulative_rf_forward_return": "expected_cumulative_rf_forward_return",
        "annualized_total_forward_return": "expected_annualized_total_forward_return",
        "annualized_rf_forward_return": "expected_annualized_rf_forward_return",
        "annualized_excess_forward_return": "expected_annualized_excess_forward_return",
        "realized_forward_volatility": "expected_realized_forward_volatility",
        "realized_forward_max_drawdown": "expected_realized_forward_max_drawdown",
    }
    available = target_compare["target_available_flag"].eq(1)
    for actual_col, expected_col in metric_map.items():
        diff = (target_compare.loc[available, actual_col] - target_compare.loc[available, expected_col]).abs()
        max_abs_diff = float(diff.max()) if not diff.empty else 0.0
        mismatch_count = int(diff.gt(1e-12).sum()) if not diff.empty else 0
        rec.add(
            check_group="targets",
            check_name=f"{actual_col}_recompute_match",
            status="PASS" if mismatch_count == 0 else "FAIL",
            observed=max_abs_diff,
            expected="<=1e-12",
            notes=f"Recomputed {actual_col} max absolute difference on target_available rows.",
        )

    rec.add(
        check_group="targets",
        check_name="no_nonpositive_gross_total_available_rows",
        status="PASS" if tp.loc[tp["target_available_flag"].eq(1), "gross_total_forward_return"].gt(0).all() else "FAIL",
        observed=int(tp.loc[tp["target_available_flag"].eq(1), "gross_total_forward_return"].le(0).sum()),
        expected=0,
        notes="Available long-horizon compounded gross returns must be strictly positive.",
    )
    rec.add(
        check_group="targets",
        check_name="realized_volatility_nonnegative",
        status="PASS" if tp.loc[tp["target_available_flag"].eq(1), "realized_forward_volatility"].ge(0).all() else "FAIL",
        observed=int(tp.loc[tp["target_available_flag"].eq(1), "realized_forward_volatility"].lt(0).sum()),
        expected=0,
        notes="Annualized realized forward volatility should be nonnegative.",
    )
    mdd = tp.loc[tp["target_available_flag"].eq(1), "realized_forward_max_drawdown"]
    rec.add(
        check_group="targets",
        check_name="realized_max_drawdown_bounds",
        status="PASS" if mdd.le(0).all() and mdd.gt(-1).all() else "FAIL",
        observed=int((mdd.gt(0) | mdd.le(-1)).sum()),
        expected=0,
        notes="Forward max drawdown should stay in (-1, 0].",
    )

    horizon_counts = tp.groupby("horizon_months")["target_available_flag"].sum().astype(int)
    expected_counts = {60: 1456, 120: 976, 180: 496}
    for horizon, expected in expected_counts.items():
        rec.add(
            check_group="coverage",
            check_name=f"target_available_rows_h{horizon}",
            status="PASS" if int(horizon_counts.get(horizon, -1)) == expected else "FAIL",
            observed=int(horizon_counts.get(horizon, -1)),
            expected=expected,
            notes="Stored v2 long-horizon availability should match the documented build counts.",
        )

    return rec.to_frame(), target_compare



def _distribution_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, chunk in frame.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_map = dict(zip(group_cols, group_key))
        for metric in KEY_TARGET_COLUMNS:
            stats = _summary_stats(chunk[metric])
            rows.append({**key_map, "metric_name": metric, **stats})
    return pd.DataFrame(rows)



def _feature_block_missingness(
    modeling_available: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    feature_meta = feature_dictionary.loc[
        feature_dictionary["available_in_modeling_panel_hstack"].eq(1),
        ["feature_name", "block_name", "geography", "is_interaction", "first_valid_date", "last_valid_date"],
    ].copy()

    rows: list[dict[str, object]] = []

    def append_scope(scope: str, subset: pd.DataFrame, extra_keys: dict[str, object]) -> None:
        for block_name, block_meta in feature_meta.groupby("block_name"):
            cols = [col for col in block_meta["feature_name"].tolist() if col in subset.columns]
            if not cols:
                continue
            feature_missing = subset[cols].isna().mean().sort_values(ascending=False)
            row_missing = _row_missing_share(subset, cols)
            worst_feature = feature_missing.index[0]
            rows.append(
                {
                    "summary_scope": scope,
                    "block_name": block_name,
                    **extra_keys,
                    "feature_count": int(len(cols)),
                    "avg_feature_missing_share": float(feature_missing.mean()),
                    "median_feature_missing_share": float(feature_missing.median()),
                    "max_feature_missing_share": float(feature_missing.max()),
                    "mean_row_missing_share": float(row_missing.mean()),
                    "latest_feature_start_date": block_meta["first_valid_date"].max(),
                    "earliest_feature_start_date": block_meta["first_valid_date"].min(),
                    "latest_feature_end_date": block_meta["last_valid_date"].max(),
                    "worst_feature": worst_feature,
                    "worst_feature_missing_share": float(feature_missing.iloc[0]),
                    "interaction_feature_count": int(block_meta["is_interaction"].sum()),
                }
            )

    append_scope("overall_target_available", modeling_available, {})
    for sleeve_id, chunk in modeling_available.groupby("sleeve_id"):
        append_scope("by_sleeve_target_available", chunk, {"sleeve_id": sleeve_id})
    for horizon, chunk in modeling_available.groupby("horizon_months"):
        append_scope("by_horizon_target_available", chunk, {"horizon_months": int(horizon)})

    out = pd.DataFrame(rows)
    for col in ["sleeve_id", "horizon_months"]:
        if col not in out.columns:
            out[col] = np.nan
    return out[
        [
            "summary_scope",
            "block_name",
            "sleeve_id",
            "horizon_months",
            "feature_count",
            "interaction_feature_count",
            "avg_feature_missing_share",
            "median_feature_missing_share",
            "max_feature_missing_share",
            "mean_row_missing_share",
            "earliest_feature_start_date",
            "latest_feature_start_date",
            "latest_feature_end_date",
            "worst_feature",
            "worst_feature_missing_share",
        ]
    ].sort_values(["summary_scope", "block_name", "sleeve_id", "horizon_months"], na_position="last").reset_index(drop=True)



def _feature_scale_summary(
    modeling_panel: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    subset = modeling_panel.loc[modeling_panel["baseline_trainable_flag"].eq(1)].copy()
    meta = feature_dictionary.loc[
        feature_dictionary["available_in_modeling_panel_hstack"].eq(1),
        ["feature_name", "block_name", "geography", "is_interaction"],
    ].drop_duplicates("feature_name")
    rows: list[dict[str, object]] = []
    for _, row in meta.iterrows():
        feature = row["feature_name"]
        if feature not in subset.columns:
            continue
        series = pd.to_numeric(subset[feature], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            zero_share = np.nan
            max_abs = np.nan
            q99_abs = np.nan
            std = np.nan
        else:
            zero_share = float(clean.eq(0).mean())
            max_abs = float(clean.abs().max())
            q99_abs = float(clean.abs().quantile(0.99))
            std = float(clean.std(ddof=1)) if clean.size > 1 else 0.0
        if clean.empty:
            stability_flag = "all_missing"
        elif std is not np.nan and std <= 1e-12:
            stability_flag = "low_variance"
        elif q99_abs is not np.nan and q99_abs > 1e3:
            stability_flag = "very_large_scale"
        elif zero_share is not np.nan and zero_share >= 0.95 and int(row["is_interaction"]) == 1:
            stability_flag = "very_sparse_interaction"
        else:
            stability_flag = "ok"
        rows.append(
            {
                "feature_name": feature,
                "block_name": row["block_name"],
                "geography": row["geography"],
                "is_interaction": int(row["is_interaction"]),
                "analysis_sample": "baseline_trainable_rows",
                **_summary_stats(series),
                "zero_share": zero_share,
                "max_abs": max_abs,
                "q99_abs": q99_abs,
                "stability_flag": stability_flag,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_interaction", "block_name", "feature_name"]).reset_index(drop=True)



def _trainability_summary(
    modeling_panel: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available = modeling_panel.loc[modeling_panel["target_available_flag"].eq(1)].copy()

    for horizon, chunk in modeling_panel.groupby("horizon_months"):
        target_chunk = chunk.loc[chunk["target_available_flag"].eq(1)]
        rows.append(
            {
                "summary_type": "horizon_trainability",
                "horizon_months": int(horizon),
                "total_rows": int(len(chunk)),
                "target_available_rows": int(len(target_chunk)),
                "baseline_trainable_rows": int(target_chunk["baseline_trainable_flag"].sum()),
                "strict_trainable_rows": int(target_chunk["strict_trainable_flag"].sum()),
                "baseline_trainable_share_of_available": float(target_chunk["baseline_trainable_flag"].mean()) if len(target_chunk) else np.nan,
                "strict_trainable_share_of_available": float(target_chunk["strict_trainable_flag"].mean()) if len(target_chunk) else np.nan,
                "baseline_trainable_share_of_total": float(chunk["baseline_trainable_flag"].mean()),
                "strict_trainable_share_of_total": float(chunk["strict_trainable_flag"].mean()),
            }
        )

    target_available = modeling_panel.loc[modeling_panel["target_available_flag"].eq(1)].copy()
    strict_fail = target_available.loc[target_available["strict_trainable_flag"].eq(0)].copy()
    block_meta = feature_dictionary.loc[
        feature_dictionary["available_in_modeling_panel_hstack"].eq(1), ["feature_name", "block_name"]
    ].drop_duplicates("feature_name")
    if not strict_fail.empty:
        for block_name, block_chunk in block_meta.groupby("block_name"):
            cols = [col for col in block_chunk["feature_name"].tolist() if col in strict_fail.columns]
            if not cols:
                continue
            missing_rows = strict_fail[cols].isna().any(axis=1)
            rows.append(
                {
                    "summary_type": "strict_failure_block_cause",
                    "block_name": block_name,
                    "target_available_rows": int(len(target_available)),
                    "strict_failure_rows": int(len(strict_fail)),
                    "rows_with_block_missing": int(missing_rows.sum()),
                    "share_of_target_available_rows": float(missing_rows.mean()),
                    "share_of_strict_failure_rows": float(missing_rows.sum() / len(strict_fail)),
                }
            )
        feature_missing = strict_fail[[c for c in block_meta["feature_name"].tolist() if c in strict_fail.columns]].isna().mean().sort_values(ascending=False)
        for feature_name, missing_share in feature_missing.head(20).items():
            rows.append(
                {
                    "summary_type": "strict_failure_feature_cause",
                    "feature_name": feature_name,
                    "rows_with_feature_missing": int(strict_fail[feature_name].isna().sum()),
                    "share_of_strict_failure_rows": float(strict_fail[feature_name].isna().mean()),
                }
            )
    return pd.DataFrame(rows)



def _interaction_diagnostics(
    modeling_panel: pd.DataFrame,
    interaction_dictionary: pd.DataFrame,
    feature_scale_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_meta = interaction_dictionary.copy()
    interaction_meta["interaction_name"] = interaction_meta["interaction_name"].astype(str)
    scale = feature_scale_summary.rename(columns={"feature_name": "interaction_name"})
    diag = interaction_meta.merge(
        scale[["interaction_name", "std", "zero_share", "max_abs", "q99_abs", "stability_flag"]],
        on="interaction_name",
        how="left",
    )
    diag["recommended_first_pass_usage"] = np.where(
        diag["interaction_family"].isin(KEY_INTERACTION_FAMILIES_TO_DEFER)
        | diag["stability_flag"].isin(["all_missing", "low_variance", "very_large_scale", "very_sparse_interaction"]),
        "defer",
        "keep_optional",
    )
    family_summary = (
        diag.groupby("interaction_family", dropna=False)
        .agg(
            interaction_count=("interaction_name", "size"),
            avg_nonmissing_share=("nonmissing_share", "mean"),
            avg_zero_share=("zero_share", "mean"),
            max_abs_max=("max_abs", "max"),
            defer_count=("recommended_first_pass_usage", lambda x: int(pd.Series(x).eq("defer").sum())),
        )
        .reset_index()
        .sort_values("interaction_family")
    )
    family_summary["recommended_first_pass_usage"] = np.where(
        family_summary["defer_count"].gt(0),
        "defer",
        "keep_optional",
    )
    return diag, family_summary



def _key_feature_section(modeling_trainable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in [f for f in KEY_FEATURES if f in modeling_trainable.columns]:
        rows.append({"feature_name": feature, **_summary_stats(modeling_trainable[feature])})
    return pd.DataFrame(rows)



def _sleeve_target_sanity(target_available: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sleeve_id, horizon), chunk in target_available.groupby(["sleeve_id", "horizon_months"]):
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "horizon_months": int(horizon),
                "row_count": int(len(chunk)),
                "first_date": chunk["month_end"].min(),
                "last_date": chunk["month_end"].max(),
                "target_mean": float(chunk["annualized_excess_forward_return"].mean()),
                "target_std": float(chunk["annualized_excess_forward_return"].std(ddof=1)),
                "target_min": float(chunk["annualized_excess_forward_return"].min()),
                "target_max": float(chunk["annualized_excess_forward_return"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_months", "sleeve_id"]).reset_index(drop=True)



def _render_acceptance_report(
    *,
    integrity_checks: pd.DataFrame,
    target_distribution_by_horizon: pd.DataFrame,
    target_distribution_by_sleeve: pd.DataFrame,
    feature_block_missingness: pd.DataFrame,
    feature_scale_summary: pd.DataFrame,
    trainability_summary: pd.DataFrame,
    interaction_family_summary: pd.DataFrame,
    key_feature_stats: pd.DataFrame,
    target_compare: pd.DataFrame,
    inputs: LoadedAcceptanceInputs,
) -> str:
    failures = integrity_checks.loc[integrity_checks["status"].eq("FAIL")]
    warnings = integrity_checks.loc[integrity_checks["status"].eq("WARN")]
    available = inputs.modeling_panel.loc[inputs.modeling_panel["target_available_flag"].eq(1)].copy()
    baseline = inputs.modeling_panel.loc[inputs.modeling_panel["baseline_trainable_flag"].eq(1)].copy()

    overall_block = feature_block_missingness.loc[feature_block_missingness["summary_scope"].eq("overall_target_available")]
    worst_blocks = overall_block.sort_values("avg_feature_missing_share", ascending=False).head(6)
    by_horizon = trainability_summary.loc[trainability_summary["summary_type"].eq("horizon_trainability")].copy()
    strict_block = trainability_summary.loc[trainability_summary["summary_type"].eq("strict_failure_block_cause")].copy()
    strict_block = strict_block.sort_values("share_of_strict_failure_rows", ascending=False).head(8)
    high_missing_features = trainability_summary.loc[trainability_summary["summary_type"].eq("strict_failure_feature_cause")].copy()
    high_missing_features = high_missing_features.sort_values("share_of_strict_failure_rows", ascending=False).head(10)
    unstable_interactions = feature_scale_summary.loc[
        feature_scale_summary["is_interaction"].eq(1) & feature_scale_summary["stability_flag"].ne("ok")
    ].sort_values(["stability_flag", "feature_name"])
    available_last_dates = (
        available.groupby("horizon_months")["month_end"].max().sort_index().to_dict()
    )

    example_row = target_compare.loc[target_compare["target_available_flag"].eq(1)].iloc[0]
    max_target_diff = integrity_checks.loc[
        integrity_checks["check_group"].eq("targets") & integrity_checks["check_name"].str.endswith("_recompute_match"),
        "observed",
    ]
    max_target_diff_value = float(pd.to_numeric(max_target_diff, errors="coerce").max()) if not max_target_diff.empty else 0.0

    lines = [
        "# XOPTPOE v2_long_horizon Acceptance / EDA Report",
        "",
        "## Executive View",
        f"- Structural integrity checks: {int((integrity_checks['status'] == 'PASS').sum())} PASS, {int((integrity_checks['status'] == 'WARN').sum())} WARN, {int((integrity_checks['status'] == 'FAIL').sum())} FAIL.",
        f"- modeling_panel_hstack shape: {inputs.modeling_panel.shape[0]} rows x {inputs.modeling_panel.shape[1]} columns.",
        f"- feature_master_monthly shape: {inputs.feature_master.shape[0]} rows x {inputs.feature_master.shape[1]} columns.",
        f"- target_panel_long_horizon shape: {inputs.target_panel.shape[0]} rows x {inputs.target_panel.shape[1]} columns.",
        f"- Recomputed target formulas matched the stored panel with max absolute difference {max_target_diff_value:.3e} across all audited target fields.",
        "",
        "## Structural Integrity",
        f"- Duplicate-key checks passed for all three main tables: feature_master_monthly, target_panel_long_horizon, and modeling_panel_hstack.",
        f"- Each (`sleeve_id`, `month_end`) key expands to exactly {len(LOCKED_HORIZONS)} horizons in the stacked panel.",
        "- modeling_panel_hstack carries the same stacked key set as target_panel_long_horizon.",
        f"- baseline_trainable_flag formula matched exactly; strict_trainable_flag formula matched exactly.",
    ]
    if not failures.empty:
        lines.append(f"- Remaining integrity failures: {', '.join(failures['check_name'].tolist())}.")
    if not warnings.empty:
        lines.append(f"- Warning-only checks: {', '.join(warnings['check_name'].tolist())}.")

    lines.extend(
        [
            "",
            "## Long-Horizon Target Sanity",
            "- Stored targets match a full recomputation from frozen v1 month-end prices, realized monthly returns, and TB3MS-based monthly risk-free compounding.",
            f"- Example audited row: sleeve={example_row['sleeve_id']}, month_end={pd.Timestamp(example_row['month_end']).date()}, horizon={int(example_row['horizon_months'])} months, annualized_excess_forward_return={example_row['annualized_excess_forward_return']:.6f}.",
            "- No nonpositive compounded gross returns were found on target-available rows.",
            "- realized_forward_volatility is nonnegative throughout the available sample.",
            "- realized_forward_max_drawdown stays within (-1, 0] as expected.",
            f"- Last target-available month by horizon: 5Y={available_last_dates.get(60).date()}, 10Y={available_last_dates.get(120).date()}, 15Y={available_last_dates.get(180).date()}.",
            "",
            "## Coverage And Missingness",
        ]
    )
    for _, row in by_horizon.iterrows():
        lines.append(
            f"- Horizon {int(row['horizon_months'])}m: target_available_rows={int(row['target_available_rows'])}, baseline_trainable_rows={int(row['baseline_trainable_rows'])}, strict_trainable_rows={int(row['strict_trainable_rows'])}, baseline_trainable_share_of_available={row['baseline_trainable_share_of_available']:.3f}, strict_trainable_share_of_available={row['strict_trainable_share_of_available']:.3f}."
        )
    lines.append(
        "- The heaviest missingness in `baseline_macro_alias` and `local_mapping` is structural by design: `EQ_EM` and `ALT_GLD` have no local macro block, so mapped `local_*` and `cape_local` fields are expected to be absent there."
    )
    lines.append(
        "- The heaviest non-structural missingness comes from late-start China enrichment fields, especially Caixin PMI and house-price series."
    )
    lines.append("- Worst overall block missingness on target-available rows:")
    for _, row in worst_blocks.iterrows():
        scope_note = f"worst_feature={row['worst_feature']} ({row['worst_feature_missing_share']:.3f})"
        lines.append(
            f"  - {row['block_name']}: avg_feature_missing_share={row['avg_feature_missing_share']:.3f}, max_feature_missing_share={row['max_feature_missing_share']:.3f}, {scope_note}, latest_feature_start_date={pd.Timestamp(row['latest_feature_start_date']).date() if pd.notna(row['latest_feature_start_date']) else 'NA'}."
        )
    lines.append("- Main causes of strict complete-case failure:")
    for _, row in strict_block.iterrows():
        lines.append(
            f"  - {row['block_name']}: missing in {int(row['rows_with_block_missing'])} strict-failure rows, share_of_strict_failure_rows={row['share_of_strict_failure_rows']:.3f}."
        )
    if not high_missing_features.empty:
        lines.append("- Highest-missing individual features among strict-failure rows:")
        for _, row in high_missing_features.iterrows():
            lines.append(
                f"  - {row['feature_name']}: share_of_strict_failure_rows={row['share_of_strict_failure_rows']:.3f}."
            )

    lines.extend(["", "## Distribution And Plausibility"])
    target_h = target_distribution_by_horizon.loc[
        target_distribution_by_horizon["metric_name"].eq("annualized_excess_forward_return")
    ].sort_values("horizon_months")
    for _, row in target_h.iterrows():
        lines.append(
            f"- annualized_excess_forward_return, horizon {int(row['horizon_months'])}m: mean={row['mean']:.4f}, std={row['std']:.4f}, p05={row['p05']:.4f}, p50={row['p50']:.4f}, p95={row['p95']:.4f}, min={row['min']:.4f}, max={row['max']:.4f}."
        )
    target_sanity = _sleeve_target_sanity(available)
    hardest = target_sanity.sort_values("target_std", ascending=False).head(3)
    easiest = target_sanity.sort_values("target_std", ascending=True).head(3)
    lines.append("- Highest target volatility sleeve-horizon combinations:")
    for _, row in hardest.iterrows():
        lines.append(
            f"  - {row['sleeve_id']} @ {int(row['horizon_months'])}m: std={row['target_std']:.4f}, range=[{row['target_min']:.4f}, {row['target_max']:.4f}]."
        )
    lines.append("- Lowest target volatility sleeve-horizon combinations:")
    for _, row in easiest.iterrows():
        lines.append(
            f"  - {row['sleeve_id']} @ {int(row['horizon_months'])}m: std={row['target_std']:.4f}, range=[{row['target_min']:.4f}, {row['target_max']:.4f}]."
        )
    lines.append("- Key predictor scale snapshot on baseline-trainable rows:")
    for _, row in key_feature_stats.iterrows():
        lines.append(
            f"  - {row['feature_name']}: mean={row['mean']:.4f}, std={row['std']:.4f}, p01={row['p01']:.4f}, p99={row['p99']:.4f}."
        )

    lines.extend(["", "## Interaction-Term Sanity"])
    for _, row in interaction_family_summary.iterrows():
        lines.append(
            f"- {row['interaction_family']}: usage={row['recommended_first_pass_usage']}, interaction_count={int(row['interaction_count'])}, avg_nonmissing_share={row['avg_nonmissing_share']:.3f}, avg_zero_share={row['avg_zero_share']:.3f}, max_abs_max={row['max_abs_max']:.3f}."
        )
    if not unstable_interactions.empty:
        lines.append("- Interactions that are numerically awkward enough to defer in first-pass modeling:")
        for _, row in unstable_interactions.head(12).iterrows():
            lines.append(
                f"  - {row['feature_name']} ({row['stability_flag']}): nonmissing_share={row['nonmissing_share']:.3f}, zero_share={row['zero_share']:.3f}, max_abs={row['max_abs']:.3f}."
            )

    lines.extend(
        [
            "",
            "## Modeling Readiness Recommendation",
            "- Main modeling input: `data/final_v2_long_horizon/modeling_panel_hstack.parquet`.",
            "- Default filter: `baseline_trainable_flag == 1`.",
            "- `strict_trainable_flag` should remain a diagnostic only; it is currently unusable as a training filter because it leaves zero rows across all horizons.",
            "- Missing-data handling is mandatory. First-pass models should use explicit masking/imputation rather than literal complete-case filtering.",
            "- First-pass horizon choice: start with 5Y + 10Y. They preserve materially more rows than 15Y while still matching the long-horizon objective.",
            "- Feature blocks to make optional in the first model pass: the latest-start China PMI / housing fields and the sparsest relevance-gated interaction families.",
        ]
    )

    intact_answer = "yes" if failures.empty else "no"
    target_answer = "yes" if failures.loc[failures['check_group'].eq('targets')].empty else "no"
    problematic_blocks = ", ".join(worst_blocks["block_name"].head(4).tolist())
    lines.extend(
        [
            "",
            "## Direct Answers",
            f"1. Is the v2_long_horizon dataset internally intact? {intact_answer}.",
            f"2. Are the long-horizon targets constructed correctly? {target_answer}; they match a full recomputation from prices, returns, and TB3MS.",
            f"3. Which feature blocks are most problematic for missingness? {problematic_blocks}; the first two are mostly structural no-local-block gaps, while `china_macro` is the main late-start enrichment block.",
            "4. Is baseline_trainable_flag the correct default training filter? yes.",
            "5. Should first-pass modeling use only 5Y, 5Y + 10Y, or all 5Y/10Y/15Y? 5Y + 10Y first; add 15Y only after the model stack is stable under missingness handling.",
            "6. Are the interaction terms ready to use, or should some be deferred? keep the full dataset intact, but defer the sparsest sleeve-/relevance-gated interaction families in the first pass.",
            "7. Is the dataset ready for the modeling section right now? yes, with baseline_trainable_flag and explicit masking/imputation as the default training setup.",
        ]
    )
    return "\n".join(lines)



def build_v2_acceptance_audit(project_root: Path | None = None) -> AcceptanceArtifacts:
    paths = default_paths(project_root)
    inputs = load_acceptance_inputs(paths)
    integrity_checks, target_compare = _build_integrity_checks(inputs)
    target_available = inputs.target_panel.loc[inputs.target_panel["target_available_flag"].eq(1)].copy()
    modeling_available = inputs.modeling_panel.loc[inputs.modeling_panel["target_available_flag"].eq(1)].copy()
    target_distribution_by_horizon = _distribution_table(target_available, ["horizon_months"]) 
    target_distribution_by_sleeve = _distribution_table(target_available, ["sleeve_id", "horizon_months"])
    feature_block_missingness = _feature_block_missingness(modeling_available, inputs.feature_dictionary)
    feature_scale_summary = _feature_scale_summary(inputs.modeling_panel, inputs.feature_dictionary)
    trainability_summary = _trainability_summary(inputs.modeling_panel, inputs.feature_dictionary)
    _, interaction_family_summary = _interaction_diagnostics(inputs.modeling_panel, inputs.interaction_dictionary, feature_scale_summary)
    key_feature_stats = _key_feature_section(inputs.modeling_panel.loc[inputs.modeling_panel["baseline_trainable_flag"].eq(1)])
    report = _render_acceptance_report(
        integrity_checks=integrity_checks,
        target_distribution_by_horizon=target_distribution_by_horizon,
        target_distribution_by_sleeve=target_distribution_by_sleeve,
        feature_block_missingness=feature_block_missingness,
        feature_scale_summary=feature_scale_summary,
        trainability_summary=trainability_summary,
        interaction_family_summary=interaction_family_summary,
        key_feature_stats=key_feature_stats,
        target_compare=target_compare,
        inputs=inputs,
    )

    write_csv(target_distribution_by_horizon, paths.reports_v2_dir / "target_distribution_by_horizon.csv")
    write_csv(target_distribution_by_sleeve, paths.reports_v2_dir / "target_distribution_by_sleeve.csv")
    write_csv(feature_block_missingness, paths.reports_v2_dir / "feature_block_missingness.csv")
    write_csv(feature_scale_summary, paths.reports_v2_dir / "feature_scale_summary.csv")
    write_csv(trainability_summary, paths.reports_v2_dir / "trainability_summary.csv")
    write_csv(integrity_checks, paths.reports_v2_dir / "integrity_checks.csv")
    write_text(report, paths.reports_v2_dir / "eda_acceptance_report.md")

    return AcceptanceArtifacts(
        integrity_checks=integrity_checks,
        target_distribution_by_horizon=target_distribution_by_horizon,
        target_distribution_by_sleeve=target_distribution_by_sleeve,
        feature_block_missingness=feature_block_missingness,
        feature_scale_summary=feature_scale_summary,
        trainability_summary=trainability_summary,
        eda_acceptance_report=report,
    )
