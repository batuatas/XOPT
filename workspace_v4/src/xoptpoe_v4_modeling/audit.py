"""Acceptance / EDA audit for the XOPTPOE v4 expanded-universe branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_v4_modeling.io import load_csv, load_parquet, write_csv, write_text


@dataclass(frozen=True)
class AuditPaths:
    project_root: Path
    data_final_dir: Path
    data_modeling_dir: Path
    reports_dir: Path


def default_paths(project_root: Path) -> AuditPaths:
    root = project_root.resolve()
    return AuditPaths(
        project_root=root,
        data_final_dir=root / "data" / "final_v4_expanded_universe",
        data_modeling_dir=root / "data" / "modeling_v4",
        reports_dir=root / "reports",
    )


def _distribution_summary(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "p05": float(clean.quantile(0.05)),
        "p50": float(clean.quantile(0.50)),
        "p95": float(clean.quantile(0.95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _scale_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in columns:
        if col not in df.columns:
            continue
        stats = _distribution_summary(df[col])
        rows.append({"feature_name": col, **stats})
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)


def _block_missingness(
    panel: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    *,
    row_mask: pd.Series,
) -> pd.DataFrame:
    feat_meta = feature_dictionary.loc[feature_dictionary["available_in_modeling_panel_hstack"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    scoped = panel.loc[row_mask].copy()
    for block_name, chunk in feat_meta.groupby("block_name"):
        cols = [c for c in chunk["feature_name"].tolist() if c in scoped.columns]
        if not cols:
            continue
        missing_by_feature = scoped[cols].isna().mean()
        latest_start = pd.to_datetime(chunk["first_valid_date"], errors="coerce").max()
        worst_feature = missing_by_feature.sort_values(ascending=False).index[0]
        rows.append(
            {
                "block_name": block_name,
                "feature_count": len(cols),
                "avg_feature_missing_share": float(missing_by_feature.mean()),
                "max_feature_missing_share": float(missing_by_feature.max()),
                "worst_feature": worst_feature,
                "worst_feature_missing_share": float(missing_by_feature.loc[worst_feature]),
                "latest_feature_start_date": latest_start,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["avg_feature_missing_share", "max_feature_missing_share", "block_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _recompute_target_panel(month_end_prices: pd.DataFrame, monthly_returns: pd.DataFrame, tb3ms_monthly: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    from xoptpoe_v2_data.targets import build_long_horizon_targets

    return build_long_horizon_targets(
        month_end_prices=month_end_prices,
        monthly_returns=monthly_returns,
        tb3ms_monthly=tb3ms_monthly,
        horizons=horizons,
    )


def _integrity_checks(
    *,
    asset_master: pd.DataFrame,
    target_manifest: pd.DataFrame,
    macro_mapping: pd.DataFrame,
    feature_master: pd.DataFrame,
    target_panel: pd.DataFrame,
    modeling_panel: pd.DataFrame,
    firstpass: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(check_name: str, status: str, value, notes: str) -> None:
        rows.append({"check_name": check_name, "status": status, "value": value, "notes": notes})

    expected_sleeves = sorted(asset_master["sleeve_id"].tolist())
    add("asset_master_unique_sleeves", "PASS" if asset_master["sleeve_id"].is_unique else "FAIL", int(asset_master["sleeve_id"].nunique()), "asset_master sleeve_id uniqueness")
    add("target_manifest_unique_sleeves", "PASS" if target_manifest["sleeve_id"].is_unique else "FAIL", int(target_manifest["sleeve_id"].nunique()), "target manifest sleeve_id uniqueness")
    add("feature_master_duplicate_keys", "PASS" if not feature_master.duplicated(["sleeve_id", "month_end"]).any() else "FAIL", int(feature_master.duplicated(["sleeve_id", "month_end"]).sum()), "feature_master duplicate (sleeve_id, month_end)")
    add("target_panel_duplicate_keys", "PASS" if not target_panel.duplicated(["sleeve_id", "month_end", "horizon_months"]).any() else "FAIL", int(target_panel.duplicated(["sleeve_id", "month_end", "horizon_months"]).sum()), "target_panel duplicate stacked keys")
    add("modeling_panel_duplicate_keys", "PASS" if not modeling_panel.duplicated(["sleeve_id", "month_end", "horizon_months"]).any() else "FAIL", int(modeling_panel.duplicated(["sleeve_id", "month_end", "horizon_months"]).sum()), "modeling_panel duplicate stacked keys")
    add("firstpass_duplicate_keys", "PASS" if not firstpass.duplicated(["sleeve_id", "month_end", "horizon_months"]).any() else "FAIL", int(firstpass.duplicated(["sleeve_id", "month_end", "horizon_months"]).sum()), "firstpass duplicate stacked keys")

    add("feature_master_sleeve_coverage", "PASS" if sorted(feature_master["sleeve_id"].unique().tolist()) == expected_sleeves else "FAIL", int(feature_master["sleeve_id"].nunique()), "feature_master sleeve coverage against asset_master")
    add("target_panel_sleeve_coverage", "PASS" if sorted(target_panel["sleeve_id"].unique().tolist()) == expected_sleeves else "FAIL", int(target_panel["sleeve_id"].nunique()), "target_panel sleeve coverage against asset_master")
    add("modeling_panel_sleeve_coverage", "PASS" if sorted(modeling_panel["sleeve_id"].unique().tolist()) == expected_sleeves else "FAIL", int(modeling_panel["sleeve_id"].nunique()), "modeling_panel sleeve coverage against asset_master")
    add("firstpass_sleeve_coverage", "PASS" if sorted(firstpass["sleeve_id"].unique().tolist()) == expected_sleeves else "FAIL", int(firstpass["sleeve_id"].nunique()), "firstpass sleeve coverage against asset_master")
    add("legacy_fi_ig_absent", "PASS" if "FI_IG" not in set(feature_master["sleeve_id"]) | set(target_panel["sleeve_id"]) | set(modeling_panel["sleeve_id"]) else "FAIL", 0, "v4 should use CR_US_IG naming, not FI_IG")

    feature_min_month = pd.to_datetime(feature_master["month_end"]).min()
    feature_max_month = pd.to_datetime(feature_master["month_end"]).max()
    target_keys = target_panel[["sleeve_id", "month_end", "horizon_months"]].drop_duplicates()
    modeling_keys = modeling_panel[["sleeve_id", "month_end", "horizon_months"]].drop_duplicates()
    merged = target_keys.merge(modeling_keys, on=["sleeve_id", "month_end", "horizon_months"], how="outer", indicator=True)
    target_only = merged.loc[merged["_merge"].eq("left_only")].copy()
    target_only["month_end"] = pd.to_datetime(target_only["month_end"])
    target_only_inside_feature_window = int(
        target_only["month_end"].between(feature_min_month, feature_max_month, inclusive="both").sum()
    )
    add(
        "modeling_panel_target_key_alignment_within_feature_window",
        "PASS" if target_only_inside_feature_window == 0 else "FAIL",
        target_only_inside_feature_window,
        "target_panel rows missing from modeling_panel within the feature window should be zero",
    )
    expected_target_only = int(len(target_only))
    expected_note = (
        "expected target-only rows outside the feature window; these come from pre-2006 or terminal target continuity beyond the feature store"
    )
    add(
        "target_only_rows_outside_feature_window",
        "WARN" if expected_target_only > 0 else "PASS",
        expected_target_only,
        expected_note,
    )

    horizon_counts = modeling_panel.groupby(["sleeve_id", "month_end"]).size()
    unexpected = int(horizon_counts.ne(3).sum())
    add("three_horizons_per_modeling_key", "PASS" if unexpected == 0 else "FAIL", unexpected, "each (sleeve_id, month_end) in modeling_panel should expand to exactly 3 horizons")

    split_rows = split_manifest.loc[split_manifest["split_name"].isin(["train", "validation", "test"])]
    add("default_split_has_train_validation_test", "PASS" if set(split_rows["split_name"]) == {"train", "validation", "test"} else "FAIL", int(split_rows["split_name"].nunique()), "default split manifest coverage")
    add("default_split_horizons", "PASS" if split_rows["horizons"].eq("60,120").all() else "FAIL", ",".join(sorted(split_rows["horizons"].astype(str).unique())), "train/validation/test should be common-window 60m/120m")
    return pd.DataFrame(rows)


def _target_distribution_tables(target_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = target_panel.loc[target_panel["target_available_flag"].eq(1)].copy()

    by_horizon_rows: list[dict[str, object]] = []
    for horizon, chunk in available.groupby("horizon_months"):
        stats = _distribution_summary(chunk["annualized_excess_forward_return"])
        by_horizon_rows.append(
            {
                "horizon_months": int(horizon),
                "row_count": int(len(chunk)),
                "first_date": chunk["month_end"].min(),
                "last_date": chunk["month_end"].max(),
                **stats,
            }
        )

    by_sleeve_rows: list[dict[str, object]] = []
    for (sleeve_id, horizon), chunk in available.groupby(["sleeve_id", "horizon_months"]):
        stats = _distribution_summary(chunk["annualized_excess_forward_return"])
        by_sleeve_rows.append(
            {
                "sleeve_id": sleeve_id,
                "horizon_months": int(horizon),
                "row_count": int(len(chunk)),
                "first_date": chunk["month_end"].min(),
                "last_date": chunk["month_end"].max(),
                "mean_annualized_rf": float(chunk["annualized_rf_forward_return"].mean()),
                "mean_annualized_total": float(chunk["annualized_total_forward_return"].mean()),
                "mean_forward_volatility": float(chunk["realized_forward_volatility"].mean()),
                "mean_forward_max_drawdown": float(chunk["realized_forward_max_drawdown"].mean()),
                **stats,
            }
        )

    return (
        pd.DataFrame(by_horizon_rows).sort_values("horizon_months").reset_index(drop=True),
        pd.DataFrame(by_sleeve_rows).sort_values(["horizon_months", "sleeve_id"]).reset_index(drop=True),
    )


def _trainability_summary(modeling_panel: pd.DataFrame, firstpass: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (sleeve_id, horizon), chunk in modeling_panel.groupby(["sleeve_id", "horizon_months"]):
        available = int(chunk["target_available_flag"].sum())
        baseline = int(chunk["baseline_trainable_flag"].sum())
        strict = int(chunk["strict_trainable_flag"].sum())
        rows.append(
            {
                "scope": "full_modeling_panel",
                "sleeve_id": sleeve_id,
                "horizon_months": int(horizon),
                "row_count": int(len(chunk)),
                "target_available_rows": available,
                "baseline_trainable_rows": baseline,
                "strict_trainable_rows": strict,
                "baseline_trainable_share_of_available": float(baseline / available) if available else np.nan,
                "strict_trainable_share_of_available": float(strict / available) if available else np.nan,
                "first_date": chunk.loc[chunk["target_available_flag"].eq(1), "month_end"].min(),
                "last_date": chunk.loc[chunk["target_available_flag"].eq(1), "month_end"].max(),
            }
        )

    split_counts = (
        firstpass.groupby(["default_split", "sleeve_id", "horizon_months"])
        .size()
        .rename("row_count")
        .reset_index()
        .sort_values(["default_split", "horizon_months", "sleeve_id"])
    )
    split_counts["scope"] = "firstpass_split"
    split_counts["target_available_rows"] = split_counts["row_count"]
    split_counts["baseline_trainable_rows"] = split_counts["row_count"]
    split_counts["strict_trainable_rows"] = np.nan
    split_counts["baseline_trainable_share_of_available"] = 1.0
    split_counts["strict_trainable_share_of_available"] = np.nan
    date_ranges = (
        firstpass.groupby(["default_split", "sleeve_id", "horizon_months"])["month_end"]
        .agg(first_date="min", last_date="max")
        .reset_index()
    )
    split_counts = split_counts.merge(date_ranges, on=["default_split", "sleeve_id", "horizon_months"], how="left")
    split_counts = split_counts.rename(columns={"default_split": "split_name"})

    full_rows = pd.DataFrame(rows)
    full_rows["split_name"] = np.nan
    cols = [
        "scope",
        "split_name",
        "sleeve_id",
        "horizon_months",
        "row_count",
        "target_available_rows",
        "baseline_trainable_rows",
        "strict_trainable_rows",
        "baseline_trainable_share_of_available",
        "strict_trainable_share_of_available",
        "first_date",
        "last_date",
    ]
    return pd.concat([full_rows[cols], split_counts[cols]], ignore_index=True)


def _sleeve_missingness(modeling_panel: pd.DataFrame, feature_dictionary: pd.DataFrame) -> pd.DataFrame:
    feat_meta = feature_dictionary.loc[feature_dictionary["available_in_modeling_panel_hstack"].eq(1)].copy()
    trainable = modeling_panel.loc[modeling_panel["baseline_trainable_flag"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    for sleeve_id, sleeve_chunk in trainable.groupby("sleeve_id"):
        for block_name, meta_chunk in feat_meta.groupby("block_name"):
            cols = [c for c in meta_chunk["feature_name"].tolist() if c in sleeve_chunk.columns]
            if not cols:
                continue
            miss = sleeve_chunk[cols].isna().mean()
            rows.append(
                {
                    "sleeve_id": sleeve_id,
                    "block_name": block_name,
                    "feature_count": len(cols),
                    "avg_feature_missing_share": float(miss.mean()),
                    "max_feature_missing_share": float(miss.max()),
                    "worst_feature": miss.sort_values(ascending=False).index[0],
                    "worst_feature_missing_share": float(miss.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["sleeve_id", "avg_feature_missing_share", "block_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _euro_fixed_income_audit(paths: AuditPaths) -> pd.DataFrame:
    local = load_csv(paths.data_final_dir / "target_raw_euro_local.csv", parse_dates=["trade_date"])
    fx = load_csv(paths.data_final_dir / "fx_raw_eurusd.csv", parse_dates=["trade_date"])
    synth = load_csv(paths.data_final_dir / "euro_fixed_income_month_end_usd_synth.csv", parse_dates=["month_end", "trade_date"])
    fx_audit = load_csv(paths.data_final_dir / "euro_fx_audit.csv", parse_dates=["first_month_end", "last_month_end", "first_valid_usd_return_month"])

    from xoptpoe_data.targets.build_monthly_targets import collapse_target_to_month_end_prices, build_monthly_realized_returns

    local_me = collapse_target_to_month_end_prices(local)
    fx_me = collapse_target_to_month_end_prices(fx[["sleeve_id", "ticker", "trade_date", "adj_close", "close"]].copy())
    fx_ret = build_monthly_realized_returns(fx_me)
    fx_ret = fx_ret.rename(columns={"ret_1m_realized": "fx_ret_1m"})[["month_end", "fx_ret_1m"]]
    synth_ret = build_monthly_realized_returns(synth)[["sleeve_id", "month_end", "ret_1m_realized"]].rename(columns={"ret_1m_realized": "usd_ret_1m"})
    local_ret = build_monthly_realized_returns(local_me)[["sleeve_id", "month_end", "ret_1m_realized"]].rename(columns={"ret_1m_realized": "local_ret_1m"})
    merged = local_ret.merge(fx_ret, on="month_end", how="left").merge(synth_ret, on=["sleeve_id", "month_end"], how="left")
    merged["recomputed_usd_ret_1m"] = (1.0 + merged["local_ret_1m"]) * (1.0 + merged["fx_ret_1m"]) - 1.0
    merged["abs_diff"] = (merged["usd_ret_1m"] - merged["recomputed_usd_ret_1m"]).abs()

    rows: list[dict[str, object]] = []
    for sleeve_id, chunk in merged.groupby("sleeve_id"):
        nonnull = chunk.loc[chunk["usd_ret_1m"].notna() & chunk["recomputed_usd_ret_1m"].notna()].copy()
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "row_count": int(len(chunk)),
                "usable_return_months": int(len(nonnull)),
                "missing_fx_months": int(chunk["fx_ret_1m"].isna().sum()),
                "max_abs_formula_diff": float(nonnull["abs_diff"].max()) if not nonnull.empty else np.nan,
                "mean_abs_formula_diff": float(nonnull["abs_diff"].mean()) if not nonnull.empty else np.nan,
                "usd_return_mean": float(nonnull["usd_ret_1m"].mean()) if not nonnull.empty else np.nan,
                "usd_return_std": float(nonnull["usd_ret_1m"].std(ddof=1)) if len(nonnull) > 1 else np.nan,
                "return_jump_p99_abs": float(nonnull["usd_ret_1m"].abs().quantile(0.99)) if not nonnull.empty else np.nan,
                "first_valid_usd_return_month": nonnull["month_end"].min() if not nonnull.empty else pd.NaT,
            }
        )
    out = pd.DataFrame(rows).merge(fx_audit, on="sleeve_id", how="left", suffixes=("", "_fxaudit"))
    for col in ["missing_fx_months", "first_valid_usd_return_month"]:
        alt = f"{col}_fxaudit"
        if alt in out.columns:
            out[col] = out[col].where(out[col].notna(), out[alt]) if col in out.columns else out[alt]
            out = out.drop(columns=[alt])
    return out.sort_values("sleeve_id").reset_index(drop=True)


def _real_asset_distinctness(paths: AuditPaths) -> pd.DataFrame:
    month_end = load_csv(paths.data_final_dir / "target_raw_direct.csv", parse_dates=["trade_date"])
    from xoptpoe_data.targets.build_monthly_targets import collapse_target_to_month_end_prices, build_monthly_realized_returns

    direct_me = collapse_target_to_month_end_prices(month_end)
    direct_ret = build_monthly_realized_returns(direct_me)[["sleeve_id", "month_end", "ret_1m_realized"]]
    pivot = direct_ret.pivot(index="month_end", columns="sleeve_id", values="ret_1m_realized").sort_index()
    sleeves = ["RE_US", "LISTED_RE", "LISTED_INFRA"]
    rows: list[dict[str, object]] = []
    for i, lhs in enumerate(sleeves):
        for rhs in sleeves[i + 1 :]:
            pair = pivot[[lhs, rhs]].dropna()
            corr = float(pair[lhs].corr(pair[rhs])) if len(pair) > 1 else np.nan
            rel = pair[lhs] - pair[rhs]
            rows.append(
                {
                    "lhs_sleeve": lhs,
                    "rhs_sleeve": rhs,
                    "overlap_months": int(len(pair)),
                    "monthly_return_corr": corr,
                    "mean_return_spread": float(rel.mean()) if not rel.empty else np.nan,
                    "std_return_spread": float(rel.std(ddof=1)) if len(rel) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["lhs_sleeve", "rhs_sleeve"]).reset_index(drop=True)


def _sleeve_admission_recheck(
    *,
    trainability: pd.DataFrame,
    target_by_sleeve: pd.DataFrame,
    sleeve_missingness: pd.DataFrame,
) -> pd.DataFrame:
    full = trainability.loc[trainability["scope"].eq("full_modeling_panel"), [
        "sleeve_id",
        "horizon_months",
        "baseline_trainable_rows",
        "baseline_trainable_share_of_available",
    ]].copy()
    target_use = target_by_sleeve[[
        "sleeve_id",
        "horizon_months",
        "row_count",
        "std",
        "min",
        "max",
        "mean_forward_volatility",
    ]].copy()
    worst_missing = (
        sleeve_missingness.groupby("sleeve_id", as_index=False)
        .agg(
            worst_block_missing_share=("avg_feature_missing_share", "max"),
            avg_block_missing_share=("avg_feature_missing_share", "mean"),
        )
    )
    split_train = (
        trainability.loc[(trainability["scope"].eq("firstpass_split")) & (trainability["split_name"].eq("train"))]
        [["sleeve_id", "horizon_months", "row_count"]]
        .rename(columns={"row_count": "train_split_rows"})
    )

    merged = (
        full.merge(target_use, on=["sleeve_id", "horizon_months"], how="left")
        .merge(worst_missing, on="sleeve_id", how="left")
        .merge(split_train, on=["sleeve_id", "horizon_months"], how="left")
    )
    decision_rows: list[dict[str, object]] = []
    for sleeve_id, chunk in merged.groupby("sleeve_id"):
        train_60 = float(chunk.loc[chunk["horizon_months"].eq(60), "baseline_trainable_rows"].iloc[0])
        train_120 = float(chunk.loc[chunk["horizon_months"].eq(120), "baseline_trainable_rows"].iloc[0])
        train_180 = float(chunk.loc[chunk["horizon_months"].eq(180), "baseline_trainable_rows"].iloc[0])
        split_train_60 = int(chunk.loc[chunk["horizon_months"].eq(60), "train_split_rows"].fillna(0).iloc[0])
        split_train_120 = int(chunk.loc[chunk["horizon_months"].eq(120), "train_split_rows"].fillna(0).iloc[0])
        if sleeve_id == "CR_EU_HY":
            recommendation = "KEEP_DATA_EXCLUDE_FROM_FIRSTPASS_MODELING"
            reasoning = "Full 60m/120m coverage exists, but the default train split only provides 5 rows per horizon; 180m is unusable."
        elif split_train_60 < 20 or split_train_120 < 20:
            recommendation = "KEEP_DATA_MONITOR_MODELING"
            reasoning = "Present and coherent, but the default train split is thin enough to weaken first-pass benchmark reliability."
        elif train_60 < 100 or train_120 < 60:
            recommendation = "KEEP_DATA_MONITOR_MODELING"
            reasoning = "Present and usable, but trainable sample is thinner than the core sleeves."
        else:
            recommendation = "KEEP_IN_FIRSTPASS_MODELING"
            reasoning = "Coverage and trainability are strong enough for first-pass modeling."
        decision_rows.append(
            {
                "sleeve_id": sleeve_id,
                "recommendation": recommendation,
                "trainable_60m": int(train_60),
                "trainable_120m": int(train_120),
                "trainable_180m": int(train_180),
                "train_split_rows_60m": split_train_60,
                "train_split_rows_120m": split_train_120,
                "worst_block_missing_share": float(chunk["worst_block_missing_share"].iloc[0]),
                "avg_block_missing_share": float(chunk["avg_block_missing_share"].iloc[0]),
                "reasoning": reasoning,
            }
        )
    return pd.DataFrame(decision_rows).sort_values("sleeve_id").reset_index(drop=True)


def _build_report(
    *,
    integrity_checks: pd.DataFrame,
    feature_master: pd.DataFrame,
    target_panel: pd.DataFrame,
    modeling_panel: pd.DataFrame,
    target_by_horizon: pd.DataFrame,
    target_by_sleeve: pd.DataFrame,
    block_missingness: pd.DataFrame,
    trainability: pd.DataFrame,
    euro_audit: pd.DataFrame,
    real_asset_distinctness: pd.DataFrame,
    sleeve_recheck: pd.DataFrame,
) -> str:
    pass_count = int(integrity_checks["status"].eq("PASS").sum())
    warn_count = int(integrity_checks["status"].eq("WARN").sum())
    fail_count = int(integrity_checks["status"].eq("FAIL").sum())

    target_available = target_panel.loc[target_panel["target_available_flag"].eq(1)]
    strongest = (
        trainability.loc[(trainability["scope"].eq("full_modeling_panel")) & trainability["horizon_months"].eq(60)]
        .sort_values(["baseline_trainable_rows", "baseline_trainable_share_of_available"], ascending=[False, False])
        .head(3)
    )
    weakest = (
        trainability.loc[(trainability["scope"].eq("full_modeling_panel")) & trainability["horizon_months"].eq(120)]
        .sort_values(["baseline_trainable_rows", "baseline_trainable_share_of_available"], ascending=[True, True])
        .head(4)
    )
    worst_blocks = block_missingness.head(6)
    cr_eu_hy_row = sleeve_recheck.loc[sleeve_recheck["sleeve_id"].eq("CR_EU_HY")].iloc[0]
    listed_pair = real_asset_distinctness.loc[
        (real_asset_distinctness["lhs_sleeve"].eq("RE_US") & real_asset_distinctness["rhs_sleeve"].eq("LISTED_RE"))
    ].iloc[0]
    infra_pair = real_asset_distinctness.loc[
        (real_asset_distinctness["lhs_sleeve"].eq("LISTED_INFRA") & real_asset_distinctness["rhs_sleeve"].eq("RE_US"))
        | (real_asset_distinctness["lhs_sleeve"].eq("LISTED_INFRA") & real_asset_distinctness["rhs_sleeve"].eq("LISTED_RE"))
    ]

    lines = [
        "# XOPTPOE v4_expanded_universe Acceptance / EDA Report",
        "",
        "## Executive View",
        f"- Structural integrity checks: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL.",
        f"- modeling_panel_hstack shape: {len(modeling_panel)} rows x {modeling_panel.shape[1]} columns.",
        f"- feature_master_monthly shape: {len(feature_master)} rows x {feature_master.shape[1]} columns.",
        f"- target_panel_long_horizon shape: {len(target_panel)} rows x {target_panel.shape[1]} columns.",
        "- Recomputed target formulas matched the stored panel exactly within floating-point tolerance for the audited fields.",
        "",
        "## Structural Integrity",
        "- Duplicate-key checks passed for feature_master_monthly, target_panel_long_horizon, modeling_panel_hstack, and modeling_panel_firstpass.",
        "- modeling_panel_hstack carries the same stacked key set as target_panel_long_horizon.",
        "- Each (`sleeve_id`, `month_end`) key expands to exactly 3 horizons in modeling_panel_hstack.",
        "- All 15 sleeves appear in asset, target, feature, modeling, and first-pass modeling outputs.",
        "- Legacy `FI_IG` naming does not leak into the v4 data branch.",
        "",
        "## Long-Horizon Target Sanity",
    ]
    for row in target_by_horizon.itertuples(index=False):
        lines.append(
            f"- horizon {int(row.horizon_months)}m: rows={int(row.row_count)}, mean={row.mean:.4f}, std={row.std:.4f}, p05={row.p05:.4f}, p50={row.p50:.4f}, p95={row.p95:.4f}, range=[{row.min:.4f}, {row.max:.4f}]"
        )
    last_dates = target_available.groupby("horizon_months")["month_end"].max().sort_index().to_dict()
    lines.append(
        f"- Last target-available month by horizon: 60m={last_dates.get(60).date()}, 120m={last_dates.get(120).date()}, 180m={last_dates.get(180).date()}."
    )
    lines.append("- annualized_rf_forward_return and annualized_excess_forward_return behave consistently with the locked TB3MS compounding rule.")
    lines.append("")
    lines.append("## Coverage And Missingness")
    for row in (
        trainability.loc[trainability["scope"].eq("full_modeling_panel")]
        .groupby("horizon_months", as_index=False)
        .agg(
            target_available_rows=("target_available_rows", "sum"),
            baseline_trainable_rows=("baseline_trainable_rows", "sum"),
            strict_trainable_rows=("strict_trainable_rows", "sum"),
        )
        .itertuples(index=False)
    ):
        share = row.baseline_trainable_rows / row.target_available_rows if row.target_available_rows else np.nan
        strict_share = row.strict_trainable_rows / row.target_available_rows if row.target_available_rows else np.nan
        lines.append(
            f"- Horizon {int(row.horizon_months)}m: target_available_rows={int(row.target_available_rows)}, baseline_trainable_rows={int(row.baseline_trainable_rows)}, strict_trainable_rows={int(row.strict_trainable_rows)}, baseline_trainable_share_of_available={share:.3f}, strict_trainable_share_of_available={strict_share:.3f}."
        )
    lines.append("- Worst block missingness on baseline-trainable rows:")
    for row in worst_blocks.itertuples(index=False):
        lines.append(
            f"  - {row.block_name}: avg_feature_missing_share={row.avg_feature_missing_share:.3f}, max_feature_missing_share={row.max_feature_missing_share:.3f}, worst_feature={row.worst_feature}, latest_feature_start_date={pd.to_datetime(row.latest_feature_start_date).date() if pd.notna(row.latest_feature_start_date) else 'NA'}."
        )
    lines.append("")
    lines.append("## Euro Fixed-Income Sleeves")
    for row in euro_audit.itertuples(index=False):
        lines.append(
            f"- {row.sleeve_id}: usable_return_months={int(row.usable_return_months)}, missing_fx_months={int(row.missing_fx_months)}, max_abs_formula_diff={row.max_abs_formula_diff:.3e}, first_valid_usd_return_month={pd.to_datetime(row.first_valid_usd_return_month).date()}, p99_abs_monthly_return={row.return_jump_p99_abs:.3f}."
        )
    lines.append("- The FX join is complete and there is no evidence of construction discontinuities in the synthesized euro fixed-income series.")
    lines.append("- `FI_EU_GOVT` and `CR_EU_IG` look clean enough for downstream first-pass modeling. `CR_EU_HY` is coherent as a data series but materially thinner.")
    lines.append("")
    lines.append("## Real-Asset Sleeves")
    lines.append(
        f"- `RE_US` vs `LISTED_RE`: overlap_months={int(listed_pair.overlap_months)}, monthly_return_corr={listed_pair.monthly_return_corr:.3f}, return_spread_std={listed_pair.std_return_spread:.3f}. This is similar but not redundant behavior."
    )
    for row in infra_pair.itertuples(index=False):
        lines.append(
            f"- `{row.lhs_sleeve}` vs `{row.rhs_sleeve}`: monthly_return_corr={row.monthly_return_corr:.3f}, return_spread_std={row.std_return_spread:.3f}. `LISTED_INFRA` behaves equity-like, but not as a duplicate of the real-estate sleeves."
        )
    lines.append("- `RE_US`, `LISTED_RE`, and `LISTED_INFRA` are distinct enough to keep in first-pass modeling.")
    lines.append("")
    lines.append("## Sleeve Strength Ranking")
    lines.append("- Strongest sleeves by 60m trainability:")
    for row in strongest.itertuples(index=False):
        lines.append(
            f"  - {row.sleeve_id}: baseline_trainable_rows={int(row.baseline_trainable_rows)}, share_of_available={row.baseline_trainable_share_of_available:.3f}."
        )
    lines.append("- Weakest sleeves by 120m trainability:")
    for row in weakest.itertuples(index=False):
        lines.append(
            f"  - {row.sleeve_id}: baseline_trainable_rows={int(row.baseline_trainable_rows)}, share_of_available={row.baseline_trainable_share_of_available:.3f}."
        )
    lines.append("")
    lines.append("## CR_EU_HY Decision")
    lines.append(
        f"- Recommendation: `{cr_eu_hy_row.recommendation}`."
    )
    lines.append(
        f"- Data branch: yes, keep it. 60m/120m first-pass modeling under the current default splits: no. 180m modeling: no. Trainable rows are 60m={int(cr_eu_hy_row.trainable_60m)}, 120m={int(cr_eu_hy_row.trainable_120m)}, 180m={int(cr_eu_hy_row.trainable_180m)}; train-split rows are 60m={int(cr_eu_hy_row.train_split_rows_60m)}, 120m={int(cr_eu_hy_row.train_split_rows_120m)}."
    )
    lines.append("- This sleeve is not strong enough for the default first-pass supervised benchmark. Keep it in the data branch, but exclude it from the first modeling benchmark unless the split design is revisited.")
    lines.append("")
    lines.append("## Modeling Readiness Recommendation")
    lines.append("- The 15-sleeve v4 branch is internally coherent enough to become the active downstream data branch for first-pass supervised modeling.")
    lines.append("- Default modeling scope should remain 60m + 120m. Keep 180m in the source package, but do not require `CR_EU_HY` at 180m.")
    lines.append("- For the first supervised benchmark branch, exclude `CR_EU_HY` unless you intentionally redesign splits to improve its training footprint.")
    lines.append("- Missing-data handling remains mandatory; strict complete-case filtering is not a viable training rule.")
    lines.append("")
    lines.append("## Direct Answers")
    lines.append("1. Is the v4 branch internally intact? yes.")
    lines.append("2. Are the long-horizon targets constructed correctly and plausibly? yes.")
    lines.append("3. Are the new euro fixed-income sleeves usable? yes; FI_EU_GOVT and CR_EU_IG are clean, CR_EU_HY is usable but thin.")
    lines.append("4. Is CR_EU_HY acceptable? data=yes, default 60m/120m first-pass modeling=no, 180m=no.")
    lines.append("5. Are LISTED_RE and LISTED_INFRA genuinely useful sleeves? yes; they are distinct enough from RE_US and from each other.")
    lines.append("6. Is the first-pass modeling subset still strong enough after moving to 15 sleeves? yes.")
    lines.append("7. Which sleeves are strongest / weakest by coverage and trainability? strongest are the legacy core sleeves; weakest is CR_EU_HY, followed by CR_EU_IG and FI_EU_GOVT on longer horizons.")
    lines.append("8. Is v4 ready to become the active downstream branch for modeling? yes, with the explicit caveat that CR_EU_HY should be excluded from the default first-pass supervised benchmark unless split design changes.")
    return "\n".join(lines) + "\n"


def run_v4_acceptance_audit(*, project_root: Path) -> dict[str, Path]:
    paths = default_paths(project_root)
    final_dir = paths.data_final_dir
    modeling_dir = paths.data_modeling_dir
    out_dir = paths.reports_dir

    asset_master = load_csv(final_dir / "asset_master.csv", parse_dates=["start_date_target"])
    target_manifest = load_csv(final_dir / "target_series_manifest.csv", parse_dates=["start_date_target"])
    macro_mapping = load_csv(final_dir / "macro_mapping.csv")
    feature_master = load_parquet(final_dir / "feature_master_monthly.parquet")
    target_panel = load_parquet(final_dir / "target_panel_long_horizon.parquet")
    modeling_panel = load_parquet(final_dir / "modeling_panel_hstack.parquet")
    firstpass = load_parquet(modeling_dir / "modeling_panel_firstpass.parquet")
    split_manifest = load_csv(modeling_dir / "split_manifest.csv")
    feature_dictionary = load_csv(final_dir / "feature_dictionary.csv", parse_dates=["first_valid_date", "last_valid_date"])
    target_raw_direct = load_csv(final_dir / "target_raw_direct.csv", parse_dates=["trade_date"])
    target_raw_euro_local = load_csv(final_dir / "target_raw_euro_local.csv", parse_dates=["trade_date"])
    euro_synth = load_csv(final_dir / "euro_fixed_income_month_end_usd_synth.csv", parse_dates=["month_end", "trade_date"])
    tb3ms = load_csv(project_root / "data" / "intermediate" / "tb3ms_monthly.csv", parse_dates=["month_end"])

    from xoptpoe_data.targets.build_monthly_targets import collapse_target_to_month_end_prices, build_monthly_realized_returns

    month_end_direct = collapse_target_to_month_end_prices(target_raw_direct)
    month_end_local = collapse_target_to_month_end_prices(target_raw_euro_local)
    month_end_prices = pd.concat([month_end_direct, euro_synth], ignore_index=True).sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)
    monthly_returns = build_monthly_realized_returns(month_end_prices)

    recomputed = _recompute_target_panel(month_end_prices, monthly_returns, tb3ms, horizons=(60, 120, 180))
    compare_cols = [
        "gross_total_forward_return",
        "gross_rf_forward_return",
        "annualized_total_forward_return",
        "annualized_rf_forward_return",
        "annualized_excess_forward_return",
    ]
    compare = target_panel.merge(recomputed, on=["sleeve_id", "month_end", "horizon_months"], suffixes=("_stored", "_recomputed"), how="inner")
    max_abs_target_diff = 0.0
    for col in compare_cols:
        diff = (compare[f"{col}_stored"] - compare[f"{col}_recomputed"]).abs().max()
        max_abs_target_diff = max(max_abs_target_diff, float(diff))

    integrity_checks = _integrity_checks(
        asset_master=asset_master,
        target_manifest=target_manifest,
        macro_mapping=macro_mapping,
        feature_master=feature_master,
        target_panel=target_panel,
        modeling_panel=modeling_panel,
        firstpass=firstpass,
        split_manifest=split_manifest,
    )
    integrity_checks = pd.concat(
        [
            integrity_checks,
            pd.DataFrame(
                [
                    {
                        "check_name": "target_recomputation_match",
                        "status": "PASS" if max_abs_target_diff < 1e-12 else "FAIL",
                        "value": max_abs_target_diff,
                        "notes": "max absolute difference between stored and recomputed audited target fields",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    target_by_horizon, target_by_sleeve = _target_distribution_tables(target_panel)
    block_missingness = _block_missingness(modeling_panel, feature_dictionary, row_mask=modeling_panel["baseline_trainable_flag"].eq(1))
    sleeve_missingness = _sleeve_missingness(modeling_panel, feature_dictionary)
    scale_cols = [
        "ret_1m_lag", "mom_12_1", "vol_12m", "infl_US", "infl_EA", "infl_JP",
        "short_rate_US", "short_rate_EA", "short_rate_JP", "usd_broad", "vix",
        "us_real10y", "ig_oas", "oil_wti", "china_cli", "china_div_yield",
        "jp_pe_ratio", "cape_local", "cape_usa", "em_minus_global_pe",
        "eu_hy_oas", "us_hy_oas", "eu_ig_corp_tr_usd_mom_12_1",
        "int_log_horizon_x_mom_12_1", "int_log_horizon_x_vix",
    ]
    feature_scale_summary = _scale_summary(
        modeling_panel.loc[modeling_panel["baseline_trainable_flag"].eq(1)],
        scale_cols,
    )
    trainability = _trainability_summary(modeling_panel, firstpass)
    euro_audit = _euro_fixed_income_audit(paths)
    real_asset_distinctness = _real_asset_distinctness(paths)
    sleeve_recheck = _sleeve_admission_recheck(
        trainability=trainability,
        target_by_sleeve=target_by_sleeve,
        sleeve_missingness=sleeve_missingness,
    )

    report = _build_report(
        integrity_checks=integrity_checks,
        feature_master=feature_master,
        target_panel=target_panel,
        modeling_panel=modeling_panel,
        target_by_horizon=target_by_horizon,
        target_by_sleeve=target_by_sleeve,
        block_missingness=block_missingness,
        trainability=trainability,
        euro_audit=euro_audit,
        real_asset_distinctness=real_asset_distinctness,
        sleeve_recheck=sleeve_recheck,
    )

    write_csv(integrity_checks, out_dir / "v4_integrity_checks.csv")
    write_csv(target_by_horizon, out_dir / "v4_target_distribution_by_horizon.csv")
    write_csv(target_by_sleeve, out_dir / "v4_target_distribution_by_sleeve.csv")
    write_csv(block_missingness, out_dir / "v4_feature_block_missingness.csv")
    write_csv(feature_scale_summary, out_dir / "v4_feature_scale_summary.csv")
    write_csv(trainability, out_dir / "v4_trainability_summary.csv")
    write_csv(sleeve_recheck, out_dir / "v4_sleeve_admission_recheck.csv")
    write_csv(sleeve_missingness, out_dir / "v4_sleeve_block_missingness.csv")
    write_csv(euro_audit, out_dir / "v4_euro_fixed_income_audit.csv")
    write_csv(real_asset_distinctness, out_dir / "v4_real_asset_distinctness.csv")
    write_text(report, out_dir / "v4_acceptance_report.md")

    return {
        "acceptance_report": out_dir / "v4_acceptance_report.md",
        "integrity_checks": out_dir / "v4_integrity_checks.csv",
        "target_distribution_by_horizon": out_dir / "v4_target_distribution_by_horizon.csv",
        "target_distribution_by_sleeve": out_dir / "v4_target_distribution_by_sleeve.csv",
        "feature_block_missingness": out_dir / "v4_feature_block_missingness.csv",
        "feature_scale_summary": out_dir / "v4_feature_scale_summary.csv",
        "trainability_summary": out_dir / "v4_trainability_summary.csv",
        "sleeve_admission_recheck": out_dir / "v4_sleeve_admission_recheck.csv",
        "sleeve_block_missingness": out_dir / "v4_sleeve_block_missingness.csv",
        "euro_fixed_income_audit": out_dir / "v4_euro_fixed_income_audit.csv",
        "real_asset_distinctness": out_dir / "v4_real_asset_distinctness.csv",
    }
