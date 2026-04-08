"""Machine-checkable QA checks and reporting for XOPTPOE v1 build."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from xoptpoe_data.config import NO_LOCAL_BLOCK_SLEEVES


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _audit(
    *,
    table_name: str,
    object_id: str,
    audit_name: str,
    passed: bool,
    audit_value: str,
    notes: str = "",
    expected_start: str | None = None,
    actual_start: str | None = None,
    expected_end: str | None = None,
    actual_end: str | None = None,
    missing_share: float | None = None,
    duplicate_count: int | None = None,
) -> dict:
    return {
        "table_name": table_name,
        "object_id": object_id,
        "audit_name": audit_name,
        "audit_result": "PASS" if passed else "FAIL",
        "audit_value": audit_value,
        "audit_timestamp": _utc_now(),
        "expected_start": expected_start,
        "actual_start": actual_start,
        "expected_end": expected_end,
        "actual_end": actual_end,
        "missing_share": missing_share,
        "duplicate_count": duplicate_count,
        "notes": notes,
    }


def run_target_qa(
    *,
    asset_master: pd.DataFrame,
    sleeve_target_raw: pd.DataFrame,
    month_end_prices: pd.DataFrame,
    target_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Run mandatory target-layer QA checks."""
    rows: list[dict] = []

    dup_cnt = int(sleeve_target_raw.duplicated(subset=["sleeve_id", "trade_date"]).sum())
    rows.append(
        _audit(
            table_name="sleeve_target_raw",
            object_id="all",
            audit_name="duplicate_primary_key",
            passed=dup_cnt == 0,
            audit_value=str(dup_cnt),
            duplicate_count=dup_cnt,
        )
    )

    miss_adj = int(sleeve_target_raw["adj_close"].isna().sum())
    rows.append(
        _audit(
            table_name="sleeve_target_raw",
            object_id="all",
            audit_name="missing_adjusted_close",
            passed=miss_adj == 0,
            audit_value=str(miss_adj),
        )
    )

    non_pos = int((sleeve_target_raw["adj_close"] <= 0).sum())
    rows.append(
        _audit(
            table_name="sleeve_target_raw",
            object_id="all",
            audit_name="non_positive_prices",
            passed=non_pos == 0,
            audit_value=str(non_pos),
        )
    )

    # Month-end collapse check: selected trade_date must match max trade_date in source month.
    raw = sleeve_target_raw.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["month_end"] = raw["trade_date"].dt.to_period("M").dt.to_timestamp("M")
    expected = raw.groupby(["sleeve_id", "month_end"], as_index=False)["trade_date"].max().rename(
        columns={"trade_date": "expected_trade_date"}
    )

    check = month_end_prices[["sleeve_id", "month_end", "trade_date"]].merge(
        expected, on=["sleeve_id", "month_end"], how="left"
    )
    bad_collapse = int((pd.to_datetime(check["trade_date"]) != pd.to_datetime(check["expected_trade_date"])).sum())
    rows.append(
        _audit(
            table_name="sleeve_month_end_prices",
            object_id="all",
            audit_name="month_end_collapse_correctness",
            passed=bad_collapse == 0,
            audit_value=str(bad_collapse),
        )
    )

    impossible = int((target_panel["ret_fwd_1m"] <= -1).sum())
    rows.append(
        _audit(
            table_name="target_panel",
            object_id="all",
            audit_name="impossible_returns",
            passed=impossible == 0,
            audit_value=str(impossible),
        )
    )

    expected_tickers = asset_master.set_index("sleeve_id")["ticker"].to_dict()
    observed = (
        sleeve_target_raw.groupby("sleeve_id")["ticker"]
        .apply(lambda s: sorted(set(map(str, s.dropna().tolist()))))
        .to_dict()
    )
    sub_fail = 0
    for sleeve_id, expected_ticker in expected_tickers.items():
        obs = observed.get(sleeve_id, [])
        if obs != [expected_ticker]:
            sub_fail += 1
    rows.append(
        _audit(
            table_name="sleeve_target_raw",
            object_id="all",
            audit_name="silent_ticker_substitution",
            passed=sub_fail == 0,
            audit_value=str(sub_fail),
        )
    )

    missing_sleeves = sorted(set(asset_master["sleeve_id"]) - set(sleeve_target_raw["sleeve_id"]))
    rows.append(
        _audit(
            table_name="sleeve_target_raw",
            object_id="all",
            audit_name="all_locked_sleeves_present",
            passed=len(missing_sleeves) == 0,
            audit_value="|".join(missing_sleeves) if missing_sleeves else "0",
            notes="Missing sleeve ids listed in audit_value when non-zero",
        )
    )

    for sleeve_id, grp in month_end_prices.groupby("sleeve_id"):
        rows.append(
            _audit(
                table_name="sleeve_month_end_prices",
                object_id=sleeve_id,
                audit_name="sample_coverage",
                passed=True,
                audit_value="ok",
                expected_start="2006-01-31",
                actual_start=str(pd.to_datetime(grp["month_end"]).min().date()),
                actual_end=str(pd.to_datetime(grp["month_end"]).max().date()),
            )
        )

    return pd.DataFrame(rows)


def run_macro_qa(
    *,
    macro_manifest: pd.DataFrame,
    macro_raw: pd.DataFrame,
    macro_state_panel: pd.DataFrame,
    global_state_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Run mandatory macro-layer QA checks."""
    rows: list[dict] = []

    dup_cnt = int(macro_raw.duplicated(subset=["series_id", "obs_date"]).sum())
    rows.append(
        _audit(
            table_name="macro_raw",
            object_id="all",
            audit_name="duplicate_primary_key",
            passed=dup_cnt == 0,
            audit_value=str(dup_cnt),
            duplicate_count=dup_cnt,
        )
    )

    manifest_freq = macro_manifest.set_index("series_id")["native_frequency"].to_dict()
    macro_raw = macro_raw.copy()
    macro_raw["expected_native_frequency"] = macro_raw["series_id"].map(manifest_freq)
    freq_mismatch = int((macro_raw["native_frequency"] != macro_raw["expected_native_frequency"]).sum())
    rows.append(
        _audit(
            table_name="macro_raw",
            object_id="all",
            audit_name="native_frequency_match",
            passed=freq_mismatch == 0,
            audit_value=str(freq_mismatch),
        )
    )

    code_mismatch = int(
        ((macro_raw["used_code"] != macro_raw["preferred_code"]) & (macro_raw["fallback_used"] != 1)).sum()
    )
    rows.append(
        _audit(
            table_name="macro_raw",
            object_id="all",
            audit_name="series_code_match_manifest",
            passed=code_mismatch == 0,
            audit_value=str(code_mismatch),
        )
    )

    required_monthly_official = (
        "US_CPI",
        "US_UNEMP",
        "US_RF3M",
        "EA_CPI",
        "EA_UNEMP",
        "EA_3M",
        "EA_10Y",
        "JP_CPI",
        "JP_UNEMP",
        "JP_3M",
        "JP_10Y",
    )
    interior_nan_count = 0
    interior_nan_details: list[str] = []
    for series_id in required_monthly_official:
        series = macro_raw.loc[macro_raw["series_id"] == series_id, ["obs_date", "value"]].copy()
        if series.empty:
            continue
        series["obs_date"] = pd.to_datetime(series["obs_date"])
        series = series.sort_values("obs_date").reset_index(drop=True)

        has_prior_valid = series["value"].notna().cumsum().gt(0)
        has_future_valid = series["value"].notna()[::-1].cumsum()[::-1].gt(0)
        interior_nan = series["value"].isna() & has_prior_valid & has_future_valid
        n_interior = int(interior_nan.sum())
        interior_nan_count += n_interior
        if n_interior > 0:
            dates = (
                series.loc[interior_nan, "obs_date"]
                .dt.date.astype(str)
                .tolist()
            )
            interior_nan_details.append(f"{series_id}:{'|'.join(dates)}")

    rows.append(
        _audit(
            table_name="macro_raw",
            object_id="required_monthly_official",
            audit_name="interior_nan_rows_flag",
            passed=True,
            audit_value=str(interior_nan_count),
            notes=(
                "FLAG: interior NaN rows in required monthly official series; "
                "as-of alignment should fallback to prior valid observation. "
                f"Details: {','.join(interior_nan_details)}"
                if interior_nan_count > 0
                else ""
            ),
        )
    )

    msp = macro_state_panel.copy()
    msp["month_end"] = pd.to_datetime(msp["month_end"])

    canonical_required = {
        "month_end",
        "infl_US",
        "unemp_US",
        "short_rate_US",
        "long_rate_US",
        "term_slope_US",
        "infl_EA",
        "unemp_EA",
        "short_rate_EA",
        "long_rate_EA",
        "term_slope_EA",
        "infl_JP",
        "unemp_JP",
        "short_rate_JP",
        "long_rate_JP",
        "term_slope_JP",
        "usd_broad",
        "vix",
        "us_real10y",
        "ig_oas",
        "oil_wti",
        "lag_policy_tag",
    }
    missing_canonical = sorted(canonical_required - set(msp.columns))
    rows.append(
        _audit(
            table_name="macro_state_panel",
            object_id="all",
            audit_name="canonical_geo_prefixed_schema_present",
            passed=len(missing_canonical) == 0,
            audit_value=str(len(missing_canonical)),
            notes="Missing canonical columns: " + ",".join(missing_canonical) if missing_canonical else "",
        )
    )

    em_like_cols = sorted([c for c in msp.columns if c.endswith("_EM") or "_EM_" in c])
    rows.append(
        _audit(
            table_name="macro_state_panel",
            object_id="all",
            audit_name="no_em_local_macro_columns",
            passed=len(em_like_cols) == 0,
            audit_value=str(len(em_like_cols)),
            notes="Unexpected EM-like columns: " + ",".join(em_like_cols) if em_like_cols else "",
        )
    )

    lineage_required = {
        "src_obs_month_end_cpi_US",
        "src_obs_month_end_unemp_US",
        "src_obs_month_end_short_rate_US",
        "src_obs_month_end_long_rate_US",
        "src_obs_month_end_cpi_EA",
        "src_obs_month_end_unemp_EA",
        "src_obs_month_end_short_rate_EA",
        "src_obs_month_end_long_rate_EA",
        "src_obs_month_end_cpi_JP",
        "src_obs_month_end_unemp_JP",
        "src_obs_month_end_short_rate_JP",
        "src_obs_month_end_long_rate_JP",
    }
    missing_lineage = sorted(lineage_required - set(msp.columns))
    rows.append(
        _audit(
            table_name="macro_state_panel",
            object_id="all",
            audit_name="source_lineage_columns_present",
            passed=len(missing_lineage) == 0,
            audit_value=str(len(missing_lineage)),
            notes="Missing lineage columns: " + ",".join(missing_lineage) if missing_lineage else "",
        )
    )

    # Lag-policy checks using source-observation lineage in canonical wide schema.
    exp_prev = msp["month_end"] - pd.offsets.MonthEnd(1)
    total_lag_viol = 0
    if missing_lineage:
        total_lag_viol = len(missing_lineage)
    else:
        for suffix in ("US", "EA", "JP"):
            cpi_obs = pd.to_datetime(msp[f"src_obs_month_end_cpi_{suffix}"])
            unemp_obs = pd.to_datetime(msp[f"src_obs_month_end_unemp_{suffix}"])
            short_obs = pd.to_datetime(msp[f"src_obs_month_end_short_rate_{suffix}"])
            long_obs = pd.to_datetime(msp[f"src_obs_month_end_long_rate_{suffix}"])

            total_lag_viol += int((cpi_obs > exp_prev).sum())
            total_lag_viol += int((unemp_obs > exp_prev).sum())
            total_lag_viol += int((short_obs > exp_prev).sum())

            # US long-rate source is market-observable at t; EA/JP are official monthly lagged by one month.
            long_limit = msp["month_end"] if suffix == "US" else exp_prev
            total_lag_viol += int((long_obs > long_limit).sum())

    rows.append(
        _audit(
            table_name="macro_state_panel",
            object_id="all",
            audit_name="lag_policy_violations",
            passed=total_lag_viol == 0,
            audit_value=str(total_lag_viol),
            notes="Canonical wide checks across US/EA/JP with US 10Y market-observable exception",
        )
    )

    stale_cols = [c for c in msp.columns if c.startswith("staleness_")]
    stale_consistency_local = int(
        (
            msp[stale_cols]
            .fillna(0)
            .gt(0)
            .any(axis=1)
            .astype(int)
            != msp["macro_stale_flag"].astype(int)
        ).sum()
    )
    rows.append(
        _audit(
            table_name="macro_state_panel",
            object_id="all",
            audit_name="stale_flag_consistency_macro_state",
            passed=stale_consistency_local == 0,
            audit_value=str(stale_consistency_local),
        )
    )

    gsp = global_state_panel.copy()
    stale_cols = [
        "usd_broad_level_staleness",
        "vix_level_staleness",
        "us_real10y_level_staleness",
        "ig_oas_level_staleness",
        "oil_wti_level_staleness",
    ]
    missing_gsp_cols = [c for c in stale_cols if c not in gsp.columns]
    if missing_gsp_cols:
        stale_consistency_global = len(missing_gsp_cols)
    else:
        stale_consistency_global = int(
            (gsp[stale_cols].fillna(0).gt(0).any(axis=1).astype(int) != gsp["global_stale_flag"].astype(int)).sum()
        )
    rows.append(
        _audit(
            table_name="global_state_panel",
            object_id="all",
            audit_name="stale_flag_consistency_global",
            passed=stale_consistency_global == 0,
            audit_value=str(stale_consistency_global),
            notes="Missing columns: " + ",".join(missing_gsp_cols) if missing_gsp_cols else "",
        )
    )

    global_stale_cols_wide = [
        "staleness_usd_broad",
        "staleness_vix",
        "staleness_us_real10y",
        "staleness_ig_oas",
        "staleness_oil_wti",
    ]
    missing_global_stale_cols = [c for c in global_stale_cols_wide if c not in msp.columns]
    if missing_global_stale_cols:
        stale_consistency_global_wide = len(missing_global_stale_cols)
    else:
        stale_consistency_global_wide = int(
            (
                msp[global_stale_cols_wide].fillna(0).gt(0).any(axis=1).astype(int)
                != msp["global_stale_flag"].astype(int)
            ).sum()
        )
    rows.append(
        _audit(
            table_name="macro_state_panel",
            object_id="all",
            audit_name="stale_flag_consistency_global_wide",
            passed=stale_consistency_global_wide == 0,
            audit_value=str(stale_consistency_global_wide),
            notes=(
                "Missing columns: " + ",".join(missing_global_stale_cols)
                if missing_global_stale_cols
                else ""
            ),
        )
    )

    dup_macro_state = int(msp.duplicated(subset=["month_end"]).sum())
    rows.append(
        _audit(
            table_name="macro_state_panel",
            object_id="all",
            audit_name="duplicate_primary_key",
            passed=dup_macro_state == 0,
            audit_value=str(dup_macro_state),
        )
    )

    return pd.DataFrame(rows)


def run_join_qa(
    *,
    feature_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    modeling_panel: pd.DataFrame,
    macro_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Run join-layer QA checks."""
    rows: list[dict] = []

    dup_model = int(modeling_panel.duplicated(subset=["sleeve_id", "month_end"]).sum())
    rows.append(
        _audit(
            table_name="modeling_panel",
            object_id="all",
            audit_name="duplicate_primary_key",
            passed=dup_model == 0,
            audit_value=str(dup_model),
            duplicate_count=dup_model,
        )
    )

    # Only feature-complete rows are expected to map into supervised modeling rows.
    eligible_features = feature_panel[feature_panel["feature_complete_flag"] == 1]
    feat_keys = eligible_features[["sleeve_id", "month_end"]].drop_duplicates()
    tgt_keys = target_panel[["sleeve_id", "month_end"]].drop_duplicates()
    merged = feat_keys.merge(tgt_keys, on=["sleeve_id", "month_end"], how="left", indicator=True)
    terminal_month = (
        feat_keys.groupby("sleeve_id", as_index=False)["month_end"]
        .max()
        .rename(columns={"month_end": "terminal_month_end"})
    )
    merged = merged.merge(terminal_month, on="sleeve_id", how="left")
    merged["is_terminal_feature_month"] = merged["month_end"] == merged["terminal_month_end"]

    missing_target_rows = int(
        ((merged["_merge"] == "left_only") & (~merged["is_terminal_feature_month"])).sum()
    )
    rows.append(
        _audit(
            table_name="modeling_join",
            object_id="all",
            audit_name="missing_target_rows",
            passed=missing_target_rows == 0,
            audit_value=str(missing_target_rows),
        )
    )

    terminal_missing = int(
        ((merged["_merge"] == "left_only") & (merged["is_terminal_feature_month"])).sum()
    )
    rows.append(
        _audit(
            table_name="modeling_join",
            object_id="all",
            audit_name="expected_terminal_missing_target_rows",
            passed=True,
            audit_value=str(terminal_missing),
            notes="One terminal feature month per sleeve may lack forward target by design",
        )
    )

    local_required_cols = [
        "local_cpi_yoy",
        "local_unemp",
        "local_3m_rate",
        "local_10y_rate",
        "local_term_slope",
    ]
    local_sleeves = set(macro_mapping.loc[macro_mapping["block_role"] == "local", "sleeve_id"])
    local_rows = eligible_features[eligible_features["sleeve_id"].isin(local_sleeves)]
    missing_local_macro = int(local_rows[local_required_cols].isna().any(axis=1).sum())
    rows.append(
        _audit(
            table_name="feature_panel",
            object_id="local_sleeves",
            audit_name="missing_mapped_local_macro_rows",
            passed=missing_local_macro == 0,
            audit_value=str(missing_local_macro),
        )
    )

    forbidden_local = int(
        (
            feature_panel["sleeve_id"].isin(NO_LOCAL_BLOCK_SLEEVES)
            & feature_panel["local_geo_block_used"].notna()
        ).sum()
    )
    rows.append(
        _audit(
            table_name="feature_panel",
            object_id="EQ_EM_ALT_GLD",
            audit_name="forbidden_local_block_attachment",
            passed=forbidden_local == 0,
            audit_value=str(forbidden_local),
        )
    )

    missing_excess = int(modeling_panel["excess_ret_fwd_1m"].isna().sum())
    rows.append(
        _audit(
            table_name="modeling_panel",
            object_id="all",
            audit_name="missing_target_after_join",
            passed=missing_excess == 0,
            audit_value=str(missing_excess),
        )
    )

    return pd.DataFrame(rows)


def write_auxiliary_reports(
    *,
    feature_panel: pd.DataFrame,
    modeling_panel: pd.DataFrame,
    reports_dir: Path,
) -> None:
    """Write machine-checkable coverage and missingness reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    coverage = (
        modeling_panel.groupby("sleeve_id", as_index=False)
        .agg(
            actual_start=("month_end", "min"),
            actual_end=("month_end", "max"),
            rows=("month_end", "count"),
            included_rows=("sample_inclusion_flag", "sum"),
        )
        .sort_values("sleeve_id")
    )
    coverage.to_csv(reports_dir / "coverage_by_sleeve.csv", index=False)

    missingness = (
        feature_panel.isna().mean().sort_values(ascending=False).rename("missing_share").reset_index()
    )
    missingness = missingness.rename(columns={"index": "feature_name"})
    missingness.to_csv(reports_dir / "feature_missingness.csv", index=False)


def write_qa_summary(audit_df: pd.DataFrame, report_path: Path) -> None:
    """Write human-readable QA markdown report."""
    fails = audit_df[audit_df["audit_result"] == "FAIL"].copy()
    pass_count = int((audit_df["audit_result"] == "PASS").sum())
    fail_count = int((audit_df["audit_result"] == "FAIL").sum())

    lines = [
        "# XOPTPOE v1 QA Summary",
        "",
        f"- PASS checks: {pass_count}",
        f"- FAIL checks: {fail_count}",
        "",
    ]

    if fail_count == 0:
        lines.append("All machine-checkable QA checks passed.")
    else:
        lines.append("## Failed Checks")
        lines.append("")
        for row in fails.itertuples(index=False):
            lines.append(
                f"- `{row.table_name}` / `{row.object_id}` / `{row.audit_name}` => {row.audit_value}"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_qa(
    *,
    asset_master: pd.DataFrame,
    macro_manifest: pd.DataFrame,
    sleeve_target_raw: pd.DataFrame,
    month_end_prices: pd.DataFrame,
    macro_raw: pd.DataFrame,
    macro_state_panel: pd.DataFrame,
    global_state_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    modeling_panel: pd.DataFrame,
    macro_mapping: pd.DataFrame,
    reports_dir: Path,
) -> pd.DataFrame:
    """Execute target/macro/join QA and write report artifacts."""
    target_audit = run_target_qa(
        asset_master=asset_master,
        sleeve_target_raw=sleeve_target_raw,
        month_end_prices=month_end_prices,
        target_panel=target_panel,
    )

    macro_audit = run_macro_qa(
        macro_manifest=macro_manifest,
        macro_raw=macro_raw,
        macro_state_panel=macro_state_panel,
        global_state_panel=global_state_panel,
    )

    join_audit = run_join_qa(
        feature_panel=feature_panel,
        target_panel=target_panel,
        modeling_panel=modeling_panel,
        macro_mapping=macro_mapping,
    )

    audit_df = pd.concat([target_audit, macro_audit, join_audit], ignore_index=True)
    write_auxiliary_reports(feature_panel=feature_panel, modeling_panel=modeling_panel, reports_dir=reports_dir)
    write_qa_summary(audit_df=audit_df, report_path=reports_dir / "qa_summary.md")
    return audit_df
