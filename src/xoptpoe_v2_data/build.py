"""Build the XOPTPOE v2 long-horizon data package."""

from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_data.config import LOCKED_SLEEVES, LOCAL_BLOCK_BY_SLEEVE, NO_LOCAL_BLOCK_SLEEVES
from xoptpoe_v2_data.config import (
    ASSET_CLASS_GROUP_BY_SLEEVE,
    EXPOSURE_REGION_BY_SLEEVE,
    LOCKED_HORIZONS,
    V2_VERSION,
    V2Paths,
    default_paths,
)
from xoptpoe_v2_data.io import load_csv, write_csv, write_parquet, write_text
from xoptpoe_v2_data.sources import build_additional_monthly_state, build_external_file_inventory
from xoptpoe_v2_data.targets import build_long_horizon_targets

ID_COLS = {"sleeve_id", "month_end", "ticker", "sleeve_name", "asset_class_group", "exposure_region"}
MONTHLY_NON_PREDICTOR_COLS = {
    "feature_complete_flag",
    "lag_policy_tag",
    "local_geo_block_used",
    "geo_block_local",
    "geo_block_global",
    "baseline_sample_inclusion_1m_flag",
    "strict_all_feature_complete_flag",
    "recommended_feature_complete_flag",
    "added_feature_missing_count",
    "added_feature_nonmissing_share",
    "any_added_feature_missing_flag",
    "proxy_flag",
}
STACKED_NON_PREDICTOR_COLS = {
    "target_available_flag",
    "forward_valid_month_count",
    "baseline_trainable_flag",
    "strict_trainable_flag",
    "gross_total_forward_return",
    "cumulative_total_forward_return",
    "gross_rf_forward_return",
    "cumulative_rf_forward_return",
    "annualized_total_forward_return",
    "annualized_rf_forward_return",
    "annualized_excess_forward_return",
    "realized_forward_volatility",
    "realized_forward_max_drawdown",
}



def _load_inputs(paths: V2Paths) -> dict[str, pd.DataFrame]:
    return {
        "feature_panel": load_csv(paths.data_intermediate_dir / "feature_panel.csv", parse_dates=["month_end"]),
        "modeling_panel": load_csv(paths.data_final_dir / "modeling_panel.csv", parse_dates=["month_end"]),
        "asset_master": load_csv(paths.data_final_dir / "asset_master.csv", parse_dates=["start_date_target"]),
        "macro_mapping": load_csv(paths.data_final_dir / "macro_mapping.csv"),
        "month_end_prices": load_csv(
            paths.data_intermediate_dir / "sleeve_month_end_prices.csv",
            parse_dates=["month_end", "trade_date"],
        ),
        "monthly_returns": load_csv(paths.data_intermediate_dir / "sleeve_monthly_returns.csv", parse_dates=["month_end"]),
        "tb3ms_monthly": load_csv(paths.data_intermediate_dir / "tb3ms_monthly.csv", parse_dates=["month_end"]),
    }



def _profile_feature(df: pd.DataFrame, feature_name: str) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, int, float]:
    series = df[feature_name]
    mask = series.notna()
    if not mask.any():
        return pd.NaT, pd.NaT, 0, 0.0
    dates = pd.to_datetime(df.loc[mask, "month_end"])
    return dates.min(), dates.max(), int(mask.sum()), float(mask.mean())



def _profile_feature_hstack(df: pd.DataFrame, feature_name: str) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, int, float]:
    series = df[feature_name]
    mask = series.notna()
    if not mask.any():
        return pd.NaT, pd.NaT, 0, 0.0
    dates = pd.to_datetime(df.loc[mask, "month_end"])
    return dates.min(), dates.max(), int(mask.sum()), float(mask.mean())



def _feature_meta_row(
    *,
    feature_name: str,
    source_file: str,
    source_column: str,
    block_name: str,
    geography: str,
    transform_type: str,
    native_frequency: str,
    lag_months: int,
    notes: str,
    monthly_flag: int,
    hstack_flag: int,
    is_interaction: int,
    monthly_df: pd.DataFrame | None,
    hstack_df: pd.DataFrame | None,
) -> dict[str, object]:
    base_df = monthly_df if monthly_flag == 1 else hstack_df
    if base_df is None:
        first_valid, last_valid, nonmissing_count, nonmissing_share = pd.NaT, pd.NaT, 0, 0.0
    elif monthly_flag == 1:
        first_valid, last_valid, nonmissing_count, nonmissing_share = _profile_feature(monthly_df, feature_name)
    else:
        first_valid, last_valid, nonmissing_count, nonmissing_share = _profile_feature_hstack(hstack_df, feature_name)

    return {
        "feature_name": feature_name,
        "source_file": source_file,
        "source_column": source_column,
        "block_name": block_name,
        "geography": geography,
        "transform_type": transform_type,
        "native_frequency": native_frequency,
        "lag_months": lag_months,
        "available_in_feature_master_monthly": monthly_flag,
        "available_in_modeling_panel_hstack": hstack_flag,
        "is_interaction": is_interaction,
        "first_valid_date": first_valid,
        "last_valid_date": last_valid,
        "nonmissing_count": nonmissing_count,
        "nonmissing_share": nonmissing_share,
        "notes": notes,
    }



def _baseline_feature_catalog(feature_master: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_cols = [
        col
        for col in feature_master.columns
        if col not in ID_COLS
        and col not in MONTHLY_NON_PREDICTOR_COLS
        and col not in {"sample_inclusion_flag", "target_quality_flag", "ret_fwd_1m", "rf_fwd_1m", "excess_ret_fwd_1m"}
    ]
    for col in baseline_cols:
        if col.startswith(("ret_", "mom_", "vol_", "maxdd_", "rel_")):
            rows.append(
                _feature_meta_row(
                    feature_name=col,
                    source_file="data/intermediate/feature_panel.csv",
                    source_column=col,
                    block_name="baseline_technical",
                    geography="SLEEVE_SPECIFIC",
                    transform_type="derived_technical",
                    native_frequency="monthly",
                    lag_months=0,
                    notes="Inherited from frozen v1 feature panel.",
                    monthly_flag=1,
                    hstack_flag=1,
                    is_interaction=0,
                    monthly_df=feature_master,
                    hstack_df=None,
                )
            )
        elif col.startswith(("infl_", "unemp_", "short_rate_", "long_rate_", "term_slope_")):
            suffix = col.rsplit("_", 1)[-1]
            rows.append(
                _feature_meta_row(
                    feature_name=col,
                    source_file="data/intermediate/feature_panel.csv",
                    source_column=col,
                    block_name="baseline_macro_canonical",
                    geography=suffix if suffix in {"US", "EA", "JP"} else "LOCAL_BLOCK",
                    transform_type="delta_1m" if col.endswith("_delta_1m") else "level_or_yoy",
                    native_frequency="monthly",
                    lag_months=1,
                    notes="Inherited canonical v1 macro feature.",
                    monthly_flag=1,
                    hstack_flag=1,
                    is_interaction=0,
                    monthly_df=feature_master,
                    hstack_df=None,
                )
            )
        elif col.startswith("local_"):
            rows.append(
                _feature_meta_row(
                    feature_name=col,
                    source_file="data/intermediate/feature_panel.csv",
                    source_column=col,
                    block_name="baseline_macro_alias",
                    geography="LOCAL_MAPPED",
                    transform_type="compatibility_alias",
                    native_frequency="monthly",
                    lag_months=1,
                    notes="Derived local alias retained from v1 compatibility layer.",
                    monthly_flag=1,
                    hstack_flag=1,
                    is_interaction=0,
                    monthly_df=feature_master,
                    hstack_df=None,
                )
            )
        elif col.startswith(("usd_broad", "vix", "us_real10y", "ig_oas", "oil_wti")) or col == "macro_stale_flag":
            rows.append(
                _feature_meta_row(
                    feature_name=col,
                    source_file="data/intermediate/feature_panel.csv",
                    source_column=col,
                    block_name="baseline_global_macro",
                    geography="GLOBAL",
                    transform_type="delta_1m" if col.endswith(("_delta_1m", "_logchg_1m", "_logchg_12m")) else "level_or_flag",
                    native_frequency="monthly",
                    lag_months=0,
                    notes="Inherited global/stress feature from frozen v1 panel.",
                    monthly_flag=1,
                    hstack_flag=1,
                    is_interaction=0,
                    monthly_df=feature_master,
                    hstack_df=None,
                )
            )
    return rows



def _manual_monthly_features(feature_master: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    panel = feature_master.copy()
    rows: list[dict[str, object]] = []
    interactions: list[dict[str, object]] = []

    panel["oecd_business_conf_local"] = np.select(
        [panel["geo_block_local"].eq("US"), panel["geo_block_local"].eq("EURO_AREA"), panel["geo_block_local"].eq("JAPAN")],
        [panel["oecd_bcicp_us"], panel["oecd_bcicp_ea"], panel["oecd_bcicp_jp"]],
        default=np.nan,
    )
    panel["oecd_consumer_conf_local"] = np.select(
        [panel["geo_block_local"].eq("US"), panel["geo_block_local"].eq("EURO_AREA"), panel["geo_block_local"].eq("JAPAN")],
        [panel["oecd_ccicp_us"], panel["oecd_ccicp_ea"], panel["oecd_ccicp_jp"]],
        default=np.nan,
    )
    panel["oecd_activity_proxy_local"] = np.select(
        [panel["geo_block_local"].eq("US"), panel["geo_block_local"].eq("EURO_AREA"), panel["geo_block_local"].eq("JAPAN")],
        [panel["oecd_li_us"], panel["oecd_bcicp_ea"], panel["oecd_li_jp"]],
        default=np.nan,
    )
    panel["cape_local"] = np.select(
        [panel["geo_block_local"].eq("US"), panel["geo_block_local"].eq("EURO_AREA"), panel["geo_block_local"].eq("JAPAN")],
        [panel["cape_usa"], panel["cape_europe"], panel["cape_japan"]],
        default=np.nan,
    )

    for col, geography, notes in [
        ("oecd_business_conf_local", "LOCAL_MAPPED", "Mapped from US/EA/JP OECD business-confidence series."),
        ("oecd_consumer_conf_local", "LOCAL_MAPPED", "Mapped from US/EA/JP OECD consumer-confidence series."),
        ("oecd_activity_proxy_local", "LOCAL_MAPPED", "Mapped local activity proxy: LI for US/JP, business-confidence proxy for EA."),
        ("cape_local", "LOCAL_MAPPED", "Mapped local CAPE from US/Europe/Japan CAPE series."),
    ]:
        rows.append(
            _feature_meta_row(
                feature_name=col,
                source_file="derived_from_monthly_state",
                source_column="mapped",
                block_name="local_mapping",
                geography=geography,
                transform_type="mapped_alias",
                native_frequency="monthly",
                lag_months=0,
                notes=notes,
                monthly_flag=1,
                hstack_flag=1,
                is_interaction=0,
                monthly_df=panel,
                hstack_df=None,
            )
        )

    panel["eq_em_relevance_flag"] = panel["sleeve_id"].eq("EQ_EM").astype(int)
    panel["jp_relevance_flag"] = panel["sleeve_id"].eq("EQ_JP").astype(int)
    panel["no_local_block_flag"] = panel["sleeve_id"].isin(NO_LOCAL_BLOCK_SLEEVES).astype(int)
    for sleeve_id in LOCKED_SLEEVES:
        panel[f"sleeve_dummy_{sleeve_id.lower()}"] = panel["sleeve_id"].eq(sleeve_id).astype(int)
    for group in sorted(set(ASSET_CLASS_GROUP_BY_SLEEVE.values())):
        panel[f"asset_group_dummy_{group}"] = panel["asset_class_group"].eq(group).astype(int)

    dummy_cols = [
        "eq_em_relevance_flag",
        "jp_relevance_flag",
        "no_local_block_flag",
        *[f"sleeve_dummy_{sleeve.lower()}" for sleeve in LOCKED_SLEEVES],
        *[f"asset_group_dummy_{group}" for group in sorted(set(ASSET_CLASS_GROUP_BY_SLEEVE.values()))],
    ]
    for col in dummy_cols:
        rows.append(
            _feature_meta_row(
                feature_name=col,
                source_file="derived_v2",
                source_column=col,
                block_name="metadata_dummy",
                geography="SLEEVE_SPECIFIC",
                transform_type="dummy_flag",
                native_frequency="monthly",
                lag_months=0,
                notes="Static sleeve/group relevance indicator for v2 interactions.",
                monthly_flag=1,
                hstack_flag=1,
                is_interaction=0,
                monthly_df=panel,
                hstack_df=None,
            )
        )

    def add_interaction(name: str, lhs: str, rhs: str, family: str, geography: str, notes: str) -> None:
        panel[name] = panel[lhs] * panel[rhs]
        interactions.append(
            {
                "interaction_name": name,
                "table_scope": "feature_master_monthly",
                "interaction_family": family,
                "lhs_feature": lhs,
                "rhs_feature": rhs,
                "geography": geography,
                "notes": notes,
            }
        )
        rows.append(
            _feature_meta_row(
                feature_name=name,
                source_file="derived_interaction",
                source_column=f"{lhs} x {rhs}",
                block_name="interaction",
                geography=geography,
                transform_type=family,
                native_frequency="monthly",
                lag_months=0,
                notes=notes,
                monthly_flag=1,
                hstack_flag=1,
                is_interaction=1,
                monthly_df=panel,
                hstack_df=None,
            )
        )

    for sleeve_id in LOCKED_SLEEVES:
        sleeve_dummy = f"sleeve_dummy_{sleeve_id.lower()}"
        add_interaction(
            f"int_{sleeve_id.lower()}_x_mom_12_1",
            "mom_12_1",
            sleeve_dummy,
            "sleeve_dummy_x_predictor",
            sleeve_id,
            f"Sleeve-specific momentum slope for {sleeve_id}.",
        )
        add_interaction(
            f"int_{sleeve_id.lower()}_x_vol_12m",
            "vol_12m",
            sleeve_dummy,
            "sleeve_dummy_x_predictor",
            sleeve_id,
            f"Sleeve-specific 12m volatility slope for {sleeve_id}.",
        )

    for group in sorted(set(ASSET_CLASS_GROUP_BY_SLEEVE.values())):
        group_dummy = f"asset_group_dummy_{group}"
        add_interaction(
            f"int_{group}_x_vix",
            "vix",
            group_dummy,
            "asset_group_dummy_x_predictor",
            group.upper(),
            f"Asset-class VIX exposure interaction for {group} sleeves.",
        )
        add_interaction(
            f"int_{group}_x_us_real10y",
            "us_real10y",
            group_dummy,
            "asset_group_dummy_x_predictor",
            group.upper(),
            f"Asset-class real-rate interaction for {group} sleeves.",
        )

    for name, lhs, rhs, family, geography, notes in [
        ("int_vix_x_cape_local", "vix", "cape_local", "stress_x_valuation", "LOCAL_MAPPED", "Stress-by-local-CAPE interaction."),
        ("int_ig_oas_x_cape_local", "ig_oas", "cape_local", "stress_x_valuation", "LOCAL_MAPPED", "Credit-stress-by-local-CAPE interaction."),
        ("int_vix_x_em_minus_global_pe", "vix", "em_minus_global_pe", "stress_x_valuation", "EM_GLOBAL", "Stress-by-EM-global valuation interaction."),
        ("int_vix_x_mom_12_1", "vix", "mom_12_1", "stress_x_momentum", "MULTI_ASSET", "Stress-by-momentum interaction."),
        ("int_ig_oas_x_rel_mom_vs_treasury", "ig_oas", "rel_mom_vs_treasury", "stress_x_momentum", "MULTI_ASSET", "Credit-stress-by-relative-momentum interaction."),
        ("int_us_real10y_x_mom_12_1", "us_real10y", "mom_12_1", "stress_x_momentum", "MULTI_ASSET", "Real-rate-by-momentum interaction."),
        ("int_china_cli_x_eq_em", "china_cli", "eq_em_relevance_flag", "china_block_x_em_relevance", "CHINA_EM", "China CLI relevance-gated to EQ_EM."),
        ("int_china_pmi_nbs_mfg_x_eq_em", "china_pmi_nbs_mfg", "eq_em_relevance_flag", "china_block_x_em_relevance", "CHINA_EM", "China PMI relevance-gated to EQ_EM."),
        ("int_china_div_yield_spread_x_eq_em", "china_div_yield_spread", "eq_em_relevance_flag", "china_block_x_em_relevance", "CHINA_EM", "China valuation spread relevance-gated to EQ_EM."),
        ("int_china_pe_spread_x_eq_em", "china_pe_spread", "eq_em_relevance_flag", "china_block_x_em_relevance", "CHINA_EM", "China PE spread relevance-gated to EQ_EM."),
        ("int_china_sse_mom_x_eq_em", "china_sse_composite_usd_mom_12_1", "eq_em_relevance_flag", "china_block_x_em_relevance", "CHINA_EM", "China market momentum relevance-gated to EQ_EM."),
        ("int_jp_oecd_alt_x_eq_jp", "jp_oecd_leading_alt", "jp_relevance_flag", "japan_block_x_jp_relevance", "JAPAN", "Japan leading-indicator relevance-gated to EQ_JP."),
        ("int_jp_pe_ratio_x_eq_jp", "jp_pe_ratio", "jp_relevance_flag", "japan_block_x_jp_relevance", "JAPAN", "Japan PE ratio relevance-gated to EQ_JP."),
        ("int_jp_buyback_mom_x_eq_jp", "jp_buyback_index_usd_mom_12_1", "jp_relevance_flag", "japan_block_x_jp_relevance", "JAPAN", "Japan buyback-index momentum relevance-gated to EQ_JP."),
        ("int_jp_tankan_actual_x_eq_jp", "jp_tankan_actual", "jp_relevance_flag", "japan_block_x_jp_relevance", "JAPAN", "Japan Tankan relevance-gated to EQ_JP."),
        ("int_cape_local_x_us_real10y", "cape_local", "us_real10y", "cape_x_real_rate", "LOCAL_MAPPED", "Local CAPE by US real-rate interaction."),
        ("int_cape_usa_x_us_real10y", "cape_usa", "us_real10y", "cape_x_real_rate", "US", "US CAPE by US real-rate interaction."),
        ("int_oecd_activity_proxy_local_x_local_term_slope", "oecd_activity_proxy_local", "local_term_slope", "cli_x_slope_or_spread", "LOCAL_MAPPED", "Local activity proxy by local term-slope interaction."),
        ("int_china_cli_x_ig_oas", "china_cli", "ig_oas", "cli_x_slope_or_spread", "CHINA_GLOBAL", "China CLI by IG-OAS spread interaction."),
    ]:
        add_interaction(name, lhs, rhs, family, geography, notes)

    return panel, rows, interactions



def _build_feature_master_monthly(inputs: dict[str, pd.DataFrame], additional_state: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    feature_panel = inputs["feature_panel"].copy()
    feature_panel["month_end"] = pd.to_datetime(feature_panel["month_end"])

    sample_flag = inputs["modeling_panel"][["sleeve_id", "month_end", "sample_inclusion_flag"]].copy()
    sample_flag.rename(columns={"sample_inclusion_flag": "baseline_sample_inclusion_1m_flag"}, inplace=True)

    asset_cols = inputs["asset_master"][["sleeve_id", "ticker", "sleeve_name", "proxy_flag"]].copy()
    feature_master = feature_panel.merge(asset_cols, on="sleeve_id", how="left")
    feature_master = feature_master.merge(sample_flag, on=["sleeve_id", "month_end"], how="left")
    feature_master["baseline_sample_inclusion_1m_flag"] = feature_master["baseline_sample_inclusion_1m_flag"].fillna(0).astype(int)

    local_map = pd.Series(LOCAL_BLOCK_BY_SLEEVE, name="geo_block_local")
    feature_master["geo_block_local"] = feature_master["sleeve_id"].map(local_map)
    feature_master["geo_block_global"] = "GLOBAL"
    feature_master["asset_class_group"] = feature_master["sleeve_id"].map(ASSET_CLASS_GROUP_BY_SLEEVE)
    feature_master["exposure_region"] = feature_master["sleeve_id"].map(EXPOSURE_REGION_BY_SLEEVE)

    feature_master = feature_master.merge(additional_state, on="month_end", how="left")
    feature_master, manual_rows, interaction_rows = _manual_monthly_features(feature_master)

    additional_feature_cols = [
        col
        for col in additional_state.columns
        if col != "month_end"
    ] + [
        "oecd_business_conf_local",
        "oecd_consumer_conf_local",
        "oecd_activity_proxy_local",
        "cape_local",
    ]
    feature_master["added_feature_missing_count"] = feature_master[additional_feature_cols].isna().sum(axis=1)
    feature_master["added_feature_nonmissing_share"] = feature_master[additional_feature_cols].notna().mean(axis=1)
    feature_master["any_added_feature_missing_flag"] = feature_master["added_feature_missing_count"].gt(0).astype(int)
    feature_master["recommended_feature_complete_flag"] = feature_master["feature_complete_flag"].astype(int)
    feature_master["strict_all_feature_complete_flag"] = (
        feature_master["feature_complete_flag"].eq(1)
        & feature_master[additional_feature_cols].notna().all(axis=1)
    ).astype(int)

    feature_master = feature_master.sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)
    if feature_master.duplicated(subset=["sleeve_id", "month_end"]).any():
        raise ValueError("feature_master_monthly has duplicate (sleeve_id, month_end) keys")

    return feature_master, manual_rows, interaction_rows



def _build_horizon_manifest() -> pd.DataFrame:
    rows = []
    for horizon in LOCKED_HORIZONS:
        years = horizon / 12.0
        rows.append(
            {
                "horizon_months": horizon,
                "horizon_years": years,
                "log_horizon_years": log(years),
                "target_label": "annualized_excess_forward_return",
                "annualization_factor": 12.0 / horizon,
                "notes": "Monthly decision dates with annualized compounded forward excess total return target.",
            }
        )
    return pd.DataFrame(rows)



def _build_stacked_panel(feature_master: pd.DataFrame, target_panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    panel = feature_master.merge(target_panel, on=["sleeve_id", "month_end"], how="left")
    panel["log_horizon_years"] = np.log(panel["horizon_years"])
    for horizon in LOCKED_HORIZONS:
        panel[f"horizon_{horizon}_flag"] = panel["horizon_months"].eq(horizon).astype(int)

    rows: list[dict[str, object]] = []
    interactions: list[dict[str, object]] = []
    for col, notes in [
        ("horizon_years", "Continuous horizon in years."),
        ("log_horizon_years", "Log horizon conditioning variable for multi-horizon models."),
        ("horizon_60_flag", "Five-year horizon dummy."),
        ("horizon_120_flag", "Ten-year horizon dummy."),
        ("horizon_180_flag", "Fifteen-year horizon dummy."),
    ]:
        rows.append(
            _feature_meta_row(
                feature_name=col,
                source_file="derived_v2_horizon",
                source_column=col,
                block_name="horizon_conditioning",
                geography="HORIZON",
                transform_type="conditioning_feature",
                native_frequency="monthly",
                lag_months=0,
                notes=notes,
                monthly_flag=0,
                hstack_flag=1,
                is_interaction=0,
                monthly_df=None,
                hstack_df=panel,
            )
        )

    def add_h_interaction(name: str, lhs: str, family: str, notes: str) -> None:
        panel[name] = panel[lhs] * panel["log_horizon_years"]
        interactions.append(
            {
                "interaction_name": name,
                "table_scope": "modeling_panel_hstack",
                "interaction_family": family,
                "lhs_feature": lhs,
                "rhs_feature": "log_horizon_years",
                "geography": "HORIZON",
                "notes": notes,
            }
        )
        rows.append(
            _feature_meta_row(
                feature_name=name,
                source_file="derived_interaction",
                source_column=f"{lhs} x log_horizon_years",
                block_name="interaction",
                geography="HORIZON",
                transform_type=family,
                native_frequency="monthly",
                lag_months=0,
                notes=notes,
                monthly_flag=0,
                hstack_flag=1,
                is_interaction=1,
                monthly_df=None,
                hstack_df=panel,
            )
        )

    for lhs, name, notes in [
        ("mom_12_1", "int_log_horizon_x_mom_12_1", "Horizon-conditioned momentum interaction."),
        ("vol_12m", "int_log_horizon_x_vol_12m", "Horizon-conditioned volatility interaction."),
        ("vix", "int_log_horizon_x_vix", "Horizon-conditioned VIX interaction."),
        ("us_real10y", "int_log_horizon_x_us_real10y", "Horizon-conditioned real-rate interaction."),
        ("cape_local", "int_log_horizon_x_cape_local", "Horizon-conditioned local CAPE interaction."),
        ("china_cli", "int_log_horizon_x_china_cli", "Horizon-conditioned China CLI interaction."),
        ("jp_pe_ratio", "int_log_horizon_x_jp_pe_ratio", "Horizon-conditioned Japan PE interaction."),
    ]:
        add_h_interaction(name, lhs, "predictor_x_log_horizon_years", notes)

    panel["baseline_trainable_flag"] = (
        panel["baseline_sample_inclusion_1m_flag"].eq(1)
        & panel["target_available_flag"].eq(1)
        & panel["proxy_flag"].eq(1)
    ).astype(int)
    panel["strict_trainable_flag"] = (
        panel["strict_all_feature_complete_flag"].eq(1)
        & panel["target_available_flag"].eq(1)
        & panel["proxy_flag"].eq(1)
    ).astype(int)

    panel = panel.sort_values(["sleeve_id", "month_end", "horizon_months"]).reset_index(drop=True)
    if panel.duplicated(subset=["sleeve_id", "month_end", "horizon_months"]).any():
        raise ValueError("modeling_panel_hstack has duplicate keys")
    return panel, rows, interactions



def _coverage_summary(feature_master: pd.DataFrame, target_panel: pd.DataFrame, stacked_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sleeve_id, chunk in feature_master.groupby("sleeve_id"):
        rows.append(
            {
                "table_name": "feature_master_monthly",
                "sleeve_id": sleeve_id,
                "horizon_months": np.nan,
                "first_date": chunk["month_end"].min(),
                "last_date": chunk["month_end"].max(),
                "row_count": int(len(chunk)),
                "target_available_rows": np.nan,
                "recommended_trainable_rows": int(chunk["recommended_feature_complete_flag"].sum()),
                "strict_trainable_rows": int(chunk["strict_all_feature_complete_flag"].sum()),
            }
        )

    available = target_panel[target_panel["target_available_flag"] == 1]
    for (sleeve_id, horizon), chunk in available.groupby(["sleeve_id", "horizon_months"]):
        rows.append(
            {
                "table_name": "target_panel_long_horizon",
                "sleeve_id": sleeve_id,
                "horizon_months": horizon,
                "first_date": chunk["month_end"].min(),
                "last_date": chunk["month_end"].max(),
                "row_count": int(len(chunk)),
                "target_available_rows": int(len(chunk)),
                "recommended_trainable_rows": np.nan,
                "strict_trainable_rows": np.nan,
            }
        )

    for (sleeve_id, horizon), chunk in stacked_panel.groupby(["sleeve_id", "horizon_months"]):
        rows.append(
            {
                "table_name": "modeling_panel_hstack",
                "sleeve_id": sleeve_id,
                "horizon_months": horizon,
                "first_date": chunk["month_end"].min(),
                "last_date": chunk["month_end"].max(),
                "row_count": int(len(chunk)),
                "target_available_rows": int(chunk["target_available_flag"].sum()),
                "recommended_trainable_rows": int(chunk["baseline_trainable_flag"].sum()),
                "strict_trainable_rows": int(chunk["strict_trainable_flag"].sum()),
            }
        )
    return pd.DataFrame(rows)



def _target_horizon_summary(target_panel: pd.DataFrame) -> pd.DataFrame:
    available = target_panel[target_panel["target_available_flag"] == 1].copy()
    grouped = available.groupby(["horizon_months", "sleeve_id"], as_index=False).agg(
        row_count=("annualized_excess_forward_return", "size"),
        first_date=("month_end", "min"),
        last_date=("month_end", "max"),
        mean_annualized_excess=("annualized_excess_forward_return", "mean"),
        std_annualized_excess=("annualized_excess_forward_return", "std"),
        mean_annualized_total=("annualized_total_forward_return", "mean"),
        mean_forward_volatility=("realized_forward_volatility", "mean"),
        mean_forward_max_drawdown=("realized_forward_max_drawdown", "mean"),
    )
    return grouped.sort_values(["horizon_months", "sleeve_id"]).reset_index(drop=True)



def _missingness_summary(feature_master: pd.DataFrame, stacked_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table_name, df, excluded in [
        ("feature_master_monthly", feature_master, ID_COLS | MONTHLY_NON_PREDICTOR_COLS),
        ("modeling_panel_hstack", stacked_panel, ID_COLS | MONTHLY_NON_PREDICTOR_COLS | STACKED_NON_PREDICTOR_COLS),
    ]:
        for col in df.columns:
            if col in excluded:
                continue
            first_valid, last_valid, nonmissing_count, nonmissing_share = _profile_feature_hstack(df, col)
            rows.append(
                {
                    "table_name": table_name,
                    "column_name": col,
                    "missing_count": int(df[col].isna().sum()),
                    "missing_share": float(df[col].isna().mean()),
                    "nonmissing_count": nonmissing_count,
                    "nonmissing_share": nonmissing_share,
                    "first_valid_date": first_valid,
                    "last_valid_date": last_valid,
                }
            )
    return pd.DataFrame(rows).sort_values(["table_name", "missing_share", "column_name"], ascending=[True, False, True]).reset_index(drop=True)



def _data_inventory(paths: V2Paths, external_inventory: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = pd.DataFrame(
        [
            {
                "file_id": "data/intermediate/feature_panel.csv",
                "container": "baseline_internal",
                "file_type": ".csv",
                "row_count": np.nan,
                "column_count": np.nan,
                "date_column": "month_end",
                "start_date": "2006-01-31",
                "end_date": "2026-02-28",
                "detected_frequency": "monthly",
                "feature_count_used": np.nan,
                "rejected_feature_count": np.nan,
                "v2_status": "used_feature_source",
                "v2_reason": "Frozen v1 monthly base feature panel.",
            },
            {
                "file_id": "data/intermediate/sleeve_month_end_prices.csv",
                "container": "baseline_internal",
                "file_type": ".csv",
                "row_count": np.nan,
                "column_count": np.nan,
                "date_column": "month_end",
                "start_date": "2006-01-31",
                "end_date": "2026-02-28",
                "detected_frequency": "monthly",
                "feature_count_used": np.nan,
                "rejected_feature_count": np.nan,
                "v2_status": "used_target_source",
                "v2_reason": "Frozen v1 sleeve month-end prices for long-horizon target construction.",
            },
            {
                "file_id": "data/intermediate/sleeve_monthly_returns.csv",
                "container": "baseline_internal",
                "file_type": ".csv",
                "row_count": np.nan,
                "column_count": np.nan,
                "date_column": "month_end",
                "start_date": "2006-01-31",
                "end_date": "2026-02-28",
                "detected_frequency": "monthly",
                "feature_count_used": np.nan,
                "rejected_feature_count": np.nan,
                "v2_status": "used_target_source",
                "v2_reason": "Frozen v1 realized monthly returns for forward volatility and drawdown labels.",
            },
            {
                "file_id": "data/intermediate/tb3ms_monthly.csv",
                "container": "baseline_internal",
                "file_type": ".csv",
                "row_count": np.nan,
                "column_count": np.nan,
                "date_column": "month_end",
                "start_date": "2006-01-31",
                "end_date": "2026-02-28",
                "detected_frequency": "monthly",
                "feature_count_used": np.nan,
                "rejected_feature_count": np.nan,
                "v2_status": "used_target_source",
                "v2_reason": "Frozen v1 monthly TB3MS series for compounded forward risk-free return.",
            },
        ]
    )
    return pd.concat([baseline_rows, external_inventory], ignore_index=True).sort_values("file_id").reset_index(drop=True)



def _render_design_report(
    *,
    feature_master: pd.DataFrame,
    target_panel: pd.DataFrame,
    stacked_panel: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    interaction_dictionary: pd.DataFrame,
    external_inventory: pd.DataFrame,
    rejected_features: pd.DataFrame,
) -> str:
    used_files = external_inventory.loc[external_inventory["v2_status"] == "used_feature_source", "file_id"].tolist()
    horizon_counts = (
        target_panel.groupby("horizon_months")["target_available_flag"].sum().astype(int).to_dict()
    )
    trainable_counts = (
        stacked_panel.groupby("horizon_months")["baseline_trainable_flag"].sum().astype(int).to_dict()
    )
    strict_trainable_total = int(stacked_panel["strict_trainable_flag"].sum())

    lines = [
        f"# XOPTPOE {V2_VERSION} Final Data Design Report",
        "",
        "## Design Summary",
        "- Frozen `v1` data outputs remain untouched.",
        "- `v2_long_horizon` is a new canonical data package for long-horizon SAA modeling.",
        "- Monthly predictors are built at sleeve-month grain and stacked across 5Y / 10Y / 15Y horizons.",
        "- No predictive models or portfolio backtests are built in this package.",
        "",
        "## Main Table Grains",
        "- `feature_master_monthly`: one row per (`sleeve_id`, `month_end`).",
        "- `target_panel_long_horizon`: one row per (`sleeve_id`, `month_end`, `horizon_months`).",
        "- `modeling_panel_hstack`: one row per (`sleeve_id`, `month_end`, `horizon_months`).",
        "",
        "## Long-Horizon Target Definition",
        "For sleeve `a`, decision month `t`, and horizon `H` in {60, 120, 180}:",
        "- compounded total forward return = `P[a, t+H] / P[a, t]`",
        "- compounded risk-free forward return = `prod_{i=1..H} (1 + TB3MS[t+i] / 1200)`",
        "- annualized total forward return = `gross_total_forward_return^(12/H) - 1`",
        "- annualized risk-free forward return = `gross_rf_forward_return^(12/H) - 1`",
        "- annualized excess forward return = `(gross_total_forward_return / gross_rf_forward_return)^(12/H) - 1`",
        "- auxiliary labels: cumulative total return, realized forward volatility, realized forward max drawdown",
        "",
        "## Source Files Used",
    ]
    for file_id in used_files:
        lines.append(f"- `{file_id}`")

    lines.extend(
        [
            "",
            "## Added Feature Blocks",
            "- OECD leading indicators: US / EA / JP / China LI, business-confidence, consumer-confidence, plus selected OECD BTS survey measures.",
            "- CAPE ratios: China, Europe, Japan, USA, plus mapped `cape_local`.",
            "- China macro block: CLI, CMI, official GDP growth, PMI variants, CPI subcomponents, industrial production, M2, fiscal flows, house-price indicators.",
            "- China valuation block: dividend yield, PE, PTB, EM-relative spreads, and China SSE composite USD market features.",
            "- Japan enrichment block: OECD leading alternative, nominal GDP, policy rate, CPI variants, labor, industrial production, Tankan, PE, TOPIX market features, buyback-index market features, alternative bond-yield / CPI file.",
            "- EM/global valuation block: EM and global dividend yield, PTB, PE, and EM-global spreads.",
            "- HY / stress block: US HY OAS/effective yield and European HY OAS/effective yield.",
            "- EU IG market block: USD-converted Euro IG total-return index history and market transforms.",
            "",
            "## China Treatment",
            "- China is included as a feature block only.",
            "- No standalone China sleeve target is added because the uploaded China workbook is a market-history feature source, not a clean investable USD adjusted-close sleeve target with consistent provenance.",
            "",
            "## Interaction Families Created",
            "- sleeve dummy x predictor",
            "- asset-class-group dummy x predictor",
            "- stress x valuation",
            "- stress x momentum",
            "- China block x EM sleeve relevance",
            "- Japan block x Japan sleeve relevance",
            "- CAPE x real-rate",
            "- CLI x slope/spread",
            "- predictor x log_horizon_years in the stacked panel",
            "",
            "## Usable Row Counts",
        ]
    )
    for horizon in LOCKED_HORIZONS:
        lines.append(
            f"- {horizon} months: target_available_rows={horizon_counts.get(horizon, 0)}, baseline_trainable_rows={trainable_counts.get(horizon, 0)}"
        )

    lines.extend(
        [
            "",
            "## Target-History Limitation",
            "- No clean benchmark/index backfill was added for the locked sleeves.",
            "- Long-horizon availability remains constrained by the ETF history inherited from frozen `v1` month-end prices.",
            "- This especially truncates 10Y and 15Y usable rows, but it is documented rather than silently backfilled with mixed target definitions.",
            "",
            "## Recommended Main Modeling Table",
            "- Use `data/final_v2_long_horizon/modeling_panel_hstack.parquet` as the primary table for the later deep-learning section.",
            "- `baseline_trainable_flag` is the recommended starting filter when using explicit imputation/masking for enrichment features.",
            f"- `strict_trainable_flag` is the literal all-features complete-case diagnostic. In the current build it yields {strict_trainable_total} long-horizon rows because the richest enrichment set only becomes fully complete after the available target window.",
        ]
    )

    if not rejected_features.empty:
        rejected_preview = rejected_features.head(10)
        lines.extend(["", "## Rejected Candidate Features", "- The following candidates were scanned but not kept because usable history within the `v2` window was too thin:"])
        for _, row in rejected_preview.iterrows():
            lines.append(
                f"- `{row['feature_name']}` from `{row['source_file']}` ({row['rejection_reason']}, nonmissing_count={int(row['nonmissing_count'])})"
            )

    lines.extend(
        [
            "",
            "## Direct Answers",
            "1. Main table grains: `feature_master_monthly` = (`sleeve_id`, `month_end`); `target_panel_long_horizon` = (`sleeve_id`, `month_end`, `horizon_months`); `modeling_panel_hstack` = (`sleeve_id`, `month_end`, `horizon_months`).",
            f"2. Uploaded files used: {', '.join(used_files)}.",
            "3. Variables added: OECD/CAPE blocks, China macro and valuation, Japan enrichment, EM/global valuation, HY/IG stress, and workbook-derived market features for China SSE, Japan buyback, and Euro IG.",
            "4. Exact long-horizon targets: annualized compounded total return, annualized compounded risk-free return, annualized compounded excess return, plus cumulative return, realized forward volatility, and realized forward max drawdown.",
            f"5. Usable target rows: 5Y={horizon_counts.get(60, 0)}, 10Y={horizon_counts.get(120, 0)}, 15Y={horizon_counts.get(180, 0)}.",
            "6. Target-history limitations remain: yes; all horizons are still limited by the ETF-based sleeve history inherited from frozen `v1`.",
            "7. China included as: feature block only, not a sleeve target.",
            f"8. Interaction families created: {', '.join(sorted(interaction_dictionary['interaction_family'].unique().tolist()))}.",
            "9. Main modeling table for later deep learning: `modeling_panel_hstack`.",
            "10. Dataset ready for the model section: yes, with documented 10Y/15Y sample limits, explicit missingness/trainability flags, and the expectation that enrichment features will need masking or imputation rather than full complete-case filtering.",
        ]
    )
    return "\n".join(lines)



def build_v2_long_horizon_dataset(project_root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Build and persist the XOPTPOE v2 long-horizon dataset package."""
    paths = default_paths(project_root)
    paths.ensure_directories()
    inputs = _load_inputs(paths)

    additional_state, additional_feature_meta, rejected_features = build_additional_monthly_state(
        paths,
        inputs["feature_panel"]["month_end"],
    )
    feature_master, manual_rows, monthly_interactions = _build_feature_master_monthly(inputs, additional_state)

    target_panel = build_long_horizon_targets(
        month_end_prices=inputs["month_end_prices"],
        monthly_returns=inputs["monthly_returns"],
        tb3ms_monthly=inputs["tb3ms_monthly"],
        horizons=LOCKED_HORIZONS,
    )
    horizon_manifest = _build_horizon_manifest()
    modeling_panel_hstack, stacked_rows, stacked_interactions = _build_stacked_panel(feature_master, target_panel)

    feature_rows = _baseline_feature_catalog(feature_master)
    for row in additional_feature_meta.to_dict("records"):
        feature_rows.append(
            _feature_meta_row(
                feature_name=row["feature_name"],
                source_file=row["source_file"],
                source_column=row["source_column"],
                block_name=row["block_name"],
                geography=row["geography"],
                transform_type=row["transform_type"],
                native_frequency=row["native_frequency"],
                lag_months=int(row["lag_months"]),
                notes=row["notes"],
                monthly_flag=1,
                hstack_flag=1,
                is_interaction=0,
                monthly_df=feature_master,
                hstack_df=None,
            )
        )
    feature_rows.extend(manual_rows)
    feature_rows.extend(stacked_rows)
    feature_dictionary = pd.DataFrame(feature_rows).drop_duplicates(subset=["feature_name"]).sort_values("feature_name").reset_index(drop=True)

    interaction_dictionary = pd.DataFrame(monthly_interactions + stacked_interactions)
    if not interaction_dictionary.empty:
        profiles = []
        for _, row in interaction_dictionary.iterrows():
            if row["table_scope"] == "feature_master_monthly":
                first_valid, last_valid, nonmissing_count, nonmissing_share = _profile_feature(feature_master, row["interaction_name"])
            else:
                first_valid, last_valid, nonmissing_count, nonmissing_share = _profile_feature_hstack(modeling_panel_hstack, row["interaction_name"])
            out = row.to_dict()
            out.update(
                {
                    "first_valid_date": first_valid,
                    "last_valid_date": last_valid,
                    "nonmissing_count": nonmissing_count,
                    "nonmissing_share": nonmissing_share,
                }
            )
            profiles.append(out)
        interaction_dictionary = pd.DataFrame(profiles).sort_values(["table_scope", "interaction_name"]).reset_index(drop=True)

    external_inventory = build_external_file_inventory(paths, additional_feature_meta, rejected_features)
    data_inventory = _data_inventory(paths, external_inventory)
    coverage_summary = _coverage_summary(feature_master, target_panel, modeling_panel_hstack)
    target_horizon_summary = _target_horizon_summary(target_panel)
    missingness_summary = _missingness_summary(feature_master, modeling_panel_hstack)
    design_report = _render_design_report(
        feature_master=feature_master,
        target_panel=target_panel,
        stacked_panel=modeling_panel_hstack,
        feature_dictionary=feature_dictionary,
        interaction_dictionary=interaction_dictionary,
        external_inventory=external_inventory,
        rejected_features=rejected_features,
    )

    write_parquet(feature_master, paths.data_final_v2_dir / "feature_master_monthly.parquet")
    write_csv(feature_master, paths.data_final_v2_dir / "feature_master_monthly.csv")
    write_parquet(target_panel, paths.data_final_v2_dir / "target_panel_long_horizon.parquet")
    write_parquet(modeling_panel_hstack, paths.data_final_v2_dir / "modeling_panel_hstack.parquet")
    write_csv(modeling_panel_hstack, paths.data_final_v2_dir / "modeling_panel_hstack.csv")
    write_csv(feature_dictionary, paths.data_final_v2_dir / "feature_dictionary.csv")
    write_csv(interaction_dictionary, paths.data_final_v2_dir / "interaction_dictionary.csv")
    write_csv(horizon_manifest, paths.data_final_v2_dir / "horizon_manifest.csv")
    write_csv(data_inventory, paths.data_final_v2_dir / "data_inventory.csv")

    write_text(design_report, paths.reports_v2_dir / "final_data_design_report.md")
    write_csv(coverage_summary, paths.reports_v2_dir / "coverage_summary.csv")
    write_csv(target_horizon_summary, paths.reports_v2_dir / "target_horizon_summary.csv")
    write_csv(missingness_summary, paths.reports_v2_dir / "missingness_summary.csv")

    return {
        "feature_master_monthly": feature_master,
        "target_panel_long_horizon": target_panel,
        "modeling_panel_hstack": modeling_panel_hstack,
        "feature_dictionary": feature_dictionary,
        "interaction_dictionary": interaction_dictionary,
        "horizon_manifest": horizon_manifest,
        "data_inventory": data_inventory,
        "coverage_summary": coverage_summary,
        "target_horizon_summary": target_horizon_summary,
        "missingness_summary": missingness_summary,
        "rejected_features": rejected_features,
    }
