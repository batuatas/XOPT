"""Build the versioned v4 expanded-universe long-horizon dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_data.features.build_features import _add_relative_features, _build_technical_features
from xoptpoe_data.targets.build_monthly_targets import (
    build_monthly_realized_returns,
    build_target_panel,
    collapse_target_to_month_end_prices,
)
from xoptpoe_data.targets.fetch_targets import YahooFinanceTargetAdapter, fetch_sleeve_target_raw
from xoptpoe_v2_data.build import (
    ID_COLS,
    MONTHLY_NON_PREDICTOR_COLS,
    STACKED_NON_PREDICTOR_COLS,
    _build_horizon_manifest,
    _build_stacked_panel,
    _coverage_summary,
    _data_inventory,
    _feature_meta_row,
    _missingness_summary,
    _profile_feature,
    _profile_feature_hstack,
    _target_horizon_summary,
)
from xoptpoe_v2_data.config import default_paths as default_v2_paths
from xoptpoe_v2_data.io import load_csv, write_csv, write_parquet, write_text
from xoptpoe_v2_data.sources import build_additional_monthly_state, build_external_file_inventory
from xoptpoe_v2_data.targets import build_long_horizon_targets
from xoptpoe_v4_data.config import (
    ASSET_CLASS_GROUP_BY_SLEEVE_V4,
    BASE_FEATURE_START,
    EURO_FIXED_INCOME_SLEEVES,
    EXPOSURE_REGION_BY_SLEEVE_V4,
    FX_TICKER_EURUSD,
    GLOBAL_REQUIRED_COLS,
    LOCAL_ALIAS_COLS,
    LOCAL_BLOCK_BY_SLEEVE_V4,
    LOCKED_HORIZONS,
    LOCKED_SLEEVES_V4,
    NO_LOCAL_CORE_SLEEVES_V4,
    TARGET_FETCH_START,
    TECH_COLS,
    TEMPLATE_SLEEVE_BY_V4_SLEEVE,
    V4_VERSION,
    V4Paths,
    default_paths,
)


def _group_slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def _load_inputs(paths: V4Paths) -> dict[str, pd.DataFrame]:
    return {
        "feature_panel": load_csv(paths.data_intermediate_dir / "feature_panel.csv", parse_dates=["month_end"]),
        "modeling_panel": load_csv(paths.data_final_dir / "modeling_panel.csv", parse_dates=["month_end"]),
        "tb3ms_monthly": load_csv(paths.data_intermediate_dir / "tb3ms_monthly.csv", parse_dates=["month_end"]),
    }


def _load_versioned_manifests(paths: V4Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_master = load_csv(
        paths.config_dir / "asset_master_seed_v4_expanded_universe.csv",
        parse_dates=["start_date_target"],
    )
    target_manifest = load_csv(
        paths.config_dir / "target_series_manifest_v4_expanded_universe.csv",
        parse_dates=["start_date_target"],
    )
    return asset_master, target_manifest


def _build_macro_mapping_v4(asset_master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sleeve_id in LOCKED_SLEEVES_V4:
        local_block = LOCAL_BLOCK_BY_SLEEVE_V4.get(sleeve_id)
        if local_block is not None:
            rows.append(
                {
                    "sleeve_id": sleeve_id,
                    "block_role": "local",
                    "geo_block": local_block,
                    "mapping_priority": 1,
                    "notes": "Versioned v4 local block mapping.",
                }
            )
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "block_role": "global",
                "geo_block": "GLOBAL",
                "mapping_priority": 2,
                "notes": "All v4 sleeves consume global/stress block.",
            }
        )
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "block_role": "usd",
                "geo_block": "GLOBAL",
                "mapping_priority": 3,
                "notes": "All final v4 targets are modeled as USD unhedged.",
            }
        )
    return pd.DataFrame(rows).sort_values(["sleeve_id", "mapping_priority"]).reset_index(drop=True)


def _fetch_target_raw_series(target_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adapter = YahooFinanceTargetAdapter()

    direct_manifest = target_manifest.loc[target_manifest["target_rule_family"].eq("USD_LISTED_ETF_DIRECT")].copy()
    euro_manifest = target_manifest.loc[target_manifest["target_rule_family"].eq("LOCAL_ETF_PLUS_FX_TO_USD")].copy()

    direct_raw = fetch_sleeve_target_raw(
        asset_master=direct_manifest[["sleeve_id", "ticker"]],
        adapter=adapter,
        start_date=TARGET_FETCH_START,
        end_date=None,
    )

    local_rows = euro_manifest[["sleeve_id", "ticker"]].drop_duplicates().sort_values("sleeve_id")
    euro_local_raw = fetch_sleeve_target_raw(
        asset_master=local_rows,
        adapter=adapter,
        start_date=TARGET_FETCH_START,
        end_date=None,
    )

    fx_raw = adapter.fetch_ticker(FX_TICKER_EURUSD, start_date=TARGET_FETCH_START, end_date=None)
    fx_raw["sleeve_id"] = "FX_EURUSD"
    fx_raw["source_id"] = adapter.source_id
    return direct_raw, euro_local_raw, fx_raw


def _synthesize_usd_month_end_prices(
    *,
    local_month_end_prices: pd.DataFrame,
    fx_month_end_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fx = fx_month_end_prices[["month_end", "adj_close"]].copy().rename(columns={"adj_close": "fx_adj_close"})
    fx["fx_ret_1m"] = fx["fx_adj_close"].pct_change()

    outputs: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for sleeve_id, chunk in local_month_end_prices.groupby("sleeve_id"):
        work = chunk.sort_values("month_end").copy()
        work["local_ret_1m"] = work["adj_close"].pct_change()
        work = work.merge(fx[["month_end", "fx_adj_close", "fx_ret_1m"]], on="month_end", how="left")
        work["usd_ret_1m"] = (1.0 + work["local_ret_1m"]) * (1.0 + work["fx_ret_1m"]) - 1.0
        work["synthetic_adj_close"] = 100.0 * (1.0 + work["usd_ret_1m"].fillna(0.0)).cumprod()
        work["synthetic_close"] = work["synthetic_adj_close"]
        work["target_currency"] = "USD"
        work["construction_rule"] = "LOCAL_ETF_PLUS_FX_TO_USD"

        outputs.append(
            work[
                [
                    "sleeve_id",
                    "ticker",
                    "month_end",
                    "trade_date",
                    "synthetic_adj_close",
                    "synthetic_close",
                ]
            ].rename(columns={"synthetic_adj_close": "adj_close", "synthetic_close": "close"})
        )
        audit_rows.append(
            {
                "sleeve_id": sleeve_id,
                "first_month_end": work["month_end"].min(),
                "last_month_end": work["month_end"].max(),
                "first_valid_usd_return_month": work.loc[work["usd_ret_1m"].notna(), "month_end"].min(),
                "missing_fx_months": int(work["fx_adj_close"].isna().sum()),
                "missing_local_months": int(work["adj_close"].isna().sum()),
            }
        )

    return (
        pd.concat(outputs, ignore_index=True).sort_values(["sleeve_id", "month_end"]).reset_index(drop=True),
        pd.DataFrame(audit_rows).sort_values("sleeve_id").reset_index(drop=True),
    )


def _build_target_store_v4(target_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    direct_raw, euro_local_raw, fx_raw = _fetch_target_raw_series(target_manifest)

    direct_month_end = collapse_target_to_month_end_prices(direct_raw)
    euro_local_month_end = collapse_target_to_month_end_prices(euro_local_raw)
    fx_month_end = collapse_target_to_month_end_prices(
        fx_raw[["sleeve_id", "ticker", "trade_date", "adj_close", "close"]].copy()
    )
    euro_usd_month_end, euro_fx_audit = _synthesize_usd_month_end_prices(
        local_month_end_prices=euro_local_month_end,
        fx_month_end_prices=fx_month_end,
    )

    month_end_prices = (
        pd.concat([direct_month_end, euro_usd_month_end], ignore_index=True)
        .sort_values(["sleeve_id", "month_end"])
        .reset_index(drop=True)
    )
    monthly_returns = build_monthly_realized_returns(month_end_prices)

    if set(month_end_prices["sleeve_id"].unique()) != set(LOCKED_SLEEVES_V4):
        missing = sorted(set(LOCKED_SLEEVES_V4) - set(month_end_prices["sleeve_id"].unique()))
        raise ValueError(f"Missing v4 sleeves in month_end_prices: {missing}")

    raw_store = {
        "target_raw_direct": direct_raw,
        "target_raw_euro_local": euro_local_raw,
        "fx_raw_eurusd": fx_raw,
        "target_month_end_direct": direct_month_end,
        "target_month_end_euro_local": euro_local_month_end,
        "fx_month_end_eurusd": fx_month_end,
        "target_month_end_euro_usd_synth": euro_usd_month_end,
        "euro_fx_audit": euro_fx_audit,
    }
    return month_end_prices, monthly_returns, raw_store


def _build_feature_base_for_sleeve(
    *,
    sleeve_id: str,
    template_sleeve: str,
    base_feature_panel: pd.DataFrame,
    technical_panel: pd.DataFrame,
    max_month_end: pd.Timestamp,
) -> pd.DataFrame:
    template = base_feature_panel.loc[base_feature_panel["sleeve_id"].eq(template_sleeve)].copy()
    sleeve_tech = technical_panel.loc[technical_panel["sleeve_id"].eq(sleeve_id)].copy()

    sleeve_tech = sleeve_tech.loc[sleeve_tech["month_end"] >= pd.Timestamp(BASE_FEATURE_START)].copy()
    sleeve_tech = sleeve_tech.loc[sleeve_tech["month_end"] <= max_month_end].copy()

    work = template.drop(columns=list(TECH_COLS), errors="ignore").merge(
        sleeve_tech[["month_end", *TECH_COLS]],
        on="month_end",
        how="left",
    )
    work["sleeve_id"] = sleeve_id

    tech_ok = work[list(TECH_COLS)].notna().all(axis=1)
    work["feature_complete_flag"] = (work["feature_complete_flag"].fillna(0).astype(int).eq(1) & tech_ok).astype(int)
    return work[base_feature_panel.columns].sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)


def _build_feature_base_panel_v4(
    *,
    base_feature_panel: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    technical_panel = _add_relative_features(_build_technical_features(monthly_returns))
    max_month_end = pd.to_datetime(base_feature_panel["month_end"]).max()

    frames = [
        _build_feature_base_for_sleeve(
            sleeve_id=sleeve_id,
            template_sleeve=TEMPLATE_SLEEVE_BY_V4_SLEEVE[sleeve_id],
            base_feature_panel=base_feature_panel,
            technical_panel=technical_panel,
            max_month_end=max_month_end,
        )
        for sleeve_id in LOCKED_SLEEVES_V4
    ]
    out = pd.concat(frames, ignore_index=True).sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)
    if out.duplicated(subset=["sleeve_id", "month_end"]).any():
        raise ValueError("v4 base feature panel has duplicate (sleeve_id, month_end) keys")
    return out


def _replace_meta_row(rows: list[dict[str, object]], row: dict[str, object]) -> list[dict[str, object]]:
    rows = [r for r in rows if r.get("feature_name") != row.get("feature_name")]
    rows.append(row)
    return rows


def _add_manual_feature(
    panel: pd.DataFrame,
    rows: list[dict[str, object]],
    *,
    feature_name: str,
    block_name: str,
    geography: str,
    transform_type: str,
    notes: str,
) -> list[dict[str, object]]:
    return _replace_meta_row(
        rows,
        _feature_meta_row(
            feature_name=feature_name,
            source_file="derived_v4",
            source_column=feature_name,
            block_name=block_name,
            geography=geography,
            transform_type=transform_type,
            native_frequency="monthly",
            lag_months=0,
            notes=notes,
            monthly_flag=1,
            hstack_flag=1,
            is_interaction=0,
            monthly_df=panel,
            hstack_df=None,
        ),
    )


def _extend_manual_monthly_features_v4(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    work = panel.copy()
    rows: list[dict[str, object]] = []
    interactions: list[dict[str, object]] = []

    china_mask = work["sleeve_id"].eq("EQ_CN")
    work["oecd_business_conf_local"] = np.select(
        [
            work["geo_block_local"].eq("US"),
            work["geo_block_local"].eq("EURO_AREA"),
            work["geo_block_local"].eq("JAPAN"),
            china_mask,
        ],
        [work["oecd_bcicp_us"], work["oecd_bcicp_ea"], work["oecd_bcicp_jp"], work["oecd_bcicp_cn"]],
        default=np.nan,
    )
    work["oecd_consumer_conf_local"] = np.select(
        [
            work["geo_block_local"].eq("US"),
            work["geo_block_local"].eq("EURO_AREA"),
            work["geo_block_local"].eq("JAPAN"),
            china_mask,
        ],
        [work["oecd_ccicp_us"], work["oecd_ccicp_ea"], work["oecd_ccicp_jp"], work["oecd_ccicp_cn"]],
        default=np.nan,
    )
    work["oecd_activity_proxy_local"] = np.select(
        [
            work["geo_block_local"].eq("US"),
            work["geo_block_local"].eq("EURO_AREA"),
            work["geo_block_local"].eq("JAPAN"),
            china_mask,
        ],
        [work["oecd_li_us"], work["oecd_bcicp_ea"], work["oecd_li_jp"], work["oecd_li_cn"]],
        default=np.nan,
    )
    work["cape_local"] = np.select(
        [
            work["geo_block_local"].eq("US"),
            work["geo_block_local"].eq("EURO_AREA"),
            work["geo_block_local"].eq("JAPAN"),
            china_mask,
        ],
        [work["cape_usa"], work["cape_europe"], work["cape_japan"], work["cape_china"]],
        default=np.nan,
    )

    for feature_name, geography, notes in [
        ("oecd_business_conf_local", "LOCAL_MAPPED", "Mapped local business-confidence series, extended to China for EQ_CN."),
        ("oecd_consumer_conf_local", "LOCAL_MAPPED", "Mapped local consumer-confidence series, extended to China for EQ_CN."),
        ("oecd_activity_proxy_local", "LOCAL_MAPPED", "Mapped local activity proxy, extended to China OECD CLI for EQ_CN."),
        ("cape_local", "LOCAL_MAPPED", "Mapped local CAPE, extended to China CAPE for EQ_CN."),
    ]:
        rows = _add_manual_feature(
            work,
            rows,
            feature_name=feature_name,
            block_name="local_mapping",
            geography=geography,
            transform_type="mapped_alias",
            notes=notes,
        )

    work["eq_em_relevance_flag"] = work["sleeve_id"].eq("EQ_EM").astype(int)
    work["jp_relevance_flag"] = work["sleeve_id"].eq("EQ_JP").astype(int)
    work["no_local_block_flag"] = work["sleeve_id"].isin(NO_LOCAL_CORE_SLEEVES_V4).astype(int)
    for sleeve_id in LOCKED_SLEEVES_V4:
        work[f"sleeve_dummy_{sleeve_id.lower()}"] = work["sleeve_id"].eq(sleeve_id).astype(int)

    for label in sorted(set(ASSET_CLASS_GROUP_BY_SLEEVE_V4.values())):
        work[f"asset_group_dummy_{_group_slug(label)}"] = work["asset_class_group"].eq(label).astype(int)

    dummy_cols = [
        "eq_em_relevance_flag",
        "jp_relevance_flag",
        "no_local_block_flag",
        *[f"sleeve_dummy_{sleeve.lower()}" for sleeve in LOCKED_SLEEVES_V4],
        *[f"asset_group_dummy_{_group_slug(label)}" for label in sorted(set(ASSET_CLASS_GROUP_BY_SLEEVE_V4.values()))],
    ]
    for col in dummy_cols:
        rows = _add_manual_feature(
            work,
            rows,
            feature_name=col,
            block_name="metadata_dummy",
            geography="SLEEVE_SPECIFIC",
            transform_type="dummy_flag",
            notes="Static sleeve/group relevance indicator for v4 interactions.",
        )

    def add_interaction(name: str, lhs: str, rhs: str, family: str, geography: str, notes: str) -> None:
        work[name] = work[lhs] * work[rhs]
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
                monthly_df=work,
                hstack_df=None,
            )
        )

    for sleeve_id in LOCKED_SLEEVES_V4:
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

    for label in sorted(set(ASSET_CLASS_GROUP_BY_SLEEVE_V4.values())):
        group_slug = _group_slug(label)
        group_dummy = f"asset_group_dummy_{group_slug}"
        add_interaction(
            f"int_{group_slug}_x_vix",
            "vix",
            group_dummy,
            "asset_group_dummy_x_predictor",
            label.upper(),
            f"Asset-group VIX exposure interaction for {label} sleeves.",
        )
        add_interaction(
            f"int_{group_slug}_x_us_real10y",
            "us_real10y",
            group_dummy,
            "asset_group_dummy_x_predictor",
            label.upper(),
            f"Asset-group real-rate interaction for {label} sleeves.",
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
        if lhs in work.columns and rhs in work.columns:
            add_interaction(name, lhs, rhs, family, geography, notes)

    return work, rows, interactions


def _build_feature_master_monthly_v4(
    *,
    inputs: dict[str, pd.DataFrame],
    additional_state: pd.DataFrame,
    asset_master_v4: pd.DataFrame,
    base_feature_panel_v4: pd.DataFrame,
    sample_flags_v4: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    feature_master = base_feature_panel_v4.copy()

    sample_flag = sample_flags_v4.copy().rename(columns={"sample_inclusion_flag": "baseline_sample_inclusion_1m_flag"})
    asset_cols = asset_master_v4[
        ["sleeve_id", "ticker", "sleeve_name", "proxy_flag", "asset_class_group"]
    ].copy()
    feature_master = feature_master.merge(asset_cols, on="sleeve_id", how="left", suffixes=("", "_asset"))
    if "asset_class_group_asset" in feature_master.columns:
        feature_master = feature_master.drop(columns=["asset_class_group"])
        feature_master = feature_master.rename(columns={"asset_class_group_asset": "asset_class_group"})

    feature_master = feature_master.merge(sample_flag, on=["sleeve_id", "month_end"], how="left")
    feature_master["baseline_sample_inclusion_1m_flag"] = feature_master["baseline_sample_inclusion_1m_flag"].fillna(0).astype(int)
    feature_master["geo_block_local"] = feature_master["sleeve_id"].map(LOCAL_BLOCK_BY_SLEEVE_V4)
    feature_master["geo_block_global"] = "GLOBAL"
    feature_master["exposure_region"] = feature_master["sleeve_id"].map(EXPOSURE_REGION_BY_SLEEVE_V4)

    feature_master = feature_master.merge(additional_state, on="month_end", how="left")
    feature_master, manual_rows, interaction_rows = _extend_manual_monthly_features_v4(feature_master)

    additional_feature_cols = [col for col in additional_state.columns if col != "month_end"] + [
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


def _baseline_feature_catalog_v4(feature_master: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    excluded = ID_COLS | MONTHLY_NON_PREDICTOR_COLS | {"sample_inclusion_flag", "target_quality_flag", "ret_fwd_1m", "rf_fwd_1m", "excess_ret_fwd_1m"}
    baseline_cols = [col for col in feature_master.columns if col not in excluded]
    for col in baseline_cols:
        if col.startswith(("ret_", "mom_", "vol_", "maxdd_", "rel_")):
            block_name = "baseline_technical"
            geography = "SLEEVE_SPECIFIC"
            transform_type = "derived_technical"
            lag_months = 0
            notes = "Versioned v4 technical feature built from sleeve monthly returns."
        elif col.startswith(("infl_", "unemp_", "short_rate_", "long_rate_", "term_slope_")):
            block_name = "baseline_macro_canonical"
            geography = "GLOBAL_CANONICAL"
            transform_type = "delta_1m" if col.endswith("_delta_1m") else "level_or_yoy"
            lag_months = 1
            notes = "Inherited canonical macro feature from frozen baseline panel."
        elif col.startswith("local_"):
            block_name = "baseline_macro_alias"
            geography = "LOCAL_MAPPED"
            transform_type = "compatibility_alias"
            lag_months = 1
            notes = "Derived local alias retained for compatibility in the v4 feature layer."
        elif col.startswith(("usd_broad", "vix", "us_real10y", "ig_oas", "oil_wti")) or col == "macro_stale_flag":
            block_name = "baseline_global_macro"
            geography = "GLOBAL"
            transform_type = "delta_1m" if col.endswith(("_delta_1m", "_logchg_1m", "_logchg_12m")) else "level_or_flag"
            lag_months = 0
            notes = "Inherited global/stress feature from the frozen baseline panel."
        else:
            continue
        rows.append(
            _feature_meta_row(
                feature_name=col,
                source_file="data/intermediate/feature_panel.csv",
                source_column=col,
                block_name=block_name,
                geography=geography,
                transform_type=transform_type,
                native_frequency="monthly",
                lag_months=lag_months,
                notes=notes,
                monthly_flag=1,
                hstack_flag=1,
                is_interaction=0,
                monthly_df=feature_master,
                hstack_df=None,
            )
        )
    return rows


def _render_design_report(
    *,
    feature_master: pd.DataFrame,
    target_panel: pd.DataFrame,
    modeling_panel_hstack: pd.DataFrame,
    target_manifest: pd.DataFrame,
    raw_store: dict[str, pd.DataFrame],
    used_files: list[str],
) -> str:
    horizon_counts = target_panel.groupby("horizon_months")["target_available_flag"].sum().astype(int).to_dict()
    trainable_counts = modeling_panel_hstack.groupby("horizon_months")["baseline_trainable_flag"].sum().astype(int).to_dict()
    euro_summary = (
        target_panel.loc[target_panel["sleeve_id"].isin(sorted(EURO_FIXED_INCOME_SLEEVES))]
        .groupby(["sleeve_id", "horizon_months"])["target_available_flag"].sum()
        .astype(int)
        .reset_index()
    )

    lines = [
        f"# XOPTPOE {V4_VERSION} Final Data Design Report",
        "",
        "## Design Summary",
        "- Frozen `v1`, `v2_long_horizon`, and `v3_long_horizon_china` outputs remain untouched.",
        f"- `{V4_VERSION}` is a new versioned 15-sleeve first-build branch.",
        "- The locked v4 sleeve roster and target rules follow the written governance lock exactly.",
        "- No downstream modeling or scenario generation is performed in this package beyond first-pass modeling-prep scaffolding.",
        "",
        "## Locked Sleeve Roster",
    ]
    for row in target_manifest.itertuples(index=False):
        lines.append(f"- `{row.sleeve_id}` | `{row.asset_class_group}` | `{row.ticker}` | `{row.target_rule_family}`")
    lines.extend(
        [
            "",
            "## Euro Fixed-Income Rule",
            "- `FI_EU_GOVT`, `CR_EU_IG`, and `CR_EU_HY` are built from local-currency investable ETF returns plus month-end EUR/USD conversion to USD.",
            "- The resulting target series are synthetic USD-unhedged total-return indices used only for this locked euro fixed-income family.",
            "",
            "## LISTED_RE Interpretation",
            "- `LISTED_RE` is implemented as ex-U.S. listed real estate.",
            "- `RE_US` remains a separate U.S. real-estate sleeve and coexists with `LISTED_RE`.",
            "",
            "## Main Table Grains",
            "- `feature_master_monthly`: one row per (`sleeve_id`, `month_end`).",
            "- `target_panel_long_horizon`: one row per (`sleeve_id`, `month_end`, `horizon_months`).",
            "- `modeling_panel_hstack`: one row per (`sleeve_id`, `month_end`, `horizon_months`).",
            "",
            "## Used Sources",
        ]
    )
    for file_id in used_files:
        lines.append(f"- `{file_id}`")
    for row in target_manifest.itertuples(index=False):
        lines.append(f"- `Yahoo Finance adjusted-close history for {row.ticker}`")
    lines.append(f"- `Yahoo Finance adjusted-close history for {FX_TICKER_EURUSD}`")
    lines.extend(["", "## Usable Row Counts"])
    for horizon in LOCKED_HORIZONS:
        lines.append(
            f"- {horizon} months: total_target_rows={horizon_counts.get(horizon, 0)}, total_baseline_trainable_rows={trainable_counts.get(horizon, 0)}"
        )
    lines.extend(["", "## Euro Fixed-Income Sleeve Coverage"])
    for row in euro_summary.itertuples(index=False):
        lines.append(f"- `{row.sleeve_id}` @ {int(row.horizon_months)}m: target_available_rows={int(row.target_available_flag)}")
    lines.extend(
        [
            "",
            "## Important Limitations",
            "- The frozen macro feature backbone still begins in 2006, so earlier ETF history is used only for technical lookbacks and forward target continuity.",
            "- `CR_EU_HY` has materially shorter effective history than the rest of the roster and should be treated as the weakest long-horizon target sleeve in this first build.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_v4_expanded_universe_dataset(project_root: Path | None = None) -> dict[str, pd.DataFrame]:
    paths = default_paths(project_root)
    paths.ensure_directories()
    inputs = _load_inputs(paths)
    asset_master_v4, target_manifest_v4 = _load_versioned_manifests(paths)
    macro_mapping_v4 = _build_macro_mapping_v4(asset_master_v4)

    month_end_prices_v4, monthly_returns_v4, raw_store = _build_target_store_v4(target_manifest_v4)

    actual_starts = month_end_prices_v4.groupby("sleeve_id", as_index=False)["month_end"].min().rename(columns={"month_end": "start_date_target"})
    asset_master_v4 = asset_master_v4.drop(columns=["start_date_target"]).merge(actual_starts, on="sleeve_id", how="left")
    target_manifest_v4 = target_manifest_v4.drop(columns=["start_date_target"]).merge(actual_starts, on="sleeve_id", how="left")

    base_feature_panel_v4 = _build_feature_base_panel_v4(
        base_feature_panel=inputs["feature_panel"],
        monthly_returns=monthly_returns_v4,
    )

    additional_state, additional_feature_meta, rejected_features = build_additional_monthly_state(
        default_v2_paths(paths.project_root),
        base_feature_panel_v4["month_end"],
    )
    one_month_target_panel = build_target_panel(
        month_end_prices_v4,
        inputs["tb3ms_monthly"],
        drop_terminal_without_forward=True,
    )
    sample_flags_v4 = base_feature_panel_v4[["sleeve_id", "month_end", "feature_complete_flag"]].merge(
        one_month_target_panel[["sleeve_id", "month_end", "target_quality_flag"]],
        on=["sleeve_id", "month_end"],
        how="left",
    )
    sample_flags_v4["target_quality_flag"] = sample_flags_v4["target_quality_flag"].fillna(False)
    sample_flags_v4["sample_inclusion_flag"] = (
        sample_flags_v4["feature_complete_flag"].eq(1) & sample_flags_v4["target_quality_flag"].eq(True)
    ).astype(int)
    feature_master, manual_rows, monthly_interactions = _build_feature_master_monthly_v4(
        inputs=inputs,
        additional_state=additional_state,
        asset_master_v4=asset_master_v4,
        base_feature_panel_v4=base_feature_panel_v4,
        sample_flags_v4=sample_flags_v4[["sleeve_id", "month_end", "sample_inclusion_flag"]],
    )

    target_panel = build_long_horizon_targets(
        month_end_prices=month_end_prices_v4,
        monthly_returns=monthly_returns_v4,
        tb3ms_monthly=inputs["tb3ms_monthly"],
        horizons=LOCKED_HORIZONS,
    )
    horizon_manifest = _build_horizon_manifest()
    modeling_panel_hstack, stacked_rows, stacked_interactions = _build_stacked_panel(feature_master, target_panel)

    feature_rows = _baseline_feature_catalog_v4(feature_master)
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
    feature_dictionary = (
        pd.DataFrame(feature_rows)
        .drop_duplicates(subset=["feature_name"], keep="last")
        .sort_values("feature_name")
        .reset_index(drop=True)
    )

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
        interaction_dictionary = pd.DataFrame(profiles).sort_values("interaction_name").reset_index(drop=True)

    external_inventory = build_external_file_inventory(default_v2_paths(paths.project_root), additional_feature_meta, rejected_features)
    data_inventory = _data_inventory(default_v2_paths(paths.project_root), external_inventory)
    target_source_rows = []
    for key in ("target_raw_direct", "target_raw_euro_local", "fx_raw_eurusd"):
        frame = raw_store[key]
        target_source_rows.append(
            {
                "file_id": key,
                "container": "versioned_target_source",
                "file_type": "api",
                "row_count": float(len(frame)),
                "column_count": float(frame.shape[1]),
                "date_column": "trade_date",
                "start_date": str(pd.to_datetime(frame["trade_date"]).min().date()),
                "end_date": str(pd.to_datetime(frame["trade_date"]).max().date()),
                "detected_frequency": "daily",
                "feature_count_used": np.nan,
                "rejected_feature_count": np.nan,
                "v2_status": "used_target_source",
                "v2_reason": f"Versioned v4 target input: {key}.",
            }
        )
    data_inventory = pd.concat([data_inventory, pd.DataFrame(target_source_rows)], ignore_index=True).sort_values("file_id").reset_index(drop=True)

    coverage_summary = _coverage_summary(feature_master, target_panel, modeling_panel_hstack)
    target_horizon_summary = _target_horizon_summary(target_panel)
    missingness_summary = _missingness_summary(feature_master, modeling_panel_hstack)
    design_report = _render_design_report(
        feature_master=feature_master,
        target_panel=target_panel,
        modeling_panel_hstack=modeling_panel_hstack,
        target_manifest=target_manifest_v4,
        raw_store=raw_store,
        used_files=external_inventory.loc[external_inventory["v2_status"] == "used_feature_source", "file_id"].tolist(),
    )

    write_csv(asset_master_v4, paths.data_final_v4_dir / "asset_master.csv")
    write_csv(target_manifest_v4, paths.data_final_v4_dir / "target_series_manifest.csv")
    write_csv(macro_mapping_v4, paths.data_final_v4_dir / "macro_mapping.csv")
    write_csv(raw_store["target_raw_direct"], paths.data_final_v4_dir / "target_raw_direct.csv")
    write_csv(raw_store["target_raw_euro_local"], paths.data_final_v4_dir / "target_raw_euro_local.csv")
    write_csv(raw_store["fx_raw_eurusd"], paths.data_final_v4_dir / "fx_raw_eurusd.csv")
    write_csv(raw_store["target_month_end_euro_usd_synth"], paths.data_final_v4_dir / "euro_fixed_income_month_end_usd_synth.csv")
    write_csv(raw_store["euro_fx_audit"], paths.data_final_v4_dir / "euro_fx_audit.csv")
    write_parquet(feature_master, paths.data_final_v4_dir / "feature_master_monthly.parquet")
    write_csv(feature_master, paths.data_final_v4_dir / "feature_master_monthly.csv")
    write_parquet(target_panel, paths.data_final_v4_dir / "target_panel_long_horizon.parquet")
    write_csv(target_panel, paths.data_final_v4_dir / "target_panel_long_horizon.csv")
    write_parquet(modeling_panel_hstack, paths.data_final_v4_dir / "modeling_panel_hstack.parquet")
    write_csv(modeling_panel_hstack, paths.data_final_v4_dir / "modeling_panel_hstack.csv")
    write_csv(feature_dictionary, paths.data_final_v4_dir / "feature_dictionary.csv")
    write_csv(interaction_dictionary, paths.data_final_v4_dir / "interaction_dictionary.csv")
    write_csv(horizon_manifest, paths.data_final_v4_dir / "horizon_manifest.csv")
    write_csv(data_inventory, paths.data_final_v4_dir / "data_inventory.csv")
    write_csv(coverage_summary, paths.reports_v4_dir / "coverage_summary.csv")
    write_csv(target_horizon_summary, paths.reports_v4_dir / "target_horizon_summary.csv")
    write_csv(missingness_summary, paths.reports_v4_dir / "missingness_summary.csv")
    write_csv(rejected_features, paths.reports_v4_dir / "rejected_features.csv")
    write_text(design_report, paths.reports_v4_dir / "final_data_design_report.md")

    return {
        "asset_master": asset_master_v4,
        "target_manifest": target_manifest_v4,
        "macro_mapping": macro_mapping_v4,
        "feature_master_monthly": feature_master,
        "target_panel_long_horizon": target_panel,
        "modeling_panel_hstack": modeling_panel_hstack,
        "feature_dictionary": feature_dictionary,
        "interaction_dictionary": interaction_dictionary,
        "coverage_summary": coverage_summary,
        "target_horizon_summary": target_horizon_summary,
        "missingness_summary": missingness_summary,
    }
