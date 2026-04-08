"""Build a versioned long-horizon dataset with an investable China sleeve."""

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
    _baseline_feature_catalog,
    _build_horizon_manifest,
    _build_stacked_panel,
    _coverage_summary,
    _data_inventory,
    _feature_meta_row,
    _manual_monthly_features,
    _missingness_summary,
    _profile_feature,
    _profile_feature_hstack,
    _target_horizon_summary,
)
from xoptpoe_v2_data.io import load_csv, write_csv, write_parquet, write_text
from xoptpoe_v2_data.sources import build_additional_monthly_state, build_external_file_inventory
from xoptpoe_v2_data.targets import build_long_horizon_targets
from xoptpoe_v3_data.config import (
    ASSET_CLASS_GROUP_BY_SLEEVE_V3,
    CHINA_SLEEVE_ID,
    CHINA_START_DATE,
    CHINA_TICKER,
    EXPOSURE_REGION_BY_SLEEVE_V3,
    LOCAL_BLOCK_BY_SLEEVE_V3,
    LOCKED_HORIZONS,
    LOCKED_SLEEVES_V3,
    NO_LOCAL_CORE_SLEEVES,
    V3_VERSION,
    V3Paths,
    default_paths,
)

TECH_COLS = [
    "ret_1m_lag",
    "ret_3m_lag",
    "ret_6m_lag",
    "ret_12m_lag",
    "mom_12_1",
    "vol_3m",
    "vol_12m",
    "maxdd_12m",
    "rel_mom_vs_treasury",
    "rel_mom_vs_us_equity",
    "rel_ret_1m_vs_treasury",
    "rel_ret_1m_vs_us_equity",
]
LOCAL_ALIAS_COLS = [
    "local_cpi_yoy",
    "local_unemp",
    "local_3m_rate",
    "local_10y_rate",
    "local_term_slope",
    "local_cpi_yoy_delta_1m",
    "local_unemp_delta_1m",
    "local_3m_rate_delta_1m",
    "local_10y_rate_delta_1m",
    "local_term_slope_delta_1m",
]
GLOBAL_REQUIRED_COLS = [
    "usd_broad_level",
    "usd_broad_logchg_1m",
    "vix_level",
    "us_real10y_level",
    "ig_oas_level",
    "oil_wti_level",
]


def _load_inputs(paths: V3Paths) -> dict[str, pd.DataFrame]:
    return {
        "feature_panel": load_csv(paths.data_intermediate_dir / "feature_panel.csv", parse_dates=["month_end"]),
        "modeling_panel": load_csv(paths.data_final_dir / "modeling_panel.csv", parse_dates=["month_end"]),
        "asset_master": load_csv(paths.data_final_dir / "asset_master.csv", parse_dates=["start_date_target"]),
        "month_end_prices": load_csv(
            paths.data_intermediate_dir / "sleeve_month_end_prices.csv",
            parse_dates=["month_end", "trade_date"],
        ),
        "monthly_returns": load_csv(paths.data_intermediate_dir / "sleeve_monthly_returns.csv", parse_dates=["month_end"]),
        "tb3ms_monthly": load_csv(paths.data_intermediate_dir / "tb3ms_monthly.csv", parse_dates=["month_end"]),
        "feature_dictionary_v2": load_csv(paths.data_final_v2_dir / "feature_dictionary.csv", parse_dates=["first_valid_date", "last_valid_date"]),
    }


def _load_versioned_manifests(paths: V3Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_master = load_csv(paths.config_dir / "asset_master_seed_v3_china.csv", parse_dates=["start_date_target"])
    target_manifest = load_csv(paths.config_dir / "target_series_manifest_v3_china.csv", parse_dates=["start_date_target"])
    return asset_master, target_manifest


def _build_macro_mapping_v3(asset_master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sleeve_id in sorted(asset_master["sleeve_id"].unique()):
        local_block = LOCAL_BLOCK_BY_SLEEVE_V3.get(sleeve_id)
        if local_block is not None:
            rows.append(
                {
                    "sleeve_id": sleeve_id,
                    "block_role": "local",
                    "geo_block": local_block,
                    "mapping_priority": 1,
                    "notes": "Versioned local block mapping",
                }
            )
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "block_role": "global",
                "geo_block": "GLOBAL",
                "mapping_priority": 2,
                "notes": "All sleeves consume global/stress block",
            }
        )
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "block_role": "usd",
                "geo_block": "GLOBAL",
                "mapping_priority": 3,
                "notes": "USD applicability via DTWEXBGS",
            }
        )
    return pd.DataFrame(rows).sort_values(["sleeve_id", "mapping_priority"]).reset_index(drop=True)


def _fetch_china_target_raw(paths: V3Paths) -> pd.DataFrame:
    asset_master = pd.DataFrame(
        [
            {
                "sleeve_id": CHINA_SLEEVE_ID,
                "ticker": CHINA_TICKER,
            }
        ]
    )
    adapter = YahooFinanceTargetAdapter()
    return fetch_sleeve_target_raw(
        asset_master=asset_master,
        adapter=adapter,
        start_date="2004-01-01",
        end_date=None,
    )


def _build_china_feature_base(
    *,
    base_feature_panel: pd.DataFrame,
    base_monthly_returns: pd.DataFrame,
    china_monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    combined_returns = pd.concat([base_monthly_returns, china_monthly_returns], ignore_index=True)
    tech = _add_relative_features(_build_technical_features(combined_returns))
    china_tech = tech.loc[tech["sleeve_id"] == CHINA_SLEEVE_ID].copy()

    # Frozen baseline macro/global state exists from 2006 onward. Preserve that lower bound.
    china_tech = china_tech.loc[china_tech["month_end"] >= pd.Timestamp("2006-01-31")].copy()
    china_tech = china_tech.loc[china_tech["month_end"] <= base_feature_panel["month_end"].max()].copy()

    ref = base_feature_panel.loc[base_feature_panel["sleeve_id"] == "EQ_EM"].copy()
    shared_cols = [
        col
        for col in base_feature_panel.columns
        if col not in {"month_end", "sleeve_id", *TECH_COLS, *LOCAL_ALIAS_COLS, "feature_complete_flag", "local_geo_block_used"}
    ]
    month_shared = ref[["month_end", *shared_cols]].drop_duplicates(subset=["month_end"])

    out = china_tech.merge(month_shared, on="month_end", how="left")
    for col in LOCAL_ALIAS_COLS:
        out[col] = np.nan
    out["local_geo_block_used"] = "CHINA"

    tech_ok = out[TECH_COLS].notna().all(axis=1)
    global_ok = out[GLOBAL_REQUIRED_COLS].notna().all(axis=1)
    out["feature_complete_flag"] = (tech_ok & global_ok).astype(int)

    ordered_cols = list(base_feature_panel.columns)
    out = out[ordered_cols].sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)
    return out


def _build_china_sample_flag(
    *,
    china_feature_base: pd.DataFrame,
    china_month_end_prices: pd.DataFrame,
    tb3ms_monthly: pd.DataFrame,
) -> pd.DataFrame:
    target_panel = build_target_panel(china_month_end_prices, tb3ms_monthly, drop_terminal_without_forward=True)
    sample = china_feature_base[["sleeve_id", "month_end", "feature_complete_flag"]].merge(
        target_panel[["sleeve_id", "month_end", "target_quality_flag"]],
        on=["sleeve_id", "month_end"],
        how="left",
    )
    sample["target_quality_flag"] = sample["target_quality_flag"].fillna(False)
    sample["sample_inclusion_flag"] = (
        sample["feature_complete_flag"].eq(1) & sample["target_quality_flag"].eq(True)
    ).astype(int)
    return sample[["sleeve_id", "month_end", "sample_inclusion_flag"]]


def _replace_meta_row(rows: list[dict[str, object]], row: dict[str, object]) -> list[dict[str, object]]:
    rows = [r for r in rows if r.get("feature_name") != row.get("feature_name")]
    rows.append(row)
    return rows


def _extend_manual_monthly_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    panel, rows, interactions = _manual_monthly_features(panel)
    china_mask = panel["sleeve_id"].eq(CHINA_SLEEVE_ID)

    panel.loc[china_mask, "oecd_business_conf_local"] = panel.loc[china_mask, "oecd_bcicp_cn"]
    panel.loc[china_mask, "oecd_consumer_conf_local"] = panel.loc[china_mask, "oecd_ccicp_cn"]
    panel.loc[china_mask, "oecd_activity_proxy_local"] = panel.loc[china_mask, "oecd_li_cn"]
    panel.loc[china_mask, "cape_local"] = panel.loc[china_mask, "cape_china"]
    panel["no_local_block_flag"] = panel["sleeve_id"].isin(NO_LOCAL_CORE_SLEEVES).astype(int)
    panel["sleeve_dummy_eq_cn"] = panel["sleeve_id"].eq(CHINA_SLEEVE_ID).astype(int)
    panel["int_eq_cn_x_mom_12_1"] = panel["mom_12_1"] * panel["sleeve_dummy_eq_cn"]
    panel["int_eq_cn_x_vol_12m"] = panel["vol_12m"] * panel["sleeve_dummy_eq_cn"]

    for col, geography, notes in [
        ("oecd_business_conf_local", "LOCAL_MAPPED", "Mapped local business-confidence series, extended to China for EQ_CN."),
        ("oecd_consumer_conf_local", "LOCAL_MAPPED", "Mapped local consumer-confidence series, extended to China for EQ_CN."),
        ("oecd_activity_proxy_local", "LOCAL_MAPPED", "Mapped local activity proxy, extended to China OECD CLI for EQ_CN."),
        ("cape_local", "LOCAL_MAPPED", "Mapped local CAPE, extended to China CAPE for EQ_CN."),
        ("no_local_block_flag", "SLEEVE_SPECIFIC", "Static flag for sleeves without inherited core local macro alias coverage."),
        ("sleeve_dummy_eq_cn", "EQ_CN", "Static sleeve dummy for the versioned China sleeve."),
    ]:
        rows = _replace_meta_row(
            rows,
            _feature_meta_row(
                feature_name=col,
                source_file="derived_v3",
                source_column=col,
                block_name="metadata_dummy" if col.startswith("sleeve_dummy") or col.endswith("flag") else "local_mapping",
                geography=geography,
                transform_type="dummy_flag" if col.startswith("sleeve_dummy") or col.endswith("flag") else "mapped_alias",
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

    for name, lhs, rhs, notes in [
        ("int_eq_cn_x_mom_12_1", "mom_12_1", "sleeve_dummy_eq_cn", "Sleeve-specific momentum slope for EQ_CN."),
        ("int_eq_cn_x_vol_12m", "vol_12m", "sleeve_dummy_eq_cn", "Sleeve-specific 12m volatility slope for EQ_CN."),
    ]:
        interactions = [i for i in interactions if i.get("interaction_name") != name]
        interactions.append(
            {
                "interaction_name": name,
                "table_scope": "feature_master_monthly",
                "interaction_family": "sleeve_dummy_x_predictor",
                "lhs_feature": lhs,
                "rhs_feature": rhs,
                "geography": CHINA_SLEEVE_ID,
                "notes": notes,
            }
        )
        rows = _replace_meta_row(
            rows,
            _feature_meta_row(
                feature_name=name,
                source_file="derived_interaction",
                source_column=f"{lhs} x {rhs}",
                block_name="interaction",
                geography=CHINA_SLEEVE_ID,
                transform_type="sleeve_dummy_x_predictor",
                native_frequency="monthly",
                lag_months=0,
                notes=notes,
                monthly_flag=1,
                hstack_flag=1,
                is_interaction=1,
                monthly_df=panel,
                hstack_df=None,
            ),
        )

    return panel, rows, interactions


def _build_feature_master_monthly_v3(
    inputs: dict[str, pd.DataFrame],
    additional_state: pd.DataFrame,
    asset_master_v3: pd.DataFrame,
    base_feature_panel_v3: pd.DataFrame,
    sample_flags_v3: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    feature_master = base_feature_panel_v3.copy()

    sample_flag = sample_flags_v3.copy().rename(columns={"sample_inclusion_flag": "baseline_sample_inclusion_1m_flag"})
    asset_cols = asset_master_v3[["sleeve_id", "ticker", "sleeve_name", "proxy_flag"]].copy()
    feature_master = feature_master.merge(asset_cols, on="sleeve_id", how="left")
    feature_master = feature_master.merge(sample_flag, on=["sleeve_id", "month_end"], how="left")
    feature_master["baseline_sample_inclusion_1m_flag"] = feature_master["baseline_sample_inclusion_1m_flag"].fillna(0).astype(int)

    local_map = pd.Series(LOCAL_BLOCK_BY_SLEEVE_V3, name="geo_block_local")
    feature_master["geo_block_local"] = feature_master["sleeve_id"].map(local_map)
    feature_master["geo_block_global"] = "GLOBAL"
    feature_master["asset_class_group"] = feature_master["sleeve_id"].map(ASSET_CLASS_GROUP_BY_SLEEVE_V3)
    feature_master["exposure_region"] = feature_master["sleeve_id"].map(EXPOSURE_REGION_BY_SLEEVE_V3)

    feature_master = feature_master.merge(additional_state, on="month_end", how="left")
    feature_master, manual_rows, interaction_rows = _extend_manual_monthly_features(feature_master)

    additional_feature_cols = [
        col for col in additional_state.columns if col != "month_end"
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


def _render_design_report(
    *,
    feature_master: pd.DataFrame,
    target_panel: pd.DataFrame,
    modeling_panel_hstack: pd.DataFrame,
    used_files: list[str],
    china_month_end_prices: pd.DataFrame,
) -> str:
    horizon_counts = target_panel.groupby("horizon_months")["target_available_flag"].sum().astype(int).to_dict()
    trainable_counts = modeling_panel_hstack.groupby("horizon_months")["baseline_trainable_flag"].sum().astype(int).to_dict()
    china_horizon_counts = (
        target_panel.loc[target_panel["sleeve_id"].eq(CHINA_SLEEVE_ID)]
        .groupby("horizon_months")["target_available_flag"].sum().astype(int).to_dict()
    )
    first_china_month = pd.to_datetime(china_month_end_prices["month_end"]).min().date()
    lines = [
        f"# XOPTPOE {V3_VERSION} Final Data Design Report",
        "",
        "## Design Summary",
        "- Frozen `v1` and `v2_long_horizon` outputs remain untouched.",
        f"- `{V3_VERSION}` is a versioned extension that adds one investable China target sleeve.",
        f"- China sleeve choice is `{CHINA_SLEEVE_ID} = {CHINA_TICKER}`.",
        "- The sleeve is added as a USD-listed ETF adjusted-close target, not from the uploaded SSE workbook.",
        "",
        "## China Sleeve Decision",
        f"- `FXI` was chosen because it is a USD-listed investable ETF with history starting {first_china_month}.",
        "- This is a China large-cap proxy, not a claim of perfect full-China market representation.",
        "- `MCHI` was rejected as the default sleeve proxy for this branch because its shorter ETF history would collapse 10Y/15Y usable sample further.",
        "",
        "## Main Table Grains",
        "- `feature_master_monthly`: one row per (`sleeve_id`, `month_end`).",
        "- `target_panel_long_horizon`: one row per (`sleeve_id`, `month_end`, `horizon_months`).",
        "- `modeling_panel_hstack`: one row per (`sleeve_id`, `month_end`, `horizon_months`).",
        "",
        "## China Feature Treatment",
        "- `EQ_CN` gets the same technical and global baseline features as the other sleeves.",
        "- It also consumes the existing China macro, China valuation, China market, OECD China, and China CAPE blocks already present in the v2 monthly state.",
        "- The inherited v1 canonical local macro core remains US/EA/JP-only, so `EQ_CN` is treated as outside that frozen local-alias core for baseline completeness flags.",
        "",
        "## Used Sources",
    ]
    for file_id in used_files:
        lines.append(f"- `{file_id}`")
    lines.extend(
        [
            f"- `Yahoo Finance adjusted-close history for {CHINA_TICKER}`",
            "",
            "## Usable Row Counts",
        ]
    )
    for horizon in LOCKED_HORIZONS:
        lines.append(
            f"- {horizon} months: total_target_rows={horizon_counts.get(horizon, 0)}, total_baseline_trainable_rows={trainable_counts.get(horizon, 0)}, china_target_rows={china_horizon_counts.get(horizon, 0)}"
        )
    lines.extend(
        [
            "",
            "## Important Limitation",
            "- Adding FXI does not back-extend the frozen baseline macro core earlier than 2006.",
            "- The pre-2006 FXI history is still useful for China technical lookbacks and forward target continuity, but the monthly feature store remains anchored to the baseline macro window.",
        ]
    )
    return "\n".join(lines)


def build_v3_long_horizon_china_dataset(project_root: Path | None = None) -> dict[str, pd.DataFrame]:
    paths = default_paths(project_root)
    paths.ensure_directories()
    inputs = _load_inputs(paths)
    asset_master_v3, target_manifest_v3 = _load_versioned_manifests(paths)
    macro_mapping_v3 = _build_macro_mapping_v3(asset_master_v3)

    china_target_raw = _fetch_china_target_raw(paths)
    china_month_end_prices = collapse_target_to_month_end_prices(china_target_raw)
    china_monthly_returns = build_monthly_realized_returns(china_month_end_prices)

    china_feature_base = _build_china_feature_base(
        base_feature_panel=inputs["feature_panel"],
        base_monthly_returns=inputs["monthly_returns"],
        china_monthly_returns=china_monthly_returns,
    )
    china_sample_flag = _build_china_sample_flag(
        china_feature_base=china_feature_base,
        china_month_end_prices=china_month_end_prices,
        tb3ms_monthly=inputs["tb3ms_monthly"],
    )

    base_feature_panel_v3 = pd.concat([inputs["feature_panel"], china_feature_base], ignore_index=True)
    sample_flags_v3 = pd.concat(
        [
            inputs["modeling_panel"][["sleeve_id", "month_end", "sample_inclusion_flag"]],
            china_sample_flag,
        ],
        ignore_index=True,
    )
    month_end_prices_v3 = pd.concat([inputs["month_end_prices"], china_month_end_prices], ignore_index=True)
    monthly_returns_v3 = pd.concat([inputs["monthly_returns"], china_monthly_returns], ignore_index=True)

    additional_state, additional_feature_meta, rejected_features = build_additional_monthly_state(
        paths,
        base_feature_panel_v3["month_end"],
    )
    feature_master, manual_rows, monthly_interactions = _build_feature_master_monthly_v3(
        inputs,
        additional_state,
        asset_master_v3,
        base_feature_panel_v3,
        sample_flags_v3,
    )
    target_panel = build_long_horizon_targets(
        month_end_prices=month_end_prices_v3,
        monthly_returns=monthly_returns_v3,
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

    external_inventory = build_external_file_inventory(paths, additional_feature_meta, rejected_features)
    data_inventory = _data_inventory(paths, external_inventory)
    data_inventory = pd.concat(
        [
            data_inventory,
            pd.DataFrame(
                [
                    {
                        "file_id": f"YahooFinance:{CHINA_TICKER}",
                        "container": "versioned_target_source",
                        "file_type": "api",
                        "row_count": float(len(china_target_raw)),
                        "column_count": float(china_target_raw.shape[1]),
                        "date_column": "trade_date",
                        "start_date": str(pd.to_datetime(china_target_raw["trade_date"]).min().date()),
                        "end_date": str(pd.to_datetime(china_target_raw["trade_date"]).max().date()),
                        "detected_frequency": "daily",
                        "feature_count_used": np.nan,
                        "rejected_feature_count": np.nan,
                        "v2_status": "used_target_source",
                        "v2_reason": "Versioned China sleeve target source.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("file_id").reset_index(drop=True)

    coverage_summary = _coverage_summary(feature_master, target_panel, modeling_panel_hstack)
    target_horizon_summary = _target_horizon_summary(target_panel)
    missingness_summary = _missingness_summary(feature_master, modeling_panel_hstack)
    design_report = _render_design_report(
        feature_master=feature_master,
        target_panel=target_panel,
        modeling_panel_hstack=modeling_panel_hstack,
        used_files=external_inventory.loc[external_inventory["v2_status"] == "used_feature_source", "file_id"].tolist(),
        china_month_end_prices=china_month_end_prices,
    )

    write_csv(asset_master_v3, paths.data_final_v3_dir / "asset_master.csv")
    write_csv(target_manifest_v3, paths.data_final_v3_dir / "target_series_manifest.csv")
    write_csv(macro_mapping_v3, paths.data_final_v3_dir / "macro_mapping.csv")
    write_csv(china_target_raw, paths.data_final_v3_dir / "china_sleeve_target_raw.csv")
    write_csv(china_month_end_prices, paths.data_final_v3_dir / "china_sleeve_month_end_prices.csv")
    write_csv(china_monthly_returns, paths.data_final_v3_dir / "china_sleeve_monthly_returns.csv")
    write_parquet(feature_master, paths.data_final_v3_dir / "feature_master_monthly.parquet")
    write_csv(feature_master, paths.data_final_v3_dir / "feature_master_monthly.csv")
    write_parquet(target_panel, paths.data_final_v3_dir / "target_panel_long_horizon.parquet")
    write_csv(target_panel, paths.data_final_v3_dir / "target_panel_long_horizon.csv")
    write_parquet(modeling_panel_hstack, paths.data_final_v3_dir / "modeling_panel_hstack.parquet")
    write_csv(modeling_panel_hstack, paths.data_final_v3_dir / "modeling_panel_hstack.csv")
    write_csv(feature_dictionary, paths.data_final_v3_dir / "feature_dictionary.csv")
    write_csv(interaction_dictionary, paths.data_final_v3_dir / "interaction_dictionary.csv")
    write_csv(horizon_manifest, paths.data_final_v3_dir / "horizon_manifest.csv")
    write_csv(data_inventory, paths.data_final_v3_dir / "data_inventory.csv")
    write_csv(coverage_summary, paths.reports_v3_dir / "coverage_summary.csv")
    write_csv(target_horizon_summary, paths.reports_v3_dir / "target_horizon_summary.csv")
    write_csv(missingness_summary, paths.reports_v3_dir / "missingness_summary.csv")
    write_csv(rejected_features, paths.reports_v3_dir / "rejected_features.csv")
    write_text(design_report, paths.reports_v3_dir / "final_data_design_report.md")

    return {
        "asset_master": asset_master_v3,
        "target_manifest": target_manifest_v3,
        "macro_mapping": macro_mapping_v3,
        "feature_master_monthly": feature_master,
        "target_panel_long_horizon": target_panel,
        "modeling_panel_hstack": modeling_panel_hstack,
        "feature_dictionary": feature_dictionary,
        "interaction_dictionary": interaction_dictionary,
        "coverage_summary": coverage_summary,
        "target_horizon_summary": target_horizon_summary,
        "missingness_summary": missingness_summary,
    }
