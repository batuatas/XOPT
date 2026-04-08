"""Build macro mapping, modeling panel, and sample-start diagnostics."""

from __future__ import annotations

import pandas as pd

from xoptpoe_data.config import GLOBAL_BLOCK, LOCAL_BLOCK_BY_SLEEVE, NO_LOCAL_BLOCK_SLEEVES


def build_macro_mapping(asset_master: pd.DataFrame) -> pd.DataFrame:
    """Construct locked sleeve-to-macro mapping table."""
    required = {"sleeve_id"}
    missing = required - set(asset_master.columns)
    if missing:
        raise ValueError(f"asset_master missing required columns for macro mapping: {sorted(missing)}")

    rows: list[dict] = []

    for sleeve_id in sorted(asset_master["sleeve_id"].unique()):
        local_block = LOCAL_BLOCK_BY_SLEEVE.get(sleeve_id)

        if local_block is not None:
            rows.append(
                {
                    "sleeve_id": sleeve_id,
                    "block_role": "local",
                    "geo_block": local_block,
                    "mapping_priority": 1,
                    "notes": "Locked local block mapping",
                }
            )

        rows.append(
            {
                "sleeve_id": sleeve_id,
                "block_role": "global",
                "geo_block": GLOBAL_BLOCK,
                "mapping_priority": 2,
                "notes": "All sleeves consume global/stress block",
            }
        )
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "block_role": "usd",
                "geo_block": GLOBAL_BLOCK,
                "mapping_priority": 3,
                "notes": "USD applicability via DTWEXBGS",
            }
        )

    out = pd.DataFrame(rows).sort_values(["sleeve_id", "mapping_priority"]).reset_index(drop=True)

    # Hard constraints.
    for sleeve in NO_LOCAL_BLOCK_SLEEVES:
        if ((out["sleeve_id"] == sleeve) & (out["block_role"] == "local")).any():
            raise ValueError(f"Forbidden local block mapping found for {sleeve}")

    if out.duplicated(subset=["sleeve_id", "block_role"]).any():
        raise ValueError("macro_mapping has duplicate (sleeve_id, block_role) keys")

    return out


def build_modeling_panel(
    *,
    feature_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    asset_master: pd.DataFrame,
    macro_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Join feature and target layers into final modeling panel."""
    feature = feature_panel.copy()
    feature["month_end"] = pd.to_datetime(feature["month_end"])

    target = target_panel.copy()
    target["month_end"] = pd.to_datetime(target["month_end"])

    model = feature.merge(
        target[[
            "sleeve_id",
            "month_end",
            "ret_fwd_1m",
            "rf_fwd_1m",
            "excess_ret_fwd_1m",
            "target_quality_flag",
        ]],
        on=["sleeve_id", "month_end"],
        how="inner",
    )

    model = model.merge(asset_master[["sleeve_id", "proxy_flag"]], on="sleeve_id", how="left")

    local_map = (
        macro_mapping[macro_mapping["block_role"] == "local"]
        [["sleeve_id", "geo_block"]]
        .rename(columns={"geo_block": "geo_block_local"})
    )
    global_map = (
        macro_mapping[macro_mapping["block_role"] == "global"]
        [["sleeve_id", "geo_block"]]
        .rename(columns={"geo_block": "geo_block_global"})
    )

    model = model.merge(local_map, on="sleeve_id", how="left")
    model = model.merge(global_map, on="sleeve_id", how="left")

    model["sample_inclusion_flag"] = (
        model["feature_complete_flag"].eq(1)
        & model["target_quality_flag"].eq(1)
        & model["proxy_flag"].eq(1)
    ).astype(int)

    model = model.sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)

    if model.duplicated(subset=["sleeve_id", "month_end"]).any():
        raise ValueError("modeling_panel has duplicate (sleeve_id, month_end) rows")

    return model


def compute_sample_start_report(
    *,
    sleeve_target_raw: pd.DataFrame,
    month_end_prices: pd.DataFrame,
    feature_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    modeling_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute required sample-start diagnostics for locked v1."""
    metrics = [
        {
            "metric": "first_raw_target_date",
            "date_value": pd.to_datetime(sleeve_target_raw["trade_date"]).min(),
        },
        {
            "metric": "first_month_end_price_date",
            "date_value": pd.to_datetime(month_end_prices["month_end"]).min(),
        },
        {
            "metric": "first_feature_ready_date",
            "date_value": pd.to_datetime(
                feature_panel.loc[feature_panel["feature_complete_flag"] == 1, "month_end"]
            ).min(),
        },
        {
            "metric": "first_target_ready_date",
            "date_value": pd.to_datetime(
                target_panel.loc[target_panel["target_quality_flag"] == 1, "month_end"]
            ).min(),
        },
        {
            "metric": "first_final_modeling_panel_date",
            "date_value": pd.to_datetime(
                modeling_panel.loc[modeling_panel["sample_inclusion_flag"] == 1, "month_end"]
            ).min(),
        },
    ]
    return pd.DataFrame(metrics)
