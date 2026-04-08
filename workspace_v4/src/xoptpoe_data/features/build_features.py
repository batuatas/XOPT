"""Build sleeve-month feature panel from canonical macro state and return history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xoptpoe_data.config import LOCAL_BLOCK_BY_SLEEVE, NO_LOCAL_BLOCK_SLEEVES
from xoptpoe_data.utils import cumulative_return, rolling_max_drawdown


GEO_SUFFIX_BY_BLOCK: dict[str, str] = {
    "US": "US",
    "EURO_AREA": "EA",
    "JAPAN": "JP",
}

CANONICAL_LOCAL_STATE_COLS = [
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
]

CANONICAL_LOCAL_DELTA_COLS = [
    "infl_US_delta_1m",
    "unemp_US_delta_1m",
    "short_rate_US_delta_1m",
    "long_rate_US_delta_1m",
    "term_slope_US_delta_1m",
    "infl_EA_delta_1m",
    "unemp_EA_delta_1m",
    "short_rate_EA_delta_1m",
    "long_rate_EA_delta_1m",
    "term_slope_EA_delta_1m",
    "infl_JP_delta_1m",
    "unemp_JP_delta_1m",
    "short_rate_JP_delta_1m",
    "long_rate_JP_delta_1m",
    "term_slope_JP_delta_1m",
]

CANONICAL_LOCAL_STALENESS_COLS = [
    "staleness_cpi_US",
    "staleness_unemp_US",
    "staleness_short_rate_US",
    "staleness_long_rate_US",
    "staleness_cpi_EA",
    "staleness_unemp_EA",
    "staleness_short_rate_EA",
    "staleness_long_rate_EA",
    "staleness_cpi_JP",
    "staleness_unemp_JP",
    "staleness_short_rate_JP",
    "staleness_long_rate_JP",
]

CANONICAL_GLOBAL_COLS = [
    "usd_broad",
    "vix",
    "us_real10y",
    "ig_oas",
    "oil_wti",
    "usd_broad_logchg_1m",
    "usd_broad_logchg_12m",
    "vix_delta_1m",
    "us_real10y_delta_1m",
    "ig_oas_delta_1m",
    "oil_wti_logchg_1m",
    "oil_wti_logchg_12m",
    "global_stale_flag",
    "macro_stale_flag",
    "lag_policy_tag",
]


def _build_technical_features(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    need = {"sleeve_id", "month_end", "ret_1m_realized"}
    missing = need - set(monthly_returns.columns)
    if missing:
        raise ValueError(f"monthly_returns missing required columns: {sorted(missing)}")

    work = monthly_returns[["sleeve_id", "month_end", "ret_1m_realized"]].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    work = work.sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)

    grouped = work.groupby("sleeve_id", group_keys=False)

    work["ret_1m_lag"] = work["ret_1m_realized"]
    work["ret_3m_lag"] = grouped["ret_1m_realized"].transform(lambda s: cumulative_return(s, 3))
    work["ret_6m_lag"] = grouped["ret_1m_realized"].transform(lambda s: cumulative_return(s, 6))
    work["ret_12m_lag"] = grouped["ret_1m_realized"].transform(lambda s: cumulative_return(s, 12))

    work["mom_12_1"] = grouped["ret_1m_realized"].transform(
        lambda s: cumulative_return(s.shift(1), 12)
    )

    work["vol_3m"] = grouped["ret_1m_realized"].transform(
        lambda s: s.rolling(3, min_periods=3).std(ddof=1)
    )
    work["vol_12m"] = grouped["ret_1m_realized"].transform(
        lambda s: s.rolling(12, min_periods=12).std(ddof=1)
    )
    work["maxdd_12m"] = grouped["ret_1m_realized"].transform(
        lambda s: rolling_max_drawdown(s, window=12)
    )

    return work


def _add_relative_features(tech_df: pd.DataFrame) -> pd.DataFrame:
    out = tech_df.copy()
    base = out[["sleeve_id", "month_end", "mom_12_1", "ret_1m_lag"]].copy()

    ief = base[base["sleeve_id"] == "FI_UST"][["month_end", "mom_12_1", "ret_1m_lag"]].rename(
        columns={"mom_12_1": "mom_12_1_ief", "ret_1m_lag": "ret_1m_lag_ief"}
    )
    vti = base[base["sleeve_id"] == "EQ_US"][["month_end", "mom_12_1", "ret_1m_lag"]].rename(
        columns={"mom_12_1": "mom_12_1_vti", "ret_1m_lag": "ret_1m_lag_vti"}
    )

    out = out.merge(ief, on="month_end", how="left")
    out = out.merge(vti, on="month_end", how="left")

    out["rel_mom_vs_treasury"] = out["mom_12_1"] - out["mom_12_1_ief"]
    out["rel_mom_vs_us_equity"] = out["mom_12_1"] - out["mom_12_1_vti"]
    out["rel_ret_1m_vs_treasury"] = out["ret_1m_lag"] - out["ret_1m_lag_ief"]
    out["rel_ret_1m_vs_us_equity"] = out["ret_1m_lag"] - out["ret_1m_lag_vti"]

    drop_cols = ["mom_12_1_ief", "ret_1m_lag_ief", "mom_12_1_vti", "ret_1m_lag_vti"]
    return out.drop(columns=drop_cols)


def _attach_macro_state(
    feature_df: pd.DataFrame,
    macro_state_panel: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "month_end",
        *CANONICAL_LOCAL_STATE_COLS,
        *CANONICAL_LOCAL_DELTA_COLS,
        *CANONICAL_LOCAL_STALENESS_COLS,
        *CANONICAL_GLOBAL_COLS,
    }
    missing = required - set(macro_state_panel.columns)
    if missing:
        raise ValueError(f"macro_state_panel missing required canonical columns: {sorted(missing)}")

    local_map = pd.Series(LOCAL_BLOCK_BY_SLEEVE)

    out = feature_df.copy()
    out["local_geo_block_used"] = out["sleeve_id"].map(local_map)
    out["local_geo_suffix_used"] = out["local_geo_block_used"].map(GEO_SUFFIX_BY_BLOCK)

    macro_cols = [
        "month_end",
        *CANONICAL_LOCAL_STATE_COLS,
        *CANONICAL_LOCAL_DELTA_COLS,
        *CANONICAL_LOCAL_STALENESS_COLS,
        *CANONICAL_GLOBAL_COLS,
    ]
    out = out.merge(macro_state_panel[macro_cols], on="month_end", how="left")

    # Backward-compatible local alias layer derived from canonical geo-prefixed state.
    local_alias_cols = [
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
    for col in local_alias_cols:
        out[col] = np.nan

    out["local_macro_stale_flag"] = 0

    for suffix in ("US", "EA", "JP"):
        mask = out["local_geo_suffix_used"] == suffix
        out.loc[mask, "local_cpi_yoy"] = out.loc[mask, f"infl_{suffix}"]
        out.loc[mask, "local_unemp"] = out.loc[mask, f"unemp_{suffix}"]
        out.loc[mask, "local_3m_rate"] = out.loc[mask, f"short_rate_{suffix}"]
        out.loc[mask, "local_10y_rate"] = out.loc[mask, f"long_rate_{suffix}"]
        out.loc[mask, "local_term_slope"] = out.loc[mask, f"term_slope_{suffix}"]

        out.loc[mask, "local_cpi_yoy_delta_1m"] = out.loc[mask, f"infl_{suffix}_delta_1m"]
        out.loc[mask, "local_unemp_delta_1m"] = out.loc[mask, f"unemp_{suffix}_delta_1m"]
        out.loc[mask, "local_3m_rate_delta_1m"] = out.loc[mask, f"short_rate_{suffix}_delta_1m"]
        out.loc[mask, "local_10y_rate_delta_1m"] = out.loc[mask, f"long_rate_{suffix}_delta_1m"]
        out.loc[mask, "local_term_slope_delta_1m"] = out.loc[mask, f"term_slope_{suffix}_delta_1m"]

        stale_cols = [
            f"staleness_cpi_{suffix}",
            f"staleness_unemp_{suffix}",
            f"staleness_short_rate_{suffix}",
            f"staleness_long_rate_{suffix}",
        ]
        # staleness_* columns are carried through macro_state_panel and available after merge
        available_stale = [c for c in stale_cols if c in out.columns]
        if available_stale:
            out.loc[mask, "local_macro_stale_flag"] = (
                out.loc[mask, available_stale].fillna(0).gt(0).any(axis=1).astype(int)
            )

    # Backward-compatible global alias layer.
    out["usd_broad_level"] = out["usd_broad"]
    out["vix_level"] = out["vix"]
    out["us_real10y_level"] = out["us_real10y"]
    out["ig_oas_level"] = out["ig_oas"]
    out["oil_wti_level"] = out["oil_wti"]

    out["macro_stale_flag"] = (
        out["local_macro_stale_flag"].fillna(0).astype(int)
        | out["global_stale_flag"].fillna(0).astype(int)
    )

    return out


def _feature_completeness(out: pd.DataFrame) -> pd.Series:
    tech_required = [
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
    ]

    global_required = [
        "usd_broad_level",
        "usd_broad_logchg_1m",
        "vix_level",
        "us_real10y_level",
        "ig_oas_level",
        "oil_wti_level",
    ]

    local_required = [
        "local_cpi_yoy",
        "local_unemp",
        "local_3m_rate",
        "local_10y_rate",
        "local_term_slope",
    ]

    is_no_local = out["sleeve_id"].isin(NO_LOCAL_BLOCK_SLEEVES)

    tech_ok = out[tech_required].notna().all(axis=1)
    global_ok = out[global_required].notna().all(axis=1)
    local_ok = out[local_required].notna().all(axis=1)

    # For sleeves without local mapping in locked v1, local features are intentionally absent.
    return np.where(is_no_local, tech_ok & global_ok, tech_ok & global_ok & local_ok)


def build_feature_panel(
    *,
    monthly_returns: pd.DataFrame,
    macro_state_panel: pd.DataFrame,
    global_state_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build final sleeve-month feature panel from canonical macro state."""
    # global_state_panel retained as optional compatibility argument; canonical source is macro_state_panel.
    _ = global_state_panel

    tech = _build_technical_features(monthly_returns)
    tech = _add_relative_features(tech)

    out = _attach_macro_state(tech, macro_state_panel)

    out["lag_policy_tag"] = out["lag_policy_tag"].fillna("LOCKED_V1_OFFICIAL_MONTHLY_LAG1")
    out["feature_complete_flag"] = _feature_completeness(out).astype(int)

    keep_cols = [
        "sleeve_id",
        "month_end",
        "ret_1m_lag",
        "ret_3m_lag",
        "ret_6m_lag",
        "ret_12m_lag",
        "mom_12_1",
        "vol_3m",
        "vol_12m",
        "maxdd_12m",
        *CANONICAL_LOCAL_STATE_COLS,
        *CANONICAL_LOCAL_DELTA_COLS,
        "usd_broad",
        "vix",
        "us_real10y",
        "ig_oas",
        "oil_wti",
        "usd_broad_logchg_1m",
        "usd_broad_logchg_12m",
        "vix_delta_1m",
        "us_real10y_delta_1m",
        "ig_oas_delta_1m",
        "oil_wti_logchg_1m",
        "oil_wti_logchg_12m",
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
        "usd_broad_level",
        "vix_level",
        "us_real10y_level",
        "ig_oas_level",
        "oil_wti_level",
        "rel_mom_vs_treasury",
        "rel_mom_vs_us_equity",
        "rel_ret_1m_vs_treasury",
        "rel_ret_1m_vs_us_equity",
        "feature_complete_flag",
        "macro_stale_flag",
        "lag_policy_tag",
        "local_geo_block_used",
    ]
    return out[keep_cols].sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)
