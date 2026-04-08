"""Build canonical geo-prefixed macro state panels for XOPTPOE v1."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from xoptpoe_data.config import LOCAL_BLOCKS
from xoptpoe_data.utils import log_change, months_between


LOCAL_SERIES_MAP: dict[str, dict[str, str]] = {
    "US": {
        "cpi": "US_CPI",
        "unemp": "US_UNEMP",
        "rate3m": "US_RF3M",
        "rate10y": "US_10Y",
    },
    "EURO_AREA": {
        "cpi": "EA_CPI",
        "unemp": "EA_UNEMP",
        "rate3m": "EA_3M",
        "rate10y": "EA_10Y",
    },
    "JAPAN": {
        "cpi": "JP_CPI",
        "unemp": "JP_UNEMP",
        "rate3m": "JP_3M",
        "rate10y": "JP_10Y",
    },
}

GEO_SUFFIX_BY_BLOCK: dict[str, str] = {
    "US": "US",
    "EURO_AREA": "EA",
    "JAPAN": "JP",
}

GLOBAL_SERIES_MAP: dict[str, str] = {
    "usd_broad": "USD_BROAD",
    "vix": "VIX",
    "us_real10y": "US_REAL10Y",
    "ig_oas": "IG_OAS",
    "oil_wti": "OIL_WTI",
}


@dataclass
class SeriesMonthly:
    """Standardized monthly representation for one series."""

    series_id: str
    native_frequency: str
    df: pd.DataFrame


def _series_frame(macro_raw: pd.DataFrame, series_id: str) -> pd.DataFrame:
    subset = macro_raw.loc[macro_raw["series_id"] == series_id, ["obs_date", "value", "native_frequency"]].copy()
    if subset.empty:
        raise ValueError(f"macro_raw is missing series_id={series_id}")
    subset["obs_date"] = pd.to_datetime(subset["obs_date"])
    subset = subset.sort_values("obs_date").reset_index(drop=True)
    return subset


def _to_monthly_last(subset: pd.DataFrame) -> SeriesMonthly:
    freq = str(subset["native_frequency"].iloc[0]).lower()
    work = subset.copy()

    if freq == "daily":
        work["month_end"] = work["obs_date"].dt.to_period("M").dt.to_timestamp("M")
        monthly = work.sort_values("obs_date").groupby("month_end", as_index=False).last()
        monthly = monthly.rename(columns={"obs_date": "source_obs_date"})
        monthly["source_obs_month_end"] = monthly["source_obs_date"].dt.to_period("M").dt.to_timestamp("M")
        return SeriesMonthly(
            series_id="",
            native_frequency=freq,
            df=monthly[["month_end", "value", "source_obs_month_end"]],
        )

    if freq == "monthly":
        work["source_obs_month_end"] = work["obs_date"].dt.to_period("M").dt.to_timestamp("M")
        monthly = work.sort_values("obs_date").groupby("source_obs_month_end", as_index=False).last()
        monthly = monthly.rename(columns={"source_obs_month_end": "month_end"})
        monthly["source_obs_month_end"] = monthly["month_end"]
        return SeriesMonthly(
            series_id="",
            native_frequency=freq,
            df=monthly[["month_end", "value", "source_obs_month_end"]],
        )

    raise ValueError(f"Unsupported native_frequency={freq}")


def _asof_on_month_end(
    monthly_df: pd.DataFrame,
    month_ends: pd.Series,
    *,
    lag_months: int,
) -> pd.DataFrame:
    """Align series to row month_end with optional lag and staleness tracking."""
    month_end_values = (
        pd.Series(pd.to_datetime(month_ends))
        .dropna()
        .sort_values()
        .drop_duplicates()
        .to_numpy()
    )
    timeline = pd.DataFrame({"month_end": month_end_values})
    monthly = monthly_df.copy().sort_values("month_end")
    # Ignore NaN-valued source rows so as-of alignment falls back to the last
    # valid released observation (with staleness tracked accordingly).
    monthly = monthly[monthly["value"].notna()].copy()

    if lag_months > 0:
        timeline["lookup_month_end"] = timeline["month_end"] - pd.offsets.MonthEnd(lag_months)
    else:
        timeline["lookup_month_end"] = timeline["month_end"]

    aligned = pd.merge_asof(
        timeline.sort_values("lookup_month_end"),
        monthly.rename(columns={"month_end": "series_month_end"}),
        left_on="lookup_month_end",
        right_on="series_month_end",
        direction="backward",
    )

    aligned["staleness_months"] = months_between(
        aligned["lookup_month_end"], aligned["series_month_end"]
    )
    aligned.loc[aligned["series_month_end"].isna(), "staleness_months"] = pd.NA

    return aligned[["month_end", "value", "series_month_end", "staleness_months"]]


def _local_component(
    macro_raw: pd.DataFrame,
    series_id: str,
    month_ends: pd.Series,
    *,
    lag_months: int,
) -> pd.DataFrame:
    raw = _series_frame(macro_raw, series_id)
    monthly = _to_monthly_last(raw).df
    return _asof_on_month_end(monthly, month_ends, lag_months=lag_months)


def _build_local_block_state(
    macro_raw: pd.DataFrame,
    month_ends: pd.Series,
    geo_block: str,
) -> pd.DataFrame:
    """Build geo-prefixed local state variables for one geo block."""
    ids = LOCAL_SERIES_MAP[geo_block]
    suffix = GEO_SUFFIX_BY_BLOCK[geo_block]

    cpi = _local_component(macro_raw, ids["cpi"], month_ends, lag_months=1)
    unemp = _local_component(macro_raw, ids["unemp"], month_ends, lag_months=1)
    short_rate = _local_component(macro_raw, ids["rate3m"], month_ends, lag_months=1)

    # US_10Y is daily (market observable at t); EA/JP 10Y are monthly and lagged.
    long_rate_lag = 0 if ids["rate10y"] == "US_10Y" else 1
    long_rate = _local_component(macro_raw, ids["rate10y"], month_ends, lag_months=long_rate_lag)

    out = pd.DataFrame({"month_end": pd.to_datetime(month_ends)})
    out[f"cpi_level_lagged_{suffix}"] = cpi["value"].values
    out[f"unemp_{suffix}"] = unemp["value"].values
    out[f"short_rate_{suffix}"] = short_rate["value"].values
    out[f"long_rate_{suffix}"] = long_rate["value"].values

    out[f"infl_{suffix}"] = 100.0 * (
        out[f"cpi_level_lagged_{suffix}"] / out[f"cpi_level_lagged_{suffix}"].shift(12) - 1.0
    )
    out[f"term_slope_{suffix}"] = out[f"long_rate_{suffix}"] - out[f"short_rate_{suffix}"]

    out[f"infl_{suffix}_delta_1m"] = out[f"infl_{suffix}"].diff(1)
    out[f"unemp_{suffix}_delta_1m"] = out[f"unemp_{suffix}"].diff(1)
    out[f"short_rate_{suffix}_delta_1m"] = out[f"short_rate_{suffix}"].diff(1)
    out[f"long_rate_{suffix}_delta_1m"] = out[f"long_rate_{suffix}"].diff(1)
    out[f"term_slope_{suffix}_delta_1m"] = out[f"term_slope_{suffix}"].diff(1)

    out[f"src_obs_month_end_cpi_{suffix}"] = cpi["series_month_end"].values
    out[f"src_obs_month_end_unemp_{suffix}"] = unemp["series_month_end"].values
    out[f"src_obs_month_end_short_rate_{suffix}"] = short_rate["series_month_end"].values
    out[f"src_obs_month_end_long_rate_{suffix}"] = long_rate["series_month_end"].values

    out[f"staleness_cpi_{suffix}"] = cpi["staleness_months"].values
    out[f"staleness_unemp_{suffix}"] = unemp["staleness_months"].values
    out[f"staleness_short_rate_{suffix}"] = short_rate["staleness_months"].values
    out[f"staleness_long_rate_{suffix}"] = long_rate["staleness_months"].values

    return out.drop(columns=[f"cpi_level_lagged_{suffix}"])


def _build_global_state(macro_raw: pd.DataFrame, month_ends: pd.Series) -> pd.DataFrame:
    """Build canonical global/stress block state variables."""
    out = pd.DataFrame({"month_end": pd.to_datetime(month_ends)})

    for var_name, series_id in GLOBAL_SERIES_MAP.items():
        aligned = _local_component(macro_raw, series_id, month_ends, lag_months=0)
        out[var_name] = aligned["value"].values
        out[f"src_obs_month_end_{var_name}"] = aligned["series_month_end"].values
        out[f"staleness_{var_name}"] = aligned["staleness_months"].values

    out = out.sort_values("month_end").reset_index(drop=True)

    out["usd_broad_logchg_1m"] = log_change(out["usd_broad"], periods=1)
    out["usd_broad_logchg_12m"] = log_change(out["usd_broad"], periods=12)
    out["vix_delta_1m"] = out["vix"].diff(1)
    out["us_real10y_delta_1m"] = out["us_real10y"].diff(1)
    out["ig_oas_delta_1m"] = out["ig_oas"].diff(1)
    out["oil_wti_logchg_1m"] = log_change(out["oil_wti"], periods=1)
    out["oil_wti_logchg_12m"] = log_change(out["oil_wti"], periods=12)

    global_stale_cols = [f"staleness_{name}" for name in GLOBAL_SERIES_MAP]
    out["global_stale_flag"] = out[global_stale_cols].fillna(0).gt(0).any(axis=1).astype(int)
    return out


def build_macro_state_panel(
    *,
    macro_raw: pd.DataFrame,
    month_ends: pd.Series,
    lag_policy_tag: str,
) -> pd.DataFrame:
    """Build canonical wide monthly macro_state_panel with geo-prefixed variables."""
    month_end_index = (
        pd.Series(pd.to_datetime(month_ends))
        .dropna()
        .sort_values()
        .drop_duplicates()
        .to_numpy()
    )

    out = pd.DataFrame({"month_end": month_end_index})

    for geo_block in LOCAL_BLOCKS:
        block = _build_local_block_state(macro_raw=macro_raw, month_ends=month_end_index, geo_block=geo_block)
        out = out.merge(block, on="month_end", how="left")

    global_block = _build_global_state(macro_raw=macro_raw, month_ends=month_end_index)
    out = out.merge(global_block, on="month_end", how="left")

    stale_cols = [col for col in out.columns if col.startswith("staleness_")]
    out["macro_stale_flag"] = out[stale_cols].fillna(0).gt(0).any(axis=1).astype(int)
    out["lag_policy_tag"] = lag_policy_tag

    local_state_cols: list[str] = []
    local_delta_cols: list[str] = []
    local_src_cols: list[str] = []
    local_stale_cols: list[str] = []
    for suffix in ("US", "EA", "JP"):
        local_state_cols.extend(
            [
                f"infl_{suffix}",
                f"unemp_{suffix}",
                f"short_rate_{suffix}",
                f"long_rate_{suffix}",
                f"term_slope_{suffix}",
            ]
        )
        local_delta_cols.extend(
            [
                f"infl_{suffix}_delta_1m",
                f"unemp_{suffix}_delta_1m",
                f"short_rate_{suffix}_delta_1m",
                f"long_rate_{suffix}_delta_1m",
                f"term_slope_{suffix}_delta_1m",
            ]
        )
        local_src_cols.extend(
            [
                f"src_obs_month_end_cpi_{suffix}",
                f"src_obs_month_end_unemp_{suffix}",
                f"src_obs_month_end_short_rate_{suffix}",
                f"src_obs_month_end_long_rate_{suffix}",
            ]
        )
        local_stale_cols.extend(
            [
                f"staleness_cpi_{suffix}",
                f"staleness_unemp_{suffix}",
                f"staleness_short_rate_{suffix}",
                f"staleness_long_rate_{suffix}",
            ]
        )

    global_state_cols = ["usd_broad", "vix", "us_real10y", "ig_oas", "oil_wti"]
    global_change_cols = [
        "usd_broad_logchg_1m",
        "usd_broad_logchg_12m",
        "vix_delta_1m",
        "us_real10y_delta_1m",
        "ig_oas_delta_1m",
        "oil_wti_logchg_1m",
        "oil_wti_logchg_12m",
    ]
    global_src_cols = [
        "src_obs_month_end_usd_broad",
        "src_obs_month_end_vix",
        "src_obs_month_end_us_real10y",
        "src_obs_month_end_ig_oas",
        "src_obs_month_end_oil_wti",
    ]
    global_stale_cols = [
        "staleness_usd_broad",
        "staleness_vix",
        "staleness_us_real10y",
        "staleness_ig_oas",
        "staleness_oil_wti",
    ]

    ordered_cols = [
        "month_end",
        *local_state_cols,
        *local_delta_cols,
        *global_state_cols,
        *global_change_cols,
        "global_stale_flag",
        "macro_stale_flag",
        "lag_policy_tag",
        *local_src_cols,
        *local_stale_cols,
        *global_src_cols,
        *global_stale_cols,
    ]
    return out[ordered_cols].sort_values("month_end").reset_index(drop=True)


def build_global_state_panel(macro_state_panel: pd.DataFrame) -> pd.DataFrame:
    """Build compatibility global_state_panel from canonical macro_state_panel."""
    required_cols = {
        "month_end",
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
        "staleness_usd_broad",
        "staleness_vix",
        "staleness_us_real10y",
        "staleness_ig_oas",
        "staleness_oil_wti",
        "global_stale_flag",
    }
    missing = required_cols - set(macro_state_panel.columns)
    if missing:
        raise ValueError(f"macro_state_panel missing columns required for global_state_panel: {sorted(missing)}")

    out = macro_state_panel[
        [
            "month_end",
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
            "staleness_usd_broad",
            "staleness_vix",
            "staleness_us_real10y",
            "staleness_ig_oas",
            "staleness_oil_wti",
            "global_stale_flag",
        ]
    ].copy()

    out = out.rename(
        columns={
            "usd_broad": "usd_broad_level",
            "vix": "vix_level",
            "us_real10y": "us_real10y_level",
            "ig_oas": "ig_oas_level",
            "oil_wti": "oil_wti_level",
            "staleness_usd_broad": "usd_broad_level_staleness",
            "staleness_vix": "vix_level_staleness",
            "staleness_us_real10y": "us_real10y_level_staleness",
            "staleness_ig_oas": "ig_oas_level_staleness",
            "staleness_oil_wti": "oil_wti_level_staleness",
        }
    )

    return out.sort_values("month_end").reset_index(drop=True)
