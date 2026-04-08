"""Build month-end target price tables and forward excess-return targets."""

from __future__ import annotations

import pandas as pd

from xoptpoe_data.utils import collapse_daily_to_month_end


def collapse_target_to_month_end_prices(sleeve_target_raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily sleeve prices to last available trading day per month."""
    required = {"sleeve_id", "ticker", "trade_date", "adj_close", "close"}
    missing = required - set(sleeve_target_raw.columns)
    if missing:
        raise ValueError(f"sleeve_target_raw missing required columns: {sorted(missing)}")

    collapsed = collapse_daily_to_month_end(
        sleeve_target_raw,
        group_cols=["sleeve_id", "ticker"],
        date_col="trade_date",
        value_cols=["adj_close", "close"],
    )

    if collapsed["adj_close"].isna().any():
        raise ValueError("Missing adj_close on selected month-end target rows")

    if (collapsed["adj_close"] <= 0).any():
        raise ValueError("Non-positive adj_close detected on selected month-end target rows")

    return collapsed[["sleeve_id", "ticker", "month_end", "trade_date", "adj_close", "close"]]


def build_monthly_realized_returns(month_end_prices: pd.DataFrame) -> pd.DataFrame:
    """Build realized monthly returns anchored at month-end t."""
    work = month_end_prices[["sleeve_id", "ticker", "month_end", "adj_close"]].copy()
    work = work.sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)

    work["ret_1m_realized"] = work.groupby("sleeve_id")["adj_close"].pct_change()
    return work


def extract_tb3ms_monthly(macro_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract raw TB3MS monthly series (observation-month indexed)."""
    need = {"series_id", "obs_date", "value"}
    missing = need - set(macro_raw.columns)
    if missing:
        raise ValueError(f"macro_raw missing required columns for TB3MS extraction: {sorted(missing)}")

    tb3 = macro_raw.loc[macro_raw["series_id"] == "US_RF3M", ["obs_date", "value"]].copy()
    if tb3.empty:
        raise ValueError("US_RF3M (TB3MS) not found in macro_raw")

    tb3["obs_date"] = pd.to_datetime(tb3["obs_date"])
    tb3["month_end"] = tb3["obs_date"].dt.to_period("M").dt.to_timestamp("M")
    tb3 = tb3.sort_values("month_end").drop_duplicates(subset=["month_end"], keep="last")
    tb3 = tb3.rename(columns={"value": "tb3ms"})
    return tb3[["month_end", "tb3ms"]]


def build_target_panel(
    month_end_prices: pd.DataFrame,
    tb3ms_monthly: pd.DataFrame,
    *,
    drop_terminal_without_forward: bool = True,
) -> pd.DataFrame:
    """Create next-month excess-return targets from month-end prices and TB3MS."""
    req_price = {"sleeve_id", "month_end", "adj_close"}
    missing = req_price - set(month_end_prices.columns)
    if missing:
        raise ValueError(f"month_end_prices missing required columns: {sorted(missing)}")

    req_rf = {"month_end", "tb3ms"}
    missing_rf = req_rf - set(tb3ms_monthly.columns)
    if missing_rf:
        raise ValueError(f"tb3ms_monthly missing required columns: {sorted(missing_rf)}")

    rf_map = tb3ms_monthly.copy()
    rf_map["month_end"] = pd.to_datetime(rf_map["month_end"])
    rf_map = rf_map.sort_values("month_end").set_index("month_end")["tb3ms"]

    work = month_end_prices[["sleeve_id", "month_end", "adj_close"]].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    work = work.sort_values(["sleeve_id", "month_end"]).reset_index(drop=True)

    work["price_t"] = work["adj_close"]
    work["price_t1"] = work.groupby("sleeve_id")["price_t"].shift(-1)
    work["next_month_end"] = work.groupby("sleeve_id")["month_end"].shift(-1)

    work["ret_fwd_1m"] = work["price_t1"] / work["price_t"] - 1.0
    work["rf_fwd_1m"] = work["next_month_end"].map(rf_map) / 1200.0
    work["excess_ret_fwd_1m"] = work["ret_fwd_1m"] - work["rf_fwd_1m"]

    work["target_quality_flag"] = (
        work["price_t"].gt(0)
        & work["price_t1"].gt(0)
        & work["ret_fwd_1m"].gt(-1.0)
        & work["rf_fwd_1m"].notna()
        & work["excess_ret_fwd_1m"].notna()
    )

    if drop_terminal_without_forward:
        work = work[work["price_t1"].notna()].copy()

    out_cols = [
        "sleeve_id",
        "month_end",
        "price_t",
        "price_t1",
        "ret_fwd_1m",
        "rf_fwd_1m",
        "excess_ret_fwd_1m",
        "target_quality_flag",
        "next_month_end",
    ]
    out = work[out_cols].reset_index(drop=True)
    return out
