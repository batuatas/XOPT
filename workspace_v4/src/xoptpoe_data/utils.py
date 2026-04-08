"""Shared utility functions for transformations and validation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return dataframe with datetime64 column conversion applied."""
    out = df.copy()
    out[col] = pd.to_datetime(out[col])
    return out


def month_end_from_date(series: pd.Series) -> pd.Series:
    """Convert any date-like series to calendar month-end timestamps."""
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp("M")


def collapse_daily_to_month_end(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    date_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Collapse daily rows to last available observation in each calendar month."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work["month_end"] = month_end_from_date(work[date_col])
    work = work.sort_values(group_cols + [date_col])

    idx = work.groupby(group_cols + ["month_end"], as_index=False)[date_col].idxmax()[date_col]
    out = work.loc[idx, group_cols + ["month_end", date_col] + value_cols].sort_values(
        group_cols + ["month_end"]
    )
    out = out.reset_index(drop=True)
    return out


def log_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Compute log-change over periods with positive-value guard."""
    s = series.astype(float)
    if (s <= 0).any():
        return pd.Series(np.nan, index=s.index)
    return np.log(s / s.shift(periods))


def cumulative_return(series: pd.Series, window: int) -> pd.Series:
    """Rolling cumulative return from simple returns."""
    return (
        (1.0 + series)
        .rolling(window=window, min_periods=window)
        .apply(np.prod, raw=True)
        .astype(float)
        - 1.0
    )


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Compute integer month distance between two month-end date series."""
    later_p = pd.to_datetime(later).dt.to_period("M")
    earlier_p = pd.to_datetime(earlier).dt.to_period("M")
    return later_p.astype(int) - earlier_p.astype(int)


def rolling_max_drawdown(returns: pd.Series, window: int = 12) -> pd.Series:
    """Trailing max drawdown from monthly returns (negative or zero)."""

    def _window_maxdd(x: np.ndarray) -> float:
        wealth = np.cumprod(1.0 + x)
        running_max = np.maximum.accumulate(wealth)
        dd = wealth / running_max - 1.0
        return float(np.min(dd))

    return returns.rolling(window=window, min_periods=window).apply(_window_maxdd, raw=True)


def assert_no_duplicates(df: pd.DataFrame, key_cols: list[str], table_name: str) -> None:
    """Raise error when duplicate keys are found."""
    dupes = df.duplicated(subset=key_cols)
    if dupes.any():
        count = int(dupes.sum())
        raise ValueError(f"{table_name} has {count} duplicate rows for key {key_cols}")
