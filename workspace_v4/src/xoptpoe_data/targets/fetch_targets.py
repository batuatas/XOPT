"""Fetch daily sleeve target histories from a swappable adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from xoptpoe_data.utils import utc_now_iso


class TargetFetchError(RuntimeError):
    """Raised when target acquisition fails."""


class TargetDataAdapter(ABC):
    """Interface for target-data adapters."""

    source_id: str

    @abstractmethod
    def fetch_ticker(self, ticker: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        """Return daily price history for one ticker."""


@dataclass
class YahooFinanceTargetAdapter(TargetDataAdapter):
    """Default target adapter using Yahoo Finance via yfinance."""

    source_id: str = "SRC_YAHOO_FINANCE"

    def fetch_ticker(self, ticker: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise TargetFetchError(
                "yfinance is required for YahooFinanceTargetAdapter; install it explicitly"
            ) from exc

        # yfinance uses an exclusive end date; add one day when end_date is provided.
        yf_end = None
        if end_date:
            yf_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        hist = yf.download(
            tickers=ticker,
            start=start_date,
            end=yf_end,
            auto_adjust=False,
            progress=False,
            actions=False,
            group_by="column",
            threads=False,
        )
        if hist.empty:
            raise TargetFetchError(f"No target history returned for ticker={ticker}")

        if isinstance(hist.columns, pd.MultiIndex):
            if ticker not in hist.columns.get_level_values(-1):
                raise TargetFetchError(
                    f"Ticker substitution detected: requested {ticker}, returned columns {hist.columns}"
                )
            hist = hist.xs(ticker, axis=1, level=-1)

        col_map = {
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        out = hist.reset_index().rename(columns=col_map)

        required_cols = {"trade_date", "close", "adj_close"}
        missing = required_cols - set(out.columns)
        if missing:
            raise TargetFetchError(f"Missing required target columns for {ticker}: {sorted(missing)}")

        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out["ticker"] = ticker
        return out[[c for c in ["ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume"] if c in out.columns]]


def fetch_sleeve_target_raw(
    *,
    asset_master: pd.DataFrame,
    adapter: TargetDataAdapter,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch daily target histories for all locked sleeves."""
    required_cols = {"sleeve_id", "ticker"}
    missing = required_cols - set(asset_master.columns)
    if missing:
        raise ValueError(f"asset_master missing columns needed for target fetch: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    downloaded_at = utc_now_iso()

    for row in asset_master.sort_values("sleeve_id").itertuples(index=False):
        sleeve_id = getattr(row, "sleeve_id")
        ticker = getattr(row, "ticker")
        frame = adapter.fetch_ticker(ticker=ticker, start_date=start_date, end_date=end_date)
        frame["sleeve_id"] = sleeve_id
        frame["source_id"] = adapter.source_id
        frame["download_timestamp"] = downloaded_at
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out = out[[
        "sleeve_id",
        "ticker",
        "trade_date",
        "adj_close",
        "close",
        "source_id",
        "download_timestamp",
        "open",
        "high",
        "low",
        "volume",
    ]]
    out = out.sort_values(["sleeve_id", "trade_date"]).reset_index(drop=True)

    # Hard fail if any locked sleeve is missing.
    expected = set(asset_master["sleeve_id"])
    actual = set(out["sleeve_id"])
    missing_sleeves = expected - actual
    if missing_sleeves:
        raise TargetFetchError(f"Missing fetched sleeves: {sorted(missing_sleeves)}")

    return out
