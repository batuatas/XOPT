"""Fetch macro backbone time series using manifest-locked codes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from xoptpoe_data.utils import utc_now_iso


class MacroFetchError(RuntimeError):
    """Raised when macro acquisition fails."""


class MacroDataAdapter(ABC):
    """Interface for macro adapters."""

    source_id: str

    @abstractmethod
    def fetch_series(self, code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        """Return a DataFrame with columns: obs_date, value."""


@dataclass
class FredMacroAdapter(MacroDataAdapter):
    """Default macro adapter using FRED series via pandas-datareader."""

    source_id: str = "SRC_FRED_API"

    def fetch_series(self, code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        try:
            from pandas_datareader import data as web
        except ImportError as exc:
            raise MacroFetchError(
                "pandas-datareader is required for FredMacroAdapter; install it explicitly"
            ) from exc

        end = end_date or pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        try:
            series_df = web.DataReader(code, "fred", start=start_date, end=end)
        except Exception as exc:  # noqa: BLE001
            raise MacroFetchError(f"Failed to fetch FRED code {code}: {exc}") from exc

        if series_df.empty:
            raise MacroFetchError(f"No observations returned for FRED code {code}")

        out = series_df.reset_index()
        if code not in out.columns:
            if len(out.columns) == 2:
                value_col = [c for c in out.columns if c.lower() not in {"date", "index"}][0]
                out = out.rename(columns={value_col: "value"})
            else:
                raise MacroFetchError(f"Unexpected FRED payload columns for code {code}: {list(out.columns)}")
        else:
            out = out.rename(columns={code: "value"})

        date_col = "DATE" if "DATE" in out.columns else "Date" if "Date" in out.columns else "index"
        out = out.rename(columns={date_col: "obs_date"})
        out["obs_date"] = pd.to_datetime(out["obs_date"])
        return out[["obs_date", "value"]]


def fetch_macro_raw(
    *,
    macro_manifest: pd.DataFrame,
    adapter: MacroDataAdapter,
    start_date: str,
    end_date: str | None = None,
    allow_fallback: bool = False,
    series_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Fetch all manifest macro series with explicit fallback control."""
    required_cols = {
        "series_id",
        "variable_name",
        "geo_block",
        "preferred_code",
        "fallback_code",
        "native_frequency",
    }
    missing = required_cols - set(macro_manifest.columns)
    if missing:
        raise ValueError(f"macro_manifest missing required columns: {sorted(missing)}")

    manifest = macro_manifest.copy()
    if series_filter is not None:
        manifest = manifest[manifest["series_id"].isin(series_filter)].copy()

    if manifest.empty:
        raise ValueError("No macro series selected for fetching")

    downloaded_at = utc_now_iso()
    records: list[pd.DataFrame] = []

    for row in manifest.sort_values("series_id").itertuples(index=False):
        series_id = getattr(row, "series_id")
        variable_name = getattr(row, "variable_name")
        state_var_name = getattr(row, "state_var_name", None)
        geo_block = getattr(row, "geo_block")
        preferred_code = str(getattr(row, "preferred_code")).strip()
        fallback_code_raw = getattr(row, "fallback_code")
        fallback_code = "" if pd.isna(fallback_code_raw) else str(fallback_code_raw).strip()
        native_frequency = getattr(row, "native_frequency")

        used_code = preferred_code
        fallback_used = 0

        try:
            fetched = adapter.fetch_series(code=preferred_code, start_date=start_date, end_date=end_date)
        except MacroFetchError as preferred_err:
            if fallback_code and fallback_code.lower() != "none" and allow_fallback:
                fetched = adapter.fetch_series(code=fallback_code, start_date=start_date, end_date=end_date)
                used_code = fallback_code
                fallback_used = 1
            else:
                raise MacroFetchError(
                    f"Failed preferred macro code for series_id={series_id} code={preferred_code}. "
                    "Fallback is disabled or unavailable."
                ) from preferred_err

        fetched["series_id"] = series_id
        fetched["variable_name"] = variable_name
        fetched["state_var_name"] = state_var_name
        fetched["geo_block"] = geo_block
        fetched["native_frequency"] = native_frequency
        fetched["source_id"] = adapter.source_id
        fetched["download_timestamp"] = downloaded_at
        fetched["preferred_code"] = preferred_code
        fetched["fallback_code"] = fallback_code
        fetched["used_code"] = used_code
        fetched["fallback_used"] = fallback_used

        records.append(fetched)

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(["series_id", "obs_date"]).reset_index(drop=True)

    out = out[[
        "series_id",
        "variable_name",
        "state_var_name",
        "geo_block",
        "obs_date",
        "value",
        "native_frequency",
        "source_id",
        "download_timestamp",
        "preferred_code",
        "fallback_code",
        "used_code",
        "fallback_used",
    ]]

    # Hard check: one manifest series should map to one output series id with non-empty data.
    missing_series = set(manifest["series_id"]) - set(out["series_id"])
    if missing_series:
        raise MacroFetchError(f"Missing fetched macro series: {sorted(missing_series)}")

    return out
