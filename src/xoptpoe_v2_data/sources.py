"""External source ingestion for the XOPTPOE v2 long-horizon build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import warnings
import zipfile

import numpy as np
import pandas as pd

from xoptpoe_v2_data.config import V2Paths

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DATE_COLUMN_CANDIDATES: tuple[str, ...] = ("Date", "observation_date", "TIME_PERIOD")
EXCLUDED_SOURCE_REASONS: dict[str, tuple[str, str]] = {
    "country_level_macro_data/Historic-cape-ratios.csv": (
        "excluded_duplicate",
        "Duplicate of the top-level CAPE upload.",
    ),
    "country_level_macro_data/china_labour_market.csv": (
        "excluded_fragile",
        "Mostly annual and sparse; not cleanly usable in a monthly feature store.",
    ),
    "country_level_macro_data/japan_10y_bond_topix_dividends.csv": (
        "excluded_inconsistent",
        "Contents duplicate the yield/CPI file and do not match the file name reliably.",
    ),
}


@dataclass(frozen=True)
class SeriesSpec:
    """Specification for one aligned monthly source series."""

    feature_name: str
    source_file: str
    block_name: str
    geography: str
    native_frequency: str
    lag_months: int
    loader_kind: str
    transform_type: str = "level"
    date_col: str = "Date"
    value_col: str | None = None
    zip_member: str | None = None
    left_col: str | None = None
    right_col: str | None = None
    oecd_area: str | None = None
    oecd_measure: str | None = None
    oecd_adjustment: str | None = None
    minimum_valid_months: int = 1
    derivations: tuple[str, ...] = ("delta_1m",)
    excel_price_col_index: int | None = None
    notes: str = ""



def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    dayfirst_mask = text.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", na=False)
    month_mask = text.str.match(r"^\d{4}-\d{2}$", na=False)
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    iso_dt_mask = text.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", na=False)
    named_month_mask = text.str.match(r"^\d{1,2}-[A-Za-z]{3}-\d{4}( \d{2}:\d{2})?(:\d{2})?$", na=False)
    other_mask = ~(dayfirst_mask | month_mask | iso_mask | iso_dt_mask | named_month_mask)

    if dayfirst_mask.any():
        parsed.loc[dayfirst_mask] = pd.to_datetime(text.loc[dayfirst_mask], errors="coerce", dayfirst=True)
    if month_mask.any():
        parsed.loc[month_mask] = pd.to_datetime(text.loc[month_mask], errors="coerce", format="%Y-%m")
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], errors="coerce", format="%Y-%m-%d")
    if iso_dt_mask.any():
        parsed.loc[iso_dt_mask] = pd.to_datetime(
            text.loc[iso_dt_mask], errors="coerce", format="%Y-%m-%d %H:%M:%S"
        )
    if named_month_mask.any():
        parsed.loc[named_month_mask] = pd.to_datetime(text.loc[named_month_mask], errors="coerce")
    if other_mask.any():
        parsed.loc[other_mask] = pd.to_datetime(text.loc[other_mask], errors="coerce")
    return parsed



def _to_month_end(values: pd.Series) -> pd.Series:
    return _parse_dates(values).dt.to_period("M").dt.to_timestamp("M")



def _infer_frequency(dates: pd.Series) -> str:
    clean = pd.Series(pd.to_datetime(dates, errors="coerce")).dropna().drop_duplicates().sort_values()
    if clean.empty or len(clean) < 3:
        return "unknown"
    diffs = clean.diff().dt.days.dropna()
    if diffs.empty:
        return "unknown"
    median_days = float(diffs.median())
    if median_days <= 7:
        return "daily"
    if median_days <= 40:
        return "monthly"
    if median_days <= 120:
        return "quarterly"
    if median_days <= 400:
        return "annual"
    return "irregular"



def _collapse_last_valid_by_month(df: pd.DataFrame, *, date_col: str, value_col: str) -> pd.DataFrame:
    work = df[[date_col, value_col]].copy()
    work[date_col] = _parse_dates(work[date_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(columns=["source_month_end", "value"])
    work["source_month_end"] = work[date_col].dt.to_period("M").dt.to_timestamp("M")
    return (
        work.groupby("source_month_end", as_index=False)
        .agg(value=(value_col, "last"))
        .sort_values("source_month_end")
        .reset_index(drop=True)
    )



def _align_monthly_series(source_df: pd.DataFrame, *, row_months: pd.Index, lag_months: int) -> pd.Series:
    row_index = pd.Index(sorted(pd.to_datetime(row_months).unique()), name="month_end")
    if source_df.empty:
        return pd.Series(index=row_index, dtype=float)

    target = pd.DataFrame({"month_end": row_index})
    target["cutoff"] = target["month_end"] - pd.offsets.DateOffset(months=lag_months)
    aligned = pd.merge_asof(
        target.sort_values("cutoff"),
        source_df.sort_values("source_month_end"),
        left_on="cutoff",
        right_on="source_month_end",
        direction="backward",
    )
    return aligned.set_index("month_end")["value"].reindex(row_index)



def _read_top_csv(path: Path, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = str(path)
    if key not in cache:
        cache[key] = pd.read_csv(path)
    return cache[key].copy()



def _read_zip_csv(zip_path: Path, member: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = f"{zip_path}::{member}"
    if key not in cache:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as handle:
                cache[key] = pd.read_csv(handle)
    return cache[key].copy()



def _extract_workbook_price(zip_path: Path, member: str, price_col_index: int) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        payload = zf.read(member)
    with tempfile.NamedTemporaryFile(suffix=Path(member).suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        raw = pd.read_excel(tmp_path, sheet_name=0, header=None)
    finally:
        tmp_path.unlink(missing_ok=True)

    first_col = _parse_dates(raw.iloc[:, 0])
    value_col = pd.to_numeric(raw.iloc[:, price_col_index], errors="coerce")
    work = pd.DataFrame({"date": first_col, "value": value_col})
    work = work.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return work



def _load_oecd_leading_series(spec: SeriesSpec, top_level_path: Path, row_months: pd.Index, cache: dict[str, pd.DataFrame]) -> pd.Series:
    df = _read_top_csv(top_level_path, cache)
    chunk = df.loc[
        (df["REF_AREA"] == spec.oecd_area)
        & (df["MEASURE"] == spec.oecd_measure)
        & (df["ADJUSTMENT"] == spec.oecd_adjustment),
        ["TIME_PERIOD", "OBS_VALUE"],
    ].copy()
    chunk.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"}, inplace=True)
    collapsed = _collapse_last_valid_by_month(chunk, date_col="date", value_col="value")
    return _align_monthly_series(collapsed, row_months=row_months, lag_months=spec.lag_months)



def _load_oecd_bts_series(spec: SeriesSpec, zip_path: Path, row_months: pd.Index, cache: dict[str, pd.DataFrame]) -> pd.Series:
    df = _read_zip_csv(zip_path, spec.zip_member or "", cache)
    chunk = df.loc[
        (df["REF_AREA"] == spec.oecd_area)
        & (df["MEASURE"] == spec.oecd_measure)
        & (df["ADJUSTMENT"] == spec.oecd_adjustment),
        ["TIME_PERIOD", "OBS_VALUE"],
    ].copy()
    chunk.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"}, inplace=True)
    collapsed = _collapse_last_valid_by_month(chunk, date_col="date", value_col="value")
    return _align_monthly_series(collapsed, row_months=row_months, lag_months=spec.lag_months)



def _load_series(spec: SeriesSpec, paths: V2Paths, row_months: pd.Index, cache: dict[str, pd.DataFrame]) -> pd.Series:
    top_oecd = paths.data_external_dir / "OECD Leading Indicatorscsv.csv"
    zip_path = paths.data_external_dir / "country_level_macro_data.zip"

    if spec.loader_kind == "top_oecd_leading":
        return _load_oecd_leading_series(spec, top_oecd, row_months, cache)

    if spec.loader_kind == "zip_oecd_bts":
        return _load_oecd_bts_series(spec, zip_path, row_months, cache)

    if spec.loader_kind == "top_csv_column":
        df = _read_top_csv(paths.project_root / spec.source_file, cache)
        collapsed = _collapse_last_valid_by_month(df, date_col=spec.date_col, value_col=spec.value_col or "")
        return _align_monthly_series(collapsed, row_months=row_months, lag_months=spec.lag_months)

    if spec.loader_kind == "zip_csv_column":
        df = _read_zip_csv(zip_path, spec.zip_member or "", cache)
        collapsed = _collapse_last_valid_by_month(df, date_col=spec.date_col, value_col=spec.value_col or "")
        return _align_monthly_series(collapsed, row_months=row_months, lag_months=spec.lag_months)

    if spec.loader_kind == "zip_csv_spread":
        df = _read_zip_csv(zip_path, spec.zip_member or "", cache)
        spread = pd.to_numeric(df[spec.value_col or ""], errors="coerce")
        if spec.left_col and spec.right_col:
            left = pd.to_numeric(df[spec.left_col], errors="coerce")
            right = pd.to_numeric(df[spec.right_col], errors="coerce")
            value = spread.where(spread.notna(), left - right)
        else:
            value = spread
        work = pd.DataFrame({"date": df[spec.date_col], "value": value})
        collapsed = _collapse_last_valid_by_month(work, date_col="date", value_col="value")
        return _align_monthly_series(collapsed, row_months=row_months, lag_months=spec.lag_months)

    if spec.loader_kind == "zip_excel_price":
        price_df = _extract_workbook_price(zip_path, spec.zip_member or "", spec.excel_price_col_index or 1)
        collapsed = _collapse_last_valid_by_month(price_df, date_col="date", value_col="value")
        return _align_monthly_series(collapsed, row_months=row_months, lag_months=spec.lag_months)

    raise ValueError(f"Unsupported loader_kind: {spec.loader_kind}")



def _derive_delta(series: pd.Series) -> pd.Series:
    return series.diff(1)



def _derive_market_features(series: pd.Series) -> dict[str, pd.Series]:
    positive = series.where(series > 0)
    log_values = np.log(positive)
    return {
        "logchg_1m": log_values.diff(1),
        "logchg_12m": log_values.diff(12),
        "mom_12_1": log_values.shift(1) - log_values.shift(13),
    }



def _derive_yoy(series: pd.Series) -> pd.Series:
    base = series.shift(12)
    return np.where(base.abs() > 1e-12, series / base - 1.0, np.nan)



def candidate_series_specs() -> list[SeriesSpec]:
    """Return the full v2 candidate series specification list."""
    specs: list[SeriesSpec] = []

    for area, suffix, measure in [
        ("USA", "us", "LI"),
        ("USA", "us", "BCICP"),
        ("USA", "us", "CCICP"),
        ("EA20", "ea", "BCICP"),
        ("EA20", "ea", "CCICP"),
        ("JPN", "jp", "LI"),
        ("JPN", "jp", "BCICP"),
        ("JPN", "jp", "CCICP"),
        ("CHN", "cn", "LI"),
        ("CHN", "cn", "BCICP"),
        ("CHN", "cn", "CCICP"),
    ]:
        specs.append(
            SeriesSpec(
                feature_name=f"oecd_{measure.lower()}_{suffix}",
                source_file="data/external/akif_candidates/OECD Leading Indicatorscsv.csv",
                block_name="oecd_leading",
                geography={"us": "US", "ea": "EURO_AREA", "jp": "JAPAN", "cn": "CHINA"}[suffix],
                native_frequency="monthly",
                lag_months=1,
                loader_kind="top_oecd_leading",
                oecd_area=area,
                oecd_measure=measure,
                oecd_adjustment="AA",
                derivations=("delta_1m",),
                notes="Top-level OECD leading-indicator file.",
            )
        )

    for area, suffix, measure, name in [
        ("USA", "us", "CURT", "oecd_bts_capacity_util_us"),
        ("USA", "us", "EM", "oecd_bts_employment_us"),
        ("USA", "us", "OB", "oecd_bts_order_books_us"),
        ("USA", "us", "OI", "oecd_bts_orders_inflow_us"),
        ("USA", "us", "PR", "oecd_bts_production_us"),
        ("USA", "us", "XR", "oecd_bts_export_orders_us"),
        ("EA20", "ea", "BU", "oecd_bts_business_situation_ea"),
        ("EA20", "ea", "DE", "oecd_bts_demand_evolution_ea"),
        ("EA20", "ea", "EM", "oecd_bts_employment_ea"),
        ("EA20", "ea", "OB", "oecd_bts_order_books_ea"),
        ("EA20", "ea", "OD", "oecd_bts_order_demand_ea"),
        ("EA20", "ea", "PR", "oecd_bts_production_ea"),
        ("EA20", "ea", "SP", "oecd_bts_selling_prices_ea"),
        ("EA20", "ea", "VS", "oecd_bts_volume_stocks_ea"),
        ("JPN", "jp", "SP", "oecd_bts_selling_prices_jp"),
    ]:
        specs.append(
            SeriesSpec(
                feature_name=name,
                source_file="country_level_macro_data/OECD.SDD.STES,DSD_STES@DF_BTS,4.0+CHN+JPN+EA20+USA.M........csv",
                zip_member="country_level_macro_data/OECD.SDD.STES,DSD_STES@DF_BTS,4.0+CHN+JPN+EA20+USA.M........csv",
                block_name="oecd_bts",
                geography={"us": "US", "ea": "EURO_AREA", "jp": "JAPAN"}[suffix],
                native_frequency="monthly",
                lag_months=1,
                loader_kind="zip_oecd_bts",
                oecd_area=area,
                oecd_measure=measure,
                oecd_adjustment="Y",
                minimum_valid_months=36,
                derivations=("delta_1m",),
                notes="Monthly OECD business-tendency-survey series.",
            )
        )

    for name, geography, column in [
        ("cape_china", "CHINA", "China"),
        ("cape_europe", "EURO_AREA", "Europe"),
        ("cape_japan", "JAPAN", "Japan"),
        ("cape_usa", "US", "USA"),
    ]:
        specs.append(
            SeriesSpec(
                feature_name=name,
                source_file="data/external/akif_candidates/Historic-cape-ratios (1).csv",
                block_name="cape",
                geography=geography,
                native_frequency="monthly",
                lag_months=0,
                loader_kind="top_csv_column",
                date_col="Date",
                value_col=column,
                derivations=("delta_1m",),
            )
        )

    china_zip = {
        "source_file": "country_level_macro_data.zip",
    }

    specs.extend(
        [
            SeriesSpec("china_cli", "country_level_macro_data/china_cli.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="COMPOSITE LEADING IN", zip_member="country_level_macro_data/china_cli.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cmi", "country_level_macro_data/china_gdp_cmi.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="CMI", zip_member="country_level_macro_data/china_gdp_cmi.csv", derivations=("delta_1m",)),
            SeriesSpec("china_gdp_official", "country_level_macro_data/china_gdp_cmi.csv", "china_macro", "CHINA", "quarterly", 4, "zip_csv_column", date_col="Date", value_col="Official GDP", zip_member="country_level_macro_data/china_gdp_cmi.csv", derivations=("delta_1m",), minimum_valid_months=24),
            SeriesSpec("china_pmi_nbs_mfg", "country_level_macro_data/china_pmi.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="NBS manufacturing", zip_member="country_level_macro_data/china_pmi.csv", derivations=("delta_1m",)),
            SeriesSpec("china_pmi_nbs_nonmfg", "country_level_macro_data/china_pmi.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="NBS non-manufacturing", zip_member="country_level_macro_data/china_pmi.csv", derivations=("delta_1m",)),
            SeriesSpec("china_pmi_caixin_mfg", "country_level_macro_data/china_pmi.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Caixin manufacturing", zip_member="country_level_macro_data/china_pmi.csv", derivations=("delta_1m",)),
            SeriesSpec("china_pmi_caixin_services", "country_level_macro_data/china_pmi.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Caixin services", zip_member="country_level_macro_data/china_pmi.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_food", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Food", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_clothing", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Clothing", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_residence", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Residence", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_household_articles", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Household articles & services", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_transport_comm", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Transport and communication", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_education_culture", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Education culture & recreation", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_health_care", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Health care", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_headline", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Headline", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_cpi_core", "country_level_macro_data/china_cpi_data.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Core", zip_member="country_level_macro_data/china_cpi_data.csv", derivations=("delta_1m",)),
            SeriesSpec("china_ip_growth", "country_level_macro_data/china_industrial_production.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="INDUSTRIAL PROD.: GR", zip_member="country_level_macro_data/china_industrial_production.csv", derivations=("delta_1m",)),
            SeriesSpec("china_m2_yoy", "country_level_macro_data/china_fiscal_socioeconomic.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="MONEY SUPPLY - M2 YOY", zip_member="country_level_macro_data/china_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("china_fiscal_expenditure", "country_level_macro_data/china_fiscal_socioeconomic.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Expenditure", zip_member="country_level_macro_data/china_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("china_fiscal_revenue", "country_level_macro_data/china_fiscal_socioeconomic.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Revenue", zip_member="country_level_macro_data/china_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("china_houseprice_100cities", "country_level_macro_data/china_fiscal_socioeconomic.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="China Index Academy avg 100 cities", zip_member="country_level_macro_data/china_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("china_houseprice_nbs_new70", "country_level_macro_data/china_fiscal_socioeconomic.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="NBS avg 70 cities newly built", zip_member="country_level_macro_data/china_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("china_houseprice_nbs_existing70", "country_level_macro_data/china_fiscal_socioeconomic.csv", "china_macro", "CHINA", "monthly", 1, "zip_csv_column", date_col="Date", value_col="NBS avg 70 cities existing", zip_member="country_level_macro_data/china_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("china_div_yield", "country_level_macro_data/China_dividend_yield.csv", "china_valuation", "CHINA", "monthly", 0, "zip_csv_column", date_col="Date", value_col="China", zip_member="country_level_macro_data/China_dividend_yield.csv", derivations=("delta_1m",)),
            SeriesSpec("china_div_yield_em", "country_level_macro_data/China_dividend_yield.csv", "china_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Emerging markets", zip_member="country_level_macro_data/China_dividend_yield.csv", derivations=("delta_1m",)),
            SeriesSpec("china_div_yield_spread", "country_level_macro_data/China_dividend_yield.csv", "china_valuation", "CHINA", "monthly", 0, "zip_csv_spread", date_col="Date", value_col="Spread (RHS)", left_col="China", right_col="Emerging markets", zip_member="country_level_macro_data/China_dividend_yield.csv", derivations=("delta_1m",)),
            SeriesSpec("china_pe", "country_level_macro_data/China_price_to_earnings.csv", "china_valuation", "CHINA", "monthly", 0, "zip_csv_column", date_col="Date", value_col="China", zip_member="country_level_macro_data/China_price_to_earnings.csv", derivations=("delta_1m",)),
            SeriesSpec("china_pe_em", "country_level_macro_data/China_price_to_earnings.csv", "china_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Emerging markets", zip_member="country_level_macro_data/China_price_to_earnings.csv", derivations=("delta_1m",)),
            SeriesSpec("china_pe_spread", "country_level_macro_data/China_price_to_earnings.csv", "china_valuation", "CHINA", "monthly", 0, "zip_csv_spread", date_col="Date", value_col="Spread (RHS)", left_col="China", right_col="Emerging markets", zip_member="country_level_macro_data/China_price_to_earnings.csv", derivations=("delta_1m",)),
            SeriesSpec("china_ptb", "country_level_macro_data/china & em price-to-book ratio.csv", "china_valuation", "CHINA", "monthly", 0, "zip_csv_column", date_col="Date", value_col="China", zip_member="country_level_macro_data/china & em price-to-book ratio.csv", derivations=("delta_1m",)),
            SeriesSpec("china_ptb_em", "country_level_macro_data/china & em price-to-book ratio.csv", "china_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Emerging markets", zip_member="country_level_macro_data/china & em price-to-book ratio.csv", derivations=("delta_1m",)),
            SeriesSpec("china_ptb_spread", "country_level_macro_data/china & em price-to-book ratio.csv", "china_valuation", "CHINA", "monthly", 0, "zip_csv_spread", date_col="Date", value_col="Spread (RHS)", left_col="China", right_col="Emerging markets", zip_member="country_level_macro_data/china & em price-to-book ratio.csv", derivations=("delta_1m",)),
            SeriesSpec("china_sse_composite_usd", "country_level_macro_data/China Price History_SSE_Composite.xlsx", "china_market", "CHINA", "monthly", 0, "zip_excel_price", zip_member="country_level_macro_data/China Price History_SSE_Composite.xlsx", transform_type="market_price_level", derivations=("market_transforms",), excel_price_col_index=1),
            SeriesSpec("jp_oecd_leading_alt", "country_level_macro_data/JPNLOLITONOSTSAM.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="observation_date", value_col="JPNLOLITONOSTSAM", zip_member="country_level_macro_data/JPNLOLITONOSTSAM.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_ngdp_level", "country_level_macro_data/JaPaN GDP.csv", "japan_enrichment", "JAPAN", "quarterly", 4, "zip_csv_column", date_col="observation_date", value_col="JPNNGDP", zip_member="country_level_macro_data/JaPaN GDP.csv", minimum_valid_months=24, derivations=("yoy",)),
            SeriesSpec("jp_policy_rate", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Target policy rate", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_unemployment_rate_alt", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="UNEMPLOYMENT RATE", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_cpi_headline_alt", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="CPI Headline", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_cpi_core_ex_fresh", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="CPI Core ex fresh food", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_cpi_core_ex_food_energy", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="CPI Core ex fresh food and energy", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_unemployment_level", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="UNEMPLOYMENT LEVEL", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_employed_persons", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="EMPLOYED PERSONS", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_ip_3m", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="IP Three-month", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_ip_12m", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="IP Twelve-month", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_ip_3mma", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="IP Three-month 3MMA", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_ip_12m_alt", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="IP Twelve-month 2", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_short_rate_alt", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Short-term interest rate", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_long_rate_alt", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="Long-term government bond yield", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_gdp_quarterly", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "quarterly", 4, "zip_csv_column", date_col="Date", value_col="GDP Quarterly", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", minimum_valid_months=24, derivations=("delta_1m",)),
            SeriesSpec("jp_gdp_four_quarter", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "quarterly", 4, "zip_csv_column", date_col="Date", value_col="GDP Four-quarter", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", minimum_valid_months=24, derivations=("delta_1m",)),
            SeriesSpec("jp_tankan_actual", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "quarterly", 4, "zip_csv_column", date_col="Date", value_col="Tankan Actual", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", minimum_valid_months=24, derivations=("delta_1m",)),
            SeriesSpec("jp_tankan_forecast", "country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", "japan_enrichment", "JAPAN", "quarterly", 4, "zip_csv_column", date_col="Date", value_col="Tankan Forecast", zip_member="country_level_macro_data/japan_macro_fiscal_socioeconomic.csv", minimum_valid_months=24, derivations=("delta_1m",)),
            SeriesSpec("jp_pe_ratio", "country_level_macro_data/japan_pe_topix_index.csv", "japan_enrichment", "JAPAN", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Price to earnings", zip_member="country_level_macro_data/japan_pe_topix_index.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_topix_index", "country_level_macro_data/japan_pe_topix_index.csv", "japan_enrichment", "JAPAN", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Topix index (RHS)", zip_member="country_level_macro_data/japan_pe_topix_index.csv", transform_type="market_price_level", derivations=("market_transforms",)),
            SeriesSpec("jp_buyback_index_usd", "country_level_macro_data/japan buyback yield.xlsx", "japan_enrichment", "JAPAN", "monthly", 0, "zip_excel_price", zip_member="country_level_macro_data/japan buyback yield.xlsx", transform_type="market_price_level", derivations=("market_transforms",), excel_price_col_index=1),
            SeriesSpec("jp_10y_yield_file", "country_level_macro_data/japan_10y_bond_yield_cpi.csv", "japan_enrichment", "JAPAN", "daily", 0, "zip_csv_column", date_col="Date", value_col="Ten-year government bond yield", zip_member="country_level_macro_data/japan_10y_bond_yield_cpi.csv", derivations=("delta_1m",)),
            SeriesSpec("jp_cpi_file", "country_level_macro_data/japan_10y_bond_yield_cpi.csv", "japan_enrichment", "JAPAN", "monthly", 1, "zip_csv_column", date_col="Date", value_col="CPI", zip_member="country_level_macro_data/japan_10y_bond_yield_cpi.csv", derivations=("delta_1m",), minimum_valid_months=24),
            SeriesSpec("em_div_yield", "country_level_macro_data/em_global_div_yield.csv", "em_global_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Emerging markets", zip_member="country_level_macro_data/em_global_div_yield.csv", derivations=("delta_1m",)),
            SeriesSpec("global_div_yield", "country_level_macro_data/em_global_div_yield.csv", "em_global_valuation", "GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Global index", zip_member="country_level_macro_data/em_global_div_yield.csv", derivations=("delta_1m",)),
            SeriesSpec("em_minus_global_div_yield", "country_level_macro_data/em_global_div_yield.csv", "em_global_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_spread", date_col="Date", value_col="Spread (RHS)", left_col="Emerging markets", right_col="Global index", zip_member="country_level_macro_data/em_global_div_yield.csv", derivations=("delta_1m",)),
            SeriesSpec("em_ptb", "country_level_macro_data/em_global_ptb.csv", "em_global_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Emerging markets", zip_member="country_level_macro_data/em_global_ptb.csv", derivations=("delta_1m",)),
            SeriesSpec("global_ptb", "country_level_macro_data/em_global_ptb.csv", "em_global_valuation", "GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Global index", zip_member="country_level_macro_data/em_global_ptb.csv", derivations=("delta_1m",)),
            SeriesSpec("em_minus_global_ptb", "country_level_macro_data/em_global_ptb.csv", "em_global_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_spread", date_col="Date", value_col="Spread (RHS)", left_col="Emerging markets", right_col="Global index", zip_member="country_level_macro_data/em_global_ptb.csv", derivations=("delta_1m",)),
            SeriesSpec("em_pe", "country_level_macro_data/em_global_pte.csv", "em_global_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Emerging markets", zip_member="country_level_macro_data/em_global_pte.csv", derivations=("delta_1m",)),
            SeriesSpec("global_pe", "country_level_macro_data/em_global_pte.csv", "em_global_valuation", "GLOBAL", "monthly", 0, "zip_csv_column", date_col="Date", value_col="Global index", zip_member="country_level_macro_data/em_global_pte.csv", derivations=("delta_1m",)),
            SeriesSpec("em_minus_global_pe", "country_level_macro_data/em_global_pte.csv", "em_global_valuation", "EM_GLOBAL", "monthly", 0, "zip_csv_spread", date_col="Date", value_col="Spread (RHS)", left_col="Emerging markets", right_col="Global index", zip_member="country_level_macro_data/em_global_pte.csv", derivations=("delta_1m",)),
            SeriesSpec("us_hy_oas", "country_level_macro_data/BAMLH0A0HYM2.csv", "credit_stress", "US", "daily", 0, "zip_csv_column", date_col="observation_date", value_col="BAMLH0A0HYM2", zip_member="country_level_macro_data/BAMLH0A0HYM2.csv", derivations=("delta_1m",)),
            SeriesSpec("us_hy_eff_yield", "country_level_macro_data/BAMLH0A0HYM2EY.csv", "credit_stress", "US", "daily", 0, "zip_csv_column", date_col="observation_date", value_col="BAMLH0A0HYM2EY", zip_member="country_level_macro_data/BAMLH0A0HYM2EY.csv", derivations=("delta_1m",)),
            SeriesSpec("eu_hy_oas", "country_level_macro_data/BAMLHE00EHYIOAS.csv", "credit_stress", "EURO_AREA", "daily", 0, "zip_csv_column", date_col="observation_date", value_col="BAMLHE00EHYIOAS", zip_member="country_level_macro_data/BAMLHE00EHYIOAS.csv", derivations=("delta_1m",)),
            SeriesSpec("eu_hy_eff_yield", "country_level_macro_data/BAMLHE00EHYIEY.csv", "credit_stress", "EURO_AREA", "daily", 0, "zip_csv_column", date_col="observation_date", value_col="BAMLHE00EHYIEY", zip_member="country_level_macro_data/BAMLHE00EHYIEY.csv", derivations=("delta_1m",)),
            SeriesSpec("eu_ig_corp_tr_usd", "country_level_macro_data/EU_IG_.xlsx", "eu_ig_market", "EURO_AREA", "monthly", 0, "zip_excel_price", zip_member="country_level_macro_data/EU_IG_.xlsx", transform_type="market_price_level", derivations=("market_transforms",), excel_price_col_index=2),
        ]
    )

    return specs



def build_additional_monthly_state(paths: V2Paths, row_months: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the aligned monthly additional feature state and its dictionaries."""
    state = pd.DataFrame({"month_end": pd.Index(sorted(pd.to_datetime(row_months).unique()))})
    cache: dict[str, pd.DataFrame] = {}
    feature_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []

    for spec in candidate_series_specs():
        series = _load_series(spec, paths, state["month_end"], cache)
        nonmissing_count = int(series.notna().sum())
        if nonmissing_count < spec.minimum_valid_months:
            rejected_rows.append(
                {
                    "feature_name": spec.feature_name,
                    "source_file": spec.source_file,
                    "block_name": spec.block_name,
                    "geography": spec.geography,
                    "rejection_reason": f"nonmissing_count<{spec.minimum_valid_months}",
                    "nonmissing_count": nonmissing_count,
                }
            )
            continue

        state[spec.feature_name] = series.to_numpy(dtype=float)
        first_valid = series.dropna().index.min()
        last_valid = series.dropna().index.max()
        feature_rows.append(
            {
                "feature_name": spec.feature_name,
                "table_scope": "feature_master_monthly",
                "source_file": spec.source_file,
                "source_column": spec.value_col or spec.oecd_measure or "trade_price",
                "block_name": spec.block_name,
                "geography": spec.geography,
                "transform_type": spec.transform_type,
                "native_frequency": spec.native_frequency,
                "lag_months": spec.lag_months,
                "first_valid_date": first_valid,
                "last_valid_date": last_valid,
                "nonmissing_count": nonmissing_count,
                "nonmissing_share": float(series.notna().mean()),
                "notes": spec.notes,
            }
        )

        if "delta_1m" in spec.derivations:
            derived = _derive_delta(series)
            col = f"{spec.feature_name}_delta_1m"
            state[col] = derived.to_numpy(dtype=float)
            feature_rows.append(
                {
                    "feature_name": col,
                    "table_scope": "feature_master_monthly",
                    "source_file": spec.source_file,
                    "source_column": spec.value_col or spec.oecd_measure or "trade_price",
                    "block_name": spec.block_name,
                    "geography": spec.geography,
                    "transform_type": "delta_1m",
                    "native_frequency": spec.native_frequency,
                    "lag_months": spec.lag_months,
                    "first_valid_date": derived.dropna().index.min() if derived.notna().any() else pd.NaT,
                    "last_valid_date": derived.dropna().index.max() if derived.notna().any() else pd.NaT,
                    "nonmissing_count": int(derived.notna().sum()),
                    "nonmissing_share": float(derived.notna().mean()),
                    "notes": f"Derived from {spec.feature_name}",
                }
            )

        if "market_transforms" in spec.derivations:
            for suffix, derived in _derive_market_features(series).items():
                col = f"{spec.feature_name}_{suffix}"
                state[col] = derived.to_numpy(dtype=float)
                feature_rows.append(
                    {
                        "feature_name": col,
                        "table_scope": "feature_master_monthly",
                        "source_file": spec.source_file,
                        "source_column": spec.value_col or "trade_price",
                        "block_name": spec.block_name,
                        "geography": spec.geography,
                        "transform_type": suffix,
                        "native_frequency": spec.native_frequency,
                        "lag_months": spec.lag_months,
                        "first_valid_date": derived.dropna().index.min() if derived.notna().any() else pd.NaT,
                        "last_valid_date": derived.dropna().index.max() if derived.notna().any() else pd.NaT,
                        "nonmissing_count": int(derived.notna().sum()),
                        "nonmissing_share": float(derived.notna().mean()),
                        "notes": f"Derived from {spec.feature_name}",
                    }
                )

        if "yoy" in spec.derivations:
            derived = pd.Series(_derive_yoy(series), index=series.index, dtype=float)
            col = f"{spec.feature_name}_yoy"
            state[col] = derived.to_numpy(dtype=float)
            feature_rows.append(
                {
                    "feature_name": col,
                    "table_scope": "feature_master_monthly",
                    "source_file": spec.source_file,
                    "source_column": spec.value_col or "value",
                    "block_name": spec.block_name,
                    "geography": spec.geography,
                    "transform_type": "yoy_change",
                    "native_frequency": spec.native_frequency,
                    "lag_months": spec.lag_months,
                    "first_valid_date": derived.dropna().index.min() if derived.notna().any() else pd.NaT,
                    "last_valid_date": derived.dropna().index.max() if derived.notna().any() else pd.NaT,
                    "nonmissing_count": int(derived.notna().sum()),
                    "nonmissing_share": float(derived.notna().mean()),
                    "notes": f"Derived from {spec.feature_name}",
                }
            )

    feature_meta = pd.DataFrame(feature_rows).sort_values("feature_name").reset_index(drop=True)
    if rejected_rows:
        rejected = pd.DataFrame(rejected_rows).sort_values(["source_file", "feature_name"]).reset_index(drop=True)
    else:
        rejected = pd.DataFrame(
            columns=[
                "feature_name",
                "source_file",
                "block_name",
                "geography",
                "rejection_reason",
                "nonmissing_count",
            ]
        )
    return state, feature_meta, rejected



def build_external_file_inventory(paths: V2Paths, feature_meta: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    """Inventory top-level and zipped external files, including v2 usage status."""
    used_counts = feature_meta.groupby("source_file").size().to_dict()
    rejected_counts = rejected.groupby("source_file").size().to_dict() if not rejected.empty else {}
    records: list[dict[str, object]] = []
    top_level_files = [
        paths.data_external_dir / "OECD Leading Indicatorscsv.csv",
        paths.data_external_dir / "Historic-cape-ratios (1).csv",
    ]
    zip_path = paths.data_external_dir / "country_level_macro_data.zip"

    for path in top_level_files:
        df = pd.read_csv(path)
        date_col = next((col for col in DATE_COLUMN_CANDIDATES if col in df.columns), None)
        dates = _parse_dates(df[date_col]) if date_col else pd.Series(dtype="datetime64[ns]")
        file_id = str(path.relative_to(paths.project_root))
        status = "used_feature_source" if file_id in used_counts else "unused_not_selected"
        reason = "Used in v2 feature store." if status == "used_feature_source" else "Not selected directly."
        records.append(
            {
                "file_id": file_id,
                "container": "top_level",
                "file_type": path.suffix.lower(),
                "row_count": int(df.shape[0]),
                "column_count": int(df.shape[1]),
                "date_column": date_col or "",
                "start_date": dates.min().date().isoformat() if not dates.dropna().empty else "",
                "end_date": dates.max().date().isoformat() if not dates.dropna().empty else "",
                "detected_frequency": _infer_frequency(dates),
                "feature_count_used": int(used_counts.get(file_id, 0)),
                "rejected_feature_count": int(rejected_counts.get(file_id, 0)),
                "v2_status": status,
                "v2_reason": reason,
            }
        )

    with zipfile.ZipFile(zip_path) as zf:
        for member in sorted(zf.namelist()):
            if member.endswith("/") or member.startswith("__MACOSX/") or member.endswith(".DS_Store"):
                continue
            suffix = Path(member).suffix.lower()
            file_id = member
            status = "used_feature_source" if file_id in used_counts else EXCLUDED_SOURCE_REASONS.get(file_id, ("unused_not_selected", "Not selected for v2."))[0]
            reason = "Used in v2 feature store." if file_id in used_counts else EXCLUDED_SOURCE_REASONS.get(file_id, (None, "Not selected for v2."))[1]

            row_count = np.nan
            col_count = np.nan
            date_col = ""
            start_date = ""
            end_date = ""
            frequency = "not_parsed"

            if suffix == ".csv":
                with zf.open(member) as handle:
                    df = pd.read_csv(handle)
                row_count = int(df.shape[0])
                col_count = int(df.shape[1])
                date_col = next((col for col in DATE_COLUMN_CANDIDATES if col in df.columns), "")
                dates = _parse_dates(df[date_col]) if date_col else pd.Series(dtype="datetime64[ns]")
                start_date = dates.min().date().isoformat() if not dates.dropna().empty else ""
                end_date = dates.max().date().isoformat() if not dates.dropna().empty else ""
                frequency = _infer_frequency(dates)
            elif suffix == ".xlsx" and file_id in used_counts:
                price_col = 2 if file_id.endswith("EU_IG_.xlsx") else 1
                parsed = _extract_workbook_price(zip_path, member, price_col)
                row_count = int(parsed.shape[0])
                col_count = 2
                date_col = "Date"
                start_date = parsed["date"].min().date().isoformat() if not parsed.empty else ""
                end_date = parsed["date"].max().date().isoformat() if not parsed.empty else ""
                frequency = _infer_frequency(parsed["date"])

            records.append(
                {
                    "file_id": file_id,
                    "container": "zip_member",
                    "file_type": suffix,
                    "row_count": row_count,
                    "column_count": col_count,
                    "date_column": date_col,
                    "start_date": start_date,
                    "end_date": end_date,
                    "detected_frequency": frequency,
                    "feature_count_used": int(used_counts.get(file_id, 0)),
                    "rejected_feature_count": int(rejected_counts.get(file_id, 0)),
                    "v2_status": status,
                    "v2_reason": reason,
                }
            )

    return pd.DataFrame(records).sort_values("file_id").reset_index(drop=True)
