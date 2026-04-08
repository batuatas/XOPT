"""Experimental China and enhanced-indicator extension for XOPTPOE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from xoptpoe_modeling.eda import infer_feature_columns
from xoptpoe_modeling.io import write_csv, write_text
from xoptpoe_modeling.rolling_robustness import MODEL_SET, RollingConfig, run_feature_set_experiments


DATE_COLUMN_CANDIDATES: tuple[str, ...] = ("Date", "observation_date", "TIME_PERIOD")


@dataclass(frozen=True)
class ExperimentalConfig:
    """Config for the experimental extension run."""

    project_root: Path
    min_train_months: int = 96
    validation_months: int = 24
    test_months: int = 24
    step_months: int = 12
    random_state: int = 42


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    dayfirst_mask = text.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", na=False)
    month_mask = text.str.match(r"^\d{4}-\d{2}$", na=False)
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    other_mask = ~(dayfirst_mask | month_mask | iso_mask)

    if dayfirst_mask.any():
        parsed.loc[dayfirst_mask] = pd.to_datetime(text.loc[dayfirst_mask], errors="coerce", dayfirst=True)
    if month_mask.any():
        parsed.loc[month_mask] = pd.to_datetime(text.loc[month_mask], errors="coerce", format="%Y-%m")
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], errors="coerce", format="%Y-%m-%d")
    if other_mask.any():
        parsed.loc[other_mask] = pd.to_datetime(text.loc[other_mask], errors="coerce")
    return parsed


def _to_month_end(values: pd.Series) -> pd.Series:
    parsed = _parse_dates(values)
    return parsed.dt.to_period("M").dt.to_timestamp("M")


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


def _read_csv(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols)


def _read_zip_csv(zip_path: Path, member: str, *, usecols: list[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            return pd.read_csv(handle, usecols=usecols)


def _collapse_last_valid_by_month(df: pd.DataFrame, *, date_col: str, value_col: str) -> pd.DataFrame:
    work = df[[date_col, value_col]].copy()
    work[date_col] = _parse_dates(work[date_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    work["source_month_end"] = work[date_col].dt.to_period("M").dt.to_timestamp("M")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(columns=["source_month_end", "value"])
    out = (
        work.groupby("source_month_end", as_index=False)
        .agg(value=(value_col, "last"))
        .sort_values("source_month_end")
        .reset_index(drop=True)
    )
    return out


def _align_monthly_series(
    source_df: pd.DataFrame,
    *,
    row_months: pd.Index,
    lag_months: int,
) -> pd.Series:
    if source_df.empty:
        return pd.Series(index=row_months, dtype=float)

    target = pd.DataFrame({"month_end": pd.Index(sorted(row_months.unique()))})
    target["cutoff"] = target["month_end"] - pd.offsets.DateOffset(months=lag_months)

    aligned = pd.merge_asof(
        target.sort_values("cutoff"),
        source_df.sort_values("source_month_end"),
        left_on="cutoff",
        right_on="source_month_end",
        direction="backward",
    )
    return aligned.set_index("month_end")["value"].reindex(row_months)


def _read_oecd_li_monthly(source_path: Path) -> pd.DataFrame:
    df = _read_csv(
        source_path,
        usecols=["REF_AREA", "MEASURE", "ADJUSTMENT", "TIME_PERIOD", "OBS_VALUE"],
    )
    df["obs_date"] = _to_month_end(df["TIME_PERIOD"])
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    def select_series(area: str, preferences: list[tuple[str, str]]) -> pd.DataFrame:
        for measure, adjustment in preferences:
            chunk = df.loc[
                (df["REF_AREA"] == area)
                & (df["MEASURE"] == measure)
                & (df["ADJUSTMENT"] == adjustment),
                ["obs_date", "OBS_VALUE"],
            ].dropna()
            if not chunk.empty:
                return chunk.rename(columns={"OBS_VALUE": "value", "obs_date": "date"})
        return pd.DataFrame(columns=["date", "value"])

    return pd.DataFrame(
        {
            "source_month_end": pd.Index(sorted(df["obs_date"].dropna().unique())),
        }
    )


def _selected_candidate_inventory() -> pd.DataFrame:
    rows = [
        {
            "item_name": "OECD leading indicators top-level CSV",
            "status": "used_experimentally",
            "category": "oecd_leading",
            "reason": "Used for US/Japan CLI and EA20 business-confidence proxy in cross-country bundle.",
        },
        {
            "item_name": "Historic CAPE ratios top-level CSV",
            "status": "used_experimentally",
            "category": "valuation",
            "reason": "Used for US/Europe/Japan local CAPE and China CAPE in valuation bundles.",
        },
        {
            "item_name": "china_cli.csv",
            "status": "used_experimentally",
            "category": "china_macro",
            "reason": "Clean monthly China leading indicator; conservatively lagged one month for EM-targeted features.",
        },
        {
            "item_name": "china_pmi.csv",
            "status": "used_experimentally",
            "category": "china_macro",
            "reason": "Monthly China PMI usable with one-month lag; NBS manufacturing chosen as compact activity proxy.",
        },
        {
            "item_name": "china_cpi_data.csv",
            "status": "used_experimentally",
            "category": "china_macro",
            "reason": "Monthly headline CPI usable with one-month lag; subcomponents not adopted.",
        },
        {
            "item_name": "china_industrial_production.csv",
            "status": "used_experimentally",
            "category": "china_macro",
            "reason": "Monthly industrial-production growth usable with one-month lag.",
        },
        {
            "item_name": "China_dividend_yield.csv",
            "status": "used_experimentally",
            "category": "china_valuation",
            "reason": "Used as China-vs-EM dividend-yield relative valuation signal for EQ_EM.",
        },
        {
            "item_name": "China_price_to_earnings.csv",
            "status": "used_experimentally",
            "category": "china_valuation",
            "reason": "Used as China-vs-EM PE relative valuation signal for EQ_EM.",
        },
        {
            "item_name": "china & em price-to-book ratio.csv",
            "status": "used_experimentally",
            "category": "china_valuation",
            "reason": "Used as China-vs-EM PTB relative valuation signal for EQ_EM.",
        },
        {
            "item_name": "japan_pe_topix_index.csv",
            "status": "used_experimentally",
            "category": "japan_enrichment",
            "reason": "Used as Japan local valuation enrichment.",
        },
        {
            "item_name": "em_global_div_yield.csv",
            "status": "used_experimentally",
            "category": "global_valuation",
            "reason": "Used for EM-vs-global dividend-yield spread.",
        },
        {
            "item_name": "em_global_ptb.csv",
            "status": "used_experimentally",
            "category": "global_valuation",
            "reason": "Used for EM-vs-global price-to-book spread.",
        },
        {
            "item_name": "em_global_pte.csv",
            "status": "used_experimentally",
            "category": "global_valuation",
            "reason": "Used for EM-vs-global price-to-earnings spread.",
        },
        {
            "item_name": "BAMLH0A0HYM2 / BAMLH0A0HYM2EY",
            "status": "used_experimentally",
            "category": "credit_stress",
            "reason": "Used as additional US HY spread/yield stress features.",
        },
        {
            "item_name": "BAMLHE00EHYIOAS / BAMLHE00EHYIEY",
            "status": "used_experimentally",
            "category": "credit_stress",
            "reason": "Used as additional European HY spread/yield stress features.",
        },
        {
            "item_name": "china_gdp_cmi.csv",
            "status": "usable_but_not_used",
            "category": "china_macro",
            "reason": "Quarterly/sparse timing makes conservative alignment possible but fragile; omitted from compact bundles.",
        },
        {
            "item_name": "japan_macro_fiscal_socioeconomic.csv",
            "status": "usable_but_not_used",
            "category": "japan_enrichment",
            "reason": "Contains mixed-frequency overlaps with baseline and timing-ambiguous series; not clean enough for disciplined first pass.",
        },
        {
            "item_name": "JaPaN GDP.csv",
            "status": "usable_but_not_used",
            "category": "japan_enrichment",
            "reason": "Quarterly GDP requires extra conservative lag and added complexity; omitted.",
        },
        {
            "item_name": "JPNLOLITONOSTSAM.csv",
            "status": "usable_but_not_used",
            "category": "japan_enrichment",
            "reason": "Useful OECD Japan leading indicator, but redundant with the top-level OECD file already selected.",
        },
        {
            "item_name": "Historic-cape-ratios.csv inside zip",
            "status": "usable_but_not_used",
            "category": "valuation",
            "reason": "Duplicate of top-level CAPE upload; kept out to avoid duplicate source paths.",
        },
        {
            "item_name": "country_level_macro_data OECD BTS CSV",
            "status": "usable_but_not_used",
            "category": "oecd_leading",
            "reason": "Large and broad survey file with many overlapping measures; useful for later follow-up, not first compact pass.",
        },
        {
            "item_name": "china_labour_market.csv",
            "status": "too_fragile",
            "category": "china_macro",
            "reason": "Mostly annual and sparse; not suitable for monthly modeling extension.",
        },
        {
            "item_name": "china_fiscal_socioeconomic.csv",
            "status": "too_fragile",
            "category": "china_macro",
            "reason": "Mixed annual/monthly/governance content with heavy sparsity and unclear release timing.",
        },
        {
            "item_name": "japan_10y_bond_yield_cpi.csv",
            "status": "too_fragile",
            "category": "japan_enrichment",
            "reason": "Overlaps baseline Japan rate/CPI backbone and mixes daily/monthly coverage awkwardly.",
        },
        {
            "item_name": "japan_10y_bond_topix_dividends.csv",
            "status": "too_fragile",
            "category": "japan_enrichment",
            "reason": "File contents appear inconsistent with file name; not reliable enough for automated use.",
        },
        {
            "item_name": "China Price History_SSE_Composite.xlsx",
            "status": "too_fragile",
            "category": "china_sleeve_candidate",
            "reason": "Local index price history is not a USD adjusted-close investable sleeve target and does not support baseline-style target construction.",
        },
        {
            "item_name": "japan buyback yield.xlsx",
            "status": "too_fragile",
            "category": "japan_enrichment",
            "reason": "Workbook appears to be price-history style export rather than clean buyback-yield time series.",
        },
        {
            "item_name": "EU_IG_.xlsx",
            "status": "too_fragile",
            "category": "credit_stress",
            "reason": "Total-return price-history workbook, not a clean spread/yield indicator for compact experimental use.",
        },
    ]
    return pd.DataFrame(rows)


def _build_file_inventory(config: ExperimentalConfig) -> pd.DataFrame:
    external_root = config.project_root / "data" / "external" / "akif_candidates"
    zip_path = external_root / "country_level_macro_data.zip"
    records: list[dict[str, object]] = []

    top_level_files = [
        external_root / "OECD Leading Indicatorscsv.csv",
        external_root / "Historic-cape-ratios (1).csv",
    ]
    for path in top_level_files:
        df = pd.read_csv(path)
        date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in df.columns), None)
        dates = _parse_dates(df[date_col]) if date_col is not None else pd.Series(dtype="datetime64[ns]")
        records.append(
            {
                "file_id": str(path.relative_to(config.project_root)),
                "container": "top_level",
                "file_type": path.suffix.lower(),
                "row_count": int(df.shape[0]),
                "column_count": int(df.shape[1]),
                "date_column": date_col or "",
                "start_date": dates.min().date().isoformat() if not dates.dropna().empty else "",
                "end_date": dates.max().date().isoformat() if not dates.dropna().empty else "",
                "detected_frequency": _infer_frequency(dates),
                "columns_preview": ", ".join(df.columns[:6].tolist()),
            }
        )

    with zipfile.ZipFile(zip_path) as zf:
        for member in sorted(zf.namelist()):
            if member.endswith("/") or member.startswith("__MACOSX/") or member.endswith(".DS_Store"):
                continue
            suffix = Path(member).suffix.lower()
            if suffix == ".csv":
                with zf.open(member) as handle:
                    df = pd.read_csv(handle)
                date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in df.columns), None)
                dates = _parse_dates(df[date_col]) if date_col is not None else pd.Series(dtype="datetime64[ns]")
                records.append(
                    {
                        "file_id": member,
                        "container": "zip_member",
                        "file_type": suffix,
                        "row_count": int(df.shape[0]),
                        "column_count": int(df.shape[1]),
                        "date_column": date_col or "",
                        "start_date": dates.min().date().isoformat() if not dates.dropna().empty else "",
                        "end_date": dates.max().date().isoformat() if not dates.dropna().empty else "",
                        "detected_frequency": _infer_frequency(dates),
                        "columns_preview": ", ".join(df.columns[:6].tolist()),
                    }
                )
            else:
                records.append(
                    {
                        "file_id": member,
                        "container": "zip_member",
                        "file_type": suffix,
                        "row_count": np.nan,
                        "column_count": np.nan,
                        "date_column": "",
                        "start_date": "",
                        "end_date": "",
                        "detected_frequency": "not_parsed",
                        "columns_preview": "binary workbook or non-csv asset",
                    }
                )

    return pd.DataFrame(records).sort_values("file_id").reset_index(drop=True)


def _load_series_from_top_level(
    path: Path,
    *,
    date_col: str,
    value_col: str,
    lag_months: int,
    row_months: pd.Index,
) -> pd.Series:
    df = _read_csv(path, usecols=[date_col, value_col]).rename(columns={date_col: "date", value_col: "value"})
    collapsed = _collapse_last_valid_by_month(df, date_col="date", value_col="value")
    return _align_monthly_series(collapsed, row_months=row_months, lag_months=lag_months)


def _load_series_from_zip(
    zip_path: Path,
    member: str,
    *,
    date_col: str,
    value_col: str,
    lag_months: int,
    row_months: pd.Index,
) -> pd.Series:
    df = _read_zip_csv(zip_path, member, usecols=[date_col, value_col]).rename(
        columns={date_col: "date", value_col: "value"}
    )
    collapsed = _collapse_last_valid_by_month(df, date_col="date", value_col="value")
    return _align_monthly_series(collapsed, row_months=row_months, lag_months=lag_months)


def _load_spread_series_from_zip(
    zip_path: Path,
    member: str,
    *,
    value_col: str,
    alt_left: str,
    alt_right: str,
    row_months: pd.Index,
) -> pd.Series:
    df = _read_zip_csv(zip_path, member, usecols=["Date", value_col, alt_left, alt_right]).copy()
    spread = pd.to_numeric(df[value_col], errors="coerce")
    left = pd.to_numeric(df[alt_left], errors="coerce")
    right = pd.to_numeric(df[alt_right], errors="coerce")
    value = spread.where(spread.notna(), left - right)
    work = pd.DataFrame({"date": df["Date"], "value": value})
    collapsed = _collapse_last_valid_by_month(work, date_col="date", value_col="value")
    return _align_monthly_series(collapsed, row_months=row_months, lag_months=0)


def _load_oecd_series(source_path: Path, row_months: pd.Index) -> dict[str, pd.Series]:
    df = _read_csv(
        source_path,
        usecols=["REF_AREA", "MEASURE", "ADJUSTMENT", "TIME_PERIOD", "OBS_VALUE"],
    )
    df["obs_date"] = _to_month_end(df["TIME_PERIOD"])
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    def select(area: str, preferences: list[tuple[str, str]]) -> pd.Series:
        for measure, adjustment in preferences:
            chunk = df.loc[
                (df["REF_AREA"] == area)
                & (df["MEASURE"] == measure)
                & (df["ADJUSTMENT"] == adjustment),
                ["obs_date", "OBS_VALUE"],
            ].dropna()
            if chunk.empty:
                continue
            collapsed = (
                chunk.rename(columns={"obs_date": "source_month_end", "OBS_VALUE": "value"})
                .sort_values("source_month_end")
                .reset_index(drop=True)
            )
            return _align_monthly_series(collapsed, row_months=row_months, lag_months=1)
        return pd.Series(index=row_months, dtype=float)

    return {
        "state_oecd_li_us": select("USA", [("LI", "AA"), ("LI", "TR"), ("LI", "NOR")]),
        "state_oecd_bc_ea": select("EA20", [("BCICP", "AA"), ("CCICP", "AA")]),
        "state_oecd_li_jp": select("JPN", [("LI", "AA"), ("LI", "TR"), ("LI", "NOR")]),
    }


def _build_experimental_monthly_state(config: ExperimentalConfig, row_months: pd.Index) -> pd.DataFrame:
    project_root = config.project_root
    external_root = project_root / "data" / "external" / "akif_candidates"
    zip_path = external_root / "country_level_macro_data.zip"

    state = pd.DataFrame({"month_end": row_months}).drop_duplicates().sort_values("month_end").reset_index(drop=True)
    state = state.set_index("month_end")

    oecd_state = _load_oecd_series(external_root / "OECD Leading Indicatorscsv.csv", row_months=state.index)
    for col, series in oecd_state.items():
        state[col] = series

    state["state_china_cli"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/china_cli.csv",
        date_col="Date",
        value_col="COMPOSITE LEADING IN",
        lag_months=1,
        row_months=state.index,
    )
    state["state_china_pmi_nbs_mfg"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/china_pmi.csv",
        date_col="Date",
        value_col="NBS manufacturing",
        lag_months=1,
        row_months=state.index,
    )
    state["state_china_cpi_headline"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/china_cpi_data.csv",
        date_col="Date",
        value_col="Headline",
        lag_months=1,
        row_months=state.index,
    )
    state["state_china_ip_growth"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/china_industrial_production.csv",
        date_col="Date",
        value_col="INDUSTRIAL PROD.: GR",
        lag_months=1,
        row_months=state.index,
    )

    state["state_china_div_yield_spread"] = _load_spread_series_from_zip(
        zip_path,
        "country_level_macro_data/China_dividend_yield.csv",
        value_col="Spread (RHS)",
        alt_left="China",
        alt_right="Emerging markets",
        row_months=state.index,
    )
    state["state_china_pe_spread"] = _load_spread_series_from_zip(
        zip_path,
        "country_level_macro_data/China_price_to_earnings.csv",
        value_col="Spread (RHS)",
        alt_left="China",
        alt_right="Emerging markets",
        row_months=state.index,
    )
    state["state_china_ptb_spread"] = _load_spread_series_from_zip(
        zip_path,
        "country_level_macro_data/china & em price-to-book ratio.csv",
        value_col="Spread (RHS)",
        alt_left="China",
        alt_right="Emerging markets",
        row_months=state.index,
    )

    cape_path = external_root / "Historic-cape-ratios (1).csv"
    state["state_cape_china"] = _load_series_from_top_level(
        cape_path,
        date_col="Date",
        value_col="China",
        lag_months=0,
        row_months=state.index,
    )
    state["state_cape_europe"] = _load_series_from_top_level(
        cape_path,
        date_col="Date",
        value_col="Europe",
        lag_months=0,
        row_months=state.index,
    )
    state["state_cape_japan"] = _load_series_from_top_level(
        cape_path,
        date_col="Date",
        value_col="Japan",
        lag_months=0,
        row_months=state.index,
    )
    state["state_cape_usa"] = _load_series_from_top_level(
        cape_path,
        date_col="Date",
        value_col="USA",
        lag_months=0,
        row_months=state.index,
    )

    state["state_japan_pe"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/japan_pe_topix_index.csv",
        date_col="Date",
        value_col="Price to earnings",
        lag_months=0,
        row_months=state.index,
    )

    state["state_em_div_yield_spread"] = _load_spread_series_from_zip(
        zip_path,
        "country_level_macro_data/em_global_div_yield.csv",
        value_col="Spread (RHS)",
        alt_left="Emerging markets",
        alt_right="Global index",
        row_months=state.index,
    )
    state["state_em_ptb_spread"] = _load_spread_series_from_zip(
        zip_path,
        "country_level_macro_data/em_global_ptb.csv",
        value_col="Spread (RHS)",
        alt_left="Emerging markets",
        alt_right="Global index",
        row_months=state.index,
    )
    state["state_em_pe_spread"] = _load_spread_series_from_zip(
        zip_path,
        "country_level_macro_data/em_global_pte.csv",
        value_col="Spread (RHS)",
        alt_left="Emerging markets",
        alt_right="Global index",
        row_months=state.index,
    )

    state["state_us_hy_oas"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/BAMLH0A0HYM2.csv",
        date_col="observation_date",
        value_col="BAMLH0A0HYM2",
        lag_months=0,
        row_months=state.index,
    )
    state["state_us_hy_ey"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/BAMLH0A0HYM2EY.csv",
        date_col="observation_date",
        value_col="BAMLH0A0HYM2EY",
        lag_months=0,
        row_months=state.index,
    )
    state["state_eu_hy_oas"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/BAMLHE00EHYIOAS.csv",
        date_col="observation_date",
        value_col="BAMLHE00EHYIOAS",
        lag_months=0,
        row_months=state.index,
    )
    state["state_eu_hy_ey"] = _load_series_from_zip(
        zip_path,
        "country_level_macro_data/BAMLHE00EHYIEY.csv",
        date_col="observation_date",
        value_col="BAMLHE00EHYIEY",
        lag_months=0,
        row_months=state.index,
    )

    return state.reset_index()


def _map_local_feature(panel: pd.DataFrame, *, us_col: str, ea_col: str, jp_col: str) -> pd.Series:
    return pd.Series(
        np.select(
            [
                panel["geo_block_local"].eq("US"),
                panel["geo_block_local"].eq("EURO_AREA"),
                panel["geo_block_local"].eq("JAPAN"),
            ],
            [
                panel[us_col],
                panel[ea_col],
                panel[jp_col],
            ],
            default=0.0,
        ),
        index=panel.index,
        dtype=float,
    )


def _build_experimental_panel(baseline_panel: pd.DataFrame, monthly_state: pd.DataFrame) -> pd.DataFrame:
    panel = baseline_panel.merge(monthly_state, on="month_end", how="left", validate="many_to_one")

    eq_em_mask = panel["sleeve_id"].eq("EQ_EM")
    panel["exp_china_cli_eq_em"] = np.where(eq_em_mask, panel["state_china_cli"], 0.0)
    panel["exp_china_pmi_nbs_mfg_eq_em"] = np.where(eq_em_mask, panel["state_china_pmi_nbs_mfg"], 0.0)
    panel["exp_china_cpi_headline_eq_em"] = np.where(eq_em_mask, panel["state_china_cpi_headline"], 0.0)
    panel["exp_china_ip_growth_eq_em"] = np.where(eq_em_mask, panel["state_china_ip_growth"], 0.0)
    panel["exp_china_div_yield_spread_eq_em"] = np.where(eq_em_mask, panel["state_china_div_yield_spread"], 0.0)
    panel["exp_china_pe_spread_eq_em"] = np.where(eq_em_mask, panel["state_china_pe_spread"], 0.0)
    panel["exp_china_ptb_spread_eq_em"] = np.where(eq_em_mask, panel["state_china_ptb_spread"], 0.0)
    panel["exp_china_cape_eq_em"] = np.where(eq_em_mask, panel["state_cape_china"], 0.0)

    panel["exp_local_leading_proxy"] = _map_local_feature(
        panel,
        us_col="state_oecd_li_us",
        ea_col="state_oecd_bc_ea",
        jp_col="state_oecd_li_jp",
    )
    panel["exp_local_cape"] = _map_local_feature(
        panel,
        us_col="state_cape_usa",
        ea_col="state_cape_europe",
        jp_col="state_cape_japan",
    )
    panel["exp_japan_pe_local"] = np.where(panel["geo_block_local"].eq("JAPAN"), panel["state_japan_pe"], 0.0)

    panel["exp_em_div_yield_spread_global"] = panel["state_em_div_yield_spread"]
    panel["exp_em_ptb_spread_global"] = panel["state_em_ptb_spread"]
    panel["exp_em_pe_spread_global"] = panel["state_em_pe_spread"]
    panel["exp_us_hy_oas_global"] = panel["state_us_hy_oas"]
    panel["exp_us_hy_ey_global"] = panel["state_us_hy_ey"]
    panel["exp_eu_hy_oas_global"] = panel["state_eu_hy_oas"]
    panel["exp_eu_hy_ey_global"] = panel["state_eu_hy_ey"]

    drop_cols = [c for c in panel.columns if c.startswith("state_")]
    return panel.drop(columns=drop_cols)


def _bundle_definitions() -> dict[str, list[str]]:
    bundles = {
        "baseline_only": [],
        "baseline_plus_china_macro": [
            "exp_china_cli_eq_em",
            "exp_china_pmi_nbs_mfg_eq_em",
            "exp_china_cpi_headline_eq_em",
            "exp_china_ip_growth_eq_em",
        ],
        "baseline_plus_china_macro_valuation": [
            "exp_china_cli_eq_em",
            "exp_china_pmi_nbs_mfg_eq_em",
            "exp_china_cpi_headline_eq_em",
            "exp_china_ip_growth_eq_em",
            "exp_china_div_yield_spread_eq_em",
            "exp_china_pe_spread_eq_em",
            "exp_china_ptb_spread_eq_em",
            "exp_china_cape_eq_em",
        ],
        "baseline_plus_oecd_cape": [
            "exp_local_leading_proxy",
            "exp_local_cape",
            "exp_japan_pe_local",
        ],
        "baseline_plus_global_valuation_credit": [
            "exp_em_div_yield_spread_global",
            "exp_em_ptb_spread_global",
            "exp_em_pe_spread_global",
            "exp_us_hy_oas_global",
            "exp_us_hy_ey_global",
            "exp_eu_hy_oas_global",
            "exp_eu_hy_ey_global",
        ],
    }
    bundles["baseline_plus_combined_experimental"] = sorted(
        set(
            bundles["baseline_plus_china_macro_valuation"]
            + bundles["baseline_plus_oecd_cape"]
            + bundles["baseline_plus_global_valuation_credit"]
        )
    )
    return bundles


def _feature_set_mapping(panel: pd.DataFrame) -> dict[str, list[str]]:
    baseline_features = sorted([c for c in infer_feature_columns(panel) if not c.startswith("exp_")])
    bundles = _bundle_definitions()
    feature_sets: dict[str, list[str]] = {}
    for bundle_name, added in bundles.items():
        if bundle_name == "baseline_only":
            feature_sets[bundle_name] = baseline_features
        else:
            feature_sets[bundle_name] = sorted(set(baseline_features + added))
    return feature_sets


def _summarize_bundle_features(
    panel: pd.DataFrame,
    *,
    bundle_defs: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    late_panel = panel.loc[panel["month_end"] >= pd.Timestamp("2015-01-31")].copy()
    descriptions = {
        "baseline_only": "Frozen baseline feature set only.",
        "baseline_plus_china_macro": "Adds China activity/inflation features targeted to EQ_EM only.",
        "baseline_plus_china_macro_valuation": "Adds China macro plus China-vs-EM valuation features for EQ_EM.",
        "baseline_plus_oecd_cape": "Adds local OECD leading proxy, local CAPE, and Japan PE enrichment.",
        "baseline_plus_global_valuation_credit": "Adds EM/global valuation spreads and US/EU HY stress features.",
        "baseline_plus_combined_experimental": "Union of all experimental feature additions.",
    }
    scope = {
        "baseline_only": "all baseline sleeves",
        "baseline_plus_china_macro": "EQ_EM targeted",
        "baseline_plus_china_macro_valuation": "EQ_EM targeted",
        "baseline_plus_oecd_cape": "US/EA/JP local-block sleeves",
        "baseline_plus_global_valuation_credit": "all sleeves",
        "baseline_plus_combined_experimental": "mixed global + targeted",
    }

    for bundle_name, added_cols in bundle_defs.items():
        if not added_cols:
            rows.append(
                {
                    "bundle_name": bundle_name,
                    "added_feature_count": 0,
                    "first_usable_month": "",
                    "late_sample_missing_share_mean": 0.0,
                    "scope": scope[bundle_name],
                    "description": descriptions[bundle_name],
                    "added_features": "",
                }
            )
            continue

        usable_months = []
        missing_shares = []
        for col in added_cols:
            non_null = panel.loc[panel[col].notna(), "month_end"]
            usable_months.append(non_null.min() if not non_null.empty else pd.NaT)
            missing_shares.append(float(late_panel[col].isna().mean()))
        first_usable = min([d for d in usable_months if pd.notna(d)], default=pd.NaT)
        rows.append(
            {
                "bundle_name": bundle_name,
                "added_feature_count": len(added_cols),
                "first_usable_month": first_usable.date().isoformat() if pd.notna(first_usable) else "",
                "late_sample_missing_share_mean": float(np.mean(missing_shares)),
                "scope": scope[bundle_name],
                "description": descriptions[bundle_name],
                "added_features": ", ".join(added_cols),
            }
        )

    return pd.DataFrame(rows).sort_values("bundle_name").reset_index(drop=True)


def _summarize_model_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = fold_metrics.loc[
        fold_metrics["feature_set"] == "baseline_only",
        ["fold_id", "model", "validation_rmse", "test_rmse", "test_spearman_ic", "test_top3_lift"],
    ].rename(
        columns={
            "validation_rmse": "baseline_validation_rmse",
            "test_rmse": "baseline_test_rmse",
            "test_spearman_ic": "baseline_test_spearman_ic",
            "test_top3_lift": "baseline_test_top3_lift",
        }
    )
    work = fold_metrics.merge(baseline, on=["fold_id", "model"], how="left", validate="many_to_one")
    work["validation_rmse_delta_vs_baseline"] = work["validation_rmse"] - work["baseline_validation_rmse"]
    work["test_rmse_delta_vs_baseline"] = work["test_rmse"] - work["baseline_test_rmse"]
    work["test_spearman_delta_vs_baseline"] = work["test_spearman_ic"] - work["baseline_test_spearman_ic"]
    work["test_top3_lift_delta_vs_baseline"] = work["test_top3_lift"] - work["baseline_test_top3_lift"]
    work["beats_baseline_test_rmse"] = work["test_rmse_delta_vs_baseline"] < 0

    grouped = (
        work.groupby(["feature_set", "model"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            feature_count=("feature_count", "max"),
            validation_rmse_mean=("validation_rmse", "mean"),
            validation_rmse_std=("validation_rmse", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_oos_r2_mean=("test_oos_r2", "mean"),
            test_corr_mean=("test_corr", "mean"),
            test_spearman_ic_mean=("test_spearman_ic", "mean"),
            test_top3_lift_mean=("test_top3_lift", "mean"),
            validation_rmse_delta_vs_baseline_mean=("validation_rmse_delta_vs_baseline", "mean"),
            test_rmse_delta_vs_baseline_mean=("test_rmse_delta_vs_baseline", "mean"),
            test_spearman_delta_vs_baseline_mean=("test_spearman_delta_vs_baseline", "mean"),
            test_top3_lift_delta_vs_baseline_mean=("test_top3_lift_delta_vs_baseline", "mean"),
            beat_baseline_test_rmse_fold_share=("beats_baseline_test_rmse", "mean"),
        )
        .sort_values(["model", "test_rmse_mean", "feature_set"])
        .reset_index(drop=True)
    )
    grouped.loc[grouped["feature_set"] == "baseline_only", "validation_rmse_delta_vs_baseline_mean"] = 0.0
    grouped.loc[grouped["feature_set"] == "baseline_only", "test_rmse_delta_vs_baseline_mean"] = 0.0
    grouped.loc[grouped["feature_set"] == "baseline_only", "test_spearman_delta_vs_baseline_mean"] = 0.0
    grouped.loc[grouped["feature_set"] == "baseline_only", "test_top3_lift_delta_vs_baseline_mean"] = 0.0
    grouped.loc[grouped["feature_set"] == "baseline_only", "beat_baseline_test_rmse_fold_share"] = np.nan
    return grouped


def _render_inventory_md(inventory: pd.DataFrame) -> str:
    lines = ["# Experimental Data Inventory", ""]
    lines.append("## Files Inspected")
    for row in inventory.itertuples(index=False):
        lines.append(
            f"- `{row.file_id}`: type={row.file_type}, rows={row.row_count}, cols={row.column_count}, "
            f"date_col=`{row.date_column}`, freq={row.detected_frequency}, "
            f"coverage={row.start_date or 'n/a'} to {row.end_date or 'n/a'}"
        )
    lines.append("")
    return "\n".join(lines)


def _render_candidate_assessment_md(assessment: pd.DataFrame) -> str:
    lines = ["# Experimental Candidate Assessment", ""]
    for status in ["used_experimentally", "usable_but_not_used", "too_fragile"]:
        lines.append(f"## {status.replace('_', ' ').title()}")
        subset = assessment.loc[assessment["status"] == status].copy()
        for row in subset.itertuples(index=False):
            lines.append(f"- {row.item_name} [{row.category}]: {row.reason}")
        lines.append("")
    return "\n".join(lines)


def _render_recommendation_md(
    *,
    assessment: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> str:
    baseline_rows = model_summary.loc[model_summary["feature_set"] == "baseline_only"].copy()
    experimental_rows = model_summary.loc[model_summary["feature_set"] != "baseline_only"].copy()
    china_rows = experimental_rows.loc[
        experimental_rows["feature_set"].isin(
            ["baseline_plus_china_macro", "baseline_plus_china_macro_valuation", "baseline_plus_combined_experimental"]
        )
    ].copy()
    best_china = china_rows.sort_values("test_rmse_delta_vs_baseline_mean").iloc[0]

    usable_files = assessment.loc[assessment["status"] == "used_experimentally", "item_name"].tolist()
    keepable = experimental_rows.loc[
        (experimental_rows["test_rmse_delta_vs_baseline_mean"] < 0)
        & (experimental_rows["beat_baseline_test_rmse_fold_share"] >= 0.50)
    ].copy()

    if keepable.empty:
        best_keep_text = "No experimental bundle beat the same-model baseline often enough to justify promotion."
        single_best = experimental_rows.sort_values(
            ["validation_rmse_mean", "test_rmse_mean", "test_spearman_ic_mean"],
            ascending=[True, True, False],
        ).iloc[0]
    else:
        row = keepable.sort_values("test_rmse_delta_vs_baseline_mean").iloc[0]
        best_keep_text = (
            f"The strongest experimental candidate was `{row['feature_set']}` with `{row['model']}`, "
            f"but it still needs to stay experimental until confirmed by a broader benchmark process."
        )
        single_best = row

    china_material = bool(
        (china_rows["test_rmse_delta_vs_baseline_mean"] < -0.0005).any()
        and (china_rows["beat_baseline_test_rmse_fold_share"] >= 0.50).any()
    )
    china_answer = (
        "China-related additions showed material, repeatable improvement."
        if china_material
        else "China-related additions did not show material, repeatable improvement in this compact rolling test."
    )

    lines = ["# Experimental Recommendation", ""]
    lines.append("## Direct Answers")
    lines.append(f"1. Genuinely usable uploaded files: {', '.join(usable_files)}.")
    lines.append(f"2. Experimentally worth keeping: {best_keep_text}")
    lines.append(f"3. China-related additions: {china_answer}")
    lines.append(
        "4. China recommendation: use only as an experimental macro/valuation block for now; "
        "do not add a standalone China sleeve from the uploaded data."
    )
    lines.append(
        f"5. Single best experimental extension to carry forward: `{single_best['feature_set']}` "
        f"with `{single_best['model']}` as the most defensible next test."
    )
    lines.append(
        "6. Main-design graduation: none of these changes should enter the locked main design yet; "
        "they should remain experimental until they beat the frozen baseline more consistently."
    )
    lines.append("")
    lines.append("## China Sleeve Assessment")
    lines.append(
        "- The uploaded SSE Composite workbook is not a USD adjusted-close investable sleeve target, "
        "so it does not support a baseline-consistent China sleeve on its own."
    )
    lines.append(
        "- A standalone China sleeve is technically feasible only as a separate ETF-target experiment "
        "using the existing swappable target adapter, not from these uploaded files alone."
    )
    lines.append("")
    return "\n".join(lines)


def run_experimental_extension(config: ExperimentalConfig) -> dict[str, Path]:
    """Run the full experimental extension workflow."""
    project_root = config.project_root
    reports_dir = project_root / "reports"
    experimental_dir = project_root / "data" / "experimental"
    reports_dir.mkdir(parents=True, exist_ok=True)
    experimental_dir.mkdir(parents=True, exist_ok=True)

    baseline_panel = pd.read_csv(
        project_root / "data" / "modeling" / "modeling_panel_filtered.csv",
        parse_dates=["month_end"],
    ).sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)
    row_months = pd.Index(sorted(baseline_panel["month_end"].unique()))

    file_inventory = _build_file_inventory(config)
    candidate_assessment = _selected_candidate_inventory()
    monthly_state = _build_experimental_monthly_state(config, row_months)
    experimental_panel = _build_experimental_panel(baseline_panel, monthly_state)

    bundle_defs = _bundle_definitions()
    feature_sets = _feature_set_mapping(experimental_panel)
    rolling_cfg = RollingConfig(
        min_train_months=config.min_train_months,
        validation_months=config.validation_months,
        test_months=config.test_months,
        step_months=config.step_months,
        random_state=config.random_state,
    )
    fold_metrics, test_predictions, fold_manifest, _ = run_feature_set_experiments(
        experimental_panel,
        config=rolling_cfg,
        model_names=MODEL_SET,
        feature_sets=feature_sets,
    )

    feature_comparison = _summarize_bundle_features(experimental_panel, bundle_defs=bundle_defs)
    model_comparison = _summarize_model_comparison(fold_metrics)

    inventory_md = _render_inventory_md(file_inventory)
    assessment_md = _render_candidate_assessment_md(candidate_assessment)
    recommendation_md = _render_recommendation_md(
        assessment=candidate_assessment,
        model_summary=model_comparison,
    )

    outputs = {
        "experimental_panel": experimental_dir / "experimental_panel.csv",
        "experimental_monthly_state": experimental_dir / "experimental_monthly_state.csv",
        "experimental_file_inventory": experimental_dir / "experimental_file_inventory.csv",
        "experimental_model_fold_metrics": experimental_dir / "experimental_model_fold_metrics.csv",
        "experimental_test_predictions": experimental_dir / "experimental_test_predictions.csv",
        "experimental_fold_manifest": experimental_dir / "experimental_fold_manifest.csv",
        "experimental_data_inventory_report": reports_dir / "experimental_data_inventory.md",
        "experimental_candidate_assessment_report": reports_dir / "experimental_candidate_assessment.md",
        "experimental_feature_comparison_report": reports_dir / "experimental_feature_comparison.csv",
        "experimental_model_comparison_report": reports_dir / "experimental_model_comparison.csv",
        "experimental_recommendation_report": reports_dir / "experimental_recommendation.md",
    }

    write_csv(experimental_panel, outputs["experimental_panel"])
    write_csv(monthly_state, outputs["experimental_monthly_state"])
    write_csv(file_inventory, outputs["experimental_file_inventory"])
    write_csv(fold_metrics, outputs["experimental_model_fold_metrics"])
    write_csv(test_predictions, outputs["experimental_test_predictions"])
    write_csv(fold_manifest, outputs["experimental_fold_manifest"])
    write_text(inventory_md, outputs["experimental_data_inventory_report"])
    write_text(assessment_md, outputs["experimental_candidate_assessment_report"])
    write_csv(feature_comparison, outputs["experimental_feature_comparison_report"])
    write_csv(model_comparison, outputs["experimental_model_comparison_report"])
    write_text(recommendation_md, outputs["experimental_recommendation_report"])

    return outputs
