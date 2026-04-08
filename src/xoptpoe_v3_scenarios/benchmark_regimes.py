"""Hybrid regime classifier for the locked robust 5Y benchmark scenario layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .state import BASE_STATE_VARIABLES


_Z_LOW = -0.35
_Z_HIGH = 0.35


@dataclass(frozen=True)
class HybridRegimeClassifier:
    """Interpretable hybrid regime classifier with external anchors."""

    variable_names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    nfci_monthly: pd.DataFrame
    nfci_loose_threshold: float
    nfci_tight_threshold: float
    recession_periods: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]

    def zscores(self, state_vector: np.ndarray) -> pd.Series:
        x = np.asarray(state_vector, dtype=float)
        z = (x - self.means) / self.stds
        return pd.Series(z, index=list(self.variable_names), dtype=float)

    def internal_scores(self, state_vector: np.ndarray) -> dict[str, float]:
        z = self.zscores(state_vector)
        growth = float(np.mean([-z["unemp_US"], -z["unemp_EA"], -z["unemp_JP"]]))
        inflation = float(np.mean([z["infl_US"], z["infl_EA"], z["infl_JP"], 0.5 * z["oil_wti"]]))
        market_stress = float(np.mean([z["vix"], z["ig_oas"], z["usd_broad"]]))
        rates = float(
            np.mean(
                [
                    z["short_rate_US"],
                    z["long_rate_US"],
                    z["short_rate_EA"],
                    z["long_rate_EA"],
                    z["short_rate_JP"],
                    z["long_rate_JP"],
                    z["us_real10y"],
                ]
            )
        )
        return {
            "growth_score": growth,
            "inflation_score": inflation,
            "market_stress_score": market_stress,
            "rates_score": rates,
        }

    def external_context(self, anchor_date: pd.Timestamp) -> dict[str, object]:
        month_end = pd.Timestamp(anchor_date).to_period("M").to_timestamp("M")
        row = self.nfci_monthly.loc[self.nfci_monthly["month_end"].eq(month_end)]
        if row.empty:
            row = self.nfci_monthly.loc[self.nfci_monthly["month_end"].le(month_end)].tail(1)
        if row.empty:
            raise ValueError(f"No NFCI value available on or before {month_end.date()}")
        nfci_value = float(row["nfci"].iloc[0])
        if nfci_value <= self.nfci_loose_threshold:
            nfci_bucket = "loose"
        elif nfci_value >= self.nfci_tight_threshold:
            nfci_bucket = "tight"
        else:
            nfci_bucket = "neutral"
        recession_overlay = "recession" if _in_recession(month_end, self.recession_periods) else "non-recession"
        return {
            "anchor_month_end": month_end,
            "nfci_value": nfci_value,
            "nfci_bucket": nfci_bucket,
            "recession_overlay": recession_overlay,
        }

    def classify(self, state_vector: np.ndarray, anchor_date: pd.Timestamp) -> dict[str, object]:
        scores = self.internal_scores(state_vector)
        external = self.external_context(anchor_date)
        growth_bucket = _bucket(scores["growth_score"], low_label="low", high_label="high")
        inflation_bucket = _bucket(scores["inflation_score"], low_label="low", high_label="high")
        market_stress_bucket = _bucket(scores["market_stress_score"], low_label="low", high_label="high")
        rates_bucket = _bucket(scores["rates_score"], low_label="easy", high_label="tight")
        regime_label = _label_regime(
            growth_bucket=growth_bucket,
            inflation_bucket=inflation_bucket,
            market_stress_bucket=market_stress_bucket,
            rates_bucket=rates_bucket,
            nfci_bucket=str(external["nfci_bucket"]),
            recession_overlay=str(external["recession_overlay"]),
        )
        return {
            **scores,
            **external,
            "growth_bucket": growth_bucket,
            "inflation_bucket": inflation_bucket,
            "market_stress_bucket": market_stress_bucket,
            "rates_bucket": rates_bucket,
            "regime_label": regime_label,
        }


def _bucket(value: float, *, low_label: str, high_label: str) -> str:
    if value <= _Z_LOW:
        return low_label
    if value >= _Z_HIGH:
        return high_label
    return "neutral"


def _label_regime(
    *,
    growth_bucket: str,
    inflation_bucket: str,
    market_stress_bucket: str,
    rates_bucket: str,
    nfci_bucket: str,
    recession_overlay: str,
) -> str:
    if recession_overlay == "recession":
        if nfci_bucket == "tight" or market_stress_bucket == "high":
            return "recessionary stress"
        return "disinflationary slowdown"
    if nfci_bucket == "tight" or market_stress_bucket == "high":
        if inflation_bucket == "high" or rates_bucket == "tight":
            return "higher-for-longer tightness"
        return "high-stress defensive"
    if growth_bucket == "high" and inflation_bucket in {"low", "neutral"} and market_stress_bucket != "high":
        return "soft landing"
    if growth_bucket == "high" and inflation_bucket == "high" and nfci_bucket == "loose" and market_stress_bucket != "high":
        return "risk-on reflation"
    if growth_bucket == "low" and inflation_bucket == "low":
        return "disinflationary slowdown"
    if growth_bucket == "high" and nfci_bucket == "loose" and market_stress_bucket == "low":
        return "risk-on growth"
    if inflation_bucket == "high" and rates_bucket == "tight":
        return "higher-for-longer tightness"
    return "mixed mid-cycle"


def _in_recession(month_end: pd.Timestamp, periods: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]) -> bool:
    for start, end in periods:
        if start <= month_end <= end:
            return True
    return False


def load_nfci_monthly(project_root: Path) -> pd.DataFrame:
    """Load NFCI and align it to month-end using the last observation in each month."""
    path = project_root / "NFCI (1).csv"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    if {"observation_date", "NFCI"} - set(raw.columns):
        raise ValueError("NFCI file must contain observation_date and NFCI columns")
    work = raw.loc[:, ["observation_date", "NFCI"]].copy()
    work["observation_date"] = pd.to_datetime(work["observation_date"])
    work["nfci"] = pd.to_numeric(work["NFCI"], errors="coerce")
    work = work.dropna(subset=["observation_date", "nfci"]).sort_values("observation_date").reset_index(drop=True)
    work["month_end"] = work["observation_date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = work.groupby("month_end", as_index=False).tail(1).reset_index(drop=True)
    monthly = monthly.loc[:, ["month_end", "observation_date", "nfci"]].sort_values("month_end").reset_index(drop=True)
    return monthly


def load_recession_periods(project_root: Path) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Parse monthly NBER peak/trough dates from the provided markdown note."""
    path = project_root / "Recessiondating.md"
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}),\s*(\d{4}-\d{2}-\d{2})")
    periods = []
    for peak_raw, trough_raw in pattern.findall(text):
        start = pd.Timestamp(peak_raw).to_period("M").to_timestamp("M")
        end = pd.Timestamp(trough_raw).to_period("M").to_timestamp("M")
        periods.append((start, end))
    if not periods:
        raise ValueError("No recession peak/trough pairs were parsed from Recessiondating.md")
    return tuple(periods)


def fit_hybrid_regime_classifier(feature_master_monthly: pd.DataFrame, project_root: Path) -> HybridRegimeClassifier:
    """Fit the hybrid regime classifier for the robust-only question framework."""
    monthly = (
        feature_master_monthly.groupby("month_end", as_index=False)
        .first()
        .sort_values("month_end")
        .reset_index(drop=True)
    )
    frame = monthly.loc[:, list(BASE_STATE_VARIABLES)].apply(pd.to_numeric, errors="coerce")
    means = frame.mean().to_numpy(dtype=float)
    stds = frame.std(ddof=1).fillna(0.0).to_numpy(dtype=float)
    stds = np.where(np.isfinite(stds) & (stds > 1e-8), stds, 1.0)

    nfci_monthly = load_nfci_monthly(project_root)
    nfci_loose = float(nfci_monthly["nfci"].quantile(1.0 / 3.0))
    nfci_tight = float(nfci_monthly["nfci"].quantile(2.0 / 3.0))
    recession_periods = load_recession_periods(project_root)
    return HybridRegimeClassifier(
        variable_names=tuple(BASE_STATE_VARIABLES),
        means=means.astype(float),
        stds=stds.astype(float),
        nfci_monthly=nfci_monthly,
        nfci_loose_threshold=nfci_loose,
        nfci_tight_threshold=nfci_tight,
        recession_periods=recession_periods,
    )


def hybrid_regime_manifest(classifier: HybridRegimeClassifier) -> pd.DataFrame:
    """Machine-readable regime framework description."""
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "component_type": "external_anchor",
            "component_name": "NFCI",
            "dimension": "financial_conditions",
            "source": "NFCI (1).csv",
            "aggregation_rule": "last observation within each calendar month, aligned to that month_end",
            "low_threshold": classifier.nfci_loose_threshold,
            "high_threshold": classifier.nfci_tight_threshold,
            "bucket_labels": "loose / neutral / tight",
            "notes": "Provided file is already monthly in practice; monthly alignment rule remains explicit and generic.",
        }
    )
    rows.append(
        {
            "component_type": "external_anchor",
            "component_name": "NBER_recession_overlay",
            "dimension": "recession_overlay",
            "source": "Recessiondating.md",
            "aggregation_rule": "month is recessionary when its month_end lies between NBER peak and trough month_end inclusive",
            "low_threshold": np.nan,
            "high_threshold": np.nan,
            "bucket_labels": "non-recession / recession",
            "notes": "Historical overlay only; not treated as predictive truth.",
        }
    )
    for variable_name, dimension, role in (
        ("unemp_US", "growth", "lower supports stronger growth"),
        ("unemp_EA", "growth", "lower supports stronger growth"),
        ("unemp_JP", "growth", "lower supports stronger growth"),
        ("infl_US", "inflation", "higher raises inflation score"),
        ("infl_EA", "inflation", "higher raises inflation score"),
        ("infl_JP", "inflation", "higher raises inflation score"),
        ("oil_wti", "inflation", "higher supports reflation pressure"),
        ("vix", "market_stress", "higher raises internal market stress"),
        ("ig_oas", "market_stress", "higher raises internal market stress"),
        ("usd_broad", "market_stress", "higher raises internal market stress"),
        ("short_rate_US", "rates", "higher tightens conditions"),
        ("long_rate_US", "rates", "higher tightens conditions"),
        ("short_rate_EA", "rates", "higher tightens conditions"),
        ("long_rate_EA", "rates", "higher tightens conditions"),
        ("short_rate_JP", "rates", "higher tightens conditions"),
        ("long_rate_JP", "rates", "higher tightens conditions"),
        ("us_real10y", "rates", "higher tightens real-rate conditions"),
    ):
        rows.append(
            {
                "component_type": "internal_dimension",
                "component_name": variable_name,
                "dimension": dimension,
                "source": "feature_master_monthly.parquet",
                "aggregation_rule": "monthly z-score against full v3 history",
                "low_threshold": _Z_LOW,
                "high_threshold": _Z_HIGH,
                "bucket_labels": "low / neutral / high, except rates = easy / neutral / tight",
                "notes": role,
            }
        )
    return pd.DataFrame(rows)
