"""Interpretable regime scoring and labeling for v3 scenario states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .state import BASE_STATE_VARIABLES


_BUCKET_LOW = -0.35
_BUCKET_HIGH = 0.35


@dataclass(frozen=True)
class RegimeClassifier:
    """Transparent scorecard over the manipulated macro state."""

    variable_names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray

    def zscores(self, state_vector: np.ndarray) -> pd.Series:
        x = np.asarray(state_vector, dtype=float)
        z = (x - self.means) / self.stds
        return pd.Series(z, index=list(self.variable_names), dtype=float)

    def dimension_scores(self, state_vector: np.ndarray) -> dict[str, float]:
        z = self.zscores(state_vector)
        growth = float(np.mean([-z["unemp_US"], -z["unemp_EA"], -z["unemp_JP"]]))
        inflation = float(np.mean([z["infl_US"], z["infl_EA"], z["infl_JP"], 0.5 * z["oil_wti"]]))
        stress = float(np.mean([z["vix"], z["ig_oas"], z["usd_broad"]]))
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
            "stress_score": stress,
            "rates_score": rates,
        }

    def classify(self, state_vector: np.ndarray) -> dict[str, object]:
        scores = self.dimension_scores(state_vector)
        growth_bucket = _bucket(scores["growth_score"], low_label="low", high_label="high")
        inflation_bucket = _bucket(scores["inflation_score"], low_label="low", high_label="high")
        stress_bucket = _bucket(scores["stress_score"], low_label="low", high_label="high")
        rates_bucket = _bucket(scores["rates_score"], low_label="easy", high_label="tight")
        label = _label_regime(growth_bucket, inflation_bucket, stress_bucket, rates_bucket)
        return {
            **scores,
            "growth_bucket": growth_bucket,
            "inflation_bucket": inflation_bucket,
            "stress_bucket": stress_bucket,
            "rates_bucket": rates_bucket,
            "regime_label": label,
        }


def _bucket(value: float, *, low_label: str, high_label: str) -> str:
    if value <= _BUCKET_LOW:
        return low_label
    if value >= _BUCKET_HIGH:
        return high_label
    return "neutral"


def _label_regime(growth: str, inflation: str, stress: str, rates: str) -> str:
    if stress == "high":
        if inflation == "high" and growth == "low":
            return "stagflation-like stress"
        if rates == "tight":
            return "high-stress defensive"
        return "risk-off stress"
    if growth == "high" and inflation in {"low", "neutral"} and stress == "low":
        return "soft landing"
    if growth == "high" and inflation == "high" and stress in {"low", "neutral"}:
        return "risk-on reflation"
    if growth == "low" and inflation == "low" and stress in {"low", "neutral"}:
        return "disinflationary slowdown"
    if growth in {"low", "neutral"} and inflation == "high" and rates == "tight":
        return "higher-for-longer tightness"
    if growth == "low" and inflation == "high":
        return "stagflation-like"
    if growth == "high" and rates == "easy" and stress == "low":
        return "risk-on growth"
    return "mixed mid-cycle"


def fit_regime_classifier(feature_master_monthly: pd.DataFrame) -> RegimeClassifier:
    """Fit historical z-score anchors for the first-pass state."""
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
    return RegimeClassifier(
        variable_names=tuple(BASE_STATE_VARIABLES),
        means=means.astype(float),
        stds=stds.astype(float),
    )


def regime_manifest_df() -> pd.DataFrame:
    """Machine-readable description of the regime taxonomy."""
    rows = []
    for variable_name, state_block, sign_role in (
        ("unemp_US", "growth", "lower supports stronger growth"),
        ("unemp_EA", "growth", "lower supports stronger growth"),
        ("unemp_JP", "growth", "lower supports stronger growth"),
        ("infl_US", "inflation", "higher raises inflation score"),
        ("infl_EA", "inflation", "higher raises inflation score"),
        ("infl_JP", "inflation", "higher raises inflation score"),
        ("oil_wti", "inflation", "higher supports reflation pressure"),
        ("vix", "stress", "higher raises stress"),
        ("ig_oas", "stress", "higher raises stress"),
        ("usd_broad", "stress", "higher raises stress/tightness"),
        ("short_rate_US", "rates", "higher tightens financial conditions"),
        ("long_rate_US", "rates", "higher tightens financial conditions"),
        ("short_rate_EA", "rates", "higher tightens financial conditions"),
        ("long_rate_EA", "rates", "higher tightens financial conditions"),
        ("short_rate_JP", "rates", "higher tightens financial conditions"),
        ("long_rate_JP", "rates", "higher tightens financial conditions"),
        ("us_real10y", "rates", "higher tightens real-rate conditions"),
    ):
        rows.append(
            {
                "variable_name": variable_name,
                "dimension": state_block,
                "scoring_role": sign_role,
                "low_threshold_z": _BUCKET_LOW,
                "high_threshold_z": _BUCKET_HIGH,
                "label_mapping": "low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates",
            }
        )
    return pd.DataFrame(rows)
