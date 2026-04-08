"""
regime.py

Regime classification and labelling for generated macro states.

Labels each generated macro state m with an interpretable regime name
based on:
  - NFCI (financial conditions stress proxy via ig_oas + vix)
  - Growth regime (via short_rate_US, unemp_US)
  - Inflation regime (via infl_US)
  - Rates regime (via long_rate_US, us_real10y)

Regime labels (mutually exclusive, in priority order):
  1. recession_stress         — NFCI proxy high + growth falling
  2. high_stress              — NFCI proxy high (stress without confirmed recession)
  3. higher_for_longer        — high real rates + elevated inflation + moderate stress
  4. inflationary_expansion   — high inflation + low stress + low unemp
  5. soft_landing             — inflation falling, rates falling, low stress
  6. disinflationary_slowdown — low growth + low inflation + moderate stress
  7. risk_off_defensive       — vix high, ig_oas high, below-median growth
  8. mid_cycle_neutral        — default if none of the above

Also computes a 3-regime simplification:
  - expansion (labels 4 + 5)
  - stress (labels 1 + 2 + 7)
  - rate_transition (labels 3 + 6)
  - neutral (label 8)

NFCI data is used to calibrate historical thresholds for ig_oas and vix
(the two macro state variables most correlated with NFCI).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS

# Regime label constants
REGIME_RECESSION_STRESS = "recession_stress"
REGIME_HIGH_STRESS = "high_stress"
REGIME_HIGHER_FOR_LONGER = "higher_for_longer"
REGIME_INFLATIONARY_EXPANSION = "inflationary_expansion"
REGIME_SOFT_LANDING = "soft_landing"
REGIME_DISINFL_SLOWDOWN = "disinflationary_slowdown"
REGIME_RISK_OFF = "risk_off_defensive"
REGIME_NEUTRAL = "mid_cycle_neutral"

SIMPLE_REGIME_MAP = {
    REGIME_RECESSION_STRESS: "stress",
    REGIME_HIGH_STRESS: "stress",
    REGIME_RISK_OFF: "stress",
    REGIME_HIGHER_FOR_LONGER: "rate_transition",
    REGIME_DISINFL_SLOWDOWN: "rate_transition",
    REGIME_INFLATIONARY_EXPANSION: "expansion",
    REGIME_SOFT_LANDING: "expansion",
    REGIME_NEUTRAL: "neutral",
}

# NBER recession dates (peak -> trough)
NBER_RECESSIONS = [
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


def load_nfci(nfci_path: str | Path) -> pd.DataFrame:
    """
    Load and resample NFCI data to monthly frequency.

    NFCI CSV: columns [observation_date, NFCI], weekly frequency.
    Returns monthly end-of-month NFCI.
    """
    nfci = pd.read_csv(nfci_path)
    nfci.columns = [c.strip() for c in nfci.columns]
    nfci["observation_date"] = pd.to_datetime(nfci["observation_date"])
    nfci["NFCI"] = pd.to_numeric(nfci["NFCI"], errors="coerce")
    nfci = nfci.dropna(subset=["NFCI"])
    nfci = nfci.set_index("observation_date").sort_index()
    # Resample to monthly end-of-month
    nfci_monthly = nfci["NFCI"].resample("ME").last().dropna()
    return pd.DataFrame({"month_end": nfci_monthly.index, "NFCI": nfci_monthly.values})


def compute_regime_thresholds(
    feature_master: pd.DataFrame,
    nfci_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Compute historical percentile thresholds for regime classification.

    Uses training-period data (pre-2016) to set thresholds.
    """
    fm = feature_master.drop_duplicates(subset="month_end").copy()
    fm["month_end"] = pd.to_datetime(fm["month_end"])
    train = fm[fm["month_end"] < pd.Timestamp("2016-01-01")]

    mac_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}

    thresholds: dict[str, float] = {}

    for col in MACRO_STATE_COLS:
        vals = train[col].dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            thresholds[f"{col}_p25"] = float(np.percentile(vals, 25))
            thresholds[f"{col}_p50"] = float(np.percentile(vals, 50))
            thresholds[f"{col}_p75"] = float(np.percentile(vals, 75))
            thresholds[f"{col}_p90"] = float(np.percentile(vals, 90))

    # NFCI-based thresholds (if available)
    if nfci_df is not None:
        nfci_train = nfci_df[pd.to_datetime(nfci_df["month_end"]) < pd.Timestamp("2016-01-01")]
        nfci_vals = nfci_train["NFCI"].dropna().to_numpy(dtype=float)
        if len(nfci_vals) > 0:
            thresholds["nfci_p75"] = float(np.percentile(nfci_vals, 75))
            thresholds["nfci_p90"] = float(np.percentile(nfci_vals, 90))
            thresholds["nfci_p50"] = float(np.percentile(nfci_vals, 50))
            thresholds["nfci_mean"] = float(np.mean(nfci_vals))

    return thresholds


def classify_single_state(
    m: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[str, str, dict[str, float]]:
    """
    Classify a single macro state into a regime label.

    Returns
    -------
    label : str — full regime label
    simple_label : str — simplified 4-regime label
    signals : dict — intermediate signal values used for classification
    """
    mac_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}

    def get(col: str) -> float:
        return float(m[mac_idx[col]]) if col in mac_idx else np.nan

    # Key signals
    ig_oas_val = get("ig_oas")
    vix_val = get("vix")
    infl_us = get("infl_US")
    short_us = get("short_rate_US")
    long_us = get("long_rate_US")
    us_real10y = get("us_real10y")
    unemp_us = get("unemp_US")

    # NFCI proxy: standardized average of ig_oas and vix (both stress indicators)
    ig_p50 = thresholds.get("ig_oas_p50", 1.0)
    ig_p75 = thresholds.get("ig_oas_p75", 1.5)
    ig_p90 = thresholds.get("ig_oas_p90", 2.5)
    vix_p50 = thresholds.get("vix_p50", 18.0)
    vix_p75 = thresholds.get("vix_p75", 22.0)
    vix_p90 = thresholds.get("vix_p90", 30.0)

    infl_p50 = thresholds.get("infl_US_p50", 2.0)
    infl_p75 = thresholds.get("infl_US_p75", 3.0)
    short_p75 = thresholds.get("short_rate_US_p75", 3.0)
    short_p25 = thresholds.get("short_rate_US_p25", 0.5)
    unemp_p75 = thresholds.get("unemp_US_p75", 6.5)
    real10y_p75 = thresholds.get("us_real10y_p75", 1.0)
    real10y_p50 = thresholds.get("us_real10y_p50", 0.3)

    stress_high = (ig_oas_val > ig_p90) or (vix_val > vix_p90)
    stress_moderate = (ig_oas_val > ig_p75) or (vix_val > vix_p75)
    growth_weak = unemp_us > unemp_p75
    infl_high = infl_us > infl_p75
    infl_elevated = infl_us > infl_p50
    rates_high = (short_us > short_p75) or (us_real10y > real10y_p75)

    signals = {
        "ig_oas": ig_oas_val,
        "vix": vix_val,
        "infl_US": infl_us,
        "short_rate_US": short_us,
        "us_real10y": us_real10y,
        "unemp_US": unemp_us,
        "stress_high": float(stress_high),
        "stress_moderate": float(stress_moderate),
    }

    # Priority classification
    if stress_high and growth_weak:
        label = REGIME_RECESSION_STRESS
    elif stress_high:
        label = REGIME_HIGH_STRESS
    elif rates_high and infl_elevated and not stress_high:
        label = REGIME_HIGHER_FOR_LONGER
    elif infl_high and not stress_high and not growth_weak:
        label = REGIME_INFLATIONARY_EXPANSION
    elif not infl_high and not rates_high and not stress_moderate:
        label = REGIME_SOFT_LANDING
    elif not infl_elevated and growth_weak and not stress_high:
        label = REGIME_DISINFL_SLOWDOWN
    elif stress_moderate and growth_weak:
        label = REGIME_RISK_OFF
    else:
        label = REGIME_NEUTRAL

    simple_label = SIMPLE_REGIME_MAP.get(label, "neutral")
    return label, simple_label, signals


def classify_sample_set(
    m_samples: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Classify a set of sampled macro states.

    Parameters
    ----------
    m_samples : np.ndarray, shape (N, STATE_DIM)
    thresholds : dict from compute_regime_thresholds()

    Returns
    -------
    DataFrame with columns: [macro state cols..., regime_label, simple_regime, + signals]
    """
    rows = []
    for i in range(len(m_samples)):
        m = m_samples[i]
        label, simple, signals = classify_single_state(m, thresholds)
        row = {col: float(m[j]) for j, col in enumerate(MACRO_STATE_COLS)}
        row["regime_label"] = label
        row["simple_regime"] = simple
        row.update({f"sig_{k}": v for k, v in signals.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def add_recession_overlay(
    df: pd.DataFrame,
    date_col: str = "anchor_date",
) -> pd.DataFrame:
    """Add a boolean `in_recession` column based on NBER recession dates."""
    if date_col not in df.columns:
        df["in_recession"] = False
        return df

    dates = pd.to_datetime(df[date_col])
    in_rec = pd.Series(False, index=df.index)
    for peak, trough in NBER_RECESSIONS:
        mask = (dates >= pd.Timestamp(peak)) & (dates <= pd.Timestamp(trough))
        in_rec = in_rec | mask
    df = df.copy()
    df["in_recession"] = in_rec
    return df


def regime_summary(classified_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize regime label frequency and mean portfolio outcomes."""
    if classified_df.empty:
        return pd.DataFrame()

    weight_cols = [c for c in classified_df.columns if c.startswith("w_")]
    regime_cols = ["regime_label", "simple_regime"]

    groups = classified_df.groupby("regime_label")
    rows = []
    for regime, chunk in groups:
        row: dict = {
            "regime_label": regime,
            "simple_regime": SIMPLE_REGIME_MAP.get(str(regime), "neutral"),
            "count": len(chunk),
            "share": len(chunk) / len(classified_df),
        }
        for col in ["port_return", "port_risk", "port_entropy", "G_value"]:
            if col in chunk.columns:
                row[f"mean_{col}"] = float(chunk[col].mean())
        for col in weight_cols:
            row[f"mean_{col}"] = float(chunk[col].mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
