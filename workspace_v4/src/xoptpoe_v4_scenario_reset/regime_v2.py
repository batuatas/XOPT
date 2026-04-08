"""
regime_v2.py

Two-layer regime system for v4 scenario reset.

Layer A: Economic dimension scores (continuous, -1 to +1)
  - dim_growth: based on unemp_US, short_rate_US
  - dim_inflation: based on infl_US, infl_EA
  - dim_policy: based on us_real10y, term_slope_US
  - dim_stress: based on ig_oas, vix
  - dim_fin_cond: based on ig_oas, us_real10y combined

Layer B: Readable conference labels (mapped from dimensional scores)
  - soft_landing
  - higher_for_longer
  - reflation_risk_on
  - risk_off_stress
  - mixed_mid_cycle
  - disinflationary_slowdown
  - high_stress_defensive

Regime transition logic:
  - anchor_regime: regime of the anchor state m0
  - scenario_regime: regime of the generated scenario state
  - key_macro_shifts: top 3 variable changes by standardized magnitude
  - allocation_implication: narrative of what the shift means for portfolio
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS


# ---------------------------------------------------------------------------
# Default percentile thresholds (populated from data; these are fallback values)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "infl_US": {"low": 2.0, "high": 4.5},
    "infl_EA": {"low": 1.5, "high": 4.0},
    "short_rate_US": {"low": 1.0, "high": 4.0},
    "unemp_US": {"low": 4.0, "high": 6.0},
    "ig_oas": {"low": 0.9, "high": 1.7},
    "us_real10y": {"low": -0.5, "high": 1.5},
    "vix": {"low": 16.0, "high": 25.0},
    "term_slope_US": {"low": 0.0, "high": 2.0},
}

# Implication narratives for regime transitions
REGIME_IMPLICATIONS: dict[str, str] = {
    "soft_landing": (
        "Balanced growth with low stress — model favors diversified equities, "
        "reduced defensives. Bond allocation moderate."
    ),
    "higher_for_longer": (
        "Persistent inflation with high policy rates — gold elevated as real rate hedge, "
        "equity concentration reduced, short-duration tilt."
    ),
    "reflation_risk_on": (
        "Reflationary expansion — equity overweight, commodities/gold moderate, "
        "credit spreads tight. Model sees strong cross-asset returns."
    ),
    "risk_off_stress": (
        "Financial stress / risk-off — defensives elevated (gold, govts), "
        "equity reduced. Return expectations compressed."
    ),
    "mixed_mid_cycle": (
        "Mid-cycle with mixed signals — diversified portfolio, no strong tilt. "
        "EW departure low, model return close to EW."
    ),
    "disinflationary_slowdown": (
        "Disinflation with growth softening — bonds favored, equity overweight fading, "
        "gold moderate. Policy accommodation likely."
    ),
    "high_stress_defensive": (
        "Extreme stress — maximum defensives, gold as safe haven elevated, "
        "credit spreads wide. Return ceiling from compression."
    ),
}


def build_regime_thresholds(
    feature_master: pd.DataFrame,
    low_pct: float = 33.0,
    high_pct: float = 67.0,
) -> dict[str, dict[str, float]]:
    """
    Compute percentile thresholds from historical data.

    Parameters
    ----------
    feature_master : pd.DataFrame
    low_pct, high_pct : float
        Percentile levels for low/high thresholds (default: 33rd and 67th).

    Returns
    -------
    thresholds : dict[str, dict[str, float]]
        For each variable: {"low": p33, "high": p67}
    """
    fm = feature_master.drop_duplicates(subset="month_end").copy()
    thresholds = {}
    for col in MACRO_STATE_COLS:
        if col not in fm.columns:
            thresholds[col] = DEFAULT_THRESHOLDS.get(col, {"low": 0.0, "high": 1.0})
            continue
        vals = fm[col].dropna().to_numpy(dtype=float)
        if len(vals) < 5:
            thresholds[col] = DEFAULT_THRESHOLDS.get(col, {"low": 0.0, "high": 1.0})
        else:
            thresholds[col] = {
                "low": float(np.percentile(vals, low_pct)),
                "high": float(np.percentile(vals, high_pct)),
            }
    return thresholds


def _score_var(val: float, low: float, high: float) -> float:
    """
    Map a scalar value to [-1, +1] based on thresholds.

    val < low  -> score = -1 (below-median territory)
    val > high -> score = +1 (above-median territory)
    in between -> linear interpolation
    """
    if high <= low:
        return 0.0
    if val <= low:
        return -1.0
    if val >= high:
        return +1.0
    return 2.0 * (val - low) / (high - low) - 1.0


def score_dimensions(
    m: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Compute 5 economic dimension scores from macro state m.

    Parameters
    ----------
    m : np.ndarray, shape (STATE_DIM,)
    thresholds : dict from build_regime_thresholds()

    Returns
    -------
    dims : dict with keys:
        dim_growth, dim_inflation, dim_policy, dim_stress, dim_fin_cond
        All values in [-1, +1].
    """
    col_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}

    def get(col: str) -> float:
        if col in col_idx:
            return float(m[col_idx[col]])
        return 0.0

    def score(col: str) -> float:
        val = get(col)
        thr = thresholds.get(col, DEFAULT_THRESHOLDS.get(col, {"low": 0.0, "high": 1.0}))
        return _score_var(val, thr["low"], thr["high"])

    # dim_growth: high unemp = negative growth signal; high short_rate = tightening = negative
    # We flip sign: low unemp (< low threshold) = good growth
    unemp_score = -score("unemp_US")   # negate: high unemp = bad growth
    rate_growth = -score("short_rate_US") * 0.3  # mild drag from high rates
    dim_growth = np.clip(0.7 * unemp_score + 0.3 * rate_growth, -1, 1)

    # dim_inflation: high infl_US + high infl_EA = inflationary
    dim_inflation = np.clip(0.7 * score("infl_US") + 0.3 * score("infl_EA"), -1, 1)

    # dim_policy: high real yield = restrictive policy; steep slope = accommodative
    real_score = score("us_real10y")
    slope_score = score("term_slope_US")
    dim_policy = np.clip(0.6 * real_score - 0.4 * slope_score, -1, 1)

    # dim_stress: high ig_oas + high vix = stress
    dim_stress = np.clip(0.5 * score("ig_oas") + 0.5 * score("vix"), -1, 1)

    # dim_fin_cond: combined financial conditions
    dim_fin_cond = np.clip(0.4 * score("ig_oas") + 0.4 * score("us_real10y") + 0.2 * score("vix"), -1, 1)

    return {
        "dim_growth": float(dim_growth),
        "dim_inflation": float(dim_inflation),
        "dim_policy": float(dim_policy),
        "dim_stress": float(dim_stress),
        "dim_fin_cond": float(dim_fin_cond),
    }


def classify_regime_v2(
    m: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> tuple[str, dict[str, float]]:
    """
    Classify a macro state into a conference-readable regime label.

    Decision tree based on dimension scores:
    1. Very high stress → high_stress_defensive
    2. High stress      → risk_off_stress
    3. High inflation + high policy → higher_for_longer
    4. High growth + positive inflation → reflation_risk_on
    5. Negative growth + low inflation → disinflationary_slowdown
    6. Low stress + moderate growth → soft_landing
    7. Everything else  → mixed_mid_cycle

    Returns
    -------
    regime_label : str
    dim_scores : dict
    """
    dims = score_dimensions(m, thresholds)

    g = dims["dim_growth"]
    inf = dims["dim_inflation"]
    pol = dims["dim_policy"]
    stress = dims["dim_stress"]

    # Rule-based classification (priority order)
    if stress > 0.7:
        label = "high_stress_defensive"
    elif stress > 0.35:
        label = "risk_off_stress"
    elif inf > 0.4 and pol > 0.3:
        label = "higher_for_longer"
    elif g > 0.3 and inf > 0.1:
        label = "reflation_risk_on"
    elif g < -0.3 and inf < -0.1:
        label = "disinflationary_slowdown"
    elif g > 0.1 and stress < 0.1:
        label = "soft_landing"
    else:
        label = "mixed_mid_cycle"

    return label, dims


def compute_regime_transition(
    m_anchor: np.ndarray,
    m_scenario: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    scales: np.ndarray,
) -> dict:
    """
    Compute regime transition diagnostics between anchor and scenario states.

    Parameters
    ----------
    m_anchor : np.ndarray, shape (D,)
    m_scenario : np.ndarray, shape (D,)
    thresholds : dict
    scales : np.ndarray, shape (D,)
        Per-variable historical std for standardization.

    Returns
    -------
    transition : dict with:
        anchor_regime, scenario_regime,
        key_shifts: list of (col, raw_shift, std_shift) for top 3 shifts,
        implication: narrative string,
        regime_changed: bool
    """
    anchor_regime, anchor_dims = classify_regime_v2(m_anchor, thresholds)
    scenario_regime, scenario_dims = classify_regime_v2(m_scenario, thresholds)

    # Compute standardized shifts
    raw_shifts = m_scenario - m_anchor
    std_shifts = raw_shifts / np.maximum(scales, 1e-8)

    # Top 3 shifts by absolute standardized magnitude
    abs_std = np.abs(std_shifts)
    top_idx = np.argsort(abs_std)[::-1][:3]

    key_shifts = []
    for i in top_idx:
        col = MACRO_STATE_COLS[i]
        key_shifts.append({
            "variable": col,
            "raw_shift": float(raw_shifts[i]),
            "std_shift": float(std_shifts[i]),
        })

    implication = REGIME_IMPLICATIONS.get(
        scenario_regime,
        "Regime implication not catalogued."
    )

    return {
        "anchor_regime": anchor_regime,
        "scenario_regime": scenario_regime,
        "anchor_dims": anchor_dims,
        "scenario_dims": scenario_dims,
        "key_shifts": key_shifts,
        "implication": implication,
        "regime_changed": anchor_regime != scenario_regime,
    }
