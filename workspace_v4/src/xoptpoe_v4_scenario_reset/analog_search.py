"""
analog_search.py

Stage 1: Historical Analog Candidate Generation.

Finds historical months (from feature_master_monthly.parquet) whose
macro state is directionally consistent with a question's target regime.

For each of the 4 questions, we define a regime filter function that
selects historical states by thresholds on interpretable macro variables.

Returns ranked candidate states (up to K=30) sorted by proximity to
the question regime center.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS


# ---------------------------------------------------------------------------
# Per-question regime filter functions
# ---------------------------------------------------------------------------
# Each filter takes a dict (or Series) of macro variable values
# and returns True if the historical month is regime-directionally
# consistent with the question target.

def _q1_gold_favorable_filter(row: dict) -> bool:
    """
    Q1_gold_threshold: Gold activation regime.

    Gold is favored by:
      - Low real yields (us_real10y low or negative)
      - Elevated financial stress (vix > 20 or ig_oas elevated)
      - OR moderately high inflation concern (infl_US > 3.5)

    Historical months where gold would plausibly be a material allocation.
    """
    real10y = float(row.get("us_real10y", 0.0))
    vix = float(row.get("vix", 15.0))
    ig_oas = float(row.get("ig_oas", 1.0))
    infl_US = float(row.get("infl_US", 2.0))
    short_rate_US = float(row.get("short_rate_US", 2.0))

    # Condition 1: Negative or very low real yield (gold as inflation hedge)
    low_real_yield = real10y < 0.5

    # Condition 2: Financial stress (gold as safe haven)
    stress_signal = vix > 22.0 or ig_oas > 1.5

    # Condition 3: High inflation concern (gold as store of value)
    high_infl = infl_US > 3.5

    # Condition 4: ZIRP / very low rates (gold alternative to cash)
    zirp = short_rate_US < 0.5

    return low_real_yield or stress_signal or high_infl or zirp


def _q2_ew_deviation_filter(row: dict) -> bool:
    """
    Q2_ew_departure: Equal-weight departure conditions.

    Strong EW departure occurs when:
      - High dispersion across asset classes (inflation shock + rate response)
      - Model has strong differentiated signals across sleeves
      - Conditions: high inflation AND (high rates OR wide spreads)
    """
    infl_US = float(row.get("infl_US", 2.0))
    short_rate_US = float(row.get("short_rate_US", 2.0))
    ig_oas = float(row.get("ig_oas", 1.0))
    vix = float(row.get("vix", 15.0))
    term_slope_US = float(row.get("term_slope_US", 1.5))

    # High inflation + tight policy = differentiated cross-asset signals
    inflationary_tightening = infl_US > 4.0 and short_rate_US > 3.0

    # Credit stress also drives dispersion
    credit_stress = ig_oas > 1.5 and infl_US > 3.0

    # Inverted curve (model sees regime shift)
    inverted_curve = term_slope_US < 0.0 and infl_US > 3.0

    # High vol differentiated environment
    high_vol_inflation = vix > 25.0 and infl_US > 3.5

    return inflationary_tightening or credit_stress or inverted_curve or high_vol_inflation


def _q3_return_discipline_filter(row: dict) -> bool:
    """
    Q3_return_discipline: Return with discipline.

    Balanced growth environment where model earns decent return
    without excessive concentration:
      - Moderate inflation (not too hot, not too cold)
      - Low stress
      - Moderate positive real yields
    """
    infl_US = float(row.get("infl_US", 2.0))
    vix = float(row.get("vix", 15.0))
    ig_oas = float(row.get("ig_oas", 1.0))
    us_real10y = float(row.get("us_real10y", 0.5))
    unemp_US = float(row.get("unemp_US", 4.5))

    # Goldilocks: moderate inflation, low stress
    goldilocks = 2.0 < infl_US < 5.0 and vix < 20.0 and ig_oas < 1.3

    # Or: low inflation with positive real yields (bond-friendly growth)
    low_infl_growth = infl_US < 3.0 and us_real10y > 0.5 and unemp_US < 5.5

    return goldilocks or low_infl_growth


def _q4_return_ceiling_filter(row: dict) -> bool:
    """
    Q4_return_ceiling: Return ceiling conditions.

    Either:
      A. Strong growth scenario (what would 5%+ look like)
      B. Stressed scenario (risk premium compression / return ceiling)
    """
    infl_US = float(row.get("infl_US", 2.0))
    vix = float(row.get("vix", 15.0))
    ig_oas = float(row.get("ig_oas", 1.0))
    us_real10y = float(row.get("us_real10y", 0.5))
    short_rate_US = float(row.get("short_rate_US", 2.0))

    # Path A: Strong growth (low stress, positive real yield, moderate inflation)
    strong_growth = vix < 16.0 and ig_oas < 0.9 and us_real10y > 1.0 and infl_US < 4.0

    # Path B: High stress (risk premium compression, return ceiling from defensive rotation)
    risk_off = vix > 30.0 or ig_oas > 2.0

    # Path C: Late cycle with inverted real curve
    late_cycle = short_rate_US > 4.0 and us_real10y < 0.0

    return strong_growth or risk_off or late_cycle


# ---------------------------------------------------------------------------
# Regime filter registry
# ---------------------------------------------------------------------------
ANALOG_FILTERS: dict[str, Callable[[dict], bool]] = {
    "Q1_gold_threshold": _q1_gold_favorable_filter,
    "Q2_ew_departure": _q2_ew_deviation_filter,
    "Q3_return_discipline": _q3_return_discipline_filter,
    "Q4_return_ceiling": _q4_return_ceiling_filter,
}

# Question regime center vectors (in MACRO_STATE_COLS space)
# Used to rank candidates by proximity to the "ideal" regime state
REGIME_CENTERS: dict[str, dict[str, float]] = {
    "Q1_gold_threshold": {
        "us_real10y": -0.5,
        "vix": 25.0,
        "ig_oas": 1.8,
        "infl_US": 4.0,
        "short_rate_US": 0.5,
    },
    "Q2_ew_departure": {
        "infl_US": 6.0,
        "short_rate_US": 4.5,
        "ig_oas": 1.8,
        "vix": 28.0,
        "term_slope_US": -0.5,
    },
    "Q3_return_discipline": {
        "infl_US": 3.0,
        "vix": 15.0,
        "ig_oas": 1.0,
        "us_real10y": 1.0,
        "unemp_US": 4.5,
    },
    "Q4_return_ceiling": {
        "vix": 14.0,
        "ig_oas": 0.8,
        "us_real10y": 1.5,
        "infl_US": 2.5,
        "short_rate_US": 3.0,
    },
}


def _standardize_state(
    m: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Standardize a state vector by per-variable scale."""
    return m / np.maximum(scales, 1e-8)


def find_analogs(
    feature_master: pd.DataFrame,
    question_id: str,
    anchor_date: pd.Timestamp,
    m0: np.ndarray,
    scales: np.ndarray,
    K: int = 30,
) -> pd.DataFrame:
    """
    Stage 1: Find historical analog macro states for a given question.

    Parameters
    ----------
    feature_master : pd.DataFrame
        Full feature_master_monthly.parquet
    question_id : str
        One of the 4 question IDs (must be in ANALOG_FILTERS)
    anchor_date : pd.Timestamp
        The anchor date (excluded from search)
    m0 : np.ndarray, shape (STATE_DIM,)
        Anchor macro state (used to ensure non-redundancy)
    scales : np.ndarray, shape (STATE_DIM,)
        Per-variable scale factors for distance computation
    K : int
        Maximum number of candidates to return

    Returns
    -------
    pd.DataFrame
        Each row is a candidate analog state with columns:
        month_end, <MACRO_STATE_COLS>, distance_from_m0
        Sorted by distance_from_regime_center ascending (most regime-relevant first).
        Excludes anchor date and any date within 3 months of anchor.
    """
    if question_id not in ANALOG_FILTERS:
        raise ValueError(
            f"Unknown question_id={question_id}. "
            f"Valid options: {list(ANALOG_FILTERS.keys())}"
        )

    filter_fn = ANALOG_FILTERS[question_id]
    regime_center_vars = REGIME_CENTERS[question_id]

    # Deduplicate: one row per date
    fm = feature_master.drop_duplicates(subset="month_end").copy()
    fm["month_end"] = pd.to_datetime(fm["month_end"])
    fm = fm.sort_values("month_end").reset_index(drop=True)

    # Exclude anchor date and ±3 months window
    anchor_ts = pd.Timestamp(anchor_date)
    exclude_window = pd.DateOffset(months=3)
    fm = fm[
        (fm["month_end"] < anchor_ts - exclude_window) |
        (fm["month_end"] > anchor_ts + exclude_window)
    ].copy()

    if fm.empty:
        return pd.DataFrame()

    # Apply regime filter
    passed = []
    for _, row in fm.iterrows():
        row_dict = row.to_dict()
        if filter_fn(row_dict):
            passed.append(row_dict)

    if not passed:
        # Fallback: return nearest historical states without filter
        passed = [row.to_dict() for _, row in fm.iterrows()]

    candidates_df = pd.DataFrame(passed)

    # Fill any missing MACRO_STATE_COLS with m0 values
    for i, col in enumerate(MACRO_STATE_COLS):
        if col not in candidates_df.columns:
            candidates_df[col] = m0[i]

    # Build macro state matrix for distance computation
    state_mat = candidates_df[MACRO_STATE_COLS].fillna(0.0).to_numpy(dtype=float)

    # Compute distance from regime center (in standardized space)
    # Use only variables present in the regime center definition
    regime_col_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}

    dist_to_center = np.zeros(len(candidates_df))
    center_weight = 0.0
    for col, center_val in regime_center_vars.items():
        if col in regime_col_idx:
            i = regime_col_idx[col]
            scale_i = float(scales[i])
            diff = (state_mat[:, i] - center_val) / max(scale_i, 1e-8)
            dist_to_center += diff ** 2
            center_weight += 1.0

    if center_weight > 0:
        dist_to_center = np.sqrt(dist_to_center / center_weight)

    # Also compute distance from m0 (want diverse from anchor)
    m0_std = _standardize_state(m0, scales)
    state_std = state_mat / np.maximum(scales, 1e-8)[np.newaxis, :]
    dist_from_m0 = np.sqrt(np.sum((state_std - m0_std) ** 2, axis=1))

    candidates_df = candidates_df.copy()
    candidates_df["dist_to_regime_center"] = dist_to_center
    candidates_df["distance_from_m0"] = dist_from_m0

    # Sort: prefer closer to regime center
    candidates_df = candidates_df.sort_values("dist_to_regime_center").reset_index(drop=True)

    # Keep top K
    candidates_df = candidates_df.head(K)

    # Return with key columns
    out_cols = ["month_end"] + MACRO_STATE_COLS + ["dist_to_regime_center", "distance_from_m0"]
    existing_out = [c for c in out_cols if c in candidates_df.columns]
    return candidates_df[existing_out].copy()
