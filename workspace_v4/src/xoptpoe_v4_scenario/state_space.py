"""
state_space.py

Defines the compact manipulated macro-financial state for v4 scenario generation.

The "state" m is a subset of columns from feature_master_monthly.parquet that are:
  - Global (identical across all 14 sleeves at a given date)
  - Non-interaction (base macro features, not products)
  - Interpretable (rates, spreads, valuation, vol, currencies)

This module provides:
  - MACRO_STATE_COLS: the canonical ordered list of state columns
  - load_state: extract m0 from the feature master at a given date
  - build_feature_matrix: reconstruct the full (14, F) feature matrix given a perturbed state
  - box_constraints: historical lower/upper bounds with slack
  - state_scales: per-variable historical std for L2 anchor scaling
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical macro state definition
# ---------------------------------------------------------------------------
# These are the GLOBAL_CANONICAL + GLOBAL features that appear in
# core_plus_interactions and are interpretable economy/market variables.
# Delta-1m variants are included because they carry signal for the model
# and represent the "change dimension" of the state.
# Admin flags (macro_stale_flag, *_level duplicates) are excluded.
# ---------------------------------------------------------------------------
MACRO_STATE_COLS: list[str] = [
    # --- Inflation (US, EA, JP) ---
    "infl_US",
    "infl_EA",
    "infl_JP",
    # --- Short rates ---
    "short_rate_US",
    "short_rate_EA",
    "short_rate_JP",
    # --- Long rates ---
    "long_rate_US",
    "long_rate_EA",
    "long_rate_JP",
    # --- Term slopes ---
    "term_slope_US",
    "term_slope_EA",
    "term_slope_JP",
    # --- Unemployment ---
    "unemp_US",
    "unemp_EA",
    # --- Credit / financial conditions ---
    "ig_oas",           # IG credit spread — core financial conditions indicator
    "us_real10y",       # US 10Y real yield — key for gold, equities, real assets
    # --- Equity/market sentiment ---
    "vix",              # Volatility — stress indicator
    # --- Commodities / currencies ---
    "oil_wti",          # Oil price level
    "usd_broad",        # Broad USD level
]

# Number of state dimensions
STATE_DIM: int = len(MACRO_STATE_COLS)

# Interaction features whose macro component is in MACRO_STATE_COLS
# These get recomputed when state is perturbed.
# Format: {feature_col: (macro_col, sleeve_dummy_or_None)}
# "asset_group_dummy_x_predictor" family: dummy is 0/1 per sleeve, fixed
# "stress_x_momentum": ig_oas * rel_mom, vix * mom_12_1 — sleeve-specific momentum is fixed
INTERACTION_MAP: dict[str, str] = {
    # x_us_real10y interactions — macro component is us_real10y
    "int_alternative_x_us_real10y": "us_real10y",
    "int_credit_x_us_real10y": "us_real10y",
    "int_equity_x_us_real10y": "us_real10y",
    "int_fixed_income_x_us_real10y": "us_real10y",
    "int_real_asset_x_us_real10y": "us_real10y",
    "int_cape_usa_x_us_real10y": "us_real10y",
    "int_log_horizon_x_us_real10y": "us_real10y",
    # x_vix interactions — macro component is vix
    "int_alternative_x_vix": "vix",
    "int_credit_x_vix": "vix",
    "int_equity_x_vix": "vix",
    "int_fixed_income_x_vix": "vix",
    "int_real_asset_x_vix": "vix",
    "int_log_horizon_x_vix": "vix",
    "int_vix_x_em_minus_global_pe": "vix",
    "int_vix_x_mom_12_1": "vix",
    # ig_oas interactions
    "int_china_cli_x_ig_oas": "ig_oas",
    "int_ig_oas_x_rel_mom_vs_treasury": "ig_oas",
    "int_ig_oas_x_cape_local": "ig_oas",
    # us_real10y x momentum
    "int_us_real10y_x_mom_12_1": "us_real10y",
}


def load_state(
    date: pd.Timestamp,
    feature_master: pd.DataFrame,
    sleeve_order: Sequence[str],
    modeling_panel: pd.DataFrame | None = None,
    horizon_months: int = 60,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract the compact macro state m0 at a given date.

    Returns
    -------
    m0 : np.ndarray, shape (STATE_DIM,)
        Compact macro state vector at date.
    feature_matrix : pd.DataFrame
        Full (len(sleeve_order), all_feature_cols) matrix at that date,
        indexed by sleeve_id.

    Notes
    -----
    If modeling_panel is provided, the feature matrix is sourced from it
    (filtered to horizon_months). This is required because horizon-related
    features (horizon_60_flag, log_horizon_years, int_log_horizon_x_*)
    exist only in modeling_panel, not in feature_master.
    """
    date = pd.Timestamp(date)

    # --- Find macro state from feature_master ---
    month_rows = feature_master[feature_master["month_end"] == date].copy()
    if month_rows.empty:
        # Try fuzzy match — find closest month_end
        available = pd.to_datetime(feature_master["month_end"].unique())
        closest = available[np.argmin(np.abs(available - date))]
        month_rows = feature_master[feature_master["month_end"] == closest].copy()
        if month_rows.empty:
            raise KeyError(f"No feature_master rows found near date={date}")

    month_rows_idx = month_rows.set_index("sleeve_id")
    # Extract macro state from any sleeve row (same across all sleeves)
    first_row = month_rows_idx.iloc[0]
    m0 = np.array([float(first_row[col]) for col in MACRO_STATE_COLS], dtype=float)

    # --- Build feature matrix ---
    if modeling_panel is not None:
        # Source from modeling_panel — has all features including horizon features
        panel_rows = modeling_panel[
            (modeling_panel["month_end"] == date) &
            (modeling_panel["horizon_months"] == horizon_months)
        ].copy()
        if panel_rows.empty:
            # Try fuzzy match on date
            available_mp = pd.to_datetime(modeling_panel["month_end"].unique())
            closest_mp = available_mp[np.argmin(np.abs(available_mp - date))]
            panel_rows = modeling_panel[
                (modeling_panel["month_end"] == closest_mp) &
                (modeling_panel["horizon_months"] == horizon_months)
            ].copy()
        if panel_rows.empty:
            raise KeyError(f"No modeling_panel rows found for date={date}, horizon={horizon_months}")
        panel_rows = panel_rows.set_index("sleeve_id")
        available_sleeves = [s for s in sleeve_order if s in panel_rows.index]
        feature_matrix = panel_rows.loc[available_sleeves]
    else:
        # Fallback: feature_master only (missing horizon features — may cause issues)
        available_sleeves = [s for s in sleeve_order if s in month_rows_idx.index]
        feature_matrix = month_rows_idx.loc[available_sleeves]

    return m0, feature_matrix


def build_feature_matrix(
    m_perturbed: np.ndarray,
    feature_matrix_base: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """
    Rebuild the (n_sleeves, n_features) feature matrix after perturbing the macro state.

    For each feature column:
    - If it's in MACRO_STATE_COLS: replace with perturbed value (same for all sleeves)
    - If it's an interaction involving a MACRO_STATE_COLS variable:
        recompute as perturbed_macro_val * (base_value / base_macro_val) to preserve
        the non-macro component exactly, OR simply replace the macro part.
        We use: new_interaction = base_interaction * (m_perturbed_macro / m_base_macro)
        This is exact for linear interactions of the form: interaction = macro_val * sleeve_val.
    - Otherwise: keep base value unchanged.

    Parameters
    ----------
    m_perturbed : np.ndarray, shape (STATE_DIM,)
    feature_matrix_base : pd.DataFrame
        shape (n_sleeves, all_feature_cols), indexed by sleeve_id
    feature_columns : list[str]
        Ordered list of feature columns the model expects (core_plus_interactions set)

    Returns
    -------
    X : np.ndarray, shape (n_sleeves, len(feature_columns))
    """
    macro_col_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}
    m_base = np.array(
        [float(feature_matrix_base.iloc[0][col]) for col in MACRO_STATE_COLS],
        dtype=float,
    )

    X = feature_matrix_base[feature_columns].to_numpy(dtype=float).copy()

    for j, feat in enumerate(feature_columns):
        if feat in macro_col_idx:
            # Pure macro column: replace with perturbed value (broadcast across sleeves)
            X[:, j] = m_perturbed[macro_col_idx[feat]]

        elif feat in INTERACTION_MAP:
            # Interaction: feat = macro_component * sleeve_component
            # Recompute as: new = (m_perturbed_macro / m_base_macro) * base_interaction
            macro_col = INTERACTION_MAP[feat]
            if macro_col in macro_col_idx:
                i_macro = macro_col_idx[macro_col]
                base_macro_val = m_base[i_macro]
                new_macro_val = m_perturbed[i_macro]
                if abs(base_macro_val) > 1e-12:
                    X[:, j] = X[:, j] * (new_macro_val / base_macro_val)
                else:
                    # Base macro is ~0; use additive update: just replace with 0 * sleeve or perturbed * sleeve
                    # Can't rescale from 0; keep base interaction (conservative)
                    pass

        # else: sleeve-specific feature, held fixed — no change to X[:, j]

    return X.astype(np.float32)


class FastFeatureBuilder:
    """
    Precomputed feature matrix builder for fast MALA perturbation.

    Precomputes at init:
      - X_base: (n_sleeves, n_features) base numpy array
      - macro_indices: list of (feature_col_idx, state_idx) for pure macro cols
      - interaction_indices: list of (feature_col_idx, state_idx, base_macro_val, X_base_col)
        for interaction cols

    At call time, only applies the perturbation — no DataFrame overhead.
    """

    def __init__(
        self,
        feature_matrix_base: pd.DataFrame,
        feature_columns: list[str],
    ):
        macro_col_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}
        self.m_base = np.array(
            [float(feature_matrix_base.iloc[0][col]) for col in MACRO_STATE_COLS],
            dtype=float,
        )
        self.X_base = feature_matrix_base[feature_columns].to_numpy(dtype=np.float32).copy()
        self.n_sleeves = self.X_base.shape[0]

        # Precompute which columns to update
        self._macro_updates: list[tuple[int, int]] = []       # (feat_col_j, state_i)
        self._interaction_updates: list[tuple[int, int, float]] = []  # (feat_col_j, state_i, base_val)

        for j, feat in enumerate(feature_columns):
            if feat in macro_col_idx:
                self._macro_updates.append((j, macro_col_idx[feat]))
            elif feat in INTERACTION_MAP:
                macro_col = INTERACTION_MAP[feat]
                if macro_col in macro_col_idx:
                    i_macro = macro_col_idx[macro_col]
                    base_macro_val = self.m_base[i_macro]
                    self._interaction_updates.append((j, i_macro, float(base_macro_val)))

    def __call__(self, m_perturbed: np.ndarray) -> np.ndarray:
        """
        Build perturbed feature matrix. Returns (n_sleeves, n_features) float32 array.
        ~10x faster than build_feature_matrix() for repeated calls.
        """
        X = self.X_base.copy()

        # Pure macro columns: broadcast scalar across all sleeves
        for j, i in self._macro_updates:
            X[:, j] = m_perturbed[i]

        # Interaction columns: rescale by (new_macro / base_macro)
        for j, i_macro, base_val in self._interaction_updates:
            if abs(base_val) > 1e-12:
                X[:, j] = self.X_base[:, j] * (m_perturbed[i_macro] / base_val)
            # else: keep base interaction (zero base — no safe rescaling)

        return X


def box_constraints(
    feature_master: pd.DataFrame,
    slack_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute [a, b] box constraints for the macro state.

    a = historical_min - slack * historical_std
    b = historical_max + slack * historical_std

    Returns
    -------
    a, b : np.ndarray, shape (STATE_DIM,)
    """
    # Use one row per date (any sleeve; macro cols are identical across sleeves)
    # Take first sleeve per date
    fm_one = feature_master.drop_duplicates(subset="month_end")

    a_vals = np.zeros(STATE_DIM)
    b_vals = np.zeros(STATE_DIM)
    for i, col in enumerate(MACRO_STATE_COLS):
        col_vals = fm_one[col].dropna().to_numpy(dtype=float)
        if len(col_vals) == 0:
            a_vals[i] = -1e6
            b_vals[i] = 1e6
            continue
        col_min = float(col_vals.min())
        col_max = float(col_vals.max())
        col_std = float(col_vals.std(ddof=1)) if len(col_vals) > 1 else 1.0
        a_vals[i] = col_min - slack_multiplier * col_std
        b_vals[i] = col_max + slack_multiplier * col_std

    return a_vals, b_vals


def state_scales(
    feature_master: pd.DataFrame,
    train_end: pd.Timestamp | None = None,
) -> np.ndarray:
    """
    Per-variable historical std, used for L2 anchor scaling.

    If train_end is provided, computes std only on training-period data.

    Returns
    -------
    scales : np.ndarray, shape (STATE_DIM,), all > 0
    """
    fm_one = feature_master.drop_duplicates(subset="month_end")
    if train_end is not None:
        fm_one = fm_one[pd.to_datetime(fm_one["month_end"]) <= pd.Timestamp(train_end)]

    scales = np.ones(STATE_DIM)
    for i, col in enumerate(MACRO_STATE_COLS):
        col_vals = fm_one[col].dropna().to_numpy(dtype=float)
        if len(col_vals) > 1:
            std = float(col_vals.std(ddof=1))
            scales[i] = max(std, 1e-6)
    return scales
