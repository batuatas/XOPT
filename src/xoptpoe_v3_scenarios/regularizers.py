"""Scenario priors and plausibility penalties for the v3 scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .state import BASE_STATE_VARIABLES, ScenarioAnchor


@dataclass(frozen=True)
class BoundsSpec:
    """Soft box constraints derived from historical monthly state support."""

    lower: np.ndarray
    upper: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class Var1Prior:
    """Gaussian VAR(1) innovation prior over the manipulated state."""

    intercept: np.ndarray
    transition: np.ndarray
    covariance: np.ndarray
    covariance_inv: np.ndarray

    def conditional_mean(self, previous_state: np.ndarray) -> np.ndarray:
        return self.intercept + self.transition @ previous_state

    def energy(self, current_state: np.ndarray, previous_state: np.ndarray) -> float:
        diff = np.asarray(current_state, dtype=float) - self.conditional_mean(previous_state)
        return float(0.5 * diff.T @ self.covariance_inv @ diff)


@dataclass(frozen=True)
class ScenarioRegularizer:
    """Combined prior used by the scenario scaffold."""

    bounds: BoundsSpec
    var1: Var1Prior
    anchor_reference: np.ndarray
    previous_reference: np.ndarray
    anchor_weight: float = 0.25
    var1_weight: float = 1.0

    def in_bounds(self, state_vector: np.ndarray) -> bool:
        x = np.asarray(state_vector, dtype=float)
        return bool(np.all(x >= self.bounds.lower) and np.all(x <= self.bounds.upper))

    def project(self, state_vector: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(state_vector, dtype=float), self.bounds.lower, self.bounds.upper)

    def anchor_energy(self, state_vector: np.ndarray) -> float:
        diff = (np.asarray(state_vector, dtype=float) - self.anchor_reference) / self.bounds.scale
        return float(0.5 * diff.T @ diff)

    def total_energy(self, state_vector: np.ndarray) -> float:
        if not self.in_bounds(state_vector):
            return float("inf")
        x = np.asarray(state_vector, dtype=float)
        return self.anchor_weight * self.anchor_energy(x) + self.var1_weight * self.var1.energy(x, self.previous_reference)


def fit_bounds(feature_master_monthly: pd.DataFrame) -> BoundsSpec:
    """Fit conservative bounds from historical monthly support."""
    monthly = feature_master_monthly.groupby("month_end", as_index=False).first()
    frame = monthly[list(BASE_STATE_VARIABLES)].apply(pd.to_numeric, errors="coerce")
    means = frame.mean().to_numpy(dtype=float)
    stds = frame.std(ddof=1).fillna(0.0).to_numpy(dtype=float)
    mins = frame.min().to_numpy(dtype=float)
    maxs = frame.max().to_numpy(dtype=float)
    scale = np.where(np.isfinite(stds) & (stds > 1e-8), stds, np.maximum(np.abs(means), 1.0))
    lower = mins - scale
    upper = maxs + scale
    return BoundsSpec(lower=lower.astype(float), upper=upper.astype(float), scale=scale.astype(float))


def fit_var1_prior(feature_master_monthly: pd.DataFrame) -> Var1Prior:
    """Fit a plain VAR(1) Gaussian innovation prior over the base state."""
    monthly = feature_master_monthly.groupby("month_end", as_index=False).first().sort_values("month_end").reset_index(drop=True)
    monthly = monthly[list(BASE_STATE_VARIABLES)].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    x = monthly.to_numpy(dtype=float)
    if len(x) < 24:
        raise ValueError("Need at least 24 monthly observations to fit a stable VAR(1) prior")

    x_t = x[:-1]
    x_tp1 = x[1:]
    design = np.column_stack([np.ones(len(x_t), dtype=float), x_t])
    try:
        coeffs, *_ = np.linalg.lstsq(design, x_tp1, rcond=None)
    except np.linalg.LinAlgError:
        ridge = 1e-6
        gram = design.T @ design + ridge * np.eye(design.shape[1], dtype=float)
        coeffs = np.linalg.solve(gram, design.T @ x_tp1)
    intercept = coeffs[0]
    transition = coeffs[1:].T
    residual = x_tp1 - design @ coeffs
    covariance = np.cov(residual, rowvar=False)
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0 + np.eye(covariance.shape[0]) * 1e-6
    covariance_inv = np.linalg.pinv(covariance)
    return Var1Prior(
        intercept=intercept.astype(float),
        transition=transition.astype(float),
        covariance=covariance.astype(float),
        covariance_inv=covariance_inv.astype(float),
    )


def build_regularizer(
    feature_master_monthly: pd.DataFrame,
    anchor: ScenarioAnchor,
) -> ScenarioRegularizer:
    """Fit the first-pass v3 regularizer stack."""
    bounds = fit_bounds(feature_master_monthly)
    var1 = fit_var1_prior(feature_master_monthly)
    return ScenarioRegularizer(
        bounds=bounds,
        var1=var1,
        anchor_reference=np.asarray(anchor.current_base_state, dtype=float),
        previous_reference=np.asarray(anchor.previous_base_state, dtype=float),
    )
