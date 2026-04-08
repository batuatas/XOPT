"""Robust long-only optimization layers and SAA risk estimation."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from cvxpylayers.torch import CvxpyLayer

from xoptpoe_v2_models.data import SLEEVE_ORDER


warnings.filterwarnings(
    "ignore",
    message="Converting A to a CSC \\(compressed sparse column\\) matrix; may take a while\\.",
    category=UserWarning,
)


@dataclass(frozen=True)
class RiskConfig:
    """Trailing sleeve-risk estimation settings."""

    lookback_months: int = 60
    min_months: int = 12
    ewma_beta: float = 0.94
    diagonal_shrinkage: float = 0.10
    ridge: float = 1e-6
    annualize_factor: float = 12.0


@dataclass(frozen=True)
class OptimizerConfig:
    """Robust mean-variance portfolio configuration."""

    lambda_risk: float
    kappa: float
    omega_type: str

    def key(self) -> tuple[float, float, str]:
        return (float(self.lambda_risk), float(self.kappa), str(self.omega_type))



def candidate_optimizer_grid() -> list[OptimizerConfig]:
    """Compact first-pass grid centered on the paper's stronger robust regimes."""
    configs: list[OptimizerConfig] = []
    for lambda_risk in (5.0, 10.0):
        for kappa in (0.1, 1.0):
            for omega_type in ("diag", "identity"):
                configs.append(OptimizerConfig(lambda_risk=lambda_risk, kappa=kappa, omega_type=omega_type))
    return configs



def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(np.asarray(matrix, dtype=float))
    eigvals = np.clip(eigvals, 1e-12, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T



def estimate_ewma_covariance(
    excess_history: pd.DataFrame,
    month_end: pd.Timestamp,
    *,
    config: RiskConfig,
) -> np.ndarray:
    """Estimate trailing annualized EWMA covariance for sleeve excess returns."""
    hist = excess_history.loc[excess_history.index < pd.Timestamp(month_end), list(SLEEVE_ORDER)].copy()
    hist = hist.dropna(how="any")
    if hist.empty:
        return np.eye(len(SLEEVE_ORDER), dtype=float)

    panel = hist.tail(config.lookback_months)
    n_obs = len(panel)
    if n_obs == 0:
        return np.eye(len(SLEEVE_ORDER), dtype=float)

    weights = np.power(config.ewma_beta, np.arange(n_obs - 1, -1, -1, dtype=float))
    weights = weights / weights.sum()

    x = panel.to_numpy(dtype=float)
    mean = np.sum(x * weights[:, None], axis=0)
    centered = x - mean
    denom = max(1e-12, 1.0 - float(np.sum(weights**2)))
    cov = (centered * weights[:, None]).T @ centered / denom

    diag = np.diag(np.diag(cov))
    cov = (1.0 - config.diagonal_shrinkage) * cov + config.diagonal_shrinkage * diag
    cov = cov + np.eye(cov.shape[0]) * float(config.ridge)
    cov = (cov + cov.T) / 2.0
    cov = cov * float(config.annualize_factor)
    return cov.astype(float)



def omega_from_sigma(sigma: np.ndarray, omega_type: str) -> np.ndarray:
    """Construct the uncertainty matrix from the estimated covariance."""
    sigma = np.asarray(sigma, dtype=float)
    if omega_type == "diag":
        return np.diag(np.clip(np.diag(sigma), 1e-8, None))
    if omega_type == "identity":
        return np.eye(sigma.shape[0], dtype=float)
    raise ValueError(f"Unsupported omega_type: {omega_type}")


class RobustOptimizerCache:
    """Cache cvxpylayers with month-specific constant risk matrices."""

    def __init__(
        self,
        *,
        sigma_by_month: dict[pd.Timestamp, np.ndarray],
    ) -> None:
        self.sigma_by_month = sigma_by_month
        self._layer_cache: dict[tuple[pd.Timestamp, float, float, str], CvxpyLayer] = {}

    def sigma_for_month(self, month_end: pd.Timestamp) -> np.ndarray:
        return np.asarray(self.sigma_by_month[pd.Timestamp(month_end)], dtype=float)

    def omega_for_month(self, month_end: pd.Timestamp, omega_type: str) -> np.ndarray:
        return omega_from_sigma(self.sigma_for_month(month_end), omega_type)

    def get_layer(self, month_end: pd.Timestamp, config: OptimizerConfig) -> CvxpyLayer:
        key = (pd.Timestamp(month_end), *config.key())
        if key not in self._layer_cache:
            sigma = self.sigma_for_month(month_end)
            omega = self.omega_for_month(month_end, config.omega_type)
            sigma_sqrt = _matrix_sqrt_psd(sigma)
            omega_sqrt = _matrix_sqrt_psd(omega)

            w = cp.Variable(len(SLEEVE_ORDER))
            mu = cp.Parameter(len(SLEEVE_ORDER))
            objective = cp.Minimize(
                -mu @ w
                + float(config.kappa) * cp.norm(omega_sqrt @ w, 2)
                + 0.5 * float(config.lambda_risk) * cp.sum_squares(sigma_sqrt @ w)
            )
            problem = cp.Problem(objective, [cp.sum(w) == 1.0, w >= 0.0])
            if not problem.is_dpp():
                raise ValueError("Robust optimizer problem is not DPP with month-specific constants")
            self._layer_cache[key] = CvxpyLayer(problem, parameters=[mu], variables=[w])
        return self._layer_cache[key]

    def solve(self, month_end: pd.Timestamp, mu: torch.Tensor, config: OptimizerConfig) -> torch.Tensor:
        layer = self.get_layer(month_end, config)
        weights, = layer(mu)
        return weights



def build_sigma_map(
    months: list[pd.Timestamp],
    *,
    excess_history: pd.DataFrame,
    risk_config: RiskConfig,
) -> dict[pd.Timestamp, np.ndarray]:
    """Precompute trailing covariance matrices for each decision month."""
    out: dict[pd.Timestamp, np.ndarray] = {}
    for month_end in sorted(pd.Timestamp(m) for m in months):
        out[month_end] = estimate_ewma_covariance(excess_history, month_end, config=risk_config)
    return out
