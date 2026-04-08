"""Long-only, fully-invested portfolio weight construction rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_long_only(raw: np.ndarray) -> np.ndarray:
    """Project nonnegative vector to simplex with sum=1 fallback to equal weight."""
    vec = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    total = float(vec.sum())
    if total <= 0:
        return np.repeat(1.0 / len(vec), len(vec))
    return vec / total


def equal_weight(scores: pd.Series) -> pd.Series:
    """Equal-weight portfolio regardless of predictions."""
    n = len(scores)
    weights = np.repeat(1.0 / n, n)
    return pd.Series(weights, index=scores.index, dtype=float)


def top_k_equal(scores: pd.Series, *, k: int) -> pd.Series:
    """Top-k long-only allocation with equal weights on selected sleeves."""
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(scores)
    k_eff = min(k, n)
    top_idx = scores.sort_values(ascending=False).index[:k_eff]
    out = pd.Series(0.0, index=scores.index, dtype=float)
    out.loc[top_idx] = 1.0 / k_eff
    return out


def score_positive(scores: pd.Series) -> pd.Series:
    """Allocate proportionally to positive predicted scores."""
    weights = _normalize_long_only(scores.to_numpy(dtype=float))
    return pd.Series(weights, index=scores.index, dtype=float)


def mean_variance_clipped(
    scores: pd.Series,
    covariance: pd.DataFrame | np.ndarray,
    *,
    ridge: float = 1e-3,
    risk_aversion: float = 1.0,
) -> pd.Series:
    """Regularized mean-variance heuristic with long-only clipping."""
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be > 0")
    mu = scores.to_numpy(dtype=float)

    cov = np.asarray(covariance, dtype=float)
    n = len(mu)
    if cov.shape != (n, n):
        raise ValueError(f"covariance has shape {cov.shape}, expected {(n, n)}")

    reg_cov = cov + np.eye(n) * float(ridge)
    # Unconstrained direction, then clipped to long-only simplex.
    raw = np.linalg.solve(reg_cov, mu / risk_aversion)
    weights = _normalize_long_only(raw)
    return pd.Series(weights, index=scores.index, dtype=float)
