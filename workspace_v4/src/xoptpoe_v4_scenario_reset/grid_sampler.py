"""
grid_sampler.py

Stage 1b: Latin Hypercube Sampling within VAR(1) plausibility box.

Generates N=200 diverse candidate macro states by LHS within the
VAR(1) one-step prediction box [mu - 3*sigma, mu + 3*sigma].

This provides broad exploration beyond the historical analog set,
catching off-calendar scenarios that are still dynamically consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def latin_hypercube_sample(
    n_samples: int,
    a: np.ndarray,
    b: np.ndarray,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    Pure-numpy Latin Hypercube Sampling in D-dimensional box [a, b].

    For each of D dimensions, creates n_samples evenly-spaced strata
    [0/n, 1/n, ..., (n-1)/n], shuffles each dimension independently,
    then scales to [a, b].

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    a : np.ndarray, shape (D,)
        Lower bounds.
    b : np.ndarray, shape (D,)
        Upper bounds.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : np.ndarray, shape (n_samples, D)
        LHS samples within [a, b].
    """
    rng = np.random.default_rng(rng_seed)
    D = len(a)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Create uniform LHS: for each dimension, strata centers = (k + U) / n, k=0..n-1
    # where U ~ Uniform[0,1]. Then shuffle each column independently.
    samples_unit = np.zeros((n_samples, D), dtype=float)
    for d in range(D):
        # Stratum boundaries: [k/n, (k+1)/n] for k = 0, ..., n-1
        # Draw one uniform sample per stratum
        stratum_starts = np.arange(n_samples, dtype=float) / n_samples
        u = rng.uniform(0.0, 1.0 / n_samples, size=n_samples)
        unit_samples = stratum_starts + u
        # Shuffle this dimension
        rng.shuffle(unit_samples)
        samples_unit[:, d] = unit_samples

    # Scale to [a, b]
    samples = a[np.newaxis, :] + samples_unit * (b - a)[np.newaxis, :]
    return samples


def filter_by_plausibility(
    candidates: np.ndarray,
    prior,                        # VAR1Prior instance
    m0: np.ndarray,
    historical_md_series: np.ndarray,
    max_mahalanobis_pct: float = 90.0,
) -> np.ndarray:
    """
    Filter candidate states by VAR(1) plausibility.

    Keeps only candidates whose Mahalanobis distance from the VAR(1)
    one-step prediction (given m0) is within the max_mahalanobis_pct-th
    percentile of historical one-step Mahalanobis distances.

    Parameters
    ----------
    candidates : np.ndarray, shape (N, D)
        Candidate macro states to filter.
    prior : VAR1Prior
        Fitted VAR(1) prior.
    m0 : np.ndarray, shape (D,)
        Current anchor state (conditioning state for prediction).
    historical_md_series : np.ndarray, shape (T,)
        Historical one-step Mahalanobis distances (from prior.historical_mahalanobis).
    max_mahalanobis_pct : float
        Percentile threshold (default: 90th percentile).

    Returns
    -------
    filtered : np.ndarray, shape (M, D) where M <= N
        Candidates passing the plausibility filter.
    """
    m0 = np.asarray(m0, dtype=float)
    candidates = np.asarray(candidates, dtype=float)

    # Compute threshold from historical distribution
    md_threshold = float(np.percentile(historical_md_series, max_mahalanobis_pct))

    # Compute Mahalanobis for each candidate
    mu_pred = prior.predict_next(m0)
    keep = []
    for i in range(len(candidates)):
        diff = candidates[i] - mu_pred
        md2 = float(diff @ prior.Q_inv @ diff)
        md = float(np.sqrt(max(md2, 0.0)))
        if md <= md_threshold:
            keep.append(i)

    if len(keep) == 0:
        # If nothing passes, return top 20% by Mahalanobis
        mds = []
        for i in range(len(candidates)):
            diff = candidates[i] - mu_pred
            md2 = float(diff @ prior.Q_inv @ diff)
            mds.append(float(np.sqrt(max(md2, 0.0))))
        mds_arr = np.array(mds)
        n_keep = max(1, len(candidates) // 5)
        keep = list(np.argsort(mds_arr)[:n_keep])

    return candidates[keep]


def generate_lhs_candidates(
    prior,             # VAR1Prior
    m0: np.ndarray,
    feature_master: pd.DataFrame,
    macro_cols: list[str],
    n_samples: int = 200,
    n_sigma: float = 3.0,
    rng_seed: int = 42,
    plausibility_pct: float = 90.0,
) -> np.ndarray:
    """
    Generate LHS candidates within VAR(1) plausibility box.

    Parameters
    ----------
    prior : VAR1Prior
    m0 : np.ndarray, shape (D,)
    feature_master : pd.DataFrame
    macro_cols : list[str]
    n_samples : int
    n_sigma : float
        Width of box in innovation std devs (default: 3.0)
    rng_seed : int
    plausibility_pct : float
        Percentile cutoff for plausibility filter.

    Returns
    -------
    filtered_candidates : np.ndarray, shape (M, D)
    """
    # Box from VAR(1) prediction
    a_var1, b_var1 = prior.box_constraints_from_prediction(m0, n_sigma=n_sigma)

    # LHS sample
    raw_candidates = latin_hypercube_sample(n_samples, a_var1, b_var1, rng_seed=rng_seed)

    # Compute historical Mahalanobis for threshold calibration
    hist_md_df = prior.historical_mahalanobis(feature_master, macro_cols)
    hist_mds = hist_md_df["mahalanobis"].to_numpy(dtype=float)

    # Filter by plausibility
    filtered = filter_by_plausibility(
        raw_candidates, prior, m0,
        historical_md_series=hist_mds,
        max_mahalanobis_pct=plausibility_pct,
    )

    return filtered
