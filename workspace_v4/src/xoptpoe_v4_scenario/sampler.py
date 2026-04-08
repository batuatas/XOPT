"""
sampler.py  —  v3

Preconditioned MALA to fix anisotropic loss landscape.

Root cause of ESS min=4.9 despite 15-20% acceptance:
  The VAR(1) Q_inv has max eigenvalue=100, creating a landscape where
  some macro dims (ig_oas, us_real10y, vix) have gradients ~100x larger
  than others.  Isotropic MALA oscillates in those dims without mixing.

Fix: diagonal preconditioning by historical std (scales vector).
  Standard MALA:       m_prop = m - eta * grad + sqrt(2*tau*eta) * xi
  Preconditioned MALA: m_prop = m - eta * P * grad + sqrt(2*tau*eta) * sqrt(P) * xi
  where P = diag(scales^2)  (scales = per-dim historical std)

This is equivalent to sampling in the normalized space z = m / scales,
then transforming back.  All dims become ~unit variance.

The MH correction is updated accordingly to maintain detailed balance.

Expected improvement: ESS min 4.9 -> 50-150.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _clamp(m: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(m, a, b)


def mala_chain(
    m_init: np.ndarray,
    G: Callable[[np.ndarray], float],
    gradG: Callable[[np.ndarray], np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    n_steps: int = 500,
    eta: float = 0.15,
    tau: float = 2.0,
    precond: np.ndarray | None = None,   # NEW: diagonal preconditioner (scales^2)
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run a single preconditioned MALA chain from m_init.

    Preconditioned update rule:
      m_prop = m - eta * P * grad(m) + sqrt(2 * tau * eta) * sqrt(P) * xi
      xi ~ N(0, I)

    where P = diag(precond) = diag(scales^2).

    MH log-acceptance ratio (corrected for preconditioner):
      A = (G(m) - G(m_prop)) / tau
      B = (||m_prop - m + eta*P*grad(m)||_{P^{-1}}^2
           - ||m - m_prop + eta*P*grad(m_prop)||_{P^{-1}}^2) / (8*tau*eta)

    When precond=None, falls back to standard isotropic MALA (P=I).

    Parameters
    ----------
    precond : np.ndarray, shape (D,), or None
        Diagonal preconditioner values.  Pass scales**2 from state_scales().
        None = isotropic (standard MALA).

    Returns
    -------
    m_last : np.ndarray, shape (D,)
    trajectory : np.ndarray, shape (n_steps, D)
    acceptance_rate : float
    """
    if rng is None:
        rng = np.random.default_rng()

    D = len(m_init)
    m = _clamp(np.asarray(m_init, dtype=float).copy(), a, b)
    trajectory = np.zeros((n_steps, D), dtype=float)

    # Build preconditioner arrays
    if precond is not None:
        P = np.asarray(precond, dtype=float)          # shape (D,)
        P = np.maximum(P, 1e-8)                        # safety floor
        sqrt_P = np.sqrt(P)                            # for noise scaling
        inv_P = 1.0 / P                                # for MH correction
    else:
        P = np.ones(D, dtype=float)
        sqrt_P = np.ones(D, dtype=float)
        inv_P = np.ones(D, dtype=float)

    G_curr = G(m)
    grad_curr = gradG(m)

    accepted = 0
    noise_scale = float(np.sqrt(2.0 * tau * eta))

    for step in range(n_steps):
        xi = rng.standard_normal(D)

        # Preconditioned proposal
        m_prop = m - eta * P * grad_curr + noise_scale * sqrt_P * xi
        m_prop = _clamp(m_prop, a, b)

        G_prop = G(m_prop)
        grad_prop = gradG(m_prop)

        # MH acceptance ratio with preconditioner correction
        A = (G_curr - G_prop) / tau

        # Proposal density correction: q(m|m_prop) / q(m_prop|m)
        # Forward:  m_prop ~ N(m - eta*P*grad_curr, 2*tau*eta*P)
        # Reverse:  m      ~ N(m_prop - eta*P*grad_prop, 2*tau*eta*P)
        diff_fwd = m_prop - m + eta * P * grad_curr    # shape (D,)
        diff_rev = m - m_prop + eta * P * grad_prop    # shape (D,)

        # Mahalanobis norm under P^{-1}: ||v||_{P^{-1}}^2 = v^T P^{-1} v
        norm_fwd = float(np.dot(diff_fwd * inv_P, diff_fwd))
        norm_rev = float(np.dot(diff_rev * inv_P, diff_rev))
        B = (norm_fwd - norm_rev) / (8.0 * tau * eta)

        log_alpha = min(0.0, A + B)
        u = float(rng.uniform())

        if np.log(u + 1e-300) < log_alpha:
            m = m_prop
            G_curr = G_prop
            grad_curr = grad_prop
            accepted += 1

        trajectory[step] = m

    acceptance_rate = accepted / n_steps
    if verbose:
        print(f"  MALA acceptance rate: {acceptance_rate:.2%}")

    return m, trajectory, acceptance_rate


def run_mala_chains(
    G: Callable[[np.ndarray], float],
    gradG: Callable[[np.ndarray], np.ndarray],
    m0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_seeds: int = 5,
    n_steps: int = 500,
    eta: float = 0.15,
    tau: float = 2.0,
    warmup_frac: float = 0.40,
    seed: int = 42,
    precond: np.ndarray | None = None,   # NEW: pass scales**2 here
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Run n_seeds independent preconditioned MALA chains.

    Parameters
    ----------
    precond : np.ndarray, shape (D,), or None
        Diagonal preconditioner.  Pass scales**2 from state_scales().
        Strongly recommended — fixes ESS collapse in anisotropic landscapes.

    Returns
    -------
    trajectories : list of np.ndarray, each shape (n_post_warmup, D)
    acceptance_rates : list of float
    """
    rng = np.random.default_rng(seed)
    D = len(m0)
    warmup = int(n_steps * warmup_frac)

    trajectories = []
    acceptance_rates = []
    n_near = n_seeds // 2

    for i in range(n_seeds):
        if i < n_near:
            noise = rng.standard_normal(D) * 0.1 * np.maximum(b - a, 1e-6)
            m_start = np.clip(m0 + noise, a, b)
            start_type = "near m0"
        else:
            m_start = a + (b - a) * rng.uniform(size=D)
            start_type = "random"

        if verbose:
            print(f"  Chain {i+1}/{n_seeds} starting {start_type} | "
                  f"tau={tau:.3f}, eta={eta:.4f}, "
                  f"precond={'yes' if precond is not None else 'no'}")

        _, traj, acc_rate = mala_chain(
            m_init=m_start,
            G=G,
            gradG=gradG,
            a=a,
            b=b,
            n_steps=n_steps,
            eta=eta,
            tau=tau,
            precond=precond,
            rng=rng,
            verbose=False,
        )
        trajectories.append(traj[warmup:])
        acceptance_rates.append(acc_rate)

        if verbose:
            print(f"    acceptance_rate={acc_rate:.2%} | "
                  f"post-warmup steps={len(traj[warmup:])}")

    mean_acc = float(np.mean(acceptance_rates))
    if verbose:
        print(f"  Mean acceptance rate across {n_seeds} chains: {mean_acc:.2%}")
        if mean_acc < 0.10:
            print("  WARNING: acceptance rate very low — reduce eta or increase tau")
        elif mean_acc > 0.60:
            print("  NOTE: acceptance rate high — increase eta for more exploration")

    return trajectories, acceptance_rates


def thin_only(
    trajectories: list[np.ndarray],
    thinning: int = 5,
) -> np.ndarray:
    """Return every k-th sample from all chains without G filtering."""
    all_samples = []
    for traj in trajectories:
        all_samples.append(traj[::thinning])
    if not all_samples:
        return np.empty((0, trajectories[0].shape[1]))
    return np.vstack(all_samples)


def filter_trajectories(
    trajectories: list[np.ndarray],
    G: Callable[[np.ndarray], float],
    G_threshold: float | None,
    thinning: int = 5,
) -> np.ndarray:
    """
    Filter and thin trajectory samples.

    G_threshold=None -> keep ALL thinned samples (proper posterior mode).
    G_threshold=float -> keep only samples where G(m) < threshold.
    """
    if G_threshold is None:
        return thin_only(trajectories, thinning=thinning)

    all_samples = []
    for traj in trajectories:
        for t in range(0, len(traj), thinning):
            m = traj[t]
            if G(m) < G_threshold:
                all_samples.append(m)

    if not all_samples:
        return np.empty((0, trajectories[0].shape[1]))
    return np.vstack(all_samples)


def compute_effective_sample_size(
    samples: np.ndarray,
    max_lag: int = 50,
) -> np.ndarray:
    """
    Estimate ESS per dimension via autocorrelation.
    ESS_i = N / (1 + 2 * sum_{k=1}^{K} rho_k_i)
    """
    N, D = samples.shape
    ess = np.zeros(D)
    for d in range(D):
        x = samples[:, d] - samples[:, d].mean()
        var = float(np.var(x))
        if var < 1e-12:
            ess[d] = 1.0
            continue
        rho_sum = 0.0
        for k in range(1, min(max_lag + 1, N)):
            rho_k = float(np.mean(x[:-k] * x[k:])) / var
            if abs(rho_k) < 0.05:
                break
            rho_sum += rho_k
        ess[d] = N / max(1.0 + 2.0 * rho_sum, 1.0)
    return ess
