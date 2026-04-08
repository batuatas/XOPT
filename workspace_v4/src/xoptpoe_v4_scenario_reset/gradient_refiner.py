"""
gradient_refiner.py

Stage 2: Bounded Gradient Descent Refinement.

For each candidate macro state from Stage 1, runs a short deterministic
gradient descent (no stochastic noise) to improve the question objective
while staying within the VAR(1) plausibility box.

This is a focused local improvement, NOT a sampler.
Deterministic: reproducible results.
Fast: 50-100 steps per candidate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _project_to_box(m: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Project m onto [a, b] box (element-wise clip)."""
    return np.clip(m, a, b)


def refine_candidate(
    m_init: np.ndarray,
    G: Callable[[np.ndarray], float],
    gradG: Callable[[np.ndarray], np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    n_steps: int = 60,
    lr: float = 0.003,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Pure gradient descent with box projection.

    Starting from m_init, takes n_steps gradient descent steps
    to minimize G(m), projecting back onto [a, b] after each step.

    Uses backtracking line search to avoid divergence.
    No stochastic noise — deterministic and reproducible.

    Parameters
    ----------
    m_init : np.ndarray, shape (D,)
        Initial macro state.
    G : callable
        Objective function G(m) -> float (minimize).
    gradG : callable
        Gradient of G: gradG(m) -> np.ndarray, shape (D,)
    a, b : np.ndarray, shape (D,)
        Box constraints.
    n_steps : int
        Number of gradient descent steps.
    lr : float
        Learning rate (step size). Reduced by backtracking if G increases.

    Returns
    -------
    m_refined : np.ndarray, shape (D,)
        Refined macro state.
    G_final : float
        Objective value at refined state.
    trajectory : list[float]
        G values along the descent path (for convergence diagnostics).
    """
    m = np.asarray(m_init, dtype=float).copy()
    m = _project_to_box(m, a, b)

    G_current = G(m)
    trajectory = [G_current]

    for step in range(n_steps):
        grad = gradG(m)

        # Backtracking line search: try full step, reduce if G doesn't decrease
        step_lr = lr
        m_candidate = _project_to_box(m - step_lr * grad, a, b)
        G_candidate = G(m_candidate)

        n_backtrack = 0
        while G_candidate >= G_current and n_backtrack < 3:
            step_lr *= 0.5
            m_candidate = _project_to_box(m - step_lr * grad, a, b)
            G_candidate = G(m_candidate)
            n_backtrack += 1

        # Accept step (even if no improvement — gradient descent continues)
        m = m_candidate
        G_current = G_candidate
        trajectory.append(G_current)

        # Early stopping: if gradient is very small, we're at a local minimum
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-6:
            break

    return m, G_current, trajectory


def refine_batch(
    candidates_m: list[np.ndarray] | np.ndarray,
    G: Callable[[np.ndarray], float],
    gradG: Callable[[np.ndarray], np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    n_steps: int = 60,
    lr: float = 0.003,
) -> list[tuple[np.ndarray, float, bool]]:
    """
    Refine all candidates in batch.

    Parameters
    ----------
    candidates_m : list of np.ndarray or array, shape (N, D)
        Initial candidate macro states.
    G : callable
    gradG : callable
    a, b : np.ndarray, shape (D,)
    n_steps : int
    lr : float

    Returns
    -------
    results : list of (m_refined, G_final, converged)
        For each input candidate:
        - m_refined: final macro state after refinement
        - G_final: final objective value
        - converged: True if G decreased monotonically in last 10 steps
    """
    if isinstance(candidates_m, np.ndarray) and candidates_m.ndim == 2:
        candidates_list = [candidates_m[i] for i in range(len(candidates_m))]
    else:
        candidates_list = list(candidates_m)

    results = []
    for m_init in candidates_list:
        try:
            m_refined, G_final, trajectory = refine_candidate(
                m_init, G, gradG, a, b, n_steps=n_steps, lr=lr
            )
            # Converged = final G is lower than initial G
            converged = G_final < trajectory[0] if len(trajectory) > 1 else False
            results.append((m_refined, G_final, converged))
        except Exception:
            # If refinement fails, return initial state as-is
            G_init = G(np.asarray(m_init, dtype=float))
            results.append((np.asarray(m_init, dtype=float), G_init, False))

    return results
