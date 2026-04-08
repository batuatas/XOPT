"""Small bounded MALA implementation for the v3 scenario scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MalaResult:
    """Trace summary from a small MALA run."""

    states: np.ndarray
    energies: np.ndarray
    acceptance_rate: float


def finite_difference_gradient(
    energy_fn,
    x: np.ndarray,
    *,
    step: float = 1e-4,
) -> np.ndarray:
    """Central-difference gradient for non-autodiff scenario pipelines."""
    base = np.asarray(x, dtype=float)
    grad = np.zeros_like(base)
    for idx in range(len(base)):
        direction = np.zeros_like(base)
        direction[idx] = step
        e_plus = float(energy_fn(base + direction))
        e_minus = float(energy_fn(base - direction))
        grad[idx] = (e_plus - e_minus) / (2.0 * step)
    return grad


def _gaussian_logpdf(x: np.ndarray, mean: np.ndarray, variance: float) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)
    dim = len(diff)
    return float(-0.5 * dim * np.log(2.0 * np.pi * variance) - 0.5 * diff.T @ diff / variance)


def run_bounded_mala(
    *,
    start: np.ndarray,
    energy_fn,
    project_fn,
    gradient_fn,
    step_size: float,
    n_steps: int,
    random_seed: int = 42,
) -> MalaResult:
    """Run a tiny bounded MALA chain for smoke testing."""
    rng = np.random.default_rng(random_seed)
    current = np.asarray(project_fn(start), dtype=float)
    current_energy = float(energy_fn(current))
    states = [current.copy()]
    energies = [current_energy]
    accept_count = 0

    variance = float(step_size**2)
    drift_scale = 0.5 * variance

    for _ in range(int(n_steps)):
        grad = np.asarray(gradient_fn(current), dtype=float)
        mean_forward = current - drift_scale * grad
        proposal = mean_forward + step_size * rng.normal(size=current.shape[0])
        proposal = np.asarray(project_fn(proposal), dtype=float)
        proposal_energy = float(energy_fn(proposal))
        if not np.isfinite(proposal_energy):
            states.append(current.copy())
            energies.append(current_energy)
            continue

        grad_prop = np.asarray(gradient_fn(proposal), dtype=float)
        mean_reverse = proposal - drift_scale * grad_prop
        log_alpha = (
            -proposal_energy
            + current_energy
            + _gaussian_logpdf(current, mean_reverse, variance)
            - _gaussian_logpdf(proposal, mean_forward, variance)
        )
        if np.log(rng.uniform()) < min(0.0, log_alpha):
            current = proposal
            current_energy = proposal_energy
            accept_count += 1

        states.append(current.copy())
        energies.append(current_energy)

    return MalaResult(
        states=np.asarray(states, dtype=float),
        energies=np.asarray(energies, dtype=float),
        acceptance_rate=float(accept_count / max(1, n_steps)),
    )
