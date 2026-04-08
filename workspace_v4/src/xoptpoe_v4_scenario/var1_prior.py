"""
var1_prior.py

VAR(1) macro prior for v4 scenario generation.

Adapted from workspace_v4/var1_regularizer.py to operate on the v4
MACRO_STATE_COLS defined in state_space.py.

Provides:
  - VAR1Prior: fitted VAR(1) model over the compact macro state
  - mahalanobis_regularizer: G_reg(m) = 0.5 * (m - mu_pred)' Q_inv (m - mu_pred)
  - box_constraints_var1: tighter [mu - 3sigma, mu + 3sigma] box from the one-step prediction

Usage:
  prior = VAR1Prior.fit(states_train)
  mu_pred = prior.predict_next(m_t)
  G_reg = prior.regularizer(m_candidate, m_t, l2reg=0.5)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SIGMA_JITTER: float = 0.01   # Tikhonov regularization for Q inversion
# NOTE: The 19-dim macro state includes term_slope = long_rate - short_rate,
# creating near-linear dependencies that make Q near-singular. A jitter of 0.01
# (≈ min historical innovation variance) keeps Q_inv eigenvalues ≤ 100,
# preventing the MALA sampler from rejecting all proposals.


@dataclass
class VAR1Prior:
    """Fitted VAR(1): m_{t+1} = c + A @ m_t + eps, eps ~ N(0, Q)"""
    c: np.ndarray          # shape (D,)
    A: np.ndarray          # shape (D, D)
    Q: np.ndarray          # innovation covariance, shape (D, D)
    Q_inv: np.ndarray      # Q^{-1}, shape (D, D)
    Q_logdet: float
    macro_cols: list[str]

    # ------------------------------------------------------------------ #
    def predict_next(self, m_t: np.ndarray) -> np.ndarray:
        """Conditional mean of m_{t+1} given m_t."""
        return self.c + self.A @ np.asarray(m_t, dtype=float)

    def mahalanobis_sq(self, m_next: np.ndarray, m_t: np.ndarray) -> float:
        """Squared Mahalanobis distance of m_next from the VAR(1) prediction given m_t."""
        mu = self.predict_next(m_t)
        diff = np.asarray(m_next, dtype=float) - mu
        return float(diff @ self.Q_inv @ diff)

    def regularizer(
        self,
        m_candidate: np.ndarray,
        m_anchor: np.ndarray,
        l2reg: float = 0.5,
    ) -> float:
        """
        G_reg(m) = l2reg * 0.5 * (m - mu_pred)' Q_inv (m - mu_pred)

        This replaces the simple L2 anchor term with a dynamically consistent
        Mahalanobis distance from the VAR(1) one-step prediction.
        """
        mu = self.predict_next(m_anchor)
        diff = np.asarray(m_candidate, dtype=float) - mu
        return float(l2reg * 0.5 * (diff @ self.Q_inv @ diff))

    def regularizer_grad(
        self,
        m_candidate: np.ndarray,
        m_anchor: np.ndarray,
        l2reg: float = 0.5,
    ) -> np.ndarray:
        """
        Analytical gradient of regularizer w.r.t. m_candidate.
        ∇_m G_reg = l2reg * Q_inv @ (m - mu_pred)
        """
        mu = self.predict_next(m_anchor)
        diff = np.asarray(m_candidate, dtype=float) - mu
        return l2reg * (self.Q_inv @ diff)

    def box_constraints_from_prediction(
        self,
        m_t: np.ndarray,
        n_sigma: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Tighter box constraints centered on VAR(1) prediction.
        a = mu_pred - n_sigma * sqrt(diag(Q))
        b = mu_pred + n_sigma * sqrt(diag(Q))
        """
        mu = self.predict_next(m_t)
        vol = np.sqrt(np.maximum(np.diag(self.Q), 0.0))
        return mu - n_sigma * vol, mu + n_sigma * vol

    def log_density(self, m_next: np.ndarray, m_t: np.ndarray) -> float:
        """Gaussian log-density under the VAR(1) innovation distribution."""
        mu = self.predict_next(m_t)
        diff = np.asarray(m_next, dtype=float) - mu
        D = len(diff)
        md2 = float(diff @ self.Q_inv @ diff)
        return float(-0.5 * (D * np.log(2.0 * np.pi) + self.Q_logdet + md2))

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        macro_cols: list[str],
        jitter: float = SIGMA_JITTER,
    ) -> "VAR1Prior":
        """
        Fit VAR(1) by OLS: m_{t+1} = c + A @ m_t + eps.

        Parameters
        ----------
        states : np.ndarray, shape (T, D)
            Time-ordered monthly macro state matrix (one row per month).
        macro_cols : list[str]
            Column names for the D dimensions.
        jitter : float
            Small diagonal regularization for Q.
        """
        if len(states) < 3:
            raise ValueError("Need at least 3 time steps to fit VAR(1).")
        x_t = states[:-1]
        x_tp1 = states[1:]
        design = np.column_stack([np.ones(len(x_t)), x_t])
        beta, *_ = np.linalg.lstsq(design, x_tp1, rcond=None)
        c = beta[0, :]
        A = beta[1:, :].T
        residuals = x_tp1 - design @ beta
        Q_raw = np.cov(residuals, rowvar=False, ddof=1)
        Q = Q_raw + jitter * np.eye(Q_raw.shape[0])
        Q = 0.5 * (Q + Q.T)
        Q_inv = np.linalg.pinv(Q)
        sign, Q_logdet = np.linalg.slogdet(Q)
        if sign <= 0:
            Q_logdet = float(np.sum(np.log(np.maximum(np.linalg.eigvalsh(Q), 1e-30))))
        return cls(c=c, A=A, Q=Q, Q_inv=Q_inv, Q_logdet=float(Q_logdet), macro_cols=list(macro_cols))

    @classmethod
    def fit_from_feature_master(
        cls,
        feature_master: pd.DataFrame,
        macro_cols: list[str],
        train_end: pd.Timestamp | None = None,
        jitter: float = SIGMA_JITTER,
    ) -> "VAR1Prior":
        """
        Convenience: fit VAR(1) directly from feature_master_monthly.parquet.

        Uses only training-period rows (up to train_end) and deduplicates
        to one row per date (macro cols are identical across sleeves).
        """
        fm = feature_master.drop_duplicates(subset="month_end").copy()
        fm["month_end"] = pd.to_datetime(fm["month_end"])
        fm = fm.sort_values("month_end").reset_index(drop=True)
        if train_end is not None:
            fm = fm[fm["month_end"] <= pd.Timestamp(train_end)]

        # Build state matrix; fill any NaN with column median
        state_df = fm[macro_cols].copy()
        for col in macro_cols:
            med = state_df[col].median()
            state_df[col] = state_df[col].fillna(med)
        states = state_df.to_numpy(dtype=float)
        return cls.fit(states, macro_cols=macro_cols, jitter=jitter)

    def historical_mahalanobis(
        self,
        feature_master: pd.DataFrame,
        macro_cols: list[str],
    ) -> pd.DataFrame:
        """
        Compute in-sample one-step Mahalanobis distances over the full history.
        Useful for percentile-calibrating the plausibility filter.
        """
        fm = feature_master.drop_duplicates(subset="month_end").copy()
        fm["month_end"] = pd.to_datetime(fm["month_end"])
        fm = fm.sort_values("month_end").reset_index(drop=True)
        state_df = fm[macro_cols].copy()
        for col in macro_cols:
            state_df[col] = state_df[col].fillna(state_df[col].median())
        states = state_df.to_numpy(dtype=float)

        rows = []
        for i in range(1, len(states)):
            mu = self.predict_next(states[i - 1])
            diff = states[i] - mu
            md2 = float(diff @ self.Q_inv @ diff)
            rows.append({
                "month_end": fm["month_end"].iloc[i],
                "mahalanobis": float(np.sqrt(max(md2, 0.0))),
                "mahalanobis_sq": md2,
                "log_density": self.log_density(states[i], states[i - 1]),
            })
        return pd.DataFrame(rows)
