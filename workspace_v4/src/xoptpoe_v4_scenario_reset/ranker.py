"""
ranker.py

Stage 3: Candidate Ranking and Selection.

Ranks refined candidates by a composite score:
  score = alpha * objective_rank + beta * plausibility_rank + gamma * diversity_bonus

Then selects top N ensuring regime diversity (at least 2 distinct regimes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_candidates(
    candidates_df: pd.DataFrame,
    prior,           # VAR1Prior instance
    m0: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> pd.DataFrame:
    """
    Compute composite score for each candidate.

    Score components:
    - objective_rank: rank by G_final (lower G = better, so lower rank = better)
    - plausibility_rank: rank by Mahalanobis distance from VAR(1) prediction
    - diversity_bonus: reward for being far from other candidates (encourages spread)

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Must have columns: <MACRO_STATE_COLS>, G_final
    prior : VAR1Prior
    m0 : np.ndarray, shape (D,)
    alpha, beta, gamma : float
        Weights for composite score (must sum to 1.0)

    Returns
    -------
    candidates_df with added columns:
        mahalanobis_dist, objective_rank, plausibility_rank, diversity_bonus, composite_score
    """
    from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS

    df = candidates_df.copy()
    N = len(df)

    if N == 0:
        return df

    # Extract macro state matrix
    state_mat = df[MACRO_STATE_COLS].fillna(0.0).to_numpy(dtype=float)

    # --- Plausibility: Mahalanobis distance from VAR(1) prediction ---
    mu_pred = prior.predict_next(np.asarray(m0, dtype=float))
    mds = []
    for i in range(N):
        diff = state_mat[i] - mu_pred
        md2 = float(diff @ prior.Q_inv @ diff)
        mds.append(float(np.sqrt(max(md2, 0.0))))
    df["mahalanobis_dist"] = mds

    # --- Objective rank: rank by G_final (lower = better) ---
    if "G_final" in df.columns:
        df["objective_rank"] = df["G_final"].rank(ascending=True, method="min")
    else:
        df["objective_rank"] = np.arange(1, N + 1)

    # --- Plausibility rank: rank by Mahalanobis (lower = better) ---
    df["plausibility_rank"] = df["mahalanobis_dist"].rank(ascending=True, method="min")

    # --- Diversity bonus: average distance to nearest neighbor ---
    # Higher = more isolated = more diverse contribution
    diversity_scores = np.zeros(N)
    for i in range(N):
        dists = np.sqrt(np.sum((state_mat - state_mat[i]) ** 2, axis=1))
        dists[i] = np.inf  # exclude self
        nn_dist = float(np.min(dists))
        diversity_scores[i] = nn_dist

    # Convert to rank (higher distance = better diversity = lower rank number in diversity)
    # We want gamma * diversity to REDUCE the composite score (lower is better overall)
    diversity_ranks = pd.Series(diversity_scores).rank(ascending=False, method="min").to_numpy()
    df["diversity_bonus"] = diversity_scores
    df["diversity_rank"] = diversity_ranks

    # --- Composite score: normalize each rank to [0, 1] then weight ---
    # Lower composite score = better candidate
    obj_norm = (df["objective_rank"].to_numpy() - 1) / max(N - 1, 1)
    plaus_norm = (df["plausibility_rank"].to_numpy() - 1) / max(N - 1, 1)
    div_norm = (df["diversity_rank"].to_numpy() - 1) / max(N - 1, 1)  # lower rank = more diverse

    df["composite_score"] = alpha * obj_norm + beta * plaus_norm + gamma * div_norm

    return df.sort_values("composite_score").reset_index(drop=True)


def select_diverse(
    scored_df: pd.DataFrame,
    n_select: int = 6,
    min_regimes: int = 2,
    regime_col: str = "regime_label",
) -> pd.DataFrame:
    """
    Select top n_select candidates ensuring regime diversity.

    Strategy:
    1. Take the top-scored candidate unconditionally.
    2. For each subsequent slot:
       - If we have fewer than min_regimes distinct regimes so far,
         try to pick the best-scoring candidate from an unseen regime.
       - Otherwise, pick the next best by composite_score.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of score_candidates (sorted by composite_score ascending).
    n_select : int
        Number of candidates to select.
    min_regimes : int
        Minimum distinct regime labels in selection.
    regime_col : str
        Column name for regime labels.

    Returns
    -------
    selected_df : pd.DataFrame
        Up to n_select rows from scored_df.
    """
    if scored_df.empty:
        return scored_df

    # Ensure sorted
    df = scored_df.sort_values("composite_score").reset_index(drop=True)
    n_available = len(df)

    selected_indices = []
    seen_regimes = set()

    # Greedy selection
    for _ in range(n_select):
        if len(selected_indices) >= n_available:
            break

        remaining = [i for i in range(n_available) if i not in selected_indices]
        if not remaining:
            break

        n_current_regimes = len(seen_regimes)

        if n_current_regimes < min_regimes and regime_col in df.columns:
            # Try to find a candidate from a new regime first
            new_regime_candidates = [
                i for i in remaining
                if df[regime_col].iloc[i] not in seen_regimes
            ]
            if new_regime_candidates:
                # Pick best-scoring from new-regime pool
                chosen = new_regime_candidates[0]  # already sorted by composite_score
            else:
                # No new regimes available; take best remaining
                chosen = remaining[0]
        else:
            # Take best remaining by composite score
            chosen = remaining[0]

        selected_indices.append(chosen)
        if regime_col in df.columns:
            seen_regimes.add(df[regime_col].iloc[chosen])

    return df.iloc[selected_indices].reset_index(drop=True)
