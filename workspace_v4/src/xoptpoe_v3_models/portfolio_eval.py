"""Portfolio-construction diagnostics for the v3 long-horizon shared-horizon setup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from xoptpoe_v3_models.data import SLEEVE_ORDER, aggregate_horizon_values
from xoptpoe_v3_models.optim_layers import OptimizerConfig, RobustOptimizerCache


@dataclass(frozen=True)
class PortfolioRunResult:
    """Decision-period portfolio returns and sleeve weights."""

    returns: pd.DataFrame
    weights: pd.DataFrame



def build_monthly_signal_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level 60m/120m predictions into a single monthly SAA signal."""
    required = {"split", "month_end", "sleeve_id", "horizon_months", "y_true", "y_pred", "benchmark_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns for portfolio evaluation: {sorted(missing)}")

    split_values = predictions["split"].dropna().unique().tolist()
    if len(split_values) != 1:
        raise ValueError("build_monthly_signal_panel expects a single split at a time")
    split_name = str(split_values[0])

    pred_panel = aggregate_horizon_values(predictions, "y_pred")
    true_panel = aggregate_horizon_values(predictions, "y_true")
    bench_panel = aggregate_horizon_values(predictions, "benchmark_pred")

    out = pred_panel.merge(true_panel, on=["month_end", "sleeve_id"], how="inner", validate="1:1")
    out = out.merge(bench_panel, on=["month_end", "sleeve_id"], how="inner", validate="1:1")
    out["split"] = split_name
    out = out.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)
    return out



def run_portfolio_evaluation(
    *,
    signal_panel: pd.DataFrame,
    optimizer_cache: RobustOptimizerCache,
    optimizer_config: OptimizerConfig,
    model_strategy_name: str,
) -> PortfolioRunResult:
    """Evaluate equal-weight and model-driven long-only portfolios on monthly decision instances."""
    required = {"split", "month_end", "sleeve_id", "y_pred", "y_true"}
    missing = required - set(signal_panel.columns)
    if missing:
        raise ValueError(f"signal_panel missing required columns: {sorted(missing)}")

    returns_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    previous_weights: dict[str, np.ndarray] = {}

    split_name = str(signal_panel["split"].iloc[0])
    for month_end, chunk in signal_panel.groupby("month_end", sort=True):
        ordered = chunk.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
        if ordered[["y_pred", "y_true"]].isna().any().any():
            raise ValueError(f"Missing aggregated signal values at month_end={month_end}")

        mu_pred = torch.tensor(ordered["y_pred"].to_numpy(dtype=np.float32), dtype=torch.float32)
        model_weights = optimizer_cache.solve(pd.Timestamp(month_end), mu_pred, optimizer_config).detach().cpu().numpy()
        eq_weights = np.repeat(1.0 / len(SLEEVE_ORDER), len(SLEEVE_ORDER))

        strategies = {
            "equal_weight": eq_weights,
            model_strategy_name: model_weights,
        }
        realized = ordered["y_true"].to_numpy(dtype=float)
        for strategy_name, weights in strategies.items():
            weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
            weights = weights / weights.sum()
            port_return = float(np.dot(weights, realized))
            prev = previous_weights.get(strategy_name)
            turnover = 0.0 if prev is None else float(0.5 * np.abs(weights - prev).sum())
            previous_weights[strategy_name] = weights.copy()
            returns_rows.append(
                {
                    "split": split_name,
                    "month_end": pd.Timestamp(month_end),
                    "strategy": strategy_name,
                    "portfolio_annualized_excess_return": port_return,
                    "turnover": turnover,
                }
            )
            for sleeve_id, weight in zip(SLEEVE_ORDER, weights, strict=True):
                weight_rows.append(
                    {
                        "split": split_name,
                        "month_end": pd.Timestamp(month_end),
                        "strategy": strategy_name,
                        "sleeve_id": sleeve_id,
                        "weight": float(weight),
                    }
                )

    returns_df = pd.DataFrame(returns_rows).sort_values(["strategy", "month_end"]).reset_index(drop=True)
    returns_df["gross_return"] = np.maximum(1.0 + returns_df["portfolio_annualized_excess_return"], 1e-6)
    returns_df["cum_nav"] = returns_df.groupby("strategy")["gross_return"].cumprod()
    running_peak = returns_df.groupby("strategy")["cum_nav"].cummax()
    returns_df["drawdown"] = returns_df["cum_nav"] / running_peak - 1.0

    weights_df = pd.DataFrame(weight_rows).sort_values(["strategy", "month_end", "sleeve_id"]).reset_index(drop=True)
    return PortfolioRunResult(returns=returns_df, weights=weights_df)



def summarize_portfolio_metrics(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy-level decision-period portfolio diagnostics."""
    required = {
        "split",
        "strategy",
        "month_end",
        "portfolio_annualized_excess_return",
        "turnover",
        "drawdown",
        "cum_nav",
    }
    missing = required - set(returns_df.columns)
    if missing:
        raise ValueError(f"returns_df missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (split_name, strategy_name), chunk in returns_df.groupby(["split", "strategy"], as_index=False):
        values = chunk["portfolio_annualized_excess_return"].to_numpy(dtype=float)
        mean_return = float(np.mean(values))
        volatility = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        sharpe = float(mean_return / volatility) if np.isfinite(volatility) and volatility > 0.0 else float("nan")
        rows.append(
            {
                "split": split_name,
                "strategy": strategy_name,
                "month_count": int(chunk["month_end"].nunique()),
                "avg_return": mean_return,
                "volatility": volatility,
                "sharpe": sharpe,
                "max_drawdown": float(chunk["drawdown"].min()),
                "avg_turnover": float(chunk["turnover"].mean()),
                "ending_nav": float(chunk["cum_nav"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "strategy"]).reset_index(drop=True)
