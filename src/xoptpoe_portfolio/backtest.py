"""Monthly long-only portfolio backtest engine for XOPTPOE v1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xoptpoe_modeling.eda import TARGET_COL
from xoptpoe_portfolio.weights import equal_weight, mean_variance_clipped, score_positive, top_k_equal


@dataclass(frozen=True)
class BacktestConfig:
    """Config for simple monthly portfolio backtests."""

    top_k: int = 3
    cov_lookback_months: int = 60
    cov_min_months: int = 24
    mv_ridge: float = 1e-3
    mv_risk_aversion: float = 1.0


def choose_signal_model(
    *,
    metrics_overall: pd.DataFrame,
    requested_model: str | None = None,
) -> str:
    """Choose prediction model by explicit request or best validation RMSE."""
    if requested_model is not None:
        if requested_model not in set(metrics_overall["model"].unique()):
            raise ValueError(f"Requested model '{requested_model}' not found in baseline metrics")
        return requested_model

    validation = metrics_overall.loc[metrics_overall["split"] == "validation"].copy()
    if validation.empty:
        raise ValueError("No validation rows found in baseline metrics")
    return str(validation.sort_values("rmse").iloc[0]["model"])


def _pivot_predictions(
    *,
    predictions_test: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot predictions and realized targets by month-end x sleeve."""
    required = {"model", "month_end", "sleeve_id", "y_true", "y_pred"}
    missing = required - set(predictions_test.columns)
    if missing:
        raise ValueError(f"predictions_test missing required columns: {sorted(missing)}")

    work = predictions_test.loc[predictions_test["model"] == model_name].copy()
    if work.empty:
        raise ValueError(f"No test predictions found for model={model_name}")

    work["month_end"] = pd.to_datetime(work["month_end"])
    dup_cnt = int(work.duplicated(subset=["month_end", "sleeve_id"]).sum())
    if dup_cnt > 0:
        raise ValueError(f"Duplicate prediction keys for model={model_name}: {dup_cnt}")

    pred = work.pivot(index="month_end", columns="sleeve_id", values="y_pred").sort_index()
    true = work.pivot(index="month_end", columns="sleeve_id", values="y_true").sort_index()
    return pred, true


def _estimate_covariance(
    *,
    realized_history: pd.DataFrame,
    current_month: pd.Timestamp,
    sleeves: list[str],
    lookback_months: int,
    min_months: int,
) -> pd.DataFrame:
    """Estimate trailing covariance from realized historical excess returns."""
    hist = realized_history.loc[realized_history["month_end"] < current_month].copy()
    if hist.empty:
        return pd.DataFrame(np.eye(len(sleeves)), index=sleeves, columns=sleeves)

    trailing_months = sorted(hist["month_end"].unique())[-lookback_months:]
    hist = hist.loc[hist["month_end"].isin(trailing_months)]
    panel = hist.pivot(index="month_end", columns="sleeve_id", values=TARGET_COL)
    panel = panel.reindex(columns=sleeves)

    if panel.shape[0] < min_months:
        return pd.DataFrame(np.eye(len(sleeves)), index=sleeves, columns=sleeves)

    cov = panel.cov(min_periods=min_months).fillna(0.0)
    # Guard against numerical issues by setting tiny diagonal floor.
    diag_floor = np.maximum(np.diag(cov.to_numpy()), 1e-8)
    cov.values[np.diag_indices_from(cov.values)] = diag_floor
    return cov


def _validate_weight_vector(weights: pd.Series, *, atol: float = 1e-8) -> None:
    """Ensure weight vector satisfies long-only + fully-invested constraints."""
    if (weights < -atol).any():
        raise ValueError("Weight vector contains negative values")
    if abs(float(weights.sum()) - 1.0) > atol:
        raise ValueError(f"Weight vector is not fully invested, sum={weights.sum()}")


def run_portfolio_backtest(
    *,
    predictions_test: pd.DataFrame,
    realized_history: pd.DataFrame,
    signal_model: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run monthly backtest for simple long-only portfolio variants."""
    pred_mat, true_mat = _pivot_predictions(predictions_test=predictions_test, model_name=signal_model)
    months = list(pred_mat.index)
    sleeves = sorted(pred_mat.columns.tolist())

    # Ensure complete monthly cross-sections in test panel.
    if pred_mat.isna().any().any() or true_mat.isna().any().any():
        raise ValueError("Test prediction panel has missing values; cannot run full-invested cross-sectional backtest")

    strategies = ("equal_weight", "top_k_equal", "score_positive", "mv_clipped")
    prev_w: dict[str, pd.Series] = {}
    returns_rows: list[dict] = []
    weights_rows: list[dict] = []

    hist = realized_history.copy()
    hist["month_end"] = pd.to_datetime(hist["month_end"])

    for month_end in months:
        pred_scores = pred_mat.loc[month_end, sleeves]
        realized = true_mat.loc[month_end, sleeves]

        cov = _estimate_covariance(
            realized_history=hist,
            current_month=month_end,
            sleeves=sleeves,
            lookback_months=config.cov_lookback_months,
            min_months=config.cov_min_months,
        )

        weight_map = {
            "equal_weight": equal_weight(pred_scores),
            "top_k_equal": top_k_equal(pred_scores, k=config.top_k),
            "score_positive": score_positive(pred_scores),
            "mv_clipped": mean_variance_clipped(
                pred_scores,
                cov,
                ridge=config.mv_ridge,
                risk_aversion=config.mv_risk_aversion,
            ),
        }

        for strategy in strategies:
            w = weight_map[strategy].reindex(sleeves)
            _validate_weight_vector(w)

            port_ret = float(np.dot(w.to_numpy(dtype=float), realized.to_numpy(dtype=float)))
            if strategy in prev_w:
                turnover = float(0.5 * np.abs(w.to_numpy() - prev_w[strategy].to_numpy()).sum())
            else:
                turnover = 0.0
            prev_w[strategy] = w

            returns_rows.append(
                {
                    "month_end": month_end,
                    "strategy": strategy,
                    "signal_model": signal_model,
                    "portfolio_excess_return": port_ret,
                    "turnover": turnover,
                }
            )
            for sleeve_id, weight in w.items():
                weights_rows.append(
                    {
                        "month_end": month_end,
                        "strategy": strategy,
                        "signal_model": signal_model,
                        "sleeve_id": sleeve_id,
                        "weight": float(weight),
                    }
                )

    returns_df = pd.DataFrame(returns_rows).sort_values(["strategy", "month_end"]).reset_index(drop=True)
    weights_df = pd.DataFrame(weights_rows).sort_values(["strategy", "month_end", "sleeve_id"]).reset_index(drop=True)

    # Derived performance paths.
    returns_df["gross_return"] = 1.0 + returns_df["portfolio_excess_return"]
    returns_df["cum_nav"] = returns_df.groupby("strategy")["gross_return"].cumprod()
    running_peak = returns_df.groupby("strategy")["cum_nav"].cummax()
    returns_df["drawdown"] = returns_df["cum_nav"] / running_peak - 1.0
    return returns_df, weights_df


def summarize_performance(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy-level performance and benchmark-relative stats."""
    required = {"month_end", "strategy", "portfolio_excess_return", "turnover", "drawdown", "cum_nav"}
    missing = required - set(returns_df.columns)
    if missing:
        raise ValueError(f"returns_df missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    benchmark = (
        returns_df.loc[returns_df["strategy"] == "equal_weight", ["month_end", "portfolio_excess_return"]]
        .rename(columns={"portfolio_excess_return": "benchmark_ret"})
        .set_index("month_end")
    )

    for strategy, grp in returns_df.groupby("strategy"):
        r = grp["portfolio_excess_return"].to_numpy(dtype=float)
        months = int(len(grp))
        mean_r = float(np.mean(r))
        vol_r = float(np.std(r, ddof=1)) if months > 1 else float("nan")
        sharpe = float((mean_r / vol_r) * np.sqrt(12.0)) if vol_r and vol_r > 0 else float("nan")
        max_dd = float(grp["drawdown"].min())
        avg_turnover = float(grp["turnover"].mean())
        end_nav = float(grp["cum_nav"].iloc[-1])

        cmp = grp.set_index("month_end").join(benchmark, how="left")
        active = cmp["portfolio_excess_return"] - cmp["benchmark_ret"]
        active_mean = float(active.mean())
        active_te = float(active.std(ddof=1))
        info_ratio = float((active_mean / active_te) * np.sqrt(12.0)) if active_te and active_te > 0 else float("nan")

        rows.append(
            {
                "strategy": strategy,
                "months": months,
                "avg_monthly_return": mean_r,
                "vol_monthly": vol_r,
                "sharpe_annualized": sharpe,
                "max_drawdown": max_dd,
                "avg_turnover": avg_turnover,
                "cum_nav_end": end_nav,
                "avg_active_return_vs_equal_weight": active_mean,
                "tracking_error_vs_equal_weight": active_te,
                "information_ratio_vs_equal_weight": info_ratio,
            }
        )

    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def summarize_weights(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize portfolio weight distributions by strategy and sleeve."""
    required = {"strategy", "sleeve_id", "weight"}
    missing = required - set(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df missing required columns: {sorted(missing)}")

    out = (
        weights_df.groupby(["strategy", "sleeve_id"], as_index=False)
        .agg(
            avg_weight=("weight", "mean"),
            std_weight=("weight", "std"),
            min_weight=("weight", "min"),
            max_weight=("weight", "max"),
            nonzero_share=("weight", lambda s: float((s > 1e-12).mean())),
        )
        .sort_values(["strategy", "sleeve_id"])
        .reset_index(drop=True)
    )
    return out
