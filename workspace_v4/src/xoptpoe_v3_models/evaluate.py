"""Prediction-metric evaluation helpers for XOPTPOE v3 models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error



def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])



def _safe_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true >= 0.0) == (y_pred >= 0.0)))



def _safe_oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_benchmark: np.ndarray) -> float:
    sse_model = float(np.sum((y_true - y_pred) ** 2))
    sse_benchmark = float(np.sum((y_true - y_benchmark) ** 2))
    if sse_benchmark <= 0.0:
        return float("nan")
    return 1.0 - (sse_model / sse_benchmark)



def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_benchmark: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    corr = _safe_corr(y_true, y_pred)
    dir_acc = _safe_directional_accuracy(y_true, y_pred)
    oos_r2 = _safe_oos_r2(y_true, y_pred, y_benchmark)
    return {
        "rmse": rmse,
        "mae": mae,
        "oos_r2_vs_naive": oos_r2,
        "corr": corr,
        "directional_accuracy": dir_acc,
    }



def evaluate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate pooled, sleeve-level, and horizon-level prediction metrics."""
    required = {"split", "sleeve_id", "horizon_months", "y_true", "y_pred", "benchmark_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    records_overall: list[dict[str, object]] = []
    for split_name, chunk in predictions.groupby("split", as_index=False):
        metrics = regression_metrics(
            chunk["y_true"].to_numpy(dtype=float),
            chunk["y_pred"].to_numpy(dtype=float),
            chunk["benchmark_pred"].to_numpy(dtype=float),
        )
        records_overall.append({"split": split_name, "row_count": int(len(chunk)), **metrics})

    records_sleeve: list[dict[str, object]] = []
    for (split_name, sleeve_id), chunk in predictions.groupby(["split", "sleeve_id"], as_index=False):
        metrics = regression_metrics(
            chunk["y_true"].to_numpy(dtype=float),
            chunk["y_pred"].to_numpy(dtype=float),
            chunk["benchmark_pred"].to_numpy(dtype=float),
        )
        records_sleeve.append(
            {"split": split_name, "sleeve_id": sleeve_id, "row_count": int(len(chunk)), **metrics}
        )

    records_horizon: list[dict[str, object]] = []
    for (split_name, horizon_months), chunk in predictions.groupby(["split", "horizon_months"], as_index=False):
        metrics = regression_metrics(
            chunk["y_true"].to_numpy(dtype=float),
            chunk["y_pred"].to_numpy(dtype=float),
            chunk["benchmark_pred"].to_numpy(dtype=float),
        )
        records_horizon.append(
            {
                "split": split_name,
                "horizon_months": int(horizon_months),
                "row_count": int(len(chunk)),
                **metrics,
            }
        )

    overall = pd.DataFrame(records_overall).sort_values("split").reset_index(drop=True)
    by_sleeve = pd.DataFrame(records_sleeve).sort_values(["split", "sleeve_id"]).reset_index(drop=True)
    by_horizon = pd.DataFrame(records_horizon).sort_values(["split", "horizon_months"]).reset_index(drop=True)
    return overall, by_sleeve, by_horizon
