"""Evaluation utilities for baseline prediction experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true >= 0) == (y_pred >= 0)))


def _safe_oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_benchmark: np.ndarray) -> float:
    sse_model = float(np.sum((y_true - y_pred) ** 2))
    sse_bench = float(np.sum((y_true - y_benchmark) ** 2))
    if sse_bench == 0:
        return float("nan")
    return 1.0 - (sse_model / sse_bench)


def regression_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_benchmark: np.ndarray,
) -> dict[str, float]:
    """Compute baseline metrics for one prediction set."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    corr = _safe_corr(y_true, y_pred)
    dir_acc = _safe_directional_accuracy(y_true, y_pred)
    oos_r2 = _safe_oos_r2(y_true, y_pred, y_benchmark)
    return {
        "rmse": rmse,
        "mae": mae,
        "oos_r2": oos_r2,
        "corr": corr,
        "directional_accuracy": dir_acc,
    }


def evaluate_overall(
    *,
    predictions: pd.DataFrame,
    train_target: pd.Series,
) -> pd.DataFrame:
    """Evaluate pooled panel metrics by model and split."""
    required = {"model", "split", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    train_mean = float(train_target.mean())
    records: list[dict] = []

    grouped = predictions.groupby(["model", "split"], as_index=False)
    for row in grouped:
        (model_name, split_name), chunk = row
        y_true = chunk["y_true"].to_numpy(dtype=float)
        y_pred = chunk["y_pred"].to_numpy(dtype=float)
        y_bench = np.full_like(y_true, train_mean, dtype=float)
        metrics = regression_metrics(y_true=y_true, y_pred=y_pred, y_benchmark=y_bench)
        records.append(
            {
                "model": model_name,
                "split": split_name,
                "row_count": int(len(chunk)),
                **metrics,
            }
        )

    out = pd.DataFrame(records)
    split_order = pd.Categorical(out["split"], categories=["validation", "test"], ordered=True)
    out["split"] = split_order
    return out.sort_values(["split", "rmse", "model"]).reset_index(drop=True)


def evaluate_by_sleeve(
    *,
    predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Evaluate sleeve-level metrics by model and split."""
    required = {"model", "split", "sleeve_id", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    sleeve_benchmark = train_df.groupby("sleeve_id")[target_col].mean().to_dict()
    records: list[dict] = []

    grouped = predictions.groupby(["model", "split", "sleeve_id"], as_index=False)
    for row in grouped:
        (model_name, split_name, sleeve_id), chunk = row
        y_true = chunk["y_true"].to_numpy(dtype=float)
        y_pred = chunk["y_pred"].to_numpy(dtype=float)
        bench_value = float(sleeve_benchmark[sleeve_id])
        y_bench = np.full_like(y_true, bench_value, dtype=float)
        metrics = regression_metrics(y_true=y_true, y_pred=y_pred, y_benchmark=y_bench)
        records.append(
            {
                "model": model_name,
                "split": split_name,
                "sleeve_id": sleeve_id,
                "row_count": int(len(chunk)),
                **metrics,
            }
        )

    out = pd.DataFrame(records)
    out["split"] = pd.Categorical(out["split"], categories=["validation", "test"], ordered=True)
    return out.sort_values(["split", "model", "sleeve_id"]).reset_index(drop=True)
