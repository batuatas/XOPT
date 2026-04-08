"""Prediction-focused ablations and diagnostics for XOPTPOE v2 long-horizon modeling."""

from __future__ import annotations

import copy
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import LinAlgWarning
from scipy.stats import ConstantInputWarning
from sklearn.linear_model import ElasticNet, Ridge
from torch.optim import AdamW

from xoptpoe_v2_modeling.features import FEATURE_SET_ORDER
from xoptpoe_v2_models.data import (
    SLEEVE_ORDER,
    TARGET_COL,
    LoadedModelingInputs,
    build_sleeve_horizon_benchmark,
    load_modeling_inputs,
)
from xoptpoe_v2_models.evaluate import regression_metrics
from xoptpoe_v2_models.losses import mse_loss
from xoptpoe_v2_models.networks import PredictorMLP
from xoptpoe_v2_models.preprocess import fit_preprocessor


warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)


HORIZON_MODES: dict[str, tuple[int, ...]] = {
    "shared_60_120": (60, 120),
    "separate_60": (60,),
    "separate_120": (120,),
}

DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
ELASTIC_ALPHA_GRID: tuple[float, ...] = (0.0005, 0.001, 0.005, 0.01, 0.05)
ELASTIC_L1_GRID: tuple[float, ...] = (0.2, 0.5, 0.8)


@dataclass(frozen=True)
class PredictionRunArtifacts:
    """Artifacts from one supervised prediction experiment."""

    experiment_name: str
    model_name: str
    feature_set_name: str
    horizon_mode: str
    horizons: tuple[int, ...]
    predictions_validation: pd.DataFrame
    predictions_test: pd.DataFrame
    metrics_overall: pd.DataFrame
    metrics_by_horizon: pd.DataFrame
    metrics_by_sleeve: pd.DataFrame
    selected_params: dict[str, object]
    input_feature_count: int
    transformed_feature_count: int


@dataclass(frozen=True)
class PredictionAblationResult:
    """Final outputs for the prediction-only ablation study."""

    ablation_results: pd.DataFrame
    horizon_specific_summary: pd.DataFrame
    feature_block_summary: pd.DataFrame
    sleeve_difficulty: pd.DataFrame
    report_text: str


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subset_frame(frame: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = frame.loc[frame["horizon_months"].isin(horizons)].copy()
    if out.empty:
        raise ValueError(f"No rows left after filtering to horizons={horizons}")
    out["month_end"] = pd.to_datetime(out["month_end"])
    out = out.sort_values(["month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)
    return out


def _subset_inputs(inputs: LoadedModelingInputs, horizons: tuple[int, ...]) -> LoadedModelingInputs:
    return LoadedModelingInputs(
        train_df=_subset_frame(inputs.train_df, horizons),
        validation_df=_subset_frame(inputs.validation_df, horizons),
        test_df=_subset_frame(inputs.test_df, horizons),
        feature_manifest=inputs.feature_manifest,
        feature_columns=inputs.feature_columns,
        monthly_excess_history=inputs.monthly_excess_history,
    )


def _with_feature_columns(inputs: LoadedModelingInputs, feature_columns: list[str]) -> LoadedModelingInputs:
    return LoadedModelingInputs(
        train_df=inputs.train_df,
        validation_df=inputs.validation_df,
        test_df=inputs.test_df,
        feature_manifest=inputs.feature_manifest,
        feature_columns=list(feature_columns),
        monthly_excess_history=inputs.monthly_excess_history,
    )


def _prediction_frame(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    benchmark: pd.DataFrame,
    *,
    split_name: str,
) -> pd.DataFrame:
    out = frame[["month_end", "sleeve_id", "horizon_months", TARGET_COL]].copy()
    out["split"] = split_name
    out["y_true"] = out[TARGET_COL].astype(float)
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    out = out.drop(columns=[TARGET_COL])
    out = out.merge(benchmark, on=["sleeve_id", "horizon_months"], how="left", validate="m:1")
    return out.sort_values(["month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)


def _quantile_thresholds(train_df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    thresholds: dict[int, dict[str, np.ndarray]] = {}
    for horizon_months, chunk in train_df.groupby("horizon_months"):
        values = chunk[TARGET_COL].to_numpy(dtype=float)
        thresholds[int(horizon_months)] = {
            "tercile": np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]),
            "quintile": np.quantile(values, [0.2, 0.4, 0.6, 0.8]),
        }
    return thresholds


def _apply_bins(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(int)


def _classification_accuracy(
    frame: pd.DataFrame,
    thresholds_by_horizon: dict[int, dict[str, np.ndarray]],
    kind: str,
) -> float:
    correct = 0
    total = 0
    for horizon_months, chunk in frame.groupby("horizon_months"):
        thresholds = thresholds_by_horizon[int(horizon_months)][kind]
        true_bins = _apply_bins(chunk["y_true"].to_numpy(dtype=float), thresholds)
        pred_bins = _apply_bins(chunk["y_pred"].to_numpy(dtype=float), thresholds)
        correct += int(np.sum(true_bins == pred_bins))
        total += int(len(chunk))
    return float(correct / total) if total > 0 else float("nan")


def _mean_rank_ic(frame: pd.DataFrame) -> tuple[float, float]:
    spearman_values: list[float] = []
    pearson_values: list[float] = []
    for _, chunk in frame.groupby(["month_end", "horizon_months"], sort=True):
        corr_s = chunk["y_pred"].corr(chunk["y_true"], method="spearman")
        corr_p = chunk["y_pred"].corr(chunk["y_true"], method="pearson")
        if pd.notna(corr_s):
            spearman_values.append(float(corr_s))
        if pd.notna(corr_p):
            pearson_values.append(float(corr_p))
    mean_s = float(np.mean(spearman_values)) if spearman_values else float("nan")
    mean_p = float(np.mean(pearson_values)) if pearson_values else float("nan")
    return mean_s, mean_p


def _diagnostic_metrics(
    frame: pd.DataFrame,
    thresholds_by_horizon: dict[int, dict[str, np.ndarray]],
) -> dict[str, float]:
    base = regression_metrics(
        frame["y_true"].to_numpy(dtype=float),
        frame["y_pred"].to_numpy(dtype=float),
        frame["benchmark_pred"].to_numpy(dtype=float),
    )
    sign_accuracy = float(np.mean((frame["y_true"] >= 0.0) == (frame["y_pred"] >= 0.0)))
    tercile_accuracy = _classification_accuracy(frame, thresholds_by_horizon, kind="tercile")
    quintile_accuracy = _classification_accuracy(frame, thresholds_by_horizon, kind="quintile")
    rank_ic_spearman, rank_ic_pearson = _mean_rank_ic(frame)
    return {
        **base,
        "sign_accuracy": sign_accuracy,
        "tercile_accuracy": tercile_accuracy,
        "quintile_accuracy": quintile_accuracy,
        "rank_ic_spearman": rank_ic_spearman,
        "rank_ic_pearson": rank_ic_pearson,
    }


def _evaluate_predictions(
    predictions: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = _quantile_thresholds(train_df)

    rows_overall: list[dict[str, object]] = []
    for split_name, chunk in predictions.groupby("split", as_index=False):
        metrics = _diagnostic_metrics(chunk, thresholds)
        rows_overall.append(
            {
                "split": split_name,
                "row_count": int(len(chunk)),
                "month_count": int(chunk["month_end"].nunique()),
                **metrics,
            }
        )

    rows_horizon: list[dict[str, object]] = []
    for (split_name, horizon_months), chunk in predictions.groupby(["split", "horizon_months"], as_index=False):
        metrics = _diagnostic_metrics(chunk, thresholds)
        rows_horizon.append(
            {
                "split": split_name,
                "horizon_months": int(horizon_months),
                "row_count": int(len(chunk)),
                "month_count": int(chunk["month_end"].nunique()),
                **metrics,
            }
        )

    rows_sleeve: list[dict[str, object]] = []
    for (split_name, sleeve_id), chunk in predictions.groupby(["split", "sleeve_id"], as_index=False):
        base = regression_metrics(
            chunk["y_true"].to_numpy(dtype=float),
            chunk["y_pred"].to_numpy(dtype=float),
            chunk["benchmark_pred"].to_numpy(dtype=float),
        )
        sign_accuracy = float(np.mean((chunk["y_true"] >= 0.0) == (chunk["y_pred"] >= 0.0)))
        rows_sleeve.append(
            {
                "split": split_name,
                "sleeve_id": sleeve_id,
                "row_count": int(len(chunk)),
                "sign_accuracy": sign_accuracy,
                **base,
            }
        )

    overall = pd.DataFrame(rows_overall).sort_values("split").reset_index(drop=True)
    by_horizon = pd.DataFrame(rows_horizon).sort_values(["split", "horizon_months"]).reset_index(drop=True)
    by_sleeve = pd.DataFrame(rows_sleeve).sort_values(["split", "sleeve_id"]).reset_index(drop=True)
    return overall, by_horizon, by_sleeve


def _fit_preprocessed_inputs(inputs: LoadedModelingInputs) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = fit_preprocessor(
        inputs.train_df,
        feature_manifest=inputs.feature_manifest,
        feature_columns=inputs.feature_columns,
    )
    x_train, _ = preprocessor.transform(inputs.train_df)
    x_val, _ = preprocessor.transform(inputs.validation_df)
    x_test, _ = preprocessor.transform(inputs.test_df)
    return preprocessor, x_train, x_val, x_test


def _run_naive_experiment(
    *,
    inputs: LoadedModelingInputs,
    feature_set_name: str,
    horizon_mode: str,
    horizons: tuple[int, ...],
) -> PredictionRunArtifacts:
    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    val_benchmark = inputs.validation_df[["sleeve_id", "horizon_months"]].merge(
        benchmark,
        on=["sleeve_id", "horizon_months"],
        how="left",
        validate="m:1",
    )["benchmark_pred"].to_numpy(dtype=float)
    test_benchmark = inputs.test_df[["sleeve_id", "horizon_months"]].merge(
        benchmark,
        on=["sleeve_id", "horizon_months"],
        how="left",
        validate="m:1",
    )["benchmark_pred"].to_numpy(dtype=float)
    pred_val = _prediction_frame(inputs.validation_df, val_benchmark, benchmark, split_name="validation")
    pred_test = _prediction_frame(inputs.test_df, test_benchmark, benchmark, split_name="test")
    combined = pd.concat([pred_val, pred_test], ignore_index=True)
    metrics_overall, metrics_by_horizon, metrics_by_sleeve = _evaluate_predictions(combined, inputs.train_df)
    return PredictionRunArtifacts(
        experiment_name=f"naive_mean__{horizon_mode}",
        model_name="naive_mean",
        feature_set_name=feature_set_name,
        horizon_mode=horizon_mode,
        horizons=horizons,
        predictions_validation=pred_val,
        predictions_test=pred_test,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_sleeve=metrics_by_sleeve,
        selected_params={},
        input_feature_count=0,
        transformed_feature_count=0,
    )


def _run_ridge_experiment(
    *,
    inputs: LoadedModelingInputs,
    feature_set_name: str,
    horizon_mode: str,
    horizons: tuple[int, ...],
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> PredictionRunArtifacts:
    preprocessor, x_train, x_val, x_test = _fit_preprocessed_inputs(inputs)
    y_train = inputs.train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = inputs.validation_df[TARGET_COL].to_numpy(dtype=float)

    best_alpha = alpha_grid[0]
    best_val_rmse = float("inf")
    for alpha in alpha_grid:
        model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=42)
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_alpha = float(alpha)

    model = Ridge(alpha=best_alpha, fit_intercept=True, random_state=42)
    model.fit(x_train, y_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    pred_val_df = _prediction_frame(inputs.validation_df, pred_val, benchmark, split_name="validation")
    pred_test_df = _prediction_frame(inputs.test_df, pred_test, benchmark, split_name="test")
    combined = pd.concat([pred_val_df, pred_test_df], ignore_index=True)
    metrics_overall, metrics_by_horizon, metrics_by_sleeve = _evaluate_predictions(combined, inputs.train_df)
    return PredictionRunArtifacts(
        experiment_name=f"ridge__{feature_set_name}__{horizon_mode}",
        model_name="ridge",
        feature_set_name=feature_set_name,
        horizon_mode=horizon_mode,
        horizons=horizons,
        predictions_validation=pred_val_df,
        predictions_test=pred_test_df,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_sleeve=metrics_by_sleeve,
        selected_params={"alpha": best_alpha},
        input_feature_count=len(inputs.feature_columns),
        transformed_feature_count=len(preprocessor.feature_names),
    )


def _run_elastic_net_experiment(
    *,
    inputs: LoadedModelingInputs,
    feature_set_name: str,
    horizon_mode: str,
    horizons: tuple[int, ...],
) -> PredictionRunArtifacts:
    preprocessor, x_train, x_val, x_test = _fit_preprocessed_inputs(inputs)
    y_train = inputs.train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = inputs.validation_df[TARGET_COL].to_numpy(dtype=float)

    best_alpha = ELASTIC_ALPHA_GRID[0]
    best_l1 = ELASTIC_L1_GRID[0]
    best_val_rmse = float("inf")
    for alpha in ELASTIC_ALPHA_GRID:
        for l1_ratio in ELASTIC_L1_GRID:
            model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), fit_intercept=True, max_iter=20000, random_state=42)
            model.fit(x_train, y_train)
            pred_val = model.predict(x_val)
            rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_alpha = float(alpha)
                best_l1 = float(l1_ratio)

    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, fit_intercept=True, max_iter=20000, random_state=42)
    model.fit(x_train, y_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    pred_val_df = _prediction_frame(inputs.validation_df, pred_val, benchmark, split_name="validation")
    pred_test_df = _prediction_frame(inputs.test_df, pred_test, benchmark, split_name="test")
    combined = pd.concat([pred_val_df, pred_test_df], ignore_index=True)
    metrics_overall, metrics_by_horizon, metrics_by_sleeve = _evaluate_predictions(combined, inputs.train_df)
    return PredictionRunArtifacts(
        experiment_name=f"elastic_net__{feature_set_name}__{horizon_mode}",
        model_name="elastic_net",
        feature_set_name=feature_set_name,
        horizon_mode=horizon_mode,
        horizons=horizons,
        predictions_validation=pred_val_df,
        predictions_test=pred_test_df,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_sleeve=metrics_by_sleeve,
        selected_params={"alpha": best_alpha, "l1_ratio": best_l1},
        input_feature_count=len(inputs.feature_columns),
        transformed_feature_count=len(preprocessor.feature_names),
    )


def _run_mlp_experiment(
    *,
    inputs: LoadedModelingInputs,
    feature_set_name: str,
    horizon_mode: str,
    horizons: tuple[int, ...],
    model_name: str,
    hidden_dims: tuple[int, ...],
    dropout: float,
    random_seed: int,
    max_epochs: int = 50,
    patience: int = 10,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-5,
) -> PredictionRunArtifacts:
    _set_seeds(random_seed)
    preprocessor, x_train, x_val, x_test = _fit_preprocessed_inputs(inputs)
    y_train = inputs.train_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_val = inputs.validation_df[TARGET_COL].to_numpy(dtype=np.float32)

    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.float32)
    val_x_tensor = torch.tensor(x_val, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val, dtype=torch.float32)
    test_x_tensor = torch.tensor(x_test, dtype=torch.float32)

    model = PredictorMLP(input_dim=x_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_train = model(train_x_tensor)
        train_loss = mse_loss(pred_train, train_y_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(val_x_tensor)
            val_loss = mse_loss(pred_val, val_y_tensor)

        val_loss_value = float(val_loss.detach().cpu().item())
        improved = val_loss_value < (best_val_loss - 1e-8)
        if improved:
            best_val_loss = val_loss_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} failed to produce a valid best state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val = model(val_x_tensor).cpu().numpy()
        pred_test = model(test_x_tensor).cpu().numpy()

    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    pred_val_df = _prediction_frame(inputs.validation_df, pred_val, benchmark, split_name="validation")
    pred_test_df = _prediction_frame(inputs.test_df, pred_test, benchmark, split_name="test")
    combined = pd.concat([pred_val_df, pred_test_df], ignore_index=True)
    metrics_overall, metrics_by_horizon, metrics_by_sleeve = _evaluate_predictions(combined, inputs.train_df)
    return PredictionRunArtifacts(
        experiment_name=f"{model_name}__{feature_set_name}__{horizon_mode}",
        model_name=model_name,
        feature_set_name=feature_set_name,
        horizon_mode=horizon_mode,
        horizons=horizons,
        predictions_validation=pred_val_df,
        predictions_test=pred_test_df,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_sleeve=metrics_by_sleeve,
        selected_params={"best_epoch": best_epoch, "hidden_dims": "-".join(str(v) for v in hidden_dims), "dropout": dropout},
        input_feature_count=len(inputs.feature_columns),
        transformed_feature_count=len(preprocessor.feature_names),
    )


def _result_row(artifacts: PredictionRunArtifacts) -> dict[str, object]:
    val_row = artifacts.metrics_overall.loc[artifacts.metrics_overall["split"].eq("validation")].iloc[0]
    test_row = artifacts.metrics_overall.loc[artifacts.metrics_overall["split"].eq("test")].iloc[0]
    return {
        "experiment_name": artifacts.experiment_name,
        "model_name": artifacts.model_name,
        "feature_set_name": artifacts.feature_set_name,
        "horizon_mode": artifacts.horizon_mode,
        "horizons": "+".join(str(v) for v in artifacts.horizons),
        "input_feature_count": artifacts.input_feature_count,
        "transformed_feature_count": artifacts.transformed_feature_count,
        "selected_params": str(artifacts.selected_params),
        "validation_rmse": float(val_row["rmse"]),
        "validation_mae": float(val_row["mae"]),
        "validation_corr": float(val_row["corr"]),
        "validation_oos_r2": float(val_row["oos_r2_vs_naive"]),
        "validation_sign_accuracy": float(val_row["sign_accuracy"]),
        "validation_tercile_accuracy": float(val_row["tercile_accuracy"]),
        "validation_quintile_accuracy": float(val_row["quintile_accuracy"]),
        "validation_rank_ic_spearman": float(val_row["rank_ic_spearman"]),
        "test_rmse": float(test_row["rmse"]),
        "test_mae": float(test_row["mae"]),
        "test_corr": float(test_row["corr"]),
        "test_oos_r2": float(test_row["oos_r2_vs_naive"]),
        "test_sign_accuracy": float(test_row["sign_accuracy"]),
        "test_tercile_accuracy": float(test_row["tercile_accuracy"]),
        "test_quintile_accuracy": float(test_row["quintile_accuracy"]),
        "test_rank_ic_spearman": float(test_row["rank_ic_spearman"]),
    }


def _horizon_rows(artifacts: PredictionRunArtifacts) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in artifacts.metrics_by_horizon.to_dict("records"):
        rows.append(
            {
                "experiment_name": artifacts.experiment_name,
                "model_name": artifacts.model_name,
                "feature_set_name": artifacts.feature_set_name,
                "horizon_mode": artifacts.horizon_mode,
                **row,
            }
        )
    return rows


def _sleeve_rows(artifacts: PredictionRunArtifacts) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in artifacts.metrics_by_sleeve.to_dict("records"):
        rows.append(
            {
                "experiment_name": artifacts.experiment_name,
                "model_name": artifacts.model_name,
                "feature_set_name": artifacts.feature_set_name,
                "horizon_mode": artifacts.horizon_mode,
                **row,
            }
        )
    return rows


def _select_best_experiment(results_df: pd.DataFrame) -> pd.Series:
    ranked = results_df.loc[results_df["model_name"].ne("naive_mean")].sort_values(
        ["validation_rmse", "validation_corr"], ascending=[True, False]
    )
    return ranked.iloc[0]


def _block_drop_summary(
    *,
    base_inputs: LoadedModelingInputs,
) -> pd.DataFrame:
    base_manifest = base_inputs.feature_manifest.copy()
    baseline_features = list(base_inputs.feature_columns)
    block_map = (
        base_manifest.loc[base_manifest["feature_name"].isin(baseline_features)]
        .groupby("block_name")["feature_name"]
        .apply(list)
        .to_dict()
    )

    rows: list[dict[str, object]] = []
    for horizon_mode, horizons in HORIZON_MODES.items():
        subset_inputs = _subset_inputs(base_inputs, horizons)
        baseline_run = _run_ridge_experiment(
            inputs=subset_inputs,
            feature_set_name="core_plus_enrichment",
            horizon_mode=horizon_mode,
            horizons=horizons,
        )
        baseline_row = _result_row(baseline_run)
        for block_name, features in sorted(block_map.items()):
            remaining = [feature for feature in baseline_features if feature not in set(features)]
            drop_inputs = _with_feature_columns(subset_inputs, remaining)
            drop_run = _run_ridge_experiment(
                inputs=drop_inputs,
                feature_set_name="core_plus_enrichment_minus_block",
                horizon_mode=horizon_mode,
                horizons=horizons,
            )
            drop_row = _result_row(drop_run)
            rows.append(
                {
                    "horizon_mode": horizon_mode,
                    "block_name": block_name,
                    "dropped_feature_count": int(len(features)),
                    "baseline_validation_rmse": baseline_row["validation_rmse"],
                    "drop_validation_rmse": drop_row["validation_rmse"],
                    "delta_validation_rmse": float(drop_row["validation_rmse"] - baseline_row["validation_rmse"]),
                    "baseline_test_rmse": baseline_row["test_rmse"],
                    "drop_test_rmse": drop_row["test_rmse"],
                    "delta_test_rmse": float(drop_row["test_rmse"] - baseline_row["test_rmse"]),
                    "baseline_validation_corr": baseline_row["validation_corr"],
                    "drop_validation_corr": drop_row["validation_corr"],
                    "delta_validation_corr": float(drop_row["validation_corr"] - baseline_row["validation_corr"]),
                    "baseline_test_corr": baseline_row["test_corr"],
                    "drop_test_corr": drop_row["test_corr"],
                    "delta_test_corr": float(drop_row["test_corr"] - baseline_row["test_corr"]),
                    "baseline_test_sign_accuracy": baseline_row["test_sign_accuracy"],
                    "drop_test_sign_accuracy": drop_row["test_sign_accuracy"],
                    "delta_test_sign_accuracy": float(drop_row["test_sign_accuracy"] - baseline_row["test_sign_accuracy"]),
                    "baseline_test_rank_ic_spearman": baseline_row["test_rank_ic_spearman"],
                    "drop_test_rank_ic_spearman": drop_row["test_rank_ic_spearman"],
                    "delta_test_rank_ic_spearman": float(drop_row["test_rank_ic_spearman"] - baseline_row["test_rank_ic_spearman"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon_mode", "delta_validation_rmse", "block_name"], ascending=[True, False, True]).reset_index(drop=True)


def _render_report(
    *,
    ablation_results: pd.DataFrame,
    horizon_specific: pd.DataFrame,
    feature_block_summary: pd.DataFrame,
    sleeve_difficulty: pd.DataFrame,
) -> str:
    best_overall = _select_best_experiment(ablation_results)
    shared_best = ablation_results.loc[ablation_results["horizon_mode"].eq("shared_60_120")].sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).iloc[0]
    sep60_best = ablation_results.loc[ablation_results["horizon_mode"].eq("separate_60")].sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).iloc[0]
    sep120_best = ablation_results.loc[ablation_results["horizon_mode"].eq("separate_120")].sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).iloc[0]

    horizon_test = horizon_specific.loc[horizon_specific["split"].eq("test")].copy()
    horizon_test_best = horizon_test.sort_values(["rmse", "corr"], ascending=[True, False]).groupby("horizon_months", as_index=False).first()

    helpful_blocks = feature_block_summary.loc[feature_block_summary["delta_validation_rmse"] > 0.0].sort_values(["delta_validation_rmse", "delta_test_rmse"], ascending=[False, False]).head(6)
    harmful_blocks = feature_block_summary.loc[feature_block_summary["delta_validation_rmse"] < 0.0].sort_values(["delta_validation_rmse", "delta_test_rmse"], ascending=[True, True]).head(6)

    carry_forward_sleeves = sleeve_difficulty.loc[
        sleeve_difficulty["split"].eq("test")
        & sleeve_difficulty["experiment_name"].eq(str(best_overall["experiment_name"]))
    ].sort_values(["rmse", "corr"], ascending=[False, True]).head(8)

    lines: list[str] = []
    lines.append("# XOPTPOE v2 Prediction Ablation Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Prediction-only study. Data, feature engineering, and portfolio layers stayed unchanged.")
    lines.append("- Compared shared vs separate horizons, four prepared feature sets, smaller-vs-paper MLPs, and ridge / elastic net benchmarks on the frozen train/validation/test splits.")
    lines.append("")
    lines.append("## Best Validation Configuration")
    lines.append(
        f"- {best_overall['experiment_name']}: validation_rmse={best_overall['validation_rmse']:.4f}, test_rmse={best_overall['test_rmse']:.4f}, test_corr={best_overall['test_corr']:.4f}, test_sign_accuracy={best_overall['test_sign_accuracy']:.4f}."
    )
    lines.append("")
    lines.append("## Horizon Pooling")
    lines.append(
        f"- Best shared model: {shared_best['experiment_name']} with validation_rmse={shared_best['validation_rmse']:.4f}, test_rmse={shared_best['test_rmse']:.4f}, test_corr={shared_best['test_corr']:.4f}."
    )
    lines.append(
        f"- Best separate 60m model: {sep60_best['experiment_name']} with validation_rmse={sep60_best['validation_rmse']:.4f}, test_rmse={sep60_best['test_rmse']:.4f}, test_corr={sep60_best['test_corr']:.4f}."
    )
    lines.append(
        f"- Best separate 120m model: {sep120_best['experiment_name']} with validation_rmse={sep120_best['validation_rmse']:.4f}, test_rmse={sep120_best['test_rmse']:.4f}, test_corr={sep120_best['test_corr']:.4f}."
    )
    lines.append("")
    lines.append("## Horizon Predictability")
    for row in horizon_test_best.itertuples(index=False):
        lines.append(
            f"- {int(row.horizon_months)}m best test result: experiment={row.experiment_name}, rmse={row.rmse:.4f}, corr={row.corr:.4f}, sign_accuracy={row.sign_accuracy:.4f}, rank_ic_spearman={row.rank_ic_spearman:.4f}."
        )
    lines.append("")
    lines.append("## Feature-Set Comparison")
    feature_summary = (
        ablation_results.loc[ablation_results['model_name'].isin(['ridge', 'elastic_net', 'paper_mlp', 'small_mlp'])]
        .groupby('feature_set_name', as_index=False)
        .agg(
            best_validation_rmse=('validation_rmse', 'min'),
            best_test_rmse=('test_rmse', 'min'),
            best_test_corr=('test_corr', 'max'),
        )
        .sort_values(['best_validation_rmse', 'best_test_corr'], ascending=[True, False])
    )
    for row in feature_summary.itertuples(index=False):
        lines.append(
            f"- {row.feature_set_name}: best_validation_rmse={row.best_validation_rmse:.4f}, best_test_rmse={row.best_test_rmse:.4f}, best_test_corr={row.best_test_corr:.4f}."
        )
    lines.append("")
    lines.append("## Model-Size Comparison")
    model_summary = (
        ablation_results.loc[ablation_results['feature_set_name'].isin(['core_baseline', 'core_plus_enrichment', 'core_plus_interactions', 'full_firstpass'])]
        .groupby('model_name', as_index=False)
        .agg(
            best_validation_rmse=('validation_rmse', 'min'),
            best_test_rmse=('test_rmse', 'min'),
            best_test_corr=('test_corr', 'max'),
        )
        .sort_values(['best_validation_rmse', 'best_test_corr'], ascending=[True, False])
    )
    for row in model_summary.itertuples(index=False):
        lines.append(
            f"- {row.model_name}: best_validation_rmse={row.best_validation_rmse:.4f}, best_test_rmse={row.best_test_rmse:.4f}, best_test_corr={row.best_test_corr:.4f}."
        )
    lines.append("")
    lines.append("## Feature Block Drop Diagnostics")
    lines.append("- Blocks whose removal worsened validation RMSE are more likely helping the default ridge baseline.")
    for row in helpful_blocks.itertuples(index=False):
        lines.append(
            f"- helpful: {row.horizon_mode}, block={row.block_name}, delta_validation_rmse={row.delta_validation_rmse:.4f}, delta_test_rmse={row.delta_test_rmse:.4f}, delta_test_corr={row.delta_test_corr:.4f}."
        )
    lines.append("- Blocks whose removal improved validation RMSE are more likely adding noise.")
    for row in harmful_blocks.itertuples(index=False):
        lines.append(
            f"- noisy: {row.horizon_mode}, block={row.block_name}, delta_validation_rmse={row.delta_validation_rmse:.4f}, delta_test_rmse={row.delta_test_rmse:.4f}, delta_test_corr={row.delta_test_corr:.4f}."
        )
    lines.append("")
    lines.append("## Sleeve Difficulty")
    lines.append(f"- Reported for the carry-forward baseline `{best_overall['experiment_name']}` only.")
    for row in carry_forward_sleeves.itertuples(index=False):
        lines.append(
            f"- hard sleeve: experiment={row.experiment_name}, sleeve={row.sleeve_id}, rmse={row.rmse:.4f}, corr={row.corr:.4f}, sign_accuracy={row.sign_accuracy:.4f}."
        )
    return "\n".join(lines) + "\n"


def run_prediction_ablation(
    *,
    project_root: Path,
    random_seed: int = 42,
) -> PredictionAblationResult:
    project_root = project_root.resolve()
    inputs_cache = {
        feature_set_name: load_modeling_inputs(project_root, feature_set_name=feature_set_name)
        for feature_set_name in FEATURE_SET_ORDER
    }

    result_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    sleeve_rows: list[dict[str, object]] = []

    for horizon_mode, horizons in HORIZON_MODES.items():
        naive_inputs = _subset_inputs(inputs_cache['core_plus_enrichment'], horizons)
        naive_run = _run_naive_experiment(
            inputs=naive_inputs,
            feature_set_name='none',
            horizon_mode=horizon_mode,
            horizons=horizons,
        )
        result_rows.append(_result_row(naive_run))
        horizon_rows.extend(_horizon_rows(naive_run))
        sleeve_rows.extend(_sleeve_rows(naive_run))

        for feature_set_name in FEATURE_SET_ORDER:
            inputs = _subset_inputs(inputs_cache[feature_set_name], horizons)
            ridge_run = _run_ridge_experiment(
                inputs=inputs,
                feature_set_name=feature_set_name,
                horizon_mode=horizon_mode,
                horizons=horizons,
            )
            elastic_run = _run_elastic_net_experiment(
                inputs=inputs,
                feature_set_name=feature_set_name,
                horizon_mode=horizon_mode,
                horizons=horizons,
            )
            paper_mlp_run = _run_mlp_experiment(
                inputs=inputs,
                feature_set_name=feature_set_name,
                horizon_mode=horizon_mode,
                horizons=horizons,
                model_name='paper_mlp',
                hidden_dims=(32, 16, 8),
                dropout=0.5,
                random_seed=random_seed,
            )
            small_mlp_run = _run_mlp_experiment(
                inputs=inputs,
                feature_set_name=feature_set_name,
                horizon_mode=horizon_mode,
                horizons=horizons,
                model_name='small_mlp',
                hidden_dims=(16, 8),
                dropout=0.5,
                random_seed=random_seed + 17,
            )
            for run in (ridge_run, elastic_run, paper_mlp_run, small_mlp_run):
                result_rows.append(_result_row(run))
                horizon_rows.extend(_horizon_rows(run))
                sleeve_rows.extend(_sleeve_rows(run))

    ablation_results = pd.DataFrame(result_rows).sort_values(
        ['validation_rmse', 'validation_corr'], ascending=[True, False]
    ).reset_index(drop=True)
    horizon_specific_summary = pd.DataFrame(horizon_rows).sort_values(
        ['split', 'horizon_months', 'rmse', 'corr'], ascending=[True, True, True, False]
    ).reset_index(drop=True)
    sleeve_difficulty = pd.DataFrame(sleeve_rows).sort_values(
        ['split', 'rmse', 'corr'], ascending=[True, False, True]
    ).reset_index(drop=True)

    block_summary = _block_drop_summary(
        base_inputs=inputs_cache['core_plus_enrichment'],
    )

    report_text = _render_report(
        ablation_results=ablation_results,
        horizon_specific=horizon_specific_summary,
        feature_block_summary=block_summary,
        sleeve_difficulty=sleeve_difficulty,
    )

    return PredictionAblationResult(
        ablation_results=ablation_results,
        horizon_specific_summary=horizon_specific_summary,
        feature_block_summary=block_summary,
        sleeve_difficulty=sleeve_difficulty,
        report_text=report_text,
    )
