"""Controlled v4 supervised prediction benchmark campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge

from xoptpoe_v3_models.preprocess import fit_preprocessor
from xoptpoe_v4_modeling.features import FEATURE_SET_ORDER, feature_columns_for_set
from xoptpoe_v4_modeling.io import load_csv, load_parquet, write_csv, write_parquet, write_text
from xoptpoe_v4_models.data import (
    DEFAULT_EXCLUDED_SLEEVES,
    SLEEVE_ORDER,
    TARGET_COL,
    LoadedModelingInputs,
    build_sleeve_horizon_benchmark,
    default_paths,
    load_modeling_inputs,
)


HORIZON_MODES: dict[str, tuple[int, ...]] = {
    "shared_60_120": (60, 120),
    "separate_60": (60,),
    "separate_120": (120,),
}

FIXED_MODELS: tuple[str, ...] = ("naive_mean", "ridge", "elastic_net", "random_forest", "gradient_boosting")
ROLLING_MODELS: tuple[str, ...] = ("naive_mean", "ridge", "elastic_net")
MODEL_FAMILY_MAP: dict[str, str] = {
    "naive_mean": "naive",
    "ridge": "linear",
    "elastic_net": "linear",
    "random_forest": "tree",
    "gradient_boosting": "tree",
}
NEW_SLEEVES: tuple[str, ...] = ("FI_EU_GOVT", "CR_EU_IG", "CR_US_HY", "LISTED_RE", "LISTED_INFRA")
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
ELASTIC_ALPHA_GRID: tuple[float, ...] = (0.0005, 0.001, 0.005, 0.01, 0.05)
ELASTIC_L1_GRID: tuple[float, ...] = (0.2, 0.5, 0.8)


@dataclass(frozen=True)
class PredictionRunArtifacts:
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
class RollingConfig:
    min_train_months: int = 48
    validation_months: int = 12
    test_months: int = 12
    step_months: int = 12
    min_train_months_by_mode: dict[str, int] | None = None


@dataclass(frozen=True)
class RollingFold:
    fold_id: int
    horizon_mode: str
    horizons: tuple[int, ...]
    min_train_months: int
    train_months: tuple[pd.Timestamp, ...]
    validation_months: tuple[pd.Timestamp, ...]
    test_months: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class PredictionBenchmarkOutputs:
    metrics: pd.DataFrame
    by_sleeve: pd.DataFrame
    feature_set_summary: pd.DataFrame
    rolling_summary: pd.DataFrame
    china_diagnostics: pd.DataFrame
    new_sleeve_diagnostics: pd.DataFrame
    predictions_validation: pd.DataFrame
    predictions_test: pd.DataFrame
    rolling_fold_metrics: pd.DataFrame
    rolling_fold_manifest: pd.DataFrame
    report_text: str


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
    )


def _prediction_frame(frame: pd.DataFrame, y_pred: np.ndarray, benchmark: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
    out = frame[["month_end", "sleeve_id", "horizon_months", TARGET_COL]].copy()
    out["split"] = split_name
    out["y_true"] = out[TARGET_COL].astype(float)
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    out = out.drop(columns=[TARGET_COL])
    out = out.merge(benchmark, on=["sleeve_id", "horizon_months"], how="left", validate="m:1")
    return out.sort_values(["month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_benchmark: np.ndarray) -> float:
    sse_model = float(np.sum((y_true - y_pred) ** 2))
    sse_benchmark = float(np.sum((y_true - y_benchmark) ** 2))
    if sse_benchmark <= 0.0:
        return float("nan")
    return 1.0 - (sse_model / sse_benchmark)


def _mean_rank_ic(frame: pd.DataFrame) -> tuple[float, float]:
    spearman_values: list[float] = []
    pearson_values: list[float] = []
    for _, chunk in frame.groupby(["month_end", "horizon_months"], sort=True):
        if len(chunk) < 2:
            continue
        if chunk["y_pred"].nunique() < 2 or chunk["y_true"].nunique() < 2:
            continue
        corr_s = chunk["y_pred"].corr(chunk["y_true"], method="spearman")
        corr_p = chunk["y_pred"].corr(chunk["y_true"], method="pearson")
        if pd.notna(corr_s):
            spearman_values.append(float(corr_s))
        if pd.notna(corr_p):
            pearson_values.append(float(corr_p))
    mean_s = float(np.mean(spearman_values)) if spearman_values else float("nan")
    mean_p = float(np.mean(pearson_values)) if pearson_values else float("nan")
    return mean_s, mean_p


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


def _classification_accuracy(frame: pd.DataFrame, thresholds_by_horizon: dict[int, dict[str, np.ndarray]], kind: str) -> float:
    correct = 0
    total = 0
    for horizon_months, chunk in frame.groupby("horizon_months"):
        thresholds = thresholds_by_horizon[int(horizon_months)][kind]
        true_bins = _apply_bins(chunk["y_true"].to_numpy(dtype=float), thresholds)
        pred_bins = _apply_bins(chunk["y_pred"].to_numpy(dtype=float), thresholds)
        correct += int(np.sum(true_bins == pred_bins))
        total += int(len(chunk))
    return float(correct / total) if total > 0 else float("nan")


def _diagnostic_metrics(frame: pd.DataFrame, thresholds_by_horizon: dict[int, dict[str, np.ndarray]]) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    y_benchmark = frame["benchmark_pred"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    corr = _safe_corr(y_true, y_pred)
    sign_accuracy = float(np.mean((y_true >= 0.0) == (y_pred >= 0.0)))
    tercile_accuracy = _classification_accuracy(frame, thresholds_by_horizon, "tercile")
    quintile_accuracy = _classification_accuracy(frame, thresholds_by_horizon, "quintile")
    rank_ic_spearman, rank_ic_pearson = _mean_rank_ic(frame)
    return {
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "oos_r2_vs_naive": _safe_oos_r2(y_true, y_pred, y_benchmark),
        "sign_accuracy": sign_accuracy,
        "tercile_accuracy": tercile_accuracy,
        "quintile_accuracy": quintile_accuracy,
        "rank_ic_spearman": rank_ic_spearman,
        "rank_ic_pearson": rank_ic_pearson,
    }


def _evaluate_predictions(predictions: pd.DataFrame, train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = _quantile_thresholds(train_df)

    rows_overall: list[dict[str, object]] = []
    for split_name, chunk in predictions.groupby("split", as_index=False):
        rows_overall.append({"split": split_name, "row_count": int(len(chunk)), **_diagnostic_metrics(chunk, thresholds)})

    rows_horizon: list[dict[str, object]] = []
    for (split_name, horizon_months), chunk in predictions.groupby(["split", "horizon_months"], as_index=False):
        rows_horizon.append(
            {
                "split": split_name,
                "horizon_months": int(horizon_months),
                "row_count": int(len(chunk)),
                **_diagnostic_metrics(chunk, thresholds),
            }
        )

    rows_sleeve: list[dict[str, object]] = []
    for (split_name, sleeve_id), chunk in predictions.groupby(["split", "sleeve_id"], as_index=False):
        rows_sleeve.append(
            {
                "split": split_name,
                "sleeve_id": sleeve_id,
                "row_count": int(len(chunk)),
                **_diagnostic_metrics(chunk, thresholds),
            }
        )

    return (
        pd.DataFrame(rows_overall).sort_values("split").reset_index(drop=True),
        pd.DataFrame(rows_horizon).sort_values(["split", "horizon_months"]).reset_index(drop=True),
        pd.DataFrame(rows_sleeve).sort_values(["split", "sleeve_id"]).reset_index(drop=True),
    )


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


def _run_naive_experiment(*, inputs: LoadedModelingInputs, feature_set_name: str, horizon_mode: str, horizons: tuple[int, ...]) -> PredictionRunArtifacts:
    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    pred_val = inputs.validation_df[["sleeve_id", "horizon_months"]].merge(benchmark, on=["sleeve_id", "horizon_months"], how="left", validate="m:1")["benchmark_pred"].to_numpy(dtype=float)
    pred_test = inputs.test_df[["sleeve_id", "horizon_months"]].merge(benchmark, on=["sleeve_id", "horizon_months"], how="left", validate="m:1")["benchmark_pred"].to_numpy(dtype=float)
    pred_val_df = _prediction_frame(inputs.validation_df, pred_val, benchmark, split_name="validation")
    pred_test_df = _prediction_frame(inputs.test_df, pred_test, benchmark, split_name="test")
    combined = pd.concat([pred_val_df, pred_test_df], ignore_index=True)
    metrics_overall, metrics_by_horizon, metrics_by_sleeve = _evaluate_predictions(combined, inputs.train_df)
    return PredictionRunArtifacts(
        experiment_name=f"naive_mean__{horizon_mode}",
        model_name="naive_mean",
        feature_set_name=feature_set_name,
        horizon_mode=horizon_mode,
        horizons=horizons,
        predictions_validation=pred_val_df,
        predictions_test=pred_test_df,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_sleeve=metrics_by_sleeve,
        selected_params={},
        input_feature_count=0,
        transformed_feature_count=0,
    )


def _run_ridge_experiment(*, inputs: LoadedModelingInputs, feature_set_name: str, horizon_mode: str, horizons: tuple[int, ...]) -> PredictionRunArtifacts:
    preprocessor, x_train, x_val, x_test = _fit_preprocessed_inputs(inputs)
    y_train = inputs.train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = inputs.validation_df[TARGET_COL].to_numpy(dtype=float)

    best_alpha = DEFAULT_ALPHA_GRID[0]
    best_val_rmse = float("inf")
    for alpha in DEFAULT_ALPHA_GRID:
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd", random_state=42)
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_alpha = float(alpha)

    model = Ridge(alpha=best_alpha, fit_intercept=True, solver="svd", random_state=42)
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


def _run_elastic_net_experiment(*, inputs: LoadedModelingInputs, feature_set_name: str, horizon_mode: str, horizons: tuple[int, ...]) -> PredictionRunArtifacts:
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


def _run_random_forest_experiment(*, inputs: LoadedModelingInputs, feature_set_name: str, horizon_mode: str, horizons: tuple[int, ...]) -> PredictionRunArtifacts:
    preprocessor, x_train, x_val, x_test = _fit_preprocessed_inputs(inputs)
    y_train = inputs.train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = inputs.validation_df[TARGET_COL].to_numpy(dtype=float)

    grid = (
        {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 10},
    )
    best_params = grid[0]
    best_val_rmse = float("inf")
    for params in grid:
        model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_params = params

    model = RandomForestRegressor(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=best_params["max_depth"],
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    pred_val_df = _prediction_frame(inputs.validation_df, pred_val, benchmark, split_name="validation")
    pred_test_df = _prediction_frame(inputs.test_df, pred_test, benchmark, split_name="test")
    combined = pd.concat([pred_val_df, pred_test_df], ignore_index=True)
    metrics_overall, metrics_by_horizon, metrics_by_sleeve = _evaluate_predictions(combined, inputs.train_df)
    return PredictionRunArtifacts(
        experiment_name=f"random_forest__{feature_set_name}__{horizon_mode}",
        model_name="random_forest",
        feature_set_name=feature_set_name,
        horizon_mode=horizon_mode,
        horizons=horizons,
        predictions_validation=pred_val_df,
        predictions_test=pred_test_df,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_sleeve=metrics_by_sleeve,
        selected_params=best_params,
        input_feature_count=len(inputs.feature_columns),
        transformed_feature_count=len(preprocessor.feature_names),
    )


def _run_gradient_boosting_experiment(*, inputs: LoadedModelingInputs, feature_set_name: str, horizon_mode: str, horizons: tuple[int, ...]) -> PredictionRunArtifacts:
    preprocessor, x_train, x_val, x_test = _fit_preprocessed_inputs(inputs)
    y_train = inputs.train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = inputs.validation_df[TARGET_COL].to_numpy(dtype=float)

    grid = (
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 250, "learning_rate": 0.03, "max_depth": 2},
    )
    best_params = grid[0]
    best_val_rmse = float("inf")
    for params in grid:
        model = GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_params = params

    model = GradientBoostingRegressor(
        n_estimators=int(best_params["n_estimators"]),
        learning_rate=float(best_params["learning_rate"]),
        max_depth=int(best_params["max_depth"]),
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(x_train, y_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    pred_val_df = _prediction_frame(inputs.validation_df, pred_val, benchmark, split_name="validation")
    pred_test_df = _prediction_frame(inputs.test_df, pred_test, benchmark, split_name="test")
    combined = pd.concat([pred_val_df, pred_test_df], ignore_index=True)
    metrics_overall, metrics_by_horizon, metrics_by_sleeve = _evaluate_predictions(combined, inputs.train_df)
    return PredictionRunArtifacts(
        experiment_name=f"gradient_boosting__{feature_set_name}__{horizon_mode}",
        model_name="gradient_boosting",
        feature_set_name=feature_set_name,
        horizon_mode=horizon_mode,
        horizons=horizons,
        predictions_validation=pred_val_df,
        predictions_test=pred_test_df,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_sleeve=metrics_by_sleeve,
        selected_params=best_params,
        input_feature_count=len(inputs.feature_columns),
        transformed_feature_count=len(preprocessor.feature_names),
    )


def _result_row(artifacts: PredictionRunArtifacts) -> dict[str, object]:
    val_row = artifacts.metrics_overall.loc[artifacts.metrics_overall["split"].eq("validation")].iloc[0]
    test_row = artifacts.metrics_overall.loc[artifacts.metrics_overall["split"].eq("test")].iloc[0]
    return {
        "experiment_name": artifacts.experiment_name,
        "model_name": artifacts.model_name,
        "model_family": MODEL_FAMILY_MAP[artifacts.model_name],
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


def _sleeve_rows(artifacts: PredictionRunArtifacts) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in artifacts.metrics_by_sleeve.to_dict("records"):
        rows.append(
            {
                "experiment_name": artifacts.experiment_name,
                "model_name": artifacts.model_name,
                "model_family": MODEL_FAMILY_MAP[artifacts.model_name],
                "feature_set_name": artifacts.feature_set_name,
                "horizon_mode": artifacts.horizon_mode,
                **row,
            }
        )
    return rows


def _append_prediction_metadata(artifacts: PredictionRunArtifacts) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_val = artifacts.predictions_validation.copy()
    pred_test = artifacts.predictions_test.copy()
    for frame in (pred_val, pred_test):
        frame["experiment_name"] = artifacts.experiment_name
        frame["model_name"] = artifacts.model_name
        frame["model_family"] = MODEL_FAMILY_MAP[artifacts.model_name]
        frame["feature_set_name"] = artifacts.feature_set_name if artifacts.model_name != "naive_mean" else "no_features"
        frame["horizon_mode"] = artifacts.horizon_mode
        frame["selected_params"] = str(artifacts.selected_params)
    return pred_val, pred_test


def _is_complete_month(chunk: pd.DataFrame, horizons: tuple[int, ...]) -> bool:
    expected = len(SLEEVE_ORDER) * len(horizons)
    if len(chunk) != expected:
        return False
    counts = chunk.groupby("sleeve_id")["horizon_months"].nunique()
    return bool(counts.reindex(list(SLEEVE_ORDER)).fillna(0).eq(len(horizons)).all())


def _subset_panel_for_mode(panel: pd.DataFrame, horizon_mode: str) -> tuple[pd.DataFrame, tuple[int, ...]]:
    horizons = HORIZON_MODES[horizon_mode]
    work = panel.loc[panel["horizon_months"].isin(horizons) & ~panel["sleeve_id"].isin(DEFAULT_EXCLUDED_SLEEVES)].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    valid_months = []
    for month_end, chunk in work.groupby("month_end", sort=True):
        if _is_complete_month(chunk, horizons):
            valid_months.append(pd.Timestamp(month_end))
    out = work.loc[work["month_end"].isin(valid_months)].copy()
    out = out.sort_values(["month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No complete rows available for horizon_mode={horizon_mode}")
    return out, horizons


def _generate_folds(panel: pd.DataFrame, horizon_mode: str, config: RollingConfig) -> list[RollingFold]:
    subset, horizons = _subset_panel_for_mode(panel, horizon_mode)
    months = tuple(sorted(pd.Timestamp(v) for v in subset["month_end"].drop_duplicates().tolist()))
    min_train_months = config.min_train_months
    if config.min_train_months_by_mode and horizon_mode in config.min_train_months_by_mode:
        min_train_months = int(config.min_train_months_by_mode[horizon_mode])
    total = min_train_months + config.validation_months + config.test_months
    folds: list[RollingFold] = []
    start = 0
    fold_id = 1
    while start + total <= len(months):
        folds.append(
            RollingFold(
                fold_id=fold_id,
                horizon_mode=horizon_mode,
                horizons=horizons,
                min_train_months=min_train_months,
                train_months=months[start : start + min_train_months],
                validation_months=months[start + min_train_months : start + min_train_months + config.validation_months],
                test_months=months[start + min_train_months + config.validation_months : start + total],
            )
        )
        start += config.step_months
        fold_id += 1
    if not folds:
        raise ValueError(f"No rolling folds generated for {horizon_mode}")
    return folds


def _make_loaded_inputs(*, panel: pd.DataFrame, feature_manifest: pd.DataFrame, feature_set_name: str, fold: RollingFold) -> LoadedModelingInputs:
    feature_columns = feature_columns_for_set(feature_manifest, feature_set_name)
    return LoadedModelingInputs(
        train_df=panel.loc[panel["month_end"].isin(fold.train_months)].copy(),
        validation_df=panel.loc[panel["month_end"].isin(fold.validation_months)].copy(),
        test_df=panel.loc[panel["month_end"].isin(fold.test_months)].copy(),
        feature_manifest=feature_manifest,
        feature_columns=feature_columns,
    )


def _run_model(*, model_name: str, inputs: LoadedModelingInputs, feature_set_name: str, horizon_mode: str, horizons: tuple[int, ...]) -> PredictionRunArtifacts:
    if model_name == "naive_mean":
        return _run_naive_experiment(inputs=inputs, feature_set_name="none", horizon_mode=horizon_mode, horizons=horizons)
    if model_name == "ridge":
        return _run_ridge_experiment(inputs=inputs, feature_set_name=feature_set_name, horizon_mode=horizon_mode, horizons=horizons)
    if model_name == "elastic_net":
        return _run_elastic_net_experiment(inputs=inputs, feature_set_name=feature_set_name, horizon_mode=horizon_mode, horizons=horizons)
    if model_name == "random_forest":
        return _run_random_forest_experiment(inputs=inputs, feature_set_name=feature_set_name, horizon_mode=horizon_mode, horizons=horizons)
    if model_name == "gradient_boosting":
        return _run_gradient_boosting_experiment(inputs=inputs, feature_set_name=feature_set_name, horizon_mode=horizon_mode, horizons=horizons)
    raise ValueError(f"Unsupported model_name={model_name}")


def _run_fixed_split(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs_cache = {
        feature_set_name: load_modeling_inputs(project_root, feature_set_name=feature_set_name)
        for feature_set_name in FEATURE_SET_ORDER
    }
    metric_rows: list[dict[str, object]] = []
    sleeve_rows: list[dict[str, object]] = []
    preds_val: list[pd.DataFrame] = []
    preds_test: list[pd.DataFrame] = []

    for horizon_mode, horizons in HORIZON_MODES.items():
        naive_inputs = _subset_inputs(inputs_cache["core_plus_enrichment"], horizons)
        naive_run = _run_naive_experiment(inputs=naive_inputs, feature_set_name="none", horizon_mode=horizon_mode, horizons=horizons)
        metric_rows.append(_result_row(naive_run))
        sleeve_rows.extend(_sleeve_rows(naive_run))
        pval, ptest = _append_prediction_metadata(naive_run)
        preds_val.append(pval)
        preds_test.append(ptest)

        for feature_set_name in FEATURE_SET_ORDER:
            inputs = _subset_inputs(inputs_cache[feature_set_name], horizons)
            for model_name in FIXED_MODELS:
                if model_name == "naive_mean":
                    continue
                run = _run_model(
                    model_name=model_name,
                    inputs=inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode=horizon_mode,
                    horizons=horizons,
                )
                metric_rows.append(_result_row(run))
                sleeve_rows.extend(_sleeve_rows(run))
                pval, ptest = _append_prediction_metadata(run)
                preds_val.append(pval)
                preds_test.append(ptest)

    metrics = pd.DataFrame(metric_rows).sort_values(["horizon_mode", "validation_rmse", "validation_corr"], ascending=[True, True, False]).reset_index(drop=True)
    metrics["selected_by_validation_within_mode"] = False
    metrics["selected_by_validation_within_mode_feature"] = False
    for horizon_mode, chunk in metrics.groupby("horizon_mode"):
        idx = chunk.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).index[0]
        metrics.loc[idx, "selected_by_validation_within_mode"] = True
    for (horizon_mode, feature_set_name), chunk in metrics.loc[metrics["feature_set_name"] != "none"].groupby(["horizon_mode", "feature_set_name"]):
        idx = chunk.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).index[0]
        metrics.loc[idx, "selected_by_validation_within_mode_feature"] = True
    by_sleeve = pd.DataFrame(sleeve_rows).sort_values(["split", "experiment_name", "rmse", "corr"], ascending=[True, True, False, True]).reset_index(drop=True)
    by_sleeve["rmse_rank_worst_first"] = by_sleeve.groupby(["experiment_name", "split"])["rmse"].rank(method="dense", ascending=False)
    by_sleeve["corr_rank_worst_first"] = by_sleeve.groupby(["experiment_name", "split"])["corr"].rank(method="dense", ascending=True)
    return metrics, by_sleeve, pd.concat(preds_val, ignore_index=True), pd.concat(preds_test, ignore_index=True)


def _run_rolling_summary(project_root: Path, config: RollingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_paths(project_root)
    panel = load_parquet(paths.data_out_dir / "modeling_panel_firstpass.parquet")
    panel = panel.loc[~panel["sleeve_id"].isin(DEFAULT_EXCLUDED_SLEEVES)].copy()
    panel["month_end"] = pd.to_datetime(panel["month_end"])
    feature_manifest = load_csv(paths.feature_manifest, parse_dates=["first_valid_date", "last_valid_date"])

    fold_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        mode_panel, horizons = _subset_panel_for_mode(panel, horizon_mode)
        folds = _generate_folds(mode_panel, horizon_mode, config)
        for fold in folds:
            manifest_rows.append(
                {
                    "horizon_mode": horizon_mode,
                    "fold_id": fold.fold_id,
                    "min_train_months": fold.min_train_months,
                    "train_start": min(fold.train_months).date().isoformat(),
                    "train_end": max(fold.train_months).date().isoformat(),
                    "validation_start": min(fold.validation_months).date().isoformat(),
                    "validation_end": max(fold.validation_months).date().isoformat(),
                    "test_start": min(fold.test_months).date().isoformat(),
                    "test_end": max(fold.test_months).date().isoformat(),
                    "train_month_count": len(fold.train_months),
                    "validation_month_count": len(fold.validation_months),
                    "test_month_count": len(fold.test_months),
                }
            )
            for feature_set_name in FEATURE_SET_ORDER:
                inputs = _make_loaded_inputs(panel=mode_panel, feature_manifest=feature_manifest, feature_set_name=feature_set_name, fold=fold)
                for model_name in ROLLING_MODELS:
                    run = _run_model(
                        model_name=model_name,
                        inputs=inputs,
                        feature_set_name=feature_set_name,
                        horizon_mode=fold.horizon_mode,
                        horizons=fold.horizons,
                    )
                    row = _result_row(run)
                    row["fold_id"] = fold.fold_id
                    row["min_train_months"] = fold.min_train_months
                    row["train_start"] = min(fold.train_months).date().isoformat()
                    row["train_end"] = max(fold.train_months).date().isoformat()
                    row["validation_start"] = min(fold.validation_months).date().isoformat()
                    row["validation_end"] = max(fold.validation_months).date().isoformat()
                    row["test_start"] = min(fold.test_months).date().isoformat()
                    row["test_end"] = max(fold.test_months).date().isoformat()
                    fold_rows.append(row)

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics["test_rmse_delta_vs_naive"] = np.nan
    fold_metrics["beat_naive_rmse"] = False
    for (horizon_mode, fold_id), chunk in fold_metrics.groupby(["horizon_mode", "fold_id"]):
        naive_rmse = float(chunk.loc[chunk["model_name"] == "naive_mean", "test_rmse"].iloc[0])
        idx = chunk.index
        fold_metrics.loc[idx, "test_rmse_delta_vs_naive"] = fold_metrics.loc[idx, "test_rmse"] - naive_rmse
        fold_metrics.loc[idx, "beat_naive_rmse"] = fold_metrics.loc[idx, "test_rmse"] < naive_rmse

    summary = (
        fold_metrics.groupby(["experiment_name", "model_name", "model_family", "feature_set_name", "horizon_mode", "horizons"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            mean_validation_rmse=("validation_rmse", "mean"),
            std_validation_rmse=("validation_rmse", "std"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_validation_corr=("validation_corr", "mean"),
            mean_test_corr=("test_corr", "mean"),
            std_test_corr=("test_corr", "std"),
            mean_validation_oos_r2=("validation_oos_r2", "mean"),
            mean_test_oos_r2=("test_oos_r2", "mean"),
            mean_test_sign_accuracy=("test_sign_accuracy", "mean"),
            mean_test_rank_ic_spearman=("test_rank_ic_spearman", "mean"),
            mean_test_rmse_delta_vs_naive=("test_rmse_delta_vs_naive", "mean"),
            beat_naive_rmse_fold_share=("beat_naive_rmse", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["validation_to_test_rmse_decay"] = summary["mean_test_rmse"] - summary["mean_validation_rmse"]
    summary = summary.sort_values(["horizon_mode", "mean_validation_rmse", "mean_test_rmse"], ascending=[True, True, True]).reset_index(drop=True)
    manifest = pd.DataFrame(manifest_rows).sort_values(["horizon_mode", "fold_id"]).reset_index(drop=True)
    return summary, fold_metrics, manifest


def _build_feature_set_summary(fixed_metrics: pd.DataFrame, rolling_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fixed_use = fixed_metrics.loc[fixed_metrics["feature_set_name"] != "none"].copy()
    rolling_use = rolling_summary.loc[rolling_summary["feature_set_name"] != "none"].copy()
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        for feature_set_name in FEATURE_SET_ORDER:
            fixed_chunk = fixed_use.loc[(fixed_use["horizon_mode"] == horizon_mode) & (fixed_use["feature_set_name"] == feature_set_name)].copy()
            rolling_chunk = rolling_use.loc[(rolling_use["horizon_mode"] == horizon_mode) & (rolling_use["feature_set_name"] == feature_set_name)].copy()
            if fixed_chunk.empty:
                continue
            fixed_best = fixed_chunk.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).iloc[0]
            row = {
                "horizon_mode": horizon_mode,
                "feature_set_name": feature_set_name,
                "fixed_best_experiment": fixed_best["experiment_name"],
                "fixed_best_model": fixed_best["model_name"],
                "fixed_best_model_family": fixed_best["model_family"],
                "fixed_validation_rmse": float(fixed_best["validation_rmse"]),
                "fixed_test_rmse": float(fixed_best["test_rmse"]),
                "fixed_test_corr": float(fixed_best["test_corr"]),
                "fixed_test_rank_ic_spearman": float(fixed_best["test_rank_ic_spearman"]),
            }
            if not rolling_chunk.empty:
                rolling_best = rolling_chunk.sort_values(["mean_validation_rmse", "mean_test_rmse", "mean_test_corr"], ascending=[True, True, False]).iloc[0]
                row.update(
                    {
                        "rolling_best_experiment": rolling_best["experiment_name"],
                        "rolling_best_model": rolling_best["model_name"],
                        "rolling_best_model_family": rolling_best["model_family"],
                        "rolling_mean_validation_rmse": float(rolling_best["mean_validation_rmse"]),
                        "rolling_mean_test_rmse": float(rolling_best["mean_test_rmse"]),
                        "rolling_std_test_rmse": float(rolling_best["std_test_rmse"]),
                        "rolling_mean_test_corr": float(rolling_best["mean_test_corr"]),
                        "rolling_beat_naive_rmse_fold_share": float(rolling_best["beat_naive_rmse_fold_share"]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon_mode", "rolling_mean_validation_rmse", "fixed_validation_rmse"], ascending=[True, True, True], na_position="last").reset_index(drop=True)


def _merge_fixed_and_rolling(fixed_metrics: pd.DataFrame, rolling_summary: pd.DataFrame) -> pd.DataFrame:
    return fixed_metrics.merge(
        rolling_summary[
            [
                "experiment_name",
                "fold_count",
                "mean_validation_rmse",
                "std_validation_rmse",
                "mean_test_rmse",
                "std_test_rmse",
                "mean_test_corr",
                "std_test_corr",
                "mean_test_oos_r2",
                "mean_test_sign_accuracy",
                "mean_test_rank_ic_spearman",
                "mean_test_rmse_delta_vs_naive",
                "beat_naive_rmse_fold_share",
                "validation_to_test_rmse_decay",
            ]
        ],
        on="experiment_name",
        how="left",
        validate="1:1",
    ).sort_values(["horizon_mode", "validation_rmse", "test_rmse"], ascending=[True, True, True]).reset_index(drop=True)


def _build_china_diagnostics(metrics: pd.DataFrame, by_sleeve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        selected = metrics.loc[metrics["selected_by_validation_within_mode"] & metrics["horizon_mode"].eq(horizon_mode)].iloc[0]
        chunk = by_sleeve.loc[
            (by_sleeve["experiment_name"] == selected["experiment_name"]) &
            (by_sleeve["split"] == "test") &
            (by_sleeve["sleeve_id"] == "EQ_CN")
        ]
        if chunk.empty:
            continue
        row = chunk.iloc[0]
        rows.append(
            {
                "diagnostic_type": "eq_cn_selected_winner",
                "horizon_mode": horizon_mode,
                "experiment_name": selected["experiment_name"],
                "model_name": selected["model_name"],
                "feature_set_name": selected["feature_set_name"],
                "sleeve_id": "EQ_CN",
                "rmse": float(row["rmse"]),
                "corr": float(row["corr"]),
                "sign_accuracy": float(row["sign_accuracy"]),
                "oos_r2_vs_naive": float(row["oos_r2_vs_naive"]),
                "rmse_rank_worst_first": float(row["rmse_rank_worst_first"]),
                "corr_rank_worst_first": float(row["corr_rank_worst_first"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_mode"]).reset_index(drop=True)


def _build_new_sleeve_diagnostics(metrics: pd.DataFrame, by_sleeve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        selected = metrics.loc[metrics["selected_by_validation_within_mode"] & metrics["horizon_mode"].eq(horizon_mode)].iloc[0]
        chunk = by_sleeve.loc[
            (by_sleeve["experiment_name"] == selected["experiment_name"]) &
            (by_sleeve["split"] == "test") &
            (by_sleeve["sleeve_id"].isin(NEW_SLEEVES))
        ].copy()
        if chunk.empty:
            continue
        chunk["diagnostic_type"] = "new_sleeve_selected_winner"
        chunk["horizon_mode"] = horizon_mode
        chunk["selected_experiment_name"] = selected["experiment_name"]
        chunk["selected_model_name"] = selected["model_name"]
        chunk["selected_feature_set_name"] = selected["feature_set_name"]
        rows.extend(chunk.to_dict("records"))
    return pd.DataFrame(rows).sort_values(["horizon_mode", "sleeve_id"]).reset_index(drop=True)


def _winner_line(row: pd.Series, prefix: str) -> str:
    return f"- {prefix}: {row['experiment_name']} | validation_rmse={row['validation_rmse']:.4f} | test_rmse={row['test_rmse']:.4f} | test_corr={row['test_corr']:.4f}"


def _render_report(*, metrics: pd.DataFrame, feature_set_summary: pd.DataFrame, rolling_summary: pd.DataFrame, china_diagnostics: pd.DataFrame, new_sleeve_diagnostics: pd.DataFrame) -> str:
    best_60 = metrics.loc[metrics["selected_by_validation_within_mode"] & metrics["horizon_mode"].eq("separate_60")].iloc[0]
    best_120 = metrics.loc[metrics["selected_by_validation_within_mode"] & metrics["horizon_mode"].eq("separate_120")].iloc[0]
    best_shared = metrics.loc[metrics["selected_by_validation_within_mode"] & metrics["horizon_mode"].eq("shared_60_120")].iloc[0]

    rolling_best_60 = rolling_summary.loc[rolling_summary["horizon_mode"].eq("separate_60")].sort_values(["mean_validation_rmse", "mean_test_rmse", "mean_test_corr"], ascending=[True, True, False]).iloc[0]
    rolling_best_120 = rolling_summary.loc[rolling_summary["horizon_mode"].eq("separate_120")].sort_values(["mean_validation_rmse", "mean_test_rmse", "mean_test_corr"], ascending=[True, True, False]).iloc[0]
    rolling_best_shared = rolling_summary.loc[rolling_summary["horizon_mode"].eq("shared_60_120")].sort_values(["mean_validation_rmse", "mean_test_rmse", "mean_test_corr"], ascending=[True, True, False]).iloc[0]

    family_summary = (
        metrics.groupby("model_family", as_index=False)
        .agg(best_validation_rmse=("validation_rmse", "min"), best_test_rmse=("test_rmse", "min"), best_test_corr=("test_corr", "max"))
        .sort_values(["best_validation_rmse", "best_test_corr"], ascending=[True, False])
    )

    strongest_overall = metrics.loc[metrics["selected_by_validation_within_mode"]].sort_values(["test_rmse", "test_corr"], ascending=[True, False]).iloc[0]

    lines: list[str] = []
    lines.append("# XOPTPOE v4 Prediction Benchmark Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Active paths only: `data/final_v4_expanded_universe/`, `data/modeling_v4/`, `reports/v4_expanded_universe/`.")
    lines.append("- Frozen v1/v2/v3 branches were not touched.")
    lines.append("- Default benchmark roster excludes `CR_EU_HY` by governance lock because the accepted split design leaves it with only 5 train rows per horizon.")
    lines.append("- Fixed-split evidence compares naive, regularized linear, and compact tree baselines across the accepted first-pass feature sets.")
    lines.append("- Rolling evidence is intentionally limited to naive and linear baselines.")
    lines.append("- Rolling windows use 48 train months for `separate_60`, but shorten to 24 train months for `separate_120` and `shared_60_120` because the accepted 14-sleeve common window is only 71 complete months for those modes.")
    lines.append("")
    lines.append("## Fixed-Split Winners")
    lines.append(_winner_line(best_60, "Best 60m validation winner"))
    lines.append(_winner_line(best_120, "Best 120m validation winner"))
    lines.append(_winner_line(best_shared, "Best shared 60m+120m benchmark"))
    lines.append("")
    lines.append("## Rolling Winners")
    lines.append(f"- separate_60: {rolling_best_60['experiment_name']} | mean_test_rmse={rolling_best_60['mean_test_rmse']:.4f} | mean_test_corr={rolling_best_60['mean_test_corr']:.4f} | beat_naive_fold_share={rolling_best_60['beat_naive_rmse_fold_share']:.2f}")
    lines.append(f"- separate_120: {rolling_best_120['experiment_name']} | mean_test_rmse={rolling_best_120['mean_test_rmse']:.4f} | mean_test_corr={rolling_best_120['mean_test_corr']:.4f} | beat_naive_fold_share={rolling_best_120['beat_naive_rmse_fold_share']:.2f}")
    lines.append(f"- shared_60_120: {rolling_best_shared['experiment_name']} | mean_test_rmse={rolling_best_shared['mean_test_rmse']:.4f} | mean_test_corr={rolling_best_shared['mean_test_corr']:.4f} | beat_naive_fold_share={rolling_best_shared['beat_naive_rmse_fold_share']:.2f}")
    lines.append("")
    lines.append("## Model-Family Comparison")
    for row in family_summary.itertuples(index=False):
        lines.append(f"- {row.model_family}: best_validation_rmse={row.best_validation_rmse:.4f}, best_test_rmse={row.best_test_rmse:.4f}, best_test_corr={row.best_test_corr:.4f}.")
    lines.append("")
    lines.append("## Feature-Set Readout")
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        chunk = feature_set_summary.loc[feature_set_summary["horizon_mode"] == horizon_mode]
        best = chunk.sort_values(["rolling_mean_validation_rmse", "fixed_validation_rmse"], ascending=[True, True], na_position="last").iloc[0]
        lines.append(f"- {horizon_mode}: best feature set={best['feature_set_name']} | fixed_best={best['fixed_best_experiment']} | rolling_best={best.get('rolling_best_experiment', '')}")
    lines.append("")
    lines.append("## China Diagnostics")
    for row in china_diagnostics.itertuples(index=False):
        lines.append(f"- EQ_CN | {row.horizon_mode}: experiment={row.experiment_name}, rmse={row.rmse:.4f}, corr={row.corr:.4f}, sign_accuracy={row.sign_accuracy:.4f}, rmse_rank_worst_first={row.rmse_rank_worst_first:.0f}.")
    lines.append("")
    lines.append("## New Sleeve Diagnostics")
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        chunk = new_sleeve_diagnostics.loc[new_sleeve_diagnostics["horizon_mode"] == horizon_mode].copy()
        if chunk.empty:
            continue
        lines.append(f"- {horizon_mode}: selected winner={chunk['selected_experiment_name'].iloc[0]}")
        for row in chunk.sort_values(["rmse", "corr"], ascending=[False, True]).itertuples(index=False):
            lines.append(f"  - sleeve={row.sleeve_id}, rmse={row.rmse:.4f}, corr={row.corr:.4f}, sign_accuracy={row.sign_accuracy:.4f}, rmse_rank_worst_first={row.rmse_rank_worst_first:.0f}.")
    lines.append("")
    lines.append("## Practical Interpretation")
    lines.append("- Separate-horizon models remain the primary benchmark; shared 60m+120m remains an ablation and is not clearly stronger.")
    lines.append("- Linear models are the strongest practical family in v4. Trees are serviceable, but they do not displace ridge / elastic net as the default benchmark anchor.")
    lines.append("- The richer v4 universe broadens the benchmark problem. It adds usable sleeves, but the new non-US fixed-income sleeves are harder than the legacy core.")
    lines.append(f"- Single strongest benchmark to beat next: {strongest_overall['experiment_name']} with test_rmse={strongest_overall['test_rmse']:.4f}, test_corr={strongest_overall['test_corr']:.4f}.")
    return "\n".join(lines) + "\n"


def run_prediction_benchmark_v4(*, project_root: Path, rolling_config: RollingConfig | None = None) -> PredictionBenchmarkOutputs:
    root = project_root.resolve()
    cfg = rolling_config or RollingConfig(
        min_train_months=48,
        validation_months=12,
        test_months=12,
        step_months=12,
        min_train_months_by_mode={"separate_120": 24, "shared_60_120": 24},
    )
    fixed_metrics, by_sleeve, preds_val, preds_test = _run_fixed_split(root)
    rolling_summary, rolling_fold_metrics, rolling_fold_manifest = _run_rolling_summary(root, cfg)
    feature_set_summary = _build_feature_set_summary(fixed_metrics, rolling_summary)
    metrics = _merge_fixed_and_rolling(fixed_metrics, rolling_summary)
    china_diagnostics = _build_china_diagnostics(metrics, by_sleeve)
    new_sleeve_diagnostics = _build_new_sleeve_diagnostics(metrics, by_sleeve)
    report_text = _render_report(
        metrics=metrics,
        feature_set_summary=feature_set_summary,
        rolling_summary=rolling_summary,
        china_diagnostics=china_diagnostics,
        new_sleeve_diagnostics=new_sleeve_diagnostics,
    )
    return PredictionBenchmarkOutputs(
        metrics=metrics,
        by_sleeve=by_sleeve,
        feature_set_summary=feature_set_summary,
        rolling_summary=rolling_summary,
        china_diagnostics=china_diagnostics,
        new_sleeve_diagnostics=new_sleeve_diagnostics,
        predictions_validation=preds_val,
        predictions_test=preds_test,
        rolling_fold_metrics=rolling_fold_metrics,
        rolling_fold_manifest=rolling_fold_manifest,
        report_text=report_text,
    )


def write_prediction_benchmark_v4_outputs(*, project_root: Path, outputs: PredictionBenchmarkOutputs) -> None:
    paths = default_paths(project_root)
    reports_dir = paths.reports_dir
    write_text(outputs.report_text, reports_dir / "v4_prediction_benchmark_report.md")
    write_csv(outputs.metrics, reports_dir / "v4_prediction_benchmark_metrics.csv")
    write_csv(outputs.by_sleeve, reports_dir / "v4_prediction_benchmark_by_sleeve.csv")
    write_csv(outputs.feature_set_summary, reports_dir / "v4_prediction_feature_set_summary.csv")
    write_csv(outputs.rolling_summary, reports_dir / "v4_prediction_rolling_summary.csv")
    write_csv(outputs.china_diagnostics, reports_dir / "v4_china_prediction_diagnostics.csv")
    write_csv(outputs.new_sleeve_diagnostics, reports_dir / "v4_new_sleeve_prediction_diagnostics.csv")
    write_csv(outputs.rolling_fold_metrics, reports_dir / "v4_prediction_rolling_fold_metrics.csv")
    write_csv(outputs.rolling_fold_manifest, reports_dir / "v4_prediction_rolling_fold_manifest.csv")
    write_parquet(outputs.predictions_validation, paths.data_out_dir / "predictions_validation_v4_benchmark.parquet")
    write_parquet(outputs.predictions_test, paths.data_out_dir / "predictions_test_v4_benchmark.parquet")
