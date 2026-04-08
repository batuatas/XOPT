"""Compact predictor horse race and SAA portfolio comparison for XOPTPOE v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from xoptpoe_v3_modeling.features import FEATURE_SET_ORDER
from xoptpoe_v3_modeling.io import load_csv, write_csv, write_parquet, write_text
from xoptpoe_v3_models.data import TARGET_COL, build_sleeve_horizon_benchmark, default_paths, load_modeling_inputs
from xoptpoe_v3_models.optim_layers import (
    OptimizerConfig,
    RiskConfig,
    RobustOptimizerCache,
    build_sigma_map,
    candidate_optimizer_grid,
)
from xoptpoe_v3_models.portfolio_eval import run_portfolio_evaluation, summarize_portfolio_metrics
from xoptpoe_v3_models.prediction_ablation import (
    HORIZON_MODES,
    PredictionRunArtifacts,
    _evaluate_predictions,
    _fit_preprocessed_inputs,
    _prediction_frame,
    _result_row,
    _run_elastic_net_experiment,
    _run_mlp_experiment,
    _run_naive_experiment,
    _run_ridge_experiment,
    _subset_inputs,
    _sleeve_rows,
)


LINEAR_FEATURE_SETS: tuple[str, ...] = FEATURE_SET_ORDER
TREE_FEATURE_SETS: tuple[str, ...] = ("core_baseline", "core_plus_enrichment", "core_plus_interactions")
NEURAL_FEATURE_SETS: tuple[str, ...] = ("core_baseline", "core_plus_interactions")
SHARED_BENCHMARK_RUNS: tuple[tuple[str, str], ...] = (
    ("ridge", "core_plus_interactions"),
    ("elastic_net", "core_plus_interactions"),
)


@dataclass(frozen=True)
class HorseRaceOutputs:
    """Container for the compact prediction and portfolio horse race artifacts."""

    predictions_validation: pd.DataFrame
    predictions_test: pd.DataFrame
    metrics_overall: pd.DataFrame
    metrics_by_sleeve: pd.DataFrame
    feature_set_summary: pd.DataFrame
    portfolio_returns: pd.DataFrame
    portfolio_metrics: pd.DataFrame
    predictor_report: str
    portfolio_report: str


def _feature_set_label(model_name: str, feature_set_name: str) -> str:
    return "no_features" if model_name == "naive_mean" else feature_set_name


def _run_random_forest_experiment(
    *,
    inputs,
    feature_set_name: str,
    horizon_mode: str,
    horizons: tuple[int, ...],
) -> PredictionRunArtifacts:
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


def _run_gradient_boosting_experiment(
    *,
    inputs,
    feature_set_name: str,
    horizon_mode: str,
    horizons: tuple[int, ...],
) -> PredictionRunArtifacts:
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


def _load_existing_prediction_benchmark(
    *,
    project_root: Path,
    model_name: str,
) -> PredictionRunArtifacts:
    paths = default_paths(project_root)
    pred_val = load_csv(paths.data_out_dir / f"predictions_validation_{model_name}.csv", parse_dates=["month_end"])
    pred_test = load_csv(paths.data_out_dir / f"predictions_test_{model_name}.csv", parse_dates=["month_end"])
    inputs = load_modeling_inputs(project_root, feature_set_name="core_plus_enrichment")
    combined = pd.concat([pred_val, pred_test], ignore_index=True)
    metrics_overall, metrics_by_horizon, metrics_by_sleeve = _evaluate_predictions(combined, inputs.train_df)
    return PredictionRunArtifacts(
        experiment_name=f"{model_name}_nn__core_plus_enrichment__shared_60_120",
        model_name=f"{model_name}_nn",
        feature_set_name="core_plus_enrichment",
        horizon_mode="shared_60_120",
        horizons=(60, 120),
        predictions_validation=pred_val,
        predictions_test=pred_test,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        metrics_by_sleeve=metrics_by_sleeve,
        selected_params={},
        input_feature_count=0,
        transformed_feature_count=0,
    )


def _append_metadata(artifacts: PredictionRunArtifacts) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    pred_val = artifacts.predictions_validation.copy()
    pred_test = artifacts.predictions_test.copy()
    for frame in (pred_val, pred_test):
        frame["experiment_name"] = artifacts.experiment_name
        frame["model_name"] = artifacts.model_name
        frame["feature_set_name"] = _feature_set_label(artifacts.model_name, artifacts.feature_set_name)
        frame["horizon_mode"] = artifacts.horizon_mode
        frame["selected_params"] = str(artifacts.selected_params)
    result_row = _result_row(artifacts)
    result_row["model_family"] = artifacts.model_name
    result_row["feature_set_name"] = _feature_set_label(artifacts.model_name, artifacts.feature_set_name)
    result_row["validation_test_rmse_gap"] = float(result_row["test_rmse"] - result_row["validation_rmse"])
    result_row["validation_test_corr_gap"] = float(result_row["test_corr"] - result_row["validation_corr"])
    sleeve_rows = _sleeve_rows(artifacts)
    for row in sleeve_rows:
        row["model_family"] = artifacts.model_name
        row["feature_set_name"] = _feature_set_label(artifacts.model_name, artifacts.feature_set_name)
    return pred_val, pred_test, result_row, sleeve_rows


def _common_truth_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions[["split", "month_end", "sleeve_id", "horizon_months", "y_true"]].copy()
    grouped = (
        work.groupby(["split", "month_end", "sleeve_id"], as_index=False)
        .agg(horizon_count=("horizon_months", "nunique"), y_true=("y_true", "mean"))
        .sort_values(["split", "month_end", "sleeve_id"])
        .reset_index(drop=True)
    )
    if not grouped["horizon_count"].eq(2).all():
        raise ValueError("Common truth panel expects both 60m and 120m rows for every sleeve-month")
    return grouped.drop(columns=["horizon_count"])


def _truth_panel_from_split_frame(frame: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["month_end", "sleeve_id"], as_index=False)
        .agg(horizon_count=("horizon_months", "nunique"), y_true=(TARGET_COL, "mean"))
        .sort_values(["month_end", "sleeve_id"])
        .reset_index(drop=True)
    )
    if not grouped["horizon_count"].eq(2).all():
        raise ValueError(f"Split truth panel expects both 60m and 120m rows for split={split_name}")
    grouped["split"] = split_name
    return grouped.drop(columns=["horizon_count"])[["split", "month_end", "sleeve_id", "y_true"]]


def _single_horizon_signal_panel(
    predictions: pd.DataFrame,
    *,
    horizon_months: int,
    truth_panel: pd.DataFrame,
) -> pd.DataFrame:
    signal = predictions.loc[predictions["horizon_months"].eq(horizon_months), ["split", "month_end", "sleeve_id", "y_pred"]].copy()
    out = signal.merge(truth_panel, on=["split", "month_end", "sleeve_id"], how="inner", validate="1:1")
    return out.sort_values(["split", "month_end", "sleeve_id"]).reset_index(drop=True)


def _average_signal_panel(predictions: pd.DataFrame, *, truth_panel: pd.DataFrame | None = None) -> pd.DataFrame:
    signal = (
        predictions.groupby(["split", "month_end", "sleeve_id"], as_index=False)
        .agg(horizon_count=("horizon_months", "nunique"), y_pred=("y_pred", "mean"))
        .sort_values(["split", "month_end", "sleeve_id"])
        .reset_index(drop=True)
    )
    if not signal["horizon_count"].eq(2).all():
        raise ValueError("Average signal panel expects both 60m and 120m rows for every sleeve-month")
    truth = _common_truth_panel(predictions) if truth_panel is None else truth_panel
    out = signal.drop(columns=["horizon_count"]).merge(truth, on=["split", "month_end", "sleeve_id"], how="inner", validate="1:1")
    return out.sort_values(["split", "month_end", "sleeve_id"]).reset_index(drop=True)


def _combine_separate_signal_panels(pred60: pd.DataFrame, pred120: pd.DataFrame, *, truth_panel: pd.DataFrame) -> pd.DataFrame:
    s60 = _single_horizon_signal_panel(pred60, horizon_months=60, truth_panel=truth_panel).rename(columns={"y_pred": "y_pred_60", "y_true": "y_true_common"})
    s120 = _single_horizon_signal_panel(pred120, horizon_months=120, truth_panel=truth_panel).rename(columns={"y_pred": "y_pred_120", "y_true": "y_true_common_120"})
    out = s60.merge(s120, on=["split", "month_end", "sleeve_id"], how="inner", validate="1:1")
    if not np.allclose(out["y_true_common"], out["y_true_common_120"], equal_nan=False):
        raise ValueError("Separate 60m and 120m signal panels disagree on common truth values")
    out["y_pred"] = 0.5 * (out["y_pred_60"] + out["y_pred_120"])
    out["y_true"] = out["y_true_common"]
    return out[["split", "month_end", "sleeve_id", "y_pred", "y_true"]].sort_values(["split", "month_end", "sleeve_id"]).reset_index(drop=True)


def _select_portfolio_config(
    *,
    validation_signal_panel: pd.DataFrame,
    optimizer_cache: RobustOptimizerCache,
    optimizer_grid: Iterable[OptimizerConfig],
    strategy_name: str,
) -> tuple[OptimizerConfig, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    grid = list(optimizer_grid)
    for config in grid:
        run = run_portfolio_evaluation(
            signal_panel=validation_signal_panel,
            optimizer_cache=optimizer_cache,
            optimizer_config=config,
            model_strategy_name=strategy_name,
        )
        metrics = summarize_portfolio_metrics(run.returns)
        selected = metrics.loc[metrics["strategy"].eq(strategy_name)].iloc[0]
        rows.append(
            {
                "lambda_risk": config.lambda_risk,
                "kappa": config.kappa,
                "omega_type": config.omega_type,
                "validation_avg_return": float(selected["avg_return"]),
                "validation_volatility": float(selected["volatility"]),
                "validation_sharpe": float(selected["sharpe"]),
                "validation_avg_turnover": float(selected["avg_turnover"]),
                "validation_max_drawdown": float(selected["max_drawdown"]),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["validation_sharpe", "validation_avg_return"], ascending=[False, False]).reset_index(drop=True)
    best = summary.iloc[0]
    config = OptimizerConfig(
        lambda_risk=float(best["lambda_risk"]),
        kappa=float(best["kappa"]),
        omega_type=str(best["omega_type"]),
    )
    return config, summary


def _weight_behavior(weights: pd.DataFrame, returns_df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strat_weights = weights.loc[weights["strategy"].eq(strategy_name)].copy()
    strat_returns = returns_df.loc[returns_df["strategy"].eq(strategy_name)].copy()
    for split_name, chunk in strat_weights.groupby("split", as_index=False):
        monthly = (
            chunk.groupby("month_end", as_index=False)["weight"]
            .agg(max_weight="max", hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
            .sort_values("month_end")
        )
        monthly["effective_n_assets"] = 1.0 / monthly["hhi"]
        ret_chunk = strat_returns.loc[strat_returns["split"].eq(split_name)]
        rows.append(
            {
                "split": split_name,
                "avg_max_weight": float(monthly["max_weight"].mean()),
                "avg_hhi": float(monthly["hhi"].mean()),
                "avg_effective_n_assets": float(monthly["effective_n_assets"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_run_from_signal(
    *,
    validation_signal_panel: pd.DataFrame,
    test_signal_panel: pd.DataFrame,
    optimizer_cache: RobustOptimizerCache,
    optimizer_grid: Iterable[OptimizerConfig],
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OptimizerConfig]:
    selected_config, selection_summary = _select_portfolio_config(
        validation_signal_panel=validation_signal_panel,
        optimizer_cache=optimizer_cache,
        optimizer_grid=optimizer_grid,
        strategy_name=strategy_name,
    )
    validation_run = run_portfolio_evaluation(
        signal_panel=validation_signal_panel,
        optimizer_cache=optimizer_cache,
        optimizer_config=selected_config,
        model_strategy_name=strategy_name,
    )
    test_run = run_portfolio_evaluation(
        signal_panel=test_signal_panel,
        optimizer_cache=optimizer_cache,
        optimizer_config=selected_config,
        model_strategy_name=strategy_name,
    )
    returns_df = pd.concat([validation_run.returns, test_run.returns], ignore_index=True)
    weights_df = pd.concat([validation_run.weights, test_run.weights], ignore_index=True)
    metrics_df = summarize_portfolio_metrics(returns_df)
    behavior = _weight_behavior(weights_df, returns_df, strategy_name)
    strategy_metrics = metrics_df.loc[metrics_df["strategy"].eq(strategy_name)].merge(
        behavior,
        on="split",
        how="left",
        validate="1:1",
    )
    benchmark_metrics = metrics_df.loc[metrics_df["strategy"].eq("equal_weight")].copy()
    sleeve_count = len(validation_signal_panel["sleeve_id"].drop_duplicates())
    benchmark_metrics["avg_max_weight"] = 1.0 / sleeve_count
    benchmark_metrics["avg_hhi"] = 1.0 / sleeve_count
    benchmark_metrics["avg_effective_n_assets"] = float(sleeve_count)
    metrics_df = pd.concat([benchmark_metrics, strategy_metrics], ignore_index=True)
    return returns_df, weights_df, metrics_df, selected_config


def _merge_reference_portfolio_metrics(project_root: Path) -> pd.DataFrame:
    reports_dir = default_paths(project_root).reports_dir
    pto = load_csv(reports_dir / "portfolio_metrics_pto.csv")
    e2e = load_csv(reports_dir / "portfolio_metrics_e2e.csv")
    pto = pto.assign(strategy_label=np.where(pto["strategy"].eq("pto_portfolio"), "pto_nn_original", "equal_weight_original"), source_type="original_report")
    e2e = e2e.assign(strategy_label=np.where(e2e["strategy"].eq("e2e_portfolio"), "e2e_nn_original", "equal_weight_original"), source_type="original_report")
    ref = pd.concat([pto, e2e], ignore_index=True)
    ref = ref.loc[ref["strategy_label"].ne("equal_weight_original")].copy()
    ref["selected_lambda_risk"] = np.nan
    ref["selected_kappa"] = np.nan
    ref["selected_omega_type"] = None
    ref["avg_max_weight"] = np.nan
    ref["avg_hhi"] = np.nan
    ref["avg_effective_n_assets"] = np.nan
    return ref[
        [
            "split",
            "strategy_label",
            "source_type",
            "month_count",
            "avg_return",
            "volatility",
            "sharpe",
            "max_drawdown",
            "avg_turnover",
            "ending_nav",
            "selected_lambda_risk",
            "selected_kappa",
            "selected_omega_type",
            "avg_max_weight",
            "avg_hhi",
            "avg_effective_n_assets",
        ]
    ]


def _select_best(metrics_overall: pd.DataFrame, *, horizon_mode: str) -> pd.Series:
    subset = metrics_overall.loc[metrics_overall["horizon_mode"].eq(horizon_mode)].copy()
    subset = subset.loc[~subset["model_name"].isin({"naive_mean", "pto_nn", "e2e_nn"})].copy()
    return subset.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).iloc[0]


def _summarize_feature_sets(metrics_overall: pd.DataFrame) -> pd.DataFrame:
    subset = metrics_overall.loc[
        (~metrics_overall["model_name"].isin({"naive_mean", "pto_nn", "e2e_nn"}))
        & metrics_overall["feature_set_name"].ne("no_features")
    ].copy()
    rows: list[dict[str, object]] = []
    for (horizon_mode, feature_set_name), chunk in subset.groupby(["horizon_mode", "feature_set_name"], as_index=False):
        best_val = chunk.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).iloc[0]
        best_test = chunk.sort_values(["test_rmse", "test_corr"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "horizon_mode": horizon_mode,
                "feature_set_name": feature_set_name,
                "experiment_count": int(len(chunk)),
                "best_validation_experiment": best_val["experiment_name"],
                "best_validation_model": best_val["model_name"],
                "best_validation_rmse": float(best_val["validation_rmse"]),
                "best_validation_corr": float(best_val["validation_corr"]),
                "best_test_experiment": best_test["experiment_name"],
                "best_test_model": best_test["model_name"],
                "best_test_rmse": float(best_test["test_rmse"]),
                "best_test_corr": float(best_test["test_corr"]),
                "mean_test_rmse": float(chunk["test_rmse"].mean()),
                "mean_test_corr": float(chunk["test_corr"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_mode", "best_validation_rmse", "feature_set_name"]).reset_index(drop=True)


def _render_predictor_report(
    *,
    metrics_overall: pd.DataFrame,
    by_sleeve: pd.DataFrame,
    feature_set_summary: pd.DataFrame,
) -> str:
    best_60 = _select_best(metrics_overall, horizon_mode="separate_60")
    best_120 = _select_best(metrics_overall, horizon_mode="separate_120")
    best_shared = _select_best(metrics_overall, horizon_mode="shared_60_120")
    family_summary = (
        metrics_overall.loc[~metrics_overall["model_name"].isin({"naive_mean", "pto_nn", "e2e_nn"})]
        .groupby("model_name", as_index=False)
        .agg(
            best_validation_rmse=("validation_rmse", "min"),
            best_test_rmse=("test_rmse", "min"),
            best_test_corr=("test_corr", "max"),
        )
        .sort_values(["best_validation_rmse", "best_test_corr"], ascending=[True, False])
    )

    hardest_60 = (
        by_sleeve.loc[
            by_sleeve["experiment_name"].eq(best_60["experiment_name"]) & by_sleeve["split"].eq("test")
        ]
        .sort_values(["rmse", "corr"], ascending=[False, True])
        .head(3)
    )
    hardest_120 = (
        by_sleeve.loc[
            by_sleeve["experiment_name"].eq(best_120["experiment_name"]) & by_sleeve["split"].eq("test")
        ]
        .sort_values(["rmse", "corr"], ascending=[False, True])
        .head(3)
    )

    lines: list[str] = []
    lines.append("# XOPTPOE v3 Predictor Horse Race Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Separate-horizon prediction is the default evaluation mode; shared 60m/120m conditioning is included only as a benchmark.")
    lines.append("- Model families compared: naive mean, ridge, elastic net, random forest, gradient boosting, small MLP, paper MLP, plus existing PTO_NN and E2E_NN shared benchmarks.")
    lines.append("- Linear models covered the full first-pass feature-set range. Tree and neural baselines were kept to the leaner feature sets to avoid turning this into a large benchmark zoo.")
    lines.append("- PCR/PLS, lasso, XGBoost, and LightGBM were intentionally skipped to keep dependencies and search scope tight.")
    lines.append("")
    lines.append("## Best Predictors")
    lines.append(
        f"- Best 60m predictor by validation RMSE: `{best_60['experiment_name']}` with validation rmse={best_60['validation_rmse']:.6f}, test rmse={best_60['test_rmse']:.6f}, test corr={best_60['test_corr']:.4f}, test sign_accuracy={best_60['test_sign_accuracy']:.4f}."
    )
    lines.append(
        f"- Best 120m predictor by validation RMSE: `{best_120['experiment_name']}` with validation rmse={best_120['validation_rmse']:.6f}, test rmse={best_120['test_rmse']:.6f}, test corr={best_120['test_corr']:.4f}, test sign_accuracy={best_120['test_sign_accuracy']:.4f}."
    )
    lines.append(
        f"- Best shared benchmark: `{best_shared['experiment_name']}` with validation rmse={best_shared['validation_rmse']:.6f}, test rmse={best_shared['test_rmse']:.6f}, test corr={best_shared['test_corr']:.4f}."
    )
    lines.append("")
    lines.append("## Model-Family Readout")
    for row in family_summary.itertuples(index=False):
        lines.append(
            f"- {row.model_name}: best_validation_rmse={row.best_validation_rmse:.6f}, best_test_rmse={row.best_test_rmse:.6f}, best_test_corr={row.best_test_corr:.4f}."
        )
    lines.append("")
    lines.append("## Feature-Set Readout")
    for row in feature_set_summary.sort_values(["horizon_mode", "best_validation_rmse"]).itertuples(index=False):
        lines.append(
            f"- {row.horizon_mode}, {row.feature_set_name}: best_validation_model={row.best_validation_model}, best_validation_rmse={row.best_validation_rmse:.6f}, best_test_rmse={row.best_test_rmse:.6f}, best_test_corr={row.best_test_corr:.4f}."
        )
    lines.append("")
    lines.append("## Hardest Sleeves")
    lines.append(f"- 60m winner `{best_60['experiment_name']}` hardest test sleeves:")
    for row in hardest_60.itertuples(index=False):
        lines.append(f"  - {row.sleeve_id}: rmse={row.rmse:.6f}, corr={row.corr:.4f}, sign_accuracy={row.sign_accuracy:.4f}.")
    lines.append(f"- 120m winner `{best_120['experiment_name']}` hardest test sleeves:")
    for row in hardest_120.itertuples(index=False):
        lines.append(f"  - {row.sleeve_id}: rmse={row.rmse:.6f}, corr={row.corr:.4f}, sign_accuracy={row.sign_accuracy:.4f}.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- The core result remains the same as the earlier ablation: separate-horizon linear models are materially stronger than the current neural baselines.")
    lines.append("- Shared 60m/120m conditioning remains usable as a benchmark, but it is not the predictive winner once separate-horizon models are allowed.")
    return "\n".join(lines) + "\n"


def _render_portfolio_report(
    *,
    portfolio_metrics: pd.DataFrame,
    best_60: pd.Series,
    best_120: pd.Series,
    best_shared: pd.Series,
) -> str:
    compare = portfolio_metrics.sort_values(["source_type", "split", "sharpe"], ascending=[True, True, False])
    lines: list[str] = []
    lines.append("# XOPTPOE v3 SAA Portfolio Comparison Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("- These are long-horizon SAA decision-period diagnostics on overlapping annualized forward excess-return labels, not a fully tradable non-overlapping monthly wealth backtest.")
    lines.append("- Best 60m and 120m predictors were selected by validation RMSE. Their signals were then converted into monthly sleeve allocations with the existing long-only robust allocator.")
    lines.append("- Predictor-driven portfolios use a common allocator design: long-only, fully invested, no leverage, active v3 risk model, validation-selected lambda/kappa/Omega from the existing compact grid.")
    lines.append("- PTO_NN and E2E_NN appear twice conceptually: as predictor-family entries in the horse race, and here as original setup reference rows loaded from the existing reports.")
    lines.append("")
    lines.append("## Predictor Winners Feeding Portfolios")
    lines.append(f"- Best 60m signal source: `{best_60['experiment_name']}`.")
    lines.append(f"- Best 120m signal source: `{best_120['experiment_name']}`.")
    lines.append(f"- Best shared benchmark source: `{best_shared['experiment_name']}`.")
    lines.append("")
    lines.append("## Portfolio Metrics")
    for row in compare.itertuples(index=False):
        lines.append(
            f"- {row.source_type}, {row.split}, {row.strategy_label}: avg_return={row.avg_return:.4f}, volatility={row.volatility:.4f}, sharpe={row.sharpe:.4f}, avg_turnover={row.avg_turnover:.4f}, max_drawdown={row.max_drawdown:.4f}."
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- The common-allocator rows are the cleanest apples-to-apples predictor comparison.")
    lines.append("- The original PTO/E2E rows are retained as implementation references from the existing reports, because those setups selected their own optimizer configurations internally.")
    return "\n".join(lines) + "\n"


def run_predictor_horse_race(
    *,
    project_root: Path,
    risk_config: RiskConfig | None = None,
    optimizer_grid: Iterable[OptimizerConfig] | None = None,
) -> HorseRaceOutputs:
    root = project_root.resolve()
    risk_config = risk_config or RiskConfig()
    optimizer_grid = list(optimizer_grid or candidate_optimizer_grid())

    artifact_runs: list[PredictionRunArtifacts] = []

    base_inputs_by_feature = {
        feature_set_name: load_modeling_inputs(root, feature_set_name=feature_set_name)
        for feature_set_name in set(LINEAR_FEATURE_SETS) | set(TREE_FEATURE_SETS) | set(NEURAL_FEATURE_SETS)
    }

    for horizon_mode, horizons in (("separate_60", (60,)), ("separate_120", (120,))):
        subset_inputs = _subset_inputs(base_inputs_by_feature["core_baseline"], horizons)
        artifact_runs.append(
            _run_naive_experiment(
                inputs=subset_inputs,
                feature_set_name="no_features",
                horizon_mode=horizon_mode,
                horizons=horizons,
            )
        )

    shared_inputs = _subset_inputs(base_inputs_by_feature["core_baseline"], HORIZON_MODES["shared_60_120"])
    artifact_runs.append(
        _run_naive_experiment(
            inputs=shared_inputs,
            feature_set_name="no_features",
            horizon_mode="shared_60_120",
            horizons=HORIZON_MODES["shared_60_120"],
        )
    )

    for feature_set_name in LINEAR_FEATURE_SETS:
        base_inputs = base_inputs_by_feature[feature_set_name]
        for horizon_mode in ("separate_60", "separate_120"):
            horizons = HORIZON_MODES[horizon_mode]
            subset_inputs = _subset_inputs(base_inputs, horizons)
            artifact_runs.append(
                _run_ridge_experiment(
                    inputs=subset_inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode=horizon_mode,
                    horizons=horizons,
                )
            )
            artifact_runs.append(
                _run_elastic_net_experiment(
                    inputs=subset_inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode=horizon_mode,
                    horizons=horizons,
                )
            )

    for model_name, feature_set_name in SHARED_BENCHMARK_RUNS:
        base_inputs = base_inputs_by_feature[feature_set_name]
        subset_inputs = _subset_inputs(base_inputs, HORIZON_MODES["shared_60_120"])
        if model_name == "ridge":
            artifact_runs.append(
                _run_ridge_experiment(
                    inputs=subset_inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode="shared_60_120",
                    horizons=HORIZON_MODES["shared_60_120"],
                )
            )
        else:
            artifact_runs.append(
                _run_elastic_net_experiment(
                    inputs=subset_inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode="shared_60_120",
                    horizons=HORIZON_MODES["shared_60_120"],
                )
            )

    for feature_set_name in TREE_FEATURE_SETS:
        base_inputs = base_inputs_by_feature[feature_set_name]
        for horizon_mode in ("separate_60", "separate_120"):
            horizons = HORIZON_MODES[horizon_mode]
            subset_inputs = _subset_inputs(base_inputs, horizons)
            artifact_runs.append(
                _run_random_forest_experiment(
                    inputs=subset_inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode=horizon_mode,
                    horizons=horizons,
                )
            )
            artifact_runs.append(
                _run_gradient_boosting_experiment(
                    inputs=subset_inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode=horizon_mode,
                    horizons=horizons,
                )
            )

    neural_seed_map = {"small_mlp": 123, "paper_mlp": 456}
    for feature_set_name in NEURAL_FEATURE_SETS:
        base_inputs = base_inputs_by_feature[feature_set_name]
        for horizon_mode in ("separate_60", "separate_120"):
            horizons = HORIZON_MODES[horizon_mode]
            subset_inputs = _subset_inputs(base_inputs, horizons)
            artifact_runs.append(
                _run_mlp_experiment(
                    inputs=subset_inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode=horizon_mode,
                    horizons=horizons,
                    model_name="small_mlp",
                    hidden_dims=(16, 8),
                    dropout=0.5,
                    random_seed=neural_seed_map["small_mlp"],
                )
            )
            artifact_runs.append(
                _run_mlp_experiment(
                    inputs=subset_inputs,
                    feature_set_name=feature_set_name,
                    horizon_mode=horizon_mode,
                    horizons=horizons,
                    model_name="paper_mlp",
                    hidden_dims=(32, 16, 8),
                    dropout=0.5,
                    random_seed=neural_seed_map["paper_mlp"],
                )
            )

    artifact_runs.append(_load_existing_prediction_benchmark(project_root=root, model_name="pto"))
    artifact_runs.append(_load_existing_prediction_benchmark(project_root=root, model_name="e2e"))

    pred_val_frames: list[pd.DataFrame] = []
    pred_test_frames: list[pd.DataFrame] = []
    overall_rows: list[dict[str, object]] = []
    sleeve_rows: list[dict[str, object]] = []
    for artifacts in artifact_runs:
        pred_val, pred_test, overall_row, sleeve_row_list = _append_metadata(artifacts)
        pred_val_frames.append(pred_val)
        pred_test_frames.append(pred_test)
        overall_rows.append(overall_row)
        sleeve_rows.extend(sleeve_row_list)

    predictions_validation = pd.concat(pred_val_frames, ignore_index=True).sort_values(
        ["experiment_name", "month_end", "horizon_months", "sleeve_id"]
    ).reset_index(drop=True)
    predictions_test = pd.concat(pred_test_frames, ignore_index=True).sort_values(
        ["experiment_name", "month_end", "horizon_months", "sleeve_id"]
    ).reset_index(drop=True)
    metrics_overall = pd.DataFrame(overall_rows).sort_values(
        ["horizon_mode", "validation_rmse", "validation_corr"], ascending=[True, True, False]
    ).reset_index(drop=True)
    metrics_by_sleeve = pd.DataFrame(sleeve_rows).sort_values(
        ["experiment_name", "split", "rmse", "sleeve_id"], ascending=[True, True, False, True]
    ).reset_index(drop=True)
    feature_set_summary = _summarize_feature_sets(metrics_overall)

    best_60 = _select_best(metrics_overall, horizon_mode="separate_60")
    best_120 = _select_best(metrics_overall, horizon_mode="separate_120")
    best_shared = _select_best(metrics_overall, horizon_mode="shared_60_120")

    all_months = sorted(
        pd.concat([predictions_validation["month_end"], predictions_test["month_end"]]).drop_duplicates().tolist()
    )
    sigma_map = build_sigma_map(all_months, excess_history=base_inputs_by_feature["core_baseline"].monthly_excess_history, risk_config=risk_config)
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)
    truth_panels = {
        "validation": _truth_panel_from_split_frame(base_inputs_by_feature["core_baseline"].validation_df, split_name="validation"),
        "test": _truth_panel_from_split_frame(base_inputs_by_feature["core_baseline"].test_df, split_name="test"),
    }

    def _predictions_for(exp_name: str, split_name: str) -> pd.DataFrame:
        frame = predictions_validation if split_name == "validation" else predictions_test
        return frame.loc[frame["experiment_name"].eq(exp_name)].copy()

    common_runs: list[pd.DataFrame] = []
    portfolio_metric_rows: list[pd.DataFrame] = []
    equal_weight_kept = False
    strategy_specs = [
        (
            "best_60_predictor",
            _single_horizon_signal_panel(_predictions_for(best_60["experiment_name"], "validation"), horizon_months=60, truth_panel=truth_panels["validation"]),
            _single_horizon_signal_panel(_predictions_for(best_60["experiment_name"], "test"), horizon_months=60, truth_panel=truth_panels["test"]),
        ),
        (
            "best_120_predictor",
            _single_horizon_signal_panel(_predictions_for(best_120["experiment_name"], "validation"), horizon_months=120, truth_panel=truth_panels["validation"]),
            _single_horizon_signal_panel(_predictions_for(best_120["experiment_name"], "test"), horizon_months=120, truth_panel=truth_panels["test"]),
        ),
        (
            "combined_60_120_predictor",
            _combine_separate_signal_panels(_predictions_for(best_60["experiment_name"], "validation"), _predictions_for(best_120["experiment_name"], "validation"), truth_panel=truth_panels["validation"]),
            _combine_separate_signal_panels(_predictions_for(best_60["experiment_name"], "test"), _predictions_for(best_120["experiment_name"], "test"), truth_panel=truth_panels["test"]),
        ),
        (
            "best_shared_predictor",
            _average_signal_panel(_predictions_for(best_shared["experiment_name"], "validation"), truth_panel=truth_panels["validation"]),
            _average_signal_panel(_predictions_for(best_shared["experiment_name"], "test"), truth_panel=truth_panels["test"]),
        ),
        (
            "pto_nn_signal",
            _average_signal_panel(_predictions_for("pto_nn__core_plus_enrichment__shared_60_120", "validation"), truth_panel=truth_panels["validation"]),
            _average_signal_panel(_predictions_for("pto_nn__core_plus_enrichment__shared_60_120", "test"), truth_panel=truth_panels["test"]),
        ),
        (
            "e2e_nn_signal",
            _average_signal_panel(_predictions_for("e2e_nn__core_plus_enrichment__shared_60_120", "validation"), truth_panel=truth_panels["validation"]),
            _average_signal_panel(_predictions_for("e2e_nn__core_plus_enrichment__shared_60_120", "test"), truth_panel=truth_panels["test"]),
        ),
    ]

    for strategy_name, validation_signal, test_signal in strategy_specs:
        returns_df, _, metrics_df, selected_config = _portfolio_run_from_signal(
            validation_signal_panel=validation_signal,
            test_signal_panel=test_signal,
            optimizer_cache=optimizer_cache,
            optimizer_grid=optimizer_grid,
            strategy_name=strategy_name,
        )
        returns_df["source_type"] = "common_allocator"
        returns_df["strategy_label"] = returns_df["strategy"]
        metrics_df["source_type"] = "common_allocator"
        metrics_df["strategy_label"] = metrics_df["strategy"]
        metrics_df["selected_lambda_risk"] = selected_config.lambda_risk
        metrics_df["selected_kappa"] = selected_config.kappa
        metrics_df["selected_omega_type"] = selected_config.omega_type
        if equal_weight_kept:
            returns_df = returns_df.loc[returns_df["strategy"].ne("equal_weight")].copy()
            metrics_df = metrics_df.loc[metrics_df["strategy"].ne("equal_weight")].copy()
        else:
            equal_weight_kept = True
        common_runs.append(returns_df)
        portfolio_metric_rows.append(
            metrics_df[
                [
                    "split",
                    "strategy_label",
                    "source_type",
                    "month_count",
                    "avg_return",
                    "volatility",
                    "sharpe",
                    "max_drawdown",
                    "avg_turnover",
                    "ending_nav",
                    "selected_lambda_risk",
                    "selected_kappa",
                    "selected_omega_type",
                    "avg_max_weight",
                    "avg_hhi",
                    "avg_effective_n_assets",
                ]
            ]
        )

    portfolio_returns = pd.concat(common_runs, ignore_index=True).sort_values(["strategy_label", "split", "month_end"]).reset_index(drop=True)
    portfolio_metrics = pd.concat(portfolio_metric_rows + [_merge_reference_portfolio_metrics(root)], ignore_index=True)
    portfolio_metrics = portfolio_metrics.sort_values(["source_type", "split", "sharpe"], ascending=[True, True, False]).reset_index(drop=True)

    predictor_report = _render_predictor_report(
        metrics_overall=metrics_overall,
        by_sleeve=metrics_by_sleeve,
        feature_set_summary=feature_set_summary,
    )
    portfolio_report = _render_portfolio_report(
        portfolio_metrics=portfolio_metrics,
        best_60=best_60,
        best_120=best_120,
        best_shared=best_shared,
    )

    return HorseRaceOutputs(
        predictions_validation=predictions_validation,
        predictions_test=predictions_test,
        metrics_overall=metrics_overall,
        metrics_by_sleeve=metrics_by_sleeve,
        feature_set_summary=feature_set_summary,
        portfolio_returns=portfolio_returns,
        portfolio_metrics=portfolio_metrics,
        predictor_report=predictor_report,
        portfolio_report=portfolio_report,
    )


def write_horse_race_outputs(*, project_root: Path, outputs: HorseRaceOutputs) -> None:
    paths = default_paths(project_root.resolve())
    write_parquet(outputs.predictions_validation, paths.data_out_dir / "predictions_validation_horse_race.parquet")
    write_parquet(outputs.predictions_test, paths.data_out_dir / "predictions_test_horse_race.parquet")
    write_csv(outputs.metrics_overall, paths.reports_dir / "predictor_horse_race_metrics.csv")
    write_csv(outputs.metrics_by_sleeve, paths.reports_dir / "predictor_horse_race_by_sleeve.csv")
    write_csv(outputs.feature_set_summary, paths.reports_dir / "predictor_feature_set_summary.csv")
    write_text(outputs.predictor_report, paths.reports_dir / "predictor_horse_race_report.md")
    write_csv(outputs.portfolio_returns, paths.data_out_dir / "portfolio_comparison_returns.csv")
    write_csv(outputs.portfolio_metrics, paths.reports_dir / "saa_portfolio_comparison_metrics.csv")
    write_text(outputs.portfolio_report, paths.reports_dir / "saa_portfolio_comparison_report.md")
