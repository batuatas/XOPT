"""Narrow tuning and diagnostic study for XOPTPOE v3 PTO/E2E models."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from xoptpoe_v3_models.data import (
    DEFAULT_HORIZONS,
    SLEEVE_ORDER,
    TARGET_COL,
    LoadedModelingInputs,
    build_sleeve_horizon_benchmark,
    load_modeling_inputs,
)
from xoptpoe_v3_models.evaluate import evaluate_predictions
from xoptpoe_v3_models.losses import mse_loss, objective_score
from xoptpoe_v3_models.networks import PredictorMLP
from xoptpoe_v3_models.optim_layers import (
    OptimizerConfig,
    RiskConfig,
    RobustOptimizerCache,
    build_sigma_map,
)
from xoptpoe_v3_models.portfolio_eval import run_portfolio_evaluation, summarize_portfolio_metrics
from xoptpoe_v3_models.preprocess import fit_preprocessor


@dataclass(frozen=True)
class ModelRunArtifacts:
    """Unified outputs from a PTO or E2E training run."""

    model_type: str
    horizons: tuple[int, ...]
    predictions_validation: pd.DataFrame
    predictions_test: pd.DataFrame
    metrics_overall: pd.DataFrame
    metrics_by_sleeve: pd.DataFrame
    metrics_by_horizon: pd.DataFrame
    training_history: pd.DataFrame
    portfolio_metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    portfolio_weights: pd.DataFrame
    selection_summary: pd.DataFrame
    selected_config: OptimizerConfig
    risk_config: RiskConfig
    feature_set_name: str
    selected_objective: str | None = None


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
    expected_rows = out["month_end"].nunique() * len(horizons) * len(SLEEVE_ORDER)
    if len(out) != expected_rows:
        raise ValueError(
            f"Filtered frame does not form a complete sleeve x horizon panel for horizons={horizons}: rows={len(out)}, expected={expected_rows}"
        )
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


def _reshape_monthly(
    frame: pd.DataFrame,
    values: np.ndarray | torch.Tensor,
    horizons: tuple[int, ...],
) -> tuple[list[pd.Timestamp], np.ndarray | torch.Tensor]:
    month_count = int(frame["month_end"].nunique())
    expected_rows = month_count * len(horizons) * len(SLEEVE_ORDER)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Split frame does not match expected stacked shape for horizons={horizons}: rows={len(frame)}, expected={expected_rows}"
        )
    months = [pd.Timestamp(v) for v in frame["month_end"].drop_duplicates().tolist()]
    if isinstance(values, torch.Tensor):
        reshaped = values.reshape(month_count, len(horizons), len(SLEEVE_ORDER))
        return months, reshaped.mean(dim=1)
    arr = np.asarray(values)
    reshaped = arr.reshape(month_count, len(horizons), len(SLEEVE_ORDER))
    return months, reshaped.mean(axis=1)


def _build_signal_panel(predictions: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    required = {"split", "month_end", "sleeve_id", "horizon_months", "y_true", "y_pred", "benchmark_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    work = predictions[["split", "month_end", "sleeve_id", "horizon_months", "y_true", "y_pred", "benchmark_pred"]].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    split_values = work["split"].dropna().unique().tolist()
    if len(split_values) != 1:
        raise ValueError("_build_signal_panel expects a single split at a time")
    split_name = str(split_values[0])

    expected = len(horizons)
    grouped = (
        work.groupby(["month_end", "sleeve_id"], as_index=False)
        .agg(
            horizon_count=("horizon_months", "nunique"),
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            benchmark_pred=("benchmark_pred", "mean"),
        )
        .sort_values(["month_end", "sleeve_id"])
        .reset_index(drop=True)
    )
    if not grouped["horizon_count"].eq(expected).all():
        raise ValueError(f"Signal aggregation expected {expected} horizon rows for every sleeve-month")
    grouped["split"] = split_name
    return grouped.drop(columns=["horizon_count"])


def _decision_score_over_months(
    *,
    months: list[pd.Timestamp],
    predicted_monthly: torch.Tensor,
    realized_monthly: torch.Tensor,
    sigma_map: dict[pd.Timestamp, np.ndarray],
    optimizer_cache: RobustOptimizerCache,
    optimizer_config: OptimizerConfig,
    objective_name: str,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for idx, month_end in enumerate(months):
        weights = optimizer_cache.solve(month_end, predicted_monthly[idx], optimizer_config)
        sigma_tensor = torch.tensor(sigma_map[pd.Timestamp(month_end)], dtype=predicted_monthly.dtype)
        score = objective_score(
            objective_name,
            weights=weights,
            realized_returns=realized_monthly[idx],
            sigma=sigma_tensor,
            lambda_risk=optimizer_config.lambda_risk,
        )
        scores.append(score)
    return torch.stack(scores).mean()


def _prepare_inputs(
    *,
    project_root: Path,
    feature_set_name: str,
    horizons: tuple[int, ...],
) -> tuple[LoadedModelingInputs, object, np.ndarray, np.ndarray, np.ndarray]:
    base_inputs = load_modeling_inputs(project_root, feature_set_name=feature_set_name)
    inputs = _subset_inputs(base_inputs, horizons)
    preprocessor = fit_preprocessor(
        inputs.train_df,
        feature_manifest=inputs.feature_manifest,
        feature_columns=inputs.feature_columns,
    )
    x_train, _ = preprocessor.transform(inputs.train_df)
    x_val, _ = preprocessor.transform(inputs.validation_df)
    x_test, _ = preprocessor.transform(inputs.test_df)
    return inputs, preprocessor, x_train, x_val, x_test


def run_pto_experiment(
    *,
    project_root: Path,
    feature_set_name: str,
    horizons: tuple[int, ...],
    random_seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    risk_config: RiskConfig,
    optimizer_grid: list[OptimizerConfig],
) -> ModelRunArtifacts:
    inputs, preprocessor, x_train, x_val, x_test = _prepare_inputs(
        project_root=project_root,
        feature_set_name=feature_set_name,
        horizons=horizons,
    )

    _set_seeds(random_seed)
    y_train = inputs.train_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_val = inputs.validation_df[TARGET_COL].to_numpy(dtype=np.float32)

    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.float32)
    val_x_tensor = torch.tensor(x_val, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val, dtype=torch.float32)

    model = PredictorMLP(input_dim=x_train.shape[1])
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = patience
    history_rows: list[dict[str, object]] = []

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

        train_loss_value = float(train_loss.detach().cpu().item())
        val_loss_value = float(val_loss.detach().cpu().item())
        improved = val_loss_value < (best_val_loss - 1e-8)
        if improved:
            best_val_loss = val_loss_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_value,
                "val_loss": val_loss_value,
                "is_best_epoch": int(improved),
            }
        )
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("PTO training did not produce a valid best state")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_pred = model(val_x_tensor).cpu().numpy()
        test_pred = model(torch.tensor(x_test, dtype=torch.float32)).cpu().numpy()

    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    pred_val_df = _prediction_frame(inputs.validation_df, val_pred, benchmark, split_name="validation")
    pred_test_df = _prediction_frame(inputs.test_df, test_pred, benchmark, split_name="test")

    all_months = sorted(pd.concat([pred_val_df["month_end"], pred_test_df["month_end"]]).drop_duplicates().tolist())
    sigma_map = build_sigma_map(all_months, excess_history=inputs.monthly_excess_history, risk_config=risk_config)
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    selection_rows: list[dict[str, object]] = []
    val_signal_panel = _build_signal_panel(pred_val_df, horizons)
    for config in optimizer_grid:
        portfolio_run = run_portfolio_evaluation(
            signal_panel=val_signal_panel,
            optimizer_cache=optimizer_cache,
            optimizer_config=config,
            model_strategy_name="pto_portfolio",
        )
        metrics = summarize_portfolio_metrics(portfolio_run.returns)
        selected = metrics.loc[metrics["strategy"].eq("pto_portfolio")].iloc[0]
        selection_rows.append(
            {
                "lambda_risk": config.lambda_risk,
                "kappa": config.kappa,
                "omega_type": config.omega_type,
                "validation_avg_return": float(selected["avg_return"]),
                "validation_volatility": float(selected["volatility"]),
                "validation_sharpe": float(selected["sharpe"]),
                "validation_max_drawdown": float(selected["max_drawdown"]),
                "validation_avg_turnover": float(selected["avg_turnover"]),
            }
        )

    selection_summary = pd.DataFrame(selection_rows).sort_values(
        ["validation_sharpe", "validation_avg_return"], ascending=[False, False]
    ).reset_index(drop=True)
    best_row = selection_summary.iloc[0]
    selected_config = OptimizerConfig(
        lambda_risk=float(best_row["lambda_risk"]),
        kappa=float(best_row["kappa"]),
        omega_type=str(best_row["omega_type"]),
    )

    portfolio_runs = []
    for pred_df in (pred_val_df, pred_test_df):
        signal_panel = _build_signal_panel(pred_df, horizons)
        split_run = run_portfolio_evaluation(
            signal_panel=signal_panel,
            optimizer_cache=optimizer_cache,
            optimizer_config=selected_config,
            model_strategy_name="pto_portfolio",
        )
        portfolio_runs.append(split_run)
    portfolio_returns = pd.concat([run.returns for run in portfolio_runs], ignore_index=True)
    portfolio_weights = pd.concat([run.weights for run in portfolio_runs], ignore_index=True)
    portfolio_metrics = summarize_portfolio_metrics(portfolio_returns)

    weight_lookup = portfolio_weights.loc[portfolio_weights["strategy"].eq("pto_portfolio")].rename(
        columns={"weight": "portfolio_weight"}
    )[["split", "month_end", "sleeve_id", "portfolio_weight"]]
    pred_val_df = pred_val_df.merge(weight_lookup, on=["split", "month_end", "sleeve_id"], how="left", validate="m:1")
    pred_test_df = pred_test_df.merge(weight_lookup, on=["split", "month_end", "sleeve_id"], how="left", validate="m:1")
    pred_val_df["selected_lambda_risk"] = selected_config.lambda_risk
    pred_val_df["selected_kappa"] = selected_config.kappa
    pred_val_df["selected_omega_type"] = selected_config.omega_type
    pred_test_df["selected_lambda_risk"] = selected_config.lambda_risk
    pred_test_df["selected_kappa"] = selected_config.kappa
    pred_test_df["selected_omega_type"] = selected_config.omega_type

    metrics_overall, metrics_by_sleeve, metrics_by_horizon = evaluate_predictions(
        pd.concat([pred_val_df, pred_test_df], ignore_index=True)
    )
    training_history = pd.DataFrame(history_rows)
    training_history["best_epoch_final"] = int(best_epoch)

    return ModelRunArtifacts(
        model_type="pto",
        horizons=horizons,
        predictions_validation=pred_val_df,
        predictions_test=pred_test_df,
        metrics_overall=metrics_overall,
        metrics_by_sleeve=metrics_by_sleeve,
        metrics_by_horizon=metrics_by_horizon,
        training_history=training_history,
        portfolio_metrics=portfolio_metrics,
        portfolio_returns=portfolio_returns,
        portfolio_weights=portfolio_weights,
        selection_summary=selection_summary,
        selected_config=selected_config,
        risk_config=risk_config,
        feature_set_name=feature_set_name,
    )


def run_e2e_experiment(
    *,
    project_root: Path,
    feature_set_name: str,
    horizons: tuple[int, ...],
    random_seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    risk_config: RiskConfig,
    optimizer_grid: list[OptimizerConfig],
    training_objective: str = "utility",
) -> ModelRunArtifacts:
    inputs, preprocessor, x_train, x_val, x_test = _prepare_inputs(
        project_root=project_root,
        feature_set_name=feature_set_name,
        horizons=horizons,
    )

    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    val_x_tensor = torch.tensor(x_val, dtype=torch.float32)
    test_x_tensor = torch.tensor(x_test, dtype=torch.float32)

    y_train_rows = inputs.train_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_val_rows = inputs.validation_df[TARGET_COL].to_numpy(dtype=np.float32)
    train_months, y_train_monthly = _reshape_monthly(inputs.train_df, y_train_rows, horizons)
    val_months, y_val_monthly = _reshape_monthly(inputs.validation_df, y_val_rows, horizons)
    y_train_monthly_tensor = torch.tensor(y_train_monthly, dtype=torch.float32)
    y_val_monthly_tensor = torch.tensor(y_val_monthly, dtype=torch.float32)

    sigma_map = build_sigma_map(
        sorted(set(train_months + val_months + [pd.Timestamp(v) for v in inputs.test_df["month_end"].drop_duplicates().tolist()])),
        excess_history=inputs.monthly_excess_history,
        risk_config=risk_config,
    )
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)
    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)

    selection_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    best_bundle: dict[str, object] | None = None
    candidate_id = 0

    for optimizer_config in optimizer_grid:
        candidate_id += 1
        _set_seeds(random_seed + candidate_id)
        model = PredictorMLP(input_dim=x_train.shape[1])
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_score = float("-inf")
        best_epoch = 0
        patience_left = patience

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred_train_rows = model(train_x_tensor)
            _, pred_train_monthly = _reshape_monthly(inputs.train_df, pred_train_rows, horizons)
            train_score = _decision_score_over_months(
                months=train_months,
                predicted_monthly=pred_train_monthly,
                realized_monthly=y_train_monthly_tensor,
                sigma_map=sigma_map,
                optimizer_cache=optimizer_cache,
                optimizer_config=optimizer_config,
                objective_name=training_objective,
            )
            train_loss = -train_score
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val_rows = model(val_x_tensor)
                _, pred_val_monthly = _reshape_monthly(inputs.validation_df, pred_val_rows, horizons)
                val_score = _decision_score_over_months(
                    months=val_months,
                    predicted_monthly=pred_val_monthly,
                    realized_monthly=y_val_monthly_tensor,
                    sigma_map=sigma_map,
                    optimizer_cache=optimizer_cache,
                    optimizer_config=optimizer_config,
                    objective_name=training_objective,
                )

            train_score_value = float(train_score.detach().cpu().item())
            val_score_value = float(val_score.detach().cpu().item())
            improved = val_score_value > (best_val_score + 1e-8)
            if improved:
                best_val_score = val_score_value
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_left = patience
            else:
                patience_left -= 1

            history_rows.append(
                {
                    "candidate_id": candidate_id,
                    "training_objective": training_objective,
                    "lambda_risk": optimizer_config.lambda_risk,
                    "kappa": optimizer_config.kappa,
                    "omega_type": optimizer_config.omega_type,
                    "epoch": epoch,
                    "train_score": train_score_value,
                    "val_score": val_score_value,
                    "is_best_epoch": int(improved),
                }
            )
            if patience_left <= 0:
                break

        if best_state is None:
            raise RuntimeError(f"E2E candidate {candidate_id} did not produce a valid best state")
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            val_pred_rows = model(val_x_tensor).cpu().numpy()
            test_pred_rows = model(test_x_tensor).cpu().numpy()

        pred_val_df = _prediction_frame(inputs.validation_df, val_pred_rows, benchmark, split_name="validation")
        pred_test_df = _prediction_frame(inputs.test_df, test_pred_rows, benchmark, split_name="test")

        val_signal_panel = _build_signal_panel(pred_val_df, horizons)
        val_portfolio_run = run_portfolio_evaluation(
            signal_panel=val_signal_panel,
            optimizer_cache=optimizer_cache,
            optimizer_config=optimizer_config,
            model_strategy_name="e2e_portfolio",
        )
        val_portfolio_metrics = summarize_portfolio_metrics(val_portfolio_run.returns)
        val_selected = val_portfolio_metrics.loc[val_portfolio_metrics["strategy"].eq("e2e_portfolio")].iloc[0]

        selection_row = {
            "candidate_id": candidate_id,
            "training_objective": training_objective,
            "lambda_risk": optimizer_config.lambda_risk,
            "kappa": optimizer_config.kappa,
            "omega_type": optimizer_config.omega_type,
            "best_epoch": best_epoch,
            "validation_score": best_val_score,
            "validation_avg_return": float(val_selected["avg_return"]),
            "validation_volatility": float(val_selected["volatility"]),
            "validation_portfolio_sharpe": float(val_selected["sharpe"]),
            "validation_max_drawdown": float(val_selected["max_drawdown"]),
            "validation_avg_turnover": float(val_selected["avg_turnover"]),
        }
        selection_rows.append(selection_row)

        candidate_rank_tuple = (
            float(val_selected["sharpe"]),
            best_val_score,
        )
        current_best_tuple = None
        if best_bundle is not None:
            current_best_tuple = (
                float(best_bundle["selection_row"]["validation_portfolio_sharpe"]),
                float(best_bundle["selection_row"]["validation_score"]),
            )
        if current_best_tuple is None or candidate_rank_tuple > current_best_tuple:
            test_signal_panel = _build_signal_panel(pred_test_df, horizons)
            test_portfolio_run = run_portfolio_evaluation(
                signal_panel=test_signal_panel,
                optimizer_cache=optimizer_cache,
                optimizer_config=optimizer_config,
                model_strategy_name="e2e_portfolio",
            )
            portfolio_returns = pd.concat([val_portfolio_run.returns, test_portfolio_run.returns], ignore_index=True)
            portfolio_weights = pd.concat([val_portfolio_run.weights, test_portfolio_run.weights], ignore_index=True)
            portfolio_metrics = summarize_portfolio_metrics(portfolio_returns)
            weight_lookup = portfolio_weights.loc[portfolio_weights["strategy"].eq("e2e_portfolio")].rename(
                columns={"weight": "portfolio_weight"}
            )[["split", "month_end", "sleeve_id", "portfolio_weight"]]
            pred_val_best = pred_val_df.merge(weight_lookup, on=["split", "month_end", "sleeve_id"], how="left", validate="m:1")
            pred_test_best = pred_test_df.merge(weight_lookup, on=["split", "month_end", "sleeve_id"], how="left", validate="m:1")
            pred_val_best["selected_lambda_risk"] = optimizer_config.lambda_risk
            pred_val_best["selected_kappa"] = optimizer_config.kappa
            pred_val_best["selected_omega_type"] = optimizer_config.omega_type
            pred_val_best["selected_training_objective"] = training_objective
            pred_test_best["selected_lambda_risk"] = optimizer_config.lambda_risk
            pred_test_best["selected_kappa"] = optimizer_config.kappa
            pred_test_best["selected_omega_type"] = optimizer_config.omega_type
            pred_test_best["selected_training_objective"] = training_objective
            best_bundle = {
                "predictions_validation": pred_val_best,
                "predictions_test": pred_test_best,
                "portfolio_returns": portfolio_returns,
                "portfolio_weights": portfolio_weights,
                "portfolio_metrics": portfolio_metrics,
                "selected_config": optimizer_config,
                "selected_objective": training_objective,
                "selection_row": selection_row,
            }

    if best_bundle is None:
        raise RuntimeError("No valid E2E candidate was selected")

    selection_summary = pd.DataFrame(selection_rows).sort_values(
        ["validation_portfolio_sharpe", "validation_score"], ascending=[False, False]
    ).reset_index(drop=True)
    training_history = pd.DataFrame(history_rows)
    combined_predictions = pd.concat(
        [best_bundle["predictions_validation"], best_bundle["predictions_test"]],
        ignore_index=True,
    )
    metrics_overall, metrics_by_sleeve, metrics_by_horizon = evaluate_predictions(combined_predictions)

    return ModelRunArtifacts(
        model_type="e2e",
        horizons=horizons,
        predictions_validation=best_bundle["predictions_validation"],
        predictions_test=best_bundle["predictions_test"],
        metrics_overall=metrics_overall,
        metrics_by_sleeve=metrics_by_sleeve,
        metrics_by_horizon=metrics_by_horizon,
        training_history=training_history,
        portfolio_metrics=best_bundle["portfolio_metrics"],
        portfolio_returns=best_bundle["portfolio_returns"],
        portfolio_weights=best_bundle["portfolio_weights"],
        selection_summary=selection_summary,
        selected_config=best_bundle["selected_config"],
        risk_config=risk_config,
        feature_set_name=feature_set_name,
        selected_objective=str(best_bundle["selected_objective"]),
    )


def _split_portfolio_metric(portfolio_metrics: pd.DataFrame, split: str, strategy: str, column: str) -> float:
    chunk = portfolio_metrics.loc[
        portfolio_metrics["split"].eq(split) & portfolio_metrics["strategy"].eq(strategy),
        column,
    ]
    return float(chunk.iloc[0]) if not chunk.empty else float("nan")


def _split_prediction_metric(metrics_overall: pd.DataFrame, split: str, column: str) -> float:
    chunk = metrics_overall.loc[metrics_overall["split"].eq(split), column]
    return float(chunk.iloc[0]) if not chunk.empty else float("nan")


def _monthly_rank_ic(signal_panel: pd.DataFrame) -> tuple[float, float]:
    spearman_values: list[float] = []
    pearson_values: list[float] = []
    for _, chunk in signal_panel.groupby("month_end", sort=True):
        corr_s = chunk["y_pred"].corr(chunk["y_true"], method="spearman")
        corr_p = chunk["y_pred"].corr(chunk["y_true"], method="pearson")
        if pd.notna(corr_s):
            spearman_values.append(float(corr_s))
        if pd.notna(corr_p):
            pearson_values.append(float(corr_p))
    mean_s = float(np.mean(spearman_values)) if spearman_values else float("nan")
    mean_p = float(np.mean(pearson_values)) if pearson_values else float("nan")
    return mean_s, mean_p


def _prediction_dispersion(signal_panel: pd.DataFrame) -> float:
    monthly = signal_panel.groupby("month_end")["y_pred"].std(ddof=1)
    return float(monthly.mean()) if len(monthly) else float("nan")


def _weight_behavior_rows(*, model_name: str, run: ModelRunArtifacts) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strategy_names = [name for name in run.portfolio_weights["strategy"].drop_duplicates().tolist()]
    for split in ("validation", "test"):
        split_weights = run.portfolio_weights.loc[run.portfolio_weights["split"].eq(split)].copy()
        split_returns = run.portfolio_returns.loc[run.portfolio_returns["split"].eq(split)].copy()
        for strategy in strategy_names:
            w = split_weights.loc[split_weights["strategy"].eq(strategy)].copy()
            if w.empty:
                continue
            pred_source = run.predictions_validation if split == "validation" else run.predictions_test
            signal_panel = _build_signal_panel(pred_source, run.horizons)
            disp = _prediction_dispersion(signal_panel)
            spearman_ic, pearson_ic = _monthly_rank_ic(signal_panel)

            monthly_w = (
                w.groupby(["month_end", "strategy"], as_index=False)
                .agg(
                    max_weight=("weight", "max"),
                    hhi=("weight", lambda x: float(np.square(np.asarray(x, dtype=float)).sum())),
                )
                .sort_values("month_end")
            )
            monthly_w["effective_n_assets"] = 1.0 / monthly_w["hhi"]
            ret_chunk = split_returns.loc[split_returns["strategy"].eq(strategy)]
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "strategy": strategy,
                    "record_type": "overall",
                    "sleeve_id": None,
                    "avg_max_weight": float(monthly_w["max_weight"].mean()),
                    "avg_hhi": float(monthly_w["hhi"].mean()),
                    "avg_effective_n_assets": float(monthly_w["effective_n_assets"].mean()),
                    "avg_turnover": float(ret_chunk["turnover"].mean()),
                    "avg_return": float(ret_chunk["portfolio_annualized_excess_return"].mean()),
                    "volatility": float(ret_chunk["portfolio_annualized_excess_return"].std(ddof=1)),
                    "sharpe": float(_split_portfolio_metric(run.portfolio_metrics, split, strategy, "sharpe")),
                    "max_drawdown": float(_split_portfolio_metric(run.portfolio_metrics, split, strategy, "max_drawdown")),
                    "ending_nav": float(_split_portfolio_metric(run.portfolio_metrics, split, strategy, "ending_nav")),
                    "avg_prediction_dispersion": disp,
                    "avg_rank_ic_spearman": spearman_ic,
                    "avg_rank_ic_pearson": pearson_ic,
                    "month_count": int(ret_chunk["month_end"].nunique()),
                }
            )

            top_weights = w.loc[w.groupby("month_end")["weight"].idxmax()].copy()
            sleeve_summary = (
                w.groupby("sleeve_id", as_index=False)
                .agg(
                    avg_weight=("weight", "mean"),
                    weight_ge_20_freq=("weight", lambda x: float(np.mean(np.asarray(x, dtype=float) >= 0.20))),
                    weight_ge_15_freq=("weight", lambda x: float(np.mean(np.asarray(x, dtype=float) >= 0.15))),
                )
                .merge(
                    top_weights.groupby("sleeve_id", as_index=False).size().rename(columns={"size": "top_count"}),
                    on="sleeve_id",
                    how="left",
                )
                .fillna({"top_count": 0.0})
            )
            month_count = max(1, int(w["month_end"].nunique()))
            sleeve_summary["top_weight_frequency"] = sleeve_summary["top_count"] / month_count
            for row in sleeve_summary.itertuples(index=False):
                rows.append(
                    {
                        "model_name": model_name,
                        "split": split,
                        "strategy": strategy,
                        "record_type": "sleeve",
                        "sleeve_id": row.sleeve_id,
                        "avg_max_weight": float("nan"),
                        "avg_hhi": float("nan"),
                        "avg_effective_n_assets": float("nan"),
                        "avg_turnover": float("nan"),
                        "avg_return": float("nan"),
                        "volatility": float("nan"),
                        "sharpe": float("nan"),
                        "max_drawdown": float("nan"),
                        "ending_nav": float("nan"),
                        "avg_prediction_dispersion": float("nan"),
                        "avg_rank_ic_spearman": float("nan"),
                        "avg_rank_ic_pearson": float("nan"),
                        "month_count": month_count,
                        "avg_weight": float(row.avg_weight),
                        "weight_ge_20_freq": float(row.weight_ge_20_freq),
                        "weight_ge_15_freq": float(row.weight_ge_15_freq),
                        "top_weight_frequency": float(row.top_weight_frequency),
                    }
                )
    return pd.DataFrame(rows)


def _tuning_result_row(
    *,
    stage_name: str,
    run: ModelRunArtifacts,
    selection_row: pd.Series,
    max_epochs: int,
    patience: int,
    selected_flag: int,
) -> dict[str, object]:
    val_signal = _build_signal_panel(run.predictions_validation, run.horizons)
    test_signal = _build_signal_panel(run.predictions_test, run.horizons)
    val_rank_ic_s, _ = _monthly_rank_ic(val_signal)
    test_rank_ic_s, _ = _monthly_rank_ic(test_signal)
    return {
        "stage": stage_name,
        "selected_flag": int(selected_flag),
        "training_objective": run.selected_objective,
        "horizon_mode": "+".join(str(h) for h in run.horizons),
        "lambda_risk": float(selection_row["lambda_risk"]),
        "kappa": float(selection_row["kappa"]),
        "omega_type": str(selection_row["omega_type"]),
        "ewma_beta": float(run.risk_config.ewma_beta),
        "diagonal_shrinkage": float(run.risk_config.diagonal_shrinkage),
        "lookback_months": int(run.risk_config.lookback_months),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "best_epoch": int(selection_row.get("best_epoch", np.nan)) if pd.notna(selection_row.get("best_epoch", np.nan)) else np.nan,
        "validation_score": float(selection_row.get("validation_score", np.nan)),
        "validation_sharpe": float(_split_portfolio_metric(run.portfolio_metrics, "validation", "e2e_portfolio", "sharpe")),
        "validation_avg_return": float(_split_portfolio_metric(run.portfolio_metrics, "validation", "e2e_portfolio", "avg_return")),
        "validation_volatility": float(_split_portfolio_metric(run.portfolio_metrics, "validation", "e2e_portfolio", "volatility")),
        "validation_avg_turnover": float(_split_portfolio_metric(run.portfolio_metrics, "validation", "e2e_portfolio", "avg_turnover")),
        "validation_rmse": float(_split_prediction_metric(run.metrics_overall, "validation", "rmse")),
        "validation_mae": float(_split_prediction_metric(run.metrics_overall, "validation", "mae")),
        "validation_corr": float(_split_prediction_metric(run.metrics_overall, "validation", "corr")),
        "validation_oos_r2": float(_split_prediction_metric(run.metrics_overall, "validation", "oos_r2_vs_naive")),
        "validation_rank_ic_spearman": val_rank_ic_s,
        "validation_avg_prediction_dispersion": _prediction_dispersion(val_signal),
        "test_sharpe": float(_split_portfolio_metric(run.portfolio_metrics, "test", "e2e_portfolio", "sharpe")),
        "test_avg_return": float(_split_portfolio_metric(run.portfolio_metrics, "test", "e2e_portfolio", "avg_return")),
        "test_volatility": float(_split_portfolio_metric(run.portfolio_metrics, "test", "e2e_portfolio", "volatility")),
        "test_avg_turnover": float(_split_portfolio_metric(run.portfolio_metrics, "test", "e2e_portfolio", "avg_turnover")),
        "test_rmse": float(_split_prediction_metric(run.metrics_overall, "test", "rmse")),
        "test_mae": float(_split_prediction_metric(run.metrics_overall, "test", "mae")),
        "test_corr": float(_split_prediction_metric(run.metrics_overall, "test", "corr")),
        "test_oos_r2": float(_split_prediction_metric(run.metrics_overall, "test", "oos_r2_vs_naive")),
        "test_rank_ic_spearman": test_rank_ic_s,
        "test_avg_prediction_dispersion": _prediction_dispersion(test_signal),
    }


def _build_e2e_optimizer_grid() -> list[OptimizerConfig]:
    configs: list[OptimizerConfig] = []
    for lambda_risk in (5.0, 10.0):
        for kappa in (0.05, 0.10, 0.25):
            for omega_type in ("identity", "diag"):
                configs.append(
                    OptimizerConfig(
                        lambda_risk=lambda_risk,
                        kappa=kappa,
                        omega_type=omega_type,
                    )
                )
    return configs


def _build_pto_optimizer_grid() -> list[OptimizerConfig]:
    configs: list[OptimizerConfig] = []
    for lambda_risk in (5.0, 10.0):
        for kappa in (0.1, 1.0):
            for omega_type in ("identity", "diag"):
                configs.append(
                    OptimizerConfig(
                        lambda_risk=lambda_risk,
                        kappa=kappa,
                        omega_type=omega_type,
                    )
                )
    return configs


def _risk_presets() -> dict[str, RiskConfig]:
    return {
        "paper": RiskConfig(ewma_beta=0.94, diagonal_shrinkage=0.10),
        "smoother": RiskConfig(ewma_beta=0.97, diagonal_shrinkage=0.20),
    }


def _stage_two_schedule() -> tuple[int, int]:
    return (25, 8)


def _stage_one_schedule() -> tuple[int, int]:
    return (15, 5)


@dataclass(frozen=True)
class TuningStudyResult:
    """Final outputs for the narrow tuning and diagnostic study."""

    tuning_results: pd.DataFrame
    horizon_ablation: pd.DataFrame
    optimizer_behavior: pd.DataFrame
    report_text: str


def run_e2e_tuning_and_diagnostics(
    *,
    project_root: Path,
    feature_set_name: str = "core_plus_enrichment",
    random_seed: int = 42,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-5,
) -> TuningStudyResult:
    project_root = project_root.resolve()
    optimizer_grid = _build_e2e_optimizer_grid()
    risk_presets = _risk_presets()
    stage1_epochs, stage1_patience = _stage_one_schedule()
    stage2_epochs, stage2_patience = _stage_two_schedule()

    tuning_rows: list[dict[str, object]] = []
    stage1_runs: list[tuple[str, OptimizerConfig, ModelRunArtifacts]] = []
    selected_run: ModelRunArtifacts | None = None

    for preset_name, risk_config in risk_presets.items():
        for optimizer_config in optimizer_grid:
            run = run_e2e_experiment(
                project_root=project_root,
                feature_set_name=feature_set_name,
                horizons=DEFAULT_HORIZONS,
                random_seed=random_seed,
                max_epochs=stage1_epochs,
                patience=stage1_patience,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                risk_config=risk_config,
                optimizer_grid=[optimizer_config],
                training_objective="utility",
            )
            selection_row = run.selection_summary.iloc[0]
            tuning_rows.append(
                {
                    **_tuning_result_row(
                        stage_name=f"stage1_{preset_name}",
                        run=run,
                        selection_row=selection_row,
                        max_epochs=stage1_epochs,
                        patience=stage1_patience,
                        selected_flag=0,
                    ),
                    "risk_preset": preset_name,
                }
            )
            stage1_runs.append((preset_name, optimizer_config, run))

    stage1_df = pd.DataFrame(tuning_rows).sort_values(
        ["validation_sharpe", "validation_score"], ascending=[False, False]
    ).reset_index(drop=True)
    top_stage2 = stage1_df.head(4)

    stage2_runs: list[tuple[str, OptimizerConfig, ModelRunArtifacts]] = []
    for row in top_stage2.itertuples(index=False):
        risk_config = risk_presets[str(row.risk_preset)]
        optimizer_config = OptimizerConfig(
            lambda_risk=float(row.lambda_risk),
            kappa=float(row.kappa),
            omega_type=str(row.omega_type),
        )
        run = run_e2e_experiment(
            project_root=project_root,
            feature_set_name=feature_set_name,
            horizons=DEFAULT_HORIZONS,
            random_seed=random_seed + 200,
            max_epochs=stage2_epochs,
            patience=stage2_patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            risk_config=risk_config,
            optimizer_grid=[optimizer_config],
            training_objective="utility",
        )
        selection_row = run.selection_summary.iloc[0]
        tuning_rows.append(
            {
                **_tuning_result_row(
                    stage_name=f"stage2_{row.risk_preset}",
                    run=run,
                    selection_row=selection_row,
                    max_epochs=stage2_epochs,
                    patience=stage2_patience,
                    selected_flag=0,
                ),
                "risk_preset": str(row.risk_preset),
            }
        )
        stage2_runs.append((str(row.risk_preset), optimizer_config, run))

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["validation_sharpe", "validation_score"], ascending=[False, False]
    ).reset_index(drop=True)
    selected_idx = int(tuning_df.index[0])
    tuning_df.loc[selected_idx, "selected_flag"] = 1
    selected_key = (
        str(tuning_df.loc[selected_idx, "risk_preset"]),
        float(tuning_df.loc[selected_idx, "lambda_risk"]),
        float(tuning_df.loc[selected_idx, "kappa"]),
        str(tuning_df.loc[selected_idx, "omega_type"]),
        int(tuning_df.loc[selected_idx, "max_epochs"]),
        int(tuning_df.loc[selected_idx, "patience"]),
    )

    all_candidate_runs: list[tuple[str, OptimizerConfig, ModelRunArtifacts, tuple[int, int]]] = []
    all_candidate_runs.extend((name, config, run, (stage1_epochs, stage1_patience)) for name, config, run in stage1_runs)
    all_candidate_runs.extend((name, config, run, (stage2_epochs, stage2_patience)) for name, config, run in stage2_runs)
    for preset_name, optimizer_config, run, schedule in all_candidate_runs:
        key = (
            preset_name,
            float(optimizer_config.lambda_risk),
            float(optimizer_config.kappa),
            str(optimizer_config.omega_type),
            int(schedule[0]),
            int(schedule[1]),
        )
        if key == selected_key:
            selected_run = run
            break
    if selected_run is None:
        raise RuntimeError("Could not resolve the selected E2E tuning run")

    selected_risk_config = selected_run.risk_config
    pto_shared = run_pto_experiment(
        project_root=project_root,
        feature_set_name=feature_set_name,
        horizons=DEFAULT_HORIZONS,
        random_seed=random_seed,
        max_epochs=50,
        patience=10,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        risk_config=selected_risk_config,
        optimizer_grid=_build_pto_optimizer_grid(),
    )
    pto_60 = run_pto_experiment(
        project_root=project_root,
        feature_set_name=feature_set_name,
        horizons=(60,),
        random_seed=random_seed,
        max_epochs=50,
        patience=10,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        risk_config=selected_risk_config,
        optimizer_grid=_build_pto_optimizer_grid(),
    )
    pto_120 = run_pto_experiment(
        project_root=project_root,
        feature_set_name=feature_set_name,
        horizons=(120,),
        random_seed=random_seed,
        max_epochs=50,
        patience=10,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        risk_config=selected_risk_config,
        optimizer_grid=_build_pto_optimizer_grid(),
    )
    e2e_60 = run_e2e_experiment(
        project_root=project_root,
        feature_set_name=feature_set_name,
        horizons=(60,),
        random_seed=random_seed + 500,
        max_epochs=int(tuning_df.loc[selected_idx, "max_epochs"]),
        patience=int(tuning_df.loc[selected_idx, "patience"]),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        risk_config=selected_risk_config,
        optimizer_grid=[selected_run.selected_config],
        training_objective="utility",
    )
    e2e_120 = run_e2e_experiment(
        project_root=project_root,
        feature_set_name=feature_set_name,
        horizons=(120,),
        random_seed=random_seed + 600,
        max_epochs=int(tuning_df.loc[selected_idx, "max_epochs"]),
        patience=int(tuning_df.loc[selected_idx, "patience"]),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        risk_config=selected_risk_config,
        optimizer_grid=[selected_run.selected_config],
        training_objective="utility",
    )

    def build_ablation_row(model_name: str, setup_name: str, run: ModelRunArtifacts) -> dict[str, object]:
        strategy_name = f"{run.model_type}_portfolio"
        return {
            "model_name": model_name,
            "setup": setup_name,
            "horizon_mode": "+".join(str(h) for h in run.horizons),
            "selected_lambda_risk": float(run.selected_config.lambda_risk),
            "selected_kappa": float(run.selected_config.kappa),
            "selected_omega_type": str(run.selected_config.omega_type),
            "ewma_beta": float(run.risk_config.ewma_beta),
            "diagonal_shrinkage": float(run.risk_config.diagonal_shrinkage),
            "validation_rmse": _split_prediction_metric(run.metrics_overall, "validation", "rmse"),
            "validation_corr": _split_prediction_metric(run.metrics_overall, "validation", "corr"),
            "validation_sharpe": _split_portfolio_metric(run.portfolio_metrics, "validation", strategy_name, "sharpe"),
            "validation_avg_return": _split_portfolio_metric(run.portfolio_metrics, "validation", strategy_name, "avg_return"),
            "validation_volatility": _split_portfolio_metric(run.portfolio_metrics, "validation", strategy_name, "volatility"),
            "test_rmse": _split_prediction_metric(run.metrics_overall, "test", "rmse"),
            "test_corr": _split_prediction_metric(run.metrics_overall, "test", "corr"),
            "test_sharpe": _split_portfolio_metric(run.portfolio_metrics, "test", strategy_name, "sharpe"),
            "test_avg_return": _split_portfolio_metric(run.portfolio_metrics, "test", strategy_name, "avg_return"),
            "test_volatility": _split_portfolio_metric(run.portfolio_metrics, "test", strategy_name, "volatility"),
            "test_avg_turnover": _split_portfolio_metric(run.portfolio_metrics, "test", strategy_name, "avg_turnover"),
        }

    horizon_ablation = pd.DataFrame(
        [
            build_ablation_row("pto", "shared_60_120", pto_shared),
            build_ablation_row("pto", "separate_60", pto_60),
            build_ablation_row("pto", "separate_120", pto_120),
            build_ablation_row("e2e", "shared_60_120", selected_run),
            build_ablation_row("e2e", "separate_60", e2e_60),
            build_ablation_row("e2e", "separate_120", e2e_120),
        ]
    ).sort_values(["model_name", "setup"]).reset_index(drop=True)

    optimizer_behavior = pd.concat(
        [
            _weight_behavior_rows(model_name="pto_shared", run=pto_shared),
            _weight_behavior_rows(model_name="e2e_shared_tuned", run=selected_run),
            _weight_behavior_rows(model_name="e2e_60", run=e2e_60),
            _weight_behavior_rows(model_name="e2e_120", run=e2e_120),
        ],
        ignore_index=True,
    )

    pto_test = pto_shared.portfolio_metrics.loc[
        pto_shared.portfolio_metrics["split"].eq("test") & pto_shared.portfolio_metrics["strategy"].eq("pto_portfolio")
    ].iloc[0]
    e2e_test = selected_run.portfolio_metrics.loc[
        selected_run.portfolio_metrics["split"].eq("test") & selected_run.portfolio_metrics["strategy"].eq("e2e_portfolio")
    ].iloc[0]
    eq_test = selected_run.portfolio_metrics.loc[
        selected_run.portfolio_metrics["split"].eq("test") & selected_run.portfolio_metrics["strategy"].eq("equal_weight")
    ].iloc[0]

    pto_signal_test = _build_signal_panel(pto_shared.predictions_test, pto_shared.horizons)
    e2e_signal_test = _build_signal_panel(selected_run.predictions_test, selected_run.horizons)
    pto_rank_ic_test, _ = _monthly_rank_ic(pto_signal_test)
    e2e_rank_ic_test, _ = _monthly_rank_ic(e2e_signal_test)
    pto_disp_test = _prediction_dispersion(pto_signal_test)
    e2e_disp_test = _prediction_dispersion(e2e_signal_test)

    shared_rows = horizon_ablation.loc[horizon_ablation["setup"].eq("shared_60_120")].sort_values("model_name")
    best_shared_setup = shared_rows.loc[shared_rows["test_sharpe"].idxmax()]
    best_separate_setup = horizon_ablation.loc[
        horizon_ablation["model_name"].eq("e2e") & horizon_ablation["setup"].ne("shared_60_120")
    ].sort_values("test_sharpe", ascending=False).iloc[0]

    lines: list[str] = []
    lines.append("# XOPTPOE v3 E2E Tuning Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Narrow second-pass study only. Data and feature pipelines stayed frozen.")
    lines.append("- E2E search kept the paper-style utility objective and tuned only around lambda, kappa, Omega, and two nearby EWMA/shrinkage risk presets.")
    lines.append("- PTO was re-run only as a matched benchmark for diagnostics, using its original compact allocation grid rather than a new PTO tuning sweep.")
    lines.append(f"- Stage 1 searched {len(stage1_runs)} shared-horizon candidates at {stage1_epochs} epochs / patience {stage1_patience}; stage 2 re-ran the top {len(stage2_runs)} at {stage2_epochs} epochs / patience {stage2_patience}.")
    lines.append("")
    lines.append("## Selected Tuned E2E Configuration")
    lines.append(
        f"- risk_preset={tuning_df.loc[selected_idx, 'risk_preset']}, lambda={tuning_df.loc[selected_idx, 'lambda_risk']}, kappa={tuning_df.loc[selected_idx, 'kappa']}, omega={tuning_df.loc[selected_idx, 'omega_type']}, ewma_beta={tuning_df.loc[selected_idx, 'ewma_beta']}, shrinkage={tuning_df.loc[selected_idx, 'diagonal_shrinkage']}, epochs={int(tuning_df.loc[selected_idx, 'max_epochs'])}, patience={int(tuning_df.loc[selected_idx, 'patience'])}."
    )
    lines.append(
        f"- validation Sharpe={tuning_df.loc[selected_idx, 'validation_sharpe']:.4f}, test Sharpe={tuning_df.loc[selected_idx, 'test_sharpe']:.4f}, test avg return={tuning_df.loc[selected_idx, 'test_avg_return']:.4f}, test volatility={tuning_df.loc[selected_idx, 'test_volatility']:.4f}."
    )
    lines.append("")
    lines.append("## Shared PTO vs Tuned Shared E2E")
    lines.append(
        f"- PTO test: avg return={float(pto_test['avg_return']):.4f}, volatility={float(pto_test['volatility']):.4f}, Sharpe={float(pto_test['sharpe']):.4f}, turnover={float(pto_test['avg_turnover']):.4f}."
    )
    lines.append(
        f"- Tuned E2E test: avg return={float(e2e_test['avg_return']):.4f}, volatility={float(e2e_test['volatility']):.4f}, Sharpe={float(e2e_test['sharpe']):.4f}, turnover={float(e2e_test['avg_turnover']):.4f}."
    )
    lines.append(
        f"- Equal weight test: avg return={float(eq_test['avg_return']):.4f}, volatility={float(eq_test['volatility']):.4f}, Sharpe={float(eq_test['sharpe']):.4f}."
    )
    lines.append("")
    lines.append("## Signal vs Risk-Control Diagnostics")
    lines.append(
        f"- PTO aggregated test rank IC={pto_rank_ic_test:.4f}, prediction dispersion={pto_disp_test:.4f}."
    )
    lines.append(
        f"- E2E aggregated test rank IC={e2e_rank_ic_test:.4f}, prediction dispersion={e2e_disp_test:.4f}."
    )
    lines.append("- If Sharpe rises while average return and rank IC do not, the gain is risk-control-driven rather than signal-driven.")
    lines.append("")
    lines.append("## Horizon Ablation")
    lines.append(
        f"- Best shared setup on test Sharpe: {best_shared_setup['model_name']} with Sharpe={best_shared_setup['test_sharpe']:.4f}, avg return={best_shared_setup['test_avg_return']:.4f}."
    )
    lines.append(
        f"- Best separate-horizon E2E setup on test Sharpe: {best_separate_setup['setup']} with Sharpe={best_separate_setup['test_sharpe']:.4f}, avg return={best_separate_setup['test_avg_return']:.4f}."
    )
    lines.append("")
    lines.append("## Takeaways")
    lines.append("- The tuning objective remained utility-focused, in line with the stronger regime reported in the paper.")
    lines.append("- Shared-vs-separate comparisons were run without changing the frozen data or feature design.")
    lines.append("- Optimizer behavior summary reports concentration, turnover, predicted-signal dispersion, and sleeve top-weight frequencies so the Sharpe source can be diagnosed directly.")
    report_text = "\n".join(lines) + "\n"

    return TuningStudyResult(
        tuning_results=tuning_df,
        horizon_ablation=horizon_ablation,
        optimizer_behavior=optimizer_behavior,
        report_text=report_text,
    )
