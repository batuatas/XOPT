"""End-to-end predict-and-optimize workflow for XOPTPOE v2 long-horizon SAA."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from xoptpoe_v2_models.data import (
    DEFAULT_HORIZONS,
    SLEEVE_ORDER,
    TARGET_COL,
    LoadedModelingInputs,
    build_sleeve_horizon_benchmark,
    load_modeling_inputs,
)
from xoptpoe_v2_models.evaluate import evaluate_predictions
from xoptpoe_v2_models.losses import objective_score
from xoptpoe_v2_models.networks import PredictorMLP
from xoptpoe_v2_models.optim_layers import (
    OptimizerConfig,
    RiskConfig,
    RobustOptimizerCache,
    build_sigma_map,
    candidate_optimizer_grid,
)
from xoptpoe_v2_models.portfolio_eval import (
    build_monthly_signal_panel,
    run_portfolio_evaluation,
    summarize_portfolio_metrics,
)
from xoptpoe_v2_models.preprocess import fit_preprocessor


TRAINING_OBJECTIVES: tuple[str, ...] = ("return", "utility", "sharpe")


@dataclass(frozen=True)
class E2EResult:
    """Final E2E artifacts."""

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
    selected_objective: str
    report_text: str
    feature_set_name: str
    preprocessor_feature_names: list[str]



def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



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



def _reshape_monthly(frame: pd.DataFrame, values: np.ndarray | torch.Tensor) -> tuple[list[pd.Timestamp], np.ndarray | torch.Tensor]:
    month_count = int(frame["month_end"].nunique())
    expected_rows = month_count * len(DEFAULT_HORIZONS) * len(SLEEVE_ORDER)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Split frame does not match expected stacked shape for shared-horizon aggregation: rows={len(frame)}, expected={expected_rows}"
        )
    months = [pd.Timestamp(v) for v in frame["month_end"].drop_duplicates().tolist()]
    if isinstance(values, torch.Tensor):
        reshaped = values.reshape(month_count, len(DEFAULT_HORIZONS), len(SLEEVE_ORDER))
        return months, reshaped.mean(dim=1)
    arr = np.asarray(values)
    reshaped = arr.reshape(month_count, len(DEFAULT_HORIZONS), len(SLEEVE_ORDER))
    return months, reshaped.mean(axis=1)



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



def _render_e2e_report(
    *,
    feature_set_name: str,
    input_feature_count: int,
    searched_objectives: tuple[str, ...],
    selection_summary: pd.DataFrame,
    selected_objective: str,
    selected_config: OptimizerConfig,
    metrics_overall: pd.DataFrame,
    metrics_by_horizon: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
) -> str:
    top_candidates = selection_summary.sort_values(
        ["validation_portfolio_sharpe", "validation_score"], ascending=[False, False]
    ).head(5)
    lines: list[str] = []
    lines.append("# XOPTPOE v2 E2E / PAO Report")
    lines.append("")
    lines.append("## Implemented PAO Logic")
    lines.append("- Shared 60m/120m predictor uses the same paper-style MLP as PTO: `32 -> 16 -> 8`, ReLU, batch normalization, dropout `0.5`.")
    lines.append("- Training is decision-focused: row-level predictions are aggregated into one monthly sleeve signal and passed through a differentiable robust optimizer built with CVXPY + cvxpylayers.")
    objective_text = ", ".join(f"`{name}`" for name in searched_objectives)
    lines.append(f"- Candidate training objectives searched in this run: {objective_text}.")
    lines.append("- Feature set used: `{}` with {} transformed inputs after missing indicators.".format(feature_set_name, input_feature_count))
    lines.append("")
    lines.append("## SAA Adaptations")
    lines.append("- The paper’s stock-ranking universe step is removed because the SAA universe is the fixed 8-sleeve investable universe.")
    lines.append("- A single monthly SAA decision is formed by averaging 60m and 120m horizon-conditioned predictions sleeve-by-sleeve before optimization.")
    lines.append("- The optimizer still follows the paper’s robust long-only mean-variance form, but Sigma is estimated from sleeve-level trailing monthly excess returns rather than stock-level firm panels.")
    lines.append("- Metrics are reported on long-horizon annualized excess-return labels, so portfolio Sharpe is computed from decision-period annualized outcomes without an extra monthly-to-annual scaling factor.")
    lines.append("")
    lines.append("## Selected Candidate")
    lines.append(f"- Selected training objective: `{selected_objective}`.")
    lines.append(f"- Selected validation config: lambda={selected_config.lambda_risk}, kappa={selected_config.kappa}, omega={selected_config.omega_type}.")
    lines.append("")
    lines.append("## Top Validation Candidates")
    for row in top_candidates.itertuples(index=False):
        lines.append(
            f"- objective={row.training_objective}, lambda={row.lambda_risk}, kappa={row.kappa}, omega={row.omega_type}, validation_portfolio_sharpe={row.validation_portfolio_sharpe:.4f}, validation_score={row.validation_score:.4f}, best_epoch={int(row.best_epoch)}."
        )
    lines.append("")
    lines.append("## Prediction Metrics")
    for row in metrics_overall.itertuples(index=False):
        lines.append(
            f"- {row.split}: rmse={row.rmse:.6f}, mae={row.mae:.6f}, oos_r2_vs_naive={row.oos_r2_vs_naive:.4f}, corr={row.corr:.4f}, directional_accuracy={row.directional_accuracy:.4f}."
        )
    lines.append("- By horizon:")
    for row in metrics_by_horizon.itertuples(index=False):
        lines.append(
            f"  - {row.split}, {int(row.horizon_months)}m: rmse={row.rmse:.6f}, corr={row.corr:.4f}, oos_r2_vs_naive={row.oos_r2_vs_naive:.4f}."
        )
    lines.append("")
    lines.append("## Portfolio Metrics")
    for row in portfolio_metrics.itertuples(index=False):
        lines.append(
            f"- {row.split}, {row.strategy}: avg_return={row.avg_return:.4f}, volatility={row.volatility:.4f}, sharpe={row.sharpe:.4f}, max_drawdown={row.max_drawdown:.4f}, avg_turnover={row.avg_turnover:.4f}."
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Model selection used validation portfolio Sharpe as the common selector across PAO candidates after early stopping on each candidate’s own decision objective.")
    lines.append("- Portfolio metrics are decision-period diagnostics on overlapping long-horizon annualized outcomes, not a fully non-overlapping wealth backtest.")
    return "\n".join(lines) + "\n"



def run_v2_e2e(
    *,
    project_root: Path,
    feature_set_name: str,
    training_objectives: tuple[str, ...] = TRAINING_OBJECTIVES,
    random_seed: int = 42,
    max_epochs: int = 50,
    patience: int = 10,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-5,
    risk_config: RiskConfig | None = None,
) -> E2EResult:
    """Train the shared-horizon E2E model and evaluate its portfolio layer."""
    risk_config = risk_config or RiskConfig()
    inputs: LoadedModelingInputs = load_modeling_inputs(project_root, feature_set_name=feature_set_name)

    preprocessor = fit_preprocessor(
        inputs.train_df,
        feature_manifest=inputs.feature_manifest,
        feature_columns=inputs.feature_columns,
    )
    x_train, _ = preprocessor.transform(inputs.train_df)
    x_val, _ = preprocessor.transform(inputs.validation_df)
    x_test, _ = preprocessor.transform(inputs.test_df)

    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    val_x_tensor = torch.tensor(x_val, dtype=torch.float32)
    test_x_tensor = torch.tensor(x_test, dtype=torch.float32)

    y_train_rows = inputs.train_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_val_rows = inputs.validation_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_test_rows = inputs.test_df[TARGET_COL].to_numpy(dtype=np.float32)

    train_months, y_train_monthly = _reshape_monthly(inputs.train_df, y_train_rows)
    val_months, y_val_monthly = _reshape_monthly(inputs.validation_df, y_val_rows)
    test_months, y_test_monthly = _reshape_monthly(inputs.test_df, y_test_rows)
    y_train_monthly_tensor = torch.tensor(y_train_monthly, dtype=torch.float32)
    y_val_monthly_tensor = torch.tensor(y_val_monthly, dtype=torch.float32)

    sigma_map = build_sigma_map(
        sorted(set(train_months + val_months + test_months)),
        excess_history=inputs.monthly_excess_history,
        risk_config=risk_config,
    )
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)
    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)

    selection_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    best_bundle: dict[str, object] | None = None
    candidate_id = 0

    for objective_name in training_objectives:
        for optimizer_config in candidate_optimizer_grid():
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
                _, pred_train_monthly = _reshape_monthly(inputs.train_df, pred_train_rows)
                train_score = _decision_score_over_months(
                    months=train_months,
                    predicted_monthly=pred_train_monthly,
                    realized_monthly=y_train_monthly_tensor,
                    sigma_map=sigma_map,
                    optimizer_cache=optimizer_cache,
                    optimizer_config=optimizer_config,
                    objective_name=objective_name,
                )
                train_loss = -train_score
                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred_val_rows = model(val_x_tensor)
                    _, pred_val_monthly = _reshape_monthly(inputs.validation_df, pred_val_rows)
                    val_score = _decision_score_over_months(
                        months=val_months,
                        predicted_monthly=pred_val_monthly,
                        realized_monthly=y_val_monthly_tensor,
                        sigma_map=sigma_map,
                        optimizer_cache=optimizer_cache,
                        optimizer_config=optimizer_config,
                        objective_name=objective_name,
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
                        "training_objective": objective_name,
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

            val_signal_panel = build_monthly_signal_panel(pred_val_df)
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
                "training_objective": objective_name,
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
                test_signal_panel = build_monthly_signal_panel(pred_test_df)
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
                pred_val_best = pred_val_df.merge(
                    weight_lookup,
                    on=["split", "month_end", "sleeve_id"],
                    how="left",
                    validate="m:1",
                )
                pred_test_best = pred_test_df.merge(
                    weight_lookup,
                    on=["split", "month_end", "sleeve_id"],
                    how="left",
                    validate="m:1",
                )
                pred_val_best["selected_lambda_risk"] = optimizer_config.lambda_risk
                pred_val_best["selected_kappa"] = optimizer_config.kappa
                pred_val_best["selected_omega_type"] = optimizer_config.omega_type
                pred_val_best["selected_training_objective"] = objective_name
                pred_test_best["selected_lambda_risk"] = optimizer_config.lambda_risk
                pred_test_best["selected_kappa"] = optimizer_config.kappa
                pred_test_best["selected_omega_type"] = optimizer_config.omega_type
                pred_test_best["selected_training_objective"] = objective_name
                best_bundle = {
                    "predictions_validation": pred_val_best,
                    "predictions_test": pred_test_best,
                    "portfolio_returns": portfolio_returns,
                    "portfolio_weights": portfolio_weights,
                    "portfolio_metrics": portfolio_metrics,
                    "selected_config": optimizer_config,
                    "selected_objective": objective_name,
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
    report_text = _render_e2e_report(
        feature_set_name=feature_set_name,
        input_feature_count=len(preprocessor.feature_names),
        searched_objectives=training_objectives,
        selection_summary=selection_summary,
        selected_objective=str(best_bundle["selected_objective"]),
        selected_config=best_bundle["selected_config"],
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        portfolio_metrics=best_bundle["portfolio_metrics"],
    )

    return E2EResult(
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
        selected_objective=str(best_bundle["selected_objective"]),
        report_text=report_text,
        feature_set_name=feature_set_name,
        preprocessor_feature_names=preprocessor.feature_names,
    )
