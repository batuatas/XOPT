"""Predict-then-optimize workflow for XOPTPOE v3 long-horizon SAA."""

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
    TARGET_COL,
    LoadedModelingInputs,
    build_sleeve_horizon_benchmark,
    default_paths,
    load_modeling_inputs,
)
from xoptpoe_v3_models.evaluate import evaluate_predictions
from xoptpoe_v3_models.losses import mse_loss
from xoptpoe_v3_models.networks import PredictorMLP
from xoptpoe_v3_models.optim_layers import (
    OptimizerConfig,
    RiskConfig,
    RobustOptimizerCache,
    build_sigma_map,
    candidate_optimizer_grid,
)
from xoptpoe_v3_models.portfolio_eval import (
    build_monthly_signal_panel,
    run_portfolio_evaluation,
    summarize_portfolio_metrics,
)
from xoptpoe_v3_models.preprocess import fit_preprocessor


@dataclass(frozen=True)
class PTOResult:
    """Final PTO artifacts."""

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



def _render_pto_report(
    *,
    feature_set_name: str,
    input_feature_count: int,
    training_history: pd.DataFrame,
    selection_summary: pd.DataFrame,
    selected_config: OptimizerConfig,
    metrics_overall: pd.DataFrame,
    metrics_by_horizon: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
) -> str:
    best_epoch = int(training_history.loc[training_history["is_best_epoch"].eq(1), "epoch"].iloc[-1])
    lines: list[str] = []
    lines.append("# XOPTPOE v3 PTO Report")
    lines.append("")
    lines.append("## Implemented PTO Logic")
    lines.append("- Shared 60m/120m predictor trained on row-level annualized excess targets only; no optimization layer in training.")
    lines.append("- Paper-style MLP architecture: hidden layers `32 -> 16 -> 8`, ReLU activations, batch normalization, dropout `0.5`.")
    lines.append("- Training hyperparameters mirrored from the paper where applicable: AdamW, learning rate `5e-5`, weight decay `1e-5`, max epochs `50`, patience `10`.")
    lines.append("- Feature set used: `{}` with {} transformed inputs after missing indicators.".format(feature_set_name, input_feature_count))
    lines.append("")
    lines.append("## SAA Adaptations")
    lines.append("- The paper’s firm-level 1m stock ranking step is removed; the investable universe is the active 9-sleeve v3 XOPTPOE universe with EQ_CN included.")
    lines.append("- The shared model still predicts horizon-conditioned 60m and 120m annualized excess returns row by row.")
    lines.append("- For downstream SAA portfolio construction, the two horizon predictions are averaged sleeve-by-sleeve into one monthly signal before optimization.")
    lines.append("- Risk estimation uses trailing monthly sleeve excess returns from the active v3-compatible 9-sleeve history, combining the frozen baseline sleeves with the versioned China sleeve returns, and keeps the expanding-to-60m EWMA window to avoid collapsing the long-horizon split.")
    lines.append("")
    lines.append("## Optimizer Layer")
    lines.append(f"- Selected validation config: lambda={selected_config.lambda_risk}, kappa={selected_config.kappa}, omega={selected_config.omega_type}.")
    lines.append("- Portfolio problem solved: maximize `w' mu_hat - kappa * sqrt(w' Omega w) - (lambda/2) * w' Sigma w`, subject to `sum(w)=1`, `w>=0`.")
    lines.append("- Sigma: annualized EWMA covariance of trailing monthly sleeve excess returns, beta `0.94`, diagonal shrinkage `0.10`, ridge `1e-6`, window cap `60` months.")
    lines.append("- Omega candidates searched: `diag(Sigma)` and `I`.")
    lines.append("")
    lines.append("## Predictor Training")
    lines.append(f"- Best validation-MSE epoch: `{best_epoch}`.")
    lines.append(f"- Final best validation MSE: `{training_history['val_loss'].min():.6f}`.")
    lines.append("")
    lines.append("## Allocation Selection Summary")
    top_candidates = selection_summary.sort_values("validation_sharpe", ascending=False).head(5)
    for row in top_candidates.itertuples(index=False):
        lines.append(
            f"- lambda={row.lambda_risk}, kappa={row.kappa}, omega={row.omega_type}: validation_sharpe={row.validation_sharpe:.4f}, validation_avg_return={row.validation_avg_return:.4f}."
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
    lines.append("- Portfolio metrics are decision-period diagnostics on overlapping long-horizon annualized outcomes, not a fully non-overlapping wealth backtest.")
    return "\n".join(lines) + "\n"



def run_v3_pto(
    *,
    project_root: Path,
    feature_set_name: str,
    random_seed: int = 42,
    max_epochs: int = 50,
    patience: int = 10,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-5,
    risk_config: RiskConfig | None = None,
) -> PTOResult:
    """Train the shared-horizon PTO model and evaluate its portfolio layer."""
    risk_config = risk_config or RiskConfig()
    _set_seeds(random_seed)
    inputs: LoadedModelingInputs = load_modeling_inputs(project_root, feature_set_name=feature_set_name)

    preprocessor = fit_preprocessor(
        inputs.train_df,
        feature_manifest=inputs.feature_manifest,
        feature_columns=inputs.feature_columns,
    )
    x_train, _ = preprocessor.transform(inputs.train_df)
    x_val, _ = preprocessor.transform(inputs.validation_df)
    x_test, _ = preprocessor.transform(inputs.test_df)
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
        train_pred = model(train_x_tensor).cpu().numpy()
        val_pred = model(val_x_tensor).cpu().numpy()
        test_pred = model(torch.tensor(x_test, dtype=torch.float32)).cpu().numpy()

    benchmark = build_sleeve_horizon_benchmark(inputs.train_df)
    pred_train_df = _prediction_frame(inputs.train_df, train_pred, benchmark, split_name="train")
    pred_val_df = _prediction_frame(inputs.validation_df, val_pred, benchmark, split_name="validation")
    pred_test_df = _prediction_frame(inputs.test_df, test_pred, benchmark, split_name="test")

    all_months = sorted(pd.concat([pred_val_df["month_end"], pred_test_df["month_end"]]).drop_duplicates().tolist())
    sigma_map = build_sigma_map(all_months, excess_history=inputs.monthly_excess_history, risk_config=risk_config)
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    selection_rows: list[dict[str, object]] = []
    val_signal_panel = build_monthly_signal_panel(pred_val_df)
    for config in candidate_optimizer_grid():
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
        split_signal_panel = build_monthly_signal_panel(pred_df)
        split_run = run_portfolio_evaluation(
            signal_panel=split_signal_panel,
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
    report_text = _render_pto_report(
        feature_set_name=feature_set_name,
        input_feature_count=len(preprocessor.feature_names),
        training_history=training_history,
        selection_summary=selection_summary,
        selected_config=selected_config,
        metrics_overall=metrics_overall,
        metrics_by_horizon=metrics_by_horizon,
        portfolio_metrics=portfolio_metrics,
    )

    return PTOResult(
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
        report_text=report_text,
        feature_set_name=feature_set_name,
        preprocessor_feature_names=preprocessor.feature_names,
    )


run_v2_pto = run_v3_pto
