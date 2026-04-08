"""Controlled v3 portfolio benchmark and allocation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from xoptpoe_v3_modeling.io import load_csv, load_parquet, write_csv, write_parquet, write_text
from xoptpoe_v3_models.data import SLEEVE_ORDER, default_paths, load_modeling_inputs
from xoptpoe_v3_models.horse_race import (
    _combine_separate_signal_panels,
    _portfolio_run_from_signal,
    _single_horizon_signal_panel,
    _truth_panel_from_split_frame,
)
from xoptpoe_v3_models.optim_layers import (
    OptimizerConfig,
    RiskConfig,
    RobustOptimizerCache,
    build_sigma_map,
    candidate_optimizer_grid,
)
from xoptpoe_v3_models.portfolio_eval import build_monthly_signal_panel, summarize_portfolio_metrics
from xoptpoe_v3_models.prediction_ablation import (
    HORIZON_MODES,
    _run_elastic_net_experiment,
    _run_ridge_experiment,
    _subset_inputs,
)


BEST_60_EXPERIMENT = "elastic_net__full_firstpass__separate_60"
BEST_120_EXPERIMENT = "ridge__full_firstpass__separate_120"
BEST_SHARED_EXPERIMENT = "ridge__full_firstpass__shared_60_120"
PTO_EXPERIMENT = "pto_nn__core_plus_enrichment__shared_60_120"
E2E_EXPERIMENT = "e2e_nn__core_plus_enrichment__shared_60_120"


@dataclass(frozen=True)
class PortfolioBenchmarkOutputs:
    returns_panel: pd.DataFrame
    metrics: pd.DataFrame
    by_sleeve: pd.DataFrame
    attribution_panel: pd.DataFrame
    china_diagnostics: pd.DataFrame
    comparison: pd.DataFrame
    report_text: str


@dataclass(frozen=True)
class StrategyMetadata:
    strategy_label: str
    strategy_group: str
    signal_source: str
    allocation_rule: str


def _base_signal_meta() -> dict[str, StrategyMetadata]:
    return {
        "equal_weight": StrategyMetadata("equal_weight", "benchmark", "equal_weight", "equal_weight"),
        "best_60_predictor": StrategyMetadata("best_60_predictor", "supervised_common_allocator", "best_60_predictor", "robust_allocator"),
        "best_120_predictor": StrategyMetadata("best_120_predictor", "supervised_common_allocator", "best_120_predictor", "robust_allocator"),
        "combined_60_120_predictor": StrategyMetadata("combined_60_120_predictor", "supervised_common_allocator", "combined_60_120_predictor", "robust_allocator"),
        "best_shared_predictor": StrategyMetadata("best_shared_predictor", "supervised_common_allocator", "best_shared_predictor", "robust_allocator"),
        "pto_nn_signal": StrategyMetadata("pto_nn_signal", "neural_common_allocator", "pto_nn_signal", "robust_allocator"),
        "e2e_nn_signal": StrategyMetadata("e2e_nn_signal", "neural_common_allocator", "e2e_nn_signal", "robust_allocator"),
        "combined_top_k_equal": StrategyMetadata("combined_top_k_equal", "supervised_concentration_control", "combined_60_120_predictor", "top_k_equal"),
        "combined_top_k_capped": StrategyMetadata("combined_top_k_capped", "supervised_concentration_control", "combined_60_120_predictor", "top_k_capped"),
        "combined_score_positive_capped": StrategyMetadata("combined_score_positive_capped", "supervised_concentration_control", "combined_60_120_predictor", "score_positive_capped"),
        "combined_diversified_cap": StrategyMetadata("combined_diversified_cap", "supervised_concentration_control", "combined_60_120_predictor", "diversified_cap"),
    }


def _strip_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    needed = ["month_end", "sleeve_id", "horizon_months", "split", "y_true", "y_pred", "benchmark_pred"]
    missing = [col for col in needed if col not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame missing required columns: {missing}")
    out = frame[needed].copy()
    out["month_end"] = pd.to_datetime(out["month_end"])
    return out.sort_values(["split", "month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)


def _load_horse_race_predictions(project_root: Path, experiment_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = default_paths(project_root)
    pred_val = load_parquet(paths.data_out_dir / "predictions_validation_horse_race.parquet")
    pred_test = load_parquet(paths.data_out_dir / "predictions_test_horse_race.parquet")
    val = pred_val.loc[pred_val["experiment_name"].eq(experiment_name)].copy()
    test = pred_test.loc[pred_test["experiment_name"].eq(experiment_name)].copy()
    if val.empty or test.empty:
        raise KeyError(experiment_name)
    return _strip_prediction_columns(val), _strip_prediction_columns(test)


def _load_neural_predictions(project_root: Path, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = default_paths(project_root)
    val = load_csv(paths.data_out_dir / f"predictions_validation_{model_name}.csv", parse_dates=["month_end"])
    test = load_csv(paths.data_out_dir / f"predictions_test_{model_name}.csv", parse_dates=["month_end"])
    return _strip_prediction_columns(val), _strip_prediction_columns(test)


def _rerun_linear_experiment(project_root: Path, experiment_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name, feature_set_name, horizon_mode = experiment_name.split("__")
    inputs = load_modeling_inputs(project_root, feature_set_name=feature_set_name)
    subset_inputs = _subset_inputs(inputs, HORIZON_MODES[horizon_mode])
    if model_name == "ridge":
        run = _run_ridge_experiment(
            inputs=subset_inputs,
            feature_set_name=feature_set_name,
            horizon_mode=horizon_mode,
            horizons=HORIZON_MODES[horizon_mode],
        )
    elif model_name == "elastic_net":
        run = _run_elastic_net_experiment(
            inputs=subset_inputs,
            feature_set_name=feature_set_name,
            horizon_mode=horizon_mode,
            horizons=HORIZON_MODES[horizon_mode],
        )
    else:
        raise ValueError(f"Unsupported rerun experiment: {experiment_name}")
    return _strip_prediction_columns(run.predictions_validation), _strip_prediction_columns(run.predictions_test)


def _load_or_rerun_predictions(project_root: Path, experiment_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if experiment_name == PTO_EXPERIMENT:
        return _load_neural_predictions(project_root, "pto")
    if experiment_name == E2E_EXPERIMENT:
        return _load_neural_predictions(project_root, "e2e")
    try:
        return _load_horse_race_predictions(project_root, experiment_name)
    except KeyError:
        return _rerun_linear_experiment(project_root, experiment_name)


def _signal_maps(project_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    inputs = load_modeling_inputs(project_root, feature_set_name="core_baseline")
    truth_validation = _truth_panel_from_split_frame(inputs.validation_df, split_name="validation")
    truth_test = _truth_panel_from_split_frame(inputs.test_df, split_name="test")

    best60_val, best60_test = _load_or_rerun_predictions(project_root, BEST_60_EXPERIMENT)
    best120_val, best120_test = _load_or_rerun_predictions(project_root, BEST_120_EXPERIMENT)
    shared_val, shared_test = _load_or_rerun_predictions(project_root, BEST_SHARED_EXPERIMENT)
    pto_val, pto_test = _load_or_rerun_predictions(project_root, PTO_EXPERIMENT)
    e2e_val, e2e_test = _load_or_rerun_predictions(project_root, E2E_EXPERIMENT)

    validation_signals = {
        "best_60_predictor": _single_horizon_signal_panel(best60_val, horizon_months=60, truth_panel=truth_validation),
        "best_120_predictor": _single_horizon_signal_panel(best120_val, horizon_months=120, truth_panel=truth_validation),
        "combined_60_120_predictor": _combine_separate_signal_panels(best60_val, best120_val, truth_panel=truth_validation),
        "best_shared_predictor": build_monthly_signal_panel(shared_val),
        "pto_nn_signal": build_monthly_signal_panel(pto_val),
        "e2e_nn_signal": build_monthly_signal_panel(e2e_val),
    }
    test_signals = {
        "best_60_predictor": _single_horizon_signal_panel(best60_test, horizon_months=60, truth_panel=truth_test),
        "best_120_predictor": _single_horizon_signal_panel(best120_test, horizon_months=120, truth_panel=truth_test),
        "combined_60_120_predictor": _combine_separate_signal_panels(best60_test, best120_test, truth_panel=truth_test),
        "best_shared_predictor": build_monthly_signal_panel(shared_test),
        "pto_nn_signal": build_monthly_signal_panel(pto_test),
        "e2e_nn_signal": build_monthly_signal_panel(e2e_test),
    }
    for mapping in (validation_signals, test_signals):
        for frame in mapping.values():
            frame["month_end"] = pd.to_datetime(frame["month_end"])
    return validation_signals, test_signals


def _normalize_long_only(values: np.ndarray) -> np.ndarray:
    vec = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(vec.sum())
    if total <= 0:
        return np.repeat(1.0 / len(vec), len(vec))
    return vec / total


def _project_with_cap(values: np.ndarray, *, max_weight: float, max_iter: int = 100) -> np.ndarray:
    n = len(values)
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be in (0,1]")
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"cap {max_weight} infeasible for {n} sleeves")
    w = _normalize_long_only(values)
    for _ in range(max_iter):
        over = w > max_weight + 1e-12
        if not np.any(over):
            break
        excess = float(np.sum(w[over] - max_weight))
        w[over] = max_weight
        under = w < max_weight - 1e-12
        if not np.any(under):
            w = np.repeat(1.0 / n, n)
            break
        under_sum = float(np.sum(w[under]))
        if under_sum <= 0:
            w[under] = 1.0 / float(np.sum(under))
        else:
            w[under] = w[under] + excess * (w[under] / under_sum)
    w = np.clip(w, 0.0, max_weight)
    return _normalize_long_only(w)


def _weights_equal(scores: pd.Series) -> pd.Series:
    n = len(scores)
    return pd.Series(np.repeat(1.0 / n, n), index=scores.index, dtype=float)


def _weights_top_k_equal(scores: pd.Series, *, k: int) -> pd.Series:
    n = len(scores)
    k_eff = min(max(k, 1), n)
    top_idx = scores.sort_values(ascending=False).index[:k_eff]
    out = pd.Series(0.0, index=scores.index, dtype=float)
    out.loc[top_idx] = 1.0 / k_eff
    return out


def _weights_top_k_capped(scores: pd.Series, *, k: int, max_weight: float) -> pd.Series:
    n = len(scores)
    k_eff = min(max(k, 1), n)
    min_required = int(np.ceil(1.0 / max_weight))
    select_n = min(max(k_eff, min_required), n)
    top_idx = scores.sort_values(ascending=False).index[:select_n]
    raw = np.clip(scores.loc[top_idx].to_numpy(dtype=float), 0.0, None)
    if float(raw.sum()) <= 0:
        raw = np.ones_like(raw, dtype=float)
    capped = _project_with_cap(_normalize_long_only(raw), max_weight=max_weight)
    out = pd.Series(0.0, index=scores.index, dtype=float)
    out.loc[top_idx] = capped
    return out


def _weights_score_positive_capped(scores: pd.Series, *, max_weight: float) -> pd.Series:
    raw = np.clip(scores.to_numpy(dtype=float), 0.0, None)
    if float(raw.sum()) <= 0:
        raw = np.repeat(1.0 / len(raw), len(raw))
    capped = _project_with_cap(_normalize_long_only(raw), max_weight=max_weight)
    return pd.Series(capped, index=scores.index, dtype=float)


def _weights_diversified_cap(scores: pd.Series, *, top_n: int, max_weight: float) -> pd.Series:
    n = len(scores)
    n_eff = min(max(top_n, 1), n)
    top_idx = scores.sort_values(ascending=False).index[:n_eff]
    raw = np.clip(scores.loc[top_idx].to_numpy(dtype=float), 0.0, None)
    if float(raw.sum()) <= 0:
        raw = np.linspace(n_eff, 1, num=n_eff, dtype=float)
    capped = _project_with_cap(_normalize_long_only(raw), max_weight=max_weight)
    out = pd.Series(0.0, index=scores.index, dtype=float)
    out.loc[top_idx] = capped
    return out


def _run_direct_weight_strategy(
    *,
    validation_signal_panel: pd.DataFrame,
    test_signal_panel: pd.DataFrame,
    strategy_label: str,
    weight_builder: Callable[[pd.Series], pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for split_name, signal_panel in (("validation", validation_signal_panel), ("test", test_signal_panel)):
        prev_weights: np.ndarray | None = None
        for month_end, chunk in signal_panel.groupby("month_end", sort=True):
            ordered = chunk.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
            scores = pd.Series(ordered["y_pred"].to_numpy(dtype=float), index=ordered["sleeve_id"].tolist(), dtype=float)
            weights = weight_builder(scores).reindex(list(SLEEVE_ORDER)).to_numpy(dtype=float)
            weights = _normalize_long_only(weights)
            realized = ordered["y_true"].to_numpy(dtype=float)
            port_return = float(np.dot(weights, realized))
            turnover = 0.0 if prev_weights is None else float(0.5 * np.abs(weights - prev_weights).sum())
            prev_weights = weights.copy()
            returns_rows.append(
                {
                    "split": split_name,
                    "month_end": pd.Timestamp(month_end),
                    "strategy_label": strategy_label,
                    "portfolio_annualized_excess_return": port_return,
                    "turnover": turnover,
                }
            )
            for sleeve_id, weight in zip(SLEEVE_ORDER, weights, strict=True):
                weight_rows.append(
                    {
                        "split": split_name,
                        "month_end": pd.Timestamp(month_end),
                        "strategy_label": strategy_label,
                        "sleeve_id": sleeve_id,
                        "weight": float(weight),
                    }
                )
    returns_df = pd.DataFrame(returns_rows).sort_values(["split", "month_end"]).reset_index(drop=True)
    returns_df["gross_return"] = np.maximum(1.0 + returns_df["portfolio_annualized_excess_return"], 1e-6)
    returns_df["cum_nav"] = returns_df.groupby(["split", "strategy_label"])["gross_return"].cumprod()
    returns_df["running_peak"] = returns_df.groupby(["split", "strategy_label"])["cum_nav"].cummax()
    returns_df["drawdown"] = returns_df["cum_nav"] / returns_df["running_peak"] - 1.0
    returns_df = returns_df.drop(columns=["running_peak"])
    weights_df = pd.DataFrame(weight_rows).sort_values(["split", "month_end", "sleeve_id"]).reset_index(drop=True)
    return returns_df, weights_df


def _attach_meta(frame: pd.DataFrame, meta: StrategyMetadata) -> pd.DataFrame:
    out = frame.copy()
    out["strategy_group"] = meta.strategy_group
    out["signal_source"] = meta.signal_source
    out["allocation_rule"] = meta.allocation_rule
    return out


def _strategy_metrics_from_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split_name, strategy_label), chunk in returns_df.groupby(["split", "strategy_label"], as_index=False):
        values = chunk["portfolio_annualized_excess_return"].to_numpy(dtype=float)
        vol = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        sharpe = float(np.mean(values) / vol) if np.isfinite(vol) and vol > 0 else float("nan")
        rows.append(
            {
                "split": split_name,
                "strategy_label": strategy_label,
                "month_count": int(chunk["month_end"].nunique()),
                "avg_return": float(np.mean(values)),
                "volatility": vol,
                "sharpe": sharpe,
                "max_drawdown": float(chunk["drawdown"].min()),
                "avg_turnover": float(chunk["turnover"].mean()),
                "ending_nav": float(chunk["cum_nav"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def _build_panels_and_metrics(
    *,
    project_root: Path,
    validation_signals: dict[str, pd.DataFrame],
    test_signals: dict[str, pd.DataFrame],
    risk_config: RiskConfig,
    optimizer_grid: list[OptimizerConfig],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_map = _base_signal_meta()
    inputs = load_modeling_inputs(project_root, feature_set_name="core_baseline")
    months = sorted(
        pd.concat(
            [
                inputs.validation_df["month_end"].drop_duplicates(),
                inputs.test_df["month_end"].drop_duplicates(),
            ],
            ignore_index=True,
        ).tolist()
    )
    sigma_map = build_sigma_map(months, excess_history=inputs.monthly_excess_history, risk_config=risk_config)
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    returns_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    equal_returns: pd.DataFrame | None = None
    equal_weights: pd.DataFrame | None = None

    common_strategies = [
        "best_60_predictor",
        "best_120_predictor",
        "combined_60_120_predictor",
        "best_shared_predictor",
        "pto_nn_signal",
        "e2e_nn_signal",
    ]
    for strategy_label in common_strategies:
        returns_df, weights_df, metrics_df, selected_config = _portfolio_run_from_signal(
            validation_signal_panel=validation_signals[strategy_label],
            test_signal_panel=test_signals[strategy_label],
            optimizer_cache=optimizer_cache,
            optimizer_grid=optimizer_grid,
            strategy_name=strategy_label,
        )
        if equal_returns is None:
            eq_ret = returns_df.loc[returns_df["strategy"].eq("equal_weight")].copy()
            eq_w = weights_df.loc[weights_df["strategy"].eq("equal_weight")].copy()
            eq_ret = eq_ret.rename(columns={"strategy": "strategy_label"})
            eq_w = eq_w.rename(columns={"strategy": "strategy_label"})
            equal_returns = _attach_meta(eq_ret, meta_map["equal_weight"])
            equal_weights = _attach_meta(eq_w, meta_map["equal_weight"])
        strat_ret = returns_df.loc[returns_df["strategy"].eq(strategy_label)].copy().rename(columns={"strategy": "strategy_label"})
        strat_w = weights_df.loc[weights_df["strategy"].eq(strategy_label)].copy().rename(columns={"strategy": "strategy_label"})
        strat_ret = _attach_meta(strat_ret, meta_map[strategy_label])
        strat_w = _attach_meta(strat_w, meta_map[strategy_label])
        metrics_keep = metrics_df.loc[metrics_df["strategy"].eq(strategy_label)].copy()
        metrics_keep["strategy_label"] = metrics_keep["strategy"]
        metrics_keep = metrics_keep.drop(columns=["strategy"])
        metrics_keep["selected_lambda_risk"] = selected_config.lambda_risk
        metrics_keep["selected_kappa"] = selected_config.kappa
        metrics_keep["selected_omega_type"] = selected_config.omega_type
        metrics_keep = _attach_meta(metrics_keep, meta_map[strategy_label])
        returns_frames.append(strat_ret)
        weight_frames.append(strat_w)
        metric_frames.append(metrics_keep)

    if equal_returns is None or equal_weights is None:
        raise RuntimeError("Equal-weight benchmark was not generated")

    metric_frames.append(
        _attach_meta(
            _strategy_metrics_from_returns(equal_returns).assign(
                selected_lambda_risk=np.nan,
                selected_kappa=np.nan,
                selected_omega_type=None,
                avg_max_weight=1.0 / len(SLEEVE_ORDER),
                avg_hhi=1.0 / len(SLEEVE_ORDER),
                avg_effective_n_assets=float(len(SLEEVE_ORDER)),
            ),
            meta_map["equal_weight"],
        )
    )
    returns_frames.append(equal_returns)
    weight_frames.append(equal_weights)

    heuristic_builders: dict[str, Callable[[pd.Series], pd.Series]] = {
        "combined_top_k_equal": lambda s: _weights_top_k_equal(s, k=3),
        "combined_top_k_capped": lambda s: _weights_top_k_capped(s, k=3, max_weight=0.45),
        "combined_score_positive_capped": lambda s: _weights_score_positive_capped(s, max_weight=0.35),
        "combined_diversified_cap": lambda s: _weights_diversified_cap(s, top_n=5, max_weight=0.30),
    }
    for strategy_label, builder in heuristic_builders.items():
        returns_df, weights_df = _run_direct_weight_strategy(
            validation_signal_panel=validation_signals["combined_60_120_predictor"],
            test_signal_panel=test_signals["combined_60_120_predictor"],
            strategy_label=strategy_label,
            weight_builder=builder,
        )
        returns_df = _attach_meta(returns_df, meta_map[strategy_label])
        weights_df = _attach_meta(weights_df, meta_map[strategy_label])
        metrics_df = _strategy_metrics_from_returns(returns_df).assign(
            selected_lambda_risk=np.nan,
            selected_kappa=np.nan,
            selected_omega_type=None,
        )
        metrics_df = _attach_meta(metrics_df, meta_map[strategy_label])
        returns_frames.append(returns_df)
        weight_frames.append(weights_df)
        metric_frames.append(metrics_df)

    returns_panel = pd.concat(returns_frames, ignore_index=True).sort_values(["strategy_label", "split", "month_end"]).reset_index(drop=True)
    weights_panel = pd.concat(weight_frames, ignore_index=True).sort_values(["strategy_label", "split", "month_end", "sleeve_id"]).reset_index(drop=True)

    concentration = (
        weights_panel.groupby(["strategy_label", "split", "month_end"], as_index=False)["weight"]
        .agg(max_weight="max", hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
        .sort_values(["strategy_label", "split", "month_end"])
        .reset_index(drop=True)
    )
    concentration["effective_n_sleeves"] = 1.0 / concentration["hhi"]
    returns_panel = returns_panel.merge(concentration, on=["strategy_label", "split", "month_end"], how="left", validate="1:1")

    metrics_panel = pd.concat(metric_frames, ignore_index=True)
    behavior = (
        concentration.groupby(["strategy_label", "split"], as_index=False)
        .agg(
            avg_max_weight=("max_weight", "mean"),
            avg_hhi=("hhi", "mean"),
            avg_effective_n_assets=("effective_n_sleeves", "mean"),
        )
    )
    metrics_panel = metrics_panel.drop(columns=["avg_max_weight", "avg_hhi", "avg_effective_n_assets"], errors="ignore").merge(
        behavior, on=["strategy_label", "split"], how="left", validate="1:1"
    )
    return returns_panel, weights_panel, metrics_panel


def _build_summary_outputs(
    *,
    validation_signals: dict[str, pd.DataFrame],
    test_signals: dict[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    weights_panel: pd.DataFrame,
    metrics_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth_full = pd.concat(
        [
            validation_signals["combined_60_120_predictor"][["split", "month_end", "sleeve_id", "y_true"]],
            test_signals["combined_60_120_predictor"][["split", "month_end", "sleeve_id", "y_true"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    signal_rows: list[pd.DataFrame] = []
    for signal_name, panel in validation_signals.items():
        tmp = panel.copy()
        tmp["signal_source"] = signal_name
        signal_rows.append(tmp)
    for signal_name, panel in test_signals.items():
        tmp = panel.copy()
        tmp["signal_source"] = signal_name
        signal_rows.append(tmp)
    signals_df = pd.concat(signal_rows, ignore_index=True)

    meta_map = _base_signal_meta()
    signal_source_map = {k: v.signal_source for k, v in meta_map.items()}
    weights_panel = weights_panel.copy()
    weights_panel["source_signal_lookup"] = weights_panel["signal_source"]
    weights_panel = weights_panel.merge(
        signals_df[["split", "month_end", "sleeve_id", "signal_source", "y_pred"]].rename(columns={"y_pred": "predicted_signal"}),
        left_on=["split", "month_end", "sleeve_id", "source_signal_lookup"],
        right_on=["split", "month_end", "sleeve_id", "signal_source"],
        how="left",
        validate="m:1",
    )
    weights_panel = weights_panel.drop(columns=["source_signal_lookup", "signal_source_y"], errors="ignore")
    if "signal_source_x" in weights_panel.columns:
        weights_panel = weights_panel.rename(columns={"signal_source_x": "signal_source"})

    weights_panel = weights_panel.merge(truth_full, on=["split", "month_end", "sleeve_id"], how="left", validate="m:1")
    eq_weights = weights_panel.loc[
        weights_panel["strategy_label"].eq("equal_weight"),
        ["split", "month_end", "sleeve_id", "weight"],
    ].rename(columns={"weight": "equal_weight_ref"})
    weights_panel = weights_panel.merge(eq_weights, on=["split", "month_end", "sleeve_id"], how="left", validate="m:1")
    weights_panel["active_weight_vs_equal_weight"] = weights_panel["weight"] - weights_panel["equal_weight_ref"]
    weights_panel["sleeve_contribution"] = weights_panel["weight"] * weights_panel["y_true"]
    weights_panel["active_contribution_vs_equal_weight"] = weights_panel["active_weight_vs_equal_weight"] * weights_panel["y_true"]
    weights_panel["top_weight_flag"] = (
        weights_panel.groupby(["strategy_label", "split", "month_end"])["weight"].rank(method="first", ascending=False).eq(1)
    ).astype(int)
    weights_panel["nonzero_alloc_flag"] = weights_panel["weight"] > 1e-10
    weights_panel = weights_panel.rename(columns={"y_true": "realized_outcome"})

    ref_map = {
        "equal_weight": "equal_weight",
        "pto_nn_signal": "pto_nn_signal",
        "e2e_nn_signal": "e2e_nn_signal",
    }
    returns_panel = returns_panel.copy()
    for label, ref_strategy in ref_map.items():
        ref = returns_panel.loc[
            returns_panel["strategy_label"].eq(ref_strategy),
            ["split", "month_end", "portfolio_annualized_excess_return"],
        ].rename(columns={"portfolio_annualized_excess_return": f"ref_{label}"})
        returns_panel = returns_panel.merge(ref, on=["split", "month_end"], how="left", validate="m:1")
    returns_panel["active_return_vs_equal_weight"] = returns_panel["portfolio_annualized_excess_return"] - returns_panel["ref_equal_weight"]
    returns_panel["active_return_vs_pto_nn_signal"] = returns_panel["portfolio_annualized_excess_return"] - returns_panel["ref_pto_nn_signal"]
    returns_panel["active_return_vs_e2e_nn_signal"] = returns_panel["portfolio_annualized_excess_return"] - returns_panel["ref_e2e_nn_signal"]

    summary_rows: list[dict[str, object]] = []
    for (strategy_label, split_name), chunk in returns_panel.groupby(["strategy_label", "split"], as_index=False):
        meta = meta_map[strategy_label]
        weights_chunk = weights_panel.loc[
            weights_panel["strategy_label"].eq(strategy_label) & weights_panel["split"].eq(split_name)
        ]
        top_sleeve_freq = (
            weights_chunk.loc[weights_chunk["top_weight_flag"].eq(1)]
            .groupby("sleeve_id", as_index=False)
            .size()
            .sort_values(["size", "sleeve_id"], ascending=[False, True])
        )
        top_sleeve = None
        top_sleeve_share = float("nan")
        if not top_sleeve_freq.empty:
            top_sleeve = str(top_sleeve_freq.iloc[0]["sleeve_id"])
            top_sleeve_share = float(top_sleeve_freq.iloc[0]["size"] / chunk["month_end"].nunique())
        active_positive = chunk.loc[chunk["active_return_vs_equal_weight"] > 0.0, "active_return_vs_equal_weight"].sort_values(ascending=False)
        positive_share = float(active_positive.head(5).sum() / active_positive.sum()) if active_positive.sum() > 0 else float("nan")
        sleeve_active = (
            weights_chunk.groupby("sleeve_id", as_index=False)["active_contribution_vs_equal_weight"]
            .sum()
            .assign(abs_active=lambda x: np.abs(x["active_contribution_vs_equal_weight"]))
            .sort_values("abs_active", ascending=False)
        )
        top2_share = float(sleeve_active.head(2)["abs_active"].sum() / sleeve_active["abs_active"].sum()) if sleeve_active["abs_active"].sum() > 0 else float("nan")
        summary_rows.append(
            {
                "strategy_label": strategy_label,
                "strategy_group": meta.strategy_group,
                "signal_source": meta.signal_source,
                "allocation_rule": meta.allocation_rule,
                "split": split_name,
                "month_count": int(chunk["month_end"].nunique()),
                "avg_return": float(chunk["portfolio_annualized_excess_return"].mean()),
                "volatility": float(chunk["portfolio_annualized_excess_return"].std(ddof=1)),
                "sharpe": float(chunk["portfolio_annualized_excess_return"].mean() / chunk["portfolio_annualized_excess_return"].std(ddof=1)) if float(chunk["portfolio_annualized_excess_return"].std(ddof=1)) > 0 else float("nan"),
                "max_drawdown": float(chunk["drawdown"].min()),
                "avg_turnover": float(chunk["turnover"].mean()),
                "avg_max_weight": float(chunk["max_weight"].mean()),
                "avg_hhi": float(chunk["hhi"].mean()),
                "avg_effective_n_sleeves": float(chunk["effective_n_sleeves"].mean()),
                "avg_active_return_vs_equal_weight": float(chunk["active_return_vs_equal_weight"].mean()),
                "avg_active_return_vs_pto_nn_signal": float(chunk["active_return_vs_pto_nn_signal"].mean()),
                "avg_active_return_vs_e2e_nn_signal": float(chunk["active_return_vs_e2e_nn_signal"].mean()),
                "top5_positive_active_month_share": positive_share,
                "top2_sleeve_active_share_abs": top2_share,
                "top_weight_sleeve": top_sleeve,
                "top_weight_sleeve_frequency": top_sleeve_share,
            }
        )
    summary = pd.DataFrame(summary_rows).merge(
        metrics_panel[
            [
                "strategy_label",
                "strategy_group",
                "signal_source",
                "allocation_rule",
                "split",
                "selected_lambda_risk",
                "selected_kappa",
                "selected_omega_type",
            ]
        ],
        on=["strategy_label", "strategy_group", "signal_source", "allocation_rule", "split"],
        how="left",
        validate="1:1",
    )
    summary = summary.sort_values(["split", "sharpe"], ascending=[True, False]).reset_index(drop=True)

    sleeve_rows: list[dict[str, object]] = []
    for (strategy_label, split_name, sleeve_id), chunk in weights_panel.groupby(["strategy_label", "split", "sleeve_id"], as_index=False):
        meta = meta_map[strategy_label]
        total_active = float(chunk["active_contribution_vs_equal_weight"].sum())
        sleeve_rows.append(
            {
                "strategy_label": strategy_label,
                "strategy_group": meta.strategy_group,
                "signal_source": meta.signal_source,
                "allocation_rule": meta.allocation_rule,
                "split": split_name,
                "sleeve_id": sleeve_id,
                "month_count": int(chunk["month_end"].nunique()),
                "avg_weight": float(chunk["weight"].mean()),
                "max_weight_observed": float(chunk["weight"].max()),
                "avg_active_weight_vs_equal_weight": float(chunk["active_weight_vs_equal_weight"].mean()),
                "top_weight_frequency": float(chunk["top_weight_flag"].mean()),
                "nonzero_allocation_share": float(chunk["nonzero_alloc_flag"].mean()),
                "avg_predicted_signal": float(chunk["predicted_signal"].mean()),
                "avg_realized_outcome": float(chunk["realized_outcome"].mean()),
                "total_contribution": float(chunk["sleeve_contribution"].sum()),
                "avg_monthly_contribution": float(chunk["sleeve_contribution"].mean()),
                "total_active_contribution_vs_equal_weight": total_active,
                "avg_monthly_active_contribution_vs_equal_weight": float(chunk["active_contribution_vs_equal_weight"].mean()),
                "abs_total_active_contribution": abs(total_active),
            }
        )
    by_sleeve = pd.DataFrame(sleeve_rows)
    for (strategy_label, split_name), idx in by_sleeve.groupby(["strategy_label", "split"]).groups.items():
        denom = float(by_sleeve.loc[idx, "abs_total_active_contribution"].sum())
        by_sleeve.loc[idx, "abs_active_contribution_share"] = by_sleeve.loc[idx, "abs_total_active_contribution"] / denom if denom > 0 else np.nan
    by_sleeve = by_sleeve.sort_values(["strategy_label", "split", "abs_total_active_contribution"], ascending=[True, True, False]).reset_index(drop=True)

    china_diag = by_sleeve.loc[by_sleeve["sleeve_id"].eq("EQ_CN")].copy().sort_values(["split", "strategy_label"]).reset_index(drop=True)
    return summary, by_sleeve, weights_panel, returns_panel, china_diag


def _build_model_vs_benchmark(summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    test_supervised = summary.loc[
        summary["split"].eq("test") & summary["strategy_group"].isin(["supervised_common_allocator", "supervised_concentration_control"])
    ].copy()
    strongest = test_supervised.sort_values(["sharpe", "avg_return"], ascending=[False, False]).iloc[0]
    strongest_label = str(strongest["strategy_label"])

    rows: list[dict[str, object]] = []
    for split_name, chunk in summary.groupby("split"):
        eq = chunk.loc[chunk["strategy_label"].eq("equal_weight")].iloc[0]
        pto = chunk.loc[chunk["strategy_label"].eq("pto_nn_signal")].iloc[0]
        e2e = chunk.loc[chunk["strategy_label"].eq("e2e_nn_signal")].iloc[0]
        bench = chunk.loc[chunk["strategy_label"].eq(strongest_label)].iloc[0]
        for row in chunk.itertuples(index=False):
            rows.append(
                {
                    "split": split_name,
                    "strategy_label": row.strategy_label,
                    "strategy_group": row.strategy_group,
                    "signal_source": row.signal_source,
                    "allocation_rule": row.allocation_rule,
                    "avg_return": float(row.avg_return),
                    "volatility": float(row.volatility),
                    "sharpe": float(row.sharpe),
                    "avg_turnover": float(row.avg_turnover),
                    "benchmark_strategy_label": strongest_label,
                    "delta_avg_return_vs_equal_weight": float(row.avg_return - eq.avg_return),
                    "delta_sharpe_vs_equal_weight": float(row.sharpe - eq.sharpe),
                    "delta_volatility_vs_equal_weight": float(row.volatility - eq.volatility),
                    "delta_avg_return_vs_pto_nn_signal": float(row.avg_return - pto.avg_return),
                    "delta_sharpe_vs_pto_nn_signal": float(row.sharpe - pto.sharpe),
                    "delta_volatility_vs_pto_nn_signal": float(row.volatility - pto.volatility),
                    "delta_avg_return_vs_e2e_nn_signal": float(row.avg_return - e2e.avg_return),
                    "delta_sharpe_vs_e2e_nn_signal": float(row.sharpe - e2e.sharpe),
                    "delta_volatility_vs_e2e_nn_signal": float(row.volatility - e2e.volatility),
                    "delta_avg_return_vs_supervised_benchmark": float(row.avg_return - bench.avg_return),
                    "delta_sharpe_vs_supervised_benchmark": float(row.sharpe - bench.sharpe),
                    "delta_volatility_vs_supervised_benchmark": float(row.volatility - bench.volatility),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "sharpe"], ascending=[True, False]).reset_index(drop=True), strongest_label


def _render_report(
    *,
    summary: pd.DataFrame,
    by_sleeve: pd.DataFrame,
    comparison: pd.DataFrame,
    strongest_benchmark_label: str,
) -> str:
    test_summary = summary.loc[summary["split"].eq("test")].copy().sort_values(["sharpe", "avg_return"], ascending=[False, False])
    supervised_test = test_summary.loc[test_summary["strategy_group"].isin(["supervised_common_allocator", "supervised_concentration_control"])].copy()
    best_test = supervised_test.iloc[0]
    best_60 = test_summary.loc[test_summary["strategy_label"].eq("best_60_predictor")].iloc[0]
    best_120 = test_summary.loc[test_summary["strategy_label"].eq("best_120_predictor")].iloc[0]
    combined = test_summary.loc[test_summary["strategy_label"].eq("combined_60_120_predictor")].iloc[0]
    shared = test_summary.loc[test_summary["strategy_label"].eq("best_shared_predictor")].iloc[0]
    pto = test_summary.loc[test_summary["strategy_label"].eq("pto_nn_signal")].iloc[0]
    e2e = test_summary.loc[test_summary["strategy_label"].eq("e2e_nn_signal")].iloc[0]

    top_concentration = by_sleeve.loc[
        by_sleeve["strategy_label"].eq(str(strongest_benchmark_label)) & by_sleeve["split"].eq("test")
    ].head(5)
    eq_cn_rows = by_sleeve.loc[by_sleeve["sleeve_id"].eq("EQ_CN") & by_sleeve["split"].eq("test")].copy()

    lines: list[str] = []
    lines.append("# XOPTPOE v3 Portfolio Benchmark Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Active v3 paths only. v1/v2 were not touched.")
    lines.append("- These are long-horizon SAA decision-period diagnostics on overlapping annualized forward excess-return labels, not a clean tradable monthly wealth backtest.")
    lines.append("- Signal sources follow the latest v3 prediction benchmark results.")
    lines.append("- Combined 60m+120m uses the existing project convention: a simple equal average of the two raw predicted annualized excess-return signals.")
    lines.append("- Common-allocator rows use the same long-only robust allocator with validation-selected lambda/kappa/Omega. Concentration-control rows use fixed transparent heuristic caps.")
    lines.append("")
    lines.append("## Benchmark Signals")
    lines.append(f"- 60m benchmark: `{BEST_60_EXPERIMENT}`")
    lines.append(f"- 120m benchmark: `{BEST_120_EXPERIMENT}`")
    lines.append(f"- shared benchmark: `{BEST_SHARED_EXPERIMENT}`")
    lines.append("")
    lines.append("## Test Portfolio Readout")
    for row in test_summary.itertuples(index=False):
        lines.append(
            f"- {row.strategy_label}: avg_return={row.avg_return:.4f}, volatility={row.volatility:.4f}, sharpe={row.sharpe:.4f}, "
            f"avg_turnover={row.avg_turnover:.4f}, avg_max_weight={row.avg_max_weight:.4f}, avg_effective_n={row.avg_effective_n_sleeves:.4f}."
        )
    lines.append("")
    lines.append("## Signal Choice")
    lines.append(
        f"- best_60_predictor: sharpe={best_60.sharpe:.4f}, avg_return={best_60.avg_return:.4f}, top_weight_sleeve={best_60.top_weight_sleeve}, top_weight_freq={best_60.top_weight_sleeve_frequency:.4f}."
    )
    lines.append(
        f"- best_120_predictor: sharpe={best_120.sharpe:.4f}, avg_return={best_120.avg_return:.4f}, top_weight_sleeve={best_120.top_weight_sleeve}, top_weight_freq={best_120.top_weight_sleeve_frequency:.4f}."
    )
    lines.append(
        f"- combined_60_120_predictor: sharpe={combined.sharpe:.4f}, avg_return={combined.avg_return:.4f}, top_weight_sleeve={combined.top_weight_sleeve}, top_weight_freq={combined.top_weight_sleeve_frequency:.4f}."
    )
    lines.append(
        f"- best_shared_predictor: sharpe={shared.sharpe:.4f}, avg_return={shared.avg_return:.4f}, top_weight_sleeve={shared.top_weight_sleeve}, top_weight_freq={shared.top_weight_sleeve_frequency:.4f}."
    )
    lines.append("")
    lines.append("## China Sleeve Readout")
    for row in eq_cn_rows.itertuples(index=False):
        lines.append(
            f"- {row.strategy_label}: avg_weight={row.avg_weight:.4f}, max_weight={row.max_weight_observed:.4f}, nonzero_alloc_share={row.nonzero_allocation_share:.4f}, "
            f"top_weight_frequency={row.top_weight_frequency:.4f}, total_active_contribution_vs_equal={row.total_active_contribution_vs_equal_weight:.4f}."
        )
    lines.append("")
    lines.append("## Concentration And Attribution")
    lines.append(f"- strongest supervised benchmark on test: `{strongest_benchmark_label}`")
    for row in top_concentration.itertuples(index=False):
        lines.append(
            f"- {row.sleeve_id}: avg_weight={row.avg_weight:.4f}, total_active_contribution_vs_equal={row.total_active_contribution_vs_equal_weight:.4f}, "
            f"abs_active_share={row.abs_active_contribution_share:.4f}."
        )
    lines.append("")
    lines.append("## Neural Comparison")
    lines.append(
        f"- PTO common-allocator: avg_return={pto.avg_return:.4f}, sharpe={pto.sharpe:.4f}, avg_turnover={pto.avg_turnover:.4f}."
    )
    lines.append(
        f"- E2E common-allocator: avg_return={e2e.avg_return:.4f}, sharpe={e2e.sharpe:.4f}, avg_turnover={e2e.avg_turnover:.4f}."
    )
    lines.append(
        f"- strongest supervised benchmark vs PTO: delta_avg_return={best_test.avg_return - pto.avg_return:.4f}, delta_sharpe={best_test.sharpe - pto.sharpe:.4f}."
    )
    lines.append(
        f"- strongest supervised benchmark vs E2E: delta_avg_return={best_test.avg_return - e2e.avg_return:.4f}, delta_sharpe={best_test.sharpe - e2e.sharpe:.4f}."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- If a strategy wins on Sharpe here, that is a decision-period SAA allocation diagnostic, not proof of a tradable overlapping-label wealth process.")
    lines.append("- The critical distinction is whether the apparent gain comes from broad signal use or from concentration in EQ_US / a few sleeves / a few months.")
    return "\n".join(lines) + "\n"


def run_portfolio_benchmark_v3(
    *,
    project_root: Path,
    risk_config: RiskConfig | None = None,
    optimizer_grid: list[OptimizerConfig] | None = None,
) -> PortfolioBenchmarkOutputs:
    root = project_root.resolve()
    risk_config = risk_config or RiskConfig()
    optimizer_grid = list(optimizer_grid or candidate_optimizer_grid())
    validation_signals, test_signals = _signal_maps(root)
    returns_panel, weights_panel, metrics_panel = _build_panels_and_metrics(
        project_root=root,
        validation_signals=validation_signals,
        test_signals=test_signals,
        risk_config=risk_config,
        optimizer_grid=optimizer_grid,
    )
    summary, by_sleeve, attribution_panel, returns_panel, china_diag = _build_summary_outputs(
        validation_signals=validation_signals,
        test_signals=test_signals,
        returns_panel=returns_panel,
        weights_panel=weights_panel,
        metrics_panel=metrics_panel,
    )
    comparison, strongest_benchmark_label = _build_model_vs_benchmark(summary)
    report_text = _render_report(
        summary=summary,
        by_sleeve=by_sleeve,
        comparison=comparison,
        strongest_benchmark_label=strongest_benchmark_label,
    )
    return PortfolioBenchmarkOutputs(
        returns_panel=returns_panel,
        metrics=summary,
        by_sleeve=by_sleeve,
        attribution_panel=attribution_panel,
        china_diagnostics=china_diag,
        comparison=comparison,
        report_text=report_text,
    )


def write_portfolio_benchmark_v3_outputs(*, project_root: Path, outputs: PortfolioBenchmarkOutputs) -> None:
    paths = default_paths(project_root)
    write_text(outputs.report_text, paths.reports_dir / "portfolio_benchmark_v3_report.md")
    write_csv(outputs.metrics, paths.reports_dir / "portfolio_benchmark_v3_metrics.csv")
    write_csv(outputs.by_sleeve, paths.reports_dir / "portfolio_benchmark_v3_by_sleeve.csv")
    write_csv(outputs.attribution_panel, paths.reports_dir / "portfolio_benchmark_v3_attribution.csv")
    write_csv(outputs.china_diagnostics, paths.reports_dir / "china_portfolio_diagnostics_v3.csv")
    write_csv(outputs.comparison, paths.reports_dir / "model_vs_benchmark_v3_comparison.csv")
    write_parquet(outputs.returns_panel, paths.data_out_dir / "portfolio_benchmark_v3_returns.parquet")
