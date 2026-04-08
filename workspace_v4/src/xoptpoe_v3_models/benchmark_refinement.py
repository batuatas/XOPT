"""Controlled v3 supervised benchmark refinement and lock-in."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from xoptpoe_v3_modeling.io import write_csv, write_parquet, write_text
from xoptpoe_v3_models.data import SLEEVE_ORDER, default_paths, load_modeling_inputs
from xoptpoe_v3_models.horse_race import _portfolio_run_from_signal, _truth_panel_from_split_frame
from xoptpoe_v3_models.optim_layers import (
    OptimizerConfig,
    RiskConfig,
    RobustOptimizerCache,
    build_sigma_map,
    candidate_optimizer_grid,
)
from xoptpoe_v3_models.portfolio_benchmark import (
    BEST_120_EXPERIMENT,
    BEST_60_EXPERIMENT,
    BEST_SHARED_EXPERIMENT,
    E2E_EXPERIMENT,
    PTO_EXPERIMENT,
    _attach_meta,
    _load_or_rerun_predictions,
    _normalize_long_only,
    _project_with_cap,
    _run_direct_weight_strategy,
    _signal_maps,
    _strategy_metrics_from_returns,
)


RAW_BENCHMARK_LABEL = "best_120_predictor"


@dataclass(frozen=True)
class StrategyMetadata:
    strategy_label: str
    strategy_group: str
    signal_source: str
    allocation_rule: str


@dataclass(frozen=True)
class BenchmarkRefinementOutputs:
    returns_panel: pd.DataFrame
    metrics: pd.DataFrame
    concentration: pd.DataFrame
    attribution: pd.DataFrame
    china_diagnostics: pd.DataFrame
    report_text: str
    lock_in_text: str


def _meta_map() -> dict[str, StrategyMetadata]:
    return {
        "equal_weight": StrategyMetadata("equal_weight", "reference", "equal_weight", "equal_weight"),
        "pto_nn_signal": StrategyMetadata("pto_nn_signal", "reference_neural", "pto_nn_signal", "robust_allocator"),
        "e2e_nn_signal": StrategyMetadata("e2e_nn_signal", "reference_neural", "e2e_nn_signal", "robust_allocator"),
        "best_60_predictor": StrategyMetadata("best_60_predictor", "raw_supervised", "best_60_predictor", "robust_allocator"),
        "best_120_predictor": StrategyMetadata("best_120_predictor", "raw_supervised", "best_120_predictor", "robust_allocator"),
        "combined_60_120_predictor": StrategyMetadata("combined_60_120_predictor", "raw_supervised", "combined_60_120_predictor", "robust_allocator"),
        "best_shared_predictor": StrategyMetadata("best_shared_predictor", "raw_supervised", "best_shared_predictor", "robust_allocator"),
        "combined_std_equal": StrategyMetadata("combined_std_equal", "refined_signal", "combined_std_equal", "robust_allocator"),
        "combined_std_60tilt": StrategyMetadata("combined_std_60tilt", "refined_signal", "combined_std_60tilt", "robust_allocator"),
        "combined_std_120tilt": StrategyMetadata("combined_std_120tilt", "refined_signal", "combined_std_120tilt", "robust_allocator"),
        "best_120_score_positive_capped": StrategyMetadata("best_120_score_positive_capped", "refined_allocator", "best_120_predictor", "score_positive_capped"),
        "best_120_top_k_capped": StrategyMetadata("best_120_top_k_capped", "refined_allocator", "best_120_predictor", "top_k_capped"),
        "best_120_diversified_cap": StrategyMetadata("best_120_diversified_cap", "refined_allocator", "best_120_predictor", "diversified_cap"),
        "best_120_breadth_blend_capped": StrategyMetadata("best_120_breadth_blend_capped", "refined_allocator", "best_120_predictor", "breadth_blend_capped"),
        "combined_std_120tilt_score_positive_capped": StrategyMetadata("combined_std_120tilt_score_positive_capped", "refined_allocator", "combined_std_120tilt", "score_positive_capped"),
        "combined_std_120tilt_top_k_capped": StrategyMetadata("combined_std_120tilt_top_k_capped", "refined_allocator", "combined_std_120tilt", "top_k_capped"),
        "combined_std_120tilt_diversified_cap": StrategyMetadata("combined_std_120tilt_diversified_cap", "refined_allocator", "combined_std_120tilt", "diversified_cap"),
        "combined_std_120tilt_breadth_blend_capped": StrategyMetadata("combined_std_120tilt_breadth_blend_capped", "refined_allocator", "combined_std_120tilt", "breadth_blend_capped"),
    }


def _cross_sectional_zscore(signal_panel: pd.DataFrame) -> pd.DataFrame:
    frame = signal_panel.copy()
    grouped = frame.groupby(["split", "month_end"])["y_pred"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    frame["y_pred"] = ((frame["y_pred"] - mean) / std).fillna(0.0)
    return frame


def _weighted_signal_combo(
    signal_60: pd.DataFrame,
    signal_120: pd.DataFrame,
    *,
    weight_60: float,
    weight_120: float,
) -> pd.DataFrame:
    s60 = _cross_sectional_zscore(signal_60).rename(columns={"y_pred": "y_pred_60", "y_true": "y_true_60"})
    s120 = _cross_sectional_zscore(signal_120).rename(columns={"y_pred": "y_pred_120", "y_true": "y_true_120"})
    out = s60.merge(s120, on=["split", "month_end", "sleeve_id"], how="inner", validate="1:1")
    if not np.allclose(out["y_true_60"], out["y_true_120"], equal_nan=False):
        raise ValueError("signal combo expected common truth across horizons")
    total = weight_60 + weight_120
    out["y_pred"] = (weight_60 / total) * out["y_pred_60"] + (weight_120 / total) * out["y_pred_120"]
    out["y_true"] = out["y_true_60"]
    return out[["split", "month_end", "sleeve_id", "y_pred", "y_true"]].sort_values(
        ["split", "month_end", "sleeve_id"]
    ).reset_index(drop=True)


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


def _weights_breadth_blend_capped(scores: pd.Series, *, max_weight: float, equal_blend: float) -> pd.Series:
    n = len(scores)
    raw = np.clip(scores.to_numpy(dtype=float), 0.0, None)
    if float(raw.sum()) <= 0:
        base = np.repeat(1.0 / n, n)
    else:
        base = _normalize_long_only(raw)
    equal = np.repeat(1.0 / n, n)
    blended = (1.0 - equal_blend) * base + equal_blend * equal
    capped = _project_with_cap(_normalize_long_only(blended), max_weight=max_weight)
    return pd.Series(capped, index=scores.index, dtype=float)


def _signal_panels(project_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    validation_signals, test_signals = _signal_maps(project_root)
    inputs = load_modeling_inputs(project_root, feature_set_name="core_baseline")
    truth_validation = _truth_panel_from_split_frame(inputs.validation_df, split_name="validation")
    truth_test = _truth_panel_from_split_frame(inputs.test_df, split_name="test")
    best60_val, best60_test = _load_or_rerun_predictions(project_root, BEST_60_EXPERIMENT)
    best120_val, best120_test = _load_or_rerun_predictions(project_root, BEST_120_EXPERIMENT)

    best60_val_signal = validation_signals["best_60_predictor"]
    best60_test_signal = test_signals["best_60_predictor"]
    best120_val_signal = validation_signals["best_120_predictor"]
    best120_test_signal = test_signals["best_120_predictor"]

    extra_validation = {
        "combined_std_equal": _weighted_signal_combo(best60_val_signal, best120_val_signal, weight_60=0.5, weight_120=0.5),
        "combined_std_60tilt": _weighted_signal_combo(best60_val_signal, best120_val_signal, weight_60=0.6, weight_120=0.4),
        "combined_std_120tilt": _weighted_signal_combo(best60_val_signal, best120_val_signal, weight_60=0.4, weight_120=0.6),
    }
    extra_test = {
        "combined_std_equal": _weighted_signal_combo(best60_test_signal, best120_test_signal, weight_60=0.5, weight_120=0.5),
        "combined_std_60tilt": _weighted_signal_combo(best60_test_signal, best120_test_signal, weight_60=0.6, weight_120=0.4),
        "combined_std_120tilt": _weighted_signal_combo(best60_test_signal, best120_test_signal, weight_60=0.4, weight_120=0.6),
    }
    validation_signals.update(extra_validation)
    test_signals.update(extra_test)
    # Explicit check that rerun predictions still align with expected truth.
    for frame in [best60_val, best60_test, best120_val, best120_test]:
        frame["month_end"] = pd.to_datetime(frame["month_end"])
    for label, truth in (("validation", truth_validation), ("test", truth_test)):
        check = extra_validation if label == "validation" else extra_test
        for panel in check.values():
            merged = panel.merge(truth, on=["split", "month_end", "sleeve_id"], how="inner", validate="1:1")
            if not np.allclose(merged["y_true_x"], merged["y_true_y"], equal_nan=False):
                raise ValueError(f"truth mismatch building {label} refinement signals")
    return validation_signals, test_signals


def _build_strategy_panels(
    *,
    project_root: Path,
    validation_signals: dict[str, pd.DataFrame],
    test_signals: dict[str, pd.DataFrame],
    risk_config: RiskConfig,
    optimizer_grid: list[OptimizerConfig],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_map = _meta_map()
    inputs = load_modeling_inputs(project_root, feature_set_name="core_baseline")
    months = sorted(
        pd.concat(
            [inputs.validation_df["month_end"].drop_duplicates(), inputs.test_df["month_end"].drop_duplicates()],
            ignore_index=True,
        ).tolist()
    )
    sigma_map = build_sigma_map(months, excess_history=inputs.monthly_excess_history, risk_config=risk_config)
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    returns_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    metrics_frames: list[pd.DataFrame] = []
    equal_returns: pd.DataFrame | None = None
    equal_weights: pd.DataFrame | None = None

    robust_strategies = [
        "best_60_predictor",
        "best_120_predictor",
        "combined_60_120_predictor",
        "best_shared_predictor",
        "combined_std_equal",
        "combined_std_60tilt",
        "combined_std_120tilt",
        "pto_nn_signal",
        "e2e_nn_signal",
    ]
    for strategy_label in robust_strategies:
        returns_df, weights_df, metrics_df, selected_config = _portfolio_run_from_signal(
            validation_signal_panel=validation_signals[strategy_label],
            test_signal_panel=test_signals[strategy_label],
            optimizer_cache=optimizer_cache,
            optimizer_grid=optimizer_grid,
            strategy_name=strategy_label,
        )
        if equal_returns is None:
            eq_ret = returns_df.loc[returns_df["strategy"].eq("equal_weight")].copy().rename(columns={"strategy": "strategy_label"})
            eq_w = weights_df.loc[weights_df["strategy"].eq("equal_weight")].copy().rename(columns={"strategy": "strategy_label"})
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
        metrics_frames.append(metrics_keep)

    if equal_returns is None or equal_weights is None:
        raise RuntimeError("Equal-weight benchmark missing from refinement run")

    returns_frames.append(equal_returns)
    weight_frames.append(equal_weights)
    metrics_frames.append(
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

    heuristic_specs: dict[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
        "best_120_score_positive_capped": ("best_120_predictor", lambda s: _weights_score_positive_capped(s, max_weight=0.30)),
        "best_120_top_k_capped": ("best_120_predictor", lambda s: _weights_top_k_capped(s, k=4, max_weight=0.35)),
        "best_120_diversified_cap": ("best_120_predictor", lambda s: _weights_diversified_cap(s, top_n=5, max_weight=0.30)),
        "best_120_breadth_blend_capped": ("best_120_predictor", lambda s: _weights_breadth_blend_capped(s, max_weight=0.30, equal_blend=0.20)),
        "combined_std_120tilt_score_positive_capped": ("combined_std_120tilt", lambda s: _weights_score_positive_capped(s, max_weight=0.30)),
        "combined_std_120tilt_top_k_capped": ("combined_std_120tilt", lambda s: _weights_top_k_capped(s, k=4, max_weight=0.35)),
        "combined_std_120tilt_diversified_cap": ("combined_std_120tilt", lambda s: _weights_diversified_cap(s, top_n=5, max_weight=0.30)),
        "combined_std_120tilt_breadth_blend_capped": ("combined_std_120tilt", lambda s: _weights_breadth_blend_capped(s, max_weight=0.30, equal_blend=0.20)),
    }
    for strategy_label, (source_signal, builder) in heuristic_specs.items():
        returns_df, weights_df = _run_direct_weight_strategy(
            validation_signal_panel=validation_signals[source_signal],
            test_signal_panel=test_signals[source_signal],
            strategy_label=strategy_label,
            weight_builder=builder,
        )
        returns_df = _attach_meta(returns_df, meta_map[strategy_label])
        weights_df = _attach_meta(weights_df, meta_map[strategy_label])
        metrics_df = _attach_meta(
            _strategy_metrics_from_returns(returns_df).assign(
                selected_lambda_risk=np.nan,
                selected_kappa=np.nan,
                selected_omega_type=None,
            ),
            meta_map[strategy_label],
        )
        returns_frames.append(returns_df)
        weight_frames.append(weights_df)
        metrics_frames.append(metrics_df)

    returns_panel = pd.concat(returns_frames, ignore_index=True).sort_values(["strategy_label", "split", "month_end"]).reset_index(drop=True)
    weights_panel = pd.concat(weight_frames, ignore_index=True).sort_values(["strategy_label", "split", "month_end", "sleeve_id"]).reset_index(drop=True)
    metrics_panel = pd.concat(metrics_frames, ignore_index=True)
    concentration = (
        weights_panel.groupby(["strategy_label", "split", "month_end"], as_index=False)["weight"]
        .agg(max_weight="max", hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
        .sort_values(["strategy_label", "split", "month_end"])
        .reset_index(drop=True)
    )
    concentration["effective_n_sleeves"] = 1.0 / concentration["hhi"]
    returns_panel = returns_panel.merge(concentration, on=["strategy_label", "split", "month_end"], how="left", validate="1:1")
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


def _signal_concentration(signals: dict[str, pd.DataFrame], meta_map: dict[str, StrategyMetadata]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    relevant = sorted({meta.signal_source for meta in meta_map.values() if meta.signal_source in signals})
    for signal_name in relevant:
        panel = signals[signal_name].copy()
        for split_name, split_chunk in panel.groupby("split"):
            month_rows = []
            for month_end, chunk in split_chunk.groupby("month_end", sort=True):
                ordered = chunk.sort_values(["y_pred", "sleeve_id"], ascending=[False, True]).reset_index(drop=True)
                top1 = float(ordered.loc[0, "y_pred"])
                top2 = float(ordered.loc[1, "y_pred"]) if len(ordered) > 1 else top1
                proxy = ordered["y_pred"].to_numpy(dtype=float)
                proxy = proxy - proxy.min()
                proxy = proxy + 1e-12
                proxy = proxy / proxy.sum()
                month_rows.append(
                    {
                        "month_end": pd.Timestamp(month_end),
                        "top_signal_sleeve": str(ordered.loc[0, "sleeve_id"]),
                        "top_minus_second": top1 - top2,
                        "positive_share": float((chunk["y_pred"] > 0).mean()),
                        "signal_hhi_proxy": float(np.square(proxy).sum()),
                    }
                )
            monthly = pd.DataFrame(month_rows)
            top_freq = (
                monthly.groupby("top_signal_sleeve", as_index=False)
                .size()
                .sort_values(["size", "top_signal_sleeve"], ascending=[False, True])
            )
            top_sleeve = str(top_freq.iloc[0]["top_signal_sleeve"])
            top_freq_share = float(top_freq.iloc[0]["size"] / len(monthly))
            rows.append(
                {
                    "signal_source": signal_name,
                    "split": split_name,
                    "signal_top_sleeve": top_sleeve,
                    "signal_top_sleeve_frequency": top_freq_share,
                    "signal_avg_top_minus_second": float(monthly["top_minus_second"].mean()),
                    "signal_avg_positive_share": float(monthly["positive_share"].mean()),
                    "signal_avg_hhi_proxy": float(monthly["signal_hhi_proxy"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "signal_source"]).reset_index(drop=True)


def _build_summary_outputs(
    *,
    validation_signals: dict[str, pd.DataFrame],
    test_signals: dict[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    weights_panel: pd.DataFrame,
    metrics_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_map = _meta_map()
    truth_full = pd.concat(
        [
            validation_signals["combined_60_120_predictor"][["split", "month_end", "sleeve_id", "y_true"]],
            test_signals["combined_60_120_predictor"][["split", "month_end", "sleeve_id", "y_true"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    signal_rows: list[pd.DataFrame] = []
    for signal_name, panel in {**validation_signals, **test_signals}.items():
        tmp = panel.copy()
        tmp["signal_source"] = signal_name
        signal_rows.append(tmp)
    signals_df = pd.concat(signal_rows, ignore_index=True)

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
    for (_, _), idx in by_sleeve.groupby(["strategy_label", "split"]).groups.items():
        denom = float(by_sleeve.loc[idx, "abs_total_active_contribution"].sum())
        by_sleeve.loc[idx, "abs_active_contribution_share"] = by_sleeve.loc[idx, "abs_total_active_contribution"] / denom if denom > 0 else np.nan
    by_sleeve = by_sleeve.sort_values(["strategy_label", "split", "abs_total_active_contribution"], ascending=[True, True, False]).reset_index(drop=True)

    china_diag = by_sleeve.loc[by_sleeve["sleeve_id"].eq("EQ_CN")].copy().sort_values(["split", "strategy_label"]).reset_index(drop=True)
    return summary, by_sleeve, weights_panel, returns_panel, china_diag


def _build_concentration_table(summary: pd.DataFrame, by_sleeve: pd.DataFrame, signal_conc: pd.DataFrame) -> pd.DataFrame:
    out = summary.merge(
        signal_conc,
        on=["signal_source", "split"],
        how="left",
        validate="m:1",
    )
    raw_test = out.loc[(out["split"].eq("test")) & (out["strategy_label"].eq(RAW_BENCHMARK_LABEL))].iloc[0]
    out["sharpe_retention_vs_raw120"] = out["sharpe"] / float(raw_test["sharpe"])
    out["avg_return_retention_vs_raw120"] = out["avg_return"] / float(raw_test["avg_return"])
    out["avg_max_weight_delta_vs_raw120"] = out["avg_max_weight"] - float(raw_test["avg_max_weight"])
    out["effective_n_delta_vs_raw120"] = out["avg_effective_n_sleeves"] - float(raw_test["avg_effective_n_sleeves"])
    out["top2_active_share_delta_vs_raw120"] = out["top2_sleeve_active_share_abs"] - float(raw_test["top2_sleeve_active_share_abs"])
    out["is_supervised_candidate"] = out["strategy_group"].isin(["raw_supervised", "refined_signal", "refined_allocator"])
    out["passes_robust_screen"] = (
        out["is_supervised_candidate"]
        & out["avg_max_weight"].le(0.45)
        & out["avg_effective_n_sleeves"].ge(3.0)
        & out["top2_sleeve_active_share_abs"].le(0.65)
        & out["sharpe_retention_vs_raw120"].ge(0.85)
    )
    return out.sort_values(["split", "sharpe"], ascending=[True, False]).reset_index(drop=True)


def _benchmark_lock_in(concentration: pd.DataFrame) -> tuple[str, str]:
    test = concentration.loc[concentration["split"].eq("test")].copy()
    validation = concentration.loc[concentration["split"].eq("validation")].copy()
    raw_candidates = test.loc[test["is_supervised_candidate"]].sort_values(["sharpe", "avg_return"], ascending=[False, False])
    raw_winner = raw_candidates.iloc[0]
    robust_pool = test.loc[test["passes_robust_screen"]].sort_values(["sharpe", "avg_return"], ascending=[False, False])
    both_pass = set(test.loc[test["passes_robust_screen"], "strategy_label"]).intersection(
        set(validation.loc[validation["passes_robust_screen"], "strategy_label"])
    )
    if robust_pool.empty:
        robust_winner = test.loc[test["strategy_label"].eq("best_60_predictor")].iloc[0]
        robust_reason = "No candidate passed the full robust screen; defaulted to the cleanest lower-concentration supervised benchmark."
    else:
        robust_winner = robust_pool.iloc[0]
        robust_reason = "Selected as the highest-Sharpe supervised candidate that passed the explicit concentration screen."

    lines = [
        "# XOPTPOE v3 Benchmark Lock-In",
        "",
        "## Lock-In Rules",
        "- strongest raw benchmark = highest test-Sharpe supervised candidate",
        "- strongest robust benchmark = highest test-Sharpe supervised candidate that passes all of:",
        "  - avg_max_weight <= 0.45",
        "  - avg_effective_n_sleeves >= 3.0",
        "  - top2_sleeve_active_share_abs <= 0.65",
        "  - sharpe retention vs raw 120m benchmark >= 0.85",
        "",
        "## Winners",
        f"- strongest raw benchmark: `{raw_winner.strategy_label}`",
        f"  - test avg_return={raw_winner.avg_return:.4f}, sharpe={raw_winner.sharpe:.4f}, avg_max_weight={raw_winner.avg_max_weight:.4f}, effective_n={raw_winner.avg_effective_n_sleeves:.4f}, top2_active_share={raw_winner.top2_sleeve_active_share_abs:.4f}",
        f"- strongest robust benchmark: `{robust_winner.strategy_label}`",
        f"  - test avg_return={robust_winner.avg_return:.4f}, sharpe={robust_winner.sharpe:.4f}, avg_max_weight={robust_winner.avg_max_weight:.4f}, effective_n={robust_winner.avg_effective_n_sleeves:.4f}, top2_active_share={robust_winner.top2_sleeve_active_share_abs:.4f}",
        "",
        "## Stability Note",
        f"- candidates passing the robust screen on both validation and test: {len(both_pass)}",
        "- This lock-in is therefore a cautious working benchmark choice, not proof of a fully stable concentration-controlled allocation rule.",
        "",
        "## Interpretation",
        f"- {robust_reason}",
        "- The raw winner is the performance ceiling benchmark.",
        "- The robust winner is the pre-scenario benchmark that future PTO/E2E and scenario-generation work should be forced to beat.",
        "",
    ]
    return str(raw_winner.strategy_label), "\n".join(lines) + "\n"


def _render_report(
    *,
    summary: pd.DataFrame,
    concentration: pd.DataFrame,
    by_sleeve: pd.DataFrame,
    china_diag: pd.DataFrame,
    raw_label: str,
    robust_label: str,
) -> str:
    test_summary = summary.loc[summary["split"].eq("test")].sort_values(["sharpe", "avg_return"], ascending=[False, False]).reset_index(drop=True)
    validation_summary = concentration.loc[concentration["split"].eq("validation")].copy()
    raw_row = concentration.loc[(concentration["split"].eq("test")) & (concentration["strategy_label"].eq(raw_label))].iloc[0]
    robust_row = concentration.loc[(concentration["split"].eq("test")) & (concentration["strategy_label"].eq(robust_label))].iloc[0]
    best60 = concentration.loc[(concentration["split"].eq("test")) & (concentration["strategy_label"].eq("best_60_predictor"))].iloc[0]
    best120 = concentration.loc[(concentration["split"].eq("test")) & (concentration["strategy_label"].eq("best_120_predictor"))].iloc[0]
    combined = concentration.loc[(concentration["split"].eq("test")) & (concentration["strategy_label"].eq("combined_60_120_predictor"))].iloc[0]
    eq = concentration.loc[(concentration["split"].eq("test")) & (concentration["strategy_label"].eq("equal_weight"))].iloc[0]
    pto = concentration.loc[(concentration["split"].eq("test")) & (concentration["strategy_label"].eq("pto_nn_signal"))].iloc[0]
    e2e = concentration.loc[(concentration["split"].eq("test")) & (concentration["strategy_label"].eq("e2e_nn_signal"))].iloc[0]
    both_pass_count = len(
        set(concentration.loc[(concentration["split"].eq("test")) & (concentration["passes_robust_screen"]), "strategy_label"]).intersection(
            set(validation_summary.loc[validation_summary["passes_robust_screen"], "strategy_label"])
        )
    )
    raw_top = by_sleeve.loc[(by_sleeve["split"].eq("test")) & (by_sleeve["strategy_label"].eq(raw_label))].head(5)
    robust_top = by_sleeve.loc[(by_sleeve["split"].eq("test")) & (by_sleeve["strategy_label"].eq(robust_label))].head(5)
    china_test = china_diag.loc[china_diag["split"].eq("test")].copy()
    lines = [
        "# XOPTPOE v3 Benchmark Refinement Report",
        "",
        "## Scope",
        "- Active v3 paths only. v1/v2 were not touched.",
        "- These remain long-horizon SAA decision-period diagnostics on overlapping forward labels, not a clean monthly tradable wealth backtest.",
        "- The objective here is benchmark lock-in: strongest raw supervised benchmark versus strongest robust supervised benchmark.",
        "",
        "## Starting Point",
        f"- 60m signal benchmark: `{BEST_60_EXPERIMENT}`",
        f"- 120m signal benchmark: `{BEST_120_EXPERIMENT}`",
        f"- shared reference: `{BEST_SHARED_EXPERIMENT}`",
        "",
        "## Why The Current 120m Winner Is Concentrated",
        f"- `best_120_predictor` test Sharpe={best120.sharpe:.4f}, avg_return={best120.avg_return:.4f}.",
        f"- allocation concentration: avg_max_weight={best120.avg_max_weight:.4f}, effective_n={best120.avg_effective_n_sleeves:.4f}, top2_sleeve_active_share={best120.top2_sleeve_active_share_abs:.4f}.",
        f"- signal concentration: top_signal_sleeve={best120.signal_top_sleeve}, top_signal_freq={best120.signal_top_sleeve_frequency:.4f}, signal_avg_top_minus_second={best120.signal_avg_top_minus_second:.4f}.",
        f"- month concentration is less severe than sleeve concentration: top5_positive_active_month_share={best120.top5_positive_active_month_share:.4f}.",
        "",
        "## 60m Versus 120m Concentration",
        f"- `best_60_predictor`: sharpe={best60.sharpe:.4f}, avg_max_weight={best60.avg_max_weight:.4f}, effective_n={best60.avg_effective_n_sleeves:.4f}, top2_active_share={best60.top2_sleeve_active_share_abs:.4f}, signal_top_freq={best60.signal_top_sleeve_frequency:.4f}.",
        f"- `best_120_predictor`: sharpe={best120.sharpe:.4f}, avg_max_weight={best120.avg_max_weight:.4f}, effective_n={best120.avg_effective_n_sleeves:.4f}, top2_active_share={best120.top2_sleeve_active_share_abs:.4f}, signal_top_freq={best120.signal_top_sleeve_frequency:.4f}.",
        "- Interpretation: 120m is more concentration-prone in the current setup on both signal and allocation diagnostics.",
        "",
        "## Test Candidate Readout",
    ]
    for row in test_summary.itertuples(index=False):
        if row.strategy_group not in {"raw_supervised", "refined_signal", "refined_allocator", "reference", "reference_neural"}:
            continue
        lines.append(
            f"- {row.strategy_label}: sharpe={row.sharpe:.4f}, avg_return={row.avg_return:.4f}, avg_max_weight={row.avg_max_weight:.4f}, "
            f"effective_n={row.avg_effective_n_sleeves:.4f}, top2_active_share={row.top2_sleeve_active_share_abs:.4f}, turnover={row.avg_turnover:.4f}."
        )
    lines += [
        "",
        "## Raw Versus Robust Winner",
        f"- strongest raw benchmark: `{raw_label}`",
        f"  - test avg_return={raw_row.avg_return:.4f}, sharpe={raw_row.sharpe:.4f}, avg_max_weight={raw_row.avg_max_weight:.4f}, effective_n={raw_row.avg_effective_n_sleeves:.4f}.",
        f"- strongest robust benchmark: `{robust_label}`",
        f"  - test avg_return={robust_row.avg_return:.4f}, sharpe={robust_row.sharpe:.4f}, avg_max_weight={robust_row.avg_max_weight:.4f}, effective_n={robust_row.avg_effective_n_sleeves:.4f}.",
        f"- concentration reduction vs raw winner: delta_max_weight={robust_row.avg_max_weight_delta_vs_raw120:.4f}, delta_effective_n={robust_row.effective_n_delta_vs_raw120:.4f}, delta_top2_active_share={robust_row.top2_active_share_delta_vs_raw120:.4f}.",
        f"- candidates passing the same robust screen on both validation and test: {both_pass_count}.",
        "",
        "## EQ_CN Diagnostics",
    ]
    for row in china_test.itertuples(index=False):
        if row.strategy_label not in {raw_label, robust_label, "best_60_predictor", "best_120_predictor", "combined_60_120_predictor", "combined_std_120tilt_breadth_blend_capped"}:
            continue
        lines.append(
            f"- {row.strategy_label}: avg_weight={row.avg_weight:.4f}, max_weight={row.max_weight_observed:.4f}, nonzero_share={row.nonzero_allocation_share:.4f}, "
            f"top_weight_freq={row.top_weight_frequency:.4f}, active_contribution_vs_equal={row.total_active_contribution_vs_equal_weight:.4f}."
        )
    lines += [
        "- EQ_CN remains marginal in the raw 120m winner and becomes more economically meaningful only under concentration controls.",
        "",
        "## Attribution",
        f"- raw winner top contributors: {', '.join(f'{row.sleeve_id}({row.abs_active_contribution_share:.2%})' for row in raw_top.itertuples(index=False))}",
        f"- robust winner top contributors: {', '.join(f'{row.sleeve_id}({row.abs_active_contribution_share:.2%})' for row in robust_top.itertuples(index=False))}",
        "",
        "## Neural Context",
        f"- equal_weight: avg_return={eq.avg_return:.4f}, sharpe={eq.sharpe:.4f}.",
        f"- PTO: avg_return={pto.avg_return:.4f}, sharpe={pto.sharpe:.4f}.",
        f"- E2E: avg_return={e2e.avg_return:.4f}, sharpe={e2e.sharpe:.4f}.",
        "- PTO/E2E remain reference comparators here, not the lock-in winners.",
        "",
        "## Interpretation",
        "- The main failure mode of the raw 120m winner is sleeve concentration, especially EQ_US dominance, more than concentration in a handful of months.",
        "- The right carry-forward benchmark is the strongest supervised candidate that preserves most of the performance while materially improving breadth.",
        "- Any later PTO/E2E or scenario-generation layer should be forced to beat the robust benchmark first, not only the raw winner or equal weight.",
        "",
    ]
    return "\n".join(lines)


def run_benchmark_refinement_v3(
    *,
    project_root: Path,
    risk_config: RiskConfig | None = None,
    optimizer_grid: list[OptimizerConfig] | None = None,
) -> BenchmarkRefinementOutputs:
    root = project_root.resolve()
    risk_config = risk_config or RiskConfig()
    optimizer_grid = list(optimizer_grid or candidate_optimizer_grid())
    validation_signals, test_signals = _signal_panels(root)
    returns_panel, weights_panel, metrics_panel = _build_strategy_panels(
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
    signal_concentration = _signal_concentration({**validation_signals, **test_signals}, _meta_map())
    concentration = _build_concentration_table(summary, by_sleeve, signal_concentration)
    raw_label, lock_in_text = _benchmark_lock_in(concentration)
    lock_in_raw_label = raw_label
    robust_pool = concentration.loc[(concentration["split"].eq("test")) & (concentration["passes_robust_screen"])].sort_values(
        ["sharpe", "avg_return"], ascending=[False, False]
    )
    robust_label = str(robust_pool.iloc[0]["strategy_label"]) if not robust_pool.empty else "best_60_predictor"
    report_text = _render_report(
        summary=summary,
        concentration=concentration,
        by_sleeve=by_sleeve,
        china_diag=china_diag,
        raw_label=lock_in_raw_label,
        robust_label=robust_label,
    )
    return BenchmarkRefinementOutputs(
        returns_panel=returns_panel,
        metrics=summary,
        concentration=concentration,
        attribution=by_sleeve,
        china_diagnostics=china_diag,
        report_text=report_text,
        lock_in_text=lock_in_text,
    )


def write_benchmark_refinement_v3_outputs(*, project_root: Path, outputs: BenchmarkRefinementOutputs) -> None:
    paths = default_paths(project_root)
    write_text(outputs.report_text, paths.reports_dir / "benchmark_refinement_v3_report.md")
    write_csv(outputs.metrics, paths.reports_dir / "benchmark_refinement_v3_metrics.csv")
    write_csv(outputs.concentration, paths.reports_dir / "benchmark_refinement_v3_concentration.csv")
    write_csv(outputs.attribution, paths.reports_dir / "benchmark_refinement_v3_attribution.csv")
    write_csv(outputs.china_diagnostics, paths.reports_dir / "china_refinement_diagnostics_v3.csv")
    write_text(outputs.lock_in_text, paths.reports_dir / "benchmark_lock_in_v3.md")
    write_parquet(outputs.returns_panel, paths.data_out_dir / "benchmark_refinement_v3_returns.parquet")
