from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet, Ridge

from xoptpoe_v3_modeling.features import feature_columns_for_set
from xoptpoe_v3_models.data import SLEEVE_ORDER, load_modeling_inputs
from xoptpoe_v3_models.optim_layers import OptimizerConfig, RiskConfig, RobustOptimizerCache, build_sigma_map
from xoptpoe_v3_models.preprocess import fit_preprocessor


RECENT_5Y_TARGET_ANCHORS: tuple[str, ...] = (
    "2021-12-31",
    "2022-12-31",
    "2023-12-31",
    "2024-12-31",
    "2025-12-31",
)


@dataclass(frozen=True)
class PlotPaths:
    project_root: Path
    reports_dir: Path
    plots_dir: Path
    data_dir: Path
    final_dir: Path
    intermediate_dir: Path


@dataclass(frozen=True)
class RecentSnapshotBundle:
    prediction_frame: pd.DataFrame
    weight_frame: pd.DataFrame
    fit_note: str


@dataclass(frozen=True)
class AnnualWealthPathBundle:
    path_frame: pd.DataFrame
    weight_frame: pd.DataFrame
    anchor_frame: pd.DataFrame
    fit_note: str


@dataclass(frozen=True)
class PlotContext:
    paths: PlotPaths
    benchmark_stack: pd.DataFrame
    portfolio_metrics: pd.DataFrame
    portfolio_by_sleeve: pd.DataFrame
    portfolio_attribution: pd.DataFrame
    china_portfolio: pd.DataFrame
    prediction_metrics: pd.DataFrame
    prediction_by_sleeve: pd.DataFrame
    china_prediction: pd.DataFrame
    predictions_test: pd.DataFrame
    portfolio_returns: pd.DataFrame
    refinement_concentration: pd.DataFrame
    active_60_prediction: str
    active_120_prediction: str
    raw_portfolio_benchmark: str
    robust_portfolio_benchmark: str
    sleeve_order: tuple[str, ...]
    sleeve_wealth: pd.DataFrame
    recent_5y_snapshot: RecentSnapshotBundle
    annual_rebalance_5y_path: AnnualWealthPathBundle


def default_paths(project_root: Path) -> PlotPaths:
    root = project_root.resolve()
    return PlotPaths(
        project_root=root,
        reports_dir=root / "reports" / "v3_long_horizon_china",
        plots_dir=root / "reports" / "v3_long_horizon_china" / "plots",
        data_dir=root / "data" / "modeling_v3",
        final_dir=root / "data" / "final_v3_long_horizon_china",
        intermediate_dir=root / "data" / "intermediate",
    )


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _stack_value(stack: pd.DataFrame, item: str) -> str:
    row = stack.loc[stack["stack_item"].eq(item)]
    if row.empty:
        raise KeyError(f"stack item missing: {item}")
    return str(row.iloc[0]["object_name"])


def _parse_selected_params(text: str) -> dict[str, object]:
    parsed = ast.literal_eval(str(text))
    if not isinstance(parsed, dict):
        raise ValueError(f"selected_params is not a dict: {text}")
    return parsed


def _resolve_anchor_months(available_months: pd.Series) -> tuple[pd.Timestamp, ...]:
    months = sorted(pd.Timestamp(v) for v in pd.to_datetime(available_months).dropna().unique())
    resolved: list[pd.Timestamp] = []
    for target in RECENT_5Y_TARGET_ANCHORS:
        ts = pd.Timestamp(target)
        if ts in months:
            resolved.append(ts)
            continue
        prior = [m for m in months if m <= ts]
        if not prior:
            raise ValueError(f"No available month on or before requested anchor {target}")
        resolved.append(prior[-1])
    return tuple(resolved)


def _load_sleeve_wealth(paths: PlotPaths) -> pd.DataFrame:
    base = _load_frame(paths.intermediate_dir / "sleeve_monthly_returns.csv")
    china = _load_frame(paths.final_dir / "china_sleeve_monthly_returns.csv")
    returns = pd.concat([base, china], ignore_index=True)
    returns["month_end"] = pd.to_datetime(returns["month_end"])
    returns["ret_1m_realized"] = pd.to_numeric(returns["ret_1m_realized"], errors="coerce")
    panel = (
        returns.loc[returns["month_end"] >= pd.Timestamp("2006-01-31")]
        .pivot(index="month_end", columns="sleeve_id", values="ret_1m_realized")
        .sort_index()
        .reindex(columns=list(SLEEVE_ORDER))
    )
    wealth = (1.0 + panel.fillna(0.0)).cumprod()
    wealth = wealth.divide(wealth.iloc[0], axis=1)
    out = wealth.reset_index().melt(id_vars="month_end", var_name="sleeve_id", value_name="wealth_index")
    out["month_end"] = pd.to_datetime(out["month_end"])
    return out.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)


def _load_total_return_panel(paths: PlotPaths) -> pd.DataFrame:
    base = _load_frame(paths.intermediate_dir / "sleeve_monthly_returns.csv")
    china = _load_frame(paths.final_dir / "china_sleeve_monthly_returns.csv")
    returns = pd.concat([base, china], ignore_index=True)
    returns["month_end"] = pd.to_datetime(returns["month_end"])
    returns["ret_1m_realized"] = pd.to_numeric(returns["ret_1m_realized"], errors="coerce")
    panel = (
        returns.pivot(index="month_end", columns="sleeve_id", values="ret_1m_realized")
        .sort_index()
        .reindex(columns=list(SLEEVE_ORDER))
    )
    if panel.columns.isna().any():
        raise ValueError("total return panel could not be aligned to the active v3 sleeve order")
    return panel.astype(float)


def _build_annual_rebalance_5y_path(
    *,
    paths: PlotPaths,
    benchmark_stack: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
) -> AnnualWealthPathBundle:
    active_60_prediction = _stack_value(benchmark_stack, "strongest_60m_prediction_benchmark")
    metrics_row = prediction_metrics.loc[prediction_metrics["experiment_name"].eq(active_60_prediction)]
    if metrics_row.empty:
        raise KeyError(f"prediction metrics missing active 60m benchmark row: {active_60_prediction}")
    metrics_row = metrics_row.iloc[0]

    model_name, feature_set_name, horizon_mode = active_60_prediction.split("__")
    if horizon_mode != "separate_60":
        raise ValueError(f"annual walk-forward path expects a separate_60 benchmark, got {active_60_prediction}")

    full_panel = _load_frame(paths.final_dir / "modeling_panel_hstack.parquet")
    full_panel["month_end"] = pd.to_datetime(full_panel["month_end"])
    feature_manifest = _load_frame(paths.data_dir / "feature_set_manifest.csv")
    feature_columns = feature_columns_for_set(feature_manifest, feature_set_name)
    total_return_panel = _load_total_return_panel(paths)

    train_pool = full_panel.loc[
        full_panel["horizon_months"].eq(60)
        & full_panel["baseline_trainable_flag"].eq(1)
        & full_panel["target_available_flag"].eq(1)
        & full_panel["annualized_excess_forward_return"].notna()
    ].copy()
    if train_pool.empty:
        raise ValueError("No labeled 60m rows available for annual walk-forward path")

    score_pool = full_panel.loc[full_panel["horizon_months"].eq(60)].copy()
    valid_anchor_months = []
    last_rebalance = total_return_panel.index.max() - pd.offsets.MonthEnd(12)
    for month_end, chunk in score_pool.groupby("month_end", sort=True):
        ts = pd.Timestamp(month_end)
        if ts.month != 12:
            continue
        if ts < pd.Timestamp("2014-12-31") or ts > last_rebalance:
            continue
        if set(chunk["sleeve_id"]) != set(SLEEVE_ORDER):
            continue
        valid_anchor_months.append(ts)
    anchor_months = tuple(valid_anchor_months)
    if len(anchor_months) < 3:
        raise ValueError("Too few year-end anchor months available for annual walk-forward path")

    portfolio_metrics = _load_frame(paths.reports_dir / "portfolio_benchmark_v3_metrics.csv")
    config_row = portfolio_metrics.loc[
        portfolio_metrics["strategy_label"].eq("best_60_predictor") & portfolio_metrics["split"].eq("test")
    ]
    if config_row.empty:
        raise KeyError("best_60_predictor allocator settings not found in portfolio benchmark metrics")
    config_row = config_row.iloc[0]
    optimizer_config = OptimizerConfig(
        lambda_risk=float(config_row["selected_lambda_risk"]),
        kappa=float(config_row["selected_kappa"]),
        omega_type=str(config_row["selected_omega_type"]),
    )

    monthly_excess_history = load_modeling_inputs(
        paths.project_root,
        feature_set_name="core_baseline",
    ).monthly_excess_history
    sigma_map = build_sigma_map(
        list(anchor_months),
        excess_history=monthly_excess_history,
        risk_config=RiskConfig(
            lookback_months=60,
            min_months=12,
            ewma_beta=0.94,
            diagonal_shrinkage=0.10,
            ridge=1e-6,
        ),
    )
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)
    params = _parse_selected_params(metrics_row["selected_params"])

    path_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    nav_by_strategy = {"best_60_predictor": 1.0, "equal_weight": 1.0}

    for anchor in anchor_months:
        cutoff = pd.Timestamp(anchor) - pd.offsets.MonthEnd(60)
        train_df = train_pool.loc[train_pool["month_end"].le(cutoff)].copy()
        if len(train_df) < len(SLEEVE_ORDER) * 12:
            continue

        score_df = score_pool.loc[score_pool["month_end"].eq(anchor)].copy()
        ordered = score_df.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
        if ordered["sleeve_id"].isna().any():
            raise ValueError(f"Anchor {anchor.date()} is missing sleeves for walk-forward scoring")

        preprocessor = fit_preprocessor(
            train_df,
            feature_manifest=feature_manifest,
            feature_columns=feature_columns,
        )
        x_train, _ = preprocessor.transform(train_df)
        x_score, _ = preprocessor.transform(ordered)
        y_train = train_df["annualized_excess_forward_return"].to_numpy(dtype=float)

        if model_name == "elastic_net":
            model = ElasticNet(
                alpha=float(params["alpha"]),
                l1_ratio=float(params["l1_ratio"]),
                fit_intercept=True,
                max_iter=20000,
                random_state=42,
            )
        elif model_name == "ridge":
            model = Ridge(alpha=float(params["alpha"]), fit_intercept=True, random_state=42)
        else:
            raise ValueError(f"Unsupported annual walk-forward benchmark model: {model_name}")

        model.fit(x_train, y_train)
        ordered["y_pred"] = model.predict(x_score)
        mu = torch.tensor(ordered["y_pred"].to_numpy(dtype=np.float32), dtype=torch.float32)
        weights = optimizer_cache.solve(anchor, mu, optimizer_config).detach().cpu().numpy().astype(float)
        weights = np.clip(weights, 0.0, None)
        weights = weights / weights.sum()

        anchor_rows.extend(
            [
                {
                    "anchor_month_end": pd.Timestamp(anchor),
                    "strategy_label": "best_60_predictor",
                    "wealth_index": float(nav_by_strategy["best_60_predictor"]),
                },
                {
                    "anchor_month_end": pd.Timestamp(anchor),
                    "strategy_label": "equal_weight",
                    "wealth_index": float(nav_by_strategy["equal_weight"]),
                },
            ]
        )

        for sleeve_id, pred, weight in zip(
            ordered["sleeve_id"].tolist(),
            ordered["y_pred"].to_numpy(dtype=float),
            weights,
            strict=True,
        ):
            weight_rows.append(
                {
                    "anchor_month_end": pd.Timestamp(anchor),
                    "sleeve_id": sleeve_id,
                    "predicted_annualized_excess_return": float(pred),
                    "portfolio_weight": float(weight),
                }
            )

        next_anchor = pd.Timestamp(anchor) + pd.offsets.YearEnd(1)
        hold_months = total_return_panel.index[(total_return_panel.index > pd.Timestamp(anchor)) & (total_return_panel.index <= next_anchor)]
        hold_panel = total_return_panel.loc[hold_months, list(SLEEVE_ORDER)].copy()
        hold_panel = hold_panel.dropna(how="any")
        for month_end, row in hold_panel.iterrows():
            port_ret = float(np.dot(weights, row.to_numpy(dtype=float)))
            eq_ret = float(row.mean())
            nav_by_strategy["best_60_predictor"] *= 1.0 + port_ret
            nav_by_strategy["equal_weight"] *= 1.0 + eq_ret
            path_rows.extend(
                [
                    {
                        "month_end": pd.Timestamp(month_end),
                        "strategy_label": "best_60_predictor",
                        "ret_1m_realized": port_ret,
                        "wealth_index": float(nav_by_strategy["best_60_predictor"]),
                        "anchor_month_end": pd.Timestamp(anchor),
                    },
                    {
                        "month_end": pd.Timestamp(month_end),
                        "strategy_label": "equal_weight",
                        "ret_1m_realized": eq_ret,
                        "wealth_index": float(nav_by_strategy["equal_weight"]),
                        "anchor_month_end": pd.Timestamp(anchor),
                    },
                ]
            )

    path_frame = pd.DataFrame(path_rows).sort_values(["strategy_label", "month_end"]).reset_index(drop=True)
    weight_frame = pd.DataFrame(weight_rows).sort_values(["anchor_month_end", "sleeve_id"]).reset_index(drop=True)
    anchor_frame = pd.DataFrame(anchor_rows).sort_values(["strategy_label", "anchor_month_end"]).reset_index(drop=True)
    if path_frame.empty or weight_frame.empty:
        raise ValueError("annual walk-forward path could not be constructed from the available v3 data")
    fit_note = (
        "Annual walk-forward illustration: each December, refit the locked 60m benchmark on then-observable labeled history, "
        "solve the locked robust allocator, and hold the resulting weights through the next calendar year of realized monthly sleeve returns."
    )
    return AnnualWealthPathBundle(
        path_frame=path_frame,
        weight_frame=weight_frame,
        anchor_frame=anchor_frame,
        fit_note=fit_note,
    )


def _fit_recent_active_60_snapshot(
    *,
    paths: PlotPaths,
    benchmark_stack: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
) -> RecentSnapshotBundle:
    active_60_prediction = _stack_value(benchmark_stack, "strongest_60m_prediction_benchmark")
    metrics_row = prediction_metrics.loc[prediction_metrics["experiment_name"].eq(active_60_prediction)]
    if metrics_row.empty:
        raise KeyError(f"prediction metrics missing active 60m benchmark row: {active_60_prediction}")
    metrics_row = metrics_row.iloc[0]

    model_name, feature_set_name, horizon_mode = active_60_prediction.split("__")
    if horizon_mode != "separate_60":
        raise ValueError(f"recent 5Y snapshot expects a separate_60 benchmark, got {active_60_prediction}")

    full_panel = _load_frame(paths.final_dir / "modeling_panel_hstack.parquet")
    full_panel["month_end"] = pd.to_datetime(full_panel["month_end"])
    feature_manifest = _load_frame(paths.data_dir / "feature_set_manifest.csv")
    feature_columns = feature_columns_for_set(feature_manifest, feature_set_name)

    train_df = full_panel.loc[
        full_panel["horizon_months"].eq(60)
        & full_panel["baseline_trainable_flag"].eq(1)
        & full_panel["target_available_flag"].eq(1)
        & full_panel["annualized_excess_forward_return"].notna()
    ].copy()
    if train_df.empty:
        raise ValueError("No labeled 60m rows available for recent snapshot fit")

    anchor_months = _resolve_anchor_months(
        full_panel.loc[full_panel["horizon_months"].eq(60), "month_end"]
    )
    snapshot_df = full_panel.loc[
        full_panel["horizon_months"].eq(60) & full_panel["month_end"].isin(anchor_months)
    ].copy()
    snapshot_df = snapshot_df.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)
    expected_rows = len(anchor_months) * len(SLEEVE_ORDER)
    if len(snapshot_df) != expected_rows:
        raise ValueError(
            f"Recent snapshot rows incomplete: expected {expected_rows}, observed {len(snapshot_df)}"
        )

    preprocessor = fit_preprocessor(
        train_df,
        feature_manifest=feature_manifest,
        feature_columns=feature_columns,
    )
    x_train, _ = preprocessor.transform(train_df)
    x_snap, _ = preprocessor.transform(snapshot_df)
    y_train = train_df["annualized_excess_forward_return"].to_numpy(dtype=float)
    params = _parse_selected_params(metrics_row["selected_params"])

    if model_name == "elastic_net":
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            fit_intercept=True,
            max_iter=20000,
            random_state=42,
        )
    elif model_name == "ridge":
        model = Ridge(alpha=float(params["alpha"]), fit_intercept=True, random_state=42)
    else:
        raise ValueError(f"Unsupported recent snapshot benchmark model: {model_name}")

    model.fit(x_train, y_train)
    snapshot_df["y_pred"] = model.predict(x_snap)

    portfolio_metrics = _load_frame(paths.reports_dir / "portfolio_benchmark_v3_metrics.csv")
    config_row = portfolio_metrics.loc[
        portfolio_metrics["strategy_label"].eq("best_60_predictor") & portfolio_metrics["split"].eq("validation")
    ]
    if config_row.empty:
        config_row = portfolio_metrics.loc[
            portfolio_metrics["strategy_label"].eq("best_60_predictor") & portfolio_metrics["split"].eq("test")
        ]
    if config_row.empty:
        raise KeyError("best_60_predictor allocator settings not found in portfolio benchmark metrics")
    config_row = config_row.iloc[0]
    optimizer_config = OptimizerConfig(
        lambda_risk=float(config_row["selected_lambda_risk"]),
        kappa=float(config_row["selected_kappa"]),
        omega_type=str(config_row["selected_omega_type"]),
    )

    monthly_excess_history = load_modeling_inputs(paths.project_root, feature_set_name="core_baseline").monthly_excess_history
    sigma_map = build_sigma_map(
        list(anchor_months),
        excess_history=monthly_excess_history,
        risk_config=RiskConfig(
            lookback_months=60,
            min_months=12,
            ewma_beta=0.94,
            diagonal_shrinkage=0.10,
            ridge=1e-6,
        ),
    )
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    weight_rows: list[dict[str, object]] = []
    for month_end, chunk in snapshot_df.groupby("month_end", sort=True):
        ordered = chunk.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
        mu = torch.tensor(ordered["y_pred"].to_numpy(dtype=np.float32), dtype=torch.float32)
        weights = optimizer_cache.solve(pd.Timestamp(month_end), mu, optimizer_config).detach().cpu().numpy()
        for sleeve_id, weight, pred in zip(
            ordered["sleeve_id"].tolist(),
            weights,
            ordered["y_pred"].to_numpy(dtype=float),
            strict=True,
        ):
            weight_rows.append(
                {
                    "month_end": pd.Timestamp(month_end),
                    "sleeve_id": sleeve_id,
                    "portfolio_weight": float(weight),
                    "predicted_annualized_excess_return": float(pred),
                }
            )

    prediction_frame = snapshot_df[["month_end", "sleeve_id", "y_pred"]].rename(
        columns={"y_pred": "predicted_annualized_excess_return"}
    )
    weight_frame = pd.DataFrame(weight_rows).sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)
    fit_note = (
        "Benchmark refit on all labeled 60m history with locked hyperparameters, then applied to recent unlabeled feature dates."
    )
    return RecentSnapshotBundle(
        prediction_frame=prediction_frame.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True),
        weight_frame=weight_frame,
        fit_note=fit_note,
    )


def load_plot_context(project_root: Path) -> PlotContext:
    paths = default_paths(project_root)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)

    benchmark_stack = _load_frame(paths.reports_dir / "final_benchmark_stack_v3.csv")
    portfolio_metrics = _load_frame(paths.reports_dir / "portfolio_benchmark_v3_metrics.csv")
    portfolio_by_sleeve = _load_frame(paths.reports_dir / "portfolio_benchmark_v3_by_sleeve.csv")
    portfolio_attribution = _load_frame(paths.reports_dir / "portfolio_benchmark_v3_attribution.csv")
    china_portfolio = _load_frame(paths.reports_dir / "china_portfolio_diagnostics_v3.csv")
    prediction_metrics = _load_frame(paths.reports_dir / "prediction_benchmark_v3_metrics.csv")
    prediction_by_sleeve = _load_frame(paths.reports_dir / "prediction_benchmark_v3_by_sleeve.csv")
    china_prediction = _load_frame(paths.reports_dir / "china_prediction_diagnostics_v3.csv")
    predictions_test = _load_frame(paths.data_dir / "predictions_test_horse_race.parquet")
    portfolio_returns = _load_frame(paths.data_dir / "portfolio_benchmark_v3_returns.parquet")

    refinement_path = paths.reports_dir / "benchmark_refinement_v3_concentration.csv"
    refinement_concentration = _load_frame(refinement_path) if refinement_path.exists() else portfolio_metrics.copy()

    active_60_prediction = _stack_value(benchmark_stack, "strongest_60m_prediction_benchmark")
    active_120_prediction = _stack_value(benchmark_stack, "strongest_raw_prediction_benchmark")
    raw_portfolio_benchmark = _stack_value(benchmark_stack, "strongest_raw_portfolio_benchmark")
    robust_portfolio_benchmark = _stack_value(benchmark_stack, "strongest_robust_portfolio_benchmark")

    for frame_name in [
        "portfolio_metrics",
        "portfolio_by_sleeve",
        "portfolio_attribution",
        "china_portfolio",
        "prediction_by_sleeve",
        "china_prediction",
        "predictions_test",
        "portfolio_returns",
        "refinement_concentration",
    ]:
        frame = locals()[frame_name]
        if "month_end" in frame.columns:
            frame["month_end"] = pd.to_datetime(frame["month_end"])

    sleeve_wealth = _load_sleeve_wealth(paths)
    recent_5y_snapshot = _fit_recent_active_60_snapshot(
        paths=paths,
        benchmark_stack=benchmark_stack,
        prediction_metrics=prediction_metrics,
    )
    annual_rebalance_5y_path = _build_annual_rebalance_5y_path(
        paths=paths,
        benchmark_stack=benchmark_stack,
        prediction_metrics=prediction_metrics,
    )

    return PlotContext(
        paths=paths,
        benchmark_stack=benchmark_stack,
        portfolio_metrics=portfolio_metrics,
        portfolio_by_sleeve=portfolio_by_sleeve,
        portfolio_attribution=portfolio_attribution,
        china_portfolio=china_portfolio,
        prediction_metrics=prediction_metrics,
        prediction_by_sleeve=prediction_by_sleeve,
        china_prediction=china_prediction,
        predictions_test=predictions_test,
        portfolio_returns=portfolio_returns,
        refinement_concentration=refinement_concentration,
        active_60_prediction=active_60_prediction,
        active_120_prediction=active_120_prediction,
        raw_portfolio_benchmark=raw_portfolio_benchmark,
        robust_portfolio_benchmark=robust_portfolio_benchmark,
        sleeve_order=tuple(SLEEVE_ORDER),
        sleeve_wealth=sleeve_wealth,
        recent_5y_snapshot=recent_5y_snapshot,
        annual_rebalance_5y_path=annual_rebalance_5y_path,
    )
