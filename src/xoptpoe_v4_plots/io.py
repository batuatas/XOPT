from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from xoptpoe_data.targets.build_monthly_targets import build_monthly_realized_returns, collapse_target_to_month_end_prices
from xoptpoe_v3_models.preprocess import fit_preprocessor
from xoptpoe_v4_modeling.features import feature_columns_for_set
from xoptpoe_v4_modeling.io import load_csv, load_parquet, write_parquet
from xoptpoe_v4_models.data import SLEEVE_ORDER, load_modeling_inputs
from xoptpoe_v4_models.optim_layers import OptimizerConfig, RiskConfig, RobustOptimizerCache, build_sigma_map


BEST_60_EXPERIMENT = "elastic_net__core_plus_interactions__separate_60"
BEST_60_TUNED_LABEL = "best_60_tuned_robust"
BEST_60_RAW_LABEL = "best_60_predictor"
TUNED_OPTIMIZER_CONFIG = OptimizerConfig(lambda_risk=8.0, kappa=0.10, omega_type="identity")
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
    modeling_dir: Path
    final_dir: Path
    reports_root: Path


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
    prediction_metrics: pd.DataFrame
    prediction_by_sleeve: pd.DataFrame
    predictions_test: pd.DataFrame
    tuned_metrics: pd.DataFrame
    tuned_by_sleeve: pd.DataFrame
    tuned_weights: pd.DataFrame
    tuned_returns: pd.DataFrame
    diagnostic_predictions_panel: pd.DataFrame
    diagnostic_weights_panel: pd.DataFrame
    diagnostic_returns_panel: pd.DataFrame
    recent_5y_snapshot: RecentSnapshotBundle
    annual_rebalance_5y_path: AnnualWealthPathBundle


def default_paths(project_root: Path) -> PlotPaths:
    root = project_root.resolve()
    return PlotPaths(
        project_root=root,
        reports_dir=root / "reports" / "v4_expanded_universe",
        plots_dir=root / "reports" / "v4_expanded_universe" / "plots",
        modeling_dir=root / "data" / "modeling_v4",
        final_dir=root / "data" / "final_v4_expanded_universe",
        reports_root=root / "reports",
    )


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
    recent_2026 = [m for m in months if m.year == 2026]
    if recent_2026:
        latest_2026 = recent_2026[-1]
        if latest_2026 not in resolved:
            resolved.append(latest_2026)
    return tuple(resolved)


def _load_total_return_panel(paths: PlotPaths) -> pd.DataFrame:
    direct_raw = load_csv(paths.final_dir / "target_raw_direct.csv", parse_dates=["trade_date"])
    euro_synth = load_csv(paths.final_dir / "euro_fixed_income_month_end_usd_synth.csv", parse_dates=["month_end", "trade_date"])
    direct_month_end = collapse_target_to_month_end_prices(direct_raw)
    euro_month_end = euro_synth[["sleeve_id", "ticker", "month_end", "trade_date", "adj_close", "close"]].copy()
    prices = pd.concat([direct_month_end, euro_month_end], ignore_index=True)
    prices = prices.loc[prices["sleeve_id"].isin(SLEEVE_ORDER)].copy()
    prices["month_end"] = pd.to_datetime(prices["month_end"])
    monthly_returns = build_monthly_realized_returns(prices)
    panel = (
        monthly_returns.pivot(index="month_end", columns="sleeve_id", values="ret_1m_realized")
        .sort_index()
        .reindex(columns=list(SLEEVE_ORDER))
    )
    if panel.columns.isna().any():
        raise ValueError("total return panel could not be aligned to the active v4 sleeve order")
    return panel.astype(float)


def _best_60_prediction_frames(paths: PlotPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_val = load_parquet(paths.modeling_dir / "predictions_validation_v4_benchmark.parquet")
    pred_test = load_parquet(paths.modeling_dir / "predictions_test_v4_benchmark.parquet")
    keep_val = pred_val["experiment_name"].eq(BEST_60_EXPERIMENT) & pred_val["horizon_mode"].eq("separate_60") & pred_val["horizon_months"].eq(60)
    keep_test = pred_test["experiment_name"].eq(BEST_60_EXPERIMENT) & pred_test["horizon_mode"].eq("separate_60") & pred_test["horizon_months"].eq(60)
    val = pred_val.loc[keep_val].copy()
    test = pred_test.loc[keep_test].copy()
    for frame in (val, test):
        frame["month_end"] = pd.to_datetime(frame["month_end"])
        frame["split"] = frame["split"].astype(str)
    return val.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True), test.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)


def _optimizer_cache(paths: PlotPaths, months: list[pd.Timestamp]) -> RobustOptimizerCache:
    monthly_excess_history = load_modeling_inputs(paths.project_root, feature_set_name="core_baseline").monthly_excess_history
    if monthly_excess_history is None:
        raise RuntimeError("monthly_excess_history missing for tuned v4 plotting context")
    sigma_map = build_sigma_map(months, excess_history=monthly_excess_history, risk_config=RiskConfig())
    return RobustOptimizerCache(sigma_by_month=sigma_map)


def _normalize(weights: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.repeat(1.0 / len(arr), len(arr))
    return arr / total


def _summarize_strategy(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    concentration = (
        weights.groupby(["strategy_label", "split", "month_end"], as_index=False)["weight"]
        .agg(max_weight="max", hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
        .sort_values(["strategy_label", "split", "month_end"])
    )
    concentration["effective_n_sleeves"] = 1.0 / concentration["hhi"]
    returns = returns.merge(concentration, on=["strategy_label", "split", "month_end"], how="left", validate="1:1")
    eq_ref = returns.loc[returns["strategy_label"].eq("equal_weight"), ["split", "month_end", "portfolio_annualized_excess_return"]].rename(
        columns={"portfolio_annualized_excess_return": "ref_equal_weight"}
    )
    returns = returns.merge(eq_ref, on=["split", "month_end"], how="left", validate="m:1")
    returns["active_return_vs_equal_weight"] = returns["portfolio_annualized_excess_return"] - returns["ref_equal_weight"]

    rows: list[dict[str, object]] = []
    for (strategy_label, split_name), chunk in returns.groupby(["strategy_label", "split"]):
        weights_chunk = weights.loc[(weights["strategy_label"].eq(strategy_label)) & (weights["split"].eq(split_name))]
        top_freq = weights_chunk.loc[weights_chunk["top_weight_flag"].eq(1)].groupby("sleeve_id").size().sort_values(ascending=False)
        active = (
            weights_chunk.groupby("sleeve_id", as_index=False)["active_contribution_vs_equal_weight"]
            .sum()
            .assign(abs_active=lambda x: np.abs(x["active_contribution_vs_equal_weight"]))
            .sort_values("abs_active", ascending=False)
        )
        top2 = float(active.head(2)["abs_active"].sum() / active["abs_active"].sum()) if float(active["abs_active"].sum()) > 0 else np.nan
        rows.append(
            {
                "strategy_label": strategy_label,
                "split": split_name,
                "month_count": int(chunk["month_end"].nunique()),
                "avg_return": float(chunk["portfolio_annualized_excess_return"].mean()),
                "volatility": float(chunk["portfolio_annualized_excess_return"].std(ddof=1)),
                "sharpe": float(chunk["portfolio_annualized_excess_return"].mean() / chunk["portfolio_annualized_excess_return"].std(ddof=1)),
                "max_drawdown": float(chunk["drawdown"].min()),
                "avg_turnover": float(chunk["turnover"].mean()),
                "avg_max_weight": float(chunk["max_weight"].mean()),
                "avg_hhi": float(chunk["hhi"].mean()),
                "avg_effective_n_sleeves": float(chunk["effective_n_sleeves"].mean()),
                "avg_active_return_vs_equal_weight": float(chunk["active_return_vs_equal_weight"].mean()),
                "top_weight_sleeve": str(top_freq.index[0]) if len(top_freq) else None,
                "top_weight_sleeve_frequency": float(top_freq.iloc[0] / chunk["month_end"].nunique()) if len(top_freq) else np.nan,
                "top2_sleeve_active_share_abs": top2,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "strategy_label"]).reset_index(drop=True)


def _by_sleeve(weights: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (strategy_label, split_name, sleeve_id), chunk in weights.groupby(["strategy_label", "split", "sleeve_id"]):
        total_active = float(chunk["active_contribution_vs_equal_weight"].sum())
        rows.append(
            {
                "strategy_label": strategy_label,
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
    out = pd.DataFrame(rows)
    for (_, _), idx in out.groupby(["strategy_label", "split"]).groups.items():
        denom = float(out.loc[idx, "abs_total_active_contribution"].sum())
        out.loc[idx, "abs_active_contribution_share"] = out.loc[idx, "abs_total_active_contribution"] / denom if denom > 0 else np.nan
    return out.sort_values(["strategy_label", "split", "abs_total_active_contribution"], ascending=[True, True, False]).reset_index(drop=True)


def _run_tuned_benchmark(validation_pred: pd.DataFrame, test_pred: pd.DataFrame, *, paths: PlotPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(pd.concat([validation_pred["month_end"], test_pred["month_end"]], ignore_index=True).drop_duplicates().tolist())
    optimizer_cache = _optimizer_cache(paths, months)

    weight_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    for split_name, frame in (("validation", validation_pred), ("test", test_pred)):
        prev_weights: np.ndarray | None = None
        for month_end, chunk in frame.groupby("month_end", sort=True):
            ordered = chunk.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
            scores = ordered["y_pred"].to_numpy(dtype=float)
            weights = _normalize(optimizer_cache.solve(pd.Timestamp(month_end), scores, TUNED_OPTIMIZER_CONFIG))
            realized = ordered["y_true"].to_numpy(dtype=float)
            ret = float(np.dot(weights, realized))
            turnover = 0.0 if prev_weights is None else float(0.5 * np.abs(weights - prev_weights).sum())
            prev_weights = weights.copy()
            return_rows.append(
                {
                    "split": split_name,
                    "month_end": pd.Timestamp(month_end),
                    "strategy_label": BEST_60_TUNED_LABEL,
                    "portfolio_annualized_excess_return": ret,
                    "turnover": turnover,
                }
            )
            return_rows.append(
                {
                    "split": split_name,
                    "month_end": pd.Timestamp(month_end),
                    "strategy_label": "equal_weight",
                    "portfolio_annualized_excess_return": float(realized.mean()),
                    "turnover": 0.0,
                }
            )
            for sleeve_id, weight, pred, truth in zip(
                ordered["sleeve_id"].tolist(),
                weights,
                ordered["y_pred"].to_numpy(dtype=float),
                realized,
            ):
                weight_rows.append(
                    {
                        "split": split_name,
                        "month_end": pd.Timestamp(month_end),
                        "strategy_label": BEST_60_TUNED_LABEL,
                        "sleeve_id": sleeve_id,
                        "weight": float(weight),
                        "predicted_signal": float(pred),
                        "realized_outcome": float(truth),
                    }
                )
                weight_rows.append(
                    {
                        "split": split_name,
                        "month_end": pd.Timestamp(month_end),
                        "strategy_label": "equal_weight",
                        "sleeve_id": sleeve_id,
                        "weight": float(1.0 / len(SLEEVE_ORDER)),
                        "predicted_signal": np.nan,
                        "realized_outcome": float(truth),
                    }
                )

    returns = pd.DataFrame(return_rows).sort_values(["strategy_label", "split", "month_end"]).reset_index(drop=True)
    returns["gross_return"] = np.maximum(1.0 + returns["portfolio_annualized_excess_return"], 1e-6)
    returns["cum_nav"] = returns.groupby(["strategy_label", "split"])["gross_return"].cumprod()
    returns["running_peak"] = returns.groupby(["strategy_label", "split"])["cum_nav"].cummax()
    returns["drawdown"] = returns["cum_nav"] / returns["running_peak"] - 1.0
    returns = returns.drop(columns=["running_peak"])

    weights = pd.DataFrame(weight_rows).sort_values(["strategy_label", "split", "month_end", "sleeve_id"]).reset_index(drop=True)
    eq_ref = weights.loc[weights["strategy_label"].eq("equal_weight"), ["split", "month_end", "sleeve_id", "weight"]].rename(columns={"weight": "equal_weight_ref"})
    weights = weights.merge(eq_ref, on=["split", "month_end", "sleeve_id"], how="left", validate="m:1")
    weights["active_weight_vs_equal_weight"] = weights["weight"] - weights["equal_weight_ref"]
    weights["sleeve_contribution"] = weights["weight"] * weights["realized_outcome"]
    weights["active_contribution_vs_equal_weight"] = weights["active_weight_vs_equal_weight"] * weights["realized_outcome"]
    weights["top_weight_flag"] = (
        weights.groupby(["strategy_label", "split", "month_end"])["weight"].rank(method="first", ascending=False).eq(1)
    ).astype(int)
    weights["nonzero_alloc_flag"] = (weights["weight"] > 1e-10).astype(int)
    return returns, weights, _summarize_strategy(weights, returns)


def _fit_best60_model_scores(
    *,
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, object],
) -> pd.DataFrame:
    preprocessor = fit_preprocessor(train_df, feature_manifest=feature_manifest, feature_columns=feature_columns)
    x_train, _ = preprocessor.transform(train_df)
    x_score, _ = preprocessor.transform(score_df)
    y_train = train_df["annualized_excess_forward_return"].to_numpy(dtype=float)
    model = ElasticNet(alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), fit_intercept=True, max_iter=20000, random_state=42)
    model.fit(x_train, y_train)
    ordered = score_df.copy()
    ordered["y_pred"] = model.predict(x_score)
    return ordered


def _fit_recent_active_60_snapshot(paths: PlotPaths, prediction_metrics: pd.DataFrame) -> RecentSnapshotBundle:
    metrics_row = prediction_metrics.loc[prediction_metrics["experiment_name"].eq(BEST_60_EXPERIMENT)].iloc[0]
    params = _parse_selected_params(metrics_row["selected_params"])
    full_panel = load_parquet(paths.final_dir / "modeling_panel_hstack.parquet")
    full_panel["month_end"] = pd.to_datetime(full_panel["month_end"])
    full_panel = full_panel.loc[full_panel["sleeve_id"].isin(SLEEVE_ORDER)].copy()
    feature_manifest = load_csv(paths.modeling_dir / "feature_set_manifest.csv", parse_dates=["first_valid_date", "last_valid_date"])
    feature_columns = feature_columns_for_set(feature_manifest, "core_plus_interactions")

    train_df = full_panel.loc[
        full_panel["horizon_months"].eq(60)
        & full_panel["baseline_trainable_flag"].eq(1)
        & full_panel["target_available_flag"].eq(1)
        & full_panel["annualized_excess_forward_return"].notna()
    ].copy()
    anchor_months = _resolve_anchor_months(full_panel.loc[full_panel["horizon_months"].eq(60), "month_end"])
    snapshot_df = full_panel.loc[full_panel["horizon_months"].eq(60) & full_panel["month_end"].isin(anchor_months)].copy()
    snapshot_df = snapshot_df.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)
    expected_rows = len(anchor_months) * len(SLEEVE_ORDER)
    if len(snapshot_df) != expected_rows:
        raise ValueError(f"Recent snapshot rows incomplete: expected {expected_rows}, observed {len(snapshot_df)}")

    scored = _fit_best60_model_scores(
        train_df=train_df,
        score_df=snapshot_df,
        feature_manifest=feature_manifest,
        feature_columns=feature_columns,
        params=params,
    )
    optimizer_cache = _optimizer_cache(paths, list(anchor_months))

    weight_rows: list[dict[str, object]] = []
    for month_end, chunk in scored.groupby("month_end", sort=True):
        ordered = chunk.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
        weights = _normalize(optimizer_cache.solve(pd.Timestamp(month_end), ordered["y_pred"].to_numpy(dtype=float), TUNED_OPTIMIZER_CONFIG))
        for sleeve_id, weight, pred in zip(
            ordered["sleeve_id"].tolist(),
            weights,
            ordered["y_pred"].to_numpy(dtype=float),
        ):
            weight_rows.append(
                {
                    "month_end": pd.Timestamp(month_end),
                    "sleeve_id": sleeve_id,
                    "portfolio_weight": float(weight),
                    "predicted_annualized_excess_return": float(pred),
                }
            )

    fit_note = "Benchmark refit on all labeled 60m history with locked hyperparameters, then mapped through the tuned robust v4 allocator."
    return RecentSnapshotBundle(
        prediction_frame=scored[["month_end", "sleeve_id", "y_pred"]].rename(columns={"y_pred": "predicted_annualized_excess_return"}).sort_values(["month_end", "sleeve_id"]).reset_index(drop=True),
        weight_frame=pd.DataFrame(weight_rows).sort_values(["month_end", "sleeve_id"]).reset_index(drop=True),
        fit_note=fit_note,
    )


def _build_annual_rebalance_5y_path(paths: PlotPaths, prediction_metrics: pd.DataFrame) -> AnnualWealthPathBundle:
    metrics_row = prediction_metrics.loc[prediction_metrics["experiment_name"].eq(BEST_60_EXPERIMENT)].iloc[0]
    params = _parse_selected_params(metrics_row["selected_params"])
    full_panel = load_parquet(paths.final_dir / "modeling_panel_hstack.parquet")
    full_panel["month_end"] = pd.to_datetime(full_panel["month_end"])
    full_panel = full_panel.loc[full_panel["sleeve_id"].isin(SLEEVE_ORDER)].copy()
    feature_manifest = load_csv(paths.modeling_dir / "feature_set_manifest.csv", parse_dates=["first_valid_date", "last_valid_date"])
    feature_columns = feature_columns_for_set(feature_manifest, "core_plus_interactions")
    total_return_panel = _load_total_return_panel(paths)

    train_pool = full_panel.loc[
        full_panel["horizon_months"].eq(60)
        & full_panel["baseline_trainable_flag"].eq(1)
        & full_panel["target_available_flag"].eq(1)
        & full_panel["annualized_excess_forward_return"].notna()
    ].copy()
    score_pool = full_panel.loc[full_panel["horizon_months"].eq(60)].copy()

    valid_anchor_months: list[pd.Timestamp] = []
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

    optimizer_cache = _optimizer_cache(paths, list(anchor_months))

    path_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    nav_by_strategy = {BEST_60_TUNED_LABEL: 1.0, "equal_weight": 1.0}

    for anchor in anchor_months:
        cutoff = pd.Timestamp(anchor) - pd.offsets.MonthEnd(60)
        train_df = train_pool.loc[train_pool["month_end"].le(cutoff)].copy()
        if len(train_df) < len(SLEEVE_ORDER) * 12:
            continue
        score_df = score_pool.loc[score_pool["month_end"].eq(anchor)].copy()
        ordered = score_df.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
        ordered = _fit_best60_model_scores(
            train_df=train_df,
            score_df=ordered,
            feature_manifest=feature_manifest,
            feature_columns=feature_columns,
            params=params,
        )
        weights = _normalize(optimizer_cache.solve(anchor, ordered["y_pred"].to_numpy(dtype=float), TUNED_OPTIMIZER_CONFIG))

        anchor_rows.extend(
            [
                {"anchor_month_end": pd.Timestamp(anchor), "strategy_label": BEST_60_TUNED_LABEL, "wealth_index": float(nav_by_strategy[BEST_60_TUNED_LABEL])},
                {"anchor_month_end": pd.Timestamp(anchor), "strategy_label": "equal_weight", "wealth_index": float(nav_by_strategy["equal_weight"])},
            ]
        )
        for sleeve_id, pred, weight in zip(ordered["sleeve_id"].tolist(), ordered["y_pred"].to_numpy(dtype=float), weights):
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
        hold_panel = total_return_panel.loc[hold_months, list(SLEEVE_ORDER)].copy().dropna(how="any")
        for month_end, row in hold_panel.iterrows():
            port_ret = float(np.dot(weights, row.to_numpy(dtype=float)))
            eq_ret = float(row.mean())
            nav_by_strategy[BEST_60_TUNED_LABEL] *= 1.0 + port_ret
            nav_by_strategy["equal_weight"] *= 1.0 + eq_ret
            path_rows.extend(
                [
                    {
                        "month_end": pd.Timestamp(month_end),
                        "strategy_label": BEST_60_TUNED_LABEL,
                        "ret_1m_realized": port_ret,
                        "wealth_index": float(nav_by_strategy[BEST_60_TUNED_LABEL]),
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

    fit_note = (
        "Annual walk-forward illustration: each December, refit the locked 60m benchmark on then-observable labeled history, "
        "solve the tuned robust v4 allocator, and hold through the next calendar year of realized sleeve returns."
    )
    return AnnualWealthPathBundle(
        path_frame=pd.DataFrame(path_rows).sort_values(["strategy_label", "month_end"]).reset_index(drop=True),
        weight_frame=pd.DataFrame(weight_rows).sort_values(["anchor_month_end", "sleeve_id"]).reset_index(drop=True),
        anchor_frame=pd.DataFrame(anchor_rows).sort_values(["strategy_label", "anchor_month_end"]).reset_index(drop=True),
        fit_note=fit_note,
    )


def load_plot_context(project_root: Path) -> PlotContext:
    paths = default_paths(project_root)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    prediction_metrics = load_csv(paths.reports_root / "v4_prediction_benchmark_metrics.csv")
    prediction_by_sleeve = load_csv(paths.reports_root / "v4_prediction_benchmark_by_sleeve.csv")
    predictions_test = load_parquet(paths.modeling_dir / "predictions_test_v4_benchmark.parquet")

    best60_val, best60_test = _best_60_prediction_frames(paths)
    tuned_returns, tuned_weights, tuned_metrics = _run_tuned_benchmark(best60_val, best60_test, paths=paths)
    tuned_by_sleeve = _by_sleeve(tuned_weights)
    recent_5y_snapshot = _fit_recent_active_60_snapshot(paths, prediction_metrics)
    annual_rebalance_5y_path = _build_annual_rebalance_5y_path(paths, prediction_metrics)

    diagnostic_predictions_panel = pd.concat(
        [
            best60_test.assign(panel_type="historical_test_60"),
            recent_5y_snapshot.prediction_frame.assign(panel_type="recent_snapshot_5y"),
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostic_weights_panel = pd.concat(
        [
            tuned_weights.assign(panel_type="historical_portfolio_eval"),
            recent_5y_snapshot.weight_frame.assign(strategy_label=BEST_60_TUNED_LABEL, split="recent_snapshot", panel_type="recent_snapshot_5y"),
            annual_rebalance_5y_path.weight_frame.assign(strategy_label=BEST_60_TUNED_LABEL, split="annual_walkforward", panel_type="annual_walkforward"),
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostic_returns_panel = pd.concat(
        [
            tuned_returns.assign(panel_type="historical_portfolio_eval"),
            annual_rebalance_5y_path.path_frame.assign(panel_type="annual_walkforward"),
        ],
        ignore_index=True,
        sort=False,
    )

    for frame in (diagnostic_predictions_panel, diagnostic_weights_panel, diagnostic_returns_panel):
        if "month_end" in frame.columns:
            frame["month_end"] = pd.to_datetime(frame["month_end"], errors="coerce")
        if "anchor_month_end" in frame.columns:
            frame["anchor_month_end"] = pd.to_datetime(frame["anchor_month_end"], errors="coerce")

    write_parquet(diagnostic_predictions_panel, paths.reports_dir / "diagnostic_predictions_panel_v4.parquet")
    write_parquet(diagnostic_weights_panel, paths.reports_dir / "diagnostic_weights_panel_v4.parquet")
    write_parquet(diagnostic_returns_panel, paths.reports_dir / "diagnostic_returns_panel_v4.parquet")

    return PlotContext(
        paths=paths,
        prediction_metrics=prediction_metrics,
        prediction_by_sleeve=prediction_by_sleeve,
        predictions_test=predictions_test,
        tuned_metrics=tuned_metrics,
        tuned_by_sleeve=tuned_by_sleeve,
        tuned_weights=tuned_weights,
        tuned_returns=tuned_returns,
        diagnostic_predictions_panel=diagnostic_predictions_panel,
        diagnostic_weights_panel=diagnostic_weights_panel,
        diagnostic_returns_panel=diagnostic_returns_panel,
        recent_5y_snapshot=recent_5y_snapshot,
        annual_rebalance_5y_path=annual_rebalance_5y_path,
    )
