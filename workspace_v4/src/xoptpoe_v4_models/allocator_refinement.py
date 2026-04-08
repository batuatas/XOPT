"""Second-stage allocator refinement for the fixed v4 best-60 predictor."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_v4_modeling.features import feature_columns_for_set
from xoptpoe_v4_modeling.io import load_csv, load_parquet, write_csv, write_text
from xoptpoe_v4_models.data import SLEEVE_ORDER, load_modeling_inputs
from xoptpoe_v4_models.optim_layers import OptimizerConfig, RiskConfig, RobustOptimizerCache, build_sigma_map
from xoptpoe_v4_models.portfolio_benchmark import _signal_maps
from xoptpoe_v4_models.portfolio_eval import run_portfolio_evaluation, summarize_portfolio_metrics
from xoptpoe_v4_plots.io import BEST_60_EXPERIMENT, _fit_best60_model_scores, _load_total_return_panel, default_paths as plot_default_paths
BEST_60_LABEL = "best_60_predictor"


@dataclass(frozen=True)
class AllocatorRefinementOutputs:
    results: pd.DataFrame
    weights_summary: pd.DataFrame
    attribution_summary: pd.DataFrame
    wealth_paths: pd.DataFrame
    report_text: str


def allocator_refinement_grid() -> list[OptimizerConfig]:
    return [
        OptimizerConfig(lambda_risk=10.0, kappa=0.10, omega_type="diag"),
        OptimizerConfig(lambda_risk=10.0, kappa=0.15, omega_type="diag"),
        OptimizerConfig(lambda_risk=10.0, kappa=0.20, omega_type="diag"),
        OptimizerConfig(lambda_risk=12.0, kappa=0.15, omega_type="diag"),
        OptimizerConfig(lambda_risk=12.0, kappa=0.20, omega_type="diag"),
        OptimizerConfig(lambda_risk=15.0, kappa=0.15, omega_type="diag"),
        OptimizerConfig(lambda_risk=15.0, kappa=0.20, omega_type="diag"),
        OptimizerConfig(lambda_risk=15.0, kappa=0.25, omega_type="diag"),
        OptimizerConfig(lambda_risk=20.0, kappa=0.20, omega_type="diag"),
        OptimizerConfig(lambda_risk=8.0, kappa=0.05, omega_type="identity"),
        OptimizerConfig(lambda_risk=8.0, kappa=0.10, omega_type="identity"),
        OptimizerConfig(lambda_risk=10.0, kappa=0.05, omega_type="identity"),
        OptimizerConfig(lambda_risk=10.0, kappa=0.10, omega_type="identity"),
        OptimizerConfig(lambda_risk=12.0, kappa=0.10, omega_type="identity"),
        OptimizerConfig(lambda_risk=12.0, kappa=0.15, omega_type="identity"),
    ]


def _config_label(config: OptimizerConfig) -> str:
    lam = int(config.lambda_risk) if float(config.lambda_risk).is_integer() else config.lambda_risk
    return f"lam{lam:g}_kap{config.kappa:g}_{config.omega_type}"


def _normalize(weights: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.repeat(1.0 / len(arr), len(arr))
    return arr / total


def _split_monthly_concentration(weights: pd.DataFrame) -> pd.DataFrame:
    out = (
        weights.groupby("month_end", as_index=False)["weight"]
        .agg(max_weight="max", hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
        .sort_values("month_end")
        .reset_index(drop=True)
    )
    out["effective_n_sleeves"] = 1.0 / out["hhi"]
    return out


def _evaluate_test_panel(
    *,
    signal_panel: pd.DataFrame,
    optimizer_cache: RobustOptimizerCache,
    optimizer_config: OptimizerConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    run = run_portfolio_evaluation(
        signal_panel=signal_panel,
        optimizer_cache=optimizer_cache,
        optimizer_config=optimizer_config,
        model_strategy_name=BEST_60_LABEL,
    )
    metrics = summarize_portfolio_metrics(run.returns)
    model_metric = metrics.loc[metrics["strategy"].eq(BEST_60_LABEL)].iloc[0]
    model_returns = run.returns.loc[run.returns["strategy"].eq(BEST_60_LABEL)].copy().reset_index(drop=True)
    eq_returns = (
        run.returns.loc[run.returns["strategy"].eq("equal_weight"), ["month_end", "portfolio_annualized_excess_return"]]
        .rename(columns={"portfolio_annualized_excess_return": "equal_weight_return"})
        .reset_index(drop=True)
    )
    model_returns = model_returns.merge(eq_returns, on="month_end", how="left", validate="1:1")
    model_returns["active_return_vs_equal_weight"] = (
        model_returns["portfolio_annualized_excess_return"] - model_returns["equal_weight_return"]
    )

    weights = run.weights.loc[run.weights["strategy"].eq(BEST_60_LABEL)].copy().reset_index(drop=True)
    truth = signal_panel[["month_end", "sleeve_id", "y_true"]].drop_duplicates().rename(columns={"y_true": "realized_outcome"})
    weights = weights.merge(truth, on=["month_end", "sleeve_id"], how="left", validate="1:1")
    weights["equal_weight_ref"] = 1.0 / len(SLEEVE_ORDER)
    weights["active_weight_vs_equal_weight"] = weights["weight"] - weights["equal_weight_ref"]
    weights["sleeve_contribution"] = weights["weight"] * weights["realized_outcome"]
    weights["active_contribution_vs_equal_weight"] = weights["active_weight_vs_equal_weight"] * weights["realized_outcome"]
    weights["top_weight_flag"] = (
        weights.groupby("month_end")["weight"].rank(method="first", ascending=False).eq(1)
    ).astype(int)
    weights["nonzero_alloc_flag"] = (weights["weight"] > 1e-10).astype(int)

    concentration = _split_monthly_concentration(weights)
    top_weight_counts = (
        weights.loc[weights["top_weight_flag"].eq(1)]
        .groupby("sleeve_id", as_index=False)
        .size()
        .sort_values(["size", "sleeve_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    top_weight_sleeve = str(top_weight_counts.iloc[0]["sleeve_id"])
    top_weight_frequency = float(top_weight_counts.iloc[0]["size"] / len(concentration))

    sleeve_active = (
        weights.groupby("sleeve_id", as_index=False)["active_contribution_vs_equal_weight"]
        .sum()
        .assign(abs_active=lambda x: np.abs(x["active_contribution_vs_equal_weight"]))
        .sort_values(["abs_active", "sleeve_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    abs_total = float(sleeve_active["abs_active"].sum())
    top_sleeve_active_contribution_share_abs = float(sleeve_active.iloc[0]["abs_active"] / abs_total) if abs_total > 0 else float("nan")

    summary_row = {
        "config_label": _config_label(optimizer_config),
        "lambda_risk": float(optimizer_config.lambda_risk),
        "kappa": float(optimizer_config.kappa),
        "omega_type": str(optimizer_config.omega_type),
        "avg_return": float(model_metric["avg_return"]),
        "volatility": float(model_metric["volatility"]),
        "sharpe": float(model_metric["sharpe"]),
        "avg_turnover": float(model_metric["avg_turnover"]),
        "avg_active_return_vs_equal_weight": float(model_returns["active_return_vs_equal_weight"].mean()),
        "avg_max_weight": float(concentration["max_weight"].mean()),
        "avg_effective_n_sleeves": float(concentration["effective_n_sleeves"].mean()),
        "top_weight_sleeve": top_weight_sleeve,
        "top_weight_sleeve_frequency": top_weight_frequency,
        "top_sleeve_active_contribution_share_abs": top_sleeve_active_contribution_share_abs,
    }
    return model_returns, weights, summary_row


def _weights_summary(weights: pd.DataFrame, *, config: OptimizerConfig) -> pd.DataFrame:
    out = (
        weights.groupby("sleeve_id", as_index=False)
        .agg(
            avg_weight=("weight", "mean"),
            max_weight_observed=("weight", "max"),
            top_weight_frequency=("top_weight_flag", "mean"),
            nonzero_allocation_share=("nonzero_alloc_flag", "mean"),
            avg_active_weight_vs_equal_weight=("active_weight_vs_equal_weight", "mean"),
            avg_realized_outcome=("realized_outcome", "mean"),
        )
        .sort_values(["avg_weight", "sleeve_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out.insert(0, "config_label", _config_label(config))
    out.insert(0, "omega_type", config.omega_type)
    out.insert(0, "kappa", float(config.kappa))
    out.insert(0, "lambda_risk", float(config.lambda_risk))
    return out


def _attribution_summary(weights: pd.DataFrame, *, config: OptimizerConfig) -> pd.DataFrame:
    out = (
        weights.groupby("sleeve_id", as_index=False)
        .agg(
            total_contribution=("sleeve_contribution", "sum"),
            total_active_contribution_vs_equal_weight=("active_contribution_vs_equal_weight", "sum"),
            avg_monthly_active_contribution_vs_equal_weight=("active_contribution_vs_equal_weight", "mean"),
        )
        .assign(abs_total_active_contribution=lambda x: np.abs(x["total_active_contribution_vs_equal_weight"]))
        .sort_values(["abs_total_active_contribution", "sleeve_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    abs_total = float(out["abs_total_active_contribution"].sum())
    out["abs_active_contribution_share"] = out["abs_total_active_contribution"] / abs_total if abs_total > 0 else np.nan
    out.insert(0, "config_label", _config_label(config))
    out.insert(0, "omega_type", config.omega_type)
    out.insert(0, "kappa", float(config.kappa))
    out.insert(0, "lambda_risk", float(config.lambda_risk))
    return out


def _annual_walkforward_bundle(
    *,
    project_root: Path,
    optimizer_config: OptimizerConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = plot_default_paths(project_root)
    pred_metrics = load_csv(paths.reports_root / "v4_prediction_benchmark_metrics.csv")
    params = ast.literal_eval(str(pred_metrics.loc[pred_metrics["experiment_name"].eq(BEST_60_EXPERIMENT), "selected_params"].iloc[0]))
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

    monthly_excess_history = load_modeling_inputs(project_root, feature_set_name="core_baseline").monthly_excess_history
    if monthly_excess_history is None:
        raise RuntimeError("monthly_excess_history missing for allocator refinement")
    sigma_map = build_sigma_map(list(anchor_months), excess_history=monthly_excess_history, risk_config=RiskConfig())
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    path_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    nav_model = 1.0
    nav_equal = 1.0
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
        weights = _normalize(optimizer_cache.solve(anchor, ordered["y_pred"].to_numpy(dtype=float), optimizer_config))
        for sleeve_id, weight in zip(ordered["sleeve_id"].tolist(), weights):
            weight_rows.append({"anchor_month_end": pd.Timestamp(anchor), "sleeve_id": sleeve_id, "weight": float(weight)})

        next_anchor = pd.Timestamp(anchor) + pd.offsets.YearEnd(1)
        hold_months = total_return_panel.index[(total_return_panel.index > pd.Timestamp(anchor)) & (total_return_panel.index <= next_anchor)]
        hold_panel = total_return_panel.loc[hold_months, list(SLEEVE_ORDER)].copy().dropna(how="any")
        for month_end, row in hold_panel.iterrows():
            port_ret = float(np.dot(weights, row.to_numpy(dtype=float)))
            eq_ret = float(row.mean())
            nav_model *= 1.0 + port_ret
            nav_equal *= 1.0 + eq_ret
            path_rows.extend(
                [
                    {
                        "config_label": _config_label(optimizer_config),
                        "lambda_risk": float(optimizer_config.lambda_risk),
                        "kappa": float(optimizer_config.kappa),
                        "omega_type": str(optimizer_config.omega_type),
                        "month_end": pd.Timestamp(month_end),
                        "strategy_label": "model",
                        "wealth_index": float(nav_model),
                        "ret_1m_realized": port_ret,
                        "anchor_month_end": pd.Timestamp(anchor),
                    },
                    {
                        "config_label": _config_label(optimizer_config),
                        "lambda_risk": float(optimizer_config.lambda_risk),
                        "kappa": float(optimizer_config.kappa),
                        "omega_type": str(optimizer_config.omega_type),
                        "month_end": pd.Timestamp(month_end),
                        "strategy_label": "equal_weight",
                        "wealth_index": float(nav_equal),
                        "ret_1m_realized": eq_ret,
                        "anchor_month_end": pd.Timestamp(anchor),
                    },
                ]
            )

    weights_df = pd.DataFrame(weight_rows)
    concentration = (
        weights_df.groupby("anchor_month_end", as_index=False)["weight"]
        .agg(max_weight="max", hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
        .sort_values("anchor_month_end")
        .reset_index(drop=True)
    )
    concentration["effective_n_sleeves"] = 1.0 / concentration["hhi"]
    top_counts = (
        weights_df.loc[weights_df.groupby("anchor_month_end")["weight"].idxmax()]
        .groupby("sleeve_id")
        .size()
        .sort_values(ascending=False)
    )
    wealth_summary = {
        "end_wealth_model": nav_model,
        "end_wealth_equal_weight": nav_equal,
        "end_wealth_minus_equal_weight": nav_model - nav_equal,
        "walkforward_avg_max_weight": float(concentration["max_weight"].mean()),
        "walkforward_effective_n_sleeves": float(concentration["effective_n_sleeves"].mean()),
        "walkforward_top_weight_sleeve": str(top_counts.index[0]),
        "walkforward_top_weight_frequency": float(top_counts.iloc[0] / len(concentration)),
    }
    return pd.DataFrame(path_rows).sort_values(["strategy_label", "month_end"]).reset_index(drop=True), wealth_summary


def _render_report(*, results: pd.DataFrame, weights_summary: pd.DataFrame, attribution_summary: pd.DataFrame) -> str:
    ordered = results.sort_values(["end_wealth_minus_equal_weight", "sharpe", "avg_effective_n_sleeves"], ascending=[False, False, False]).reset_index(drop=True)
    raw_perf = ordered.iloc[0]
    div_best = results.sort_values(["avg_max_weight", "avg_effective_n_sleeves", "end_wealth_minus_equal_weight"], ascending=[True, False, False]).iloc[0]
    anchor = results.loc[results["config_label"].eq("lam10_kap0.1_diag")].iloc[0]
    feasible = results.loc[
        (results["end_wealth_minus_equal_weight"] >= 0.0)
        & (results["avg_max_weight"] < anchor["avg_max_weight"])
        & (results["avg_effective_n_sleeves"] > anchor["avg_effective_n_sleeves"])
    ].copy()
    if feasible.empty:
        balanced = results.sort_values(["end_wealth_minus_equal_weight", "avg_max_weight", "avg_effective_n_sleeves"], ascending=[False, True, False]).iloc[0]
    else:
        feasible = feasible.loc[
            (feasible["avg_max_weight"] <= anchor["avg_max_weight"] - 0.20)
            & (feasible["avg_effective_n_sleeves"] >= anchor["avg_effective_n_sleeves"] + 2.0)
        ].copy()
        if feasible.empty:
            feasible = results.loc[
                (results["end_wealth_minus_equal_weight"] >= 0.0)
                & (results["avg_max_weight"] < anchor["avg_max_weight"])
                & (results["avg_effective_n_sleeves"] > anchor["avg_effective_n_sleeves"])
            ].copy()
        feasible["balanced_score"] = (
            0.9 * feasible["end_wealth_minus_equal_weight"]
            - 0.6 * feasible["avg_max_weight"]
            + 0.16 * feasible["avg_effective_n_sleeves"]
            + 0.10 * feasible["sharpe"]
        )
        balanced = feasible.sort_values(["balanced_score", "end_wealth_minus_equal_weight", "avg_effective_n_sleeves"], ascending=[False, False, False]).iloc[0]

    eq_cn = weights_summary.loc[(weights_summary["config_label"].eq(str(balanced["config_label"]))) & (weights_summary["sleeve_id"].eq("EQ_CN"))].iloc[0]
    top_bal = attribution_summary.loc[attribution_summary["config_label"].eq(str(balanced["config_label"]))].head(5)

    lines = [
        "# XOPTPOE v4 Allocator Refinement Report",
        "",
        "## Scope",
        "- Fixed predictor: `elastic_net__core_plus_interactions__separate_60`.",
        "- Fixed branch: accepted v4 benchmark roster and current walk-forward logic.",
        "- Only allocator settings moved in a narrow second-stage grid between the raw anchor and the over-defensive identity setting.",
        "",
        "## Grid",
    ]
    for row in results[["config_label", "lambda_risk", "kappa", "omega_type"]].drop_duplicates().itertuples(index=False):
        lines.append(f"- `{row.config_label}`: lambda={row.lambda_risk:.2f}, kappa={row.kappa:.2f}, omega={row.omega_type}.")
    lines.extend(["", "## Candidate Summary"])
    for row in ordered.itertuples(index=False):
        lines.append(
            f"- {row.config_label}: end_wealth={row.end_wealth_model:.3f}, ew_end={row.end_wealth_equal_weight:.3f}, delta={row.end_wealth_minus_equal_weight:.3f}, sharpe={row.sharpe:.3f}, avg_max_weight={row.avg_max_weight:.3f}, eff_n={row.avg_effective_n_sleeves:.2f}, top_weight={row.top_weight_sleeve}."
        )
    lines.extend(
        [
            "",
            "## Decisions",
            f"- Raw best-performance setting: `{raw_perf.config_label}` with end wealth {raw_perf.end_wealth_model:.3f} versus equal weight {raw_perf.end_wealth_equal_weight:.3f}.",
            f"- Best diversification setting: `{div_best.config_label}` with avg max weight {div_best.avg_max_weight:.3f} and effective N {div_best.avg_effective_n_sleeves:.2f}.",
            f"- Best balanced carry-forward setting: `{balanced.config_label}` with end wealth delta {balanced.end_wealth_minus_equal_weight:.3f}, avg max weight {balanced.avg_max_weight:.3f}, and effective N {balanced.avg_effective_n_sleeves:.2f}.",
            f"- Versus the raw anchor `lam10_kap0.1_diag`, the balanced setting changes avg max weight by {balanced.avg_max_weight - anchor.avg_max_weight:.3f} and effective N by {balanced.avg_effective_n_sleeves - anchor.avg_effective_n_sleeves:.2f}.",
            "",
            "## China Readout",
            f"- Under `{balanced.config_label}`, EQ_CN avg_weight={eq_cn.avg_weight:.4f}, max_weight={eq_cn.max_weight_observed:.4f}, top_weight_frequency={eq_cn.top_weight_frequency:.4f}.",
            "",
            "## Balanced Attribution",
        ]
    )
    for row in top_bal.itertuples(index=False):
        lines.append(
            f"- {row.sleeve_id}: total_active_contribution_vs_equal={row.total_active_contribution_vs_equal_weight:.4f}, abs_active_share={row.abs_active_contribution_share:.4f}."
        )
    return "\n".join(lines) + "\n"


def run_allocator_refinement_v4(
    *,
    project_root: Path,
    risk_config: RiskConfig | None = None,
    refinement_grid: list[OptimizerConfig] | None = None,
) -> AllocatorRefinementOutputs:
    root = project_root.resolve()
    risk_config = risk_config or RiskConfig()
    refinement_grid = list(refinement_grid or allocator_refinement_grid())

    validation_signals, test_signals, monthly_excess_history = _signal_maps(root)
    months = sorted(
        pd.concat(
            [
                validation_signals[BEST_60_LABEL]["month_end"].drop_duplicates(),
                test_signals[BEST_60_LABEL]["month_end"].drop_duplicates(),
            ],
            ignore_index=True,
        ).tolist()
    )
    sigma_map = build_sigma_map(months, excess_history=monthly_excess_history, risk_config=risk_config)
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    results_rows: list[dict[str, object]] = []
    weights_frames: list[pd.DataFrame] = []
    attribution_frames: list[pd.DataFrame] = []
    wealth_frames: list[pd.DataFrame] = []

    for config in refinement_grid:
        _, weights, test_summary = _evaluate_test_panel(
            signal_panel=test_signals[BEST_60_LABEL],
            optimizer_cache=optimizer_cache,
            optimizer_config=config,
        )
        wealth_path, wealth_summary = _annual_walkforward_bundle(project_root=root, optimizer_config=config)
        result_row = {**test_summary, **wealth_summary}
        results_rows.append(result_row)
        weights_frames.append(_weights_summary(weights, config=config))
        attribution_frames.append(_attribution_summary(weights, config=config))
        wealth_frames.append(wealth_path)

    results = pd.DataFrame(results_rows).sort_values(
        ["end_wealth_minus_equal_weight", "sharpe", "avg_effective_n_sleeves"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    weights_summary = pd.concat(weights_frames, ignore_index=True).sort_values(
        ["config_label", "avg_weight", "sleeve_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    attribution_summary = pd.concat(attribution_frames, ignore_index=True).sort_values(
        ["config_label", "abs_active_contribution_share", "sleeve_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    wealth_paths = pd.concat(wealth_frames, ignore_index=True).sort_values(["config_label", "strategy_label", "month_end"]).reset_index(drop=True)
    report_text = _render_report(results=results, weights_summary=weights_summary, attribution_summary=attribution_summary)
    return AllocatorRefinementOutputs(
        results=results,
        weights_summary=weights_summary,
        attribution_summary=attribution_summary,
        wealth_paths=wealth_paths,
        report_text=report_text,
    )


def write_allocator_refinement_outputs(project_root: Path, outputs: AllocatorRefinementOutputs) -> None:
    reports_dir = project_root.resolve() / "reports"
    write_csv(outputs.results, reports_dir / "v4_allocator_refinement_results.csv")
    write_csv(outputs.weights_summary, reports_dir / "v4_allocator_refinement_weights_summary.csv")
    write_csv(outputs.attribution_summary, reports_dir / "v4_allocator_refinement_attribution_summary.csv")
    write_csv(outputs.wealth_paths, reports_dir / "v4_allocator_refinement_wealth_paths.csv")
    write_text(outputs.report_text, reports_dir / "v4_allocator_refinement_report.md")
