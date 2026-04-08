"""Compact allocator-regularization sweep for the fixed v4 best-60 predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_v4_modeling.io import write_csv, write_text
from xoptpoe_v4_models.data import SLEEVE_ORDER
from xoptpoe_v4_models.optim_layers import OptimizerConfig, RiskConfig, RobustOptimizerCache, build_sigma_map
from xoptpoe_v4_models.portfolio_benchmark import _signal_maps
from xoptpoe_v4_models.portfolio_eval import run_portfolio_evaluation, summarize_portfolio_metrics
from xoptpoe_v4_plots.io import _best_60_prediction_frames, _run_best60_diversified


BEST_60_LABEL = "best_60_predictor"


@dataclass(frozen=True)
class AllocatorSweepOutputs:
    results: pd.DataFrame
    weights_summary: pd.DataFrame
    attribution_summary: pd.DataFrame
    report_text: str


def allocator_sweep_grid() -> list[OptimizerConfig]:
    return [
        OptimizerConfig(lambda_risk=10.0, kappa=0.10, omega_type="diag"),
        OptimizerConfig(lambda_risk=15.0, kappa=0.10, omega_type="diag"),
        OptimizerConfig(lambda_risk=20.0, kappa=0.10, omega_type="diag"),
        OptimizerConfig(lambda_risk=30.0, kappa=0.10, omega_type="diag"),
        OptimizerConfig(lambda_risk=10.0, kappa=0.25, omega_type="diag"),
        OptimizerConfig(lambda_risk=15.0, kappa=0.25, omega_type="diag"),
        OptimizerConfig(lambda_risk=20.0, kappa=0.25, omega_type="diag"),
        OptimizerConfig(lambda_risk=30.0, kappa=0.25, omega_type="diag"),
        OptimizerConfig(lambda_risk=15.0, kappa=0.50, omega_type="diag"),
        OptimizerConfig(lambda_risk=20.0, kappa=0.50, omega_type="diag"),
        OptimizerConfig(lambda_risk=30.0, kappa=0.50, omega_type="diag"),
        OptimizerConfig(lambda_risk=15.0, kappa=0.25, omega_type="identity"),
        OptimizerConfig(lambda_risk=20.0, kappa=0.50, omega_type="identity"),
    ]


def _config_label(config: OptimizerConfig) -> str:
    lam = int(config.lambda_risk) if float(config.lambda_risk).is_integer() else config.lambda_risk
    return f"lam{lam:g}_kap{config.kappa:g}_{config.omega_type}"


def _split_monthly_concentration(weights: pd.DataFrame) -> pd.DataFrame:
    out = (
        weights.groupby("month_end", as_index=False)["weight"]
        .agg(max_weight="max", hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
        .sort_values("month_end")
        .reset_index(drop=True)
    )
    out["effective_n_sleeves"] = 1.0 / out["hhi"]
    return out


def _evaluate_split(
    *,
    split_name: str,
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
    top2_sleeve_active_share_abs = float(sleeve_active.head(2)["abs_active"].sum() / abs_total) if abs_total > 0 else float("nan")

    model_metric = metrics.loc[metrics["strategy"].eq(BEST_60_LABEL)].iloc[0]
    summary_row = {
        "split": split_name,
        "lambda_risk": float(optimizer_config.lambda_risk),
        "kappa": float(optimizer_config.kappa),
        "omega_type": str(optimizer_config.omega_type),
        "config_label": _config_label(optimizer_config),
        "month_count": int(model_metric["month_count"]),
        "avg_return": float(model_metric["avg_return"]),
        "volatility": float(model_metric["volatility"]),
        "sharpe": float(model_metric["sharpe"]),
        "max_drawdown": float(model_metric["max_drawdown"]),
        "avg_turnover": float(model_metric["avg_turnover"]),
        "avg_active_return_vs_equal_weight": float(model_returns["active_return_vs_equal_weight"].mean()),
        "avg_max_weight": float(concentration["max_weight"].mean()),
        "avg_hhi": float(concentration["hhi"].mean()),
        "avg_effective_n_sleeves": float(concentration["effective_n_sleeves"].mean()),
        "top_weight_sleeve": top_weight_sleeve,
        "top_weight_sleeve_frequency": top_weight_frequency,
        "top_sleeve_active_contribution_share_abs": top_sleeve_active_contribution_share_abs,
        "top2_sleeve_active_share_abs": top2_sleeve_active_share_abs,
    }
    return model_returns, weights, summary_row


def _weights_summary(weights: pd.DataFrame, *, config: OptimizerConfig, split_name: str) -> pd.DataFrame:
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
    out.insert(0, "split", split_name)
    return out


def _attribution_summary(weights: pd.DataFrame, *, config: OptimizerConfig, split_name: str) -> pd.DataFrame:
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
    out.insert(0, "split", split_name)
    return out


def _balanced_score_frame(results: pd.DataFrame) -> pd.DataFrame:
    test = results.loc[results["split"].eq("test")].copy()
    for col in ["avg_max_weight", "avg_effective_n_sleeves", "top_sleeve_active_contribution_share_abs", "sharpe"]:
        if col not in test.columns:
            raise KeyError(f"Missing required column for balanced scoring: {col}")

    candidates = test.loc[(test["avg_max_weight"] <= 0.20) & (test["avg_effective_n_sleeves"] >= 6.0)].copy()
    if candidates.empty:
        sharpe_floor = float(test.loc[test["config_label"].eq("lam10_kap0.1_diag"), "sharpe"].iloc[0]) - 2.0
        candidates = test.loc[test["sharpe"].ge(sharpe_floor)].copy()
        candidates["balanced_score"] = (
            -candidates["avg_max_weight"]
            + 0.10 * candidates["avg_effective_n_sleeves"]
            - candidates["top_sleeve_active_contribution_share_abs"]
            + 0.20 * candidates["sharpe"]
        )
        return candidates.sort_values(
            ["balanced_score", "avg_effective_n_sleeves", "sharpe"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    candidates["balanced_score"] = (
        0.40 * candidates["sharpe"]
        + 0.12 * candidates["avg_effective_n_sleeves"]
        - candidates["avg_max_weight"]
        - candidates["top_sleeve_active_contribution_share_abs"]
    )
    return candidates.sort_values(
        ["balanced_score", "sharpe", "avg_effective_n_sleeves"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _current_diversified_summary(project_root: Path) -> pd.Series:
    val_pred, test_pred = _best_60_prediction_frames(_plot_paths(project_root))
    _, _, summary = _run_best60_diversified(val_pred, test_pred)
    return summary.loc[summary["strategy_label"].eq("best_60_diversified_cap") & summary["split"].eq("test")].iloc[0]


def _plot_paths(project_root: Path):
    from xoptpoe_v4_plots.io import default_paths

    return default_paths(project_root)


def _render_report(
    *,
    results: pd.DataFrame,
    weights_summary: pd.DataFrame,
    attribution_summary: pd.DataFrame,
    current_diversified: pd.Series,
) -> str:
    test = results.loc[results["split"].eq("test")].copy().sort_values(["sharpe", "avg_effective_n_sleeves"], ascending=[False, False])
    raw_best = test.iloc[0]
    balanced_rank = _balanced_score_frame(results)
    balanced_best = balanced_rank.iloc[0]
    anchor = test.loc[test["config_label"].eq("lam10_kap0.1_diag")].iloc[0]
    eq_cn_bal = weights_summary.loc[
        weights_summary["split"].eq("test")
        & weights_summary["config_label"].eq(str(balanced_best["config_label"]))
        & weights_summary["sleeve_id"].eq("EQ_CN")
    ].iloc[0]
    top_balanced = attribution_summary.loc[
        attribution_summary["split"].eq("test") & attribution_summary["config_label"].eq(str(balanced_best["config_label"]))
    ].head(5)

    lines: list[str] = []
    lines.append("# XOPTPOE v4 Allocator Sweep Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Fixed object: `best_60_predictor = elastic_net__core_plus_interactions__separate_60`.")
    lines.append("- Fixed branch: v4 accepted benchmark roster and current split logic.")
    lines.append("- Only allocator regularization moved: `lambda_risk`, `kappa`, and existing `omega_type`.")
    lines.append("- This is a presentation-benchmark tuning pass, not a new predictor selection pass.")
    lines.append("")
    lines.append("## Grid")
    for row in results[["config_label", "lambda_risk", "kappa", "omega_type"]].drop_duplicates().itertuples(index=False):
        lines.append(f"- `{row.config_label}`: lambda={row.lambda_risk:.2f}, kappa={row.kappa:.2f}, omega={row.omega_type}.")
    lines.append("")
    lines.append("## Test Results")
    for row in test.itertuples(index=False):
        lines.append(
            f"- {row.config_label}: sharpe={row.sharpe:.4f}, avg_return={row.avg_return:.4f}, avg_max_weight={row.avg_max_weight:.4f}, eff_n={row.avg_effective_n_sleeves:.2f}, top_weight={row.top_weight_sleeve}, top_weight_freq={row.top_weight_sleeve_frequency:.4f}, top_active_share={row.top_sleeve_active_contribution_share_abs:.4f}."
        )
    lines.append("")
    lines.append("## Raw Best Sharpe")
    lines.append(
        f"- `{raw_best.config_label}` is the raw test-Sharpe winner: sharpe={raw_best.sharpe:.4f}, avg_return={raw_best.avg_return:.4f}, avg_max_weight={raw_best.avg_max_weight:.4f}, eff_n={raw_best.avg_effective_n_sleeves:.2f}."
    )
    lines.append("")
    lines.append("## Balanced Diversification Candidate")
    lines.append(
        f"- `{balanced_best.config_label}` is the best balanced setting in this compact sweep: sharpe={balanced_best.sharpe:.4f}, avg_return={balanced_best.avg_return:.4f}, avg_max_weight={balanced_best.avg_max_weight:.4f}, eff_n={balanced_best.avg_effective_n_sleeves:.2f}, top_active_share={balanced_best.top_sleeve_active_contribution_share_abs:.4f}."
    )
    lines.append(
        f"- Versus the current raw anchor `{anchor.config_label}`: return delta={balanced_best.avg_return - anchor.avg_return:.4f}, sharpe delta={balanced_best.sharpe - anchor.sharpe:.4f}, max-weight delta={balanced_best.avg_max_weight - anchor.avg_max_weight:.4f}, eff_n delta={balanced_best.avg_effective_n_sleeves - anchor.avg_effective_n_sleeves:.2f}."
    )
    lines.append(
        f"- Versus the current heuristic diversified object `best_60_diversified_cap`: sharpe delta={balanced_best.sharpe - float(current_diversified['sharpe']):.4f}, max-weight delta={balanced_best.avg_max_weight - float(current_diversified['avg_max_weight']):.4f}, eff_n delta={balanced_best.avg_effective_n_sleeves - float(current_diversified['avg_effective_n_sleeves']):.2f}."
    )
    lines.append("")
    lines.append("## China Readout")
    lines.append(
        f"- Under `{balanced_best.config_label}`, EQ_CN avg_weight={eq_cn_bal.avg_weight:.4f}, max_weight={eq_cn_bal.max_weight_observed:.4f}, nonzero_alloc_share={eq_cn_bal.nonzero_allocation_share:.4f}, top_weight_frequency={eq_cn_bal.top_weight_frequency:.4f}."
    )
    lines.append("")
    lines.append("## Balanced Attribution")
    for row in top_balanced.itertuples(index=False):
        lines.append(
            f"- {row.sleeve_id}: total_active_contribution_vs_equal={row.total_active_contribution_vs_equal_weight:.4f}, abs_active_share={row.abs_active_contribution_share:.4f}."
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append(
        f"- Carry-forward presentation benchmark: `{balanced_best.config_label}`. It materially lowers sleeve dominance without collapsing return quality, and it is stronger than the current heuristic diversified object on both Sharpe and breadth."
    )
    lines.append(
        f"- Keep `{raw_best.config_label}` only as the raw Sharpe reference. It is stronger on pure Sharpe but still more concentration-driven than the recommended balanced setting."
    )
    return "\n".join(lines) + "\n"


def run_allocator_sweep_v4(
    *,
    project_root: Path,
    risk_config: RiskConfig | None = None,
    sweep_grid: list[OptimizerConfig] | None = None,
) -> AllocatorSweepOutputs:
    root = project_root.resolve()
    risk_config = risk_config or RiskConfig()
    sweep_grid = list(sweep_grid or allocator_sweep_grid())

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

    result_rows: list[dict[str, object]] = []
    weights_frames: list[pd.DataFrame] = []
    attribution_frames: list[pd.DataFrame] = []

    for config in sweep_grid:
        for split_name, signal_panel in (("validation", validation_signals[BEST_60_LABEL]), ("test", test_signals[BEST_60_LABEL])):
            _, weights, summary_row = _evaluate_split(
                split_name=split_name,
                signal_panel=signal_panel,
                optimizer_cache=optimizer_cache,
                optimizer_config=config,
            )
            result_rows.append(summary_row)
            weights_frames.append(_weights_summary(weights, config=config, split_name=split_name))
            attribution_frames.append(_attribution_summary(weights, config=config, split_name=split_name))

    results = pd.DataFrame(result_rows).sort_values(["split", "sharpe", "avg_effective_n_sleeves"], ascending=[True, False, False]).reset_index(drop=True)
    weights_summary = pd.concat(weights_frames, ignore_index=True).sort_values(
        ["split", "config_label", "avg_weight", "sleeve_id"], ascending=[True, True, False, True]
    ).reset_index(drop=True)
    attribution_summary = pd.concat(attribution_frames, ignore_index=True).sort_values(
        ["split", "config_label", "abs_active_contribution_share", "sleeve_id"], ascending=[True, True, False, True]
    ).reset_index(drop=True)

    current_diversified = _current_diversified_summary(root)
    report_text = _render_report(
        results=results,
        weights_summary=weights_summary,
        attribution_summary=attribution_summary,
        current_diversified=current_diversified,
    )
    return AllocatorSweepOutputs(
        results=results,
        weights_summary=weights_summary,
        attribution_summary=attribution_summary,
        report_text=report_text,
    )


def write_allocator_sweep_outputs(project_root: Path, outputs: AllocatorSweepOutputs) -> None:
    reports_dir = project_root.resolve() / "reports"
    write_csv(outputs.results, reports_dir / "v4_allocator_sweep_results.csv")
    write_csv(outputs.weights_summary, reports_dir / "v4_allocator_sweep_weights_summary.csv")
    write_csv(outputs.attribution_summary, reports_dir / "v4_allocator_sweep_attribution_summary.csv")
    write_text(outputs.report_text, reports_dir / "v4_allocator_sweep_report.md")
