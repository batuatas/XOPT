#!/usr/bin/env python3
"""Run simple long-only monthly portfolio backtests for XOPTPOE v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_portfolio.backtest import (  # noqa: E402
    BacktestConfig,
    choose_signal_model,
    run_portfolio_backtest,
    summarize_performance,
    summarize_weights,
)
from xoptpoe_modeling.io import write_csv, write_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-only portfolio backtest on test predictions")
    parser.add_argument(
        "--signal-model",
        default=None,
        help="Model name from baseline_metrics_overall.csv. Default: best validation RMSE model.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k count for top_k_equal strategy")
    parser.add_argument("--cov-lookback-months", type=int, default=60, help="Covariance trailing window")
    parser.add_argument("--cov-min-months", type=int, default=24, help="Minimum months for covariance estimation")
    parser.add_argument("--mv-ridge", type=float, default=1e-3, help="Ridge added to covariance diagonal")
    parser.add_argument("--mv-risk-aversion", type=float, default=1.0, help="Risk-aversion scale in MV heuristic")
    return parser.parse_args()


def _build_report(
    *,
    signal_model: str,
    config: BacktestConfig,
    perf: pd.DataFrame,
    wsum: pd.DataFrame,
) -> str:
    best_by_sharpe = perf.sort_values("sharpe_annualized", ascending=False).iloc[0]
    eq = perf.loc[perf["strategy"] == "equal_weight"].iloc[0]
    non_eq = perf.loc[perf["strategy"] != "equal_weight"].copy()
    added_value = bool(
        (non_eq["sharpe_annualized"] > float(eq["sharpe_annualized"])).any()
        or (non_eq["avg_monthly_return"] > float(eq["avg_monthly_return"])).any()
    )

    top_alloc = wsum.sort_values("avg_weight", ascending=False).head(10)

    lines: list[str] = []
    lines.append("# XOPTPOE v1 Portfolio Backtest Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- signal_model: `{signal_model}`")
    lines.append("- constraints: long-only, fully invested, no leverage, monthly rebalancing")
    lines.append(
        f"- config: top_k={config.top_k}, cov_lookback_months={config.cov_lookback_months}, "
        f"cov_min_months={config.cov_min_months}, mv_ridge={config.mv_ridge}, "
        f"mv_risk_aversion={config.mv_risk_aversion}"
    )
    lines.append("")
    lines.append("## Backtest Variants")
    lines.append("- `equal_weight`: 1/8 each month")
    lines.append(f"- `top_k_equal`: equal weight over top-{config.top_k} predicted sleeves each month")
    lines.append("- `score_positive`: weights proportional to clipped positive prediction scores")
    lines.append("- `mv_clipped`: regularized mean-variance heuristic + long-only clipping")
    lines.append("")
    lines.append("## Performance Summary")
    for row in perf.sort_values("sharpe_annualized", ascending=False).itertuples(index=False):
        lines.append(
            f"- {row.strategy}: avg_ret={row.avg_monthly_return:.6f}, vol={row.vol_monthly:.6f}, "
            f"sharpe={row.sharpe_annualized:.4f}, max_dd={row.max_drawdown:.4f}, turnover={row.avg_turnover:.4f}"
        )
    lines.append("")
    lines.append("## Best Strategy")
    lines.append(
        f"- best_by_sharpe: `{best_by_sharpe.strategy}` "
        f"(Sharpe={best_by_sharpe.sharpe_annualized:.4f}, avg_ret={best_by_sharpe.avg_monthly_return:.6f})"
    )
    lines.append("")
    lines.append("## Value Add vs Equal Weight")
    lines.append(
        f"- equal_weight: avg_ret={eq['avg_monthly_return']:.6f}, sharpe={eq['sharpe_annualized']:.4f}"
    )
    lines.append(
        "- prediction_layer_adds_value_over_equal_weight: "
        + ("YES" if added_value else "NO")
    )
    lines.append("")
    lines.append("## Weight Diagnostics")
    lines.append("- highest average sleeve allocations across strategies:")
    for row in top_alloc.itertuples(index=False):
        lines.append(
            f"  - {row.strategy}/{row.sleeve_id}: avg={row.avg_weight:.4f}, "
            f"min={row.min_weight:.4f}, max={row.max_weight:.4f}, nonzero_share={row.nonzero_share:.2%}"
        )
    lines.append("")
    lines.append("## Next Improvements")
    lines.append("- Add transaction-cost assumptions and net-return reporting.")
    lines.append("- Calibrate top-k, score transforms, and MV risk-aversion with rolling validation.")
    lines.append("- Add covariance shrinkage alternatives and turnover penalties.")
    lines.append("- Evaluate model-ensemble signals instead of single-model predictions.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    predictions_test = pd.read_csv(PROJECT_ROOT / "data" / "modeling" / "predictions_test.csv", parse_dates=["month_end"])
    metrics_overall = pd.read_csv(PROJECT_ROOT / "reports" / "baseline_metrics_overall.csv")
    realized_history = pd.read_csv(PROJECT_ROOT / "data" / "modeling" / "modeling_panel_filtered.csv", parse_dates=["month_end"])

    signal_model = choose_signal_model(metrics_overall=metrics_overall, requested_model=args.signal_model)
    cfg = BacktestConfig(
        top_k=args.top_k,
        cov_lookback_months=args.cov_lookback_months,
        cov_min_months=args.cov_min_months,
        mv_ridge=args.mv_ridge,
        mv_risk_aversion=args.mv_risk_aversion,
    )

    returns_df, weights_df = run_portfolio_backtest(
        predictions_test=predictions_test,
        realized_history=realized_history,
        signal_model=signal_model,
        config=cfg,
    )

    perf = summarize_performance(returns_df)
    wsum = summarize_weights(weights_df)
    report_text = _build_report(signal_model=signal_model, config=cfg, perf=perf, wsum=wsum)

    write_csv(returns_df, PROJECT_ROOT / "data" / "modeling" / "portfolio_returns.csv")
    write_csv(perf, PROJECT_ROOT / "reports" / "portfolio_performance_summary.csv")
    write_csv(wsum, PROJECT_ROOT / "reports" / "portfolio_weights_summary.csv")
    write_text(report_text, PROJECT_ROOT / "reports" / "portfolio_backtest_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
