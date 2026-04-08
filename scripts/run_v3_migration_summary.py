#!/usr/bin/env python3
"""Summarize the downstream migration from v2 to the active v3 China branch."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v3_modeling.io import write_csv, write_text  # noqa: E402


def _select_best(metrics_df: pd.DataFrame, horizon_mode: str) -> pd.Series:
    subset = metrics_df.loc[metrics_df["horizon_mode"].eq(horizon_mode)].copy()
    subset = subset.loc[~subset["model_name"].isin({"naive_mean", "pto_nn", "e2e_nn"})].copy()
    return subset.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).iloc[0]


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> int:
    root = PROJECT_ROOT
    reports_v2 = root / "reports" / "v2_long_horizon"
    reports_v3 = root / "reports" / "v3_long_horizon_china"

    metrics_v2 = _load_csv(reports_v2 / "predictor_horse_race_metrics.csv")
    metrics_v3 = _load_csv(reports_v3 / "predictor_horse_race_metrics.csv")
    port_v2 = _load_csv(reports_v2 / "saa_portfolio_comparison_metrics.csv")
    port_v3 = _load_csv(reports_v3 / "saa_portfolio_comparison_metrics.csv")
    china_pred = _load_csv(reports_v3 / "china_sleeve_prediction_summary.csv")
    china_port = _load_csv(reports_v3 / "china_sleeve_portfolio_summary.csv")
    integrity = _load_csv(reports_v3 / "downstream_integrity_checks.csv")

    predictor_rows: list[dict[str, object]] = []
    slot_map = {
        "best_60_predictor": "separate_60",
        "best_120_predictor": "separate_120",
        "best_shared_predictor": "shared_60_120",
    }
    for version, frame in (("v2", metrics_v2), ("v3", metrics_v3)):
        for benchmark_slot, horizon_mode in slot_map.items():
            row = _select_best(frame, horizon_mode)
            predictor_rows.append(
                {
                    "version": version,
                    "benchmark_slot": benchmark_slot,
                    "experiment_name": row["experiment_name"],
                    "model_name": row["model_name"],
                    "feature_set_name": row["feature_set_name"],
                    "validation_rmse": float(row["validation_rmse"]),
                    "test_rmse": float(row["test_rmse"]),
                    "test_corr": float(row["test_corr"]),
                    "test_sign_accuracy": float(row["test_sign_accuracy"]),
                }
            )
    predictor_cmp = pd.DataFrame(predictor_rows).sort_values(["benchmark_slot", "version"]).reset_index(drop=True)

    strategy_order = [
        "equal_weight",
        "best_60_predictor",
        "best_120_predictor",
        "combined_60_120_predictor",
        "best_shared_predictor",
        "pto_nn_signal",
        "e2e_nn_signal",
    ]
    portfolio_rows: list[dict[str, object]] = []
    for version, frame in (("v2", port_v2), ("v3", port_v3)):
        subset = frame.loc[
            frame["source_type"].eq("common_allocator")
            & frame["strategy_label"].isin(strategy_order)
        ].copy()
        subset["version"] = version
        portfolio_rows.append(subset)
    portfolio_cmp = (
        pd.concat(portfolio_rows, ignore_index=True)
        .sort_values(["split", "strategy_label", "version"])
        .reset_index(drop=True)
    )

    model_vs_benchmark = port_v3.loc[
        port_v3["source_type"].eq("common_allocator")
        & port_v3["strategy_label"].isin(strategy_order)
    ].copy()
    ref_cols = ["split", "strategy_label", "avg_return", "volatility", "sharpe", "avg_turnover"]
    refs = {}
    for ref in ("equal_weight", "pto_nn_signal", "e2e_nn_signal"):
        refs[ref] = model_vs_benchmark.loc[model_vs_benchmark["strategy_label"].eq(ref), ref_cols].rename(
            columns={
                "avg_return": f"{ref}_avg_return",
                "volatility": f"{ref}_volatility",
                "sharpe": f"{ref}_sharpe",
                "avg_turnover": f"{ref}_avg_turnover",
            }
        )
    out = model_vs_benchmark.copy()
    for ref, ref_df in refs.items():
        out = out.merge(ref_df.drop(columns=["strategy_label"]), on="split", how="left", validate="m:1")
        out[f"delta_avg_return_vs_{ref}"] = out["avg_return"] - out[f"{ref}_avg_return"]
        out[f"delta_sharpe_vs_{ref}"] = out["sharpe"] - out[f"{ref}_sharpe"]
        out[f"delta_volatility_vs_{ref}"] = out["volatility"] - out[f"{ref}_volatility"]
        out[f"delta_turnover_vs_{ref}"] = out["avg_turnover"] - out[f"{ref}_avg_turnover"]
    model_vs_benchmark = out.sort_values(["split", "sharpe"], ascending=[True, False]).reset_index(drop=True)

    best60_v2 = predictor_cmp.loc[
        predictor_cmp["version"].eq("v2") & predictor_cmp["benchmark_slot"].eq("best_60_predictor")
    ].iloc[0]
    best60_v3 = predictor_cmp.loc[
        predictor_cmp["version"].eq("v3") & predictor_cmp["benchmark_slot"].eq("best_60_predictor")
    ].iloc[0]
    best120_v2 = predictor_cmp.loc[
        predictor_cmp["version"].eq("v2") & predictor_cmp["benchmark_slot"].eq("best_120_predictor")
    ].iloc[0]
    best120_v3 = predictor_cmp.loc[
        predictor_cmp["version"].eq("v3") & predictor_cmp["benchmark_slot"].eq("best_120_predictor")
    ].iloc[0]
    best_port_v3 = portfolio_cmp.loc[
        portfolio_cmp["version"].eq("v3")
        & portfolio_cmp["source_type"].eq("common_allocator")
        & portfolio_cmp["split"].eq("test")
    ].sort_values(["sharpe", "avg_return"], ascending=[False, False]).iloc[0]
    eq_v2 = portfolio_cmp.loc[
        portfolio_cmp["version"].eq("v2")
        & portfolio_cmp["source_type"].eq("common_allocator")
        & portfolio_cmp["split"].eq("test")
        & portfolio_cmp["strategy_label"].eq("equal_weight")
    ].iloc[0]
    eq_v3 = portfolio_cmp.loc[
        portfolio_cmp["version"].eq("v3")
        & portfolio_cmp["source_type"].eq("common_allocator")
        & portfolio_cmp["split"].eq("test")
        & portfolio_cmp["strategy_label"].eq("equal_weight")
    ].iloc[0]
    eq_cn_best_shared = china_port.loc[
        china_port["split"].eq("test") & china_port["strategy_label"].eq("best_shared_predictor")
    ].iloc[0]
    eq_cn_best60 = china_port.loc[
        china_port["split"].eq("test") & china_port["strategy_label"].eq("best_60_predictor")
    ].iloc[0]
    eq_cn_pred_shared = china_pred.loc[
        china_pred["split"].eq("test") & china_pred["strategy_label"].eq("best_shared_predictor")
    ].iloc[0]

    lines: list[str] = []
    lines.append("# XOPTPOE v3 Downstream Migration Summary")
    lines.append("")
    lines.append("## Status")
    lines.append("- v3 is the active downstream branch.")
    lines.append("- v1 and v2 remain untouched benchmark branches.")
    lines.append(f"- Integrity checks: {int((integrity['status'] == 'PASS').sum())} PASS, {int((integrity['status'] == 'FAIL').sum())} FAIL.")
    lines.append("")
    lines.append("## What Changed Versus v2")
    lines.append(
        f"- Best 60m predictor changed from `{best60_v2['experiment_name']}` in v2 to `{best60_v3['experiment_name']}` in v3."
    )
    lines.append(
        f"- Best 120m predictor remained a full-firstpass ridge winner, with v2 `{best120_v2['experiment_name']}` and v3 `{best120_v3['experiment_name']}`."
    )
    lines.append(
        f"- Equal-weight test Sharpe moved from {eq_v2['sharpe']:.4f} in the 8-sleeve branch to {eq_v3['sharpe']:.4f} in the 9-sleeve branch; the equal-weight max weight fell from {eq_v2['avg_max_weight']:.4f} to {eq_v3['avg_max_weight']:.4f} because the benchmark is now 1/9 instead of 1/8."
    )
    lines.append("")
    lines.append("## China Sleeve Readout")
    lines.append(
        f"- EQ_CN predictive quality is mixed: in the best shared predictor on the test split, rmse={eq_cn_pred_shared['rmse']:.4f}, corr={eq_cn_pred_shared['corr']:.4f}, sign_accuracy={eq_cn_pred_shared['sign_accuracy']:.4f}."
    )
    lines.append(
        f"- EQ_CN remains marginal in the strongest allocation diagnostics: best_shared avg_weight={eq_cn_best_shared['avg_weight']:.4f}, max_weight={eq_cn_best_shared['max_weight']:.4f}; best_60 avg_weight={eq_cn_best60['avg_weight']:.4f}."
    )
    lines.append(
        f"- EQ_CN never becomes the top-weight sleeve in the monitored test strategies; top_weight_frequency stays {eq_cn_best_shared['top_weight_frequency']:.4f} in best_shared and {eq_cn_best60['top_weight_frequency']:.4f} in best_60."
    )
    lines.append("")
    lines.append("## Active Benchmark")
    lines.append(
        f"- Best test portfolio behavior in v3 is `{best_port_v3['strategy_label']}` with avg_return={best_port_v3['avg_return']:.4f}, sharpe={best_port_v3['sharpe']:.4f}, avg_turnover={best_port_v3['avg_turnover']:.4f}."
    )
    lines.append("- PTO and E2E should now be judged against the strongest v3 supervised benchmark rather than against equal weight alone.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Adding EQ_CN made the active branch coherent as a 9-sleeve system, but it did not make China a dominant allocation sleeve in the current benchmark stack.")
    lines.append("- The best supervised predictors remain linear and separate-horizon. PTO/E2E still lag those benchmarks on prediction quality.")
    return_text = "\n".join(lines) + "\n"

    write_csv(predictor_cmp, reports_v3 / "v2_vs_v3_predictor_comparison.csv")
    write_csv(portfolio_cmp, reports_v3 / "v2_vs_v3_portfolio_comparison.csv")
    write_csv(model_vs_benchmark, reports_v3 / "model_vs_benchmark_comparison.csv")
    write_text(return_text, reports_v3 / "v3_migration_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
