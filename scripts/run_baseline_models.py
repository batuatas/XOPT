#!/usr/bin/env python3
"""Run first-pass baseline prediction models on frozen XOPTPOE splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_modeling.baselines import run_baseline_suite  # noqa: E402
from xoptpoe_modeling.eda import TARGET_COL  # noqa: E402
from xoptpoe_modeling.evaluate import evaluate_by_sleeve, evaluate_overall  # noqa: E402
from xoptpoe_modeling.io import write_csv, write_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline models for XOPTPOE v1")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _render_report(
    *,
    feature_columns: list[str],
    model_manifest: pd.DataFrame,
    metrics_overall: pd.DataFrame,
    metrics_by_sleeve: pd.DataFrame,
    best_model: str,
) -> str:
    val_metrics = metrics_overall[metrics_overall["split"] == "validation"].copy()
    test_metrics = metrics_overall[metrics_overall["split"] == "test"].copy()
    best_val_row = val_metrics[val_metrics["model"] == best_model].iloc[0]
    best_test_row = test_metrics[test_metrics["model"] == best_model].iloc[0]

    best_test_by_sleeve = metrics_by_sleeve[
        (metrics_by_sleeve["split"] == "test") & (metrics_by_sleeve["model"] == best_model)
    ].copy()
    easiest = best_test_by_sleeve.sort_values("rmse").head(3)
    hardest = best_test_by_sleeve.sort_values("rmse", ascending=False).head(3)

    readiness_note = (
        "Signal quality looks strong enough to justify cautious portfolio-construction prototyping."
        if (best_test_row["oos_r2"] > 0 and best_test_row["corr"] > 0.10)
        else "Signal quality is weak/modest; proceed to portfolio construction only as an exploratory prototype."
    )

    lines: list[str] = []
    lines.append("# XOPTPOE v1 Baseline Model Report")
    lines.append("")
    lines.append("## Models Implemented")
    for row in model_manifest.itertuples(index=False):
        lines.append(f"- {row.model} ({row.family}) with {row.hyperparams}")
    lines.append("")
    lines.append("## Feature Set Used")
    lines.append(f"- target: `{TARGET_COL}`")
    lines.append(f"- numeric feature count: `{len(feature_columns)}`")
    lines.append("- pooled models also include sleeve one-hot indicators.")
    lines.append("- numeric features:")
    for col in feature_columns:
        lines.append(f"  - {col}")
    lines.append("")
    lines.append("## Validation Metrics (Model Selection)")
    for row in val_metrics.sort_values("rmse").itertuples(index=False):
        lines.append(
            f"- {row.model}: RMSE={row.rmse:.6f}, MAE={row.mae:.6f}, "
            f"OOS_R2={row.oos_r2:.6f}, Corr={row.corr:.6f}, DirAcc={row.directional_accuracy:.6f}"
        )
    lines.append(f"- best_validation_model: `{best_model}` (lowest validation RMSE)")
    lines.append("")
    lines.append("## Test Metrics (Frozen After Validation Choice)")
    for row in test_metrics.sort_values("rmse").itertuples(index=False):
        lines.append(
            f"- {row.model}: RMSE={row.rmse:.6f}, MAE={row.mae:.6f}, "
            f"OOS_R2={row.oos_r2:.6f}, Corr={row.corr:.6f}, DirAcc={row.directional_accuracy:.6f}"
        )
    lines.append("")
    lines.append("## Sleeve-Level Difficulty (Best Model On Test)")
    lines.append("- easiest sleeves by RMSE:")
    for row in easiest.itertuples(index=False):
        lines.append(
            f"  - {row.sleeve_id}: RMSE={row.rmse:.6f}, OOS_R2={row.oos_r2:.6f}, Corr={row.corr:.6f}"
        )
    lines.append("- hardest sleeves by RMSE:")
    for row in hardest.itertuples(index=False):
        lines.append(
            f"  - {row.sleeve_id}: RMSE={row.rmse:.6f}, OOS_R2={row.oos_r2:.6f}, Corr={row.corr:.6f}"
        )
    lines.append("")
    lines.append("## Readiness")
    lines.append(f"- {readiness_note}")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    train_path = PROJECT_ROOT / "data" / "modeling" / "train_split.csv"
    validation_path = PROJECT_ROOT / "data" / "modeling" / "validation_split.csv"
    test_path = PROJECT_ROOT / "data" / "modeling" / "test_split.csv"

    train_df = pd.read_csv(train_path, parse_dates=["month_end"])
    validation_df = pd.read_csv(validation_path, parse_dates=["month_end"])
    test_df = pd.read_csv(test_path, parse_dates=["month_end"])

    pred_val, pred_test, model_manifest, feature_columns = run_baseline_suite(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        random_state=args.random_state,
    )

    pred_val = pred_val.sort_values(["model", "month_end", "sleeve_id"]).reset_index(drop=True)
    pred_test = pred_test.sort_values(["model", "month_end", "sleeve_id"]).reset_index(drop=True)

    combined_predictions = pd.concat([pred_val, pred_test], ignore_index=True)
    metrics_overall = evaluate_overall(predictions=combined_predictions, train_target=train_df[TARGET_COL])
    metrics_by_sleeve = evaluate_by_sleeve(
        predictions=combined_predictions,
        train_df=train_df,
        target_col=TARGET_COL,
    )

    validation_metrics = metrics_overall[metrics_overall["split"] == "validation"].copy()
    best_model = validation_metrics.sort_values("rmse").iloc[0]["model"]

    report_text = _render_report(
        feature_columns=feature_columns,
        model_manifest=model_manifest,
        metrics_overall=metrics_overall,
        metrics_by_sleeve=metrics_by_sleeve,
        best_model=best_model,
    )

    write_csv(pred_val, PROJECT_ROOT / "data" / "modeling" / "predictions_validation.csv")
    write_csv(pred_test, PROJECT_ROOT / "data" / "modeling" / "predictions_test.csv")
    write_csv(metrics_overall, PROJECT_ROOT / "reports" / "baseline_metrics_overall.csv")
    write_csv(metrics_by_sleeve, PROJECT_ROOT / "reports" / "baseline_metrics_by_sleeve.csv")
    write_text(report_text, PROJECT_ROOT / "reports" / "baseline_model_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
