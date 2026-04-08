#!/usr/bin/env python3
"""Run compact rolling robustness upgrades for modeling + portfolio prototype."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_modeling.io import write_csv, write_text  # noqa: E402
from xoptpoe_modeling.rolling_robustness import (  # noqa: E402
    MODEL_SET,
    RollingConfig,
    evaluate_concentration_controls,
    run_feature_set_experiments,
    select_best_feature_set_per_model,
    summarize_feature_set_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling robustness checks for XOPTPOE prototype")
    parser.add_argument("--min-train-months", type=int, default=96)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=24)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-k-cap", type=float, default=0.30)
    parser.add_argument("--score-cap", type=float, default=0.35)
    parser.add_argument("--diversify-top-n", type=int, default=5)
    parser.add_argument("--diversify-cap", type=float, default=0.25)
    return parser.parse_args()


def _render_report(
    *,
    cfg: RollingConfig,
    fold_manifest: pd.DataFrame,
    feature_summary: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    selected_feature_set_by_model: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append("# XOPTPOE v1 Rolling Robustness Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Focused robustness-improvement pass; data pipeline and locked dataset design unchanged.")
    lines.append("- Compact model set: naive_sleeve_mean, ridge_pooled, elastic_net_pooled.")
    lines.append("- Feature-set experiments: technical-only, macro-only, technical+global macro, full set.")
    lines.append("")
    lines.append("## Rolling Setup")
    lines.append(
        f"- expanding folds: min_train_months={cfg.min_train_months}, "
        f"validation_months={cfg.validation_months}, test_months={cfg.test_months}, step_months={cfg.step_months}"
    )
    lines.append(f"- fold count: {len(fold_manifest)}")
    if not fold_manifest.empty:
        first = fold_manifest.iloc[0]
        last = fold_manifest.iloc[-1]
        lines.append(
            f"- first fold test range: {first['test_start']} to {first['test_end']}; "
            f"last fold test range: {last['test_start']} to {last['test_end']}"
        )
    lines.append("")

    lines.append("## Feature-Set Robustness")
    for model_name in MODEL_SET:
        chunk = feature_summary.loc[feature_summary["model"] == model_name].copy()
        if chunk.empty:
            continue
        best_row = chunk.sort_values(["validation_rmse_mean", "test_rmse_mean"]).iloc[0]
        lines.append(
            f"- {model_name}: best feature set = `{best_row['feature_set']}` "
            f"(val_rmse={best_row['validation_rmse_mean']:.4f}, "
            f"test_rmse={best_row['test_rmse_mean']:.4f}, "
            f"test_spearman={best_row['test_spearman_ic_mean']:.4f})"
        )

    compact_candidates = feature_summary.loc[
        feature_summary["feature_set"].isin(["technical_only", "technical_plus_global_macro"])
    ]
    if not compact_candidates.empty:
        best_compact = compact_candidates.sort_values("test_rmse_mean").iloc[0]
        lines.append(
            f"- best compact feature-set/model pair by test RMSE: "
            f"`{best_compact['model']}` + `{best_compact['feature_set']}` "
            f"(test_rmse={best_compact['test_rmse_mean']:.4f}, "
            f"test_spearman={best_compact['test_spearman_ic_mean']:.4f})"
        )
    lines.append("")

    lines.append("## Concentration Control")
    lines.append(
        f"- evaluated variants: equal_weight, top_k_equal, top_k_capped(k={cfg.top_k}, cap={cfg.top_k_max_weight:.2f}), "
        f"score_positive_capped(cap={cfg.score_max_weight:.2f}), "
        f"top_k_diversified_cap(top_n={max(cfg.top_k, cfg.diversify_top_n)}, cap={cfg.diversify_max_weight:.2f})"
    )
    lines.append("- selected feature set per model for this stage:")
    for model_name in MODEL_SET:
        if model_name not in selected_feature_set_by_model:
            continue
        lines.append(f"  - {model_name}: {selected_feature_set_by_model[model_name]}")

    for model_name in MODEL_SET:
        chunk = concentration_summary.loc[concentration_summary["model"] == model_name].copy()
        if chunk.empty:
            continue
        best_port = chunk.sort_values("sharpe_annualized_mean", ascending=False).iloc[0]
        baseline = chunk.loc[chunk["strategy"] == "top_k_equal"]
        if baseline.empty:
            continue
        base_row = baseline.iloc[0]
        lines.append(
            f"- {model_name}: best concentration-aware strategy `{best_port['strategy']}` "
            f"(Sharpe={best_port['sharpe_annualized_mean']:.3f}, "
            f"avg_max_weight={best_port['avg_max_weight_mean']:.3f}). "
            f"top_k_equal baseline avg_max_weight={base_row['avg_max_weight_mean']:.3f}."
        )
    lines.append("")

    # Stable-signal decision rule (compact and explicit).
    non_naive = feature_summary.loc[feature_summary["model"] != "naive_sleeve_mean"].copy()
    stable_candidates = non_naive.loc[
        (non_naive["test_rmse_delta_vs_naive_mean"] < 0)
        & (non_naive["beat_naive_rmse_fold_share"] >= 0.50)
        & (non_naive["test_spearman_ic_mean"] > 0.05)
    ]
    has_stable_signal = not stable_candidates.empty

    lines.append("## Decision")
    if has_stable_signal:
        best_stable = stable_candidates.sort_values("test_rmse_delta_vs_naive_mean").iloc[0]
        lines.append(
            "- There is a stable predictive signal candidate under rolling checks "
            f"(`{best_stable['model']}` + `{best_stable['feature_set']}`), but it should be carried forward cautiously."
        )
        lines.append(
            "After these robustness upgrades, there is a stable signal worth building on, "
            "with concentration controls kept as default guardrails."
        )
    else:
        lines.append(
            "- Rolling evidence does not show a sufficiently stable non-naive signal across folds."
        )
        lines.append(
            "After these robustness upgrades, the project should pivot toward simpler benchmark-driven "
            "allocation rules unless further predictive signal improvements are demonstrated."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    cfg = RollingConfig(
        min_train_months=args.min_train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
        random_state=args.random_state,
        top_k=args.top_k,
        top_k_max_weight=args.top_k_cap,
        score_max_weight=args.score_cap,
        diversify_top_n=args.diversify_top_n,
        diversify_max_weight=args.diversify_cap,
    )

    panel = pd.read_csv(PROJECT_ROOT / "data" / "modeling" / "modeling_panel_filtered.csv", parse_dates=["month_end"])
    fold_metrics, test_predictions, fold_manifest, feature_sets = run_feature_set_experiments(panel, config=cfg)
    feature_summary = summarize_feature_set_comparison(fold_metrics)
    selected_by_model = select_best_feature_set_per_model(feature_summary)
    concentration_summary = evaluate_concentration_controls(
        test_predictions=test_predictions,
        selected_feature_set_by_model=selected_by_model,
        config=cfg,
    )

    report_text = _render_report(
        cfg=cfg,
        fold_manifest=fold_manifest,
        feature_summary=feature_summary,
        concentration_summary=concentration_summary,
        selected_feature_set_by_model=selected_by_model,
    )

    write_csv(feature_summary, PROJECT_ROOT / "reports" / "feature_set_comparison.csv")
    write_csv(concentration_summary, PROJECT_ROOT / "reports" / "concentration_control_summary.csv")
    write_text(report_text, PROJECT_ROOT / "reports" / "rolling_robustness_report.md")

    # Keep this available for manual debugging without adding scope to required outputs.
    _ = feature_sets  # silence lint-style tools; explicitly retained for future extension.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
