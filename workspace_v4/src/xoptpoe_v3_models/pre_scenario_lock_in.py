"""Final v3 pre-scenario lock-in synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xoptpoe_v3_modeling.io import load_csv, write_csv, write_text
from xoptpoe_v3_models.data import default_paths


@dataclass(frozen=True)
class PreScenarioLockInOutputs:
    benchmark_stack: pd.DataFrame
    scenario_candidates: pd.DataFrame
    china_role: pd.DataFrame
    carryforward_decisions: pd.DataFrame
    manifest: pd.DataFrame
    report_text: str
    readiness_text: str


def _load_inputs(project_root: Path) -> dict[str, pd.DataFrame]:
    paths = default_paths(project_root)
    return {
        "pred_metrics": load_csv(paths.reports_dir / "prediction_benchmark_v3_metrics.csv"),
        "pred_rolling": load_csv(paths.reports_dir / "prediction_rolling_v3_summary.csv"),
        "pred_china": load_csv(paths.reports_dir / "china_prediction_diagnostics_v3.csv"),
        "portfolio_metrics": load_csv(paths.reports_dir / "portfolio_benchmark_v3_metrics.csv"),
        "portfolio_sleeve": load_csv(paths.reports_dir / "portfolio_benchmark_v3_by_sleeve.csv"),
        "portfolio_china": load_csv(paths.reports_dir / "china_portfolio_diagnostics_v3.csv"),
        "refine_metrics": load_csv(paths.reports_dir / "benchmark_refinement_v3_metrics.csv"),
        "refine_conc": load_csv(paths.reports_dir / "benchmark_refinement_v3_concentration.csv"),
        "refine_attr": load_csv(paths.reports_dir / "benchmark_refinement_v3_attribution.csv"),
        "refine_china": load_csv(paths.reports_dir / "china_refinement_diagnostics_v3.csv"),
        "split_manifest": load_csv(paths.data_out_dir / "split_manifest.csv"),
    }


def _first_row(df: pd.DataFrame, **filters: object) -> pd.Series:
    out = df.copy()
    for key, value in filters.items():
        out = out.loc[out[key].eq(value)]
    if out.empty:
        raise KeyError(f"no row found for {filters}")
    return out.iloc[0]


def _build_benchmark_stack(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pred_metrics = data["pred_metrics"]
    pred_rolling = data["pred_rolling"]
    refine_conc = data["refine_conc"]

    raw_pred = _first_row(pred_metrics, experiment_name="ridge__full_firstpass__separate_120")
    robust_pred = _first_row(pred_rolling, experiment_name="ridge__full_firstpass__separate_120")
    best_60 = _first_row(pred_metrics, experiment_name="elastic_net__full_firstpass__separate_60")
    best_60_roll = _first_row(pred_rolling, experiment_name="ridge__full_firstpass__separate_60")
    shared_roll = _first_row(pred_rolling, experiment_name="ridge__full_firstpass__shared_60_120")

    raw_port = _first_row(refine_conc, split="test", strategy_label="best_120_predictor")
    robust_port = _first_row(refine_conc, split="test", strategy_label="best_60_predictor")
    capped_tradeoff = _first_row(refine_conc, split="test", strategy_label="combined_std_120tilt_top_k_capped")
    pto = _first_row(refine_conc, split="test", strategy_label="pto_nn_signal")
    e2e = _first_row(refine_conc, split="test", strategy_label="e2e_nn_signal")

    rows = [
        {
            "stack_item": "strongest_raw_prediction_benchmark",
            "object_name": str(raw_pred["experiment_name"]),
            "object_type": "predictor",
            "horizon_scope": "120m",
            "status": "active benchmark",
            "selection_basis": "fixed-split overall raw prediction winner and strongest 120m benchmark",
            "key_metric_1": "validation_rmse",
            "key_value_1": float(raw_pred["validation_rmse"]),
            "key_metric_2": "test_rmse",
            "key_value_2": float(raw_pred["test_rmse"]),
            "source_artifact": "prediction_benchmark_v3_metrics.csv",
            "notes": "Raw and robust prediction winner are the same overall object in v3.",
        },
        {
            "stack_item": "strongest_robust_prediction_benchmark",
            "object_name": str(robust_pred["experiment_name"]),
            "object_type": "predictor",
            "horizon_scope": "120m",
            "status": "active robust benchmark",
            "selection_basis": "best rolling mean_test_rmse and strongest rolling stability",
            "key_metric_1": "mean_test_rmse",
            "key_value_1": float(robust_pred["mean_test_rmse"]),
            "key_metric_2": "mean_test_corr",
            "key_value_2": float(robust_pred["mean_test_corr"]),
            "source_artifact": "prediction_rolling_v3_summary.csv",
            "notes": "This is the cleanest robust prediction anchor.",
        },
        {
            "stack_item": "strongest_60m_prediction_benchmark",
            "object_name": str(best_60["experiment_name"]),
            "object_type": "predictor",
            "horizon_scope": "60m",
            "status": "active benchmark",
            "selection_basis": "fixed-split 60m validation winner",
            "key_metric_1": "validation_rmse",
            "key_value_1": float(best_60["validation_rmse"]),
            "key_metric_2": "test_corr",
            "key_value_2": float(best_60["test_corr"]),
            "source_artifact": "prediction_benchmark_v3_metrics.csv",
            "notes": "Carry with ridge__full_firstpass__separate_60 as robustness cross-check.",
        },
        {
            "stack_item": "strongest_60m_prediction_robustness_crosscheck",
            "object_name": str(best_60_roll["experiment_name"]),
            "object_type": "predictor",
            "horizon_scope": "60m",
            "status": "reference ceiling",
            "selection_basis": "rolling 60m winner",
            "key_metric_1": "mean_test_rmse",
            "key_value_1": float(best_60_roll["mean_test_rmse"]),
            "key_metric_2": "mean_test_corr",
            "key_value_2": float(best_60_roll["mean_test_corr"]),
            "source_artifact": "prediction_rolling_v3_summary.csv",
            "notes": "Cross-check for 60m robustness, not the default named 60m benchmark.",
        },
        {
            "stack_item": "strongest_shared_prediction_benchmark",
            "object_name": str(shared_roll["experiment_name"]),
            "object_type": "predictor",
            "horizon_scope": "60m+120m",
            "status": "comparator only",
            "selection_basis": "shared rolling winner",
            "key_metric_1": "mean_test_rmse",
            "key_value_1": float(shared_roll["mean_test_rmse"]),
            "key_metric_2": "mean_test_corr",
            "key_value_2": float(shared_roll["mean_test_corr"]),
            "source_artifact": "prediction_rolling_v3_summary.csv",
            "notes": "Shared remains ablation/reference, not the default carry-forward winner.",
        },
        {
            "stack_item": "strongest_raw_portfolio_benchmark",
            "object_name": str(raw_port["strategy_label"]),
            "object_type": "portfolio",
            "horizon_scope": "120m-led",
            "status": "reference ceiling",
            "selection_basis": "highest test-Sharpe supervised portfolio candidate",
            "key_metric_1": "sharpe",
            "key_value_1": float(raw_port["sharpe"]),
            "key_metric_2": "avg_return",
            "key_value_2": float(raw_port["avg_return"]),
            "source_artifact": "benchmark_refinement_v3_concentration.csv",
            "notes": "Performance ceiling only; too concentrated for primary carry-forward use.",
        },
        {
            "stack_item": "strongest_robust_portfolio_benchmark",
            "object_name": str(robust_port["strategy_label"]),
            "object_type": "portfolio",
            "horizon_scope": "60m-led",
            "status": "active robust benchmark",
            "selection_basis": "highest-Sharpe supervised candidate passing explicit concentration screen on test",
            "key_metric_1": "sharpe",
            "key_value_1": float(robust_port["sharpe"]),
            "key_metric_2": "avg_max_weight",
            "key_value_2": float(robust_port["avg_max_weight"]),
            "source_artifact": "benchmark_refinement_v3_concentration.csv",
            "notes": "Primary pre-scenario benchmark to beat, with explicit stability caveat.",
        },
        {
            "stack_item": "concentration_controlled_supervised_variant",
            "object_name": str(capped_tradeoff["strategy_label"]),
            "object_type": "portfolio",
            "horizon_scope": "combined 60m/120m",
            "status": "comparator only",
            "selection_basis": "best 120-led concentration-control tradeoff",
            "key_metric_1": "sharpe",
            "key_value_1": float(capped_tradeoff["sharpe"]),
            "key_metric_2": "avg_max_weight",
            "key_value_2": float(capped_tradeoff["avg_max_weight"]),
            "source_artifact": "benchmark_refinement_v3_concentration.csv",
            "notes": "Useful comparator for concentration reduction without collapsing raw return.",
        },
        {
            "stack_item": "pto_status",
            "object_name": str(pto["strategy_label"]),
            "object_type": "neural comparator",
            "horizon_scope": "60m+120m shared",
            "status": "comparator only",
            "selection_basis": "paper-faithful neural reference",
            "key_metric_1": "sharpe",
            "key_value_1": float(pto["sharpe"]),
            "key_metric_2": "avg_return",
            "key_value_2": float(pto["avg_return"]),
            "source_artifact": "benchmark_refinement_v3_concentration.csv",
            "notes": "Keep as supervised-neural reference, not as an active contender.",
        },
        {
            "stack_item": "e2e_status",
            "object_name": str(e2e["strategy_label"]),
            "object_type": "neural comparator",
            "horizon_scope": "60m+120m shared",
            "status": "comparator only",
            "selection_basis": "decision-focused neural reference",
            "key_metric_1": "sharpe",
            "key_value_1": float(e2e["sharpe"]),
            "key_metric_2": "avg_return",
            "key_value_2": float(e2e["avg_return"]),
            "source_artifact": "benchmark_refinement_v3_concentration.csv",
            "notes": "Useful because it shows risk-control-driven gains without winning on prediction or raw return.",
        },
        {
            "stack_item": "china_status",
            "object_name": "EQ_CN",
            "object_type": "active sleeve",
            "horizon_scope": "both",
            "status": "active benchmark",
            "selection_basis": "active v3 universe design",
            "key_metric_1": "active_branch_sleeves",
            "key_value_1": 9.0,
            "key_metric_2": "carryforward_decision",
            "key_value_2": 1.0,
            "source_artifact": "final_data_design_report.md",
            "notes": "Remain active regardless of current marginal portfolio weight.",
        },
    ]
    return pd.DataFrame(rows)


def _build_china_role(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pred_china = data["pred_china"]
    refine_china = data["refine_china"]

    eq_cn_60 = _first_row(pred_china, diagnostic_type="eq_cn_selected_winner", horizon_mode="separate_60")
    eq_cn_120 = _first_row(pred_china, diagnostic_type="eq_cn_selected_winner", horizon_mode="separate_120")
    eq_cn_shared = _first_row(pred_china, diagnostic_type="eq_cn_selected_winner", horizon_mode="shared_60_120")

    drop_120 = pred_china.loc[pred_china["diagnostic_type"].eq("china_feature_drop_by_sleeve") & pred_china["horizon_mode"].eq("separate_120")]
    drop_60 = pred_china.loc[pred_china["diagnostic_type"].eq("china_feature_drop_by_sleeve") & pred_china["horizon_mode"].eq("separate_60")]
    raw_port = _first_row(refine_china, split="test", strategy_label="best_120_predictor")
    robust_port = _first_row(refine_china, split="test", strategy_label="best_60_predictor")
    breadth_port = _first_row(refine_china, split="test", strategy_label="combined_std_120tilt_top_k_capped")

    rows = [
        {
            "assessment_area": "structural",
            "question": "did_adding_eq_cn_matter_structurally",
            "answer": "yes",
            "evidence": "v3 is a 9-sleeve active branch with EQ_CN fully included in prediction, portfolio, and diagnostics outputs.",
        },
        {
            "assessment_area": "prediction",
            "question": "did_eq_cn_matter_predictively",
            "answer": "mixed_modest_yes",
            "evidence": (
                f"EQ_CN selected-winner metrics: 60m rmse={eq_cn_60['rmse']:.4f}, corr={eq_cn_60['corr']:.4f}; "
                f"120m rmse={eq_cn_120['rmse']:.4f}, corr={eq_cn_120['corr']:.4f}. "
                f"China-feature-drop hurts 120m EQ_CN/EQ_EM/EQ_EZ/EQ_JP/EQ_US more consistently than 60m."
            ),
        },
        {
            "assessment_area": "prediction",
            "question": "do_china_features_help_only_eq_cn",
            "answer": "no",
            "evidence": (
                "At 120m, removing China features worsens RMSE for EQ_CN, EQ_EM, EQ_EZ, EQ_JP, and EQ_US; "
                "at 60m, effects are mixed and less robust."
            ),
        },
        {
            "assessment_area": "allocation",
            "question": "does_eq_cn_matter_in_raw_winner_allocations",
            "answer": "mostly_no",
            "evidence": (
                f"best_120_predictor test EQ_CN avg_weight={raw_port['avg_weight']:.4f}, "
                f"nonzero_share={raw_port['nonzero_allocation_share']:.4f}, "
                f"active_contribution_vs_equal={raw_port['total_active_contribution_vs_equal_weight']:.4f}."
            ),
        },
        {
            "assessment_area": "allocation",
            "question": "does_eq_cn_matter_in_robust_winner_allocations",
            "answer": "small_but_nonzero",
            "evidence": (
                f"best_60_predictor test EQ_CN avg_weight={robust_port['avg_weight']:.4f}, "
                f"nonzero_share={robust_port['nonzero_allocation_share']:.4f}, "
                f"active_contribution_vs_equal={robust_port['total_active_contribution_vs_equal_weight']:.4f}."
            ),
        },
        {
            "assessment_area": "allocation",
            "question": "does_china_help_diversification_in_carryforward_benchmarks",
            "answer": "not_materially",
            "evidence": (
                f"In the raw and robust carryforward winners EQ_CN stays underweight; "
                f"the stronger concentration-control comparator raises EQ_CN only to avg_weight={breadth_port['avg_weight']:.4f}."
            ),
        },
        {
            "assessment_area": "final_role",
            "question": "is_eq_cn_central_or_marginal",
            "answer": "marginal_but_active",
            "evidence": "EQ_CN is a valid active sleeve and a useful structural addition, but it is still mostly marginal in the carry-forward allocation winners.",
        },
        {
            "assessment_area": "decision",
            "question": "should_eq_cn_remain_active",
            "answer": "yes",
            "evidence": "EQ_CN remains part of the active v3 system unless a concrete implementation bug is found; no such bug was found.",
        },
    ]
    return pd.DataFrame(rows)


def _build_carryforward_decisions(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    refine_conc = data["refine_conc"]
    pred_metrics = data["pred_metrics"]
    rows = [
        {
            "object_name": "best_60_predictor",
            "object_type": "portfolio benchmark",
            "status": "active robust benchmark",
            "horizon_scope": "60m-led",
            "reason": "highest-Sharpe supervised candidate passing the explicit concentration screen on test",
            "source_artifact": "benchmark_refinement_v3_concentration.csv",
        },
        {
            "object_name": "best_120_predictor",
            "object_type": "portfolio benchmark",
            "status": "reference ceiling",
            "horizon_scope": "120m-led",
            "reason": "highest raw supervised portfolio result, but too concentrated to carry forward as the primary benchmark",
            "source_artifact": "benchmark_refinement_v3_concentration.csv",
        },
        {
            "object_name": "ridge__full_firstpass__shared_60_120 / best_shared_predictor",
            "object_type": "shared predictor and portfolio reference",
            "status": "comparator only",
            "horizon_scope": "60m+120m",
            "reason": "shared remains useful as ablation/reference, not as the default carry-forward winner",
            "source_artifact": "prediction_rolling_v3_summary.csv",
        },
        {
            "object_name": "PTO",
            "object_type": "neural comparator",
            "status": "comparator only",
            "horizon_scope": "60m+120m shared",
            "reason": "paper-faithful neural reference; weaker than supervised benchmarks on prediction and raw portfolio return",
            "source_artifact": "pto_report.md",
        },
        {
            "object_name": "E2E",
            "object_type": "neural comparator",
            "status": "comparator only",
            "horizon_scope": "60m+120m shared",
            "reason": "still useful because it shows risk-control-driven behavior, but it does not beat the supervised benchmark stack",
            "source_artifact": "e2e_report.md",
        },
        {
            "object_name": "combined_std_120tilt_top_k_capped",
            "object_type": "concentration-controlled supervised variant",
            "status": "comparator only",
            "horizon_scope": "combined 60m/120m",
            "reason": "best compact concentration-control tradeoff; useful to interrogate benchmark robustness but not the locked primary benchmark",
            "source_artifact": "benchmark_refinement_v3_concentration.csv",
        },
    ]
    return pd.DataFrame(rows)


def _build_scenario_candidates(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        {
            "candidate_name": "ridge__full_firstpass__separate_120",
            "candidate_type": "predictor",
            "horizon_relevance": "120m",
            "carryforward_role": "active benchmark",
            "why_explain": "strongest raw and robust prediction object overall",
            "what_we_learn": "what drives the cleanest long-horizon predictive signal",
            "behavior_type": "genuine predictive signal",
        },
        {
            "candidate_name": "elastic_net__full_firstpass__separate_60",
            "candidate_type": "predictor",
            "horizon_relevance": "60m",
            "carryforward_role": "active benchmark",
            "why_explain": "named 60m benchmark used by the active best_60 portfolio signal",
            "what_we_learn": "what drives the cleaner medium-long-horizon signal that survives concentration screening at portfolio level",
            "behavior_type": "genuine predictive signal",
        },
        {
            "candidate_name": "best_60_predictor",
            "candidate_type": "portfolio",
            "horizon_relevance": "60m-led",
            "carryforward_role": "active robust benchmark",
            "why_explain": "locked robust portfolio benchmark",
            "what_we_learn": "how predictive signal translates into a lower-concentration SAA allocation",
            "behavior_type": "genuine signal plus moderated concentration",
        },
        {
            "candidate_name": "best_120_predictor",
            "candidate_type": "portfolio",
            "horizon_relevance": "120m-led",
            "carryforward_role": "reference ceiling",
            "why_explain": "strongest raw portfolio result and concentration failure case",
            "what_we_learn": "whether the strongest 120m signal is economically intuitive or mostly an EQ_US concentration channel",
            "behavior_type": "genuine signal mixed with concentrated allocation behavior",
        },
        {
            "candidate_name": "combined_std_120tilt_top_k_capped",
            "candidate_type": "portfolio",
            "horizon_relevance": "both",
            "carryforward_role": "comparator only",
            "why_explain": "best compact concentration-control tradeoff on the 120-led side",
            "what_we_learn": "how much concentration can be reduced before performance degrades materially",
            "behavior_type": "mixture of signal use and explicit breadth control",
        },
        {
            "candidate_name": "e2e_nn_signal",
            "candidate_type": "neural comparator",
            "horizon_relevance": "both",
            "carryforward_role": "comparator only",
            "why_explain": "most informative neural reference",
            "what_we_learn": "whether apparent gains come from real signal or from defensive risk control",
            "behavior_type": "mostly risk-control-driven behavior",
        },
    ]
    return pd.DataFrame(rows)


def _build_manifest(
    *,
    benchmark_stack: pd.DataFrame,
    china_role: pd.DataFrame,
    carryforward: pd.DataFrame,
    scenario_candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active_paths = [
        ("active_data_root", "data/final_v3_long_horizon_china/"),
        ("active_modeling_root", "data/modeling_v3/"),
        ("active_reports_root", "reports/v3_long_horizon_china/"),
    ]
    for name, value in active_paths:
        rows.append(
            {
                "manifest_section": "active_paths",
                "item_name": name,
                "item_value": value,
                "status": "active",
                "source_artifact": "",
                "notes": "",
            }
        )
    for row in benchmark_stack.itertuples(index=False):
        rows.append(
            {
                "manifest_section": "benchmark_stack",
                "item_name": row.stack_item,
                "item_value": row.object_name,
                "status": row.status,
                "source_artifact": row.source_artifact,
                "notes": row.notes,
            }
        )
    for row in carryforward.itertuples(index=False):
        rows.append(
            {
                "manifest_section": "carryforward_decision",
                "item_name": row.object_name,
                "item_value": row.object_type,
                "status": row.status,
                "source_artifact": row.source_artifact,
                "notes": row.reason,
            }
        )
    for row in scenario_candidates.itertuples(index=False):
        rows.append(
            {
                "manifest_section": "scenario_candidate",
                "item_name": row.candidate_name,
                "item_value": row.candidate_type,
                "status": row.carryforward_role,
                "source_artifact": "",
                "notes": row.what_we_learn,
            }
        )
    for row in china_role.itertuples(index=False):
        rows.append(
            {
                "manifest_section": "china_role",
                "item_name": row.question,
                "item_value": row.answer,
                "status": "informational",
                "source_artifact": "",
                "notes": row.evidence,
            }
        )
    rows.append(
        {
            "manifest_section": "major_caveat",
            "item_name": "robust_screen_both_validation_test",
            "item_value": "0 candidates passed",
            "status": "caveat",
            "source_artifact": "benchmark_lock_in_v3.md",
            "notes": "Benchmark lock-in is governance-quality and cautious, not proof of a fully stable concentration-controlled winner.",
        }
    )
    return pd.DataFrame(rows)


def _render_report(
    *,
    benchmark_stack: pd.DataFrame,
    china_role: pd.DataFrame,
    carryforward: pd.DataFrame,
    scenario_candidates: pd.DataFrame,
) -> str:
    raw_pred = _first_row(benchmark_stack, stack_item="strongest_raw_prediction_benchmark")
    robust_pred = _first_row(benchmark_stack, stack_item="strongest_robust_prediction_benchmark")
    raw_port = _first_row(benchmark_stack, stack_item="strongest_raw_portfolio_benchmark")
    robust_port = _first_row(benchmark_stack, stack_item="strongest_robust_portfolio_benchmark")
    shared = _first_row(benchmark_stack, stack_item="strongest_shared_prediction_benchmark")
    pto = _first_row(benchmark_stack, stack_item="pto_status")
    e2e = _first_row(benchmark_stack, stack_item="e2e_status")
    lines = [
        "# XOPTPOE v3 Pre-Scenario Lock-In Report",
        "",
        "## Scope",
        "- Active branch only: v3_long_horizon_china.",
        "- This is a synthesis and lock-in pass. No new data design, target construction, or benchmark sweep was introduced.",
        "- These remain long-horizon SAA decision diagnostics, not a clean overlapping-month wealth backtest.",
        "",
        "## Final Benchmark Synthesis",
        f"- strongest raw prediction benchmark: `{raw_pred['object_name']}`",
        f"- strongest robust prediction benchmark: `{robust_pred['object_name']}`",
        f"- strongest raw portfolio benchmark: `{raw_port['object_name']}`",
        f"- strongest robust portfolio benchmark: `{robust_port['object_name']}`",
        f"- strongest shared benchmark: `{shared['object_name']}` -> {shared['status']}",
        f"- PTO status: {pto['status']}",
        f"- E2E status: {e2e['status']}",
        "",
        "## Raw Versus Robust",
        "- Prediction side: raw and robust winner are effectively the same overall object at 120m.",
        "- Portfolio side: raw and robust winners differ. The raw winner is too concentrated; the robust winner is the carry-forward benchmark.",
        "",
        "## Final Carry-Forward Decision",
    ]
    for row in carryforward.itertuples(index=False):
        lines.append(f"- {row.object_name}: {row.status} | {row.reason}")
    lines += [
        "",
        "## China Interpretation",
    ]
    for row in china_role.itertuples(index=False):
        lines.append(f"- {row.question}: {row.answer} | {row.evidence}")
    lines += [
        "",
        "## Scenario Candidate Set",
    ]
    for row in scenario_candidates.itertuples(index=False):
        lines.append(
            f"- {row.candidate_name}: type={row.candidate_type}, horizon={row.horizon_relevance}, role={row.carryforward_role}, "
            f"learn={row.what_we_learn}, behavior={row.behavior_type}"
        )
    lines += [
        "",
        "## Practical Decision",
        "- Carry forward the robust 60m-led portfolio benchmark as the primary governance object.",
        "- Keep the raw 120m-led portfolio benchmark as the ceiling/reference object to explain concentration.",
        "- Carry the 120m prediction benchmark and the 60m prediction benchmark into the next stage because the next stage should explain both predictors and portfolios.",
        "",
    ]
    return "\n".join(lines)


def _render_readiness(
    *,
    benchmark_stack: pd.DataFrame,
    china_role: pd.DataFrame,
) -> str:
    raw_port = _first_row(benchmark_stack, stack_item="strongest_raw_portfolio_benchmark")
    robust_port = _first_row(benchmark_stack, stack_item="strongest_robust_portfolio_benchmark")
    lines = [
        "# XOPTPOE v3 Pre-Scenario Readiness",
        "",
        "## Readiness Decision",
        "- Ready for scenario generation as disciplined project governance: yes.",
        "- Ready as a claim that benchmark robustness is fully solved: no.",
        "",
        "## Why",
        f"- raw portfolio benchmark `{raw_port['object_name']}` is still too concentrated to explain as the only primary object.",
        f"- robust portfolio benchmark `{robust_port['object_name']}` is a better carry-forward object, but no candidate passed the explicit robust screen on both validation and test.",
        "- This means the next stage should start from a cautious benchmark stack, not from a claim of final scientific certainty.",
        "",
        "## Next-Stage Focus",
        "- explain both predictors and portfolios",
        "- focus on both 60m and 120m, with different roles",
        "- use the robust benchmark as the primary benchmark to beat",
        "- keep the raw benchmark as the concentration/reference ceiling",
        "",
        "## China",
        "- EQ_CN remains an active sleeve in the system.",
        "- China is still mostly marginal in the carry-forward winners, but that is not a reason to remove it.",
        "",
    ]
    return "\n".join(lines)


def run_pre_scenario_lock_in_v3(*, project_root: Path) -> PreScenarioLockInOutputs:
    root = project_root.resolve()
    data = _load_inputs(root)
    benchmark_stack = _build_benchmark_stack(data)
    china_role = _build_china_role(data)
    carryforward = _build_carryforward_decisions(data)
    scenario_candidates = _build_scenario_candidates(data)
    manifest = _build_manifest(
        benchmark_stack=benchmark_stack,
        china_role=china_role,
        carryforward=carryforward,
        scenario_candidates=scenario_candidates,
    )
    report_text = _render_report(
        benchmark_stack=benchmark_stack,
        china_role=china_role,
        carryforward=carryforward,
        scenario_candidates=scenario_candidates,
    )
    readiness_text = _render_readiness(
        benchmark_stack=benchmark_stack,
        china_role=china_role,
    )
    return PreScenarioLockInOutputs(
        benchmark_stack=benchmark_stack,
        scenario_candidates=scenario_candidates,
        china_role=china_role,
        carryforward_decisions=carryforward,
        manifest=manifest,
        report_text=report_text,
        readiness_text=readiness_text,
    )


def write_pre_scenario_lock_in_v3_outputs(*, project_root: Path, outputs: PreScenarioLockInOutputs) -> None:
    paths = default_paths(project_root)
    write_text(outputs.report_text, paths.reports_dir / "pre_scenario_lock_in_v3_report.md")
    write_text(outputs.readiness_text, paths.reports_dir / "pre_scenario_readiness_v3.md")
    write_csv(outputs.benchmark_stack, paths.reports_dir / "final_benchmark_stack_v3.csv")
    write_csv(outputs.scenario_candidates, paths.reports_dir / "scenario_candidate_summary_v3.csv")
    write_csv(outputs.china_role, paths.reports_dir / "china_role_final_v3.csv")
    write_csv(outputs.carryforward_decisions, paths.reports_dir / "model_carryforward_decision_v3.csv")
    write_csv(outputs.manifest, paths.data_out_dir / "final_benchmark_manifest_v3.csv")
