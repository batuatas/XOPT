"""Case-based conference synthesis for the locked robust 5Y scenario benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark_question_run import (
    _baseline_bundle,
    _build_context,
    _run_single_question,
)
from .benchmark_question_set import build_robust_benchmark_question_set
from .benchmark_regimes import fit_hybrid_regime_classifier
from .io import default_paths, load_active_artifacts


@dataclass(frozen=True)
class ConferenceCaseSpec:
    case_id: str
    case_role: str
    anchor_date: str
    question_id: str
    case_label: str
    why_selected: str


def _pick_cases() -> list[ConferenceCaseSpec]:
    return [
        ConferenceCaseSpec(
            case_id="upside_soft_landing",
            case_role="upside",
            anchor_date="2021-12-31",
            question_id="q_soft_landing",
            case_label="Upside case",
            why_selected="Largest positive benchmark-return response among the final conference questions, with only limited extra concentration.",
        ),
        ConferenceCaseSpec(
            case_id="breadth_return_with_breadth",
            case_role="breadth",
            anchor_date="2023-12-31",
            question_id="q_return_with_breadth",
            case_label="Breadth case",
            why_selected="Cleanest reduction in concentration with a visible effective-N improvement and only a modest return give-up.",
        ),
        ConferenceCaseSpec(
            case_id="adverse_higher_for_longer",
            case_role="adverse",
            anchor_date="2022-12-31",
            question_id="q_higher_for_longer",
            case_label="Adverse case",
            why_selected="Clearest restrictive macro narrative with a negative benchmark-return response under a tighter, more inflationary backdrop.",
        ),
    ]


def _delta_pp(value: float) -> float:
    return 100.0 * float(value)


def _pct(value: float) -> float:
    return 100.0 * float(value)


def _reconstruct_case(project_root: Path, classifier, spec: ConferenceCaseSpec) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    context = _build_context(project_root, spec.anchor_date, classifier)
    baseline = _baseline_bundle(context)
    questions = {q.spec.question_id: q for q in build_robust_benchmark_question_set(context, baseline, classifier)}
    question = questions[spec.question_id]
    run = _run_single_question(context, baseline, question)
    best_row = run["best_row"]

    baseline_eval = run["baseline_evaluation"]
    scenario_eval = best_row["evaluation"]
    baseline_weights = baseline_eval["weights_df"].copy()
    scenario_weights = scenario_eval["weights_df"].copy()

    base_state = np.asarray(context.anchor.current_base_state, dtype=float)
    scenario_state = np.asarray(best_row["state_vector"], dtype=float)
    shift_std = (scenario_state - base_state) / context.regularizer.bounds.scale

    top_idx = np.argsort(np.abs(shift_std))[::-1][:8]
    shift_rows = []
    for rank, idx in enumerate(top_idx, start=1):
        variable_name = context.state_spec.base_variables[idx]
        shift_rows.append(
            {
                "case_id": spec.case_id,
                "case_role": spec.case_role,
                "anchor_date": pd.Timestamp(spec.anchor_date),
                "question_id": spec.question_id,
                "question_label": question.spec.short_label,
                "variable_name": variable_name,
                "shift_rank": rank,
                "baseline_value": float(base_state[idx]),
                "scenario_value": float(scenario_state[idx]),
                "shift_std_units_signed": float(shift_std[idx]),
                "shift_std_units_abs": float(abs(shift_std[idx])),
            }
        )

    weight_rows = []
    for phase, frame in (("baseline", baseline_weights), ("scenario", scenario_weights)):
        work = frame.copy()
        work["case_id"] = spec.case_id
        work["case_role"] = spec.case_role
        work["anchor_date"] = pd.Timestamp(spec.anchor_date)
        work["question_id"] = spec.question_id
        work["case_phase"] = phase
        weight_rows.append(work)

    ordered_base = baseline_weights.sort_values("weight", ascending=False)
    ordered_scen = scenario_weights.sort_values("weight", ascending=False)
    regime_label = str(scenario_eval["regime_label"])
    interpretation = {
        "upside": "A looser, growth-supportive state raises long-run return without turning the benchmark into a radically different portfolio.",
        "breadth": "The benchmark can be made broader, but the gain comes from giving up some return rather than from a free diversification win.",
        "adverse": "A tighter, inflationary state trims return and nudges the benchmark toward a more defensive posture rather than a risk-on allocation.",
    }[spec.case_role]

    case_row = {
        "case_id": spec.case_id,
        "case_role": spec.case_role,
        "case_label": spec.case_label,
        "anchor_date": pd.Timestamp(spec.anchor_date),
        "question_id": spec.question_id,
        "question_label": question.spec.short_label,
        "regime_label": regime_label,
        "nfci_bucket": str(scenario_eval["nfci_bucket"]),
        "recession_overlay": str(scenario_eval["recession_overlay"]),
        "why_selected": spec.why_selected,
        "baseline_predicted_return": float(baseline_eval["portfolio_predicted_return"]),
        "scenario_predicted_return": float(scenario_eval["portfolio_predicted_return"]),
        "delta_predicted_return": float(scenario_eval["portfolio_predicted_return"] - baseline_eval["portfolio_predicted_return"]),
        "baseline_max_weight": float(baseline_eval["portfolio_max_weight"]),
        "scenario_max_weight": float(scenario_eval["portfolio_max_weight"]),
        "delta_max_weight": float(scenario_eval["portfolio_max_weight"] - baseline_eval["portfolio_max_weight"]),
        "baseline_effective_n": float(baseline_eval["portfolio_effective_n"]),
        "scenario_effective_n": float(scenario_eval["portfolio_effective_n"]),
        "delta_effective_n": float(scenario_eval["portfolio_effective_n"] - baseline_eval["portfolio_effective_n"]),
        "top_weight_sleeve_before": str(ordered_base["sleeve_id"].iloc[0]),
        "top_weight_sleeve_after": str(ordered_scen["sleeve_id"].iloc[0]),
        "eq_cn_weight_before": float(baseline_weights.loc[baseline_weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
        "eq_cn_weight_after": float(scenario_weights.loc[scenario_weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
        "short_interpretation": interpretation,
    }
    return case_row, pd.concat(weight_rows, ignore_index=True), pd.DataFrame(shift_rows)


def _build_report(case_summary: pd.DataFrame, macro_fingerprint: pd.DataFrame) -> str:
    top_shifts = (
        macro_fingerprint.loc[macro_fingerprint["shift_rank"].le(3)]
        .groupby("case_id")["variable_name"]
        .apply(lambda s: ", ".join(s.tolist()))
        .to_dict()
    )
    lines = [
        "# Scenario Case Conference Report",
        "",
        "## Scope",
        "- Active branch only: `v3_long_horizon_china`.",
        "- Active scenario object only: the locked robust 5Y benchmark (`best_60_predictor`).",
        "- These are scenario-conditioned, plausibility-regularized, anchor-local benchmark diagnostics for long-horizon SAA.",
        "",
        "## Selected Cases",
    ]
    for _, row in case_summary.iterrows():
        lines.extend(
            [
                f"### {row['case_label']}",
                f"- Anchor: `{pd.Timestamp(row['anchor_date']).date()}`",
                f"- Source question: `{row['question_label']}`",
                f"- Regime label: `{row['regime_label']}` with NFCI bucket `{row['nfci_bucket']}` and recession overlay `{row['recession_overlay']}`",
                f"- Benchmark response: predicted return {_pct(row['baseline_predicted_return']):.2f}% -> {_pct(row['scenario_predicted_return']):.2f}%, max weight {_pct(row['baseline_max_weight']):.2f}% -> {_pct(row['scenario_max_weight']):.2f}%, effective N {row['baseline_effective_n']:.2f} -> {row['scenario_effective_n']:.2f}",
                f"- Macro fingerprint: {top_shifts.get(row['case_id'], '')}",
                f"- Why selected: {row['why_selected']}",
                f"- Interpretation: {row['short_interpretation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Visual Story",
            "- Use the case overview figure first: it shows which scenarios matter at the benchmark level.",
            "- Use the allocation-change figure second: it proves the benchmark changes across the full 9-sleeve system, not just one highlighted sleeve.",
            "- Use the macro-fingerprint figure third: it explains what state shifts actually define the generated scenario.",
            "- Keep China implicit inside the full allocation figure. It is present, but it does not dominate any selected case.",
        ]
    )
    return "\n".join(lines)


def build_case_conference_outputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    root = project_root.resolve()
    paths = default_paths(root)
    artifacts = load_active_artifacts(paths)
    classifier = fit_hybrid_regime_classifier(artifacts["feature_master_monthly"], root)

    case_rows: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    shift_frames: list[pd.DataFrame] = []
    for spec in _pick_cases():
        case_row, weight_df, shift_df = _reconstruct_case(root, classifier, spec)
        case_rows.append(case_row)
        weight_frames.append(weight_df)
        shift_frames.append(shift_df)

    case_summary = pd.DataFrame(case_rows)
    case_summary["anchor_date"] = pd.to_datetime(case_summary["anchor_date"])
    case_summary = case_summary.sort_values(
        by="case_role",
        key=lambda s: s.map({"upside": 0, "breadth": 1, "adverse": 2}),
    ).reset_index(drop=True)
    weight_summary = pd.concat(weight_frames, ignore_index=True)
    macro_fingerprint = pd.concat(shift_frames, ignore_index=True)
    report_text = _build_report(case_summary, macro_fingerprint)
    return case_summary, weight_summary, macro_fingerprint, report_text
