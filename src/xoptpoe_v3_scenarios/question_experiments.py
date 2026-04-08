"""Focused conference-grade scenario question experiments for v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from xoptpoe_v3_models.data import SLEEVE_ORDER

from .experiments import (
    ANCHOR_DATES,
    _accepted_indices,
    _build_context,
    _forward_difference_gradient,
    _portfolio_for_state,
)
from .io import default_paths, load_active_artifacts
from .mala import run_bounded_mala
from .question_sets import (
    DEFENSIVE_SLEEVES,
    RISK_ON_SLEEVES,
    ScenarioQuestionRunSpec,
    build_question_set,
)
from .regimes import fit_regime_classifier, regime_manifest_df


@dataclass(frozen=True)
class ScenarioQuestionOutputs:
    framework_report: str
    question_manifest: pd.DataFrame
    regime_manifest: pd.DataFrame
    question_results: pd.DataFrame
    regime_summary: pd.DataFrame
    conference_notes: str
    state_shift_summary: pd.DataFrame
    selected_questions: pd.DataFrame


def _weight(weights_df: pd.DataFrame, sleeve_id: str) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].eq(sleeve_id), "weight"].iloc[0])


def _pred(weights_df: pd.DataFrame, sleeve_id: str) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].eq(sleeve_id), "predicted_return"].iloc[0])


def _rank(weights_df: pd.DataFrame, sleeve_id: str) -> int:
    ordered = weights_df.sort_values("weight", ascending=False).reset_index(drop=True)
    return int(ordered.index[ordered["sleeve_id"].eq(sleeve_id)][0] + 1)


def _share(weights_df: pd.DataFrame, sleeves: tuple[str, ...]) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].isin(list(sleeves)), "weight"].sum())


def _single_portfolio_eval(context, classifier, state_vector: np.ndarray, portfolio_id: str) -> dict[str, object]:
    weights, summary = _portfolio_for_state(context, state_vector, portfolio_id)
    regime = classifier.classify(state_vector)
    out: dict[str, object] = {
        **summary,
        **regime,
        "weights_df": weights,
        "top_weight_sleeve": str(weights.sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
        "risky_share": _share(weights, RISK_ON_SLEEVES),
        "defensive_share": _share(weights, DEFENSIVE_SLEEVES),
        "eq_cn_weight": _weight(weights, "EQ_CN"),
        "eq_cn_predicted_return": _pred(weights, "EQ_CN"),
        "eq_cn_rank": _rank(weights, "EQ_CN"),
    }
    for sleeve in SLEEVE_ORDER:
        out[f"weight_{sleeve}"] = _weight(weights, sleeve)
        out[f"pred_{sleeve}"] = _pred(weights, sleeve)
    out["soft_landing_alignment"] = float(
        -(
            (regime["growth_score"] - 0.8) ** 2
            + (regime["inflation_score"] - 0.0) ** 2
            + (regime["stress_score"] + 0.8) ** 2
            + (regime["rates_score"] - 0.2) ** 2
        )
    )
    out["higher_for_longer_alignment"] = float(
        -(
            (regime["growth_score"] + 0.4) ** 2
            + (regime["inflation_score"] - 0.8) ** 2
            + (regime["stress_score"] - 0.5) ** 2
            + (regime["rates_score"] - 0.9) ** 2
        )
    )
    return out


def _dual_portfolio_eval(context, classifier, state_vector: np.ndarray) -> dict[str, object]:
    w60, s60 = _portfolio_for_state(context, state_vector, "best_60_predictor")
    w120, s120 = _portfolio_for_state(context, state_vector, "best_120_predictor")
    merged = w60.merge(w120, on=["month_end", "sleeve_id"], suffixes=("_60", "_120"), validate="1:1")
    regime = classifier.classify(state_vector)
    return {
        **regime,
        "allocation_gap_l1": float(np.abs(merged["weight_60"] - merged["weight_120"]).sum()),
        "return_gap": float(s60["portfolio_predicted_return"] - s120["portfolio_predicted_return"]),
        "abs_return_gap": float(abs(s60["portfolio_predicted_return"] - s120["portfolio_predicted_return"])),
        "summary_60": s60,
        "summary_120": s120,
        "weights_60": w60,
        "weights_120": w120,
        "eq_cn_weight_60": _weight(w60, "EQ_CN"),
        "eq_cn_weight_120": _weight(w120, "EQ_CN"),
        "top_weight_sleeve_60": str(w60.sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
        "top_weight_sleeve_120": str(w120.sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
    }


def _baseline_bundle(context, classifier) -> dict[str, object]:
    state = np.asarray(context.anchor.current_base_state, dtype=float)
    return {
        "state": state,
        "best_60_predictor": _single_portfolio_eval(context, classifier, state, "best_60_predictor"),
        "best_120_predictor": _single_portfolio_eval(context, classifier, state, "best_120_predictor"),
        "pair": _dual_portfolio_eval(context, classifier, state),
    }


def _evaluate_question(context, classifier, question: ScenarioQuestionRunSpec, state_vector: np.ndarray) -> dict[str, object]:
    if question.spec.question_type == "single_portfolio":
        return _single_portfolio_eval(context, classifier, state_vector, question.spec.candidate_name)
    if question.spec.question_type == "dual_portfolio":
        return _dual_portfolio_eval(context, classifier, state_vector)
    raise ValueError(f"Unsupported question_type: {question.spec.question_type}")


def _run_single_question(context, classifier, baseline_bundle: dict[str, object], question: ScenarioQuestionRunSpec) -> dict[str, object]:
    base_state = np.asarray(baseline_bundle["state"], dtype=float)
    seed_payload = f"{context.anchor.month_end.date()}::{question.spec.question_id}".encode("utf-8")
    random_seed = int(zlib.adler32(seed_payload) % (2**32 - 1))
    total_energy = lambda x: float(question.probe.energy(context.regularizer.project(x)) + context.regularizer.total_energy(context.regularizer.project(x)))
    gradient = lambda x: _forward_difference_gradient(total_energy, context.regularizer.project(x), step=1e-3)
    chain = run_bounded_mala(
        start=base_state,
        energy_fn=total_energy,
        project_fn=context.regularizer.project,
        gradient_fn=gradient,
        step_size=question.spec.step_size,
        n_steps=question.spec.steps,
        random_seed=random_seed,
    )
    accepted = _accepted_indices(chain.states)
    retained = accepted if accepted else [len(chain.states) - 1]
    eval_rows = []
    for idx in retained:
        state = np.asarray(chain.states[idx], dtype=float)
        evaluation = _evaluate_question(context, classifier, question, state)
        eval_rows.append(
            {
                "chain_index": int(idx),
                "state_vector": state,
                "evaluation": evaluation,
                "selection_metric": float(question.selection_metric(evaluation)),
                "probe_energy": float(question.probe.energy(state)),
                "plausibility_energy": float(context.regularizer.total_energy(state)),
                "total_energy": float(total_energy(state)),
            }
        )
    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        raise ValueError(f"No retained states for {question.spec.question_id} at {context.anchor.month_end.date()}")
    ascending = question.spec.response_direction == "min"
    best_row = eval_df.sort_values(["selection_metric", "total_energy"], ascending=[ascending, True]).iloc[0]
    base_eval = baseline_bundle[question.spec.candidate_name] if question.spec.question_type == "single_portfolio" else baseline_bundle["pair"]
    return {
        "baseline_evaluation": base_eval,
        "best_row": best_row,
        "scenario_df": eval_df,
        "accepted_count": int(round(chain.acceptance_rate * question.spec.steps)),
    }


def _top_shift_summary(context, state_vector: np.ndarray, *, top_n: int = 5) -> list[tuple[str, float]]:
    shifts = (np.asarray(state_vector, dtype=float) - np.asarray(context.anchor.current_base_state, dtype=float)) / context.regularizer.bounds.scale
    ordered = pd.Series(shifts, index=list(context.state_spec.base_variables)).abs().sort_values(ascending=False).head(top_n)
    return [(name, float(shifts[list(context.state_spec.base_variables).index(name)])) for name in ordered.index]


def _question_result_row(anchor_date: pd.Timestamp, question: ScenarioQuestionRunSpec, baseline_eval: dict[str, object], scenario_eval: dict[str, object], *, scenario_count: int, plausibility_metric: float) -> dict[str, object]:
    if question.spec.question_type == "single_portfolio":
        row = {
            "anchor_date": anchor_date,
            "question_id": question.spec.question_id,
            "question_family": question.spec.question_family,
            "short_label": question.spec.short_label,
            "candidate_name": question.spec.candidate_name,
            "candidate_2": question.spec.candidate_2,
            "baseline_primary_metric": float(question.selection_metric(baseline_eval)),
            "scenario_primary_metric": float(question.selection_metric(scenario_eval)),
            "baseline_predicted_return": float(baseline_eval["portfolio_predicted_return"]),
            "scenario_predicted_return": float(scenario_eval["portfolio_predicted_return"]),
            "baseline_max_weight": float(baseline_eval["portfolio_max_weight"]),
            "scenario_max_weight": float(scenario_eval["portfolio_max_weight"]),
            "baseline_effective_n": float(baseline_eval["portfolio_effective_n"]),
            "scenario_effective_n": float(scenario_eval["portfolio_effective_n"]),
            "eq_cn_weight_before": float(baseline_eval["eq_cn_weight"]),
            "eq_cn_weight_after": float(scenario_eval["eq_cn_weight"]),
            "eq_cn_rank_before": int(baseline_eval["eq_cn_rank"]),
            "eq_cn_rank_after": int(scenario_eval["eq_cn_rank"]),
            "top_weight_sleeve_before": str(baseline_eval["top_weight_sleeve"]),
            "top_weight_sleeve_after": str(scenario_eval["top_weight_sleeve"]),
            "scenario_regime_label": str(scenario_eval["regime_label"]),
            "growth_bucket": str(scenario_eval["growth_bucket"]),
            "inflation_bucket": str(scenario_eval["inflation_bucket"]),
            "stress_bucket": str(scenario_eval["stress_bucket"]),
            "rates_bucket": str(scenario_eval["rates_bucket"]),
            "scenario_count": int(scenario_count),
            "plausibility_metric": float(plausibility_metric),
            "notes": question.spec.notes,
        }
        row["delta_predicted_return"] = row["scenario_predicted_return"] - row["baseline_predicted_return"]
        row["delta_max_weight"] = row["scenario_max_weight"] - row["baseline_max_weight"]
        row["delta_effective_n"] = row["scenario_effective_n"] - row["baseline_effective_n"]
        row["delta_eq_cn_weight"] = row["eq_cn_weight_after"] - row["eq_cn_weight_before"]
        for sleeve in SLEEVE_ORDER:
            row[f"weight_before_{sleeve}"] = float(baseline_eval[f"weight_{sleeve}"])
            row[f"weight_after_{sleeve}"] = float(scenario_eval[f"weight_{sleeve}"])
            row[f"pred_before_{sleeve}"] = float(baseline_eval[f"pred_{sleeve}"])
            row[f"pred_after_{sleeve}"] = float(scenario_eval[f"pred_{sleeve}"])
        return row
    row = {
        "anchor_date": anchor_date,
        "question_id": question.spec.question_id,
        "question_family": question.spec.question_family,
        "short_label": question.spec.short_label,
        "candidate_name": question.spec.candidate_name,
        "candidate_2": question.spec.candidate_2,
        "baseline_primary_metric": float(question.selection_metric(baseline_eval)),
        "scenario_primary_metric": float(question.selection_metric(scenario_eval)),
        "baseline_predicted_return": float(baseline_eval["summary_60"]["portfolio_predicted_return"]),
        "scenario_predicted_return": float(scenario_eval["summary_60"]["portfolio_predicted_return"]),
        "baseline_max_weight": float(baseline_eval["summary_60"]["portfolio_max_weight"]),
        "scenario_max_weight": float(scenario_eval["summary_60"]["portfolio_max_weight"]),
        "baseline_effective_n": float(baseline_eval["summary_60"]["portfolio_effective_n"]),
        "scenario_effective_n": float(scenario_eval["summary_60"]["portfolio_effective_n"]),
        "eq_cn_weight_before": float(baseline_eval["eq_cn_weight_60"]),
        "eq_cn_weight_after": float(scenario_eval["eq_cn_weight_60"]),
        "eq_cn_rank_before": int(_rank(baseline_eval["weights_60"], "EQ_CN")),
        "eq_cn_rank_after": int(_rank(scenario_eval["weights_60"], "EQ_CN")),
        "top_weight_sleeve_before": str(baseline_eval["top_weight_sleeve_60"]),
        "top_weight_sleeve_after": str(scenario_eval["top_weight_sleeve_60"]),
        "scenario_regime_label": str(scenario_eval["regime_label"]),
        "growth_bucket": str(scenario_eval["growth_bucket"]),
        "inflation_bucket": str(scenario_eval["inflation_bucket"]),
        "stress_bucket": str(scenario_eval["stress_bucket"]),
        "rates_bucket": str(scenario_eval["rates_bucket"]),
        "scenario_count": int(scenario_count),
        "plausibility_metric": float(plausibility_metric),
        "notes": question.spec.notes,
    }
    row["delta_predicted_return"] = row["scenario_predicted_return"] - row["baseline_predicted_return"]
    row["delta_max_weight"] = row["scenario_max_weight"] - row["baseline_max_weight"]
    row["delta_effective_n"] = row["scenario_effective_n"] - row["baseline_effective_n"]
    row["delta_eq_cn_weight"] = row["eq_cn_weight_after"] - row["eq_cn_weight_before"]
    for sleeve in SLEEVE_ORDER:
        row[f"weight_before_{sleeve}"] = float(_weight(baseline_eval["weights_60"], sleeve))
        row[f"weight_after_{sleeve}"] = float(_weight(scenario_eval["weights_60"], sleeve))
        row[f"pred_before_{sleeve}"] = float(_pred(baseline_eval["weights_60"], sleeve))
        row[f"pred_after_{sleeve}"] = float(_pred(scenario_eval["weights_60"], sleeve))
    return row


def _build_framework_report(question_manifest: pd.DataFrame, regime_manifest: pd.DataFrame, question_results: pd.DataFrame, variable_summary: pd.DataFrame) -> str:
    recommended = question_results.groupby("question_id", as_index=False).agg(
        question_family=("question_family", "first"),
        short_label=("short_label", "first"),
        mean_regime=("scenario_regime_label", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        mean_return_change=("delta_predicted_return", "mean"),
        mean_eq_cn_change=("delta_eq_cn_weight", "mean"),
        mean_max_weight_change=("delta_max_weight", "mean"),
    )
    return "\n".join(
        [
            "# v3 Scenario Question Framework",
            "",
            "## Scope",
            "- Active branch only: v3_long_horizon_china.",
            "- This pass upgrades the scenario layer from a generic probe menu into a compact conference-grade question set.",
            "- These remain plausibility-regularized, anchor-local, model-implied diagnostics for long-horizon SAA benchmark portfolios.",
            "",
            "## Final Question Menu",
            question_manifest[["question_id", "question_family", "short_label", "candidate_name", "question_text"]].to_markdown(index=False),
            "",
            "## Regime Taxonomy",
            "- Dimensions: growth, inflation, stress, rates/financial conditions.",
            "- Buckets: low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates.",
            regime_manifest.head(12).to_markdown(index=False),
            "",
            "## First-Pass Findings",
            recommended[["question_id", "question_family", "short_label", "mean_regime", "mean_return_change", "mean_max_weight_change"]].to_markdown(index=False),
            "",
            "## Repeating State Variables",
            variable_summary.to_markdown(index=False),
            "",
            "## Interpretation",
            "- The strongest questions are the ones that move return or allocation in a recognizable way while remaining anchor-local and plausible.",
            "- In this first pass, regime labels are driven more by the anchor backdrop than by large cross-question regime jumps. That is consistent with local scenario diagnostics rather than regime-switch forecasting.",
            "- The robust 60m benchmark remains the main interpretation object; the raw 120m ceiling remains the comparison object.",
            "- China remains included, but it does not become a dominant scenario driver in the first pass.",
        ]
    )


def _build_conference_notes(question_results: pd.DataFrame, selected_questions: pd.DataFrame) -> str:
    lines = ["# Scenario Question Notes", ""]
    for _, row in selected_questions.iterrows():
        subset = question_results.loc[question_results["question_id"].eq(row["question_id"])].copy()
        regime = subset["scenario_regime_label"].mode().iloc[0] if not subset.empty else "mixed mid-cycle"
        mean_return_change = float(subset["delta_predicted_return"].mean()) if not subset.empty else float("nan")
        mean_max_weight_change = float(subset["delta_max_weight"].mean()) if not subset.empty else float("nan")
        lines.append(
            f"- {row['short_label']}: across anchors, this question most often lands in a '{regime}' regime, shifts predicted return by about {mean_return_change:+.2%}, and shifts max weight by about {mean_max_weight_change:+.2%}."
        )
    lines.extend(
        [
            "- The robust 60m benchmark gives cleaner answers than the raw 120m ceiling.",
            "- The raw 120m ceiling remains useful because it makes concentration risk visible rather than because it is the main carry-forward object.",
            "- China remains a secondary sleeve under the improved question set as well.",
            "- The right public framing is still benchmark-conditioned, plausibility-regularized state diagnostics, not causal macro truth.",
        ]
    )
    return "\n".join(lines)


def run_scenario_question_set(project_root: Path) -> ScenarioQuestionOutputs:
    """Run the compact conference-grade question set on the active v3 anchors."""
    paths = default_paths(project_root)
    artifacts = load_active_artifacts(paths)
    classifier = fit_regime_classifier(artifacts["feature_master_monthly"])

    question_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    question_manifest_rows: list[dict[str, object]] = []

    for anchor_date in ANCHOR_DATES:
        context = _build_context(project_root, anchor_date)
        baseline = _baseline_bundle(context, classifier)
        questions = build_question_set(context, baseline, classifier)

        for question in questions:
            if not question_manifest_rows:
                pass
            question_manifest_rows.append({
                "question_id": question.spec.question_id,
                "question_family": question.spec.question_family,
                "short_label": question.spec.short_label,
                "candidate_name": question.spec.candidate_name,
                "candidate_2": question.spec.candidate_2,
                "horizon": question.spec.horizon,
                "question_text": question.spec.question_text,
                "why_it_matters": question.spec.why_it_matters,
                "question_type": question.spec.question_type,
                "primary_metric_label": question.spec.primary_metric_label,
                "response_direction": question.spec.response_direction,
                "steps": question.spec.steps,
                "step_size": question.spec.step_size,
                "notes": question.spec.notes,
            })
            run = _run_single_question(context, classifier, baseline, question)
            best_row = run["best_row"]
            best_state = np.asarray(best_row["state_vector"], dtype=float)
            base_eval = run["baseline_evaluation"]
            scenario_eval = best_row["evaluation"]

            question_rows.append(
                _question_result_row(
                    pd.Timestamp(anchor_date),
                    question,
                    base_eval,
                    scenario_eval,
                    scenario_count=int(len(run["scenario_df"])),
                    plausibility_metric=float(best_row["plausibility_energy"]),
                )
            )

            regime_rows.append(
                {
                    "anchor_date": pd.Timestamp(anchor_date),
                    "question_id": question.spec.question_id,
                    "short_label": question.spec.short_label,
                    "candidate_name": question.spec.candidate_name,
                    "baseline_regime_label": str(classifier.classify(baseline["state"])["regime_label"]),
                    "scenario_regime_label": str(scenario_eval["regime_label"]),
                    "growth_score": float(scenario_eval["growth_score"]),
                    "inflation_score": float(scenario_eval["inflation_score"]),
                    "stress_score": float(scenario_eval["stress_score"]),
                    "rates_score": float(scenario_eval["rates_score"]),
                    "growth_bucket": str(scenario_eval["growth_bucket"]),
                    "inflation_bucket": str(scenario_eval["inflation_bucket"]),
                    "stress_bucket": str(scenario_eval["stress_bucket"]),
                    "rates_bucket": str(scenario_eval["rates_bucket"]),
                    "eq_cn_weight_before": float(question_rows[-1]["eq_cn_weight_before"]),
                    "eq_cn_weight_after": float(question_rows[-1]["eq_cn_weight_after"]),
                }
            )

            top_shifts = _top_shift_summary(context, best_state)
            for variable_name, shift in top_shifts:
                state_rows.append(
                    {
                        "anchor_date": pd.Timestamp(anchor_date),
                        "question_id": question.spec.question_id,
                        "short_label": question.spec.short_label,
                        "candidate_name": question.spec.candidate_name,
                        "variable_name": variable_name,
                        "shift_std_units": float(shift),
                    }
                )

    question_manifest = pd.DataFrame(question_manifest_rows).drop_duplicates(subset=["question_id"]).sort_values("question_id").reset_index(drop=True)
    question_results = pd.DataFrame(question_rows).sort_values(["anchor_date", "question_id"]).reset_index(drop=True)
    regime_summary = pd.DataFrame(regime_rows).sort_values(["anchor_date", "question_id"]).reset_index(drop=True)
    state_shift_summary = pd.DataFrame(state_rows).sort_values(["question_id", "anchor_date", "variable_name"]).reset_index(drop=True)
    regime_manifest = regime_manifest_df()

    variable_importance = (
        state_shift_summary.assign(abs_shift=lambda df: df["shift_std_units"].abs())
        .groupby("variable_name", as_index=False)
        .agg(
            avg_abs_shift_std_units=("abs_shift", "mean"),
            case_count=("question_id", "count"),
        )
        .sort_values(["avg_abs_shift_std_units", "case_count"], ascending=[False, False])
        .head(10)
        .reset_index(drop=True)
    )

    selected_questions = (
        question_results.groupby("question_id", as_index=False)
        .agg(
            short_label=("short_label", "first"),
            question_family=("question_family", "first"),
            candidate_name=("candidate_name", "first"),
            avg_plausibility=("plausibility_metric", "mean"),
            avg_return_change=("delta_predicted_return", "mean"),
            avg_max_weight_change=("delta_max_weight", "mean"),
            avg_effective_n_change=("delta_effective_n", "mean"),
        )
        .sort_values(["question_family", "avg_plausibility"], ascending=[True, True])
        .reset_index(drop=True)
    )

    recommended_ids = [
        "q_robust_double_digit",
        "q_raw_ceiling_upside",
        "q_raw_deconcentration",
        "q_gold_tilt",
        "q_robust_raw_disagreement",
        "q_em_tilt",
    ]
    selected_questions = selected_questions.loc[selected_questions["question_id"].isin(recommended_ids)].copy()
    selected_questions["recommended_for_conference"] = 1

    framework_report = _build_framework_report(question_manifest, regime_manifest, question_results, variable_importance)
    conference_notes = _build_conference_notes(question_results, selected_questions)

    return ScenarioQuestionOutputs(
        framework_report=framework_report,
        question_manifest=question_manifest,
        regime_manifest=regime_manifest,
        question_results=question_results,
        regime_summary=regime_summary,
        conference_notes=conference_notes,
        state_shift_summary=state_shift_summary,
        selected_questions=selected_questions,
    )
