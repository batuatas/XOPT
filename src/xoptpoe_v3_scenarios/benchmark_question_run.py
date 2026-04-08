"""One-benchmark scenario question runner for the locked robust 5Y benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from xoptpoe_v3_models.data import SLEEVE_ORDER

from .benchmark_question_set import RobustBenchmarkQuestionRunSpec, build_robust_benchmark_question_set
from .benchmark_regimes import HybridRegimeClassifier, fit_hybrid_regime_classifier, hybrid_regime_manifest
from .io import default_paths, load_active_artifacts
from .mala import run_bounded_mala
from .pipelines import build_default_candidate_specs, build_scenario_rows, fit_portfolio_candidate, fit_predictor_candidate
from .regularizers import build_regularizer
from .state import build_anchor_context, default_state_spec


ANCHOR_DATES: tuple[str, ...] = ("2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31")

RISK_ON_SLEEVES = ("EQ_US", "EQ_EZ", "EQ_JP", "EQ_CN", "EQ_EM", "RE_US")
DEFENSIVE_SLEEVES = ("FI_UST", "FI_IG", "ALT_GLD")


@dataclass(frozen=True)
class RobustScenarioContext:
    project_root: Path
    paths: object
    state_spec: object
    anchor: object
    regularizer: object
    portfolio: object
    classifier: HybridRegimeClassifier


@dataclass(frozen=True)
class RobustScenarioOutputs:
    framework_report: str
    question_manifest: pd.DataFrame
    results: pd.DataFrame
    summary: pd.DataFrame
    selected_questions: pd.DataFrame
    regime_manifest: pd.DataFrame
    takeaways: str
    state_shift_summary: pd.DataFrame


def _weight(weights_df: pd.DataFrame, sleeve_id: str) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].eq(sleeve_id), "weight"].iloc[0])


def _pred(weights_df: pd.DataFrame, sleeve_id: str) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].eq(sleeve_id), "predicted_return"].iloc[0])


def _rank(weights_df: pd.DataFrame, sleeve_id: str) -> int:
    ordered = weights_df.sort_values("weight", ascending=False).reset_index(drop=True)
    return int(ordered.index[ordered["sleeve_id"].eq(sleeve_id)][0] + 1)


def _share(weights_df: pd.DataFrame, sleeves: tuple[str, ...]) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].isin(list(sleeves)), "weight"].sum())


def _forward_difference_gradient(energy_fn, x: np.ndarray, *, step: float = 1e-3) -> np.ndarray:
    base = np.asarray(x, dtype=float)
    base_energy = float(energy_fn(base))
    grad = np.zeros_like(base)
    for idx in range(len(base)):
        bumped = base.copy()
        bumped[idx] = bumped[idx] + step
        grad[idx] = (float(energy_fn(bumped)) - base_energy) / step
    return grad


def _accepted_indices(states: np.ndarray, *, tol: float = 1e-10) -> list[int]:
    accepted = []
    for idx in range(1, len(states)):
        if float(np.max(np.abs(states[idx] - states[idx - 1]))) > tol:
            accepted.append(idx)
    return accepted


def _build_context(project_root: Path, anchor_date: str, classifier: HybridRegimeClassifier) -> RobustScenarioContext:
    paths = default_paths(project_root)
    artifacts = load_active_artifacts(paths)
    state_spec = default_state_spec()
    anchor = build_anchor_context(
        artifacts["modeling_panel_hstack"],
        artifacts["feature_master_monthly"],
        month_end=pd.Timestamp(anchor_date),
        spec=state_spec,
    )
    regularizer = build_regularizer(artifacts["feature_master_monthly"], anchor)
    predictor_specs, portfolio_specs = build_default_candidate_specs(paths)
    predictor_spec = next(spec for spec in predictor_specs if spec.candidate_id == "predictor_60_anchor")
    portfolio_spec = next(spec for spec in portfolio_specs if spec.candidate_id == "best_60_predictor")
    predictor = fit_predictor_candidate(paths, predictor_spec, anchor_month_end=anchor.month_end)
    portfolio = fit_portfolio_candidate(paths, portfolio_spec, predictor, anchor_month_end=anchor.month_end)
    return RobustScenarioContext(
        project_root=project_root,
        paths=paths,
        state_spec=state_spec,
        anchor=anchor,
        regularizer=regularizer,
        portfolio=portfolio,
        classifier=classifier,
    )


def _evaluate_benchmark(context: RobustScenarioContext, state_vector: np.ndarray) -> dict[str, object]:
    frame = build_scenario_rows(context.anchor, context.state_spec, state_vector, predictor=context.portfolio.predictor)
    weights, summary = context.portfolio.evaluate(frame)
    regime = context.classifier.classify(state_vector, context.anchor.month_end)
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
    scores = context.classifier.internal_scores(state_vector)
    external = context.classifier.external_context(context.anchor.month_end)
    out["soft_landing_alignment"] = float(
        -(
            (scores["growth_score"] - 0.7) ** 2
            + (scores["inflation_score"] + 0.1) ** 2
            + (scores["market_stress_score"] + 0.5) ** 2
            + (scores["rates_score"] - 0.1) ** 2
            + (0.4 if external["nfci_bucket"] == "tight" else 0.0)
            + (0.5 if external["recession_overlay"] == "recession" else 0.0)
        )
    )
    out["higher_for_longer_alignment"] = float(
        -(
            (scores["growth_score"] + 0.1) ** 2
            + (scores["inflation_score"] - 0.8) ** 2
            + (scores["market_stress_score"] - 0.2) ** 2
            + (scores["rates_score"] - 0.9) ** 2
            + (0.3 if external["nfci_bucket"] == "loose" else 0.0)
        )
    )
    for sleeve in SLEEVE_ORDER:
        out[f"weight_{sleeve}"] = _weight(weights, sleeve)
        out[f"pred_{sleeve}"] = _pred(weights, sleeve)
    return out


def _baseline_bundle(context: RobustScenarioContext) -> dict[str, object]:
    state = np.asarray(context.anchor.current_base_state, dtype=float)
    evaluation = _evaluate_benchmark(context, state)
    return {"state": state, "evaluation": evaluation}


def _run_single_question(context: RobustScenarioContext, baseline: dict[str, object], question: RobustBenchmarkQuestionRunSpec) -> dict[str, object]:
    base_state = np.asarray(baseline["state"], dtype=float)
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
    rows = []
    for idx in retained:
        state = np.asarray(chain.states[idx], dtype=float)
        evaluation = _evaluate_benchmark(context, state)
        rows.append(
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
    eval_df = pd.DataFrame(rows)
    ascending = question.spec.response_direction == "min"
    best_row = eval_df.sort_values(["selection_metric", "total_energy"], ascending=[ascending, True]).iloc[0]
    return {
        "baseline_evaluation": baseline["evaluation"],
        "best_row": best_row,
        "scenario_count": int(len(eval_df)),
        "accepted_count": int(round(chain.acceptance_rate * question.spec.steps)),
    }


def _top_shift_summary(context: RobustScenarioContext, state_vector: np.ndarray, *, top_n: int = 5) -> list[tuple[str, float]]:
    shifts = (np.asarray(state_vector, dtype=float) - np.asarray(context.anchor.current_base_state, dtype=float)) / context.regularizer.bounds.scale
    ordered = pd.Series(shifts, index=list(context.state_spec.base_variables)).abs().sort_values(ascending=False).head(top_n)
    return [(name, float(shifts[list(context.state_spec.base_variables).index(name)])) for name in ordered.index]


def _result_row(anchor_date: pd.Timestamp, question: RobustBenchmarkQuestionRunSpec, baseline_eval: dict[str, object], scenario_eval: dict[str, object], *, scenario_count: int, plausibility_metric: float) -> dict[str, object]:
    row = {
        "anchor_date": anchor_date,
        "question_id": question.spec.question_id,
        "question_group": question.spec.question_group,
        "question_family": question.spec.question_family,
        "audience_question": question.spec.audience_question,
        "short_label": question.spec.short_label,
        "recommended_for_conference": int(question.spec.recommended_for_conference),
        "target_value": question.spec.target_value,
        "baseline_predicted_return": float(baseline_eval["portfolio_predicted_return"]),
        "scenario_predicted_return": float(scenario_eval["portfolio_predicted_return"]),
        "baseline_max_weight": float(baseline_eval["portfolio_max_weight"]),
        "scenario_max_weight": float(scenario_eval["portfolio_max_weight"]),
        "baseline_effective_n": float(baseline_eval["portfolio_effective_n"]),
        "scenario_effective_n": float(scenario_eval["portfolio_effective_n"]),
        "baseline_hhi": float(baseline_eval["portfolio_hhi"]),
        "scenario_hhi": float(scenario_eval["portfolio_hhi"]),
        "top_weight_sleeve_before": str(baseline_eval["top_weight_sleeve"]),
        "top_weight_sleeve_after": str(scenario_eval["top_weight_sleeve"]),
        "eq_cn_weight_before": float(baseline_eval["eq_cn_weight"]),
        "eq_cn_weight_after": float(scenario_eval["eq_cn_weight"]),
        "eq_cn_rank_before": int(baseline_eval["eq_cn_rank"]),
        "eq_cn_rank_after": int(scenario_eval["eq_cn_rank"]),
        "gold_weight_before": float(baseline_eval["weight_ALT_GLD"]),
        "gold_weight_after": float(scenario_eval["weight_ALT_GLD"]),
        "eq_us_weight_before": float(baseline_eval["weight_EQ_US"]),
        "eq_us_weight_after": float(scenario_eval["weight_EQ_US"]),
        "eq_em_weight_before": float(baseline_eval["weight_EQ_EM"]),
        "eq_em_weight_after": float(scenario_eval["weight_EQ_EM"]),
        "nfci_value": float(scenario_eval["nfci_value"]),
        "nfci_bucket": str(scenario_eval["nfci_bucket"]),
        "recession_overlay": str(scenario_eval["recession_overlay"]),
        "growth_bucket": str(scenario_eval["growth_bucket"]),
        "inflation_bucket": str(scenario_eval["inflation_bucket"]),
        "market_stress_bucket": str(scenario_eval["market_stress_bucket"]),
        "rates_bucket": str(scenario_eval["rates_bucket"]),
        "scenario_regime_label": str(scenario_eval["regime_label"]),
        "scenario_count": int(scenario_count),
        "plausibility_metric": float(plausibility_metric),
        "notes": question.spec.notes,
    }
    row["delta_predicted_return"] = row["scenario_predicted_return"] - row["baseline_predicted_return"]
    row["delta_max_weight"] = row["scenario_max_weight"] - row["baseline_max_weight"]
    row["delta_effective_n"] = row["scenario_effective_n"] - row["baseline_effective_n"]
    row["delta_hhi"] = row["scenario_hhi"] - row["baseline_hhi"]
    row["delta_gold_weight"] = row["gold_weight_after"] - row["gold_weight_before"]
    row["delta_eq_us_weight"] = row["eq_us_weight_after"] - row["eq_us_weight_before"]
    row["delta_eq_em_weight"] = row["eq_em_weight_after"] - row["eq_em_weight_before"]
    row["delta_eq_cn_weight"] = row["eq_cn_weight_after"] - row["eq_cn_weight_before"]
    if question.spec.target_value is not None:
        row["return_gap_to_target_before"] = abs(row["baseline_predicted_return"] - float(question.spec.target_value))
        row["return_gap_to_target_after"] = abs(row["scenario_predicted_return"] - float(question.spec.target_value))
    else:
        row["return_gap_to_target_before"] = np.nan
        row["return_gap_to_target_after"] = np.nan
    return row


def _build_framework_report(question_manifest: pd.DataFrame, regime_manifest: pd.DataFrame, summary: pd.DataFrame, selected_questions: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# v3 Scenario Regime Framework",
            "",
            "## Scope",
            "- Active branch only: v3_long_horizon_china.",
            "- Active scenario object only: the locked robust 5Y benchmark (`best_60_predictor`).",
            "- NFCI is used as the external financial-conditions anchor; NBER recession dates are used as a historical overlay only.",
            "- These remain plausibility-regularized, anchor-local, model-implied scenario diagnostics for long-horizon SAA.",
            "",
            "## Final Question Menu",
            question_manifest[
                [
                    "question_id",
                    "question_group",
                    "question_family",
                    "short_label",
                    "audience_question",
                    "recommended_for_conference",
                ]
            ].to_markdown(index=False),
            "",
            "## Hybrid Regime Taxonomy",
            regime_manifest.to_markdown(index=False),
            "",
            "## Selected Conference Questions",
            selected_questions[
                [
                    "question_id",
                    "question_family",
                    "short_label",
                    "avg_delta_return",
                    "avg_delta_max_weight",
                    "modal_regime",
                    "nfci_bucket_mode",
                ]
            ].to_markdown(index=False),
            "",
            "## Interpretation",
            "- This framework is centered on one benchmark object, not on a benchmark horse race.",
            "- NFCI makes the stress/conditions axis easier to explain to an allocator audience.",
            "- The recession overlay is historical context, not a predictive label.",
            "- China remains in the system, but it is not a main-deck question unless the scenario evidence makes it economically material.",
        ]
    )


def _build_takeaways(summary: pd.DataFrame, results: pd.DataFrame) -> str:
    lines = [
        "# Scenario Conference Takeaways",
        "",
        "- The locked robust 5Y benchmark is now the single reported scenario object.",
        "- NFCI-based conditions and recession overlay make the regime language easier to explain on stage.",
        "- The strongest conference questions are the ones that change return or allocation in a visible way without turning the story back into a model comparison exercise.",
        "",
        "## Main Questions",
    ]
    for _, row in summary.loc[summary["recommended_for_conference"].eq(1)].sort_values(["question_family", "question_id"]).iterrows():
        lines.append(
            f"- {row['short_label']}: modal regime `{row['modal_regime']}`, NFCI backdrop `{row['nfci_bucket_mode']}`, average return change {float(row['avg_delta_return']):+.2%}, average max-weight change {float(row['avg_delta_max_weight']):+.2%}."
        )
    lines.extend(
        [
            "",
            "## China",
            f"- Maximum EQ_CN weight change across the focused batch: {float(results['delta_eq_cn_weight'].max()):+.2%}.",
            f"- Average EQ_CN weight change across the focused batch: {float(results['delta_eq_cn_weight'].mean()):+.2%}.",
            "- China remains active but secondary in the one-benchmark story.",
        ]
    )
    return "\n".join(lines)


def run_robust_regime_questions(project_root: Path) -> RobustScenarioOutputs:
    """Run the focused one-benchmark scenario batch."""
    root = project_root.resolve()
    paths = default_paths(root)
    artifacts = load_active_artifacts(paths)
    classifier = fit_hybrid_regime_classifier(artifacts["feature_master_monthly"], root)

    manifest_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    state_shift_rows: list[dict[str, object]] = []

    for anchor_date in ANCHOR_DATES:
        context = _build_context(root, anchor_date, classifier)
        baseline = _baseline_bundle(context)
        questions = build_robust_benchmark_question_set(context, baseline, classifier)

        for question in questions:
            manifest_rows.append(
                {
                    "question_id": question.spec.question_id,
                    "question_group": question.spec.question_group,
                    "question_family": question.spec.question_family,
                    "audience_question": question.spec.audience_question,
                    "short_label": question.spec.short_label,
                    "recommended_for_conference": int(question.spec.recommended_for_conference),
                    "target_value": question.spec.target_value,
                    "primary_metric_label": question.spec.primary_metric_label,
                    "response_direction": question.spec.response_direction,
                    "steps": question.spec.steps,
                    "step_size": question.spec.step_size,
                    "notes": question.spec.notes,
                }
            )
            run = _run_single_question(context, baseline, question)
            best_row = run["best_row"]
            best_state = np.asarray(best_row["state_vector"], dtype=float)
            baseline_eval = run["baseline_evaluation"]
            scenario_eval = best_row["evaluation"]
            result_rows.append(
                _result_row(
                    pd.Timestamp(anchor_date),
                    question,
                    baseline_eval,
                    scenario_eval,
                    scenario_count=int(run["scenario_count"]),
                    plausibility_metric=float(best_row["plausibility_energy"]),
                )
            )
            for variable_name, shift in _top_shift_summary(context, best_state):
                state_shift_rows.append(
                    {
                        "anchor_date": pd.Timestamp(anchor_date),
                        "question_id": question.spec.question_id,
                        "short_label": question.spec.short_label,
                        "variable_name": variable_name,
                        "shift_std_units": float(shift),
                    }
                )

    question_manifest = pd.DataFrame(manifest_rows).drop_duplicates(subset=["question_id"]).sort_values(["recommended_for_conference", "question_family", "question_id"], ascending=[False, True, True]).reset_index(drop=True)
    results = pd.DataFrame(result_rows).sort_values(["anchor_date", "question_id"]).reset_index(drop=True)
    state_shift_summary = pd.DataFrame(state_shift_rows).sort_values(["question_id", "anchor_date", "variable_name"]).reset_index(drop=True)
    regime_manifest = hybrid_regime_manifest(classifier)

    summary = (
        results.groupby(["question_id", "question_group", "question_family", "short_label", "audience_question", "recommended_for_conference"], as_index=False)
        .agg(
            avg_delta_return=("delta_predicted_return", "mean"),
            avg_delta_max_weight=("delta_max_weight", "mean"),
            avg_delta_effective_n=("delta_effective_n", "mean"),
            avg_delta_gold_weight=("delta_gold_weight", "mean"),
            avg_delta_eq_us_weight=("delta_eq_us_weight", "mean"),
            avg_delta_eq_em_weight=("delta_eq_em_weight", "mean"),
            avg_delta_eq_cn_weight=("delta_eq_cn_weight", "mean"),
            avg_gap_to_target_after=("return_gap_to_target_after", "mean"),
            modal_regime=("scenario_regime_label", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            nfci_bucket_mode=("nfci_bucket", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            recession_overlay_mode=("recession_overlay", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        )
        .sort_values(["recommended_for_conference", "question_family", "question_id"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    selected_questions = summary.loc[summary["recommended_for_conference"].eq(1)].copy().reset_index(drop=True)
    framework_report = _build_framework_report(question_manifest, regime_manifest, summary, selected_questions)
    takeaways = _build_takeaways(summary, results)
    return RobustScenarioOutputs(
        framework_report=framework_report,
        question_manifest=question_manifest,
        results=results,
        summary=summary,
        selected_questions=selected_questions,
        regime_manifest=regime_manifest,
        takeaways=takeaways,
        state_shift_summary=state_shift_summary,
    )
