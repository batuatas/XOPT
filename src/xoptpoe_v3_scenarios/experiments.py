"""First real v3 scenario experiments on top of the active benchmark stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_v3_models.data import SLEEVE_ORDER

from .io import default_paths, load_active_artifacts
from .mala import run_bounded_mala
from .pipelines import (
    FittedPortfolioCandidate,
    FittedPredictorCandidate,
    build_default_candidate_specs,
    build_scenario_rows,
    fit_portfolio_candidate,
    fit_predictor_candidate,
)
from .probes import (
    ChinaRoleProbe,
    PredictionDisagreementProbe,
    ReturnConcentrationProbe,
    SimilarReturnDistinctAllocationProbe,
    TargetReturnProbe,
)
from .regularizers import ScenarioRegularizer, build_regularizer
from .state import BASE_STATE_VARIABLES, ScenarioAnchor, ScenarioStateSpec, build_anchor_context, default_state_spec


ANCHOR_DATES: tuple[str, ...] = ("2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31")


@dataclass(frozen=True)
class ExperimentPaths:
    project_root: Path
    reports_root: Path
    plots_root: Path
    modeling_root: Path


@dataclass
class ExperimentContext:
    paths: object
    state_spec: ScenarioStateSpec
    anchor: ScenarioAnchor
    regularizer: ScenarioRegularizer
    predictors: dict[str, FittedPredictorCandidate]
    portfolios: dict[str, FittedPortfolioCandidate]


@dataclass(frozen=True)
class ScenarioExperimentOutputs:
    report_text: str
    summary: pd.DataFrame
    state_shift_summary: pd.DataFrame
    portfolio_response_summary: pd.DataFrame
    representative_cases: pd.DataFrame
    china_diagnostics: pd.DataFrame
    experiment_manifest: pd.DataFrame
    representative_state_table: pd.DataFrame


def _cross_sectional_zscore(scores: pd.Series) -> pd.Series:
    mean = float(scores.mean())
    std = float(scores.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-8:
        return pd.Series(0.0, index=scores.index, dtype=float)
    return (scores - mean) / std


def _normalize_long_only(raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.repeat(1.0 / len(clipped), len(clipped))
    return clipped / total


def _project_with_cap(weights: np.ndarray, *, max_weight: float) -> np.ndarray:
    w = _normalize_long_only(weights)
    for _ in range(100):
        over = w > max_weight + 1e-12
        if not np.any(over):
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if not np.any(under):
            break
        w[under] = w[under] + excess * w[under] / w[under].sum()
    return _normalize_long_only(w)


def _weights_top_k_capped(scores: pd.Series, *, k: int, max_weight: float) -> pd.Series:
    n = len(scores)
    min_required = int(np.ceil(1.0 / max_weight))
    select_n = min(max(k, min_required), n)
    top_idx = scores.sort_values(ascending=False).index[:select_n]
    raw = np.clip(scores.loc[top_idx].to_numpy(dtype=float), 0.0, None)
    if float(raw.sum()) <= 0:
        raw = np.ones_like(raw, dtype=float)
    capped = _project_with_cap(_normalize_long_only(raw), max_weight=max_weight)
    out = pd.Series(0.0, index=scores.index, dtype=float)
    out.loc[top_idx] = capped
    return out


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


def _scenario_case_label(probe_id: str, anchor_date: pd.Timestamp) -> str:
    family = probe_id.replace("probe_", "").replace("_", " ")
    return f"{anchor_date.year} | {family}"


def _short_interpretation(shift_row: pd.Series) -> str:
    pieces = []
    for variable_name in shift_row.index[:3]:
        shift = float(shift_row[variable_name])
        if abs(shift) < 1e-8:
            continue
        direction = "higher" if shift > 0 else "lower"
        pieces.append(f"{direction} {variable_name}")
    return ", ".join(pieces) if pieces else "very small move around anchor"


def _build_context(project_root: Path, anchor_date: str) -> ExperimentContext:
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

    predictor_map = {
        spec.candidate_id: fit_predictor_candidate(paths, spec, anchor_month_end=anchor.month_end)
        for spec in predictor_specs
        if spec.candidate_id != "predictor_shared_anchor"
    }
    portfolio_predictor_map = {
        "best_60_predictor": predictor_map["predictor_60_anchor"],
        "best_120_predictor": predictor_map["predictor_120_anchor"],
    }
    portfolio_map = {
        spec.candidate_id: fit_portfolio_candidate(
            paths,
            spec,
            portfolio_predictor_map[spec.candidate_id],
            anchor_month_end=anchor.month_end,
        )
        for spec in portfolio_specs
    }
    return ExperimentContext(
        paths=paths,
        state_spec=state_spec,
        anchor=anchor,
        regularizer=regularizer,
        predictors=predictor_map,
        portfolios=portfolio_map,
    )


def _predict_for_state(
    context: ExperimentContext,
    state_vector: np.ndarray,
    predictor_id: str,
) -> pd.DataFrame:
    predictor = context.predictors[predictor_id]
    frame = build_scenario_rows(context.anchor, context.state_spec, state_vector, predictor=predictor)
    return predictor.predict(frame)


def _portfolio_for_state(
    context: ExperimentContext,
    state_vector: np.ndarray,
    portfolio_id: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    portfolio = context.portfolios[portfolio_id]
    frame = build_scenario_rows(context.anchor, context.state_spec, state_vector, predictor=portfolio.predictor)
    return portfolio.evaluate(frame)


def _combined_top_k_capped_for_state(context: ExperimentContext, state_vector: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    pred_60 = _predict_for_state(context, state_vector, "predictor_60_anchor")
    pred_120 = _predict_for_state(context, state_vector, "predictor_120_anchor")
    s60 = pred_60.set_index("sleeve_id")["y_pred"].reindex(list(SLEEVE_ORDER))
    s120 = pred_120.set_index("sleeve_id")["y_pred"].reindex(list(SLEEVE_ORDER))
    combo = 0.4 * _cross_sectional_zscore(s60) + 0.6 * _cross_sectional_zscore(s120)
    weights = _weights_top_k_capped(combo, k=4, max_weight=0.35).reindex(list(SLEEVE_ORDER))
    weights_df = pd.DataFrame(
        {
            "month_end": context.anchor.month_end,
            "sleeve_id": list(SLEEVE_ORDER),
            "weight": weights.to_numpy(dtype=float),
            "predicted_return": combo.to_numpy(dtype=float),
        }
    )
    hhi = float(np.square(weights_df["weight"].to_numpy(dtype=float)).sum())
    summary = {
        "portfolio_predicted_return": float(np.dot(weights_df["weight"], weights_df["predicted_return"])),
        "portfolio_hhi": hhi,
        "portfolio_effective_n": float(1.0 / hhi),
        "portfolio_max_weight": float(weights_df["weight"].max()),
    }
    return weights_df, summary


def _baseline_bundle(context: ExperimentContext) -> dict[str, object]:
    state = np.asarray(context.anchor.current_base_state, dtype=float)
    pred60 = _predict_for_state(context, state, "predictor_60_anchor")
    pred120 = _predict_for_state(context, state, "predictor_120_anchor")
    port60_w, port60_s = _portfolio_for_state(context, state, "best_60_predictor")
    port120_w, port120_s = _portfolio_for_state(context, state, "best_120_predictor")
    topk_w, topk_s = _combined_top_k_capped_for_state(context, state)
    return {
        "state": state,
        "pred60": pred60,
        "pred120": pred120,
        "port60_weights": port60_w,
        "port60_summary": port60_s,
        "port120_weights": port120_w,
        "port120_summary": port120_s,
        "combined_topk_weights": topk_w,
        "combined_topk_summary": topk_s,
    }


def _target_delta(predictions: pd.DataFrame, *, lower: float, upper: float) -> float:
    spread = float(predictions["y_pred"].std(ddof=1))
    if not np.isfinite(spread):
        spread = lower
    return float(np.clip(0.5 * spread, lower, upper))


def _eq_cn_rank(weights: pd.DataFrame) -> int:
    ordered = weights.sort_values("weight", ascending=False).reset_index(drop=True)
    return int(ordered.index[ordered["sleeve_id"].eq("EQ_CN")][0] + 1)


def _build_probe_definitions(context: ExperimentContext, baseline: dict[str, object]) -> list[dict[str, object]]:
    delta60 = _target_delta(baseline["pred60"], lower=0.005, upper=0.015)
    delta120 = _target_delta(baseline["pred120"], lower=0.005, upper=0.020)
    baseline_120 = baseline["port120_summary"]
    concentration_weight = float(0.75 * baseline_120["portfolio_predicted_return"] / max(baseline_120["portfolio_hhi"], 1e-6))

    return [
        {
            "probe_id": "probe_60_target_up",
            "candidate_name": "best_60_predictor",
            "probe_family": "target_return",
            "response_metric": "portfolio_predicted_return",
            "response_direction": "max",
            "notes": f"target = baseline + {delta60:.4f}",
            "step_size": 0.07,
            "steps": 2,
            "probe": TargetReturnProbe(
                probe_id="probe_60_target_up",
                portfolio=context.portfolios["best_60_predictor"],
                anchor=context.anchor,
                state_spec=context.state_spec,
                target_return=float(baseline["port60_summary"]["portfolio_predicted_return"] + delta60),
            ),
        },
        {
            "probe_id": "probe_60_target_down",
            "candidate_name": "best_60_predictor",
            "probe_family": "target_return",
            "response_metric": "portfolio_predicted_return",
            "response_direction": "min",
            "notes": f"target = baseline - {delta60:.4f}",
            "step_size": 0.07,
            "steps": 2,
            "probe": TargetReturnProbe(
                probe_id="probe_60_target_down",
                portfolio=context.portfolios["best_60_predictor"],
                anchor=context.anchor,
                state_spec=context.state_spec,
                target_return=float(baseline["port60_summary"]["portfolio_predicted_return"] - delta60),
            ),
        },
        {
            "probe_id": "probe_120_target_up",
            "candidate_name": "best_120_predictor",
            "probe_family": "target_return",
            "response_metric": "portfolio_predicted_return",
            "response_direction": "max",
            "notes": f"target = baseline + {delta120:.4f}",
            "step_size": 0.07,
            "steps": 2,
            "probe": TargetReturnProbe(
                probe_id="probe_120_target_up",
                portfolio=context.portfolios["best_120_predictor"],
                anchor=context.anchor,
                state_spec=context.state_spec,
                target_return=float(baseline["port120_summary"]["portfolio_predicted_return"] + delta120),
            ),
        },
        {
            "probe_id": "probe_120_target_down",
            "candidate_name": "best_120_predictor",
            "probe_family": "target_return",
            "response_metric": "portfolio_predicted_return",
            "response_direction": "min",
            "notes": f"target = baseline - {delta120:.4f}",
            "step_size": 0.07,
            "steps": 2,
            "probe": TargetReturnProbe(
                probe_id="probe_120_target_down",
                portfolio=context.portfolios["best_120_predictor"],
                anchor=context.anchor,
                state_spec=context.state_spec,
                target_return=float(baseline["port120_summary"]["portfolio_predicted_return"] - delta120),
            ),
        },
        {
            "probe_id": "probe_60_120_allocation_disagreement",
            "candidate_name": "best_60_predictor_vs_best_120_predictor",
            "probe_family": "benchmark_difference",
            "response_metric": "allocation_gap_l1",
            "response_direction": "max",
            "notes": "Similar portfolio predicted return, maximally different allocations.",
            "step_size": 0.08,
            "steps": 2,
            "probe": SimilarReturnDistinctAllocationProbe(
                probe_id="probe_60_120_allocation_disagreement",
                portfolio_a=context.portfolios["best_60_predictor"],
                portfolio_b=context.portfolios["best_120_predictor"],
                anchor=context.anchor,
                state_spec=context.state_spec,
                diff_reward=0.35,
            ),
        },
        {
            "probe_id": "probe_60_120_prediction_disagreement",
            "candidate_name": "elastic_net__full_firstpass__separate_60_vs_ridge__full_firstpass__separate_120",
            "probe_family": "benchmark_difference",
            "response_metric": "prediction_gap_mean_abs",
            "response_direction": "max",
            "notes": "Maximize disagreement in cross-sectional predicted sleeve returns.",
            "step_size": 0.08,
            "steps": 2,
            "probe": PredictionDisagreementProbe(
                probe_id="probe_60_120_prediction_disagreement",
                predictor_a=context.predictors["predictor_60_anchor"],
                predictor_b=context.predictors["predictor_120_anchor"],
                anchor=context.anchor,
                state_spec=context.state_spec,
                disagreement_weight=1.0,
            ),
        },
        {
            "probe_id": "probe_120_deconcentration",
            "candidate_name": "best_120_predictor",
            "probe_family": "deconcentration",
            "response_metric": "portfolio_hhi",
            "response_direction": "min",
            "notes": f"return-concentration tradeoff with concentration_weight={concentration_weight:.4f}",
            "step_size": 0.07,
            "steps": 2,
            "probe": ReturnConcentrationProbe(
                probe_id="probe_120_deconcentration",
                portfolio=context.portfolios["best_120_predictor"],
                anchor=context.anchor,
                state_spec=context.state_spec,
                concentration_weight=concentration_weight,
            ),
        },
        {
            "probe_id": "probe_60_china_role",
            "candidate_name": "best_60_predictor",
            "probe_family": "china_role",
            "response_metric": "eq_cn_weight",
            "response_direction": "max",
            "notes": "Increase EQ_CN relevance in the robust 60m portfolio without ignoring portfolio quality.",
            "step_size": 0.08,
            "steps": 2,
            "probe": ChinaRoleProbe(
                probe_id="probe_60_china_role",
                portfolio=context.portfolios["best_60_predictor"],
                anchor=context.anchor,
                state_spec=context.state_spec,
                eq_cn_weight_weight=1.0,
                return_weight=0.25,
                concentration_penalty=0.10,
            ),
        },
    ]


def _primary_metric_value(probe_def: dict[str, object], evaluation: dict[str, object]) -> float:
    metric = str(probe_def["response_metric"])
    return float(evaluation[metric])


def _evaluate_probe_state(
    context: ExperimentContext,
    probe_def: dict[str, object],
    state_vector: np.ndarray,
) -> dict[str, object]:
    probe_id = str(probe_def["probe_id"])
    out: dict[str, object] = {}
    if probe_id.startswith("probe_60_target") or probe_id == "probe_60_china_role":
        weights, summary = _portfolio_for_state(context, state_vector, "best_60_predictor")
        eq_cn_weight = float(weights.loc[weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0])
        eq_cn_pred = float(weights.loc[weights["sleeve_id"].eq("EQ_CN"), "predicted_return"].iloc[0])
        out.update(summary)
        out["eq_cn_weight"] = eq_cn_weight
        out["eq_cn_predicted_return"] = eq_cn_pred
        out["eq_cn_rank"] = _eq_cn_rank(weights)
        out["top_weight_sleeve"] = str(weights.sort_values("weight", ascending=False)["sleeve_id"].iloc[0])
        out["weights_df"] = weights
    elif probe_id.startswith("probe_120_target") or probe_id == "probe_120_deconcentration":
        weights, summary = _portfolio_for_state(context, state_vector, "best_120_predictor")
        eq_cn_weight = float(weights.loc[weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0])
        eq_cn_pred = float(weights.loc[weights["sleeve_id"].eq("EQ_CN"), "predicted_return"].iloc[0])
        out.update(summary)
        out["eq_cn_weight"] = eq_cn_weight
        out["eq_cn_predicted_return"] = eq_cn_pred
        out["eq_cn_rank"] = _eq_cn_rank(weights)
        out["top_weight_sleeve"] = str(weights.sort_values("weight", ascending=False)["sleeve_id"].iloc[0])
        out["weights_df"] = weights
    elif probe_id == "probe_60_120_allocation_disagreement":
        w60, s60 = _portfolio_for_state(context, state_vector, "best_60_predictor")
        w120, s120 = _portfolio_for_state(context, state_vector, "best_120_predictor")
        merged = w60.merge(w120, on=["month_end", "sleeve_id"], suffixes=("_60", "_120"), validate="1:1")
        gap = float(np.abs(merged["weight_60"] - merged["weight_120"]).sum())
        out.update(
            {
                "allocation_gap_l1": gap,
                "summary_60": s60,
                "summary_120": s120,
                "weights_60": w60,
                "weights_120": w120,
                "eq_cn_weight_60": float(w60.loc[w60["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
                "eq_cn_weight_120": float(w120.loc[w120["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
            }
        )
    elif probe_id == "probe_60_120_prediction_disagreement":
        p60 = _predict_for_state(context, state_vector, "predictor_60_anchor")
        p120 = _predict_for_state(context, state_vector, "predictor_120_anchor")
        merged = p60.merge(p120, on=["month_end", "sleeve_id"], suffixes=("_60", "_120"), validate="1:1")
        gap = float(np.mean(np.abs(merged["y_pred_60"] - merged["y_pred_120"])))
        out.update(
            {
                "prediction_gap_mean_abs": gap,
                "pred60": p60,
                "pred120": p120,
            }
        )
    else:
        raise ValueError(f"Unsupported probe_id: {probe_id}")
    return out


def _run_single_probe(
    context: ExperimentContext,
    baseline: dict[str, object],
    probe_def: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    probe = probe_def["probe"]
    base_state = np.asarray(baseline["state"], dtype=float)

    total_energy = lambda x: float(probe.energy(context.regularizer.project(x)) + context.regularizer.total_energy(context.regularizer.project(x)))
    gradient = lambda x: _forward_difference_gradient(total_energy, context.regularizer.project(x), step=1e-3)

    mala_result = run_bounded_mala(
        start=base_state,
        energy_fn=total_energy,
        project_fn=context.regularizer.project,
        gradient_fn=gradient,
        step_size=float(probe_def["step_size"]),
        n_steps=int(probe_def["steps"]),
        random_seed=42,
    )
    accepted = _accepted_indices(mala_result.states)
    retained_indices = accepted if accepted else [len(mala_result.states) - 1]

    rows = []
    for idx in retained_indices:
        state = mala_result.states[idx]
        evaluation = _evaluate_probe_state(context, probe_def, state)
        rows.append(
            {
                "chain_index": int(idx),
                "probe_energy": float(probe.energy(state)),
                "plausibility_energy": float(context.regularizer.total_energy(state)),
                "total_energy": float(total_energy(state)),
                "state_vector": state,
                "evaluation": evaluation,
                "primary_metric_value": _primary_metric_value(probe_def, evaluation),
            }
        )

    state_df = pd.DataFrame(rows)
    direction = str(probe_def["response_direction"])
    ascending = direction == "min"
    best_row = state_df.sort_values(["primary_metric_value", "total_energy"], ascending=[ascending, True]).iloc[0]
    accepted_count = int(round(mala_result.acceptance_rate * int(probe_def["steps"])))
    result = {
        "accepted_count": accepted_count,
        "scenario_count": int(len(state_df)),
        "best_row": best_row,
        "state_df": state_df,
        "mala_result": mala_result,
        "baseline_evaluation": _evaluate_probe_state(context, probe_def, base_state),
    }
    return result, state_df


def _build_report(
    summary_df: pd.DataFrame,
    representative_cases: pd.DataFrame,
    state_shift_summary: pd.DataFrame,
) -> str:
    top_cases = representative_cases.head(12)
    largest_shifts = (
        state_shift_summary.loc[state_shift_summary["included_in_representative_summary"].eq(1)]
        .groupby("variable_name", as_index=False)["shift_magnitude_std_units"]
        .mean()
        .sort_values("shift_magnitude_std_units", ascending=False)
        .head(10)
    )
    return "\n".join(
        [
            "# v3 Scenario Experiment Report",
            "",
            "## Scope",
            "- Active branch only: v3_long_horizon_china.",
            "- These are model-based, plausibility-regularized state perturbation diagnostics for long-horizon SAA benchmarks.",
            "- They are not a causal macro truth exercise and not a live-investment backtest.",
            "",
            "## Experiment Design",
            "- Anchors: 2021-12-31, 2022-12-31, 2023-12-31, 2024-12-31.",
            "- Manipulated state: the locked 17-variable canonical macro block from the scaffold.",
            "- Benchmarks actually evaluated: best_60_predictor, best_120_predictor, elastic_net__full_firstpass__separate_60, ridge__full_firstpass__separate_120.",
            "- Shared predictor and combined_std_120tilt_top_k_capped are retained as comparators/reference objects, but the first-pass probes stay focused on the supervised 60m/120m carry-forward stack.",
            "- E2E remains out of the real run set because the active v3 artifacts still do not include a clean scenario-ready persisted model object.",
            "",
            "## Representative Cases",
            top_cases.to_markdown(index=False),
            "",
            "## Variables That Move Most Consistently",
            largest_shifts.to_markdown(index=False),
            "",
            "## Interpretation",
            "- Target-return probes show which macro shifts raise or lower the model-implied portfolio return while staying plausible under the prior.",
            "- Benchmark-difference probes separate predictor disagreement from allocation disagreement.",
            "- Deconcentration probes show whether the 120m ceiling can be made less EQ_US-heavy without collapsing predicted return.",
            "- China-role probes ask when EQ_CN becomes more meaningful inside the robust 60m benchmark, not whether China should exist as a sleeve.",
        ]
    )


def run_scenario_experiments(project_root: Path) -> ScenarioExperimentOutputs:
    """Run the controlled first-pass v3 scenario suite."""
    summary_rows: list[dict[str, object]] = []
    state_shift_rows: list[dict[str, object]] = []
    portfolio_response_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    china_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    representative_state_rows: list[dict[str, object]] = []

    for anchor_date in ANCHOR_DATES:
        context = _build_context(project_root, anchor_date)
        baseline = _baseline_bundle(context)
        probe_defs = _build_probe_definitions(context, baseline)

        for probe_def in probe_defs:
            run_result, state_df = _run_single_probe(context, baseline, probe_def)
            best_row = run_result["best_row"]
            baseline_eval = run_result["baseline_evaluation"]
            direction = str(probe_def["response_direction"])
            response_name = str(probe_def["response_metric"])
            best_value = float(best_row["primary_metric_value"])
            mean_value = float(state_df["primary_metric_value"].mean())
            plausibility_mean = float(state_df["plausibility_energy"].mean())
            accepted_count = int(run_result["accepted_count"])

            summary_rows.append(
                {
                    "anchor_date": pd.Timestamp(anchor_date),
                    "probe_id": probe_def["probe_id"],
                    "candidate_name": probe_def["candidate_name"],
                    "baseline_response": float(_primary_metric_value(probe_def, baseline_eval)),
                    "scenario_response_mean": mean_value,
                    "scenario_response_best": best_value,
                    "scenario_count": int(run_result["scenario_count"]),
                    "plausibility_summary": plausibility_mean,
                    "notes": probe_def["notes"],
                }
            )

            best_state = np.asarray(best_row["state_vector"], dtype=float)
            best_eval = best_row["evaluation"]
            state_shift = (best_state - baseline["state"]) / context.regularizer.bounds.scale
            ordered_shift = pd.Series(state_shift, index=list(BASE_STATE_VARIABLES)).abs().sort_values(ascending=False)
            top_shift_names = set(ordered_shift.head(6).index.tolist())
            for idx, variable_name in enumerate(BASE_STATE_VARIABLES):
                scenario_values = state_df["state_vector"].apply(lambda x: float(np.asarray(x, dtype=float)[idx]))
                shift_std = float((best_state[idx] - baseline["state"][idx]) / context.regularizer.bounds.scale[idx])
                state_shift_rows.append(
                    {
                        "anchor_date": pd.Timestamp(anchor_date),
                        "probe_id": probe_def["probe_id"],
                        "variable_name": variable_name,
                        "baseline_value": float(baseline["state"][idx]),
                        "scenario_mean": float(scenario_values.mean()),
                        "scenario_median": float(scenario_values.median()),
                        "scenario_best": float(best_state[idx]),
                        "shift_direction": "up" if shift_std > 0 else ("down" if shift_std < 0 else "flat"),
                        "shift_magnitude_std_units": float(abs(shift_std)),
                        "included_in_representative_summary": int(variable_name in top_shift_names),
                    }
                )

            if probe_def["probe_id"] in {"probe_60_target_up", "probe_60_target_down", "probe_60_china_role"}:
                baseline_weights = baseline["port60_weights"]
                scenario_weights = best_eval["weights_df"]
                portfolio_response_rows.append(
                    {
                        "candidate_name": "best_60_predictor",
                        "anchor_date": pd.Timestamp(anchor_date),
                        "probe_id": probe_def["probe_id"],
                        "predicted_return_change": float(best_eval["portfolio_predicted_return"] - baseline["port60_summary"]["portfolio_predicted_return"]),
                        "portfolio_return_change": float(best_eval["portfolio_predicted_return"] - baseline["port60_summary"]["portfolio_predicted_return"]),
                        "max_weight_change": float(best_eval["portfolio_max_weight"] - baseline["port60_summary"]["portfolio_max_weight"]),
                        "hhi_change": float(best_eval["portfolio_hhi"] - baseline["port60_summary"]["portfolio_hhi"]),
                        "effective_n_change": float(best_eval["portfolio_effective_n"] - baseline["port60_summary"]["portfolio_effective_n"]),
                        "eq_cn_weight_change": float(best_eval["eq_cn_weight"] - float(baseline_weights.loc[baseline_weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0])),
                        "top_weight_sleeve_before": str(baseline_weights.sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
                        "top_weight_sleeve_after": str(best_eval["top_weight_sleeve"]),
                    }
                )
                china_rows.append(
                    {
                        "candidate_name": "best_60_predictor",
                        "anchor_date": pd.Timestamp(anchor_date),
                        "probe_id": probe_def["probe_id"],
                        "eq_cn_predicted_return_before": float(baseline_weights.loc[baseline_weights["sleeve_id"].eq("EQ_CN"), "predicted_return"].iloc[0]),
                        "eq_cn_predicted_return_after": float(best_eval["eq_cn_predicted_return"]),
                        "eq_cn_weight_before": float(baseline_weights.loc[baseline_weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
                        "eq_cn_weight_after": float(best_eval["eq_cn_weight"]),
                        "eq_cn_rank_before": _eq_cn_rank(baseline_weights),
                        "eq_cn_rank_after": int(best_eval["eq_cn_rank"]),
                        "whether_eq_cn_became_material": int(best_eval["eq_cn_weight"] >= 0.05),
                    }
                )
            elif probe_def["probe_id"] in {"probe_120_target_up", "probe_120_target_down", "probe_120_deconcentration"}:
                baseline_weights = baseline["port120_weights"]
                scenario_weights = best_eval["weights_df"]
                portfolio_response_rows.append(
                    {
                        "candidate_name": "best_120_predictor",
                        "anchor_date": pd.Timestamp(anchor_date),
                        "probe_id": probe_def["probe_id"],
                        "predicted_return_change": float(best_eval["portfolio_predicted_return"] - baseline["port120_summary"]["portfolio_predicted_return"]),
                        "portfolio_return_change": float(best_eval["portfolio_predicted_return"] - baseline["port120_summary"]["portfolio_predicted_return"]),
                        "max_weight_change": float(best_eval["portfolio_max_weight"] - baseline["port120_summary"]["portfolio_max_weight"]),
                        "hhi_change": float(best_eval["portfolio_hhi"] - baseline["port120_summary"]["portfolio_hhi"]),
                        "effective_n_change": float(best_eval["portfolio_effective_n"] - baseline["port120_summary"]["portfolio_effective_n"]),
                        "eq_cn_weight_change": float(best_eval["eq_cn_weight"] - float(baseline_weights.loc[baseline_weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0])),
                        "top_weight_sleeve_before": str(baseline_weights.sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
                        "top_weight_sleeve_after": str(best_eval["top_weight_sleeve"]),
                    }
                )
                china_rows.append(
                    {
                        "candidate_name": "best_120_predictor",
                        "anchor_date": pd.Timestamp(anchor_date),
                        "probe_id": probe_def["probe_id"],
                        "eq_cn_predicted_return_before": float(baseline_weights.loc[baseline_weights["sleeve_id"].eq("EQ_CN"), "predicted_return"].iloc[0]),
                        "eq_cn_predicted_return_after": float(best_eval["eq_cn_predicted_return"]),
                        "eq_cn_weight_before": float(baseline_weights.loc[baseline_weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
                        "eq_cn_weight_after": float(best_eval["eq_cn_weight"]),
                        "eq_cn_rank_before": _eq_cn_rank(baseline_weights),
                        "eq_cn_rank_after": int(best_eval["eq_cn_rank"]),
                        "whether_eq_cn_became_material": int(best_eval["eq_cn_weight"] >= 0.05),
                    }
                )
            elif probe_def["probe_id"] == "probe_60_120_allocation_disagreement":
                for candidate_name, summary_key, weight_key, baseline_summary_key, baseline_weight_key in (
                    ("best_60_predictor", "summary_60", "weights_60", "port60_summary", "port60_weights"),
                    ("best_120_predictor", "summary_120", "weights_120", "port120_summary", "port120_weights"),
                ):
                    portfolio_response_rows.append(
                        {
                            "candidate_name": candidate_name,
                            "anchor_date": pd.Timestamp(anchor_date),
                            "probe_id": probe_def["probe_id"],
                            "predicted_return_change": float(best_eval[summary_key]["portfolio_predicted_return"] - baseline[baseline_summary_key]["portfolio_predicted_return"]),
                            "portfolio_return_change": float(best_eval[summary_key]["portfolio_predicted_return"] - baseline[baseline_summary_key]["portfolio_predicted_return"]),
                            "max_weight_change": float(best_eval[summary_key]["portfolio_max_weight"] - baseline[baseline_summary_key]["portfolio_max_weight"]),
                            "hhi_change": float(best_eval[summary_key]["portfolio_hhi"] - baseline[baseline_summary_key]["portfolio_hhi"]),
                            "effective_n_change": float(best_eval[summary_key]["portfolio_effective_n"] - baseline[baseline_summary_key]["portfolio_effective_n"]),
                            "eq_cn_weight_change": float(
                                best_eval[weight_key].loc[best_eval[weight_key]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]
                                - baseline[baseline_weight_key].loc[baseline[baseline_weight_key]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]
                            ),
                            "top_weight_sleeve_before": str(baseline[baseline_weight_key].sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
                            "top_weight_sleeve_after": str(best_eval[weight_key].sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
                        }
                    )
                china_rows.extend(
                    [
                        {
                            "candidate_name": "best_60_predictor",
                            "anchor_date": pd.Timestamp(anchor_date),
                            "probe_id": probe_def["probe_id"],
                            "eq_cn_predicted_return_before": float(baseline["port60_weights"].loc[baseline["port60_weights"]["sleeve_id"].eq("EQ_CN"), "predicted_return"].iloc[0]),
                            "eq_cn_predicted_return_after": float(best_eval["weights_60"].loc[best_eval["weights_60"]["sleeve_id"].eq("EQ_CN"), "predicted_return"].iloc[0]),
                            "eq_cn_weight_before": float(baseline["port60_weights"].loc[baseline["port60_weights"]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
                            "eq_cn_weight_after": float(best_eval["weights_60"].loc[best_eval["weights_60"]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
                            "eq_cn_rank_before": _eq_cn_rank(baseline["port60_weights"]),
                            "eq_cn_rank_after": _eq_cn_rank(best_eval["weights_60"]),
                            "whether_eq_cn_became_material": int(float(best_eval["weights_60"].loc[best_eval["weights_60"]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]) >= 0.05),
                        },
                        {
                            "candidate_name": "best_120_predictor",
                            "anchor_date": pd.Timestamp(anchor_date),
                            "probe_id": probe_def["probe_id"],
                            "eq_cn_predicted_return_before": float(baseline["port120_weights"].loc[baseline["port120_weights"]["sleeve_id"].eq("EQ_CN"), "predicted_return"].iloc[0]),
                            "eq_cn_predicted_return_after": float(best_eval["weights_120"].loc[best_eval["weights_120"]["sleeve_id"].eq("EQ_CN"), "predicted_return"].iloc[0]),
                            "eq_cn_weight_before": float(baseline["port120_weights"].loc[baseline["port120_weights"]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
                            "eq_cn_weight_after": float(best_eval["weights_120"].loc[best_eval["weights_120"]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
                            "eq_cn_rank_before": _eq_cn_rank(baseline["port120_weights"]),
                            "eq_cn_rank_after": _eq_cn_rank(best_eval["weights_120"]),
                            "whether_eq_cn_became_material": int(float(best_eval["weights_120"].loc[best_eval["weights_120"]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]) >= 0.05),
                        },
                    ]
                )
            elif probe_def["probe_id"] == "probe_60_120_prediction_disagreement":
                w60, s60 = _portfolio_for_state(context, best_state, "best_60_predictor")
                w120, s120 = _portfolio_for_state(context, best_state, "best_120_predictor")
                for candidate_name, summary_val, weight_val, baseline_summary_key, baseline_weight_key in (
                    ("best_60_predictor", s60, w60, "port60_summary", "port60_weights"),
                    ("best_120_predictor", s120, w120, "port120_summary", "port120_weights"),
                ):
                    portfolio_response_rows.append(
                        {
                            "candidate_name": candidate_name,
                            "anchor_date": pd.Timestamp(anchor_date),
                            "probe_id": probe_def["probe_id"],
                            "predicted_return_change": float(summary_val["portfolio_predicted_return"] - baseline[baseline_summary_key]["portfolio_predicted_return"]),
                            "portfolio_return_change": float(summary_val["portfolio_predicted_return"] - baseline[baseline_summary_key]["portfolio_predicted_return"]),
                            "max_weight_change": float(summary_val["portfolio_max_weight"] - baseline[baseline_summary_key]["portfolio_max_weight"]),
                            "hhi_change": float(summary_val["portfolio_hhi"] - baseline[baseline_summary_key]["portfolio_hhi"]),
                            "effective_n_change": float(summary_val["portfolio_effective_n"] - baseline[baseline_summary_key]["portfolio_effective_n"]),
                            "eq_cn_weight_change": float(
                                weight_val.loc[weight_val["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]
                                - baseline[baseline_weight_key].loc[baseline[baseline_weight_key]["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]
                            ),
                            "top_weight_sleeve_before": str(baseline[baseline_weight_key].sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
                            "top_weight_sleeve_after": str(weight_val.sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
                        }
                    )
                for label, pred_key, baseline_pred in (
                    ("elastic_net__full_firstpass__separate_60", "pred60", baseline["pred60"]),
                    ("ridge__full_firstpass__separate_120", "pred120", baseline["pred120"]),
                ):
                    china_rows.append(
                        {
                            "candidate_name": label,
                            "anchor_date": pd.Timestamp(anchor_date),
                            "probe_id": probe_def["probe_id"],
                            "eq_cn_predicted_return_before": float(baseline_pred.loc[baseline_pred["sleeve_id"].eq("EQ_CN"), "y_pred"].iloc[0]),
                            "eq_cn_predicted_return_after": float(best_eval[pred_key].loc[best_eval[pred_key]["sleeve_id"].eq("EQ_CN"), "y_pred"].iloc[0]),
                            "eq_cn_weight_before": np.nan,
                            "eq_cn_weight_after": np.nan,
                            "eq_cn_rank_before": int(baseline_pred.sort_values("y_pred", ascending=False).reset_index(drop=True).index[baseline_pred.sort_values("y_pred", ascending=False).reset_index(drop=True)["sleeve_id"].eq("EQ_CN")][0] + 1),
                            "eq_cn_rank_after": int(best_eval[pred_key].sort_values("y_pred", ascending=False).reset_index(drop=True).index[best_eval[pred_key].sort_values("y_pred", ascending=False).reset_index(drop=True)["sleeve_id"].eq("EQ_CN")][0] + 1),
                            "whether_eq_cn_became_material": 0,
                        }
                    )

            top_shift_series = pd.Series(best_state - baseline["state"], index=list(BASE_STATE_VARIABLES)).abs().sort_values(ascending=False)
            representative_rows.append(
                {
                    "case_id": f"{pd.Timestamp(anchor_date).date()}__{probe_def['probe_id']}",
                    "anchor_date": pd.Timestamp(anchor_date),
                    "probe_id": probe_def["probe_id"],
                    "candidate_name": probe_def["candidate_name"],
                    "baseline_score": float(_primary_metric_value(probe_def, baseline_eval)),
                    "scenario_score": best_value,
                    "plausibility_metric": float(best_row["plausibility_energy"]),
                    "short_case_label": _scenario_case_label(probe_def["probe_id"], pd.Timestamp(anchor_date)),
                    "short_case_interpretation": _short_interpretation(top_shift_series),
                }
            )
            for idx, variable_name in enumerate(BASE_STATE_VARIABLES):
                representative_state_rows.append(
                    {
                        "case_id": f"{pd.Timestamp(anchor_date).date()}__{probe_def['probe_id']}",
                        "anchor_date": pd.Timestamp(anchor_date),
                        "probe_id": probe_def["probe_id"],
                        "candidate_name": probe_def["candidate_name"],
                        "variable_name": variable_name,
                        "baseline_value": float(baseline["state"][idx]),
                        "scenario_value": float(best_state[idx]),
                        "shift_std_units": float((best_state[idx] - baseline["state"][idx]) / context.regularizer.bounds.scale[idx]),
                    }
                )

            manifest_rows.append(
                {
                    "anchor_date": pd.Timestamp(anchor_date),
                    "probe_id": probe_def["probe_id"],
                    "candidate_name": probe_def["candidate_name"],
                    "sampler": "bounded_mala",
                    "steps": int(probe_def["steps"]),
                    "step_size": float(probe_def["step_size"]),
                    "accepted_count": accepted_count,
                    "success_flag": 1,
                    "notes": probe_def["notes"],
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["anchor_date", "probe_id"]).reset_index(drop=True)
    state_shift_df = pd.DataFrame(state_shift_rows).sort_values(["anchor_date", "probe_id", "variable_name"]).reset_index(drop=True)
    portfolio_response_df = pd.DataFrame(portfolio_response_rows).sort_values(["anchor_date", "probe_id", "candidate_name"]).reset_index(drop=True)
    representative_cases_df = pd.DataFrame(representative_rows).sort_values(["anchor_date", "probe_id"]).reset_index(drop=True)
    china_df = pd.DataFrame(china_rows).sort_values(["anchor_date", "probe_id", "candidate_name"]).reset_index(drop=True)
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["anchor_date", "probe_id"]).reset_index(drop=True)
    representative_state_df = pd.DataFrame(representative_state_rows).sort_values(["anchor_date", "probe_id", "variable_name"]).reset_index(drop=True)
    report_text = _build_report(summary_df, representative_cases_df, state_shift_df)
    return ScenarioExperimentOutputs(
        report_text=report_text,
        summary=summary_df,
        state_shift_summary=state_shift_df,
        portfolio_response_summary=portfolio_response_df,
        representative_cases=representative_cases_df,
        china_diagnostics=china_df,
        experiment_manifest=manifest_df,
        representative_state_table=representative_state_df,
    )
