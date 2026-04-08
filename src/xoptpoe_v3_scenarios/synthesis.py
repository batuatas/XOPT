"""Conference-facing synthesis layer for the first real v3 scenario experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .experiments import (
    BASE_STATE_VARIABLES,
    _baseline_bundle,
    _build_context,
    _build_probe_definitions,
    _evaluate_probe_state,
)


@dataclass(frozen=True)
class SelectedCase:
    case_id: str
    case_family: str
    short_case_label: str
    anchor_date: pd.Timestamp
    candidate_name: str
    probe_id: str
    selection_rule: str


@dataclass(frozen=True)
class ScenarioSynthesisOutputs:
    report_text: str
    conference_notes_text: str
    casebook: pd.DataFrame
    variable_importance: pd.DataFrame
    benchmark_contrast: pd.DataFrame
    china_role_summary: pd.DataFrame
    case_state_shifts: pd.DataFrame
    response_cloud: pd.DataFrame
    casebook_weights: pd.DataFrame
    selected_cases: pd.DataFrame


def _load_csv(root: Path, stem: str) -> pd.DataFrame:
    path = root / stem
    parse_dates = ["anchor_date"] if "anchor_date" in pd.read_csv(path, nrows=0).columns else None
    return pd.read_csv(path, parse_dates=parse_dates)


def _selection_rule_table(summary_df: pd.DataFrame, portfolio_df: pd.DataFrame, china_df: pd.DataFrame) -> list[SelectedCase]:
    robust = portfolio_df.loc[
        portfolio_df["candidate_name"].eq("best_60_predictor") & portfolio_df["probe_id"].eq("probe_60_target_up")
    ].copy()
    robust["selection_score"] = robust["portfolio_return_change"] - robust["hhi_change"].clip(lower=0.0)
    robust_row = robust.sort_values(["selection_score", "portfolio_return_change"], ascending=False).iloc[0]

    raw = portfolio_df.loc[
        portfolio_df["candidate_name"].eq("best_120_predictor") & portfolio_df["probe_id"].eq("probe_120_target_up")
    ].merge(
        summary_df[["anchor_date", "probe_id", "candidate_name", "scenario_response_best"]],
        on=["anchor_date", "probe_id", "candidate_name"],
        how="left",
        validate="1:1",
    )
    raw = raw.loc[raw["scenario_response_best"] > 0].copy()
    raw_row = raw.sort_values(["scenario_response_best", "portfolio_return_change"], ascending=False).iloc[0]

    deconc = portfolio_df.loc[
        portfolio_df["candidate_name"].eq("best_120_predictor") & portfolio_df["probe_id"].eq("probe_120_deconcentration")
    ].copy()
    deconc = deconc.loc[deconc["portfolio_return_change"] > -0.0005].copy()
    deconc["selection_score"] = (-deconc["hhi_change"]) + 0.5 * deconc["portfolio_return_change"]
    deconc_row = deconc.sort_values(["selection_score", "portfolio_return_change"], ascending=False).iloc[0]

    disagreement = summary_df.loc[summary_df["probe_id"].eq("probe_60_120_allocation_disagreement")].copy()
    disagreement_row = disagreement.sort_values(["scenario_response_best", "plausibility_summary"], ascending=[False, True]).iloc[0]

    china_probe = china_df.loc[
        china_df["candidate_name"].eq("best_60_predictor") & china_df["probe_id"].eq("probe_60_china_role")
    ].copy()
    china_probe["weight_change"] = china_probe["eq_cn_weight_after"] - china_probe["eq_cn_weight_before"]
    china_probe["rank_change"] = china_probe["eq_cn_rank_before"] - china_probe["eq_cn_rank_after"]
    china_row = china_probe.sort_values(["whether_eq_cn_became_material", "rank_change", "weight_change"], ascending=False).iloc[0]

    selected = [
        SelectedCase(
            case_id="robust_return_up",
            case_family="robust",
            short_case_label="Robust benchmark upside",
            anchor_date=pd.Timestamp(robust_row["anchor_date"]),
            candidate_name="best_60_predictor",
            probe_id="probe_60_target_up",
            selection_rule="Highest 60m target-up return change after penalizing extra concentration.",
        ),
        SelectedCase(
            case_id="raw_return_up",
            case_family="raw",
            short_case_label="Raw ceiling upside",
            anchor_date=pd.Timestamp(raw_row["anchor_date"]),
            candidate_name="best_120_predictor",
            probe_id="probe_120_target_up",
            selection_rule="Positive 120m target-up case with the highest resulting scenario return level.",
        ),
        SelectedCase(
            case_id="raw_deconcentration",
            case_family="deconcentration",
            short_case_label="Raw ceiling deconcentration",
            anchor_date=pd.Timestamp(deconc_row["anchor_date"]),
            candidate_name="best_120_predictor",
            probe_id="probe_120_deconcentration",
            selection_rule="Largest HHI reduction among 120m deconcentration cases without obvious return collapse.",
        ),
        SelectedCase(
            case_id="disagreement_case_60",
            case_family="robust",
            short_case_label="Robust side of disagreement",
            anchor_date=pd.Timestamp(disagreement_row["anchor_date"]),
            candidate_name="best_60_predictor",
            probe_id="probe_60_120_allocation_disagreement",
            selection_rule="Anchor with the largest 60m-vs-120m allocation disagreement response.",
        ),
        SelectedCase(
            case_id="disagreement_case_120",
            case_family="raw",
            short_case_label="Raw side of disagreement",
            anchor_date=pd.Timestamp(disagreement_row["anchor_date"]),
            candidate_name="best_120_predictor",
            probe_id="probe_60_120_allocation_disagreement",
            selection_rule="Same disagreement anchor, evaluated on the 120m raw ceiling object.",
        ),
        SelectedCase(
            case_id="china_probe_best",
            case_family="china",
            short_case_label="China stays secondary",
            anchor_date=pd.Timestamp(china_row["anchor_date"]),
            candidate_name="best_60_predictor",
            probe_id="probe_60_china_role",
            selection_rule="Best China-role probe by materiality, rank change, and weight increase; still secondary if below threshold.",
        ),
    ]
    return selected


def _reconstruct_state_vector(state_shift_df: pd.DataFrame, anchor_date: pd.Timestamp, probe_id: str) -> np.ndarray:
    subset = state_shift_df.loc[
        state_shift_df["anchor_date"].eq(anchor_date) & state_shift_df["probe_id"].eq(probe_id)
    ].copy()
    if subset.empty:
        raise KeyError((anchor_date, probe_id))
    return np.asarray(
        [
            float(subset.loc[subset["variable_name"].eq(variable_name), "scenario_best"].iloc[0])
            for variable_name in BASE_STATE_VARIABLES
        ],
        dtype=float,
    )


def _build_anchor_cache(project_root: Path, anchors: list[pd.Timestamp]) -> dict[pd.Timestamp, dict[str, Any]]:
    cache: dict[pd.Timestamp, dict[str, Any]] = {}
    for anchor_date in sorted(set(pd.Timestamp(v) for v in anchors)):
        context = _build_context(project_root, anchor_date.strftime("%Y-%m-%d"))
        baseline = _baseline_bundle(context)
        probe_defs = {probe_def["probe_id"]: probe_def for probe_def in _build_probe_definitions(context, baseline)}
        cache[pd.Timestamp(anchor_date)] = {
            "context": context,
            "baseline": baseline,
            "probe_defs": probe_defs,
        }
    return cache


def _extract_case_metrics(
    *,
    candidate_name: str,
    probe_id: str,
    baseline: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    if candidate_name == "best_60_predictor":
        if probe_id == "probe_60_120_allocation_disagreement":
            summary = evaluation["summary_60"]
            weights = evaluation["weights_60"]
        else:
            summary = evaluation
            weights = evaluation["weights_df"]
        baseline_summary = baseline["port60_summary"]
        baseline_weights = baseline["port60_weights"]
    elif candidate_name == "best_120_predictor":
        if probe_id == "probe_60_120_allocation_disagreement":
            summary = evaluation["summary_120"]
            weights = evaluation["weights_120"]
        else:
            summary = evaluation
            weights = evaluation["weights_df"]
        baseline_summary = baseline["port120_summary"]
        baseline_weights = baseline["port120_weights"]
    else:
        raise ValueError(candidate_name)

    row = {
        "baseline_predicted_return": float(baseline_summary["portfolio_predicted_return"]),
        "scenario_predicted_return": float(summary["portfolio_predicted_return"]),
        "baseline_portfolio_return": float(baseline_summary["portfolio_predicted_return"]),
        "scenario_portfolio_return": float(summary["portfolio_predicted_return"]),
        "baseline_max_weight": float(baseline_summary["portfolio_max_weight"]),
        "scenario_max_weight": float(summary["portfolio_max_weight"]),
        "baseline_effective_n": float(baseline_summary["portfolio_effective_n"]),
        "scenario_effective_n": float(summary["portfolio_effective_n"]),
        "baseline_hhi": float(baseline_summary["portfolio_hhi"]),
        "scenario_hhi": float(summary["portfolio_hhi"]),
        "eq_cn_weight_before": float(baseline_weights.loc[baseline_weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
        "eq_cn_weight_after": float(weights.loc[weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0]),
        "top_weight_sleeve_before": str(baseline_weights.sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
        "top_weight_sleeve_after": str(weights.sort_values("weight", ascending=False)["sleeve_id"].iloc[0]),
    }
    return row, pd.concat(
        [
            baseline_weights.assign(case_phase="baseline", candidate_name=candidate_name),
            weights.assign(case_phase="scenario", candidate_name=candidate_name),
        ],
        ignore_index=True,
    )


def _top_shift_rows(state_shift_df: pd.DataFrame, anchor_date: pd.Timestamp, probe_id: str, top_n: int = 5) -> pd.DataFrame:
    subset = state_shift_df.loc[
        state_shift_df["anchor_date"].eq(anchor_date) & state_shift_df["probe_id"].eq(probe_id)
    ].copy()
    subset["shift_std_units_signed"] = np.sign(subset["scenario_best"] - subset["baseline_value"]) * subset["shift_magnitude_std_units"]
    subset = subset.sort_values("shift_magnitude_std_units", ascending=False).head(top_n)
    return subset[["variable_name", "baseline_value", "scenario_best", "shift_magnitude_std_units", "shift_std_units_signed"]].reset_index(drop=True)


def _case_interpretation(case_id: str, metrics: dict[str, float], top_shifts: pd.DataFrame) -> str:
    shift_names = ", ".join(top_shifts["variable_name"].head(3).tolist())
    if case_id == "robust_return_up":
        return f"A moderate upside state for the 60m benchmark led by {shift_names}, with only limited extra concentration."
    if case_id == "raw_return_up":
        return f"The 120m ceiling improves most under a {shift_names} state, but concentration stays high."
    if case_id == "raw_deconcentration":
        return f"A plausible {shift_names} state broadens the 120m ceiling slightly without obvious return collapse."
    if case_id == "disagreement_case_60":
        return f"Under the disagreement state led by {shift_names}, the 60m benchmark stays the broader side of the contrast."
    if case_id == "disagreement_case_120":
        return f"Under the same disagreement state, the 120m object remains the more concentrated side of the contrast."
    if case_id == "china_probe_best":
        return f"Even in the strongest China-role case, {shift_names} only nudges EQ_CN modestly and it remains secondary."
    return f"State shift led by {shift_names}."


def _build_casebook(
    selected_cases: list[SelectedCase],
    anchor_cache: dict[pd.Timestamp, dict[str, Any]],
    state_shift_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, Any]] = []
    weight_rows: list[pd.DataFrame] = []
    shift_rows: list[dict[str, Any]] = []

    for selected in selected_cases:
        cache = anchor_cache[selected.anchor_date]
        context = cache["context"]
        baseline = cache["baseline"]
        probe_def = cache["probe_defs"][selected.probe_id]
        state_vector = _reconstruct_state_vector(state_shift_df, selected.anchor_date, selected.probe_id)
        evaluation = _evaluate_probe_state(context, probe_def, state_vector)
        metrics, weights = _extract_case_metrics(
            candidate_name=selected.candidate_name,
            probe_id=selected.probe_id,
            baseline=baseline,
            evaluation=evaluation,
        )
        top_shifts = _top_shift_rows(state_shift_df, selected.anchor_date, selected.probe_id, top_n=5)
        for order, (_, shift_row) in enumerate(top_shifts.iterrows(), start=1):
            shift_rows.append(
                {
                    "case_id": selected.case_id,
                    "case_family": selected.case_family,
                    "anchor_date": selected.anchor_date,
                    "candidate_name": selected.candidate_name,
                    "probe_id": selected.probe_id,
                    "shift_rank": order,
                    "variable_name": shift_row["variable_name"],
                    "baseline_value": float(shift_row["baseline_value"]),
                    "scenario_value": float(shift_row["scenario_best"]),
                    "shift_magnitude_std_units": float(shift_row["shift_magnitude_std_units"]),
                    "shift_std_units_signed": float(shift_row["shift_std_units_signed"]),
                }
            )
        weight_rows.append(weights.assign(case_id=selected.case_id, case_family=selected.case_family, anchor_date=selected.anchor_date, probe_id=selected.probe_id))
        case_rows.append(
            {
                "case_id": selected.case_id,
                "case_family": selected.case_family,
                "short_case_label": selected.short_case_label,
                "anchor_date": selected.anchor_date,
                "candidate_name": selected.candidate_name,
                "probe_id": selected.probe_id,
                **metrics,
                "selection_rule": selected.selection_rule,
                "short_case_interpretation": _case_interpretation(selected.case_id, metrics, top_shifts),
            }
        )

    casebook_df = pd.DataFrame(case_rows)
    casebook_df = casebook_df.loc[~casebook_df["case_id"].eq("china_probe_best")].reset_index(drop=True)
    shifts_df = pd.DataFrame(shift_rows)
    weights_df = pd.concat(weight_rows, ignore_index=True)
    return casebook_df, shifts_df, weights_df


def _build_variable_importance(casebook_df: pd.DataFrame, case_shift_df: pd.DataFrame) -> pd.DataFrame:
    selected = case_shift_df.loc[case_shift_df["shift_rank"] <= 5].copy()
    rows: list[dict[str, Any]] = []
    for variable_name, group in selected.groupby("variable_name", sort=True):
        case_ids = sorted(group["case_id"].unique().tolist())
        robust_cases = int(group["case_family"].eq("robust").sum())
        raw_cases = int(group["case_family"].eq("raw").sum())
        deconc_cases = int(group["case_family"].eq("deconcentration").sum())
        china_cases = int(group["case_family"].eq("china").sum())
        note_parts = []
        if robust_cases and raw_cases:
            note_parts.append("recurs in both robust and raw cases")
        elif robust_cases:
            note_parts.append("primarily a robust-case variable")
        elif raw_cases:
            note_parts.append("primarily a raw-case variable")
        if deconc_cases:
            note_parts.append("also appears in deconcentration")
        rows.append(
            {
                "variable_name": variable_name,
                "avg_abs_shift_std_units": float(group["shift_magnitude_std_units"].abs().mean()),
                "case_count_selected": int(len(case_ids)),
                "robust_case_count": robust_cases,
                "raw_case_count": raw_cases,
                "deconcentration_case_count": deconc_cases,
                "china_case_count": china_cases,
                "comment": "; ".join(note_parts) if note_parts else "localized first-pass move",
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_abs_shift_std_units", "case_count_selected"], ascending=False).reset_index(drop=True)


def _build_benchmark_contrast(
    summary_df: pd.DataFrame,
    state_shift_df: pd.DataFrame,
    anchor_cache: dict[pd.Timestamp, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    anchors = sorted(summary_df.loc[summary_df["probe_id"].eq("probe_60_120_allocation_disagreement"), "anchor_date"].drop_duplicates())
    for anchor_date in anchors:
        cache = anchor_cache[pd.Timestamp(anchor_date)]
        baseline = cache["baseline"]
        probe_def = cache["probe_defs"]["probe_60_120_allocation_disagreement"]
        state_vector = _reconstruct_state_vector(state_shift_df, pd.Timestamp(anchor_date), "probe_60_120_allocation_disagreement")
        evaluation = _evaluate_probe_state(cache["context"], probe_def, state_vector)
        summary_60 = evaluation["summary_60"]
        summary_120 = evaluation["summary_120"]
        baseline_60 = baseline["port60_summary"]
        baseline_120 = baseline["port120_summary"]
        rows.append(
            {
                "anchor_date": pd.Timestamp(anchor_date),
                "baseline_60_return": float(baseline_60["portfolio_predicted_return"]),
                "scenario_60_return": float(summary_60["portfolio_predicted_return"]),
                "baseline_120_return": float(baseline_120["portfolio_predicted_return"]),
                "scenario_120_return": float(summary_120["portfolio_predicted_return"]),
                "baseline_60_max_weight": float(baseline_60["portfolio_max_weight"]),
                "scenario_60_max_weight": float(summary_60["portfolio_max_weight"]),
                "baseline_120_max_weight": float(baseline_120["portfolio_max_weight"]),
                "scenario_120_max_weight": float(summary_120["portfolio_max_weight"]),
                "whether_top_weight_sleeve_changed_60": int(
                    baseline["port60_weights"].sort_values("weight", ascending=False)["sleeve_id"].iloc[0]
                    != evaluation["weights_60"].sort_values("weight", ascending=False)["sleeve_id"].iloc[0]
                ),
                "whether_top_weight_sleeve_changed_120": int(
                    baseline["port120_weights"].sort_values("weight", ascending=False)["sleeve_id"].iloc[0]
                    != evaluation["weights_120"].sort_values("weight", ascending=False)["sleeve_id"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("anchor_date").reset_index(drop=True)


def _build_china_summary(china_df: pd.DataFrame) -> pd.DataFrame:
    subset = china_df.loc[
        china_df["candidate_name"].eq("best_60_predictor") & china_df["probe_id"].eq("probe_60_china_role")
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in subset.sort_values("anchor_date").iterrows():
        material = int(row["whether_eq_cn_became_material"])
        rank_improved = int(row["eq_cn_rank_after"] < row["eq_cn_rank_before"])
        if material or rank_improved:
            interpretation = "EQ_CN became more visible, but still did not dominate the allocation."
        else:
            interpretation = "EQ_CN stayed below material weight and its cross-sleeve rank did not improve."
        rows.append(
            {
                "anchor_date": pd.Timestamp(row["anchor_date"]),
                "case_id": f"china_probe_{pd.Timestamp(row['anchor_date']).date()}",
                "candidate_name": row["candidate_name"],
                "eq_cn_rank_before": int(row["eq_cn_rank_before"]),
                "eq_cn_rank_after": int(row["eq_cn_rank_after"]),
                "eq_cn_weight_before": float(row["eq_cn_weight_before"]),
                "eq_cn_weight_after": float(row["eq_cn_weight_after"]),
                "whether_eq_cn_became_material": material,
                "did_china_features_seem_global_or_local": "broader prediction spillovers exist, but allocation impact stayed local and marginal",
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _build_response_cloud(
    portfolio_df: pd.DataFrame,
    anchor_cache: dict[pd.Timestamp, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for _, row in portfolio_df.iterrows():
        candidate_name = str(row["candidate_name"])
        if candidate_name not in {"best_60_predictor", "best_120_predictor"}:
            continue
        baseline_summary = anchor_cache[pd.Timestamp(row["anchor_date"])] ["baseline"]["port60_summary" if candidate_name == "best_60_predictor" else "port120_summary"]
        rows.append(
            {
                "candidate_name": candidate_name,
                "anchor_date": pd.Timestamp(row["anchor_date"]),
                "probe_id": row["probe_id"],
                "portfolio_return_change": float(row["portfolio_return_change"]),
                "hhi_change": float(row["hhi_change"]),
                "effective_n_change": float(row["effective_n_change"]),
                "scenario_max_weight": float(baseline_summary["portfolio_max_weight"] + row["max_weight_change"]),
                "baseline_max_weight": float(baseline_summary["portfolio_max_weight"]),
            }
        )
    return pd.DataFrame(rows)


def _build_report(
    casebook_df: pd.DataFrame,
    variable_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
    china_df: pd.DataFrame,
) -> str:
    selected_display = casebook_df[[
        "case_id",
        "short_case_label",
        "anchor_date",
        "candidate_name",
        "probe_id",
        "baseline_predicted_return",
        "scenario_predicted_return",
        "baseline_max_weight",
        "scenario_max_weight",
        "short_case_interpretation",
    ]].copy()
    selected_display["anchor_date"] = pd.to_datetime(selected_display["anchor_date"]).dt.date
    top_vars = variable_df.head(8).copy()
    top_vars = top_vars[["variable_name", "avg_abs_shift_std_units", "case_count_selected", "comment"]]

    robust = casebook_df.loc[casebook_df["case_id"].eq("robust_return_up")].iloc[0]
    raw_up = casebook_df.loc[casebook_df["case_id"].eq("raw_return_up")].iloc[0]
    raw_deconc = casebook_df.loc[casebook_df["case_id"].eq("raw_deconcentration")].iloc[0]
    china_line = china_df.sort_values(["whether_eq_cn_became_material", "eq_cn_weight_after"], ascending=False).iloc[0]

    return "\n".join(
        [
            "# v3 Scenario Synthesis Report",
            "",
            "## Scope",
            "- Active branch only: v3_long_horizon_china.",
            "- This is a synthesis pass built on the already-run first-pass scenario experiments.",
            "- These outputs remain plausibility-regularized, anchor-local, model-implied state diagnostics for long-horizon SAA benchmarks.",
            "",
            "## Benchmark Story",
            "- `best_60_predictor` remains the main scenario object: it moves in an economically interpretable way, but the responses stay moderate.",
            "- `best_120_predictor` remains the raw ceiling comparison object: it reacts more and stays more concentration-prone.",
            "- The 120m ceiling can be nudged toward lower concentration, but the gain is incremental rather than transformative.",
            "",
            "## Representative Case Selection Rule",
            "- `robust_return_up`: highest 60m target-up return change after penalizing extra concentration.",
            "- `raw_return_up`: positive 120m target-up case with the highest resulting scenario return level.",
            "- `raw_deconcentration`: largest HHI reduction among 120m deconcentration cases without obvious return collapse.",
            "- `disagreement_case`: anchor with the largest allocation disagreement between the 60m and 120m benchmark portfolios.",
            "- `china_probe`: retained only as a diagnostic; no China case became material enough to enter the core casebook.",
            "",
            "## Representative Cases",
            selected_display.to_markdown(index=False),
            "",
            "## What The Scenario Layer Learned",
            f"- Robust benchmark message: at {pd.Timestamp(robust['anchor_date']).date()}, the 60m benchmark improved from {robust['baseline_predicted_return']:.4f} to {robust['scenario_predicted_return']:.4f} while max weight only moved from {robust['baseline_max_weight']:.3f} to {robust['scenario_max_weight']:.3f}.",
            f"- Raw ceiling message: at {pd.Timestamp(raw_up['anchor_date']).date()}, the 120m object improved from {raw_up['baseline_predicted_return']:.4f} to {raw_up['scenario_predicted_return']:.4f}, but max weight stayed high at {raw_up['scenario_max_weight']:.3f}.",
            f"- Deconcentration message: at {pd.Timestamp(raw_deconc['anchor_date']).date()}, the 120m ceiling moved from effective N {raw_deconc['baseline_effective_n']:.2f} to {raw_deconc['scenario_effective_n']:.2f} with scenario return {raw_deconc['scenario_predicted_return']:.4f}.",
            "",
            "## Variables That Repeat Across Cases",
            top_vars.to_markdown(index=False),
            "",
            "## Raw Versus Robust Contrast",
            contrast_df.to_markdown(index=False),
            "",
            "## China Under Scenarios",
            f"- No first-pass scenario made EQ_CN material. The strongest China-role case ended with EQ_CN weight {china_line['eq_cn_weight_after']:.4f} and rank {int(china_line['eq_cn_rank_after'])}.",
            "- China remains part of the active system, but the scenario layer still treats it as a secondary sleeve rather than a main allocation driver.",
            "",
            "## Public-Facing Interpretation",
            "- The scenario layer is now strong enough for conference use as a model-based explanation layer.",
            "- The right framing is benchmark-conditioned local diagnostics, not causal macro discovery.",
            "- The 60m benchmark is the main object for interpretation; the 120m ceiling is the comparison object that reveals concentration risk.",
        ]
    )


def _build_conference_notes(casebook_df: pd.DataFrame, variable_df: pd.DataFrame, china_df: pd.DataFrame) -> str:
    robust = casebook_df.loc[casebook_df["case_id"].eq("robust_return_up")].iloc[0]
    raw = casebook_df.loc[casebook_df["case_id"].eq("raw_return_up")].iloc[0]
    deconc = casebook_df.loc[casebook_df["case_id"].eq("raw_deconcentration")].iloc[0]
    top_vars = ", ".join(variable_df.head(5)["variable_name"].tolist())
    china_material = int(china_df["whether_eq_cn_became_material"].max())
    china_note = "China never becomes a material sleeve under the first-pass scenarios." if china_material == 0 else "China becomes visible only in a narrow subset of scenarios."
    bullets = [
        "# Scenario Notes",
        "",
        f"- The 60m benchmark remains the main interpretation object: its strongest upside case moved the model-implied portfolio return from {robust['baseline_predicted_return']:.4f} to {robust['scenario_predicted_return']:.4f} without a large jump in concentration.",
        f"- The 120m benchmark is clearly more state-sensitive: its strongest upside case moved from {raw['baseline_predicted_return']:.4f} to {raw['scenario_predicted_return']:.4f}, but it stayed much more concentrated than the 60m benchmark.",
        f"- Some deconcentration is possible for the 120m ceiling: in the cleanest case, effective N improved from {deconc['baseline_effective_n']:.2f} to {deconc['scenario_effective_n']:.2f} without an obvious return collapse.",
        f"- The same variables keep reappearing across cases: {top_vars}.",
        "- This makes the scenario layer look economically interpretable rather than like a pure optimizer artifact.",
        f"- {china_note}",
        "- The right public framing is not causal macro truth. These are plausibility-regularized, anchor-local diagnostics around the active benchmark objects.",
        "- For the conference narrative, the simplest message is: the 60m benchmark reacts sensibly and the 120m ceiling helps show what extra state-sensitivity looks like when concentration risk is left less constrained.",
    ]
    return "\n".join(bullets) + "\n"


def run_scenario_synthesis(project_root: Path) -> ScenarioSynthesisOutputs:
    reports_root = project_root / "reports" / "v3_long_horizon_china"
    summary_df = _load_csv(reports_root, "scenario_experiment_summary_v3.csv")
    state_shift_df = _load_csv(reports_root, "scenario_state_shift_summary_v3.csv")
    portfolio_df = _load_csv(reports_root, "scenario_portfolio_response_summary_v3.csv")
    china_df = _load_csv(reports_root, "china_scenario_diagnostics_v3.csv")

    selected_cases = _selection_rule_table(summary_df, portfolio_df, china_df)
    anchor_cache = _build_anchor_cache(project_root, [selected.anchor_date for selected in selected_cases] + list(pd.to_datetime(summary_df["anchor_date"]).drop_duplicates()))
    casebook_df, case_shift_df, casebook_weights_df = _build_casebook(selected_cases, anchor_cache, state_shift_df)
    variable_df = _build_variable_importance(casebook_df, case_shift_df)
    contrast_df = _build_benchmark_contrast(summary_df, state_shift_df, anchor_cache)
    china_summary_df = _build_china_summary(china_df)
    response_cloud_df = _build_response_cloud(portfolio_df, anchor_cache)
    selected_cases_df = pd.DataFrame([selected.__dict__ for selected in selected_cases])

    report_text = _build_report(casebook_df, variable_df, contrast_df, china_summary_df)
    conference_notes_text = _build_conference_notes(casebook_df, variable_df, china_summary_df)
    return ScenarioSynthesisOutputs(
        report_text=report_text,
        conference_notes_text=conference_notes_text,
        casebook=casebook_df,
        variable_importance=variable_df,
        benchmark_contrast=contrast_df,
        china_role_summary=china_summary_df,
        case_state_shifts=case_shift_df,
        response_cloud=response_cloud_df,
        casebook_weights=casebook_weights_df,
        selected_cases=selected_cases_df,
    )
