"""Dry-run audit for the v3 scenario-generation scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import default_paths, load_active_artifacts
from .mala import finite_difference_gradient, run_bounded_mala
from .pipelines import (
    PortfolioCandidateSpec,
    PredictorCandidateSpec,
    build_default_candidate_specs,
    build_scenario_rows,
    fit_portfolio_candidate,
    fit_predictor_candidate,
)
from .probes import SimilarReturnDistinctAllocationProbe, TargetReturnProbe, build_default_probe_specs, ReturnConcentrationProbe
from .regularizers import build_regularizer
from .state import build_anchor_context, build_state_manifest, default_state_spec


@dataclass(frozen=True)
class AuditResult:
    """Structured outputs from the scenario scaffold audit."""

    scenario_candidate_manifest: pd.DataFrame
    scenario_state_manifest: pd.DataFrame
    scenario_probe_manifest: pd.DataFrame
    scenario_scaffold_check: pd.DataFrame
    pipeline_audit_report: str
    adaptation_plan_report: str


def _component_row(component: str, status: str, details: str) -> dict[str, str]:
    return {"component": component, "status": status, "details": details}


def _build_candidate_manifest(
    predictor_specs: list[PredictorCandidateSpec],
    portfolio_specs: list[PortfolioCandidateSpec],
) -> pd.DataFrame:
    rows = []
    for spec in predictor_specs:
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_type": "predictor anchor",
                "source_model_name": spec.source_model_name,
                "horizon": ",".join(str(h) for h in spec.horizons),
                "role_in_scenario_stage": spec.role_in_scenario_stage,
                "active_flag": int(spec.role_in_scenario_stage != "comparator only"),
                "notes": f"{spec.model_name} using {spec.feature_set_name}",
            }
        )
    for spec in portfolio_specs:
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_type": "portfolio benchmark",
                "source_model_name": spec.source_model_name,
                "horizon": str(spec.horizon),
                "role_in_scenario_stage": spec.role_in_scenario_stage,
                "active_flag": int(spec.role_in_scenario_stage == "active robust benchmark"),
                "notes": f"locked allocator lambda={spec.optimizer_config.lambda_risk}, kappa={spec.optimizer_config.kappa}, omega={spec.optimizer_config.omega_type}",
            }
        )
    rows.extend(
        [
            {
                "candidate_id": "e2e_nn_signal",
                "candidate_type": "neural comparator",
                "source_model_name": "e2e_nn_signal",
                "horizon": "60,120",
                "role_in_scenario_stage": "comparator only",
                "active_flag": 0,
                "notes": "Carried forward for interpretation, but not instantiated in the first scaffold because no scenario-ready persisted model object ships in the active v3 artifacts.",
            },
            {
                "candidate_id": "combined_std_120tilt_top_k_capped",
                "candidate_type": "portfolio comparator",
                "source_model_name": "combined_std_120tilt_top_k_capped",
                "horizon": "60,120",
                "role_in_scenario_stage": "comparator only",
                "active_flag": 0,
                "notes": "Useful concentration-control comparator, but not the first active target for scenario probing.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _build_probe_manifest() -> pd.DataFrame:
    return pd.DataFrame([probe.__dict__ for probe in build_default_probe_specs()])


def _pipeline_audit_report(anchor_date: pd.Timestamp, checks: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# v3 Scenario Pipeline Audit",
            "",
            "## Old Mehmet Pipeline: What The Code Actually Does",
            "- `mehmet/probe_eval.py` is the old scenario core. It defines a macro-state probing function `G`, wraps trained PTO/PAO models, and evaluates robust long-only allocations under manipulated macro conditions.",
            "- `mehmet/script1.py` targets benchmark return, `mehmet/script2.py` targets entropy/diversification, and `mehmet/script3.py` compares two trained decision pipelines under the same macro perturbation.",
            "- The manipulated variable in the old code is a low-dimensional macro state `m`; firm characteristics and the trailing covariance matrix stay fixed at the anchor date.",
            "- The old Gibbs / POE target is an energy over macro states. In the benchmark-return case it is squared distance to a return target; in the entropy case it is negative portfolio entropy; in the contrast case it trades off similar return against different risk behavior.",
            "- Gradients in the old code are taken with respect to the macro state, not with respect to portfolio weights or historical returns.",
            "- The old scripts call an external `end2endportfolio.src.langevin` MALA implementation, but that dependency is not present in this repo. The v3 scaffold therefore ships its own small bounded MALA implementation.",
            "",
            "## `var1_regularizer.py`: What It Does",
            "- It fits a Gaussian VAR(1) on the old 9-variable macro state and computes innovation Mahalanobis energies.",
            "- That object is a dynamic plausibility prior: it penalizes macro states that are unlikely relative to the previous month, rather than acting as a pure static L2 distance.",
            "- The idea survives into v3, but the state vector has to change because the active project no longer uses the old Goyal-Welch 9-variable setup.",
            "",
            "## What Was Reused Directly",
            "- Low-dimensional manipulated state.",
            "- Fixed anchor convention: hold non-manipulated inputs fixed at a chosen month-end.",
            "- Energy-style probes over the full prediction-plus-allocation pipeline.",
            "- Dynamic plausibility regularization via a fitted VAR(1) prior.",
            "- MALA-style local exploration.",
            "",
            "## What Had To Change For v3 Long-Horizon SAA",
            "- The old firm-level 1-month TAA design was removed. v3 scenarios operate on sleeve-level rows from `data/final_v3_long_horizon_china/modeling_panel_hstack.parquet`.",
            "- Top-K stock selection, firm characteristic interactions, and 1-step realized return objectives are not reused.",
            "- The active targets are now the locked v3 supervised prediction anchors and their downstream SAA portfolios, not the old archived TAA models.",
            "- The first manipulated state is the interpretable canonical macro block only; enrichments such as `china_cli`, `jp_pe_ratio`, `cape_local`, `mom_12_1`, and `vol_12m` are held fixed at the anchor in this first pass.",
            "",
            "## Dry-Run Anchor",
            f"- Smoke test anchor date: `{anchor_date.date()}`.",
            "- This anchor is recent enough to fit both 60m and 120m predictors using only labels observable by the anchor date, while still allowing both scenario horizons to be instantiated from the active stacked panel.",
            "",
            "## Scaffold Check Summary",
            checks.to_markdown(index=False),
            "",
            "## Important Caveat",
            "- The old prompt list referenced `reports/v3_long_horizon_china/final_benchmark_manifest_v3.csv`, but the active manifest in this repo is `data/modeling_v3/final_benchmark_manifest_v3.csv`. The scaffold resolves that actual path explicitly.",
        ]
    )


def _adaptation_plan_report(anchor_date: pd.Timestamp) -> str:
    return "\n".join(
        [
            "# v3 Scenario Adaptation Plan",
            "",
            "## Active Target Pipelines",
            "- Primary robust portfolio benchmark: `best_60_predictor` built from `elastic_net__full_firstpass__separate_60` plus the locked robust allocator (`lambda=10`, `kappa=0.1`, `omega=identity`).",
            "- Raw ceiling portfolio benchmark: `best_120_predictor` built from `ridge__full_firstpass__separate_120` plus the locked robust allocator (`lambda=10`, `kappa=0.1`, `omega=diag`).",
            "- Prediction anchors carried forward directly: `elastic_net__full_firstpass__separate_60` and `ridge__full_firstpass__separate_120`.",
            "- Shared predictor remains comparator only; E2E remains comparator only in this scaffold because the active artifacts do not include a clean persisted scenario-ready neural model object.",
            "",
            "## First Manipulated State",
            "- Manipulate only the canonical macro base state: US / EA / JP inflation, unemployment, short rate, long rate, plus `usd_broad`, `vix`, `us_real10y`, `ig_oas`, and `oil_wti`.",
            "- Rebuild derived deltas, term slopes, 1m/12m log changes, and selected active interaction terms from that state.",
            "- Hold enrichments fixed at the anchor date in the first pass. This preserves interpretability and avoids turning the first MALA state into a high-dimensional opaque feature vector.",
            "",
            "## Anchor Convention",
            f"- Choose an anchor month-end `t`; the audit uses `{anchor_date.date()}`.",
            "- Hold all non-manipulated features at their actual anchor-date values.",
            "- Refit the locked supervised predictor specification on all labels observable by `t` for the relevant horizon(s).",
            "- Feed the manipulated state through the fitted predictor and then, when relevant, the locked robust allocator.",
            "- Evaluate probe energies on that anchor-date pipeline state only. This is a conditional scenario design, not a re-estimation of the full historical experiment zoo.",
            "",
            "## First Implemented Probe Families",
            "- `probe_60_target_return`: target predicted annualized excess return for the robust 60m portfolio.",
            "- `probe_120_deconcentration`: keep the raw 120m ceiling attractive while penalizing HHI concentration.",
            "- `probe_60_120_allocation_contrast`: similar predicted portfolio return, different sleeve allocations.",
            "- `probe_60_vs_e2e_disagreement`: specified in the manifest but deferred until a scenario-ready E2E object is materialized.",
            "",
            "## Recommended First Workflow",
            "1. Select an anchor date and load the actual v3 stacked rows.",
            "2. Fit the locked supervised prediction anchors using only labels observable by the anchor.",
            "3. Instantiate the locked robust allocator for the 60m and 120m portfolio benchmarks.",
            "4. Build the combined regularizer: historical support bounds + VAR(1) plausibility prior + anchor-distance term.",
            "5. Evaluate probe energies and finite-difference gradients at the actual anchor state.",
            "6. Run a very small bounded MALA smoke test before any large scenario batch is launched.",
        ]
    )


def run_scenario_scaffold_audit(
    project_root: Path,
    *,
    anchor_month_end: str = "2024-12-31",
) -> AuditResult:
    """Build manifests, fit the first scenario targets, and run a dry-run smoke test."""
    paths = default_paths(project_root)
    artifacts = load_active_artifacts(paths)
    spec = default_state_spec()

    checks: list[dict[str, str]] = []
    checks.append(
        _component_row(
            "active_paths",
            "PASS",
            f"Using {paths.data_root}, {paths.modeling_root}, and {paths.reports_root}.",
        )
    )

    predictor_specs, portfolio_specs = build_default_candidate_specs(paths)
    candidate_manifest = _build_candidate_manifest(predictor_specs, portfolio_specs)
    checks.append(
        _component_row(
            "candidate_manifest",
            "PASS",
            f"{len(candidate_manifest)} scenario candidates resolved from active v3 artifacts.",
        )
    )

    state_manifest = build_state_manifest(artifacts["feature_manifest"], spec)
    missing_state = [
        name
        for name in spec.base_variables
        if name not in set(artifacts["feature_master_monthly"].columns)
    ]
    if missing_state:
        raise ValueError(f"Feature master is missing required state variables: {missing_state}")
    checks.append(
        _component_row(
            "state_manifest",
            "PASS",
            f"{int(state_manifest['included_in_first_pass'].sum())} state variables included in the first pass.",
        )
    )

    anchor = build_anchor_context(
        artifacts["modeling_panel_hstack"],
        artifacts["feature_master_monthly"],
        month_end=pd.Timestamp(anchor_month_end),
        spec=spec,
    )
    checks.append(
        _component_row(
            "anchor_context",
            "PASS",
            f"Anchor {anchor.month_end.date()} loaded with {len(anchor.anchor_rows)} stacked rows across horizons {anchor.scenario_horizons}.",
        )
    )

    fitted_predictors = {
        predictor_spec.candidate_id: fit_predictor_candidate(
            paths,
            predictor_spec,
            anchor_month_end=anchor.month_end,
        )
        for predictor_spec in predictor_specs
        if predictor_spec.role_in_scenario_stage != "comparator only" or predictor_spec.candidate_id == "predictor_shared_anchor"
    }
    checks.append(
        _component_row(
            "predictor_fit",
            "PASS",
            ", ".join(
                f"{candidate_id}:{fitted.training_rows} rows"
                for candidate_id, fitted in fitted_predictors.items()
            ),
        )
    )

    portfolio_predictor_map = {
        "best_60_predictor": fitted_predictors["predictor_60_anchor"],
        "best_120_predictor": fitted_predictors["predictor_120_anchor"],
    }
    fitted_portfolios = {
        portfolio_spec.candidate_id: fit_portfolio_candidate(
            paths,
            portfolio_spec,
            portfolio_predictor_map[portfolio_spec.candidate_id],
            anchor_month_end=anchor.month_end,
        )
        for portfolio_spec in portfolio_specs
    }
    checks.append(
        _component_row(
            "portfolio_fit",
            "PASS",
            ", ".join(fitted_portfolios.keys()),
        )
    )

    regularizer = build_regularizer(artifacts["feature_master_monthly"], anchor)
    checks.append(
        _component_row(
            "regularizer",
            "PASS",
            f"Bounds + VAR(1) prior built on {len(spec.base_variables)} manipulable state variables.",
        )
    )

    baseline_state = np.asarray(anchor.current_base_state, dtype=float)
    baseline_rows_60 = build_scenario_rows(anchor, spec, baseline_state, predictor=fitted_predictors["predictor_60_anchor"])
    baseline_pred_60 = fitted_portfolios["best_60_predictor"].evaluate(baseline_rows_60)[1]["portfolio_predicted_return"]
    probes = {
        "probe_60_target_return": TargetReturnProbe(
            probe_id="probe_60_target_return",
            portfolio=fitted_portfolios["best_60_predictor"],
            anchor=anchor,
            state_spec=spec,
            target_return=float(baseline_pred_60 + 0.01),
        ),
        "probe_120_deconcentration": ReturnConcentrationProbe(
            probe_id="probe_120_deconcentration",
            portfolio=fitted_portfolios["best_120_predictor"],
            anchor=anchor,
            state_spec=spec,
            concentration_weight=0.50,
        ),
        "probe_60_120_allocation_contrast": SimilarReturnDistinctAllocationProbe(
            probe_id="probe_60_120_allocation_contrast",
            portfolio_a=fitted_portfolios["best_60_predictor"],
            portfolio_b=fitted_portfolios["best_120_predictor"],
            anchor=anchor,
            state_spec=spec,
            diff_reward=0.35,
        ),
    }
    probe_manifest = _build_probe_manifest()
    checks.append(
        _component_row(
            "probe_manifest",
            "PASS",
            f"{int(probe_manifest['implemented_flag'].sum())} implemented probe definitions and 1 deferred E2E comparator probe.",
        )
    )

    probe_rows = []
    gradient_probe_ids = {"probe_60_target_return"}
    for probe_id, probe in probes.items():
        energy = float(probe.energy(baseline_state))
        if probe_id in gradient_probe_ids:
            grad = finite_difference_gradient(probe.energy, baseline_state, step=5e-4)
            grad_norm = float(np.linalg.norm(grad))
        else:
            grad_norm = float("nan")
        probe_rows.append((probe_id, energy, grad_norm))
    checks.append(
        _component_row(
            "probe_evaluation",
            "PASS",
            "; ".join(
                f"{probe_id}: energy={energy:.6f}"
                + (f", grad_norm={grad_norm:.6f}" if np.isfinite(grad_norm) else ", grad_norm=skipped_in_smoke_test")
                for probe_id, energy, grad_norm in probe_rows
            ),
        )
    )

    smoke_probe = probes["probe_60_target_return"]
    smoke_energy = lambda x: smoke_probe.energy(regularizer.project(x)) + regularizer.total_energy(regularizer.project(x))
    smoke_grad = lambda x: finite_difference_gradient(smoke_energy, regularizer.project(x), step=1e-4)
    mala_result = run_bounded_mala(
        start=baseline_state,
        energy_fn=smoke_energy,
        project_fn=regularizer.project,
        gradient_fn=smoke_grad,
        step_size=0.04,
        n_steps=2,
        random_seed=42,
    )
    checks.append(
        _component_row(
            "mala_smoke_test",
            "PASS",
            f"acceptance_rate={mala_result.acceptance_rate:.3f}, final_energy={float(mala_result.energies[-1]):.6f}, steps={len(mala_result.energies) - 1}",
        )
    )

    check_df = pd.DataFrame(checks)
    return AuditResult(
        scenario_candidate_manifest=candidate_manifest,
        scenario_state_manifest=state_manifest,
        scenario_probe_manifest=probe_manifest,
        scenario_scaffold_check=check_df,
        pipeline_audit_report=_pipeline_audit_report(anchor.month_end, check_df),
        adaptation_plan_report=_adaptation_plan_report(anchor.month_end),
    )
