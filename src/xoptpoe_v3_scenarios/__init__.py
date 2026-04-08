"""Scenario-generation scaffold for the active v3 long-horizon China branch."""

from .audit import run_scenario_scaffold_audit
from .io import ScenarioPaths, default_paths
from .pipelines import (
    FittedPortfolioCandidate,
    FittedPredictorCandidate,
    fit_portfolio_candidate,
    fit_predictor_candidate,
)
from .probes import build_default_probe_specs
from .state import (
    BASE_STATE_VARIABLES,
    DERIVED_STATE_VARIABLES,
    ScenarioAnchor,
    ScenarioStateSpec,
    build_anchor_context,
    build_state_manifest,
    default_state_spec,
)

__all__ = [
    "BASE_STATE_VARIABLES",
    "DERIVED_STATE_VARIABLES",
    "FittedPortfolioCandidate",
    "FittedPredictorCandidate",
    "ScenarioAnchor",
    "ScenarioPaths",
    "ScenarioStateSpec",
    "build_anchor_context",
    "build_default_probe_specs",
    "build_state_manifest",
    "default_paths",
    "default_state_spec",
    "fit_portfolio_candidate",
    "fit_predictor_candidate",
    "run_scenario_scaffold_audit",
]
