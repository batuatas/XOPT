"""Explicit first-pass scenario probe definitions for the active v3 stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipelines import FittedPortfolioCandidate, FittedPredictorCandidate
from .state import ScenarioAnchor, ScenarioStateSpec


@dataclass(frozen=True)
class ProbeSpec:
    """Machine-readable probe configuration."""

    probe_id: str
    probe_family: str
    candidate_1: str
    candidate_2: str
    horizon: str
    objective_description: str
    implemented_flag: int
    notes: str


class ScenarioProbe:
    """Base probe interface for finite-difference scenario exploration."""

    probe_id: str

    def energy(self, state_vector: np.ndarray) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class PortfolioObjectiveProbe(ScenarioProbe):
    """Generic single-portfolio probe driven by a custom objective."""

    def __init__(
        self,
        *,
        probe_id: str,
        portfolio: FittedPortfolioCandidate,
        anchor: ScenarioAnchor,
        state_spec: ScenarioStateSpec,
        objective_fn,
    ) -> None:
        self.probe_id = probe_id
        self.portfolio = portfolio
        self.anchor = anchor
        self.state_spec = state_spec
        self.objective_fn = objective_fn

    def energy(self, state_vector: np.ndarray) -> float:
        from .pipelines import build_scenario_rows

        frame = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.portfolio.predictor)
        weights, summary = self.portfolio.evaluate(frame)
        return float(self.objective_fn(weights, summary, np.asarray(state_vector, dtype=float)))


class DualPortfolioObjectiveProbe(ScenarioProbe):
    """Generic two-portfolio probe driven by a custom objective."""

    def __init__(
        self,
        *,
        probe_id: str,
        portfolio_a: FittedPortfolioCandidate,
        portfolio_b: FittedPortfolioCandidate,
        anchor: ScenarioAnchor,
        state_spec: ScenarioStateSpec,
        objective_fn,
    ) -> None:
        self.probe_id = probe_id
        self.portfolio_a = portfolio_a
        self.portfolio_b = portfolio_b
        self.anchor = anchor
        self.state_spec = state_spec
        self.objective_fn = objective_fn

    def energy(self, state_vector: np.ndarray) -> float:
        from .pipelines import build_scenario_rows

        frame_a = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.portfolio_a.predictor)
        frame_b = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.portfolio_b.predictor)
        weights_a, summary_a = self.portfolio_a.evaluate(frame_a)
        weights_b, summary_b = self.portfolio_b.evaluate(frame_b)
        return float(
            self.objective_fn(
                weights_a,
                summary_a,
                weights_b,
                summary_b,
                np.asarray(state_vector, dtype=float),
            )
        )


class TargetReturnProbe(ScenarioProbe):
    """Low energy when a portfolio hits a user-chosen target predicted return."""

    def __init__(
        self,
        *,
        probe_id: str,
        portfolio: FittedPortfolioCandidate,
        anchor: ScenarioAnchor,
        state_spec: ScenarioStateSpec,
        target_return: float,
    ) -> None:
        self.probe_id = probe_id
        self.portfolio = portfolio
        self.anchor = anchor
        self.state_spec = state_spec
        self.target_return = float(target_return)

    def energy(self, state_vector: np.ndarray) -> float:
        from .pipelines import build_scenario_rows

        frame = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.portfolio.predictor)
        _, summary = self.portfolio.evaluate(frame)
        return float((summary["portfolio_predicted_return"] - self.target_return) ** 2)


class ReturnConcentrationProbe(ScenarioProbe):
    """Trade off predicted portfolio return against concentration."""

    def __init__(
        self,
        *,
        probe_id: str,
        portfolio: FittedPortfolioCandidate,
        anchor: ScenarioAnchor,
        state_spec: ScenarioStateSpec,
        concentration_weight: float = 0.50,
    ) -> None:
        self.probe_id = probe_id
        self.portfolio = portfolio
        self.anchor = anchor
        self.state_spec = state_spec
        self.concentration_weight = float(concentration_weight)

    def energy(self, state_vector: np.ndarray) -> float:
        from .pipelines import build_scenario_rows

        frame = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.portfolio.predictor)
        _, summary = self.portfolio.evaluate(frame)
        return float(-summary["portfolio_predicted_return"] + self.concentration_weight * summary["portfolio_hhi"])


class SimilarReturnDistinctAllocationProbe(ScenarioProbe):
    """Low energy when two portfolios have similar return but visibly different weights."""

    def __init__(
        self,
        *,
        probe_id: str,
        portfolio_a: FittedPortfolioCandidate,
        portfolio_b: FittedPortfolioCandidate,
        anchor: ScenarioAnchor,
        state_spec: ScenarioStateSpec,
        diff_reward: float = 0.35,
    ) -> None:
        self.probe_id = probe_id
        self.portfolio_a = portfolio_a
        self.portfolio_b = portfolio_b
        self.anchor = anchor
        self.state_spec = state_spec
        self.diff_reward = float(diff_reward)

    def energy(self, state_vector: np.ndarray) -> float:
        from .pipelines import build_scenario_rows

        frame_a = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.portfolio_a.predictor)
        frame_b = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.portfolio_b.predictor)
        weights_a, summary_a = self.portfolio_a.evaluate(frame_a)
        weights_b, summary_b = self.portfolio_b.evaluate(frame_b)
        merged = weights_a.merge(weights_b, on=["month_end", "sleeve_id"], suffixes=("_a", "_b"), validate="1:1")
        weight_gap = float(np.abs(merged["weight_a"].to_numpy(dtype=float) - merged["weight_b"].to_numpy(dtype=float)).sum())
        return_gap = float(summary_a["portfolio_predicted_return"] - summary_b["portfolio_predicted_return"])
        return float(return_gap**2 - self.diff_reward * weight_gap)


class PredictionDisagreementProbe(ScenarioProbe):
    """Low energy when two predictors imply very different cross-sectional views."""

    def __init__(
        self,
        *,
        probe_id: str,
        predictor_a: FittedPredictorCandidate,
        predictor_b: FittedPredictorCandidate,
        anchor: ScenarioAnchor,
        state_spec: ScenarioStateSpec,
        disagreement_weight: float = 1.0,
    ) -> None:
        self.probe_id = probe_id
        self.predictor_a = predictor_a
        self.predictor_b = predictor_b
        self.anchor = anchor
        self.state_spec = state_spec
        self.disagreement_weight = float(disagreement_weight)

    def energy(self, state_vector: np.ndarray) -> float:
        from .pipelines import build_scenario_rows

        frame_a = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.predictor_a)
        frame_b = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.predictor_b)
        pred_a = self.predictor_a.predict(frame_a)[["sleeve_id", "y_pred"]].rename(columns={"y_pred": "y_pred_a"})
        pred_b = self.predictor_b.predict(frame_b)[["sleeve_id", "y_pred"]].rename(columns={"y_pred": "y_pred_b"})
        merged = pred_a.merge(pred_b, on="sleeve_id", how="inner", validate="1:1")
        spread = float(np.mean(np.abs(merged["y_pred_a"].to_numpy(dtype=float) - merged["y_pred_b"].to_numpy(dtype=float))))
        return float(-self.disagreement_weight * spread)


class ChinaRoleProbe(ScenarioProbe):
    """Low energy when EQ_CN becomes more meaningful without ignoring portfolio quality."""

    def __init__(
        self,
        *,
        probe_id: str,
        portfolio: FittedPortfolioCandidate,
        anchor: ScenarioAnchor,
        state_spec: ScenarioStateSpec,
        eq_cn_weight_weight: float = 1.0,
        return_weight: float = 0.25,
        concentration_penalty: float = 0.10,
    ) -> None:
        self.probe_id = probe_id
        self.portfolio = portfolio
        self.anchor = anchor
        self.state_spec = state_spec
        self.eq_cn_weight_weight = float(eq_cn_weight_weight)
        self.return_weight = float(return_weight)
        self.concentration_penalty = float(concentration_penalty)

    def energy(self, state_vector: np.ndarray) -> float:
        from .pipelines import build_scenario_rows

        frame = build_scenario_rows(self.anchor, self.state_spec, state_vector, predictor=self.portfolio.predictor)
        weights, summary = self.portfolio.evaluate(frame)
        eq_cn_weight = float(weights.loc[weights["sleeve_id"].eq("EQ_CN"), "weight"].iloc[0])
        return float(
            -self.eq_cn_weight_weight * eq_cn_weight
            - self.return_weight * summary["portfolio_predicted_return"]
            + self.concentration_penalty * summary["portfolio_hhi"]
        )


def build_default_probe_specs() -> list[ProbeSpec]:
    """Return the locked first-pass v3 probe menu."""
    return [
        ProbeSpec(
            probe_id="probe_60_target_return",
            probe_family="single_pipeline_target",
            candidate_1="best_60_predictor",
            candidate_2="",
            horizon="60m",
            objective_description="States that make the robust 60m portfolio hit a target predicted annualized excess return.",
            implemented_flag=1,
            notes="Audit dry-run uses target = baseline predicted portfolio return + 1 percentage point.",
        ),
        ProbeSpec(
            probe_id="probe_120_deconcentration",
            probe_family="single_pipeline_tradeoff",
            candidate_1="best_120_predictor",
            candidate_2="",
            horizon="120m",
            objective_description="States that keep the raw 120m portfolio attractive while penalizing concentration.",
            implemented_flag=1,
            notes="First-pass energy is predicted return minus an HHI penalty.",
        ),
        ProbeSpec(
            probe_id="probe_60_120_allocation_contrast",
            probe_family="cross_pipeline_contrast",
            candidate_1="best_60_predictor",
            candidate_2="best_120_predictor",
            horizon="both",
            objective_description="States where 60m and 120m portfolios have similar predicted return but distinct allocations.",
            implemented_flag=1,
            notes="Direct adaptation of the old similar-return / distinct-risk contrast idea to long-horizon SAA weights.",
        ),
        ProbeSpec(
            probe_id="probe_60_vs_e2e_disagreement",
            probe_family="cross_pipeline_contrast",
            candidate_1="best_60_predictor",
            candidate_2="e2e_nn_signal",
            horizon="both",
            objective_description="States where the robust supervised benchmark and E2E comparator disagree most.",
            implemented_flag=0,
            notes="Deferred because the active v3 E2E reports do not ship a clean persisted scenario-ready model object.",
        ),
    ]
