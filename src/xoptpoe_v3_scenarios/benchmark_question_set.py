"""Conference-grade question set for the locked robust 5Y benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .benchmark_regimes import HybridRegimeClassifier
from .probes import PortfolioObjectiveProbe


@dataclass(frozen=True)
class RobustBenchmarkQuestionSpec:
    """Audience-first scenario question specification."""

    question_id: str
    question_group: str
    question_family: str
    audience_question: str
    short_label: str
    recommended_for_conference: int
    target_value: float | None
    primary_metric_label: str
    response_direction: str
    steps: int
    step_size: float
    notes: str


@dataclass(frozen=True)
class RobustBenchmarkQuestionRunSpec:
    """Runtime question spec with attached probe and metric selector."""

    spec: RobustBenchmarkQuestionSpec
    probe: object
    selection_metric: Callable[[dict[str, object]], float]


def _weight(weights_df, sleeve_id: str) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].eq(sleeve_id), "weight"].iloc[0])


def _pred(weights_df, sleeve_id: str) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].eq(sleeve_id), "predicted_return"].iloc[0])


def _share(weights_df, sleeve_ids: tuple[str, ...]) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].isin(list(sleeve_ids)), "weight"].sum())


def _regime_distance(
    scores: dict[str, float],
    *,
    growth: float,
    inflation: float,
    market_stress: float,
    rates: float,
) -> float:
    return float(
        (scores["growth_score"] - growth) ** 2
        + (scores["inflation_score"] - inflation) ** 2
        + (scores["market_stress_score"] - market_stress) ** 2
        + (scores["rates_score"] - rates) ** 2
    )


def build_robust_benchmark_question_set(context, baseline: dict[str, object], classifier: HybridRegimeClassifier) -> list[RobustBenchmarkQuestionRunSpec]:
    """Build the final one-benchmark question menu."""
    portfolio = context.portfolio
    baseline_summary = baseline["evaluation"]
    baseline_hhi = float(baseline_summary["portfolio_hhi"])
    baseline_max_weight = float(baseline_summary["portfolio_max_weight"])

    questions: list[RobustBenchmarkQuestionRunSpec] = []

    def add(spec: RobustBenchmarkQuestionSpec, probe: object, selection_metric: Callable[[dict[str, object]], float]) -> None:
        questions.append(RobustBenchmarkQuestionRunSpec(spec=spec, probe=probe, selection_metric=selection_metric))

    def return_target_objective(target: float, concentration_weight: float = 0.15) -> Callable:
        return lambda w, s, x, t=target, cw=concentration_weight: 40.0 * (s["portfolio_predicted_return"] - t) ** 2 + cw * s["portfolio_hhi"]

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_target_10",
            question_group="g_strong_long_run_return",
            question_family="return-target question",
            audience_question="What regime would justify a 10% annualized long-run return assumption for the locked benchmark?",
            short_label="10% long-run return",
            recommended_for_conference=0,
            target_value=0.10,
            primary_metric_label="return_gap_to_target",
            response_direction="min",
            steps=2,
            step_size=0.09,
            notes="Target-return question anchored on a strong double-digit strategic outlook.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_target_10",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=return_target_objective(0.10, concentration_weight=0.18),
        ),
        lambda e: abs(float(e["portfolio_predicted_return"]) - 0.10),
    )

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_house_view_7",
            question_group="g_house_view_ladder",
            question_family="return-target question",
            audience_question="What regime would justify a 7% annualized strategic return assumption?",
            short_label="7% house view",
            recommended_for_conference=0,
            target_value=0.07,
            primary_metric_label="return_gap_to_target",
            response_direction="min",
            steps=1,
            step_size=0.09,
            notes="House-view ladder question: optimistic but still institutional.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_house_view_7",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=return_target_objective(0.07, concentration_weight=0.12),
        ),
        lambda e: abs(float(e["portfolio_predicted_return"]) - 0.07),
    )

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_house_view_6",
            question_group="g_house_view_ladder",
            question_family="return-target question",
            audience_question="What regime would justify a 6% annualized strategic return assumption?",
            short_label="6% house view",
            recommended_for_conference=0,
            target_value=0.06,
            primary_metric_label="return_gap_to_target",
            response_direction="min",
            steps=1,
            step_size=0.09,
            notes="House-view ladder question: more conservative strategic assumption.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_house_view_6",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=return_target_objective(0.06, concentration_weight=0.10),
        ),
        lambda e: abs(float(e["portfolio_predicted_return"]) - 0.06),
    )

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_return_with_breadth",
            question_group="g_return_with_breadth",
            question_family="risk/governance question",
            audience_question="What regime improves expected long-run return without making the benchmark much more concentrated?",
            short_label="Return with breadth",
            recommended_for_conference=1,
            target_value=None,
            primary_metric_label="return_breadth_score",
            response_direction="max",
            steps=2,
            step_size=0.085,
            notes="Risk/governance question for the single locked benchmark.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_return_with_breadth",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x, base_hhi=baseline_hhi, base_max=baseline_max_weight: -s["portfolio_predicted_return"] + 0.18 * s["portfolio_hhi"] + 0.12 * max(0.0, s["portfolio_max_weight"] - base_max) + 0.08 * max(0.0, s["portfolio_hhi"] - base_hhi),
        ),
        lambda e: float(e["portfolio_predicted_return"] - 0.25 * e["portfolio_hhi"]),
    )

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_gold_tilt",
            question_group="g_gold_tilt",
            question_family="allocation-tilt question",
            audience_question="What regime would make gold more attractive inside the locked benchmark?",
            short_label="Gold more attractive",
            recommended_for_conference=1,
            target_value=None,
            primary_metric_label="weight_ALT_GLD",
            response_direction="max",
            steps=1,
            step_size=0.085,
            notes="Allocation-tilt question around the benchmark's defensive real-asset sleeve.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_gold_tilt",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -1.5 * _weight(w, "ALT_GLD") - 0.25 * _pred(w, "ALT_GLD") - 0.08 * s["portfolio_predicted_return"] + 0.08 * s["portfolio_hhi"],
        ),
        lambda e: float(e["weight_ALT_GLD"]),
    )

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_us_equity_tilt",
            question_group="g_us_equity_tilt",
            question_family="allocation-tilt question",
            audience_question="What regime would make US equities more attractive inside the locked benchmark?",
            short_label="US equities more attractive",
            recommended_for_conference=1,
            target_value=None,
            primary_metric_label="weight_EQ_US",
            response_direction="max",
            steps=1,
            step_size=0.085,
            notes="Allocation-tilt question around the core US sleeve.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_us_equity_tilt",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -1.3 * _weight(w, "EQ_US") - 0.20 * _pred(w, "EQ_US") - 0.08 * s["portfolio_predicted_return"] + 0.10 * s["portfolio_hhi"],
        ),
        lambda e: float(e["weight_EQ_US"]),
    )

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_em_equity_tilt",
            question_group="g_em_equity_tilt",
            question_family="allocation-tilt question",
            audience_question="What regime would make EM equities more attractive inside the locked benchmark?",
            short_label="EM equities more attractive",
            recommended_for_conference=0,
            target_value=None,
            primary_metric_label="weight_EQ_EM",
            response_direction="max",
            steps=1,
            step_size=0.085,
            notes="Allocation-tilt question for non-US cyclical risk appetite.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_em_equity_tilt",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -1.5 * _weight(w, "EQ_EM") - 0.25 * _pred(w, "EQ_EM") - 0.06 * s["portfolio_predicted_return"] + 0.08 * s["portfolio_hhi"],
        ),
        lambda e: float(e["weight_EQ_EM"]),
    )

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_soft_landing",
            question_group="g_regime_narratives",
            question_family="regime-narrative question",
            audience_question="What does a soft-landing regime look like for the locked benchmark?",
            short_label="Soft landing",
            recommended_for_conference=1,
            target_value=None,
            primary_metric_label="soft_landing_alignment",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Regime-narrative question combining internal dimensions with the anchor-level conditions backdrop.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_soft_landing",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x, clf=classifier, ad=context.anchor.month_end: _regime_distance(
                clf.internal_scores(x),
                growth=0.7,
                inflation=-0.1,
                market_stress=-0.5,
                rates=0.1,
            ) + (0.40 if clf.external_context(ad)["nfci_bucket"] == "tight" else 0.0) + (0.50 if clf.external_context(ad)["recession_overlay"] == "recession" else 0.0) - 0.10 * s["portfolio_predicted_return"] + 0.05 * s["portfolio_hhi"],
        ),
        lambda e: float(e["soft_landing_alignment"]),
    )

    add(
        RobustBenchmarkQuestionSpec(
            question_id="q_higher_for_longer",
            question_group="g_regime_narratives",
            question_family="regime-narrative question",
            audience_question="What does a higher-for-longer regime look like for the locked benchmark?",
            short_label="Higher-for-longer",
            recommended_for_conference=1,
            target_value=None,
            primary_metric_label="higher_for_longer_alignment",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Regime-narrative question for a tight-rates, sticky-inflation backdrop.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_higher_for_longer",
            portfolio=portfolio,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x, clf=classifier, ad=context.anchor.month_end: _regime_distance(
                clf.internal_scores(x),
                growth=-0.1,
                inflation=0.8,
                market_stress=0.2,
                rates=0.9,
            ) - 0.05 * _share(w, ("FI_UST", "FI_IG", "ALT_GLD")) - 0.04 * s["portfolio_predicted_return"] + (0.0 if clf.external_context(ad)["nfci_bucket"] != "loose" else 0.30),
        ),
        lambda e: float(e["higher_for_longer_alignment"]),
    )

    return questions
