"""Curated conference-grade scenario questions for the active v3 benchmark stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .probes import DualPortfolioObjectiveProbe, PortfolioObjectiveProbe
from .regimes import RegimeClassifier


RISK_ON_SLEEVES = ("EQ_US", "EQ_EZ", "EQ_JP", "EQ_CN", "EQ_EM", "RE_US")
DEFENSIVE_SLEEVES = ("FI_UST", "FI_IG", "ALT_GLD")


@dataclass(frozen=True)
class ScenarioQuestionSpec:
    """Machine-readable conference question."""

    question_id: str
    question_family: str
    short_label: str
    candidate_name: str
    candidate_2: str
    horizon: str
    question_text: str
    why_it_matters: str
    question_type: str
    primary_metric_label: str
    response_direction: str
    steps: int
    step_size: float
    notes: str


@dataclass(frozen=True)
class ScenarioQuestionRunSpec:
    """Runtime question specification with attached probe and selection rule."""

    spec: ScenarioQuestionSpec
    probe: object
    selection_metric: Callable[[dict[str, object]], float]


def _weight(weights_df, sleeve_id: str) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].eq(sleeve_id), "weight"].iloc[0])


def _pred(weights_df, sleeve_id: str) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].eq(sleeve_id), "predicted_return"].iloc[0])


def _share(weights_df, sleeve_ids: tuple[str, ...]) -> float:
    return float(weights_df.loc[weights_df["sleeve_id"].isin(list(sleeve_ids)), "weight"].sum())


def _regime_distance(scores: dict[str, float], *, growth: float, inflation: float, stress: float, rates: float) -> float:
    return float(
        (scores["growth_score"] - growth) ** 2
        + (scores["inflation_score"] - inflation) ** 2
        + (scores["stress_score"] - stress) ** 2
        + (scores["rates_score"] - rates) ** 2
    )


def build_question_set(context, baseline: dict[str, object], classifier: RegimeClassifier) -> list[ScenarioQuestionRunSpec]:
    """Build the compact conference-grade scenario question menu."""
    baseline_60 = baseline["best_60_predictor"]
    baseline_120 = baseline["best_120_predictor"]
    deconc_weight = max(0.10, 0.75 * abs(baseline_120["portfolio_predicted_return"]) / max(baseline_120["portfolio_hhi"], 1e-6))
    deconc_return_floor = 0.90 * baseline_120["portfolio_predicted_return"]

    best60 = context.portfolios["best_60_predictor"]
    best120 = context.portfolios["best_120_predictor"]

    questions: list[ScenarioQuestionRunSpec] = []

    def add(spec: ScenarioQuestionSpec, probe: object, selection_metric: Callable[[dict[str, object]], float]) -> None:
        questions.append(ScenarioQuestionRunSpec(spec=spec, probe=probe, selection_metric=selection_metric))

    add(
        ScenarioQuestionSpec(
            question_id="q_robust_double_digit",
            question_family="outcome_target",
            short_label="Robust double-digit outlook",
            candidate_name="best_60_predictor",
            candidate_2="",
            horizon="60m",
            question_text="What plausible regime keeps the robust 5Y benchmark in double-digit annualized excess-return territory without materially higher concentration?",
            why_it_matters="This is the cleanest benchmark-governance question for the carry-forward portfolio.",
            question_type="single_portfolio",
            primary_metric_label="portfolio_predicted_return",
            response_direction="max",
            steps=2,
            step_size=0.075,
            notes="Maximize robust 60m return with a live concentration penalty.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_robust_double_digit",
            portfolio=best60,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -s["portfolio_predicted_return"] + 0.25 * s["portfolio_hhi"],
        ),
        lambda e: float(e["portfolio_predicted_return"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_raw_ceiling_upside",
            question_family="outcome_target",
            short_label="Raw ceiling upside",
            candidate_name="best_120_predictor",
            candidate_2="",
            horizon="120m",
            question_text="What plausible regime supports the raw 10Y ceiling benchmark when expected return is pushed higher?",
            why_it_matters="This is the most direct way to ask what powers the concentrated raw ceiling.",
            question_type="single_portfolio",
            primary_metric_label="portfolio_predicted_return",
            response_direction="max",
            steps=2,
            step_size=0.075,
            notes="Maximize 120m return with only a light concentration penalty.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_raw_ceiling_upside",
            portfolio=best120,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -s["portfolio_predicted_return"] + 0.10 * s["portfolio_hhi"],
        ),
        lambda e: float(e["portfolio_predicted_return"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_raw_deconcentration",
            question_family="benchmark_disagreement",
            short_label="Raw ceiling deconcentration",
            candidate_name="best_120_predictor",
            candidate_2="",
            horizon="120m",
            question_text="What plausible regime lets the raw 10Y ceiling deconcentrate without giving up much expected return?",
            why_it_matters="Senior finance audiences will ask whether the raw ceiling can be made less one-sided.",
            question_type="single_portfolio",
            primary_metric_label="portfolio_hhi",
            response_direction="min",
            steps=2,
            step_size=0.075,
            notes=f"Positive concentration penalty with a soft return floor; concentration_weight={deconc_weight:.4f}, return_floor={deconc_return_floor:.4f}.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_raw_deconcentration",
            portfolio=best120,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x, cw=deconc_weight, floor=deconc_return_floor: cw * s["portfolio_hhi"] + 1.25 * max(0.0, floor - s["portfolio_predicted_return"]) - 0.15 * s["portfolio_predicted_return"],
        ),
        lambda e: float(e["portfolio_hhi"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_gold_tilt",
            question_family="allocation_tilt",
            short_label="Gold-supporting regime",
            candidate_name="best_60_predictor",
            candidate_2="",
            horizon="60m",
            question_text="What plausible regime makes the robust benchmark allocate more meaningfully to gold?",
            why_it_matters="Gold is a recognizable macro sleeve and gives the audience an intuitive allocation question.",
            question_type="single_portfolio",
            primary_metric_label="weight_ALT_GLD",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Increase gold weight while keeping portfolio quality positive.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_gold_tilt",
            portfolio=best60,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -1.2 * _weight(w, "ALT_GLD") - 0.20 * _pred(w, "ALT_GLD") - 0.10 * s["portfolio_predicted_return"] + 0.08 * s["portfolio_hhi"],
        ),
        lambda e: float(e["weight_ALT_GLD"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_us_equity_tilt",
            question_family="allocation_tilt",
            short_label="US equity tilt",
            candidate_name="best_60_predictor",
            candidate_2="",
            horizon="60m",
            question_text="What plausible regime makes US equities the clearest overweight inside the robust benchmark?",
            why_it_matters="US equities remain the core strategic sleeve, so this is easy to explain on stage.",
            question_type="single_portfolio",
            primary_metric_label="weight_EQ_US",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Tilt the robust benchmark toward EQ_US without ignoring portfolio quality.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_us_equity_tilt",
            portfolio=best60,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -1.0 * _weight(w, "EQ_US") - 0.15 * _pred(w, "EQ_US") - 0.10 * s["portfolio_predicted_return"] + 0.12 * s["portfolio_hhi"],
        ),
        lambda e: float(e["weight_EQ_US"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_em_tilt",
            question_family="allocation_tilt",
            short_label="EM-supporting regime",
            candidate_name="best_60_predictor",
            candidate_2="",
            horizon="60m",
            question_text="What plausible regime makes EM equities more attractive inside the robust benchmark?",
            why_it_matters="This is a clean test of whether the framework can find a risk-on allocation outside the default US core.",
            question_type="single_portfolio",
            primary_metric_label="weight_EQ_EM",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Increase EM weight while retaining positive portfolio return.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_em_tilt",
            portfolio=best60,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -1.2 * _weight(w, "EQ_EM") - 0.20 * _pred(w, "EQ_EM") - 0.05 * s["portfolio_predicted_return"] + 0.08 * s["portfolio_hhi"],
        ),
        lambda e: float(e["weight_EQ_EM"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_china_materiality",
            question_family="allocation_tilt",
            short_label="China materiality",
            candidate_name="best_60_predictor",
            candidate_2="",
            horizon="60m",
            question_text="What plausible regime makes China become a more meaningful sleeve rather than staying marginal?",
            why_it_matters="China is in the system, so this answers the natural follow-up without letting China dominate the talk.",
            question_type="single_portfolio",
            primary_metric_label="weight_EQ_CN",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Increase EQ_CN weight and predicted return with only a mild concentration penalty.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_china_materiality",
            portfolio=best60,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x: -1.4 * _weight(w, "EQ_CN") - 0.25 * _pred(w, "EQ_CN") - 0.05 * s["portfolio_predicted_return"] + 0.05 * s["portfolio_hhi"],
        ),
        lambda e: float(e["weight_EQ_CN"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_robust_raw_disagreement",
            question_family="benchmark_disagreement",
            short_label="Robust vs raw disagreement",
            candidate_name="best_60_predictor",
            candidate_2="best_120_predictor",
            horizon="both",
            question_text="What plausible regime makes the robust 5Y benchmark and the raw 10Y ceiling disagree most in allocation behavior?",
            why_it_matters="This is the cleanest side-by-side question for showing why the raw and robust objects are not the same thing.",
            question_type="dual_portfolio",
            primary_metric_label="allocation_gap_l1",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Keep predicted returns similar while increasing allocation distance.",
        ),
        DualPortfolioObjectiveProbe(
            probe_id="q_robust_raw_disagreement",
            portfolio_a=best60,
            portfolio_b=best120,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda wa, sa, wb, sb, x: (sa["portfolio_predicted_return"] - sb["portfolio_predicted_return"]) ** 2 - 0.40 * float(np.abs(wa["weight"].to_numpy(dtype=float) - wb["weight"].to_numpy(dtype=float)).sum()),
        ),
        lambda e: float(e["allocation_gap_l1"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_soft_landing_outlook",
            question_family="strategic_outlook",
            short_label="Soft landing outlook",
            candidate_name="best_60_predictor",
            candidate_2="",
            horizon="60m",
            question_text="What plausible regime gives the robust benchmark a soft-landing-style strategic outlook?",
            why_it_matters="This turns the scenario layer into language that senior allocators immediately recognize.",
            question_type="single_portfolio",
            primary_metric_label="soft_landing_alignment",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Target high growth, low stress, moderate inflation, neutral-to-slightly-tight rates.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_soft_landing_outlook",
            portfolio=best60,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x, clf=classifier: _regime_distance(clf.dimension_scores(x), growth=0.8, inflation=0.0, stress=-0.8, rates=0.2) - 0.20 * s["portfolio_predicted_return"] + 0.06 * s["portfolio_hhi"] + 0.05 * max(0.0, _share(w, DEFENSIVE_SLEEVES) - _share(w, RISK_ON_SLEEVES)),
        ),
        lambda e: float(e["soft_landing_alignment"]),
    )

    add(
        ScenarioQuestionSpec(
            question_id="q_higher_for_longer_defensive",
            question_family="strategic_outlook",
            short_label="Higher-for-longer defensive",
            candidate_name="best_60_predictor",
            candidate_2="",
            horizon="60m",
            question_text="What plausible regime gives the robust benchmark a higher-for-longer defensive strategic outlook?",
            why_it_matters="This is the clearest strategic contrast to soft landing and is easy to explain on stage.",
            question_type="single_portfolio",
            primary_metric_label="higher_for_longer_alignment",
            response_direction="max",
            steps=2,
            step_size=0.08,
            notes="Target low growth, high inflation, tighter rates, and more defensive allocation share.",
        ),
        PortfolioObjectiveProbe(
            probe_id="q_higher_for_longer_defensive",
            portfolio=best60,
            anchor=context.anchor,
            state_spec=context.state_spec,
            objective_fn=lambda w, s, x, clf=classifier: _regime_distance(clf.dimension_scores(x), growth=-0.4, inflation=0.8, stress=0.5, rates=0.9) - 0.12 * _share(w, DEFENSIVE_SLEEVES) - 0.08 * s["portfolio_predicted_return"] + 0.04 * max(0.0, _share(w, RISK_ON_SLEEVES) - _share(w, DEFENSIVE_SLEEVES)),
        ),
        lambda e: float(e["higher_for_longer_alignment"]),
    )

    return questions
