"""Active v3 benchmark predictor and portfolio wrappers for scenario work."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet, Ridge

from xoptpoe_v3_modeling.features import feature_columns_for_set
from xoptpoe_v3_models.data import SLEEVE_ORDER, load_modeling_inputs
from xoptpoe_v3_models.optim_layers import (
    OptimizerConfig,
    RiskConfig,
    RobustOptimizerCache,
    build_sigma_map,
)
from xoptpoe_v3_models.preprocess import FittedPreprocessor, fit_preprocessor

from .io import ScenarioPaths
from .state import ScenarioAnchor, ScenarioStateSpec, apply_state_vector


TARGET_COL = "annualized_excess_forward_return"


@dataclass(frozen=True)
class PredictorCandidateSpec:
    """Locked supervised prediction benchmark specification."""

    candidate_id: str
    source_model_name: str
    model_name: str
    feature_set_name: str
    horizons: tuple[int, ...]
    selected_params: dict[str, object]
    role_in_scenario_stage: str


@dataclass(frozen=True)
class PortfolioCandidateSpec:
    """Locked portfolio benchmark specification."""

    candidate_id: str
    source_model_name: str
    horizon: int
    optimizer_config: OptimizerConfig
    role_in_scenario_stage: str


@dataclass
class FittedPredictorCandidate:
    """Fitted supervised benchmark ready for scenario scoring."""

    spec: PredictorCandidateSpec
    preprocessor: FittedPreprocessor
    model: object
    feature_columns: list[str]
    training_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x, _ = self.preprocessor.transform(frame)
        pred = np.asarray(self.model.predict(x), dtype=float)
        out = frame[["month_end", "sleeve_id", "horizon_months"]].copy()
        out["y_pred"] = pred
        out["candidate_id"] = self.spec.candidate_id
        return out.sort_values(["horizon_months", "sleeve_id"]).reset_index(drop=True)


@dataclass
class FittedPortfolioCandidate:
    """Predictor plus locked allocator configuration."""

    spec: PortfolioCandidateSpec
    predictor: FittedPredictorCandidate
    optimizer_cache: RobustOptimizerCache

    def evaluate(self, scenario_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        pred = self.predictor.predict(
            scenario_rows.loc[scenario_rows["horizon_months"].eq(self.spec.horizon)].copy()
        )
        ordered = pred.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
        if ordered["y_pred"].isna().any():
            raise ValueError(f"Scenario rows for {self.spec.candidate_id} are missing sleeves")
        mu = torch.tensor(ordered["y_pred"].to_numpy(dtype=np.float32), dtype=torch.float32)
        weights = (
            self.optimizer_cache.solve(pd.Timestamp(ordered["month_end"].iloc[0]), mu, self.spec.optimizer_config)
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
        weights = np.clip(weights, 0.0, None)
        weights = weights / weights.sum()
        out = ordered[["month_end", "sleeve_id"]].copy()
        out["weight"] = weights
        out["predicted_return"] = ordered["y_pred"].to_numpy(dtype=float)
        hhi = float(np.square(weights).sum())
        summary = {
            "portfolio_predicted_return": float(np.dot(weights, ordered["y_pred"].to_numpy(dtype=float))),
            "portfolio_hhi": hhi,
            "portfolio_effective_n": float(1.0 / hhi),
            "portfolio_max_weight": float(weights.max()),
        }
        return out, summary


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value)


def _parse_selected_params(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    return dict(ast.literal_eval(text))


def _parse_horizons(value: object) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if isinstance(value, float) and value.is_integer():
        return (int(value),)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple()
        if "+" in text and all(part.strip().isdigit() for part in text.split("+")):
            return tuple(int(part.strip()) for part in text.split("+"))
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (int, np.integer)):
            return (int(parsed),)
        return tuple(int(x) for x in parsed)
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    raise ValueError(f"Unsupported horizon encoding: {value!r}")


def _source_row(prediction_metrics: pd.DataFrame, experiment_name: str) -> pd.Series:
    rows = prediction_metrics.loc[prediction_metrics["experiment_name"].eq(experiment_name)].copy()
    if rows.empty:
        raise ValueError(f"Prediction metrics do not contain benchmark experiment '{experiment_name}'")
    return rows.iloc[0]


def build_default_candidate_specs(paths: ScenarioPaths) -> tuple[list[PredictorCandidateSpec], list[PortfolioCandidateSpec]]:
    """Build the active first-pass scenario candidates from locked v3 artifacts."""
    prediction_metrics = pd.read_csv(paths.prediction_metrics)
    portfolio_metrics = pd.read_csv(paths.portfolio_metrics)

    predictor_specs = []
    for candidate_id, source_model_name, role in (
        ("predictor_60_anchor", "elastic_net__full_firstpass__separate_60", "active benchmark"),
        ("predictor_120_anchor", "ridge__full_firstpass__separate_120", "active benchmark"),
        ("predictor_shared_anchor", "ridge__full_firstpass__shared_60_120", "comparator only"),
    ):
        row = _source_row(prediction_metrics, source_model_name)
        horizons = _parse_horizons(row["horizons"])
        predictor_specs.append(
            PredictorCandidateSpec(
                candidate_id=candidate_id,
                source_model_name=source_model_name,
                model_name=str(row["model_name"]),
                feature_set_name=str(row["feature_set_name"]),
                horizons=horizons,
                selected_params=_parse_selected_params(row["selected_params"]),
                role_in_scenario_stage=role,
            )
        )

    portfolio_specs = []
    for candidate_id, source_model_name, horizon, role in (
        ("best_60_predictor", "elastic_net__full_firstpass__separate_60", 60, "active robust benchmark"),
        ("best_120_predictor", "ridge__full_firstpass__separate_120", 120, "reference ceiling"),
    ):
        row = portfolio_metrics.loc[
            portfolio_metrics["strategy_label"].eq(candidate_id) & portfolio_metrics["split"].eq("test")
        ].iloc[0]
        portfolio_specs.append(
            PortfolioCandidateSpec(
                candidate_id=candidate_id,
                source_model_name=source_model_name,
                horizon=int(horizon),
                optimizer_config=OptimizerConfig(
                    lambda_risk=float(row["selected_lambda_risk"]),
                    kappa=float(row["selected_kappa"]),
                    omega_type=str(row["selected_omega_type"]),
                ),
                role_in_scenario_stage=role,
            )
        )
    return predictor_specs, portfolio_specs


def _eligible_training_rows(frame: pd.DataFrame, anchor_month_end: pd.Timestamp, horizons: tuple[int, ...]) -> pd.DataFrame:
    work = frame.copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    work = work.loc[work["horizon_months"].isin(horizons)].copy()
    if "baseline_trainable_flag" in work.columns:
        work = work.loc[work["baseline_trainable_flag"].eq(1)].copy()
    work = work.loc[work[TARGET_COL].notna()].copy()
    maturity = work.apply(
        lambda row: pd.Timestamp(row["month_end"]) + pd.DateOffset(months=int(row["horizon_months"])),
        axis=1,
    )
    work = work.loc[maturity <= pd.Timestamp(anchor_month_end)].copy()
    if work.empty:
        raise ValueError(f"No eligible training rows remain for horizons={horizons} at anchor={anchor_month_end.date()}")
    return work.sort_values(["month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)


def _instantiate_model(model_name: str, selected_params: dict[str, object]) -> object:
    if model_name == "ridge":
        return Ridge(alpha=float(selected_params["alpha"]), fit_intercept=True, random_state=42)
    if model_name == "elastic_net":
        return ElasticNet(
            alpha=float(selected_params["alpha"]),
            l1_ratio=float(selected_params["l1_ratio"]),
            fit_intercept=True,
            max_iter=20000,
            random_state=42,
        )
    raise ValueError(f"Unsupported first-pass scenario predictor model: {model_name}")


def fit_predictor_candidate(
    paths: ScenarioPaths,
    spec: PredictorCandidateSpec,
    *,
    anchor_month_end: pd.Timestamp,
) -> FittedPredictorCandidate:
    """Fit the locked supervised predictor on all labels observable by the anchor date."""
    inputs = load_modeling_inputs(paths.project_root, feature_set_name=spec.feature_set_name)
    train_df = _eligible_training_rows(
        pd.concat([inputs.train_df, inputs.validation_df, inputs.test_df], ignore_index=True),
        pd.Timestamp(anchor_month_end),
        spec.horizons,
    )
    feature_columns = feature_columns_for_set(inputs.feature_manifest, spec.feature_set_name)
    preprocessor = fit_preprocessor(
        train_df,
        feature_manifest=inputs.feature_manifest,
        feature_columns=feature_columns,
    )
    x_train, _ = preprocessor.transform(train_df)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    model = _instantiate_model(spec.model_name, spec.selected_params)
    model.fit(x_train, y_train)
    return FittedPredictorCandidate(
        spec=spec,
        preprocessor=preprocessor,
        model=model,
        feature_columns=feature_columns,
        training_rows=int(len(train_df)),
    )


def fit_portfolio_candidate(
    paths: ScenarioPaths,
    spec: PortfolioCandidateSpec,
    predictor: FittedPredictorCandidate,
    *,
    anchor_month_end: pd.Timestamp,
) -> FittedPortfolioCandidate:
    """Attach the locked v3 robust allocator to a fitted predictor."""
    inputs = load_modeling_inputs(paths.project_root, feature_set_name=predictor.spec.feature_set_name)
    sigma_map = build_sigma_map(
        [_as_timestamp(anchor_month_end)],
        excess_history=inputs.monthly_excess_history,
        risk_config=RiskConfig(),
    )
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)
    return FittedPortfolioCandidate(
        spec=spec,
        predictor=predictor,
        optimizer_cache=optimizer_cache,
    )


def build_scenario_rows(
    anchor: ScenarioAnchor,
    spec: ScenarioStateSpec,
    state_vector: np.ndarray,
    *,
    predictor: FittedPredictorCandidate | None = None,
) -> pd.DataFrame:
    """Create scenario rows and, when available, trim them to the predictor horizon set."""
    out = apply_state_vector(anchor, spec, state_vector)
    if predictor is not None:
        out = out.loc[out["horizon_months"].isin(predictor.spec.horizons)].copy()
    missing = [col for col in (predictor.feature_columns if predictor else []) if col not in out.columns]
    if missing:
        raise ValueError(f"Scenario rows are missing predictor columns: {missing[:10]}")
    return out.sort_values(["horizon_months", "sleeve_id"]).reset_index(drop=True)
