"""Baseline models for first-pass XOPTPOE v1 return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xoptpoe_modeling.eda import TARGET_COL, infer_feature_columns


@dataclass
class DesignMatrices:
    """Container for split-aligned feature matrices."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    numeric_feature_columns: list[str]
    model_feature_columns: list[str]


@dataclass
class BaselineModelResult:
    """Predictions and metadata for one fitted baseline."""

    model_name: str
    family: str
    hyperparams: dict[str, Any]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def prepare_design_matrices(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> DesignMatrices:
    """Build leak-safe matrices with train-only imputation and sleeve dummies."""
    for name, frame in (
        ("train_df", train_df),
        ("validation_df", validation_df),
        ("test_df", test_df),
    ):
        if TARGET_COL not in frame.columns:
            raise ValueError(f"{name} missing target column: {TARGET_COL}")

    numeric_cols = feature_columns if feature_columns is not None else infer_feature_columns(train_df)
    missing_features = sorted(set(numeric_cols) - set(train_df.columns))
    if missing_features:
        raise ValueError(f"Requested feature columns missing in train split: {missing_features}")

    # Keep only features that exist in all splits.
    common_cols = [c for c in numeric_cols if c in validation_df.columns and c in test_df.columns]
    if not common_cols:
        raise ValueError("No common numeric feature columns found across splits")

    X_train_num = train_df[common_cols].copy()
    X_val_num = validation_df[common_cols].copy()
    X_test_num = test_df[common_cols].copy()

    # Train-only median imputation.
    medians = X_train_num.median(numeric_only=True)
    X_train_num = X_train_num.fillna(medians)
    X_val_num = X_val_num.fillna(medians)
    X_test_num = X_test_num.fillna(medians)

    # Sleeve identity can materially improve pooled baselines.
    sleeve_train = pd.get_dummies(train_df["sleeve_id"], prefix="sleeve", dtype=float)
    sleeve_val = pd.get_dummies(validation_df["sleeve_id"], prefix="sleeve", dtype=float)
    sleeve_test = pd.get_dummies(test_df["sleeve_id"], prefix="sleeve", dtype=float)
    sleeve_cols = sorted(sleeve_train.columns.tolist())
    sleeve_val = sleeve_val.reindex(columns=sleeve_cols, fill_value=0.0)
    sleeve_test = sleeve_test.reindex(columns=sleeve_cols, fill_value=0.0)

    X_train = pd.concat([X_train_num, sleeve_train], axis=1)
    X_val = pd.concat([X_val_num, sleeve_val], axis=1)
    X_test = pd.concat([X_test_num, sleeve_test], axis=1)

    y_train = train_df[TARGET_COL].astype(float).copy()
    y_val = validation_df[TARGET_COL].astype(float).copy()
    y_test = test_df[TARGET_COL].astype(float).copy()

    return DesignMatrices(
        train_frame=train_df.copy(),
        validation_frame=validation_df.copy(),
        test_frame=test_df.copy(),
        X_train=X_train,
        X_validation=X_val,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_val,
        y_test=y_test,
        numeric_feature_columns=common_cols,
        model_feature_columns=X_train.columns.tolist(),
    )


def _to_prediction_frame(
    *,
    model_name: str,
    split_name: str,
    frame: pd.DataFrame,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    out = frame[["sleeve_id", "month_end", TARGET_COL]].copy()
    out = out.rename(columns={TARGET_COL: "y_true"})
    out["y_pred"] = y_pred.astype(float)
    out["model"] = model_name
    out["split"] = split_name
    out["residual"] = out["y_true"] - out["y_pred"]
    return out[["model", "split", "sleeve_id", "month_end", "y_true", "y_pred", "residual"]]


def _validation_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _fit_ridge(design: DesignMatrices, *, random_state: int) -> BaselineModelResult:
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = alphas[0]
    best_rmse = float("inf")

    for alpha in alphas:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(design.X_train, design.y_train)
        pred_val = model.predict(design.X_validation)
        rmse = _validation_rmse(design.y_validation, pred_val)
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    final_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha)),
        ]
    )
    final_model.fit(design.X_train, design.y_train)
    pred_val = final_model.predict(design.X_validation)
    pred_test = final_model.predict(design.X_test)

    return BaselineModelResult(
        model_name="ridge_pooled",
        family="ridge",
        hyperparams={"alpha": best_alpha, "validation_rmse": best_rmse, "random_state": random_state},
        validation_predictions=_to_prediction_frame(
            model_name="ridge_pooled",
            split_name="validation",
            frame=design.validation_frame,
            y_pred=pred_val,
        ),
        test_predictions=_to_prediction_frame(
            model_name="ridge_pooled",
            split_name="test",
            frame=design.test_frame,
            y_pred=pred_test,
        ),
    )


def _fit_elastic_net(design: DesignMatrices, *, random_state: int) -> BaselineModelResult:
    alphas = [0.0005, 0.001, 0.005, 0.01, 0.05]
    l1_ratios = [0.2, 0.5, 0.8]
    best_params = {"alpha": alphas[0], "l1_ratio": l1_ratios[0]}
    best_rmse = float("inf")

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "elastic_net",
                        ElasticNet(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            max_iter=20000,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            model.fit(design.X_train, design.y_train)
            pred_val = model.predict(design.X_validation)
            rmse = _validation_rmse(design.y_validation, pred_val)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

    final_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "elastic_net",
                ElasticNet(
                    alpha=best_params["alpha"],
                    l1_ratio=best_params["l1_ratio"],
                    max_iter=20000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    final_model.fit(design.X_train, design.y_train)
    pred_val = final_model.predict(design.X_validation)
    pred_test = final_model.predict(design.X_test)

    return BaselineModelResult(
        model_name="elastic_net_pooled",
        family="elastic_net",
        hyperparams={
            "alpha": best_params["alpha"],
            "l1_ratio": best_params["l1_ratio"],
            "validation_rmse": best_rmse,
            "random_state": random_state,
        },
        validation_predictions=_to_prediction_frame(
            model_name="elastic_net_pooled",
            split_name="validation",
            frame=design.validation_frame,
            y_pred=pred_val,
        ),
        test_predictions=_to_prediction_frame(
            model_name="elastic_net_pooled",
            split_name="test",
            frame=design.test_frame,
            y_pred=pred_test,
        ),
    )


def _fit_random_forest(design: DesignMatrices, *, random_state: int) -> BaselineModelResult:
    param_grid = [
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 5},
        {"n_estimators": 400, "max_depth": 6, "min_samples_leaf": 5},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3},
    ]
    best_params = param_grid[0]
    best_rmse = float("inf")

    for params in param_grid:
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(design.X_train, design.y_train)
        pred_val = model.predict(design.X_validation)
        rmse = _validation_rmse(design.y_validation, pred_val)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    final_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(design.X_train, design.y_train)
    pred_val = final_model.predict(design.X_validation)
    pred_test = final_model.predict(design.X_test)

    return BaselineModelResult(
        model_name="random_forest_pooled",
        family="random_forest",
        hyperparams={**best_params, "validation_rmse": best_rmse, "random_state": random_state},
        validation_predictions=_to_prediction_frame(
            model_name="random_forest_pooled",
            split_name="validation",
            frame=design.validation_frame,
            y_pred=pred_val,
        ),
        test_predictions=_to_prediction_frame(
            model_name="random_forest_pooled",
            split_name="test",
            frame=design.test_frame,
            y_pred=pred_test,
        ),
    )


def _naive_sleeve_mean(design: DesignMatrices) -> BaselineModelResult:
    mean_by_sleeve = design.train_frame.groupby("sleeve_id")[TARGET_COL].mean()
    global_mean = float(design.train_frame[TARGET_COL].mean())

    val_pred = (
        design.validation_frame["sleeve_id"]
        .map(mean_by_sleeve)
        .fillna(global_mean)
        .to_numpy(dtype=float)
    )
    test_pred = (
        design.test_frame["sleeve_id"]
        .map(mean_by_sleeve)
        .fillna(global_mean)
        .to_numpy(dtype=float)
    )

    return BaselineModelResult(
        model_name="naive_sleeve_mean",
        family="naive",
        hyperparams={"type": "historical_mean_by_sleeve"},
        validation_predictions=_to_prediction_frame(
            model_name="naive_sleeve_mean",
            split_name="validation",
            frame=design.validation_frame,
            y_pred=val_pred,
        ),
        test_predictions=_to_prediction_frame(
            model_name="naive_sleeve_mean",
            split_name="test",
            frame=design.test_frame,
            y_pred=test_pred,
        ),
    )


def _naive_last_return(design: DesignMatrices) -> BaselineModelResult:
    if "ret_1m_lag" not in design.validation_frame.columns or "ret_1m_lag" not in design.test_frame.columns:
        raise ValueError("ret_1m_lag is required for naive_last_return baseline")

    val_pred = design.validation_frame["ret_1m_lag"].astype(float).to_numpy()
    test_pred = design.test_frame["ret_1m_lag"].astype(float).to_numpy()

    return BaselineModelResult(
        model_name="naive_last_return",
        family="naive",
        hyperparams={"type": "predict_ret_1m_lag"},
        validation_predictions=_to_prediction_frame(
            model_name="naive_last_return",
            split_name="validation",
            frame=design.validation_frame,
            y_pred=val_pred,
        ),
        test_predictions=_to_prediction_frame(
            model_name="naive_last_return",
            split_name="test",
            frame=design.test_frame,
            y_pred=test_pred,
        ),
    )


def run_baseline_suite(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Train all baselines and return predictions + model metadata."""
    design = prepare_design_matrices(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    results = [
        _naive_sleeve_mean(design),
        _naive_last_return(design),
        _fit_ridge(design, random_state=random_state),
        _fit_elastic_net(design, random_state=random_state),
        _fit_random_forest(design, random_state=random_state),
    ]

    pred_val = pd.concat([r.validation_predictions for r in results], ignore_index=True)
    pred_test = pd.concat([r.test_predictions for r in results], ignore_index=True)
    model_manifest = pd.DataFrame(
        [
            {
                "model": r.model_name,
                "family": r.family,
                "hyperparams": str(r.hyperparams),
            }
            for r in results
        ]
    )
    return pred_val, pred_test, model_manifest, design.numeric_feature_columns
