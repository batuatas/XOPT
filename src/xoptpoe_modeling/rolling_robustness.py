"""Compact rolling robustness evaluation for XOPTPOE v1 modeling prototype."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xoptpoe_modeling.baselines import DesignMatrices, prepare_design_matrices
from xoptpoe_modeling.eda import TARGET_COL, infer_feature_columns
from xoptpoe_modeling.evaluate import regression_metrics


MODEL_SET: tuple[str, ...] = ("naive_sleeve_mean", "ridge_pooled", "elastic_net_pooled")


@dataclass(frozen=True)
class RollingConfig:
    """Config for expanding-window rolling robustness checks."""

    min_train_months: int = 96
    validation_months: int = 24
    test_months: int = 24
    step_months: int = 12
    random_state: int = 42
    top_k: int = 3
    top_k_max_weight: float = 0.30
    score_max_weight: float = 0.35
    diversify_top_n: int = 5
    diversify_max_weight: float = 0.25


@dataclass(frozen=True)
class FoldDefinition:
    """One rolling fold with contiguous train/validation/test month blocks."""

    fold_id: int
    train_months: list[pd.Timestamp]
    validation_months: list[pd.Timestamp]
    test_months: list[pd.Timestamp]


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float("nan")
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    if xr.std(ddof=0) == 0 or yr.std(ddof=0) == 0:
        return float("nan")
    return float(xr.corr(yr))


def _ranking_metrics(monthly_pred: pd.DataFrame) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for _, chunk in monthly_pred.groupby("month_end"):
        if chunk.empty:
            continue
        ic = _safe_spearman(chunk["y_pred"], chunk["y_true"])
        k = min(3, len(chunk))
        top_mask = chunk["y_pred"].rank(method="first", ascending=False) <= k
        lift = float(chunk.loc[top_mask, "y_true"].mean() - chunk["y_true"].mean())
        rows.append({"spearman_ic": ic, "top3_lift": lift})

    if not rows:
        return {"spearman_ic_mean": float("nan"), "top3_lift_mean": float("nan")}
    out = pd.DataFrame(rows)
    return {
        "spearman_ic_mean": float(out["spearman_ic"].mean()),
        "top3_lift_mean": float(out["top3_lift"].mean()),
    }


def _validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"month_end", "sleeve_id", TARGET_COL}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"modeling panel missing required columns: {sorted(missing)}")

    work = panel.copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    dup_cnt = int(work.duplicated(subset=["month_end", "sleeve_id"]).sum())
    if dup_cnt > 0:
        raise ValueError(f"modeling panel has duplicate (month_end, sleeve_id) keys: {dup_cnt}")
    return work.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)


def generate_folds(panel: pd.DataFrame, config: RollingConfig) -> list[FoldDefinition]:
    """Generate expanding-window folds with fixed validation/test horizons."""
    work = _validate_panel(panel)
    months = pd.Index(sorted(work["month_end"].unique()))
    fold_list: list[FoldDefinition] = []

    train_end_idx = config.min_train_months - 1
    fold_id = 1
    while True:
        val_start = train_end_idx + 1
        val_end = val_start + config.validation_months - 1
        test_start = val_end + 1
        test_end = test_start + config.test_months - 1
        if test_end >= len(months):
            break

        fold_list.append(
            FoldDefinition(
                fold_id=fold_id,
                train_months=list(months[: train_end_idx + 1]),
                validation_months=list(months[val_start : val_end + 1]),
                test_months=list(months[test_start : test_end + 1]),
            )
        )
        fold_id += 1
        train_end_idx += config.step_months

    if not fold_list:
        raise ValueError("No rolling folds generated; adjust RollingConfig windows")
    return fold_list


def _prefix_match(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def build_feature_sets(panel: pd.DataFrame) -> dict[str, list[str]]:
    """Build compact feature-set variants for robustness checks."""
    all_features = infer_feature_columns(panel)

    technical_prefixes = ("ret_", "mom_", "vol_", "maxdd_", "rel_")
    global_prefixes = ("usd_broad", "vix", "us_real10y", "ig_oas", "oil_wti")
    macro_prefixes = (
        "infl_",
        "unemp_",
        "short_rate_",
        "long_rate_",
        "term_slope_",
        "local_",
        "usd_broad",
        "vix",
        "us_real10y",
        "ig_oas",
        "oil_wti",
        "macro_stale_flag",
    )

    technical = sorted([c for c in all_features if _prefix_match(c, technical_prefixes)])
    global_macro = sorted(
        [c for c in all_features if _prefix_match(c, global_prefixes) or c == "macro_stale_flag"]
    )
    macro_only = sorted([c for c in all_features if _prefix_match(c, macro_prefixes)])
    tech_plus_global = sorted(set(technical + global_macro))
    full = sorted(all_features)

    feature_sets = {
        "technical_only": technical,
        "macro_only": macro_only,
        "technical_plus_global_macro": tech_plus_global,
        "full_current_set": full,
    }
    for set_name, cols in feature_sets.items():
        if not cols:
            raise ValueError(f"Feature set '{set_name}' is empty")
    return feature_sets


def _fit_naive(design: DesignMatrices) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean_by_sleeve = design.train_frame.groupby("sleeve_id")[TARGET_COL].mean()
    global_mean = float(design.train_frame[TARGET_COL].mean())
    pred_val = (
        design.validation_frame["sleeve_id"].map(mean_by_sleeve).fillna(global_mean).to_numpy(dtype=float)
    )
    pred_test = design.test_frame["sleeve_id"].map(mean_by_sleeve).fillna(global_mean).to_numpy(dtype=float)
    return pred_val, pred_test, {"type": "historical_mean_by_sleeve"}


def _fit_ridge(design: DesignMatrices) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = alphas[0]
    best_rmse = float("inf")
    for alpha in alphas:
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        model.fit(design.X_train, design.y_train)
        pred_val = model.predict(design.X_validation)
        rmse = float(np.sqrt(mean_squared_error(design.y_validation, pred_val)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    final_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
    final_model.fit(design.X_train, design.y_train)
    pred_val = final_model.predict(design.X_validation)
    pred_test = final_model.predict(design.X_test)
    return pred_val, pred_test, {"alpha": best_alpha, "validation_rmse": best_rmse}


def _fit_elastic_net(
    design: DesignMatrices, *, random_state: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    alphas = [0.0005, 0.001, 0.005, 0.01, 0.05]
    l1_ratios = [0.2, 0.5, 0.8]
    best_alpha = alphas[0]
    best_l1 = l1_ratios[0]
    best_rmse = float("inf")

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = Pipeline(
                [
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
            rmse = float(np.sqrt(mean_squared_error(design.y_validation, pred_val)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
                best_l1 = l1_ratio

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elastic_net",
                ElasticNet(
                    alpha=best_alpha,
                    l1_ratio=best_l1,
                    max_iter=20000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    final_model.fit(design.X_train, design.y_train)
    pred_val = final_model.predict(design.X_validation)
    pred_test = final_model.predict(design.X_test)
    return pred_val, pred_test, {"alpha": best_alpha, "l1_ratio": best_l1, "validation_rmse": best_rmse}


def _fit_model(
    *,
    model_name: str,
    design: DesignMatrices,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if model_name == "naive_sleeve_mean":
        return _fit_naive(design)
    if model_name == "ridge_pooled":
        return _fit_ridge(design)
    if model_name == "elastic_net_pooled":
        return _fit_elastic_net(design, random_state=random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def _eval_split(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_benchmark: np.ndarray,
) -> dict[str, float]:
    return regression_metrics(y_true=y_true, y_pred=y_pred, y_benchmark=y_benchmark)


def run_feature_set_experiments(
    panel: pd.DataFrame,
    *,
    config: RollingConfig,
    model_names: tuple[str, ...] = MODEL_SET,
    feature_sets: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Run rolling fold experiments across feature sets and compact model set."""
    work = _validate_panel(panel)
    folds = generate_folds(work, config)
    resolved_feature_sets = feature_sets if feature_sets is not None else build_feature_sets(work)

    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    fold_manifest_rows: list[dict[str, Any]] = []

    for fold in folds:
        train_df = work.loc[work["month_end"].isin(fold.train_months)].copy()
        val_df = work.loc[work["month_end"].isin(fold.validation_months)].copy()
        test_df = work.loc[work["month_end"].isin(fold.test_months)].copy()

        fold_manifest_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_start": min(fold.train_months).date().isoformat(),
                "train_end": max(fold.train_months).date().isoformat(),
                "validation_start": min(fold.validation_months).date().isoformat(),
                "validation_end": max(fold.validation_months).date().isoformat(),
                "test_start": min(fold.test_months).date().isoformat(),
                "test_end": max(fold.test_months).date().isoformat(),
                "train_months": len(fold.train_months),
                "validation_months": len(fold.validation_months),
                "test_months": len(fold.test_months),
            }
        )

        for feature_set_name, feature_cols in resolved_feature_sets.items():
            design = prepare_design_matrices(
                train_df=train_df,
                validation_df=val_df,
                test_df=test_df,
                feature_columns=feature_cols,
            )

            train_mean = float(design.y_train.mean())
            bench_val = np.full(shape=design.y_validation.shape, fill_value=train_mean, dtype=float)
            bench_test = np.full(shape=design.y_test.shape, fill_value=train_mean, dtype=float)

            for model_name in model_names:
                pred_val, pred_test, hyperparams = _fit_model(
                    model_name=model_name,
                    design=design,
                    random_state=config.random_state,
                )

                val_metrics = _eval_split(
                    y_true=design.y_validation.to_numpy(dtype=float),
                    y_pred=np.asarray(pred_val, dtype=float),
                    y_benchmark=bench_val,
                )
                test_metrics = _eval_split(
                    y_true=design.y_test.to_numpy(dtype=float),
                    y_pred=np.asarray(pred_test, dtype=float),
                    y_benchmark=bench_test,
                )

                val_frame = design.validation_frame[["month_end", "sleeve_id", TARGET_COL]].copy()
                val_frame = val_frame.rename(columns={TARGET_COL: "y_true"})
                val_frame["y_pred"] = pred_val

                test_frame = design.test_frame[["month_end", "sleeve_id", TARGET_COL]].copy()
                test_frame = test_frame.rename(columns={TARGET_COL: "y_true"})
                test_frame["y_pred"] = pred_test

                val_rank = _ranking_metrics(val_frame)
                test_rank = _ranking_metrics(test_frame)

                fold_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "feature_count": len(feature_cols),
                        "validation_rmse": val_metrics["rmse"],
                        "validation_mae": val_metrics["mae"],
                        "validation_oos_r2": val_metrics["oos_r2"],
                        "validation_corr": val_metrics["corr"],
                        "validation_directional_accuracy": val_metrics["directional_accuracy"],
                        "validation_spearman_ic": val_rank["spearman_ic_mean"],
                        "validation_top3_lift": val_rank["top3_lift_mean"],
                        "test_rmse": test_metrics["rmse"],
                        "test_mae": test_metrics["mae"],
                        "test_oos_r2": test_metrics["oos_r2"],
                        "test_corr": test_metrics["corr"],
                        "test_directional_accuracy": test_metrics["directional_accuracy"],
                        "test_spearman_ic": test_rank["spearman_ic_mean"],
                        "test_top3_lift": test_rank["top3_lift_mean"],
                        "rmse_test_minus_validation": test_metrics["rmse"] - val_metrics["rmse"],
                        "hyperparams": str(hyperparams),
                    }
                )

                test_frame = test_frame.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)
                for row in test_frame.itertuples(index=False):
                    pred_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "month_end": pd.to_datetime(row.month_end),
                            "sleeve_id": row.sleeve_id,
                            "y_true": float(row.y_true),
                            "y_pred": float(row.y_pred),
                        }
                    )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(["fold_id", "feature_set", "model"]).reset_index(drop=True)
    test_predictions = (
        pd.DataFrame(pred_rows).sort_values(["fold_id", "model", "feature_set", "month_end", "sleeve_id"]).reset_index(drop=True)
    )
    fold_manifest = pd.DataFrame(fold_manifest_rows).sort_values("fold_id").reset_index(drop=True)
    return fold_metrics, test_predictions, fold_manifest, resolved_feature_sets


def summarize_feature_set_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics to compare feature sets/models."""
    if fold_metrics.empty:
        raise ValueError("fold_metrics is empty")

    naive = fold_metrics.loc[
        fold_metrics["model"] == "naive_sleeve_mean",
        ["fold_id", "feature_set", "test_rmse"],
    ].rename(columns={"test_rmse": "test_rmse_naive"})
    work = fold_metrics.merge(naive, on=["fold_id", "feature_set"], how="left", validate="many_to_one")
    work["test_rmse_delta_vs_naive"] = work["test_rmse"] - work["test_rmse_naive"]
    work["beats_naive_rmse"] = work["test_rmse_delta_vs_naive"] < 0

    grouped = (
        work.groupby(["feature_set", "model"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            feature_count=("feature_count", "max"),
            validation_rmse_mean=("validation_rmse", "mean"),
            validation_rmse_std=("validation_rmse", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_oos_r2_mean=("test_oos_r2", "mean"),
            test_corr_mean=("test_corr", "mean"),
            test_directional_accuracy_mean=("test_directional_accuracy", "mean"),
            test_spearman_ic_mean=("test_spearman_ic", "mean"),
            test_top3_lift_mean=("test_top3_lift", "mean"),
            rmse_test_minus_validation_mean=("rmse_test_minus_validation", "mean"),
            test_rmse_delta_vs_naive_mean=("test_rmse_delta_vs_naive", "mean"),
            beat_naive_rmse_fold_share=("beats_naive_rmse", "mean"),
        )
    )
    grouped["beat_naive_rmse_fold_share"] = grouped["beat_naive_rmse_fold_share"].astype(float)
    grouped.loc[grouped["model"] == "naive_sleeve_mean", "test_rmse_delta_vs_naive_mean"] = 0.0
    grouped.loc[grouped["model"] == "naive_sleeve_mean", "beat_naive_rmse_fold_share"] = np.nan
    return grouped.sort_values(["model", "test_rmse_mean"]).reset_index(drop=True)


def select_best_feature_set_per_model(summary_df: pd.DataFrame) -> dict[str, str]:
    """Pick one feature set per model by lowest mean validation RMSE."""
    selected: dict[str, str] = {}
    for model_name, chunk in summary_df.groupby("model"):
        row = chunk.sort_values(["validation_rmse_mean", "test_rmse_mean"]).iloc[0]
        selected[model_name] = str(row["feature_set"])
    return selected


def _normalize_long_only(values: np.ndarray) -> np.ndarray:
    vec = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(vec.sum())
    if total <= 0:
        return np.repeat(1.0 / len(vec), len(vec))
    return vec / total


def _project_with_cap(values: np.ndarray, *, max_weight: float, max_iter: int = 100) -> np.ndarray:
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be in (0, 1]")

    n = len(values)
    if n == 0:
        return values
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"Infeasible cap {max_weight} for {n} assets (cap*n < 1)")

    w = _normalize_long_only(values)
    for _ in range(max_iter):
        over = w > max_weight + 1e-12
        if not np.any(over):
            break
        excess = float(np.sum(w[over] - max_weight))
        w[over] = max_weight

        under = w < max_weight - 1e-12
        if not np.any(under):
            w = np.repeat(1.0 / n, n)
            break

        under_sum = float(np.sum(w[under]))
        if under_sum <= 0:
            w[under] = 1.0 / float(np.sum(under))
        else:
            w[under] = w[under] + excess * (w[under] / under_sum)

    w = np.clip(w, 0.0, max_weight)
    return _normalize_long_only(w)


def _weights_equal(scores: pd.Series) -> pd.Series:
    n = len(scores)
    return pd.Series(np.repeat(1.0 / n, n), index=scores.index, dtype=float)


def _weights_top_k_equal(scores: pd.Series, *, k: int) -> pd.Series:
    n = len(scores)
    k_eff = min(max(k, 1), n)
    top_idx = scores.sort_values(ascending=False).index[:k_eff]
    out = pd.Series(0.0, index=scores.index, dtype=float)
    out.loc[top_idx] = 1.0 / k_eff
    return out


def _weights_top_k_capped(scores: pd.Series, *, k: int, max_weight: float) -> pd.Series:
    n = len(scores)
    k_eff = min(max(k, 1), n)
    min_required = int(np.ceil(1.0 / max_weight))
    select_n = min(max(k_eff, min_required), n)
    top_idx = scores.sort_values(ascending=False).index[:select_n]

    raw = np.clip(scores.loc[top_idx].to_numpy(dtype=float), 0.0, None)
    if float(raw.sum()) <= 0:
        raw = np.ones_like(raw, dtype=float)
    raw = _normalize_long_only(raw)
    capped = _project_with_cap(raw, max_weight=max_weight)

    out = pd.Series(0.0, index=scores.index, dtype=float)
    out.loc[top_idx] = capped
    return out


def _weights_score_positive_capped(scores: pd.Series, *, max_weight: float) -> pd.Series:
    raw = np.clip(scores.to_numpy(dtype=float), 0.0, None)
    if float(raw.sum()) <= 0:
        raw = np.repeat(1.0 / len(raw), len(raw))
    base = _normalize_long_only(raw)
    capped = _project_with_cap(base, max_weight=max_weight)
    return pd.Series(capped, index=scores.index, dtype=float)


def _weights_top_k_diversified_cap(
    scores: pd.Series,
    *,
    top_n: int,
    max_weight: float,
) -> pd.Series:
    n = len(scores)
    n_eff = min(max(top_n, 1), n)
    top_idx = scores.sort_values(ascending=False).index[:n_eff]

    raw_scores = np.clip(scores.loc[top_idx].to_numpy(dtype=float), 0.0, None)
    if float(raw_scores.sum()) <= 0:
        raw_scores = np.linspace(n_eff, 1, num=n_eff, dtype=float)
    raw_scores = _normalize_long_only(raw_scores)
    capped = _project_with_cap(raw_scores, max_weight=max_weight)

    out = pd.Series(0.0, index=scores.index, dtype=float)
    out.loc[top_idx] = capped
    return out


def _strategy_metrics(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eq = (
        returns_df.loc[returns_df["strategy"] == "equal_weight", ["month_end", "portfolio_return"]]
        .rename(columns={"portfolio_return": "equal_weight_return"})
        .set_index("month_end")
    )
    for strategy, grp in returns_df.groupby("strategy"):
        r = grp["portfolio_return"].to_numpy(dtype=float)
        months = len(grp)
        mean_r = float(np.mean(r))
        vol_r = float(np.std(r, ddof=1)) if months > 1 else float("nan")
        sharpe = float((mean_r / vol_r) * sqrt(12.0)) if vol_r and vol_r > 0 else float("nan")
        nav = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(nav)
        drawdown = nav / peak - 1.0
        max_dd = float(np.min(drawdown))

        cmp = grp.set_index("month_end").join(eq, how="left")
        active = cmp["portfolio_return"] - cmp["equal_weight_return"]

        rows.append(
            {
                "strategy": strategy,
                "months": int(months),
                "avg_monthly_return": mean_r,
                "vol_monthly": vol_r,
                "sharpe_annualized": sharpe,
                "max_drawdown": max_dd,
                "avg_turnover": float(grp["turnover"].mean()),
                "avg_max_weight": float(grp["max_weight"].mean()),
                "avg_weight_hhi": float(grp["weight_hhi"].mean()),
                "avg_active_return_vs_equal_weight": float(active.mean()),
            }
        )
    return pd.DataFrame(rows)


def _run_portfolio_for_fold(
    fold_preds: pd.DataFrame,
    *,
    config: RollingConfig,
) -> pd.DataFrame:
    strategies = (
        "equal_weight",
        "top_k_equal",
        "top_k_capped",
        "score_positive_capped",
        "top_k_diversified_cap",
    )
    prev_weights: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []

    months = sorted(fold_preds["month_end"].unique())
    sleeves = sorted(fold_preds["sleeve_id"].unique())
    for month_end in months:
        chunk = fold_preds.loc[fold_preds["month_end"] == month_end].copy()
        scores = chunk.set_index("sleeve_id")["y_pred"].reindex(sleeves)
        realized = chunk.set_index("sleeve_id")["y_true"].reindex(sleeves)

        weight_map = {
            "equal_weight": _weights_equal(scores),
            "top_k_equal": _weights_top_k_equal(scores, k=config.top_k),
            "top_k_capped": _weights_top_k_capped(
                scores,
                k=config.top_k,
                max_weight=config.top_k_max_weight,
            ),
            "score_positive_capped": _weights_score_positive_capped(
                scores,
                max_weight=config.score_max_weight,
            ),
            "top_k_diversified_cap": _weights_top_k_diversified_cap(
                scores,
                top_n=max(config.diversify_top_n, config.top_k),
                max_weight=config.diversify_max_weight,
            ),
        }

        for strategy in strategies:
            w = weight_map[strategy]
            ret = float(np.dot(w.to_numpy(dtype=float), realized.to_numpy(dtype=float)))
            if strategy in prev_weights:
                turnover = float(0.5 * np.abs(w.to_numpy(dtype=float) - prev_weights[strategy].to_numpy(dtype=float)).sum())
            else:
                turnover = 0.0
            prev_weights[strategy] = w

            rows.append(
                {
                    "month_end": pd.to_datetime(month_end),
                    "strategy": strategy,
                    "portfolio_return": ret,
                    "turnover": turnover,
                    "max_weight": float(w.max()),
                    "weight_hhi": float(np.square(w.to_numpy(dtype=float)).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["strategy", "month_end"]).reset_index(drop=True)


def evaluate_concentration_controls(
    *,
    test_predictions: pd.DataFrame,
    selected_feature_set_by_model: dict[str, str],
    config: RollingConfig,
) -> pd.DataFrame:
    """Evaluate concentration-controlled portfolio variants on rolling test folds."""
    required = {"fold_id", "model", "feature_set", "month_end", "sleeve_id", "y_true", "y_pred"}
    missing = required - set(test_predictions.columns)
    if missing:
        raise ValueError(f"test_predictions missing required columns: {sorted(missing)}")

    fold_summary_rows: list[dict[str, Any]] = []
    for model_name, feature_set in selected_feature_set_by_model.items():
        chunk = test_predictions.loc[
            (test_predictions["model"] == model_name)
            & (test_predictions["feature_set"] == feature_set)
        ].copy()
        for fold_id, fold_chunk in chunk.groupby("fold_id"):
            returns_df = _run_portfolio_for_fold(fold_chunk, config=config)
            strat_summary = _strategy_metrics(returns_df)
            for row in strat_summary.itertuples(index=False):
                fold_summary_rows.append(
                    {
                        "fold_id": int(fold_id),
                        "model": model_name,
                        "feature_set": feature_set,
                        "strategy": row.strategy,
                        "months": int(row.months),
                        "avg_monthly_return": float(row.avg_monthly_return),
                        "vol_monthly": float(row.vol_monthly),
                        "sharpe_annualized": float(row.sharpe_annualized),
                        "max_drawdown": float(row.max_drawdown),
                        "avg_turnover": float(row.avg_turnover),
                        "avg_max_weight": float(row.avg_max_weight),
                        "avg_weight_hhi": float(row.avg_weight_hhi),
                        "avg_active_return_vs_equal_weight": float(row.avg_active_return_vs_equal_weight),
                    }
                )

    fold_summary = pd.DataFrame(fold_summary_rows)
    if fold_summary.empty:
        raise ValueError("No concentration-control fold summaries produced")

    out = (
        fold_summary.groupby(["model", "feature_set", "strategy"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            avg_monthly_return_mean=("avg_monthly_return", "mean"),
            avg_monthly_return_std=("avg_monthly_return", "std"),
            vol_monthly_mean=("vol_monthly", "mean"),
            sharpe_annualized_mean=("sharpe_annualized", "mean"),
            sharpe_annualized_std=("sharpe_annualized", "std"),
            max_drawdown_mean=("max_drawdown", "mean"),
            avg_turnover_mean=("avg_turnover", "mean"),
            avg_max_weight_mean=("avg_max_weight", "mean"),
            avg_weight_hhi_mean=("avg_weight_hhi", "mean"),
            avg_active_return_vs_equal_weight_mean=("avg_active_return_vs_equal_weight", "mean"),
        )
        .sort_values(["model", "strategy"])
        .reset_index(drop=True)
    )
    return out
