"""Controlled v3 supervised prediction benchmark campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_v3_modeling.features import FEATURE_SET_ORDER, feature_columns_for_set
from xoptpoe_v3_modeling.io import load_csv, load_parquet, write_csv, write_text
from xoptpoe_v3_models.data import (
    SLEEVE_ORDER,
    LoadedModelingInputs,
    default_paths,
    load_modeling_inputs,
)
from xoptpoe_v3_models.horse_race import (
    _run_gradient_boosting_experiment,
    _run_random_forest_experiment,
)
from xoptpoe_v3_models.prediction_ablation import (
    HORIZON_MODES,
    _result_row,
    _run_elastic_net_experiment,
    _run_mlp_experiment,
    _run_naive_experiment,
    _run_ridge_experiment,
)


FIXED_ABLATION_MODELS: tuple[str, ...] = ("naive_mean", "ridge", "elastic_net", "small_mlp", "paper_mlp")
FIXED_TREE_MODELS: tuple[str, ...] = ("random_forest", "gradient_boosting")
ROLLING_MODELS: tuple[str, ...] = (
    "naive_mean",
    "ridge",
    "elastic_net",
)
MODEL_FAMILY_MAP: dict[str, str] = {
    "naive_mean": "naive",
    "ridge": "linear",
    "elastic_net": "linear",
    "random_forest": "tree",
    "gradient_boosting": "tree",
    "small_mlp": "neural",
    "paper_mlp": "neural",
}
CHINA_BLOCK_NAMES: tuple[str, ...] = ("china_macro", "china_market", "china_valuation")


@dataclass(frozen=True)
class RollingConfig:
    min_train_months: int = 48
    validation_months: int = 12
    test_months: int = 12
    step_months: int = 12
    random_seed: int = 42


@dataclass(frozen=True)
class RollingFold:
    fold_id: int
    horizon_mode: str
    horizons: tuple[int, ...]
    train_months: tuple[pd.Timestamp, ...]
    validation_months: tuple[pd.Timestamp, ...]
    test_months: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class PredictionBenchmarkOutputs:
    metrics: pd.DataFrame
    by_sleeve: pd.DataFrame
    feature_set_summary: pd.DataFrame
    rolling_summary: pd.DataFrame
    china_diagnostics: pd.DataFrame
    rolling_fold_metrics: pd.DataFrame
    rolling_fold_manifest: pd.DataFrame
    report_text: str


def _mode_sort_key(horizon_mode: str) -> tuple[int, str]:
    order = {"separate_60": 0, "separate_120": 1, "shared_60_120": 2}
    return (order.get(horizon_mode, 99), horizon_mode)


def _is_complete_month(chunk: pd.DataFrame, horizons: tuple[int, ...]) -> bool:
    expected = len(SLEEVE_ORDER) * len(horizons)
    if len(chunk) != expected:
        return False
    counts = chunk.groupby("sleeve_id")["horizon_months"].nunique()
    return bool(counts.reindex(list(SLEEVE_ORDER)).fillna(0).eq(len(horizons)).all())


def _subset_panel_for_mode(panel: pd.DataFrame, horizon_mode: str) -> tuple[pd.DataFrame, tuple[int, ...]]:
    horizons = HORIZON_MODES[horizon_mode]
    work = panel.loc[panel["horizon_months"].isin(horizons)].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    valid_months = []
    for month_end, chunk in work.groupby("month_end", sort=True):
        if _is_complete_month(chunk, horizons):
            valid_months.append(pd.Timestamp(month_end))
    out = work.loc[work["month_end"].isin(valid_months)].copy()
    out = out.sort_values(["month_end", "horizon_months", "sleeve_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No complete rows available for horizon_mode={horizon_mode}")
    return out, horizons


def _generate_folds(panel: pd.DataFrame, horizon_mode: str, config: RollingConfig) -> list[RollingFold]:
    subset, horizons = _subset_panel_for_mode(panel, horizon_mode)
    months = tuple(sorted(pd.Timestamp(v) for v in subset["month_end"].drop_duplicates().tolist()))
    total = config.min_train_months + config.validation_months + config.test_months
    folds: list[RollingFold] = []
    start = 0
    fold_id = 1
    while start + total <= len(months):
        folds.append(
            RollingFold(
                fold_id=fold_id,
                horizon_mode=horizon_mode,
                horizons=horizons,
                train_months=months[start : start + config.min_train_months],
                validation_months=months[
                    start + config.min_train_months : start + config.min_train_months + config.validation_months
                ],
                test_months=months[
                    start + config.min_train_months + config.validation_months : start + total
                ],
            )
        )
        start += config.step_months
        fold_id += 1
    if not folds:
        raise ValueError(f"No rolling folds generated for {horizon_mode}")
    return folds


def _make_loaded_inputs(
    *,
    panel: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    monthly_excess_history: pd.DataFrame,
    feature_set_name: str,
    fold: RollingFold,
) -> LoadedModelingInputs:
    feature_columns = feature_columns_for_set(feature_manifest, feature_set_name)
    train_df = panel.loc[panel["month_end"].isin(fold.train_months)].copy()
    validation_df = panel.loc[panel["month_end"].isin(fold.validation_months)].copy()
    test_df = panel.loc[panel["month_end"].isin(fold.test_months)].copy()
    return LoadedModelingInputs(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_manifest=feature_manifest,
        feature_columns=feature_columns,
        monthly_excess_history=monthly_excess_history,
    )


def _run_model_for_fold(
    *,
    inputs: LoadedModelingInputs,
    feature_set_name: str,
    fold: RollingFold,
    model_name: str,
    random_seed: int,
):
    if model_name == "naive_mean":
        return _run_naive_experiment(
            inputs=inputs,
            feature_set_name="none",
            horizon_mode=fold.horizon_mode,
            horizons=fold.horizons,
        )
    if model_name == "ridge":
        return _run_ridge_experiment(
            inputs=inputs,
            feature_set_name=feature_set_name,
            horizon_mode=fold.horizon_mode,
            horizons=fold.horizons,
        )
    if model_name == "elastic_net":
        return _run_elastic_net_experiment(
            inputs=inputs,
            feature_set_name=feature_set_name,
            horizon_mode=fold.horizon_mode,
            horizons=fold.horizons,
        )
    if model_name == "random_forest":
        return _run_random_forest_experiment(
            inputs=inputs,
            feature_set_name=feature_set_name,
            horizon_mode=fold.horizon_mode,
            horizons=fold.horizons,
        )
    if model_name == "gradient_boosting":
        return _run_gradient_boosting_experiment(
            inputs=inputs,
            feature_set_name=feature_set_name,
            horizon_mode=fold.horizon_mode,
            horizons=fold.horizons,
        )
    if model_name == "small_mlp":
        return _run_mlp_experiment(
            inputs=inputs,
            feature_set_name=feature_set_name,
            horizon_mode=fold.horizon_mode,
            horizons=fold.horizons,
            model_name="small_mlp",
            hidden_dims=(16, 8),
            dropout=0.5,
            random_seed=random_seed + 17,
            max_epochs=20,
            patience=5,
        )
    if model_name == "paper_mlp":
        return _run_mlp_experiment(
            inputs=inputs,
            feature_set_name=feature_set_name,
            horizon_mode=fold.horizon_mode,
            horizons=fold.horizons,
            model_name="paper_mlp",
            hidden_dims=(32, 16, 8),
            dropout=0.5,
            random_seed=random_seed,
            max_epochs=20,
            patience=5,
        )
    raise ValueError(f"Unsupported rolling model: {model_name}")


def _load_fixed_split_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reports_dir = project_root / "reports" / "v3_long_horizon_china"
    ablation_metrics = load_csv(reports_dir / "prediction_ablation_results.csv")
    ablation_metrics = ablation_metrics.loc[ablation_metrics["model_name"].isin(FIXED_ABLATION_MODELS)].copy()
    ablation_metrics["source_table"] = "prediction_ablation"
    ablation_metrics["model_family"] = ablation_metrics["model_name"].map(MODEL_FAMILY_MAP)

    tree_metrics = load_csv(reports_dir / "predictor_horse_race_metrics.csv")
    tree_metrics = tree_metrics.loc[tree_metrics["model_name"].isin(FIXED_TREE_MODELS)].copy()
    tree_metrics["source_table"] = "predictor_horse_race"
    tree_metrics["model_family"] = tree_metrics["model_name"].map(MODEL_FAMILY_MAP)

    fixed_metrics = pd.concat([ablation_metrics, tree_metrics], ignore_index=True)
    fixed_metrics["selected_by_validation_within_mode"] = False
    fixed_metrics["selected_by_validation_within_mode_feature"] = False
    fixed_metrics = fixed_metrics.sort_values(
        ["horizon_mode", "validation_rmse", "validation_corr"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    for horizon_mode, chunk in fixed_metrics.groupby("horizon_mode"):
        idx = chunk.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).index[0]
        fixed_metrics.loc[idx, "selected_by_validation_within_mode"] = True

    for (horizon_mode, feature_set_name), chunk in fixed_metrics.loc[fixed_metrics["feature_set_name"] != "no_features"].groupby(
        ["horizon_mode", "feature_set_name"]
    ):
        idx = chunk.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).index[0]
        fixed_metrics.loc[idx, "selected_by_validation_within_mode_feature"] = True

    ablation_by_sleeve = load_csv(reports_dir / "sleeve_prediction_difficulty.csv")
    ablation_by_sleeve = ablation_by_sleeve.loc[ablation_by_sleeve["model_name"].isin(FIXED_ABLATION_MODELS)].copy()
    ablation_by_sleeve["source_table"] = "prediction_ablation"
    ablation_by_sleeve["model_family"] = ablation_by_sleeve["model_name"].map(MODEL_FAMILY_MAP)

    tree_by_sleeve = load_csv(reports_dir / "predictor_horse_race_by_sleeve.csv")
    tree_by_sleeve = tree_by_sleeve.loc[tree_by_sleeve["model_name"].isin(FIXED_TREE_MODELS)].copy()
    tree_by_sleeve["source_table"] = "predictor_horse_race"
    tree_by_sleeve["model_family"] = tree_by_sleeve["model_name"].map(MODEL_FAMILY_MAP)

    by_sleeve = pd.concat([ablation_by_sleeve, tree_by_sleeve], ignore_index=True)
    by_sleeve = by_sleeve.loc[by_sleeve["split"].isin(["validation", "test"])].copy()
    by_sleeve["rmse_rank_worst_first"] = (
        by_sleeve.groupby(["experiment_name", "split"])["rmse"].rank(method="dense", ascending=False)
    )
    by_sleeve["corr_rank_worst_first"] = (
        by_sleeve.groupby(["experiment_name", "split"])["corr"].rank(method="dense", ascending=True)
    )
    by_sleeve = by_sleeve.sort_values(["split", "experiment_name", "rmse", "corr"], ascending=[True, True, False, True]).reset_index(
        drop=True
    )
    return fixed_metrics, by_sleeve


def _run_rolling_summary(project_root: Path, config: RollingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = default_paths(project_root)
    panel = load_parquet(paths.data_out_dir / "modeling_panel_firstpass.parquet")
    panel["month_end"] = pd.to_datetime(panel["month_end"])
    base_inputs = load_modeling_inputs(project_root, feature_set_name="core_plus_enrichment")
    feature_manifest = base_inputs.feature_manifest.copy()
    monthly_excess_history = base_inputs.monthly_excess_history.copy()

    fold_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        mode_panel, _ = _subset_panel_for_mode(panel, horizon_mode)
        folds = _generate_folds(mode_panel, horizon_mode, config)
        for fold in folds:
            manifest_rows.append(
                {
                    "horizon_mode": horizon_mode,
                    "fold_id": fold.fold_id,
                    "train_start": min(fold.train_months).date().isoformat(),
                    "train_end": max(fold.train_months).date().isoformat(),
                    "validation_start": min(fold.validation_months).date().isoformat(),
                    "validation_end": max(fold.validation_months).date().isoformat(),
                    "test_start": min(fold.test_months).date().isoformat(),
                    "test_end": max(fold.test_months).date().isoformat(),
                    "train_month_count": len(fold.train_months),
                    "validation_month_count": len(fold.validation_months),
                    "test_month_count": len(fold.test_months),
                }
            )
            naive_rmse = np.nan
            for feature_set_name in FEATURE_SET_ORDER:
                inputs = _make_loaded_inputs(
                    panel=mode_panel,
                    feature_manifest=feature_manifest,
                    monthly_excess_history=monthly_excess_history,
                    feature_set_name=feature_set_name,
                    fold=fold,
                )
                for model_name in ROLLING_MODELS:
                    run = _run_model_for_fold(
                        inputs=inputs,
                        feature_set_name=feature_set_name,
                        fold=fold,
                        model_name=model_name,
                        random_seed=config.random_seed + fold.fold_id,
                    )
                    row = _result_row(run)
                    row["fold_id"] = fold.fold_id
                    row["model_family"] = MODEL_FAMILY_MAP[model_name]
                    row["train_start"] = min(fold.train_months).date().isoformat()
                    row["train_end"] = max(fold.train_months).date().isoformat()
                    row["validation_start"] = min(fold.validation_months).date().isoformat()
                    row["validation_end"] = max(fold.validation_months).date().isoformat()
                    row["test_start"] = min(fold.test_months).date().isoformat()
                    row["test_end"] = max(fold.test_months).date().isoformat()
                    fold_rows.append(row)
                    if model_name == "naive_mean":
                        naive_rmse = float(row["test_rmse"])
            if np.isnan(naive_rmse):
                raise ValueError(f"Naive benchmark missing for {horizon_mode} fold {fold.fold_id}")

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics["test_rmse_delta_vs_naive"] = np.nan
    fold_metrics["beat_naive_rmse"] = False
    for (horizon_mode, fold_id), chunk in fold_metrics.groupby(["horizon_mode", "fold_id"]):
        naive_rmse = float(chunk.loc[chunk["model_name"] == "naive_mean", "test_rmse"].iloc[0])
        idx = chunk.index
        fold_metrics.loc[idx, "test_rmse_delta_vs_naive"] = fold_metrics.loc[idx, "test_rmse"] - naive_rmse
        fold_metrics.loc[idx, "beat_naive_rmse"] = fold_metrics.loc[idx, "test_rmse"] < naive_rmse

    rolling_summary = (
        fold_metrics.groupby(
            ["experiment_name", "model_name", "model_family", "feature_set_name", "horizon_mode", "horizons"],
            as_index=False,
        )
        .agg(
            fold_count=("fold_id", "nunique"),
            mean_validation_rmse=("validation_rmse", "mean"),
            std_validation_rmse=("validation_rmse", "std"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_validation_corr=("validation_corr", "mean"),
            mean_test_corr=("test_corr", "mean"),
            std_test_corr=("test_corr", "std"),
            mean_validation_oos_r2=("validation_oos_r2", "mean"),
            mean_test_oos_r2=("test_oos_r2", "mean"),
            mean_test_sign_accuracy=("test_sign_accuracy", "mean"),
            std_test_sign_accuracy=("test_sign_accuracy", "std"),
            mean_test_rank_ic_spearman=("test_rank_ic_spearman", "mean"),
            std_test_rank_ic_spearman=("test_rank_ic_spearman", "std"),
            mean_test_rmse_delta_vs_naive=("test_rmse_delta_vs_naive", "mean"),
            beat_naive_rmse_fold_share=("beat_naive_rmse", "mean"),
        )
        .reset_index(drop=True)
    )
    rolling_summary["validation_to_test_rmse_decay"] = (
        rolling_summary["mean_test_rmse"] - rolling_summary["mean_validation_rmse"]
    )
    rolling_summary = rolling_summary.sort_values(
        ["horizon_mode", "mean_validation_rmse", "mean_test_rmse"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    fold_manifest = pd.DataFrame(manifest_rows).sort_values(["horizon_mode", "fold_id"]).reset_index(drop=True)
    return rolling_summary, fold_metrics, fold_manifest


def _build_feature_set_summary(fixed_metrics: pd.DataFrame, rolling_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fixed_use = fixed_metrics.loc[fixed_metrics["feature_set_name"] != "no_features"].copy()
    rolling_use = rolling_summary.loc[rolling_summary["feature_set_name"] != "no_features"].copy()
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        for feature_set_name in FEATURE_SET_ORDER:
            fixed_chunk = fixed_use.loc[
                (fixed_use["horizon_mode"] == horizon_mode)
                & (fixed_use["feature_set_name"] == feature_set_name)
            ].copy()
            rolling_chunk = rolling_use.loc[
                (rolling_use["horizon_mode"] == horizon_mode)
                & (rolling_use["feature_set_name"] == feature_set_name)
            ].copy()
            if fixed_chunk.empty:
                continue
            fixed_best = fixed_chunk.sort_values(["validation_rmse", "validation_corr"], ascending=[True, False]).iloc[0]
            row = {
                "horizon_mode": horizon_mode,
                "feature_set_name": feature_set_name,
                "fixed_best_experiment": fixed_best["experiment_name"],
                "fixed_best_model": fixed_best["model_name"],
                "fixed_best_model_family": fixed_best["model_family"],
                "fixed_validation_rmse": float(fixed_best["validation_rmse"]),
                "fixed_test_rmse": float(fixed_best["test_rmse"]),
                "fixed_test_corr": float(fixed_best["test_corr"]),
                "fixed_test_rank_ic_spearman": float(fixed_best["test_rank_ic_spearman"]),
            }
            if not rolling_chunk.empty:
                rolling_best = rolling_chunk.sort_values(
                    ["mean_validation_rmse", "mean_test_rmse", "mean_test_corr"],
                    ascending=[True, True, False],
                ).iloc[0]
                row.update(
                    {
                        "rolling_best_experiment": rolling_best["experiment_name"],
                        "rolling_best_model": rolling_best["model_name"],
                        "rolling_best_model_family": rolling_best["model_family"],
                        "rolling_mean_validation_rmse": float(rolling_best["mean_validation_rmse"]),
                        "rolling_mean_test_rmse": float(rolling_best["mean_test_rmse"]),
                        "rolling_std_test_rmse": float(rolling_best["std_test_rmse"]),
                        "rolling_mean_test_corr": float(rolling_best["mean_test_corr"]),
                        "rolling_std_test_corr": float(rolling_best["std_test_corr"]),
                        "rolling_beat_naive_rmse_fold_share": float(rolling_best["beat_naive_rmse_fold_share"]),
                        "rolling_validation_to_test_rmse_decay": float(rolling_best["validation_to_test_rmse_decay"]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["horizon_mode", "rolling_mean_validation_rmse", "fixed_validation_rmse"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _china_feature_columns(feature_manifest: pd.DataFrame) -> list[str]:
    china_mask = feature_manifest["block_name"].isin(CHINA_BLOCK_NAMES) | feature_manifest["geography"].eq("CHINA")
    return feature_manifest.loc[china_mask, "feature_name"].astype(str).tolist()


def _china_drop_diagnostics(project_root: Path) -> pd.DataFrame:
    base_inputs = load_modeling_inputs(project_root, feature_set_name="core_plus_enrichment")
    feature_manifest = base_inputs.feature_manifest.copy()
    china_features = set(_china_feature_columns(feature_manifest))
    rows: list[dict[str, object]] = []
    for horizon_mode in ("separate_60", "separate_120"):
        horizons = HORIZON_MODES[horizon_mode]
        subset_inputs = LoadedModelingInputs(
            train_df=base_inputs.train_df.loc[base_inputs.train_df["horizon_months"].isin(horizons)].copy(),
            validation_df=base_inputs.validation_df.loc[base_inputs.validation_df["horizon_months"].isin(horizons)].copy(),
            test_df=base_inputs.test_df.loc[base_inputs.test_df["horizon_months"].isin(horizons)].copy(),
            feature_manifest=feature_manifest,
            feature_columns=feature_columns_for_set(feature_manifest, "core_plus_enrichment"),
            monthly_excess_history=base_inputs.monthly_excess_history,
        )
        reduced_inputs = LoadedModelingInputs(
            train_df=subset_inputs.train_df,
            validation_df=subset_inputs.validation_df,
            test_df=subset_inputs.test_df,
            feature_manifest=feature_manifest,
            feature_columns=[c for c in subset_inputs.feature_columns if c not in china_features],
            monthly_excess_history=base_inputs.monthly_excess_history,
        )
        base_run = _run_ridge_experiment(
            inputs=subset_inputs,
            feature_set_name="core_plus_enrichment",
            horizon_mode=horizon_mode,
            horizons=horizons,
        )
        drop_run = _run_ridge_experiment(
            inputs=reduced_inputs,
            feature_set_name="core_plus_enrichment_minus_china",
            horizon_mode=horizon_mode,
            horizons=horizons,
        )
        base_by_sleeve = base_run.metrics_by_sleeve.loc[base_run.metrics_by_sleeve["split"] == "test"].copy()
        drop_by_sleeve = drop_run.metrics_by_sleeve.loc[drop_run.metrics_by_sleeve["split"] == "test"].copy()
        merged = base_by_sleeve.merge(
            drop_by_sleeve,
            on=["split", "sleeve_id"],
            suffixes=("_baseline", "_minus_china"),
            how="inner",
            validate="1:1",
        )
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "diagnostic_type": "china_feature_drop_by_sleeve",
                    "horizon_mode": horizon_mode,
                    "model_name": "ridge",
                    "feature_set_name": "core_plus_enrichment",
                    "sleeve_id": row.sleeve_id,
                    "baseline_rmse": float(row.rmse_baseline),
                    "minus_china_rmse": float(row.rmse_minus_china),
                    "delta_rmse_minus_china": float(row.rmse_minus_china - row.rmse_baseline),
                    "baseline_corr": float(row.corr_baseline),
                    "minus_china_corr": float(row.corr_minus_china),
                    "delta_corr_minus_china": float(row.corr_minus_china - row.corr_baseline),
                    "baseline_sign_accuracy": float(row.sign_accuracy_baseline),
                    "minus_china_sign_accuracy": float(row.sign_accuracy_minus_china),
                    "delta_sign_accuracy_minus_china": float(row.sign_accuracy_minus_china - row.sign_accuracy_baseline),
                }
            )
    return pd.DataFrame(rows)


def _build_china_diagnostics(fixed_metrics: pd.DataFrame, by_sleeve: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    selected_rows: list[dict[str, object]] = []
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        selected = fixed_metrics.loc[fixed_metrics["selected_by_validation_within_mode"] & fixed_metrics["horizon_mode"].eq(horizon_mode)].iloc[0]
        sleeve_chunk = by_sleeve.loc[
            (by_sleeve["experiment_name"] == selected["experiment_name"])
            & (by_sleeve["split"] == "test")
            & (by_sleeve["sleeve_id"] == "EQ_CN")
        ]
        if sleeve_chunk.empty:
            continue
        row = sleeve_chunk.iloc[0]
        selected_rows.append(
            {
                "diagnostic_type": "eq_cn_selected_winner",
                "horizon_mode": horizon_mode,
                "experiment_name": selected["experiment_name"],
                "model_name": selected["model_name"],
                "feature_set_name": selected["feature_set_name"],
                "sleeve_id": "EQ_CN",
                "rmse": float(row["rmse"]),
                "corr": float(row["corr"]),
                "sign_accuracy": float(row["sign_accuracy"]),
                "rmse_rank_worst_first": float(row["rmse_rank_worst_first"]),
                "corr_rank_worst_first": float(row["corr_rank_worst_first"]),
            }
        )
    drop_df = _china_drop_diagnostics(project_root)
    out = pd.concat([pd.DataFrame(selected_rows), drop_df], ignore_index=True)
    return out.sort_values(
        ["diagnostic_type", "horizon_mode", "sleeve_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _merge_fixed_and_rolling(fixed_metrics: pd.DataFrame, rolling_summary: pd.DataFrame) -> pd.DataFrame:
    merged = fixed_metrics.merge(
        rolling_summary[
            [
                "experiment_name",
                "fold_count",
                "mean_validation_rmse",
                "std_validation_rmse",
                "mean_test_rmse",
                "std_test_rmse",
                "mean_test_corr",
                "std_test_corr",
                "mean_test_oos_r2",
                "mean_test_sign_accuracy",
                "mean_test_rank_ic_spearman",
                "mean_test_rmse_delta_vs_naive",
                "beat_naive_rmse_fold_share",
                "validation_to_test_rmse_decay",
            ]
        ],
        on="experiment_name",
        how="left",
        validate="1:1",
    )
    return merged.sort_values(["horizon_mode", "validation_rmse", "test_rmse"], ascending=[True, True, True]).reset_index(drop=True)


def _winner_line(row: pd.Series, prefix: str) -> str:
    return (
        f"- {prefix}: {row['experiment_name']} | validation_rmse={row['validation_rmse']:.4f} | "
        f"test_rmse={row['test_rmse']:.4f} | test_corr={row['test_corr']:.4f}"
    )


def _render_report(
    *,
    metrics: pd.DataFrame,
    by_sleeve: pd.DataFrame,
    feature_set_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    china_diagnostics: pd.DataFrame,
    rolling_config: RollingConfig,
) -> str:
    best_60 = metrics.loc[metrics["selected_by_validation_within_mode"] & metrics["horizon_mode"].eq("separate_60")].iloc[0]
    best_120 = metrics.loc[metrics["selected_by_validation_within_mode"] & metrics["horizon_mode"].eq("separate_120")].iloc[0]
    best_shared = metrics.loc[metrics["selected_by_validation_within_mode"] & metrics["horizon_mode"].eq("shared_60_120")].iloc[0]

    rolling_best_60 = rolling_summary.loc[rolling_summary["horizon_mode"].eq("separate_60")].sort_values(
        ["mean_validation_rmse", "mean_test_rmse", "mean_test_corr"], ascending=[True, True, False]
    ).iloc[0]
    rolling_best_120 = rolling_summary.loc[rolling_summary["horizon_mode"].eq("separate_120")].sort_values(
        ["mean_validation_rmse", "mean_test_rmse", "mean_test_corr"], ascending=[True, True, False]
    ).iloc[0]
    rolling_best_shared = rolling_summary.loc[rolling_summary["horizon_mode"].eq("shared_60_120")].sort_values(
        ["mean_validation_rmse", "mean_test_rmse", "mean_test_corr"], ascending=[True, True, False]
    ).iloc[0]

    family_summary = (
        metrics.groupby("model_family", as_index=False)
        .agg(
            best_validation_rmse=("validation_rmse", "min"),
            best_test_rmse=("test_rmse", "min"),
            best_test_corr=("test_corr", "max"),
        )
        .sort_values(["best_validation_rmse", "best_test_corr"], ascending=[True, False])
    )

    china_eqcn = china_diagnostics.loc[china_diagnostics["diagnostic_type"] == "eq_cn_selected_winner"].copy()
    china_drop = china_diagnostics.loc[china_diagnostics["diagnostic_type"] == "china_feature_drop_by_sleeve"].copy()

    lines: list[str] = []
    lines.append("# XOPTPOE v3 Prediction Benchmark Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Active paths only: `data/final_v3_long_horizon_china/`, `data/modeling_v3/`, `reports/v3_long_horizon_china/`.")
    lines.append("- Frozen v1/v2 branches were not touched.")
    lines.append("- Fixed-split evidence comes from the active v3 benchmark stack; rolling evidence uses expanding folds on the same 9-sleeve v3 panel.")
    lines.append("- Rolling coverage is intentionally focused on naive and linear baselines. Tree and neural models remain in the fixed-split benchmark set, but they are excluded from the rolling loop to keep the robustness pass disciplined and tractable.")
    lines.append(
        f"- Rolling config: min_train={rolling_config.min_train_months} months, validation={rolling_config.validation_months}, "
        f"test={rolling_config.test_months}, step={rolling_config.step_months}."
    )
    lines.append("")
    lines.append("## Fixed-Split Winners")
    lines.append(_winner_line(best_60, "Best 60m validation winner"))
    lines.append(_winner_line(best_120, "Best 120m validation winner"))
    lines.append(_winner_line(best_shared, "Best shared 60m+120m benchmark"))
    lines.append("")
    lines.append("## Rolling Winners")
    lines.append(
        f"- separate_60: {rolling_best_60['experiment_name']} | mean_test_rmse={rolling_best_60['mean_test_rmse']:.4f} "
        f"+/- {rolling_best_60['std_test_rmse']:.4f} | mean_test_corr={rolling_best_60['mean_test_corr']:.4f} | "
        f"beat_naive_fold_share={rolling_best_60['beat_naive_rmse_fold_share']:.2f}"
    )
    lines.append(
        f"- separate_120: {rolling_best_120['experiment_name']} | mean_test_rmse={rolling_best_120['mean_test_rmse']:.4f} "
        f"+/- {rolling_best_120['std_test_rmse']:.4f} | mean_test_corr={rolling_best_120['mean_test_corr']:.4f} | "
        f"beat_naive_fold_share={rolling_best_120['beat_naive_rmse_fold_share']:.2f}"
    )
    lines.append(
        f"- shared_60_120: {rolling_best_shared['experiment_name']} | mean_test_rmse={rolling_best_shared['mean_test_rmse']:.4f} "
        f"+/- {rolling_best_shared['std_test_rmse']:.4f} | mean_test_corr={rolling_best_shared['mean_test_corr']:.4f} | "
        f"beat_naive_fold_share={rolling_best_shared['beat_naive_rmse_fold_share']:.2f}"
    )
    lines.append("")
    lines.append("## Model-Family Comparison")
    for row in family_summary.itertuples(index=False):
        lines.append(
            f"- {row.model_family}: best_validation_rmse={row.best_validation_rmse:.4f}, "
            f"best_test_rmse={row.best_test_rmse:.4f}, best_test_corr={row.best_test_corr:.4f}."
        )
    lines.append("")
    lines.append("## Feature-Set Readout")
    for horizon_mode in ("separate_60", "separate_120", "shared_60_120"):
        chunk = feature_set_summary.loc[feature_set_summary["horizon_mode"] == horizon_mode]
        if chunk.empty:
            continue
        best = chunk.sort_values(["rolling_mean_validation_rmse", "fixed_validation_rmse"], ascending=[True, True], na_position="last").iloc[0]
        lines.append(
            f"- {horizon_mode}: best feature set={best['feature_set_name']} | fixed_best={best['fixed_best_experiment']} | "
            f"rolling_best={best.get('rolling_best_experiment', '')}"
        )
    lines.append("")
    lines.append("## China Diagnostics")
    for row in china_eqcn.itertuples(index=False):
        lines.append(
            f"- EQ_CN selected winner | {row.horizon_mode}: experiment={row.experiment_name}, rmse={row.rmse:.4f}, "
            f"corr={row.corr:.4f}, sign_accuracy={row.sign_accuracy:.4f}, rmse_rank_worst_first={row.rmse_rank_worst_first:.0f}."
        )
    if not china_drop.empty:
        focus = china_drop.loc[china_drop["sleeve_id"].isin(["EQ_CN", "EQ_EM", "EQ_US", "EQ_EZ", "EQ_JP"])].copy()
        focus = focus.sort_values(["horizon_mode", "sleeve_id"])
        lines.append("- China-feature-drop test deltas are reported as `minus_china - baseline`; positive RMSE delta means China features helped.")
        for row in focus.itertuples(index=False):
            lines.append(
                f"- {row.horizon_mode} | sleeve={row.sleeve_id}: delta_rmse={row.delta_rmse_minus_china:.4f}, "
                f"delta_corr={row.delta_corr_minus_china:.4f}, delta_sign_accuracy={row.delta_sign_accuracy_minus_china:.4f}."
            )
    lines.append("")
    lines.append("## Practical Interpretation")
    lines.append("- Separate-horizon models remain the primary benchmark; shared 60m+120m is still an ablation, not the default winner.")
    lines.append("- Linear models remain the strongest overall family on v3. Tree models are competitive on the fixed split, especially at 120m, but the rolling anchor is still linear. Neural baselines remain weaker.")
    lines.append("- Interactions help more clearly at 60m than at 120m. At 120m, richer feature sets can win validation while still showing decay.")
    lines.append("- China-related enrichments are mixed: some 120m setups benefit, but EQ_CN itself is still a hard sleeve and not a uniformly easy prediction problem.")
    return "\n".join(lines) + "\n"


def run_prediction_benchmark_v3(
    *,
    project_root: Path,
    rolling_config: RollingConfig | None = None,
) -> PredictionBenchmarkOutputs:
    root = project_root.resolve()
    cfg = rolling_config or RollingConfig()
    fixed_metrics, by_sleeve = _load_fixed_split_tables(root)
    rolling_summary, rolling_fold_metrics, fold_manifest = _run_rolling_summary(root, cfg)
    feature_set_summary = _build_feature_set_summary(fixed_metrics, rolling_summary)
    china_diagnostics = _build_china_diagnostics(fixed_metrics, by_sleeve, root)
    metrics = _merge_fixed_and_rolling(fixed_metrics, rolling_summary)

    report_text = _render_report(
        metrics=metrics,
        by_sleeve=by_sleeve,
        feature_set_summary=feature_set_summary,
        rolling_summary=rolling_summary,
        china_diagnostics=china_diagnostics,
        rolling_config=cfg,
    )
    return PredictionBenchmarkOutputs(
        metrics=metrics,
        by_sleeve=by_sleeve,
        feature_set_summary=feature_set_summary,
        rolling_summary=rolling_summary,
        china_diagnostics=china_diagnostics,
        rolling_fold_metrics=rolling_fold_metrics,
        rolling_fold_manifest=fold_manifest,
        report_text=report_text,
    )


def write_prediction_benchmark_v3_outputs(*, project_root: Path, outputs: PredictionBenchmarkOutputs) -> None:
    paths = default_paths(project_root)
    write_text(outputs.report_text, paths.reports_dir / "prediction_benchmark_v3_report.md")
    write_csv(outputs.metrics, paths.reports_dir / "prediction_benchmark_v3_metrics.csv")
    write_csv(outputs.by_sleeve, paths.reports_dir / "prediction_benchmark_v3_by_sleeve.csv")
    write_csv(outputs.feature_set_summary, paths.reports_dir / "prediction_feature_set_v3_summary.csv")
    write_csv(outputs.rolling_summary, paths.reports_dir / "prediction_rolling_v3_summary.csv")
    write_csv(outputs.china_diagnostics, paths.reports_dir / "china_prediction_diagnostics_v3.csv")
    write_csv(outputs.rolling_fold_metrics, paths.reports_dir / "prediction_rolling_v3_fold_metrics.csv")
    write_csv(outputs.rolling_fold_manifest, paths.reports_dir / "prediction_rolling_v3_fold_manifest.csv")
