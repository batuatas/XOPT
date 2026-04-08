"""Prediction diagnostics for the selected v2 SAA model comparators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xoptpoe_v2_models.evaluate import regression_metrics


SELECTED_SHARED_PTO = 'pto_nn__core_plus_enrichment__shared_60_120'
SELECTED_SHARED_E2E = 'e2e_nn__core_plus_enrichment__shared_60_120'


@dataclass(frozen=True)
class PredictionDiagnostics:
    selected_experiments: dict[str, str]
    diagnostic_panel: pd.DataFrame
    summary: pd.DataFrame
    monthly_summary: pd.DataFrame
    top_predicted_frequency: pd.DataFrame


_KEY_ORDER = (
    'best_60_predictor',
    'best_120_predictor',
    'best_shared_predictor',
    'pto_nn_signal',
    'e2e_nn_signal',
)


def select_key_experiments(metrics_df: pd.DataFrame) -> dict[str, str]:
    eligible = metrics_df.loc[~metrics_df['model_name'].isin(['naive_mean'])].copy()
    best_60 = eligible.loc[eligible['horizon_mode'].eq('separate_60')].sort_values(
        ['validation_rmse', 'validation_corr'], ascending=[True, False]
    ).iloc[0]['experiment_name']
    best_120 = eligible.loc[eligible['horizon_mode'].eq('separate_120')].sort_values(
        ['validation_rmse', 'validation_corr'], ascending=[True, False]
    ).iloc[0]['experiment_name']
    best_shared = eligible.loc[eligible['horizon_mode'].eq('shared_60_120')].sort_values(
        ['validation_rmse', 'validation_corr'], ascending=[True, False]
    ).iloc[0]['experiment_name']
    return {
        'best_60_predictor': str(best_60),
        'best_120_predictor': str(best_120),
        'best_shared_predictor': str(best_shared),
        'pto_nn_signal': SELECTED_SHARED_PTO,
        'e2e_nn_signal': SELECTED_SHARED_E2E,
    }


def _top_predicted_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work['rank_within_month'] = work.groupby(['strategy_label', 'split', 'month_end', 'horizon_months'])['y_pred'].rank(method='first', ascending=False)
    top = work.loc[work['rank_within_month'].eq(1)].copy()
    freq = (
        top.groupby(['strategy_label', 'split', 'horizon_months', 'sleeve_id'], as_index=False)
        .size()
        .rename(columns={'size': 'top_month_count'})
    )
    month_counts = top.groupby(['strategy_label', 'split', 'horizon_months'], as_index=False)['month_end'].nunique().rename(columns={'month_end': 'month_count'})
    freq = freq.merge(month_counts, on=['strategy_label', 'split', 'horizon_months'], how='left', validate='m:1')
    freq['top_predicted_frequency'] = freq['top_month_count'] / freq['month_count']
    return freq.sort_values(['strategy_label', 'split', 'horizon_months', 'top_predicted_frequency', 'sleeve_id'], ascending=[True, True, True, False, True]).reset_index(drop=True)


def _monthly_rank_sign(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (strategy_label, split_name, horizon_months, month_end), chunk in frame.groupby(
        ['strategy_label', 'split', 'horizon_months', 'month_end'],
        as_index=False,
    ):
        spearman = float(chunk['y_pred'].corr(chunk['y_true'], method='spearman'))
        pearson = float(chunk['y_pred'].corr(chunk['y_true'], method='pearson'))
        sign_acc = float(np.mean((chunk['y_true'] >= 0.0) == (chunk['y_pred'] >= 0.0)))
        rows.append(
            {
                'strategy_label': strategy_label,
                'split': split_name,
                'horizon_months': int(horizon_months),
                'month_end': pd.Timestamp(month_end),
                'rank_ic_spearman': spearman,
                'rank_ic_pearson': pearson,
                'sign_accuracy': sign_acc,
                'predicted_dispersion': float(chunk['y_pred'].std(ddof=1)),
                'realized_dispersion': float(chunk['y_true'].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values(['strategy_label', 'split', 'horizon_months', 'month_end']).reset_index(drop=True)


def build_prediction_diagnostics(
    *,
    predictions_validation: pd.DataFrame,
    predictions_test: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> PredictionDiagnostics:
    selected = select_key_experiments(metrics_df)
    combined = pd.concat([predictions_validation, predictions_test], ignore_index=True)

    frames: list[pd.DataFrame] = []
    for strategy_label in _KEY_ORDER:
        experiment_name = selected[strategy_label]
        chunk = combined.loc[combined['experiment_name'].eq(experiment_name)].copy()
        chunk['strategy_label'] = strategy_label
        if strategy_label == 'best_60_predictor':
            chunk = chunk.loc[chunk['horizon_months'].eq(60)].copy()
        elif strategy_label == 'best_120_predictor':
            chunk = chunk.loc[chunk['horizon_months'].eq(120)].copy()
        chunk['month_end'] = pd.to_datetime(chunk['month_end'])
        chunk['residual'] = chunk['y_true'] - chunk['y_pred']
        chunk['abs_error'] = np.abs(chunk['residual'])
        chunk['squared_error'] = np.square(chunk['residual'])
        chunk['sign_correct'] = ((chunk['y_true'] >= 0.0) == (chunk['y_pred'] >= 0.0)).astype(int)
        frames.append(chunk)

    panel = pd.concat(frames, ignore_index=True).sort_values(
        ['strategy_label', 'split', 'month_end', 'horizon_months', 'sleeve_id']
    ).reset_index(drop=True)

    top_freq = _top_predicted_frequency(panel)
    monthly = _monthly_rank_sign(panel)

    rows: list[dict[str, object]] = []
    for (strategy_label, split_name, horizon_months), chunk in panel.groupby(['strategy_label', 'split', 'horizon_months'], as_index=False):
        benchmark = chunk['benchmark_pred'].to_numpy(dtype=float) if 'benchmark_pred' in chunk.columns else np.zeros(len(chunk))
        base = regression_metrics(
            chunk['y_true'].to_numpy(dtype=float),
            chunk['y_pred'].to_numpy(dtype=float),
            benchmark,
        )
        monthly_chunk = monthly.loc[
            monthly['strategy_label'].eq(strategy_label)
            & monthly['split'].eq(split_name)
            & monthly['horizon_months'].eq(horizon_months)
        ]
        top_chunk = top_freq.loc[
            top_freq['strategy_label'].eq(strategy_label)
            & top_freq['split'].eq(split_name)
            & top_freq['horizon_months'].eq(horizon_months)
        ].sort_values(['top_predicted_frequency', 'sleeve_id'], ascending=[False, True])
        top_sleeve = None
        top_sleeve_freq = float('nan')
        if not top_chunk.empty:
            top_sleeve = str(top_chunk.iloc[0]['sleeve_id'])
            top_sleeve_freq = float(top_chunk.iloc[0]['top_predicted_frequency'])
        rows.append(
            {
                'strategy_label': strategy_label,
                'split': split_name,
                'horizon_months': int(horizon_months),
                'row_count': int(len(chunk)),
                'month_count': int(chunk['month_end'].nunique()),
                'rmse': float(base['rmse']),
                'mae': float(base['mae']),
                'corr': float(base['corr']),
                'oos_r2_vs_naive': float(base['oos_r2_vs_naive']),
                'sign_accuracy': float(chunk['sign_correct'].mean()),
                'residual_mean': float(chunk['residual'].mean()),
                'residual_std': float(chunk['residual'].std(ddof=1)),
                'abs_error_mean': float(chunk['abs_error'].mean()),
                'rank_ic_spearman_mean': float(monthly_chunk['rank_ic_spearman'].mean()),
                'rank_ic_spearman_std': float(monthly_chunk['rank_ic_spearman'].std(ddof=1)),
                'sign_accuracy_monthly_mean': float(monthly_chunk['sign_accuracy'].mean()),
                'predicted_dispersion_mean': float(monthly_chunk['predicted_dispersion'].mean()),
                'top_predicted_sleeve': top_sleeve,
                'top_predicted_sleeve_frequency': top_sleeve_freq,
            }
        )
    summary = pd.DataFrame(rows).sort_values(['strategy_label', 'split', 'horizon_months']).reset_index(drop=True)
    return PredictionDiagnostics(
        selected_experiments=selected,
        diagnostic_panel=panel,
        summary=summary,
        monthly_summary=monthly,
        top_predicted_frequency=top_freq,
    )
