"""Portfolio diagnostics for the selected v2 SAA signal comparators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_v2_models.data import SLEEVE_ORDER, TARGET_COL, load_modeling_inputs
from xoptpoe_v2_models.optim_layers import OptimizerConfig, RiskConfig, RobustOptimizerCache, build_sigma_map
from xoptpoe_v2_models.portfolio_eval import run_portfolio_evaluation


@dataclass(frozen=True)
class PortfolioDiagnostics:
    weights_panel: pd.DataFrame
    returns_panel: pd.DataFrame
    summary: pd.DataFrame
    sleeve_attribution_summary: pd.DataFrame
    best_worst_months: pd.DataFrame


_DIAGNOSTIC_STRATEGIES = (
    'equal_weight',
    'best_60_predictor',
    'best_120_predictor',
    'best_shared_predictor',
    'combined_60_120_predictor',
    'pto_nn_signal',
    'e2e_nn_signal',
)


def _truth_panel(frame: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(['month_end', 'sleeve_id'], as_index=False)
        .agg(horizon_count=('horizon_months', 'nunique'), y_true=(TARGET_COL, 'mean'))
        .sort_values(['month_end', 'sleeve_id'])
        .reset_index(drop=True)
    )
    if not grouped['horizon_count'].eq(2).all():
        raise ValueError(f'truth panel for split={split_name} is not a full 60m/120m panel')
    grouped['split'] = split_name
    return grouped.drop(columns=['horizon_count'])[['split', 'month_end', 'sleeve_id', 'y_true']]


def _average_signal(predictions: pd.DataFrame, truth_panel: pd.DataFrame) -> pd.DataFrame:
    signal = (
        predictions.groupby(['split', 'month_end', 'sleeve_id'], as_index=False)
        .agg(horizon_count=('horizon_months', 'nunique'), y_pred=('y_pred', 'mean'))
        .sort_values(['split', 'month_end', 'sleeve_id'])
        .reset_index(drop=True)
    )
    if not signal['horizon_count'].eq(2).all():
        raise ValueError('average signal expected both 60m and 120m rows for every sleeve-month')
    out = signal.drop(columns=['horizon_count']).merge(
        truth_panel,
        on=['split', 'month_end', 'sleeve_id'],
        how='inner',
        validate='1:1',
    )
    return out.sort_values(['split', 'month_end', 'sleeve_id']).reset_index(drop=True)


def _single_horizon_signal(predictions: pd.DataFrame, *, horizon_months: int, truth_panel: pd.DataFrame) -> pd.DataFrame:
    signal = predictions.loc[predictions['horizon_months'].eq(horizon_months), ['split', 'month_end', 'sleeve_id', 'y_pred']].copy()
    out = signal.merge(truth_panel, on=['split', 'month_end', 'sleeve_id'], how='inner', validate='1:1')
    return out.sort_values(['split', 'month_end', 'sleeve_id']).reset_index(drop=True)


def _combine_signals(pred60: pd.DataFrame, pred120: pd.DataFrame, truth_panel: pd.DataFrame) -> pd.DataFrame:
    s60 = _single_horizon_signal(pred60, horizon_months=60, truth_panel=truth_panel).rename(columns={'y_pred': 'y_pred_60'})
    s120 = _single_horizon_signal(pred120, horizon_months=120, truth_panel=truth_panel).rename(columns={'y_pred': 'y_pred_120', 'y_true': 'y_true_120'})
    out = s60.merge(s120, on=['split', 'month_end', 'sleeve_id'], how='inner', validate='1:1')
    if not np.allclose(out['y_true'], out['y_true_120'], equal_nan=False):
        raise ValueError('combined signal truth mismatch across 60m and 120m panels')
    out['y_pred'] = 0.5 * (out['y_pred_60'] + out['y_pred_120'])
    return out[['split', 'month_end', 'sleeve_id', 'y_pred', 'y_true']].sort_values(['split', 'month_end', 'sleeve_id']).reset_index(drop=True)


def _extract_config_map(portfolio_metrics: pd.DataFrame) -> dict[str, OptimizerConfig]:
    subset = portfolio_metrics.loc[portfolio_metrics['source_type'].eq('common_allocator')].copy()
    out: dict[str, OptimizerConfig] = {}
    for strategy_label, chunk in subset.groupby('strategy_label'):
        if strategy_label == 'equal_weight':
            continue
        row = chunk.iloc[0]
        out[str(strategy_label)] = OptimizerConfig(
            lambda_risk=float(row['selected_lambda_risk']),
            kappa=float(row['selected_kappa']),
            omega_type=str(row['selected_omega_type']),
        )
    return out


def _prediction_dispersion(signal_panel: pd.DataFrame) -> float:
    monthly = signal_panel.groupby('month_end')['y_pred'].std(ddof=1)
    return float(monthly.mean()) if len(monthly) else float('nan')


def _mean_rank_ic(signal_panel: pd.DataFrame) -> float:
    vals: list[float] = []
    for _, chunk in signal_panel.groupby('month_end', sort=True):
        corr = chunk['y_pred'].corr(chunk['y_true'], method='spearman')
        if pd.notna(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else float('nan')


def _validate_returns(reconstructed: pd.DataFrame, reference: pd.DataFrame) -> None:
    comp = reconstructed.merge(
        reference[['split', 'month_end', 'strategy_label', 'portfolio_annualized_excess_return']],
        on=['split', 'month_end', 'strategy_label'],
        how='inner',
        validate='1:1',
        suffixes=('_recon', '_ref'),
    )
    if comp.empty:
        raise ValueError('No overlap between reconstructed and reference returns for diagnostics')
    max_diff = float(np.abs(comp['portfolio_annualized_excess_return_recon'] - comp['portfolio_annualized_excess_return_ref']).max())
    if max_diff > 1e-8:
        raise ValueError(f'Reconstructed returns do not match saved portfolio comparison returns; max_diff={max_diff:.3e}')


def build_portfolio_diagnostics(
    *,
    project_root: Path,
    predictions_panel: pd.DataFrame,
    selected_experiments: dict[str, str],
    portfolio_metrics_reference: pd.DataFrame,
    portfolio_returns_reference: pd.DataFrame,
) -> PortfolioDiagnostics:
    inputs = load_modeling_inputs(project_root, feature_set_name='core_baseline')
    truth_by_split = {
        'validation': _truth_panel(inputs.validation_df, split_name='validation'),
        'test': _truth_panel(inputs.test_df, split_name='test'),
    }
    sigma_map = build_sigma_map(
        sorted(pd.concat([truth_by_split['validation']['month_end'], truth_by_split['test']['month_end']]).drop_duplicates().tolist()),
        excess_history=inputs.monthly_excess_history,
        risk_config=RiskConfig(),
    )
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)
    config_map = _extract_config_map(portfolio_metrics_reference)

    signal_panels: dict[str, dict[str, pd.DataFrame]] = {}
    for split_name in ('validation', 'test'):
        truth = truth_by_split[split_name]
        split_preds = predictions_panel.loc[predictions_panel['split'].eq(split_name)].copy()
        best60 = split_preds.loc[split_preds['experiment_name'].eq(selected_experiments['best_60_predictor'])].copy()
        best120 = split_preds.loc[split_preds['experiment_name'].eq(selected_experiments['best_120_predictor'])].copy()
        shared = split_preds.loc[split_preds['experiment_name'].eq(selected_experiments['best_shared_predictor'])].copy()
        pto = split_preds.loc[split_preds['experiment_name'].eq(selected_experiments['pto_nn_signal'])].copy()
        e2e = split_preds.loc[split_preds['experiment_name'].eq(selected_experiments['e2e_nn_signal'])].copy()
        signal_panels[split_name] = {
            'best_60_predictor': _single_horizon_signal(best60, horizon_months=60, truth_panel=truth),
            'best_120_predictor': _single_horizon_signal(best120, horizon_months=120, truth_panel=truth),
            'combined_60_120_predictor': _combine_signals(best60, best120, truth),
            'best_shared_predictor': _average_signal(shared, truth),
            'pto_nn_signal': _average_signal(pto, truth),
            'e2e_nn_signal': _average_signal(e2e, truth),
        }

    returns_frames: list[pd.DataFrame] = []
    weights_frames: list[pd.DataFrame] = []
    equal_weight_seen_splits: set[str] = set()
    strategy_monthly_rows: list[dict[str, object]] = []

    for strategy_label in _DIAGNOSTIC_STRATEGIES:
        if strategy_label == 'equal_weight':
            continue
        config = config_map[strategy_label]
        for split_name in ('validation', 'test'):
            signal_panel = signal_panels[split_name][strategy_label]
            run = run_portfolio_evaluation(
                signal_panel=signal_panel,
                optimizer_cache=optimizer_cache,
                optimizer_config=config,
                model_strategy_name=strategy_label,
            )
            returns_df = run.returns.copy()
            weights_df = run.weights.copy()
            returns_df['strategy_label'] = returns_df['strategy']
            weights_df['strategy_label'] = weights_df['strategy']
            if split_name in equal_weight_seen_splits:
                returns_df = returns_df.loc[returns_df['strategy'].ne('equal_weight')].copy()
                weights_df = weights_df.loc[weights_df['strategy'].ne('equal_weight')].copy()
            else:
                equal_weight_seen_splits.add(split_name)
            returns_frames.append(returns_df)
            weights_frames.append(weights_df)

            strat_returns = returns_df.loc[returns_df['strategy_label'].eq(strategy_label)].copy()
            strat_weights = weights_df.loc[weights_df['strategy_label'].eq(strategy_label)].copy()
            signal_chunk = signal_panel.copy()
            monthly_stats = (
                strat_weights.groupby('month_end', as_index=False)['weight']
                .agg(max_weight='max', hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
                .sort_values('month_end')
            )
            monthly_stats['effective_n_sleeves'] = 1.0 / monthly_stats['hhi']
            for row in monthly_stats.itertuples(index=False):
                strategy_monthly_rows.append(
                    {
                        'split': split_name,
                        'strategy_label': strategy_label,
                        'month_end': pd.Timestamp(row.month_end),
                        'avg_prediction_dispersion': _prediction_dispersion(signal_chunk.loc[signal_chunk['month_end'].eq(pd.Timestamp(row.month_end))]),
                        'rank_ic_spearman': _mean_rank_ic(signal_chunk.loc[signal_chunk['month_end'].eq(pd.Timestamp(row.month_end))]),
                        'max_weight': float(row.max_weight),
                        'hhi': float(row.hhi),
                        'effective_n_sleeves': float(row.effective_n_sleeves),
                    }
                )

    returns_panel = pd.concat(returns_frames, ignore_index=True).sort_values(['strategy_label', 'split', 'month_end']).reset_index(drop=True)
    weights_panel = pd.concat(weights_frames, ignore_index=True).sort_values(['strategy_label', 'split', 'month_end', 'sleeve_id']).reset_index(drop=True)
    returns_panel['month_end'] = pd.to_datetime(returns_panel['month_end'])
    weights_panel['month_end'] = pd.to_datetime(weights_panel['month_end'])

    _validate_returns(
        returns_panel.loc[returns_panel['strategy_label'].isin(_DIAGNOSTIC_STRATEGIES)],
        portfolio_returns_reference.loc[portfolio_returns_reference['strategy_label'].isin(_DIAGNOSTIC_STRATEGIES)],
    )

    equal_weight_weights = weights_panel.loc[weights_panel['strategy_label'].eq('equal_weight'), ['split', 'month_end', 'sleeve_id', 'weight']].rename(columns={'weight': 'equal_weight_ref'})
    weights_panel = weights_panel.merge(equal_weight_weights, on=['split', 'month_end', 'sleeve_id'], how='left', validate='m:1')
    weights_panel['active_weight_vs_equal_weight'] = weights_panel['weight'] - weights_panel['equal_weight_ref']

    truth_full = pd.concat(list(truth_by_split.values()), ignore_index=True)
    weights_panel = weights_panel.merge(truth_full, on=['split', 'month_end', 'sleeve_id'], how='left', validate='m:1')
    weights_panel['sleeve_contribution'] = weights_panel['weight'] * weights_panel['y_true']
    weights_panel['active_contribution_vs_equal_weight'] = weights_panel['active_weight_vs_equal_weight'] * weights_panel['y_true']
    weights_panel['top_weight_flag'] = (
        weights_panel.groupby(['split', 'strategy_label', 'month_end'])['weight'].rank(method='first', ascending=False).eq(1)
    ).astype(int)

    signal_rows = []
    for split_name, strategy_map in signal_panels.items():
        for strategy_label, signal_panel in strategy_map.items():
            chunk = signal_panel.copy()
            chunk['strategy_label'] = strategy_label
            signal_rows.append(chunk)
    signals_df = pd.concat(signal_rows, ignore_index=True)
    weights_panel = weights_panel.merge(
        signals_df[['split', 'month_end', 'strategy_label', 'sleeve_id', 'y_pred']].rename(columns={'y_pred': 'predicted_signal'}),
        on=['split', 'month_end', 'strategy_label', 'sleeve_id'],
        how='left',
        validate='m:1',
    )
    weights_panel = weights_panel.rename(columns={'y_true': 'realized_outcome'})

    concentration_all = (
        weights_panel.groupby(['split', 'strategy_label', 'month_end'], as_index=False)['weight']
        .agg(max_weight='max', hhi=lambda x: float(np.square(np.asarray(x, dtype=float)).sum()))
        .sort_values(['strategy_label', 'split', 'month_end'])
        .reset_index(drop=True)
    )
    concentration_all['effective_n_sleeves'] = 1.0 / concentration_all['hhi']

    monthly_stats_df = pd.DataFrame(strategy_monthly_rows)
    returns_panel = returns_panel.merge(monthly_stats_df, on=['split', 'strategy_label', 'month_end'], how='left', validate='1:1')
    returns_panel = returns_panel.drop(columns=['max_weight', 'hhi', 'effective_n_sleeves'], errors='ignore').merge(
        concentration_all,
        on=['split', 'strategy_label', 'month_end'],
        how='left',
        validate='1:1',
    )

    ref_map = {
        'equal_weight': 'equal_weight',
        'pto_nn_signal': 'pto_nn_signal',
        'e2e_nn_signal': 'e2e_nn_signal',
    }
    for label, ref_strategy in ref_map.items():
        ref = returns_panel.loc[returns_panel['strategy_label'].eq(ref_strategy), ['split', 'month_end', 'portfolio_annualized_excess_return']].rename(columns={'portfolio_annualized_excess_return': f'ref_{label}'})
        returns_panel = returns_panel.merge(ref, on=['split', 'month_end'], how='left', validate='m:1')
    returns_panel['active_return_vs_equal_weight'] = returns_panel['portfolio_annualized_excess_return'] - returns_panel['ref_equal_weight']
    returns_panel['active_return_vs_pto_nn_signal'] = returns_panel['portfolio_annualized_excess_return'] - returns_panel['ref_pto_nn_signal']
    returns_panel['active_return_vs_e2e_nn_signal'] = returns_panel['portfolio_annualized_excess_return'] - returns_panel['ref_e2e_nn_signal']

    summary_rows: list[dict[str, object]] = []
    for (strategy_label, split_name), chunk in returns_panel.groupby(['strategy_label', 'split'], as_index=False):
        weights_chunk = weights_panel.loc[
            weights_panel['strategy_label'].eq(strategy_label) & weights_panel['split'].eq(split_name)
        ]
        top_sleeve_freq = (
            weights_chunk.loc[weights_chunk['top_weight_flag'].eq(1)]
            .groupby('sleeve_id', as_index=False)
            .size()
            .sort_values(['size', 'sleeve_id'], ascending=[False, True])
        )
        top_sleeve = None
        top_sleeve_share = float('nan')
        if not top_sleeve_freq.empty:
            top_sleeve = str(top_sleeve_freq.iloc[0]['sleeve_id'])
            top_sleeve_share = float(top_sleeve_freq.iloc[0]['size'] / chunk['month_end'].nunique())
        active_positive = chunk.loc[chunk['active_return_vs_equal_weight'] > 0.0, 'active_return_vs_equal_weight'].sort_values(ascending=False)
        positive_share = float(active_positive.head(5).sum() / active_positive.sum()) if active_positive.sum() > 0 else float('nan')
        sleeve_active = (
            weights_chunk.groupby('sleeve_id', as_index=False)['active_contribution_vs_equal_weight']
            .sum()
            .assign(abs_active=lambda x: np.abs(x['active_contribution_vs_equal_weight']))
            .sort_values('abs_active', ascending=False)
        )
        top2_share = float(sleeve_active.head(2)['abs_active'].sum() / sleeve_active['abs_active'].sum()) if sleeve_active['abs_active'].sum() > 0 else float('nan')
        summary_rows.append(
            {
                'strategy_label': strategy_label,
                'split': split_name,
                'month_count': int(chunk['month_end'].nunique()),
                'avg_return': float(chunk['portfolio_annualized_excess_return'].mean()),
                'volatility': float(chunk['portfolio_annualized_excess_return'].std(ddof=1)),
                'sharpe': float(chunk['portfolio_annualized_excess_return'].mean() / chunk['portfolio_annualized_excess_return'].std(ddof=1)) if float(chunk['portfolio_annualized_excess_return'].std(ddof=1)) > 0 else float('nan'),
                'max_drawdown': float(chunk['drawdown'].min()),
                'avg_turnover': float(chunk['turnover'].mean()),
                'avg_max_weight': float(chunk['max_weight'].mean()),
                'avg_hhi': float(chunk['hhi'].mean()),
                'avg_effective_n_sleeves': float(chunk['effective_n_sleeves'].mean()),
                'avg_prediction_dispersion': float(chunk['avg_prediction_dispersion'].mean()),
                'avg_rank_ic_spearman': float(chunk['rank_ic_spearman'].mean()),
                'avg_active_return_vs_equal_weight': float(chunk['active_return_vs_equal_weight'].mean()),
                'avg_active_return_vs_pto_nn_signal': float(chunk['active_return_vs_pto_nn_signal'].mean()),
                'avg_active_return_vs_e2e_nn_signal': float(chunk['active_return_vs_e2e_nn_signal'].mean()),
                'top5_positive_active_month_share': positive_share,
                'top2_sleeve_active_share_abs': top2_share,
                'top_weight_sleeve': top_sleeve,
                'top_weight_sleeve_frequency': top_sleeve_share,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(['split', 'sharpe'], ascending=[True, False]).reset_index(drop=True)

    sleeve_rows: list[dict[str, object]] = []
    for (strategy_label, split_name, sleeve_id), chunk in weights_panel.groupby(['strategy_label', 'split', 'sleeve_id'], as_index=False):
        total_active = float(chunk['active_contribution_vs_equal_weight'].sum())
        sleeve_rows.append(
            {
                'strategy_label': strategy_label,
                'split': split_name,
                'sleeve_id': sleeve_id,
                'month_count': int(chunk['month_end'].nunique()),
                'avg_weight': float(chunk['weight'].mean()),
                'avg_active_weight_vs_equal_weight': float(chunk['active_weight_vs_equal_weight'].mean()),
                'top_weight_frequency': float(chunk['top_weight_flag'].mean()),
                'avg_predicted_signal': float(chunk['predicted_signal'].mean()),
                'avg_realized_outcome': float(chunk['realized_outcome'].mean()),
                'total_contribution': float(chunk['sleeve_contribution'].sum()),
                'avg_monthly_contribution': float(chunk['sleeve_contribution'].mean()),
                'total_active_contribution_vs_equal_weight': total_active,
                'avg_monthly_active_contribution_vs_equal_weight': float(chunk['active_contribution_vs_equal_weight'].mean()),
                'abs_total_active_contribution': abs(total_active),
            }
        )
    sleeve_summary = pd.DataFrame(sleeve_rows)
    for (strategy_label, split_name), idx in sleeve_summary.groupby(['strategy_label', 'split']).groups.items():
        denom = float(sleeve_summary.loc[idx, 'abs_total_active_contribution'].sum())
        if denom > 0:
            sleeve_summary.loc[idx, 'abs_active_contribution_share'] = sleeve_summary.loc[idx, 'abs_total_active_contribution'] / denom
        else:
            sleeve_summary.loc[idx, 'abs_active_contribution_share'] = np.nan
    sleeve_summary = sleeve_summary.sort_values(['strategy_label', 'split', 'abs_total_active_contribution'], ascending=[True, True, False]).reset_index(drop=True)

    month_rows: list[dict[str, object]] = []
    for (strategy_label, split_name), chunk in returns_panel.groupby(['strategy_label', 'split'], as_index=False):
        best_month = chunk.sort_values('portfolio_annualized_excess_return', ascending=False).iloc[0]
        worst_month = chunk.sort_values('portfolio_annualized_excess_return', ascending=True).iloc[0]
        month_rows.extend(
            [
                {
                    'strategy_label': strategy_label,
                    'split': split_name,
                    'record_type': 'best_month',
                    'month_end': best_month['month_end'],
                    'portfolio_annualized_excess_return': float(best_month['portfolio_annualized_excess_return']),
                    'active_return_vs_equal_weight': float(best_month['active_return_vs_equal_weight']),
                },
                {
                    'strategy_label': strategy_label,
                    'split': split_name,
                    'record_type': 'worst_month',
                    'month_end': worst_month['month_end'],
                    'portfolio_annualized_excess_return': float(worst_month['portfolio_annualized_excess_return']),
                    'active_return_vs_equal_weight': float(worst_month['active_return_vs_equal_weight']),
                },
            ]
        )
    best_worst = pd.DataFrame(month_rows).sort_values(['strategy_label', 'split', 'record_type']).reset_index(drop=True)

    return PortfolioDiagnostics(
        weights_panel=weights_panel,
        returns_panel=returns_panel,
        summary=summary,
        sleeve_attribution_summary=sleeve_summary,
        best_worst_months=best_worst,
    )
