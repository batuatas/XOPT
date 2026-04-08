"""Matplotlib plot helpers for v3 SAA diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from xoptpoe_v3_models.data import SLEEVE_ORDER


PLOT_STRATEGY_ORDER = [
    'equal_weight',
    'best_60_predictor',
    'best_120_predictor',
    'combined_60_120_predictor',
    'best_shared_predictor',
    'pto_nn_signal',
    'e2e_nn_signal',
]
PLOT_COLORS = {
    'equal_weight': '#333333',
    'best_60_predictor': '#0f766e',
    'best_120_predictor': '#1d4ed8',
    'combined_60_120_predictor': '#b45309',
    'best_shared_predictor': '#7c3aed',
    'pto_nn_signal': '#dc2626',
    'e2e_nn_signal': '#16a34a',
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _heatmap(fig_ax, matrix: pd.DataFrame, title: str, cmap: str = 'coolwarm') -> None:
    ax = fig_ax
    if matrix.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return
    values = matrix.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(values, aspect='auto', cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    ax.set_title(title)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([pd.Timestamp(idx).strftime('%Y-%m') for idx in matrix.index], fontsize=6)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(list(matrix.columns), rotation=45, ha='right', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.7)


def prediction_scatter(panel: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for sleeve_id, chunk in panel.groupby('sleeve_id'):
        ax.scatter(chunk['y_true'], chunk['y_pred'], s=20, alpha=0.7, label=sleeve_id)
    all_vals = pd.concat([panel['y_true'], panel['y_pred']], ignore_index=True)
    lo = float(all_vals.min())
    hi = float(all_vals.max())
    ax.plot([lo, hi], [lo, hi], linestyle='--', color='black', linewidth=1.0)
    ax.set_xlabel('Realized annualized excess return')
    ax.set_ylabel('Predicted annualized excess return')
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    _save(fig, path)


def prediction_scatter_by_sleeve(panel: pd.DataFrame, path: Path, title: str) -> None:
    sleeve_count = len(SLEEVE_ORDER)
    ncols = 3
    nrows = int(np.ceil(sleeve_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, sleeve_id in zip(axes, SLEEVE_ORDER):
        chunk = panel.loc[panel['sleeve_id'].eq(sleeve_id)]
        ax.scatter(chunk['y_true'], chunk['y_pred'], s=18, alpha=0.75, color='#2563eb')
        if not chunk.empty:
            all_vals = pd.concat([chunk['y_true'], chunk['y_pred']], ignore_index=True)
            lo = float(all_vals.min())
            hi = float(all_vals.max())
            ax.plot([lo, hi], [lo, hi], linestyle='--', color='black', linewidth=0.8)
        ax.set_title(sleeve_id)
    for ax in axes[sleeve_count:]:
        ax.axis('off')
    fig.suptitle(title, y=1.02)
    _save(fig, path)


def rank_ic_over_time(monthly_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    test = monthly_summary.loc[monthly_summary['split'].eq('test')].copy()
    shared_like = {'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal'}
    for strategy_label in ['best_60_predictor', 'best_120_predictor', 'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal']:
        chunk = test.loc[test['strategy_label'].eq(strategy_label)].copy()
        if strategy_label in shared_like:
            chunk = chunk.groupby('month_end', as_index=False)['rank_ic_spearman'].mean()
        else:
            chunk = chunk[['month_end', 'rank_ic_spearman']].copy()
        chunk = chunk.sort_values('month_end')
        ax.plot(chunk['month_end'], chunk['rank_ic_spearman'], label=strategy_label, color=PLOT_COLORS.get(strategy_label), linewidth=1.8)
    ax.axhline(0.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title('Test-Split Rank IC Over Time')
    ax.set_ylabel('Spearman rank IC')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def sign_accuracy_over_time(monthly_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    test = monthly_summary.loc[monthly_summary['split'].eq('test')].copy()
    shared_like = {'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal'}
    for strategy_label in ['best_60_predictor', 'best_120_predictor', 'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal']:
        chunk = test.loc[test['strategy_label'].eq(strategy_label)].copy()
        if strategy_label in shared_like:
            chunk = chunk.groupby('month_end', as_index=False)['sign_accuracy'].mean()
        else:
            chunk = chunk[['month_end', 'sign_accuracy']].copy()
        chunk = chunk.sort_values('month_end')
        ax.plot(chunk['month_end'], chunk['sign_accuracy'], label=strategy_label, color=PLOT_COLORS.get(strategy_label), linewidth=1.8)
    ax.set_title('Test-Split Sign Accuracy Over Time')
    ax.set_ylabel('Cross-sleeve sign accuracy')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def score_heatmap(panel: pd.DataFrame, path: Path, title: str, value_col: str = 'y_pred') -> None:
    matrix = panel.pivot(index='month_end', columns='sleeve_id', values=value_col).reindex(columns=list(SLEEVE_ORDER))
    fig, ax = plt.subplots(figsize=(8, 8))
    _heatmap(ax, matrix, title)
    _save(fig, path)


def weights_stacked(weights_panel: pd.DataFrame, strategy_label: str, path: Path, title: str) -> None:
    test = weights_panel.loc[
        weights_panel['split'].eq('test') & weights_panel['strategy_label'].eq(strategy_label)
    ].copy()
    matrix = test.pivot(index='month_end', columns='sleeve_id', values='weight').reindex(columns=list(SLEEVE_ORDER))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.stackplot(matrix.index, [matrix[col].to_numpy(dtype=float) for col in matrix.columns], labels=matrix.columns)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_ylabel('Weight')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    _save(fig, path)


def weights_heatmap(weights_panel: pd.DataFrame, strategy_label: str, path: Path, title: str) -> None:
    test = weights_panel.loc[
        weights_panel['split'].eq('test') & weights_panel['strategy_label'].eq(strategy_label)
    ].copy()
    matrix = test.pivot(index='month_end', columns='sleeve_id', values='weight').reindex(columns=list(SLEEVE_ORDER))
    fig, ax = plt.subplots(figsize=(8, 8))
    if matrix.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
    else:
        im = ax.imshow(matrix.to_numpy(dtype=float), aspect='auto', cmap='viridis', vmin=0.0, vmax=float(matrix.max().max()))
        ax.set_title(title)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([pd.Timestamp(idx).strftime('%Y-%m') for idx in matrix.index], fontsize=6)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(list(matrix.columns), rotation=45, ha='right', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)
    _save(fig, path)


def turnover_concentration(returns_panel: pd.DataFrame, path: Path) -> None:
    test = returns_panel.loc[
        returns_panel['split'].eq('test')
        & returns_panel['strategy_label'].isin(['best_60_predictor', 'best_120_predictor', 'combined_60_120_predictor', 'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal'])
    ].copy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for strategy_label in test['strategy_label'].drop_duplicates().tolist():
        chunk = test.loc[test['strategy_label'].eq(strategy_label)].sort_values('month_end')
        axes[0].plot(chunk['month_end'], chunk['turnover'], label=strategy_label, color=PLOT_COLORS.get(strategy_label), linewidth=1.7)
        axes[1].plot(chunk['month_end'], chunk['max_weight'], label=strategy_label, color=PLOT_COLORS.get(strategy_label), linewidth=1.7)
        axes[2].plot(chunk['month_end'], chunk['hhi'], label=strategy_label, color=PLOT_COLORS.get(strategy_label), linewidth=1.7)
    axes[0].set_title('Test-Split Turnover and Concentration')
    axes[0].set_ylabel('Turnover')
    axes[1].set_ylabel('Max weight')
    axes[2].set_ylabel('HHI')
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def portfolio_cumulative(returns_panel: pd.DataFrame, path: Path) -> None:
    test = returns_panel.loc[returns_panel['split'].eq('test')].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for strategy_label in PLOT_STRATEGY_ORDER:
        chunk = test.loc[test['strategy_label'].eq(strategy_label)].sort_values('month_end')
        if chunk.empty:
            continue
        ax.plot(chunk['month_end'], chunk['cum_nav'], label=strategy_label, color=PLOT_COLORS.get(strategy_label), linewidth=1.8)
    ax.set_title('Cumulative Decision-Period Comparison (Test Split)')
    ax.set_ylabel('Cumulated product of 1 + decision-period label')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def active_return_vs_equal(returns_panel: pd.DataFrame, path: Path) -> None:
    test = returns_panel.loc[
        returns_panel['split'].eq('test') & returns_panel['strategy_label'].ne('equal_weight')
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for strategy_label in ['best_60_predictor', 'best_120_predictor', 'combined_60_120_predictor', 'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal']:
        chunk = test.loc[test['strategy_label'].eq(strategy_label)].sort_values('month_end').copy()
        if chunk.empty:
            continue
        chunk['cum_active_vs_eq'] = chunk['active_return_vs_equal_weight'].cumsum()
        ax.plot(chunk['month_end'], chunk['cum_active_vs_eq'], label=strategy_label, color=PLOT_COLORS.get(strategy_label), linewidth=1.8)
    ax.axhline(0.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title('Cumulative Active Return vs Equal Weight (Test Split)')
    ax.set_ylabel('Cumulative active return')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def sleeve_contribution_bars(sleeve_summary: pd.DataFrame, path: Path) -> None:
    test = sleeve_summary.loc[
        sleeve_summary['split'].eq('test')
        & sleeve_summary['strategy_label'].isin(['best_60_predictor', 'best_120_predictor', 'combined_60_120_predictor', 'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal'])
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sleeves = list(SLEEVE_ORDER)
    strategies = ['best_60_predictor', 'best_120_predictor', 'combined_60_120_predictor', 'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal']
    width = 0.12
    x = np.arange(len(sleeves))
    for idx, strategy_label in enumerate(strategies):
        chunk = test.loc[test['strategy_label'].eq(strategy_label)].set_index('sleeve_id').reindex(sleeves)
        values = chunk['total_active_contribution_vs_equal_weight'].to_numpy(dtype=float)
        ax.bar(x + (idx - (len(strategies)-1)/2)*width, values, width=width, label=strategy_label, color=PLOT_COLORS.get(strategy_label))
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sleeves)
    ax.set_title('Total Active Contribution vs Equal Weight by Sleeve (Test Split)')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)
