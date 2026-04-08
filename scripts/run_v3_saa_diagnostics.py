#!/usr/bin/env python3
"""Run the compact v3 long-horizon SAA diagnostics and plot pack."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v3_diagnostics.io import default_paths, load_diagnostic_inputs, write_csv, write_parquet, write_text  # noqa: E402
from xoptpoe_v3_diagnostics.plots import (  # noqa: E402
    active_return_vs_equal,
    portfolio_cumulative,
    prediction_scatter,
    prediction_scatter_by_sleeve,
    rank_ic_over_time,
    score_heatmap,
    sign_accuracy_over_time,
    sleeve_contribution_bars,
    turnover_concentration,
    weights_heatmap,
    weights_stacked,
)
from xoptpoe_v3_diagnostics.portfolio import build_portfolio_diagnostics  # noqa: E402
from xoptpoe_v3_diagnostics.prediction import build_prediction_diagnostics  # noqa: E402
from xoptpoe_v3_diagnostics.report import build_report  # noqa: E402


def build_china_prediction_summary(horse_race_by_sleeve: pd.DataFrame, selected_experiments: dict[str, str]) -> pd.DataFrame:
    mapping = {
        'best_60_predictor': selected_experiments['best_60_predictor'],
        'best_120_predictor': selected_experiments['best_120_predictor'],
        'best_shared_predictor': selected_experiments['best_shared_predictor'],
        'pto_nn_signal': selected_experiments['pto_nn_signal'],
        'e2e_nn_signal': selected_experiments['e2e_nn_signal'],
    }
    subset = horse_race_by_sleeve.loc[
        horse_race_by_sleeve['experiment_name'].isin(mapping.values())
        & horse_race_by_sleeve['sleeve_id'].eq('EQ_CN')
    ].copy()
    reverse = {value: key for key, value in mapping.items()}
    subset['strategy_label'] = subset['experiment_name'].map(reverse)
    subset['rmse_rank_worst_first'] = subset.groupby(['experiment_name', 'split'])['rmse'].rank(
        ascending=False,
        method='min',
    )
    subset['corr_rank_worst_first'] = subset.groupby(['experiment_name', 'split'])['corr'].rank(
        ascending=True,
        method='min',
    )
    return subset.sort_values(['strategy_label', 'split']).reset_index(drop=True)


def build_china_portfolio_summary(weights_panel: pd.DataFrame, sleeve_attribution_summary: pd.DataFrame) -> pd.DataFrame:
    china_weights = (
        weights_panel.loc[weights_panel['sleeve_id'].eq('EQ_CN')]
        .groupby(['strategy_label', 'split'], as_index=False)
        .agg(
            avg_weight=('weight', 'mean'),
            max_weight=('weight', 'max'),
            avg_active_weight_vs_equal_weight=('active_weight_vs_equal_weight', 'mean'),
            avg_predicted_signal=('predicted_signal', 'mean'),
            avg_realized_outcome=('realized_outcome', 'mean'),
            top_weight_frequency=('top_weight_flag', 'mean'),
        )
    )
    china_attr = sleeve_attribution_summary.loc[
        sleeve_attribution_summary['sleeve_id'].eq('EQ_CN')
    ][
        [
            'strategy_label',
            'split',
            'total_contribution',
            'avg_monthly_contribution',
            'total_active_contribution_vs_equal_weight',
            'avg_monthly_active_contribution_vs_equal_weight',
            'abs_active_contribution_share',
        ]
    ].copy()
    return china_weights.merge(china_attr, on=['strategy_label', 'split'], how='left', validate='1:1')


def build_integrity_checks(
    *,
    prediction_panel: pd.DataFrame,
    weights_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    horse_race_by_sleeve: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({'check_name': name, 'status': 'PASS' if passed else 'FAIL', 'detail': detail})

    pred_sleeves = sorted(prediction_panel['sleeve_id'].dropna().unique().tolist())
    weight_sleeves = sorted(weights_panel['sleeve_id'].dropna().unique().tolist())
    horse_sleeves = sorted(horse_race_by_sleeve['sleeve_id'].dropna().unique().tolist())
    add('eq_cn_in_prediction_panel', 'EQ_CN' in pred_sleeves, f"sleeves={pred_sleeves}")
    add('eq_cn_in_weights_panel', 'EQ_CN' in weight_sleeves, f"sleeves={weight_sleeves}")
    add('eq_cn_in_by_sleeve_metrics', 'EQ_CN' in horse_sleeves, f"sleeves={horse_sleeves}")
    add(
        'no_duplicate_prediction_keys',
        not prediction_panel.duplicated(subset=['strategy_label', 'split', 'month_end', 'horizon_months', 'sleeve_id']).any(),
        f"duplicate_count={int(prediction_panel.duplicated(subset=['strategy_label', 'split', 'month_end', 'horizon_months', 'sleeve_id']).sum())}",
    )
    add(
        'no_duplicate_weight_keys',
        not weights_panel.duplicated(subset=['strategy_label', 'split', 'month_end', 'sleeve_id']).any(),
        f"duplicate_count={int(weights_panel.duplicated(subset=['strategy_label', 'split', 'month_end', 'sleeve_id']).sum())}",
    )
    add(
        'nine_sleeves_in_test_weights',
        int(weights_panel.loc[weights_panel['split'].eq('test'), 'sleeve_id'].nunique()) == 9,
        f"test_sleeve_count={int(weights_panel.loc[weights_panel['split'].eq('test'), 'sleeve_id'].nunique())}",
    )
    add(
        'portfolio_returns_have_eq_cn_months',
        returns_panel['month_end'].nunique() > 0,
        f"month_count={int(returns_panel['month_end'].nunique())}",
    )
    return pd.DataFrame(checks)


def main() -> int:
    paths = default_paths(PROJECT_ROOT)
    inputs = load_diagnostic_inputs(PROJECT_ROOT)

    prediction = build_prediction_diagnostics(
        predictions_validation=inputs['predictions_validation'],
        predictions_test=inputs['predictions_test'],
        metrics_df=inputs['horse_race_metrics'],
    )
    portfolio = build_portfolio_diagnostics(
        project_root=PROJECT_ROOT,
        predictions_panel=pd.concat([inputs['predictions_validation'], inputs['predictions_test']], ignore_index=True),
        selected_experiments=prediction.selected_experiments,
        portfolio_metrics_reference=inputs['portfolio_comparison_metrics'],
        portfolio_returns_reference=inputs['portfolio_comparison_returns'],
    )

    report_text = build_report(
        prediction_summary=prediction.summary,
        portfolio_summary=portfolio.summary,
        sleeve_attribution_summary=portfolio.sleeve_attribution_summary,
        best_worst_months=portfolio.best_worst_months,
        selected_experiments=prediction.selected_experiments,
    )

    write_parquet(prediction.diagnostic_panel, paths.data_out_dir / 'diagnostic_predictions_panel.parquet')
    write_parquet(portfolio.weights_panel, paths.data_out_dir / 'diagnostic_weights_panel.parquet')
    write_parquet(portfolio.returns_panel, paths.data_out_dir / 'diagnostic_returns_panel.parquet')

    write_csv(prediction.summary, paths.reports_dir / 'prediction_diagnostics_summary.csv')
    write_csv(portfolio.summary, paths.reports_dir / 'portfolio_diagnostics_summary.csv')
    write_csv(portfolio.sleeve_attribution_summary, paths.reports_dir / 'sleeve_attribution_summary.csv')
    write_csv(prediction.top_predicted_frequency, paths.reports_dir / 'top_predicted_sleeve_frequency.csv')
    write_csv(prediction.monthly_summary, paths.reports_dir / 'prediction_time_series_summary.csv')
    write_csv(portfolio.best_worst_months, paths.reports_dir / 'best_worst_months.csv')
    write_csv(
        build_china_prediction_summary(inputs['horse_race_by_sleeve'], prediction.selected_experiments),
        paths.reports_dir / 'china_sleeve_prediction_summary.csv',
    )
    write_csv(
        build_china_portfolio_summary(portfolio.weights_panel, portfolio.sleeve_attribution_summary),
        paths.reports_dir / 'china_sleeve_portfolio_summary.csv',
    )
    write_csv(
        build_integrity_checks(
            prediction_panel=prediction.diagnostic_panel,
            weights_panel=portfolio.weights_panel,
            returns_panel=portfolio.returns_panel,
            horse_race_by_sleeve=inputs['horse_race_by_sleeve'],
        ),
        paths.reports_dir / 'downstream_integrity_checks.csv',
    )
    write_text(report_text, paths.reports_dir / 'saa_diagnostics_report.md')

    best60_test = prediction.diagnostic_panel.loc[
        prediction.diagnostic_panel['strategy_label'].eq('best_60_predictor')
        & prediction.diagnostic_panel['split'].eq('test')
        & prediction.diagnostic_panel['horizon_months'].eq(60)
    ].copy()
    best120_test = prediction.diagnostic_panel.loc[
        prediction.diagnostic_panel['strategy_label'].eq('best_120_predictor')
        & prediction.diagnostic_panel['split'].eq('test')
        & prediction.diagnostic_panel['horizon_months'].eq(120)
    ].copy()

    prediction_scatter(best60_test, paths.plots_dir / 'pred_vs_realized_60m.png', 'Best 60m Predictor: Test Predicted vs Realized')
    prediction_scatter(best120_test, paths.plots_dir / 'pred_vs_realized_120m.png', 'Best 120m Predictor: Test Predicted vs Realized')
    prediction_scatter_by_sleeve(best60_test, paths.plots_dir / 'pred_vs_realized_by_sleeve_best60.png', 'Best 60m Predictor by Sleeve')
    prediction_scatter_by_sleeve(best120_test, paths.plots_dir / 'pred_vs_realized_by_sleeve_best120.png', 'Best 120m Predictor by Sleeve')
    rank_ic_over_time(prediction.monthly_summary, paths.plots_dir / 'rank_ic_over_time.png')
    sign_accuracy_over_time(prediction.monthly_summary, paths.plots_dir / 'sign_accuracy_over_time.png')
    score_heatmap(best60_test, paths.plots_dir / 'predicted_score_heatmap_best60.png', 'Best 60m Predictor Scores (Test)', value_col='y_pred')
    score_heatmap(best120_test, paths.plots_dir / 'predicted_score_heatmap_best120.png', 'Best 120m Predictor Scores (Test)', value_col='y_pred')
    score_heatmap(best60_test, paths.plots_dir / 'realized_outcome_heatmap_60m.png', 'Realized 60m Outcomes (Test)', value_col='y_true')

    weights_stacked(portfolio.weights_panel, 'best_60_predictor', paths.plots_dir / 'weights_stacked_best60.png', 'Best 60m Signal Portfolio Weights (Test)')
    weights_stacked(portfolio.weights_panel, 'combined_60_120_predictor', paths.plots_dir / 'weights_stacked_combined.png', 'Combined 60m/120m Signal Portfolio Weights (Test)')
    weights_heatmap(portfolio.weights_panel, 'best_60_predictor', paths.plots_dir / 'weights_heatmap_best60.png', 'Best 60m Signal Weights Heatmap (Test)')
    turnover_concentration(portfolio.returns_panel, paths.plots_dir / 'turnover_concentration.png')
    portfolio_cumulative(portfolio.returns_panel, paths.plots_dir / 'portfolio_comparison_cumulative.png')
    active_return_vs_equal(portfolio.returns_panel, paths.plots_dir / 'active_return_vs_equal_weight.png')
    sleeve_contribution_bars(portfolio.sleeve_attribution_summary, paths.plots_dir / 'sleeve_contribution_bars.png')
    return 0


if __name__ == '__main__':
    import pandas as pd
    raise SystemExit(main())
