"""I/O helpers for the v3 SAA diagnostics package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xoptpoe_v3_modeling.io import load_csv, load_parquet, write_csv, write_parquet, write_text


@dataclass(frozen=True)
class DiagnosticPaths:
    project_root: Path
    predictions_validation: Path
    predictions_test: Path
    portfolio_comparison_metrics: Path
    portfolio_comparison_returns: Path
    horse_race_metrics: Path
    horse_race_by_sleeve: Path
    feature_set_summary: Path
    modeling_panel_firstpass: Path
    data_out_dir: Path
    reports_dir: Path
    plots_dir: Path


def default_paths(project_root: Path) -> DiagnosticPaths:
    root = project_root.resolve()
    reports_dir = root / 'reports' / 'v3_long_horizon_china'
    return DiagnosticPaths(
        project_root=root,
        predictions_validation=root / 'data' / 'modeling_v3' / 'predictions_validation_horse_race.parquet',
        predictions_test=root / 'data' / 'modeling_v3' / 'predictions_test_horse_race.parquet',
        portfolio_comparison_metrics=reports_dir / 'saa_portfolio_comparison_metrics.csv',
        portfolio_comparison_returns=root / 'data' / 'modeling_v3' / 'portfolio_comparison_returns.csv',
        horse_race_metrics=reports_dir / 'predictor_horse_race_metrics.csv',
        horse_race_by_sleeve=reports_dir / 'predictor_horse_race_by_sleeve.csv',
        feature_set_summary=reports_dir / 'predictor_feature_set_summary.csv',
        modeling_panel_firstpass=root / 'data' / 'modeling_v3' / 'modeling_panel_firstpass.parquet',
        data_out_dir=root / 'data' / 'modeling_v3',
        reports_dir=reports_dir,
        plots_dir=reports_dir / 'plots',
    )


def load_diagnostic_inputs(project_root: Path) -> dict[str, pd.DataFrame]:
    paths = default_paths(project_root)
    return {
        'predictions_validation': load_parquet(paths.predictions_validation),
        'predictions_test': load_parquet(paths.predictions_test),
        'portfolio_comparison_metrics': load_csv(paths.portfolio_comparison_metrics),
        'portfolio_comparison_returns': load_csv(paths.portfolio_comparison_returns, parse_dates=['month_end']),
        'horse_race_metrics': load_csv(paths.horse_race_metrics),
        'horse_race_by_sleeve': load_csv(paths.horse_race_by_sleeve),
        'feature_set_summary': load_csv(paths.feature_set_summary),
        'modeling_panel_firstpass': load_parquet(paths.modeling_panel_firstpass),
    }


__all__ = [
    'DiagnosticPaths',
    'default_paths',
    'load_diagnostic_inputs',
    'load_csv',
    'load_parquet',
    'write_csv',
    'write_parquet',
    'write_text',
]
