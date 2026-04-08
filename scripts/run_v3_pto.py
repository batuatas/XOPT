#!/usr/bin/env python3
"""Run the XOPTPOE v3 long-horizon PTO model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v3_modeling.io import write_csv, write_text  # noqa: E402
from xoptpoe_v3_models.pto import run_v3_pto  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run v3 long-horizon PTO model')
    parser.add_argument('--project-root', default=str(PROJECT_ROOT))
    parser.add_argument('--feature-set', default='core_plus_enrichment')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    result = run_v3_pto(
        project_root=root,
        feature_set_name=args.feature_set,
        random_seed=args.random_seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    data_dir = root / 'data' / 'modeling_v3'
    reports_dir = root / 'reports' / 'v3_long_horizon_china'
    write_csv(result.predictions_validation, data_dir / 'predictions_validation_pto.csv')
    write_csv(result.predictions_test, data_dir / 'predictions_test_pto.csv')
    write_csv(result.metrics_overall, reports_dir / 'pto_metrics_overall.csv')
    write_csv(result.metrics_by_sleeve, reports_dir / 'pto_metrics_by_sleeve.csv')
    write_csv(result.metrics_by_horizon, reports_dir / 'pto_metrics_by_horizon.csv')
    write_csv(result.training_history, reports_dir / 'training_history_pto.csv')
    write_csv(result.portfolio_metrics, reports_dir / 'portfolio_metrics_pto.csv')
    write_csv(result.portfolio_returns, reports_dir / 'portfolio_returns_pto.csv')
    write_csv(result.selection_summary, reports_dir / 'model_selection_pto.csv')
    write_text(result.report_text, reports_dir / 'pto_report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
