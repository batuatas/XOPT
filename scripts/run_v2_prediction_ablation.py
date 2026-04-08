#!/usr/bin/env python3
"""Run the v2 long-horizon prediction ablation study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workspace_v4.src.xoptpoe_v2_modeling.io import write_csv, write_text  # noqa: E402
from xoptpoe_v2_models.prediction_ablation import run_prediction_ablation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run v2 long-horizon prediction ablation study')
    parser.add_argument('--project-root', default=str(PROJECT_ROOT))
    parser.add_argument('--random-seed', type=int, default=42)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    result = run_prediction_ablation(project_root=root, random_seed=args.random_seed)

    reports_dir = root / 'reports' / 'v2_long_horizon'
    write_csv(result.ablation_results, reports_dir / 'prediction_ablation_results.csv')
    write_csv(result.horizon_specific_summary, reports_dir / 'horizon_specific_prediction_summary.csv')
    write_csv(result.feature_block_summary, reports_dir / 'feature_block_prediction_summary.csv')
    write_csv(result.sleeve_difficulty, reports_dir / 'sleeve_prediction_difficulty.csv')
    write_text(result.report_text, reports_dir / 'prediction_ablation_report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
