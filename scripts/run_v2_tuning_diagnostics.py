#!/usr/bin/env python3
"""Run the narrow v2 PTO/E2E tuning and diagnostic study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workspace_v4.src.xoptpoe_v2_modeling.io import write_csv, write_text  # noqa: E402
from xoptpoe_v2_models.tuning import run_e2e_tuning_and_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run v2 narrow E2E tuning and diagnostics')
    parser.add_argument('--project-root', default=str(PROJECT_ROOT))
    parser.add_argument('--feature-set', default='core_plus_enrichment')
    parser.add_argument('--random-seed', type=int, default=42)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    result = run_e2e_tuning_and_diagnostics(
        project_root=root,
        feature_set_name=args.feature_set,
        random_seed=args.random_seed,
    )

    reports_dir = root / 'reports' / 'v2_long_horizon'
    write_csv(result.tuning_results, reports_dir / 'e2e_tuning_results.csv')
    write_csv(result.horizon_ablation, reports_dir / 'horizon_ablation_results.csv')
    write_csv(result.optimizer_behavior, reports_dir / 'optimizer_behavior_summary.csv')
    write_text(result.report_text, reports_dir / 'e2e_tuning_report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
