#!/usr/bin/env python3
"""Run the controlled v3 supervised prediction benchmark campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v3_models.prediction_benchmark import (  # noqa: E402
    RollingConfig,
    run_prediction_benchmark_v3,
    write_prediction_benchmark_v3_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v3 supervised prediction benchmark campaign")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--min-train-months", type=int, default=48)
    parser.add_argument("--validation-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    outputs = run_prediction_benchmark_v3(
        project_root=root,
        rolling_config=RollingConfig(
            min_train_months=args.min_train_months,
            validation_months=args.validation_months,
            test_months=args.test_months,
            step_months=args.step_months,
            random_seed=args.random_seed,
        ),
    )
    write_prediction_benchmark_v3_outputs(project_root=root, outputs=outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
