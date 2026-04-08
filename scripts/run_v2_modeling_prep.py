#!/usr/bin/env python3
"""Run first-pass modeling preparation for XOPTPOE v2_long_horizon outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_v2_modeling.prepare import run_v2_modeling_preparation  # noqa: E402



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare first-pass model-ready XOPTPOE v2 datasets")
    parser.add_argument(
        "--validation-months",
        type=int,
        default=24,
        help="Number of contiguous common-window validation months before the test block",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=24,
        help="Number of contiguous common-window final months reserved for test",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=60,
        help="Minimum common-window train span required by split validator",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    run_v2_modeling_preparation(
        project_root=PROJECT_ROOT,
        validation_months=args.validation_months,
        test_months=args.test_months,
        min_train_months=args.min_train_months,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
