#!/usr/bin/env python3
"""Run first-pass modeling preparation for frozen XOPTPOE v1 outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_modeling.prepare_dataset import run_modeling_preparation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare model-ready XOPTPOE v1 datasets")
    parser.add_argument(
        "--validation-months",
        type=int,
        default=24,
        help="Number of contiguous validation months before the test block",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=24,
        help="Number of contiguous final months reserved for test",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=60,
        help="Minimum train span required by split validator",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_modeling_preparation(
        project_root=PROJECT_ROOT,
        validation_months=args.validation_months,
        test_months=args.test_months,
        min_train_months=args.min_train_months,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
