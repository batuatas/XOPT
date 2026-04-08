#!/usr/bin/env python3
"""Run the experimental China-and-enhanced-indicators extension."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_experimental.extension import ExperimentalConfig, run_experimental_extension  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimental XOPTPOE extension")
    parser.add_argument("--min-train-months", type=int, default=96)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=24)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ExperimentalConfig(
        project_root=PROJECT_ROOT,
        min_train_months=args.min_train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
        random_state=args.random_state,
    )
    run_experimental_extension(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
