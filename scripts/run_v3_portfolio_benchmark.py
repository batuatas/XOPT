#!/usr/bin/env python3
"""Run the v3 portfolio benchmark and allocation diagnostics phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v3_models.optim_layers import RiskConfig, candidate_optimizer_grid  # noqa: E402
from xoptpoe_v3_models.portfolio_benchmark import (  # noqa: E402
    run_portfolio_benchmark_v3,
    write_portfolio_benchmark_v3_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v3 portfolio benchmark phase")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--risk-lookback-months", type=int, default=60)
    parser.add_argument("--risk-min-months", type=int, default=12)
    parser.add_argument("--risk-ewma-beta", type=float, default=0.94)
    parser.add_argument("--risk-diagonal-shrinkage", type=float, default=0.10)
    parser.add_argument("--risk-ridge", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    outputs = run_portfolio_benchmark_v3(
        project_root=root,
        risk_config=RiskConfig(
            lookback_months=args.risk_lookback_months,
            min_months=args.risk_min_months,
            ewma_beta=args.risk_ewma_beta,
            diagonal_shrinkage=args.risk_diagonal_shrinkage,
            ridge=args.risk_ridge,
        ),
        optimizer_grid=list(candidate_optimizer_grid()),
    )
    write_portfolio_benchmark_v3_outputs(project_root=root, outputs=outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
