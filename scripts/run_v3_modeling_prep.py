#!/usr/bin/env python3
"""Run the first-pass modeling preparation for XOPTPOE v3_long_horizon_china."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v3_modeling import run_v3_modeling_preparation


if __name__ == '__main__':
    run_v3_modeling_preparation(project_root=PROJECT_ROOT)
