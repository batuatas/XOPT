#!/usr/bin/env python3
"""Run first-pass modeling preparation for XOPTPOE v4_expanded_universe."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v4_modeling import run_v4_modeling_preparation


if __name__ == "__main__":
    run_v4_modeling_preparation(project_root=PROJECT_ROOT)
