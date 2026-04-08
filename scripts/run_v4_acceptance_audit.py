#!/usr/bin/env python3
"""Run the v4 expanded-universe acceptance / EDA audit."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v4_modeling.audit import run_v4_acceptance_audit


if __name__ == "__main__":
    run_v4_acceptance_audit(project_root=PROJECT_ROOT)
