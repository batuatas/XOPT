#!/usr/bin/env python3
"""Run the v4 supervised prediction benchmark layer."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v4_models.prediction_benchmark import run_prediction_benchmark_v4, write_prediction_benchmark_v4_outputs


if __name__ == "__main__":
    outputs = run_prediction_benchmark_v4(project_root=PROJECT_ROOT)
    write_prediction_benchmark_v4_outputs(project_root=PROJECT_ROOT, outputs=outputs)
