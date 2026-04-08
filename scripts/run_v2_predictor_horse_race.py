#!/usr/bin/env python3
"""Run the compact v2 predictor horse race and SAA portfolio comparison."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v2_models.horse_race import run_predictor_horse_race, write_horse_race_outputs  # noqa: E402


def main() -> None:
    project_root = PROJECT_ROOT
    outputs = run_predictor_horse_race(project_root=project_root)
    write_horse_race_outputs(project_root=project_root, outputs=outputs)


if __name__ == "__main__":
    main()
