#!/usr/bin/env python3
"""Run the second-stage allocator refinement for the fixed v4 best-60 predictor."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v4_models.allocator_refinement import run_allocator_refinement_v4, write_allocator_refinement_outputs


def main() -> None:
    outputs = run_allocator_refinement_v4(project_root=PROJECT_ROOT)
    write_allocator_refinement_outputs(PROJECT_ROOT, outputs)


if __name__ == "__main__":
    main()
