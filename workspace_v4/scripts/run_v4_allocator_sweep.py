#!/usr/bin/env python3
"""Run the compact allocator-regularization sweep for the fixed v4 best-60 predictor."""

from __future__ import annotations

from pathlib import Path

from xoptpoe_v4_models.allocator_sweep import run_allocator_sweep_v4, write_allocator_sweep_outputs


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs = run_allocator_sweep_v4(project_root=project_root)
    write_allocator_sweep_outputs(project_root, outputs)


if __name__ == "__main__":
    main()
