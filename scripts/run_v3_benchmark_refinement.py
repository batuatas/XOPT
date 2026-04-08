from __future__ import annotations

from pathlib import Path

from xoptpoe_v3_models.benchmark_refinement import (
    run_benchmark_refinement_v3,
    write_benchmark_refinement_v3_outputs,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs = run_benchmark_refinement_v3(project_root=project_root)
    write_benchmark_refinement_v3_outputs(project_root=project_root, outputs=outputs)


if __name__ == "__main__":
    main()
