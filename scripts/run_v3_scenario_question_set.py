"""Run the conference-grade v3 scenario question framework."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_v3_scenarios.question_experiments import run_scenario_question_set
from xoptpoe_v3_scenarios.question_plots import (
    plot_allocation_tilt_questions,
    plot_benchmark_question_response,
    plot_question_regime_map,
)


def main() -> None:
    project_root = PROJECT_ROOT
    reports_root = project_root / "reports" / "v3_long_horizon_china"
    plots_root = reports_root / "plots"
    reports_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    outputs = run_scenario_question_set(project_root)

    (reports_root / "scenario_question_framework_v3.md").write_text(outputs.framework_report, encoding="utf-8")
    outputs.question_manifest.to_csv(reports_root / "scenario_question_manifest_v3.csv", index=False)
    outputs.regime_manifest.to_csv(reports_root / "scenario_regime_manifest_v3.csv", index=False)
    outputs.question_results.to_csv(reports_root / "scenario_question_results_v3.csv", index=False)
    outputs.regime_summary.to_csv(reports_root / "scenario_regime_summary_v3.csv", index=False)
    outputs.state_shift_summary.to_csv(reports_root / "scenario_question_state_shifts_v3.csv", index=False)
    outputs.selected_questions.to_csv(reports_root / "scenario_selected_questions_v3.csv", index=False)
    (reports_root / "scenario_conference_question_notes_v3.md").write_text(outputs.conference_notes, encoding="utf-8")

    plot_question_regime_map(outputs.regime_summary, outputs.selected_questions, plots_root / "scenario_question_regime_map_v3")
    plot_benchmark_question_response(outputs.question_results, plots_root / "scenario_benchmark_question_response_v3")
    plot_allocation_tilt_questions(outputs.question_results, plots_root / "scenario_allocation_tilt_questions_v3")


if __name__ == "__main__":
    main()
