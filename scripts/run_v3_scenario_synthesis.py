"""Run the conference-facing synthesis pass for the first v3 scenario experiments."""

from __future__ import annotations

from pathlib import Path

from xoptpoe_v3_scenarios.conference_figures import (
    plot_china_under_scenarios_compact,
    plot_raw_deconcentration_case,
    plot_robust_vs_raw_response,
    plot_scenario_case_grid,
    plot_scenario_story_compact,
)
from xoptpoe_v3_scenarios.synthesis import run_scenario_synthesis


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_root = project_root / "reports" / "v3_long_horizon_china"
    plots_root = reports_root / "plots"
    reports_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    outputs = run_scenario_synthesis(project_root)

    (reports_root / "scenario_synthesis_report_v3.md").write_text(outputs.report_text, encoding="utf-8")
    (reports_root / "scenario_conference_notes_v3.md").write_text(outputs.conference_notes_text, encoding="utf-8")

    outputs.casebook.to_csv(reports_root / "scenario_casebook_v3.csv", index=False)
    outputs.variable_importance.to_csv(reports_root / "scenario_variable_importance_v3.csv", index=False)
    outputs.benchmark_contrast.to_csv(reports_root / "scenario_benchmark_contrast_v3.csv", index=False)
    outputs.china_role_summary.to_csv(reports_root / "scenario_china_role_summary_v3.csv", index=False)
    outputs.selected_cases.to_csv(reports_root / "scenario_selected_cases_v3.csv", index=False)
    outputs.case_state_shifts.to_csv(reports_root / "scenario_case_state_shifts_v3.csv", index=False)

    plot_scenario_story_compact(outputs.casebook, plots_root / "scenario_story_compact_v3")
    plot_scenario_case_grid(outputs.casebook, outputs.case_state_shifts, outputs.benchmark_contrast, plots_root / "scenario_case_grid_v3")
    plot_robust_vs_raw_response(outputs.response_cloud, plots_root / "robust_vs_raw_response_v3")
    plot_raw_deconcentration_case(outputs.casebook, outputs.casebook_weights, plots_root / "raw_deconcentration_case_v3")
    plot_china_under_scenarios_compact(outputs.china_role_summary, plots_root / "china_under_scenarios_compact_v3")


if __name__ == "__main__":
    main()
