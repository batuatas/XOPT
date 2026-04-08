"""Run the first real v3 scenario-generation experiments."""

from __future__ import annotations

from pathlib import Path
import shutil

from xoptpoe_v3_scenarios.experiments import run_scenario_experiments
from xoptpoe_v3_scenarios.plots import (
    plot_china_role,
    plot_portfolio_change,
    plot_return_vs_concentration,
    plot_state_shift_heatmap,
)
from xoptpoe_v3_scenarios.summaries import strongest_findings_markdown


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_root = project_root / "reports" / "v3_long_horizon_china"
    modeling_root = project_root / "data" / "modeling_v3"
    plots_root = reports_root / "plots"
    reports_root.mkdir(parents=True, exist_ok=True)
    modeling_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    outputs = run_scenario_experiments(project_root)

    (reports_root / "scenario_experiment_report_v3.md").write_text(
        outputs.report_text + "\n\n" + strongest_findings_markdown(outputs.summary, outputs.portfolio_response_summary, outputs.china_diagnostics),
        encoding="utf-8",
    )
    outputs.summary.to_csv(reports_root / "scenario_experiment_summary_v3.csv", index=False)
    outputs.state_shift_summary.to_csv(reports_root / "scenario_state_shift_summary_v3.csv", index=False)
    outputs.portfolio_response_summary.to_csv(reports_root / "scenario_portfolio_response_summary_v3.csv", index=False)
    outputs.representative_cases.to_csv(reports_root / "scenario_representative_cases_v3.csv", index=False)
    outputs.china_diagnostics.to_csv(reports_root / "china_scenario_diagnostics_v3.csv", index=False)
    outputs.experiment_manifest.to_csv(reports_root / "scenario_experiment_manifest_v3.csv", index=False)

    plot_state_shift_heatmap(outputs.representative_state_table, plots_root / "scenario_state_shift_heatmap_v3")
    plot_portfolio_change(outputs.portfolio_response_summary, plots_root / "scenario_portfolio_change_v3")
    plot_return_vs_concentration(outputs.summary, outputs.portfolio_response_summary, plots_root / "scenario_return_vs_concentration_v3")
    plot_china_role(outputs.china_diagnostics, plots_root / "china_role_under_scenarios_v3")

    benchmark_manifest_src = project_root / "data" / "modeling_v3" / "final_benchmark_manifest_v3.csv"
    benchmark_manifest_dst = reports_root / "final_benchmark_manifest_v3.csv"
    if benchmark_manifest_src.exists():
        shutil.copyfile(benchmark_manifest_src, benchmark_manifest_dst)


if __name__ == "__main__":
    main()
