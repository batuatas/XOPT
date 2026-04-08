"""Run the one-benchmark v3 scenario regime question framework."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_v3_scenarios.benchmark_question_plots import (
    plot_allocation_tilt_summary,
    plot_house_view_ladder,
    plot_regime_map,
    plot_regime_question_response,
)
from xoptpoe_v3_scenarios.benchmark_question_run import run_robust_regime_questions


def main() -> None:
    reports_root = PROJECT_ROOT / "reports" / "v3_long_horizon_china"
    plots_root = reports_root / "plots"
    reports_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    outputs = run_robust_regime_questions(PROJECT_ROOT)

    (reports_root / "scenario_regime_framework_v3.md").write_text(outputs.framework_report, encoding="utf-8")
    outputs.question_manifest.to_csv(reports_root / "scenario_regime_question_manifest_v3.csv", index=False)
    outputs.results.to_csv(reports_root / "scenario_regime_results_v3.csv", index=False)
    outputs.summary.to_csv(reports_root / "scenario_regime_summary_v3.csv", index=False)
    outputs.selected_questions.to_csv(reports_root / "scenario_selected_questions_v3.csv", index=False)
    outputs.regime_manifest.to_csv(reports_root / "scenario_regime_manifest_v3.csv", index=False)
    outputs.state_shift_summary.to_csv(reports_root / "scenario_regime_state_shift_summary_v3.csv", index=False)
    (reports_root / "scenario_conference_takeaways_v3.md").write_text(outputs.takeaways, encoding="utf-8")

    plot_regime_question_response(outputs.summary, plots_root / "scenario_regime_question_response_v3")
    plot_regime_map(outputs.results, outputs.selected_questions, plots_root / "scenario_regime_map_v3")
    plot_allocation_tilt_summary(outputs.summary, plots_root / "scenario_regime_allocation_tilts_v3")
    plot_house_view_ladder(outputs.results, plots_root / "scenario_regime_house_view_v3")


if __name__ == "__main__":
    main()
