from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_v3_scenarios.case_conference import build_case_conference_outputs
from xoptpoe_v3_scenarios.case_conference_figures import (
    plot_case_allocation_change,
    plot_case_benchmark_story,
    plot_case_comparison,
    plot_case_macro_fingerprint,
)


def main() -> None:
    reports_root = PROJECT_ROOT / "reports" / "v3_long_horizon_china"
    plots_root = reports_root / "plots"
    reports_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    case_summary, weight_summary, macro_fingerprint, report_text = build_case_conference_outputs(PROJECT_ROOT)

    (reports_root / "scenario_case_conference_report_v3.md").write_text(report_text, encoding="utf-8")
    case_summary.to_csv(reports_root / "scenario_selected_cases_v3.csv", index=False)
    case_summary.to_csv(reports_root / "scenario_case_summary_v3.csv", index=False)
    macro_fingerprint.to_csv(reports_root / "scenario_case_macro_fingerprint_v3.csv", index=False)
    weight_summary.to_csv(reports_root / "scenario_case_allocation_weights_v3.csv", index=False)

    plot_case_benchmark_story(case_summary, plots_root / "scenario_case_benchmark_story_v3")
    plot_case_allocation_change(case_summary, weight_summary, plots_root / "scenario_case_allocation_change_v3")
    plot_case_macro_fingerprint(case_summary, macro_fingerprint, plots_root / "scenario_case_macro_fingerprint_v3")
    plot_case_comparison(case_summary, plots_root / "scenario_case_comparison_v3")


if __name__ == "__main__":
    main()
