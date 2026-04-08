"""Run the first-pass v3 scenario scaffold audit and write its artifacts."""

from __future__ import annotations

from pathlib import Path

from xoptpoe_v3_scenarios.audit import run_scenario_scaffold_audit
from xoptpoe_v3_scenarios.io import default_paths


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = default_paths(project_root)
    result = run_scenario_scaffold_audit(project_root, anchor_month_end="2024-12-31")

    paths.reports_root.mkdir(parents=True, exist_ok=True)
    paths.modeling_root.mkdir(parents=True, exist_ok=True)

    (paths.reports_root / "scenario_pipeline_audit_v3.md").write_text(result.pipeline_audit_report, encoding="utf-8")
    (paths.reports_root / "scenario_adaptation_plan_v3.md").write_text(result.adaptation_plan_report, encoding="utf-8")
    result.scenario_candidate_manifest.to_csv(paths.reports_root / "scenario_candidate_manifest_v3.csv", index=False)
    result.scenario_state_manifest.to_csv(paths.reports_root / "scenario_state_manifest_v3.csv", index=False)
    result.scenario_probe_manifest.to_csv(paths.reports_root / "scenario_probe_manifest_v3.csv", index=False)
    result.scenario_scaffold_check.to_csv(paths.reports_root / "scenario_scaffold_check_v3.csv", index=False)


if __name__ == "__main__":
    main()
