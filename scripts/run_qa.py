#!/usr/bin/env python3
"""Run QA checks from already-built XOPTPOE artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_data.io import load_csv, write_csv  # noqa: E402
from xoptpoe_data.qa.run_qa import run_all_qa  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XOPTPOE QA checks")
    parser.add_argument(
        "--allow-qa-failures",
        action="store_true",
        help="Do not raise non-zero exit when failures are found",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    macro_manifest = load_csv(PROJECT_ROOT / "config" / "macro_series_manifest.csv")

    asset_master = load_csv(PROJECT_ROOT / "data" / "final" / "asset_master.csv", parse_dates=["start_date_target"])
    sleeve_target_raw = load_csv(PROJECT_ROOT / "data" / "raw" / "sleeve_target_raw.csv", parse_dates=["trade_date"])
    month_end_prices = load_csv(PROJECT_ROOT / "data" / "intermediate" / "sleeve_month_end_prices.csv", parse_dates=["month_end", "trade_date"])
    macro_raw = load_csv(PROJECT_ROOT / "data" / "raw" / "macro_raw.csv", parse_dates=["obs_date"])
    macro_state_panel = load_csv(PROJECT_ROOT / "data" / "intermediate" / "macro_state_panel.csv", parse_dates=["month_end"])
    global_state_panel = load_csv(PROJECT_ROOT / "data" / "intermediate" / "global_state_panel.csv", parse_dates=["month_end"])
    feature_panel = load_csv(PROJECT_ROOT / "data" / "intermediate" / "feature_panel.csv", parse_dates=["month_end"])
    target_panel = load_csv(PROJECT_ROOT / "data" / "final" / "target_panel.csv", parse_dates=["month_end", "next_month_end"])
    modeling_panel = load_csv(PROJECT_ROOT / "data" / "final" / "modeling_panel.csv", parse_dates=["month_end"])
    macro_mapping = load_csv(PROJECT_ROOT / "data" / "final" / "macro_mapping.csv")

    audit_df = run_all_qa(
        asset_master=asset_master,
        macro_manifest=macro_manifest,
        sleeve_target_raw=sleeve_target_raw,
        month_end_prices=month_end_prices,
        macro_raw=macro_raw,
        macro_state_panel=macro_state_panel,
        global_state_panel=global_state_panel,
        feature_panel=feature_panel,
        target_panel=target_panel,
        modeling_panel=modeling_panel,
        macro_mapping=macro_mapping,
        reports_dir=PROJECT_ROOT / "reports",
    )

    write_csv(audit_df, PROJECT_ROOT / "data" / "final" / "coverage_audit.csv")

    fail_count = int((audit_df["audit_result"] == "FAIL").sum())
    if fail_count > 0 and not args.allow_qa_failures:
        raise RuntimeError(f"QA failed with {fail_count} failing checks. See reports/qa_summary.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
