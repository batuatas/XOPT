#!/usr/bin/env python3
"""Run the full locked v1 XOPTPOE data build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xoptpoe_data.build_modeling_panel import (  # noqa: E402
    build_macro_mapping,
    build_modeling_panel,
    compute_sample_start_report,
)
from xoptpoe_data.config import default_config  # noqa: E402
from xoptpoe_data.features.build_features import build_feature_panel  # noqa: E402
from xoptpoe_data.io import (  # noqa: E402
    load_asset_master_seed,
    load_macro_manifest,
    load_source_master_seed,
    validate_locked_asset_master,
    write_csv,
)
from xoptpoe_data.macro.build_macro_state_panel import (  # noqa: E402
    build_global_state_panel,
    build_macro_state_panel,
)
from xoptpoe_data.macro.fetch_macro import FredMacroAdapter, fetch_macro_raw  # noqa: E402
from xoptpoe_data.qa.run_qa import run_all_qa, run_target_qa, write_qa_summary  # noqa: E402
from xoptpoe_data.targets.build_monthly_targets import (  # noqa: E402
    build_monthly_realized_returns,
    build_target_panel,
    collapse_target_to_month_end_prices,
    extract_tb3ms_monthly,
)
from xoptpoe_data.targets.fetch_targets import (  # noqa: E402
    YahooFinanceTargetAdapter,
    fetch_sleeve_target_raw,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build XOPTPOE locked v1 monthly dataset")
    parser.add_argument("--start-date", default="2006-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--allow-macro-fallback",
        action="store_true",
        help="Allow manifest fallback macro series codes (disabled by default)",
    )
    parser.add_argument(
        "--allow-qa-failures",
        action="store_true",
        help="Do not raise on QA failures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = default_config(project_root=PROJECT_ROOT, end_date=args.end_date)
    cfg = cfg.__class__(
        paths=cfg.paths,
        start_date=args.start_date,
        end_date=args.end_date,
        lag_policy_tag=cfg.lag_policy_tag,
        allow_macro_fallback=args.allow_macro_fallback,
    )
    cfg.paths.ensure_directories()

    asset_master = load_asset_master_seed(cfg.paths)
    source_master = load_source_master_seed(cfg.paths)
    macro_manifest = load_macro_manifest(cfg.paths)

    validate_locked_asset_master(asset_master)

    write_csv(asset_master, cfg.paths.data_final_dir / "asset_master.csv")
    write_csv(source_master, cfg.paths.data_final_dir / "source_master.csv")

    # 1) Targets: fetch raw daily prices.
    target_adapter = YahooFinanceTargetAdapter()
    sleeve_target_raw = fetch_sleeve_target_raw(
        asset_master=asset_master,
        adapter=target_adapter,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    write_csv(sleeve_target_raw, cfg.paths.data_raw_dir / "sleeve_target_raw.csv")

    # 2) Collapse to month-end prices and build monthly realized returns.
    month_end_prices = collapse_target_to_month_end_prices(sleeve_target_raw)
    monthly_returns = build_monthly_realized_returns(month_end_prices)
    write_csv(month_end_prices, cfg.paths.data_intermediate_dir / "sleeve_month_end_prices.csv")
    write_csv(monthly_returns, cfg.paths.data_intermediate_dir / "sleeve_monthly_returns.csv")

    # 3) Fetch TB3MS first to build forward rf for target panel (target QA gate prerequisite).
    macro_adapter = FredMacroAdapter()
    rf_macro_raw = fetch_macro_raw(
        macro_manifest=macro_manifest,
        adapter=macro_adapter,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        allow_fallback=cfg.allow_macro_fallback,
        series_filter={"US_RF3M"},
    )
    tb3ms_monthly = extract_tb3ms_monthly(rf_macro_raw)
    write_csv(tb3ms_monthly, cfg.paths.data_intermediate_dir / "tb3ms_monthly.csv")

    target_panel = build_target_panel(month_end_prices=month_end_prices, tb3ms_monthly=tb3ms_monthly)
    write_csv(target_panel, cfg.paths.data_final_dir / "target_panel.csv")

    # 4) Mandatory first QA gate: target layer must pass before full macro build.
    target_gate = run_target_qa(
        asset_master=asset_master,
        sleeve_target_raw=sleeve_target_raw,
        month_end_prices=month_end_prices,
        target_panel=target_panel,
    )
    write_csv(target_gate, cfg.paths.reports_dir / "target_gate_audit.csv")
    write_qa_summary(target_gate, cfg.paths.reports_dir / "target_gate_summary.md")

    gate_failures = int((target_gate["audit_result"] == "FAIL").sum())
    if gate_failures > 0:
        raise RuntimeError(
            f"Target QA gate failed with {gate_failures} failing checks. See reports/target_gate_summary.md"
        )

    # 5) Full macro backbone fetch and state construction.
    macro_raw = fetch_macro_raw(
        macro_manifest=macro_manifest,
        adapter=macro_adapter,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        allow_fallback=cfg.allow_macro_fallback,
        series_filter=None,
    )
    write_csv(macro_raw, cfg.paths.data_raw_dir / "macro_raw.csv")

    month_ends = pd.Series(sorted(pd.to_datetime(month_end_prices["month_end"]).unique()))

    macro_state_panel = build_macro_state_panel(
        macro_raw=macro_raw,
        month_ends=month_ends,
        lag_policy_tag=cfg.lag_policy_tag,
    )
    global_state_panel = build_global_state_panel(macro_state_panel)

    write_csv(macro_state_panel, cfg.paths.data_intermediate_dir / "macro_state_panel.csv")
    write_csv(global_state_panel, cfg.paths.data_intermediate_dir / "global_state_panel.csv")

    # 6) Feature panel.
    feature_panel = build_feature_panel(
        monthly_returns=monthly_returns,
        macro_state_panel=macro_state_panel,
        global_state_panel=global_state_panel,
    )
    write_csv(feature_panel, cfg.paths.data_intermediate_dir / "feature_panel.csv")

    # 7) Mapping + modeling panel.
    macro_mapping = build_macro_mapping(asset_master)
    write_csv(macro_mapping, cfg.paths.data_final_dir / "macro_mapping.csv")

    modeling_panel = build_modeling_panel(
        feature_panel=feature_panel,
        target_panel=target_panel,
        asset_master=asset_master,
        macro_mapping=macro_mapping,
    )
    write_csv(modeling_panel, cfg.paths.data_final_dir / "modeling_panel.csv")

    # 8) Required sample-start diagnostics.
    sample_start_report = compute_sample_start_report(
        sleeve_target_raw=sleeve_target_raw,
        month_end_prices=month_end_prices,
        feature_panel=feature_panel,
        target_panel=target_panel,
        modeling_panel=modeling_panel,
    )
    write_csv(sample_start_report, cfg.paths.reports_dir / "sample_start_report.csv")

    # 9) Full QA and coverage audit.
    coverage_audit = run_all_qa(
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
        reports_dir=cfg.paths.reports_dir,
    )
    write_csv(coverage_audit, cfg.paths.data_final_dir / "coverage_audit.csv")

    fail_count = int((coverage_audit["audit_result"] == "FAIL").sum())
    if fail_count > 0 and not args.allow_qa_failures:
        raise RuntimeError(f"Build completed with {fail_count} QA failures. See reports/qa_summary.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
