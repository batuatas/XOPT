"""End-to-end modeling-preparation runner for frozen XOPTPOE v1 outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from xoptpoe_modeling.eda import (
    IMPORTANT_FEATURES,
    TARGET_COL,
    correlation_diagnostics,
    infer_feature_columns,
    summarize_features,
    summarize_target,
)
from xoptpoe_modeling.io import load_modeling_panel, write_csv, write_text
from xoptpoe_modeling.splits import (
    SplitConfig,
    assign_time_splits,
    build_split_manifest,
    build_split_summary,
    filter_modeling_panel,
    split_subsets,
)


def _build_report(
    *,
    source_path: Path,
    filtered_panel: pd.DataFrame,
    split_manifest: pd.DataFrame,
    split_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    important_feature_summary: pd.DataFrame,
    target_corr: pd.DataFrame,
    pairwise_corr: pd.DataFrame,
) -> str:
    """Render a compact human-readable modeling prep report."""
    start_date = pd.to_datetime(filtered_panel["month_end"]).min().date().isoformat()
    end_date = pd.to_datetime(filtered_panel["month_end"]).max().date().isoformat()
    month_count = int(filtered_panel["month_end"].nunique())
    sleeve_count = int(filtered_panel["sleeve_id"].nunique())
    row_count = int(len(filtered_panel))

    rows_per_month = filtered_panel.groupby("month_end").size()
    min_rows_per_month = int(rows_per_month.min())
    max_rows_per_month = int(rows_per_month.max())

    overall_target = target_summary[
        (target_summary["split"] == "ALL") & (target_summary["sleeve_id"] == "ALL")
    ].iloc[0]
    by_split_target = target_summary[target_summary["sleeve_id"] == "ALL"].copy()

    top_missing = feature_summary.sort_values("missing_share", ascending=False).head(12)
    top_target_corr = target_corr.head(12)
    high_pairwise = pairwise_corr.head(12)

    suspicious_notes: list[str] = []
    if min_rows_per_month != max_rows_per_month:
        suspicious_notes.append(
            f"Row count per month varies ({min_rows_per_month} to {max_rows_per_month}); check partial cross-sections."
        )
    max_missing_important = (
        float(important_feature_summary["missing_share"].max())
        if not important_feature_summary.empty
        else 0.0
    )
    if max_missing_important > 0.05:
        suspicious_notes.append(
            f"At least one highlighted feature has >5% missingness (max={max_missing_important:.2%})."
        )
    if float(overall_target["min"]) < -0.40 or float(overall_target["max"]) > 0.40:
        suspicious_notes.append(
            "Target tails exceed +/-40% monthly; confirm behavior is intended for the use case."
        )

    lines: list[str] = []
    lines.append("# XOPTPOE v1 Modeling Preparation Report")
    lines.append("")
    lines.append("## Data Source")
    lines.append(f"- source_file: `{source_path}`")
    lines.append("")
    lines.append("## Filtering Applied")
    lines.append("- Keep rows with `sample_inclusion_flag == 1`.")
    lines.append("- Keep rows with `target_quality_flag == 1`.")
    lines.append(f"- Keep rows with non-null `{TARGET_COL}`.")
    lines.append("- Enforce no duplicate `(sleeve_id, month_end)` keys.")
    lines.append("")
    lines.append("## Filtered Panel Snapshot")
    lines.append(f"- usable_start: `{start_date}`")
    lines.append(f"- usable_end: `{end_date}`")
    lines.append(f"- month_count: `{month_count}`")
    lines.append(f"- sleeve_count: `{sleeve_count}`")
    lines.append(f"- row_count: `{row_count}`")
    lines.append(f"- rows_per_month_min_max: `{min_rows_per_month} / {max_rows_per_month}`")
    lines.append("")
    lines.append("## Time Split Design")
    lines.append("- Split method: deterministic month-block split, no random shuffling.")
    lines.append("- Default configuration: final 24 months = test, prior 24 months = validation, remainder = train.")
    lines.append("- Split manifest:")
    for row in split_manifest.itertuples(index=False):
        lines.append(
            f"  - {row.split}: {row.start_month_end} to {row.end_month_end}, "
            f"months={row.month_count}, rows={row.row_count}, sleeves={row.sleeve_count}"
        )
    lines.append("")
    lines.append("## Split Coverage by Sleeve")
    split_sleeve = split_summary[split_summary["sleeve_id"] != "ALL"]
    for split_name in ("train", "validation", "test"):
        chunk = split_sleeve[split_sleeve["split"] == split_name]
        if chunk.empty:
            continue
        lines.append(f"- {split_name}:")
        for row in chunk.itertuples(index=False):
            lines.append(
                f"  - {row.sleeve_id}: months={row.month_count}, rows={row.row_count}, "
                f"range={row.start_month_end}..{row.end_month_end}"
            )
    lines.append("")
    lines.append("## Target Diagnostics")
    lines.append(
        f"- panel target mean/std: `{overall_target['mean']:.6f} / {overall_target['std']:.6f}`"
    )
    lines.append(
        f"- panel target min/p95/max: `{overall_target['min']:.6f} / {overall_target['p95']:.6f} / {overall_target['max']:.6f}`"
    )
    lines.append("- target summary by split (all sleeves pooled):")
    for row in by_split_target.itertuples(index=False):
        lines.append(
            f"  - {row.split}: mean={row.mean:.6f}, std={row.std:.6f}, "
            f"p05={row.p05:.6f}, p50={row.p50:.6f}, p95={row.p95:.6f}"
        )
    lines.append("")
    lines.append("## Feature Missingness and Scale")
    lines.append("- top missingness features:")
    for row in top_missing.itertuples(index=False):
        lines.append(f"  - {row.feature_name}: missing={row.missing_share:.2%}")
    lines.append("- highlighted feature diagnostics:")
    for row in important_feature_summary.itertuples(index=False):
        lines.append(
            f"  - {row.feature_name}: missing={row.missing_share:.2%}, "
            f"mean={row.mean:.6f}, std={row.std:.6f}, p01={row.p01:.6f}, p99={row.p99:.6f}"
        )
    lines.append("")
    lines.append("## Correlation Diagnostics")
    lines.append("- highest absolute correlation with target:")
    for row in top_target_corr.itertuples(index=False):
        lines.append(
            f"  - {row.feature_name}: corr={row.corr_with_target:.4f}, abs={row.abs_corr_with_target:.4f}"
        )
    if high_pairwise.empty:
        lines.append("- high pairwise feature correlation (|corr| >= 0.85): none in highlighted subset.")
    else:
        lines.append("- high pairwise feature correlation (|corr| >= 0.85):")
        for row in high_pairwise.itertuples(index=False):
            lines.append(f"  - {row.feature_a} vs {row.feature_b}: corr={row.corr:.4f}")
    lines.append("")
    lines.append("## Suspicious Findings")
    if not suspicious_notes:
        lines.append("- No obvious structural red flags in filtered panel and split diagnostics.")
    else:
        for note in suspicious_notes:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("## Output Files")
    lines.append("- `data/modeling/modeling_panel_filtered.csv`")
    lines.append("- `data/modeling/train_split.csv`")
    lines.append("- `data/modeling/validation_split.csv`")
    lines.append("- `data/modeling/test_split.csv`")
    lines.append("- `data/modeling/split_manifest.csv`")
    lines.append("- `reports/split_summary.csv`")
    lines.append("- `reports/target_summary_by_sleeve.csv`")
    lines.append("- `reports/feature_summary.csv`")
    lines.append("- `reports/modeling_prep_report.md`")

    return "\n".join(lines) + "\n"


def run_modeling_preparation(
    *,
    project_root: Path,
    validation_months: int = 24,
    test_months: int = 24,
    min_train_months: int = 60,
) -> dict[str, Path]:
    """Run full modeling-preparation workflow and persist outputs."""
    source_path = project_root / "data" / "final" / "modeling_panel.csv"
    data_modeling_dir = project_root / "data" / "modeling"
    reports_dir = project_root / "reports"

    modeling_panel = load_modeling_panel(source_path)
    filtered_panel = filter_modeling_panel(modeling_panel)

    split_cfg = SplitConfig(
        validation_months=validation_months,
        test_months=test_months,
        min_train_months=min_train_months,
    )
    panel_with_splits = assign_time_splits(filtered_panel, split_cfg)
    split_frames = split_subsets(panel_with_splits)

    split_manifest = build_split_manifest(panel_with_splits)
    split_summary = build_split_summary(panel_with_splits)

    feature_cols = infer_feature_columns(panel_with_splits)
    target_summary = summarize_target(panel_with_splits)
    feature_summary = summarize_features(panel_with_splits, feature_cols=feature_cols)

    important_present = [c for c in IMPORTANT_FEATURES if c in panel_with_splits.columns]
    target_corr, pairwise_corr = correlation_diagnostics(
        panel_with_splits,
        feature_subset=important_present,
    )
    important_feature_summary = feature_summary[
        feature_summary["feature_name"].isin(important_present)
    ].copy()

    # Persist model-ready tables.
    write_csv(panel_with_splits, data_modeling_dir / "modeling_panel_filtered.csv")
    write_csv(split_frames["train"], data_modeling_dir / "train_split.csv")
    write_csv(split_frames["validation"], data_modeling_dir / "validation_split.csv")
    write_csv(split_frames["test"], data_modeling_dir / "test_split.csv")
    write_csv(split_manifest, data_modeling_dir / "split_manifest.csv")

    # Persist compact EDA reports.
    write_csv(split_summary, reports_dir / "split_summary.csv")
    write_csv(target_summary, reports_dir / "target_summary_by_sleeve.csv")
    write_csv(feature_summary, reports_dir / "feature_summary.csv")

    report_text = _build_report(
        source_path=source_path,
        filtered_panel=panel_with_splits,
        split_manifest=split_manifest,
        split_summary=split_summary,
        target_summary=target_summary,
        feature_summary=feature_summary,
        important_feature_summary=important_feature_summary,
        target_corr=target_corr,
        pairwise_corr=pairwise_corr,
    )
    report_path = reports_dir / "modeling_prep_report.md"
    write_text(report_text, report_path)

    return {
        "modeling_panel_filtered": data_modeling_dir / "modeling_panel_filtered.csv",
        "train_split": data_modeling_dir / "train_split.csv",
        "validation_split": data_modeling_dir / "validation_split.csv",
        "test_split": data_modeling_dir / "test_split.csv",
        "split_manifest": data_modeling_dir / "split_manifest.csv",
        "split_summary": reports_dir / "split_summary.csv",
        "target_summary_by_sleeve": reports_dir / "target_summary_by_sleeve.csv",
        "feature_summary": reports_dir / "feature_summary.csv",
        "modeling_prep_report": report_path,
    }
