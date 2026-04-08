"""End-to-end first-pass modeling preparation for XOPTPOE v3_long_horizon_china."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xoptpoe_v3_modeling.features import (
    DEFAULT_FEATURE_SET,
    FEATURE_SET_ORDER,
    build_feature_set_manifest,
    feature_columns_for_set,
    summarize_feature_sets,
)
from xoptpoe_v3_modeling.io import load_csv, load_parquet, write_csv, write_parquet, write_text
from xoptpoe_v3_modeling.splits import (
    EXCLUDED_SPLIT_NAME,
    SplitConfig,
    assign_default_splits,
    build_split_manifest,
    build_split_summary,
    filter_firstpass_panel,
    split_subsets,
)


@dataclass(frozen=True)
class PrepPaths:
    project_root: Path
    source_panel: Path
    feature_dictionary: Path
    interaction_dictionary: Path
    data_modeling_dir: Path
    reports_dir: Path


def default_paths(project_root: Path) -> PrepPaths:
    root = project_root.resolve()
    return PrepPaths(
        project_root=root,
        source_panel=root / "data" / "final_v3_long_horizon_china" / "modeling_panel_hstack.parquet",
        feature_dictionary=root / "data" / "final_v3_long_horizon_china" / "feature_dictionary.csv",
        interaction_dictionary=root / "data" / "final_v3_long_horizon_china" / "interaction_dictionary.csv",
        data_modeling_dir=root / "data" / "modeling_v3",
        reports_dir=root / "reports" / "v3_long_horizon_china",
    )


def _attach_feature_set_row_stats(panel, feature_manifest):
    work = panel.copy()
    for feature_set_name in FEATURE_SET_ORDER:
        cols = [c for c in feature_columns_for_set(feature_manifest, feature_set_name) if c in work.columns]
        if not cols:
            work[f"{feature_set_name}_missing_feature_count"] = 0
            work[f"{feature_set_name}_missing_feature_share"] = 0.0
            work[f"{feature_set_name}_complete_flag"] = 1
            continue
        missing_count = work[cols].isna().sum(axis=1)
        work[f"{feature_set_name}_missing_feature_count"] = missing_count.astype(int)
        work[f"{feature_set_name}_missing_feature_share"] = (missing_count / len(cols)).astype(float)
        work[f"{feature_set_name}_complete_flag"] = missing_count.eq(0).astype(int)
    return work


def _build_report(*, filtered_panel, split_panel, split_manifest, split_summary, feature_set_summary):
    subset_start = filtered_panel["month_end"].min().date().isoformat()
    subset_end = filtered_panel["month_end"].max().date().isoformat()
    excluded_tail = split_panel.loc[split_panel["default_split"].eq(EXCLUDED_SPLIT_NAME)]
    excluded_months = int(excluded_tail["month_end"].nunique()) if not excluded_tail.empty else 0
    excluded_rows = int(len(excluded_tail))

    rows = []
    rows.append("# XOPTPOE v3_long_horizon_china Modeling Preparation Report")
    rows.append("")
    rows.append("## Filtering Applied")
    rows.append("- Source table: `data/final_v3_long_horizon_china/modeling_panel_hstack.parquet`.")
    rows.append("- Keep rows with `baseline_trainable_flag == 1`.")
    rows.append("- Keep rows with `target_available_flag == 1` and non-null `annualized_excess_forward_return`.")
    rows.append("- Keep horizons `{60, 120}` only for the first-pass modeling package; `180` remains in the versioned source dataset.")
    rows.append("- Enforce no duplicate `(sleeve_id, month_end, horizon_months)` keys.")
    rows.append("")
    rows.append("## First-Pass Modeling Subset")
    rows.append(f"- filtered_row_count: `{len(filtered_panel)}`")
    rows.append(f"- filtered_month_range: `{subset_start}` to `{subset_end}`")
    rows.append(f"- sleeve_count: `{filtered_panel['sleeve_id'].nunique()}`")
    rows.append("- filtered_rows_by_horizon:")
    for horizon, row_count in filtered_panel.groupby("horizon_months").size().sort_index().items():
        rows.append(f"  - {int(horizon)}m: {int(row_count)} rows")
    rows.append("")
    rows.append("## Default Split Design")
    rows.append("- Split method: deterministic month-block split on the common `60m`/`120m` window only.")
    rows.append("- Rationale: validation and test should contain both horizons and the full 9-sleeve cross-section.")
    rows.append("- Default configuration: final 24 common months = test, prior 24 common months = validation, remainder = train.")
    rows.append("- Split manifest:")
    for row in split_manifest.itertuples(index=False):
        rows.append(
            f"  - {row.split_name}: {row.start_month_end} to {row.end_month_end}, months={row.month_count}, rows={row.row_count}, horizons={row.horizons}"
        )
    rows.append(f"- Extra filtered `60m` tail retained outside default splits: `{excluded_rows}` rows across `{excluded_months}` months.")
    rows.append("")
    rows.append("## Split Coverage")
    overall = split_summary.loc[split_summary["summary_scope"].eq("overall")]
    for row in overall.itertuples(index=False):
        rows.append(f"- {row.split_name}: rows={row.row_count}, months={row.month_count}, range={row.start_month_end}..{row.end_month_end}")
    rows.append("")
    rows.append("## Feature Sets")
    rows.append(f"- Default starting feature set: `{DEFAULT_FEATURE_SET}`.")
    overall_feature_sets = feature_set_summary.loc[feature_set_summary["summary_scope"].eq("overall")]
    for row in overall_feature_sets.itertuples(index=False):
        rows.append(
            f"- {row.feature_set_name}: features={row.feature_count}, interactions={row.interaction_feature_count}, features_with_missingness={row.features_with_missingness_count}, avg_default_split_nonmissing_share={row.avg_default_split_nonmissing_share:.3f}, min_default_split_nonmissing_share={row.min_default_split_nonmissing_share:.3f}"
        )
    rows.append("")
    rows.append("## China-Sleeve Note")
    rows.append("- `EQ_CN` is now part of the default downstream modeling branch.")
    rows.append("- Frozen `v1` and `v2` are retained only as benchmark branches, not as the active default path for new work.")
    return "\n".join(rows) + "\n"


def run_v3_modeling_preparation(*, project_root: Path, validation_months: int = 24, test_months: int = 24, min_train_months: int = 60):
    paths = default_paths(project_root)
    modeling_panel = load_parquet(paths.source_panel)
    feature_dictionary = load_csv(paths.feature_dictionary, parse_dates=["first_valid_date", "last_valid_date"])
    interaction_dictionary = load_csv(paths.interaction_dictionary, parse_dates=["first_valid_date", "last_valid_date"])

    filtered_panel = filter_firstpass_panel(modeling_panel, horizons=(60, 120))
    split_cfg = SplitConfig(
        validation_months=validation_months,
        test_months=test_months,
        min_train_months=min_train_months,
        required_horizons=(60, 120),
    )
    panel_with_splits = assign_default_splits(filtered_panel, split_cfg)
    feature_manifest = build_feature_set_manifest(panel_with_splits, feature_dictionary, interaction_dictionary)
    panel_with_splits = _attach_feature_set_row_stats(panel_with_splits, feature_manifest)

    split_frames = split_subsets(panel_with_splits)
    split_manifest = build_split_manifest(panel_with_splits)
    split_summary = build_split_summary(panel_with_splits)
    feature_set_summary = summarize_feature_sets(feature_manifest)
    report_text = _build_report(
        filtered_panel=panel_with_splits,
        split_panel=panel_with_splits,
        split_manifest=split_manifest,
        split_summary=split_summary,
        feature_set_summary=feature_set_summary,
    )

    write_parquet(panel_with_splits, paths.data_modeling_dir / "modeling_panel_firstpass.parquet")
    write_parquet(split_frames["train"], paths.data_modeling_dir / "train_split.parquet")
    write_parquet(split_frames["validation"], paths.data_modeling_dir / "validation_split.parquet")
    write_parquet(split_frames["test"], paths.data_modeling_dir / "test_split.parquet")
    write_csv(split_manifest, paths.data_modeling_dir / "split_manifest.csv")
    write_csv(feature_manifest, paths.data_modeling_dir / "feature_set_manifest.csv")
    write_csv(split_summary, paths.reports_dir / "modeling_split_summary.csv")
    write_csv(feature_set_summary, paths.reports_dir / "feature_set_summary.csv")
    write_text(report_text, paths.reports_dir / "modeling_prep_v3_report.md")

    return {
        "modeling_panel_firstpass": paths.data_modeling_dir / "modeling_panel_firstpass.parquet",
        "train_split": paths.data_modeling_dir / "train_split.parquet",
        "validation_split": paths.data_modeling_dir / "validation_split.parquet",
        "test_split": paths.data_modeling_dir / "test_split.parquet",
        "split_manifest": paths.data_modeling_dir / "split_manifest.csv",
        "feature_set_manifest": paths.data_modeling_dir / "feature_set_manifest.csv",
        "modeling_split_summary": paths.reports_dir / "modeling_split_summary.csv",
        "feature_set_summary": paths.reports_dir / "feature_set_summary.csv",
        "modeling_prep_v3_report": paths.reports_dir / "modeling_prep_v3_report.md",
    }
