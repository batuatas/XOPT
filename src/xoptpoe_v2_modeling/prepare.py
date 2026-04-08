"""End-to-end first-pass modeling preparation for XOPTPOE v2_long_horizon."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xoptpoe_v2_modeling.features import (
    DEFAULT_FEATURE_SET,
    FEATURE_SET_ORDER,
    build_feature_set_manifest,
    feature_columns_for_set,
    summarize_feature_sets,
)
from workspace_v4.src.xoptpoe_v2_modeling.io import load_csv, load_parquet, write_csv, write_parquet, write_text
from workspace_v4.src.xoptpoe_v2_modeling.splits import (
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
    """Filesystem layout for v2 modeling-prep outputs."""

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
        source_panel=root / "data" / "final_v2_long_horizon" / "modeling_panel_hstack.parquet",
        feature_dictionary=root / "data" / "final_v2_long_horizon" / "feature_dictionary.csv",
        interaction_dictionary=root / "data" / "final_v2_long_horizon" / "interaction_dictionary.csv",
        data_modeling_dir=root / "data" / "modeling_v2",
        reports_dir=root / "reports" / "v2_long_horizon",
    )



def _attach_feature_set_row_stats(panel: pd.DataFrame, feature_manifest: pd.DataFrame) -> pd.DataFrame:
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



def _build_report(
    *,
    filtered_panel: pd.DataFrame,
    split_panel: pd.DataFrame,
    split_manifest: pd.DataFrame,
    split_summary: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    feature_set_summary: pd.DataFrame,
) -> str:
    subset_start = pd.to_datetime(filtered_panel["month_end"]).min().date().isoformat()
    subset_end = pd.to_datetime(filtered_panel["month_end"]).max().date().isoformat()
    excluded_tail = split_panel.loc[split_panel["default_split"].eq(EXCLUDED_SPLIT_NAME)]
    excluded_months = int(excluded_tail["month_end"].nunique()) if not excluded_tail.empty else 0
    excluded_rows = int(len(excluded_tail))

    rows: list[str] = []
    rows.append("# XOPTPOE v2_long_horizon Modeling Preparation Report")
    rows.append("")
    rows.append("## Filtering Applied")
    rows.append("- Source table: `data/final_v2_long_horizon/modeling_panel_hstack.parquet`.")
    rows.append("- Keep rows with `baseline_trainable_flag == 1`.")
    rows.append("- Keep rows with `target_available_flag == 1` and non-null `annualized_excess_forward_return`.")
    rows.append("- Keep horizons `{60, 120}` only for the first-pass modeling package; `180` remains in the frozen source dataset.")
    rows.append("- Enforce no duplicate `(sleeve_id, month_end, horizon_months)` keys.")
    rows.append("")
    rows.append("## First-Pass Modeling Subset")
    rows.append(f"- filtered_row_count: `{len(filtered_panel)}`")
    rows.append(f"- filtered_month_range: `{subset_start}` to `{subset_end}`")
    rows.append("- filtered_rows_by_horizon:")
    for horizon, row_count in filtered_panel.groupby("horizon_months").size().sort_index().items():
        rows.append(f"  - {int(horizon)}m: {int(row_count)} rows")
    rows.append("")
    rows.append("## Default Split Design")
    rows.append("- Split method: deterministic month-block split on the common `60m`/`120m` window only.")
    rows.append("- Rationale: validation and test should contain both horizons, not only late-sample `60m` rows.")
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
        rows.append(
            f"- {row.split_name}: rows={row.row_count}, months={row.month_count}, range={row.start_month_end}..{row.end_month_end}"
        )
    rows.append("- By horizon:")
    by_horizon = split_summary.loc[split_summary["summary_scope"].eq("by_horizon")]
    for row in by_horizon.itertuples(index=False):
        rows.append(
            f"  - {row.split_name}, {int(row.horizon_months)}m: rows={row.row_count}, months={row.month_count}"
        )
    rows.append("")
    rows.append("## Feature Sets")
    rows.append(f"- Default starting feature set: `{DEFAULT_FEATURE_SET}`.")
    overall_feature_sets = feature_set_summary.loc[feature_set_summary["summary_scope"].eq("overall")]
    for row in overall_feature_sets.itertuples(index=False):
        rows.append(
            f"- {row.feature_set_name}: features={row.feature_count}, interactions={row.interaction_feature_count}, features_with_missingness={row.features_with_missingness_count}, avg_default_split_nonmissing_share={row.avg_default_split_nonmissing_share:.3f}, min_default_split_nonmissing_share={row.min_default_split_nonmissing_share:.3f}"
        )
    rows.append("- Default feature-set block composition:")
    default_blocks = feature_set_summary.loc[
        (feature_set_summary["feature_set_name"].eq(DEFAULT_FEATURE_SET))
        & (feature_set_summary["summary_scope"].eq("by_block"))
    ]
    for row in default_blocks.itertuples(index=False):
        rows.append(
            f"  - {row.block_name}: features={row.feature_count}, interactions={row.interaction_feature_count}, avg_nonmissing={row.avg_default_split_nonmissing_share:.3f}"
        )
    rows.append("")
    rows.append("## Deferred Interaction Design")
    rows.append("- Deferred by default: `china_block_x_em_relevance`, `japan_block_x_jp_relevance`, `sleeve_dummy_x_predictor`.")
    rows.append("- Deferred by name: `int_vix_x_cape_local` because it is a large-scale local-CAPE stress interaction with weaker first-pass numerical stability.")
    rows.append("- Compatibility aliases such as `baseline_macro_alias` are excluded from first-pass feature sets because the canonical macro-state columns already carry the same information with cleaner semantics.")
    rows.append("")
    rows.append("## Missingness-Ready Design")
    rows.append("- Raw NaNs are preserved in `modeling_panel_firstpass.parquet`; the prep package does not force complete-case filtering.")
    rows.append("- `feature_set_manifest.csv` carries per-feature missingness, imputation hints, and membership by feature set.")
    rows.append("- Row-level missing-feature counts, missing-feature shares, and complete flags are attached for every prepared feature set.")
    rows.append("- This is intended for later model code to combine explicit masking and train-only imputation without changing the frozen source data.")
    rows.append("")
    rows.append("## Output Files")
    rows.append("- `data/modeling_v2/modeling_panel_firstpass.parquet`")
    rows.append("- `data/modeling_v2/train_split.parquet`")
    rows.append("- `data/modeling_v2/validation_split.parquet`")
    rows.append("- `data/modeling_v2/test_split.parquet`")
    rows.append("- `data/modeling_v2/split_manifest.csv`")
    rows.append("- `data/modeling_v2/feature_set_manifest.csv`")
    rows.append("- `reports/v2_long_horizon/modeling_split_summary.csv`")
    rows.append("- `reports/v2_long_horizon/feature_set_summary.csv`")
    rows.append("- `reports/v2_long_horizon/modeling_prep_v2_report.md`")
    rows.append("")
    rows.append("## Direct Answers")
    rows.append(f"1. Final first-pass modeling subset: `{len(filtered_panel)}` rows from `{subset_start}` to `{subset_end}` after `baseline_trainable_flag == 1`, `target_available_flag == 1`, non-null annualized excess target, and horizons `60/120` only.")
    train_manifest = split_manifest.loc[split_manifest["split_name"].eq("train")].iloc[0]
    val_manifest = split_manifest.loc[split_manifest["split_name"].eq("validation")].iloc[0]
    test_manifest = split_manifest.loc[split_manifest["split_name"].eq("test")].iloc[0]
    rows.append(
        f"2. Exact train/validation/test date ranges: train `{train_manifest.start_month_end}` to `{train_manifest.end_month_end}`, validation `{val_manifest.start_month_end}` to `{val_manifest.end_month_end}`, test `{test_manifest.start_month_end}` to `{test_manifest.end_month_end}`."
    )
    rows.append(
        f"3. Rows by split: train `{train_manifest.row_count}`, validation `{val_manifest.row_count}`, test `{test_manifest.row_count}`; excluded `60m` tail outside default splits `{excluded_rows}`."
    )
    rows.append(f"4. Default starting feature set: `{DEFAULT_FEATURE_SET}`.")
    rows.append(
        "5. Default first-pass model input blocks: baseline technical, canonical macro, global stress, metadata/horizon conditioning, OECD leading/BTS, CAPE, Japan enrichment, EM-global valuation, credit stress, EU IG market, China market/valuation, and selected high-coverage China macro features."
    )
    rows.append(
        "6. Deferred interaction families: `china_block_x_em_relevance`, `japan_block_x_jp_relevance`, `sleeve_dummy_x_predictor`, plus named deferral of `int_vix_x_cape_local`."
    )
    rows.append(
        "7. Data ready for the actual model-building section: yes; the prepared tables now separate the usable first-pass subset, default multi-horizon splits, and missingness-aware feature-set manifests without changing the frozen v2 dataset."
    )
    return "\n".join(rows) + "\n"



def run_v2_modeling_preparation(
    *,
    project_root: Path,
    validation_months: int = 24,
    test_months: int = 24,
    min_train_months: int = 60,
) -> dict[str, Path]:
    """Run the first-pass v2 modeling preparation workflow and persist outputs."""
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
        feature_manifest=feature_manifest,
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
    write_text(report_text, paths.reports_dir / "modeling_prep_v2_report.md")

    return {
        "modeling_panel_firstpass": paths.data_modeling_dir / "modeling_panel_firstpass.parquet",
        "train_split": paths.data_modeling_dir / "train_split.parquet",
        "validation_split": paths.data_modeling_dir / "validation_split.parquet",
        "test_split": paths.data_modeling_dir / "test_split.parquet",
        "split_manifest": paths.data_modeling_dir / "split_manifest.csv",
        "feature_set_manifest": paths.data_modeling_dir / "feature_set_manifest.csv",
        "modeling_split_summary": paths.reports_dir / "modeling_split_summary.csv",
        "feature_set_summary": paths.reports_dir / "feature_set_summary.csv",
        "modeling_prep_v2_report": paths.reports_dir / "modeling_prep_v2_report.md",
    }
