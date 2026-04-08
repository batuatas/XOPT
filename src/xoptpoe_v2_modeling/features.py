"""Feature-set design helpers for XOPTPOE v2 long-horizon modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_SET_ORDER: tuple[str, ...] = (
    "core_baseline",
    "core_plus_enrichment",
    "core_plus_interactions",
    "full_firstpass",
)
DEFAULT_FEATURE_SET = "core_plus_enrichment"

CORE_BASELINE_BLOCKS = {
    "baseline_technical",
    "baseline_macro_canonical",
    "baseline_global_macro",
    "metadata_dummy",
    "horizon_conditioning",
}
ENRICHMENT_BLOCKS = {
    "cape",
    "oecd_leading",
    "oecd_bts",
    "japan_enrichment",
    "em_global_valuation",
    "credit_stress",
    "eu_ig_market",
    "china_market",
    "china_valuation",
    "china_macro",
}
SAFE_INTERACTION_FAMILIES = {
    "asset_group_dummy_x_predictor",
    "cape_x_real_rate",
    "cli_x_slope_or_spread",
    "predictor_x_log_horizon_years",
    "stress_x_momentum",
    "stress_x_valuation",
}
DEFERRED_INTERACTION_FAMILIES = {
    "china_block_x_em_relevance",
    "japan_block_x_jp_relevance",
    "sleeve_dummy_x_predictor",
}
DEFERRED_INTERACTION_NAMES = {"int_vix_x_cape_local"}
EXCLUDE_FROM_ALL_FEATURE_SETS = {
    "usd_broad_level",
    "vix_level",
    "us_real10y_level",
    "ig_oas_level",
    "oil_wti_level",
}
BLOCKS_EXCLUDED_FROM_ALL_FEATURE_SETS = {
    "baseline_macro_alias",
}
MIN_ENRICHMENT_NONMISSING_SHARE = 0.85
MIN_SAFE_INTERACTION_NONMISSING_SHARE = 0.85


@dataclass(frozen=True)
class FeatureSetBuild:
    """Container for feature-set manifests and summaries."""

    feature_manifest: pd.DataFrame
    feature_summary: pd.DataFrame



def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def _classify_imputation(row: pd.Series) -> str:
    if int(row["included_in_any_set"]) == 0:
        return "not_used"
    if int(row["missing_indicator_recommended"]) == 0:
        return "none_required"
    transform = str(row.get("transform_type", ""))
    feature_name = str(row["feature_name"])
    if "dummy" in transform or feature_name.endswith("_flag"):
        return "zero_fill_with_indicator"
    return "train_median_with_indicator"



def build_feature_set_manifest(
    panel_with_splits: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    interaction_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a row-level feature manifest with feature-set membership and missingness profiles."""
    work = panel_with_splits.copy()
    split_eligible = work.loc[work["default_split_eligible_flag"].eq(1)].copy()

    feature_meta = feature_dictionary.copy()
    interaction_meta = interaction_dictionary[["interaction_name", "interaction_family"]].rename(
        columns={"interaction_name": "feature_name"}
    )
    feature_meta = feature_meta.merge(interaction_meta, on="feature_name", how="left")

    rows: list[dict[str, object]] = []
    for meta in feature_meta.to_dict("records"):
        feature_name = meta["feature_name"]
        if feature_name not in work.columns:
            continue

        full_series = _coerce_numeric(work[feature_name])
        split_series = _coerce_numeric(split_eligible[feature_name])
        full_nonmissing = float(full_series.notna().mean())
        split_nonmissing = float(split_series.notna().mean()) if len(split_series) else 0.0
        split_clean = split_series.dropna()
        split_std = float(split_clean.std(ddof=1)) if len(split_clean) > 1 else 0.0
        all_missing_split = int(split_clean.empty)
        constant_split = int((not split_clean.empty) and (split_clean.nunique(dropna=True) <= 1))

        block_name = str(meta["block_name"])
        feature_name_str = str(feature_name)
        interaction_family = meta.get("interaction_family")
        interaction_family = str(interaction_family) if pd.notna(interaction_family) else ""

        exclude_reason = ""
        if feature_name_str in EXCLUDE_FROM_ALL_FEATURE_SETS:
            exclude_reason = "duplicate_global_alias"
        elif block_name in BLOCKS_EXCLUDED_FROM_ALL_FEATURE_SETS:
            exclude_reason = "compatibility_alias_duplicate_of_canonical_state"
        elif all_missing_split:
            exclude_reason = "no_coverage_in_default_split_window"
        elif constant_split:
            exclude_reason = "constant_in_default_split_window"

        include_core_baseline = int(block_name in CORE_BASELINE_BLOCKS and not exclude_reason)

        selected_china_enrichment = not (
            block_name == "china_macro" and split_nonmissing < MIN_ENRICHMENT_NONMISSING_SHARE
        )
        include_core_plus_enrichment = int(
            include_core_baseline
            or (
                block_name in ENRICHMENT_BLOCKS
                and block_name != "interaction"
                and selected_china_enrichment
                and split_nonmissing >= MIN_ENRICHMENT_NONMISSING_SHARE
                and not exclude_reason
            )
        )

        deferred_interaction = False
        deferred_reason = ""
        if int(meta.get("is_interaction", 0)) == 1:
            if interaction_family in DEFERRED_INTERACTION_FAMILIES:
                deferred_interaction = True
                deferred_reason = f"deferred_family:{interaction_family}"
            elif feature_name_str in DEFERRED_INTERACTION_NAMES:
                deferred_interaction = True
                deferred_reason = f"deferred_name:{feature_name_str}"
            elif interaction_family not in SAFE_INTERACTION_FAMILIES:
                deferred_interaction = True
                deferred_reason = f"deferred_family:{interaction_family or 'unknown'}"

        include_core_plus_interactions = int(
            include_core_plus_enrichment
            or (
                int(meta.get("is_interaction", 0)) == 1
                and not exclude_reason
                and not deferred_interaction
                and split_nonmissing >= MIN_SAFE_INTERACTION_NONMISSING_SHARE
            )
        )

        include_full_firstpass = int(
            not exclude_reason
            and (
                int(meta.get("is_interaction", 0)) == 0
                or not deferred_interaction
            )
        )

        included_in_any_set = int(
            any(
                [
                    include_core_baseline,
                    include_core_plus_enrichment,
                    include_core_plus_interactions,
                    include_full_firstpass,
                ]
            )
        )
        missing_indicator_recommended = int(included_in_any_set and split_nonmissing < 1.0)

        rows.append(
            {
                **meta,
                "interaction_family": interaction_family,
                "filtered_subset_nonmissing_share": full_nonmissing,
                "default_split_nonmissing_share": split_nonmissing,
                "default_split_std": split_std,
                "all_missing_in_default_split_flag": all_missing_split,
                "constant_in_default_split_flag": constant_split,
                "selected_china_enrichment_flag": int(block_name != "china_macro" or selected_china_enrichment),
                "excluded_from_all_sets_reason": exclude_reason,
                "deferred_interaction_reason": deferred_reason,
                "include_core_baseline": include_core_baseline,
                "include_core_plus_enrichment": include_core_plus_enrichment,
                "include_core_plus_interactions": include_core_plus_interactions,
                "include_full_firstpass": include_full_firstpass,
                "included_in_any_set": included_in_any_set,
                "missing_indicator_recommended": missing_indicator_recommended,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest["imputation_strategy_hint"] = manifest.apply(_classify_imputation, axis=1)
    manifest["default_feature_set_flag"] = manifest[f"include_{DEFAULT_FEATURE_SET}"].astype(int)
    return manifest.sort_values(["included_in_any_set", "block_name", "feature_name"], ascending=[False, True, True]).reset_index(drop=True)



def summarize_feature_sets(feature_manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature-set counts and missingness by block and overall."""
    rows: list[dict[str, object]] = []
    for feature_set_name in FEATURE_SET_ORDER:
        include_col = f"include_{feature_set_name}"
        subset = feature_manifest.loc[feature_manifest[include_col].eq(1)].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "summary_scope": "overall",
                "block_name": "ALL",
                "feature_count": int(len(subset)),
                "interaction_feature_count": int(subset["is_interaction"].sum()),
                "features_with_missingness_count": int(subset["missing_indicator_recommended"].sum()),
                "avg_default_split_nonmissing_share": float(subset["default_split_nonmissing_share"].mean()),
                "min_default_split_nonmissing_share": float(subset["default_split_nonmissing_share"].min()),
            }
        )
        grouped = (
            subset.groupby("block_name", as_index=False)
            .agg(
                feature_count=("feature_name", "size"),
                interaction_feature_count=("is_interaction", "sum"),
                features_with_missingness_count=("missing_indicator_recommended", "sum"),
                avg_default_split_nonmissing_share=("default_split_nonmissing_share", "mean"),
                min_default_split_nonmissing_share=("default_split_nonmissing_share", "min"),
            )
            .sort_values(["feature_count", "block_name"], ascending=[False, True])
        )
        for row in grouped.to_dict("records"):
            rows.append(
                {
                    "feature_set_name": feature_set_name,
                    "summary_scope": "by_block",
                    **row,
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_set_name", "summary_scope", "block_name"]).reset_index(drop=True)



def feature_columns_for_set(feature_manifest: pd.DataFrame, feature_set_name: str) -> list[str]:
    """Return ordered feature columns for a named first-pass feature set."""
    include_col = f"include_{feature_set_name}"
    if include_col not in feature_manifest.columns:
        raise ValueError(f"Unknown feature set: {feature_set_name}")
    return feature_manifest.loc[feature_manifest[include_col].eq(1), "feature_name"].tolist()
