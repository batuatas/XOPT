# XOPTPOE v3_long_horizon_china Modeling Preparation Report

## Filtering Applied
- Source table: `data/final_v3_long_horizon_china/modeling_panel_hstack.parquet`.
- Keep rows with `baseline_trainable_flag == 1`.
- Keep rows with `target_available_flag == 1` and non-null `annualized_excess_forward_return`.
- Keep horizons `{60, 120}` only for the first-pass modeling package; `180` remains in the versioned source dataset.
- Enforce no duplicate `(sleeve_id, month_end, horizon_months)` keys.

## First-Pass Modeling Subset
- filtered_row_count: `2502`
- filtered_month_range: `2007-02-28` to `2021-02-28`
- sleeve_count: `9`
- filtered_rows_by_horizon:
  - 60m: 1521 rows
  - 120m: 981 rows

## Default Split Design
- Split method: deterministic month-block split on the common `60m`/`120m` window only.
- Rationale: validation and test should contain both horizons and the full 9-sleeve cross-section.
- Default configuration: final 24 common months = test, prior 24 common months = validation, remainder = train.
- Split manifest:
  - train: 2007-02-28 to 2012-02-29, months=61, rows=1098, horizons=60,120
  - validation: 2012-03-31 to 2014-02-28, months=24, rows=432, horizons=60,120
  - test: 2014-03-31 to 2016-02-29, months=24, rows=432, horizons=60,120
  - excluded_60_only_tail: 2016-03-31 to 2021-02-28, months=60, rows=540, horizons=60
- Extra filtered `60m` tail retained outside default splits: `540` rows across `60` months.

## Split Coverage
- test: rows=432, months=24, range=2014-03-31..2016-02-29
- train: rows=1098, months=61, range=2007-02-28..2012-02-29
- validation: rows=432, months=24, range=2012-03-31..2014-02-28

## Feature Sets
- Default starting feature set: `core_plus_enrichment`.
- core_baseline: features=74, interactions=0, features_with_missingness=3, avg_default_split_nonmissing_share=1.000, min_default_split_nonmissing_share=0.991
- core_plus_enrichment: features=274, interactions=0, features_with_missingness=7, avg_default_split_nonmissing_share=0.999, min_default_split_nonmissing_share=0.872
- core_plus_interactions: features=294, interactions=20, features_with_missingness=7, avg_default_split_nonmissing_share=0.999, min_default_split_nonmissing_share=0.872
- full_firstpass: features=308, interactions=24, features_with_missingness=21, avg_default_split_nonmissing_share=0.983, min_default_split_nonmissing_share=0.495

## China-Sleeve Note
- `EQ_CN` is now part of the default downstream modeling branch.
- Frozen `v1` and `v2` are retained only as benchmark branches, not as the active default path for new work.
