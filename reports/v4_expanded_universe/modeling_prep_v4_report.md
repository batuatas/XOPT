# XOPTPOE v4_expanded_universe Modeling Preparation Report

## Filtering Applied
- Source table: `data/final_v4_expanded_universe/modeling_panel_hstack.parquet`.
- Keep rows with `baseline_trainable_flag == 1`.
- Keep rows with `target_available_flag == 1` and non-null `annualized_excess_forward_return`.
- Keep horizons `{60, 120}` only for the first-pass modeling package; `180` remains in the versioned source dataset.
- Enforce no duplicate `(sleeve_id, month_end, horizon_months)` keys.

## First-Pass Modeling Subset
- filtered_row_count: `3836`
- filtered_month_range: `2007-02-28` to `2021-02-28`
- sleeve_count: `15`
- filtered_rows_by_horizon:
  - 60m: 2368 rows
  - 120m: 1468 rows

## Default Split Design
- Split method: deterministic month-block split on the common `60m`/`120m` window only.
- Rationale: validation and test should contain both horizons and the full v4 cross-section when coverage permits.
- Default configuration: final 24 common months = test, prior 24 common months = validation, remainder = train.
- Split manifest:
  - train: 2007-02-28 to 2012-02-29, months=61, rows=1496, horizons=60,120
  - validation: 2012-03-31 to 2014-02-28, months=24, rows=720, horizons=60,120
  - test: 2014-03-31 to 2016-02-29, months=24, rows=720, horizons=60,120
  - excluded_60_only_tail: 2016-03-31 to 2021-02-28, months=60, rows=900, horizons=60
- Extra filtered `60m` tail retained outside default splits: `900` rows across `60` months.

## Split Coverage
- test: rows=720, months=24, range=2014-03-31..2016-02-29
- train: rows=1496, months=61, range=2007-02-28..2012-02-29
- validation: rows=720, months=24, range=2012-03-31..2014-02-28

## Feature Sets
- Default starting feature set: `core_plus_enrichment`.
- core_baseline: features=81, interactions=0, features_with_missingness=3, avg_default_split_nonmissing_share=1.000, min_default_split_nonmissing_share=0.994
- core_plus_enrichment: features=281, interactions=0, features_with_missingness=7, avg_default_split_nonmissing_share=0.999, min_default_split_nonmissing_share=0.912
- core_plus_interactions: features=303, interactions=22, features_with_missingness=7, avg_default_split_nonmissing_share=0.999, min_default_split_nonmissing_share=0.912
- full_firstpass: features=317, interactions=26, features_with_missingness=21, avg_default_split_nonmissing_share=0.984, min_default_split_nonmissing_share=0.551

## Locked v4 Notes
- `CR_US_IG` replaces legacy `FI_IG` naming in the v4 modeling-prep branch.
- `RE_US` and `LISTED_RE` coexist as separate sleeves.
- Euro fixed-income sleeves enter as USD-unhedged synthesized targets built from local ETF returns plus FX conversion.
