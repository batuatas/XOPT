# XOPTPOE v2_long_horizon Modeling Preparation Report

## Filtering Applied
- Source table: `data/final_v2_long_horizon/modeling_panel_hstack.parquet`.
- Keep rows with `baseline_trainable_flag == 1`.
- Keep rows with `target_available_flag == 1` and non-null `annualized_excess_forward_return`.
- Keep horizons `{60, 120}` only for the first-pass modeling package; `180` remains in the frozen source dataset.
- Enforce no duplicate `(sleeve_id, month_end, horizon_months)` keys.

## First-Pass Modeling Subset
- filtered_row_count: `2224`
- filtered_month_range: `2007-02-28` to `2021-02-28`
- filtered_rows_by_horizon:
  - 60m: 1352 rows
  - 120m: 872 rows

## Default Split Design
- Split method: deterministic month-block split on the common `60m`/`120m` window only.
- Rationale: validation and test should contain both horizons, not only late-sample `60m` rows.
- Default configuration: final 24 common months = test, prior 24 common months = validation, remainder = train.
- Split manifest:
  - train: 2007-02-28 to 2012-02-29, months=61, rows=976, horizons=60,120
  - validation: 2012-03-31 to 2014-02-28, months=24, rows=384, horizons=60,120
  - test: 2014-03-31 to 2016-02-29, months=24, rows=384, horizons=60,120
  - excluded_60_only_tail: 2016-03-31 to 2021-02-28, months=60, rows=480, horizons=60
- Extra filtered `60m` tail retained outside default splits: `480` rows across `60` months.

## Split Coverage
- test: rows=384, months=24, range=2014-03-31..2016-02-29
- train: rows=976, months=61, range=2007-02-28..2012-02-29
- validation: rows=384, months=24, range=2012-03-31..2014-02-28
- By horizon:
  - test, 60m: rows=192, months=24
  - test, 120m: rows=192, months=24
  - train, 60m: rows=488, months=61
  - train, 120m: rows=488, months=61
  - validation, 60m: rows=192, months=24
  - validation, 120m: rows=192, months=24

## Feature Sets
- Default starting feature set: `core_plus_enrichment`.
- core_baseline: features=73, interactions=0, features_with_missingness=3, avg_default_split_nonmissing_share=1.000, min_default_split_nonmissing_share=0.991
- core_plus_enrichment: features=273, interactions=0, features_with_missingness=7, avg_default_split_nonmissing_share=0.999, min_default_split_nonmissing_share=0.872
- core_plus_interactions: features=293, interactions=20, features_with_missingness=7, avg_default_split_nonmissing_share=0.999, min_default_split_nonmissing_share=0.872
- full_firstpass: features=307, interactions=24, features_with_missingness=21, avg_default_split_nonmissing_share=0.983, min_default_split_nonmissing_share=0.495
- Default feature-set block composition:
  - baseline_global_macro: features=12, interactions=0, avg_nonmissing=1.000
  - baseline_macro_canonical: features=30, interactions=0, avg_nonmissing=0.999
  - baseline_technical: features=12, interactions=0, avg_nonmissing=1.000
  - cape: features=8, interactions=0, avg_nonmissing=1.000
  - china_macro: features=36, interactions=0, avg_nonmissing=0.992
  - china_market: features=4, interactions=0, avg_nonmissing=1.000
  - china_valuation: features=18, interactions=0, avg_nonmissing=1.000
  - credit_stress: features=8, interactions=0, avg_nonmissing=1.000
  - em_global_valuation: features=18, interactions=0, avg_nonmissing=1.000
  - eu_ig_market: features=4, interactions=0, avg_nonmissing=1.000
  - horizon_conditioning: features=4, interactions=0, avg_nonmissing=1.000
  - japan_enrichment: features=52, interactions=0, avg_nonmissing=1.000
  - metadata_dummy: features=15, interactions=0, avg_nonmissing=1.000
  - oecd_bts: features=30, interactions=0, avg_nonmissing=1.000
  - oecd_leading: features=22, interactions=0, avg_nonmissing=1.000

## Deferred Interaction Design
- Deferred by default: `china_block_x_em_relevance`, `japan_block_x_jp_relevance`, `sleeve_dummy_x_predictor`.
- Deferred by name: `int_vix_x_cape_local` because it is a large-scale local-CAPE stress interaction with weaker first-pass numerical stability.
- Compatibility aliases such as `baseline_macro_alias` are excluded from first-pass feature sets because the canonical macro-state columns already carry the same information with cleaner semantics.

## Missingness-Ready Design
- Raw NaNs are preserved in `modeling_panel_firstpass.parquet`; the prep package does not force complete-case filtering.
- `feature_set_manifest.csv` carries per-feature missingness, imputation hints, and membership by feature set.
- Row-level missing-feature counts, missing-feature shares, and complete flags are attached for every prepared feature set.
- This is intended for later model code to combine explicit masking and train-only imputation without changing the frozen source data.

## Output Files
- `data/modeling_v2/modeling_panel_firstpass.parquet`
- `data/modeling_v2/train_split.parquet`
- `data/modeling_v2/validation_split.parquet`
- `data/modeling_v2/test_split.parquet`
- `data/modeling_v2/split_manifest.csv`
- `data/modeling_v2/feature_set_manifest.csv`
- `reports/v2_long_horizon/modeling_split_summary.csv`
- `reports/v2_long_horizon/feature_set_summary.csv`
- `reports/v2_long_horizon/modeling_prep_v2_report.md`

## Direct Answers
1. Final first-pass modeling subset: `2224` rows from `2007-02-28` to `2021-02-28` after `baseline_trainable_flag == 1`, `target_available_flag == 1`, non-null annualized excess target, and horizons `60/120` only.
2. Exact train/validation/test date ranges: train `2007-02-28` to `2012-02-29`, validation `2012-03-31` to `2014-02-28`, test `2014-03-31` to `2016-02-29`.
3. Rows by split: train `976`, validation `384`, test `384`; excluded `60m` tail outside default splits `480`.
4. Default starting feature set: `core_plus_enrichment`.
5. Default first-pass model input blocks: baseline technical, canonical macro, global stress, metadata/horizon conditioning, OECD leading/BTS, CAPE, Japan enrichment, EM-global valuation, credit stress, EU IG market, China market/valuation, and selected high-coverage China macro features.
6. Deferred interaction families: `china_block_x_em_relevance`, `japan_block_x_jp_relevance`, `sleeve_dummy_x_predictor`, plus named deferral of `int_vix_x_cape_local`.
7. Data ready for the actual model-building section: yes; the prepared tables now separate the usable first-pass subset, default multi-horizon splits, and missingness-aware feature-set manifests without changing the frozen v2 dataset.
