# XOPTPOE v1 Modeling Preparation Report

## Data Source
- source_file: `/Users/batuhanatas/Desktop/XOPTPOE/data/final/modeling_panel.csv`

## Filtering Applied
- Keep rows with `sample_inclusion_flag == 1`.
- Keep rows with `target_quality_flag == 1`.
- Keep rows with non-null `excess_ret_fwd_1m`.
- Enforce no duplicate `(sleeve_id, month_end)` keys.

## Filtered Panel Snapshot
- usable_start: `2007-02-28`
- usable_end: `2026-01-31`
- month_count: `228`
- sleeve_count: `8`
- row_count: `1824`
- rows_per_month_min_max: `8 / 8`

## Time Split Design
- Split method: deterministic month-block split, no random shuffling.
- Default configuration: final 24 months = test, prior 24 months = validation, remainder = train.
- Split manifest:
  - train: 2007-02-28 to 2022-01-31, months=180, rows=1440, sleeves=8
  - validation: 2022-02-28 to 2024-01-31, months=24, rows=192, sleeves=8
  - test: 2024-02-29 to 2026-01-31, months=24, rows=192, sleeves=8

## Split Coverage by Sleeve
- train:
  - ALT_GLD: months=180, rows=180, range=2007-02-28..2022-01-31
  - EQ_EM: months=180, rows=180, range=2007-02-28..2022-01-31
  - EQ_EZ: months=180, rows=180, range=2007-02-28..2022-01-31
  - EQ_JP: months=180, rows=180, range=2007-02-28..2022-01-31
  - EQ_US: months=180, rows=180, range=2007-02-28..2022-01-31
  - FI_IG: months=180, rows=180, range=2007-02-28..2022-01-31
  - FI_UST: months=180, rows=180, range=2007-02-28..2022-01-31
  - RE_US: months=180, rows=180, range=2007-02-28..2022-01-31
- validation:
  - ALT_GLD: months=24, rows=24, range=2022-02-28..2024-01-31
  - EQ_EM: months=24, rows=24, range=2022-02-28..2024-01-31
  - EQ_EZ: months=24, rows=24, range=2022-02-28..2024-01-31
  - EQ_JP: months=24, rows=24, range=2022-02-28..2024-01-31
  - EQ_US: months=24, rows=24, range=2022-02-28..2024-01-31
  - FI_IG: months=24, rows=24, range=2022-02-28..2024-01-31
  - FI_UST: months=24, rows=24, range=2022-02-28..2024-01-31
  - RE_US: months=24, rows=24, range=2022-02-28..2024-01-31
- test:
  - ALT_GLD: months=24, rows=24, range=2024-02-29..2026-01-31
  - EQ_EM: months=24, rows=24, range=2024-02-29..2026-01-31
  - EQ_EZ: months=24, rows=24, range=2024-02-29..2026-01-31
  - EQ_JP: months=24, rows=24, range=2024-02-29..2026-01-31
  - EQ_US: months=24, rows=24, range=2024-02-29..2026-01-31
  - FI_IG: months=24, rows=24, range=2024-02-29..2026-01-31
  - FI_UST: months=24, rows=24, range=2024-02-29..2026-01-31
  - RE_US: months=24, rows=24, range=2024-02-29..2026-01-31

## Target Diagnostics
- panel target mean/std: `0.004835 / 0.048659`
- panel target min/p95/max: `-0.317828 / 0.079019 / 0.306703`
- target summary by split (all sleeves pooled):
  - ALL: mean=0.004835, std=0.048659, p05=-0.074524, p50=0.004968, p95=0.079019
  - train: mean=0.004718, std=0.049922, p05=-0.075741, p50=0.004881, p95=0.079127
  - validation: mean=-0.001232, std=0.051954, p05=-0.076676, p50=-0.010064, p95=0.089908
  - test: mean=0.011781, std=0.032211, p05=-0.037463, p50=0.012651, p95=0.063782

## Feature Missingness and Scale
- top missingness features:
  - local_cpi_yoy_delta_1m: missing=25.33%
  - local_cpi_yoy: missing=25.00%
  - local_10y_rate_delta_1m: missing=25.00%
  - local_unemp: missing=25.00%
  - local_term_slope_delta_1m: missing=25.00%
  - local_term_slope: missing=25.00%
  - local_3m_rate_delta_1m: missing=25.00%
  - local_3m_rate: missing=25.00%
  - local_10y_rate: missing=25.00%
  - local_unemp_delta_1m: missing=25.00%
  - infl_US_delta_1m: missing=0.44%
  - infl_JP_delta_1m: missing=0.44%
- highlighted feature diagnostics:
  - ig_oas: missing=0.00%, mean=1.609474, std=0.923910, p01=0.790000, p99=6.040000
  - infl_EA: missing=0.00%, mean=2.131507, std=2.082868, p01=-0.315217, p99=9.927182
  - infl_JP: missing=0.00%, mean=0.223242, std=0.976127, p01=-2.249483, p99=3.481009
  - infl_US: missing=0.00%, mean=2.517136, std=1.950220, p01=-1.377943, p99=8.538171
  - mom_12_1: missing=0.00%, mean=0.071547, std=0.177272, p01=-0.452523, p99=0.556234
  - oil_wti: missing=0.00%, mean=72.887982, std=21.479958, p01=32.740000, p99=124.170000
  - ret_1m_lag: missing=0.00%, mean=0.005898, std=0.048623, p01=-0.128892, p99=0.126994
  - short_rate_EA: missing=0.00%, mean=1.084395, std=1.693316, p01=-0.560143, p99=4.965190
  - short_rate_JP: missing=0.00%, mean=0.259710, std=0.288292, p01=-0.072000, p99=0.880000
  - short_rate_US: missing=0.00%, mean=1.477237, std=1.827993, p01=0.010000, p99=5.300000
  - us_real10y: missing=0.00%, mean=0.733772, std=0.968067, p01=-1.070000, p99=2.600000
  - usd_broad: missing=0.00%, mean=106.582682, std=12.562212, p01=86.478600, p99=128.280500
  - vix: missing=0.00%, mean=19.951974, std=8.159394, p01=10.260000, p99=53.540000
  - vol_12m: missing=0.00%, mean=0.042241, std=0.023934, p01=0.010494, p99=0.132584

## Correlation Diagnostics
- highest absolute correlation with target:
  - oil_wti: corr=-0.1406, abs=0.1406
  - infl_US: corr=-0.1405, abs=0.1405
  - infl_JP: corr=-0.1158, abs=0.1158
  - infl_EA: corr=-0.0970, abs=0.0970
  - vol_12m: corr=0.0943, abs=0.0943
  - vix: corr=0.0689, abs=0.0689
  - usd_broad: corr=0.0601, abs=0.0601
  - short_rate_EA: corr=-0.0563, abs=0.0563
  - ret_1m_lag: corr=0.0351, abs=0.0351
  - mom_12_1: corr=-0.0344, abs=0.0344
  - us_real10y: corr=0.0200, abs=0.0200
  - short_rate_JP: corr=-0.0101, abs=0.0101
- high pairwise feature correlation (|corr| >= 0.85):
  - infl_US vs infl_EA: corr=0.8733

## Suspicious Findings
- No obvious structural red flags in filtered panel and split diagnostics.

## Output Files
- `data/modeling/modeling_panel_filtered.csv`
- `data/modeling/train_split.csv`
- `data/modeling/validation_split.csv`
- `data/modeling/test_split.csv`
- `data/modeling/split_manifest.csv`
- `reports/split_summary.csv`
- `reports/target_summary_by_sleeve.csv`
- `reports/feature_summary.csv`
- `reports/modeling_prep_report.md`
