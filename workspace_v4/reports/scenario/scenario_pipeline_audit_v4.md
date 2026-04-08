# Scenario Pipeline Audit — v4 First Pass

## Benchmark Object
- Predictor: `elastic_net__core_plus_interactions__separate_60`
- Allocator: `lambda_risk=8.0`, `kappa=0.10`, `omega_type=identity`
- 14-sleeve universe

## Macro State
- Dimension: 19 variables
- Variables: infl_US, infl_EA, infl_JP, short_rate_US, short_rate_EA, short_rate_JP, long_rate_US, long_rate_EA, long_rate_JP, term_slope_US, term_slope_EA, term_slope_JP, unemp_US, unemp_EA, ig_oas, us_real10y, vix, oil_wti, usd_broad

## VAR(1) Prior
- Fitted on training period up to 2016-02-29
- Used as Mahalanobis plausibility regularizer in all probing functions

## Per-Anchor Sanity Checks

### 2021-12-31
- EN alpha=0.0050, l1=0.50
- m0 key values: infl_US=6.90%, short_rate_US=0.05%, ig_oas=0.98, vix=17.22
- Benchmark predicted return: 3.97%
- Portfolio entropy: 1.954
- w_ALT_GLD: 0.095, w_EQ_US: 0.276

### 2022-12-31
- EN alpha=0.0050, l1=0.50
- m0 key values: infl_US=7.12%, short_rate_US=4.15%, ig_oas=1.38, vix=21.67
- Benchmark predicted return: 2.02%
- Portfolio entropy: 1.732
- w_ALT_GLD: 0.249, w_EQ_US: 0.137

### 2023-12-31
- EN alpha=0.0050, l1=0.50
- m0 key values: infl_US=3.13%, short_rate_US=5.27%, ig_oas=1.04, vix=12.45
- Benchmark predicted return: 2.19%
- Portfolio entropy: 1.658
- w_ALT_GLD: 0.244, w_EQ_US: 0.168

### 2024-12-31
- EN alpha=0.0050, l1=0.50
- m0 key values: infl_US=2.72%, short_rate_US=4.42%, ig_oas=0.82, vix=17.35
- Benchmark predicted return: 2.29%
- Portfolio entropy: 1.777
- w_ALT_GLD: 0.254, w_EQ_US: 0.192
