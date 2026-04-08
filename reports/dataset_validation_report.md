# XOPTPOE v1 Dataset Validation Report

Date: 2026-03-19

## 1. Sample Period and Effective Start
- first_raw_target_date: `2006-01-03`
- first_month_end_price_date: `2006-01-31`
- first_target_ready_date: `2006-01-31`
- first_feature_ready_date: `2007-02-28`
- first_final_modeling_panel_date: `2007-02-28`
- Assessment: effective start at 2007-02-28 is broadly consistent with 12m lookbacks and lagged macro construction (early warmup rows excluded).

## 2. Sleeve Coverage and Target Sanity
- duplicate keys: target_panel=0, feature_panel=0, modeling_panel=0
- no-return-impossible check (`ret_fwd_1m <= -100%`): 0
- extreme returns (`|ret_fwd_1m| > 50%`): 0
- month-end collapse mismatch vs raw last trading date: 0
- price alignment mismatches (`target_panel` vs month-end prices): price_t=0, price_t1=0
- formula mismatches: ret=0, rf=0, excess=0
- final-month handling policy (target ends one month before prices): True
- Per-sleeve stats are in `reports/return_summary_by_sleeve.csv`.

## 3. Feature Completeness and Missingness
- Highest missingness columns are local alias fields (`local_*`) and early-window deltas, which is expected given no-local sleeves (EQ_EM, ALT_GLD) and lookback warmup.
- Included-sample local alias missingness behaves as expected:
  - ALT_GLD: local_cpi_yoy missing=100.00%, local_3m_rate missing=100.00%, local_cpi_yoy_delta_1m missing=100.00%
  - EQ_EM: local_cpi_yoy missing=100.00%, local_3m_rate missing=100.00%, local_cpi_yoy_delta_1m missing=100.00%
  - EQ_EZ: local_cpi_yoy missing=0.00%, local_3m_rate missing=0.00%, local_cpi_yoy_delta_1m missing=0.44%
  - EQ_JP: local_cpi_yoy missing=0.00%, local_3m_rate missing=0.00%, local_cpi_yoy_delta_1m missing=0.44%
  - EQ_US: local_cpi_yoy missing=0.00%, local_3m_rate missing=0.00%, local_cpi_yoy_delta_1m missing=0.88%
  - FI_IG: local_cpi_yoy missing=0.00%, local_3m_rate missing=0.00%, local_cpi_yoy_delta_1m missing=0.88%
  - FI_UST: local_cpi_yoy missing=0.00%, local_3m_rate missing=0.00%, local_cpi_yoy_delta_1m missing=0.88%
  - RE_US: local_cpi_yoy missing=0.00%, local_3m_rate missing=0.00%, local_cpi_yoy_delta_1m missing=0.88%

## 4. Macro State Sanity
- canonical geo-prefixed columns present: True
- forbidden EM-local macro columns detected: 0
- local alias derivation mismatches: 0
- global alias derivation mismatches: 0
- Macro summary stats are in `reports/macro_state_summary.csv`.

## 5. Lag / Leakage / Alignment Sanity
- official monthly lag violations (obs >= row month_end): 0
- daily/global future-date violations: 0
- leakage heuristic (same-row `ret_1m_lag == ret_fwd_1m`): 0.000000%
- alignment check (`ret_1m_lag(t)` equals `ret_fwd_1m(t-1)`): 100.00%

## 6. Cross-Asset Feature Sanity
- relative-feature formula mismatches: {'rel_mom_vs_treasury': 0, 'rel_mom_vs_us_equity': 0, 'rel_ret_1m_vs_treasury': 0, 'rel_ret_1m_vs_us_equity': 0}
- relative-feature missingness: {'rel_mom_vs_treasury': 0.053719, 'rel_mom_vs_us_equity': 0.053719, 'rel_ret_1m_vs_treasury': 0.004132, 'rel_ret_1m_vs_us_equity': 0.004132}

## 7. Modeling Panel Sanity
- duplicates on (sleeve_id, month_end): 0
- `proxy_flag` unique values: [1]
- partial included months (not 0 and not full 8 sleeves): 1
- partial months detail:
  - 2025-11-30: included_rows=4
- `geo_block_local/global` by sleeve are consistent with mapping intent.

## 8. Economic Plausibility
- FI_UST_vol_less_than_equity_median: True
- FI_IG_vol_gt_FI_UST_vol: True
- GLD_vol_distinct_from_EQ_US: False
- Volatility ordering and return ranges look plausible for cross-asset monthly sleeves.

## 9. Concrete Findings
### [MEDIUM] Interior missing official US macro observations create a partial-sample month
- Evidence: `macro_raw` has interior NaNs at `US_CPI` and `US_UNEMP` on observation month `2025-10-01` (between valid surrounding months).
- Effect: US-local sleeves (`EQ_US`, `FI_UST`, `FI_IG`, `RE_US`) are excluded at 2025-11-30 (`sample_inclusion_flag=0` for 4/4 rows), creating a partial cross-section; related local delta fields are also missing at 2025-12-31 for these sleeves.
- Why this is a real issue: this is not a designed warmup artifact or locked mapping rule; it is a data continuity break in core monthly macro inputs.
- Smallest likely fix location: `src/xoptpoe_data/macro/build_macro_state_panel.py` in `_asof_on_month_end` / `_local_component` path.
- Minimal fix proposal: before `merge_asof`, drop monthly rows where `value` is NaN so as-of alignment falls back to the last valid observation and records positive staleness; add QA to fail/warn on interior NaNs in required official monthly series.

## 10. Fragile Assumptions
- EU/JP local official series show long stale runs in recent years (captured by `macro_stale_flag`), so modeling should either use stale flags actively or cap training window if freshness is required.
- `feature_complete_flag` allows inclusion even when some local delta fields are missing in isolated months; downstream model feature selection should account for this.

## 11. Readiness Verdict
- Not fully clean: one concrete medium-severity data continuity issue exists (US macro interior NaN propagation in Nov 2025).
- Modeling can proceed for exploratory first-pass work, but for a clean train-ready panel this issue should be fixed first.