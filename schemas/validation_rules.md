# XOPTPOE v1 Validation Rules

All rules are implemented in code (`src/xoptpoe_data/qa/run_qa.py`) and emitted to:
- structured audit table: `data/final/coverage_audit.csv`
- human-readable report: `reports/qa_summary.md`

## Target QA (hard failures unless explicitly downgraded)

- No duplicate keys in `sleeve_target_raw`: (`sleeve_id`,`trade_date`)
- `adj_close` non-null and strictly positive for used rows
- No non-positive selected month-end prices
- Month-end collapse correctness: selected `trade_date` equals max trade date in that calendar month for each sleeve
- Forward returns finite and `ret_fwd_1m > -1`
- Silent ticker substitution forbidden
- Coverage summary by sleeve

## Macro QA

- No duplicate keys in `macro_raw`: (`series_id`,`obs_date`)
- `used_code` must match manifest `preferred_code` unless explicit fallback is enabled and used
- `native_frequency` must match manifest
- Canonical geo-prefixed macro schema must be present in `macro_state_panel`
- No EM local macro columns (forbidden in locked v1)
- Lag policy validation in canonical lineage columns:
  - official monthly local variables use observation month `t-1`
  - US 10Y long rate is market-observable at month-end `t`
- Stale-flag consistency in canonical macro panel (`macro_stale_flag`, `global_stale_flag`)
- Compatibility global panel stale checks remain machine-checkable

## Join QA

- No duplicate keys in `modeling_panel`: (`sleeve_id`,`month_end`)
- Feature-complete rows must have matching target rows except expected terminal no-forward month
- Missing mapped local macro rows for sleeves with local blocks must be zero
- Forbidden local block usage check:
  - `EQ_EM` must not use local geo block
  - `ALT_GLD` must not use local geo block
- Missingness summary by feature in `reports/feature_missingness.csv`
- Actual sample start/end by sleeve in `reports/coverage_by_sleeve.csv`

## Sample-start diagnostics (required reporting)

The build writes `reports/sample_start_report.csv` with these machine-checkable dates:
- first raw target date
- first month-end price date
- first feature-ready date
- first target-ready date
- first final modeling-panel date
