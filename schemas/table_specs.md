# XOPTPOE v1 Table Specs

`docs/LOCKED_V1_SPEC.md` is authoritative for locked economic/content choices.

## Controlled Macro-Schema Refactor

This repository applies a controlled schema refactor for macro state representation:
- canonical macro state is now a **wide monthly geo-prefixed vector** (`macro_state_panel`)
- sleeve-level mapped aliases (`local_*`, `*_level`) remain for modeling convenience
- sleeve universe, target construction, lag rules, and mapping restrictions remain unchanged

## Output Tables

### `asset_master`
- Grain: one row per sleeve
- PK: `sleeve_id`
- Required: `sleeve_id,ticker,sleeve_name,economic_definition,exposure_region,target_currency,provider_type,proxy_flag,locked_flag,start_date_target,notes`

### `source_master`
- Grain: one row per data provider/source object
- PK: `source_id`
- Required: `source_id,provider_name,dataset_name,source_type,access_method,frequency_native,url_or_endpoint_note,fallback_source_id,notes`

### `sleeve_target_raw`
- Grain: one row per sleeve per trade date
- PK: (`sleeve_id`,`trade_date`)
- Required: `sleeve_id,trade_date,adj_close,close,source_id,download_timestamp`
- Optional used in implementation: `ticker,open,high,low,volume`

### `macro_raw`
- Grain: one row per source series per observation date
- PK: (`series_id`,`obs_date`)
- Required: `series_id,variable_name,geo_block,obs_date,value,native_frequency,source_id,download_timestamp`
- Optional used in implementation: `preferred_code,fallback_code,used_code,fallback_used,state_var_name`

### `macro_state_panel` (canonical)
- Grain: one row per month-end
- PK: `month_end`
- Required canonical local state columns:
  - US: `infl_US,unemp_US,short_rate_US,long_rate_US,term_slope_US`
  - EA: `infl_EA,unemp_EA,short_rate_EA,long_rate_EA,term_slope_EA`
  - JP: `infl_JP,unemp_JP,short_rate_JP,long_rate_JP,term_slope_JP`
- Required canonical global state columns:
  - `usd_broad,vix,us_real10y,ig_oas,oil_wti`
- Required change columns:
  - local deltas: `infl_*_delta_1m,unemp_*_delta_1m,short_rate_*_delta_1m,long_rate_*_delta_1m,term_slope_*_delta_1m`
  - global changes: `usd_broad_logchg_1m,usd_broad_logchg_12m,vix_delta_1m,us_real10y_delta_1m,ig_oas_delta_1m,oil_wti_logchg_1m,oil_wti_logchg_12m`
- Required controls: `global_stale_flag,macro_stale_flag,lag_policy_tag`
- Required lineage columns (for lag QA):
  - `src_obs_month_end_cpi_US,src_obs_month_end_unemp_US,src_obs_month_end_short_rate_US,src_obs_month_end_long_rate_US`
  - `src_obs_month_end_cpi_EA,src_obs_month_end_unemp_EA,src_obs_month_end_short_rate_EA,src_obs_month_end_long_rate_EA`
  - `src_obs_month_end_cpi_JP,src_obs_month_end_unemp_JP,src_obs_month_end_short_rate_JP,src_obs_month_end_long_rate_JP`

### `global_state_panel` (compatibility intermediate)
- Grain: one row per month-end
- PK: `month_end`
- Purpose: backward-compatible global alias layer derived from canonical `macro_state_panel`
- Required columns:
  - `usd_broad_level,usd_broad_logchg_1m,usd_broad_logchg_12m`
  - `vix_level,vix_delta_1m`
  - `us_real10y_level,us_real10y_delta_1m`
  - `ig_oas_level,ig_oas_delta_1m`
  - `oil_wti_level,oil_wti_logchg_1m,oil_wti_logchg_12m`
  - `global_stale_flag`

### `feature_panel`
- Grain: one row per sleeve per month-end
- PK: (`sleeve_id`,`month_end`)
- Required columns:
  - technicals: `ret_1m_lag,ret_3m_lag,ret_6m_lag,ret_12m_lag,mom_12_1,vol_3m,vol_12m,maxdd_12m`
  - canonical macro pass-through: geo-prefixed columns from `macro_state_panel`
  - mapped local aliases: `local_cpi_yoy,local_unemp,local_3m_rate,local_10y_rate,local_term_slope`
  - mapped global aliases: `usd_broad_level,vix_level,us_real10y_level,ig_oas_level,oil_wti_level`
  - relative: `rel_mom_vs_treasury,rel_mom_vs_us_equity`
  - controls: `feature_complete_flag,macro_stale_flag,lag_policy_tag,local_geo_block_used`

### `target_panel`
- Grain: one row per sleeve per month-end
- PK: (`sleeve_id`,`month_end`)
- Required: `sleeve_id,month_end,price_t,price_t1,ret_fwd_1m,rf_fwd_1m,excess_ret_fwd_1m,target_quality_flag`
- Optional: `next_month_end`

### `macro_mapping`
- Grain: one row per sleeve per block role
- PK: (`sleeve_id`,`block_role`)
- Required: `sleeve_id,block_role,geo_block,mapping_priority,notes`

### `modeling_panel`
- Grain: one row per sleeve per month-end
- PK: (`sleeve_id`,`month_end`)
- Required: all required feature columns + `excess_ret_fwd_1m,proxy_flag,geo_block_local,geo_block_global,sample_inclusion_flag`

### `coverage_audit`
- Grain: one row per audit result
- PK: (`table_name`,`object_id`,`audit_name`)
- Required: `table_name,object_id,audit_name,audit_result,audit_value,audit_timestamp`
- Optional: `expected_start,actual_start,expected_end,actual_end,missing_share,duplicate_count,notes`
