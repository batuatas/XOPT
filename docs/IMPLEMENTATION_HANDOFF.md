# OUTPUT 2 — MACHINE-READABLE SERIES MANIFEST

## Controlled Refactor Note (Macro Schema)

For the controlled macro-schema refactor, implementation treats the canonical monthly macro state as a geo-prefixed wide vector:
- US: `infl_US, unemp_US, short_rate_US, long_rate_US, term_slope_US`
- EA: `infl_EA, unemp_EA, short_rate_EA, long_rate_EA, term_slope_EA`
- JP: `infl_JP, unemp_JP, short_rate_JP, long_rate_JP, term_slope_JP`
- Global: `usd_broad, vix, us_real10y, ig_oas, oil_wti`

Mapped sleeve aliases (`local_*`, `*_level`) remain as compatibility/convenience fields derived from canonical state.

## A. `target_series_manifest`

| sleeve_id | ticker | sleeve_name | economic_definition | provider_type | primary_source | secondary_source | monthly_price_rule | return_formula | proxy_flag | start_date_target | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EQ_US | VTI | US equities | Broad US total stock market exposure | ETF issuer + market history | Yahoo Finance adjusted-close history for VTI | Vanguard VTI fund page | last available trading-day adjusted close in each calendar month | `adj_close_t1 / adj_close_t - 1` | 1 | 2006-01-31 | investable sleeve proxy; USD-listed; no FX conversion |
| EQ_EZ | EZU | Euro area equities | Developed-market eurozone large/mid-cap equities | ETF issuer + market history | Yahoo Finance adjusted-close history for EZU | BlackRock EZU fund page | same | same | 1 | 2006-01-31 | locked Europe definition is euro area, not generic Europe |
| EQ_JP | EWJ | Japan equities | Broad Japanese equity exposure | ETF issuer + market history | Yahoo Finance adjusted-close history for EWJ | BlackRock EWJ fund page | same | same | 1 | 2006-01-31 | USD unhedged exposure embeds JPY/USD effect |
| EQ_EM | VWO | EM equities | Broad emerging-market equities | ETF issuer + market history | Yahoo Finance adjusted-close history for VWO | Vanguard VWO fund page | same | same | 1 | 2006-01-31 | no separate EM macro block in v1 |
| FI_UST | IEF | US Treasuries 7–10Y | Intermediate US Treasury exposure | ETF issuer + market history | Yahoo Finance adjusted-close history for IEF | iShares IEF fund page | same | same | 1 | 2006-01-31 | locked v1 target uses ETF, not benchmark Treasury index |
| FI_IG | LQD | US IG credit | USD investment-grade corporate bond exposure | ETF issuer + market history | Yahoo Finance adjusted-close history for LQD | iShares LQD fund page | same | same | 1 | 2006-01-31 | ETF proxy; no separate bond-index target layer in v1 |
| ALT_GLD | GLD | Gold | Gold-bullion-linked exposure | ETF issuer + market history | Yahoo Finance adjusted-close history for GLD | SPDR GLD fund page | same | same | 1 | 2006-01-31 | investable gold sleeve; no spot-gold target series in v1 |
| RE_US | VNQ | US REITs | US listed real-estate exposure | ETF issuer + market history | Yahoo Finance adjusted-close history for VNQ | Vanguard VNQ fund page | same | same | 1 | 2006-01-31 | REIT sleeve proxy; listed real assets |

## B. `macro_series_manifest`

| series_id | variable_name | geo_block | source_provider | preferred_code | fallback_code | native_frequency | transform_rule | lag_rule | scenario_manipulable_flag | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| US_CPI | us_cpi | US | FRED/BLS | CPIAUCSL | none | monthly | `infl_us_yoy = 100*(x / lag12(x) - 1)` | use observation month t-1 at row t | 1 | headline CPI level source |
| US_UNEMP | us_unemp | US | FRED/BLS | UNRATE | none | monthly | level | use observation month t-1 at row t | 1 | unemployment rate |
| US_RF3M | us_3m | US | FRED/Board of Governors | TB3MS | DGS3MO | monthly | level; also `rf_1m = x/1200` for target construction | use observation month t-1 at row t for features; use t+1 in target subtraction | 1 | dual use: state short rate + risk-free source |
| US_10Y | us_10y | US | FRED/Board of Governors | DGS10 | GS10 | daily | month-end level | month-end observable at row t | 1 | treated as market-observable rate series |
| EA_CPI | ea_cpi | EURO_AREA | FRED/Eurostat | CP0000EZ19M086NEST | Eurostat direct HICP | monthly | `infl_ea_yoy = 100*(x / lag12(x) - 1)` | use observation month t-1 at row t | 1 | euro area HICP |
| EA_UNEMP | ea_unemp | EURO_AREA | FRED/OECD | LRHUTTTTEZM156S | OECD direct same concept | monthly | level | use observation month t-1 at row t | 1 | harmonized unemployment |
| EA_3M | ea_3m | EURO_AREA | FRED/OECD | IR3TIB01EZM156N | OECD direct same concept | monthly | level | use observation month t-1 at row t | 1 | euro-area 3m interbank rate |
| EA_10Y | ea_10y | EURO_AREA | FRED/OECD | IRLTLT01EZM156N | OECD direct same concept | monthly | level | use observation month t-1 at row t | 1 | euro-area 10y government yield |
| JP_CPI | jp_cpi | JAPAN | FRED/OECD | JPNCPIALLMINMEI | OECD direct same concept | monthly | `infl_jp_yoy = 100*(x / lag12(x) - 1)` | use observation month t-1 at row t | 1 | Japan CPI |
| JP_UNEMP | jp_unemp | JAPAN | FRED/OECD | LRHUTTTTJPM156S | OECD direct same concept | monthly | level | use observation month t-1 at row t | 1 | Japan unemployment |
| JP_3M | jp_3m | JAPAN | FRED/OECD | IR3TIB01JPM156N | OECD direct same concept | monthly | level | use observation month t-1 at row t | 1 | Japan 3m interbank rate |
| JP_10Y | jp_10y | JAPAN | FRED/OECD | IRLTLT01JPM156N | OECD direct same concept | monthly | level | use observation month t-1 at row t | 1 | Japan 10y government yield |
| USD_BROAD | usd_broad | GLOBAL | FRED/Board of Governors | DTWEXBGS | Fed H.10 direct | daily | month-end level; `d1 = log(x/lag1(x))`; `d12 = log(x/lag12(x))` | month-end observable at row t | 1 | USD block |
| VIX | vix | GLOBAL | FRED/Cboe | VIXCLS | Cboe direct | daily | month-end level; optional 1m delta | month-end observable at row t | 1 | stress block |
| US_REAL10Y | us_real10y | GLOBAL | FRED/Board of Governors | DFII10 | T10YIE only if DFII10 unavailable | daily | month-end level; optional 1m delta | month-end observable at row t | 1 | gold/rates opportunity-cost variable |
| IG_OAS | ig_oas | GLOBAL | FRED/ICE BofA | BAMLC0A0CM | BAA10YM | daily | month-end level; optional 1m delta | month-end observable at row t | 1 | credit-stress variable |
| OIL_WTI | oil_wti | GLOBAL | FRED/EIA | DCOILWTICO | MCOILWTICO | daily | month-end level; `d1 = log(x/lag1(x))`; `d12 = log(x/lag12(x))` | month-end observable at row t | 1 | oil / commodity shock proxy |

# OUTPUT 3 — FINAL TABLE SPECS

## `asset_master`

- **grain:** one row per sleeve
- **primary key:** `sleeve_id`

**required columns**
- `sleeve_id`
- `ticker`
- `sleeve_name`
- `economic_definition`
- `exposure_region`
- `target_currency`
- `provider_type`
- `proxy_flag`
- `locked_flag`
- `start_date_target`
- `notes`

**optional columns**
- `issuer_name`
- `issuer_page_url`
- `benchmark_name`
- `inception_date_fund`
- `phase`
- `deprecation_flag`

**validation rules**
- exactly 8 rows in locked v1
- `sleeve_id` unique
- `ticker` unique
- `target_currency = 'USD'` for all rows
- `proxy_flag = 1` for all rows
- no nulls in required columns

---

## `source_master`

- **grain:** one row per source or provider object
- **primary key:** `source_id`

**required columns**
- `source_id`
- `provider_name`
- `dataset_name`
- `source_type`
- `access_method`
- `frequency_native`
- `url_or_endpoint_note`
- `fallback_source_id`
- `notes`

**optional columns**
- `license_note`
- `citation_note`
- `update_schedule_note`
- `owner`
- `stability_risk`

**validation rules**
- `source_id` unique
- every source referenced elsewhere must exist here
- `fallback_source_id` either null or valid `source_id`

---

## `sleeve_target_raw`

- **grain:** one row per sleeve per trade date
- **primary key:** (`sleeve_id`, `trade_date`)

**required columns**
- `sleeve_id`
- `trade_date`
- `adj_close`
- `close`
- `source_id`
- `download_timestamp`

**optional columns**
- `open`
- `high`
- `low`
- `volume`
- `currency`
- `split_flag`
- `dividend_flag`
- `raw_payload_hash`

**validation rules**
- no duplicate (`sleeve_id`, `trade_date`)
- `adj_close > 0`
- `close > 0` when present
- `trade_date` strictly increasing within sleeve
- at least one valid observation in every month from `start_date_target`
- no missing `adj_close` on selected month-end proxy dates

---

## `macro_raw`

- **grain:** one row per source series per observation date
- **primary key:** (`series_id`, `obs_date`)

**required columns**
- `series_id`
- `variable_name`
- `geo_block`
- `obs_date`
- `value`
- `native_frequency`
- `source_id`
- `download_timestamp`

**optional columns**
- `release_name`
- `units`
- `seasonal_adjustment`
- `release_date`
- `available_date`
- `vintage_date`
- `notes`

**validation rules**
- no duplicate (`series_id`, `obs_date`)
- `value` numeric
- `native_frequency` consistent with manifest
- `obs_date` increasing within series
- no silent code substitution

---

## `macro_state_panel`

> Implementation note: for the controlled macro-schema refactor, canonical production schema is geo-prefixed wide monthly state (see `schemas/table_specs.md`). Legacy `local_*` style names are retained only as derived compatibility aliases in `feature_panel`.

- **grain:** one row per month-end per geo block
- **primary key:** (`month_end`, `geo_block`)

**required columns**
- `month_end`
- `geo_block`
- `local_cpi_yoy`
- `local_unemp`
- `local_3m_rate`
- `local_10y_rate`
- `local_term_slope`
- `usd_broad_level`
- `usd_broad_logchg_1m`
- `usd_broad_logchg_12m`
- `vix_level`
- `vix_delta_1m`
- `us_real10y_level`
- `us_real10y_delta_1m`
- `ig_oas_level`
- `ig_oas_delta_1m`
- `oil_wti_level`
- `oil_wti_logchg_1m`
- `oil_wti_logchg_12m`
- `lag_policy_tag`

**optional columns**
- `local_cpi_yoy_delta_1m`
- `local_unemp_delta_1m`
- `local_3m_rate_delta_1m`
- `local_10y_rate_delta_1m`
- `local_term_slope_delta_1m`
- `macro_stale_flag`
- `source_coverage_note`

**validation rules**
- one row per (`month_end`, `geo_block`)
- `geo_block` in {US, EURO_AREA, JAPAN, GLOBAL}
- lagged monthly official macro must reflect the locked one-month rule
- month-end daily series must use last available trading day in month
- no future information at row date

---

## `feature_panel`

- **grain:** one row per sleeve per month-end
- **primary key:** (`sleeve_id`, `month_end`)

**required columns**
- `sleeve_id`
- `month_end`
- `ret_1m_lag`
- `ret_3m_lag`
- `ret_6m_lag`
- `ret_12m_lag`
- `mom_12_1`
- `vol_3m`
- `vol_12m`
- `maxdd_12m`
- `local_cpi_yoy`
- `local_unemp`
- `local_3m_rate`
- `local_10y_rate`
- `local_term_slope`
- `usd_broad_level`
- `usd_broad_logchg_1m`
- `vix_level`
- `us_real10y_level`
- `ig_oas_level`
- `oil_wti_level`
- `rel_mom_vs_treasury`
- `rel_mom_vs_us_equity`
- `feature_complete_flag`

**optional columns**
- all macro 1m deltas
- 12m macro changes
- stress interaction features
- Shiller block fields
- BIS add-on fields
- z-scored versions
- winsorized versions

**validation rules**
- no duplicate (`sleeve_id`, `month_end`)
- no use of forward returns in feature columns
- `vol_* >= 0`
- `maxdd_12m <= 0`
- all derived features reproducible from raw inputs
- missingness summarized in `coverage_audit`

---

## `target_panel`

- **grain:** one row per sleeve per month-end
- **primary key:** (`sleeve_id`, `month_end`)

**required columns**
- `sleeve_id`
- `month_end`
- `price_t`
- `price_t1`
- `ret_fwd_1m`
- `rf_fwd_1m`
- `excess_ret_fwd_1m`
- `target_quality_flag`

**optional columns**
- `return_calc_method`
- `next_month_end`
- `rf_source_id`
- `notes`

**validation rules**
- `price_t > 0`
- `price_t1 > 0`
- `ret_fwd_1m > -1`
- `rf_fwd_1m` finite
- `excess_ret_fwd_1m = ret_fwd_1m - rf_fwd_1m`
- final month without forward target must be excluded or flagged

---

## `modeling_panel`

- **grain:** one row per sleeve per month-end
- **primary key:** (`sleeve_id`, `month_end`)

**required columns**
- all required feature columns
- `excess_ret_fwd_1m`
- `proxy_flag`
- `geo_block_local`
- `geo_block_global`
- `sample_inclusion_flag`

**optional columns**
- split labels (`train_flag`, `val_flag`, `test_flag`)
- winsorization flags
- normalization version tags
- feature set version
- experiment tags

**validation rules**
- no duplicate primary keys
- rows only where both features and target exist
- `proxy_flag = 1` for all v1 rows
- local/global block assignments consistent with `macro_mapping`

---

## `macro_mapping`

- **grain:** one row per sleeve per block role
- **primary key:** (`sleeve_id`, `block_role`)

**required columns**
- `sleeve_id`
- `block_role`
- `geo_block`
- `mapping_priority`
- `notes`

**optional columns**
- `weight`
- `active_flag`

**validation rules**
- every sleeve must have:
  - one `global` block,
  - one `usd` applicability tag in notes or explicit role,
  - at most one `local` block
- EQ_EM must have **no** dedicated EM local block
- ALT_GLD must have no local block

---

## `coverage_audit`

- **grain:** one row per table/object/field audit result
- **primary key:** (`table_name`, `object_id`, `audit_name`)

**required columns**
- `table_name`
- `object_id`
- `audit_name`
- `audit_result`
- `audit_value`
- `audit_timestamp`

**optional columns**
- `expected_start`
- `actual_start`
- `expected_end`
- `actual_end`
- `missing_share`
- `duplicate_count`
- `notes`

**validation rules**
- one row per executed audit
- audit names standardized
- failed audits must be machine-detectable

# OUTPUT 4 — FEATURE DICTIONARY

The v1 modeling table should stay disciplined. Features below are the locked core plus a short add-on layer.

## A. Target-related

| feature_name | description | source dependency | transformation | priority | POE role |
|---|---|---|---|---|---|
| ret_fwd_1m | next-month sleeve total return | sleeve_target_raw | `price_t1 / price_t - 1` | must-have | target only |
| rf_fwd_1m | next-month monthly rf approximation | TB3MS | `TB3MS_{t+1}/1200` | must-have | target only |
| excess_ret_fwd_1m | next-month excess return | ret_fwd_1m + rf_fwd_1m | `ret_fwd_1m - rf_fwd_1m` | must-have | target only |

## B. Asset technical

| feature_name | description | source dependency | transformation | priority | POE role |
|---|---|---|---|---|---|
| ret_1m_lag | prior 1-month sleeve return | sleeve_target_raw | lagged month-end return | must-have | derived |
| ret_3m_lag | trailing 3-month cumulative return | sleeve_target_raw | cumulative trailing return | must-have | derived |
| ret_6m_lag | trailing 6-month cumulative return | sleeve_target_raw | cumulative trailing return | must-have | derived |
| ret_12m_lag | trailing 12-month cumulative return | sleeve_target_raw | cumulative trailing return | must-have | derived |
| mom_12_1 | 12-1 momentum | sleeve_target_raw | cumulative return months t-12 to t-1 excluding t | must-have | derived |
| vol_3m | trailing 3-month realized volatility | sleeve_target_raw | std of monthly returns over past 3 months | must-have | derived |
| vol_12m | trailing 12-month realized volatility | sleeve_target_raw | std of monthly returns over past 12 months | must-have | derived |
| maxdd_12m | trailing 12-month max drawdown | sleeve_target_raw | rolling peak-to-trough drawdown | must-have | derived |

## C. Asset carry / rate-sensitive

| feature_name | description | source dependency | transformation | priority | POE role |
|---|---|---|---|---|---|
| local_3m_rate | mapped local short rate | macro_state_panel | level | must-have | manipulable |
| local_10y_rate | mapped local long rate | macro_state_panel | level | must-have | manipulable |
| local_term_slope | mapped local curve slope | macro_state_panel | `local_10y_rate - local_3m_rate` | must-have | derived from manipulable |
| us_real10y_level | US 10Y real yield | DFII10 | month-end level | must-have | manipulable |
| ig_oas_level | US IG corporate OAS | BAMLC0A0CM | month-end level | must-have | manipulable |

## D. Macro levels

| feature_name | description | source dependency | transformation | priority | POE role |
|---|---|---|---|---|---|
| local_cpi_yoy | mapped local inflation rate | CPI series by block | YoY pct change | must-have | manipulable |
| local_unemp | mapped local unemployment rate | unemployment series by block | level | must-have | manipulable |
| usd_broad_level | broad USD index | DTWEXBGS | month-end level | must-have | manipulable |
| vix_level | VIX index level | VIXCLS | month-end level | must-have | manipulable |
| oil_wti_level | WTI crude oil level | DCOILWTICO | month-end level | must-have | manipulable |

## E. Macro changes

| feature_name | description | source dependency | transformation | priority | POE role |
|---|---|---|---|---|---|
| local_cpi_yoy_delta_1m | 1m change in local inflation YoY | local_cpi_yoy | first difference | must-have | derived from manipulable |
| local_unemp_delta_1m | 1m change in local unemployment | local_unemp | first difference | must-have | derived from manipulable |
| local_3m_rate_delta_1m | 1m change in local short rate | local_3m_rate | first difference | must-have | derived from manipulable |
| local_10y_rate_delta_1m | 1m change in local 10Y rate | local_10y_rate | first difference | must-have | derived from manipulable |
| local_term_slope_delta_1m | 1m change in local slope | local_term_slope | first difference | must-have | derived from manipulable |
| usd_broad_logchg_1m | 1m USD change | DTWEXBGS | log change | must-have | derived from manipulable |
| usd_broad_logchg_12m | 12m USD change | DTWEXBGS | log change | nice-to-have | derived from manipulable |
| vix_delta_1m | 1m change in VIX | VIXCLS | first difference | must-have | derived from manipulable |
| us_real10y_delta_1m | 1m change in real 10Y yield | DFII10 | first difference | must-have | derived from manipulable |
| ig_oas_delta_1m | 1m change in IG OAS | BAMLC0A0CM | first difference | must-have | derived from manipulable |
| oil_wti_logchg_1m | 1m oil change | DCOILWTICO | log change | must-have | derived from manipulable |
| oil_wti_logchg_12m | 12m oil change | DCOILWTICO | log change | nice-to-have | derived from manipulable |

## F. Stress / global

| feature_name | description | source dependency | transformation | priority | POE role |
|---|---|---|---|---|---|
| stress_usd_vix_interaction | joint USD-strength / equity-stress term | DTWEXBGS + VIXCLS | standardized product term | nice-to-have | derived |
| stress_oas_vix_interaction | joint credit-stress / equity-stress term | BAMLC0A0CM + VIXCLS | standardized product term | nice-to-have | derived |
| gold_realrate_interaction | gold opportunity-cost interaction | DFII10 + sleeve identity | product term or sleeve-specific use | nice-to-have | derived |

## G. Cross-asset relative features

These features are built after all sleeve monthly returns exist.

| feature_name | description | source dependency | transformation | priority | POE role |
|---|---|---|---|---|---|
| rel_mom_vs_treasury | current sleeve momentum minus IEF momentum | sleeve_target_raw | `mom_12_1(sleeve) - mom_12_1(IEF)` | must-have | derived |
| rel_mom_vs_us_equity | current sleeve momentum minus VTI momentum | sleeve_target_raw | `mom_12_1(sleeve) - mom_12_1(VTI)` | must-have | derived |
| rel_ret_1m_vs_treasury | sleeve 1m lagged return minus IEF 1m lagged return | sleeve_target_raw | difference | nice-to-have | derived |
| rel_ret_1m_vs_us_equity | sleeve 1m lagged return minus VTI 1m lagged return | sleeve_target_raw | difference | nice-to-have | derived |

## H. Quality / control flags

| feature_name | description | source dependency | transformation | priority | POE role |
|---|---|---|---|---|---|
| proxy_flag | sleeve target is proxy-based | asset_master | copy-through | must-have | control only |
| feature_complete_flag | all required core features present | feature_panel build | boolean | must-have | control only |
| macro_stale_flag | macro row used a stale carried-forward value beyond tolerance | macro_state_panel build | boolean | must-have | control only |
| sample_inclusion_flag | row passes all inclusion rules | modeling join stage | boolean | must-have | control only |
| lag_policy_tag | records locked lag policy version | macro_state_panel | string tag | must-have | control only |

# OUTPUT 5 — CODING-AGENT BUILD BRIEF

## Build brief for the coding agent

You are implementing the locked v1 dataset repository for XOPTPOE. The design is fixed. Do not silently substitute assets, sources, timing rules, or mappings.

### 1) Build this first
Start with the target layer.

1. create `asset_master` with exactly these 8 sleeves and tickers:
   - VTI
   - EZU
   - EWJ
   - VWO
   - IEF
   - LQD
   - GLD
   - VNQ

2. fetch daily price history for all 8 sleeves into `sleeve_target_raw`
3. validate adjusted-close coverage from at least 2006-01 onward
4. collapse daily prices to month-end prices
5. build forward monthly returns

Do not fetch macro data before confirming the sleeve layer is intact.

### 2) Validate this first
The first QA gate is the target layer.

Required checks:
- all 8 tickers present
- no duplicate trade dates within ticker
- adjusted close exists on each selected month-end proxy date
- month-end collapse uses last available trading day in calendar month
- first usable month-end is at or before 2006-01-31
- forward return formula works and does not create impossible values

If any of these fail, stop and fix the target layer before doing macro work.

### 3) These design choices are locked and must not change silently
Do not change any of the following without an explicit design update:

- use **EZU**, not VGK or another Europe proxy
- use **VWO**, not a China sleeve and not an EM local macro block
- use **USD-denominated ETF adjusted closes** for all sleeve targets
- do **not** add an FX conversion layer to target construction
- use **TB3MS/1200** for the monthly rf approximation
- row date is **calendar month-end**
- target is **next-month excess return**
- official monthly macro uses **one observation month lag**
- daily market-state series use **month-end t**
- use **latest revised macro values**
- downstream intent is long-only, fully invested; no leverage assumptions in data layer

### 4) Modularize these parts
Keep the repo modular at the following boundaries:

- `targets/`
  - raw sleeve fetch
  - daily-to-month-end collapse
  - forward return builder

- `macro/`
  - raw macro fetch
  - lag-policy application
  - state-panel builder

- `features/`
  - asset technical features
  - macro feature merge
  - relative-feature builder

- `qa/`
  - coverage checks
  - duplicate-key checks
  - lag leakage checks
  - missingness reports

- `schemas/`
  - manifest loader
  - table column specs
  - validation rules

### 5) Mandatory QA checks
Implement QA as code, not as ad hoc notebook inspection.

Minimum required checks:

**Target QA**
- duplicate primary keys
- missing adjusted close
- non-positive prices
- month-end collapse correctness
- impossible returns

**Macro QA**
- duplicate primary keys
- series code matches manifest
- native frequency matches manifest
- lag policy applied correctly
- no future observation used at row date
- no stale data carried beyond allowed policy without flag

**Join QA**
- every modeling row has exactly one target row
- every modeling row has exactly one mapped macro row
- no duplicate (`sleeve_id`, `month_end`)
- missingness rates by feature
- audit of actual sample start / end by sleeve

### 6) Failure modes to watch for
The main failure modes are:

- using the wrong Europe sleeve
- silently pulling close instead of adjusted close
- collapsing to calendar month-end instead of last trading day
- using current-month official macro instead of the locked one-month lag
- mixing daily month-end market series with lagged monthly macro incorrectly
- accidentally creating an EM local macro block
- computing rf from the wrong month in the target subtraction
- leaking future month-end prices into lagged technical features
- silently replacing unavailable series with different codes

### 7) What to defer if fragile
Do not block v1 on any of the following:

- Shiller add-on block
- BIS quarterly credit block
- benchmark-index replacement of ETF targets
- institution benchmark metadata
- standalone China sleeve
- commodities ex-gold
- vintage-aware macro

If these are fragile, leave them out and keep v1 core stable.

### 8) Recommended implementation sequence
1. create manifests
2. create schema validators
3. build `asset_master`
4. fetch `sleeve_target_raw`
5. run sleeve coverage QA
6. collapse to month-end prices
7. build raw return series
8. fetch TB3MS and build rf series
9. fetch `macro_raw`
10. apply lag policy
11. build `macro_state_panel`
12. build `feature_panel`
13. build `target_panel`
14. build `macro_mapping`
15. join `modeling_panel`
16. run `coverage_audit`
17. freeze v1 outputs

### 9) Definition of done
The implementation is done when:

- all 10 locked tables are produced,
- the modeling panel is reproducible from manifests,
- the locked timing convention is enforced,
- coverage and leakage QA pass,
- and no locked design choice has been silently altered.

This is the final repo-ready implementation specification.
