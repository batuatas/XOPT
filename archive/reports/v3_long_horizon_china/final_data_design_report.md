# XOPTPOE v3_long_horizon_china Final Data Design Report

## Design Summary
- Frozen `v1` and `v2_long_horizon` outputs remain untouched.
- `v3_long_horizon_china` is a versioned extension that adds one investable China target sleeve.
- China sleeve choice is `EQ_CN = FXI`.
- The sleeve is added as a USD-listed ETF adjusted-close target, not from the uploaded SSE workbook.

## China Sleeve Decision
- `FXI` was chosen because it is a USD-listed investable ETF with history starting 2004-10-31.
- This is a China large-cap proxy, not a claim of perfect full-China market representation.
- `MCHI` was rejected as the default sleeve proxy for this branch because its shorter ETF history would collapse 10Y/15Y usable sample further.

## Main Table Grains
- `feature_master_monthly`: one row per (`sleeve_id`, `month_end`).
- `target_panel_long_horizon`: one row per (`sleeve_id`, `month_end`, `horizon_months`).
- `modeling_panel_hstack`: one row per (`sleeve_id`, `month_end`, `horizon_months`).

## China Feature Treatment
- `EQ_CN` gets the same technical and global baseline features as the other sleeves.
- It also consumes the existing China macro, China valuation, China market, OECD China, and China CAPE blocks already present in the v2 monthly state.
- The inherited v1 canonical local macro core remains US/EA/JP-only, so `EQ_CN` is treated as outside that frozen local-alias core for baseline completeness flags.

## Used Sources
- `country_level_macro_data/BAMLH0A0HYM2.csv`
- `country_level_macro_data/BAMLH0A0HYM2EY.csv`
- `country_level_macro_data/BAMLHE00EHYIEY.csv`
- `country_level_macro_data/BAMLHE00EHYIOAS.csv`
- `country_level_macro_data/China Price History_SSE_Composite.xlsx`
- `country_level_macro_data/China_dividend_yield.csv`
- `country_level_macro_data/China_price_to_earnings.csv`
- `country_level_macro_data/EU_IG_.xlsx`
- `country_level_macro_data/JPNLOLITONOSTSAM.csv`
- `country_level_macro_data/JaPaN GDP.csv`
- `country_level_macro_data/OECD.SDD.STES,DSD_STES@DF_BTS,4.0+CHN+JPN+EA20+USA.M........csv`
- `country_level_macro_data/china & em price-to-book ratio.csv`
- `country_level_macro_data/china_cli.csv`
- `country_level_macro_data/china_cpi_data.csv`
- `country_level_macro_data/china_fiscal_socioeconomic.csv`
- `country_level_macro_data/china_gdp_cmi.csv`
- `country_level_macro_data/china_industrial_production.csv`
- `country_level_macro_data/china_pmi.csv`
- `country_level_macro_data/em_global_div_yield.csv`
- `country_level_macro_data/em_global_ptb.csv`
- `country_level_macro_data/em_global_pte.csv`
- `country_level_macro_data/japan buyback yield.xlsx`
- `country_level_macro_data/japan_10y_bond_yield_cpi.csv`
- `country_level_macro_data/japan_macro_fiscal_socioeconomic.csv`
- `country_level_macro_data/japan_pe_topix_index.csv`
- `data/external/akif_candidates/Historic-cape-ratios (1).csv`
- `data/external/akif_candidates/OECD Leading Indicatorscsv.csv`
- `Yahoo Finance adjusted-close history for FXI`

## Usable Row Counts
- 60 months: total_target_rows=1639, total_baseline_trainable_rows=1521, china_target_rows=183
- 120 months: total_target_rows=1099, total_baseline_trainable_rows=981, china_target_rows=123
- 180 months: total_target_rows=559, total_baseline_trainable_rows=441, china_target_rows=63

## Important Limitation
- Adding FXI does not back-extend the frozen baseline macro core earlier than 2006.
- The pre-2006 FXI history is still useful for China technical lookbacks and forward target continuity, but the monthly feature store remains anchored to the baseline macro window.