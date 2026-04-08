# XOPTPOE v4_expanded_universe Final Data Design Report

## Design Summary
- Frozen `v1`, `v2_long_horizon`, and `v3_long_horizon_china` outputs remain untouched.
- `v4_expanded_universe` is a new versioned 15-sleeve first-build branch.
- The locked v4 sleeve roster and target rules follow the written governance lock exactly.
- No downstream modeling or scenario generation is performed in this package beyond first-pass modeling-prep scaffolding.

## Locked Sleeve Roster
- `EQ_US` | `Equity` | `VTI` | `USD_LISTED_ETF_DIRECT`
- `EQ_EZ` | `Equity` | `EZU` | `USD_LISTED_ETF_DIRECT`
- `EQ_JP` | `Equity` | `EWJ` | `USD_LISTED_ETF_DIRECT`
- `EQ_CN` | `Equity` | `FXI` | `USD_LISTED_ETF_DIRECT`
- `EQ_EM` | `Equity` | `VWO` | `USD_LISTED_ETF_DIRECT`
- `FI_UST` | `Fixed Income` | `IEF` | `USD_LISTED_ETF_DIRECT`
- `FI_EU_GOVT` | `Fixed Income` | `IBGM.AS` | `LOCAL_ETF_PLUS_FX_TO_USD`
- `CR_US_IG` | `Credit` | `LQD` | `USD_LISTED_ETF_DIRECT`
- `CR_EU_IG` | `Credit` | `IEAC.L` | `LOCAL_ETF_PLUS_FX_TO_USD`
- `CR_US_HY` | `Credit` | `HYG` | `USD_LISTED_ETF_DIRECT`
- `CR_EU_HY` | `Credit` | `IHYG.L` | `LOCAL_ETF_PLUS_FX_TO_USD`
- `RE_US` | `Real Asset` | `VNQ` | `USD_LISTED_ETF_DIRECT`
- `LISTED_RE` | `Real Asset` | `RWX` | `USD_LISTED_ETF_DIRECT`
- `LISTED_INFRA` | `Real Asset` | `IGF` | `USD_LISTED_ETF_DIRECT`
- `ALT_GLD` | `Alternative` | `GLD` | `USD_LISTED_ETF_DIRECT`

## Euro Fixed-Income Rule
- `FI_EU_GOVT`, `CR_EU_IG`, and `CR_EU_HY` are built from local-currency investable ETF returns plus month-end EUR/USD conversion to USD.
- The resulting target series are synthetic USD-unhedged total-return indices used only for this locked euro fixed-income family.

## LISTED_RE Interpretation
- `LISTED_RE` is implemented as ex-U.S. listed real estate.
- `RE_US` remains a separate U.S. real-estate sleeve and coexists with `LISTED_RE`.

## Main Table Grains
- `feature_master_monthly`: one row per (`sleeve_id`, `month_end`).
- `target_panel_long_horizon`: one row per (`sleeve_id`, `month_end`, `horizon_months`).
- `modeling_panel_hstack`: one row per (`sleeve_id`, `month_end`, `horizon_months`).

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
- `Yahoo Finance adjusted-close history for VTI`
- `Yahoo Finance adjusted-close history for EZU`
- `Yahoo Finance adjusted-close history for EWJ`
- `Yahoo Finance adjusted-close history for FXI`
- `Yahoo Finance adjusted-close history for VWO`
- `Yahoo Finance adjusted-close history for IEF`
- `Yahoo Finance adjusted-close history for IBGM.AS`
- `Yahoo Finance adjusted-close history for LQD`
- `Yahoo Finance adjusted-close history for IEAC.L`
- `Yahoo Finance adjusted-close history for HYG`
- `Yahoo Finance adjusted-close history for IHYG.L`
- `Yahoo Finance adjusted-close history for VNQ`
- `Yahoo Finance adjusted-close history for RWX`
- `Yahoo Finance adjusted-close history for IGF`
- `Yahoo Finance adjusted-close history for GLD`
- `Yahoo Finance adjusted-close history for EURUSD=X`

## Usable Row Counts
- 60 months: total_target_rows=2572, total_baseline_trainable_rows=2368
- 120 months: total_target_rows=1672, total_baseline_trainable_rows=1468
- 180 months: total_target_rows=772, total_baseline_trainable_rows=575

## Euro Fixed-Income Sleeve Coverage
- `CR_EU_HY` @ 60m: target_available_rows=126
- `CR_EU_HY` @ 120m: target_available_rows=66
- `CR_EU_HY` @ 180m: target_available_rows=6
- `CR_EU_IG` @ 60m: target_available_rows=144
- `CR_EU_IG` @ 120m: target_available_rows=84
- `CR_EU_IG` @ 180m: target_available_rows=24
- `FI_EU_GOVT` @ 60m: target_available_rows=158
- `FI_EU_GOVT` @ 120m: target_available_rows=98
- `FI_EU_GOVT` @ 180m: target_available_rows=38

## Important Limitations
- The frozen macro feature backbone still begins in 2006, so earlier ETF history is used only for technical lookbacks and forward target continuity.
- `CR_EU_HY` has materially shorter effective history than the rest of the roster and should be treated as the weakest long-horizon target sleeve in this first build.
