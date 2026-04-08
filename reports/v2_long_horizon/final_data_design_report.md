# XOPTPOE v2_long_horizon Final Data Design Report

## Design Summary
- Frozen `v1` data outputs remain untouched.
- `v2_long_horizon` is a new canonical data package for long-horizon SAA modeling.
- Monthly predictors are built at sleeve-month grain and stacked across 5Y / 10Y / 15Y horizons.
- No predictive models or portfolio backtests are built in this package.

## Main Table Grains
- `feature_master_monthly`: one row per (`sleeve_id`, `month_end`).
- `target_panel_long_horizon`: one row per (`sleeve_id`, `month_end`, `horizon_months`).
- `modeling_panel_hstack`: one row per (`sleeve_id`, `month_end`, `horizon_months`).

## Long-Horizon Target Definition
For sleeve `a`, decision month `t`, and horizon `H` in {60, 120, 180}:
- compounded total forward return = `P[a, t+H] / P[a, t]`
- compounded risk-free forward return = `prod_{i=1..H} (1 + TB3MS[t+i] / 1200)`
- annualized total forward return = `gross_total_forward_return^(12/H) - 1`
- annualized risk-free forward return = `gross_rf_forward_return^(12/H) - 1`
- annualized excess forward return = `(gross_total_forward_return / gross_rf_forward_return)^(12/H) - 1`
- auxiliary labels: cumulative total return, realized forward volatility, realized forward max drawdown

## Source Files Used
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

## Added Feature Blocks
- OECD leading indicators: US / EA / JP / China LI, business-confidence, consumer-confidence, plus selected OECD BTS survey measures.
- CAPE ratios: China, Europe, Japan, USA, plus mapped `cape_local`.
- China macro block: CLI, CMI, official GDP growth, PMI variants, CPI subcomponents, industrial production, M2, fiscal flows, house-price indicators.
- China valuation block: dividend yield, PE, PTB, EM-relative spreads, and China SSE composite USD market features.
- Japan enrichment block: OECD leading alternative, nominal GDP, policy rate, CPI variants, labor, industrial production, Tankan, PE, TOPIX market features, buyback-index market features, alternative bond-yield / CPI file.
- EM/global valuation block: EM and global dividend yield, PTB, PE, and EM-global spreads.
- HY / stress block: US HY OAS/effective yield and European HY OAS/effective yield.
- EU IG market block: USD-converted Euro IG total-return index history and market transforms.

## China Treatment
- China is included as a feature block only.
- No standalone China sleeve target is added because the uploaded China workbook is a market-history feature source, not a clean investable USD adjusted-close sleeve target with consistent provenance.

## Interaction Families Created
- sleeve dummy x predictor
- asset-class-group dummy x predictor
- stress x valuation
- stress x momentum
- China block x EM sleeve relevance
- Japan block x Japan sleeve relevance
- CAPE x real-rate
- CLI x slope/spread
- predictor x log_horizon_years in the stacked panel

## Usable Row Counts
- 60 months: target_available_rows=1456, baseline_trainable_rows=1352
- 120 months: target_available_rows=976, baseline_trainable_rows=872
- 180 months: target_available_rows=496, baseline_trainable_rows=392

## Target-History Limitation
- No clean benchmark/index backfill was added for the locked sleeves.
- Long-horizon availability remains constrained by the ETF history inherited from frozen `v1` month-end prices.
- This especially truncates 10Y and 15Y usable rows, but it is documented rather than silently backfilled with mixed target definitions.

## Recommended Main Modeling Table
- Use `data/final_v2_long_horizon/modeling_panel_hstack.parquet` as the primary table for the later deep-learning section.
- `baseline_trainable_flag` is the recommended starting filter when using explicit imputation/masking for enrichment features.
- `strict_trainable_flag` is the literal all-features complete-case diagnostic. In the current build it yields 0 long-horizon rows because the richest enrichment set only becomes fully complete after the available target window.

## Direct Answers
1. Main table grains: `feature_master_monthly` = (`sleeve_id`, `month_end`); `target_panel_long_horizon` = (`sleeve_id`, `month_end`, `horizon_months`); `modeling_panel_hstack` = (`sleeve_id`, `month_end`, `horizon_months`).
2. Uploaded files used: country_level_macro_data/BAMLH0A0HYM2.csv, country_level_macro_data/BAMLH0A0HYM2EY.csv, country_level_macro_data/BAMLHE00EHYIEY.csv, country_level_macro_data/BAMLHE00EHYIOAS.csv, country_level_macro_data/China Price History_SSE_Composite.xlsx, country_level_macro_data/China_dividend_yield.csv, country_level_macro_data/China_price_to_earnings.csv, country_level_macro_data/EU_IG_.xlsx, country_level_macro_data/JPNLOLITONOSTSAM.csv, country_level_macro_data/JaPaN GDP.csv, country_level_macro_data/OECD.SDD.STES,DSD_STES@DF_BTS,4.0+CHN+JPN+EA20+USA.M........csv, country_level_macro_data/china & em price-to-book ratio.csv, country_level_macro_data/china_cli.csv, country_level_macro_data/china_cpi_data.csv, country_level_macro_data/china_fiscal_socioeconomic.csv, country_level_macro_data/china_gdp_cmi.csv, country_level_macro_data/china_industrial_production.csv, country_level_macro_data/china_pmi.csv, country_level_macro_data/em_global_div_yield.csv, country_level_macro_data/em_global_ptb.csv, country_level_macro_data/em_global_pte.csv, country_level_macro_data/japan buyback yield.xlsx, country_level_macro_data/japan_10y_bond_yield_cpi.csv, country_level_macro_data/japan_macro_fiscal_socioeconomic.csv, country_level_macro_data/japan_pe_topix_index.csv, data/external/akif_candidates/Historic-cape-ratios (1).csv, data/external/akif_candidates/OECD Leading Indicatorscsv.csv.
3. Variables added: OECD/CAPE blocks, China macro and valuation, Japan enrichment, EM/global valuation, HY/IG stress, and workbook-derived market features for China SSE, Japan buyback, and Euro IG.
4. Exact long-horizon targets: annualized compounded total return, annualized compounded risk-free return, annualized compounded excess return, plus cumulative return, realized forward volatility, and realized forward max drawdown.
5. Usable target rows: 5Y=1456, 10Y=976, 15Y=496.
6. Target-history limitations remain: yes; all horizons are still limited by the ETF-based sleeve history inherited from frozen `v1`.
7. China included as: feature block only, not a sleeve target.
8. Interaction families created: asset_group_dummy_x_predictor, cape_x_real_rate, china_block_x_em_relevance, cli_x_slope_or_spread, japan_block_x_jp_relevance, predictor_x_log_horizon_years, sleeve_dummy_x_predictor, stress_x_momentum, stress_x_valuation.
9. Main modeling table for later deep learning: `modeling_panel_hstack`.
10. Dataset ready for the model section: yes, with documented 10Y/15Y sample limits, explicit missingness/trainability flags, and the expectation that enrichment features will need masking or imputation rather than full complete-case filtering.