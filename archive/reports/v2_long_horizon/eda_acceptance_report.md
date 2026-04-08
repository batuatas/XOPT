# XOPTPOE v2_long_horizon Acceptance / EDA Report

## Executive View
- Structural integrity checks: 25 PASS, 0 WARN, 0 FAIL.
- modeling_panel_hstack shape: 5808 rows x 386 columns.
- feature_master_monthly shape: 1936 rows x 360 columns.
- target_panel_long_horizon shape: 5808 rows x 15 columns.
- Recomputed target formulas matched the stored panel with max absolute difference 0.000e+00 across all audited target fields.

## Structural Integrity
- Duplicate-key checks passed for all three main tables: feature_master_monthly, target_panel_long_horizon, and modeling_panel_hstack.
- Each (`sleeve_id`, `month_end`) key expands to exactly 3 horizons in the stacked panel.
- modeling_panel_hstack carries the same stacked key set as target_panel_long_horizon.
- baseline_trainable_flag formula matched exactly; strict_trainable_flag formula matched exactly.

## Long-Horizon Target Sanity
- Stored targets match a full recomputation from frozen v1 month-end prices, realized monthly returns, and TB3MS-based monthly risk-free compounding.
- Example audited row: sleeve=ALT_GLD, month_end=2006-01-31, horizon=60 months, annualized_excess_forward_return=0.156046.
- No nonpositive compounded gross returns were found on target-available rows.
- realized_forward_volatility is nonnegative throughout the available sample.
- realized_forward_max_drawdown stays within (-1, 0] as expected.
- Last target-available month by horizon: 5Y=2021-02-28, 10Y=2016-02-29, 15Y=2011-02-28.

## Coverage And Missingness
- Horizon 60m: target_available_rows=1456, baseline_trainable_rows=1352, strict_trainable_rows=0, baseline_trainable_share_of_available=0.929, strict_trainable_share_of_available=0.000.
- Horizon 120m: target_available_rows=976, baseline_trainable_rows=872, strict_trainable_rows=0, baseline_trainable_share_of_available=0.893, strict_trainable_share_of_available=0.000.
- Horizon 180m: target_available_rows=496, baseline_trainable_rows=392, strict_trainable_rows=0, baseline_trainable_share_of_available=0.790, strict_trainable_share_of_available=0.000.
- The heaviest missingness in `baseline_macro_alias` and `local_mapping` is structural by design: `EQ_EM` and `ALT_GLD` have no local macro block, so mapped `local_*` and `cape_local` fields are expected to be absent there.
- The heaviest non-structural missingness comes from late-start China enrichment fields, especially Caixin PMI and house-price series.
- Worst overall block missingness on target-available rows:
  - baseline_macro_alias: avg_feature_missing_share=0.273, max_feature_missing_share=0.336, worst_feature=local_cpi_yoy_delta_1m (0.336), latest_feature_start_date=2007-03-31.
  - local_mapping: avg_feature_missing_share=0.250, max_feature_missing_share=0.250, worst_feature=cape_local (0.250), latest_feature_start_date=2006-01-31.
  - china_macro: avg_feature_missing_share=0.172, max_feature_missing_share=1.000, worst_feature=china_pmi_caixin_services_delta_1m (1.000), latest_feature_start_date=2023-06-30.
  - interaction: avg_feature_missing_share=0.073, max_feature_missing_share=0.256, worst_feature=int_oecd_activity_proxy_local_x_local_term_slope (0.256), latest_feature_start_date=2007-02-28.
  - baseline_technical: avg_feature_missing_share=0.061, max_feature_missing_share=0.107, worst_feature=mom_12_1 (0.107), latest_feature_start_date=2007-02-28.
  - china_market: avg_feature_missing_share=0.053, max_feature_missing_share=0.107, worst_feature=china_sse_composite_usd_mom_12_1 (0.107), latest_feature_start_date=2007-02-28.
- Main causes of strict complete-case failure:
  - china_macro: missing in 2928 strict-failure rows, share_of_strict_failure_rows=1.000.
  - baseline_macro_alias: missing in 984 strict-failure rows, share_of_strict_failure_rows=0.336.
  - interaction: missing in 966 strict-failure rows, share_of_strict_failure_rows=0.330.
  - local_mapping: missing in 732 strict-failure rows, share_of_strict_failure_rows=0.250.
  - baseline_macro_canonical: missing in 336 strict-failure rows, share_of_strict_failure_rows=0.115.
  - baseline_technical: missing in 312 strict-failure rows, share_of_strict_failure_rows=0.107.
  - china_market: missing in 312 strict-failure rows, share_of_strict_failure_rows=0.107.
  - japan_enrichment: missing in 312 strict-failure rows, share_of_strict_failure_rows=0.107.
- Highest-missing individual features among strict-failure rows:
  - china_pmi_caixin_mfg_delta_1m: share_of_strict_failure_rows=1.000.
  - china_pmi_caixin_services: share_of_strict_failure_rows=1.000.
  - china_pmi_caixin_mfg: share_of_strict_failure_rows=1.000.
  - china_pmi_caixin_services_delta_1m: share_of_strict_failure_rows=1.000.
  - china_houseprice_100cities_delta_1m: share_of_strict_failure_rows=0.541.
  - china_houseprice_100cities: share_of_strict_failure_rows=0.536.
  - china_houseprice_nbs_new70_delta_1m: share_of_strict_failure_rows=0.514.
  - china_houseprice_nbs_existing70_delta_1m: share_of_strict_failure_rows=0.514.
  - china_houseprice_nbs_new70: share_of_strict_failure_rows=0.508.
  - china_houseprice_nbs_existing70: share_of_strict_failure_rows=0.508.

## Distribution And Plausibility
- annualized_excess_forward_return, horizon 60m: mean=0.0474, std=0.0588, p05=-0.0486, p50=0.0451, p95=0.1565, min=-0.1302, max=0.2945.
- annualized_excess_forward_return, horizon 120m: mean=0.0462, std=0.0374, p05=-0.0060, p50=0.0417, p95=0.1263, min=-0.0216, max=0.1785.
- annualized_excess_forward_return, horizon 180m: mean=0.0464, std=0.0315, p05=0.0067, p50=0.0425, p95=0.1194, min=-0.0177, max=0.1475.
- Highest target volatility sleeve-horizon combinations:
  - ALT_GLD @ 60m: std=0.0738, range=[-0.0816, 0.2145].
  - RE_US @ 60m: std=0.0597, range=[-0.0347, 0.2945].
  - EQ_EZ @ 60m: std=0.0590, range=[-0.1302, 0.1701].
- Lowest target volatility sleeve-horizon combinations:
  - ALT_GLD @ 180m: std=0.0085, range=[0.0317, 0.0711].
  - FI_IG @ 180m: std=0.0086, range=[0.0200, 0.0495].
  - FI_UST @ 180m: std=0.0119, range=[0.0052, 0.0410].
- Key predictor scale snapshot on baseline-trainable rows:
  - ret_1m_lag: mean=0.0050, std=0.0550, p01=-0.1718, p99=0.1381.
  - mom_12_1: mean=0.0669, std=0.2030, p01=-0.5237, p99=0.6769.
  - vol_12m: mean=0.0460, std=0.0287, p01=0.0105, p99=0.1440.
  - infl_US: mean=1.8419, std=1.4610, p01=-1.4838, p99=5.3080.
  - infl_EA: mean=1.5412, std=1.1367, p01=-0.6070, p99=4.0330.
  - infl_JP: mean=0.2538, std=1.1999, p01=-2.2564, p99=3.5941.
  - short_rate_US: mean=0.8893, std=1.4047, p01=0.0100, p99=4.9800.
  - short_rate_EA: mean=1.3187, std=1.7792, p01=-0.5091, p99=5.0192.
  - short_rate_JP: mean=0.3832, std=0.2737, p01=-0.0550, p99=0.8800.
  - usd_broad: mean=98.2608, std=9.6405, p01=85.7282, p99=120.4649.
  - vix: mean=21.6454, std=9.3986, p01=10.4100, p99=55.8400.
  - us_real10y: mean=0.8358, std=0.9073, p01=-1.0000, p99=2.6500.
  - ig_oas: mean=2.0022, std=1.1610, p01=0.9100, p99=6.0600.
  - oil_wti: mean=76.4359, std=23.2911, p01=32.7400, p99=127.3500.
  - china_cli: mean=100.2112, std=2.7936, p01=89.8598, p99=106.1712.
  - china_cmi: mean=8.5203, std=3.8692, p01=0.1200, p99=15.0000.
  - china_div_yield: mean=3.0446, std=1.1404, p01=0.9330, p99=5.3460.
  - china_sse_composite_usd_mom_12_1: mean=0.0912, std=0.4520, p01=-0.9930, p99=1.2053.
  - jp_pe_ratio: mean=24.4665, std=14.7221, p01=10.1016, p99=63.7422.
  - jp_tankan_actual: mean=-9.7859, std=12.9509, p01=-46.0000, p99=1.0000.
  - jp_buyback_index_usd_mom_12_1: mean=0.0153, std=0.1423, p01=-0.3655, p99=0.3206.
  - cape_local: mean=22.7580, std=6.5790, p01=12.4300, p99=54.0000.
  - cape_usa: mean=22.8545, std=4.5700, p01=13.7000, p99=34.2900.
  - em_minus_global_pe: mean=-2.0177, std=1.5520, p01=-5.2520, p99=1.9380.
  - us_hy_oas: mean=6.3520, std=3.2964, p01=2.7400, p99=18.1200.
  - eu_hy_oas: mean=6.6000, std=4.3425, p01=2.0500, p99=21.8200.
  - eu_ig_corp_tr_usd_mom_12_1: mean=0.0373, std=0.1059, p01=-0.1811, p99=0.3060.
  - int_log_horizon_x_mom_12_1: mean=0.1344, std=0.4437, p01=-1.2075, p99=1.4697.
  - int_log_horizon_x_vix: mean=44.1493, std=23.4936, p01=16.7542, p99=128.5764.
  - int_china_cli_x_eq_em: mean=12.5264, std=33.1628, p01=0.0000, p99=104.1305.
  - int_jp_pe_ratio_x_eq_jp: mean=3.0583, std=9.6224, p01=0.0000, p99=63.7422.

## Interaction-Term Sanity
- asset_group_dummy_x_predictor: usage=keep_optional, interaction_count=8, avg_nonmissing_share=1.000, avg_zero_share=0.750, max_abs_max=59.890.
- cape_x_real_rate: usage=keep_optional, interaction_count=2, avg_nonmissing_share=0.875, avg_zero_share=0.003, max_abs_max=163.028.
- china_block_x_em_relevance: usage=defer, interaction_count=5, avg_nonmissing_share=0.989, avg_zero_share=0.875, max_abs_max=106.171.
- cli_x_slope_or_spread: usage=keep_optional, interaction_count=2, avg_nonmissing_share=0.873, avg_zero_share=0.000, max_abs_max=607.684.
- japan_block_x_jp_relevance: usage=defer, interaction_count=4, avg_nonmissing_share=0.987, avg_zero_share=0.875, max_abs_max=101.404.
- predictor_x_log_horizon_years: usage=keep_optional, interaction_count=7, avg_nonmissing_share=0.950, avg_zero_share=0.000, max_abs_max=287.517.
- sleeve_dummy_x_predictor: usage=defer, interaction_count=16, avg_nonmissing_share=0.948, avg_zero_share=0.875, max_abs_max=1.106.
- stress_x_momentum: usage=keep_optional, interaction_count=3, avg_nonmissing_share=0.946, avg_zero_share=0.043, max_abs_max=31.381.
- stress_x_valuation: usage=defer, interaction_count=3, avg_nonmissing_share=0.833, avg_zero_share=0.000, max_abs_max=1363.219.
- Interactions that are numerically awkward enough to defer in first-pass modeling:
  - int_vix_x_cape_local (very_large_scale): nonmissing_share=0.750, zero_share=0.000, max_abs=1363.219.

## Modeling Readiness Recommendation
- Main modeling input: `data/final_v2_long_horizon/modeling_panel_hstack.parquet`.
- Default filter: `baseline_trainable_flag == 1`.
- `strict_trainable_flag` should remain a diagnostic only; it is currently unusable as a training filter because it leaves zero rows across all horizons.
- Missing-data handling is mandatory. First-pass models should use explicit masking/imputation rather than literal complete-case filtering.
- First-pass horizon choice: start with 5Y + 10Y. They preserve materially more rows than 15Y while still matching the long-horizon objective.
- Feature blocks to make optional in the first model pass: the latest-start China PMI / housing fields and the sparsest relevance-gated interaction families.

## Direct Answers
1. Is the v2_long_horizon dataset internally intact? yes.
2. Are the long-horizon targets constructed correctly? yes; they match a full recomputation from prices, returns, and TB3MS.
3. Which feature blocks are most problematic for missingness? baseline_macro_alias, local_mapping, china_macro, interaction; the first two are mostly structural no-local-block gaps, while `china_macro` is the main late-start enrichment block.
4. Is baseline_trainable_flag the correct default training filter? yes.
5. Should first-pass modeling use only 5Y, 5Y + 10Y, or all 5Y/10Y/15Y? 5Y + 10Y first; add 15Y only after the model stack is stable under missingness handling.
6. Are the interaction terms ready to use, or should some be deferred? keep the full dataset intact, but defer the sparsest sleeve-/relevance-gated interaction families in the first pass.
7. Is the dataset ready for the modeling section right now? yes, with baseline_trainable_flag and explicit masking/imputation as the default training setup.