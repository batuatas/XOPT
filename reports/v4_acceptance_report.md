# XOPTPOE v4_expanded_universe Acceptance / EDA Report

## Executive View
- Structural integrity checks: 16 PASS, 1 WARN, 0 FAIL.
- modeling_panel_hstack shape: 10890 rows x 410 columns.
- feature_master_monthly shape: 3630 rows x 384 columns.
- target_panel_long_horizon shape: 14175 rows x 15 columns.
- Recomputed target formulas matched the stored panel exactly within floating-point tolerance for the audited fields.

## Structural Integrity
- Duplicate-key checks passed for feature_master_monthly, target_panel_long_horizon, modeling_panel_hstack, and modeling_panel_firstpass.
- modeling_panel_hstack carries the same stacked key set as target_panel_long_horizon.
- Each (`sleeve_id`, `month_end`) key expands to exactly 3 horizons in modeling_panel_hstack.
- All 15 sleeves appear in asset, target, feature, modeling, and first-pass modeling outputs.
- Legacy `FI_IG` naming does not leak into the v4 data branch.

## Long-Horizon Target Sanity
- horizon 60m: rows=2572, mean=0.0341, std=0.0568, p05=-0.0554, p50=0.0312, p95=0.1381, range=[-0.1358, 0.2945]
- horizon 120m: rows=1672, mean=0.0346, std=0.0371, p05=-0.0189, p50=0.0314, p95=0.1089, range=[-0.0393, 0.1785]
- horizon 180m: rows=772, mean=0.0363, std=0.0329, p05=-0.0101, p50=0.0360, p95=0.0985, range=[-0.0647, 0.1475]
- Last target-available month by horizon: 60m=2021-02-28, 120m=2016-02-29, 180m=2011-02-28.
- annualized_rf_forward_return and annualized_excess_forward_return behave consistently with the locked TB3MS compounding rule.

## Coverage And Missingness
- Horizon 60m: target_available_rows=2563, baseline_trainable_rows=2368, strict_trainable_rows=0, baseline_trainable_share_of_available=0.924, strict_trainable_share_of_available=0.000.
- Horizon 120m: target_available_rows=1663, baseline_trainable_rows=1468, strict_trainable_rows=0, baseline_trainable_share_of_available=0.883, strict_trainable_share_of_available=0.000.
- Horizon 180m: target_available_rows=763, baseline_trainable_rows=575, strict_trainable_rows=0, baseline_trainable_share_of_available=0.754, strict_trainable_share_of_available=0.000.
- Worst block missingness on baseline-trainable rows:
  - baseline_macro_alias: avg_feature_missing_share=0.348, max_feature_missing_share=0.352, worst_feature=local_cpi_yoy_delta_1m, latest_feature_start_date=2007-03-31.
  - local_mapping: avg_feature_missing_share=0.273, max_feature_missing_share=0.273, worst_feature=cape_local, latest_feature_start_date=2006-01-31.
  - china_macro: avg_feature_missing_share=0.144, max_feature_missing_share=1.000, worst_feature=china_pmi_caixin_mfg_delta_1m, latest_feature_start_date=2023-06-30.
  - interaction: avg_feature_missing_share=0.022, max_feature_missing_share=0.348, worst_feature=int_oecd_activity_proxy_local_x_local_term_slope, latest_feature_start_date=2007-02-28.
  - baseline_macro_canonical: avg_feature_missing_share=0.001, max_feature_missing_share=0.006, worst_feature=infl_EA_delta_1m, latest_feature_start_date=2007-03-31.
  - baseline_global_macro: avg_feature_missing_share=0.000, max_feature_missing_share=0.000, worst_feature=ig_oas, latest_feature_start_date=2007-01-31.

## Euro Fixed-Income Sleeves
- CR_EU_HY: usable_return_months=186, missing_fx_months=0, max_abs_formula_diff=6.661e-16, first_valid_usd_return_month=2010-10-31, p99_abs_monthly_return=0.107.
- CR_EU_IG: usable_return_months=204, missing_fx_months=0, max_abs_formula_diff=7.772e-16, first_valid_usd_return_month=2009-04-30, p99_abs_monthly_return=0.085.
- FI_EU_GOVT: usable_return_months=218, missing_fx_months=0, max_abs_formula_diff=8.882e-16, first_valid_usd_return_month=2008-02-29, p99_abs_monthly_return=0.096.
- The FX join is complete and there is no evidence of construction discontinuities in the synthesized euro fixed-income series.
- `FI_EU_GOVT` and `CR_EU_IG` look clean enough for downstream first-pass modeling. `CR_EU_HY` is coherent as a data series but materially thinner.

## Real-Asset Sleeves
- `RE_US` vs `LISTED_RE`: overlap_months=231, monthly_return_corr=0.792, return_spread_std=0.040. This is similar but not redundant behavior.
- `RE_US`, `LISTED_RE`, and `LISTED_INFRA` are distinct enough to keep in first-pass modeling.

## Sleeve Strength Ranking
- Strongest sleeves by 60m trainability:
  - ALT_GLD: baseline_trainable_rows=169, share_of_available=0.929.
  - CR_US_IG: baseline_trainable_rows=169, share_of_available=0.929.
  - EQ_CN: baseline_trainable_rows=169, share_of_available=0.929.
- Weakest sleeves by 120m trainability:
  - CR_EU_HY: baseline_trainable_rows=53, share_of_available=0.803.
  - CR_EU_IG: baseline_trainable_rows=71, share_of_available=0.845.
  - FI_EU_GOVT: baseline_trainable_rows=85, share_of_available=0.867.
  - LISTED_INFRA: baseline_trainable_rows=86, share_of_available=0.869.

## CR_EU_HY Decision
- Recommendation: `KEEP_DATA_EXCLUDE_FROM_FIRSTPASS_MODELING`.
- Data branch: yes, keep it. 60m/120m first-pass modeling under the current default splits: no. 180m modeling: no. Trainable rows are 60m=113, 120m=53, 180m=0; train-split rows are 60m=5, 120m=5.
- This sleeve is not strong enough for the default first-pass supervised benchmark. Keep it in the data branch, but exclude it from the first modeling benchmark unless the split design is revisited.

## Modeling Readiness Recommendation
- The 15-sleeve v4 branch is internally coherent enough to become the active downstream data branch for first-pass supervised modeling.
- Default modeling scope should remain 60m + 120m. Keep 180m in the source package, but do not require `CR_EU_HY` at 180m.
- For the first supervised benchmark branch, exclude `CR_EU_HY` unless you intentionally redesign splits to improve its training footprint.
- Missing-data handling remains mandatory; strict complete-case filtering is not a viable training rule.

## Direct Answers
1. Is the v4 branch internally intact? yes.
2. Are the long-horizon targets constructed correctly and plausibly? yes.
3. Are the new euro fixed-income sleeves usable? yes; FI_EU_GOVT and CR_EU_IG are clean, CR_EU_HY is usable but thin.
4. Is CR_EU_HY acceptable? data=yes, default 60m/120m first-pass modeling=no, 180m=no.
5. Are LISTED_RE and LISTED_INFRA genuinely useful sleeves? yes; they are distinct enough from RE_US and from each other.
6. Is the first-pass modeling subset still strong enough after moving to 15 sleeves? yes.
7. Which sleeves are strongest / weakest by coverage and trainability? strongest are the legacy core sleeves; weakest is CR_EU_HY, followed by CR_EU_IG and FI_EU_GOVT on longer horizons.
8. Is v4 ready to become the active downstream branch for modeling? yes, with the explicit caveat that CR_EU_HY should be excluded from the default first-pass supervised benchmark unless split design changes.
