# XOPTPOE v1 Baseline Model Report

## Models Implemented
- naive_sleeve_mean (naive) with {'type': 'historical_mean_by_sleeve'}
- naive_last_return (naive) with {'type': 'predict_ret_1m_lag'}
- ridge_pooled (ridge) with {'alpha': 100.0, 'validation_rmse': 0.050994110583344056, 'random_state': 42}
- elastic_net_pooled (elastic_net) with {'alpha': 0.001, 'l1_ratio': 0.2, 'validation_rmse': 0.0506573409853505, 'random_state': 42}
- random_forest_pooled (random_forest) with {'n_estimators': 300, 'max_depth': 4, 'min_samples_leaf': 5, 'validation_rmse': 0.05185522332838286, 'random_state': 42}

## Feature Set Used
- target: `excess_ret_fwd_1m`
- numeric feature count: `70`
- pooled models also include sleeve one-hot indicators.
- numeric features:
  - ig_oas
  - ig_oas_delta_1m
  - ig_oas_level
  - infl_EA
  - infl_EA_delta_1m
  - infl_JP
  - infl_JP_delta_1m
  - infl_US
  - infl_US_delta_1m
  - local_10y_rate
  - local_10y_rate_delta_1m
  - local_3m_rate
  - local_3m_rate_delta_1m
  - local_cpi_yoy
  - local_cpi_yoy_delta_1m
  - local_term_slope
  - local_term_slope_delta_1m
  - local_unemp
  - local_unemp_delta_1m
  - long_rate_EA
  - long_rate_EA_delta_1m
  - long_rate_JP
  - long_rate_JP_delta_1m
  - long_rate_US
  - long_rate_US_delta_1m
  - macro_stale_flag
  - maxdd_12m
  - mom_12_1
  - oil_wti
  - oil_wti_level
  - oil_wti_logchg_12m
  - oil_wti_logchg_1m
  - rel_mom_vs_treasury
  - rel_mom_vs_us_equity
  - rel_ret_1m_vs_treasury
  - rel_ret_1m_vs_us_equity
  - ret_12m_lag
  - ret_1m_lag
  - ret_3m_lag
  - ret_6m_lag
  - short_rate_EA
  - short_rate_EA_delta_1m
  - short_rate_JP
  - short_rate_JP_delta_1m
  - short_rate_US
  - short_rate_US_delta_1m
  - term_slope_EA
  - term_slope_EA_delta_1m
  - term_slope_JP
  - term_slope_JP_delta_1m
  - term_slope_US
  - term_slope_US_delta_1m
  - unemp_EA
  - unemp_EA_delta_1m
  - unemp_JP
  - unemp_JP_delta_1m
  - unemp_US
  - unemp_US_delta_1m
  - us_real10y
  - us_real10y_delta_1m
  - us_real10y_level
  - usd_broad
  - usd_broad_level
  - usd_broad_logchg_12m
  - usd_broad_logchg_1m
  - vix
  - vix_delta_1m
  - vix_level
  - vol_12m
  - vol_3m

## Validation Metrics (Model Selection)
- elastic_net_pooled: RMSE=0.050657, MAE=0.041284, OOS_R2=0.056760, Corr=0.214801, DirAcc=0.598958
- ridge_pooled: RMSE=0.050994, MAE=0.041255, OOS_R2=0.044177, Corr=0.184992, DirAcc=0.557292
- random_forest_pooled: RMSE=0.051855, MAE=0.042500, OOS_R2=0.011624, Corr=0.022127, DirAcc=0.468750
- naive_sleeve_mean: RMSE=0.052178, MAE=0.043206, OOS_R2=-0.000721, Corr=0.010230, DirAcc=0.442708
- naive_last_return: RMSE=0.078206, MAE=0.062421, OOS_R2=-1.248111, Corr=-0.125170, DirAcc=0.458333
- best_validation_model: `elastic_net_pooled` (lowest validation RMSE)

## Test Metrics (Frozen After Validation Choice)
- naive_sleeve_mean: RMSE=0.032795, MAE=0.024480, OOS_R2=0.006046, Corr=0.081770, DirAcc=0.687500
- random_forest_pooled: RMSE=0.033046, MAE=0.024743, OOS_R2=-0.009226, Corr=-0.000018, DirAcc=0.687500
- elastic_net_pooled: RMSE=0.035358, MAE=0.026926, OOS_R2=-0.155408, Corr=0.106766, DirAcc=0.593750
- ridge_pooled: RMSE=0.035986, MAE=0.028016, OOS_R2=-0.196802, Corr=0.123644, DirAcc=0.505208
- naive_last_return: RMSE=0.043319, MAE=0.033809, OOS_R2=-0.734274, Corr=0.080650, DirAcc=0.536458

## Sleeve-Level Difficulty (Best Model On Test)
- easiest sleeves by RMSE:
  - FI_UST: RMSE=0.023641, OOS_R2=-1.056492, Corr=0.048105
  - FI_IG: RMSE=0.025108, OOS_R2=-1.316773, Corr=-0.086889
  - EQ_EM: RMSE=0.030147, OOS_R2=-0.360696, Corr=-0.043997
- hardest sleeves by RMSE:
  - RE_US: RMSE=0.043155, OOS_R2=-0.212632, Corr=-0.025521
  - ALT_GLD: RMSE=0.041706, OOS_R2=0.321487, Corr=0.411768
  - EQ_US: RMSE=0.040262, OOS_R2=-0.720910, Corr=-0.377774

## Readiness
- Signal quality is weak/modest; proceed to portfolio construction only as an exploratory prototype.
