# XOPTPOE v1 Rolling Robustness Report

## Scope
- Focused robustness-improvement pass; data pipeline and locked dataset design unchanged.
- Compact model set: naive_sleeve_mean, ridge_pooled, elastic_net_pooled.
- Feature-set experiments: technical-only, macro-only, technical+global macro, full set.

## Rolling Setup
- expanding folds: min_train_months=96, validation_months=24, test_months=24, step_months=12
- fold count: 8
- first fold test range: 2017-02-28 to 2019-01-31; last fold test range: 2024-02-29 to 2026-01-31

## Feature-Set Robustness
- naive_sleeve_mean: best feature set = `full_current_set` (val_rmse=0.0423, test_rmse=0.0424, test_spearman=0.0781)
- ridge_pooled: best feature set = `technical_only` (val_rmse=0.0425, test_rmse=0.0430, test_spearman=0.0468)
- elastic_net_pooled: best feature set = `full_current_set` (val_rmse=0.0418, test_rmse=0.0431, test_spearman=0.1895)
- best compact feature-set/model pair by test RMSE: `naive_sleeve_mean` + `technical_only` (test_rmse=0.0424, test_spearman=0.0781)

## Concentration Control
- evaluated variants: equal_weight, top_k_equal, top_k_capped(k=3, cap=0.30), score_positive_capped(cap=0.35), top_k_diversified_cap(top_n=5, cap=0.25)
- selected feature set per model for this stage:
  - naive_sleeve_mean: full_current_set
  - ridge_pooled: technical_only
  - elastic_net_pooled: full_current_set
- naive_sleeve_mean: best concentration-aware strategy `top_k_equal` (Sharpe=0.888, avg_max_weight=0.333). top_k_equal baseline avg_max_weight=0.333.
- ridge_pooled: best concentration-aware strategy `top_k_capped` (Sharpe=1.001, avg_max_weight=0.284). top_k_equal baseline avg_max_weight=0.333.
- elastic_net_pooled: best concentration-aware strategy `top_k_diversified_cap` (Sharpe=0.710, avg_max_weight=0.227). top_k_equal baseline avg_max_weight=0.333.

## Decision
- Rolling evidence does not show a sufficiently stable non-naive signal across folds.
After these robustness upgrades, the project should pivot toward simpler benchmark-driven allocation rules unless further predictive signal improvements are demonstrated.
