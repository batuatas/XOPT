# XOPTPOE v1 Robustness Report

## Scope
- This is a robustness/diagnostics audit of the existing modeling + portfolio prototype outputs only.
- No data-pipeline redesign or schema change was performed.
- Test window audited: 2024-02-29 to 2026-01-31 (24 months).

## 1) Prediction Robustness
- Validation-selected model: `elastic_net_pooled`.
- Validation RMSE: elastic_net_pooled=0.050657 vs naive_sleeve_mean=0.052178 (selected model better in validation).
- Test RMSE: elastic_net_pooled=0.035358 vs naive_sleeve_mean=0.032795 (naive better in test by 0.002563).
- Test OOS R^2: elastic_net_pooled=-0.1554 vs naive_sleeve_mean=0.0060.
- Ranking quality (mean test Spearman IC): elastic_net_pooled=0.2212, naive_sleeve_mean=0.0942.
- Ranking lift (top-3 minus cross-section mean, test): elastic_net_pooled=0.0086, naive_sleeve_mean=0.0056.
- Top-1 predicted sleeve concentration on test (elastic_net_pooled): dominant=EQ_JP, share=83.33%, HHI=0.722.
- Sleeve-level stability check is mixed: several sleeves change materially between validation and test (see `metrics_by_sleeve` deltas in this report section notes).

## 2) Portfolio Robustness
- top_k_equal (elastic_net_pooled): avg_ret=0.0204, vol=0.0272, Sharpe=2.595, avg_turnover=0.097.
- mv_clipped (elastic_net_pooled): avg_ret=0.0152, vol=0.0254, Sharpe=2.081, avg_turnover=0.235.
- equal_weight (elastic_net_pooled): avg_ret=0.0118, vol=0.0204, Sharpe=2.000, avg_turnover=0.000.
- score_positive (elastic_net_pooled): avg_ret=0.0122, vol=0.0250, Sharpe=1.696, avg_turnover=0.265.
- Outperformance concentration (top_k_equal vs equal_weight, elastic_net_pooled): top 3 positive active months contribute 34.12% of all positive active return; top 5 contribute 51.24%.
- Subperiod stability (top_k_equal, elastic_net_pooled): first-half avg=0.0094 (Sharpe=1.254), second-half avg=0.0314 (Sharpe=4.402).
- Sleeve concentration in top-k selection frequency:
  - EQ_JP: selected in 100.00% of test months
  - ALT_GLD: selected in 95.83% of test months
  - EQ_EZ: selected in 58.33% of test months
  - EQ_EM: selected in 45.83% of test months
  - EQ_US: selected in 0.00% of test months
  - FI_IG: selected in 0.00% of test months
  - FI_UST: selected in 0.00% of test months
  - RE_US: selected in 0.00% of test months

## 3) Sensitivity Checks
- Top-k sensitivity (fixed selected model) written to `reports/topk_sensitivity_summary.csv`.
- Best top-k by Sharpe under selected model: k=4, Sharpe=2.599, avg_ret=0.0184.
- Model sensitivity (naive_sleeve_mean, ridge_pooled, elastic_net_pooled; all strategies, k=3) written to `reports/model_sensitivity_summary.csv`.
- Best non-equal-weight strategy across checked models: mv_clipped with signal=naive_sleeve_mean (Sharpe=2.788).
- Transaction-cost stress (10 bps and 25 bps turnover haircuts) included in both sensitivity CSVs.

## 4) Practical Interpretation
- The prototype contains a tradable-looking ranking signal in some windows, but point-forecast skill is weak and unstable across validation/test.
- Portfolio gains are present in the test window for top-k variants, but concentration and short sample length make regime overfitting risk material.
- Turnover haircuts reduce but do not universally eliminate the observed edge in this 24-month window; robustness across longer periods is unproven.

## Final Answer
The current outperformance is promising but not robust enough yet to fully trust; treat it as fragile/provisional and require additional model validation before relying on it for production portfolio construction.
