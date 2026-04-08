# XOPTPOE v1 Portfolio Backtest Report

## Setup
- signal_model: `elastic_net_pooled`
- constraints: long-only, fully invested, no leverage, monthly rebalancing
- config: top_k=3, cov_lookback_months=60, cov_min_months=24, mv_ridge=0.001, mv_risk_aversion=1.0

## Backtest Variants
- `equal_weight`: 1/8 each month
- `top_k_equal`: equal weight over top-3 predicted sleeves each month
- `score_positive`: weights proportional to clipped positive prediction scores
- `mv_clipped`: regularized mean-variance heuristic + long-only clipping

## Performance Summary
- top_k_equal: avg_ret=0.020385, vol=0.027210, sharpe=2.5952, max_dd=-0.0452, turnover=0.0972
- mv_clipped: avg_ret=0.015236, vol=0.025357, sharpe=2.0814, max_dd=-0.0444, turnover=0.2349
- equal_weight: avg_ret=0.011781, vol=0.020410, sharpe=1.9996, max_dd=-0.0521, turnover=0.0000
- score_positive: avg_ret=0.012244, vol=0.025013, sharpe=1.6956, max_dd=-0.0580, turnover=0.2649

## Best Strategy
- best_by_sharpe: `top_k_equal` (Sharpe=2.5952, avg_ret=0.020385)

## Value Add vs Equal Weight
- equal_weight: avg_ret=0.011781, sharpe=1.9996
- prediction_layer_adds_value_over_equal_weight: YES

## Weight Diagnostics
- highest average sleeve allocations across strategies:
  - mv_clipped/EQ_JP: avg=0.4575, min=0.1747, max=0.7244, nonzero_share=100.00%
  - score_positive/EQ_JP: avg=0.3550, min=0.1499, max=1.0000, nonzero_share=100.00%
  - top_k_equal/EQ_JP: avg=0.3333, min=0.3333, max=0.3333, nonzero_share=100.00%
  - top_k_equal/ALT_GLD: avg=0.3194, min=0.0000, max=0.3333, nonzero_share=95.83%
  - mv_clipped/ALT_GLD: avg=0.2468, min=0.1285, max=0.3293, nonzero_share=100.00%
  - top_k_equal/EQ_EZ: avg=0.1944, min=0.0000, max=0.3333, nonzero_share=58.33%
  - score_positive/ALT_GLD: avg=0.1734, min=0.0000, max=0.2893, nonzero_share=91.67%
  - top_k_equal/EQ_EM: avg=0.1528, min=0.0000, max=0.3333, nonzero_share=45.83%
  - score_positive/EQ_EZ: avg=0.1493, min=0.0000, max=0.3133, nonzero_share=91.67%
  - score_positive/EQ_EM: avg=0.1456, min=0.0000, max=0.3032, nonzero_share=91.67%

## Next Improvements
- Add transaction-cost assumptions and net-return reporting.
- Calibrate top-k, score transforms, and MV risk-aversion with rolling validation.
- Add covariance shrinkage alternatives and turnover penalties.
- Evaluate model-ensemble signals instead of single-model predictions.
