# XOPTPOE v3 SAA Portfolio Comparison Report

## Setup
- These are long-horizon SAA decision-period diagnostics on overlapping annualized forward excess-return labels, not a fully tradable non-overlapping monthly wealth backtest.
- Best 60m and 120m predictors were selected by validation RMSE. Their signals were then converted into monthly sleeve allocations with the existing long-only robust allocator.
- Predictor-driven portfolios use a common allocator design: long-only, fully invested, no leverage, active v3 risk model, validation-selected lambda/kappa/Omega from the existing compact grid.
- PTO_NN and E2E_NN appear twice conceptually: as predictor-family entries in the horse race, and here as original setup reference rows loaded from the existing reports.

## Predictor Winners Feeding Portfolios
- Best 60m signal source: `elastic_net__full_firstpass__separate_60`.
- Best 120m signal source: `ridge__full_firstpass__separate_120`.
- Best shared benchmark source: `elastic_net__core_plus_interactions__shared_60_120`.

## Portfolio Metrics
- common_allocator, test, best_shared_predictor: avg_return=0.0695, volatility=0.0108, sharpe=6.4291, avg_turnover=0.0750, max_drawdown=0.0000.
- common_allocator, test, best_120_predictor: avg_return=0.0767, volatility=0.0124, sharpe=6.1859, avg_turnover=0.0840, max_drawdown=0.0000.
- common_allocator, test, best_60_predictor: avg_return=0.0513, volatility=0.0088, sharpe=5.8271, avg_turnover=0.0553, max_drawdown=0.0000.
- common_allocator, test, combined_60_120_predictor: avg_return=0.0694, volatility=0.0121, sharpe=5.7325, avg_turnover=0.1040, max_drawdown=0.0000.
- common_allocator, test, e2e_nn_signal: avg_return=0.0374, volatility=0.0077, sharpe=4.8521, avg_turnover=0.1116, max_drawdown=0.0000.
- common_allocator, test, pto_nn_signal: avg_return=0.0386, volatility=0.0106, sharpe=3.6312, avg_turnover=0.0330, max_drawdown=0.0000.
- common_allocator, test, equal_weight: avg_return=0.0406, volatility=0.0150, sharpe=2.7019, avg_turnover=0.0000, max_drawdown=0.0000.
- common_allocator, validation, best_120_predictor: avg_return=0.0638, volatility=0.0055, sharpe=11.6044, avg_turnover=0.0296, max_drawdown=0.0000.
- common_allocator, validation, best_shared_predictor: avg_return=0.0588, volatility=0.0063, sharpe=9.2918, avg_turnover=0.0249, max_drawdown=0.0000.
- common_allocator, validation, combined_60_120_predictor: avg_return=0.0607, volatility=0.0071, sharpe=8.5989, avg_turnover=0.0377, max_drawdown=0.0000.
- common_allocator, validation, best_60_predictor: avg_return=0.0538, volatility=0.0072, sharpe=7.4668, avg_turnover=0.0390, max_drawdown=0.0000.
- common_allocator, validation, pto_nn_signal: avg_return=0.0321, volatility=0.0045, sharpe=7.1834, avg_turnover=0.0258, max_drawdown=0.0000.
- common_allocator, validation, e2e_nn_signal: avg_return=0.0305, volatility=0.0045, sharpe=6.7188, avg_turnover=0.1009, max_drawdown=0.0000.
- common_allocator, validation, equal_weight: avg_return=0.0404, volatility=0.0079, sharpe=5.1422, avg_turnover=0.0000, max_drawdown=0.0000.
- original_report, test, e2e_nn_original: avg_return=0.0374, volatility=0.0077, sharpe=4.8521, avg_turnover=0.1116, max_drawdown=0.0000.
- original_report, test, pto_nn_original: avg_return=0.0386, volatility=0.0106, sharpe=3.6312, avg_turnover=0.0330, max_drawdown=0.0000.
- original_report, validation, pto_nn_original: avg_return=0.0321, volatility=0.0045, sharpe=7.1834, avg_turnover=0.0258, max_drawdown=0.0000.
- original_report, validation, e2e_nn_original: avg_return=0.0305, volatility=0.0045, sharpe=6.7188, avg_turnover=0.1009, max_drawdown=0.0000.

## Notes
- The common-allocator rows are the cleanest apples-to-apples predictor comparison.
- The original PTO/E2E rows are retained as implementation references from the existing reports, because those setups selected their own optimizer configurations internally.
