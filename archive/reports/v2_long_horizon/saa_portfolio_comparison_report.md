# XOPTPOE v2 SAA Portfolio Comparison Report

## Setup
- These are long-horizon SAA decision-period diagnostics on overlapping annualized forward excess-return labels, not a fully tradable non-overlapping monthly wealth backtest.
- Best 60m and 120m predictors were selected by validation RMSE. Their signals were then converted into monthly sleeve allocations with the existing long-only robust allocator.
- Predictor-driven portfolios use a common allocator design: long-only, fully invested, no leverage, default v2 risk model, validation-selected lambda/kappa/Omega from the existing compact grid.
- PTO_NN and E2E_NN appear twice conceptually: as predictor-family entries in the horse race, and here as original setup reference rows loaded from the existing reports.

## Predictor Winners Feeding Portfolios
- Best 60m signal source: `elastic_net__core_plus_interactions__separate_60`.
- Best 120m signal source: `ridge__full_firstpass__separate_120`.
- Best shared benchmark source: `elastic_net__core_plus_interactions__shared_60_120`.

## Portfolio Metrics
- common_allocator, test, best_60_predictor: avg_return=0.0618, volatility=0.0084, sharpe=7.3482, avg_turnover=0.0493, max_drawdown=0.0000.
- common_allocator, test, combined_60_120_predictor: avg_return=0.0687, volatility=0.0101, sharpe=6.7970, avg_turnover=0.0647, max_drawdown=0.0000.
- common_allocator, test, e2e_nn_signal: avg_return=0.0334, volatility=0.0052, sharpe=6.4479, avg_turnover=0.0984, max_drawdown=0.0000.
- common_allocator, test, best_shared_predictor: avg_return=0.0679, volatility=0.0106, sharpe=6.3899, avg_turnover=0.0779, max_drawdown=0.0000.
- common_allocator, test, best_120_predictor: avg_return=0.0758, volatility=0.0122, sharpe=6.2171, avg_turnover=0.0836, max_drawdown=0.0000.
- common_allocator, test, pto_nn_signal: avg_return=0.0437, volatility=0.0135, sharpe=3.2350, avg_turnover=0.0253, max_drawdown=0.0000.
- common_allocator, test, equal_weight: avg_return=0.0444, volatility=0.0138, sharpe=3.2092, avg_turnover=0.0000, max_drawdown=0.0000.
- common_allocator, validation, best_120_predictor: avg_return=0.0636, volatility=0.0056, sharpe=11.3076, avg_turnover=0.0299, max_drawdown=0.0000.
- common_allocator, validation, combined_60_120_predictor: avg_return=0.0590, volatility=0.0062, sharpe=9.4619, avg_turnover=0.0229, max_drawdown=0.0000.
- common_allocator, validation, best_shared_predictor: avg_return=0.0584, volatility=0.0068, sharpe=8.5765, avg_turnover=0.0271, max_drawdown=0.0000.
- common_allocator, validation, e2e_nn_signal: avg_return=0.0351, volatility=0.0045, sharpe=7.8486, avg_turnover=0.1015, max_drawdown=0.0000.
- common_allocator, validation, best_60_predictor: avg_return=0.0545, volatility=0.0072, sharpe=7.6063, avg_turnover=0.0183, max_drawdown=0.0000.
- common_allocator, validation, pto_nn_signal: avg_return=0.0394, volatility=0.0060, sharpe=6.5258, avg_turnover=0.0231, max_drawdown=0.0000.
- common_allocator, validation, equal_weight: avg_return=0.0425, volatility=0.0075, sharpe=5.6765, avg_turnover=0.0000, max_drawdown=0.0000.
- original_report, test, e2e_nn_original: avg_return=0.0340, volatility=0.0061, sharpe=5.6016, avg_turnover=0.1316, max_drawdown=0.0000.
- original_report, test, pto_nn_original: avg_return=0.0437, volatility=0.0135, sharpe=3.2350, avg_turnover=0.0253, max_drawdown=0.0000.
- original_report, validation, e2e_nn_original: avg_return=0.0371, volatility=0.0049, sharpe=7.5412, avg_turnover=0.1399, max_drawdown=0.0000.
- original_report, validation, pto_nn_original: avg_return=0.0394, volatility=0.0060, sharpe=6.5258, avg_turnover=0.0231, max_drawdown=0.0000.

## Notes
- The common-allocator rows are the cleanest apples-to-apples predictor comparison.
- The original PTO/E2E rows are retained as implementation references from the existing reports, because those setups selected their own optimizer configurations internally.
