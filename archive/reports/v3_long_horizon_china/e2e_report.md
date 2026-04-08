# XOPTPOE v3 E2E / PAO Report

## Implemented PAO Logic
- Shared 60m/120m predictor uses the same paper-style MLP as PTO: `32 -> 16 -> 8`, ReLU, batch normalization, dropout `0.5`.
- Training is decision-focused: row-level predictions are aggregated into one monthly sleeve signal and passed through a differentiable robust optimizer built with CVXPY + cvxpylayers.
- Candidate training objectives searched in this run: `utility`.
- Feature set used: `core_plus_enrichment` with 281 transformed inputs after missing indicators.

## SAA Adaptations
- The paper’s stock-ranking universe step is removed because the SAA universe is the active 9-sleeve v3 investable universe with EQ_CN included.
- A single monthly SAA decision is formed by averaging 60m and 120m horizon-conditioned predictions sleeve-by-sleeve before optimization.
- The optimizer still follows the paper’s robust long-only mean-variance form, but Sigma is estimated from v3 sleeve-level trailing monthly excess returns across all 9 sleeves rather than stock-level firm panels.
- Metrics are reported on long-horizon annualized excess-return labels, so portfolio Sharpe is computed from decision-period annualized outcomes without an extra monthly-to-annual scaling factor.

## Selected Candidate
- Selected training objective: `utility`.
- Selected validation config: lambda=10.0, kappa=0.1, omega=identity.

## Top Validation Candidates
- objective=utility, lambda=10.0, kappa=0.1, omega=identity, validation_portfolio_sharpe=6.7188, validation_score=0.0159, best_epoch=10.
- objective=utility, lambda=10.0, kappa=1.0, omega=identity, validation_portfolio_sharpe=6.1377, validation_score=-0.0194, best_epoch=4.
- objective=utility, lambda=5.0, kappa=0.1, omega=identity, validation_portfolio_sharpe=5.7699, validation_score=0.0209, best_epoch=14.
- objective=utility, lambda=5.0, kappa=1.0, omega=diag, validation_portfolio_sharpe=5.5859, validation_score=0.0203, best_epoch=14.
- objective=utility, lambda=5.0, kappa=1.0, omega=identity, validation_portfolio_sharpe=5.5658, validation_score=0.0050, best_epoch=2.

## Prediction Metrics
- test: rmse=0.134494, mae=0.121944, oos_r2_vs_naive=-10.8608, corr=0.0894, directional_accuracy=0.8634.
- validation: rmse=0.201662, mae=0.192415, oos_r2_vs_naive=-23.7054, corr=-0.0447, directional_accuracy=0.8657.
- By horizon:
  - test, 60m: rmse=0.125328, corr=0.1728, oos_r2_vs_naive=-8.7800.
  - test, 120m: rmse=0.143073, corr=0.0306, oos_r2_vs_naive=-13.1749.
  - validation, 60m: rmse=0.189243, corr=-0.0151, oos_r2_vs_naive=-13.0733.
  - validation, 120m: rmse=0.213358, corr=-0.0271, oos_r2_vs_naive=-59.9047.

## Portfolio Metrics
- test, e2e_portfolio: avg_return=0.0374, volatility=0.0077, sharpe=4.8521, max_drawdown=0.0000, avg_turnover=0.1116.
- test, equal_weight: avg_return=0.0406, volatility=0.0150, sharpe=2.7019, max_drawdown=0.0000, avg_turnover=0.0000.
- validation, e2e_portfolio: avg_return=0.0305, volatility=0.0045, sharpe=6.7188, max_drawdown=0.0000, avg_turnover=0.1009.
- validation, equal_weight: avg_return=0.0404, volatility=0.0079, sharpe=5.1422, max_drawdown=0.0000, avg_turnover=0.0000.

## Notes
- Model selection used validation portfolio Sharpe as the common selector across PAO candidates after early stopping on each candidate’s own decision objective.
- Portfolio metrics are decision-period diagnostics on overlapping long-horizon annualized outcomes, not a fully non-overlapping wealth backtest.
