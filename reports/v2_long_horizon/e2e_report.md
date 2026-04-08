# XOPTPOE v2 E2E / PAO Report

## Implemented PAO Logic
- Shared 60m/120m predictor uses the same paper-style MLP as PTO: `32 -> 16 -> 8`, ReLU, batch normalization, dropout `0.5`.
- Training is decision-focused: row-level predictions are aggregated into one monthly sleeve signal and passed through a differentiable robust optimizer built with CVXPY + cvxpylayers.
- Candidate training objectives searched in this run: `utility`.
- Feature set used: `core_plus_enrichment` with 280 transformed inputs after missing indicators.

## SAA Adaptations
- The paper’s stock-ranking universe step is removed because the SAA universe is the fixed 8-sleeve investable universe.
- A single monthly SAA decision is formed by averaging 60m and 120m horizon-conditioned predictions sleeve-by-sleeve before optimization.
- The optimizer still follows the paper’s robust long-only mean-variance form, but Sigma is estimated from sleeve-level trailing monthly excess returns rather than stock-level firm panels.
- Metrics are reported on long-horizon annualized excess-return labels, so portfolio Sharpe is computed from decision-period annualized outcomes without an extra monthly-to-annual scaling factor.

## Selected Candidate
- Selected training objective: `utility`.
- Selected validation config: lambda=5.0, kappa=0.1, omega=identity.

## Top Validation Candidates
- objective=utility, lambda=5.0, kappa=0.1, omega=identity, validation_portfolio_sharpe=7.5412, validation_score=0.0282, best_epoch=7.
- objective=utility, lambda=10.0, kappa=0.1, omega=identity, validation_portfolio_sharpe=7.0270, validation_score=0.0145, best_epoch=1.
- objective=utility, lambda=10.0, kappa=1.0, omega=identity, validation_portfolio_sharpe=6.1832, validation_score=-0.0109, best_epoch=15.
- objective=utility, lambda=5.0, kappa=1.0, omega=identity, validation_portfolio_sharpe=6.1618, validation_score=0.0109, best_epoch=1.
- objective=utility, lambda=10.0, kappa=1.0, omega=diag, validation_portfolio_sharpe=5.7245, validation_score=0.0128, best_epoch=1.

## Prediction Metrics
- test: rmse=0.113002, mae=0.095646, oos_r2_vs_naive=-7.6476, corr=-0.1586, directional_accuracy=0.1667.
- validation: rmse=0.155353, mae=0.140541, oos_r2_vs_naive=-14.5092, corr=-0.0519, directional_accuracy=0.1042.
- By horizon:
  - test, 60m: rmse=0.120902, corr=-0.0624, oos_r2_vs_naive=-8.5416.
  - test, 120m: rmse=0.104506, corr=-0.2454, oos_r2_vs_naive=-6.6839.
  - validation, 60m: rmse=0.168960, corr=-0.0438, oos_r2_vs_naive=-10.5605.
  - validation, 120m: rmse=0.140433, corr=-0.0200, oos_r2_vs_naive=-29.6769.

## Portfolio Metrics
- test, e2e_portfolio: avg_return=0.0340, volatility=0.0061, sharpe=5.6016, max_drawdown=0.0000, avg_turnover=0.1316.
- test, equal_weight: avg_return=0.0444, volatility=0.0138, sharpe=3.2092, max_drawdown=0.0000, avg_turnover=0.0000.
- validation, e2e_portfolio: avg_return=0.0371, volatility=0.0049, sharpe=7.5412, max_drawdown=0.0000, avg_turnover=0.1399.
- validation, equal_weight: avg_return=0.0425, volatility=0.0075, sharpe=5.6765, max_drawdown=0.0000, avg_turnover=0.0000.

## Notes
- Model selection used validation portfolio Sharpe as the common selector across PAO candidates after early stopping on each candidate’s own decision objective.
- Portfolio metrics are decision-period diagnostics on overlapping long-horizon annualized outcomes, not a fully non-overlapping wealth backtest.
