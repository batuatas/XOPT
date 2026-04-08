# XOPTPOE v2 PTO Report

## Implemented PTO Logic
- Shared 60m/120m predictor trained on row-level annualized excess targets only; no optimization layer in training.
- Paper-style MLP architecture: hidden layers `32 -> 16 -> 8`, ReLU activations, batch normalization, dropout `0.5`.
- Training hyperparameters mirrored from the paper where applicable: AdamW, learning rate `5e-5`, weight decay `1e-5`, max epochs `50`, patience `10`.
- Feature set used: `core_plus_enrichment` with 280 transformed inputs after missing indicators.

## SAA Adaptations
- The paper’s firm-level 1m stock ranking step is removed; the investable universe is the fixed 8-sleeve XOPTPOE universe.
- The shared model still predicts horizon-conditioned 60m and 120m annualized excess returns row by row.
- For downstream SAA portfolio construction, the two horizon predictions are averaged sleeve-by-sleeve into one monthly signal before optimization.
- Risk estimation uses trailing monthly sleeve excess returns from the frozen v1 history, with an expanding-to-60m EWMA window instead of adding a new 60m sample burn-in that would collapse the long-horizon split.

## Optimizer Layer
- Selected validation config: lambda=10.0, kappa=1.0, omega=identity.
- Portfolio problem solved: maximize `w' mu_hat - kappa * sqrt(w' Omega w) - (lambda/2) * w' Sigma w`, subject to `sum(w)=1`, `w>=0`.
- Sigma: annualized EWMA covariance of trailing monthly sleeve excess returns, beta `0.94`, diagonal shrinkage `0.10`, ridge `1e-6`, window cap `60` months.
- Omega candidates searched: `diag(Sigma)` and `I`.

## Predictor Training
- Best validation-MSE epoch: `24`.
- Final best validation MSE: `0.010306`.

## Allocation Selection Summary
- lambda=10.0, kappa=1.0, omega=identity: validation_sharpe=6.5258, validation_avg_return=0.0394.
- lambda=5.0, kappa=1.0, omega=identity: validation_sharpe=6.1246, validation_avg_return=0.0408.
- lambda=10.0, kappa=0.1, omega=identity: validation_sharpe=4.9826, validation_avg_return=0.0318.
- lambda=5.0, kappa=0.1, omega=identity: validation_sharpe=4.6511, validation_avg_return=0.0342.
- lambda=10.0, kappa=1.0, omega=diag: validation_sharpe=4.5335, validation_avg_return=0.0250.

## Prediction Metrics
- test: rmse=0.126891, mae=0.109682, oos_r2_vs_naive=-9.9040, corr=-0.0070, directional_accuracy=0.2344.
- validation: rmse=0.101518, mae=0.082682, oos_r2_vs_naive=-5.6227, corr=0.0394, directional_accuracy=0.3516.
- By horizon:
  - test, 60m: rmse=0.123982, corr=-0.0278, oos_r2_vs_naive=-9.0340.
  - test, 120m: rmse=0.129735, corr=0.0040, oos_r2_vs_naive=-10.8418.
  - validation, 60m: rmse=0.088306, corr=-0.0067, oos_r2_vs_naive=-2.1578.
  - validation, 120m: rmse=0.113198, corr=-0.0592, oos_r2_vs_naive=-18.9320.

## Portfolio Metrics
- test, equal_weight: avg_return=0.0444, volatility=0.0138, sharpe=3.2092, max_drawdown=0.0000, avg_turnover=0.0000.
- test, pto_portfolio: avg_return=0.0437, volatility=0.0135, sharpe=3.2350, max_drawdown=0.0000, avg_turnover=0.0253.
- validation, equal_weight: avg_return=0.0425, volatility=0.0075, sharpe=5.6765, max_drawdown=0.0000, avg_turnover=0.0000.
- validation, pto_portfolio: avg_return=0.0394, volatility=0.0060, sharpe=6.5258, max_drawdown=0.0000, avg_turnover=0.0231.

## Notes
- Portfolio metrics are decision-period diagnostics on overlapping long-horizon annualized outcomes, not a fully non-overlapping wealth backtest.
