# XOPTPOE v3 PTO Report

## Implemented PTO Logic
- Shared 60m/120m predictor trained on row-level annualized excess targets only; no optimization layer in training.
- Paper-style MLP architecture: hidden layers `32 -> 16 -> 8`, ReLU activations, batch normalization, dropout `0.5`.
- Training hyperparameters mirrored from the paper where applicable: AdamW, learning rate `5e-5`, weight decay `1e-5`, max epochs `50`, patience `10`.
- Feature set used: `core_plus_enrichment` with 281 transformed inputs after missing indicators.

## SAA Adaptations
- The paper’s firm-level 1m stock ranking step is removed; the investable universe is the active 9-sleeve v3 XOPTPOE universe with EQ_CN included.
- The shared model still predicts horizon-conditioned 60m and 120m annualized excess returns row by row.
- For downstream SAA portfolio construction, the two horizon predictions are averaged sleeve-by-sleeve into one monthly signal before optimization.
- Risk estimation uses trailing monthly sleeve excess returns from the active v3-compatible 9-sleeve history, combining the frozen baseline sleeves with the versioned China sleeve returns, and keeps the expanding-to-60m EWMA window to avoid collapsing the long-horizon split.

## Optimizer Layer
- Selected validation config: lambda=10.0, kappa=0.1, omega=identity.
- Portfolio problem solved: maximize `w' mu_hat - kappa * sqrt(w' Omega w) - (lambda/2) * w' Sigma w`, subject to `sum(w)=1`, `w>=0`.
- Sigma: annualized EWMA covariance of trailing monthly sleeve excess returns, beta `0.94`, diagonal shrinkage `0.10`, ridge `1e-6`, window cap `60` months.
- Omega candidates searched: `diag(Sigma)` and `I`.

## Predictor Training
- Best validation-MSE epoch: `1`.
- Final best validation MSE: `0.007209`.

## Allocation Selection Summary
- lambda=10.0, kappa=0.1, omega=identity: validation_sharpe=7.1834, validation_avg_return=0.0321.
- lambda=5.0, kappa=0.1, omega=identity: validation_sharpe=7.1656, validation_avg_return=0.0329.
- lambda=10.0, kappa=1.0, omega=identity: validation_sharpe=6.0191, validation_avg_return=0.0377.
- lambda=5.0, kappa=1.0, omega=identity: validation_sharpe=5.6292, validation_avg_return=0.0388.
- lambda=10.0, kappa=0.1, omega=diag: validation_sharpe=5.5597, validation_avg_return=0.0274.

## Prediction Metrics
- test: rmse=0.065165, mae=0.053797, oos_r2_vs_naive=-1.7844, corr=0.0066, directional_accuracy=0.3634.
- validation: rmse=0.084905, mae=0.073160, oos_r2_vs_naive=-3.3794, corr=-0.0777, directional_accuracy=0.1412.
- By horizon:
  - test, 60m: rmse=0.063294, corr=-0.0199, oos_r2_vs_naive=-1.4944.
  - test, 120m: rmse=0.066983, corr=-0.0103, oos_r2_vs_naive=-2.1069.
  - validation, 60m: rmse=0.092085, corr=-0.1622, oos_r2_vs_naive=-2.3322.
  - validation, 120m: rmse=0.077059, corr=-0.1404, oos_r2_vs_naive=-6.9448.

## Portfolio Metrics
- test, equal_weight: avg_return=0.0406, volatility=0.0150, sharpe=2.7019, max_drawdown=0.0000, avg_turnover=0.0000.
- test, pto_portfolio: avg_return=0.0386, volatility=0.0106, sharpe=3.6312, max_drawdown=0.0000, avg_turnover=0.0330.
- validation, equal_weight: avg_return=0.0404, volatility=0.0079, sharpe=5.1422, max_drawdown=0.0000, avg_turnover=0.0000.
- validation, pto_portfolio: avg_return=0.0321, volatility=0.0045, sharpe=7.1834, max_drawdown=0.0000, avg_turnover=0.0258.

## Notes
- Portfolio metrics are decision-period diagnostics on overlapping long-horizon annualized outcomes, not a fully non-overlapping wealth backtest.
