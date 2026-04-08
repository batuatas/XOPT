# XOPTPOE v2 E2E Tuning Report

## Scope
- Narrow second-pass study only. Data and feature pipelines stayed frozen.
- E2E search kept the paper-style utility objective and tuned only around lambda, kappa, Omega, and two nearby EWMA/shrinkage risk presets.
- PTO was re-run only as a matched benchmark for diagnostics, using its original compact allocation grid rather than a new PTO tuning sweep.
- Stage 1 searched 24 shared-horizon candidates at 15 epochs / patience 5; stage 2 re-ran the top 4 at 25 epochs / patience 8.

## Selected Tuned E2E Configuration
- risk_preset=smoother, lambda=10.0, kappa=0.25, omega=identity, ewma_beta=0.97, shrinkage=0.2, epochs=15, patience=5.
- validation Sharpe=7.4116, test Sharpe=3.5505, test avg return=0.0413, test volatility=0.0116.

## Shared PTO vs Tuned Shared E2E
- PTO test: avg return=0.0435, volatility=0.0135, Sharpe=3.2340, turnover=0.0251.
- Tuned E2E test: avg return=0.0413, volatility=0.0116, Sharpe=3.5505, turnover=0.0386.
- Equal weight test: avg return=0.0444, volatility=0.0138, Sharpe=3.2092.

## Signal vs Risk-Control Diagnostics
- PTO aggregated test rank IC=-0.0397, prediction dispersion=0.0263.
- E2E aggregated test rank IC=-0.2183, prediction dispersion=0.0115.
- If Sharpe rises while average return and rank IC do not, the gain is risk-control-driven rather than signal-driven.

## Horizon Ablation
- Best shared setup on test Sharpe: e2e with Sharpe=3.5505, avg return=0.0413.
- Best separate-horizon E2E setup on test Sharpe: separate_120 with Sharpe=4.4860, avg return=0.0367.

## Takeaways
- The tuning objective remained utility-focused, in line with the stronger regime reported in the paper.
- Shared-vs-separate comparisons were run without changing the frozen data or feature design.
- Optimizer behavior summary reports concentration, turnover, predicted-signal dispersion, and sleeve top-weight frequencies so the Sharpe source can be diagnosed directly.
