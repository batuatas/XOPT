# Portfolio Stack

## Purpose

This project does not stop at prediction quality. It evaluates how long-horizon sleeve predictions map into a benchmark allocation object.

The accepted portfolio layer is still a benchmark layer:

- long-only
- fully invested
- robust
- interpretable

It is not yet PTO/E2E.
It is not yet scenario generation.

## Portfolio Construction Logic

Predicted sleeve-level long-horizon excess returns feed a robust long-only allocator.

The allocator family used in v4 is:

- maximize expected return
- penalize robustness / uncertainty
- penalize covariance risk
- require nonnegative fully invested weights

## Accepted Portfolio Benchmark Context

The earlier raw strongest portfolio benchmark was more concentrated than desired for presentation and later scenario storytelling.

The project therefore ran an allocator refinement pass around the fixed 5Y predictor and selected a more balanced carry-forward object.

## Locked Active Presentation Benchmark Object

This is the currently locked benchmark object for presentation and current downstream benchmark interpretation:

- Predictor:
  - `elastic_net__core_plus_interactions__separate_60`
- Portfolio label:
  - `best_60_tuned_robust`
- Allocator:
  - `lambda_risk = 8.0`
  - `kappa = 0.10`
  - `omega_type = identity`

## Why This Object Was Chosen

This object was chosen because it is a better middle ground than both extremes:

- less concentrated than the raw strongest 5Y benchmark
- stronger and less defensive than the over-diversified alternatives
- good enough to remain above equal weight on the realized walk-forward path
- more balanced and interpretable for presentation and later scenario work

In plain terms:

- the raw winner was too concentrated
- the heavily regularized alternative was too defensive
- the locked tuned object is the chosen compromise

## Current Portfolio Interpretation

Use this language when describing the benchmark:

- it is a long-horizon SAA decision diagnostic
- it is not a clean monthly trading backtest
- the wealth-path figure uses honest walk-forward annual refits and next-year holds

## What Still Matters About Concentration

Even after tuning, the benchmark is not an equal-weight-like object. It is still a signal-driven strategic benchmark.

That means:

- concentration is reduced, not eliminated
- `EQ_US` remains an important sleeve
- the portfolio is more readable and more defensible than the raw concentrated winner

## Copied Portfolio Artifacts

- [v4_portfolio_benchmark_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_portfolio_benchmark_report.md)
- [v4_portfolio_benchmark_metrics.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_portfolio_benchmark_metrics.csv)
- [v4_portfolio_benchmark_by_sleeve.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_portfolio_benchmark_by_sleeve.csv)
- [v4_portfolio_benchmark_attribution.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_portfolio_benchmark_attribution.csv)
- [v4_allocator_refinement_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_allocator_refinement_report.md)
- [v4_allocator_refinement_results.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_allocator_refinement_results.csv)
- [v4_allocator_refinement_wealth_paths.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_allocator_refinement_wealth_paths.csv)
- [portfolio_benchmark_returns.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/portfolio_benchmark_returns.parquet)

## Active Conference Graphics

The current presentation graphics that correspond to this locked object are:

- [graphic01_why_long_horizon_saa_v4.png](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/plots/graphic01_why_long_horizon_saa_v4.png)
- [graphic02_universe_and_target_v4.png](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/plots/graphic02_universe_and_target_v4.png)
- [graphic03_features_to_ai_prediction_v4.png](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/plots/graphic03_features_to_ai_prediction_v4.png)
- [graphic04_prediction_evidence_v4.png](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/plots/graphic04_prediction_evidence_v4.png)
- [graphic05_benchmark_behavior_v4.png](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/plots/graphic05_benchmark_behavior_v4.png)
