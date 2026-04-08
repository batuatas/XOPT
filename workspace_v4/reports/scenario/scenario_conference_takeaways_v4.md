# Scenario Conference Takeaways — v4 First Pass

## Overview
Generated macro scenarios using MALA sampling with VAR(1) plausibility prior,
probing the locked v4 SAA benchmark: elastic_net__core_plus_interactions__separate_60,
best_60_tuned_robust (lambda_risk=8, kappa=0.1, omega=identity).

## A. Gold Allocation Question

**Gold-favorable scenarios** (A1, maximize gold weight):
- Mean ALT_GLD weight in generated scenarios: 23.3%
- Most common regimes: {'higher_for_longer': 134, 'high_stress': 58, 'inflationary_expansion': 48}
- Key macro signals in gold-favorable states:
  - infl_US: 6.24
  - infl_EA: 6.31
  - infl_JP: -0.04
  - short_rate_US: 3.32
  - short_rate_EA: 3.05
  - short_rate_JP: 0.15
  - long_rate_US: 3.63
  - long_rate_EA: 2.79
  - long_rate_JP: 0.76
  - term_slope_US: 0.31

**Gold-adverse scenarios** (A2, minimize gold weight):
- Mean ALT_GLD weight: 21.5%
- Most common regimes: {'higher_for_longer': 86, 'high_stress': 66, 'inflationary_expansion': 43}

## B. Equal-Weight vs Model Question

- Scenarios where model most clearly beats equal weight:
  - Mean predicted portfolio return: 2.1%
  - Dominant regimes: {'higher_for_longer': 87, 'high_stress': 57, 'inflationary_expansion': 18}

## C. House-View Return Question

**10% house-view target:**
  - Valid scenarios: 177
  - Mean model predicted return: 2.2%
  - Dominant regimes: {'higher_for_longer': 81, 'inflationary_expansion': 48}
**6% house-view target:**
  - Valid scenarios: 170
  - Mean model predicted return: 2.1%
  - Dominant regimes: {'higher_for_longer': 148, 'inflationary_expansion': 21}
**7% house-view target:**
  - Valid scenarios: 234
  - Mean model predicted return: 2.4%
  - Dominant regimes: {'higher_for_longer': 123, 'inflationary_expansion': 65}

## D. Diversification Question

- Macro conditions that maximize portfolio entropy:
  - Mean entropy: 1.780
  - Dominant regimes: {'higher_for_longer': 120, 'inflationary_expansion': 22, 'high_stress': 22}

## Methodology Note

- Sampler: MALA (Metropolis-Adjusted Langevin Algorithm)
- MALA parameters: eta=0.005, tau=0.5, n_steps=200, n_seeds=4
- Plausibility: VAR(1) Mahalanobis regularizer fitted on training period
- Regime classification: threshold-based on NFCI + NBER overlays
- State dimension: 19 global macro-financial variables
- Gradient: central finite differences (epsilon=1e-4)

## Limitations of First Pass

- ElasticNet is refit for each anchor date on available training data.
  The model used here approximates the benchmark; the exact benchmark model
  parameters from the accepted prediction_benchmark run should be used for
  production scenarios.
- Realized 60m returns are unavailable for 2023-12 and 2024-12 anchors
  (5-year windows ending 2028-12 and 2029-12 have not happened yet).
  House-view probes use model-predicted returns, which is conservative.
- The regime classifier is rule-based on 8 thresholds; a data-driven
  classifier (HMM or k-means on the macro state) is the next step.