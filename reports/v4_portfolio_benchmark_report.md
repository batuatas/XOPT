# XOPTPOE v4 Portfolio Benchmark Report

## Scope
- Active v4 paths only. v1/v2/v3 were not touched.
- These are long-horizon SAA decision-period diagnostics on overlapping annualized forward excess-return labels, not a clean tradable monthly wealth backtest.
- `CR_EU_HY` remains excluded from the default supervised benchmark roster by governance lock.
- Combined 60m+120m uses the project convention: a simple equal average of the selected 60m and 120m predicted annualized excess-return signals.
- Common-allocator rows use the same long-only robust allocator family with validation-selected lambda/kappa/Omega. One transparent concentration-control variant is included for interpretation.

## Benchmark Signals
- 60m benchmark: `elastic_net__core_plus_interactions__separate_60`
- 120m benchmark: `ridge__core_plus_interactions__separate_120`
- shared benchmark: `elastic_net__core_plus_interactions__shared_60_120`

## Test Portfolio Readout
- best_60_predictor: avg_return=0.0645, volatility=0.0089, sharpe=7.2771, avg_turnover=0.0495, avg_max_weight=0.5742, avg_effective_n=2.0605.
- combined_60_120_predictor: avg_return=0.0707, volatility=0.0102, sharpe=6.9545, avg_turnover=0.0638, avg_max_weight=0.6300, avg_effective_n=2.0188.
- best_shared_predictor: avg_return=0.0692, volatility=0.0105, sharpe=6.5664, avg_turnover=0.0713, avg_max_weight=0.6165, avg_effective_n=2.0088.
- best_120_predictor: avg_return=0.0767, volatility=0.0117, sharpe=6.5473, avg_turnover=0.0797, avg_max_weight=0.6714, avg_effective_n=1.9262.
- combined_diversified_cap: avg_return=0.0573, volatility=0.0109, sharpe=5.2731, avg_turnover=0.0716, avg_max_weight=0.3000, avg_effective_n=4.3803.
- equal_weight: avg_return=0.0296, volatility=0.0133, sharpe=2.2251, avg_turnover=0.0000, avg_max_weight=0.0714, avg_effective_n=14.0000.

## Signal Choice
- best_60_predictor: sharpe=7.2771, avg_return=0.0645, top_weight_sleeve=EQ_US, top_weight_freq=0.9167.
- best_120_predictor: sharpe=6.5473, avg_return=0.0767, top_weight_sleeve=EQ_US, top_weight_freq=1.0000.
- combined_60_120_predictor: sharpe=6.9545, avg_return=0.0707, top_weight_sleeve=EQ_US, top_weight_freq=1.0000.
- best_shared_predictor: sharpe=6.5664, avg_return=0.0692, top_weight_sleeve=EQ_US, top_weight_freq=1.0000.
- combined_diversified_cap: sharpe=5.2731, avg_return=0.0573, top_weight_sleeve=EQ_US, top_weight_freq=1.0000.

## China Sleeve Readout
- best_120_predictor: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=1.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0179.
- best_60_predictor: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=1.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0179.
- best_shared_predictor: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=0.8333, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0179.
- combined_60_120_predictor: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=0.8750, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0179.
- combined_diversified_cap: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=0.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0179.
- equal_weight: avg_weight=0.0714, max_weight=0.0714, nonzero_alloc_share=1.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=0.0000.

## New Sleeve Readout
- LISTED_INFRA: avg_weight=0.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0621.
- CR_US_HY: avg_weight=0.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0492.
- LISTED_RE: avg_weight=0.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=0.0132.
- CR_EU_IG: avg_weight=0.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=0.0079.
- FI_EU_GOVT: avg_weight=0.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=0.0052.

## Concentration And Attribution
- strongest supervised portfolio benchmark on test: `best_60_predictor`
- EQ_US: avg_weight=0.5713, total_active_contribution_vs_equal=1.2482, abs_active_share=0.6762.
- ALT_GLD: avg_weight=0.0014, total_active_contribution_vs_equal=-0.1035, abs_active_share=0.0560.
- EQ_JP: avg_weight=0.0000, total_active_contribution_vs_equal=-0.0733, abs_active_share=0.0397.
- FI_UST: avg_weight=0.3888, total_active_contribution_vs_equal=0.0663, abs_active_share=0.0359.
- LISTED_INFRA: avg_weight=0.0000, total_active_contribution_vs_equal=-0.0621, abs_active_share=0.0337.

## Interpretation
- Strong portfolio behavior here means better decision-period allocation diagnostics under the shared 60m/120m portfolio objective, not proof of a tradable monthly wealth process.
- The key question is whether gains come from broad sleeve use or from concentration in a small subset of sleeves and months.
- Single strongest supervised portfolio benchmark to beat next: `best_60_predictor`.
