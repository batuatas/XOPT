# XOPTPOE v3 Portfolio Benchmark Report

## Scope
- Active v3 paths only. v1/v2 were not touched.
- These are long-horizon SAA decision-period diagnostics on overlapping annualized forward excess-return labels, not a clean tradable monthly wealth backtest.
- Signal sources follow the latest v3 prediction benchmark results.
- Combined 60m+120m uses the existing project convention: a simple equal average of the two raw predicted annualized excess-return signals.
- Common-allocator rows use the same long-only robust allocator with validation-selected lambda/kappa/Omega. Concentration-control rows use fixed transparent heuristic caps.

## Benchmark Signals
- 60m benchmark: `elastic_net__full_firstpass__separate_60`
- 120m benchmark: `ridge__full_firstpass__separate_120`
- shared benchmark: `ridge__full_firstpass__shared_60_120`

## Test Portfolio Readout
- best_120_predictor: avg_return=0.0767, volatility=0.0124, sharpe=6.1859, avg_turnover=0.0840, avg_max_weight=0.6844, avg_effective_n=1.8878.
- best_60_predictor: avg_return=0.0513, volatility=0.0088, sharpe=5.8271, avg_turnover=0.0553, avg_max_weight=0.2957, avg_effective_n=4.7671.
- combined_60_120_predictor: avg_return=0.0694, volatility=0.0121, sharpe=5.7325, avg_turnover=0.1040, avg_max_weight=0.6028, avg_effective_n=2.1885.
- combined_top_k_capped: avg_return=0.0709, volatility=0.0128, sharpe=5.5632, avg_turnover=0.0476, avg_max_weight=0.4423, avg_effective_n=2.7635.
- best_shared_predictor: avg_return=0.0663, volatility=0.0120, sharpe=5.5384, avg_turnover=0.0985, avg_max_weight=0.5535, avg_effective_n=2.4182.
- combined_top_k_equal: avg_return=0.0640, volatility=0.0128, sharpe=5.0129, avg_turnover=0.0556, avg_max_weight=0.3333, avg_effective_n=3.0000.
- e2e_nn_signal: avg_return=0.0374, volatility=0.0077, sharpe=4.8521, avg_turnover=0.1116, avg_max_weight=0.2995, avg_effective_n=4.7996.
- combined_score_positive_capped: avg_return=0.0571, volatility=0.0123, sharpe=4.6568, avg_turnover=0.0741, avg_max_weight=0.3114, avg_effective_n=5.0105.
- combined_diversified_cap: avg_return=0.0581, volatility=0.0132, sharpe=4.4049, avg_turnover=0.0520, avg_max_weight=0.3000, avg_effective_n=4.2819.
- pto_nn_signal: avg_return=0.0386, volatility=0.0106, sharpe=3.6312, avg_turnover=0.0330, avg_max_weight=0.2663, avg_effective_n=5.8597.
- equal_weight: avg_return=0.0406, volatility=0.0150, sharpe=2.7019, avg_turnover=0.0000, avg_max_weight=0.1111, avg_effective_n=9.0000.

## Signal Choice
- best_60_predictor: sharpe=5.8271, avg_return=0.0513, top_weight_sleeve=EQ_US, top_weight_freq=1.0000.
- best_120_predictor: sharpe=6.1859, avg_return=0.0767, top_weight_sleeve=EQ_US, top_weight_freq=1.0000.
- combined_60_120_predictor: sharpe=5.7325, avg_return=0.0694, top_weight_sleeve=EQ_US, top_weight_freq=1.0000.
- best_shared_predictor: sharpe=5.5384, avg_return=0.0663, top_weight_sleeve=EQ_US, top_weight_freq=1.0000.

## China Sleeve Readout
- best_120_predictor: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=0.1250, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0279.
- best_60_predictor: avg_weight=0.0205, max_weight=0.0872, nonzero_alloc_share=0.5000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0138.
- best_shared_predictor: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=0.2083, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0279.
- combined_60_120_predictor: avg_weight=0.0000, max_weight=0.0001, nonzero_alloc_share=0.1667, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0279.
- combined_diversified_cap: avg_weight=0.0448, max_weight=0.1672, nonzero_alloc_share=0.3333, top_weight_frequency=0.0000, total_active_contribution_vs_equal=0.0179.
- combined_score_positive_capped: avg_weight=0.0569, max_weight=0.1372, nonzero_alloc_share=0.9583, top_weight_frequency=0.0000, total_active_contribution_vs_equal=0.0064.
- combined_top_k_capped: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=0.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0279.
- combined_top_k_equal: avg_weight=0.0000, max_weight=0.0000, nonzero_alloc_share=0.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0279.
- e2e_nn_signal: avg_weight=0.0073, max_weight=0.0338, nonzero_alloc_share=0.6667, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0278.
- equal_weight: avg_weight=0.1111, max_weight=0.1111, nonzero_alloc_share=1.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=0.0000.
- pto_nn_signal: avg_weight=0.0540, max_weight=0.0994, nonzero_alloc_share=1.0000, top_weight_frequency=0.0000, total_active_contribution_vs_equal=-0.0255.

## Concentration And Attribution
- strongest supervised benchmark on test: `best_120_predictor`
- EQ_US: avg_weight=0.6844, total_active_contribution_vs_equal=1.4338, abs_active_share=0.7004.
- ALT_GLD: avg_weight=0.0005, total_active_contribution_vs_equal=-0.1647, abs_active_share=0.0805.
- EQ_JP: avg_weight=0.0000, total_active_contribution_vs_equal=-0.1139, abs_active_share=0.0557.
- RE_US: avg_weight=0.0137, total_active_contribution_vs_equal=-0.1072, abs_active_share=0.0524.
- EQ_EM: avg_weight=0.0000, total_active_contribution_vs_equal=-0.0886, abs_active_share=0.0433.

## Neural Comparison
- PTO common-allocator: avg_return=0.0386, sharpe=3.6312, avg_turnover=0.0330.
- E2E common-allocator: avg_return=0.0374, sharpe=4.8521, avg_turnover=0.1116.
- strongest supervised benchmark vs PTO: delta_avg_return=0.0381, delta_sharpe=2.5546.
- strongest supervised benchmark vs E2E: delta_avg_return=0.0393, delta_sharpe=1.3337.

## Interpretation
- If a strategy wins on Sharpe here, that is a decision-period SAA allocation diagnostic, not proof of a tradable overlapping-label wealth process.
- The critical distinction is whether the apparent gain comes from broad signal use or from concentration in EQ_US / a few sleeves / a few months.
