# XOPTPOE v3 Benchmark Refinement Report

## Scope
- Active v3 paths only. v1/v2 were not touched.
- These remain long-horizon SAA decision-period diagnostics on overlapping forward labels, not a clean monthly tradable wealth backtest.
- The objective here is benchmark lock-in: strongest raw supervised benchmark versus strongest robust supervised benchmark.

## Starting Point
- 60m signal benchmark: `elastic_net__full_firstpass__separate_60`
- 120m signal benchmark: `ridge__full_firstpass__separate_120`
- shared reference: `ridge__full_firstpass__shared_60_120`

## Why The Current 120m Winner Is Concentrated
- `best_120_predictor` test Sharpe=6.1859, avg_return=0.0767.
- allocation concentration: avg_max_weight=0.6844, effective_n=1.8878, top2_sleeve_active_share=0.7809.
- signal concentration: top_signal_sleeve=EQ_US, top_signal_freq=1.0000, signal_avg_top_minus_second=0.0403.
- month concentration is less severe than sleeve concentration: top5_positive_active_month_share=0.2889.

## 60m Versus 120m Concentration
- `best_60_predictor`: sharpe=5.8271, avg_max_weight=0.2957, effective_n=4.7671, top2_active_share=0.6445, signal_top_freq=0.6667.
- `best_120_predictor`: sharpe=6.1859, avg_max_weight=0.6844, effective_n=1.8878, top2_active_share=0.7809, signal_top_freq=1.0000.
- Interpretation: 120m is more concentration-prone in the current setup on both signal and allocation diagnostics.

## Test Candidate Readout
- best_120_predictor: sharpe=6.1859, avg_return=0.0767, avg_max_weight=0.6844, effective_n=1.8878, top2_active_share=0.7809, turnover=0.0840.
- best_60_predictor: sharpe=5.8271, avg_return=0.0513, avg_max_weight=0.2957, effective_n=4.7671, top2_active_share=0.6445, turnover=0.0553.
- combined_std_60tilt: sharpe=5.7346, avg_return=0.0879, avg_max_weight=0.7161, effective_n=1.6777, top2_active_share=0.7484, turnover=0.0434.
- combined_60_120_predictor: sharpe=5.7325, avg_return=0.0694, avg_max_weight=0.6028, effective_n=2.1885, top2_active_share=0.7622, turnover=0.1040.
- combined_std_equal: sharpe=5.7313, avg_return=0.0911, avg_max_weight=0.7678, effective_n=1.5512, top2_active_share=0.7749, turnover=0.0501.
- combined_std_120tilt: sharpe=5.6947, avg_return=0.0946, avg_max_weight=0.8269, effective_n=1.4036, top2_active_share=0.8086, turnover=0.0597.
- best_shared_predictor: sharpe=5.5384, avg_return=0.0663, avg_max_weight=0.5535, effective_n=2.4182, top2_active_share=0.7238, turnover=0.0985.
- combined_std_120tilt_top_k_capped: sharpe=5.1467, avg_return=0.0649, avg_max_weight=0.3500, effective_n=3.0655, top2_active_share=0.5829, turnover=0.0197.
- best_120_top_k_capped: sharpe=5.0998, avg_return=0.0632, avg_max_weight=0.3500, effective_n=3.4732, top2_active_share=0.6280, turnover=0.0383.
- e2e_nn_signal: sharpe=4.8521, avg_return=0.0374, avg_max_weight=0.2995, effective_n=4.7996, top2_active_share=0.3455, turnover=0.1116.
- combined_std_120tilt_breadth_blend_capped: sharpe=4.6594, avg_return=0.0589, avg_max_weight=0.3000, effective_n=4.6405, top2_active_share=0.6739, turnover=0.0302.
- best_120_diversified_cap: sharpe=4.5038, avg_return=0.0593, avg_max_weight=0.3000, effective_n=4.0816, top2_active_share=0.6041, turnover=0.0644.
- best_120_score_positive_capped: sharpe=4.2678, avg_return=0.0566, avg_max_weight=0.2944, effective_n=4.8720, top2_active_share=0.6385, turnover=0.0772.
- best_120_breadth_blend_capped: sharpe=3.9218, avg_return=0.0552, avg_max_weight=0.2842, effective_n=5.6066, top2_active_share=0.6713, turnover=0.0552.
- pto_nn_signal: sharpe=3.6312, avg_return=0.0386, avg_max_weight=0.2663, effective_n=5.8597, top2_active_share=0.3643, turnover=0.0330.
- combined_std_120tilt_diversified_cap: sharpe=3.3064, avg_return=0.0524, avg_max_weight=0.2292, effective_n=4.6027, top2_active_share=0.5031, turnover=0.1262.
- combined_std_120tilt_score_positive_capped: sharpe=2.8712, avg_return=0.0508, avg_max_weight=0.1993, effective_n=6.9169, top2_active_share=0.5414, turnover=0.0776.
- equal_weight: sharpe=2.7019, avg_return=0.0406, avg_max_weight=0.1111, effective_n=9.0000, top2_active_share=nan, turnover=0.0000.

## Raw Versus Robust Winner
- strongest raw benchmark: `best_120_predictor`
  - test avg_return=0.0767, sharpe=6.1859, avg_max_weight=0.6844, effective_n=1.8878.
- strongest robust benchmark: `best_60_predictor`
  - test avg_return=0.0513, sharpe=5.8271, avg_max_weight=0.2957, effective_n=4.7671.
- concentration reduction vs raw winner: delta_max_weight=-0.3887, delta_effective_n=2.8793, delta_top2_active_share=-0.1364.
- candidates passing the same robust screen on both validation and test: 0.

## EQ_CN Diagnostics
- best_120_predictor: avg_weight=0.0000, max_weight=0.0000, nonzero_share=0.1250, top_weight_freq=0.0000, active_contribution_vs_equal=-0.0279.
- best_60_predictor: avg_weight=0.0205, max_weight=0.0872, nonzero_share=0.5000, top_weight_freq=0.0000, active_contribution_vs_equal=-0.0138.
- combined_60_120_predictor: avg_weight=0.0000, max_weight=0.0001, nonzero_share=0.1667, top_weight_freq=0.0000, active_contribution_vs_equal=-0.0279.
- combined_std_120tilt_breadth_blend_capped: avg_weight=0.0428, max_weight=0.0872, nonzero_share=1.0000, top_weight_freq=0.0000, active_contribution_vs_equal=-0.0122.
- EQ_CN remains marginal in the raw 120m winner and becomes more economically meaningful only under concentration controls.

## Attribution
- raw winner top contributors: EQ_US(70.04%), ALT_GLD(8.05%), EQ_JP(5.57%), RE_US(5.24%), EQ_EM(4.33%)
- robust winner top contributors: EQ_US(50.48%), ALT_GLD(13.97%), EQ_EM(9.74%), EQ_EZ(9.66%), RE_US(5.87%)

## Neural Context
- equal_weight: avg_return=0.0406, sharpe=2.7019.
- PTO: avg_return=0.0386, sharpe=3.6312.
- E2E: avg_return=0.0374, sharpe=4.8521.
- PTO/E2E remain reference comparators here, not the lock-in winners.

## Interpretation
- The main failure mode of the raw 120m winner is sleeve concentration, especially EQ_US dominance, more than concentration in a handful of months.
- The right carry-forward benchmark is the strongest supervised candidate that preserves most of the performance while materially improving breadth.
- Any later PTO/E2E or scenario-generation layer should be forced to beat the robust benchmark first, not only the raw winner or equal weight.
