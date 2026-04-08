# XOPTPOE v3 Benchmark Lock-In

## Lock-In Rules
- strongest raw benchmark = highest test-Sharpe supervised candidate
- strongest robust benchmark = highest test-Sharpe supervised candidate that passes all of:
  - avg_max_weight <= 0.45
  - avg_effective_n_sleeves >= 3.0
  - top2_sleeve_active_share_abs <= 0.65
  - sharpe retention vs raw 120m benchmark >= 0.85

## Winners
- strongest raw benchmark: `best_120_predictor`
  - test avg_return=0.0767, sharpe=6.1859, avg_max_weight=0.6844, effective_n=1.8878, top2_active_share=0.7809
- strongest robust benchmark: `best_60_predictor`
  - test avg_return=0.0513, sharpe=5.8271, avg_max_weight=0.2957, effective_n=4.7671, top2_active_share=0.6445

## Stability Note
- candidates passing the robust screen on both validation and test: 0
- This lock-in is therefore a cautious working benchmark choice, not proof of a fully stable concentration-controlled allocation rule.

## Interpretation
- Selected as the highest-Sharpe supervised candidate that passed the explicit concentration screen.
- The raw winner is the performance ceiling benchmark.
- The robust winner is the pre-scenario benchmark that future PTO/E2E and scenario-generation work should be forced to beat.

