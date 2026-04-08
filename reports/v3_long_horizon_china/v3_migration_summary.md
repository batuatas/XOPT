# XOPTPOE v3 Downstream Migration Summary

## Status
- v3 is the active downstream branch.
- v1 and v2 remain untouched benchmark branches.
- Integrity checks: 7 PASS, 0 FAIL.

## What Changed Versus v2
- Best 60m predictor changed from `elastic_net__core_plus_interactions__separate_60` in v2 to `elastic_net__full_firstpass__separate_60` in v3.
- Best 120m predictor remained a full-firstpass ridge winner, with v2 `ridge__full_firstpass__separate_120` and v3 `ridge__full_firstpass__separate_120`.
- Equal-weight test Sharpe moved from 3.2092 in the 8-sleeve branch to 2.7019 in the 9-sleeve branch; the equal-weight max weight fell from 0.1250 to 0.1111 because the benchmark is now 1/9 instead of 1/8.

## China Sleeve Readout
- EQ_CN predictive quality is mixed: in the best shared predictor on the test split, rmse=0.0309, corr=0.6582, sign_accuracy=0.5625.
- EQ_CN remains marginal in the strongest allocation diagnostics: best_shared avg_weight=0.0000, max_weight=0.0000; best_60 avg_weight=0.0205.
- EQ_CN never becomes the top-weight sleeve in the monitored test strategies; top_weight_frequency stays 0.0000 in best_shared and 0.0000 in best_60.

## Active Benchmark
- Best test portfolio behavior in v3 is `best_shared_predictor` with avg_return=0.0695, sharpe=6.4291, avg_turnover=0.0750.
- PTO and E2E should now be judged against the strongest v3 supervised benchmark rather than against equal weight alone.

## Interpretation
- Adding EQ_CN made the active branch coherent as a 9-sleeve system, but it did not make China a dominant allocation sleeve in the current benchmark stack.
- The best supervised predictors remain linear and separate-horizon. PTO/E2E still lag those benchmarks on prediction quality.
