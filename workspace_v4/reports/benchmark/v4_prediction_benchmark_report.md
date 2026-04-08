# XOPTPOE v4 Prediction Benchmark Report

## Scope
- Active paths only: `data/final_v4_expanded_universe/`, `data/modeling_v4/`, `reports/v4_expanded_universe/`.
- Frozen v1/v2/v3 branches were not touched.
- Default benchmark roster excludes `CR_EU_HY` by governance lock because the accepted split design leaves it with only 5 train rows per horizon.
- Fixed-split evidence compares naive, regularized linear, and compact tree baselines across the accepted first-pass feature sets.
- Rolling evidence is intentionally limited to naive and linear baselines.
- Rolling windows use 48 train months for `separate_60`, but shorten to 24 train months for `separate_120` and `shared_60_120` because the accepted 14-sleeve common window is only 71 complete months for those modes.

## Fixed-Split Winners
- Best 60m validation winner: elastic_net__core_plus_interactions__separate_60 | validation_rmse=0.0252 | test_rmse=0.0308 | test_corr=0.6329
- Best 120m validation winner: ridge__core_plus_interactions__separate_120 | validation_rmse=0.0213 | test_rmse=0.0353 | test_corr=0.5673
- Best shared 60m+120m benchmark: elastic_net__core_plus_interactions__shared_60_120 | validation_rmse=0.0249 | test_rmse=0.0328 | test_corr=0.6002

## Rolling Winners
- separate_60: ridge__full_firstpass__separate_60 | mean_test_rmse=0.0438 | mean_test_corr=0.6281 | beat_naive_fold_share=0.60
- separate_120: ridge__full_firstpass__separate_120 | mean_test_rmse=0.0226 | mean_test_corr=0.8453 | beat_naive_fold_share=1.00
- shared_60_120: elastic_net__core_plus_enrichment__shared_60_120 | mean_test_rmse=0.0272 | mean_test_corr=0.7230 | beat_naive_fold_share=1.00

## Model-Family Comparison
- linear: best_validation_rmse=0.0213, best_test_rmse=0.0301, best_test_corr=0.6746.
- tree: best_validation_rmse=0.0262, best_test_rmse=0.0325, best_test_corr=0.6654.
- naive: best_validation_rmse=0.0305, best_test_rmse=0.0391, best_test_corr=0.5795.

## Feature-Set Readout
- separate_60: best feature set=full_firstpass | fixed_best=ridge__full_firstpass__separate_60 | rolling_best=ridge__full_firstpass__separate_60
- separate_120: best feature set=full_firstpass | fixed_best=ridge__full_firstpass__separate_120 | rolling_best=ridge__full_firstpass__separate_120
- shared_60_120: best feature set=core_plus_enrichment | fixed_best=random_forest__core_plus_enrichment__shared_60_120 | rolling_best=elastic_net__core_plus_enrichment__shared_60_120

## China Diagnostics
- EQ_CN | separate_120: experiment=ridge__core_plus_interactions__separate_120, rmse=0.0222, corr=0.3793, sign_accuracy=0.5417, rmse_rank_worst_first=7.
- EQ_CN | separate_60: experiment=elastic_net__core_plus_interactions__separate_60, rmse=0.0299, corr=0.6832, sign_accuracy=0.8333, rmse_rank_worst_first=5.
- EQ_CN | shared_60_120: experiment=elastic_net__core_plus_interactions__shared_60_120, rmse=0.0318, corr=0.6870, sign_accuracy=0.5625, rmse_rank_worst_first=5.

## New Sleeve Diagnostics
- separate_60: selected winner=elastic_net__core_plus_interactions__separate_60
  - sleeve=LISTED_RE, rmse=0.0619, corr=0.0124, sign_accuracy=0.6667, rmse_rank_worst_first=1.
  - sleeve=LISTED_INFRA, rmse=0.0353, corr=0.2685, sign_accuracy=0.9167, rmse_rank_worst_first=3.
  - sleeve=CR_US_HY, rmse=0.0191, corr=0.7692, sign_accuracy=1.0000, rmse_rank_worst_first=11.
  - sleeve=FI_EU_GOVT, rmse=0.0131, corr=0.9039, sign_accuracy=0.7083, rmse_rank_worst_first=12.
  - sleeve=CR_EU_IG, rmse=0.0095, corr=0.9194, sign_accuracy=0.8750, rmse_rank_worst_first=14.
- separate_120: selected winner=ridge__core_plus_interactions__separate_120
  - sleeve=LISTED_INFRA, rmse=0.0304, corr=-0.2562, sign_accuracy=1.0000, rmse_rank_worst_first=4.
  - sleeve=LISTED_RE, rmse=0.0263, corr=0.1173, sign_accuracy=0.3750, rmse_rank_worst_first=5.
  - sleeve=CR_EU_IG, rmse=0.0226, corr=-0.2858, sign_accuracy=0.9167, rmse_rank_worst_first=6.
  - sleeve=FI_EU_GOVT, rmse=0.0153, corr=-0.2582, sign_accuracy=0.9583, rmse_rank_worst_first=13.
  - sleeve=CR_US_HY, rmse=0.0125, corr=-0.0867, sign_accuracy=0.9583, rmse_rank_worst_first=14.
- shared_60_120: selected winner=elastic_net__core_plus_interactions__shared_60_120
  - sleeve=LISTED_RE, rmse=0.0547, corr=0.2925, sign_accuracy=0.3750, rmse_rank_worst_first=2.
  - sleeve=FI_EU_GOVT, rmse=0.0252, corr=0.7189, sign_accuracy=0.3542, rmse_rank_worst_first=9.
  - sleeve=LISTED_INFRA, rmse=0.0195, corr=0.4063, sign_accuracy=0.9583, rmse_rank_worst_first=11.
  - sleeve=CR_US_HY, rmse=0.0190, corr=0.7557, sign_accuracy=1.0000, rmse_rank_worst_first=12.
  - sleeve=CR_EU_IG, rmse=0.0126, corr=0.8414, sign_accuracy=0.8750, rmse_rank_worst_first=14.

## Practical Interpretation
- Separate-horizon models remain the primary benchmark; shared 60m+120m remains an ablation and is not clearly stronger.
- Linear models are the strongest practical family in v4. Trees are serviceable, but they do not displace ridge / elastic net as the default benchmark anchor.
- The richer v4 universe broadens the benchmark problem. It adds usable sleeves, but the new non-US fixed-income sleeves are harder than the legacy core.
- Single strongest benchmark to beat next: elastic_net__core_plus_interactions__separate_60 with test_rmse=0.0308, test_corr=0.6329.
