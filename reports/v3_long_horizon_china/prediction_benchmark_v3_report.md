# XOPTPOE v3 Prediction Benchmark Report

## Scope
- Active paths only: `data/final_v3_long_horizon_china/`, `data/modeling_v3/`, `reports/v3_long_horizon_china/`.
- Frozen v1/v2 branches were not touched.
- Fixed-split evidence comes from the active v3 benchmark stack; rolling evidence uses expanding folds on the same 9-sleeve v3 panel.
- Rolling coverage is intentionally focused on naive and linear baselines. Tree and neural models remain in the fixed-split benchmark set, but they are excluded from the rolling loop to keep the robustness pass disciplined and tractable.
- Rolling config: min_train=48 months, validation=12, test=12, step=12.

## Fixed-Split Winners
- Best 60m validation winner: elastic_net__full_firstpass__separate_60 | validation_rmse=0.0296 | test_rmse=0.0374 | test_corr=0.4957
- Best 120m validation winner: ridge__full_firstpass__separate_120 | validation_rmse=0.0207 | test_rmse=0.0390 | test_corr=0.4819
- Best shared 60m+120m benchmark: elastic_net__core_plus_interactions__shared_60_120 | validation_rmse=0.0263 | test_rmse=0.0333 | test_corr=0.5604

## Rolling Winners
- separate_60: ridge__full_firstpass__separate_60 | mean_test_rmse=0.0466 +/- 0.0304 | mean_test_corr=0.6763 | beat_naive_fold_share=0.78
- separate_120: ridge__full_firstpass__separate_120 | mean_test_rmse=0.0267 +/- 0.0063 | mean_test_corr=0.8242 | beat_naive_fold_share=1.00
- shared_60_120: ridge__full_firstpass__shared_60_120 | mean_test_rmse=0.0308 +/- 0.0076 | mean_test_corr=0.7019 | beat_naive_fold_share=1.00

## Model-Family Comparison
- linear: best_validation_rmse=0.0207, best_test_rmse=0.0277, best_test_corr=0.6902.
- tree: best_validation_rmse=0.0263, best_test_rmse=0.0323, best_test_corr=0.6758.
- naive: best_validation_rmse=0.0273, best_test_rmse=0.0380, best_test_corr=0.5222.
- neural: best_validation_rmse=0.0683, best_test_rmse=0.0638, best_test_corr=0.1628.

## Feature-Set Readout
- separate_60: best feature set=full_firstpass | fixed_best=elastic_net__full_firstpass__separate_60 | rolling_best=ridge__full_firstpass__separate_60
- separate_120: best feature set=full_firstpass | fixed_best=ridge__full_firstpass__separate_120 | rolling_best=ridge__full_firstpass__separate_120
- shared_60_120: best feature set=full_firstpass | fixed_best=ridge__full_firstpass__shared_60_120 | rolling_best=ridge__full_firstpass__shared_60_120

## China Diagnostics
- EQ_CN selected winner | separate_120: experiment=ridge__full_firstpass__separate_120, rmse=0.0276, corr=0.1262, sign_accuracy=0.5417, rmse_rank_worst_first=4.
- EQ_CN selected winner | separate_60: experiment=elastic_net__full_firstpass__separate_60, rmse=0.0393, corr=0.3284, sign_accuracy=0.7083, rmse_rank_worst_first=4.
- EQ_CN selected winner | shared_60_120: experiment=elastic_net__core_plus_interactions__shared_60_120, rmse=0.0309, corr=0.6582, sign_accuracy=0.5625, rmse_rank_worst_first=3.
- China-feature-drop test deltas are reported as `minus_china - baseline`; positive RMSE delta means China features helped.
- separate_120 | sleeve=EQ_CN: delta_rmse=0.0017, delta_corr=-0.0943, delta_sign_accuracy=-0.0417.
- separate_120 | sleeve=EQ_EM: delta_rmse=0.0024, delta_corr=-0.1367, delta_sign_accuracy=0.0000.
- separate_120 | sleeve=EQ_EZ: delta_rmse=0.0011, delta_corr=-0.3539, delta_sign_accuracy=0.0417.
- separate_120 | sleeve=EQ_JP: delta_rmse=0.0011, delta_corr=-0.1362, delta_sign_accuracy=0.0000.
- separate_120 | sleeve=EQ_US: delta_rmse=0.0004, delta_corr=-0.0585, delta_sign_accuracy=0.0000.
- separate_60 | sleeve=EQ_CN: delta_rmse=-0.0013, delta_corr=-0.0679, delta_sign_accuracy=-0.0417.
- separate_60 | sleeve=EQ_EM: delta_rmse=-0.0021, delta_corr=-0.0273, delta_sign_accuracy=0.0417.
- separate_60 | sleeve=EQ_EZ: delta_rmse=-0.0000, delta_corr=-0.0007, delta_sign_accuracy=0.0000.
- separate_60 | sleeve=EQ_JP: delta_rmse=-0.0028, delta_corr=-0.0535, delta_sign_accuracy=0.0000.
- separate_60 | sleeve=EQ_US: delta_rmse=0.0011, delta_corr=-0.0167, delta_sign_accuracy=0.0000.

## Practical Interpretation
- Separate-horizon models remain the primary benchmark; shared 60m+120m is still an ablation, not the default winner.
- Linear models remain the strongest overall family on v3. Tree models are competitive on the fixed split, especially at 120m, but the rolling anchor is still linear. Neural baselines remain weaker.
- Interactions help more clearly at 60m than at 120m. At 120m, richer feature sets can win validation while still showing decay.
- China-related enrichments are mixed: some 120m setups benefit, but EQ_CN itself is still a hard sleeve and not a uniformly easy prediction problem.
