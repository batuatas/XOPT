# XOPTPOE v3 Prediction Ablation Report

## Scope
- Prediction-only study. Data, feature engineering, and portfolio layers stayed unchanged.
- Compared shared vs separate horizons, four prepared feature sets, smaller-vs-paper MLPs, and ridge / elastic net benchmarks on the frozen train/validation/test splits.

## Best Validation Configuration
- ridge__full_firstpass__separate_120: validation_rmse=0.0207, test_rmse=0.0390, test_corr=0.4819, test_sign_accuracy=0.7639.

## Horizon Pooling
- Best shared model: elastic_net__core_plus_interactions__shared_60_120 with validation_rmse=0.0263, test_rmse=0.0333, test_corr=0.5604.
- Best separate 60m model: elastic_net__full_firstpass__separate_60 with validation_rmse=0.0296, test_rmse=0.0374, test_corr=0.4957.
- Best separate 120m model: ridge__full_firstpass__separate_120 with validation_rmse=0.0207, test_rmse=0.0390, test_corr=0.4819.

## Horizon Predictability
- 60m best test result: experiment=elastic_net__core_plus_interactions__separate_60, rmse=0.0277, corr=0.6902, sign_accuracy=0.9167, rank_ic_spearman=0.5458.
- 120m best test result: experiment=elastic_net__core_plus_enrichment__separate_120, rmse=0.0327, corr=0.5915, sign_accuracy=0.8333, rank_ic_spearman=0.2986.

## Feature-Set Comparison
- full_firstpass: best_validation_rmse=0.0207, best_test_rmse=0.0335, best_test_corr=0.5651.
- core_plus_interactions: best_validation_rmse=0.0219, best_test_rmse=0.0277, best_test_corr=0.6902.
- core_baseline: best_validation_rmse=0.0244, best_test_rmse=0.0325, best_test_corr=0.6240.
- core_plus_enrichment: best_validation_rmse=0.0258, best_test_rmse=0.0317, best_test_corr=0.5935.

## Model-Size Comparison
- ridge: best_validation_rmse=0.0207, best_test_rmse=0.0325, best_test_corr=0.6194.
- elastic_net: best_validation_rmse=0.0219, best_test_rmse=0.0277, best_test_corr=0.6902.
- paper_mlp: best_validation_rmse=0.0683, best_test_rmse=0.0638, best_test_corr=0.0777.
- small_mlp: best_validation_rmse=0.0686, best_test_rmse=0.1474, best_test_corr=0.1628.

## Feature Block Drop Diagnostics
- Blocks whose removal worsened validation RMSE are more likely helping the default ridge baseline.
- helpful: separate_120, block=metadata_dummy, delta_validation_rmse=0.0160, delta_test_rmse=0.0066, delta_test_corr=-0.4346.
- helpful: shared_60_120, block=metadata_dummy, delta_validation_rmse=0.0067, delta_test_rmse=0.0054, delta_test_corr=-0.3178.
- helpful: separate_120, block=china_macro, delta_validation_rmse=0.0019, delta_test_rmse=0.0012, delta_test_corr=-0.0041.
- helpful: separate_120, block=baseline_macro_canonical, delta_validation_rmse=0.0010, delta_test_rmse=0.0014, delta_test_corr=-0.0021.
- helpful: separate_60, block=em_global_valuation, delta_validation_rmse=0.0004, delta_test_rmse=-0.0001, delta_test_corr=-0.0012.
- helpful: separate_120, block=eu_ig_market, delta_validation_rmse=0.0003, delta_test_rmse=0.0001, delta_test_corr=-0.0018.
- Blocks whose removal improved validation RMSE are more likely adding noise.
- noisy: separate_60, block=baseline_technical, delta_validation_rmse=-0.0072, delta_test_rmse=0.0034, delta_test_corr=-0.0555.
- noisy: shared_60_120, block=baseline_technical, delta_validation_rmse=-0.0046, delta_test_rmse=0.0032, delta_test_corr=-0.0498.
- noisy: separate_60, block=metadata_dummy, delta_validation_rmse=-0.0035, delta_test_rmse=0.0029, delta_test_corr=-0.2656.
- noisy: separate_120, block=china_valuation, delta_validation_rmse=-0.0004, delta_test_rmse=-0.0002, delta_test_corr=-0.0063.
- noisy: separate_120, block=baseline_technical, delta_validation_rmse=-0.0003, delta_test_rmse=0.0019, delta_test_corr=-0.0450.
- noisy: separate_60, block=baseline_global_macro, delta_validation_rmse=-0.0002, delta_test_rmse=0.0001, delta_test_corr=-0.0122.

## Sleeve Difficulty
- Reported for the carry-forward baseline `ridge__full_firstpass__separate_120` only.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=ALT_GLD, rmse=0.0869, corr=-0.4869, sign_accuracy=0.4167.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=EQ_EZ, rmse=0.0445, corr=-0.4998, sign_accuracy=0.7083.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=EQ_EM, rmse=0.0312, corr=-0.1229, sign_accuracy=0.8333.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=EQ_CN, rmse=0.0276, corr=0.1262, sign_accuracy=0.5417.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=RE_US, rmse=0.0255, corr=0.6888, sign_accuracy=1.0000.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=FI_UST, rmse=0.0234, corr=0.8119, sign_accuracy=0.3750.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=FI_IG, rmse=0.0209, corr=0.4644, sign_accuracy=1.0000.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=EQ_US, rmse=0.0206, corr=-0.1323, sign_accuracy=1.0000.
