# XOPTPOE v2 Prediction Ablation Report

## Scope
- Prediction-only study. Data, feature engineering, and portfolio layers stayed unchanged.
- Compared shared vs separate horizons, four prepared feature sets, smaller-vs-paper MLPs, and ridge / elastic net benchmarks on the frozen train/validation/test splits.

## Best Validation Configuration
- ridge__full_firstpass__separate_120: validation_rmse=0.0164, test_rmse=0.0427, test_corr=0.4090, test_sign_accuracy=0.7500.

## Horizon Pooling
- Best shared model: elastic_net__core_plus_interactions__shared_60_120 with validation_rmse=0.0246, test_rmse=0.0347, test_corr=0.5049.
- Best separate 60m model: elastic_net__core_plus_interactions__separate_60 with validation_rmse=0.0304, test_rmse=0.0262, test_corr=0.7048.
- Best separate 120m model: ridge__full_firstpass__separate_120 with validation_rmse=0.0164, test_rmse=0.0427, test_corr=0.4090.

## Horizon Predictability
- 60m best test result: experiment=elastic_net__core_plus_interactions__separate_60, rmse=0.0262, corr=0.7048, sign_accuracy=0.9219, rank_ic_spearman=0.5714.
- 120m best test result: experiment=elastic_net__core_baseline__separate_120, rmse=0.0316, corr=0.5772, sign_accuracy=0.8750, rank_ic_spearman=0.2103.

## Feature-Set Comparison
- full_firstpass: best_validation_rmse=0.0164, best_test_rmse=0.0340, best_test_corr=0.5663.
- core_plus_interactions: best_validation_rmse=0.0186, best_test_rmse=0.0262, best_test_corr=0.7048.
- core_baseline: best_validation_rmse=0.0239, best_test_rmse=0.0297, best_test_corr=0.6438.
- core_plus_enrichment: best_validation_rmse=0.0251, best_test_rmse=0.0312, best_test_corr=0.5908.

## Model-Size Comparison
- ridge: best_validation_rmse=0.0164, best_test_rmse=0.0315, best_test_corr=0.5972.
- elastic_net: best_validation_rmse=0.0172, best_test_rmse=0.0262, best_test_corr=0.7048.
- small_mlp: best_validation_rmse=0.0500, best_test_rmse=0.0635, best_test_corr=0.2858.
- paper_mlp: best_validation_rmse=0.0891, best_test_rmse=0.0849, best_test_corr=0.2990.

## Feature Block Drop Diagnostics
- Blocks whose removal worsened validation RMSE are more likely helping the default ridge baseline.
- helpful: separate_120, block=metadata_dummy, delta_validation_rmse=0.0148, delta_test_rmse=0.0022, delta_test_corr=-0.2823.
- helpful: shared_60_120, block=metadata_dummy, delta_validation_rmse=0.0067, delta_test_rmse=0.0023, delta_test_corr=-0.2626.
- helpful: separate_120, block=china_macro, delta_validation_rmse=0.0016, delta_test_rmse=0.0003, delta_test_corr=-0.0081.
- helpful: separate_120, block=baseline_macro_canonical, delta_validation_rmse=0.0008, delta_test_rmse=-0.0000, delta_test_corr=0.0067.
- helpful: separate_120, block=oecd_bts, delta_validation_rmse=0.0004, delta_test_rmse=-0.0006, delta_test_corr=0.0183.
- helpful: separate_120, block=japan_enrichment, delta_validation_rmse=0.0003, delta_test_rmse=0.0001, delta_test_corr=-0.0012.
- Blocks whose removal improved validation RMSE are more likely adding noise.
- noisy: separate_60, block=baseline_technical, delta_validation_rmse=-0.0089, delta_test_rmse=0.0003, delta_test_corr=-0.0187.
- noisy: shared_60_120, block=baseline_technical, delta_validation_rmse=-0.0058, delta_test_rmse=0.0021, delta_test_corr=-0.0118.
- noisy: separate_60, block=metadata_dummy, delta_validation_rmse=-0.0013, delta_test_rmse=0.0010, delta_test_corr=-0.2976.
- noisy: separate_120, block=baseline_technical, delta_validation_rmse=-0.0008, delta_test_rmse=0.0010, delta_test_corr=-0.0007.
- noisy: separate_120, block=china_valuation, delta_validation_rmse=-0.0003, delta_test_rmse=0.0008, delta_test_corr=-0.0155.
- noisy: separate_120, block=credit_stress, delta_validation_rmse=-0.0002, delta_test_rmse=0.0000, delta_test_corr=0.0008.

## Sleeve Difficulty
- Reported for the carry-forward baseline `ridge__full_firstpass__separate_120` only.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=ALT_GLD, rmse=0.0914, corr=-0.5879, sign_accuracy=0.3750.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=EQ_EZ, rmse=0.0502, corr=-0.5884, sign_accuracy=0.5417.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=EQ_EM, rmse=0.0363, corr=-0.2307, sign_accuracy=0.6250.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=EQ_US, rmse=0.0241, corr=-0.2044, sign_accuracy=1.0000.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=FI_UST, rmse=0.0223, corr=0.8205, sign_accuracy=0.5417.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=RE_US, rmse=0.0213, corr=0.7076, sign_accuracy=1.0000.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=EQ_JP, rmse=0.0208, corr=0.0517, sign_accuracy=1.0000.
- hard sleeve: experiment=ridge__full_firstpass__separate_120, sleeve=FI_IG, rmse=0.0202, corr=0.3932, sign_accuracy=0.9167.
