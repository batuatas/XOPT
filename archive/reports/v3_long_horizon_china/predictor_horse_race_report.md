# XOPTPOE v3 Predictor Horse Race Report

## Scope
- Separate-horizon prediction is the default evaluation mode; shared 60m/120m conditioning is included only as a benchmark.
- Model families compared: naive mean, ridge, elastic net, random forest, gradient boosting, small MLP, paper MLP, plus existing PTO_NN and E2E_NN shared benchmarks.
- Linear models covered the full first-pass feature-set range. Tree and neural baselines were kept to the leaner feature sets to avoid turning this into a large benchmark zoo.
- PCR/PLS, lasso, XGBoost, and LightGBM were intentionally skipped to keep dependencies and search scope tight.

## Best Predictors
- Best 60m predictor by validation RMSE: `elastic_net__full_firstpass__separate_60` with validation rmse=0.029565, test rmse=0.037379, test corr=0.4957, test sign_accuracy=0.8287.
- Best 120m predictor by validation RMSE: `ridge__full_firstpass__separate_120` with validation rmse=0.020693, test rmse=0.038987, test corr=0.4819, test sign_accuracy=0.7639.
- Best shared benchmark: `elastic_net__core_plus_interactions__shared_60_120` with validation rmse=0.026255, test rmse=0.033251, test corr=0.5604.

## Model-Family Readout
- ridge: best_validation_rmse=0.020693, best_test_rmse=0.032536, best_test_corr=0.6194.
- elastic_net: best_validation_rmse=0.021866, best_test_rmse=0.027662, best_test_corr=0.6902.
- gradient_boosting: best_validation_rmse=0.026292, best_test_rmse=0.032256, best_test_corr=0.6530.
- random_forest: best_validation_rmse=0.026939, best_test_rmse=0.032413, best_test_corr=0.6758.
- small_mlp: best_validation_rmse=0.166021, best_test_rmse=0.229591, best_test_corr=0.3787.
- paper_mlp: best_validation_rmse=0.299306, best_test_rmse=0.360222, best_test_corr=0.0500.

## Feature-Set Readout
- separate_120, full_firstpass: best_validation_model=ridge, best_validation_rmse=0.020693, best_test_rmse=0.035953, best_test_corr=0.5278.
- separate_120, core_plus_interactions: best_validation_model=elastic_net, best_validation_rmse=0.021866, best_test_rmse=0.032413, best_test_corr=0.6758.
- separate_120, core_baseline: best_validation_model=ridge, best_validation_rmse=0.024392, best_test_rmse=0.032871, best_test_corr=0.5861.
- separate_120, core_plus_enrichment: best_validation_model=ridge, best_validation_rmse=0.025777, best_test_rmse=0.032710, best_test_corr=0.5915.
- separate_60, full_firstpass: best_validation_model=elastic_net, best_validation_rmse=0.029565, best_test_rmse=0.033469, best_test_corr=0.5651.
- separate_60, core_plus_interactions: best_validation_model=elastic_net, best_validation_rmse=0.030013, best_test_rmse=0.027662, best_test_corr=0.6902.
- separate_60, core_baseline: best_validation_model=random_forest, best_validation_rmse=0.032678, best_test_rmse=0.032256, best_test_corr=0.5912.
- separate_60, core_plus_enrichment: best_validation_model=random_forest, best_validation_rmse=0.037517, best_test_rmse=0.031728, best_test_corr=0.5897.
- shared_60_120, core_plus_interactions: best_validation_model=elastic_net, best_validation_rmse=0.026255, best_test_rmse=0.033251, best_test_corr=0.5604.

## Hardest Sleeves
- 60m winner `elastic_net__full_firstpass__separate_60` hardest test sleeves:
  - ALT_GLD: rmse=0.061672, corr=0.7932, sign_accuracy=0.5000.
  - RE_US: rmse=0.050771, corr=-0.4401, sign_accuracy=0.9583.
  - EQ_EM: rmse=0.042511, corr=0.4480, sign_accuracy=0.6250.
- 120m winner `ridge__full_firstpass__separate_120` hardest test sleeves:
  - ALT_GLD: rmse=0.086948, corr=-0.4869, sign_accuracy=0.4167.
  - EQ_EZ: rmse=0.044540, corr=-0.4998, sign_accuracy=0.7083.
  - EQ_EM: rmse=0.031160, corr=-0.1229, sign_accuracy=0.8333.

## Interpretation
- The core result remains the same as the earlier ablation: separate-horizon linear models are materially stronger than the current neural baselines.
- Shared 60m/120m conditioning remains usable as a benchmark, but it is not the predictive winner once separate-horizon models are allowed.
