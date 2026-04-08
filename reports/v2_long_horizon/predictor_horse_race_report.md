# XOPTPOE v2 Predictor Horse Race Report

## Scope
- Separate-horizon prediction is the default evaluation mode; shared 60m/120m conditioning is included only as a benchmark.
- Model families compared: naive mean, ridge, elastic net, random forest, gradient boosting, small MLP, paper MLP, plus existing PTO_NN and E2E_NN shared benchmarks.
- Linear models covered the full first-pass feature-set range. Tree and neural baselines were kept to the leaner feature sets to avoid turning this into a large benchmark zoo.
- PCR/PLS, lasso, XGBoost, and LightGBM were intentionally skipped to keep dependencies and search scope tight.

## Best Predictors
- Best 60m predictor by validation RMSE: `elastic_net__core_plus_interactions__separate_60` with validation rmse=0.030395, test rmse=0.026182, test corr=0.7048, test sign_accuracy=0.9219.
- Best 120m predictor by validation RMSE: `ridge__full_firstpass__separate_120` with validation rmse=0.016406, test rmse=0.042685, test corr=0.4090, test sign_accuracy=0.7500.
- Best shared benchmark: `elastic_net__core_plus_interactions__shared_60_120` with validation rmse=0.024601, test rmse=0.034731, test corr=0.5049.

## Model-Family Readout
- ridge: best_validation_rmse=0.016406, best_test_rmse=0.031494, best_test_corr=0.5972.
- elastic_net: best_validation_rmse=0.017235, best_test_rmse=0.026182, best_test_corr=0.7048.
- gradient_boosting: best_validation_rmse=0.023534, best_test_rmse=0.030608, best_test_corr=0.6116.
- random_forest: best_validation_rmse=0.025563, best_test_rmse=0.031118, best_test_corr=0.6191.
- paper_mlp: best_validation_rmse=0.082061, best_test_rmse=0.087995, best_test_corr=0.2173.
- small_mlp: best_validation_rmse=0.118092, best_test_rmse=0.216543, best_test_corr=0.4144.

## Feature-Set Readout
- separate_120, full_firstpass: best_validation_model=ridge, best_validation_rmse=0.016406, best_test_rmse=0.039607, best_test_corr=0.4287.
- separate_120, core_plus_interactions: best_validation_model=ridge, best_validation_rmse=0.018640, best_test_rmse=0.031118, best_test_corr=0.6191.
- separate_120, core_baseline: best_validation_model=elastic_net, best_validation_rmse=0.023945, best_test_rmse=0.031597, best_test_corr=0.5772.
- separate_120, core_plus_enrichment: best_validation_model=gradient_boosting, best_validation_rmse=0.024750, best_test_rmse=0.031838, best_test_corr=0.6116.
- separate_60, core_plus_interactions: best_validation_model=elastic_net, best_validation_rmse=0.030395, best_test_rmse=0.026182, best_test_corr=0.7048.
- separate_60, core_baseline: best_validation_model=random_forest, best_validation_rmse=0.031551, best_test_rmse=0.030030, best_test_corr=0.6438.
- separate_60, full_firstpass: best_validation_model=elastic_net, best_validation_rmse=0.032091, best_test_rmse=0.034032, best_test_corr=0.5663.
- separate_60, core_plus_enrichment: best_validation_model=random_forest, best_validation_rmse=0.036418, best_test_rmse=0.032927, best_test_corr=0.5051.
- shared_60_120, core_plus_interactions: best_validation_model=elastic_net, best_validation_rmse=0.024601, best_test_rmse=0.034731, best_test_corr=0.5049.

## Hardest Sleeves
- 60m winner `elastic_net__core_plus_interactions__separate_60` hardest test sleeves:
  - ALT_GLD: rmse=0.042548, corr=0.4912, sign_accuracy=0.8333.
  - RE_US: rmse=0.031372, corr=-0.1763, sign_accuracy=0.9583.
  - EQ_EM: rmse=0.030722, corr=0.8952, sign_accuracy=0.8333.
- 120m winner `ridge__full_firstpass__separate_120` hardest test sleeves:
  - ALT_GLD: rmse=0.091428, corr=-0.5879, sign_accuracy=0.3750.
  - EQ_EZ: rmse=0.050210, corr=-0.5884, sign_accuracy=0.5417.
  - EQ_EM: rmse=0.036342, corr=-0.2307, sign_accuracy=0.6250.

## Interpretation
- The core result remains the same as the earlier ablation: separate-horizon linear models are materially stronger than the current neural baselines.
- Shared 60m/120m conditioning remains usable as a benchmark, but it is not the predictive winner once separate-horizon models are allowed.
