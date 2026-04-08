# XOPTPOE v3 SAA Diagnostics Report

## Scope And Caveat
- These outputs are long-horizon SAA decision-period diagnostics built on the active v3 China-sleeve dataset and the current horse-race / PTO / E2E outputs.
- They are not presented as a fully tradable non-overlapping monthly wealth backtest, because the labels are overlapping long-horizon annualized outcomes.
- The purpose is interpretability: what the models predict, which sleeves they favor, how concentrated the allocations are, and where the apparent gains come from.

## Most Informative Diagnostics For Discussion
- The prediction scatter and heatmaps show whether the strongest 60m and 120m models are actually separating sleeves in economically sensible ways.
- The rank-IC-over-time plot is the cleanest way to compare signal quality with the shared neural baselines.
- The weight stack/heatmap and turnover-concentration panel show whether portfolio gains come from broad signal use or a small number of concentrated bets.
- The active-return and sleeve-contribution diagnostics show whether the apparent portfolio edge comes from many months/sleeves or only a narrow subset.

## Prediction Readout
- Best 60m predictor: `elastic_net__full_firstpass__separate_60`. Test rmse=0.037379, corr=0.4957, sign_accuracy=0.8287, mean rank IC=0.5132. Top predicted sleeve most often: EQ_US (0.6667).
- Best 120m predictor: `ridge__full_firstpass__separate_120`. Test rmse=0.038987, corr=0.4819, sign_accuracy=0.7639, mean rank IC=0.3000. Top predicted sleeve most often: EQ_US (1.0000).
- Best shared benchmark: `elastic_net__core_plus_interactions__shared_60_120`. Test rmse=0.030019, corr=0.6233, mean rank IC=0.5181.
- PTO_NN shared signal: avg test rmse=0.065138, avg corr=-0.0151, avg rank IC=-0.0983, prediction dispersion=0.0053.
- E2E_NN shared signal: avg test rmse=0.134201, avg corr=0.1017, avg rank IC=-0.0285, prediction dispersion=0.0178.

## Portfolio Behavior Readout
- equal_weight: avg_return=0.0406, vol=0.0150, sharpe=2.7019, avg_turnover=0.0000, avg_max_weight=0.1111, avg_effective_n=9.0000.
- best_60_predictor: avg_return=0.0513, vol=0.0088, sharpe=5.8271, avg_turnover=0.0553, avg_max_weight=0.2957, avg_effective_n=4.7671.
- best_120_predictor: avg_return=0.0767, vol=0.0124, sharpe=6.1859, avg_turnover=0.0840, avg_max_weight=0.6844, avg_effective_n=1.8878.
- combined_60_120_predictor: avg_return=0.0694, vol=0.0121, sharpe=5.7325, avg_turnover=0.1040, avg_max_weight=0.6028, avg_effective_n=2.1885.
- best_shared_predictor: avg_return=0.0695, vol=0.0108, sharpe=6.4291, avg_turnover=0.0750, avg_max_weight=0.6084, avg_effective_n=2.1300.
- pto_nn_signal: avg_return=0.0386, vol=0.0106, sharpe=3.6312, avg_turnover=0.0330, avg_max_weight=0.2663, avg_effective_n=5.8597.
- e2e_nn_signal: avg_return=0.0374, vol=0.0077, sharpe=4.8521, avg_turnover=0.1116, avg_max_weight=0.2995, avg_effective_n=4.7996.

## Signal Versus Risk-Control Interpretation
- The strongest separate-horizon predictor remains the 60m elastic-net model. It has the best test rmse/corr mix and its portfolio diagnostics show strong active return with moderate turnover.
- The current 120m winner by validation is not the cleanest out-of-sample predictor, so its portfolio should be interpreted more cautiously than the 60m winner.
- E2E still looks much more like risk control than superior prediction. Its test prediction metrics remain weak, its prediction dispersion is low, and its portfolio volatility is suppressed materially more than equal weight.
- The combined 60m/120m predictor helps on portfolio behavior relative to best shared, PTO_NN, and equal weight in this decision-period diagnostic, which is consistent with horizon diversification helping the signal layer.

## Sleeve And Month Drivers
- Largest absolute active-contribution sleeves for the combined 60m/120m signal on the test split:
  - EQ_US: total_active_contribution_vs_equal_weight=1.2317; avg_weight=0.6028; top_weight_frequency=1.0000.
  - ALT_GLD: total_active_contribution_vs_equal_weight=-0.1660; avg_weight=0.0000; top_weight_frequency=0.0000.
  - EQ_JP: total_active_contribution_vs_equal_weight=-0.1139; avg_weight=0.0000; top_weight_frequency=0.0000.
- Best and worst test months by strategy are included in the diagnostic tables; these are decision-period label outcomes, not tradable single-month realized PnL statements.

## Concentration Readout
- best_60_predictor: top5_positive_active_month_share=0.3623, top2_sleeve_active_share_abs=0.6445, most frequent top-weight sleeve=EQ_US (1.0000).
- combined_60_120_predictor: top5_positive_active_month_share=0.3197, top2_sleeve_active_share_abs=0.7622, most frequent top-weight sleeve=EQ_US (1.0000).
- best_120_predictor: top5_positive_active_month_share=0.2889, top2_sleeve_active_share_abs=0.7809, most frequent top-weight sleeve=EQ_US (1.0000).
- e2e_nn_signal: top5_positive_active_month_share=0.7731, top2_sleeve_active_share_abs=0.3455, most frequent top-weight sleeve=FI_UST (0.6250).

## Bottom Line
- For professor/Akif discussion, the clearest story is: prediction quality is strongest in the separate 60m elastic-net setup; the combined 60m/120m signal gives the most compelling portfolio behavior among the common-allocator diagnostics; and E2E still looks primarily volatility-suppressing rather than signal-superior.
- The system is not yet ready to jump directly into a POE/scenario-generation stage. The predictive layer is improving, but the 120m model choice and the translation from prediction quality into robust portfolio behavior still need more discipline before moving downstream.
