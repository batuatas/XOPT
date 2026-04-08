# XOPTPOE v2 SAA Diagnostics Report

## Scope And Caveat
- These outputs are long-horizon SAA decision-period diagnostics built on the frozen v2 dataset and the existing horse-race / PTO / E2E outputs.
- They are not presented as a fully tradable non-overlapping monthly wealth backtest, because the labels are overlapping long-horizon annualized outcomes.
- The purpose is interpretability: what the models predict, which sleeves they favor, how concentrated the allocations are, and where the apparent gains come from.

## Most Informative Diagnostics For Discussion
- The prediction scatter and heatmaps show whether the strongest 60m and 120m models are actually separating sleeves in economically sensible ways.
- The rank-IC-over-time plot is the cleanest way to compare signal quality with the shared neural baselines.
- The weight stack/heatmap and turnover-concentration panel show whether portfolio gains come from broad signal use or a small number of concentrated bets.
- The active-return and sleeve-contribution diagnostics show whether the apparent portfolio edge comes from many months/sleeves or only a narrow subset.

## Prediction Readout
- Best 60m predictor: `elastic_net__core_plus_interactions__separate_60`. Test rmse=0.026182, corr=0.7048, sign_accuracy=0.9219, mean rank IC=0.5714. Top predicted sleeve most often: EQ_US (1.0000).
- Best 120m predictor: `ridge__full_firstpass__separate_120`. Test rmse=0.042685, corr=0.4090, sign_accuracy=0.7500, mean rank IC=0.1786. Top predicted sleeve most often: EQ_US (1.0000).
- Best shared benchmark: `elastic_net__core_plus_interactions__shared_60_120`. Test rmse=0.030285, corr=0.6174, mean rank IC=0.5367.
- PTO_NN shared signal: avg test rmse=0.126859, avg corr=-0.0119, avg rank IC=-0.0580, prediction dispersion=0.0268.
- E2E_NN shared signal: avg test rmse=0.112704, avg corr=-0.1539, avg rank IC=-0.2693, prediction dispersion=0.0271.

## Portfolio Behavior Readout
- equal_weight: avg_return=0.0444, vol=0.0138, sharpe=3.2092, avg_turnover=0.0000, avg_max_weight=0.1250, avg_effective_n=8.0000.
- best_60_predictor: avg_return=0.0618, vol=0.0084, sharpe=7.3482, avg_turnover=0.0493, avg_max_weight=0.5519, avg_effective_n=2.1020.
- best_120_predictor: avg_return=0.0758, vol=0.0122, sharpe=6.2171, avg_turnover=0.0836, avg_max_weight=0.6753, avg_effective_n=1.9192.
- combined_60_120_predictor: avg_return=0.0687, vol=0.0101, sharpe=6.7970, avg_turnover=0.0647, avg_max_weight=0.6115, avg_effective_n=2.0457.
- best_shared_predictor: avg_return=0.0679, vol=0.0106, sharpe=6.3899, avg_turnover=0.0779, avg_max_weight=0.5904, avg_effective_n=2.1938.
- pto_nn_signal: avg_return=0.0437, vol=0.0135, sharpe=3.2350, avg_turnover=0.0253, avg_max_weight=0.1475, avg_effective_n=7.9050.
- e2e_nn_signal: avg_return=0.0334, vol=0.0052, sharpe=6.4479, avg_turnover=0.0984, avg_max_weight=0.3629, avg_effective_n=3.6598.

## Signal Versus Risk-Control Interpretation
- The strongest separate-horizon predictor remains the 60m elastic-net model. It has the best test rmse/corr mix and its portfolio diagnostics show strong active return with moderate turnover.
- The current 120m winner by validation is not the cleanest out-of-sample predictor, so its portfolio should be interpreted more cautiously than the 60m winner.
- E2E still looks much more like risk control than superior prediction. Its test prediction metrics remain weak, its prediction dispersion is low, and its portfolio volatility is suppressed materially more than equal weight.
- The combined 60m/120m predictor helps on portfolio behavior relative to best shared, PTO_NN, and equal weight in this decision-period diagnostic, which is consistent with horizon diversification helping the signal layer.

## Sleeve And Month Drivers
- Largest absolute active-contribution sleeves for the combined 60m/120m signal on the test split:
  - EQ_US: total_active_contribution_vs_equal_weight=1.2148; avg_weight=0.6115; top_weight_frequency=1.0000.
  - ALT_GLD: total_active_contribution_vs_equal_weight=-0.1860; avg_weight=0.0003; top_weight_frequency=0.0000.
  - EQ_JP: total_active_contribution_vs_equal_weight=-0.1282; avg_weight=0.0000; top_weight_frequency=0.0000.
- Best and worst test months by strategy are included in the diagnostic tables; these are decision-period label outcomes, not tradable single-month realized PnL statements.

## Concentration Readout
- best_60_predictor: top5_positive_active_month_share=0.3233, top2_sleeve_active_share_abs=0.6840, most frequent top-weight sleeve=EQ_US (0.8750).
- combined_60_120_predictor: top5_positive_active_month_share=0.3056, top2_sleeve_active_share_abs=0.7277, most frequent top-weight sleeve=EQ_US (1.0000).
- best_120_predictor: top5_positive_active_month_share=0.2965, top2_sleeve_active_share_abs=0.7708, most frequent top-weight sleeve=EQ_US (1.0000).
- e2e_nn_signal: top5_positive_active_month_share=1.0000, top2_sleeve_active_share_abs=0.4109, most frequent top-weight sleeve=FI_UST (0.7083).

## Bottom Line
- For professor/Akif discussion, the clearest story is: prediction quality is strongest in the separate 60m elastic-net setup; the combined 60m/120m signal gives the most compelling portfolio behavior among the common-allocator diagnostics; and E2E still looks primarily volatility-suppressing rather than signal-superior.
- The system is not yet ready to jump directly into a POE/scenario-generation stage. The predictive layer is improving, but the 120m model choice and the translation from prediction quality into robust portfolio behavior still need more discipline before moving downstream.
