# XOPTPOE v3 Pre-Scenario Lock-In Report

## Scope
- Active branch only: v3_long_horizon_china.
- This is a synthesis and lock-in pass. No new data design, target construction, or benchmark sweep was introduced.
- These remain long-horizon SAA decision diagnostics, not a clean overlapping-month wealth backtest.

## Final Benchmark Synthesis
- strongest raw prediction benchmark: `ridge__full_firstpass__separate_120`
- strongest robust prediction benchmark: `ridge__full_firstpass__separate_120`
- strongest raw portfolio benchmark: `best_120_predictor`
- strongest robust portfolio benchmark: `best_60_predictor`
- strongest shared benchmark: `ridge__full_firstpass__shared_60_120` -> comparator only
- PTO status: comparator only
- E2E status: comparator only

## Raw Versus Robust
- Prediction side: raw and robust winner are effectively the same overall object at 120m.
- Portfolio side: raw and robust winners differ. The raw winner is too concentrated; the robust winner is the carry-forward benchmark.

## Final Carry-Forward Decision
- best_60_predictor: active robust benchmark | highest-Sharpe supervised candidate passing the explicit concentration screen on test
- best_120_predictor: reference ceiling | highest raw supervised portfolio result, but too concentrated to carry forward as the primary benchmark
- ridge__full_firstpass__shared_60_120 / best_shared_predictor: comparator only | shared remains useful as ablation/reference, not as the default carry-forward winner
- PTO: comparator only | paper-faithful neural reference; weaker than supervised benchmarks on prediction and raw portfolio return
- E2E: comparator only | still useful because it shows risk-control-driven behavior, but it does not beat the supervised benchmark stack
- combined_std_120tilt_top_k_capped: comparator only | best compact concentration-control tradeoff; useful to interrogate benchmark robustness but not the locked primary benchmark

## China Interpretation
- did_adding_eq_cn_matter_structurally: yes | v3 is a 9-sleeve active branch with EQ_CN fully included in prediction, portfolio, and diagnostics outputs.
- did_eq_cn_matter_predictively: mixed_modest_yes | EQ_CN selected-winner metrics: 60m rmse=0.0393, corr=0.3284; 120m rmse=0.0276, corr=0.1262. China-feature-drop hurts 120m EQ_CN/EQ_EM/EQ_EZ/EQ_JP/EQ_US more consistently than 60m.
- do_china_features_help_only_eq_cn: no | At 120m, removing China features worsens RMSE for EQ_CN, EQ_EM, EQ_EZ, EQ_JP, and EQ_US; at 60m, effects are mixed and less robust.
- does_eq_cn_matter_in_raw_winner_allocations: mostly_no | best_120_predictor test EQ_CN avg_weight=0.0000, nonzero_share=0.1250, active_contribution_vs_equal=-0.0279.
- does_eq_cn_matter_in_robust_winner_allocations: small_but_nonzero | best_60_predictor test EQ_CN avg_weight=0.0205, nonzero_share=0.5000, active_contribution_vs_equal=-0.0138.
- does_china_help_diversification_in_carryforward_benchmarks: not_materially | In the raw and robust carryforward winners EQ_CN stays underweight; the stronger concentration-control comparator raises EQ_CN only to avg_weight=0.0058.
- is_eq_cn_central_or_marginal: marginal_but_active | EQ_CN is a valid active sleeve and a useful structural addition, but it is still mostly marginal in the carry-forward allocation winners.
- should_eq_cn_remain_active: yes | EQ_CN remains part of the active v3 system unless a concrete implementation bug is found; no such bug was found.

## Scenario Candidate Set
- ridge__full_firstpass__separate_120: type=predictor, horizon=120m, role=active benchmark, learn=what drives the cleanest long-horizon predictive signal, behavior=genuine predictive signal
- elastic_net__full_firstpass__separate_60: type=predictor, horizon=60m, role=active benchmark, learn=what drives the cleaner medium-long-horizon signal that survives concentration screening at portfolio level, behavior=genuine predictive signal
- best_60_predictor: type=portfolio, horizon=60m-led, role=active robust benchmark, learn=how predictive signal translates into a lower-concentration SAA allocation, behavior=genuine signal plus moderated concentration
- best_120_predictor: type=portfolio, horizon=120m-led, role=reference ceiling, learn=whether the strongest 120m signal is economically intuitive or mostly an EQ_US concentration channel, behavior=genuine signal mixed with concentrated allocation behavior
- combined_std_120tilt_top_k_capped: type=portfolio, horizon=both, role=comparator only, learn=how much concentration can be reduced before performance degrades materially, behavior=mixture of signal use and explicit breadth control
- e2e_nn_signal: type=neural comparator, horizon=both, role=comparator only, learn=whether apparent gains come from real signal or from defensive risk control, behavior=mostly risk-control-driven behavior

## Practical Decision
- Carry forward the robust 60m-led portfolio benchmark as the primary governance object.
- Keep the raw 120m-led portfolio benchmark as the ceiling/reference object to explain concentration.
- Carry the 120m prediction benchmark and the 60m prediction benchmark into the next stage because the next stage should explain both predictors and portfolios.
