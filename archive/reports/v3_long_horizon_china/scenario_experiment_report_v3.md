# v3 Scenario Experiment Report

## Scope
- Active branch only: v3_long_horizon_china.
- These are model-based, plausibility-regularized state perturbation diagnostics for long-horizon SAA benchmarks.
- They are not a causal macro truth exercise and not a live-investment backtest.

## Experiment Design
- Anchors: 2021-12-31, 2022-12-31, 2023-12-31, 2024-12-31.
- Manipulated state: the locked 17-variable canonical macro block from the scaffold.
- Benchmarks actually evaluated: best_60_predictor, best_120_predictor, elastic_net__full_firstpass__separate_60, ridge__full_firstpass__separate_120.
- Shared predictor and combined_std_120tilt_top_k_capped are retained as comparators/reference objects, but the first-pass probes stay focused on the supervised 60m/120m carry-forward stack.
- E2E remains out of the real run set because the active v3 artifacts still do not include a clean scenario-ready persisted model object.

## Representative Cases
| case_id                                          | anchor_date         | probe_id                             | candidate_name                                                                  |   baseline_score |   scenario_score |   plausibility_metric | short_case_label                      | short_case_interpretation                        |
|:-------------------------------------------------|:--------------------|:-------------------------------------|:--------------------------------------------------------------------------------|-----------------:|-----------------:|----------------------:|:--------------------------------------|:-------------------------------------------------|
| 2021-12-31__probe_120_deconcentration            | 2021-12-31 00:00:00 | probe_120_deconcentration            | best_120_predictor                                                              |       0.410678   |       0.409568   |               7.71577 | 2021 | 120 deconcentration            | higher usd_broad, higher infl_EA, higher ig_oas  |
| 2021-12-31__probe_120_target_down                | 2021-12-31 00:00:00 | probe_120_target_down                | best_120_predictor                                                              |       0.0727549  |       0.0736252  |               7.71655 | 2021 | 120 target down                | higher usd_broad, higher infl_EA, higher ig_oas  |
| 2021-12-31__probe_120_target_up                  | 2021-12-31 00:00:00 | probe_120_target_up                  | best_120_predictor                                                              |       0.0727549  |       0.0736253  |               7.71649 | 2021 | 120 target up                  | higher usd_broad, higher infl_EA, higher ig_oas  |
| 2021-12-31__probe_60_120_allocation_disagreement | 2021-12-31 00:00:00 | probe_60_120_allocation_disagreement | best_60_predictor_vs_best_120_predictor                                         |       0.790604   |       0.782309   |               7.95728 | 2021 | 60 120 allocation disagreement | higher usd_broad, higher infl_EA, higher ig_oas  |
| 2021-12-31__probe_60_120_prediction_disagreement | 2021-12-31 00:00:00 | probe_60_120_prediction_disagreement | elastic_net__full_firstpass__separate_60_vs_ridge__full_firstpass__separate_120 |       0.0378936  |       0.0399919  |               7.9578  | 2021 | 60 120 prediction disagreement | higher usd_broad, higher infl_EA, higher ig_oas  |
| 2021-12-31__probe_60_china_role                  | 2021-12-31 00:00:00 | probe_60_china_role                  | best_60_predictor                                                               |       0.0148392  |       0.012201   |               7.95683 | 2021 | 60 china role                  | higher usd_broad, higher infl_EA, higher ig_oas  |
| 2021-12-31__probe_60_target_down                 | 2021-12-31 00:00:00 | probe_60_target_down                 | best_60_predictor                                                               |       0.120272   |       0.123639   |               7.71652 | 2021 | 60 target down                 | higher usd_broad, higher infl_EA, higher ig_oas  |
| 2021-12-31__probe_60_target_up                   | 2021-12-31 00:00:00 | probe_60_target_up                   | best_60_predictor                                                               |       0.120272   |       0.123639   |               7.71652 | 2021 | 60 target up                   | higher usd_broad, higher infl_EA, higher ig_oas  |
| 2022-12-31__probe_120_deconcentration            | 2022-12-31 00:00:00 | probe_120_deconcentration            | best_120_predictor                                                              |       0.58528    |       0.585374   |               7.31256 | 2022 | 120 deconcentration            | higher unemp_EA, higher infl_EA, higher vix      |
| 2022-12-31__probe_120_target_down                | 2022-12-31 00:00:00 | probe_120_target_down                | best_120_predictor                                                              |       0.00568299 |       0.00549463 |               7.31385 | 2022 | 120 target down                | higher unemp_EA, higher infl_EA, higher vix      |
| 2022-12-31__probe_120_target_up                  | 2022-12-31 00:00:00 | probe_120_target_up                  | best_120_predictor                                                              |       0.00568299 |       0.00568805 |               5.44185 | 2022 | 120 target up                  | higher usd_broad, higher unemp_EA, higher ig_oas |
| 2022-12-31__probe_60_120_allocation_disagreement | 2022-12-31 00:00:00 | probe_60_120_allocation_disagreement | best_60_predictor_vs_best_120_predictor                                         |       1.04014    |       1.03885    |               7.94648 | 2022 | 60 120 allocation disagreement | higher unemp_EA, higher infl_EA, higher vix      |

## Variables That Move Most Consistently
| variable_name   |   shift_magnitude_std_units |
|:----------------|----------------------------:|
| short_rate_JP   |                   0.222948  |
| long_rate_JP    |                   0.13395   |
| ig_oas          |                   0.08252   |
| unemp_JP        |                   0.0781168 |
| unemp_EA        |                   0.0626023 |
| infl_EA         |                   0.057373  |
| long_rate_US    |                   0.0553242 |
| us_real10y      |                   0.0426492 |
| infl_JP         |                   0.0419285 |
| unemp_US        |                   0.0406255 |

## Interpretation
- Target-return probes show which macro shifts raise or lower the model-implied portfolio return while staying plausible under the prior.
- Benchmark-difference probes separate predictor disagreement from allocation disagreement.
- Deconcentration probes show whether the 120m ceiling can be made less EQ_US-heavy without collapsing predicted return.
- China-role probes ask when EQ_CN becomes more meaningful inside the robust 60m benchmark, not whether China should exist as a sleeve.

## Compact Findings
- Most favorable 60m return-up case: 2024-12-31 with scenario_response_best=0.1628 vs baseline=0.1618.
- Most favorable 120m return-up case: 2021-12-31 with scenario_response_best=0.0736 vs baseline=0.0728.
- Largest 120m deconcentration move: 2024-12-31 with hhi_change=-0.0013 and predicted_return_change=0.0016.
- Largest EQ_CN role increase in the robust benchmark: 2022-12-31 with weight 0.0220 -> 0.0223.
