# v3 Scenario Synthesis Report

## Scope
- Active branch only: v3_long_horizon_china.
- This is a synthesis pass built on the already-run first-pass scenario experiments.
- These outputs remain plausibility-regularized, anchor-local, model-implied state diagnostics for long-horizon SAA benchmarks.

## Benchmark Story
- `best_60_predictor` remains the main scenario object: it moves in an economically interpretable way, but the responses stay moderate.
- `best_120_predictor` remains the raw ceiling comparison object: it reacts more and stays more concentration-prone.
- The 120m ceiling can be nudged toward lower concentration, but the gain is incremental rather than transformative.

## Representative Case Selection Rule
- `robust_return_up`: highest 60m target-up return change after penalizing extra concentration.
- `raw_return_up`: positive 120m target-up case with the highest resulting scenario return level.
- `raw_deconcentration`: largest HHI reduction among 120m deconcentration cases without obvious return collapse.
- `disagreement_case`: anchor with the largest allocation disagreement between the 60m and 120m benchmark portfolios.
- `china_probe`: retained only as a diagnostic; no China case became material enough to enter the core casebook.

## Representative Cases
| case_id               | short_case_label            | anchor_date   | candidate_name     | probe_id                             |   baseline_predicted_return |   scenario_predicted_return |   baseline_max_weight |   scenario_max_weight | short_case_interpretation                                                                                                            |
|:----------------------|:----------------------------|:--------------|:-------------------|:-------------------------------------|----------------------------:|----------------------------:|----------------------:|----------------------:|:-------------------------------------------------------------------------------------------------------------------------------------|
| robust_return_up      | Robust benchmark upside     | 2021-12-31    | best_60_predictor  | probe_60_target_up                   |                   0.120272  |                  0.123639   |              0.291512 |              0.29203  | A moderate upside state for the 60m benchmark led by short_rate_JP, ig_oas, long_rate_JP, with only limited extra concentration.     |
| raw_return_up         | Raw ceiling upside          | 2021-12-31    | best_120_predictor | probe_120_target_up                  |                   0.0727549 |                  0.0736253  |              0.485081 |              0.481196 | The 120m ceiling improves most under a short_rate_JP, ig_oas, long_rate_JP state, but concentration stays high.                      |
| raw_deconcentration   | Raw ceiling deconcentration | 2024-12-31    | best_120_predictor | probe_120_deconcentration            |                   0.0318866 |                  0.0334649  |              0.419115 |              0.420104 | A plausible short_rate_JP, long_rate_JP, unemp_JP state broadens the 120m ceiling slightly without obvious return collapse.          |
| disagreement_case_60  | Robust side of disagreement | 2023-12-31    | best_60_predictor  | probe_60_120_allocation_disagreement |                   0.157046  |                  0.158389   |              0.393172 |              0.397247 | Under the disagreement state led by long_rate_JP, short_rate_JP, unemp_JP, the 60m benchmark stays the broader side of the contrast. |
| disagreement_case_120 | Raw side of disagreement    | 2023-12-31    | best_120_predictor | probe_60_120_allocation_disagreement |                  -0.0103349 |                 -0.00777187 |              0.65982  |              0.662668 | Under the same disagreement state, the 120m object remains the more concentrated side of the contrast.                               |

## What The Scenario Layer Learned
- Robust benchmark message: at 2021-12-31, the 60m benchmark improved from 0.1203 to 0.1236 while max weight only moved from 0.292 to 0.292.
- Raw ceiling message: at 2021-12-31, the 120m object improved from 0.0728 to 0.0736, but max weight stayed high at 0.481.
- Deconcentration message: at 2024-12-31, the 120m ceiling moved from effective N 3.07 to 3.08 with scenario return 0.0335.

## Variables That Repeat Across Cases
| variable_name   |   avg_abs_shift_std_units |   case_count_selected | comment                                                              |
|:----------------|--------------------------:|----------------------:|:---------------------------------------------------------------------|
| short_rate_JP   |                 0.227206  |                     6 | recurs in both robust and raw cases; also appears in deconcentration |
| long_rate_JP    |                 0.142654  |                     6 | recurs in both robust and raw cases; also appears in deconcentration |
| unemp_JP        |                 0.0889298 |                     3 | recurs in both robust and raw cases; also appears in deconcentration |
| ig_oas          |                 0.0833851 |                     6 | recurs in both robust and raw cases; also appears in deconcentration |
| infl_EA         |                 0.0687074 |                     1 | also appears in deconcentration                                      |
| unemp_EA        |                 0.0610779 |                     1 | localized first-pass move                                            |
| long_rate_US    |                 0.0585724 |                     4 | recurs in both robust and raw cases                                  |
| unemp_US        |                 0.0436017 |                     1 | localized first-pass move                                            |

## Raw Versus Robust Contrast
| anchor_date         |   baseline_60_return |   scenario_60_return |   baseline_120_return |   scenario_120_return |   baseline_60_max_weight |   scenario_60_max_weight |   baseline_120_max_weight |   scenario_120_max_weight |   whether_top_weight_sleeve_changed_60 |   whether_top_weight_sleeve_changed_120 |
|:--------------------|---------------------:|---------------------:|----------------------:|----------------------:|-------------------------:|-------------------------:|--------------------------:|--------------------------:|---------------------------------------:|----------------------------------------:|
| 2021-12-31 00:00:00 |             0.120272 |             0.124197 |            0.0727549  |            0.0738401  |                 0.291512 |                 0.292098 |                  0.485081 |                  0.480575 |                                      0 |                                       0 |
| 2022-12-31 00:00:00 |             0.13363  |             0.134052 |            0.00568299 |            0.00546792 |                 0.294181 |                 0.294865 |                  0.739162 |                  0.739392 |                                      0 |                                       0 |
| 2023-12-31 00:00:00 |             0.157046 |             0.158389 |           -0.0103349  |           -0.00777187 |                 0.393172 |                 0.397247 |                  0.65982  |                  0.662668 |                                      0 |                                       0 |
| 2024-12-31 00:00:00 |             0.161798 |             0.16281  |            0.0318866  |            0.0338282  |                 0.454807 |                 0.457342 |                  0.419115 |                  0.420233 |                                      0 |                                       0 |

## China Under Scenarios
- No first-pass scenario made EQ_CN material. The strongest China-role case ended with EQ_CN weight 0.0409 and rank 6.
- China remains part of the active system, but the scenario layer still treats it as a secondary sleeve rather than a main allocation driver.

## Public-Facing Interpretation
- The scenario layer is now strong enough for conference use as a model-based explanation layer.
- The right framing is benchmark-conditioned local diagnostics, not causal macro discovery.
- The 60m benchmark is the main object for interpretation; the 120m ceiling is the comparison object that reveals concentration risk.