# XOPTPOE v3 Conference Plot Index

- Active branch only: `v3_long_horizon_china`.
- This index reflects the **final conference-facing story**, not the full internal diagnostics set.
- Main deck should stay selective. The scenario layer is the conceptual climax.
- Historical portfolio figures remain long-horizon SAA decision diagnostics, not a clean tradable monthly wealth backtest.

## Final Main-Deck Recommendation

| Order | Filename | Intended slide title | One-line interpretation | Evidence type |
| --- | --- | --- | --- | --- |
| 1 | `reports/v3_long_horizon_china/plots/benchmark_60_overview.png` | A credible 5Y benchmark is in place | The active 60m benchmark improves on equal weight and the neural comparators without making the talk about benchmark horse races. | historical validation |
| 2 | `reports/v3_long_horizon_china/plots/prediction_scatter_60.png` | 5Y forecasts align with realized outcomes | The 60m predictor is noisy but clearly not random, which is enough to justify interpreting it. | historical validation |
| 3 | `reports/v3_long_horizon_china/plots/portfolio_wealth_path_best_60.png` | The benchmark rolls into an interpretable portfolio path | The annual 5Y benchmark produces an economically meaningful realized path with readable allocations rather than random rotation. | historical decision-period validation |
| 4 | `reports/v3_long_horizon_china/plots/recent_prediction_snapshot_5y.png` | Current 5Y forecasts are differentiated across sleeves | The model has a non-uniform forward-looking view today, which motivates asking what macro states support those forecasts. | recent forward-looking snapshot |
| 5 | `reports/v3_long_horizon_china/plots/recent_portfolio_snapshot_5y.png` | Those forecasts map into diversified allocations | The active benchmark converts forward-looking forecasts into a readable multi-sleeve SAA allocation, which is the object scenarios should probe. | recent forward-looking snapshot |
| 6 | `reports/v3_long_horizon_china/plots/scenario_story_compact_v3.png` | Scenarios move the robust and raw benchmarks differently | The 60m benchmark moves moderately while the 120m ceiling remains more concentration-prone, which is the core scenario contrast. | scenario synthesis |
| 7 | `reports/v3_long_horizon_china/plots/scenario_case_grid_v3.png` | A small set of state variables explains most of the scenario behavior | Across representative cases, a small recurring macro subset drives the model-implied benchmark responses. | scenario synthesis |

## Appendix Recommendation

| Filename | Intended slide title | Why appendix only |
| --- | --- | --- |
| `reports/v3_long_horizon_china/plots/prediction_rank_spread_60.png` | Predicted leaders outperform predicted laggards | Useful backup, not needed in main flow. |
| `reports/v3_long_horizon_china/plots/allocation_heatmap_best_60.png` | The historical 5Y benchmark allocates across multiple sleeves | Too diagnostic for the main deck. |
| `reports/v3_long_horizon_china/plots/active_contribution_best_60.png` | A few sleeves drive most active behavior | Appendix only. |
| `reports/v3_long_horizon_china/plots/rolling_prediction_quality_60.png` | Prediction quality is persistent, not one-month noise | Appendix only. |
| `reports/v3_long_horizon_china/plots/sleeve_wealth_paths_actual.png` | The sleeve universe spans distinct market histories | Only if extra context is needed. |
| `reports/v3_long_horizon_china/plots/robust_vs_raw_response_v3.png` | The raw ceiling reacts more and stays more concentrated | Good speaker backup after the main scenario slides. |
| `reports/v3_long_horizon_china/plots/raw_deconcentration_case_v3.png` | Some deconcentration is possible, but only modestly | Appendix only. |
| `reports/v3_long_horizon_china/plots/china_under_scenarios_compact_v3.png` | China remains secondary under first-pass scenarios | Appendix only; do not headline China. |

## Drop From The Talk

| Filename | Reason |
| --- | --- |
| `reports/v3_long_horizon_china/plots/benchmark_story_compact.png` | Superseded by the 60m-centered story. |
| `reports/v3_long_horizon_china/plots/china_in_context.png` | Do not use in this talk. |
| `reports/v3_long_horizon_china/plots/strategy_path_illustrative.png` | Do not use. |
| `reports/v3_long_horizon_china/plots/scenario_state_shift_heatmap_v3.png` | Superseded by scenario_case_grid_v3. |
| `reports/v3_long_horizon_china/plots/scenario_portfolio_change_v3.png` | Superseded by cleaner synthesis figures. |

## Best Final 5–7 Figures For The Main Deck
1. `benchmark_60_overview.png`
2. `prediction_scatter_60.png`
3. `portfolio_wealth_path_best_60.png`
4. `recent_prediction_snapshot_5y.png`
5. `recent_portfolio_snapshot_5y.png`
6. `scenario_story_compact_v3.png`
7. `scenario_case_grid_v3.png`

## Compressed 5-Slide Fallback
If slide time is tight, keep:
1. `benchmark_60_overview.png`
2. `prediction_scatter_60.png`
3. `recent_portfolio_snapshot_5y.png`
4. `scenario_story_compact_v3.png`
5. `scenario_case_grid_v3.png`

## Most Important Bridge Figure
- `recent_portfolio_snapshot_5y.png`
- Why: it converts current model forecasts into the exact benchmark allocation object that the scenario layer then probes.

## China Recommendation
- No dedicated main-deck slide.
- Keep China as a brief mention in the verbal narrative and use `china_under_scenarios_compact_v3.png` only in the appendix if asked.
