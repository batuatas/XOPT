# Final Conference Story v3

## Main-Deck Recommendation
This section should stay compact. The right target is **7 slides maximum**, with the scenario layer as the conceptual climax rather than the modeling layer.

## Recommended Main-Deck Order

### Slide 1: A credible 5Y benchmark is in place
- Figure: `reports/v3_long_horizon_china/plots/benchmark_60_overview.png`
- Why it deserves slide time: The active 60m benchmark improves on equal weight and the neural comparators without making the talk about benchmark horse races.
- Verbal transition: Start with one compact credibility slide and move on quickly so the talk does not become a benchmark horse race.

### Slide 2: 5Y forecasts align with realized outcomes
- Figure: `reports/v3_long_horizon_china/plots/prediction_scatter_60.png`
- Why it deserves slide time: The 60m predictor is noisy but clearly not random, which is enough to justify interpreting it.
- Verbal transition: After establishing there is a benchmark, show one clean prediction-quality figure to make the signal believable.

### Slide 3: The benchmark rolls into an interpretable portfolio path
- Figure: `reports/v3_long_horizon_china/plots/portfolio_wealth_path_best_60.png`
- Why it deserves slide time: The annual 5Y benchmark produces an economically meaningful realized path with readable allocations rather than random rotation.
- Verbal transition: Once the signal looks credible, show that it maps into economically meaningful portfolio behavior rather than arbitrary rotation.

### Slide 4: Current 5Y forecasts are differentiated across sleeves
- Figure: `reports/v3_long_horizon_china/plots/recent_prediction_snapshot_5y.png`
- Why it deserves slide time: The model has a non-uniform forward-looking view today, which motivates asking what macro states support those forecasts.
- Verbal transition: Then pivot from historical validation to the present: what is the model saying now?

### Slide 5: Those forecasts map into diversified allocations
- Figure: `reports/v3_long_horizon_china/plots/recent_portfolio_snapshot_5y.png`
- Why it deserves slide time: The active benchmark converts forward-looking forecasts into a readable multi-sleeve SAA allocation, which is the object scenarios should probe.
- Verbal transition: Use the current allocation snapshot as the bridge into scenarios: these are the benchmark objects we now want to explain.

### Slide 6: Scenarios move the robust and raw benchmarks differently
- Figure: `reports/v3_long_horizon_china/plots/scenario_story_compact_v3.png`
- Why it deserves slide time: The 60m benchmark moves moderately while the 120m ceiling remains more concentration-prone, which is the core scenario contrast.
- Verbal transition: Now introduce the scenario layer with one compact contrast slide rather than a diagnostic dump.

### Slide 7: A small set of state variables explains most of the scenario behavior
- Figure: `reports/v3_long_horizon_china/plots/scenario_case_grid_v3.png`
- Why it deserves slide time: Across representative cases, a small recurring macro subset drives the model-implied benchmark responses.
- Verbal transition: Close the section by showing that a small recurring macro subset explains most of the benchmark-conditioned scenario behavior.

## Appendix Figures
Use these only if questioned on robustness, concentration, or China specifically.

- `reports/v3_long_horizon_china/plots/prediction_rank_spread_60.png`
  Why appendix only: Useful backup, not needed in main flow.

- `reports/v3_long_horizon_china/plots/allocation_heatmap_best_60.png`
  Why appendix only: Too diagnostic for the main deck.

- `reports/v3_long_horizon_china/plots/active_contribution_best_60.png`
  Why appendix only: Appendix only.

- `reports/v3_long_horizon_china/plots/rolling_prediction_quality_60.png`
  Why appendix only: Appendix only.

- `reports/v3_long_horizon_china/plots/sleeve_wealth_paths_actual.png`
  Why appendix only: Only if extra context is needed.

- `reports/v3_long_horizon_china/plots/robust_vs_raw_response_v3.png`
  Why appendix only: Good speaker backup after the main scenario slides.

- `reports/v3_long_horizon_china/plots/raw_deconcentration_case_v3.png`
  Why appendix only: Appendix only.

- `reports/v3_long_horizon_china/plots/china_under_scenarios_compact_v3.png`
  Why appendix only: Appendix only; do not headline China.

## Drop From The Talk
These figures either overemphasize benchmark politics, overemphasize China, or still look too diagnostic for a finance-conference deck.

- `reports/v3_long_horizon_china/plots/benchmark_story_compact.png`
  Reason: Superseded by the 60m-centered story.

- `reports/v3_long_horizon_china/plots/china_in_context.png`
  Reason: Do not use in this talk.

- `reports/v3_long_horizon_china/plots/strategy_path_illustrative.png`
  Reason: Do not use.

- `reports/v3_long_horizon_china/plots/scenario_state_shift_heatmap_v3.png`
  Reason: Superseded by scenario_case_grid_v3.

- `reports/v3_long_horizon_china/plots/scenario_portfolio_change_v3.png`
  Reason: Superseded by cleaner synthesis figures.

## Cleanest Main-Deck Story
1. We built a serious long-horizon SAA benchmark on a 9-sleeve investable universe.
2. The active 60m benchmark is credible enough to interpret.
3. Its allocations are economically meaningful rather than random.
4. The current forecasts and current portfolio are therefore worth explaining.
5. The scenario layer then asks which plausible macro states support those outcomes.

## Compressed 5-Slide Fallback
If Ilker needs a shorter conference version, use this order:

1. `reports/v3_long_horizon_china/plots/benchmark_60_overview.png`
   Why it survives: fastest credibility slide.
2. `reports/v3_long_horizon_china/plots/prediction_scatter_60.png`
   Why it survives: strongest single prediction-quality figure.
3. `reports/v3_long_horizon_china/plots/recent_portfolio_snapshot_5y.png`
   Why it survives: strongest bridge into scenario generation.
4. `reports/v3_long_horizon_china/plots/scenario_story_compact_v3.png`
   Why it survives: cleanest raw-vs-robust scenario contrast.
5. `reports/v3_long_horizon_china/plots/scenario_case_grid_v3.png`
   Why it survives: best single slide showing recurring state drivers.

## Answers
- Strongest main-deck figures: `benchmark_60_overview`, `prediction_scatter_60`, `portfolio_wealth_path_best_60`, `recent_prediction_snapshot_5y`, `recent_portfolio_snapshot_5y`, `scenario_story_compact_v3`, `scenario_case_grid_v3`.
- Strongest single bridge figure into scenario generation: `recent_portfolio_snapshot_5y`.
- Most important scenario figure: `scenario_story_compact_v3` if only one scenario slide survives; `scenario_case_grid_v3` if two scenario slides are available.
- China should not get its own main-deck slide. Keep it appendix only.
- The deck was slightly too benchmark-heavy before this pass. The recommended sequence fixes that by reducing benchmark taxonomy and making scenarios the conceptual endpoint.
- The resulting order is presentation-ready for Ilker.
