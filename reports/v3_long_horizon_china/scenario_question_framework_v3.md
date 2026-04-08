# v3 Scenario Question Framework

## Scope
- Active branch only: v3_long_horizon_china.
- This pass upgrades the scenario layer from a generic probe menu into a compact conference-grade question set.
- These remain plausibility-regularized, anchor-local, model-implied diagnostics for long-horizon SAA benchmark portfolios.

## Final Question Menu
| question_id                   | question_family        | short_label                 | candidate_name     | question_text                                                                                                                                   |
|:------------------------------|:-----------------------|:----------------------------|:-------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| q_china_materiality           | allocation_tilt        | China materiality           | best_60_predictor  | What plausible regime makes China become a more meaningful sleeve rather than staying marginal?                                                 |
| q_em_tilt                     | allocation_tilt        | EM-supporting regime        | best_60_predictor  | What plausible regime makes EM equities more attractive inside the robust benchmark?                                                            |
| q_gold_tilt                   | allocation_tilt        | Gold-supporting regime      | best_60_predictor  | What plausible regime makes the robust benchmark allocate more meaningfully to gold?                                                            |
| q_higher_for_longer_defensive | strategic_outlook      | Higher-for-longer defensive | best_60_predictor  | What plausible regime gives the robust benchmark a higher-for-longer defensive strategic outlook?                                               |
| q_raw_ceiling_upside          | outcome_target         | Raw ceiling upside          | best_120_predictor | What plausible regime supports the raw 10Y ceiling benchmark when expected return is pushed higher?                                             |
| q_raw_deconcentration         | benchmark_disagreement | Raw ceiling deconcentration | best_120_predictor | What plausible regime lets the raw 10Y ceiling deconcentrate without giving up much expected return?                                            |
| q_robust_double_digit         | outcome_target         | Robust double-digit outlook | best_60_predictor  | What plausible regime keeps the robust 5Y benchmark in double-digit annualized excess-return territory without materially higher concentration? |
| q_robust_raw_disagreement     | benchmark_disagreement | Robust vs raw disagreement  | best_60_predictor  | What plausible regime makes the robust 5Y benchmark and the raw 10Y ceiling disagree most in allocation behavior?                               |
| q_soft_landing_outlook        | strategic_outlook      | Soft landing outlook        | best_60_predictor  | What plausible regime gives the robust benchmark a soft-landing-style strategic outlook?                                                        |
| q_us_equity_tilt              | allocation_tilt        | US equity tilt              | best_60_predictor  | What plausible regime makes US equities the clearest overweight inside the robust benchmark?                                                    |

## Regime Taxonomy
- Dimensions: growth, inflation, stress, rates/financial conditions.
- Buckets: low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates.
| variable_name   | dimension   | scoring_role                         |   low_threshold_z |   high_threshold_z | label_mapping                                                                |
|:----------------|:------------|:-------------------------------------|------------------:|-------------------:|:-----------------------------------------------------------------------------|
| unemp_US        | growth      | lower supports stronger growth       |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| unemp_EA        | growth      | lower supports stronger growth       |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| unemp_JP        | growth      | lower supports stronger growth       |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| infl_US         | inflation   | higher raises inflation score        |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| infl_EA         | inflation   | higher raises inflation score        |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| infl_JP         | inflation   | higher raises inflation score        |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| oil_wti         | inflation   | higher supports reflation pressure   |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| vix             | stress      | higher raises stress                 |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| ig_oas          | stress      | higher raises stress                 |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| usd_broad       | stress      | higher raises stress/tightness       |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| short_rate_US   | rates       | higher tightens financial conditions |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |
| long_rate_US    | rates       | higher tightens financial conditions |             -0.35 |               0.35 | low/neutral/high for growth, inflation, stress; easy/neutral/tight for rates |

## First-Pass Findings
| question_id                   | question_family        | short_label                 | mean_regime     |   mean_return_change |   mean_max_weight_change |
|:------------------------------|:-----------------------|:----------------------------|:----------------|---------------------:|-------------------------:|
| q_china_materiality           | allocation_tilt        | China materiality           | mixed mid-cycle |         -0.0013697   |             -0.0051349   |
| q_em_tilt                     | allocation_tilt        | EM-supporting regime        | mixed mid-cycle |          0.000946076 |              0.00134863  |
| q_gold_tilt                   | allocation_tilt        | Gold-supporting regime      | mixed mid-cycle |          0.000987406 |              0.00532554  |
| q_higher_for_longer_defensive | strategic_outlook      | Higher-for-longer defensive | mixed mid-cycle |          0.00104606  |              0.00253828  |
| q_raw_ceiling_upside          | outcome_target         | Raw ceiling upside          | mixed mid-cycle |          0.000359561 |             -0.000922248 |
| q_raw_deconcentration         | benchmark_disagreement | Raw ceiling deconcentration | mixed mid-cycle |          0.000351599 |              0.00068823  |
| q_robust_double_digit         | outcome_target         | Robust double-digit outlook | mixed mid-cycle |          0.000460459 |             -0.00190539  |
| q_robust_raw_disagreement     | benchmark_disagreement | Robust vs raw disagreement  | mixed mid-cycle |          0.00108374  |              0.00327172  |
| q_soft_landing_outlook        | strategic_outlook      | Soft landing outlook        | mixed mid-cycle |         -0.00188516  |             -0.00669799  |
| q_us_equity_tilt              | allocation_tilt        | US equity tilt              | mixed mid-cycle |          0.000218613 |             -8.67863e-05 |

## Repeating State Variables
| variable_name   |   avg_abs_shift_std_units |   case_count |
|:----------------|--------------------------:|-------------:|
| long_rate_JP    |                 0.184295  |           11 |
| short_rate_JP   |                 0.160456  |           10 |
| infl_JP         |                 0.115626  |            9 |
| long_rate_US    |                 0.0931216 |            5 |
| long_rate_EA    |                 0.0863982 |            3 |
| short_rate_EA   |                 0.0794988 |            5 |
| infl_EA         |                 0.0783045 |            2 |
| unemp_EA        |                 0.0773831 |            4 |
| ig_oas          |                 0.0352321 |           34 |
| unemp_JP        |                 0.0310189 |           33 |

## Interpretation
- The strongest questions are the ones that move return or allocation in a recognizable way while remaining anchor-local and plausible.
- In this first pass, regime labels are driven more by the anchor backdrop than by large cross-question regime jumps. That is consistent with local scenario diagnostics rather than regime-switch forecasting.
- The robust 60m benchmark remains the main interpretation object; the raw 120m ceiling remains the comparison object.
- China remains included, but it does not become a dominant scenario driver in the first pass.