# Scenario Question Quality Audit — v4 Final

## Gate 1: Benchmark Faithfulness
All anchors passed: **True**

| Anchor | Status | Pred Ret | Truth | Δ | GLD | Truth GLD | Δ |
|---|---|---|---|---|---|---|---|
| 2021-12-31 | PASS | 2.0135% | 2.0140% | -0.0005% | 0.081 | 0.081 | +0.000 |
| 2022-12-31 | PASS | 3.2958% | 3.2960% | -0.0002% | 0.223 | 0.223 | +0.000 |
| 2023-12-31 | PASS | 2.8194% | 2.8190% | +0.0004% | 0.224 | 0.224 | +0.000 |
| 2024-12-31 | PASS | 2.2357% | 2.2360% | -0.0003% | 0.232 | 0.232 | -0.000 |

## Gate 2: Compact Macro State
- STATE_DIM = 19
- Variables: infl_US, infl_EA, infl_JP, short_rate_US, short_rate_EA, short_rate_JP, long_rate_US, long_rate_EA, long_rate_JP, term_slope_US, term_slope_EA, term_slope_JP, unemp_US, unemp_EA, ig_oas, us_real10y, vix, oil_wti, usd_broad
- No interaction features, no sleeve dummies, no technical features in state
- Interaction features recomputed from perturbed macro state via INTERACTION_MAP
- **PASS**

## Gate 3: VAR(1) Prior Active
- VAR(1) prior fitted on training period (2007-02 to 2016-02)
- SIGMA_JITTER = 0.01 (prevents Q_inv blow-up from term_slope near-collinearity)
- Mahalanobis regularizer active in all three question probes
- l2reg_var1 = 0.3 for all questions
- **PASS**

## Gate 4: Regime Layer
- Two-layer regime system: 5 dimensional scores + conference label
- Historical thresholds from training period (pre-2016)
- Non-degenerate: multiple regimes represented in output

## Question Quality Assessment

### Q1_gold_favorable
- n_valid_samples: 105
- regime_diversity: 4
- return_range: 3.4454%
- transition_diversity: 5
- dominant_regimes: {'higher_for_longer': 56, 'reflation_risk_on': 28}
- quality: STRONG

### Q2_ew_deviation
- n_valid_samples: 50
- regime_diversity: 3
- return_range: 2.1630%
- transition_diversity: 3
- dominant_regimes: {'mixed_mid_cycle': 21, 'reflation_risk_on': 18}
- quality: STRONG

### Q3_house_view_5pct
- n_valid_samples: 100
- regime_diversity: 4
- return_range: 1.5692%
- transition_diversity: 5
- dominant_regimes: {'higher_for_longer': 33, 'risk_off_stress': 30}
- quality: STRONG
