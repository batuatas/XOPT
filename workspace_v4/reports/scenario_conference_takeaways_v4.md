# Scenario Conference Takeaways — v4 Final

## Benchmark
- Predictor: `elastic_net__core_plus_interactions__separate_60`
- Allocator: Robust MVO (λ=8.0, κ=0.10, Ω=identity)
- Training: 2007-02 to 2021-02 (2368 rows, expanding window)
- Feature rows: `modeling_panel_hstack` (contemporary anchor rows, NOT stale Feb 2021)
- Gate 1 all passed: True

## Anchor-Truth Baseline Values
| Anchor | Pred Return | ALT_GLD | EQ_US | Regime |
|---|---|---|---|---|
| 2021-12-31 | 2.014% | 0.081 | 0.256 | reflation_risk_on |
| 2022-12-31 | 3.296% | 0.223 | 0.137 | higher_for_longer |
| 2023-12-31 | 2.819% | 0.224 | 0.163 | higher_for_longer |
| 2024-12-31 | 2.236% | 0.232 | 0.185 | mixed_mid_cycle |

## Top 3 Regime-Question Stories

### Story 1: Q1_gold_favorable

**Anchor: 2021-12-31**
- Anchor regime: `reflation_risk_on` | Anchor pred_return: 2.014%
- Generated scenarios: n=42, regime_diversity=2
- Mean pred_return in scenarios: 2.127%
- Mean ALT_GLD weight in scenarios: 0.083
- Most common regime transition: `same_regime`
- Key dimension changes: dim_policy: neutral -> easy; dim_fin_cond: neutral -> loose
- Best scenario state shifts:
  - oil_wti: +27.4659
  - usd_broad: -13.1139
  - infl_US: +2.3643

**Anchor: 2022-12-31**
- Anchor regime: `higher_for_longer` | Anchor pred_return: 3.296%
- Generated scenarios: n=21, regime_diversity=2
- Mean pred_return in scenarios: 3.022%
- Mean ALT_GLD weight in scenarios: 0.234
- Most common regime transition: `same_regime`
- Key dimension changes: dim_stress: low -> moderate
- Best scenario state shifts:
  - oil_wti: -14.4524
  - usd_broad: -5.3789
  - vix: +3.1592

**Anchor: 2023-12-31**
- Anchor regime: `higher_for_longer` | Anchor pred_return: 2.819%
- Generated scenarios: n=21, regime_diversity=1
- Mean pred_return in scenarios: 2.484%
- Mean ALT_GLD weight in scenarios: 0.245
- Most common regime transition: `same_regime`
- Key dimension changes: dim_stress: low -> moderate
- Best scenario state shifts:
  - oil_wti: -24.1833
  - usd_broad: -5.8150
  - vix: +4.3901

**Anchor: 2024-12-31**
- Anchor regime: `mixed_mid_cycle` | Anchor pred_return: 2.236%
- Generated scenarios: n=21, regime_diversity=2
- Mean pred_return in scenarios: 1.856%
- Mean ALT_GLD weight in scenarios: 0.245
- Most common regime transition: `mixed_mid_cycle -> higher_for_longer`
- Key dimension changes: dim_inflation: neutral -> high; dim_stress: low -> moderate
- Best scenario state shifts:
  - usd_broad: -13.9056
  - oil_wti: -10.4814
  - vix: -9.7969

**Conference takeaway:**
The model's gold allocation is driven by real-rate compression and credit spread widening.
2021→2022 transition: negative real yields and rising IG OAS are the primary drivers.
Gold appears in the benchmark only when stressed financial conditions dominate growth risk.

### Story 2: Q2_ew_deviation

**Anchor: 2021-12-31**
- Anchor regime: `reflation_risk_on` | Anchor pred_return: 2.014%
- Generated scenarios: n=18, regime_diversity=1
- Mean pred_return in scenarios: 1.422%
- Mean ALT_GLD weight in scenarios: 0.090
- Most common regime transition: `same_regime`
- Key dimension changes: dim_policy: neutral -> easy; dim_fin_cond: neutral -> loose
- Best scenario state shifts:
  - oil_wti: -17.4875
  - vix: -7.9494
  - usd_broad: -7.5660

**Anchor: 2022-12-31**
- Anchor regime: `higher_for_longer` | Anchor pred_return: 3.296%
- Generated scenarios: n=4, regime_diversity=1
- Mean pred_return in scenarios: 3.134%
- Mean ALT_GLD weight in scenarios: 0.231
- Most common regime transition: `same_regime`
- Key dimension changes: dim_stress: low -> moderate
- Best scenario state shifts:
  - oil_wti: +58.6015
  - usd_broad: -14.5532
  - vix: -11.3421

**Anchor: 2023-12-31**
- Anchor regime: `higher_for_longer` | Anchor pred_return: 2.819%
- Generated scenarios: n=21, regime_diversity=1
- Mean pred_return in scenarios: 2.515%
- Mean ALT_GLD weight in scenarios: 0.244
- Most common regime transition: `higher_for_longer -> mixed_mid_cycle`
- Key dimension changes: dim_inflation: high -> neutral; dim_stress: low -> moderate
- Best scenario state shifts:
  - oil_wti: -7.8569
  - vix: -5.7540
  - ig_oas: +1.7935

**Anchor: 2024-12-31**
- Anchor regime: `mixed_mid_cycle` | Anchor pred_return: 2.236%
- Generated scenarios: n=7, regime_diversity=1
- Mean pred_return in scenarios: 1.887%
- Mean ALT_GLD weight in scenarios: 0.257
- Most common regime transition: `mixed_mid_cycle -> higher_for_longer`
- Key dimension changes: dim_inflation: neutral -> high; dim_stress: low -> moderate
- Best scenario state shifts:
  - usd_broad: +5.9781
  - vix: +5.9427
  - oil_wti: -5.9291

**Conference takeaway:**
The model tilts most away from equal weight in environments where return dispersion
across sleeves is highest — typically when rates are rising and credit spreads are
widening (fixed income and credit better differentiated from equity).

### Story 3: Q3_house_view_5pct

**Anchor: 2021-12-31**
- Anchor regime: `reflation_risk_on` | Anchor pred_return: 2.014%
- Generated scenarios: n=16, regime_diversity=1
- Mean pred_return in scenarios: 2.382%
- Mean ALT_GLD weight in scenarios: 0.075
- Most common regime transition: `same_regime`
- Key dimension changes: dim_policy: neutral -> easy; dim_fin_cond: neutral -> loose
- Best scenario state shifts:
  - usd_broad: -8.6500
  - oil_wti: -5.0607
  - term_slope_EA: -1.2890

**Anchor: 2022-12-31**
- Anchor regime: `higher_for_longer` | Anchor pred_return: 3.296%
- Generated scenarios: n=21, regime_diversity=2
- Mean pred_return in scenarios: 3.067%
- Mean ALT_GLD weight in scenarios: 0.231
- Most common regime transition: `same_regime`
- Key dimension changes: dim_stress: low -> moderate
- Best scenario state shifts:
  - oil_wti: +24.1928
  - usd_broad: -7.4308
  - short_rate_EA: +2.0001

**Anchor: 2023-12-31**
- Anchor regime: `higher_for_longer` | Anchor pred_return: 2.819%
- Generated scenarios: n=21, regime_diversity=1
- Mean pred_return in scenarios: 2.532%
- Mean ALT_GLD weight in scenarios: 0.244
- Most common regime transition: `higher_for_longer -> mixed_mid_cycle`
- Key dimension changes: dim_inflation: high -> neutral; dim_stress: low -> moderate
- Best scenario state shifts:
  - oil_wti: -17.0628
  - vix: -7.4001
  - usd_broad: -4.7107

**Anchor: 2024-12-31**
- Anchor regime: `mixed_mid_cycle` | Anchor pred_return: 2.236%
- Generated scenarios: n=42, regime_diversity=2
- Mean pred_return in scenarios: 1.922%
- Mean ALT_GLD weight in scenarios: 0.246
- Most common regime transition: `mixed_mid_cycle -> risk_off_stress`
- Key dimension changes: dim_inflation: neutral -> high
- Best scenario state shifts:
  - oil_wti: +43.9556
  - vix: +20.8420
  - usd_broad: -12.0286

**Conference takeaway:**
Reaching a 5% predicted annualized excess return requires a macro state shift
toward positive real yields, moderate inflation, and low financial stress.
The current 2-3% baseline reflects a world of compressed return premia where
no single regime delivers clean 6-10% strategic house-view assumptions.

## Methodology

- Sampler: MALA (pure numpy, correct MH acceptance ratio)
- Parameters: η=0.008, τ=0.8, 120 steps, 3 chains, warmup=30%
- Plausibility: VAR(1) Mahalanobis regularizer (l2reg=0.3, JITTER=0.01)
- Regime: two-layer (5 dimensional scores + conference label)
- Regime transition analysis: per-sample anchor→scenario tracking
- State: 19-dim compact macro-financial
- Gradient: central finite differences (ε=1e-4)

## Important Caveats

- Predicted returns (mu_hat @ w) are the model's own 5Y forward estimates, not realized.
- All four anchor dates share the same fitted model (train_end=2021-02-28).
- The 2-3% baseline predicted returns reflect the model trained on a low-rate era;
  they are not claims about expected future returns.
- Scenario states are locally plausible (VAR(1) regularized) but not guaranteed
  to be achievable; they are probing functions, not forecasts.
- No conclusions from earlier mismatched baseline (Feb 2021 stale feature rows) survive.