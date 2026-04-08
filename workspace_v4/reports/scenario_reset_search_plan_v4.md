# Scenario Reset Search Plan — v4

**Date**: 2026-03-30
**Framework**: XOPTPOE v4
**Benchmark**: `best_60_tuned_robust` (λ=8.0, κ=0.10, ω=identity)

---

## Chosen Pipeline: Historical Analog + Latin Hypercube + Bounded Gradient Descent

This document describes the concrete 3-stage search pipeline implemented for the v4 scenario reset.

---

## Stage 1A: Historical Analog Candidate Generation

**Method**: Filter `feature_master_monthly.parquet` for historical months whose macro state is regime-directionally consistent with the question target.

**Why**: Historical states are plausible by construction — they actually occurred. They provide diverse starting points qualitatively different from the anchor, covering crisis states, inflationary episodes, ZIRP periods, and tightening cycles that MALA would never explore.

**Per-question analog filters** (applied to each historical month's macro state):

| Question | Filter Logic | Target Regime |
|---|---|---|
| Q1_gold_threshold | (us_real10y < 0.5) OR (vix > 22) OR (ig_oas > 1.5) OR (infl_US > 3.5) OR (short_rate_US < 0.5) | Gold-favorable: low real yields, stress, inflation |
| Q2_ew_departure | (infl_US > 4.0 AND short_rate_US > 3.0) OR (ig_oas > 1.5 AND vix > 20) OR (infl_US > 5.0 AND ig_oas > 1.2) | High cross-asset dispersion: inflationary tightening |
| Q3_return_discipline | (2.0 < infl_US < 5.0 AND vix < 20 AND ig_oas < 1.3) OR (short_rate_US > 1.0 AND short_rate_US < 4.5 AND ig_oas < 1.0) | Balanced growth: model operational sweet spot |
| Q4_return_ceiling | (infl_US > 6.0 AND vix > 25) OR (short_rate_US > 5.0) OR (vix < 12 AND ig_oas < 0.8 AND infl_US < 3.0) | Extreme states where return ceiling binding |

**Output**: Top K=20 matching historical states, sorted by standardized Euclidean distance from anchor m0 (furthest first — maximizes diversity).

---

## Stage 1B: Latin Hypercube Sampling

**Method**: Generate N=60 LHS candidates within the VAR(1) plausibility box `[μ_{t+1} ± 3σ]`.

**LHS construction** (pure numpy):
1. For each of D=19 macro dimensions, create N strata `[k/N, (k+1)/N]`, k=0..N-1
2. Draw one uniform sample per stratum
3. Shuffle each dimension independently
4. Scale to `[μ_{t+1} - 3σ, μ_{t+1} + 3σ]`

**Plausibility filter**: Keep candidates where Mahalanobis distance from VAR(1) prediction is ≤ 90th percentile of historical one-step Mahalanobis distances (calibrated on full history). This removes dynamically implausible states while retaining regime-diverse coverage.

**Output**: 40-60 filtered LHS candidates.

---

## Stage 2: Bounded Gradient Descent Refinement

**Method**: For the top N_REFINE=15 candidates from Stage 1 (ranked by quick G evaluation), run deterministic gradient descent.

**Algorithm**:
```
for step in 1..15:
    G0 = G(m)
    grad = forward_difference_gradient(G, m, epsilon=1e-3)  # 19 evals
    m_new = clip(m - lr * grad, a_box, b_box)
    backtrack if G(m_new) > G0  # up to 3 halvings
    m = m_new
```

**Key design choices**:
- Forward differences (19 evals/step) instead of central differences (38 evals/step) — 2x faster
- Pure task-loss objective (no regularizer) — gradient points toward scenario target, not back to anchor
- Box constraints from VAR(1) plausibility box — maintains dynamic consistency
- Deterministic — reproducible results

**Output**: 15 refined macro states with final G values and convergence flags.

---

## Stage 3: Ranking and Selection

**Composite score** (lower = better):
```
score = 0.5 * objective_rank + 0.3 * plausibility_rank + 0.2 * regime_diversity_bonus
```

where:
- `objective_rank`: rank of G_final (1 = lowest G = best)
- `plausibility_rank`: rank of Mahalanobis distance from VAR(1) prediction
- `regime_diversity_bonus`: negative bonus for underrepresented regimes (encourages diverse selection)

**Selection rule**: Take top N_SELECT=5, ensuring ≥2 distinct regime labels represented (hard constraint — if top 5 all share same regime, replace worst with best from next regime).

---

## Implementation Notes

**Modules** (in `src/xoptpoe_v4_scenario_reset/`):
- `analog_search.py` — Stage 1A filters + `find_analogs()`
- `grid_sampler.py` — Stage 1B LHS + plausibility filter
- `gradient_refiner.py` — Stage 2 gradient descent
- `ranker.py` — Stage 3 scoring and selection
- `regime_v2.py` — Two-layer regime classification

**Main script**: `scripts/scenario/run_v4_scenario_reset.py`

**Total pipeline calls per question-anchor**: ~(80 quick-eval) + (15 × 15 steps × 20 evals) = ~4,580 — approximately 20× fewer than MALA.

---

## Anchors and Questions

**Anchors**: 2021-12-31, 2022-12-31, 2023-12-31, 2024-12-31

**Questions**:
1. Q1_gold_threshold — Gold activation threshold
2. Q2_ew_departure — Equal-weight departure (concentrated portfolio)
3. Q3_return_discipline — Return improvement without excessive concentration
4. Q4_return_ceiling — 5%+ return scenario and why it fails

**Total runs**: 4 anchors × 4 questions = 16 question-anchor combinations
**Total selected scenarios**: up to 80 (5 per combination)
