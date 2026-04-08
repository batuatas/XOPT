# Scenario Post-Reconciliation Check v4

## Reconciliation Status: PASS (after fix)

The scenario baseline now correctly uses `modeling_panel_hstack.parquet` for feature row lookup.
The anchor-truth is the expanding-window refit on 2007-02-28 to 2021-02-28.

---

## Check 1: Return scale PASS
- Target: `annualized_excess_forward_return`, 60m horizon, decimal annualized
- Predicted returns 2-4% ann. excess — plausible for diversified risk-averse SAA
- No cumulative/annualized confusion; no gross/excess confusion

---

## Check 2: Manipulated state — macro-only PASS
- STATE_DIM = 19
- All 19 variables are macro-financial: inflation, rates, term slopes, unemployment, credit spreads, VIX, oil, USD
- Zero interaction features or technical features in the manipulated state
- Interaction feature columns in the 303-feature model ARE correctly recomputed from the perturbed macro state via INTERACTION_MAP in state_space.py

---

## Check 3: VAR(1) prior integration PASS
- `var1_prior.py` is correctly wired: fitted on training period (2016-02-29 cutoff), active Mahalanobis penalty
- SIGMA_JITTER = 0.01 (fixed from prior session — prevents Q_inv blow-up from term_slope near-collinearity)
- Q_inv condition number: 4814 (acceptable)
- G_reg(m0) at 2021-12-31 ≈ 37.9 — realistic baseline energy
- Gradient of regularizer is non-zero and well-scaled
- **The SIGMA_JITTER=0.01 fix from the prior session must be preserved — do not revert to 1e-10**

---

## Check 4: Regime layer PASS (non-degenerate)
- 76 threshold keys computed from historical data
- Test classifications:
  - 2008-10-31 (GFC): `high_stress` ✓
  - 2020-03-31 (COVID): `high_stress` ✓
  - 2022-06-30 (rate hike): `inflationary_expansion` ✓
  - 2021-12-31 (pre-hike): `inflationary_expansion` ✓
- Labels are economically sensible and non-degenerate

---

## Check 5: Prior scenario conclusions — NOT carry-forward
All scenario results from the prior run (produced with stale Feb-2021 feature rows) must be discarded.
The re-run with `modeling_panel_hstack` will produce new, correct results.

---

## Known remaining issues
1. **term_slope redundancy in state**: term_slope_US = long_rate_US - short_rate_US.
   Including both creates near-linear dependence in Q. The SIGMA_JITTER=0.01 mitigates this
   but does not eliminate the redundancy. Could be reduced to 16-dim state by dropping 3 term_slope
   vars (they can be derived). Deferred — acceptable for now.

2. **Training data ends 2021-02-28 for all four anchors**: All four recent anchor dates share
   the same model. This means the scenario engine probes the same allocator at four different
   macro states, not four progressively updated allocators. This is the correct design for
   probing the *same locked benchmark*, and is NOT a bug.

---

## Readiness Assessment
The scenario engine is **ready for a fresh run** after the `modeling_panel_hstack` fix.
The following components are confirmed correct:
- Feature row lookup at anchor dates ✓ (fixed)
- Training window for expanding refit ✓
- Locked hyperparameters (alpha=0.005, l1=0.5) ✓
- Robust MVO allocator (λ=8.0, κ=0.10, Ω=I) ✓
- VAR(1) prior (JITTER=0.01) ✓
- MALA sampler (ETA=0.005, TAU=0.5, 4 chains, 200 steps) ✓
- Regime classification (non-degenerate, economically sensible) ✓

Previous scenario output files in `reports/scenario/` are **stale and should be regenerated**.
