# Scenario Generation Onboarding Summary — XOPTPOE v4

---

## 1. What This Document Is For

This document is the entry point for anyone (human or AI agent) working on the scenario generation layer of the XOPTPOE v4 project. It explains:

- The locked v4 benchmark pipeline (predictor + allocator) that scenario generation operates on top of
- What scenario generation does conceptually and why it is useful
- The academic paper that defines the methodology
- The Mehmet reference implementation and what it does concretely
- The key structural differences between that implementation and the v4 setup
- What needs to be built, and how to interpret outputs

Read this document first. The companion documents are:
- `PAPER_TO_V4_ADAPTATION_MAP.md` — precise component-by-component mapping
- `MEHMET_PIPELINE_EXPLAINED.md` — technical walkthrough of the reference code
- `SCENARIO_NEXT_STEP_PLAN.md` — concrete implementation plan

---

## 2. The v4 Project Benchmark — Locked Object

### Problem Framing

XOPTPOE v4 is a **long-horizon Strategic Asset Allocation (SAA)** research pipeline. It is NOT monthly TAA, NOT stock selection, NOT short-horizon market timing.

- **Core horizon**: 5 years (60 months)
- **Universe**: 14 sleeves (asset class/region segments)
- **Workflow**: observe current macro-financial state → translate to sleeve-level 60-month expected excess returns → convert predictions into a robust long-only allocation

### Sleeve Universe (14 sleeves)

```
EQ_US, EQ_EZ, EQ_JP, EQ_CN, EQ_EM,
FI_UST, FI_EU_GOVT,
CR_US_IG, CR_EU_IG, CR_US_HY,
RE_US, LISTED_RE, LISTED_INFRA,
ALT_GLD
```

Note: `CR_EU_HY` appears in the data branch (15 sleeves) but is excluded from the active universe.

### Target Variable

Annualized 60-month forward excess return per sleeve. Euro FI sleeves use a local-return-plus-FX-to-USD conversion rule.

### Features

- **Primary source**: `data_refs/feature_master_monthly.parquet` — date-indexed monthly panel
- Feature categories: global macro, regional macro, sleeve-linked valuation/technical features, interaction terms
- Macro backbone starts 2006
- The `core_plus_interactions` feature set includes pre-built interaction terms

### Locked Predictor

```
Label: elastic_net__core_plus_interactions__separate_60
```

This is an Elastic Net model trained with walk-forward refits (annual), predicting the annualized 60-month forward excess return for each sleeve. It is the strongest fixed-split winner for the 60m target.

### Locked Allocator

```
Portfolio label:   best_60_tuned_robust
Predictor:         elastic_net__core_plus_interactions__separate_60
lambda_risk:       8.0
kappa:             0.10
omega_type:        identity   → Omega = Identity matrix
```

**Objective** (minimization form):

```
minimize  -mu @ w  +  kappa * ||omega_sqrt @ w||_2  +  0.5 * lambda_risk * ||sigma_sqrt @ w||^2
subject to  sum(w) = 1,  w >= 0
```

Equivalently, in maximization form:

```
maximize  w^T * mu_hat  -  kappa * ||Omega @ w||_2  -  (lambda_risk/2) * w^T * Sigma * w
subject to  sum(w) = 1,  w >= 0
```

**Covariance (Sigma)**:
- Type: EWMA
- Lookback: 60 months
- Decay beta: 0.94
- Diagonal shrinkage: 10%
- Ridge regularization: 1e-6

**Walk-forward**: annual refits, next-year holds (e.g., refit at end of 2016, hold allocation for 2017).

### Key Source Files

| File | Role |
|---|---|
| `src/xoptpoe_v4_models/optim_layers.py` | `RobustOptimizerCache`, `estimate_ewma_covariance`, `OptimizerConfig` |
| `src/xoptpoe_v4_models/portfolio_benchmark.py` | Full portfolio construction pipeline |
| `src/xoptpoe_v4_modeling/features.py` | Feature-set definitions and assembly |
| `data_refs/feature_master_monthly.parquet` | Macro + valuation feature time series |
| `data_refs/predictions_test_v4_benchmark.parquet` | Benchmark predictor outputs (test period) |

---

## 3. What Scenario Generation Does on Top of the Benchmark

The benchmark pipeline is a fixed decision rule: at any date t, it maps the current feature state to a portfolio allocation. Scenario generation **probes the pipeline** by asking:

> What macroeconomic environments would cause the pipeline to behave in a specific way?

Examples of probing questions:
- "What macro conditions cause the portfolio to match a target return of X?"
- "What macro conditions cause the portfolio to be maximally diversified?"
- "Under what macro conditions would stress (elevated NFCI) produce a meaningfully different allocation than a baseline?"

The methodology does NOT modify the pipeline. It leaves the predictor and allocator weights entirely frozen. Instead, it searches the **macro state space** to find distributions of macro states that satisfy the target behavior.

This is useful for:
1. **Attribution**: explaining why the pipeline recommends a given allocation
2. **Stress testing**: finding macro conditions under which the portfolio underperforms
3. **Communication**: translating quantitative allocations into narrative macro scenarios
4. **Regime labeling**: tagging generated scenario clusters with NFCI/regime labels

---

## 4. The Paper Pipeline Summary

### Reference

Ataş, Aydın, Kıral, Birbil — "Explaining Portfolio Optimization With Scenarios"

### Core Concept

Given a trained decision pipeline π (predictor + optimizer), generate macroeconomic scenarios that explain its portfolio behavior. The pipeline π is **fixed** (trained weights θ* are frozen). Scenario generation only manipulates the **input macro state** m.

The method defines a **probing distribution** over macro states:

```
p_π(m) ∝ exp(-G(m) / τ)
```

This is the Gibbs-Boltzmann distribution with temperature τ. Lower G(m) means macro state m better satisfies the target property. The distribution concentrates on macro states that are good answers to the probing question.

**Theoretical justification**: This distribution is the unique minimizer of:

```
min_p  E_p[G]  -  τ * H(p)      (entropy-regularized expected loss)
```

or equivalently:

```
min_p  E_p[G]  +  τ * KL(p || p_0)     (Bayesian update from flat prior)
```

### Probing Function G(m)

G maps a macro state m to a scalar measuring how poorly the target property is satisfied. Lower = better.

**Three canonical probing questions** from the paper:

1. **Benchmark return matching**:
   ```
   G(m) = (w*(m)^T r_{t+1} - b)^2  +  l2reg * (m - m0)^2 / scale^2
   ```
   Finds macro states where portfolio return equals target b.

2. **Entropy / diversification**:
   ```
   G(m) = -entropy(w*(m))  +  l2reg * (m - m0)^2 / scale^2
   ```
   Finds macro states where portfolio is maximally diversified (max Shannon entropy of weights).

3. **Model contrast** (requires two pipelines π1, π2):
   ```
   G(m) = (w_π1*(m)^T r - w_π2*(m)^T r)^2  +  exp(-alpha * (Sharpe_π1(m) - Sharpe_π2(m))^2) / alpha
   ```
   Finds macro states where two pipelines produce similar returns but different Sharpe ratios.

All probing functions include an **L2 regularization anchor** term that penalizes deviation from the observed macro state m0:

```
l2reg * sum((m_i - m0_i)^2 / scale_i^2)
```

This keeps generated scenarios plausible (close to historically observed states).

### MALA Sampling

To sample from p_π(m) ∝ exp(-G(m)/τ), the paper uses **MALA** (Metropolis Adjusted Langevin Algorithm):

1. Compute gradient ∇G(m^(k)) at current state
2. Propose: `m_prop = m^(k) - η * ∇G(m^(k)) + sqrt(2τη) * ξ`,  ξ ~ N(0, I)
3. Accept/reject via Metropolis-Hastings step

Parameters:
- τ (temperature): controls sharpness of distribution
- η (step size): controls proposal spread
- Multiple independent chains from random starting points for diversity
- Box constraints: m ∈ [a, b] where a = min(historical) - std, b = max(historical) + std

---

## 5. The Mehmet Implementation Summary

The Mehmet pipeline (in `/Users/batuhanatas/Desktop/XOPTPOE/mehmet/`) is the reference Python implementation of the paper methodology. It operates on a **stock-selection problem** with monthly horizon.

### What It Does Concretely

At a chosen date t:
1. Loads CRSP/Chen-Zimmermann firm-level data for a cross-section of ~60 stocks
2. Loads the observed macro state m0 ∈ R^9 (9 Goyal-Welch variables)
3. Builds EWMA covariance Sigma (60 × 60, 60-month lookback)
4. Constructs firm-level features by interaction: `interactions = C_t ⊗ [1, m]`
   - C_t shape: (60 firms, 140 firm characteristics)
   - [1, m] shape: (10,) — intercept + 9 macro variables
   - interactions shape: (60, 1400)
5. Passes interactions through FNN predictor → predicted returns mu_hat (shape: 60)
6. Solves robust MVO: `maximize w^T mu_hat - kappa ||Omega w||_2 - (λ/2) w^T Sigma w`
7. Evaluates portfolio outcomes (return, volatility, Sharpe)
8. Defines G(m) for the probing question
9. Runs MALA to sample macro states m satisfying G(m) ≈ 0

### The 9 Macro State Variables

The state m ∈ R^9 consists of Goyal-Welch macro predictors:

```
dp    — log dividend-price ratio
ep    — log earnings-price ratio
bm    — book-to-market ratio
ntis  — net equity issuance
tbl   — T-bill rate
tms   — term spread (10y - 3m)
dfy   — default yield spread (BAA - AAA)
svar  — stock variance (monthly)
infl  — inflation rate
```

### Firm-Level Feature Reconstruction

The critical operation that makes the state searchable is:

```python
mtilde = [1, m]                                    # shape: (10,)
interactions = C_t[:, None, :] * mtilde[None, :, None]  # broadcast
interactions = interactions.reshape(K, 140 * 10)   # shape: (K, 1400)
```

This means: for any new macro state m, the features can be rebuilt analytically by simple multiplication. The gradient ∇G(m) flows through this linear interaction layer into the FNN predictor.

### Three Experiments

- **script1.py**: PTO vs PAO — benchmark return probing at DATE=202002. Compares PTO and E2E portfolios.
- **script2.py**: Entropy probing at DATE=202404. Finds conditions for maximum diversification.
- **script3.py**: Contrast probing at DATE=202001. "Summer Child" vs "Winter Wolf" models — finds conditions where they produce similar returns but different Sharpe.

---

## 6. Key Differences Between Old Pipeline and v4 Setup

| Dimension | Mehmet (Paper) | v4 SAA |
|---|---|---|
| **Problem** | Monthly stock selection | 5-year horizon SAA |
| **Universe** | ~60 stocks (up to 200) | 14 sleeves |
| **Macro state** | R^9 (Goyal-Welch) | R^D (columns of `feature_master_monthly.parquet`) |
| **Feature construction** | C_t ⊗ [1, m] — explicit at inference | Pre-built interactions in parquet; macro columns updated directly |
| **Predictor** | FNN (torch, autodiff) | Elastic Net (sklearn, no autograd) |
| **Gradient of G** | Autograd through FNN | Finite differences or surrogate needed |
| **Optimizer** | cvxpylayers (differentiable) | RobustOptimizerCache (cvxpy, not needed to differentiate through) |
| **Covariance** | 60×60 stock EWMA | 14×14 sleeve EWMA |
| **E2E model** | Yes (PAO trained FNN) | Not available — only PTO elastic net |
| **Realized returns** | Next-month firm returns | 60-month annualized sleeve returns |
| **Target horizon** | 1 month | 60 months |

### Most Important Structural Difference

In Mehmet, the feature reconstruction is **explicit at inference time**: `features = C_t ⊗ [1, m]`. Perturbing m linearly changes features. Gradient flows analytically.

In v4, the features are **pre-built columns** in a parquet file. There is no explicit C_t ⊗ m operation at inference time. When the scenario engine perturbs the macro state, it modifies the **macro columns** of the feature row, while sleeve-specific valuation columns may be held fixed. The predictor (Elastic Net) then computes: `mu_hat = beta_0 + beta @ feature_row`.

This means: gradient of mu_hat with respect to the perturbed macro columns is simply the corresponding coefficients from the Elastic Net — analytically available. No autograd needed.

---

## 7. The v4 Scenario State — What to Perturb

### The Perturbed Object

The scenario state in v4 is a **subset of columns from `feature_master_monthly.parquet`** at a chosen date t — specifically, the global macro columns (and possibly regional macro columns).

The full feature row at date t has shape `(14 sleeves, F features)` where F is the number of features in the `core_plus_interactions` feature set. Each sleeve's feature row contains:
- Global macro features (same across all sleeves at date t)
- Regional macro features (sleeve-specific)
- Sleeve valuation/technical features (sleeve-specific)
- Interaction terms (macro × sleeve-specific, pre-built)

**Default strategy**: perturb only the pure global macro scalar features. Hold sleeve-specific valuation features fixed. When the macro scalar features change, the pre-built interaction columns must also be recomputed (or their effective change must be accounted for in the gradient).

### Exact Column Selection

The macro columns to include in m are identified from `data_refs/feature_dictionary.csv` and the `features.py` feature set definitions. The state vector m_v4 consists of the global macro feature values at date t.

### Perturbing m → Updating Feature Row

For each sleeve at date t:
```
feature_row_perturbed[sleeve] = update(feature_row_original[sleeve], m_perturbed)
```

Where `update` replaces the macro columns with their perturbed values and recomputes any interaction terms that involve those macro columns.

---

## 8. NFCI and Regime Interpretation Layer

### What NFCI Is

The **National Financial Conditions Index (NFCI)** is a weekly index (Federal Reserve Bank of Chicago). Data: weekly observations from 1971-01-01, column `NFCI`.

- Positive NFCI = tighter-than-average financial conditions (stress, credit tightening)
- Negative NFCI = easier-than-average financial conditions (loose credit)
- NFCI near 0 = average conditions

### Role in v4 Scenario Generation

NFCI is NOT a direct feature being perturbed. It is used as an **interpretive overlay** on generated scenarios:

1. **Cluster labeling**: After MALA produces a distribution of macro states, map each sampled state to an NFCI-consistent regime label based on its macro feature values.

2. **Regime definitions**:
   - "Soft landing": NFCI near 0 or negative, falling; growth positive
   - "Higher-for-longer": elevated rates + moderate NFCI; growth slowing
   - "Financial stress": NFCI > +1 or +2; credit spreads wide

3. **Plausibility filter**: Generated scenario clusters can be labeled by which NFCI regime they correspond to, providing a narrative structure for the output.

4. **Historical comparison**: The pairplot of generated macro states vs. historical distribution can be annotated with NFCI regime colors.

---

## 9. What Outputs Scenario Generation Should Produce for v4

For a single scenario run at date t, the outputs are:

### Quantitative Outputs
1. **Sampled macro state trajectories**: MALA chains, shape `(n_steps, D)` where D = dim of perturbed state
2. **Portfolio weights at sampled states**: shape `(n_samples, 14)` — one allocation per sampled state
3. **Portfolio outcomes at sampled states**: return, volatility, Sharpe for each sample
4. **Summary statistics**: mean/std of weights across samples, concentration measures

### Interpretive Outputs
5. **Pairplot**: scatter of sampled states vs. historical macro distribution, with m0 highlighted
6. **Regime labels**: each sample tagged with NFCI-consistent regime category
7. **Allocation heatmap**: how sleeve weights shift across sampled macro conditions
8. **Comparison to benchmark allocation**: how the scenario-implied allocation differs from the baseline

### First Target Output (Priority)
The first scenario run should produce: at the most recent available test date t*, what macro conditions would have caused the benchmark portfolio to earn a return 2% above its actual realized 60m return? — with a pairplot of those conditions and the implied portfolio weights.
