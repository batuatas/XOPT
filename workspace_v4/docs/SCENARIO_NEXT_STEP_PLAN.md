# Scenario Generation — v4 Implementation Plan

Concrete plan for building the v4 scenario generation layer on top of the locked benchmark pipeline. This document is prescriptive: it specifies exactly what to build, in what order, and how.

---

## 1. The First v4 Scenario Object — Precise Specification

### Locked Inputs (do not modify)

```
Predictor label:    elastic_net__core_plus_interactions__separate_60
Portfolio label:    best_60_tuned_robust
lambda_risk:        8.0
kappa:              0.10
omega_type:         identity   → Omega = I_{14}
Covariance:         EWMA, lookback=60m, beta=0.94, diag_shrink=0.10, ridge=1e-6
Universe:           14 sleeves [EQ_US, EQ_EZ, EQ_JP, EQ_CN, EQ_EM, FI_UST,
                                FI_EU_GOVT, CR_US_IG, CR_EU_IG, CR_US_HY,
                                RE_US, LISTED_RE, LISTED_INFRA, ALT_GLD]
```

### First Scenario Date

Use the last available walk-forward test date where:
1. A fitted Elastic Net model exists for `elastic_net__core_plus_interactions__separate_60`
2. The 60m realized returns for all 14 sleeves are available in `target_panel_long_horizon.parquet`
3. 60 months of sleeve return history exist for EWMA covariance estimation

Candidate date: first walk-forward hold period start (e.g., 2016-02-29 or the earliest refit date in the benchmark walk-forward).

### First Probing Question

Benchmark return matching:
```
G(m) = (w*(m)^T ret_60m - b_target)^2  +  l2reg * sum((m_i - m0_i)^2 / scale_i^2)
```
Where:
- `b_target = realized_portfolio_return_at_t + 0.02` (actual benchmark 60m return + 2 percentage points)
- `m0` = macro state at date t from `feature_master_monthly.parquet`
- `scale_i` = historical std of feature i over the training period

---

## 2. What Needs to Be Built

### 2.1 New Files to Create

```
src/xoptpoe_v4_scenario/
    __init__.py
    pipeline.py          # v4AllocationPipeline: wraps Elastic Net + RobustOptimizerCache
    probe_functions.py   # G_function_v4, G_entropy_v4 (returns G and numerical gradG)
    state_space.py       # load_macro_state, build_feature_row, define_box_constraints
    sampler.py           # thin wrapper around end2endportfolio.src.langevin for v4
    outputs.py           # traj_outputs_v4, produce_pairplot, produce_weight_heatmap
    var1_prior.py        # VAR(1) macro prior (port from var1_regularizer.py)
```

Experiment scripts:
```
scripts/scenario/
    run_scenario_benchmark.py    # First experiment: benchmark return matching
    run_scenario_entropy.py      # Second experiment: diversification probing
    run_scenario_stress.py       # Third experiment: NFCI stress regime probing
```

### 2.2 Dependencies Already Available

- `src/xoptpoe_v4_models/optim_layers.py` — `RobustOptimizerCache`, `estimate_ewma_covariance`
- `src/xoptpoe_v4_models/portfolio_benchmark.py` — existing benchmark portfolio construction
- `data_refs/feature_master_monthly.parquet` — macro feature time series
- `data_refs/predictions_test_v4_benchmark.parquet` — predictor outputs
- `end2endportfolio.src.langevin` — MALA sampler (from installed Mehmet package)

### 2.3 Key Functions to Implement

**`pipeline.py` — `v4AllocationPipeline`**
```python
class v4AllocationPipeline:
    def __init__(self, elastic_net_model, Sigma, kappa=0.10, lambda_risk=8.0):
        # Stores: elastic_net_model, Sigma, kappa, lambda_risk
        # Builds RobustOptimizerCache with omega_type='identity'
        # Note: NOT cvxpylayers — plain cvxpy, no differentiability needed

    def predict(self, feature_matrix):
        # feature_matrix: np.ndarray, shape (14, F)
        # Returns: np.ndarray of shape (14,) — predicted 60m excess returns

    def optimize(self, mu_hat):
        # mu_hat: np.ndarray, shape (14,)
        # Returns: np.ndarray of shape (14,) — portfolio weights w*

    def __call__(self, m_perturbed, feature_row_base, fixed_sleeve_features):
        # m_perturbed: the new macro state being evaluated
        # feature_row_base: original feature row at date t
        # fixed_sleeve_features: sleeve-specific features held constant
        # Returns: w* (np.ndarray, shape 14)
```

**`state_space.py` — Feature Row Reconstruction**
```python
def load_macro_state(date_t, feature_master_path):
    # Returns m0: np.ndarray of shape (D,) — macro columns at date t
    # Returns macro_feature_cols: list of column names that are being perturbed

def split_feature_row(feature_row, macro_cols, interaction_cols, sleeve_cols):
    # Splits a full feature row into components

def rebuild_feature_row(m_perturbed, fixed_sleeve_features, macro_cols,
                        interaction_structure):
    # Reconstructs the full (14, F) feature matrix given a new m_perturbed
    # interaction_structure: dict mapping interaction_col → (macro_col, sleeve_col)
    # For pure macro cols: direct replacement
    # For interaction cols: recompute as m_perturbed[macro_col] * fixed_sleeve_features[sleeve_col]

def define_box_constraints(feature_master_df, macro_cols):
    # a = macro_cols.min() - macro_cols.std()
    # b = macro_cols.max() + macro_cols.std()
    # Returns: a (np.ndarray, D), b (np.ndarray, D)
```

---

## 3. Gradient Strategy Decision

### Recommendation: Finite Differences

Use **central differences** for all partial derivatives of G(m):

```python
def numerical_gradG(G, m, epsilon=1e-4):
    grad = np.zeros(len(m))
    for i in range(len(m)):
        m_plus = m.copy();  m_plus[i] += epsilon
        m_minus = m.copy(); m_minus[i] -= epsilon
        grad[i] = (G(m_plus) - G(m_minus)) / (2 * epsilon)
    return grad
```

**Rationale**:
1. Elastic Net has no autograd. Implementing the analytical Jacobian requires auditing every interaction column in the `core_plus_interactions` feature set and mapping it to its constituent macro and sleeve factors — this is significant engineering work.
2. MALA is robust to noisy gradient estimates. Finite differences with `epsilon=1e-4` will be sufficiently accurate.
3. The state dimension D is expected to be moderate (10-30 global macro features). Cost is D+1 or 2D forward passes per MALA step — acceptable.
4. Forward pass cost is minimal: Elastic Net prediction on 14 rows with F columns is microseconds; EWMA covariance is precomputed; MVO solve on 14 variables is fast.

**Central vs. forward differences**: Use central differences for better accuracy (O(eps^2) vs. O(eps) error).

**When to revisit**: If D > 50 or the total MALA chain cost exceeds acceptable wall time, implement the analytical Elastic Net gradient via `coef_` extraction.

### Analytical Option (deferred)

The Elastic Net model exposes `model.coef_` (shape `(14, F)`) and `model.intercept_` (shape `(14,)`). For any feature column j that is a function of macro variable i:

```
∂mu_hat_s / ∂m_i = sum_{j: feature_j depends on m_i} coef_[s, j] * ∂feature_j / ∂m_i
```

For a pure macro column: `∂feature_j / ∂m_i = 1.0`
For an interaction column (macro_i × sleeve_val_k): `∂feature_j / ∂m_i = sleeve_val_k` (fixed)

This requires a complete column dependency map, obtainable from `data_refs/feature_dictionary.csv` and `src/xoptpoe_v4_modeling/features.py`.

---

## 4. State Space Definition for v4

### Columns to Include in m_v4

From `data_refs/feature_master_monthly.parquet`, include all columns that are:
1. **Global macro** — same value for all 14 sleeves at a given date (not sleeve-specific)
2. **Not derived** — base macro features, not pre-computed interaction terms
3. **Available with monthly frequency from at least 2006**

Candidate categories (to be confirmed against `data_refs/feature_dictionary.csv`):
- Interest rates and yield curve: short rate, term spread, credit spread
- Equity market state: dividend yield, earnings yield, P/E, market return
- Financial conditions: NFCI-related indicators, VIX or variance measures
- Economic activity: growth indicators, ISM-type composites
- Inflation and monetary: CPI growth, breakeven inflation

### Columns to Hold Fixed

- Sleeve-specific valuation features (P/E per market, credit spread per sector, etc.)
- Interaction terms that are products of macro × sleeve features (these get recomputed from the perturbed macro values when feature row is rebuilt)

### Typical Expected Dimensionality

D = 10 to 30 (depending on feature dictionary review). This is the dimensionality of m_v4 — the object being sampled by MALA.

### Feature Row Reconstruction Logic

When m_v4 changes from m0 to m_perturbed:

```
For each sleeve s (14 sleeves):
  For each feature column f in the feature row:
    if f is a pure macro column:
        feature_row_perturbed[s, f] = m_perturbed[macro_index(f)]
    elif f is a macro × sleeve interaction:
        feature_row_perturbed[s, f] = m_perturbed[macro_index(f)] * fixed_sleeve_val[s, f]
    else (pure sleeve-specific feature):
        feature_row_perturbed[s, f] = feature_row_base[s, f]   # held fixed
```

---

## 5. Probing Questions — Priority Order

### Priority 1: Benchmark Return Matching

**Why first**: Directly interpretable. Answers: "what macro conditions would have made the portfolio earn more?" Clear target value. Easy to validate (check that sampled states produce portfolio returns near b_target).

```
G_1(m) = (w*(m)^T ret_60m - b_target)^2  +  l2reg * sum((m_i - m0_i)^2 / scale_i^2)
b_target = realized_60m_return_at_t + 0.02
l2reg = 0.1
```

### Priority 2: Diversification / Entropy Probing

**Why second**: Also easily interpretable. Answers: "what macro conditions lead to maximally spread allocations?" Useful for identifying conditions under which the model hedges broadly.

```
G_2(m) = -entropy(w*(m))  +  l2reg * sum((m_i - m0_i)^2 / scale_i^2)
entropy(w) = -sum_s w_s * log(w_s + eps)
l2reg = 0.1
```

### Priority 3: Stress Regime Probing

**Why third**: Most relevant for SAA risk communication. Answers: "what macro conditions cause worst-case allocations, consistent with financial stress?"

```
G_3(m) = -portfolio_return(m)  +  l2reg * sum((m_i - m0_i)^2 / scale_i^2)
         (with NFCI-consistency constraint via additional penalty)
```

Or alternatively: restrict the MALA box constraints to the NFCI-stress region of the macro state space (upper half of NFCI-correlated macro features) and run benchmark probing within those constraints.

### Not Implemented Initially: Model Contrast

There is currently no E2E/PAO model in v4. Model contrast probing requires two trained pipelines. Deferred until an E2E model is available.

---

## 6. NFCI and Regime Labels on Top of Generated Scenarios

### NFCI Data

- Source: NFCI CSV file, weekly, column `NFCI`, from 1971-01-01
- Convert to monthly by taking end-of-month observations or month-average
- Merge with `feature_master_monthly.parquet` on date

### Regime Labeling Function

After MALA produces a set of valid macro states `{m_1, ..., m_N}`, label each state with a regime:

```python
def label_regime(m, nfci_proxy_cols, macro_feature_names, nfci_monthly):
    """
    m: np.ndarray shape (D,) — single sampled macro state
    nfci_proxy_cols: which columns of m correlate with NFCI (e.g., credit spread, VIX)
    nfci_monthly: monthly NFCI series for historical reference

    Returns: regime label string
    """
    # Construct NFCI proxy from m components
    # Compare to historical NFCI percentiles
    # Assign label:
    if nfci_proxy < nfci_25th_percentile:
        return "easy_financial_conditions"
    elif nfci_proxy < nfci_75th_percentile:
        return "neutral"
    elif nfci_proxy < nfci_95th_percentile:
        return "tightening"
    else:
        return "financial_stress"
```

### Additional Regime Labels

Beyond NFCI:
- **Growth regime**: based on macro features correlated with GDP growth proxies
- **Inflation regime**: based on inflation and real rate features
- **Risk-off / risk-on**: based on equity market features

### Output with Regime Labels

Each sampled scenario in the output table gets a column `regime_label`. The summary output shows:
- Portfolio weights conditional on regime (mean weights per regime cluster)
- Frequency of each regime in the sampled distribution (how common each regime is among scenarios that satisfy the probing question)
- Historical frequency of each regime (baseline for comparison)

---

## 7. VAR(1) Integration Plan

### Concept

The VAR(1) prior (from `var1_regularizer.py`) provides a dynamically consistent plausibility filter. Instead of L2 anchoring at m0 with fixed scales, anchor at the VAR(1) predicted distribution one step ahead.

### Data for v4

The macro feature time series in `feature_master_monthly.parquet` constitutes a multivariate monthly time series. The VAR(1) model can be fit to the macro columns over the training period.

### Implementation Steps

1. **Fit VAR(1)** on macro columns of `feature_master_monthly.parquet` over the training period (pre-2016 for walk-forward validation):
   ```
   m_{t+1} = c + A m_t + eps,   eps ~ N(0, Q)
   Fit by OLS: A, c, Q = var1_fit(macro_train)
   ```

2. **At scenario time t**, compute one-step prediction:
   ```
   m_pred = c + A @ m_t
   Q_inv = np.linalg.inv(Q + 1e-6 * I)
   ```

3. **Replace L2 anchor term** in G(m) with Mahalanobis-based regularizer:
   ```
   reg_VAR1(m) = l2reg * (m - m_pred)^T Q_inv (m - m_pred)
   ```

4. **Optionally update box constraints** to reflect the VAR(1) predicted distribution:
   - `a_var1 = m_pred - 3 * sqrt(diag(Q))`
   - `b_var1 = m_pred + 3 * sqrt(diag(Q))`

### Port from Mehmet

`var1_regularizer.py` already implements this logic for the 9 Goyal-Welch variables. The v4 port requires:
1. Replacing the data source (macro columns from `feature_master_monthly.parquet` instead of `macro_final`)
2. Adjusting the state dimension from 9 to D
3. Same OLS fitting, same Mahalanobis computation

Target file: `src/xoptpoe_v4_scenario/var1_prior.py`

---

## 8. Step-by-Step Implementation Order

### Step 1: Audit Feature Space (prerequisite — 1 day)

1. Load `data_refs/feature_master_monthly.parquet` and `data_refs/feature_dictionary.csv`
2. Identify which columns are global macro (same across all sleeves at a date)
3. Identify which columns are interaction terms (product of macro × sleeve feature)
4. Document the column dependency map: `{interaction_col: (macro_col, sleeve_col)}`
5. Define m_v4 as the list of global macro column names → record dimension D

### Step 2: Build state_space.py (1-2 days)

1. Implement `load_macro_state(date_t, feature_master_path)` → returns m0, column names
2. Implement `rebuild_feature_row(m_perturbed, date_t, feature_master_path, interaction_map)` → returns (14, F) array
3. Implement `define_box_constraints(macro_cols, training_period)` → returns a, b arrays
4. Unit test: `rebuild_feature_row(m0, ...)` should return original feature row exactly

### Step 3: Build pipeline.py (1-2 days)

1. Implement `v4AllocationPipeline.__init__` — load fitted Elastic Net, build Sigma, build `RobustOptimizerCache`
2. Implement `predict(feature_matrix)` — call `ElasticNet.predict` on (14, F) input
3. Implement `optimize(mu_hat)` — call `RobustOptimizerCache.solve(mu_hat, Sigma)`
4. Implement `__call__(m_perturbed, ...)` — calls `rebuild_feature_row`, then `predict`, then `optimize`
5. Test: `pipeline(m0, ...)` should produce weights matching the benchmark allocation at date t

### Step 4: Build probe_functions.py (1 day)

1. Implement `G_benchmark_v4(m, pipeline, ret_60m, b_target, m0, scale, l2reg)` → scalar
2. Implement `numerical_gradG(G, m, epsilon=1e-4)` → gradient array
3. Implement `G_entropy_v4(m, pipeline, m0, scale, l2reg)` → scalar
4. Unit test: verify G(m0) is finite, gradG(m0) is non-zero

### Step 5: Build sampler.py (0.5 days)

1. Thin wrapper around `torch_MALA_chain` that accepts numpy arrays and converts to/from torch
2. `run_mala_chains(G, gradG, m0, a, b, n_seeds, n_steps, eta_0, tau)` → list of trajectories

### Step 6: Build outputs.py (1 day)

1. `evaluate_trajectory(m_traj, pipeline, ret_60m, Sigma)` → DataFrame of outcomes per sample
2. `filter_valid_samples(outcomes_df, G, threshold)` → valid samples
3. `produce_pairplot(m_samples, macro_history, m0, macro_col_names)` → matplotlib figure
4. `produce_weight_heatmap(weights_samples, sleeve_names)` → matplotlib figure

### Step 7: Build run_scenario_benchmark.py (0.5 days)

1. Full experiment script using all above modules
2. Load Elastic Net model and Sigma at chosen date t
3. Compute benchmark allocation, realized return, b_target
4. Run MALA (10 chains × 500 steps)
5. Produce outputs: pairplot, weight heatmap, outcome table

### Step 8: Add NFCI Regime Labels (0.5 days)

1. Load NFCI CSV, resample to monthly
2. Implement `label_regime(m_sample, nfci_proxy_mapping)`
3. Add regime column to output tables

### Step 9: Build var1_prior.py (1 day)

1. Port `var1_regularizer.py` logic to v4 state space
2. Fit VAR(1) on training-period macro columns
3. Replace L2 anchor in G with Mahalanobis regularizer
4. Compare outputs: L2-anchor vs. VAR(1)-anchor on same scenario run

### Step 10: Run Priority 2 and 3 Experiments (1 day each)

1. `run_scenario_entropy.py` using `G_entropy_v4`
2. `run_scenario_stress.py` using NFCI-constrained box constraints + return or entropy probing

---

## 9. What the First Scenario Run Should Produce

### Required Outputs

**Console / Log**:
```
Date t: 2016-02-29
Observed macro state m0: [array of D values with column names]
Benchmark portfolio return (60m realized): X.XX%
Benchmark portfolio weights: {sleeve: weight, ...}
Target return b_target: (X.XX + 2.00)% = Y.YY%
G(m0): ZZ.ZZ  [should be > 0, since m0 rarely exactly achieves b_target]

Running 10 MALA chains × 500 steps...
Valid chains (G < threshold): N/10
```

**Saved Files**:
1. `outputs/scenario/benchmark_return/m_trajectories.npy` — shape `(n_valid_chains, 500, D)`
2. `outputs/scenario/benchmark_return/m_valid_samples.npy` — shape `(n_valid, D)`
3. `outputs/scenario/benchmark_return/outcomes.csv` — columns: `[return, volatility, sharpe, G_value, regime_label]`
4. `outputs/scenario/benchmark_return/weights.csv` — shape `(n_valid, 14)`, columns = sleeve names

**Figures**:
5. `outputs/scenario/benchmark_return/pairplot.png` — scatter matrix: sampled states (orange) vs historical (blue) vs m0 (red star), for key macro variable pairs
6. `outputs/scenario/benchmark_return/weight_heatmap.png` — sleeve allocation heatmap across sampled scenarios
7. `outputs/scenario/benchmark_return/regime_breakdown.png` — bar chart of regime label frequency among valid scenarios

### Sanity Checks (verify before declaring success)

1. All valid samples have `w >= 0` and `sum(w) = 1` (allocation constraints satisfied)
2. `portfolio_return(w*, ret_60m) ≈ b_target` for all valid samples (within tolerance)
3. Sampled m values stay within box constraints `[a, b]`
4. Pairplot shows sampled states overlapping with (but displaced from) the historical distribution — not outside the historical range
5. `G(m0) > G(m_valid)` — the valid samples achieve lower loss than the observed state
6. At least 3 of 10 chains converge (otherwise reduce temperature τ or increase chain length)
