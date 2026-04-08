# Paper-to-v4 Adaptation Map

Maps each component of the paper ("Explaining Portfolio Optimization With Scenarios") to its v4 equivalent. Purpose: precise technical guide for implementing v4 scenario generation from the reference code.

---

## 1. Paper Pipeline Components

### 1.1 State Space

- **Variable**: m ∈ R^9
- **Contents**: Goyal-Welch macro predictors: `dp, ep, bm, ntis, tbl, tms, dfy, svar, infl`
- **Source**: `macro_final` DataFrame loaded via `DataStorageEngine`, columns 2:11 at a given `yyyymm` row
- **Role**: The only object being searched/sampled. Everything else (firm chars C_t, Sigma, realized returns) is fixed.

### 1.2 Feature Construction

- **Operation**: Kronecker/interaction product at inference time
  ```python
  mtilde = torch.cat([torch.ones(1), m])       # shape: (10,)
  interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
  # C_t shape: (K, 140),  interactions shape: (K, 1400)
  ```
- **Property**: Features are a linear function of m → gradient ∇_m(features) is analytically available
- **K**: number of selected stocks at date t (typically 60)

### 1.3 Predictor

- **Model**: FNN (Feed-forward Neural Network) with torch autograd
- **Input**: interactions, shape (K, 1400)
- **Output**: raw predictions preds_raw, shape (K,); then transformed via `_transform_mu` (zscore or tanh_cap)
- **Gradient availability**: Full autograd — ∇_m flows through interactions → FNN → mu_hat

### 1.4 Optimizer

- **Implementation**: `DifferentiableRobustMVOLayer` using `cvxpylayers`
- **Objective**: `maximize w^T mu_hat - kappa ||Omega w||_2 - (λ/2) w^T Sigma w`
- **Constraints**: `sum(w) = 1`, `w >= 0`
- **Differentiability**: Yes — cvxpylayers enables gradient flow through the optimization layer (used during E2E training)
- **Parameters (Mehmet)**:
  - kappa: tuned per run
  - lambda: tuned per run
  - Omega: constructed from Sigma diagonals

### 1.5 Probing Function G(m)

Three canonical forms (see `probe_eval.py`, `G_function` and `G_contrast_function`):

```
G_benchmark(m) = (w*(m)^T r_t+1 - b)^2  +  l2reg * sum((m_i - m0_i)^2 / scale_i^2)

G_entropy(m)   = -entropy(w*(m))  +  l2reg * sum((m_i - m0_i)^2 / scale_i^2)

G_contrast(m)  = (w_π1*(m)^T r - w_π2*(m)^T r)^2
                 + exp(-alpha * (Sharpe_π1(m) - Sharpe_π2(m))^2) / alpha
```

The L2 anchor scale vector (hardcoded in `probe_eval.py`):
```python
scale = [2.5856, 3.7924, 1.5339, 0.1413, 0.2326, 0.1126, 0.0372, 0.0844, 0.0414]
```
Indexed to the 9 macro variables in order.

### 1.6 Sampling

- **Algorithm**: MALA (Metropolis Adjusted Langevin Algorithm) from `end2endportfolio.src.langevin`
- **Proposal**: `m_prop = m^(k) - η ∇G(m^(k)) + sqrt(2τη) ξ`,  ξ ~ N(0, I)
- **Accept/reject**: Metropolis-Hastings correction
- **Multiple chains**: started from random points in [a, b]
- **Steps**: 500 per chain (in scripts)

### 1.7 Plausibility

- **Box constraints**: `m ∈ [a, b]`
  - `a = macro_df.min(axis=0) - macro_df.std(axis=0)`
  - `b = macro_df.max(axis=0) + macro_df.std(axis=0)`
- **L2 anchor**: `l2reg * (m - m0)^2 / scale^2` inside G(m)
- **VAR(1) prior** (not yet integrated into main scripts): fits `m_{t+1} = c + A m_t + eps`, computes Mahalanobis distance from predicted next state

---

## 2. v4 Equivalents for Each Component

### 2.1 State Space — v4

- **Variable**: m_v4 ∈ R^D where D = number of global macro feature columns selected from `feature_master_monthly.parquet`
- **Source file**: `data_refs/feature_master_monthly.parquet` — date-indexed, one row per month
- **Column selection**: global macro columns as defined in `src/xoptpoe_v4_modeling/features.py` feature set construction
- **Role**: Same as paper — the searched/sampled object. Sleeve-specific valuation columns are held fixed.

### 2.2 Feature Construction — v4

- **No explicit C_t ⊗ m at inference time**. The `core_plus_interactions` feature set already has interaction terms pre-built into the parquet files (`modeling_panel_hstack.parquet` or equivalent).
- **At scenario time**: the scenario engine modifies the macro scalar columns in the feature row. Interaction columns that are products of macro scalars × sleeve valuation features must be recomputed from the perturbed macro values.
- **Effective operation**: for each sleeve `s`, given the perturbed macro vector m_v4:
  ```
  feature_row_perturbed[s] = [valuation_features[s],
                               m_v4,
                               m_v4 * valuation_scalar_1[s],
                               m_v4 * valuation_scalar_2[s], ...]
  ```
- **Shape**: `(14 sleeves, F)` where F = total number of features in `core_plus_interactions`

### 2.3 Predictor — v4

- **Model**: Elastic Net (sklearn `ElasticNet`)
- **Label**: `elastic_net__core_plus_interactions__separate_60`
- **Input**: feature row per sleeve, shape `(F,)`
- **Output**: predicted annualized 60m excess return per sleeve, shape `(14,)`
- **Gradient availability**: NO native autograd. See Section 5.

### 2.4 Optimizer — v4

- **Implementation**: `RobustOptimizerCache` from `src/xoptpoe_v4_models/optim_layers.py`
- **Objective** (minimization form):
  ```
  minimize  -mu @ w  +  kappa * ||omega_sqrt @ w||_2  +  0.5 * lambda_risk * ||sigma_sqrt @ w||^2
  s.t.  sum(w) = 1,  w >= 0
  ```
- **Parameters**:
  - `lambda_risk = 8.0`
  - `kappa = 0.10`
  - `omega_type = identity` → Omega = I (identity matrix, shape 14×14)
- **Covariance Sigma**: from `estimate_ewma_covariance`, 60m lookback, beta=0.94, diag_shrink=0.10, ridge=1e-6
- **Differentiability**: NOT needed for MALA in v4 — gradient only needs to flow through the predictor (Elastic Net), not through the optimizer. See Section 6.

### 2.5 Probing Function G(m) — v4

Same structural form as paper, adapted for SAA context:

```
G_benchmark(m_v4) = (w*(m_v4)^T ret_60m - b)^2
                    + l2reg * sum((m_v4_i - m0_i)^2 / scale_v4_i^2)

G_entropy(m_v4)   = -entropy(w*(m_v4))
                    + l2reg * sum((m_v4_i - m0_i)^2 / scale_v4_i^2)
```

Where:
- `ret_60m`: realized 60m annualized excess returns for the 14 sleeves at date t (from `target_panel_long_horizon.parquet`)
- `m0`: actual macro state at date t (row from `feature_master_monthly.parquet`)
- `scale_v4`: per-feature historical standard deviation from the training period

### 2.6 Sampling — v4

- **Same algorithm**: MALA from `end2endportfolio.src.langevin` — directly reusable
- **Box constraints**: `a = macro_cols.min() - macro_cols.std()`, `b = macro_cols.max() + macro_cols.std()` computed over training period
- **Multiple chains**: same strategy as Mehmet scripts

### 2.7 Plausibility — v4

- Same box constraint structure
- L2 anchor with per-feature scale = historical std of each macro column (training period)
- VAR(1) macro prior applicable — v4 has monthly macro time series in `feature_master_monthly.parquet`

---

## 3. What Changes — Technical Details

### 3.1 State Space Dimensionality

| | Paper | v4 |
|---|---|---|
| Dim | 9 | D (to be determined from feature dictionary) |
| Source | `macro_final` DataFrame | `feature_master_monthly.parquet`, macro columns only |
| Type | Monthly economic aggregates | Monthly macro + global financial indicators |

The exact columns for m_v4 must be extracted from `data_refs/feature_dictionary.csv`. The selection criterion: global macro features that are not sleeve-specific (i.e., they appear as the same value for all 14 sleeves at date t).

### 3.2 Feature Row Shape

| | Paper | v4 |
|---|---|---|
| Per-entity features | (140,) firm characteristics | (F_sleeve,) sleeve-specific valuation features |
| Macro state | (9,) Goyal-Welch | (D,) global macro from feature_master |
| Full feature vector | (1400,) = 140 × 10 | (F,) pre-built, with interaction columns already present |
| Number of entities | K stocks (~60) | 14 sleeves |

### 3.3 Predictor I/O Shape

| | Paper | v4 |
|---|---|---|
| Input | (K, 1400) — all stocks simultaneously | (14, F) — all sleeves simultaneously |
| Output | (K,) predicted next-month returns | (14,) predicted 60m annualized excess returns |
| Model type | FNN (torch) | Elastic Net (sklearn) |

### 3.4 Covariance Matrix Shape

| | Paper | v4 |
|---|---|---|
| Shape | (K, K) e.g. (60, 60) | (14, 14) |
| Estimation | `sigma.py` construct_C2 EWMA | `estimate_ewma_covariance` in `optim_layers.py` |
| Parameters | lambda=0.94, 60m, diag_shrink=0.10, ridge=1e-6 | beta=0.94, 60m, diag_shrink=0.10, ridge=1e-6 |

Parameters are identical; only the matrix size differs.

### 3.5 Realized Return Vector Shape

| | Paper | v4 |
|---|---|---|
| Shape | (K,) next-month firm returns | (14,) 60m annualized sleeve excess returns |
| Source | `y_test` in DataStorageEngine | `target_panel_long_horizon.parquet` |

### 3.6 Gradient Computation Strategy

| | Paper | v4 |
|---|---|---|
| Gradient source | Torch autograd through FNN | Finite differences (recommended) |
| Optimizer gradient | Theoretically available via cvxpylayers; often not needed in practice | Not needed — MALA only needs ∇_m G, not ∇_w |
| Gradient through Omega | Not needed for MALA | Not needed |

---

## 4. What Stays the Same

### 4.1 Gibbs-Boltzmann Formulation

The probing distribution `p_π(m) ∝ exp(-G(m)/τ)` is identical in structure. The interpretation is the same: lower G(m) → higher probability → macro state better satisfies the probing question.

### 4.2 MALA Algorithm

The MALA update rule is identical:
```
m_prop = m^(k) - η ∇G(m^(k)) + sqrt(2τη) ξ,   ξ ~ N(0, I)
```
The `torch_MALA_chain` function from `end2endportfolio.src.langevin` is directly reusable with no modification. The only change is that the gradient function `gradG` will be computed via finite differences rather than autograd.

### 4.3 Probing Function Pattern

The G(m) structure — portfolio outcome term + L2 anchor regularization — is identical. Only the portfolio outcome computation changes (different predictor, different sleeve universe).

### 4.4 Box Constraints

```
a = historical_macro.min() - historical_macro.std()
b = historical_macro.max() + historical_macro.std()
```
Applied component-wise. Same logic, different columns.

### 4.5 L2 Anchor Regularization

```
l2reg * sum((m_i - m0_i)^2 / scale_i^2)
```
Same formula. `scale_i` = historical standard deviation of feature i over training period (replaces the hardcoded 9-element vector from the paper).

### 4.6 Multiple Chain Strategy

Starting MALA chains from random points in [a, b] and filtering for chains that converge to low G(m) — same approach.

### 4.7 MVO Objective Form

Both paper and v4 solve the same robust MVO:
```
maximize  w^T mu_hat  -  kappa ||Omega w||_2  -  (λ/2) w^T Sigma w
s.t.  sum(w) = 1,  w >= 0
```
With identical parameter meanings. v4 uses `omega_type=identity` so Omega = I.

---

## 5. Gradient Challenge — Elastic Net Has No Autograd

### The Problem

MALA requires `∇_m G(m)` at each step. G(m) involves:
1. Feature reconstruction from m → feature_row(m)
2. Predictor: `mu_hat = ElasticNet.predict(feature_row(m))` — this is a numpy operation
3. Optimizer: `w* = RobustOptimizerCache.solve(mu_hat, Sigma)` — this is a cvxpy operation
4. Portfolio outcome: `R = w* @ ret_60m` — numpy scalar

None of steps 2-4 has native autograd.

### Option A: Finite Differences (Recommended for v4)

Numerically approximate each partial derivative:

```python
def gradG_fd(m, epsilon=1e-5):
    grad = np.zeros(len(m))
    g0 = G(m)
    for i in range(len(m)):
        m_plus = m.copy(); m_plus[i] += epsilon
        grad[i] = (G(m_plus) - g0) / epsilon
    return grad
```

- **Cost**: D+1 forward passes per MALA step (D = dim of state)
- **Accuracy**: Sufficient for MALA — MALA is robust to noisy gradients
- **Simplicity**: No code changes to predictor or optimizer
- **Recommendation**: Use this for the first v4 implementation

### Option B: Analytical Gradient Through Elastic Net

Elastic Net prediction is linear: `mu_hat_s = beta_0_s + beta_s @ feature_row_s(m)`.

The feature row is a known function of m (with pre-built interactions). So:
```
∂mu_hat_s / ∂m_i = beta_s @ ∂feature_row_s / ∂m_i
```

This is analytically computable if the interaction structure is known. However, it requires:
1. Extracting beta coefficients from the fitted Elastic Net
2. Implementing the exact feature reconstruction Jacobian (which columns depend on which macro variables)

This is accurate and fast but requires more engineering work. Suitable for a second implementation pass.

### Option C: Torch Surrogate Predictor

Train a small torch MLP to mimic the Elastic Net predictions on the training data. Then use torch autograd through the surrogate.

- **Risk**: Surrogate may not perfectly match the Elastic Net
- **Benefit**: Clean autograd pipeline
- **Not recommended** for first implementation — adds a training dependency

### Decision

Use **Option A (finite differences)** for the first v4 scenario generation implementation. Revisit Option B if D is large and compute time becomes a bottleneck.

---

## 6. Differentiability Diagram (v4)

```
MALA step: needs  ∇_m G(m)
           ↓
G(m) = portfolio_loss(w*(m), ret_60m) + l2reg_term(m)
                   ↓
       w*(m) = RobustMVO(mu_hat(m), Sigma)
                            ↓
               mu_hat(m) = ElasticNet(feature_row(m))
                                          ↓
                           feature_row(m) = reconstruct(m, fixed_sleeve_features)

GRADIENT PATH:
   ∇_m G  ←  finite differences on G(m)
              OR analytically:
              ∂G/∂w * ∂w/∂mu_hat * ∂mu_hat/∂m

   ∂G/∂w:       gradient of portfolio_loss w.r.t. w*
                 e.g. for G_benchmark: 2*(R - b) * ret_60m
                 (w* appears linearly in R = w*^T ret_60m)

   ∂w/∂mu_hat:  sensitivity of optimizer output to predicted returns
                 (requires KKT conditions of MVO — available analytically
                  but not needed if using finite differences on full G)

   ∂mu_hat/∂m:  Elastic Net is linear → beta coefficients
                 ∂mu_hat_s/∂m_i = beta_s[interaction_cols_involving_macro_i]

   ∂feature_row/∂m:  Jacobian of feature reconstruction
                      Pure macro columns: ∂/∂m_i = 1 for that column
                      Interaction col (macro_i × valuation_j): ∂/∂m_i = valuation_j (fixed)

NOTE: The OPTIMIZER does NOT need to be differentiated through for MALA.
      MALA only needs ∇_m G, which can be computed entirely via finite differences
      on the scalar G(m) without differentiating through w* at all.
```
