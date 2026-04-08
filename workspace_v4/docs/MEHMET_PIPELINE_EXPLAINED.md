# Mehmet Pipeline — Technical Walkthrough

Reference implementation of "Explaining Portfolio Optimization With Scenarios." All files are in `/Users/batuhanatas/Desktop/XOPTPOE/mehmet/` unless otherwise noted. This document covers every file, every data structure, and the full execution flow for a scenario generation run.

---

## 1. File Inventory

| File | Role |
|---|---|
| `probe_eval.py` | **THE CORE SCENARIO ENGINE.** Defines `AllocationPipeline`, `G_function`, `G_contrast_function`, `evaluate`, `traj_outputs`. Entry point for all scenario logic. |
| `e2e_model_defs.py` | Model definitions: `AssetPricingFNN`, `DifferentiableMVOLayer`, `DifferentiableRobustMVOLayer`, `E2EPortfolioModel`, `load_e2e_model_from_run`. Defines both PTO and E2E wrappers. |
| `sigma.py` | Data loading, EWMA Sigma construction, `construct_C` and `construct_C2` functions. Handles firm selection and covariance estimation at a given date. |
| `script1.py` | Experiment 1: PTO vs PAO benchmark return probing (DATE=202002). |
| `script2.py` | Experiment 2: PTO entropy probing (DATE=202404). |
| `script3.py` | Experiment 3: Summer Child vs Winter Wolf contrast probing (DATE=202001). |
| `var1_regularizer.py` | VAR(1) macro prior: fits `m_{t+1} = c + A m_t + eps`, computes Mahalanobis distances, generates samples for plausibility filtering. |
| `utils.py` | PSD projection, step-size schedulers: `sqrt_decay`, `harmonic_decay`, `power_decay`. |
| `dataloaders.py` | `DataStorageEngine` (loads train/val/test parquet data), `strict_metadata_alignment`. |
| `PTO.py` | Full PTO backtest pipeline (MVO backtest over time, not scenario generation). |
| `mehmet.py` | Loading and evaluating FNN model on test data; diagnostic outputs. |
| `batuhan_copy.py` | Early exploratory script — not used in production experiments. |

The MALA sampler itself lives in the installed package: `end2endportfolio.src.langevin` — specifically `torch_MALA_chain`. This is a dependency, not a local file.

---

## 2. Central Data Structures

### 2.1 Macro State

```
m  ∈  R^9   (torch.Tensor, dtype=float32 or float64)
```

The 9 Goyal-Welch macro variables, in this order:
```
Index 0: dp    — log dividend-price ratio
Index 1: ep    — log earnings-price ratio
Index 2: bm    — book-to-market ratio
Index 3: ntis  — net equity issuance
Index 4: tbl   — T-bill rate
Index 5: tms   — term spread (10y - 3m)
Index 6: dfy   — default yield spread (BAA - AAA)
Index 7: svar  — stock variance (monthly)
Index 8: infl  — inflation rate
```

Source: `data['macro_final']` DataFrame from `DataStorageEngine`. Columns: `[yyyymm, date, dp, ep, bm, ntis, tbl, tms, dfy, svar, infl]`. The macro state at date t is: `macro_df[macro_df['yyyymm'] == DATE].iloc[0, 2:]`.

### 2.2 Firm Characteristics Matrix

```
C_t  ∈  R^{K × 140}   (torch.Tensor)
```

- K: number of selected firms at date t (typically 60)
- 140 columns: Chen-Zimmermann firm characteristics (NOT macro variables — those are separate)
- Source: first 140 columns of `X_test` at date t from `DataStorageEngine`
- Fixed during scenario generation — only m is varied

### 2.3 Feature Interaction Matrix

```
interactions  ∈  R^{K × 1400}   (torch.Tensor)
```

Built inside `evaluate()` and inside G(m) at every function evaluation:
```python
mtilde = torch.cat([torch.ones(1), m])                    # shape: (10,)
interactions = C_t[:, None, :] * mtilde[None, :, None]    # shape: (K, 10, 140)
interactions = interactions.flatten(1)                     # shape: (K, 1400)
```

### 2.4 Covariance Matrix

```
Sigma  ∈  R^{K × K}   (torch.Tensor or numpy array)
```

EWMA covariance of selected firms' returns. Fixed during scenario generation.

Parameters:
- Lookback: 60 months
- Lambda (decay): 0.94
- Diagonal shrinkage: 10%
- Ridge: 1e-6

Built by `construct_C2` in `sigma.py`.

### 2.5 Omega Matrix

```
Omega  ∈  R^{K × K}   (torch.Tensor)
```

Used in the robust MVO term `kappa * ||Omega @ w||_2`. In `AllocationPipeline`, Omega is constructed from `Sigma`:
- `diagSigma` mode: Omega = diag(sqrt(diag(Sigma)))
- Other modes: identity

### 2.6 Realized Returns

```
rets_t  ∈  R^K   (torch.Tensor)
```

Realized next-month excess returns for the K selected firms at date t. Source: `y_test` at date t from `DataStorageEngine`. Used to evaluate portfolio outcomes (not passed to the predictor).

### 2.7 Portfolio Weights

```
w_star  ∈  R^K   (torch.Tensor, non-negative, sums to 1)
```

Output of the cvxpy optimizer given predicted returns.

---

## 3. Step-by-Step Execution Flow (Single Scenario Run)

Using `script1.py` as the canonical example.

### Step 1: Load Data

```python
storage = DataStorageEngine("./Data/final_data", load_train=False)
data = storage.load_dataset()
# data keys: X_test, y_test, metadata_test, macro_final
# X_test: DataFrame, rows = (date, permno) pairs, columns = 1400 features
# y_test: DataFrame or Series, rows = (date, permno) pairs, values = 1-month returns
# macro_final: DataFrame, rows = months, columns = [yyyymm, date, dp, ep, bm, ...]
```

### Step 2: Load Models

```python
# PTO model (Predict-Then-Optimize FNN)
pto_model = load_fnn_from_dir("./mehmet/fnn_v1")
# Returns AssetPricingFNN wrapped in E2EPortfolioModel with DifferentiableRobustMVOLayer

# PAO/E2E model (Predict-And-Optimize, trained end-to-end)
pao_model = load_e2e_model_from_run("./mehmet/e2e_state_dicts_bundle/runs/<run_id>/")
# Returns E2EPortfolioModel with same structure but E2E-trained weights
```

Both models are set to `.eval()` mode.

### Step 3: Construct Sigma and Firm Data

```python
DATE = 202002   # integer yyyymm format
Sigma, C_t, rets_t, permnos = construct_C2(data, DATE, ASSET_SIZE=60)
```

Inside `construct_C2`:
1. Filter firms with full 60-month return history ending at DATE
2. Select top-K firms by mean excess return over those 60 months
3. Build EWMA Sigma: `Sigma = ewma_covariance(returns_60m, lambda=0.94, diag_shrink=0.10, ridge=1e-6)`
4. `C_t = X_test.loc[DATE].iloc[:K, :140]` — first 140 columns (firm chars, NOT macro interactions)
5. `rets_t = y_test.loc[DATE].iloc[:K]` — realized returns of selected firms

After this step:
- `Sigma.shape = (60, 60)`
- `C_t.shape = (60, 140)`
- `rets_t.shape = (60,)`
- `permnos`: list of 60 CRSP firm identifiers

### Step 4: Build AllocationPipeline

```python
pi = AllocationPipeline(model, Sigma)
```

Inside `AllocationPipeline.__init__`:
1. Builds the cvxpy robust MVO problem parameterized by `b` (predicted returns):
   ```
   maximize b @ w - kappa * ||Omega @ w||_2 - (lambda/2) * w^T Sigma w
   s.t.  sum(w) = 1,  w >= 0
   ```
2. Converts to `cvxpylayers` `CvxpyLayer` for differentiable forward passes
3. Computes `Omega` from `Sigma` (diagSigma mode or identity, depending on initialization)
4. Stores model reference for access to `.predictor` and `._transform_mu`

### Step 5: Observe Actual Macro State and Set Bounds

```python
macro_df = data['macro_final']
m0 = macro_df[macro_df['yyyymm'] == DATE].iloc[0, 2:].values  # shape: (9,)
m0 = torch.tensor(m0, dtype=torch.float32)

a = torch.tensor((macro_df.iloc[:, 2:].min() - macro_df.iloc[:, 2:].std()).values)
b = torch.tensor((macro_df.iloc[:, 2:].max() + macro_df.iloc[:, 2:].std()).values)
# a.shape = b.shape = (9,)
```

### Step 6: Evaluate Pipeline at m0

```python
results, w_star = evaluate(m0, C_t, rets_t, Sigma, pi)
# results: tuple of (portfolio_return, portfolio_volatility, portfolio_sharpe)
# w_star: torch.Tensor of shape (60,)
```

Inside `evaluate(m, C_t, rets_t, Sigma, pi)`:
```python
mtilde = torch.cat([torch.ones(1), m])                         # shape: (10,)
interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)  # shape: (60, 1400)
preds_raw = pi.model.predictor(interactions)                    # FNN forward pass → (60,)
preds_std = pi.model._transform_mu(preds_raw)                   # zscore or tanh_cap → (60,)
w_star = pi.cvxlayer(preds_std)[0]                             # optimizer → (60,)
portfolio_return = (w_star * rets_t).sum()                      # scalar
portfolio_vol = torch.sqrt(w_star @ Sigma @ w_star)            # scalar
portfolio_sharpe = portfolio_return * (12 ** 0.5) / portfolio_vol  # annualized
return (portfolio_return, portfolio_vol, portfolio_sharpe), w_star
```

### Step 7: Define Probing Function G

```python
b_target = results[0].item() + 0.02   # target = actual return + 2%

G, gradG = G_function(
    pi, C_t, rets_t,
    score_function=b_target,   # benchmark return target
    anchor=m0,
    l2reg=0.1
)
```

`G_function` returns two callables:
- `G(m)`: takes m tensor → returns scalar loss
- `gradG(m)`: takes m tensor → returns gradient tensor of shape (9,) via `torch.autograd.grad`

### Step 8: MALA Sampling

```python
from end2endportfolio.src import langevin

step_schedule = utils.sqrt_decay(eta_0=0.01, t0=10)   # or harmonic_decay, power_decay
beta = tau   # temperature parameter

hypsG = (G, gradG, step_schedule, beta, a, b)

n_seeds = 10
trajectories = []
for seed in range(n_seeds):
    torch.manual_seed(seed)
    m_start = a + (b - a) * torch.rand(9)
    m_last, m_traj = langevin.torch_MALA_chain(m_start, hypsG, n_steps=500)
    trajectories.append(m_traj)
```

`torch_MALA_chain` signature:
```python
def torch_MALA_chain(m_init, hyps, n_steps):
    # hyps = (G, gradG, step_schedule, beta, a, b)
    # Returns: (m_final, trajectory)
    # trajectory: tensor of shape (n_steps, 9)
```

Each step:
1. Compute `g = gradG(m^(k))`
2. Sample `xi ~ N(0, I_9)`
3. Propose `m_prop = clip(m^(k) - eta * g + sqrt(2 * beta * eta) * xi, a, b)`
4. Compute acceptance ratio (Metropolis-Hastings)
5. Accept or reject

### Step 9: Trajectory Analysis

```python
# Subsample trajectory (every 100 steps to reduce autocorrelation)
m_samples = m_traj[::100]   # shape: (5, 9) for 500-step chain

traj_outputs(m_samples, C_t, rets_t, Sigma, pi, permnos)
```

Inside `traj_outputs`:
1. For each m in m_samples: call `evaluate(m, C_t, rets_t, Sigma, pi)`
2. Collect portfolio outcomes and weight vectors
3. Filter valid trajectories (those with G(m) below threshold)
4. Produce pairplot: scatter of m_samples vs historical macro distribution vs m0

---

## 4. AllocationPipeline Internals

Defined in `probe_eval.py`.

```python
class AllocationPipeline:
    def __init__(self, model, Sigma, kappa=0.5, lambda_risk=1.0, omega_mode='diagSigma'):
        self.model = model          # E2EPortfolioModel (wraps FNN predictor + cvxpy layer)
        self.Sigma = Sigma          # (K, K) EWMA covariance
        self.kappa = kappa
        self.lambda_risk = lambda_risk

        # Build Omega
        if omega_mode == 'diagSigma':
            self.Omega = torch.diag(torch.sqrt(torch.diag(Sigma)))  # (K, K)
        else:
            self.Omega = torch.eye(K)

        # Build cvxpylayers problem
        b_param = cp.Parameter(K)   # predicted returns (variable parameter)
        w = cp.Variable(K)
        objective = cp.Maximize(
            b_param @ w
            - kappa * cp.norm(Omega_np @ w, 2)
            - (lambda_risk / 2) * cp.quad_form(w, Sigma_np)
        )
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        self.cvxlayer = CvxpyLayer(prob, parameters=[b_param], variables=[w])

    def __call__(self, m, C_t):
        """Forward pass: macro state m → portfolio weights w*"""
        mtilde = torch.cat([torch.ones(1), m])
        interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
        preds_raw = self.model.predictor(interactions)
        preds_std = self.model._transform_mu(preds_raw)
        w_star = self.cvxlayer(preds_std)[0]
        return w_star
```

---

## 5. G_function Internals — Feature Reconstruction from m

```python
def G_function(pi, C_t, rets_t, score_function, anchor, l2reg=0.1):
    """
    Returns (G, gradG) callables.

    score_function: float (target return b), or 'entropy', or other mode
    anchor: m0 tensor of shape (9,)
    l2reg: regularization strength
    """
    scale = torch.tensor([2.5856, 3.7924, 1.5339, 0.1413, 0.2326, 0.1126, 0.0372, 0.0844, 0.0414])

    def G(m):
        # Feature reconstruction
        mtilde = torch.cat([torch.ones(1), m])                           # (10,)
        interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)  # (K, 1400)

        # Predictor forward pass
        preds = pi.model._transform_mu(pi.model.predictor(interactions))  # (K,)

        # Optimizer forward pass
        w_star = pi.cvxlayer(preds)[0]                                   # (K,)

        # Portfolio outcome
        port_return = (w_star * rets_t).sum()

        # Loss term
        if isinstance(score_function, float):
            # Benchmark return matching
            loss = (port_return - score_function) ** 2
        elif score_function == 'entropy':
            # Maximizing portfolio entropy (diversification)
            entropy = -(w_star * torch.log(w_star + 1e-8)).sum()
            loss = -entropy
        else:
            raise ValueError(f"Unknown score_function: {score_function}")

        # L2 regularization anchor
        reg = l2reg * ((m - anchor) ** 2 / scale ** 2).sum()

        return loss + reg

    def gradG(m):
        m = m.detach().requires_grad_(True)
        g = G(m)
        grad = torch.autograd.grad(g, m)[0]
        return grad.detach()

    return G, gradG
```

The gradient flows: m → mtilde → interactions (linear in m) → FNN forward pass (differentiable) → cvxpylayers optimizer (differentiable via implicit differentiation) → portfolio outcome → loss scalar → autograd backward.

---

## 6. MALA Internals

The `torch_MALA_chain` function in `end2endportfolio.src.langevin` implements:

### Parameters

| Parameter | Typical Value | Role |
|---|---|---|
| `eta_0` (initial step size) | 0.01 | Controls proposal spread |
| `beta` / `tau` (temperature) | 1.0 | Controls sharpness of Gibbs distribution |
| `n_steps` | 500 | Chain length |
| Box constraints `[a, b]` | historical min-std, max+std | Keeps samples plausible |

### Step-Size Schedules (from `utils.py`)

```python
def sqrt_decay(eta_0, t0):
    # Returns schedule function: step(t) = eta_0 * sqrt(t0 / (t0 + t))
    return lambda t: eta_0 * (t0 / (t0 + t)) ** 0.5

def harmonic_decay(eta_0, t0):
    # Returns schedule function: step(t) = eta_0 * t0 / (t0 + t)
    return lambda t: eta_0 * t0 / (t0 + t)

def power_decay(eta_0, t0, p=0.6):
    # Returns schedule function: step(t) = eta_0 * (t0 / (t0 + t))^p
    return lambda t: eta_0 * (t0 / (t0 + t)) ** p
```

### MALA Update (pseudocode)

```python
def torch_MALA_chain(m_init, hyps, n_steps):
    G, gradG, step_schedule, beta, a, b = hyps
    m = m_init.clone()
    trajectory = []

    for t in range(n_steps):
        eta = step_schedule(t)
        g = gradG(m)

        # Langevin proposal
        noise = torch.randn_like(m)
        m_prop = m - eta * g + (2 * beta * eta) ** 0.5 * noise

        # Clip to box constraints
        m_prop = torch.clamp(m_prop, a, b)

        # Metropolis-Hastings acceptance
        log_alpha = (-G(m_prop) + G(m)) / beta    # log acceptance ratio
        if torch.log(torch.rand(1)) < log_alpha:
            m = m_prop

        trajectory.append(m.clone())

    return m, torch.stack(trajectory)
```

---

## 7. VAR(1) Regularizer

`var1_regularizer.py` — implements a VAR(1) macro prior as an alternative plausibility measure.

### What It Fits

```
m_{t+1} = c + A m_t + eps,   eps ~ N(0, Q)
```

Fit by OLS/MLE on the historical macro time series. Produces:
- `c`: intercept vector, shape (9,)
- `A`: transition matrix, shape (9, 9)
- `Q`: residual covariance, shape (9, 9)

### What It Computes

Given current macro state m_t (observed at date t), the VAR(1) model predicts the expected next state:

```
m_predicted = c + A @ m_t
```

The **Mahalanobis distance** of a candidate macro state m from this prediction:

```
d(m) = sqrt((m - m_predicted)^T @ Q^{-1} @ (m - m_predicted))
```

This provides a principled plausibility measure: macro states close to the VAR(1) prediction are more likely to be realistic.

### Saved Artifacts

`var1_regularizer.py` saves:
- Fitted parameters: `c`, `A`, `Q`, `Q_inv`
- Mahalanobis distances for historical states
- Sample trajectories drawn from the VAR(1) process

### Relationship to Plausibility in v4

The VAR(1) regularizer is NOT yet integrated into the main scenario scripts. Current plausibility relies on box constraints + L2 anchor. The VAR(1) would provide a stronger, dynamically consistent plausibility filter: instead of anchoring at m0 with a fixed scale, use the Mahalanobis distance from the VAR(1) predicted distribution.

Integration plan: replace the L2 regularization term in G(m) with:

```
l2reg * d_mahalanobis(m, m_predicted_t+1)^2
```

where `m_predicted_t+1 = c + A @ m_t` is the VAR(1) one-step-ahead prediction from the observed state m_t.

---

## 8. What Each Experiment Does

### script1.py — Benchmark Return Probing (DATE=202002)

**Goal**: For both PTO and PAO pipelines, find macro conditions that would cause the portfolio to earn a return 2% above its realized return in February 2020.

- DATE = 202002 (February 2020 — pre-COVID crash)
- score_function = results[0].item() + 0.02
- Both PTO model and PAO model are loaded and probed
- Two `AllocationPipeline` objects, two `G_function` calls, two MALA runs
- Output: comparison of macro scenario distributions that "rescue" each portfolio

**Key question**: Do the two pipelines (PTO vs E2E) require different macro conditions to achieve the same return improvement?

### script2.py — Entropy Probing (DATE=202404)

**Goal**: For the PTO pipeline, find macro conditions that produce maximum portfolio diversification.

- DATE = 202404 (April 2024)
- score_function = 'entropy'
- G(m) = -entropy(w*(m)) + l2reg_term
- Finds macro states where the Elastic Net predictions are such that the MVO spreads weight evenly across assets
- Output: sampled macro states associated with high-entropy (diversified) allocations

**Interpretation**: When does the PTO pipeline "want" to diversify vs. concentrate?

### script3.py — Model Contrast Probing (DATE=202001)

**Goal**: Find macro conditions where "Summer Child" and "Winter Wolf" (two different E2E models) produce similar portfolio returns but different Sharpe ratios.

- DATE = 202001 (January 2020)
- score_function uses `G_contrast_function`
- G(m) = (w_π1*(m)^T r - w_π2*(m)^T r)^2 + exp(-alpha*(Sharpe_π1 - Sharpe_π2)^2)/alpha
- Two E2E models (named "Summer Child" and "Winter Wolf") loaded from different run directories
- Output: macro conditions where the two models disagree on risk-adjusted performance

**Interpretation**: Which macro regime reveals the most structural difference between the two trained models?

---

## 9. Output Artifacts Produced

For each experiment:

### Trajectory Files
- `m_trajectories_{experiment}.pt`: list of trajectory tensors, each shape `(n_steps, 9)`
- `m_valid_{experiment}.pt`: filtered samples (those achieving low G(m))

### Portfolio Output Tables
- `weights_at_scenarios_{experiment}.csv`: shape `(n_valid_samples, K)` — portfolio weights at each valid scenario
- `outcomes_at_scenarios_{experiment}.csv`: columns `[return, volatility, sharpe]` for each valid scenario

### Visualization
- Pairplot: scatter matrix of the 9 macro variables for sampled states (orange), historical distribution (blue), m0 at date t (red star)
- Weight heatmap: how sleeve/stock allocations shift across sampled macro states

### Reference Files
- Historical macro distribution: `macro_final` from `DataStorageEngine`
- Realized macro state m0: single row at DATE
- Realized portfolio outcome: from `evaluate(m0, C_t, rets_t, Sigma, pi)`
