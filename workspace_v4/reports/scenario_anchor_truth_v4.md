# Scenario Anchor Truth v4

## What This Is
Authoritative benchmark truth table for v4 scenario generation at anchor dates 2021–2024.
Built by refitting `elastic_net__core_plus_interactions__separate_60` on the maximum
available training window and scoring each anchor date using correct contemporary feature rows.

## Benchmark Definition
| Parameter | Value |
|---|---|
| Predictor | `elastic_net__core_plus_interactions__separate_60` |
| Hyperparameters | alpha=0.005, l1_ratio=0.5 |
| Training window | 2007-02-28 to 2021-02-28 (expanding, 2368 rows) |
| Allocator | Robust MVO: λ_risk=8.0, κ=0.10, Ω=identity |
| Risk model | EWMA covariance, lookback=60m, β=0.94, diag-shrink=10%, annualize×12 |
| Feature source | `modeling_panel_hstack.parquet` (covers 2006–2026) |

## Why train_end=2021-02-28 is correct for all four anchors
The `target_available_flag==1` constraint on 60-month forward returns requires
realised returns through the target horizon. For a 60-month horizon, data available at
an anchor in late 2021–2024 has known returns only through early 2021 (60 months before
~mid 2026 is not yet observed). Therefore the training window is **identical** for all
four anchor dates: 2007-02-28 to 2021-02-28 (n=2368 rows).

## Anchor-Date Truth Values

### 2021-12-31
| Item | Value |
|---|---|
| Training rows | 2368 |
| Feature row source | modeling_panel_hstack → resolved to 2021-12-31 |
| Predicted portfolio return | 2.0135% |
| Portfolio risk (σ) | 7.4797% |
| Pred Sharpe (ann. exc.) | 0.269 |
| Portfolio entropy | 2.041 |
| HHI | 0.1540 |
| Effective N sleeves | 6.5 |
| Max weight | 0.256 |

**Sleeve-level weights (>1%):**

| Sleeve | Weight | Pred Return |
|---|---|---|
| EQ_US | 0.256 | 9.301% |
| EQ_JP | 0.127 | 1.625% |
| EQ_CN | 0.053 | 0.751% |
| EQ_EM | 0.024 | 1.700% |
| FI_UST | 0.198 | -1.506% |
| FI_EU_GOVT | 0.045 | -2.963% |
| CR_US_IG | 0.118 | -0.728% |
| CR_US_HY | 0.083 | -0.909% |
| RE_US | 0.014 | 1.629% |
| ALT_GLD | 0.081 | -1.013% |

### 2022-12-31
| Item | Value |
|---|---|
| Training rows | 2368 |
| Feature row source | modeling_panel_hstack → resolved to 2022-12-31 |
| Predicted portfolio return | 3.2958% |
| Portfolio risk (σ) | 10.2406% |
| Pred Sharpe (ann. exc.) | 0.322 |
| Portfolio entropy | 1.846 |
| HHI | 0.1765 |
| Effective N sleeves | 5.7 |
| Max weight | 0.240 |

**Sleeve-level weights (>1%):**

| Sleeve | Weight | Pred Return |
|---|---|---|
| EQ_US | 0.137 | 7.896% |
| FI_UST | 0.240 | 1.418% |
| FI_EU_GOVT | 0.032 | 0.617% |
| CR_US_IG | 0.147 | 2.848% |
| CR_US_HY | 0.161 | 2.293% |
| RE_US | 0.035 | 5.853% |
| LISTED_INFRA | 0.026 | 3.515% |
| ALT_GLD | 0.223 | 3.477% |

### 2023-12-31
| Item | Value |
|---|---|
| Training rows | 2368 |
| Feature row source | modeling_panel_hstack → resolved to 2023-12-31 |
| Predicted portfolio return | 2.8194% |
| Portfolio risk (σ) | 10.2217% |
| Pred Sharpe (ann. exc.) | 0.276 |
| Portfolio entropy | 1.730 |
| HHI | 0.1911 |
| Effective N sleeves | 5.2 |
| Max weight | 0.238 |

**Sleeve-level weights (>1%):**

| Sleeve | Weight | Pred Return |
|---|---|---|
| EQ_US | 0.163 | 6.703% |
| FI_UST | 0.238 | 1.450% |
| CR_US_IG | 0.122 | 1.980% |
| CR_US_HY | 0.203 | 1.520% |
| RE_US | 0.012 | 4.916% |
| LISTED_INFRA | 0.038 | 2.952% |
| ALT_GLD | 0.224 | 2.941% |

### 2024-12-31
| Item | Value |
|---|---|
| Training rows | 2368 |
| Feature row source | modeling_panel_hstack → resolved to 2024-12-31 |
| Predicted portfolio return | 2.2357% |
| Portfolio risk (σ) | 8.6903% |
| Pred Sharpe (ann. exc.) | 0.257 |
| Portfolio entropy | 1.871 |
| HHI | 0.1708 |
| Effective N sleeves | 5.9 |
| Max weight | 0.232 |

**Sleeve-level weights (>1%):**

| Sleeve | Weight | Pred Return |
|---|---|---|
| EQ_US | 0.185 | 4.970% |
| FI_UST | 0.165 | 0.564% |
| CR_US_IG | 0.123 | 1.384% |
| CR_EU_IG | 0.018 | -1.569% |
| CR_US_HY | 0.190 | 1.138% |
| RE_US | 0.027 | 4.034% |
| LISTED_INFRA | 0.058 | 2.144% |
| ALT_GLD | 0.232 | 2.732% |
