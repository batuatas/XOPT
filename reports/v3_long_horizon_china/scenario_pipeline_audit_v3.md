# v3 Scenario Pipeline Audit

## Old Mehmet Pipeline: What The Code Actually Does
- `mehmet/probe_eval.py` is the old scenario core. It defines a macro-state probing function `G`, wraps trained PTO/PAO models, and evaluates robust long-only allocations under manipulated macro conditions.
- `mehmet/script1.py` targets benchmark return, `mehmet/script2.py` targets entropy/diversification, and `mehmet/script3.py` compares two trained decision pipelines under the same macro perturbation.
- The manipulated variable in the old code is a low-dimensional macro state `m`; firm characteristics and the trailing covariance matrix stay fixed at the anchor date.
- The old Gibbs / POE target is an energy over macro states. In the benchmark-return case it is squared distance to a return target; in the entropy case it is negative portfolio entropy; in the contrast case it trades off similar return against different risk behavior.
- Gradients in the old code are taken with respect to the macro state, not with respect to portfolio weights or historical returns.
- The old scripts call an external `end2endportfolio.src.langevin` MALA implementation, but that dependency is not present in this repo. The v3 scaffold therefore ships its own small bounded MALA implementation.

## `var1_regularizer.py`: What It Does
- It fits a Gaussian VAR(1) on the old 9-variable macro state and computes innovation Mahalanobis energies.
- That object is a dynamic plausibility prior: it penalizes macro states that are unlikely relative to the previous month, rather than acting as a pure static L2 distance.
- The idea survives into v3, but the state vector has to change because the active project no longer uses the old Goyal-Welch 9-variable setup.

## What Was Reused Directly
- Low-dimensional manipulated state.
- Fixed anchor convention: hold non-manipulated inputs fixed at a chosen month-end.
- Energy-style probes over the full prediction-plus-allocation pipeline.
- Dynamic plausibility regularization via a fitted VAR(1) prior.
- MALA-style local exploration.

## What Had To Change For v3 Long-Horizon SAA
- The old firm-level 1-month TAA design was removed. v3 scenarios operate on sleeve-level rows from `data/final_v3_long_horizon_china/modeling_panel_hstack.parquet`.
- Top-K stock selection, firm characteristic interactions, and 1-step realized return objectives are not reused.
- The active targets are now the locked v3 supervised prediction anchors and their downstream SAA portfolios, not the old archived TAA models.
- The first manipulated state is the interpretable canonical macro block only; enrichments such as `china_cli`, `jp_pe_ratio`, `cape_local`, `mom_12_1`, and `vol_12m` are held fixed at the anchor in this first pass.

## Dry-Run Anchor
- Smoke test anchor date: `2024-12-31`.
- This anchor is recent enough to fit both 60m and 120m predictors using only labels observable by the anchor date, while still allowing both scenario horizons to be instantiated from the active stacked panel.

## Scaffold Check Summary
| component          | status   | details                                                                                                                                                                                                                       |
|:-------------------|:---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| active_paths       | PASS     | Using /Users/batuhanatas/Desktop/XOPTPOE/data/final_v3_long_horizon_china, /Users/batuhanatas/Desktop/XOPTPOE/data/modeling_v3, and /Users/batuhanatas/Desktop/XOPTPOE/reports/v3_long_horizon_china.                         |
| candidate_manifest | PASS     | 7 scenario candidates resolved from active v3 artifacts.                                                                                                                                                                      |
| state_manifest     | PASS     | 51 state variables included in the first pass.                                                                                                                                                                                |
| anchor_context     | PASS     | Anchor 2024-12-31 loaded with 18 stacked rows across horizons (60, 120).                                                                                                                                                      |
| predictor_fit      | PASS     | predictor_60_anchor:981 rows, predictor_120_anchor:855 rows, predictor_shared_anchor:1836 rows                                                                                                                                |
| portfolio_fit      | PASS     | best_60_predictor, best_120_predictor                                                                                                                                                                                         |
| regularizer        | PASS     | Bounds + VAR(1) prior built on 17 manipulable state variables.                                                                                                                                                                |
| probe_manifest     | PASS     | 3 implemented probe definitions and 1 deferred E2E comparator probe.                                                                                                                                                          |
| probe_evaluation   | PASS     | probe_60_target_return: energy=0.000100, grad_norm=0.000918; probe_120_deconcentration: energy=0.130881, grad_norm=skipped_in_smoke_test; probe_60_120_allocation_contrast: energy=-0.314732, grad_norm=skipped_in_smoke_test |
| mala_smoke_test    | PASS     | acceptance_rate=1.000, final_energy=4.802738, steps=2                                                                                                                                                                         |

## Important Caveat
- The old prompt list referenced `reports/v3_long_horizon_china/final_benchmark_manifest_v3.csv`, but the active manifest in this repo is `data/modeling_v3/final_benchmark_manifest_v3.csv`. The scaffold resolves that actual path explicitly.