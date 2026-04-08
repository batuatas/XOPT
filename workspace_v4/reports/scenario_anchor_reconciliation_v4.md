# Scenario Anchor Reconciliation v4

## Summary
The **prior scenario baseline** used `modeling_panel_firstpass.parquet` for anchor-date feature rows.
`modeling_panel_firstpass` only covers through 2021-02-28. For anchor dates 2021-12 through 2024-12,
it fuzzy-matches to 2021-02-28 — 306 to 1402 days stale. This was the source of all material mismatch.

The **training model** (fitted on rows up to 2021-02-28 from `modeling_panel_hstack`) was identical
in both cases. The difference was purely in the **feature rows being scored at the anchor date**.

**Fix applied:** `load_data()` in `run_v4_scenario_first_pass.py` now loads `modeling_panel_hstack.parquet`.

## Did the Prior Reconciliation Use the Wrong Benchmark Reference?
**Yes.** The prior session's reconciliation compared the scenario baseline against a
`train_end=2012-02-29` fixed-split model — the original locked benchmark from the v3/v4 fixed-split
campaign (train 2007-02 to 2012-02, n=748 rows). That fixed-split model is **not the correct truth
object for 2021-2024 anchor dates** because:
- The fixed-split test window only covers 2014-03 to 2016-02
- There are no authoritative fixed-split benchmark predictions for 2021-2024
- The correct truth object for recent anchors is an expanding-window refit on all available training data

## Reconciliation Table

| Date | Truth ret | Scen ret (OLD) | Δ ret | Truth GLD | Scen GLD (OLD) | Δ GLD | Truth EQ_US | Δ EQ_US | Root cause |
|---|---|---|---|---|---|---|---|---|---|
| 2021-12-31 | 2.014% | 3.967% | +1.954% | 0.081 | 0.095 | +0.014 | 0.256 | +0.020 | Feature rows 306d stale (Feb 2021 used instead of 2021-12-31) |
| 2022-12-31 | 3.296% | 2.016% | -1.280% | 0.223 | 0.249 | +0.026 | 0.137 | +0.000 | Feature rows 671d stale (Feb 2021 used instead of 2022-12-31) |
| 2023-12-31 | 2.819% | 2.187% | -0.632% | 0.224 | 0.244 | +0.020 | 0.163 | +0.005 | Feature rows 1036d stale (Feb 2021 used instead of 2023-12-31) |
| 2024-12-31 | 2.236% | 2.288% | +0.052% | 0.232 | 0.254 | +0.022 | 0.185 | +0.007 | Feature rows 1402d stale (Feb 2021 used instead of 2024-12-31) |

## After Fix: Expected Reconciliation
After applying the `modeling_panel_hstack` fix, the scenario baseline will evaluate
feature rows from the correct anchor date (±1 month). The expanding-window truth above
becomes the scenario baseline. Remaining tolerance after fix should be <0.1% on pred_return
and <0.005 on individual sleeve weights (numerical precision only).

## Diagnosis Summary
| Issue | Root cause | Fix |
|---|---|---|
| Stale feature rows (306-1402 days) | `modeling_panel_firstpass` ends 2021-02 | Use `modeling_panel_hstack` |
| Wrong benchmark reference in prior reconciliation | Fixed-split (2012-02) is not the truth object for 2021-2024 | Use expanding-window refit as truth |
| 2021 Gold weight 9.5% vs 0% confusion | Prior session assumed fixed-split (0% GLD) was truth; expanding truth is 8.1% GLD at 2021-12-31 | Fixed-split has never seen post-2016 macro; expanding model is correct |
