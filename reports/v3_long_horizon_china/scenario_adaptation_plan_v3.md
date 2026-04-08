# v3 Scenario Adaptation Plan

## Active Target Pipelines
- Primary robust portfolio benchmark: `best_60_predictor` built from `elastic_net__full_firstpass__separate_60` plus the locked robust allocator (`lambda=10`, `kappa=0.1`, `omega=identity`).
- Raw ceiling portfolio benchmark: `best_120_predictor` built from `ridge__full_firstpass__separate_120` plus the locked robust allocator (`lambda=10`, `kappa=0.1`, `omega=diag`).
- Prediction anchors carried forward directly: `elastic_net__full_firstpass__separate_60` and `ridge__full_firstpass__separate_120`.
- Shared predictor remains comparator only; E2E remains comparator only in this scaffold because the active artifacts do not include a clean persisted scenario-ready neural model object.

## First Manipulated State
- Manipulate only the canonical macro base state: US / EA / JP inflation, unemployment, short rate, long rate, plus `usd_broad`, `vix`, `us_real10y`, `ig_oas`, and `oil_wti`.
- Rebuild derived deltas, term slopes, 1m/12m log changes, and selected active interaction terms from that state.
- Hold enrichments fixed at the anchor date in the first pass. This preserves interpretability and avoids turning the first MALA state into a high-dimensional opaque feature vector.

## Anchor Convention
- Choose an anchor month-end `t`; the audit uses `2024-12-31`.
- Hold all non-manipulated features at their actual anchor-date values.
- Refit the locked supervised predictor specification on all labels observable by `t` for the relevant horizon(s).
- Feed the manipulated state through the fitted predictor and then, when relevant, the locked robust allocator.
- Evaluate probe energies on that anchor-date pipeline state only. This is a conditional scenario design, not a re-estimation of the full historical experiment zoo.

## First Implemented Probe Families
- `probe_60_target_return`: target predicted annualized excess return for the robust 60m portfolio.
- `probe_120_deconcentration`: keep the raw 120m ceiling attractive while penalizing HHI concentration.
- `probe_60_120_allocation_contrast`: similar predicted portfolio return, different sleeve allocations.
- `probe_60_vs_e2e_disagreement`: specified in the manifest but deferred until a scenario-ready E2E object is materialized.

## Recommended First Workflow
1. Select an anchor date and load the actual v3 stacked rows.
2. Fit the locked supervised prediction anchors using only labels observable by the anchor.
3. Instantiate the locked robust allocator for the 60m and 120m portfolio benchmarks.
4. Build the combined regularizer: historical support bounds + VAR(1) plausibility prior + anchor-distance term.
5. Evaluate probe energies and finite-difference gradients at the actual anchor state.
6. Run a very small bounded MALA smoke test before any large scenario batch is launched.