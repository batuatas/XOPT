# Workflow And Reruns

## Principle

Rerun only what you actually need.

This workspace is a clean v4 handoff pack, but the copied scripts are snapshots of the original v4 scripts. Unless you deliberately retarget them, they still write to the repository’s authoritative v4 data and report directories.

## Recommended Order

Use this order for serious reruns:

1. governance and config review
2. v4 data build
3. v4 modeling prep
4. v4 acceptance audit
5. v4 supervised prediction benchmark
6. v4 supervised portfolio benchmark
7. allocator refinement if you are deliberately retuning the benchmark object
8. conference graphics rebuild

## Scripts And Their Roles

### Data and preparation

- [run_v4_expanded_universe_build.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_expanded_universe_build.py)
  - builds the accepted v4 data branch
- [run_v4_modeling_prep.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_modeling_prep.py)
  - builds the accepted first-pass modeling-prep layer
- [run_v4_acceptance_audit.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_acceptance_audit.py)
  - checks integrity, target plausibility, coverage, and sleeve usability

### Benchmark layers

- [run_v4_prediction_benchmark.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_prediction_benchmark.py)
  - runs the supervised prediction benchmark comparison
- [run_v4_portfolio_benchmark.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_portfolio_benchmark.py)
  - runs the supervised portfolio benchmark comparison

### Allocator tuning and presentation

- [run_v4_allocator_sweep.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_allocator_sweep.py)
  - broad allocator sensitivity sweep
- [run_v4_allocator_refinement.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_allocator_refinement.py)
  - narrower allocator refinement around the 5Y benchmark object
- [run_v4_conference_plots.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_conference_plots.py)
  - rebuilds the current active conference-graphics package

## What Outputs These Stages Create

### Data build

Primary outputs:

- `asset_master.csv`
- `target_series_manifest.csv`
- `macro_mapping.csv`
- `feature_master_monthly.parquet`
- `target_panel_long_horizon.parquet`
- `modeling_panel_hstack.parquet`

### Modeling prep

Primary outputs:

- `modeling_panel_firstpass.parquet`
- split files
- feature-set manifest
- modeling-prep report

### Acceptance

Primary outputs:

- acceptance report
- integrity checks
- coverage and trainability summaries
- sleeve recheck outputs

### Benchmark and portfolio layers

Primary outputs:

- prediction benchmark report and metrics
- portfolio benchmark report and metrics
- allocator refinement report and wealth-path comparisons

### Graphics

Primary outputs:

- active v4 conference graphic package

## What Is Considered Locked

Locked and not to be changed casually:

- the v4 sleeve taxonomy
- `LISTED_RE = ex-U.S. listed real estate`
- euro fixed-income local-return-plus-FX rule
- exclusion of `CR_EU_HY` from the default supervised benchmark roster
- active benchmark predictor:
  - `elastic_net__core_plus_interactions__separate_60`
- active presentation benchmark allocator:
  - `lambda_risk = 8.0`
  - `kappa = 0.10`
  - `omega_type = identity`

## What To Rerun Carefully

Rerun carefully if and only if you intentionally want a new benchmark or new accepted branch:

- allocator refinement
- prediction benchmark comparison
- portfolio benchmark comparison
- conference graphic package after changing the benchmark object

Those are not routine maintenance steps. They change the interpretation layer of the project.
