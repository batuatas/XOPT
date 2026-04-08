# Modeling Stack

## Modeling-Prep Status

The accepted first-pass modeling-prep branch is built on top of the accepted `v4_expanded_universe` data branch.

The first-pass modeling subset keeps:

- horizons `60` and `120`
- rows with `baseline_trainable_flag == 1`
- rows with `target_available_flag == 1`
- non-null annualized excess forward returns

The `180m` horizon remains in the source data branch for coverage and diagnostics, but it is not part of the accepted first-pass supervised benchmark package.

## Default Split Design

The accepted default split design is deterministic month-block splitting on the common `60m` / `120m` window:

- train: `2007-02-28` to `2012-02-29`
- validation: `2012-03-31` to `2014-02-28`
- test: `2014-03-31` to `2016-02-29`
- extra `60m` tail: `2016-03-31` to `2021-02-28`, retained in the filtered modeling panel but not used in the common split files

The split manifest copied here is:

- [split_manifest.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/split_manifest.csv)

## Feature Sets

The accepted v4 feature-set stack includes:

- `core_baseline`
- `core_plus_enrichment`
- `core_plus_interactions`
- `full_firstpass`

The copied feature-set manifest is:

- [feature_set_manifest.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/feature_set_manifest.csv)

## Default Supervised Benchmark Roster

The accepted default supervised benchmark roster excludes `CR_EU_HY`.

Reason:

- under the accepted split design, `CR_EU_HY` is present in the data branch but has only 5 train rows per horizon in the default common-window setup
- that is too thin for a serious first-pass supervised benchmark sleeve

## Accepted Supervised Benchmark Results

The strongest fixed-split winners are:

- best `60m` predictor:
  - `elastic_net__core_plus_interactions__separate_60`
- best `120m` predictor:
  - `ridge__core_plus_interactions__separate_120`
- best shared `60m+120m` predictor:
  - `elastic_net__core_plus_interactions__shared_60_120`

## Best Practical Benchmark Predictor

The single strongest supervised benchmark to beat is:

- `elastic_net__core_plus_interactions__separate_60`

Why it matters:

- it is the strongest practical benchmark in the fixed-split evidence
- it is the predictor used in the active presentation benchmark object
- it anchors the current pre-scenario visual package

## Model Family Takeaway

The accepted v4 benchmark conclusion is:

- linear models are the strongest practical family
- ridge and elastic net are the default benchmark families
- trees are secondary comparators, not the active benchmark anchor

## What Is Locked Versus Exploratory

### Locked

- accepted v4 data branch
- accepted v4 modeling-prep branch
- default supervised benchmark roster excluding `CR_EU_HY`
- active benchmark predictor:
  - `elastic_net__core_plus_interactions__separate_60`

### Exploratory or secondary

- shared `60m+120m` modeling
- additional tree-model exploration
- any neural or scenario-linked model work beyond the current benchmark layer

## Copied Benchmark Artifacts

- [v4_prediction_benchmark_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/benchmark/v4_prediction_benchmark_report.md)
- [v4_prediction_benchmark_metrics.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/benchmark/v4_prediction_benchmark_metrics.csv)
- [v4_prediction_benchmark_by_sleeve.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/benchmark/v4_prediction_benchmark_by_sleeve.csv)
- [predictions_validation_v4_benchmark.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/predictions_validation_v4_benchmark.parquet)
- [predictions_test_v4_benchmark.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/predictions_test_v4_benchmark.parquet)
