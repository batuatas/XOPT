# File Map

## Purpose

This document explains what the copied files in `workspace_v4/` are for.

It is a guide to the handoff pack, not a complete repository manifest.

## Top-Level

- [README.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/README.md)
  - entry point for a new agent

## Docs

- [PROJECT_OVERVIEW.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/PROJECT_OVERVIEW.md)
  - high-level project framing and v4 universe logic
- [DATA_AND_TARGETS.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/DATA_AND_TARGETS.md)
  - v4 sleeve universe, targets, and target-rule logic
- [MODELING_STACK.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/MODELING_STACK.md)
  - modeling-prep structure and accepted supervised benchmark conclusions
- [PORTFOLIO_STACK.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/PORTFOLIO_STACK.md)
  - portfolio-construction logic and the active benchmark object
- [WORKFLOW_AND_RERUNS.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/WORKFLOW_AND_RERUNS.md)
  - rerun order and script purposes
- [CURRENT_STATUS_AND_NEXT_STEPS.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/CURRENT_STATUS_AND_NEXT_STEPS.md)
  - current project status and recommended next moves
- [FILE_MAP.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/FILE_MAP.md)
  - this file

## Config

- [asset_master_seed_v4_expanded_universe.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/config/asset_master_seed_v4_expanded_universe.csv)
  - seed manifest for the v4 sleeve roster
- [target_series_manifest_v4_expanded_universe.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/config/target_series_manifest_v4_expanded_universe.csv)
  - seed manifest for v4 target series

## Source Modules

### Data build

- [xoptpoe_v4_data/config.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_data/config.py)
  - data-build paths and config helpers
- [xoptpoe_v4_data/build.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_data/build.py)
  - v4 data-branch construction logic

### Modeling prep

- [xoptpoe_v4_modeling/features.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_modeling/features.py)
  - feature-set definitions and column selection logic
- [xoptpoe_v4_modeling/splits.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_modeling/splits.py)
  - accepted split logic
- [xoptpoe_v4_modeling/prepare.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_modeling/prepare.py)
  - first-pass modeling-prep build logic
- [xoptpoe_v4_modeling/audit.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_modeling/audit.py)
  - acceptance-audit helpers
- [xoptpoe_v4_modeling/io.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_modeling/io.py)
  - file I/O for v4 modeling-prep artifacts

### Benchmark and portfolio modules

- [xoptpoe_v4_models/prediction_benchmark.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_models/prediction_benchmark.py)
  - supervised benchmark comparison logic
- [xoptpoe_v4_models/portfolio_benchmark.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_models/portfolio_benchmark.py)
  - supervised portfolio benchmark logic
- [xoptpoe_v4_models/allocator_sweep.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_models/allocator_sweep.py)
  - broad allocator sweep
- [xoptpoe_v4_models/allocator_refinement.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_models/allocator_refinement.py)
  - narrow allocator refinement around the active 5Y predictor
- [xoptpoe_v4_models/optim_layers.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_models/optim_layers.py)
  - robust optimizer and risk-penalty machinery
- [xoptpoe_v4_models/portfolio_eval.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_models/portfolio_eval.py)
  - portfolio diagnostics and evaluation helpers
- [xoptpoe_v4_models/data.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_models/data.py)
  - shared v4 benchmark data-loading utilities

### Graphics

- [xoptpoe_v4_plots/io.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_plots/io.py)
  - plotting context and active benchmark object definitions
- [xoptpoe_v4_plots/style.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_plots/style.py)
  - sleeve colors, strategy colors, and presentation style
- [xoptpoe_v4_plots/figures.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src/xoptpoe_v4_plots/figures.py)
  - current active conference-graphics package

## Scripts

- [run_v4_expanded_universe_build.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_expanded_universe_build.py)
- [run_v4_modeling_prep.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_modeling_prep.py)
- [run_v4_acceptance_audit.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_acceptance_audit.py)
- [run_v4_prediction_benchmark.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_prediction_benchmark.py)
- [run_v4_portfolio_benchmark.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_portfolio_benchmark.py)
- [run_v4_allocator_sweep.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_allocator_sweep.py)
- [run_v4_allocator_refinement.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_allocator_refinement.py)
- [run_v4_conference_plots.py](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts/run_v4_conference_plots.py)

These are copied snapshots of the accepted v4 run scripts. They are useful for understanding and controlled reruns, but they still point to the repository’s main v4 paths unless you retarget them.

## Reports

### Accepted and design reports

- [v4_governance_lock.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/accepted/v4_governance_lock.md)
  - locked sleeve universe and target rules
- [v4_target_definition_rules.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/accepted/v4_target_definition_rules.md)
  - explicit written target-definition rules
- [v4_acceptance_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/accepted/v4_acceptance_report.md)
  - acceptance audit conclusion for the v4 branch
- [final_data_design_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/design/final_data_design_report.md)
  - accepted v4 data-design report
- [modeling_prep_v4_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/design/modeling_prep_v4_report.md)
  - accepted v4 modeling-prep report
- [conference_plot_index_v4.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/design/conference_plot_index_v4.md)
  - current active conference-graphics index

### Benchmark and portfolio reports

- [v4_prediction_benchmark_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/benchmark/v4_prediction_benchmark_report.md)
- [v4_portfolio_benchmark_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_portfolio_benchmark_report.md)
- [v4_allocator_refinement_report.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports/portfolio/v4_allocator_refinement_report.md)

## Data References

These are copied compact artifacts intended to make the workspace readable and inspectable without dragging in every historical dataset.

- [asset_master.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/asset_master.csv)
- [target_series_manifest.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/target_series_manifest.csv)
- [macro_mapping.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/macro_mapping.csv)
- [horizon_manifest.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/horizon_manifest.csv)
- [feature_dictionary.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/feature_dictionary.csv)
- [feature_master_monthly.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/feature_master_monthly.parquet)
- [target_panel_long_horizon.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/target_panel_long_horizon.parquet)
- [modeling_panel_hstack.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/modeling_panel_hstack.parquet)
- [modeling_panel_firstpass.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/modeling_panel_firstpass.parquet)
- [predictions_validation_v4_benchmark.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/predictions_validation_v4_benchmark.parquet)
- [predictions_test_v4_benchmark.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/predictions_test_v4_benchmark.parquet)
- [portfolio_benchmark_returns.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/portfolio_benchmark_returns.parquet)
- [feature_set_manifest.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/feature_set_manifest.csv)
- [split_manifest.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/split_manifest.csv)
- [AUTHORITATIVE_ARTIFACTS.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/AUTHORITATIVE_ARTIFACTS.csv)

## Plots

The active conference visuals are copied into:

- [plots/](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/plots)
