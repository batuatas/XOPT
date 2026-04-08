# v4 Handoff Workspace

This folder is a clean v4-focused handoff pack for `XOPTPOE`.

It is designed so that a new coding agent can start here, read a small set of self-contained Markdown files, and understand the current locked v4 setup without reconstructing older project history.

## What This Workspace Contains

- locked v4 governance and design material
- accepted v4 data-build and modeling-prep context
- accepted v4 supervised prediction benchmark context
- accepted v4 portfolio benchmark context
- the final locked active presentation benchmark object
- the current v4 plotting and conference-graphics code
- compact v4 data and report artifacts that are small enough to copy cleanly

## What This Workspace Does Not Do

- it does not replace the original repository structure
- it does not delete, rename, or modify `v1`, `v2`, or `v3`
- it does not move the authoritative historical branches
- it does not retarget copied scripts automatically

The copied scripts are snapshots of the accepted v4 tooling. They still write to the repository’s authoritative v4 data and report paths unless you explicitly adapt them.

## Active v4 Lock

- Active branch status: `v4` is the active downstream branch.
- Historical branches: `v1`, `v2_long_horizon`, and `v3_long_horizon_china` remain frozen historical references.
- Active data universe: 15 sleeves in the data branch.
- Default supervised benchmark roster: 14 sleeves, excluding `CR_EU_HY`.

### Locked active benchmark object

- Predictor: `elastic_net__core_plus_interactions__separate_60`
- Portfolio label: `best_60_tuned_robust`
- Allocator:
  - `lambda_risk = 8.0`
  - `kappa = 0.10`
  - `omega_type = identity`

This is the benchmark object currently used for the presentation and conference graphics.

## Start Here

1. Read [CURRENT_STATUS_AND_NEXT_STEPS.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/CURRENT_STATUS_AND_NEXT_STEPS.md).
2. Read [PROJECT_OVERVIEW.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/PROJECT_OVERVIEW.md).
3. Read [DATA_AND_TARGETS.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/DATA_AND_TARGETS.md).
4. Read [MODELING_STACK.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/MODELING_STACK.md).
5. Read [PORTFOLIO_STACK.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/PORTFOLIO_STACK.md).
6. Use [WORKFLOW_AND_RERUNS.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/WORKFLOW_AND_RERUNS.md) before running anything.

## Directory Guide

- [docs/](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs): self-contained handoff documentation
- [config/](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/config): v4 seed manifests
- [src/](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/src): copied v4 source modules
- [scripts/](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/scripts): copied v4 run scripts
- [reports/](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports): copied accepted reports and benchmark summaries
- [data_refs/](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs): compact copied data artifacts and manifests
- [plots/](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/plots): current active conference graphics

## Practical Guidance For A New Agent

- Treat this workspace as the entry point for new v4 work.
- Do not default back to `v3`.
- Do not reopen sleeve-admission or governance debates unless explicitly asked.
- Keep the long-horizon framing intact: this project is about strategic asset allocation, not monthly tactical allocation.
- If you need the full story of what each copied file does, use [FILE_MAP.md](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/docs/FILE_MAP.md).
