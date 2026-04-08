# XOPTPOE — Decision Intelligence Platform

## Overview
XOPTPOE is a **decision intelligence platform** designed for strategic asset allocation (SAA). It implements the **Predict–Optimize–Explain (POE)** framework, allowing institutional investors to move beyond black-box output and fully understand the macroeconomic conditions that drive their optimal portfolio allocations.

Instead of generating arbitrary Monte Carlo simulations or simple factor attributions, XOPTPOE searches the macro state-space (using MALA and other scenario engines) to answer targeted, narrative-based questions. Example: *"What macro conditions would cause the optimizer to allocate 15% to gold?"*

This repository represents the transition from the original academic POE research into a structured, scalable decision engine platform.

## Repository Structure

The codebase is organized into nested modules reflecting different stages of the POE evolution:

> [!IMPORTANT]
> For LLMs and human analysts exploring this repo: **`workspace_v4/` is the active codebase.** If you are modifying code, generating scenarios, or analyzing the production-grade pipeline, look there. All other top-level directories (`data/`, `docs/`, `reports/`, etc.) exist as historical architecture or legacy assets and should be ignored for active development unless stated otherwise.

### `workspace_v4/` (Active Implementation)
- **What it is**: The current 14-sleeve strategic asset allocation pipeline.
- **Key differences from earlier versions**: Walks forward an Elastic Net model on 19 macro variables across 14 investable sleeves (Equities, Bonds, Credit, REITs, Infrastructure, Gold) on a 5-year prediction horizon.
- **Components**:
  - `src/xoptpoe_v4_data/`: Data build and feature orchestration.
  - `src/xoptpoe_v4_models/`: Robust MVO optimizer and walk-forward Elastic Net predictor.
  - `src/xoptpoe_v4_scenario/`: MALA-based scenario engine for macro-state sampling and prediction probing.

### `xoptpoe_product_brief.md` (Platform Vision)
- **What it is**: The strategic blueprint for where this repository is heading next. It re-frames XOPTPOE not just as the v4 implementation, but as a modular platform where the data layer, prediction model, optimizer, and scenario engine are all swappable.
- **Why read it**: Before proposing major architectural changes to v4, read this brief to understand the intended multi-tenant, model-agnostic product architecture.

### `mehmet/` (Reference Origin)
- **What it is**: The original research implementation referenced in the Predict-Optimize-Explain theory.
- **Key differences**: Firm-level (stock selection) monthly allocation based on neural networks (PyTorch) using end-to-end differentiable cvxpylayers.
- **Why it's here**: Kept primarily for benchmark reference and research continuity regarding the analytical gradient path versus autograd.

### `archive/` (Legacy Reports & Specs)
- Contains historical `v1`, `v2`, and `v3` design reports, logs, and locked specifications.
- Do not reference `archive/docs/LOCKED_V1_SPEC.md` for current architectural truth.

## Quick Start (workspace_v4)

If you are setting up the environment to run the v4 pipeline:

```bash
# Install dependencies
pip install -r requirements.txt

# To run a scenario probe using the v4 pipeline
cd workspace_v4
python scripts/scenario/run_v4_scenario_final.py
```
*(Please see `workspace_v4/README.md` and `workspace_v4/docs/` for detailed run instructions).*
