# Scenario Explainer Graphics Index — v4 Final

**Date**: 2026-03-30
**Output directory**: `reports/scenario_explainer_graphics_v4_final/`
**Script**: `scripts/scenario/build_scenario_explainer_graphics_v4_final.py`
**Benchmark**: `best_60_tuned_robust` (λ=8.0, κ=0.10, ω=identity)

---

## Main-Stage Figures

### Fig 01 — Three Questions About the Benchmark
**File**: `scenario_explainer_01_overview_v4.{png,pdf}`
**Purpose**: Package opener — shows all three scenario questions and their answers via actual data thumbnails.
**Storyline position**: First figure; sets the frame before any methodology.
**Speaker takeaway**: "Three questions. Three structural findings. No text boxes — each question is answered by the data below it."

---

### Fig 02 — Scenario Search: How We Find Plausible Macro Regimes
**File**: `scenario_explainer_02_search_method_v4.{png,pdf}`
**Purpose**: Explains the 3-stage reset pipeline (Historical Analog → LHS → Gradient Refinement → Ranked Selection) as a clean left-to-right flow diagram. No MALA framing, no colored boxes.
**Storyline position**: Second figure; earns credibility for the search results shown in subsequent figures.
**Speaker takeaway**: "Historical analogs give us diverse starting points. Gradient descent sharpens each one. No random walk, no hand-picked scenarios."

---

### Fig 03 — What Each Question Asks and How the Search Optimises It
**File**: `scenario_explainer_03_question_and_objective_v4.{png,pdf}`
**Purpose**: Three-column reference card showing the scalar objective G(m) for each question, what minimising it means, and what the search actually found.
**Storyline position**: Third figure; bridges methodology to results — audience sees the precise question before seeing the answer.
**Speaker takeaway**: "Each objective is a single number. The search finds the macro state that makes that number as small as possible — subject to plausibility constraints."

---

### Fig 04 — Did the Search Achieve Its Objective?
**File**: `scenario_explainer_04_search_achievement_v4.{png,pdf}`
**Purpose**: Strip-plot showing all candidates and selected scenarios vs. anchor baseline for Q1 (gold weight), Q3 (predicted return vs 4% target), and Q4 (predicted return vs 5% target). Confirms the search worked and makes the return ceiling visible.
**Storyline position**: Fourth figure; transition from "what we asked" to "what we found."
**Speaker takeaway**: "Gold search clearly separates 2021 from 2022+. Return targets of 4% and 5% are never reached — that is the finding, not the failure."

---

### Fig 05 — Plausibility Space vs Selected Scenario Values
**File**: `scenario_explainer_05_prior_vs_selected_macros_v4.{png,pdf}`
**Purpose**: For the 2022 anchor (richest regime), shows the historical range of 4 key macro variables alongside where Q1 selected scenarios land. Confirms scenarios are plausible but non-trivial.
**Storyline position**: Fifth figure; supporting context for the gold activation story.
**Speaker takeaway**: "Selected scenarios are not extreme outliers — they sit within historically observed ranges. The model is not being asked to operate in science fiction."

---

### Fig 06 — Gold Activation: From Regime Shift to Portfolio Consequence
**File**: `scenario_explainer_06_regime_to_portfolio_v4.{png,pdf}`
**Purpose**: Three-panel hero figure: (left) gold weight across all 4 anchors with regime labels; (centre) the 3 macro variables that shifted between 2021 and 2022 with magnitude callouts; (right) 2021 vs 2022 portfolio composition side-by-side.
**Storyline position**: Sixth figure; the centrepiece of the conference section — the full gold activation story in one image.
**Speaker takeaway**: "Real yields turned positive, short rates surged, credit spreads widened. The model doubled gold automatically — not because it was told to, but because the regime crossed a threshold."

---

## Appendix Figures

### Fig 07 — What Actually Changed Between Anchor Dates (Appendix)
**File**: `scenario_explainer_07_real_world_vs_model_world_v4.{png,pdf}`
**Purpose**: Two-panel appendix: (top) heatmap of the 4 key macro variables across all 4 anchor dates, showing the 2022 surge; (bottom) benchmark gold allocation response at each anchor.
**Storyline position**: Appendix backup for Fig 06 — shows the raw macro numbers for audiences who want the data behind the transition.
**Speaker takeaway**: "The 2022 column shows simultaneous increases in real yields, short rates, and inflation — triggering the gold transition."

---

### Fig 08 — Portfolio Shift Detail: Dec 2021 → Dec 2022 (Appendix)
**File**: `scenario_explainer_08_allocation_shift_detail_v4.{png,pdf}`
**Purpose**: Two-panel appendix: (left) sleeve-level weight changes 2021→2022 as a horizontal bar chart with magnitude labels; (right) return ceiling strip plot showing all Q4 candidates across all anchors.
**Storyline position**: Appendix detail for Fig 06 and Fig 04 — granular portfolio arithmetic for the gold activation and ceiling stories.
**Speaker takeaway**: "Gold +14pp, UST +5pp, EU equities and bonds to zero. The 2022 portfolio is maximally concentrated within the robust optimizer's constraints — and still can't reach 5%."

---

## Figure Order for Conference Presentation

| Slot | Figure | Label |
|---|---|---|
| 1 | Fig 01 — Three Questions | Main stage |
| 2 | Fig 02 — Search Method | Main stage |
| 3 | Fig 03 — Questions and Objectives | Main stage |
| 4 | Fig 04 — Search Achievement | Main stage |
| 5 | Fig 06 — Gold Activation Hero | Main stage |
| — | Fig 05 — Plausibility Space | Supporting / optional |
| — | Fig 07 — Macro Heatmap | Appendix |
| — | Fig 08 — Portfolio Shift Detail | Appendix |

---

## What Was Dropped vs Prior Package

| Dropped figure | Reason |
|---|---|
| `slidegraphic_07` (MALA method) | Wrong engine; MALA is not the reset method |
| `scenario_story_overview` | Text-only card boxes, no data |
| `slidegraphic_09` (Gold old) | Infographic colored boxes, rainbow palette |
| `finalgfx_05` (Gold reset) | Anchor bars nearly identical; cramped macro panels |
| `slidegraphic_10` (Barbell) | Uses degenerate Q2 results — not honest to present |
| `finalgfx_06 / slidegraphic_11` (Ceiling) | Text box right panel, cluttered gap annotations |
| `finalgfx_01` (Big picture) | References MALA; wrong framing |
| Any Q2 (EW departure) figure | Probe failed — all candidates identical; not honest |
