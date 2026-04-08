# Final Slidegraphics Index — XOPTPOE v4

Generated: 2026-03-30
Output: `reports/final_slidegraphics_v4/`  |  All figures: 16:9, 300 dpi PNG + vector PDF

---

## Figure 01 — Big Picture Pipeline
**File:** `slidegraphic_01_big_picture_v4`
**Purpose:** Section opener. Shows the full pipeline at a glance: Data -> AI -> Allocation -> Benchmark -> Scenario.
**Storyline position:** Slide 1 of the deck (after title slide).
**Speaker message:** "Here is the full pipeline. Data and features feed an AI prediction layer; predictions feed a robust allocator; the resulting benchmark is then probed by a scenario engine."

---

## Figure 02 — Investable Universe
**File:** `slidegraphic_02_investable_universe_v4`
**Purpose:** Shows the 14 sleeves grouped by asset class with proxy instruments and geography.
**Storyline position:** Early — immediately after big picture, before modeling deep-dive.
**Speaker message:** "We cover 14 sleeves across five groups. The universe is deliberately broad: equity, rates, credit, real assets, and gold."

---

## Figure 03 — Data, Features, Targets
**File:** `slidegraphic_03_data_features_targets_v4`
**Purpose:** Explains the data design: macro features, interaction construction, 5Y target definition.
**Storyline position:** Data/modeling section.
**Speaker message:** "303 features — macro, rate, stress, momentum, and their interactions — feed into separate elastic net models, one per sleeve, predicting 5Y annualized excess return."

---

## Figure 04 — AI Prediction Layer
**File:** `slidegraphic_04_ai_prediction_layer_v4`
**Purpose:** Explains the prediction model. Left: architecture card. Right: out-of-sample prediction scatter (validation period, corr=0.76).
**Storyline position:** AI/prediction section.
**Speaker message:** "The model achieves a prediction correlation of 0.76 on held-out data. The scatter is noisy — this is intentional; we rely on robust optimization to handle uncertainty downstream."

---

## Figure 05 — Allocation Layer
**File:** `slidegraphic_05_allocation_layer_v4`
**Purpose:** Explains the robust MVO optimizer. Shows the objective formula, key design choices, and a compact allocation snapshot across anchor years.
**Storyline position:** After prediction section.
**Speaker message:** "Predicted returns are not the final output. They feed a robust optimizer that penalizes risk and excessive deviation from equal weight. The resulting allocations are stable and interpretable."

---

## Figure 06 — Hero Benchmark Behavior *** STRONGEST SINGLE SLIDE ***
**File:** `slidegraphic_06_hero_benchmark_behavior_v4`
**Purpose:** The hero empirical figure. Top panel: wealth path vs equal weight (2.04x vs 1.80x). Bottom panel: annual rebalance allocation composition.
**Storyline position:** Benchmark results section — centerpiece of the deck.
**Speaker message:** "The benchmark compounds to 2.04x over the decade vs 1.80x for equal weight. The allocation composition shows a clear regime shift after 2022: gold and credit grow materially, equity is compressed."

---

## Figure 07 — Scenario Method
**File:** `slidegraphic_07_scenario_method_v4`
**Purpose:** Explains the scenario generation method: pipeline, probe objective, VAR(1) prior, MALA search, regime labels.
**Storyline position:** Opens the scenario section.
**Speaker message:** "Scenario generation is not brute-force stress testing. We use a gradient-based sampler that searches over locally plausible macro states — constrained by a VAR(1) prior — to find states that answer specific questions about the benchmark."

---

## Figure 08 — Anchor Context
**File:** `slidegraphic_08_anchor_context_v4`
**Purpose:** Shows macro variable evolution 2007-2025 with regime-shaded periods and anchor date markers.
**Storyline position:** Contextualizes anchor dates before scenario results.
**Speaker message:** "Our four anchor dates — 2021 through 2024 — span a dramatic macro transition: from negative real rates and reflation to the highest short rates in 15 years."

---

## Figure 09 — Gold Activation Threshold
**File:** `slidegraphic_09_gold_activation_v4`
**Purpose:** Strongest scenario result. Shows regime transition, real yield threshold, 3x gold jump, and portfolio composition change.
**Storyline position:** First main scenario result.
**Speaker message:** "Gold allocation is dormant when real yields are negative. Once us_real10y crosses zero — as it did in 2022 — gold triples. The model learned this as a threshold, not a gradual relationship."

---

## Figure 10 — Defensive Barbell
**File:** `slidegraphic_10_equal_weight_deviation_v4`
**Purpose:** Shows when/why the benchmark deviates from equal weight. Dot plot comparison plus regime context.
**Storyline position:** Second main scenario result.
**Speaker message:** "In risk-off stress regimes, the optimizer concentrates into a defensive barbell: FI UST, gold, and HY credit together exceed 60%. Equity is crowded out systematically."

---

## Figure 11 — House-View Gap
**File:** `slidegraphic_11_house_view_gap_v4`
**Purpose:** Shows the structural return ceiling. Scenario search cannot close the gap to a 5% house-view return assumption.
**Storyline position:** Third (and final) scenario result — closes the deck.
**Speaker message:** "The model's ceiling is 3.08%, achieved at the 2022 anchor. No locally plausible macro state reaches 5%. This is a calibration signal — the model was trained in a post-GFC low-return world, and it reflects that honestly."

---

## Summary Table

| Fig | File | Deck | Priority |
|-----|------|------|----------|
| 01 | big_picture | Opener | Required |
| 02 | investable_universe | Context | Required |
| 03 | data_features | Methodology | Required |
| 04 | ai_prediction | Methodology | Required |
| 05 | allocation_layer | Methodology | Required |
| 06 | hero_benchmark | **Results — centerpiece** | **Must-have** |
| 07 | scenario_method | Scenario intro | Required |
| 08 | anchor_context | Scenario context | Recommended |
| 09 | gold_activation | **Scenario hero** | **Must-have** |
| 10 | equal_weight_deviation | Scenario result | Required |
| 11 | house_view_gap | Scenario close | Required |

**Strongest single figure:** Fig 06 (Hero Benchmark) or Fig 09 (Gold Activation)
**Optional/appendix:** Fig 08 (Anchor Context) can be moved to appendix in a short deck.
