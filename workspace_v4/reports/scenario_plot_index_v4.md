# Scenario Plot Index — XOPTPOE v4

Generated: 2026-03-30

## Output location
`reports/scenario_plots/` — all figures in `.png` (300 dpi) and `.pdf` (vector)

---

## Figure 1 — Story Overview

**File:** `scenario_story_overview_v4.png / .pdf`

**What it shows:**
Three-column opener introducing the three scenario questions side by side. Each column shows the anchor regime, generated scenario regime (colored chips + arrow), four key data bullets, and a one-line italic takeaway. Designed as a section divider slide — not data-dense.

**Deck placement:** First slide of scenario section, before any individual story.

**Speaker message:** "We ran three focused questions through the scenario engine. Each question has a different macro driver story — I'll walk through them one by one."

---

## Figure 2 — Gold Activation Threshold *** STRONGEST SINGLE SLIDE ***

**File:** `scenario_gold_activation_v4.png / .pdf`

**What it shows:**
Four-panel (2x2) headline figure.
- Panel A: Regime transition chips — reflation -> higher-for-longer -> risk-off
- Panel B: Gold allocation bars across all four anchors (anchor baseline vs scenario mean Q1), with annotated 3x jump
- Panel C: US real 10-year yield at each anchor date — negative in 2021, positive from 2022, with zero-threshold shading
- Panel D: Stacked portfolio bar for Dec-2021 vs Dec-2022 (EQ US, FI UST, ALT GLD, CR US HY, Other)

**Deck placement:** First main story slide — leads the scenario section narrative.

**Speaker message:** "The model's gold allocation is dormant when real yields are negative. Once us_real10y crosses zero in 2022, gold triples from 8% to 22%. It's a threshold, not a gradual tilt."

---

## Figure 3 — Defensive Barbell Under Stress Regimes

**File:** `scenario_equal_weight_deviation_v4.png / .pdf`

**What it shows:**
Two-panel figure.
- Panel A: Dot plot comparing equal-weight (1/14), Dec-2022 anchor baseline, and Q2 scenario mean for 8 key sleeves, sorted by scenario weight. Barbell sleeves (FI UST, ALT GLD, CR US HY) highlighted in amber band. Total barbell weight annotated.
- Panel B: Regime context card — risk-off chip, four dimension labels (stress/policy/fin.cond./growth) as colored categorical chips, plus four macro state variables (ig_oas, us_real10y, VIX, short rate).

**Deck placement:** Second main story slide.

**Speaker message:** "When the model moves away from equal weight, it doesn't tilt gradually — it concentrates into a three-sleeve defensive barbell. FI UST, gold, and HY credit collectively take 60%+ of the portfolio, and equity compresses below 11%."

---

## Figure 4 — The Return Ceiling

**File:** `scenario_house_view_gap_v4.png / .pdf`

**What it shows:**
Two-panel figure.
- Panel A: Bar chart per anchor date showing anchor baseline return, scenario mean, and scenario max, with the 5% house-view target as a red dashed line. Gap between scenario max and target annotated with double-headed arrows. Scenario range shown as a shaded band.
- Panel B: Best-case scenario detail card — regime chip, five macro state variables for the best single scenario (Dec 2022, pred_ret=3.08%), and gap summary table (all four anchors).

**Deck placement:** Third main story slide — closes the scenario section.

**Speaker message:** "Even in the most favorable locally plausible macro state — positive real yields, moderate spreads, higher-for-longer in 2022 — the model tops out at 3.08%. The 5% house-view assumption is structurally out of reach. That's a model signal, not a calibration failure."

---

## Summary

| Fig | File stem | Strongest for | Required |
|-----|-----------|--------------|---------|
| 1 | scenario_story_overview_v4 | Section opener / orientation | Optional in short deck |
| 2 | scenario_gold_activation_v4 | Headline story — use always | Yes |
| 3 | scenario_equal_weight_deviation_v4 | Allocation logic story | Yes |
| 4 | scenario_house_view_gap_v4 | Risk/constraint framing | Yes |

**Optional 5th figure (regime legend):** Not built. The three main story figures are self-contained. The regime chips in each figure carry sufficient labeling — a separate legend would add redundancy without clarity.

**Strongest single slide:** Figure 2 (Gold Activation). Most visual, most concrete, best headline message.

**Package consistency:** All four figures share identical rcParams, color palette, regime chip style, and gridspec margins. Safe for direct insertion into a 16:9 deck without resizing.
