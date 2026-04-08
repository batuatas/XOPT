# Scenario Explainer Design Brief — v4

**Date**: 2026-03-30

---

## Critique of Existing Graphics

### finalgfx_01 (Big Picture)
**Weak**: Clean opener but says "MALA explores plausible macro neighborhoods" — wrong framing for reset. The stat row (+37%) refers to a wealth path number that doesn't appear in that figure. Feels like a letterhead, not an argument.

### finalgfx_05 (Gold Activation — reset version)
**Weak in two ways**:
1. Left bar chart is good but the three right panels (US Real 10Y, Short Rate, IG OAS) are crammed into a narrow column with tiny labels — unreadable from a projected screen.
2. The orange arrow annotation and "3×" label look handmade / infographic-ish.
3. The gold scenario bars are nearly identical to anchor bars at 2022–2024 (22% vs 22.5%) — the "activation" story is barely visible for three of four anchors.
4. Caption text too long and small.

### finalgfx_06 (Return Ceiling — reset version)
**Weak**:
1. Left panel: scatter of blue dots (scenario range) around grey anchor bars — the dots are tiny and indistinguishable.
2. Right panel: a plain text box listing macro values — no visual structure, reads like a footnote.
3. The red dashed "5% target" line is dominant but draws attention to the wrong thing (what we can't achieve) without communicating *why*.
4. The gap annotation numbers (−0.70pp, −1.90pp etc.) are placed on the chart in a way that looks like error messages.

### slidegraphic_07 (Scenario Method — MALA version)
**Critically wrong framing**:
1. Central pipeline uses five colored boxes including a purple "MALA/GS" box as the main method — this is the wrong engine.
2. "MALA Search" section below with MH accept/reject formula — should not appear in any main-stage figure.
3. The "Search Trajectory" panel shows a Markov chain ellipse — explicitly illustrates MALA, not the reset method.
4. Rainbow colored pipeline boxes (navy, purple, green, orange, red) — violates the restrained palette.
5. Too many panels, too much text — this figure tries to explain everything.

### slidegraphic_09 (Gold Activation — old version)
**Better composition than reset version** but:
1. Panel A (regime transition) uses bright green / orange / red colored boxes — feels like a traffic light infographic.
2. Panel D (portfolio composition 2021 vs 2022) uses many colors including purple and teal — rainbow feel.
3. Panel labels A/B/C/D with different widths create uneven layout.
4. The regime transition arrow flow reads like a management consulting deck.

### slidegraphic_10 (Defensive Barbell)
**Mixed**:
1. Left dot-plot panel is actually good — clean and readable.
2. Right panel uses large colored regime boxes (red "Risk-Off/Stress", coloured dimension tags) — very infographic, not editorial.
3. The scenario being shown (Dec 2022 Q2) is from the *failed* EW probe — the results are degenerate (same portfolio repeated). Cannot be used honestly.

### slidegraphic_11 (House View Gap)
**Structurally good but**:
1. Right panel is a plain orange-bordered text box — looks like a callout box from a Word document.
2. The gap numbers in red on the left chart are cluttered.
3. The best-case macro state table has no visual hierarchy.

### scenario_story_overview (Three Questions)
**Worst figure in the package**:
1. Three colored card boxes (orange, grey, red) — generic presentation template feel.
2. Bullet-point lists inside boxes — this is a slide deck, not a graphic.
3. No data visible — pure text.
4. Cannot tell at a glance what any result actually is.

---

## What Will Be Replaced

| Old figure | Problem | Replacement |
|---|---|---|
| slidegraphic_07 (MALA method) | Wrong engine, rainbow colors, too complex | New fig 02: reset method as clean flow diagram |
| scenario_story_overview | Text-only card boxes | New fig 01: overview with actual data thumbnails |
| slidegraphic_09 (Gold, old) | Infographic boxes, rainbow | New fig 06: editorial gold + regime transition |
| finalgfx_05 (Gold, reset) | Anchor bars nearly identical, cramped macro panels | Replaced by fig 06 |
| slidegraphic_10 (Barbell) | Uses degenerate Q2 results | Dropped — Q2 failed |
| finalgfx_06 / slidegraphic_11 (ceiling) | Text box right panel, cluttered gaps | New fig 04 + fig 06 |
| finalgfx_01 (big picture) | MALA reference | New fig 01 removes MALA |

---

## What Will Be Intentionally Omitted

- **Any explicit MALA formula or MALA trajectory diagram** in main figures
- **Equal-weight departure (Q2)** — probe failed; degenerate results; not honest to present
- **Panel label letters (A/B/C/D)** — creates academic paper feel, not conference feel
- **Colored regime boxes** (red/green/orange as background fills) — too infographic
- **Bullet-point text blocks** inside figures
- **Rainbow-colored pipeline boxes**
- **Gap annotation numbers** scattered across bar charts
- **Text box callouts** as right-panel replacements for actual data visualization

---

## Visual Language for the New Package

**Layout**: 16:9, two-column or single-focus. No more than 2 panels per figure unless panels are tightly integrated.

**Color**: Ink (#1A1A1A) + one blue (#1F6FBF) + gold (#D4820A) only when referring to gold. Muted grey (#888) for secondary data. No red except for a single stress indicator line where essential. No green. No purple. No orange (except gold data).

**Typography**: Large, readable titles. One-line subtitles. No paragraph text inside figures. Axis labels minimal. No internal legend boxes — use direct annotation instead.

**Data-ink principle**: Every element either carries data or creates structure. No decorative borders, gradient fills, background panels, or drop shadows.

**Figure discipline**: One primary message per figure. If a figure needs a second panel, that panel must serve the primary message, not introduce a new topic.

**Annotation style**: Single clean leader line or single label directly on the data point. No annotation tables.

---

## Questions Kept for Main Package

| Question | Status | Reason |
|---|---|---|
| Q1: Gold activation threshold | **Main stage** | Clearest structural finding, strongest visual story |
| Q2: Equal-weight departure | **Dropped** | Probe failed; all results identical |
| Q3: Return with discipline | **Supporting** | Feeds into return ceiling; merge with Q4 |
| Q4: Return ceiling | **Main stage** | Sharp quantified finding, honest |
| Allocation sensitivity | **Main stage** | Which macro variables matter most — new from state-shift data |

**Effective 3 main questions**: Gold activation, Return ceiling, Macro sensitivity (allocation driver).
