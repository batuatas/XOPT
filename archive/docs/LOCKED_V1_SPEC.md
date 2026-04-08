# XOPTPOE Locked V1 Specification

## Status

This document is the **authoritative locked v1 design specification** for the XOPTPOE data-build repository.

It is the final design-level handoff from dataset planning to implementation.

Unless there is a true implementation contradiction, the coding agent must **implement this spec without silently redesigning it**.

For lower-level manifests, schema details, feature dictionaries, and coding guidance, see:

- `docs/IMPLEMENTATION_HANDOFF.md`

If there is any conflict between files, this document is the **primary source of truth** for locked v1 design decisions.

---

## 1. Project objective

XOPTPOE is the industry-oriented implementation of the **Predict–Optimize–Explain** framework.

The academic version operates at the firm level in a tactical allocation setting.  
This repository instead builds the **minimum viable v1 monthly sleeve-level dataset** for a strategic-allocation setting.

The purpose of this repository is to produce a **clean, reproducible, implementation-ready monthly panel** that can later support:

- monthly return prediction
- downstream long-only portfolio optimization
- future POE-style macro-state interrogation and scenario generation

This repository is for the **data layer only**.

It does **not** implement:

- forecasting models
- portfolio optimization
- POE sampling / MALA scenario generation
- institutional benchmark comparison layers

---

## 2. Locked v1 design principles

The following principles are locked.

### 2.1 Investable sleeves
Targets must correspond to **investable sleeve proxies**, not abstract research-only benchmark objects.

### 2.2 USD base currency, unhedged
Target returns are modeled in **USD** and are **unhedged**.

### 2.3 Monthly frequency
The final modeling table is a **monthly panel indexed at calendar month-end**.

### 2.4 Long-only downstream intent
The downstream optimization use case is:

- long-only
- fully invested
- no leverage

This does not affect target construction directly, but it informs the v1 dataset scope.

### 2.5 Implementation realism
Prefer data choices that are:

- reproducible
- accessible
- coding-friendly
- realistic for a first working build

over more elegant but fragile alternatives.

### 2.6 POE compatibility
The macro-state design must remain suitable for later **decision-pipeline explanation** and **macro-condition generation**.

That means the dataset should preserve a clear distinction between:

- economically interpretable state variables
- derived technical or descriptive features

---

## 3. Locked v1 sleeve universe

The locked v1 sleeve universe contains exactly **8 sleeves**.

| sleeve_id | ticker | sleeve_name | economic_definition | exposure_region | target_currency |
|---|---|---|---|---|---|
| EQ_US | VTI | US equities | Broad US total stock market exposure | US | USD |
| EQ_EZ | EZU | Euro area equities | Developed-market eurozone large/mid-cap equities | EURO_AREA | USD |
| EQ_JP | EWJ | Japan equities | Broad Japanese equity exposure | JAPAN | USD |
| EQ_EM | VWO | EM equities | Broad emerging-market equity exposure | EM_GLOBAL | USD |
| FI_UST | IEF | US Treasuries 7–10Y | Intermediate US Treasury exposure | US | USD |
| FI_IG | LQD | US investment-grade credit | USD investment-grade corporate bond exposure | US | USD |
| ALT_GLD | GLD | Gold | Gold-bullion-linked exposure | GLOBAL | USD |
| RE_US | VNQ | US REITs | US listed real-estate exposure | US | USD |

### 3.1 Locked sleeve decisions

The following are explicit design locks:

- use **EZU**, not generic Europe or developed-Europe substitutes
- use **VWO** for EM exposure
- do **not** include standalone China in locked v1
- use **USD-listed ETFs** for all sleeve targets
- treat all v1 sleeves as **investable proxies**
- all locked v1 sleeves carry `proxy_flag = 1`

### 3.2 Explicitly excluded from locked v1

The following are not part of locked v1:

- standalone China equities
- broad commodities ex-gold
- non-US government bond sleeves
- high-yield credit sleeve
- benchmark-index replacement of ETF sleeves
- institutional benchmark metadata layer
- hedged return variants
- explicit FX-converted local-index target construction

---

## 4. Locked target construction

## 4.1 Target source rule

For all 8 sleeves, the locked v1 target source is:

- **USD-denominated ETF adjusted-close history**

This is the uniform v1 rule.

The target layer is intentionally standardized across sleeves rather than mixing ETF proxies and benchmark-index targets.

## 4.2 Month-end rule

For each sleeve and month:

- pull daily adjusted-close history
- define `price_t` as the **last available trading-day adjusted close in calendar month t**
- define `price_t1` as the last available trading-day adjusted close in calendar month t+1

Literal calendar-end should **not** be used when markets are closed.

## 4.3 Sleeve return formula

For sleeve `a`:

\[
R_{a,t+1} = \frac{P_{a,t+1}}{P_{a,t}} - 1
\]

This is the locked v1 monthly sleeve return.

## 4.4 Risk-free rule

Use **TB3MS** as the monthly risk-free approximation source.

\[
rf^{(1m)}_t = \frac{TB3MS_t}{1200}
\]

This is intentionally simple and reproducible for v1.

## 4.5 Excess-return target

The locked v1 supervised-learning label is:

\[
y_{a,t+1} = R_{a,t+1} - rf^{(1m)}_{t+1}
\]

## 4.6 FX rule

There is **no separate FX conversion layer** in v1 target construction.

Currency effects are embedded in the USD-denominated, unhedged ETF return series.

---

## 5. Locked macro backbone

The macro backbone combines:

- local / regional blocks
- a global / stress block

The local blocks are used only where economically and structurally justified.

## 5.1 US local block

| series_id | preferred_code | meaning |
|---|---|---|
| US_CPI | CPIAUCSL | headline CPI level |
| US_UNEMP | UNRATE | unemployment rate |
| US_RF3M | TB3MS | 3M short-rate state variable and rf source |
| US_10Y | DGS10 | 10Y Treasury yield |

## 5.2 Euro area local block

| series_id | preferred_code | meaning |
|---|---|---|
| EA_CPI | CP0000EZ19M086NEST | euro-area HICP |
| EA_UNEMP | LRHUTTTTEZM156S | euro-area harmonized unemployment |
| EA_3M | IR3TIB01EZM156N | euro-area 3M interbank rate |
| EA_10Y | IRLTLT01EZM156N | euro-area 10Y government yield |

## 5.3 Japan local block

| series_id | preferred_code | meaning |
|---|---|---|
| JP_CPI | JPNCPIALLMINMEI | Japan CPI |
| JP_UNEMP | LRHUTTTTJPM156S | Japan unemployment |
| JP_3M | IR3TIB01JPM156N | Japan 3M interbank rate |
| JP_10Y | IRLTLT01JPM156N | Japan 10Y government yield |

## 5.4 Global / stress block

| series_id | preferred_code | meaning |
|---|---|---|
| USD_BROAD | DTWEXBGS | broad USD strength |
| VIX | VIXCLS | equity-volatility / risk stress |
| US_REAL10Y | DFII10 | US 10Y real yield |
| IG_OAS | BAMLC0A0CM | US IG corporate option-adjusted spread |
| OIL_WTI | DCOILWTICO | oil / commodity shock proxy |

---

## 6. Locked asset-to-macro mapping

The macro mapping is sleeve-specific and intentionally asymmetric.

| sleeve_id | local_block | global_block | mapping rule |
|---|---|---|---|
| EQ_US | US | GLOBAL | use US local + global |
| EQ_EZ | EURO_AREA | GLOBAL | use euro area local + global |
| EQ_JP | JAPAN | GLOBAL | use Japan local + global |
| EQ_EM | none | GLOBAL | use global/USD/risk only |
| FI_UST | US | GLOBAL | US rates block is primary |
| FI_IG | US | GLOBAL | US rates + credit-stress important |
| ALT_GLD | none | GLOBAL | no local block; use USD/real-rate/stress |
| RE_US | US | GLOBAL | US rates + stress important |

## 6.1 Locked EM rule

`EQ_EM` must **not** receive a dedicated EM local macro block in v1.

Its effective state mapping comes from:

- USD strength
- global stress
- oil / commodity shock
- real-rate and credit-stress conditions

This is a deliberate design choice, not an omission.

## 6.2 Locked gold rule

`ALT_GLD` must **not** receive a local macro block.

Gold is modeled using:

- USD block
- real-rate block
- global stress
- oil / commodity context

---

## 7. Locked timing and lag policy

## 7.1 Row date

Each modeling row is indexed by **calendar month-end t**.

## 7.2 Target horizon

The target on row `t` is the realized sleeve excess return from month-end `t` to month-end `t+1`.

## 7.3 Market-observable data at row t

The following may enter row `t` using information observable through the **last available trading day in month t**:

- sleeve adjusted closes
- DGS10
- DTWEXBGS
- VIXCLS
- DFII10
- BAMLC0A0CM
- DCOILWTICO

No extra lag is required for those market-observable series.

## 7.4 Official monthly macro at row t

For official monthly macro series, row `t` must use the value for **observation month t-1**.

This is the locked conservative v1 lag rule.

The implementation must not replace this with a hand-built release-calendar approach unless explicitly instructed in a later version.

## 7.5 Quarterly variables

Quarterly variables are not part of locked v1 core.

If introduced later, they must be carried forward only after a conservative release lag.

## 7.6 Revisions and vintages

v1 uses **latest revised values**.

It is **not vintage-pure**.

## 7.7 Look-ahead prevention

The implementation must guarantee:

- no current-month official macro leakage
- no forward-return leakage into features
- no use of future month-end prices in lagged technical variables
- no silent use of wrong-month risk-free subtraction

---

## 8. Locked feature families

This section defines the design-level feature families.  
Exact column manifests and lower-level details belong in `docs/IMPLEMENTATION_HANDOFF.md`.

For controlled schema consistency, canonical macro state variables are geo-prefixed in `macro_state_panel`; sleeve-level `local_*` names in `feature_panel` are derived compatibility aliases.

## 8.1 Asset technical features

The v1 dataset must include a core technical block based on sleeve return history.

Must-have:

- 1-month lagged return
- 3-month lagged return
- 6-month lagged return
- 12-month lagged return
- 12-1 momentum
- trailing 3-month realized volatility
- trailing 12-month realized volatility
- trailing 12-month max drawdown

These are **derived** and not directly scenario-manipulable.

## 8.2 Rate / carry-sensitive features

The v1 dataset must include mapped rate and carry-sensitive state variables.

Must-have:

- local short rate
- local long rate
- local term slope
- US 10Y real yield
- IG OAS

## 8.3 Macro level features

Must-have:

- mapped local inflation
- mapped local unemployment
- broad USD level
- VIX level
- oil level

## 8.4 Macro change features

Must-have:

- 1-month change in local inflation
- 1-month change in local unemployment
- 1-month change in local short rate
- 1-month change in local long rate
- 1-month change in local term slope
- 1-month USD log change
- 1-month VIX change
- 1-month real-yield change
- 1-month IG OAS change
- 1-month oil log change

Nice-to-have:

- 12-month USD log change
- 12-month oil log change

## 8.5 Cross-asset relative features

Must-have:

- momentum relative to Treasury sleeve
- momentum relative to US equity sleeve

Nice-to-have:

- 1-month lagged return relative to Treasury sleeve
- 1-month lagged return relative to US equity sleeve

## 8.6 Quality / control flags

Must-have:

- proxy flag
- feature completeness flag
- macro staleness flag
- sample inclusion flag
- lag-policy tag

---

## 9. Scenario-manipulable vs derived variables

For later POE work, only a subset of v1 variables should be treated as directly manipulable state variables.

### 9.1 Directly scenario-manipulable
These are part of the economic state representation:

- geo-prefixed inflation state (`infl_US`, `infl_EA`, `infl_JP`)
- geo-prefixed unemployment state (`unemp_US`, `unemp_EA`, `unemp_JP`)
- geo-prefixed short-rate state (`short_rate_US`, `short_rate_EA`, `short_rate_JP`)
- geo-prefixed long-rate state (`long_rate_US`, `long_rate_EA`, `long_rate_JP`)
- geo-prefixed term-slope state (`term_slope_US`, `term_slope_EA`, `term_slope_JP`)
- broad USD index (`usd_broad`)
- VIX (`vix`)
- US real 10Y yield (`us_real10y`)
- IG OAS (`ig_oas`)
- oil WTI (`oil_wti`)

### 9.2 Derived or descriptive only
These should not be directly perturbed in the scenario engine:

- momentum variables
- rolling volatility variables
- drawdown variables
- relative-momentum features
- interaction features
- feature standardizations / z-scores
- technical transforms of raw state variables

---

## 10. Locked table architecture

The repository must produce the following tables:

- `asset_master`
- `source_master`
- `sleeve_target_raw`
- `macro_raw`
- `macro_state_panel`
- `feature_panel`
- `target_panel`
- `modeling_panel`
- `macro_mapping`
- `coverage_audit`

## 10.1 Preferred `macro_state_panel` design

To make POE state semantics explicit, the canonical implementation choice is:

- `macro_state_panel` stores **one wide row per month_end**
- canonical local geo-prefixed state variables are:
  - `infl_US`, `unemp_US`, `short_rate_US`, `long_rate_US`, `term_slope_US`
  - `infl_EA`, `unemp_EA`, `short_rate_EA`, `long_rate_EA`, `term_slope_EA`
  - `infl_JP`, `unemp_JP`, `short_rate_JP`, `long_rate_JP`, `term_slope_JP`
- canonical global/stress variables are:
  - `usd_broad`, `vix`, `us_real10y`, `ig_oas`, `oil_wti`
- sleeve-level mapped `local_*` names may remain in `feature_panel` as **derived compatibility aliases**, not canonical macro-state source variables
- no dedicated EM local block is introduced (`EQ_EM` remains global/USD/risk-driven only)
- `ALT_GLD` continues to have no local macro block

All locked v1 sleeves, targets, target construction, and timing/lag rules remain unchanged.

---

## 11. Locked build order

The build should proceed in this order:

1. create manifests and schema validators
2. create `asset_master`
3. fetch `sleeve_target_raw`
4. validate target coverage
5. collapse target prices to month-end
6. compute monthly sleeve returns
7. fetch TB3MS and construct monthly rf
8. fetch `macro_raw`
9. apply lag rules
10. build `macro_state_panel`
11. build `feature_panel`
12. build `target_panel`
13. build `macro_mapping`
14. join `modeling_panel`
15. run `coverage_audit`
16. freeze v1 outputs

## 11.1 First QA gate

The first mandatory QA gate is the **target layer**.

Implementation must validate:

- all 8 tickers are present
- no duplicate trade dates within sleeve
- adjusted close is available on used month-end proxy dates
- month-end collapse uses last trading day
- no impossible returns
- usable history exists from the intended start window

If this fails, the build must stop and resolve the target layer before proceeding.

---

## 12. Locked QA requirements

QA must be machine-checkable.

### 12.1 Target QA
Required checks:

- duplicate primary keys
- missing adjusted close
- non-positive prices
- invalid month-end collapse
- impossible returns
- silent ticker substitution

### 12.2 Macro QA
Required checks:

- duplicate primary keys
- manifest/source mismatch
- incorrect native-frequency handling
- lag-policy violations
- stale carried-forward values flagged
- no silent code substitution

### 12.3 Join QA
Required checks:

- duplicate modeling rows
- missing macro merges
- missing target merges
- inconsistent sleeve coverage
- actual sample start and end dates by sleeve
- missingness summaries by feature

---

## 13. Non-goals for locked v1

Locked v1 does **not** include:

- standalone China sleeve
- broad commodity sleeve ex-gold
- non-US bond sleeves
- benchmark-index replacement of ETF targets
- explicit FX conversion layer
- institutional benchmark metadata
- vintage-aware macro architecture
- forecasting models
- optimizer implementation
- POE sampler implementation

---

## 14. Future extensions

Possible later extensions include:

- Shiller US valuation block
- BIS leverage / credit-to-GDP variables
- benchmark-index replacement of selected ETF sleeves
- standalone China sleeve
- broader commodity sleeves
- vintage-aware macro architecture
- institutional benchmark / forecast layer
- direct POE macro-state generation layer

These are intentionally outside locked v1.

---

## 15. Implementation discipline rule

If a source, download adapter, or transformation step is fragile, the implementation must:

- fail loudly, or
- use an explicitly documented fallback

The implementation must **not silently substitute**:

- different sleeves
- different tickers
- different macro series codes
- different lag rules
- different macro mappings
- different target definitions

This file is the source of truth for locked v1.
