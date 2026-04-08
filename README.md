# XOPTPOE v1 Data Build

## Project purpose

This repository builds the **minimum viable v1 monthly dataset** for the industry translation of the **Predict–Optimize–Explain (POE)** framework.

The academic paper works with firm-level tactical allocation and pipeline-level explanation. This repository instead builds a **cross-asset, sleeve-level, strategic-allocation panel** that can support:

- monthly return prediction,
- downstream long-only portfolio optimization,
- and later POE-style macro-state interrogation / scenario generation.

The goal is not to build the full modeling or optimization stack here. The goal is to produce a **clean, reproducible, implementation-ready monthly asset-month dataset**.

## Locked v1 scope

This repo implements the following locked design choices:

- **8 sleeves**
- **USD-denominated ETF adjusted-close targets**
- **monthly month-end panel**
- **target = next-month excess return**
- **risk-free = TB3MS-based monthly approximation**
- **no separate FX conversion layer in v1 target construction**
- **local macro blocks = US, Euro area, Japan**
- **EM uses global/USD/risk mapping, not a separate EM macro block**
- **global/stress block = DTWEXBGS, VIXCLS, DFII10, BAMLC0A0CM, DCOILWTICO**
- **official monthly macro lagged by one observation month**
- **latest revised data, not vintage-pure**
- **downstream optimization intent = long-only, fully invested, no leverage**
- **POE compatibility preserved**

## Sleeve universe

The locked v1 sleeves are:

- **VTI** — US equities
- **EZU** — Euro area equities
- **EWJ** — Japan equities
- **VWO** — Emerging market equities
- **IEF** — US 7–10Y Treasuries
- **LQD** — US investment-grade corporate credit
- **GLD** — Gold
- **VNQ** — US REITs

These are treated as **investable sleeve proxies**. The target layer is intentionally uniform: every sleeve is built from a USD-listed ETF using adjusted-close price history.

## Target definition

For sleeve \(a\) and month-end \(t\):

- let \(P_{a,t}\) be the **last available adjusted close in calendar month \(t\)**
- compute sleeve return as

\[
R_{a,t+1} = \frac{P_{a,t+1}}{P_{a,t}} - 1
\]

- define the monthly risk-free approximation as

\[
rf^{(1m)}_t = \frac{\text{TB3MS}_t}{1200}
\]

- define the prediction target as

\[
y_{a,t+1} = R_{a,t+1} - rf^{(1m)}_{t+1}
\]

This is the locked v1 excess-return label.

## Macro backbone

### Local / regional blocks
- **US**: CPIAUCSL, UNRATE, TB3MS, DGS10
- **Euro area**: CP0000EZ19M086NEST, LRHUTTTTEZM156S, IR3TIB01EZM156N, IRLTLT01EZM156N
- **Japan**: JPNCPIALLMINMEI, LRHUTTTTJPM156S, IR3TIB01JPM156N, IRLTLT01JPM156N

### Global / stress block
- **DTWEXBGS** — Nominal Broad U.S. Dollar Index
- **VIXCLS** — VIX
- **DFII10** — 10Y real Treasury yield
- **BAMLC0A0CM** — ICE BofA US Corporate OAS
- **DCOILWTICO** — WTI crude oil

## Timing and lag policy

### Row date
Each modeling row is indexed by **calendar month-end \(t\)**.

### Market data allowed at row \(t\)
Allowed through the **last available trading day in month \(t\)**:
- sleeve adjusted closes,
- VIXCLS,
- DTWEXBGS,
- DFII10,
- BAMLC0A0CM,
- DCOILWTICO,
- DGS10.

### Official monthly macro allowed at row \(t\)
Use the value for **observation month \(t-1\)**.

This is the locked v1 conservative lag rule. It avoids hand-building country-specific release calendars.

### Quarterly variables
Quarterly variables are not part of the locked core. If introduced later, they must be carried forward only after a conservative release lag.

### Revisions
v1 uses **latest revised values**, not vintage-pure data.

## Table architecture

The repository should produce the following tables:

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

## Build order

1. lock `asset_master`
2. fetch raw sleeve histories
3. validate sleeve coverage and month-end collapses
4. fetch TB3MS and build risk-free series
5. fetch macro backbone
6. apply lag policy
7. build `macro_state_panel`
8. build sleeve features
9. build `target_panel`
10. join `modeling_panel`
11. run coverage / leakage / duplicate-key QA
12. freeze v1

## Quality checks

The build must include checks for:

- duplicate primary keys
- missing sleeve histories
- invalid month-end collapse
- missing adjusted-close values at used month-ends
- impossible returns (e.g. \(R \le -1\))
- macro-series staleness
- lag-policy violations
- accidental use of current-month official macro
- inconsistent sample start handling across sleeves
- silent ticker substitution

## Explicit non-goals for v1

This repo does **not** do the following in locked v1:

- no standalone China sleeve
- no commodities ex-gold sleeve
- no institution benchmark metadata layer
- no separate FX conversion layer
- no vintage-pure macro architecture
- no benchmark-index replacement for ETF sleeves
- no optimization engine
- no POE sampler yet

## Future extensions

Planned future extensions may include:

- Shiller US valuation block
- BIS leverage / credit-to-GDP block
- benchmark-index replacement for selected ETF sleeves
- standalone China sleeve
- broader commodity sleeve
- vintage-aware macro states
- institution forecast / benchmark metadata
- direct POE macro-state generation layer

## Repository implementation

Current implementation files are organized as:

- `config/` locked manifests and seed tables
- `schemas/` table specs and validation rules
- `src/xoptpoe_data/` modular pipeline code
- `scripts/run_build.py` full data build entrypoint
- `scripts/run_qa.py` QA-only re-run entrypoint

### Install dependencies

```bash
python3 -m pip install pandas numpy yfinance pandas-datareader
```

### Run full build

```bash
python3 scripts/run_build.py --end-date 2026-03-01
```

### Run QA only

```bash
python3 scripts/run_qa.py
```

### Output paths

Build outputs are written to:

- `data/raw/`
- `data/intermediate/`
- `data/final/`
- `reports/`

### Note on macro-state layout

The implementation uses a canonical **geo-prefixed wide** `macro_state_panel` with monthly rows and explicit state names, for example:
- `infl_US`, `unemp_EA`, `short_rate_JP`, `term_slope_US`
- `usd_broad`, `vix`, `us_real10y`, `ig_oas`, `oil_wti`

For backward compatibility and sleeve modeling convenience:
- mapped alias fields (`local_*`, `*_level`) are derived from canonical macro state
- `global_state_panel` is retained as a compatibility intermediate view

Locked v1 restrictions are unchanged:
- no EM dedicated local macro block
- no local block for `ALT_GLD`
