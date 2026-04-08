# XOPTPOE v3 Basic Project Info

This note is a short summary of the latest active project version:

- active branch: `v3_long_horizon_china`
- active scenario object: locked robust 5Y benchmark
- current use case: long-horizon SAA, not short-horizon TAA

## 1. Asset Classes

The active investable universe has 9 sleeves:

- `EQ_US`: US equities
- `EQ_EZ`: Euro Area equities
- `EQ_JP`: Japan equities
- `EQ_CN`: China equities
- `EQ_EM`: Emerging Market equities
- `FI_UST`: US Treasuries 7-10Y
- `FI_IG`: Investment Grade credit
- `ALT_GLD`: Gold
- `RE_US`: US REITs

Practical grouping:

- Equities: `EQ_US`, `EQ_EZ`, `EQ_JP`, `EQ_CN`, `EQ_EM`
- Fixed income: `FI_UST`, `FI_IG`
- Real assets / alternatives: `ALT_GLD`, `RE_US`

## 2. Time Horizon

The project is built around long-horizon return targets:

- `60m` = 5-year horizon
- `120m` = 10-year horizon

For the current conference story, the main active benchmark is:

- locked robust 5Y benchmark = `best_60_predictor`

So the scenario layer is centered on the 5-year benchmark as one benchmark object.

## 3. Variables Inspected In Scenario Generation

The first-pass manipulated scenario state contains 17 macro / market variables:

- US: `infl_US`, `unemp_US`, `short_rate_US`, `long_rate_US`
- Euro Area: `infl_EA`, `unemp_EA`, `short_rate_EA`, `long_rate_EA`
- Japan: `infl_JP`, `unemp_JP`, `short_rate_JP`, `long_rate_JP`
- Global / financial conditions: `usd_broad`, `vix`, `us_real10y`, `ig_oas`, `oil_wti`

These are the main scenario levers.

The scenario layer also keeps some enrichment variables fixed at the anchor date for context rather than directly manipulating them. Examples:

- `china_cli`
- `jp_pe_ratio`
- `cape_local`
- `cape_usa`
- `oecd_activity_proxy_local`
- `mom_12_1`
- `vol_12m`
- `em_minus_global_pe`
- `rel_mom_vs_treasury`

So the scenario engine is intentionally interpretable:

- manipulate a small macro state
- hold richer cross-sectional context fixed
- observe how the benchmark prediction and allocation respond

## 4. How The Regime Classifier Works

The current regime layer is a simple, interpretable hybrid classifier.

### External anchors

- `NFCI` is used for the financial-conditions / stress backdrop
- NBER recession dates are used as a historical recession overlay

NFCI handling:

- source: `NFCI (1).csv`
- alignment rule: last available observation in each calendar month, mapped to month-end
- bucketed into: `loose / neutral / tight`

Recession overlay:

- source: `Recessiondating.md`
- used only as a historical overlay: `recession / non-recession`

### Internal macro dimensions

The scenario state is summarized into four dimensions:

- Growth
- Inflation
- Stress
- Rates / conditions

The broad logic is:

- Growth: mainly from unemployment across US / Euro Area / Japan
- Inflation: mainly from regional inflation plus oil
- Stress: mainly from `vix`, `ig_oas`, `usd_broad`
- Rates: mainly from short rates, long rates, and `us_real10y`

Each dimension is bucketed into simple labels such as:

- `low / neutral / high`
- or for rates: `easy / neutral / tight`

### Final regime labels

These dimensions are combined into compact labels such as:

- `soft landing`
- `risk-on reflation`
- `higher-for-longer tightness`
- `high-stress defensive`
- `disinflationary slowdown`
- `recessionary stress`
- `risk-on growth`
- `mixed mid-cycle`

Important point:

- this is not a black-box classifier
- it is a transparent macro scorecard built for interpretation

## 5. Possible Scenario-Generation Questions

The strongest conference-style questions for the current benchmark are:

- What does a `soft landing` regime look like for the locked benchmark?
- What does a `higher-for-longer` regime look like for the locked benchmark?
- What regime makes `gold` more attractive inside the benchmark?
- What regime improves expected long-run return without making the benchmark much more concentrated?

Secondary / appendix-style questions:

- What regime makes `US equities` more attractive?
- What regime makes `EM equities` more attractive?
- What regime would justify a `6%` strategic return assumption?
- What regime would justify a `7%` strategic return assumption?
- What regime would justify a `10%` long-run return assumption?

Current practical note:

- the `6% / 7% / 10%` house-view questions are weaker in the current run because the benchmark already starts from relatively high predicted returns at the selected anchors

## 6. Current Conference Cases

The current scenario presentation is strongest when it is framed around three representative cases:

1. `Upside case`
- source question: `Soft landing`
- message: return can improve without destabilizing the benchmark

2. `Breadth case`
- source question: `Return with breadth`
- message: diversification can improve, but not for free

3. `Adverse case`
- source question: `Higher-for-longer`
- message: tighter conditions weaken the outlook and keep the benchmark more defensive

## 7. One-Line Project Description

`XOPTPOE v3` is a long-horizon, 9-sleeve strategic allocation framework that uses transparent macro-state scenario generation to ask what plausible regimes support the current robust 5Y benchmark’s return, concentration, and allocation behavior.
