# Data And Targets

## Data Branch Status

The accepted data branch is `v4_expanded_universe`.

It is a versioned v4 branch built in parallel to frozen earlier branches. It does not modify `v1`, `v2`, or `v3`.

## Active Sleeve Universe

### Full v4 data universe: 15 sleeves

- `EQ_US`
- `EQ_EZ`
- `EQ_JP`
- `EQ_CN`
- `EQ_EM`
- `FI_UST`
- `FI_EU_GOVT`
- `CR_US_IG`
- `CR_EU_IG`
- `CR_US_HY`
- `CR_EU_HY`
- `RE_US`
- `LISTED_RE`
- `LISTED_INFRA`
- `ALT_GLD`

### Default supervised benchmark roster: 14 sleeves

`CR_EU_HY` stays in the data branch but is excluded from the default supervised benchmark roster because the accepted split design leaves it with extremely weak train coverage.

That means the default supervised benchmark roster is:

- `EQ_US`
- `EQ_EZ`
- `EQ_JP`
- `EQ_CN`
- `EQ_EM`
- `FI_UST`
- `FI_EU_GOVT`
- `CR_US_IG`
- `CR_EU_IG`
- `CR_US_HY`
- `RE_US`
- `LISTED_RE`
- `LISTED_INFRA`
- `ALT_GLD`

## Taxonomy And Sleeve Definitions

- `CR_US_IG` replaces legacy `FI_IG` naming in v4
- `LISTED_RE` is locked to `ex-U.S. listed real estate`
- `RE_US` and `LISTED_RE` coexist
- `LISTED_INFRA` is global listed infrastructure

## Target Families

The long-horizon target framework keeps the established project structure:

- `60m`
- `120m`
- `180m`

For each horizon, the target family includes:

- annualized total forward return
- annualized forward risk-free return
- annualized forward excess return

The benchmark modeling focus is the annualized forward excess return target.

## Primary Benchmark Target

The main supervised benchmark target is:

- annualized 60-month excess return

This is the 5Y target used by the active benchmark predictor.

## Euro Fixed-Income Exception Rule

The following sleeves use a locked exception rule:

- `FI_EU_GOVT`
- `CR_EU_IG`
- `CR_EU_HY`

Their target rule is:

- local-currency investable ETF return plus month-end EUR/USD conversion to USD

More explicitly:

1. calculate monthly ETF return in local currency from adjusted close or equivalent total-return market series
2. convert that monthly return into unhedged USD investor experience with public month-end FX
3. use the resulting synthesized USD-unhedged monthly series as the sleeve return history

This exception is locked only for the euro fixed-income family.

## Feature Groups

The feature layer combines:

- global macro and market-state features
- regional macro blocks
- sleeve-linked valuation and technical features
- selected interaction terms

The frozen macro backbone starts in 2006. That matters because some sleeve target histories start earlier than the feature history.

## Main Data Artifacts

The most important copied artifacts in this handoff workspace are:

- [asset_master.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/asset_master.csv)
- [target_series_manifest.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/target_series_manifest.csv)
- [macro_mapping.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/macro_mapping.csv)
- [feature_dictionary.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/feature_dictionary.csv)
- [horizon_manifest.csv](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/horizon_manifest.csv)
- [feature_master_monthly.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/feature_master_monthly.parquet)
- [target_panel_long_horizon.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/target_panel_long_horizon.parquet)
- [modeling_panel_hstack.parquet](/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs/modeling_panel_hstack.parquet)

## Coverage Reality

Coverage is clean enough overall for v4 to be active, but it is not uniform.

Important facts:

- the macro feature backbone starts in 2006
- `60m` is the main training horizon
- `120m` is accepted for first-pass modeling
- `180m` is retained in the data branch but is too thin for first-pass benchmark modeling
- `CR_EU_HY` is the weakest sleeve by history sufficiency

## What A New Agent Should Assume

- the data branch is accepted
- the target definitions are locked
- the 15-sleeve data branch is real
- the 14-sleeve default supervised roster is intentional
- do not “fix” the euro rule or `LISTED_RE` definition unless explicitly asked
