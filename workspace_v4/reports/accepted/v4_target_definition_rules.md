# v4 Target Definition Rules

## Purpose

This file freezes the target-definition rules for the `v4` first-build universe.

## Rule families

### 1. `USD_LISTED_ETF_DIRECT`

Applies to:

- `EQ_US`
- `EQ_EZ`
- `EQ_JP`
- `EQ_CN`
- `EQ_EM`
- `FI_UST`
- `CR_US_IG`
- `CR_US_HY`
- `RE_US`
- `LISTED_RE`
- `LISTED_INFRA`
- `ALT_GLD`

Rule:

1. use public adjusted-close history for the chosen USD-listed ETF proxy
2. define monthly price as the last available trading-day adjusted close in each calendar month
3. compute sleeve return as `price_t1 / price_t - 1`
4. interpret the resulting series as `USD unhedged`

## 2. `LOCAL_ETF_PLUS_FX_TO_USD`

Applies to:

- `FI_EU_GOVT`
- `CR_EU_IG`
- `CR_EU_HY`

Rule:

1. use a public investable ETF with sufficient history in local currency
2. compute monthly sleeve return in the ETF currency from adjusted close or equivalent total-return market series
3. convert that sleeve return into `USD unhedged` experience using public month-end FX
4. treat the resulting series as the locked `v4` target return

Required interpretation:

- this is a narrow `v4` exception to locked `v1`
- this rule is accepted only for the euro fixed-income sleeve family
- this rule does not automatically extend to `FI_JP_GOVT`, `FI_CN_GOVT`, or other future sleeves

## Month-end rule

For all `v4` first-build sleeves:

- use the last available trading day in the calendar month
- do not force literal calendar-end observations when markets are closed

## Currency rule

- all final `v4` target returns are modeled in `USD`
- all final `v4` target returns are interpreted as `unhedged`

## Excluded first-build sleeves

No target-definition rule is locked for:

- `FI_CN_GOVT`
- `FI_JP_GOVT`

These sleeves remain outside first-build `v4`.
