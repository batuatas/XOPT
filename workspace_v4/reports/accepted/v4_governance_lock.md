# v4 Governance Lock

## Status

This document is the governance lock for the `v4` first-build universe.

It freezes:

- the exact first-build sleeve roster
- the final category taxonomy
- the locked sleeve interpretations
- the locked target-definition rule for euro fixed-income sleeves

It does not build the dataset.
It does not repoint downstream modeling.
It does not modify `v1`, `v2`, or `v3`.

`v3_long_horizon_china` remains the frozen active benchmark branch until the versioned `v4` build is completed and judged coherent.

## Locked v4 first-build universe

The locked `v4` first-build sleeve roster is:

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

Total sleeves: `15`

## Locked category labels

The final category mapping for first-build `v4` is:

- `Equity`
  - `EQ_US`
  - `EQ_EZ`
  - `EQ_JP`
  - `EQ_CN`
  - `EQ_EM`
- `Fixed Income`
  - `FI_UST`
  - `FI_EU_GOVT`
- `Credit`
  - `CR_US_IG`
  - `CR_EU_IG`
  - `CR_US_HY`
  - `CR_EU_HY`
- `Real Asset`
  - `RE_US`
  - `LISTED_RE`
  - `LISTED_INFRA`
- `Alternative`
  - `ALT_GLD`

## Locked taxonomy cleanup

- `CR_US_IG` is the official `v4` replacement for the legacy `FI_IG` naming.
- `FI_IG` remains part of frozen historical branches only.
- All corporate-credit sleeves in `v4` use the `CR_*` taxonomy.
- Sovereign bond sleeves remain in the `FI_*` taxonomy.

## Locked sleeve definitions

### Equity sleeves

- `EQ_US`: broad U.S. equity sleeve
- `EQ_EZ`: euro-area equity sleeve, not generic Europe
- `EQ_JP`: Japan equity sleeve
- `EQ_CN`: investable China equity sleeve
- `EQ_EM`: broad emerging-market equity sleeve

### Fixed income sleeves

- `FI_UST`: intermediate U.S. Treasury sleeve
- `FI_EU_GOVT`: euro-area government bond sleeve

### Credit sleeves

- `CR_US_IG`: U.S. investment-grade corporate credit sleeve
- `CR_EU_IG`: euro investment-grade corporate credit sleeve
- `CR_US_HY`: U.S. high-yield corporate credit sleeve
- `CR_EU_HY`: euro high-yield corporate credit sleeve

### Real-asset sleeves

- `RE_US`: U.S. listed real-estate / REIT sleeve
- `LISTED_RE`: `ex-U.S. listed real estate`
- `LISTED_INFRA`: `global listed infrastructure`, explicitly treated as an equity-like real-asset sleeve

### Alternative sleeve

- `ALT_GLD`: gold-bullion-linked sleeve

## Locked listed real-estate interpretation

`LISTED_RE` is locked to mean:

- `ex-U.S. listed real estate`

This is the final `v4` first-build interpretation.

The project does not treat `LISTED_RE` as:

- global listed real estate
- a synonym for `RE_US`
- a replacement for `RE_US`

## Locked coexistence rule

- `RE_US` remains in the first-build universe
- `RE_US` and `LISTED_RE` coexist in first-build `v4`

The purpose of this coexistence is to preserve:

- a clean U.S. REIT sleeve
- a distinct non-U.S. listed real-estate sleeve

## Locked euro fixed-income target rule

The following sleeves use a shared locked `v4` target-definition rule:

- `FI_EU_GOVT`
- `CR_EU_IG`
- `CR_EU_HY`

### Rule

For these sleeves, `v4` accepts:

- `local-currency investable ETF return + FX conversion to USD`

### Exact standard

1. choose a public investable ETF with sufficient history in local currency
2. use monthly returns from the ETF adjusted-close or equivalent total-return market series in the ETF trading / base currency
3. convert that monthly sleeve return into `USD unhedged` investor experience using public month-end FX
4. label the resulting target as `USD unhedged synthesized from local investable fund + FX`

### Governance meaning

This is a deliberate `v4` exception to the fully uniform locked `v1` target rule.

It is accepted because:

- the sleeve family is economically clean
- the target rule is explicit
- the rule is consistent across the euro fixed-income family
- the result still reflects investable, unhedged USD experience

### Non-extension rule

This euro fixed-income exception is locked only for:

- `FI_EU_GOVT`
- `CR_EU_IG`
- `CR_EU_HY`

It does not automatically authorize bespoke FX synthesis for other sleeves.

## Excluded from first-build v4

The following sleeves are excluded from first-build `v4`:

- `FI_CN_GOVT`
- `FI_JP_GOVT`

### Exclusion status

- `FI_CN_GOVT`: deferred
- `FI_JP_GOVT`: unresolved and out of first build

## Build-blocking status

No unresolved governance debate remains that blocks Prompt 4, provided the implementation follows this lock exactly.

There are still implementation tasks ahead, but they are no longer design blockers:

- encode the euro fixed-income FX rule in manifests and build logic
- encode the final `v4` taxonomy and sleeve roster
- keep `v3` frozen while versioning `v4`

## Final governance decision

The `v4` first-build design is now locked.

Prompt 4 should implement this exact first-build universe and these exact target-definition rules as a new versioned branch without altering frozen `v3`.
