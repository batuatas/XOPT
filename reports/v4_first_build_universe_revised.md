# Revised v4 First-Build Universe

## Decision

The first-pass `v4` build can now be expanded meaningfully.

It no longer needs to remain at the earlier 10-sleeve conservative core, provided the team accepts one explicit target-rule addition:

- euro fixed-income sleeves may use `local-currency investable ETF + month-end FX conversion to USD`

## Revised recommended first-build universe

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

## What changed versus the earlier recommendation

Upgraded into first-pass `v4`:

- `FI_EU_GOVT`
- `CR_EU_IG`
- `CR_EU_HY`
- `LISTED_RE`
- `LISTED_INFRA`

Still not in first-pass `v4`:

- `FI_JP_GOVT`
- `FI_CN_GOVT`

## Required sleeve-charter locks

1. `LISTED_RE` means `ex-U.S. listed real estate`.
2. `RE_US` stays and coexists with `LISTED_RE`.
3. `LISTED_INFRA` means `global listed infrastructure`, explicitly equity-like.
4. `FI_EU_GOVT`, `CR_EU_IG`, and `CR_EU_HY` use the same local-fund-plus-FX target policy.

## Remaining deferred sleeves

- `FI_CN_GOVT`: still too fragile

## Remaining unresolved but not fully dead

- `FI_JP_GOVT`: economically valid, but still lacks a clean long-history public target rule

## Prompt 3 recommendation

Move to Prompt 3 only if the team accepts the four sleeve-charter locks above.

If accepted, the target layer is now coherent enough to scaffold a real `v4` branch without touching `v3`.
