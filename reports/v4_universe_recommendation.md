# v4 Universe Recommendation

## Decision

Build `v4` as a `two-stage plan`, not as the full Akif universe immediately.

## Recommended first-build v4 universe

The recommended `v4 core` is:

- `EQ_US`
- `EQ_EZ`
- `EQ_JP`
- `EQ_CN`
- `EQ_EM`
- `FI_UST`
- `CR_US_IG`
- `CR_US_HY`
- `ALT_GLD`
- `RE_US`

## Why this is the right first build

- It preserves the locked project discipline: clean investable sleeves first, taxonomy expansion second.
- It extends `v3` meaningfully without importing multiple fragile target-construction problems at once.
- It avoids mixing straightforward ETF sleeves with bespoke non-US sovereign bond engineering in the same first pass.
- It creates a coherent industry-facing SAA universe that is still auditable.
- It does not admit a sleeve merely because a plausible product exists.

## Sleeves not recommended for first build

Stage-2 after clarification:

- `FI_EU_GOVT`
- `FI_JP_GOVT`
- `CR_EU_IG`
- `LISTED_RE`
- `LISTED_INFRA`

Deferred:

- `FI_CN_GOVT`
- `CR_EU_HY`

## Required open questions before any v4 build prompt

1. Is `CR_US_IG` the formal `v4` replacement name for legacy `FI_IG`?
2. For non-US fixed income sleeves, is the project willing to accept index-plus-FX target construction, or must first-pass targets remain simple public investable fund series?
3. If `LISTED_RE` is pursued later, does it mean `global listed real estate`, `ex-US listed real estate`, or `replace RE_US entirely`?
4. If `LISTED_INFRA` is pursued later, is the team comfortable treating listed infrastructure as an equity-like real-asset sleeve rather than a private-infrastructure analogue?
5. Does the team want `ALT_GLD` to keep its legacy sleeve id for continuity, or should naming be modernized only after the target layer is stable?

## Governance plan

1. Freeze `v3_long_horizon_china` as the active benchmark branch until `v4 core` is built and passes coherence review.
2. Approve the `v4 core` sleeve list and target-policy rules in writing.
3. Only then move to Prompt 2 for versioned `v4` data-build scaffolding.
4. Do not repoint downstream modeling or benchmarks until the `v4 core` dataset exists and is audited against `v3`.
