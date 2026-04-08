# XOPTPOE v4 Universe Admission Audit

## Scope and governance

- `docs/LOCKED_V1_SPEC.md` remains the primary design authority for baseline discipline: investable sleeves, USD base, unhedged return experience, monthly frequency, and reproducible public targets.
- `v3_long_horizon_china` remains the frozen active benchmark branch until a coherent `v4` is proven. This audit does not modify `v1`, `v2`, or `v3`.
- This pass is a sleeve-admission decision, not a dataset build, modeling migration, or target implementation.

## Admission standard used

Each proposed sleeve was evaluated in this order:

1. sleeve role in an institutional long-horizon SAA universe
2. admission decision for a first-pass `v4` universe
3. target-definition risk
4. history sufficiency for 5Y / 10Y / 15Y targets
5. consistency with the rest of the sleeve taxonomy
6. implementation risk

Likely proxy families are noted only as secondary comments. They do not drive admission.

Decision labels:

- `ADMIT_NOW`: clean enough for first-pass `v4` target build
- `ADMIT_WITH_CLARIFICATION`: conceptually admissible, but target or sleeve charter must be tightened before build
- `DEFER`: not clean enough for first-pass `v4`

## High-level conclusion

The project should not build the full Akif universe now. The clean governance move is a two-stage `v4` plan:

- `v4 core` first: only sleeves with clear long-horizon investable targets and low-to-moderate implementation ambiguity
- `v4 expansion` second: add non-US sovereign and non-US credit / real-asset sleeves only after exact target definitions are locked

The current repo already contains partial feature concepts for several expansion sleeves, but those are not sufficient evidence that the target layer is admission-ready. Feature availability is weaker than target clarity.

## Sleeve-by-sleeve audit

### Equities

#### EQ_US
- Sleeve role: core SAA anchor
- Decision: `ADMIT_NOW`
- Target-definition risk: very low; current investable-target rule is already coherent
- History: sufficient
- Consistency: fully aligned
- Implementation risk: low
- Secondary proxy comment: current broad US equity ETF discipline remains appropriate

#### EQ_EZ
- Sleeve role: core developed ex-US regional equity sleeve
- Decision: `ADMIT_NOW`
- Target-definition risk: low if the sleeve remains explicitly euro area rather than generic Europe
- History: sufficient
- Consistency: strong
- Implementation risk: low
- Secondary proxy comment: use the same explicit euro-area discipline already embedded in `v1`

#### EQ_JP
- Sleeve role: core developed regional equity sleeve
- Decision: `ADMIT_NOW`
- Target-definition risk: low
- History: sufficient
- Consistency: strong
- Implementation risk: low
- Secondary proxy comment: broad Japan equity ETF treatment remains clean

#### EQ_CN
- Sleeve role: acceptable dedicated China sleeve in an expanded SAA universe
- Decision: `ADMIT_NOW`
- Target-definition risk: moderate but manageable; the main risk is reopening broad-China purity versus long-history investability too early
- History: sufficient if `FXI`-style proxy discipline is retained
- Consistency: good as a versioned extension already proven in `v3`
- Implementation risk: low to medium
- Secondary proxy comment: do not reopen the `FXI` versus broader-China debate inside first-pass `v4`; lock one investable China target first

#### EQ_EM
- Sleeve role: important broad EM sleeve even with standalone China present
- Decision: `ADMIT_NOW`
- Target-definition risk: low
- History: sufficient
- Consistency: strong
- Implementation risk: low
- Secondary proxy comment: broad EM ETF treatment remains coherent

### Fixed income and credit

#### FI_UST
- Sleeve role: core duration / ballast sleeve
- Decision: `ADMIT_NOW`
- Target-definition risk: low
- History: sufficient
- Consistency: strong
- Implementation risk: low
- Secondary proxy comment: existing intermediate Treasury discipline remains valid

#### FI_EU_GOVT
- Sleeve role: sensible institutional sovereign-bond sleeve
- Decision: `ADMIT_WITH_CLARIFICATION`
- Target-definition risk: high for first-pass governance because the sleeve is economically clear but target construction is not
- History: conceptually sufficient, but only if the proxy definition is locked to a reproducible investable series
- Consistency: fits the taxonomy
- Implementation risk: medium to high
- Main issue: there is no obvious existing repo-standard USD-listed US ETF equivalent to `IEF` for a pure euro-area sovereign sleeve with long history and stable currency treatment
- Required clarification: choose one of
  - USD investor experience via a public unhedged ETF/share class
  - local-currency sovereign total-return index plus explicit FX conversion
  - a broader non-US sovereign fund, which would weaken sleeve purity and should likely be rejected

#### FI_JP_GOVT
- Sleeve role: sensible institutional sovereign-bond sleeve
- Decision: `ADMIT_WITH_CLARIFICATION`
- Target-definition risk: high; the sleeve role is valid but the target rule is still too underspecified
- History: likely sufficient only if a clean public proxy is identified
- Consistency: fits the taxonomy
- Implementation risk: high
- Main issue: Japanese government bonds are economically valid but awkward in a USD-investable, public, long-history sleeve framework
- Comment: viable as a stage-2 sleeve only after the target construction rule is explicit

#### FI_CN_GOVT
- Sleeve role: conceptually attractive but not essential to a first-pass SAA core
- Decision: `DEFER`
- Target-definition risk: very high
- History: fragile
- Consistency: would fit only if the target rule is materially more complex than the rest of the universe
- Implementation risk: high
- Why defer:
  - onshore China government bond access is structurally different from the rest of the universe
  - public ETF proxies are relatively short-history, mandate-sensitive, or awkward on currency treatment
  - a stable long-horizon target definition is materially harder than `EQ_CN`
- Bottom line: realistic later, not realistic for first-pass `v4 core`

#### CR_US_IG
- Sleeve role: core spread sleeve
- Decision: `ADMIT_NOW`
- Target-definition risk: low; this is mainly a taxonomy cleanup from `FI_IG` to `CR_US_IG`
- History: sufficient
- Consistency: strong if taxonomy is cleaned so corporates sit under `CR_*`
- Implementation risk: low
- Secondary proxy comment: first-pass `v4` should treat this as a naming rationalization, not a new economic sleeve

#### CR_EU_IG
- Sleeve role: sensible credit sleeve
- Decision: `ADMIT_WITH_CLARIFICATION`
- Target-definition risk: medium to high
- History: economic history exists, but clean public investable proxy history is not obviously long enough under a simple ETF-only rule
- Consistency: good
- Implementation risk: medium
- Main issue: easy to describe, harder to reproduce cleanly without mixing hedged ETF products, spreadsheet-only sources, or non-US share classes
- Comment: admissible in principle, but not a first-build core sleeve unless the exact target proxy is locked

#### CR_US_HY
- Sleeve role: sensible return-seeking spread sleeve
- Decision: `ADMIT_NOW`
- Target-definition risk: low
- History: sufficient with established USD high-yield ETFs
- Consistency: improves breadth of the credit bucket
- Implementation risk: low
- Secondary proxy comment: this is the cleanest genuinely new credit sleeve for `v4 core`

#### CR_EU_HY
- Sleeve role: plausible but non-essential spread sleeve
- Decision: `DEFER`
- Target-definition risk: high
- History: not clean enough under the likely public ETF options
- Consistency: acceptable in theory
- Implementation risk: high
- Why defer:
  - available ETF options are either short-history, renamed, mandate-changed, or USD-hedged in a way that complicates sleeve comparability
  - the current repo has useful EU HY state features, but not a clearly locked target object

### Real assets and alternatives

#### RE_US
- Sleeve role: strong, narrow, already-proven real-estate sleeve
- Decision: `ADMIT_NOW`
- Target-definition risk: low
- History: sufficient
- Consistency: strong as the cleanest real-estate sleeve currently available
- Implementation risk: low

#### Listed RE
- Sleeve role: valid only if the sleeve charter is made explicit
- Decision: `ADMIT_WITH_CLARIFICATION`
- Target-definition risk: medium to high because the sleeve itself is not yet fully defined
- History: likely sufficient if a global listed real estate ETF is chosen, but the exact definition is unresolved
- Consistency: unresolved because it directly interacts with `RE_US`
- Implementation risk: medium
- Core issue: `Listed RE` could mean at least three different things
  - US listed REITs, which is just `RE_US`
  - global listed real estate
  - ex-US listed real estate
- Recommendation: do not replace `RE_US` now
- Preferred governance: keep `RE_US` in `v4 core`; evaluate `Listed RE` later as a possible global replacement or companion only after overlap logic is explicitly approved

#### Listed Infra
- Sleeve role: credible industry-facing real-asset sleeve, but not a mandatory first-pass core sleeve
- Decision: `ADMIT_WITH_CLARIFICATION`
- Target-definition risk: medium; the bigger issue is sleeve charter rather than product scarcity
- History: likely sufficient
- Consistency: decent, but its overlap with equities and utilities needs explicit acceptance
- Implementation risk: medium
- Secondary proxy comment: among the harder sleeves, this is the strongest stage-2 expansion candidate after `CR_US_HY`
- Secondary proxy comment: the sleeve definition should explicitly state that this is listed equity-like infrastructure, not private infrastructure

#### ALT_GLD
- Sleeve role: clean diversifying reserve / real-asset sleeve
- Decision: `ADMIT_NOW`
- Target-definition risk: low
- History: sufficient
- Consistency: strong
- Implementation risk: low
- Naming note: the target is clean; the only open issue is taxonomy. `ALT_GLD` is defensible for continuity, but a later taxonomy pass could rename the bucket without changing the economic sleeve.

## Special design answers

### RE_US versus Listed RE

- `RE_US` should not be dropped in first-pass `v4`.
- `RE_US` should also not be silently replaced by `Listed RE`.
- Best current governance:
  - keep `RE_US` in `v4 core`
  - treat `Listed RE` as a stage-2 candidate requiring a formal sleeve charter
  - only consider replacement later if the approved target is clearly global listed real estate and the project explicitly prefers broader real-estate representation over continuity with `v1`/`v3`

### FI_CN_GOVT

- `FI_CN_GOVT` is too fragile for first-pass `v4`.
- It fails the “simple, reproducible, clean investable target” standard relative to the rest of the candidate universe.
- It should be deferred until the team is willing to accept a more bespoke target-construction rule.

### Listed RE and Listed Infra maturity

- `Listed RE`: mature concept, not yet mature sleeve definition in this project
- `Listed Infra`: mature concept and probably buildable later, but still not governance-ready for automatic first-pass inclusion

## Recommended first-build universe

Recommended `v4 core`:

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

This keeps the first build disciplined:

- 5 equity sleeves
- 3 fixed-income / credit sleeves
- 2 real-asset / alternative sleeves

It expands `v3` in a meaningful but still governable way by adding one genuinely new sleeve class: US high yield.

## Stage-2 expansion candidates

Admissible after clarification:

- `FI_EU_GOVT`
- `FI_JP_GOVT`
- `CR_EU_IG`
- `Listed RE`
- `Listed Infra`

Deferred:

- `FI_CN_GOVT`
- `CR_EU_HY`

## Governance plan from v3 to v4

1. Keep `v3_long_horizon_china` frozen as the active benchmark branch.
2. Approve a `v4 core` sleeve list before any data-build code is scaffolded.
3. Lock one target proxy candidate per sleeve, including a written currency rule.
4. Only after that, build `v4` as a new versioned branch without repointing downstream modeling.
5. Run coherence checks against `v3` before any benchmark migration decision.
6. Consider stage-2 sleeves only after the `v4 core` target layer is complete and auditable.

## Final recommendation

- Build style: `two-stage v4 plan`
- Do not build the full Akif universe now.
- Do not repoint downstream code yet.
- Prompt 2 should only start once the team explicitly accepts this `v4 core` and resolves the remaining target-definition questions for any sleeve it wants elevated into the first build.
