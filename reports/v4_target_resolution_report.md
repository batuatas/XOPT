# v4 Target Resolution Report

## Scope

This pass resolves the previously non-automatic sleeves using concrete target candidates.

It does not build `v4`.
It does not repoint downstream modeling.
It treats `v3_long_horizon_china` as the frozen active benchmark branch until `v4` is proven coherent.

## Resolution principle

The key design question is no longer just whether a sleeve is economically sensible.
It is whether the sleeve can be mapped to a clean, investable, reproducible, long-history target rule.

For this pass, the cleanest resolution is:

- keep the original `v1` discipline wherever possible: USD-listed, unhedged ETF targets
- where no clean long-history USD-listed proxy exists, allow a narrow `v4` exception only if all of the following hold:
  - the sleeve itself is economically clean
  - a long-history public fund or benchmark exists in local currency
  - USD investor experience can be defined explicitly and reproducibly
  - the rule is consistent across a sleeve family rather than ad hoc

## Recommended narrow v4 exception rule

Adopt one explicit `v4` target exception for selected non-US bond sleeves:

- use a public local-currency ETF share class with sufficient history
- calculate month-end total return from adjusted close / NAV-like market series in the fund currency
- convert into USD using public month-end FX
- label the result as `USD unhedged synthesized from local investable fund + FX`

This is a meaningful departure from locked `v1`, but it is still disciplined because:

- it remains investable in spirit
- it is explicit
- it is reproducible
- it avoids mixing in opaque spreadsheet-only target series

This exception is strong enough to rescue the euro fixed-income family.
It is not strong enough to rescue Japan government bonds or China government bonds in first-pass `v4`.

## Sleeve-by-sleeve resolution

### FI_EU_GOVT

- Resolution decision: `UPGRADE_TO_ADMIT_NOW`
- Sleeve charter: `Euro-area government bonds, duration-matched to the existing U.S. Treasury sleeve where possible`
- Preferred candidate: [`iShares € Govt Bond 7-10yr UCITS ETF (IBGM)`](https://www.ishares.com/uk/professional/en/products/251738/ishares-euro-government-bond-710yr-ucits-etf)
- Evidence:
  - fund launch date: `08-Dec-2006`
  - fund base currency: `EUR`
  - benchmark: `Bloomberg Euro Government Bond 10 Year Term Index`
- Fallback candidate: [`iShares Core € Govt Bond UCITS ETF (IEGA/IEGS family)`](https://www.ishares.com/ch/individual/en/products/251740/ishares-euro-government-bond-ucits-etf)
- Why preferred:
  - longest clean public history among the obvious euro sovereign ETF candidates found
  - duration profile is closer to `FI_UST = IEF`
  - benchmark is explicit and sovereign-only
- Earliest likely usable date: `2007-01-31`
- USD treatment rule: convert EUR sleeve returns to USD using month-end `EURUSD`
- Investability assessment: sufficiently investable for a versioned research build, even though not US-listed
- Remaining drawback:
  - this is not a pure `v1-style` USD-listed ETF target
  - implementation needs an explicit FX layer
- Final view:
  - `FI_EU_GOVT` is now clean enough to admit if the project explicitly accepts the euro-bond local-fund-plus-FX rule

### FI_JP_GOVT

- Resolution decision: `KEEP_ADMIT_WITH_CLARIFICATION`
- Sleeve charter: `Japanese government bonds, unhedged USD experience`
- Preferred candidate: [`iShares Japan Govt Bond UCITS ETF`](https://www.ishares.com/uk/professional/en/products/334023/ishares-japan-govt-bond-ucits-etf)
- Evidence:
  - fund launch date: `06-Dec-2023`
  - benchmark: `Bloomberg Japan Treasury Index`
  - fund base currency: `JPY`
- Fallback candidate: direct `Bloomberg Japan Treasury Index` plus `JPYUSD` conversion
- Why not upgraded:
  - ETF history is far too short for clean 10Y/15Y long-horizon work
  - index route is conceptually possible, but a stable public history source was not pinned down cleanly enough in this pass
- Earliest likely usable date:
  - ETF route: `2024-01-31`, which is not sufficient
  - index route: unresolved for public reproducible implementation
- USD treatment rule if admitted later: local JPY return plus `JPYUSD` FX conversion
- Final view:
  - the sleeve remains economically valid
  - first-pass `v4` still should not admit it without an explicit public index-history solution

### CR_EU_IG

- Resolution decision: `UPGRADE_TO_ADMIT_NOW`
- Sleeve charter: `Euro-denominated investment-grade corporate bonds, USD unhedged investor experience`
- Preferred candidate: [`iShares Core € Corp Bond UCITS ETF (IEAC)`](https://www.ishares.com/uk/professional/en/literature/fact-sheet/ieac-ishares-core-corp-bond-ucits-etf-fund-fact-sheet-en-gb.pdf?siteEntryPassthrough=true&switchLocale=y)
- Evidence:
  - fund launch date: `06-Mar-2009`
  - share class currency: `EUR`
  - benchmark: `BBG Euro Aggregate Corporate Index (EUR)`
- Fallback candidate: USD-hedged share-class sibling such as `IEASX` / `HEZU` family if the team refuses FX synthesis
- Why preferred:
  - long enough history
  - economically precise euro IG corporate sleeve
  - same local-fund-plus-FX rule as euro government bonds
- Earliest likely usable date: `2009-03-31`
- USD treatment rule: convert EUR sleeve returns to USD using month-end `EURUSD`
- Investability assessment: good enough for first-pass `v4`
- Remaining drawback:
  - depends on accepting the euro fixed-income exception rule
  - hedged USD share classes exist but are less aligned with unhedged sleeve discipline
- Final view:
  - `CR_EU_IG` is buildable now if `FI_EU_GOVT` is accepted under the same target rule

### LISTED_RE

- Resolution decision: `UPGRADE_TO_ADMIT_NOW`
- Recommended sleeve charter: `Listed real estate ex-U.S.`
- Preferred candidate: [`SPDR Dow Jones International Real Estate ETF (RWX)`](https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-rwx.pdf)
- Evidence:
  - inception date: `12/15/2006`
  - benchmark: `Dow Jones Global ex-U.S. Select Real Estate Securities Index`
  - USD-listed
- Fallback candidate: [`SPDR Dow Jones Global Real Estate ETF (RWO)`](https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-dow-jones-global-real-estate-etf-rwo)
- Why preferred:
  - resolves the design conflict with `RE_US`
  - keeps `RE_US` as the U.S. REIT sleeve
  - adds a genuinely new real-estate region sleeve rather than duplicating U.S. exposure inside a global fund
- Earliest likely usable date: `2007-01-31`
- USD treatment rule: none beyond existing USD ETF discipline
- Investability assessment: strong
- Remaining drawback:
  - the sleeve id `LISTED_RE` is generic; the charter should explicitly say `ex-U.S. listed real estate`
- Final view:
  - `LISTED_RE` is admissible now if it is defined as `ex-U.S. listed real estate`
  - `RE_US` should coexist with it

### LISTED_INFRA

- Resolution decision: `UPGRADE_TO_ADMIT_NOW`
- Sleeve charter: `Global listed infrastructure, equity-like real-asset sleeve`
- Preferred candidate: [`iShares Global Infrastructure ETF (IGF)`](https://www.ishares.com/us/products/239746/ishares-global-infrastructure-etf)
- Evidence:
  - fund inception: `Dec 10, 2007`
  - benchmark: `S&P Global Infrastructure Index (Net)`
  - USD-listed
- Fallback candidate: [`SPDR S&P Global Infrastructure ETF (GII)`](https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-global-infrastructure-etf-gii)
- Why preferred:
  - cleaner benchmark continuity than `GII`
  - `GII` uses linked benchmark returns from the `Macquarie Global Infrastructure 100 Index` until `5/1/2013`, then `S&P Global Infrastructure Index`
  - `IGF` still has enough history for 15Y targets as of 2026
- Earliest likely usable date: `2008-01-31`
- USD treatment rule: none beyond existing USD ETF discipline
- Investability assessment: strong
- Remaining drawback:
  - sleeve behavior is equity-like and overlaps with utilities / transportation / energy infrastructure
- Final view:
  - `LISTED_INFRA` is buildable now if the sleeve charter explicitly accepts listed equity infrastructure as the target object

### FI_CN_GOVT

- Resolution decision: `KEEP_DEFER`
- Sleeve charter attempted: `China government bonds, investable, long-history, USD-definable`
- Preferred candidate examined: [`VanEck China Bond ETF (CBON)`](https://www.vaneck.com/us/en/investments/chinaamc-china-bond-etf-cbon/overview/)
- Evidence:
  - inception date: `11/10/2014`
  - benchmark: `FTSE Chinese Broad Bond 0 – 10 Diversified Select Index`
  - benchmark composition includes `Chinese credit, governmental and quasi-governmental` issuers
- Fallback candidate examined: [`KraneShares Bloomberg China Bond Inclusion Index ETF / KBND family`](https://kraneshares.com/kcny/)
- Why still deferred:
  - `CBON` is not a clean pure government-bond sleeve
  - likely alternatives are either newer, renamed, repurposed, or still not pure sovereign
  - onshore China bond access and currency treatment remain meaningfully more bespoke than the rest of `v4`
- Earliest likely usable date of best ETF candidate: `2014-11-30`
- USD treatment rule if ever admitted: either use the USD-listed ETF directly or use local-CNY sovereign rule plus FX, but neither route is clean enough yet
- Final view:
  - after deeper search, `FI_CN_GOVT` is still out for first-pass `v4`

### CR_EU_HY

- Resolution decision: `UPGRADE_TO_ADMIT_NOW`
- Sleeve charter: `Euro-denominated high-yield corporate bonds, USD unhedged investor experience`
- Preferred candidate: [`iShares € High Yield Corp Bond UCITS ETF (HIGH / IHYG family)`](https://www.ishares.com/ch/professionals/en/products/290618/ishares-high-yield-corp-bond-ucits-etf-fund)
- Evidence:
  - fund launch date: `03-Sep-2010`
  - fund base currency: `EUR`
  - benchmark: `Markit iBoxx Euro Liquid High Yield Index (EUR)`
- Fallback candidate: USD-hedged share class `HYGU`
- Why preferred:
  - much cleaner long-history solution than the USD-hedged or newer constrained alternatives
  - consistent with the same euro credit FX rule used for `CR_EU_IG`
- Earliest likely usable date: `2010-09-30`
- USD treatment rule: convert EUR sleeve returns to USD using month-end `EURUSD`
- Investability assessment: acceptable for first-pass `v4`, though less clean than euro IG
- Remaining drawback:
  - this is the highest-risk sleeve among the euro fixed-income upgrades
  - requires accepting both local-fund-plus-FX and high-yield spread behavior
- Final view:
  - `CR_EU_HY` can be rescued into first-pass `v4`, but it should be tagged as a higher-risk admit than `FI_EU_GOVT` and `CR_EU_IG`

## Cross-sleeve resolution summary

### Sleeves upgraded into first-pass v4

- `FI_EU_GOVT`
- `CR_EU_IG`
- `LISTED_RE` as `ex-U.S. listed real estate`
- `LISTED_INFRA`
- `CR_EU_HY`

### Sleeves still not admitted

- `FI_JP_GOVT`
- `FI_CN_GOVT`

## Revised first-pass v4 universe

Recommended revised first-build universe:

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

This is materially richer than the earlier conservative core, but it is still coherent because the new admits follow two clean patterns:

- USD-listed equity / real-asset ETFs
- euro bond sleeves built under one explicit local-currency-plus-FX rule

## Final governance recommendation

- `RE_US` should coexist with `LISTED_RE`
- `LISTED_RE` should mean `ex-U.S. listed real estate`
- `FI_CN_GOVT` remains out
- `FI_JP_GOVT` remains unresolved and should stay out of first-pass `v4`
- `CR_EU_HY` can come in, but should be documented as a higher-risk admit than the rest of the upgraded sleeves

## Prompt 3 readiness

The project is ready for Prompt 3 only if the team explicitly accepts two design choices:

1. euro fixed-income sleeves may use `local-currency investable ETF + FX conversion`
2. `LISTED_RE` means `ex-U.S. listed real estate`, not global real estate and not a replacement for `RE_US`
