# Scenario Case Conference Report

## Scope
- Active branch only: `v3_long_horizon_china`.
- Active scenario object only: the locked robust 5Y benchmark (`best_60_predictor`).
- These are scenario-conditioned, plausibility-regularized, anchor-local benchmark diagnostics for long-horizon SAA.

## Selected Cases
### Upside case
- Anchor: `2021-12-31`
- Source question: `Soft landing`
- Regime label: `risk-on reflation` with NFCI bucket `loose` and recession overlay `non-recession`
- Benchmark response: predicted return 12.03% -> 12.52%, max weight 29.15% -> 29.27%, effective N 4.40 -> 4.35
- Macro fingerprint: long_rate_JP, ig_oas, unemp_JP
- Why selected: Largest positive benchmark-return response among the final conference questions, with only limited extra concentration.
- Interpretation: A looser, growth-supportive state raises long-run return without turning the benchmark into a radically different portfolio.

### Breadth case
- Anchor: `2023-12-31`
- Source question: `Return with breadth`
- Regime label: `soft landing` with NFCI bucket `neutral` and recession overlay `non-recession`
- Benchmark response: predicted return 15.70% -> 15.44%, max weight 39.32% -> 38.45%, effective N 3.92 -> 4.01
- Macro fingerprint: long_rate_JP, us_real10y, long_rate_EA
- Why selected: Cleanest reduction in concentration with a visible effective-N improvement and only a modest return give-up.
- Interpretation: The benchmark can be made broader, but the gain comes from giving up some return rather than from a free diversification win.

### Adverse case
- Anchor: `2022-12-31`
- Source question: `Higher-for-longer`
- Regime label: `higher-for-longer tightness` with NFCI bucket `neutral` and recession overlay `non-recession`
- Benchmark response: predicted return 13.36% -> 13.18%, max weight 29.42% -> 29.45%, effective N 4.57 -> 4.63
- Macro fingerprint: unemp_JP, long_rate_JP, long_rate_US
- Why selected: Clearest restrictive macro narrative with a negative benchmark-return response under a tighter, more inflationary backdrop.
- Interpretation: A tighter, inflationary state trims return and nudges the benchmark toward a more defensive posture rather than a risk-on allocation.

## Visual Story
- Use the case overview figure first: it shows which scenarios matter at the benchmark level.
- Use the allocation-change figure second: it proves the benchmark changes across the full 9-sleeve system, not just one highlighted sleeve.
- Use the macro-fingerprint figure third: it explains what state shifts actually define the generated scenario.
- Keep China implicit inside the full allocation figure. It is present, but it does not dominate any selected case.