# Project Overview

## Core Problem

`XOPTPOE` is a long-horizon strategic asset-allocation research project.

The problem is not short-horizon market timing. The project asks a different question:

- given the current macro-financial state,
- what are plausible medium-term return differences across investable sleeves,
- and how should those differences feed a disciplined strategic allocation?

The core modeling horizon is 5 years. Longer horizons of 10 years and 15 years are also built into the data layer for coverage and diagnostic use.

## Strategic Framing

The project is explicitly about long-horizon SAA:

- not monthly tactical allocation
- not a high-turnover trading system
- not a pure scenario engine without an underlying predictive benchmark

The intended logic is:

1. observe the macro-financial state
2. translate that state into sleeve-level long-horizon expected excess returns
3. convert those predictions into a robust long-only allocation
4. later stress that benchmark under coherent macro scenarios

## Why v4 Exists

Earlier versions established the long-horizon framework and the China sleeve. The current active branch is `v4`, which expands the investable universe while keeping the same long-horizon SAA objective.

The design goal of `v4` is:

- a richer investable universe that still maps cleanly to investable public targets
- stronger institutional SAA coverage than the older smaller branches
- a cleaner basis for later scenario-generation work

## Active v4 Universe Logic

The accepted v4 data branch contains 15 sleeves:

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

The default supervised benchmark roster excludes `CR_EU_HY`, leaving 14 sleeves for the accepted first-pass benchmark runs.

## Category Taxonomy

- `Equity`: `EQ_US`, `EQ_EZ`, `EQ_JP`, `EQ_CN`, `EQ_EM`
- `Fixed Income`: `FI_UST`, `FI_EU_GOVT`
- `Credit`: `CR_US_IG`, `CR_EU_IG`, `CR_US_HY`, `CR_EU_HY`
- `Real Asset`: `RE_US`, `LISTED_RE`, `LISTED_INFRA`
- `Alternative`: `ALT_GLD`

## Important Sleeve Interpretations

- `LISTED_RE` means `ex-U.S. listed real estate`
- `RE_US` remains a separate U.S. listed real-estate sleeve
- `LISTED_INFRA` is a global listed infrastructure sleeve and is explicitly treated as an equity-like real-asset sleeve
- `CR_US_IG` is the official v4 replacement for legacy `FI_IG` naming

## What The Project Is Trying To Produce

The long-run goal is a coherent sequence:

- accepted long-horizon data layer
- accepted supervised prediction benchmark
- accepted benchmark allocation object
- later PTO/E2E and scenario layers that must beat or improve on that benchmark

At the moment, the active and locked object is the v4 5Y benchmark workflow described in the rest of this workspace.
