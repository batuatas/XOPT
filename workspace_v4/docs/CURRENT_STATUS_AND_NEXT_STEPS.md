# Current Status And Next Steps

## Current Status

The project is now in an accepted `v4` state.

That means:

- `v4` is the active downstream branch
- `v1`, `v2_long_horizon`, and `v3_long_horizon_china` remain frozen historical branches
- the accepted data and modeling-prep layers are in place
- the supervised prediction benchmark has been run
- the supervised portfolio benchmark has been run
- the active conference-graphics package has been rebuilt around the locked 5Y benchmark object

## What Is Locked Right Now

### Universe and taxonomy

- active 15-sleeve v4 data branch
- default 14-sleeve supervised benchmark roster excluding `CR_EU_HY`
- `CR_US_IG` replaces legacy `FI_IG`
- `LISTED_RE` means `ex-U.S. listed real estate`

### Targets

- annualized `60m` excess return is the primary benchmark target
- `120m` remains part of accepted first-pass modeling
- `180m` remains in the data branch but not the accepted first-pass benchmark package
- euro fixed-income sleeves use the locked local-return-plus-FX rule

### Modeling

- strongest practical benchmark predictor:
  - `elastic_net__core_plus_interactions__separate_60`

### Portfolio object

- active presentation benchmark:
  - predictor: `elastic_net__core_plus_interactions__separate_60`
  - label: `best_60_tuned_robust`
  - `lambda_risk = 8.0`
  - `kappa = 0.10`
  - `omega_type = identity`

## What Is Not Locked

These areas are still natural next-step work rather than frozen decisions:

- PTO/E2E comparison on v4
- scenario-generation layers built on top of the accepted benchmark object
- alternative presentation packages if the communication goal changes

## Recommended Next Steps

For a new agent, the next productive tasks are:

1. treat this workspace as the default v4 starting point
2. preserve the current active benchmark object unless explicitly asked to change it
3. use the current conference graphics as the presentation baseline
4. if new modeling work is requested, benchmark against the locked 5Y Elastic Net object first
5. if scenario work is requested, build on the locked tuned robust portfolio object rather than the old concentrated winner

## What Not To Do By Accident

- do not fall back to `v3` as the active branch
- do not silently reintroduce `FI_IG`
- do not treat `LISTED_RE` as global real estate
- do not put `CR_EU_HY` back into the default supervised benchmark roster without an explicit split redesign
- do not swap the active presentation benchmark object without updating reports and graphics consistently

## Practical Starting Recommendation

If you are continuing the project now:

- read the docs in this workspace
- inspect the copied benchmark reports
- inspect the current five active conference graphics
- then proceed to the next requested v4 task from this clean workspace context
