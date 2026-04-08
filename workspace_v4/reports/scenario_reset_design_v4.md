# Scenario Reset Design — v4

**Date:** 2026-03-30
**Project:** XOPTPOE v4

---

## 1. What Was Wrong with MALA-First

The original scenario engine used a Metropolis-Adjusted Langevin Algorithm (MALA) sampler initialized near the anchor state `m0`. Four structural problems made the results unsuitable for conference presentation:

### 1.1 Anchor Locality

MALA is initialized with chains that start either at `m0` or very close to it (with small random perturbations). The Langevin gradient update is:

```
m_{t+1} = m_t - (eta/2) * gradG(m_t) + sqrt(eta * tau) * noise
```

At the default settings (eta=0.005, tau=0.5), the per-step displacement is small relative to the macro state's historical variation. Over 200 steps with 40% warmup, the effective exploration radius is approximately ±1.5 standard deviations from `m0`. Most financially interesting regimes (e.g., 2008-style stress, 1970s inflation) lie much further.

### 1.2 Temperature Sensitivity and Low Diversity

The temperature parameter `tau` controls the noise magnitude. A low `tau` causes chains to converge quickly to a local mode of G — typically near the anchor. A high `tau` causes random walks that reject the objective gradient. The practical operating range was narrow, and even well-tuned MALA produced samples clustered in 1-2 regimes per question (observed: 80-100% of samples in the dominant regime).

From the existing output (`scenario_question_manifest_v4.csv`):
- Q1_gold_favorable: dominant regime share 66-100%
- Q2_ew_deviation: dominant regime share 100% in 3 of 4 anchors
- Q3_house_view_5pct: dominant regime share 81-100%

This is not a sampler tuning problem — it is a structural problem with the approach.

### 1.3 Burn-in Waste

MALA requires a burn-in phase (first 40% of steps discarded) to allow chains to move away from initialization. For a 200-step run with 4 chains, this wastes 320 evaluations per question-anchor combination. Each pipeline evaluation costs ~3ms, so burn-in alone costs ~1 second per question-anchor pair — yet produces no useful samples.

### 1.4 Weak Question Design

The probe functions were designed as penalized objectives:

```
G(m) = task_loss(m) + l2reg * ||m - m0||^2 / sigma^2
```

The L2 anchor regularizer explicitly pulls samples toward `m0`, counteracting the task loss gradient. Even with VAR(1)-based Mahalanobis regularization, the anchor acts as a gravity well. This is appropriate for MALA (where samples should be near the anchor) but directly prevents the discovery of distant, structurally different regimes.

### 1.5 Thin Coverage of Objective Space

The objective space for gold (Q1) spans approximately 6%-24% gold weight across the four anchor dates. The MALA samples covered only 6-10% gold weight — a factor of 3-4x below the upper bound actually observed in the data (2022-12-31 anchor: 22.3% gold). The sampler never escaped the anchor's neighborhood.

---

## 2. Why Historical Analog + Bounded Grid + Gradient Refinement Is Better

### 2.1 Diverse Initialization from Actually-Happened Regimes

Historical analog search scans the full macro history (2006-2026) for months that are regime-directionally consistent with the question target. For Q1 (gold threshold), this finds months with negative real yields, elevated stress, or high inflation — exactly the conditions that historically produce gold-favorable allocations. These months span:

- 2008-2009: financial crisis (stress regime)
- 2010-2012: ZIRP era (zero real yields)
- 2014-2019: mixed cycles (variable gold allocation)
- 2020-2022: COVID + inflation spike

This initialization is grounded in realized macro dynamics, not Gaussian noise around the anchor.

### 2.2 No Burn-in Waste

The historical analog approach is pure lookup — no sampling, no warmup, no acceptance probability. The 30 candidates are identified in O(T) time where T is the number of historical months. There is no concept of "burn-in" because we are not running a Markov chain.

### 2.3 Latin Hypercube Sampling for Systematic Coverage

LHS with 200 samples within the VAR(1) plausibility box [mu ± 3 sigma] ensures that the entire plausible macro state space is systematically covered. LHS guarantees projection-uniformity: each marginal dimension is uniformly sampled regardless of correlations. This is far more efficient than random sampling for discovering outlier regimes.

The VAR(1) plausibility filter (90th percentile Mahalanobis) removes states that are dynamically implausible — i.e., states that cannot arise from the observed macro dynamics in a single period. This is a principled constraint, not an arbitrary box.

### 2.4 Deterministic Gradient Refinement Is Reproducible

The gradient descent (50-60 steps, no noise) is fully deterministic given the initial state. This means:
- Results are reproducible bit-for-bit across runs
- No MALA acceptance probability to tune
- No temperature parameter
- Convergence is monotone (backtracking ensures G never increases)

The refinement is a local improvement step, not a sampler. It takes a historically-grounded starting point and moves it to the nearest objective optimum within the VAR(1) plausibility box. This produces "stressed but plausible" scenarios, not arbitrary macro states.

### 2.5 Regime Diversity Enforced at Selection

The Stage 3 selection explicitly enforces minimum 2 distinct regimes among the top 6 selected candidates. This is structurally impossible to enforce in MALA without heavy modifications. With the analog+LHS initialization, diverse regimes naturally emerge from different historical periods, and the selection step ensures they are preserved.

---

## 3. The Three-Stage Pipeline in Detail

### Stage 1a: Historical Analog Candidate Generation

**Input:** feature_master_monthly.parquet, question_id, anchor_date, m0

**Process:**
1. Deduplicate feature_master to one row per date
2. Exclude anchor date ± 3 months (to avoid trivial neighbors)
3. Apply question-specific regime filter function to each historical row
4. Rank passing rows by distance from regime center (standardized Euclidean in the question's key variables)
5. Return top K=30

**Regime filter definitions:**
- Q1 gold threshold: `us_real10y < 0.5 OR vix > 22 OR ig_oas > 1.5 OR infl_US > 3.5 OR short_rate_US < 0.5`
- Q2 EW departure: `(infl_US > 4.0 AND short_rate_US > 3.0) OR (ig_oas > 1.5 AND infl_US > 3.0) OR (term_slope_US < 0 AND infl_US > 3.0)`
- Q3 return discipline: `(2.0 < infl_US < 5.0 AND vix < 20 AND ig_oas < 1.3) OR (infl_US < 3.0 AND us_real10y > 0.5 AND unemp_US < 5.5)`
- Q4 return ceiling: `(vix < 16 AND ig_oas < 0.9 AND us_real10y > 1.0) OR vix > 30 OR ig_oas > 2.0 OR (short_rate_US > 4.0 AND us_real10y < 0.0)`

### Stage 1b: Latin Hypercube Sampling

**Input:** VAR(1) prior, m0

**Process:**
1. Compute VAR(1) one-step prediction: `mu = c + A @ m0`
2. Compute per-variable innovation standard deviation: `vol = sqrt(diag(Q))`
3. Define plausibility box: `[mu - 3*vol, mu + 3*vol]`
4. Generate N=200 LHS samples within this box (pure numpy, no scipy)
5. Filter: keep samples within 90th percentile historical Mahalanobis distance

**LHS implementation:** For each of D=19 dimensions, create n strata `[k/n, (k+1)/n]`, draw one uniform sample per stratum, shuffle independently, scale to [a, b].

### Stage 2: Bounded Gradient Descent Refinement

**Input:** Combined analog + LHS candidates (up to N_REFINE_TOP=40 best by quick G evaluation)

**Process:**
1. Quick evaluate all candidates: `G(m_init)` for all
2. Sort by G ascending, take top 40
3. For each: run 60 gradient descent steps with backtracking line search
   - `m_new = project_to_box(m - lr * gradG(m), a, b)`
   - Backtracking: if `G(m_new) >= G(m)`, halve lr up to 10 times
4. Return (m_refined, G_final, converged) for each

### Stage 3: Ranking and Selection

**Composite score:**
```
score = 0.5 * objective_rank_norm + 0.3 * plausibility_rank_norm + 0.2 * diversity_rank_norm
```

Where each rank is normalized to [0, 1] and lower is better.

**Selection:**
- Sort by composite score ascending
- Fill N_SELECT=6 slots greedily
- If current selection has < min_regimes=2 distinct regimes, prioritize next candidate from an unseen regime
- Otherwise take next best by composite score

---

## 4. Why This Is Better for This Project Specifically

### 4.1 Conference Interpretability

The MALA results produced scenarios that were numerically valid but narratively empty — "slightly less inflation than 2022" is not a conference-worthy insight. The analog search grounds scenarios in specific historical episodes: "this scenario resembles 2008-Q4 stress dynamics" or "this is structurally similar to the 2014 disinflation period." Conference audiences can evaluate these comparisons directly.

### 4.2 The Gold Threshold Question Requires Off-Anchor Exploration

The gold threshold question (Q1) is specifically about finding the macro conditions under which gold goes from 6% to 22% weight. This transition cannot be discovered by sampling near the 2021 anchor (where gold was at 8%) — it requires exploring states that resemble the 2022 anchor (where gold was at 22%). The historical analog search finds exactly these states.

### 4.3 Regime Diversity Is the Primary Output

The conference narrative is about regime transitions — "under what regime does the benchmark behave differently?" MALA's tendency to stay in one regime is a fatal flaw for this purpose. The reset design's Stage 3 explicitly enforces regime diversity, ensuring each question-answer pair spans multiple macro narratives.

### 4.4 Reproducibility for Academic Reporting

The deterministic gradient refinement produces the same results on every run. MALA with random noise does not (even with fixed seeds, acceptance probability introduces run-to-run variation). For a conference paper with documented numbers, reproducibility is essential.
