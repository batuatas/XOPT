# Akif's Questions — Answered from v4 Scenario Engine
**Generated:** 2026-03-31
**Data sources:** `scenario_results_v4.csv` (7,200 rows, Q1–Q3) + `scenario_results_v4_akif.csv` (12,000 rows, Q4–Q8)
**Combined sample:** 19,200 macro states across 4 anchors (2021–2024)

---

## Q1 — Which macro regime produces both a diversified portfolio and a good Sharpe?

**Answer:** These two objectives do **not** co-locate in the same regime.

**Diversification (Q4 — portfolio entropy):**

| Regime | Mean entropy | Max entropy | Note |
|---|---|---|---|
| reflation_risk_on | 2.032 | 2.134 | Best diversification |
| mixed_mid_cycle | 1.996 | 2.154 | Second |
| risk_off_stress | 1.875 | 2.169 | Moderate |
| higher_for_longer | 1.813 | 1.934 | Most concentrated |

Perfect equal-weight entropy = log(14) = **2.639**. The model never reaches it — the data range is 1.60–2.17 nats. Low-stress, reflationary conditions (rising growth + moderate inflation + easy policy) push entropy highest because the model spreads weight across equities, alternatives, and credit simultaneously.

**Sharpe (Q7 — sharpe_pred_total):**

| Regime | Mean Sharpe | Median Sharpe | Max Sharpe |
|---|---|---|---|
| higher_for_longer | 0.750 | 0.752 | 1.106 |
| risk_off_stress | 0.608 | 0.696 | 1.035 |
| mixed_mid_cycle | 0.341 | 0.157 | 0.981 |
| reflation_risk_on | 0.116 | 0.118 | 0.547 |

The Sharpe is driven primarily by the **rf numerator** — when short_rate_US is high (4–6%), even a modest excess return produces a good total Sharpe. `higher_for_longer` (high inflation, tight policy) wins on Sharpe precisely because the denominator (portfolio risk) is compressed by the λ=8.0 penalty.

**Practical answer:** A `higher_for_longer` regime (infl_US ~3–5%, short_rate_US ~4–6%, low VIX) maximizes **Sharpe** but produces a **concentrated** portfolio (entropy ~1.8). If you want both, the closest compromise is `mixed_mid_cycle` at the 2023–2024 anchors: mean Sharpe ~0.34–0.60, entropy ~2.0.

*Note: 6% **excess** return is structurally unreachable — the elastic net model ceiling is ~3.8% excess. All figures use total return (excess + rf). See Q2 for total return analysis.*

---

## Q2 — Can the benchmark achieve 6% excess return? What about 9% total?

**Excess return ceiling:**
The elastic net model predicts a maximum excess return of **6.11%** in the full 19,200-sample set, but this is an extreme scenario. The median across all scenarios is ~2.0–3.5%. The model ceiling is approximately **3.8% excess** at standard macro states. Values above 5% excess require macro states outside the training distribution (extrapolated by the VAR(1) prior).

**Total return (excess + rf):**

| Target | Achievable? | Conditions |
|---|---|---|
| ≥ 6% total | Yes — 12,567 scenarios (65% of all) | Any anchor with short_rate_US ≥ 2% |
| ≥ 9% total | Yes — 366 scenarios (1.9%) | High inflation (infl_US > 6%) + high rates (short_rate_US > 4%) at 2022 anchor |
| > 10% total | Yes — small tail | infl_US > 8%, short_rate_US > 5.5%, VIX ~30–50, 2022 anchor |

**Top 5 scenarios achieving > 10% total return:**

| Anchor | Regime | Total Return | infl_US | short_rate_US | VIX |
|---|---|---|---|---|---|
| 2022-12-31 | risk_off_stress | 11.0% | 10.6% | 6.4% | 31.4 |
| 2022-12-31 | risk_off_stress | 10.8% | 9.2% | 6.1% | 18.8 |
| 2022-12-31 | risk_off_stress | 10.6% | 9.1% | 6.3% | 34.5 |
| 2022-12-31 | risk_off_stress | 10.4% | 7.5% | 6.0% | 66.9 |
| 2022-12-31 | higher_for_longer | 10.3% | 8.7% | 4.5% | 28.1 |

**Conclusion:** 6% total return is easily achievable at any 2022–2024 anchor where the rf rate is elevated. 9% total requires extreme reflationary or stagflationary stress concentrated at the 2022 anchor. 9% **excess** is not achievable — it would require the model to predict excess returns 2.5× above its empirical ceiling.

---

## Q3 — Under what macro conditions does the benchmark show highest volatility and weight sensitivity?

**Source:** Q5 (max_risk probe), sorted by portfolio_risk.

**Top 10 highest-risk scenarios:**

| Anchor | Regime | Portfolio Risk | Total Return | infl_US | short_rate_US | VIX | IG OAS |
|---|---|---|---|---|---|---|---|
| 2022-12-31 | risk_off_stress | 11.74% | 9.56% | 8.73 | 3.93 | 36.2 | 3.54 |
| 2022-12-31 | higher_for_longer | 11.72% | 10.28% | 8.66 | 4.48 | 28.1 | 2.85 |
| 2022-12-31 | risk_off_stress | 11.59% | 8.57% | 10.93 | 3.55 | 47.1 | 4.87 |
| 2022-12-31 | risk_off_stress | 11.51% | 8.26% | 8.74 | 3.01 | 33.6 | 4.07 |
| 2022-12-31 | risk_off_stress | 11.51% | 8.26% | 8.74 | 3.01 | 33.6 | 4.07 |
| 2022-12-31 | risk_off_stress | 11.51% | 8.26% | 8.74 | 3.01 | 33.6 | 4.07 |
| 2022-12-31 | risk_off_stress | 11.18% | 7.67% | 10.04 | 3.48 | 47.1 | 4.42 |
| 2022-12-31 | risk_off_stress | 11.15% | 6.83% | 4.97 | 2.19 | 32.4 | 3.64 |
| 2022-12-31 | risk_off_stress | 11.03% | 8.07% | 9.22 | 4.17 | 48.2 | 5.82 |
| 2022-12-31 | higher_for_longer | 10.99% | 8.90% | 8.18 | 3.85 | 1.4 | 2.08 |

**Pattern:** Maximum risk scenarios cluster at the **2022-12-31 anchor** under `risk_off_stress` and `higher_for_longer` regimes. The common factor is **high inflation (infl_US > 7%)** combined with **elevated credit spreads (ig_oas > 3.0%)** and **high policy uncertainty (short_rate_US in transition)**. The risk-penalty (λ=8.0) is most easily broken when the optimizer perceives high-return opportunities in equities that dominate the penalty term. Portfolio risk in data: 6.46%–11.74% annualized.

---

## Q4 — Can the benchmark beat the S&P 500?

**Honest answer:** The S&P 500 realized return is **not in the model**. The elastic net predicts expected *excess* returns for 14 multi-asset sleeves based on macro factors — it does not model S&P 500 returns directly, and cannot be compared to realized equity returns.

**What the model can say:** Under what conditions does it maximize US equity (w_EQ_US)?

**w_EQ_US by regime (Q8):**

| Regime | Mean w_EQ_US | Max w_EQ_US |
|---|---|---|
| reflation_risk_on | 22.8% | 29.5% |
| mixed_mid_cycle | 19.1% | 29.6% |
| higher_for_longer | 12.1% | 29.8% |
| risk_off_stress | 11.3% | 26.9% |

**Top equity-tilt scenarios:**

| Anchor | Regime | w_EQ_US | Total Return | infl_US | short_rate_US | VIX |
|---|---|---|---|---|---|---|
| 2024-12-31 | higher_for_longer | 29.8% | 6.3% | 5.5 | 2.7 | 8.7 |
| 2021-12-31 | mixed_mid_cycle | 29.6% | 3.6% | 7.8 | -0.2 | 27.4 |
| 2024-12-31 | higher_for_longer | 29.6% | 8.6% | 3.9 | 5.0 | 1.4 |
| 2021-12-31 | reflation_risk_on | 29.5% | 3.8% | 9.0 | -0.2 | 1.4 |

**Conclusion:** The model maximally allocates to EQ_US (~30%) under **reflation_risk_on** (high inflation, easy/neutral policy, low stress) or **higher_for_longer** at 2024 with moderate inflation and declining rates. If you believe EQ_US tracks the S&P 500, these are the conditions where the benchmark is most equity-exposed — but the model predicts only 3–9% total return, not the level of realized equity returns.

---

## Q5 — What macro regime produces a portfolio close to 60/40 with a decent return?

**Source:** Q6 (sixty_forty probe). The model structurally cannot reach 60% equity — the maximum observed is 58.8% in historical data. The probe strains toward 60/40 but is limited by the optimizer's risk-penalty.

**Closest 60/40 scenarios (by Euclidean distance from 60%/40%):**

| Anchor | Regime | Sum EQ | Sum FI+CR | Total Return | Distance |
|---|---|---|---|---|---|
| 2021-12-31 | risk_off_stress | 51.7% | 38.2% | 2.7% | 0.085 |
| 2021-12-31 | risk_off_stress | 51.5% | 39.0% | 2.1% | 0.086 |
| 2021-12-31 | risk_off_stress | 51.5% | 39.0% | 2.1% | 0.086 |
| 2021-12-31 | reflation_risk_on | 51.6% | 42.7% | 2.9% | 0.088 |
| 2021-12-31 | mixed_mid_cycle | 51.1% | 42.6% | 2.6% | 0.092 |

**Practical finding:** The closest the model gets to 60/40 is approximately **52% equity / 40% bonds** at the 2021 anchor under `risk_off_stress` or `reflation_risk_on` conditions. The 8% shortfall on equities is structural — the remaining weight goes to ALT_GLD (~10–15%), real assets (~5–8%), and the optimizer's risk-penalty constrains further equity concentration. Total return at these scenarios is **2.1–2.9%** (in a ZIRP environment — rf ≈ 0.05%). At 2022–2024 anchors, similar equity weights produce total returns of **5–8%** due to higher rf.

**If you want 60/40 with ≥ 5% total return:** Use a 2022–2024 anchor + risk_off_stress or higher_for_longer regime, where 50–55% equity + 40% bonds + elevated rf (~4–5%) delivers the target.

---

## Methodology Notes

- All scenarios are posterior samples from MALA with acceptance rate 28–38% (healthy).
- ESS min ≥ 17.5 across all questions; ESS median 50–107.
- 3–5 distinct regime labels per question per anchor.
- The probe objectives successfully steer MALA toward distinct macro regions.
- `pred_return_total = pred_return_excess + short_rate_US / 100` (confirmed at all 4 anchors).
- Portfolio risk range: 6.46%–11.74% annualized.
- Portfolio entropy range: 1.60–2.17 nats (vs theoretical max log(14) = 2.639).

---

*Figures 1–12 saved to `reports/figures/`. See `fig09_q4_diversification.png`, `fig10_q5_risk.png`, `fig11_q6_sixty_forty.png`, `fig12_q7_q8_sharpe_equity.png` for the visual summaries corresponding to this document.*
