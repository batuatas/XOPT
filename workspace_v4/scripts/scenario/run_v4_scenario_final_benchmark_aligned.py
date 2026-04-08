#!/usr/bin/env python3
"""
run_v4_scenario_final.py  —  v3

Changes from v2
---------------
1. Results row stores rf_rate, pred_return_excess, pred_return_total
   pred_return kept as backward-compat alias for excess.

2. audit_rows stores m0_rf_rate and m0_pred_return_total
   using the TRUE m0 value from feature_master (not perturbed scenario state).

3. Q3 probe: house_view_return_probe() now receives house_view_total=0.07
   (7% total return) instead of 0.05 excess.
   Implied target_excess at each anchor:
     2021: 7.0% - 0.05% = 6.95%  (ZIRP — hard, probe will explore boundary)
     2022: 7.0% - 4.15% = 2.85%  (achievable)
     2023: 7.0% - 5.27% = 1.73%  (achievable)
     2024: 7.0% - 4.42% = 2.58%  (achievable)

4. Gate 1 truth table updated with total return column.

5. MALA parameters unchanged: N_STEPS=1000, N_SEEDS=5, ETA=0.05,
   WARMUP=40%, THINNING=5, TAU=adaptive, PRECOND=scales**2.
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORKSPACE = Path(__file__).resolve().parent.parent.parent
REPO_SRC  = WORKSPACE.parent.parent / "src"

for p in [str(WORKSPACE / "src"), str(REPO_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from xoptpoe_v4_scenario.state_space import (
    MACRO_STATE_COLS, STATE_DIM, box_constraints, load_state, state_scales,
)
from xoptpoe_v4_scenario.var1_prior import VAR1Prior
from xoptpoe_v4_scenario.pipeline import (
    SLEEVES_14,
    benchmark_train_end,
    build_benchmark_aligned_pipeline_at_date,
)
from xoptpoe_v4_scenario.probe_functions import (
    gold_weight_probe,
    house_view_return_probe,
    make_G_and_gradG,
    build_fast_gradG,
)
from xoptpoe_v4_scenario.sampler import (
    run_mala_chains, thin_only, filter_trajectories,
    compute_effective_sample_size,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_REFS = WORKSPACE / "data_refs"
REPORTS   = WORKSPACE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

NFCI_PATH = WORKSPACE / "NFCI (1).csv"

BENCHMARK_EXPERIMENT   = "elastic_net__core_plus_interactions__separate_60"
BENCHMARK_METRICS_PATH   = REPORTS /  "benchmark/v4_prediction_benchmark_metrics.csv"
TRAIN_END_VAR1           = pd.Timestamp("2016-02-29")

ANCHOR_DATES = [
    pd.Timestamp("2021-12-31"),
    pd.Timestamp("2022-12-31"),
    pd.Timestamp("2023-12-31"),
    pd.Timestamp("2024-12-31"),
]

# MALA parameters (locked)
N_SEEDS      = 5
N_STEPS      = 1000
WARMUP_FRAC  = 0.40
ETA          = 0.05
THINNING     = 5
TAU_DIVISOR  = 5.0
TAU_MIN      = 1.0

# Gate tolerances
GATE_TOL_RET = 0.005
GATE_TOL_GLD = 0.025

# Anchor truth note
# -----------------
# The old truth table came from the frozen-Feb-2021 scenario reconstruction.
# Once we switch to benchmark-aligned per-anchor refits, those values are no
# longer a valid hard gate. Leave the table empty on the first aligned rerun;
# repopulate later from the new anchor audit if you still want a strict gate.
ANCHOR_TRUTH_EXCESS = {}

# True m0 rf rates from feature_master (confirmed via check_rf_sources.py)
# short_rate_US in % units -> divide by 100 for decimal
ANCHOR_RF_PCT = {
    pd.Timestamp("2021-12-31"): 0.050,   # 0.05%
    pd.Timestamp("2022-12-31"): 4.150,   # 4.15%
    pd.Timestamp("2023-12-31"): 5.270,   # 5.27%
    pd.Timestamp("2024-12-31"): 4.420,   # 4.42%
}

# House view total return target for Q3 probe
HOUSE_VIEW_TOTAL = 0.07   # 7% annualized total return


# ---------------------------------------------------------------------------
# Regime scoring (unchanged)
# ---------------------------------------------------------------------------

def score_regime_dimensions(m: np.ndarray, thresholds: dict) -> dict:
    mac_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}
    def get(col):
        return float(m[mac_idx[col]]) if col in mac_idx else float("nan")

    ig_oas     = get("ig_oas")
    vix        = get("vix")
    infl_us    = get("infl_US")
    short_us   = get("short_rate_US")
    us_real10y = get("us_real10y")
    unemp_us   = get("unemp_US")

    def th(k, default=None):
        return thresholds.get(k, default)

    unemp_p25 = th("unemp_US_p25", 5.0)
    unemp_p75 = th("unemp_US_p75", 7.0)
    growth = "high" if unemp_us <= unemp_p25 else \
             "low"  if unemp_us >= unemp_p75 else "neutral"

    infl_p25 = th("infl_US_p25", 1.5)
    infl_p75 = th("infl_US_p75", 3.0)
    inflation = "low"  if infl_us <= infl_p25 else \
                "high" if infl_us >= infl_p75 else "neutral"

    short_p25   = th("short_rate_US_p25", 0.5)
    short_p75   = th("short_rate_US_p75", 3.5)
    real10y_p25 = th("us_real10y_p25", -0.5)
    real10y_p75 = th("us_real10y_p75", 1.0)
    policy_tight = (short_us >= short_p75) or (us_real10y >= real10y_p75)
    policy_easy  = (short_us <= short_p25) and (us_real10y <= real10y_p25)
    policy = "tight" if policy_tight else "easy" if policy_easy else "neutral"

    ig_p75  = th("ig_oas_p75", 1.5);  ig_p90  = th("ig_oas_p90", 2.5)
    vix_p75 = th("vix_p75", 22.0);    vix_p90 = th("vix_p90", 30.0)
    stress_high = (ig_oas > ig_p90) or (vix > vix_p90)
    stress_mod  = (ig_oas > ig_p75) or (vix > vix_p75)
    stress = "high" if stress_high else "moderate" if stress_mod else "low"

    fin_cond = "loose" if (stress == "low" and policy == "easy") else \
               "tight" if (stress in ("high", "moderate") or policy == "tight") \
               else "neutral"

    if   stress_high and growth == "low":                                    label = "high_stress_defensive"
    elif stress_high:                                                        label = "risk_off_stress"
    elif inflation == "high" and policy == "tight":                          label = "higher_for_longer"
    elif inflation == "low"  and policy == "easy" and growth in ("neutral","high"): label = "soft_landing"
    elif inflation in ("high","neutral") and policy in ("neutral","easy") and stress == "low": label = "reflation_risk_on"
    elif inflation == "low"  and growth == "low" and stress in ("low","moderate"): label = "disinflationary_slowdown"
    elif stress == "moderate" and growth == "low":                           label = "risk_off_stress"
    else:                                                                    label = "mixed_mid_cycle"

    return {
        "dim_growth": growth, "dim_inflation": inflation,
        "dim_policy": policy, "dim_stress": stress,
        "dim_fin_cond": fin_cond, "regime_label": label,
        "sig_ig_oas": round(ig_oas, 3), "sig_vix": round(vix, 2),
        "sig_infl_US": round(infl_us, 3), "sig_short_rate_US": round(short_us, 3),
        "sig_us_real10y": round(us_real10y, 3), "sig_unemp_US": round(unemp_us, 3),
    }


def describe_regime_transition(m0, m_scenario, r0, rs, macro_cols):
    shifts = {col: round(float(m_scenario[i] - m0[i]), 4)
              for i, col in enumerate(macro_cols)
              if abs(float(m_scenario[i] - m0[i])) > 1e-8}
    top3 = dict(sorted(shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
    transition = "same_regime" if r0["regime_label"] == rs["regime_label"] \
                 else f"{r0['regime_label']} -> {rs['regime_label']}"
    dim_changes = [
        f"{d}: {r0.get(d)} -> {rs.get(d)}"
        for d in ["dim_growth","dim_inflation","dim_policy","dim_stress","dim_fin_cond"]
        if r0.get(d) != rs.get(d)
    ]
    return {
        "regime_transition": transition,
        "anchor_regime": r0["regime_label"],
        "scenario_regime": rs["regime_label"],
        "top_shifts": top3,
        "dim_changes": "; ".join(dim_changes) if dim_changes else "no_dimension_change",
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict:
    print("Loading data artifacts...")
    fm  = pd.read_parquet(DATA_REFS / "feature_master_monthly.parquet")
    mp  = pd.read_parquet(DATA_REFS / "modeling_panel_hstack.parquet")
    fsm = pd.read_csv(DATA_REFS / "feature_set_manifest.csv")
    tp  = pd.read_parquet(DATA_REFS / "target_panel_long_horizon.parquet")
    for df in [fm, mp, tp]:
        df["month_end"] = pd.to_datetime(df["month_end"])
    feat_cols = fsm[fsm["include_core_plus_interactions"] == 1]["feature_name"].tolist()
    print(f"  feature_master: {len(fm)} rows | modeling_panel: {len(mp)} rows")
    print(f"  features: {len(feat_cols)}")
    return {"feature_master": fm, "modeling_panel": mp,
            "feature_manifest": fsm, "target_panel": tp,
            "feature_columns": feat_cols}


def build_excess_returns(fm: pd.DataFrame) -> pd.DataFrame:
    """Use the benchmark monthly excess-return history when available."""
    try:
        from xoptpoe_v4_models.data import load_modeling_inputs
        inputs = load_modeling_inputs(WORKSPACE, feature_set_name="core_baseline")
        hist = inputs.monthly_excess_history
        if hist is not None:
            out = hist.copy()
            out.index = pd.to_datetime(out.index)
            avail = [s for s in SLEEVES_14 if s in out.columns]
            return out[avail].sort_index()
    except Exception as e:
        print(f"  WARNING: benchmark excess history unavailable; falling back to feature_master ret_1m_lag ({e})")

    if "ret_1m_lag" in fm.columns:
        pivot = fm.pivot_table(index="month_end", columns="sleeve_id",
                               values="ret_1m_lag", aggfunc="first")
        avail = [s for s in SLEEVES_14 if s in pivot.columns]
        return pivot[avail].sort_index()
    dates = pd.to_datetime(fm["month_end"].unique())
    return pd.DataFrame(0.0, index=sorted(dates), columns=list(SLEEVES_14))


def compute_thresholds(fm, nfci_df=None):
    from xoptpoe_v4_scenario.regime import compute_regime_thresholds
    return compute_regime_thresholds(fm, nfci_df)


def gate1_check(anchor, pred_return_excess, w_gld):
    truth = ANCHOR_TRUTH_EXCESS.get(anchor, {})
    if not truth:
        return True, "no_truth_reference"
    d_ret = abs(pred_return_excess - truth["pred_return"])
    d_gld = abs(w_gld - truth["w_ALT_GLD"])
    ok = (d_ret <= GATE_TOL_RET) and (d_gld <= GATE_TOL_GLD)
    msg = (f"excess_ret: {pred_return_excess:.4%} vs truth {truth['pred_return']:.4%} "
           f"(Δ={d_ret:.4%}) | w_GLD: {w_gld:.3f} vs truth {truth['w_ALT_GLD']:.3f} "
           f"(Δ={d_gld:.3f})")
    return ok, msg


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_scenario() -> None:
    print("=" * 70)
    print("v4 Scenario Engine — benchmark-aligned")
    print(f"MALA: N_STEPS={N_STEPS}, N_SEEDS={N_SEEDS}, ETA={ETA}, "
          f"WARMUP={WARMUP_FRAC:.0%}, THINNING={THINNING}")
    print(f"TAU: adaptive = G_m0 / {TAU_DIVISOR} (floor={TAU_MIN})")
    print(f"Benchmark experiment: {BENCHMARK_EXPERIMENT}")
    print(f"Q3 target: {HOUSE_VIEW_TOTAL:.0%} TOTAL return (excess + rf)")
    print("=" * 70)

    data = load_data()
    fm, mp, fsm = data["feature_master"], data["modeling_panel"], data["feature_manifest"]
    feat_cols = data["feature_columns"]

    nfci_df = None
    if NFCI_PATH.exists():
        from xoptpoe_v4_scenario.regime import load_nfci
        nfci_df = load_nfci(NFCI_PATH)

    thresholds = compute_thresholds(fm, nfci_df)

    print(f"\nFitting VAR(1) prior (train_end={TRAIN_END_VAR1.date()})...")
    var1_prior = VAR1Prior.fit_from_feature_master(
        fm, MACRO_STATE_COLS, train_end=TRAIN_END_VAR1
    )

    excess_ret = build_excess_returns(fm)
    scales     = state_scales(fm, train_end=TRAIN_END_VAR1)
    a_box, b_box = box_constraints(fm, slack_multiplier=1.0)
    precond    = scales ** 2   # diagonal preconditioner

    all_results = []
    audit_rows  = []
    gate_rows   = []

    for anchor in ANCHOR_DATES:
        train_end = benchmark_train_end(anchor)
        print(f"\n{'='*60}\nAnchor: {anchor.date()} | benchmark_train_end={train_end.date()}\n{'='*60}")

        try:
            pipeline = build_benchmark_aligned_pipeline_at_date(
                anchor_date=anchor,
                feature_master=fm,
                modeling_panel=mp,
                feature_manifest=fsm,
                feature_columns=feat_cols,
                excess_returns_monthly=excess_ret,
                benchmark_metrics_path=BENCHMARK_METRICS_PATH,
                experiment_name=BENCHMARK_EXPERIMENT,
                train_end=train_end,
                sleeve_order=SLEEVES_14,
            )
        except Exception as e:
            print(f"  ERROR building pipeline: {e}")
            continue

        m0, _ = load_state(anchor, fm, SLEEVES_14, modeling_panel=mp, horizon_months=60)
        ev0   = pipeline.evaluate_at(m0)
        w0    = ev0["w"]

        pred0_excess = ev0["pred_return_excess"]
        pred0_total  = ev0["pred_return_total"]
        rf0          = ev0["rf_rate"]
        gld0         = float(w0[SLEEVES_14.index("ALT_GLD")])

        # Cross-check: rf from evaluate_at vs locked ANCHOR_RF_PCT
        rf0_locked = ANCHOR_RF_PCT.get(anchor, 0.0) / 100.0
        if abs(rf0 - rf0_locked) > 0.001:
            print(f"  WARNING: rf mismatch — evaluate_at={rf0:.4f}, "
                  f"locked={rf0_locked:.4f}")

        gate_ok, gate_msg = gate1_check(anchor, pred0_excess, gld0)
        status = "PASS" if gate_ok else "FAIL"
        print(f"\n  GATE 1 [{status}]: {gate_msg}")
        print(f"  RF={rf0*100:.3f}%  excess={pred0_excess*100:.2f}%  "
              f"total={pred0_total*100:.2f}%")

        gate_rows.append({
            "anchor_date":            anchor.date(),
            "train_end":              train_end.date(),
            "gate1_status":           status,
            "pred_return_excess":     round(pred0_excess, 6),
            "rf_rate":                round(rf0, 6),
            "pred_return_total":      round(pred0_total, 6),
            "truth_pred_return":      ANCHOR_TRUTH_EXCESS.get(anchor, {}).get("pred_return"),
            "delta_pred_return":      round(pred0_excess -
                                       ANCHOR_TRUTH_EXCESS.get(anchor, {}).get("pred_return", pred0_excess), 6),
            "w_ALT_GLD":              round(gld0, 4),
            "truth_w_ALT_GLD":        ANCHOR_TRUTH_EXCESS.get(anchor, {}).get("w_ALT_GLD"),
            "detail":                 gate_msg,
        })

        if not gate_ok:
            print("  *** GATE 1 FAILED — skipping anchor ***")
            continue

        r0 = score_regime_dimensions(m0, thresholds)
        print(f"  Regime: {r0['regime_label']} | "
              f"growth={r0['dim_growth']}, infl={r0['dim_inflation']}, "
              f"policy={r0['dim_policy']}, stress={r0['dim_stress']}")

        audit_rows.append({
            "anchor_date":           anchor.date(),
            "train_end":             train_end.date(),
            "pred_return_excess":    round(pred0_excess, 6),
            "rf_rate":               round(rf0, 6),
            "pred_return_total":     round(pred0_total, 6),
            "w_ALT_GLD_m0":          round(gld0, 4),
            "w_EQ_US_m0":            round(float(w0[SLEEVES_14.index("EQ_US")]), 4),
            "entropy_m0":            round(ev0["entropy"], 4),
            "anchor_regime":         r0["regime_label"],
            "dim_growth":            r0["dim_growth"],
            "dim_inflation":         r0["dim_inflation"],
            "dim_policy":            r0["dim_policy"],
            "dim_stress":            r0["dim_stress"],
            "gate1_status":          status,
            **{f"w0_{s}": round(float(w0[i]), 4) for i, s in enumerate(SLEEVES_14)},
        })

        questions = _build_questions(anchor, pipeline, m0, scales, var1_prior)

        for q in questions:
            qid     = q["question_id"]
            G_q     = q["G"]
            gradG_q = q["gradG"]

            G_m0 = G_q(m0)
            tau_effective = max(G_m0 / TAU_DIVISOR, TAU_MIN)

            print(f"\n  Q: {qid}")
            print(f"    G(m0)={G_m0:.4f} | tau={tau_effective:.3f}")

            trajectories, acc_rates = run_mala_chains(
                G=G_q, gradG=gradG_q, m0=m0,
                a=a_box, b=b_box,
                n_seeds=N_SEEDS, n_steps=N_STEPS,
                eta=ETA, tau=tau_effective,
                warmup_frac=WARMUP_FRAC,
                seed=hash(qid + str(anchor)) % (2**31),
                precond=precond,
                verbose=True,
            )

            mean_acc = float(np.mean(acc_rates))
            all_samples = thin_only(trajectories, thinning=THINNING)
            n_posterior = len(all_samples)

            if n_posterior == 0:
                print("    No samples — skip")
                continue

            ess = compute_effective_sample_size(all_samples)
            ess_min    = float(ess.min())
            ess_median = float(np.median(ess))
            print(f"    n={n_posterior} | acc={mean_acc:.2%} | "
                  f"ESS min={ess_min:.1f} median={ess_median:.1f}")

            for m_s in all_samples:
                ev_s  = pipeline.evaluate_at(m_s)
                w_s   = ev_s["w"]
                rs    = score_regime_dimensions(m_s, thresholds)
                trans = describe_regime_transition(m0, m_s, r0, rs, MACRO_STATE_COLS)

                row = {
                    "question_id":          qid,
                    "anchor_date":          anchor.date(),
                    "train_end":            train_end.date(),
                    "G_value":              round(float(G_q(m_s)), 6),
                    # Return columns — both excess and total
                    "pred_return":          round(ev_s["pred_return_excess"], 6),  # backward compat
                    "pred_return_excess":   round(ev_s["pred_return_excess"], 6),
                    "rf_rate":              round(ev_s["rf_rate"], 6),
                    "pred_return_total":    round(ev_s["pred_return_total"], 6),
                    # Risk/portfolio
                    "portfolio_risk":       round(ev_s["risk"], 6),
                    "portfolio_entropy":    round(ev_s["entropy"], 4),
                    "sharpe_pred":          round(ev_s["sharpe_pred"], 4),
                    "sharpe_pred_total":    round(ev_s["sharpe_pred_total"], 4),
                    # Regime
                    "regime_label":         rs["regime_label"],
                    "dim_growth":           rs["dim_growth"],
                    "dim_inflation":        rs["dim_inflation"],
                    "dim_policy":           rs["dim_policy"],
                    "dim_stress":           rs["dim_stress"],
                    "dim_fin_cond":         rs["dim_fin_cond"],
                    "regime_transition":    trans["regime_transition"],
                    "anchor_regime":        r0["regime_label"],
                    "dim_changes":          trans["dim_changes"],
                    "top_shift_1_var":      list(trans["top_shifts"].keys())[0] if trans["top_shifts"] else "",
                    "top_shift_1_val":      list(trans["top_shifts"].values())[0] if trans["top_shifts"] else 0.0,
                    "top_shift_2_var":      list(trans["top_shifts"].keys())[1] if len(trans["top_shifts"]) > 1 else "",
                    "top_shift_2_val":      list(trans["top_shifts"].values())[1] if len(trans["top_shifts"]) > 1 else 0.0,
                    "top_shift_3_var":      list(trans["top_shifts"].keys())[2] if len(trans["top_shifts"]) > 2 else "",
                    "top_shift_3_val":      list(trans["top_shifts"].values())[2] if len(trans["top_shifts"]) > 2 else 0.0,
                    # Signals
                    "sig_ig_oas":           rs["sig_ig_oas"],
                    "sig_vix":              rs["sig_vix"],
                    "sig_infl_US":          rs["sig_infl_US"],
                    "sig_short_rate_US":    rs["sig_short_rate_US"],
                    "sig_us_real10y":       rs["sig_us_real10y"],
                    "sig_unemp_US":         rs["sig_unemp_US"],
                    # MALA diagnostics
                    "tau_effective":        round(tau_effective, 4),
                    "mean_acceptance_rate": round(mean_acc, 4),
                    "ess_min":              round(ess_min, 1),
                    "ess_median":           round(ess_median, 1),
                    # Macro state
                    **{col: round(float(m_s[i]), 5) for i, col in enumerate(MACRO_STATE_COLS)},
                    # Weights
                    **{f"w_{s}": round(float(w_s[i]), 5) for i, s in enumerate(SLEEVES_14)},
                }
                all_results.append(row)

            # Quick summary
            rows_q = [r for r in all_results
                      if r["question_id"] == qid
                      and str(r["anchor_date"]) == str(anchor.date())]
            rets_exc   = [r["pred_return_excess"] for r in rows_q]
            rets_total = [r["pred_return_total"]  for r in rows_q]
            glds       = [r["w_ALT_GLD"]          for r in rows_q]
            regs       = [r["regime_label"]        for r in rows_q]
            print(f"    excess={np.mean(rets_exc)*100:.2f}%±{np.std(rets_exc)*100:.2f}%  "
                  f"total={np.mean(rets_total)*100:.2f}%±{np.std(rets_total)*100:.2f}%  "
                  f"GLD={np.mean(glds):.3f}")
            print(f"    Regimes: {dict(Counter(regs).most_common(3))}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving outputs...")

    results_df = pd.DataFrame(all_results)
    audit_df   = pd.DataFrame(audit_rows)
    gate_df    = pd.DataFrame(gate_rows)

    all_passed = (gate_df["gate1_status"] == "PASS").all() if not gate_df.empty else False

    gate_df.to_csv(REPORTS / "scenario_anchor_truth_v4.csv", index=False)

    if not results_df.empty:
        results_df.to_csv(REPORTS / "scenario_results_v4.csv", index=False)
        print(f"  scenario_results_v4.csv ({len(results_df)} rows)")
    else:
        print("  WARNING: no results produced")
        return

    _write_regime_summary(results_df,          REPORTS / "scenario_regime_summary_v4.csv")
    _write_selected_cases(results_df,           REPORTS / "scenario_selected_cases_v4.csv")
    _write_question_manifest(results_df,        REPORTS / "scenario_question_manifest_v4.csv")
    _write_portfolio_response_summary(results_df, REPORTS / "scenario_portfolio_response_summary_v4.csv")
    _write_regime_transition_summary(results_df, REPORTS / "scenario_regime_transition_summary_v4.csv")
    _write_selected_questions(results_df,       REPORTS / "scenario_selected_questions_v4.csv")
    _write_mala_diagnostics(results_df,         REPORTS / "scenario_mala_diagnostics_v4.csv")

    print(f"\nDone. Gate 1 all_passed={all_passed}")


# ---------------------------------------------------------------------------
# Question builder — v3
# ---------------------------------------------------------------------------

def _build_questions(anchor, pipeline, m0, scales, var1_prior):
    questions = []

    # Q1: Gold maximization
    G_gold, gradG_gold = gold_weight_probe(
        pipeline, m0, scales, l2reg=0.05,
        prior=var1_prior, var1_l2reg=0.1, sign=-1.0,
    )
    questions.append({
        "question_id": "Q1_gold_favorable",
        "question_text": "What macro regime maximizes ALT_GLD allocation?",
        "G": G_gold, "gradG": gradG_gold,
    })

    # Q2: EW deviation
    def G_ew_dev(m: np.ndarray) -> float:
        ev  = pipeline.evaluate_at(m)
        w   = ev["w"]
        mu  = ev["mu_hat"]
        n   = len(mu)
        ew  = np.ones(n) / n
        task = -(float(w @ mu) - float(ew @ mu))
        reg  = var1_prior.regularizer(m, m0, l2reg=0.1)
        return task + reg

    gradG_ew = build_fast_gradG(pipeline, G_ew_dev, m0, var1_prior, 0.1)
    questions.append({
        "question_id": "Q2_ew_deviation",
        "question_text": "Under what macro conditions does the benchmark tilt most away from equal weight?",
        "G": G_ew_dev, "gradG": gradG_ew,
    })

    # Q3: House-view total return = 7%
    rf_at_m0 = float(m0[list(MACRO_STATE_COLS).index("short_rate_US")]) / 100.0
    target_excess = HOUSE_VIEW_TOTAL - rf_at_m0
    print(f"  Q3 target: {HOUSE_VIEW_TOTAL:.0%} total | "
          f"rf={rf_at_m0*100:.3f}% | implied excess target={target_excess*100:.3f}%")

    G_hv, gradG_hv = house_view_return_probe(
        pipeline, m0,
        house_view_total=HOUSE_VIEW_TOTAL,
        scales=scales, l2reg=0.05,
        prior=var1_prior, var1_l2reg=0.1,
    )
    questions.append({
        "question_id": "Q3_house_view_7pct_total",
        "question_text": f"What macro regime makes benchmark predict ~7% total return? "
                         f"(implied excess target at m0: {target_excess*100:.2f}%)",
        "G": G_hv, "gradG": gradG_hv,
    })

    return questions


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_regime_summary(results_df, path):
    rows = []
    for (qid, anchor), grp in results_df.groupby(["question_id", "anchor_date"]):
        for regime, rgrp in grp.groupby("regime_label"):
            rows.append({
                "question_id": qid, "anchor_date": anchor,
                "regime_label": regime,
                "count": len(rgrp),
                "share": round(len(rgrp) / len(grp), 3),
                "mean_pred_return_excess": round(float(rgrp["pred_return_excess"].mean()), 5),
                "mean_pred_return_total":  round(float(rgrp["pred_return_total"].mean()), 5),
                "mean_rf_rate":            round(float(rgrp["rf_rate"].mean()), 5),
                "std_pred_return_total":   round(float(rgrp["pred_return_total"].std()), 5),
                "mean_entropy":            round(float(rgrp["portfolio_entropy"].mean()), 4),
                "mean_w_ALT_GLD":          round(float(rgrp["w_ALT_GLD"].mean()), 4) if "w_ALT_GLD" in rgrp else None,
                "mean_w_EQ_US":            round(float(rgrp["w_EQ_US"].mean()), 4)   if "w_EQ_US"  in rgrp else None,
                "mean_w_FI_UST":           round(float(rgrp["w_FI_UST"].mean()), 4)  if "w_FI_UST" in rgrp else None,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_selected_cases(results_df, path):
    rows = []
    for (qid, anchor, regime), grp in results_df.groupby(
            ["question_id", "anchor_date", "regime_label"]):
        best = grp.loc[grp["G_value"].idxmin()]
        rows.append(best)
    pd.DataFrame(rows).reset_index(drop=True).to_csv(path, index=False)


def _write_question_manifest(results_df, path):
    rows = []
    for (qid, anchor), grp in results_df.groupby(["question_id", "anchor_date"]):
        rows.append({
            "question_id":             qid,
            "anchor_date":             anchor,
            "n_valid_samples":         len(grp),
            "regime_diversity":        grp["regime_label"].nunique(),
            "mean_pred_return_excess": round(float(grp["pred_return_excess"].mean()), 5),
            "mean_pred_return_total":  round(float(grp["pred_return_total"].mean()), 5),
            "mean_rf_rate":            round(float(grp["rf_rate"].mean()), 5),
            "std_pred_return_total":   round(float(grp["pred_return_total"].std()), 5),
            "mean_entropy":            round(float(grp["portfolio_entropy"].mean()), 4),
            "dominant_regime":         grp["regime_label"].value_counts().index[0],
            "dominant_regime_share":   round(float(grp["regime_label"].value_counts().iloc[0] / len(grp)), 3),
            "tau_effective":           round(float(grp["tau_effective"].iloc[0]), 4) if "tau_effective" in grp else None,
            "mean_acceptance_rate":    round(float(grp["mean_acceptance_rate"].mean()), 4) if "mean_acceptance_rate" in grp else None,
            "ess_min":                 round(float(grp["ess_min"].mean()), 1) if "ess_min" in grp else None,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_mala_diagnostics(results_df, path):
    rows = []
    for (qid, anchor), grp in results_df.groupby(["question_id", "anchor_date"]):
        row = {"question_id": qid, "anchor_date": anchor, "n_samples": len(grp)}
        for col in ["tau_effective", "mean_acceptance_rate", "ess_min", "ess_median"]:
            if col in grp.columns:
                row[col] = round(float(grp[col].iloc[0]), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_portfolio_response_summary(results_df, path):
    rows = []
    for (qid, anchor), grp in results_df.groupby(["question_id", "anchor_date"]):
        weight_cols = [c for c in grp.columns if c.startswith("w_")]
        row = {
            "question_id": qid, "anchor_date": anchor,
            "n_samples": len(grp),
            "mean_pred_return_excess":  round(float(grp["pred_return_excess"].mean()), 5),
            "std_pred_return_excess":   round(float(grp["pred_return_excess"].std()), 5),
            "mean_rf_rate":             round(float(grp["rf_rate"].mean()), 5),
            "mean_pred_return_total":   round(float(grp["pred_return_total"].mean()), 5),
            "std_pred_return_total":    round(float(grp["pred_return_total"].std()), 5),
            "p25_pred_return_total":    round(float(grp["pred_return_total"].quantile(0.25)), 5),
            "p75_pred_return_total":    round(float(grp["pred_return_total"].quantile(0.75)), 5),
            "mean_entropy":             round(float(grp["portfolio_entropy"].mean()), 4),
            "mean_sharpe_pred_total":   round(float(grp["sharpe_pred_total"].mean()), 4),
        }
        for wc in weight_cols:
            row[f"mean_{wc}"] = round(float(grp[wc].mean()), 4)
            row[f"std_{wc}"]  = round(float(grp[wc].std()),  4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_regime_transition_summary(results_df, path):
    rows = []
    for (qid, anchor), grp in results_df.groupby(["question_id", "anchor_date"]):
        for trans, tgrp in grp.groupby("regime_transition"):
            rows.append({
                "question_id": qid, "anchor_date": anchor,
                "regime_transition": trans,
                "count": len(tgrp),
                "share": round(len(tgrp) / len(grp), 3),
                "mean_pred_return_total": round(float(tgrp["pred_return_total"].mean()), 5),
                "mean_w_ALT_GLD": round(float(tgrp["w_ALT_GLD"].mean()), 4) if "w_ALT_GLD" in tgrp.columns else None,
                "dominant_top_shift": tgrp["top_shift_1_var"].value_counts().index[0]
                    if "top_shift_1_var" in tgrp.columns and len(tgrp) > 0 else "",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_selected_questions(results_df, path):
    scores = []
    for qid, grp in results_df.groupby("question_id"):
        diversity  = grp["regime_label"].nunique()
        ret_range  = float(grp["pred_return_total"].max() - grp["pred_return_total"].min())
        ret_std    = float(grp["pred_return_total"].std())
        n_samples  = len(grp)
        trans_div  = grp["regime_transition"].nunique() if "regime_transition" in grp.columns else 1
        score = diversity * ret_range * min(n_samples / 50, 3.0) * trans_div
        scores.append({
            "question_id": qid, "score": round(score, 4),
            "n_samples": n_samples, "regime_diversity": diversity,
            "transition_diversity": trans_div,
            "return_range_total": round(ret_range, 5),
            "return_std_total": round(ret_std, 5),
            "mean_pred_return_total": round(float(grp["pred_return_total"].mean()), 5),
        })
    scores_df = pd.DataFrame(scores).sort_values("score", ascending=False).head(3)
    scores_df["rank"] = range(1, len(scores_df) + 1)
    scores_df.to_csv(path, index=False)


if __name__ == "__main__":
    run_scenario()
