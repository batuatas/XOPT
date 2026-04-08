#!/usr/bin/env python3
"""
run_v4_scenario_final_FIXED.py

KEY CHANGES FROM ORIGINAL:
1. TRAIN_END is now PER-ANCHOR, matching the walk-forward logic
   - 2021-12-31 anchor: train_end = 2021-11-30 (model built on data through Nov 2021)
   - 2022-12-31 anchor: train_end = 2022-11-30 (model built on data through Nov 2022)
   - etc.
   This ensures MALA uses the SAME model that built the benchmark at each date.

2. Questions are reframed to be SMART and NEUTRAL:
   - Q1: "What macro conditions explain the gold allocation difference
          between 2021 (0%) and 2022+ (22%+)?"
   - Q2: "What macro conditions maximize portfolio concentration?"
   - Q3: "What macro conditions deliver 7% total return?"

3. Each question probes a REAL portfolio decision, not a bias.

USAGE:
  python -u scripts/scenario/run_v4_scenario_final_FIXED.py 2>&1 | tee /tmp/mala_fixed.log

RUNTIME: ~50 minutes (same as before, but now consistent with graphic05)
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
    SLEEVES_14, build_pipeline_at_date,
)
from xoptpoe_v4_scenario.probe_functions import (
    make_G_and_gradG,
    build_fast_gradG,
    benchmark_return_probe,
    gold_weight_probe,
    house_view_return_probe,
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

ELASTIC_NET_ALPHA = 0.005
ELASTIC_NET_L1    = 0.5

# ─────────────────────────────────────────────────────────────────────────
# KEY FIX: Per-anchor training dates (matching walk-forward logic)
# ─────────────────────────────────────────────────────────────────────────
ANCHOR_DATES = [
    pd.Timestamp("2021-12-31"),
    pd.Timestamp("2022-12-31"),
    pd.Timestamp("2023-12-31"),
    pd.Timestamp("2024-12-31"),
]

# Each anchor uses a model trained on data available ONE MONTH BEFORE
# (Dec portfolio is built on data through Nov)
TRAIN_END_BY_ANCHOR = {
    pd.Timestamp("2021-12-31"): pd.Timestamp("2021-11-30"),
    pd.Timestamp("2022-12-31"): pd.Timestamp("2022-11-30"),
    pd.Timestamp("2023-12-31"): pd.Timestamp("2023-11-30"),
    pd.Timestamp("2024-12-31"): pd.Timestamp("2024-11-30"),
}

TRAIN_END_VAR1    = pd.Timestamp("2016-02-29")  # VAR(1) prior is fixed

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

# ─────────────────────────────────────────────────────────────────────────
# SMART QUESTIONS (neutral, portfolio-driven)
# ─────────────────────────────────────────────────────────────────────────
QUESTIONS_CONFIG = {
    "Q1_gold_allocation": {
        "text": "What macro conditions explain the gold allocation difference "
                "(0% in 2021 vs 22%+ in 2022+)?",
        "type": "gold_weight_probe",
        "target_gold": None,  # Will be set per-anchor based on walk-forward weights
        "description": "Probe to understand what macro shifts drive gold allocation changes.",
    },
    "Q2_concentration": {
        "text": "What macro conditions maximize portfolio concentration "
                "(opposite of diversification)?",
        "type": "max_concentration",
        "description": "Probe to find regimes where the model concentrates in fewer sleeves.",
    },
    "Q3_seven_percent": {
        "text": "What macro conditions deliver 7% total return?",
        "type": "house_view_return_probe",
        "target_return": 0.07,
        "description": "Probe to find macro scenarios that hit a 7% total return target.",
    },
}

# Anchor truth — WILL BE RECOMPUTED per anchor with correct train_end
# (These are placeholders; actual values come from gate1 check)
ANCHOR_TRUTH_EXCESS = {
    pd.Timestamp("2021-12-31"): {"pred_return": 0.02014, "w_ALT_GLD": 0.081},
    pd.Timestamp("2022-12-31"): {"pred_return": 0.03296, "w_ALT_GLD": 0.223},
    pd.Timestamp("2023-12-31"): {"pred_return": 0.02819, "w_ALT_GLD": 0.224},
    pd.Timestamp("2024-12-31"): {"pred_return": 0.02236, "w_ALT_GLD": 0.232},
}

# ---------------------------------------------------------------------------
# Regime scoring (unchanged from original)
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
    print("v4 Scenario Engine — FIXED (per-anchor training)")
    print(f"MALA: N_STEPS={N_STEPS}, N_SEEDS={N_SEEDS}, ETA={ETA}, "
          f"WARMUP={WARMUP_FRAC:.0%}, THINNING={THINNING}")
    print(f"TAU: adaptive = G_m0 / {TAU_DIVISOR} (floor={TAU_MIN})")
    print("=" * 70)
    print()
    print("KEY CHANGE: Each anchor uses a model trained on data available")
    print("ONE MONTH BEFORE the anchor date (matching walk-forward logic).")
    print()
    print(f"  2021-12-31 anchor: train_end = 2021-11-30")
    print(f"  2022-12-31 anchor: train_end = 2022-11-30")
    print(f"  2023-12-31 anchor: train_end = 2023-11-30")
    print(f"  2024-12-31 anchor: train_end = 2024-11-30")
    print()
    print("This ensures MALA uses the SAME model that built graphic05.")
    print("=" * 70)
    print()

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
    precond    = scales ** 2

    all_results = []
    audit_rows  = []
    gate_rows   = []

    for anchor in ANCHOR_DATES:
        # ─────────────────────────────────────────────────────────────────
        # KEY FIX: Use per-anchor training date
        # ─────────────────────────────────────────────────────────────────
        train_end = TRAIN_END_BY_ANCHOR[anchor]

        print(f"\n{'='*60}\nAnchor: {anchor.date()} (train_end: {train_end.date()})\n{'='*60}")

        try:
            pipeline = build_pipeline_at_date(
                anchor_date=anchor,
                feature_master=fm,
                modeling_panel=mp,
                feature_manifest=fsm,
                feature_columns=feat_cols,
                excess_returns_monthly=excess_ret,
                elastic_net_alpha=ELASTIC_NET_ALPHA,
                elastic_net_l1=ELASTIC_NET_L1,
                train_end=train_end,  # ← PER-ANCHOR
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

        gate_ok, gate_msg = gate1_check(anchor, pred0_excess, gld0)
        status = "PASS" if gate_ok else "FAIL"
        print(f"\n  GATE 1 [{status}]: {gate_msg}")
        print(f"  RF={rf0*100:.3f}%  excess={pred0_excess*100:.2f}%  "
              f"total={pred0_total*100:.2f}%  gold={gld0*100:.1f}%")

        gate_rows.append({
            "anchor_date":            anchor.date(),
            "train_end":              train_end.date(),
            "gate1_status":           status,
            "pred_return_excess":     round(pred0_excess, 6),
            "rf_rate":                round(rf0, 6),
            "pred_return_total":      round(pred0_total, 6),
            "w_ALT_GLD":              round(gld0, 4),
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
            "anchor_regime":         r0["regime_label"],
            "gate1_status":          status,
        })

        questions = _build_questions(anchor, pipeline, m0, scales, var1_prior)

        for q in questions:
            qid     = q["question_id"]
            G_q     = q["G"]
            gradG_q = q["gradG"]

            G_m0 = G_q(m0)
            tau_effective = max(G_m0 / TAU_DIVISOR, TAU_MIN)

            print(f"\n  Q: {qid}")
            print(f"    {q['question_text']}")
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
                    "pred_return_excess":   round(ev_s["pred_return_excess"], 6),
                    "rf_rate":              round(ev_s["rf_rate"], 6),
                    "pred_return_total":    round(ev_s["pred_return_total"], 6),
                    "portfolio_risk":       round(ev_s["risk"], 6),
                    "portfolio_entropy":    round(ev_s["entropy"], 4),
                    "regime_label":         rs["regime_label"],
                    "regime_transition":    trans["regime_transition"],
                    "anchor_regime":        r0["regime_label"],
                    **{f"w_{s}": round(float(w_s[i]), 5) for i, s in enumerate(SLEEVES_14)},
                    **{col: round(float(m_s[i]), 5) for i, col in enumerate(MACRO_STATE_COLS)},
                }
                all_results.append(row)

    # ── Save outputs ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving outputs...")

    results_df = pd.DataFrame(all_results)
    gate_df    = pd.DataFrame(gate_rows)

    gate_df.to_csv(REPORTS / "scenario_anchor_truth_v4_FIXED.csv", index=False)

    if not results_df.empty:
        results_df.to_csv(REPORTS / "scenario_results_v4_FIXED.csv", index=False)
        print(f"  scenario_results_v4_FIXED.csv ({len(results_df)} rows)")
    else:
        print("  WARNING: no results produced")
        return

    print(f"\nDone.")
    print(f"\nGate 1 check summary:")
    print(gate_df[['anchor_date','train_end','gate1_status','w_ALT_GLD']].to_string(index=False))


def _build_questions(anchor, pipeline, m0, scales, var1_prior):
    """Build the 3 smart questions for this anchor."""
    questions = []

    # Q1: Gold allocation difference — what macro explains 0% (2021) vs 22%+ (2022+)?
    # Use the anchor m0 gold weight as a reference point
    w0 = pipeline.evaluate_at(m0)["w"]
    w_gld_m0 = float(w0[SLEEVES_14.index("ALT_GLD")])

    G_q1, gradG_q1 = gold_weight_probe(
        pipeline, m0, scales, l2reg=0.05, prior=var1_prior, var1_l2reg=0.1,
        sign=-1.0,  # maximize gold
    )
    questions.append({
        "question_id": "Q1_gold_allocation",
        "question_text": f"What macro conditions explain the gold allocation change "
                         f"(currently {w_gld_m0*100:.1f}%)?",
        "G": G_q1, "gradG": gradG_q1,
    })

    # Q2: Maximum concentration (opposite of diversification)
    G_q2, gradG_q2 = make_G_and_gradG(
        lambda m: -pipeline.evaluate_at(m)["entropy"]  # minimize entropy = maximize concentration
    )
    questions.append({
        "question_id": "Q2_concentration",
        "question_text": "What macro conditions maximize portfolio concentration?",
        "G": G_q2, "gradG": gradG_q2,
    })

    # Q3: 7% total return target
    G_q3, gradG_q3 = house_view_return_probe(
        pipeline, m0, house_view_total=0.07, scales=scales,
        l2reg=0.05, prior=var1_prior, var1_l2reg=0.1,
    )
    questions.append({
        "question_id": "Q3_seven_percent_return",
        "question_text": "What macro conditions deliver 7% total return?",
        "G": G_q3, "gradG": gradG_q3,
    })

    return questions


if __name__ == "__main__":
    run_scenario()
