#!/usr/bin/env python3
"""
run_v4_scenario_mapped.py  —  v4 scenario engine with curated anchor→question map

Each question is assigned to the anchor(s) where it is most narratively
informative, based on the realized allocation path from hero_wealth_allocation_v4.

Anchor–question rationale (from the wealth/allocation chart):
─────────────────────────────────────────────────────────────
2020-12-31  Post-COVID, rf≈0, gold just emerging, heavy FI/UST, equity moderate
  → Q9  flight_to_safety   — COVID memory fresh, what macro causes another panic?
  → Q7  stretch_excess     — rf≈0 so excess≈total, what macro lifts returns?
  → Q8  more_equity        — equity was moderate, what tilts toward equities?

2021-12-31  Inflationary expansion, rates still low, gold growing, equity rising
  → Q1  more_gold          — gold was building, what pushes it further?
  → Q5  more_diversified   — portfolio starting to concentrate in a few sleeves
  → Q7  stretch_excess     — low rates era, excess return is the key metric

2022-12-31  Rate hikes began, gold ≈25%, regime shift to higher_for_longer
  → Q4  less_gold          — gold surged, what would reduce it?
  → Q6  classic_60_40      — allocation very far from traditional, what pulls it back?
  → Q8  more_equity        — equity got compressed, what scenario restores it?

2023-12-31  Gold ≈30%, higher_for_longer, rf=5.27%
  → Q3  house_view_6pct    — rf=5.27%, 6% total = only 0.73% excess: easily achievable?
  → Q4  less_gold          — gold at 30%, what brings it down?
  → Q11 max_sharpe         — with high rf, where is the risk-adjusted sweet spot?

2024-12-31  Gold ≈33% (peak), concentrated portfolio, rf=4.42%
  → Q4  less_gold          — gold at historical peak, what reverses it?
  → Q6  classic_60_40      — furthest from 60/40, what macro pushes toward it?
  → Q10 real_asset_rotation — real/hard assets prominent, what accelerates this?

2026-02-28  Current state — forward-looking, broadest question set
  → Q1  more_gold          — what macro regime increases gold from here?
  → Q4  less_gold          — what macro regime decreases gold from here?
  → Q3  house_view_6pct    — forward-looking return target
  → Q7  stretch_excess     — push excess return higher
  → Q8  more_equity        — what pushes equity allocation up?
  → Q9  flight_to_safety   — what would trigger a safety flight today?
  → Q11 max_sharpe         — where is the current optimal frontier?
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
REPO_SRC = WORKSPACE.parent.parent / "src"

if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))
if str(REPO_SRC) not in sys.path:
    sys.path.append(str(REPO_SRC))

from xoptpoe_v4_scenario.state_space import (
    MACRO_STATE_COLS, STATE_DIM, box_constraints, load_state, state_scales,
)
from xoptpoe_v4_scenario.var1_prior import VAR1Prior
from xoptpoe_v4_scenario.pipeline import (
    SLEEVES_14,
    build_benchmark_aligned_pipeline_at_date,
    benchmark_train_end,
    validate_anchor_alignment,
)
from xoptpoe_v4_scenario.probe_functions import (
    gold_weight_probe,
    less_gold_probe,
    more_diversification_probe,
    classic_sixty_forty_probe,
    excess_return_target_probe,
    max_total_equity_probe,
    house_view_return_probe,
    make_G_and_gradG,
    flight_to_safety_probe,
    real_asset_rotation_probe,
    max_sharpe_total_probe,
)
from xoptpoe_v4_scenario.sampler import (
    run_mala_chains, thin_only, filter_trajectories,
    compute_effective_sample_size,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_REFS = WORKSPACE / "data_refs"
REPORTS = WORKSPACE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

NFCI_PATH = WORKSPACE / "NFCI (1).csv"

BENCHMARK_EXPERIMENT = "elastic_net__core_plus_interactions__separate_60"
BENCHMARK_METRICS_PATH = REPORTS / "benchmark/v4_prediction_benchmark_metrics.csv"
TRAIN_END_VAR1 = pd.Timestamp("2016-02-29")

ANCHOR_DATES = [
    pd.Timestamp("2020-12-31"),
    pd.Timestamp("2021-12-31"),
    pd.Timestamp("2022-12-31"),
    pd.Timestamp("2023-12-31"),
    pd.Timestamp("2024-12-31"),
    pd.Timestamp("2026-02-28"),
]

# MALA parameters — tuned for exploration
N_SEEDS = 5
N_STEPS = 1000
WARMUP_FRAC = 0.40
ETA = 1.5
THINNING = 5
TAU_DIVISOR = 5.0
TAU_MIN = 1.0

# Probe regularization — loosened so task gradient dominates
VAR1_L2 = 0.005
L2_REG = 0.05

# Gate tolerances
GATE_TOL_RET = 0.005
GATE_TOL_GLD = 0.025

ANCHOR_TRUTH_EXCESS: dict = {}

# Targets
HOUSE_SAA_TOTAL = 0.06     # 6% total return
TARGET_EXCESS_Q7 = 0.05    # 5% excess return

# 60/40 probe configuration
CLASSIC_EQ_TARGET = 0.60
CLASSIC_FICR_TARGET = 0.40
CLASSIC_ALT_TARGET = 0.00
CLASSIC_ALT_PENALTY = 1.0


# ---------------------------------------------------------------------------
# Anchor → Question mapping
# ---------------------------------------------------------------------------

QUESTION_ANCHOR_MAP: dict[str, list[str]] = {
    # 2020: post-COVID, rf≈0, gold emerging, heavy FI
    "2020-12-31": ["Q7_stretch_excess"],
    # 2021: inflationary expansion, gold growing, equity rising
    "2021-12-31": ["Q1_more_gold", "Q5_more_diversified"],
    # 2022: rate hikes, gold≈25%, higher_for_longer begins
    "2022-12-31": ["Q6_classic_60_40"],
    # 2024: gold≈33% peak, concentrated, rf=4.42%
    "2024-12-31": ["Q4_less_gold"],
    # 2026: current — broadest question set
    "2026-02-28": ["Q11_max_sharpe_total"],
}


# ---------------------------------------------------------------------------
# Regime scoring (unchanged)
# ---------------------------------------------------------------------------

def score_regime_dimensions(m: np.ndarray, thresholds: dict) -> dict:
    from xoptpoe_v4_scenario.regime import classify_single_state

    label, simple, signals = classify_single_state(m, thresholds)

    stress_high = signals.get("stress_high", 0.0) == 1.0
    stress_moderate = signals.get("stress_moderate", 0.0) == 1.0
    stress_desc = "high" if stress_high else "moderate" if stress_moderate else "low"

    growth_desc = "weak" if signals.get("unemp_US", 0.0) > thresholds.get("unemp_US_p75", 6.0) else "neutral/strong"
    infl_desc = "high" if signals.get("infl_US", 0.0) > thresholds.get("infl_US_p75", 3.0) else "neutral/low"

    return {
        "dim_growth": growth_desc,
        "dim_inflation": infl_desc,
        "dim_policy": "dynamic",
        "dim_stress": stress_desc,
        "dim_fin_cond": "dynamic",
        "regime_label": label,
        "simple_regime": simple,
        "sig_ig_oas": round(signals.get("ig_oas", 0.0), 3),
        "sig_vix": round(signals.get("vix", 0.0), 2),
        "sig_infl_US": round(signals.get("infl_US", 0.0), 3),
        "sig_short_rate_US": round(signals.get("short_rate_US", 0.0), 3),
        "sig_us_real10y": round(signals.get("us_real10y", 0.0), 3),
        "sig_unemp_US": round(signals.get("unemp_US", 0.0), 3),
    }


def describe_regime_transition(m0, m_scenario, r0, rs, macro_cols):
    shifts = {
        col: round(float(m_scenario[i] - m0[i]), 4)
        for i, col in enumerate(macro_cols)
        if abs(float(m_scenario[i] - m0[i])) > 1e-8
    }
    top3 = dict(sorted(shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
    transition = "same_regime" if r0["regime_label"] == rs["regime_label"] \
        else f"{r0['regime_label']} -> {rs['regime_label']}"
    dim_changes = [
        f"{d}: {r0.get(d)} -> {rs.get(d)}"
        for d in ["dim_growth", "dim_inflation", "dim_policy", "dim_stress", "dim_fin_cond"]
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
    fm = pd.read_parquet(DATA_REFS / "feature_master_monthly.parquet")
    mp = pd.read_parquet(DATA_REFS / "modeling_panel_hstack.parquet")
    fsm = pd.read_csv(DATA_REFS / "feature_set_manifest.csv")
    tp = pd.read_parquet(DATA_REFS / "target_panel_long_horizon.parquet")
    for df in [fm, mp, tp]:
        df["month_end"] = pd.to_datetime(df["month_end"])
    feat_cols = fsm[fsm["include_core_plus_interactions"] == 1]["feature_name"].tolist()
    print(f"  feature_master: {len(fm)} rows | modeling_panel: {len(mp)} rows")
    print(f"  features: {len(feat_cols)}")
    return {
        "feature_master": fm,
        "modeling_panel": mp,
        "feature_manifest": fsm,
        "target_panel": tp,
        "feature_columns": feat_cols,
    }


_EXCESS_RETURNS_SOURCE: str = "unknown"


def build_excess_returns(fm: pd.DataFrame) -> pd.DataFrame:
    global _EXCESS_RETURNS_SOURCE

    try:
        from xoptpoe_v4_models.data import load_modeling_inputs
        inputs = load_modeling_inputs(WORKSPACE, feature_set_name="core_baseline")
        hist = inputs.monthly_excess_history
        if hist is not None:
            out = hist.copy()
            out.index = pd.to_datetime(out.index)
            avail = [s for s in SLEEVES_14 if s in out.columns]
            _EXCESS_RETURNS_SOURCE = "benchmark_exact"
            return out[avail].sort_index()
    except Exception:
        pass

    if "ret_1m_lag" in fm.columns and "short_rate_US" in fm.columns:
        work = fm[["month_end", "sleeve_id", "ret_1m_lag", "short_rate_US"]].copy()
        work["month_end"] = pd.to_datetime(work["month_end"])
        work["rf_1m"] = work["short_rate_US"] / 1200.0
        work["excess_ret_1m"] = work["ret_1m_lag"] - work["rf_1m"]
        pivot = work.pivot_table(
            index="month_end", columns="sleeve_id",
            values="excess_ret_1m", aggfunc="first",
        )
        avail = [s for s in SLEEVES_14 if s in pivot.columns]
        _EXCESS_RETURNS_SOURCE = "benchmark_exact"
        return pivot[avail].sort_index()

    if "ret_1m_lag" in fm.columns:
        print("  WARNING: short_rate_US missing — using total returns as excess proxy")
        pivot = fm.pivot_table(
            index="month_end", columns="sleeve_id",
            values="ret_1m_lag", aggfunc="first",
        )
        avail = [s for s in SLEEVES_14 if s in pivot.columns]
        _EXCESS_RETURNS_SOURCE = "fallback_ret_1m_lag"
        return pivot[avail].sort_index()

    _EXCESS_RETURNS_SOURCE = "fallback_zeros"
    dates = pd.to_datetime(fm["month_end"].unique())
    return pd.DataFrame(0.0, index=sorted(dates), columns=list(SLEEVES_14))


def compute_thresholds(fm, nfci_df=None):
    from xoptpoe_v4_scenario.regime import compute_regime_thresholds
    return compute_regime_thresholds(fm, nfci_df)


# ---------------------------------------------------------------------------
# Question builder — all probes use FD gradients + VAR1_L2=0.005
# ---------------------------------------------------------------------------

def _build_all_probes(pipeline, m0, scales, var1_prior):
    """Build all probes once; return dict[question_id -> (G, gradG)]."""
    probes: dict[str, tuple] = {}

    # Q1: more gold
    G1, _ = gold_weight_probe(pipeline, m0, scales,
                               l2reg=L2_REG, prior=var1_prior,
                               var1_l2reg=VAR1_L2, sign=-1.0)
    _, grad1 = make_G_and_gradG(G1)
    probes["Q1_more_gold"] = (G1, grad1)

    # Q3: house view 6% total
    rf_at_m0 = float(m0[list(MACRO_STATE_COLS).index("short_rate_US")]) / 100.0
    implied_excess_q3 = HOUSE_SAA_TOTAL - rf_at_m0
    print(
        f"  Q3 house-style target: {HOUSE_SAA_TOTAL:.0%} total | "
        f"rf={rf_at_m0*100:.3f}% | implied excess target={implied_excess_q3*100:.3f}%"
    )
    G3, _ = house_view_return_probe(pipeline, m0,
                                     house_view_total=HOUSE_SAA_TOTAL,
                                     scales=scales, l2reg=L2_REG,
                                     prior=var1_prior, var1_l2reg=VAR1_L2)
    _, grad3 = make_G_and_gradG(G3)
    probes["Q3_house_view_6pct_total"] = (G3, grad3)

    # Q4: less gold
    G4, _ = less_gold_probe(pipeline, m0, scales,
                             l2reg=L2_REG, prior=var1_prior, var1_l2reg=VAR1_L2)
    _, grad4 = make_G_and_gradG(G4)
    probes["Q4_less_gold"] = (G4, grad4)

    # Q5: more diversified
    G5, _ = more_diversification_probe(pipeline, m0, scales,
                                        l2reg=L2_REG, prior=var1_prior,
                                        var1_l2reg=VAR1_L2)
    _, grad5 = make_G_and_gradG(G5)
    probes["Q5_more_diversified"] = (G5, grad5)

    # Q6: classic 60/40
    G6, _ = classic_sixty_forty_probe(pipeline, m0, scales,
                                       eq_target=CLASSIC_EQ_TARGET,
                                       ficr_target=CLASSIC_FICR_TARGET,
                                       alt_target=CLASSIC_ALT_TARGET,
                                       alt_penalty=CLASSIC_ALT_PENALTY,
                                       l2reg=L2_REG, prior=var1_prior,
                                       var1_l2reg=VAR1_L2)
    _, grad6 = make_G_and_gradG(G6)
    probes["Q6_classic_60_40"] = (G6, grad6)

    # Q7: stretch excess — anchor-aware
    anchor_excess = float(pipeline.evaluate_at(m0)["pred_return_excess"])
    raw_target_q7 = anchor_excess + 0.01
    target_excess_q7 = max(0.04, min(0.05, raw_target_q7))
    print(
        f"  Q7 direct excess target: anchor={anchor_excess*100:.3f}% "
        f"-> target={target_excess_q7*100:.3f}%"
    )
    G7, _ = excess_return_target_probe(pipeline, m0, target_excess=target_excess_q7,
                                        scales=scales, l2reg=L2_REG,
                                        prior=var1_prior, var1_l2reg=VAR1_L2)
    _, grad7 = make_G_and_gradG(G7)
    probes["Q7_stretch_excess"] = (G7, grad7)

    # Q8: more equity
    G8, _ = max_total_equity_probe(pipeline, m0, scales,
                                    l2reg=L2_REG, prior=var1_prior,
                                    var1_l2reg=VAR1_L2)
    _, grad8 = make_G_and_gradG(G8)
    probes["Q8_more_equity"] = (G8, grad8)

    # Q9: flight to safety
    G9, _ = flight_to_safety_probe(pipeline, m0, scales,
                                    l2reg=L2_REG, prior=var1_prior,
                                    var1_l2reg=VAR1_L2)
    _, grad9 = make_G_and_gradG(G9)
    probes["Q9_flight_to_safety"] = (G9, grad9)

    # Q10: real asset rotation
    G10, _ = real_asset_rotation_probe(pipeline, m0, scales,
                                        l2reg=L2_REG, prior=var1_prior,
                                        var1_l2reg=VAR1_L2)
    _, grad10 = make_G_and_gradG(G10)
    probes["Q10_real_asset_rotation"] = (G10, grad10)

    # Q11: max sharpe
    G11, _ = max_sharpe_total_probe(pipeline, m0, scales,
                                     l2reg=L2_REG, prior=var1_prior,
                                     var1_l2reg=VAR1_L2)
    _, grad11 = make_G_and_gradG(G11)
    probes["Q11_max_sharpe_total"] = (G11, grad11)

    return probes


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_scenario() -> None:
    print("=" * 70)
    print("v4 Scenario Engine — MAPPED (curated anchor→question)")
    print(f"MALA: N_STEPS={N_STEPS}, N_SEEDS={N_SEEDS}, ETA={ETA}, "
          f"WARMUP={WARMUP_FRAC:.0%}, THINNING={THINNING}")
    print(f"TAU: adaptive = G_m0 / {TAU_DIVISOR} (floor={TAU_MIN})")
    print(f"VAR1 regularizer: {VAR1_L2}  |  Gradients: full finite-difference")
    print(f"Benchmark experiment: {BENCHMARK_EXPERIMENT}")
    print(f"Q3 target: {HOUSE_SAA_TOTAL:.0%} TOTAL return (house/SAA style)")
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
    scales = state_scales(fm, train_end=TRAIN_END_VAR1)
    a_box, b_box = box_constraints(fm, slack_multiplier=1.0)
    precond = scales ** 2

    all_results = []
    audit_rows = []
    gate_rows = []

    for anchor in ANCHOR_DATES:
        anchor_key = str(anchor.date())
        mapped_questions = QUESTION_ANCHOR_MAP.get(anchor_key, [])
        if not mapped_questions:
            print(f"\n  Anchor {anchor_key}: no questions mapped — skip")
            continue

        train_end = benchmark_train_end(anchor)
        print(f"\n{'='*60}\nAnchor: {anchor.date()} | "
              f"benchmark_train_end={train_end.date()}\n"
              f"  Mapped questions: {', '.join(mapped_questions)}\n{'='*60}")

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

        # ── Alignment gate ──────────────────────────────────────────────
        try:
            val = validate_anchor_alignment(
                anchor=anchor,
                pipeline=pipeline,
                m0=m0,
                expected_experiment=BENCHMARK_EXPERIMENT,
                expected_train_end=train_end,
                covariance_source=_EXCESS_RETURNS_SOURCE,
                strict_2021=True,
            )
        except RuntimeError as e:
            print(f"\n  *** ALIGNMENT GATE HARD STOP ***\n  {e}\n")
            gate_rows.append({
                "anchor_date": anchor.date(),
                "train_end": train_end.date(),
                "covariance_source": _EXCESS_RETURNS_SOURCE,
                "gate1_status": "HARD_FAIL",
                "detail": str(e),
            })
            continue

        gate_rows.append({
            "anchor_date": anchor.date(),
            "train_end": train_end.date(),
            "covariance_source": _EXCESS_RETURNS_SOURCE,
            "gate1_status": val["status"],
            "pred_return_excess": val["pred_return_excess"],
            "rf_rate": val["rf_rate"],
            "pred_return_total": val["pred_return_total"],
            "w_ALT_GLD": val["w_ALT_GLD"],
            "top_sleeve": val["top_sleeve"],
            "covariance_exact": val["covariance_exact"],
            "alignment_passed": val["alignment_passed"],
        })

        if not val["alignment_passed"]:
            print("  *** ALIGNMENT GATE FAILED — skipping anchor ***")
            continue

        # Unpack validated values
        ev0 = pipeline.evaluate_at(m0)
        w0 = ev0["w"]
        pred0_excess = float(val["pred_return_excess"])
        pred0_total = float(val["pred_return_total"])
        rf0 = float(val["rf_rate"])
        gld0 = float(val["w_ALT_GLD"])

        r0 = score_regime_dimensions(m0, thresholds)
        print(f"  Regime: {r0['regime_label']} | "
              f"growth={r0['dim_growth']}, infl={r0['dim_inflation']}, "
              f"policy={r0['dim_policy']}, stress={r0['dim_stress']}")

        audit_rows.append({
            "anchor_date": anchor.date(),
            "train_end": train_end.date(),
            "covariance_source": _EXCESS_RETURNS_SOURCE,
            "pred_return_excess": round(pred0_excess, 6),
            "rf_rate": round(rf0, 6),
            "pred_return_total": round(pred0_total, 6),
            "w_ALT_GLD_m0": round(gld0, 4),
            "w_EQ_US_m0": round(float(w0[SLEEVES_14.index("EQ_US")]), 4),
            "entropy_m0": round(ev0["entropy"], 4),
            "anchor_regime": r0["regime_label"],
            "dim_growth": r0["dim_growth"],
            "dim_inflation": r0["dim_inflation"],
            "dim_policy": r0["dim_policy"],
            "dim_stress": r0["dim_stress"],
            **{f"w0_{s}": round(float(w0[i]), 4) for i, s in enumerate(SLEEVES_14)},
        })

        # Build only the probes needed for this anchor
        all_probes = _build_all_probes(pipeline, m0, scales, var1_prior)

        for qid in mapped_questions:
            if qid not in all_probes:
                print(f"\n  Q: {qid} [NOT FOUND in probe builder — skip]")
                continue

            G_q, gradG_q = all_probes[qid]
            G_m0 = G_q(m0)
            tau_effective = max(abs(G_m0) / TAU_DIVISOR, TAU_MIN)

            print(f"\n  Q: {qid}")
            print(f"    G(m0)={G_m0:.4f} | tau={tau_effective:.3f}")

            trajectories, acc_rates = run_mala_chains(
                G=G_q,
                gradG=gradG_q,
                m0=m0,
                a=a_box,
                b=b_box,
                n_seeds=N_SEEDS,
                n_steps=N_STEPS,
                eta=ETA,
                tau=tau_effective,
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
            ess_min = float(ess.min())
            ess_median = float(np.median(ess))
            print(f"    n={n_posterior} | acc={mean_acc:.2%} | "
                  f"ESS min={ess_min:.1f} median={ess_median:.1f}")

            for m_s in all_samples:
                ev_s = pipeline.evaluate_at(m_s)
                w_s = ev_s["w"]
                rs = score_regime_dimensions(m_s, thresholds)
                trans = describe_regime_transition(m0, m_s, r0, rs, MACRO_STATE_COLS)

                row = {
                    "question_id": qid,
                    "anchor_date": anchor.date(),
                    "train_end": train_end.date(),
                    "G_value": round(float(G_q(m_s)), 6),

                    # Return columns
                    "pred_return": round(ev_s["pred_return_excess"], 6),
                    "pred_return_excess": round(ev_s["pred_return_excess"], 6),
                    "rf_rate": round(ev_s["rf_rate"], 6),
                    "pred_return_total": round(ev_s["pred_return_total"], 6),

                    # Risk / portfolio
                    "portfolio_risk": round(ev_s["risk"], 6),
                    "portfolio_entropy": round(ev_s["entropy"], 4),
                    "sharpe_pred": round(ev_s["sharpe_pred"], 4),
                    "sharpe_pred_total": round(ev_s["sharpe_pred_total"], 4),

                    # Regime
                    "regime_label": rs["regime_label"],
                    "dim_growth": rs["dim_growth"],
                    "dim_inflation": rs["dim_inflation"],
                    "dim_policy": rs["dim_policy"],
                    "dim_stress": rs["dim_stress"],
                    "dim_fin_cond": rs["dim_fin_cond"],
                    "regime_transition": trans["regime_transition"],
                    "anchor_regime": r0["regime_label"],
                    "dim_changes": trans["dim_changes"],
                    "top_shift_1_var": list(trans["top_shifts"].keys())[0] if trans["top_shifts"] else "",
                    "top_shift_1_val": list(trans["top_shifts"].values())[0] if trans["top_shifts"] else 0.0,
                    "top_shift_2_var": list(trans["top_shifts"].keys())[1] if len(trans["top_shifts"]) > 1 else "",
                    "top_shift_2_val": list(trans["top_shifts"].values())[1] if len(trans["top_shifts"]) > 1 else 0.0,
                    "top_shift_3_var": list(trans["top_shifts"].keys())[2] if len(trans["top_shifts"]) > 2 else "",
                    "top_shift_3_val": list(trans["top_shifts"].values())[2] if len(trans["top_shifts"]) > 2 else 0.0,

                    # Signals
                    "sig_ig_oas": rs["sig_ig_oas"],
                    "sig_vix": rs["sig_vix"],
                    "sig_infl_US": rs["sig_infl_US"],
                    "sig_short_rate_US": rs["sig_short_rate_US"],
                    "sig_us_real10y": rs["sig_us_real10y"],
                    "sig_unemp_US": rs["sig_unemp_US"],

                    # MALA diagnostics
                    "tau_effective": round(tau_effective, 4),
                    "mean_acceptance_rate": round(mean_acc, 4),
                    "ess_min": round(ess_min, 1),
                    "ess_median": round(ess_median, 1),

                    # Macro state
                    **{col: round(float(m_s[i]), 5) for i, col in enumerate(MACRO_STATE_COLS)},

                    # Weights
                    **{f"w_{s}": round(float(w_s[i]), 5) for i, s in enumerate(SLEEVES_14)},
                }
                all_results.append(row)

            rows_q = [
                r for r in all_results
                if r["question_id"] == qid and str(r["anchor_date"]) == str(anchor.date())
            ]
            rets_exc = [r["pred_return_excess"] for r in rows_q]
            rets_total = [r["pred_return_total"] for r in rows_q]
            glds = [r["w_ALT_GLD"] for r in rows_q]
            regs = [r["regime_label"] for r in rows_q]
            print(
                f"    excess={np.mean(rets_exc)*100:.2f}%±{np.std(rets_exc)*100:.2f}%  "
                f"total={np.mean(rets_total)*100:.2f}%±{np.std(rets_total)*100:.2f}%  "
                f"GLD={np.mean(glds):.3f}"
            )
            print(f"    Regimes: {dict(Counter(regs).most_common(3))}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving outputs...")

    results_df = pd.DataFrame(all_results)
    audit_df = pd.DataFrame(audit_rows)
    gate_df = pd.DataFrame(gate_rows)

    all_passed = (gate_df["gate1_status"] == "PASS").all() if not gate_df.empty else False

    gate_df.to_csv(REPORTS / "scenario_anchor_truth_mapped_v4.csv", index=False)
    audit_df.to_csv(REPORTS / "scenario_anchor_audit_mapped_v4.csv", index=False)

    if not results_df.empty:
        results_df.to_csv(REPORTS / "scenario_results_mapped_v4.csv", index=False)
        print(f"  scenario_results_mapped_v4.csv ({len(results_df)} rows)")
    else:
        print("  WARNING: no results produced")
        return

    _write_regime_summary(results_df, REPORTS / "scenario_regime_summary_mapped_v4.csv")
    _write_selected_cases(results_df, REPORTS / "scenario_selected_cases_mapped_v4.csv")
    _write_question_manifest(results_df, REPORTS / "scenario_question_manifest_mapped_v4.csv")
    _write_portfolio_response_summary(results_df, REPORTS / "scenario_portfolio_response_summary_mapped_v4.csv")
    _write_regime_transition_summary(results_df, REPORTS / "scenario_regime_transition_summary_mapped_v4.csv")
    _write_selected_questions(results_df, REPORTS / "scenario_selected_questions_mapped_v4.csv")
    _write_mala_diagnostics(results_df, REPORTS / "scenario_mala_diagnostics_mapped_v4.csv")

    print(f"\nDone. Gate 1 all_passed={all_passed}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_regime_summary(results_df, path):
    rows = []
    for (qid, anchor), grp in results_df.groupby(["question_id", "anchor_date"]):
        for regime, rgrp in grp.groupby("regime_label"):
            rows.append({
                "question_id": qid,
                "anchor_date": anchor,
                "regime_label": regime,
                "count": len(rgrp),
                "share": round(len(rgrp) / len(grp), 3),
                "mean_pred_return_excess": round(float(rgrp["pred_return_excess"].mean()), 5),
                "mean_pred_return_total": round(float(rgrp["pred_return_total"].mean()), 5),
                "mean_rf_rate": round(float(rgrp["rf_rate"].mean()), 5),
                "std_pred_return_total": round(float(rgrp["pred_return_total"].std()), 5),
                "mean_entropy": round(float(rgrp["portfolio_entropy"].mean()), 4),
                "mean_w_ALT_GLD": round(float(rgrp["w_ALT_GLD"].mean()), 4) if "w_ALT_GLD" in rgrp else None,
                "mean_w_EQ_US": round(float(rgrp["w_EQ_US"].mean()), 4) if "w_EQ_US" in rgrp else None,
                "mean_w_FI_UST": round(float(rgrp["w_FI_UST"].mean()), 4) if "w_FI_UST" in rgrp else None,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_selected_cases(results_df, path):
    rows = []
    for (qid, anchor, regime), grp in results_df.groupby(["question_id", "anchor_date", "regime_label"]):
        best = grp.loc[grp["G_value"].idxmin()]
        rows.append(best)
    pd.DataFrame(rows).reset_index(drop=True).to_csv(path, index=False)


def _write_question_manifest(results_df, path):
    rows = []
    for (qid, anchor), grp in results_df.groupby(["question_id", "anchor_date"]):
        rows.append({
            "question_id": qid,
            "anchor_date": anchor,
            "n_valid_samples": len(grp),
            "regime_diversity": grp["regime_label"].nunique(),
            "mean_pred_return_excess": round(float(grp["pred_return_excess"].mean()), 5),
            "mean_pred_return_total": round(float(grp["pred_return_total"].mean()), 5),
            "mean_rf_rate": round(float(grp["rf_rate"].mean()), 5),
            "std_pred_return_total": round(float(grp["pred_return_total"].std()), 5),
            "mean_entropy": round(float(grp["portfolio_entropy"].mean()), 4),
            "dominant_regime": grp["regime_label"].value_counts().index[0],
            "dominant_regime_share": round(float(grp["regime_label"].value_counts().iloc[0] / len(grp)), 3),
            "tau_effective": round(float(grp["tau_effective"].iloc[0]), 4) if "tau_effective" in grp else None,
            "mean_acceptance_rate": round(float(grp["mean_acceptance_rate"].mean()), 4) if "mean_acceptance_rate" in grp else None,
            "ess_min": round(float(grp["ess_min"].mean()), 1) if "ess_min" in grp else None,
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
            "question_id": qid,
            "anchor_date": anchor,
            "n_samples": len(grp),
            "mean_pred_return_excess": round(float(grp["pred_return_excess"].mean()), 5),
            "std_pred_return_excess": round(float(grp["pred_return_excess"].std()), 5),
            "mean_rf_rate": round(float(grp["rf_rate"].mean()), 5),
            "mean_pred_return_total": round(float(grp["pred_return_total"].mean()), 5),
            "std_pred_return_total": round(float(grp["pred_return_total"].std()), 5),
            "p25_pred_return_total": round(float(grp["pred_return_total"].quantile(0.25)), 5),
            "p75_pred_return_total": round(float(grp["pred_return_total"].quantile(0.75)), 5),
            "mean_entropy": round(float(grp["portfolio_entropy"].mean()), 4),
            "mean_sharpe_pred_total": round(float(grp["sharpe_pred_total"].mean()), 4),
        }
        for wc in weight_cols:
            row[f"mean_{wc}"] = round(float(grp[wc].mean()), 4)
            row[f"std_{wc}"] = round(float(grp[wc].std()), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_regime_transition_summary(results_df, path):
    rows = []
    for (qid, anchor), grp in results_df.groupby(["question_id", "anchor_date"]):
        for trans, tgrp in grp.groupby("regime_transition"):
            rows.append({
                "question_id": qid,
                "anchor_date": anchor,
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
        diversity = grp["regime_label"].nunique()
        ret_range = float(grp["pred_return_total"].max() - grp["pred_return_total"].min())
        ret_std = float(grp["pred_return_total"].std())
        n_samples = len(grp)
        trans_div = grp["regime_transition"].nunique() if "regime_transition" in grp.columns else 1
        score = diversity * ret_range * min(n_samples / 50, 3.0) * trans_div
        scores.append({
            "question_id": qid,
            "score": round(score, 4),
            "n_samples": n_samples,
            "regime_diversity": diversity,
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