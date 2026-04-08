#!/usr/bin/env python3
"""
Quick MALA tuning diagnostic — runs Q1 (more gold) and Q4 (less gold) on
anchor 2024-12-31 only, prints gold-weight direction so we can verify
the sampler actually moves the right way.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

WORKSPACE = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))
REPO_SRC = WORKSPACE.parent.parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.append(str(REPO_SRC))

from xoptpoe_v4_scenario.state_space import (
    MACRO_STATE_COLS, STATE_DIM, box_constraints, load_state, state_scales,
)
from xoptpoe_v4_scenario.var1_prior import VAR1Prior
from xoptpoe_v4_scenario.pipeline import (
    SLEEVES_14, build_benchmark_aligned_pipeline_at_date, benchmark_train_end,
    validate_anchor_alignment,
)
from xoptpoe_v4_scenario.probe_functions import (
    gold_weight_probe, less_gold_probe, house_view_return_probe,
    excess_return_target_probe, GOLD_IDX,
)
from xoptpoe_v4_scenario.sampler import (
    run_mala_chains, thin_only, compute_effective_sample_size,
)

# ── Config ───────────────────────────────────────────────────────────
DATA_REFS = WORKSPACE / "data_refs"
REPORTS   = WORKSPACE / "reports"
NFCI_PATH = WORKSPACE / "NFCI (1).csv"
BENCHMARK_EXPERIMENT = "elastic_net__core_plus_interactions__separate_60"
BENCHMARK_METRICS_PATH = REPORTS / "benchmark/v4_prediction_benchmark_metrics.csv"
TRAIN_END_VAR1 = pd.Timestamp("2016-02-29")

ANCHOR = pd.Timestamp("2024-12-31")

# ── Tuning grid ──────────────────────────────────────────────────────
CONFIGS = [
    {"label": "CURRENT  (var1=0.10, eta=0.05)", "var1_l2reg": 0.10, "eta": 0.05},
    {"label": "LOWER_REG(var1=0.01, eta=0.05)", "var1_l2reg": 0.01, "eta": 0.05},
    {"label": "HIGHER_ET(var1=0.01, eta=0.15)", "var1_l2reg": 0.01, "eta": 0.15},
    {"label": "AGGRESSIVE(var1=0.005,eta=0.20)", "var1_l2reg": 0.005, "eta": 0.20},
]

N_SEEDS = 3
N_STEPS = 600
WARMUP_FRAC = 0.40
THINNING = 5
TAU_DIVISOR = 5.0
TAU_MIN = 1.0
HOUSE_SAA_TOTAL = 0.06


def main():
    # ── Load data ────────────────────────────────────────────────────
    fm = pd.read_parquet(DATA_REFS / "feature_master_monthly.parquet")
    mp = pd.read_parquet(DATA_REFS / "modeling_panel_hstack.parquet")
    fsm = pd.read_csv(DATA_REFS / "feature_set_manifest.csv")
    for df in [fm, mp]:
        df["month_end"] = pd.to_datetime(df["month_end"])
    feat_cols = fsm[fsm["include_core_plus_interactions"] == 1]["feature_name"].tolist()

    # excess returns
    work = fm[["month_end", "sleeve_id", "ret_1m_lag", "short_rate_US"]].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    work["rf_1m"] = work["short_rate_US"] / 1200.0
    work["excess_ret_1m"] = work["ret_1m_lag"] - work["rf_1m"]
    excess_ret = work.pivot_table(index="month_end", columns="sleeve_id",
                                   values="excess_ret_1m", aggfunc="first")
    avail = [s for s in SLEEVES_14 if s in excess_ret.columns]
    excess_ret = excess_ret[avail].sort_index()

    var1_prior = VAR1Prior.fit_from_feature_master(fm, MACRO_STATE_COLS, train_end=TRAIN_END_VAR1)
    scales = state_scales(fm, train_end=TRAIN_END_VAR1)
    a_box, b_box = box_constraints(fm, slack_multiplier=1.0)
    precond = scales ** 2

    train_end = benchmark_train_end(ANCHOR)
    pipeline = build_benchmark_aligned_pipeline_at_date(
        anchor_date=ANCHOR, feature_master=fm, modeling_panel=mp,
        feature_manifest=fsm, feature_columns=feat_cols,
        excess_returns_monthly=excess_ret,
        benchmark_metrics_path=BENCHMARK_METRICS_PATH,
        experiment_name=BENCHMARK_EXPERIMENT,
        train_end=train_end, sleeve_order=SLEEVES_14,
    )
    m0, _ = load_state(ANCHOR, fm, SLEEVES_14, modeling_panel=mp, horizon_months=60)

    ev0 = pipeline.evaluate_at(m0)
    w0 = ev0["w"]
    gld0 = float(w0[GOLD_IDX])
    excess0 = ev0["pred_return_excess"]
    total0 = ev0["pred_return_total"]

    print(f"Anchor: {ANCHOR.date()}")
    print(f"  Baseline gold={gld0*100:.1f}%  excess={excess0*100:.2f}%  total={total0*100:.2f}%")
    print(f"  Baseline weights: {', '.join(f'{s}={w0[i]*100:.1f}%' for i, s in enumerate(SLEEVES_14) if w0[i]>0.01)}")
    print()

    # ── Run each config ──────────────────────────────────────────────
    for cfg in CONFIGS:
        vl = cfg["var1_l2reg"]
        eta = cfg["eta"]
        print(f"{'='*70}")
        print(f"CONFIG: {cfg['label']}")
        print(f"{'='*70}")

        probes = {}

        # Q1: more gold (sign=-1)
        G1, grad1 = gold_weight_probe(pipeline, m0, scales,
                                       l2reg=0.05, prior=var1_prior,
                                       var1_l2reg=vl, sign=-1.0)
        probes["Q1_more_gold"] = (G1, grad1)

        # Q4: less gold (sign=+1)
        G4, grad4 = less_gold_probe(pipeline, m0, scales,
                                     l2reg=0.05, prior=var1_prior,
                                     var1_l2reg=vl)
        probes["Q4_less_gold"] = (G4, grad4)

        # Q3: house view 6% total
        G3, grad3 = house_view_return_probe(pipeline, m0,
                                             house_view_total=HOUSE_SAA_TOTAL,
                                             scales=scales, l2reg=0.05,
                                             prior=var1_prior, var1_l2reg=vl)
        probes["Q3_house_6pct"] = (G3, grad3)

        # Q7: stretch excess
        raw_t = excess0 + 0.01
        t7 = max(0.04, min(0.05, raw_t))
        G7, grad7 = excess_return_target_probe(pipeline, m0, target_excess=t7,
                                                scales=scales, l2reg=0.05,
                                                prior=var1_prior, var1_l2reg=vl)
        probes[f"Q7_excess_{t7*100:.1f}pct"] = (G7, grad7)

        for qid, (G_q, gradG_q) in probes.items():
            G_m0 = G_q(m0)
            tau = max(abs(G_m0) / TAU_DIVISOR, TAU_MIN)

            trajectories, acc_rates = run_mala_chains(
                G=G_q, gradG=gradG_q, m0=m0,
                a=a_box, b=b_box,
                n_seeds=N_SEEDS, n_steps=N_STEPS, eta=eta,
                tau=tau, warmup_frac=WARMUP_FRAC,
                seed=hash(qid) % (2**31), precond=precond, verbose=False,
            )
            mean_acc = float(np.mean(acc_rates))
            samples = thin_only(trajectories, thinning=THINNING)
            n = len(samples)
            if n == 0:
                print(f"  {qid}: NO SAMPLES")
                continue

            ess = compute_effective_sample_size(samples)

            # Evaluate portfolio stats for each sample
            golds, excesses, totals = [], [], []
            for m_s in samples:
                ev = pipeline.evaluate_at(m_s)
                golds.append(float(ev["w"][GOLD_IDX]))
                excesses.append(ev["pred_return_excess"])
                totals.append(ev["pred_return_total"])

            mg = np.mean(golds)
            me = np.mean(excesses)
            mt = np.mean(totals)
            delta_g = mg - gld0
            delta_e = me - excess0
            delta_t = mt - total0

            direction_ok = "?"
            if "more_gold" in qid:
                direction_ok = "OK" if delta_g > 0.005 else "WRONG" if delta_g < -0.005 else "~FLAT"
            elif "less_gold" in qid:
                direction_ok = "OK" if delta_g < -0.005 else "WRONG" if delta_g > 0.005 else "~FLAT"
            elif "house" in qid:
                direction_ok = "OK" if abs(mt - 0.06) < abs(total0 - 0.06) else "WRONG"
            elif "excess" in qid:
                direction_ok = "OK" if abs(me - t7) < abs(excess0 - t7) else "WRONG"

            print(
                f"  {qid:25s} | acc={mean_acc:.0%} ESS={float(np.median(ess)):.0f} "
                f"| gold={mg*100:.1f}% (Δ{delta_g*100:+.1f}%) "
                f"| excess={me*100:.2f}% (Δ{delta_e*100:+.2f}%) "
                f"| total={mt*100:.2f}% (Δ{delta_t*100:+.2f}%) "
                f"| {direction_ok}"
            )
        print()


if __name__ == "__main__":
    main()
