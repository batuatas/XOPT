#!/usr/bin/env python3
"""
run_v4_scenario_first_pass.py

First-pass v4 scenario engine run.

Implements four conference-relevant scenario questions:
  A. Gold transition: what macro shift explains 2021->2022 gold weight change?
  B. Equal-weight excess: when does model beat equal weight?
  C. House-view return: what macro regime justifies 6%, 7%, 10% SAA returns?
  D. Diversification: what macro conditions maximise portfolio entropy?

Anchor dates: 2021-12-31 and 2022-12-31 (primary pair for Gold question)
              2023-12-31 and 2024-12-31 (for house-view and EW questions)

Uses:
  - Locked benchmark: elastic_net__core_plus_interactions__separate_60
    lambda_risk=8.0, kappa=0.10, omega_type=identity
  - VAR(1) prior (var1_prior.py) for plausibility regularization
  - MALA sampler (sampler.py) — pure numpy
  - Regime classification (regime.py) using NFCI + NBER overlays

Outputs (written to workspace_v4/reports/scenario/):
  - scenario_results_v4.csv           — all valid samples with portfolio outcomes + regime labels
  - scenario_regime_summary_v4.csv    — regime frequency table per question
  - scenario_selected_cases_v4.csv    — representative selected scenarios
  - scenario_pipeline_audit_v4.md     — pipeline sanity checks at each anchor date
  - scenario_question_manifest_v4.csv — questions, anchors, targets
  - scenario_conference_takeaways_v4.md — narrative interpretations

Usage:
  cd workspace_v4
  python scripts/scenario/run_v4_scenario_first_pass.py
"""
from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent
REPO_SRC = WORKSPACE.parent.parent / "src"

for p in [str(WORKSPACE / "src"), str(REPO_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from xoptpoe_v4_scenario.state_space import (
    MACRO_STATE_COLS,
    STATE_DIM,
    box_constraints,
    load_state,
    state_scales,
)
from xoptpoe_v4_scenario.var1_prior import VAR1Prior
from xoptpoe_v4_scenario.pipeline import (
    SLEEVES_14,
    build_pipeline_at_date,
)
from xoptpoe_v4_scenario.probe_functions import (
    gold_weight_probe,
    gold_transition_probe,
    equal_weight_excess_probe,
    house_view_return_probe,
    diversification_probe,
    benchmark_return_probe,
)
from xoptpoe_v4_scenario.sampler import run_mala_chains, filter_trajectories
from xoptpoe_v4_scenario.regime import (
    load_nfci,
    compute_regime_thresholds,
    classify_sample_set,
    add_recession_overlay,
    regime_summary,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_REFS = WORKSPACE / "data_refs"
REPORTS_DIR = WORKSPACE / "reports" / "scenario"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

NFCI_PATH = WORKSPACE / "NFCI (1).csv"

# Accepted ElasticNet hyperparameters from benchmark run
# From prediction_benchmark_report: best alpha/l1_ratio for separate_60
# These are the values selected in the accepted benchmark
ELASTIC_NET_ALPHA = 0.005
ELASTIC_NET_L1 = 0.5

# MALA parameters — lean first-pass settings (~3ms/eval after optimization)
# Runtime: 4 chains × 200 steps × 38 evals × 3ms ≈ 90s per question
N_SEEDS = 4        # number of independent chains (2 near m0, 2 random)
N_STEPS = 200      # steps per chain
WARMUP_FRAC = 0.4  # discard first 40%
ETA = 0.005        # step size (slightly larger → faster exploration)
TAU = 0.5          # temperature
THINNING = 5       # keep every 5th sample

# Anchor dates
ANCHOR_DATES = [
    pd.Timestamp("2021-12-31"),
    pd.Timestamp("2022-12-31"),
    pd.Timestamp("2023-12-31"),
    pd.Timestamp("2024-12-31"),
]

# Training cutoffs for each anchor (use data up to anchor - 1 day for model fit)
TRAIN_END_VAR1 = pd.Timestamp("2016-02-29")   # fixed VAR(1) prior on training period


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_data() -> dict[str, object]:
    """Load all required data artifacts."""
    print("Loading data artifacts...")
    fm = pd.read_parquet(DATA_REFS / "feature_master_monthly.parquet")
    fm["month_end"] = pd.to_datetime(fm["month_end"])

    # Use modeling_panel_hstack: covers 2006-2026 with all 303 feature columns,
    # required for correct feature-row lookup at anchor dates 2021-2024.
    # modeling_panel_firstpass only covers to 2021-02-28 and fuzzy-matches all
    # recent anchors to Feb 2021 rows (306-1402 days stale).
    mp = pd.read_parquet(DATA_REFS / "modeling_panel_hstack.parquet")
    mp["month_end"] = pd.to_datetime(mp["month_end"])

    fsm = pd.read_csv(DATA_REFS / "feature_set_manifest.csv")

    tp = pd.read_parquet(DATA_REFS / "target_panel_long_horizon.parquet")
    tp["month_end"] = pd.to_datetime(tp["month_end"])

    # feature_columns for core_plus_interactions
    feat_cols = fsm[fsm["include_core_plus_interactions"] == 1]["feature_name"].tolist()

    print(f"  feature_master: {len(fm)} rows, {len(fm.columns)} cols")
    print(f"  modeling_panel: {len(mp)} rows")
    print(f"  core_plus_interactions features: {len(feat_cols)}")

    return {
        "feature_master": fm,
        "modeling_panel": mp,
        "feature_manifest": fsm,
        "target_panel": tp,
        "feature_columns": feat_cols,
    }


def build_sleeve_excess_returns_monthly(
    target_panel: pd.DataFrame,
    feature_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a (month_end x sleeve) monthly excess return matrix for EWMA cov estimation.

    Uses 1-month excess returns from target_panel (annualized_excess_forward_return
    at horizon=1 is the 1-month return; if unavailable, approximate from feature_master
    using ret_1m_lag as a proxy for the prior month's realized return).
    """
    tp = target_panel.copy()
    tp["month_end"] = pd.to_datetime(tp["month_end"])

    # Prefer 1m horizon if available; otherwise use sleeve momentum features
    if "horizon_months" in tp.columns:
        tp1m = tp[tp["horizon_months"] == 1].copy() if 1 in tp["horizon_months"].values else pd.DataFrame()
    else:
        tp1m = pd.DataFrame()

    if tp1m.empty:
        # Fallback: use ret_1m_lag from feature_master as realized monthly excess return proxy
        fm = feature_master.copy()
        fm["month_end"] = pd.to_datetime(fm["month_end"])
        if "ret_1m_lag" in fm.columns:
            # ret_1m_lag = prior month return on that sleeve, used as excess return proxy
            pivot = fm.pivot_table(
                index="month_end",
                columns="sleeve_id",
                values="ret_1m_lag",
                aggfunc="first",
            )
            # Keep only 14 benchmark sleeves
            available = [s for s in SLEEVES_14 if s in pivot.columns]
            return pivot[available].sort_index()
        else:
            # Last fallback: zero matrix
            dates = pd.to_datetime(fm["month_end"].unique())
            return pd.DataFrame(0.0, index=sorted(dates), columns=list(SLEEVES_14))
    else:
        # Use 1m target returns
        if "annualized_excess_forward_return" in tp1m.columns:
            ret_col = "annualized_excess_forward_return"
        else:
            ret_col = [c for c in tp1m.columns if "excess" in c.lower()][0]

        pivot = tp1m.pivot_table(
            index="month_end",
            columns="sleeve_id",
            values=ret_col,
            aggfunc="first",
        )
        available = [s for s in SLEEVES_14 if s in pivot.columns]
        return pivot[available].sort_index()


def get_realized_60m_returns(
    target_panel: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> dict[str, float]:
    """Get realized 60m annualized excess returns at anchor_date."""
    tp = target_panel.copy()
    tp["month_end"] = pd.to_datetime(tp["month_end"])

    if "horizon_months" in tp.columns:
        chunk = tp[
            (tp["month_end"] == anchor_date) &
            (tp["horizon_months"] == 60)
        ]
    else:
        chunk = pd.DataFrame()

    if chunk.empty:
        # No realized return available at this date — use NaN
        return {s: np.nan for s in SLEEVES_14}

    ret_col = "annualized_excess_forward_return"
    if ret_col not in chunk.columns:
        return {s: np.nan for s in SLEEVES_14}

    result = {}
    for sleeve in SLEEVES_14:
        row = chunk[chunk["sleeve_id"] == sleeve]
        result[sleeve] = float(row[ret_col].iloc[0]) if not row.empty else np.nan
    return result


def get_elastic_net_hyperparams(
    modeling_panel: pd.DataFrame,
    feature_columns: list[str],
    feature_manifest: pd.DataFrame,
    train_end: pd.Timestamp,
) -> tuple[float, float]:
    """
    Find best ElasticNet (alpha, l1_ratio) on train+val split.
    Quick grid search; uses fixed grid from benchmark.
    Returns accepted benchmark params if no improvement found.
    """
    from sklearn.linear_model import ElasticNet
    from xoptpoe_v4_scenario.pipeline import _fit_preprocessor_state

    TARGET_COL = "annualized_excess_forward_return"
    ALPHA_GRID = [0.0005, 0.001, 0.005, 0.01, 0.05]
    L1_GRID = [0.2, 0.5, 0.8]

    panel_60 = modeling_panel[
        (modeling_panel["horizon_months"] == 60) &
        (modeling_panel["baseline_trainable_flag"] == 1) &
        (modeling_panel["target_available_flag"] == 1)
    ].copy()
    panel_60["month_end"] = pd.to_datetime(panel_60["month_end"])

    train_df = panel_60[panel_60["month_end"] <= train_end].copy()
    if len(train_df) < 50:
        return ELASTIC_NET_ALPHA, ELASTIC_NET_L1

    # Use last 20% of training rows as internal val
    n_val = max(int(len(train_df) * 0.2), 20)
    val_df = train_df.tail(n_val)
    tr_df = train_df.iloc[:-n_val]
    if len(tr_df) < 30:
        return ELASTIC_NET_ALPHA, ELASTIC_NET_L1

    prep = _fit_preprocessor_state(tr_df, feature_manifest, feature_columns)

    def transform(df_: pd.DataFrame) -> np.ndarray:
        X_df = df_[feature_columns].copy().apply(pd.to_numeric, errors="coerce")
        for feat, fill in prep.fill_values.items():
            X_df[feat] = X_df[feat].fillna(fill)
        indicators = {}
        for feat in prep.indicator_feature_names:
            indicators[f"{feat}__missing"] = df_[feat].isna().astype(float).values
        for feat in prep.original_feature_names:
            X_df[feat] = (X_df[feat] - prep.means[feat]) / prep.stds[feat]
        for name, vals in indicators.items():
            X_df[name] = vals
        return X_df[prep.feature_names].to_numpy(dtype=np.float32)

    X_tr = transform(tr_df)
    X_val = transform(val_df)
    y_tr = tr_df[TARGET_COL].to_numpy(dtype=float)
    y_val = val_df[TARGET_COL].to_numpy(dtype=float)

    best_rmse = float("inf")
    best_alpha, best_l1 = ELASTIC_NET_ALPHA, ELASTIC_NET_L1
    for alpha in ALPHA_GRID:
        for l1 in L1_GRID:
            m = ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=True, max_iter=10000, random_state=42)
            m.fit(X_tr, y_tr)
            rmse = float(np.sqrt(np.mean((m.predict(X_val) - y_val) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha, best_l1 = alpha, l1
    return best_alpha, best_l1


# ---------------------------------------------------------------------------
# Per-date scenario run
# ---------------------------------------------------------------------------
def run_scenario_at_date(
    anchor_date: pd.Timestamp,
    data: dict,
    var1_prior: VAR1Prior,
    nfci_df: pd.DataFrame | None,
    regime_thresholds: dict[str, float],
    all_results: list[dict],
    question_manifest: list[dict],
    audit_rows: list[dict],
) -> None:
    """Run all scenario questions at a single anchor date."""
    print(f"\n{'='*60}")
    print(f"Anchor date: {anchor_date.date()}")
    print(f"{'='*60}")

    fm = data["feature_master"]
    mp = data["modeling_panel"]
    fsm = data["feature_manifest"]
    tp = data["target_panel"]
    feat_cols = data["feature_columns"]

    # --- Check if feature data exists at this date ---
    date_rows = fm[fm["month_end"] == anchor_date]
    if date_rows.empty:
        print(f"  WARNING: No feature data at {anchor_date.date()}, skipping.")
        return

    # --- Train end: use the end of the benchmark training split
    #     or anchor_date - 6 months (for 2021+, training data extends to 2021-02)
    from xoptpoe_v4_scenario.pipeline import refit_elastic_net
    train_end = min(anchor_date - pd.DateOffset(months=1),
                    pd.Timestamp("2021-02-28"))

    # --- Use locked benchmark EN hyperparams (avoids CV overhead) ---
    alpha, l1 = ELASTIC_NET_ALPHA, ELASTIC_NET_L1
    print(f"  EN alpha={alpha}, l1_ratio={l1} (locked benchmark)")

    # --- Build excess return matrix for EWMA cov ---
    excess_returns_monthly = build_sleeve_excess_returns_monthly(tp, fm)

    # --- Build pipeline ---
    print(f"  Building v4AllocationPipeline...")
    try:
        pipeline = build_pipeline_at_date(
            anchor_date=anchor_date,
            feature_master=fm,
            modeling_panel=mp,
            feature_manifest=fsm,
            feature_columns=feat_cols,
            excess_returns_monthly=excess_returns_monthly,
            elastic_net_alpha=alpha,
            elastic_net_l1=l1,
            train_end=train_end,
            sleeve_order=SLEEVES_14,
        )
    except Exception as e:
        print(f"  ERROR building pipeline: {e}")
        return

    # --- Get observed macro state ---
    m0, _ = load_state(anchor_date, fm, SLEEVES_14)
    scales = state_scales(fm, train_end=TRAIN_END_VAR1)
    a_box, b_box = box_constraints(fm, slack_multiplier=1.0)

    # --- Evaluate pipeline at m0 (sanity check) ---
    result_m0 = pipeline.evaluate_at(m0)
    w0 = result_m0["w"]
    gold_w0 = w0[SLEEVES_14.index("ALT_GLD")]
    eq_us_w0 = w0[SLEEVES_14.index("EQ_US")]
    print(f"  Benchmark weights at m0:")
    for i, s in enumerate(SLEEVES_14):
        if w0[i] > 0.02:
            print(f"    {s}: {w0[i]:.3f}")
    print(f"  Predicted portfolio return: {result_m0['pred_return']:.3%}")
    print(f"  Portfolio entropy: {result_m0['entropy']:.3f}")

    # --- Audit row ---
    audit_rows.append({
        "anchor_date": anchor_date.date(),
        "en_alpha": alpha,
        "en_l1": l1,
        "m0_ig_oas": m0[MACRO_STATE_COLS.index("ig_oas")],
        "m0_vix": m0[MACRO_STATE_COLS.index("vix")],
        "m0_infl_US": m0[MACRO_STATE_COLS.index("infl_US")],
        "m0_short_rate_US": m0[MACRO_STATE_COLS.index("short_rate_US")],
        "m0_long_rate_US": m0[MACRO_STATE_COLS.index("long_rate_US")],
        "m0_us_real10y": m0[MACRO_STATE_COLS.index("us_real10y")],
        "pred_return_m0": result_m0["pred_return"],
        "entropy_m0": result_m0["entropy"],
        "w_ALT_GLD_m0": float(gold_w0),
        "w_EQ_US_m0": float(eq_us_w0),
    })

    # --- Get 60m realized returns if available ---
    realized = get_realized_60m_returns(tp, anchor_date)
    ret_60m = np.array([realized.get(s, np.nan) for s in SLEEVES_14])
    # Where NaN, use predicted return as proxy (conservative)
    for i in range(len(ret_60m)):
        if np.isnan(ret_60m[i]):
            ret_60m[i] = float(result_m0["mu_hat"][i])

    # ----------------------------------------------------------------
    # Define all scenario questions for this date
    # ----------------------------------------------------------------
    questions = []

    # Q-A1: Maximize gold weight (find gold-favorable macro regime)
    G_gold_max, gradG_gold_max = gold_weight_probe(
        pipeline, m0, scales, l2reg=0.05,
        prior=var1_prior, var1_l2reg=0.3,
        sign=-1.0,
    )
    questions.append({
        "question_id": "A1_gold_maximize",
        "question_text": "What macro regime maximizes ALT_GLD allocation?",
        "anchor_date": anchor_date,
        "G": G_gold_max,
        "gradG": gradG_gold_max,
        "G_threshold_multiplier": 0.7,
    })

    # Q-A2: Minimize gold weight (find gold-adverse macro regime)
    G_gold_min, gradG_gold_min = gold_weight_probe(
        pipeline, m0, scales, l2reg=0.05,
        prior=var1_prior, var1_l2reg=0.3,
        sign=+1.0,
    )
    questions.append({
        "question_id": "A2_gold_minimize",
        "question_text": "What macro regime minimizes ALT_GLD allocation?",
        "anchor_date": anchor_date,
        "G": G_gold_min,
        "gradG": gradG_gold_min,
        "G_threshold_multiplier": 0.7,
    })

    # Q-B: Equal-weight excess — when does model beat EW?
    G_ew, gradG_ew = equal_weight_excess_probe(
        pipeline, m0, ret_60m, scales, l2reg=0.05,
        prior=var1_prior, var1_l2reg=0.3,
    )
    questions.append({
        "question_id": "B_equal_weight_excess",
        "question_text": "What macro conditions justify model allocation over equal weight?",
        "anchor_date": anchor_date,
        "G": G_ew,
        "gradG": gradG_ew,
        "G_threshold_multiplier": 0.5,
    })

    # Q-C: House-view return matching (6%, 7%, 10%)
    for hv_return in [0.06, 0.07, 0.10]:
        G_hv, gradG_hv = house_view_return_probe(
            pipeline, m0, hv_return, scales, l2reg=0.05,
            prior=var1_prior, var1_l2reg=0.3,
        )
        questions.append({
            "question_id": f"C_house_view_{int(hv_return*100)}pct",
            "question_text": f"What macro regime justifies {int(hv_return*100)}% annualized SAA return?",
            "anchor_date": anchor_date,
            "G": G_hv,
            "gradG": gradG_hv,
            "G_threshold_multiplier": 0.6,
        })

    # Q-D: Diversification
    G_div, gradG_div = diversification_probe(
        pipeline, m0, scales, l2reg=0.05,
        prior=var1_prior, var1_l2reg=0.3,
    )
    questions.append({
        "question_id": "D_diversification",
        "question_text": "What macro conditions maximise portfolio diversification?",
        "anchor_date": anchor_date,
        "G": G_div,
        "gradG": gradG_div,
        "G_threshold_multiplier": 0.5,
    })

    # ----------------------------------------------------------------
    # Run MALA for each question
    # ----------------------------------------------------------------
    for q in questions:
        qid = q["question_id"]
        print(f"\n  Running MALA for {qid}...")
        G_q = q["G"]
        gradG_q = q["gradG"]

        G_m0 = G_q(m0)
        print(f"    G(m0) = {G_m0:.4f}")

        # Register question in manifest
        question_manifest.append({
            "question_id": qid,
            "anchor_date": anchor_date.date(),
            "question_text": q["question_text"],
            "G_m0": round(G_m0, 6),
            "G_threshold": round(G_m0 * q["G_threshold_multiplier"], 6),
        })

        # Run chains
        trajectories = run_mala_chains(
            G=G_q,
            gradG=gradG_q,
            m0=m0,
            a=a_box,
            b=b_box,
            n_seeds=N_SEEDS,
            n_steps=N_STEPS,
            eta=ETA,
            tau=TAU,
            warmup_frac=WARMUP_FRAC,
            seed=hash(qid + str(anchor_date)) % (2**31),
            verbose=False,
        )

        # Compute trajectory G values for adaptive thresholding
        all_traj_G = []
        for traj in trajectories:
            for t in range(0, len(traj), THINNING):
                all_traj_G.append(G_q(traj[t]))
        if not all_traj_G:
            print(f"    Skipping {qid} — empty trajectories")
            continue

        G_min_traj = float(min(all_traj_G))
        G_p20_traj = float(np.percentile(all_traj_G, 20))

        # Adaptive threshold: best 30% of trajectory samples by G value
        # OR a threshold relative to G_m0 if G_m0 is a meaningful lower bound
        if G_m0 > 1e-6:
            # G_m0 is positive — use fraction of it
            G_threshold = G_m0 * q["G_threshold_multiplier"]
        else:
            # G_m0 ≈ 0 or negative — use percentile of trajectory G values
            # (captures the "good" region of the MALA-explored space)
            G_threshold = G_p20_traj

        print(f"    G(m0)={G_m0:.4f}, G_min_traj={G_min_traj:.4f}, G_p20={G_p20_traj:.4f}, threshold={G_threshold:.4f}")
        valid_samples = filter_trajectories(
            trajectories, G_q, G_threshold, thinning=THINNING
        )
        print(f"    Valid samples: {len(valid_samples)}")

        if len(valid_samples) == 0:
            # Fall back to best 50% of trajectory
            G_threshold_loose = float(np.percentile(all_traj_G, 50))
            valid_samples = filter_trajectories(
                trajectories, G_q, G_threshold_loose, thinning=THINNING
            )
            print(f"    Valid samples (loose p50 threshold={G_threshold_loose:.4f}): {len(valid_samples)}")

        if len(valid_samples) == 0:
            print(f"    Skipping {qid} — no valid samples after loosening threshold")
            continue

        # Classify regimes
        classified = classify_sample_set(valid_samples, regime_thresholds)

        # Evaluate portfolio outcomes for each valid sample
        port_returns, port_entropies, g_values = [], [], []
        weight_matrix = []
        for m_s in valid_samples:
            ev = pipeline.evaluate_at(m_s)
            port_returns.append(ev["pred_return"])
            port_entropies.append(ev["entropy"])
            g_values.append(G_q(m_s))
            weight_matrix.append(ev["w"])

        classified["port_return"] = port_returns
        classified["port_entropy"] = port_entropies
        classified["G_value"] = g_values
        for j, s in enumerate(SLEEVES_14):
            classified[f"w_{s}"] = [w[j] for w in weight_matrix]

        classified["question_id"] = qid
        classified["anchor_date"] = anchor_date.date()

        all_results.append(classified)

        print(f"    Regime breakdown: {classified['regime_label'].value_counts().to_dict()}")
        print(f"    Mean port return: {np.mean(port_returns):.3%}")
        print(f"    Mean ALT_GLD weight: {np.mean([w[SLEEVES_14.index('ALT_GLD')] for w in weight_matrix]):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("v4 Scenario Engine — First Pass")
    print("Benchmark: elastic_net__core_plus_interactions__separate_60")
    print("Allocator: lambda_risk=8.0, kappa=0.10, omega_type=identity")
    print("=" * 70)

    # Load data
    data = load_data()
    fm = data["feature_master"]

    # Load NFCI
    nfci_df = None
    if NFCI_PATH.exists():
        nfci_df = load_nfci(NFCI_PATH)
        print(f"NFCI loaded: {len(nfci_df)} monthly observations")
    else:
        print(f"WARNING: NFCI file not found at {NFCI_PATH}")

    # Compute regime thresholds from training period
    regime_thresholds = compute_regime_thresholds(fm, nfci_df)
    print(f"Regime thresholds computed from {len(fm.drop_duplicates('month_end'))} months")

    # Fit VAR(1) prior on training period (pre-2016)
    print(f"\nFitting VAR(1) prior on training period (up to {TRAIN_END_VAR1.date()})...")
    var1_prior = VAR1Prior.fit_from_feature_master(
        feature_master=fm,
        macro_cols=MACRO_STATE_COLS,
        train_end=TRAIN_END_VAR1,
    )
    print(f"  VAR(1) fitted on {STATE_DIM}-dimensional state")
    print(f"  Innovation covariance trace: {np.trace(var1_prior.Q):.4f}")

    # Compute historical Mahalanobis for reference
    mah_df = var1_prior.historical_mahalanobis(fm, MACRO_STATE_COLS)
    print(f"  Mahalanobis median/p90/p95: "
          f"{mah_df['mahalanobis'].median():.2f} / "
          f"{mah_df['mahalanobis'].quantile(0.90):.2f} / "
          f"{mah_df['mahalanobis'].quantile(0.95):.2f}")

    # Containers
    all_results = []
    question_manifest = []
    audit_rows = []

    # Run scenarios at each anchor date
    for anchor_date in ANCHOR_DATES:
        run_scenario_at_date(
            anchor_date=anchor_date,
            data=data,
            var1_prior=var1_prior,
            nfci_df=nfci_df,
            regime_thresholds=regime_thresholds,
            all_results=all_results,
            question_manifest=question_manifest,
            audit_rows=audit_rows,
        )

    # ----------------------------------------------------------------
    # Compile outputs
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Compiling outputs...")

    # 1. Question manifest
    qm_df = pd.DataFrame(question_manifest)
    qm_df.to_csv(REPORTS_DIR / "scenario_question_manifest_v4.csv", index=False)
    print(f"  Written: scenario_question_manifest_v4.csv ({len(qm_df)} questions)")

    # 2. Pipeline audit
    audit_df = pd.DataFrame(audit_rows)
    _write_pipeline_audit(audit_df, REPORTS_DIR / "scenario_pipeline_audit_v4.md")
    print(f"  Written: scenario_pipeline_audit_v4.md")

    if not all_results:
        print("  WARNING: No valid scenario results produced.")
        _write_empty_conference_takeaways(REPORTS_DIR / "scenario_conference_takeaways_v4.md")
        return

    # 3. Full results
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(REPORTS_DIR / "scenario_results_v4.csv", index=False)
    print(f"  Written: scenario_results_v4.csv ({len(results_df)} valid samples)")

    # 4. Regime summary (per question)
    regime_rows = []
    for qid, grp in results_df.groupby("question_id"):
        rs = regime_summary(grp)
        rs["question_id"] = qid
        regime_rows.append(rs)
    if regime_rows:
        regime_df = pd.concat(regime_rows, ignore_index=True)
        regime_df.to_csv(REPORTS_DIR / "scenario_regime_summary_v4.csv", index=False)
        print(f"  Written: scenario_regime_summary_v4.csv")

    # 5. Selected cases (one representative per regime per question)
    selected = _select_representative_cases(results_df)
    selected.to_csv(REPORTS_DIR / "scenario_selected_cases_v4.csv", index=False)
    print(f"  Written: scenario_selected_cases_v4.csv ({len(selected)} cases)")

    # 6. Conference takeaways
    _write_conference_takeaways(results_df, audit_df, REPORTS_DIR / "scenario_conference_takeaways_v4.md")
    print(f"  Written: scenario_conference_takeaways_v4.md")

    print("\nDone.")


def _select_representative_cases(results_df: pd.DataFrame) -> pd.DataFrame:
    """Select one high-quality representative sample per (question_id, regime_label)."""
    rows = []
    for (qid, regime), grp in results_df.groupby(["question_id", "regime_label"]):
        # Pick sample with lowest G_value (best satisfying the probe)
        if "G_value" in grp.columns:
            best = grp.loc[grp["G_value"].idxmin()]
        else:
            best = grp.iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def _write_pipeline_audit(audit_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Scenario Pipeline Audit — v4 First Pass",
        "",
        "## Benchmark Object",
        "- Predictor: `elastic_net__core_plus_interactions__separate_60`",
        "- Allocator: `lambda_risk=8.0`, `kappa=0.10`, `omega_type=identity`",
        "- 14-sleeve universe",
        "",
        "## Macro State",
        f"- Dimension: {STATE_DIM} variables",
        f"- Variables: {', '.join(MACRO_STATE_COLS)}",
        "",
        "## VAR(1) Prior",
        f"- Fitted on training period up to {TRAIN_END_VAR1.date()}",
        "- Used as Mahalanobis plausibility regularizer in all probing functions",
        "",
        "## Per-Anchor Sanity Checks",
        "",
    ]

    if audit_df.empty:
        lines.append("No audit rows available.")
    else:
        for _, row in audit_df.iterrows():
            lines.append(f"### {row.get('anchor_date', 'N/A')}")
            lines.append(f"- EN alpha={row.get('en_alpha', 'N/A'):.4f}, l1={row.get('en_l1', 'N/A'):.2f}")
            lines.append(f"- m0 key values: infl_US={row.get('m0_infl_US', 0):.2f}%, "
                         f"short_rate_US={row.get('m0_short_rate_US', 0):.2f}%, "
                         f"ig_oas={row.get('m0_ig_oas', 0):.2f}, "
                         f"vix={row.get('m0_vix', 0):.2f}")
            lines.append(f"- Benchmark predicted return: {row.get('pred_return_m0', 0)*100:.2f}%")
            lines.append(f"- Portfolio entropy: {row.get('entropy_m0', 0):.3f}")
            lines.append(f"- w_ALT_GLD: {row.get('w_ALT_GLD_m0', 0):.3f}, "
                         f"w_EQ_US: {row.get('w_EQ_US_m0', 0):.3f}")
            lines.append("")

    path.write_text("\n".join(lines))


def _write_conference_takeaways(
    results_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Scenario Conference Takeaways — v4 First Pass",
        "",
        "## Overview",
        "Generated macro scenarios using MALA sampling with VAR(1) plausibility prior,",
        "probing the locked v4 SAA benchmark: elastic_net__core_plus_interactions__separate_60,",
        "best_60_tuned_robust (lambda_risk=8, kappa=0.1, omega=identity).",
        "",
    ]

    # Gold question summary
    gold_q = results_df[results_df["question_id"].isin(["A1_gold_maximize", "A2_gold_minimize"])]
    if not gold_q.empty:
        lines.append("## A. Gold Allocation Question")
        lines.append("")
        gold_max = gold_q[gold_q["question_id"] == "A1_gold_maximize"]
        gold_min = gold_q[gold_q["question_id"] == "A2_gold_minimize"]
        if not gold_max.empty:
            mean_gold_weight_max = gold_max[f"w_ALT_GLD"].mean()
            top_regimes = gold_max["regime_label"].value_counts().head(3).to_dict()
            lines.append(f"**Gold-favorable scenarios** (A1, maximize gold weight):")
            lines.append(f"- Mean ALT_GLD weight in generated scenarios: {mean_gold_weight_max:.1%}")
            lines.append(f"- Most common regimes: {top_regimes}")
            macro_gold = {}
            for col in MACRO_STATE_COLS[:10]:
                if col in gold_max.columns:
                    macro_gold[col] = gold_max[col].mean()
            lines.append(f"- Key macro signals in gold-favorable states:")
            for k, v in macro_gold.items():
                lines.append(f"  - {k}: {v:.2f}")
        if not gold_min.empty:
            mean_gold_weight_min = gold_min[f"w_ALT_GLD"].mean()
            top_regimes_min = gold_min["regime_label"].value_counts().head(3).to_dict()
            lines.append(f"")
            lines.append(f"**Gold-adverse scenarios** (A2, minimize gold weight):")
            lines.append(f"- Mean ALT_GLD weight: {mean_gold_weight_min:.1%}")
            lines.append(f"- Most common regimes: {top_regimes_min}")
        lines.append("")

    # Equal weight question
    ew_q = results_df[results_df["question_id"] == "B_equal_weight_excess"]
    if not ew_q.empty:
        lines.append("## B. Equal-Weight vs Model Question")
        lines.append("")
        mean_port_ret = ew_q["port_return"].mean()
        top_regimes = ew_q["regime_label"].value_counts().head(3).to_dict()
        lines.append(f"- Scenarios where model most clearly beats equal weight:")
        lines.append(f"  - Mean predicted portfolio return: {mean_port_ret:.1%}")
        lines.append(f"  - Dominant regimes: {top_regimes}")
        lines.append("")

    # House view question
    hv_q = results_df[results_df["question_id"].str.startswith("C_house_view")]
    if not hv_q.empty:
        lines.append("## C. House-View Return Question")
        lines.append("")
        for qid in sorted(hv_q["question_id"].unique()):
            chunk = hv_q[hv_q["question_id"] == qid]
            hv_pct = qid.split("_")[-1].replace("pct", "%")
            mean_ret = chunk["port_return"].mean()
            top_reg = chunk["regime_label"].value_counts().head(2).to_dict()
            lines.append(f"**{hv_pct} house-view target:**")
            lines.append(f"  - Valid scenarios: {len(chunk)}")
            lines.append(f"  - Mean model predicted return: {mean_ret:.1%}")
            lines.append(f"  - Dominant regimes: {top_reg}")
        lines.append("")

    # Diversification question
    div_q = results_df[results_df["question_id"] == "D_diversification"]
    if not div_q.empty:
        lines.append("## D. Diversification Question")
        lines.append("")
        mean_ent = div_q["port_entropy"].mean()
        top_regimes = div_q["regime_label"].value_counts().head(3).to_dict()
        lines.append(f"- Macro conditions that maximize portfolio entropy:")
        lines.append(f"  - Mean entropy: {mean_ent:.3f}")
        lines.append(f"  - Dominant regimes: {top_regimes}")
        lines.append("")

    lines.extend([
        "## Methodology Note",
        "",
        "- Sampler: MALA (Metropolis-Adjusted Langevin Algorithm)",
        f"- MALA parameters: eta={ETA}, tau={TAU}, n_steps={N_STEPS}, n_seeds={N_SEEDS}",
        "- Plausibility: VAR(1) Mahalanobis regularizer fitted on training period",
        "- Regime classification: threshold-based on NFCI + NBER overlays",
        f"- State dimension: {STATE_DIM} global macro-financial variables",
        "- Gradient: central finite differences (epsilon=1e-4)",
        "",
        "## Limitations of First Pass",
        "",
        "- ElasticNet is refit for each anchor date on available training data.",
        "  The model used here approximates the benchmark; the exact benchmark model",
        "  parameters from the accepted prediction_benchmark run should be used for",
        "  production scenarios.",
        "- Realized 60m returns are unavailable for 2023-12 and 2024-12 anchors",
        "  (5-year windows ending 2028-12 and 2029-12 have not happened yet).",
        "  House-view probes use model-predicted returns, which is conservative.",
        "- The regime classifier is rule-based on 8 thresholds; a data-driven",
        "  classifier (HMM or k-means on the macro state) is the next step.",
    ])

    path.write_text("\n".join(lines))


def _write_empty_conference_takeaways(path: Path) -> None:
    path.write_text(
        "# Scenario Conference Takeaways — v4 First Pass\n\n"
        "No valid scenario results produced in this run.\n"
        "Check pipeline audit for diagnostics.\n"
    )


if __name__ == "__main__":
    main()
