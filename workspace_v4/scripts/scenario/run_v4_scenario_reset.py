#!/usr/bin/env python3
"""
run_v4_scenario_reset.py

v4 Scenario Reset Engine — Historical Analog + Bounded Grid + VAR(1)-Constrained
Gradient Refinement.

Replaces the MALA-first approach with:
  Stage 1: Historical analog candidate generation (K=30) + LHS candidates (N=200)
  Stage 2: Bounded gradient descent refinement (50-60 steps, no noise)
  Stage 3: Ranking and selection (top 6 per question-anchor, min 2 distinct regimes)

Four questions:
  Q1_gold_threshold: Gold activation threshold
  Q2_ew_departure: Equal-weight departure
  Q3_return_discipline: Return with discipline
  Q4_return_ceiling: Return ceiling

Outputs to workspace_v4/reports/:
  scenario_reset_results_v4.csv
  scenario_reset_regime_summary_v4.csv
  scenario_reset_selected_cases_v4.csv
  scenario_reset_question_manifest_v4.csv
  scenario_reset_state_shift_summary_v4.csv
  scenario_reset_portfolio_response_summary_v4.csv
  scenario_reset_regime_transition_summary_v4.csv
"""
from __future__ import annotations

import sys
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

# Existing v4 scenario imports
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
    equal_weight_excess_probe,
    house_view_return_probe,
)

# New reset module imports
from xoptpoe_v4_scenario_reset.analog_search import find_analogs, ANALOG_FILTERS
from xoptpoe_v4_scenario_reset.grid_sampler import generate_lhs_candidates
from xoptpoe_v4_scenario_reset.gradient_refiner import refine_batch
from xoptpoe_v4_scenario_reset.ranker import score_candidates, select_diverse
from xoptpoe_v4_scenario_reset.regime_v2 import (
    build_regime_thresholds,
    classify_regime_v2,
    compute_regime_transition,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_REFS = WORKSPACE / "data_refs"
REPORTS_DIR = WORKSPACE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ELASTIC_NET_ALPHA = 0.005
ELASTIC_NET_L1 = 0.5
TRAIN_END_VAR1 = pd.Timestamp("2016-02-29")
MODEL_TRAIN_END = pd.Timestamp("2021-02-28")

ANCHOR_DATES = [
    pd.Timestamp("2021-12-31"),
    pd.Timestamp("2022-12-31"),
    pd.Timestamp("2023-12-31"),
    pd.Timestamp("2024-12-31"),
]

# Anchor truth from scenario_anchor_truth_v4.csv
ANCHOR_TRUTH = {
    pd.Timestamp("2021-12-31"): {"pred_return": 0.0201, "w_ALT_GLD": 0.081, "w_EQ_US": 0.256},
    pd.Timestamp("2022-12-31"): {"pred_return": 0.0330, "w_ALT_GLD": 0.223, "w_EQ_US": 0.137},
    pd.Timestamp("2023-12-31"): {"pred_return": 0.0282, "w_ALT_GLD": 0.224, "w_EQ_US": 0.163},
    pd.Timestamp("2024-12-31"): {"pred_return": 0.0224, "w_ALT_GLD": 0.232, "w_EQ_US": 0.185},
}

TOLERANCE_PRED_RETURN = 0.005    # 0.5 pct absolute tolerance
TOLERANCE_GOLD_WEIGHT = 0.05     # 5 pct absolute tolerance

# Stage 1 & 2 parameters  (tuned for speed: ~5-8 min total)
K_ANALOGS = 20
N_LHS = 60
N_REFINE_TOP = 15      # refine top 15 combined candidates
N_GD_STEPS = 15        # 15 steps × 38 evals × 15 cands = ~8,550 evals/question
GD_LR = 0.005          # larger lr to compensate fewer steps
N_SELECT = 5
MIN_REGIMES = 2


# ---------------------------------------------------------------------------
# Question definitions
# ---------------------------------------------------------------------------
QUESTIONS = [
    {
        "question_id": "Q1_gold_threshold",
        "question_text": (
            "Gold activation threshold — what regime makes gold go "
            "from minor to material allocation?"
        ),
        "probe_type": "gold_maximize",
        "target_value": None,
    },
    {
        "question_id": "Q2_ew_departure",
        "question_text": (
            "Equal-weight departure — when does benchmark move strongly "
            "away from EW, and where does it concentrate?"
        ),
        "probe_type": "ew_excess",
        "target_value": None,
    },
    {
        "question_id": "Q3_return_discipline",
        "question_text": (
            "Return with discipline — what regime improves return "
            "without excessive concentration?"
        ),
        "probe_type": "house_view",
        "target_value": 0.04,  # target 4% return (disciplined, achievable)
    },
    {
        "question_id": "Q4_return_ceiling",
        "question_text": (
            "Return ceiling — what regime would support 5%+ annualized, "
            "and why can't benchmark reach it plausibly?"
        ),
        "probe_type": "house_view",
        "target_value": 0.05,  # target 5% return
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> dict:
    print("Loading data artifacts...")
    fm = pd.read_parquet(DATA_REFS / "feature_master_monthly.parquet")
    fm["month_end"] = pd.to_datetime(fm["month_end"])

    mp = pd.read_parquet(DATA_REFS / "modeling_panel_hstack.parquet")
    mp["month_end"] = pd.to_datetime(mp["month_end"])

    fsm = pd.read_csv(DATA_REFS / "feature_set_manifest.csv")

    tp = pd.read_parquet(DATA_REFS / "target_panel_long_horizon.parquet")
    tp["month_end"] = pd.to_datetime(tp["month_end"])

    feat_cols = fsm[fsm["include_core_plus_interactions"] == 1]["feature_name"].tolist()

    print(f"  feature_master: {len(fm)} rows")
    print(f"  modeling_panel: {len(mp)} rows")
    print(f"  core_plus_interactions features: {len(feat_cols)}")

    return {
        "feature_master": fm,
        "modeling_panel": mp,
        "feature_manifest": fsm,
        "target_panel": tp,
        "feature_columns": feat_cols,
    }


def build_excess_returns_monthly(tp: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    """Build monthly excess return matrix for EWMA covariance."""
    tp1m = pd.DataFrame()
    if "horizon_months" in tp.columns and 1 in tp["horizon_months"].values:
        tp1m = tp[tp["horizon_months"] == 1].copy()

    if not tp1m.empty and "annualized_excess_forward_return" in tp1m.columns:
        pivot = tp1m.pivot_table(
            index="month_end", columns="sleeve_id",
            values="annualized_excess_forward_return", aggfunc="first",
        )
        available = [s for s in SLEEVES_14 if s in pivot.columns]
        return pivot[available].sort_index()

    # Fallback: ret_1m_lag from feature_master
    if "ret_1m_lag" in fm.columns:
        pivot = fm.pivot_table(
            index="month_end", columns="sleeve_id",
            values="ret_1m_lag", aggfunc="first",
        )
        available = [s for s in SLEEVES_14 if s in pivot.columns]
        return pivot[available].sort_index()

    dates = pd.to_datetime(fm["month_end"].unique())
    return pd.DataFrame(0.0, index=sorted(dates), columns=list(SLEEVES_14))


def get_realized_60m_returns(tp: pd.DataFrame, anchor_date: pd.Timestamp) -> np.ndarray:
    """Get 60m realized returns at anchor date; fill NaN with zero."""
    if "horizon_months" not in tp.columns:
        return np.zeros(len(SLEEVES_14))

    chunk = tp[(tp["month_end"] == anchor_date) & (tp["horizon_months"] == 60)]
    ret_col = "annualized_excess_forward_return"
    if chunk.empty or ret_col not in chunk.columns:
        return np.zeros(len(SLEEVES_14))

    ret_60m = np.zeros(len(SLEEVES_14))
    for i, sleeve in enumerate(SLEEVES_14):
        row = chunk[chunk["sleeve_id"] == sleeve]
        if not row.empty:
            ret_60m[i] = float(row[ret_col].iloc[0])
    return ret_60m


# ---------------------------------------------------------------------------
# Baseline check
# ---------------------------------------------------------------------------
def check_anchor_baseline(pipeline, m0: np.ndarray, anchor_date: pd.Timestamp) -> dict:
    """Verify anchor truth for a given date."""
    result = pipeline.evaluate_at(m0)
    w = result["w"]
    pred_return = result["pred_return"]
    w_gold = float(w[list(SLEEVES_14).index("ALT_GLD")])
    w_eq_us = float(w[list(SLEEVES_14).index("EQ_US")])

    truth = ANCHOR_TRUTH.get(anchor_date, {})
    checks = {}
    if truth:
        pred_ret_err = abs(pred_return - truth["pred_return"])
        gold_err = abs(w_gold - truth["w_ALT_GLD"])
        checks["pred_return_err"] = pred_ret_err
        checks["gold_weight_err"] = gold_err
        checks["pred_return_ok"] = pred_ret_err <= TOLERANCE_PRED_RETURN
        checks["gold_weight_ok"] = gold_err <= TOLERANCE_GOLD_WEIGHT

        if not checks["pred_return_ok"]:
            print(
                f"  WARNING: pred_return mismatch at {anchor_date.date()}: "
                f"got {pred_return:.4f}, truth={truth['pred_return']:.4f}, "
                f"err={pred_ret_err:.4f} > tol={TOLERANCE_PRED_RETURN}"
            )
        if not checks["gold_weight_ok"]:
            print(
                f"  WARNING: gold weight mismatch at {anchor_date.date()}: "
                f"got {w_gold:.4f}, truth={truth['w_ALT_GLD']:.4f}, "
                f"err={gold_err:.4f} > tol={TOLERANCE_GOLD_WEIGHT}"
            )
        if checks.get("pred_return_ok") and checks.get("gold_weight_ok"):
            print(f"  Baseline check PASSED for {anchor_date.date()}")

    return {
        "anchor_date": anchor_date,
        "pred_return": pred_return,
        "w_ALT_GLD": w_gold,
        "w_EQ_US": w_eq_us,
        **checks,
    }


# ---------------------------------------------------------------------------
# Build probe (G, gradG) for each question type
# ---------------------------------------------------------------------------
# Numerical gradient helper (needed for pure-objective probes)
def _numerical_grad(G, m, epsilon=1e-3):
    """Forward-difference gradient (19 evals vs 38 for central diff)."""
    m = np.asarray(m, dtype=float)
    G0 = G(m)
    grad = np.zeros_like(m)
    for i in range(len(m)):
        m_plus = m.copy(); m_plus[i] += epsilon
        grad[i] = (G(m_plus) - G0) / epsilon
    return grad


def build_probe(
    question: dict,
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    prior: VAR1Prior,
    ret_60m: np.ndarray,
    refinement_mode: bool = False,
):
    """Build (G, gradG) tuple for a question.

    If refinement_mode=True, builds an objective-only probe (no regularizer)
    for gradient descent — allows refinement to maximize the question objective
    without the regularizer pulling everything toward the VAR(1) prediction.

    If refinement_mode=False, builds the full regularized probe for ranking/scoring.
    """
    probe_type = question["probe_type"]
    target = question.get("target_value")

    if refinement_mode:
        # Pure objective function — no regularizer
        # This allows the gradient to point toward the question target, not the VAR(1) anchor
        GOLD_IDX = list(SLEEVES_14).index("ALT_GLD")
        EPS = 1e-10

        if probe_type == "gold_maximize":
            def G(m):
                m = np.asarray(m, dtype=float)
                w = pipeline(m)
                return float(-w[GOLD_IDX])  # maximize gold weight
            def gradG(m): return _numerical_grad(G, m)

        elif probe_type == "ew_excess":
            ret = np.asarray(ret_60m, dtype=float)
            n = len(ret)
            ew = np.ones(n) / n
            def G(m):
                m = np.asarray(m, dtype=float)
                w = pipeline(m)
                return -(float(w @ ret) - float(ew @ ret))
            def gradG(m): return _numerical_grad(G, m)

        elif probe_type == "house_view":
            if target is None:
                target = 0.05
            _target = target
            def G(m):
                m = np.asarray(m, dtype=float)
                result = pipeline.evaluate_at(m)
                return (result["pred_return"] - _target) ** 2
            def gradG(m): return _numerical_grad(G, m)

        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

    else:
        # Full regularized probe (for ranking / plausibility scoring)
        if probe_type == "gold_maximize":
            G, gradG = gold_weight_probe(
                pipeline, m0, scales, l2reg=0.05,
                prior=prior, var1_l2reg=0.3, sign=-1.0,
            )
        elif probe_type == "ew_excess":
            G, gradG = equal_weight_excess_probe(
                pipeline, m0, ret_60m, scales, l2reg=0.05,
                prior=prior, var1_l2reg=0.3,
            )
        elif probe_type == "house_view":
            if target is None:
                target = 0.05
            G, gradG = house_view_return_probe(
                pipeline, m0, target, scales, l2reg=0.05,
                prior=prior, var1_l2reg=0.3,
            )
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

    return G, gradG


# ---------------------------------------------------------------------------
# Main per-anchor-question loop
# ---------------------------------------------------------------------------
def run_question_at_date(
    question: dict,
    anchor_date: pd.Timestamp,
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    prior: VAR1Prior,
    ret_60m: np.ndarray,
    feature_master: pd.DataFrame,
    a_box: np.ndarray,
    b_box: np.ndarray,
    regime_thresholds: dict,
) -> list[dict]:
    """
    Run a single question at a single anchor date.
    Returns list of candidate dicts (all refined candidates, before selection).
    """
    question_id = question["question_id"]
    print(f"  Question: {question_id}")

    # Build probe
    G, gradG = build_probe(question, pipeline, m0, scales, prior, ret_60m)

    # --- Stage 1a: Historical analogs ---
    analogs_df = find_analogs(
        feature_master, question_id, anchor_date, m0, scales, K=K_ANALOGS,
    )
    print(f"    Stage1a: {len(analogs_df)} historical analogs found")

    # Extract macro states from analogs
    analog_states = []
    if not analogs_df.empty:
        for _, row in analogs_df.iterrows():
            m_row = np.array([float(row.get(col, m0[i])) for i, col in enumerate(MACRO_STATE_COLS)])
            analog_states.append(m_row)

    # --- Stage 1b: LHS samples ---
    try:
        lhs_states = generate_lhs_candidates(
            prior, m0, feature_master, MACRO_STATE_COLS,
            n_samples=N_LHS, n_sigma=3.0, rng_seed=42, plausibility_pct=90.0,
        )
        print(f"    Stage1b: {len(lhs_states)} LHS candidates (post-filter)")
    except Exception as e:
        print(f"    Stage1b: LHS failed ({e}), skipping")
        lhs_states = np.empty((0, STATE_DIM))

    # --- Combine and rank by G (quick evaluation) ---
    all_m_states = analog_states + [lhs_states[i] for i in range(len(lhs_states))]

    if not all_m_states:
        print(f"    No candidates for {question_id} at {anchor_date.date()}")
        return []

    # Quick evaluate all candidates to rank
    quick_scores = []
    for m_c in all_m_states:
        try:
            g_val = G(m_c)
            quick_scores.append((m_c, g_val))
        except Exception:
            quick_scores.append((m_c, float("inf")))

    # Sort by G (ascending = better)
    quick_scores.sort(key=lambda x: x[1])

    # Take top N_REFINE_TOP
    top_candidates = [m for m, _ in quick_scores[:N_REFINE_TOP]]
    print(f"    Stage2: refining top {len(top_candidates)} candidates...")

    # --- Stage 2: Gradient refinement ---
    refined_results = refine_batch(
        top_candidates, G, gradG, a_box, b_box,
        n_steps=N_GD_STEPS, lr=GD_LR,
    )
    print(f"    Stage2: {sum(1 for _, _, c in refined_results if c)} converged")

    # --- Build candidate rows ---
    candidate_rows = []
    for i, (m_ref, g_final, converged) in enumerate(refined_results):
        # Evaluate full pipeline at refined state
        try:
            result = pipeline.evaluate_at(m_ref)
            w = result["w"]
            pred_return = result["pred_return"]
            entropy = result["entropy"]
        except Exception:
            w = np.ones(len(SLEEVES_14)) / len(SLEEVES_14)
            pred_return = 0.0
            entropy = 0.0

        # Regime classification v2
        regime_label, dim_scores = classify_regime_v2(m_ref, regime_thresholds)

        # Transition from anchor
        transition = compute_regime_transition(m0, m_ref, regime_thresholds, scales)

        row = {
            "question_id": question_id,
            "anchor_date": anchor_date.date(),
            "candidate_idx": i,
            "G_final": float(g_final),
            "converged": bool(converged),
            "pred_return": float(pred_return),
            "entropy": float(entropy),
            "regime_label": regime_label,
            "anchor_regime": transition["anchor_regime"],
            "regime_changed": bool(transition["regime_changed"]),
        }

        # Macro state
        for j, col in enumerate(MACRO_STATE_COLS):
            row[col] = float(m_ref[j])

        # Weights
        for j, sleeve in enumerate(SLEEVES_14):
            row[f"w_{sleeve}"] = float(w[j])

        # Dimension scores
        for dim_name, dim_val in dim_scores.items():
            row[dim_name] = float(dim_val)

        # Top macro shift
        if transition["key_shifts"]:
            row["top_shift_var"] = transition["key_shifts"][0]["variable"]
            row["top_shift_std"] = float(transition["key_shifts"][0]["std_shift"])

        candidate_rows.append(row)

    return candidate_rows


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("v4 Scenario Reset Engine")
    print("Historical Analog + LHS + Gradient Refinement")
    print("=" * 70)

    # Load data
    data = load_data()
    fm = data["feature_master"]
    mp = data["modeling_panel"]
    fsm = data["feature_manifest"]
    tp = data["target_panel"]
    feat_cols = data["feature_columns"]

    # Fit VAR(1) prior (fixed on pre-2016 data)
    print("\nFitting VAR(1) prior...")
    prior = VAR1Prior.fit_from_feature_master(
        fm, MACRO_STATE_COLS, train_end=TRAIN_END_VAR1
    )
    print("  VAR(1) fitted.")

    # Build regime thresholds from full history
    print("Building regime thresholds...")
    regime_thresholds = build_regime_thresholds(fm)

    # Excess returns for EWMA cov
    excess_returns_monthly = build_excess_returns_monthly(tp, fm)

    # Box constraints (historical range + slack)
    a_box, b_box = box_constraints(fm, slack_multiplier=1.0)

    # Scales for standardization
    scales = state_scales(fm, train_end=TRAIN_END_VAR1)

    # Results accumulator
    all_results = []
    all_selected = []
    manifest_rows = []
    transition_rows = []

    # -------------------------------------------------------------------------
    # Main loop: anchor dates × questions
    # -------------------------------------------------------------------------
    for anchor_date in ANCHOR_DATES:
        print(f"\n{'='*60}")
        print(f"Anchor: {anchor_date.date()}")
        print(f"{'='*60}")

        # Check data availability
        fm_rows = fm[fm["month_end"] == anchor_date]
        if fm_rows.empty:
            print(f"  No feature data at {anchor_date.date()}, skipping.")
            continue

        # Build pipeline
        train_end = min(anchor_date - pd.DateOffset(months=1), MODEL_TRAIN_END)
        print(f"  Building pipeline (train_end={train_end.date()})...")
        try:
            pipeline = build_pipeline_at_date(
                anchor_date=anchor_date,
                feature_master=fm,
                modeling_panel=mp,
                feature_manifest=fsm,
                feature_columns=feat_cols,
                excess_returns_monthly=excess_returns_monthly,
                elastic_net_alpha=ELASTIC_NET_ALPHA,
                elastic_net_l1=ELASTIC_NET_L1,
                train_end=train_end,
                sleeve_order=SLEEVES_14,
            )
        except Exception as e:
            print(f"  ERROR building pipeline: {e}")
            continue

        # Load anchor state
        m0, _ = load_state(anchor_date, fm, SLEEVES_14)

        # Baseline check
        print("  Running baseline check...")
        baseline = check_anchor_baseline(pipeline, m0, anchor_date)

        # Get 60m realized returns
        ret_60m = get_realized_60m_returns(tp, anchor_date)
        # Fill zeros with model mu_hat
        result_m0 = pipeline.evaluate_at(m0)
        for i in range(len(ret_60m)):
            if ret_60m[i] == 0.0:
                ret_60m[i] = float(result_m0["mu_hat"][i])

        # Run each question
        for question in QUESTIONS:
            qid = question["question_id"]
            print(f"\n  Running {qid}...")

            candidates = run_question_at_date(
                question, anchor_date, pipeline, m0, scales, prior,
                ret_60m, fm, a_box, b_box, regime_thresholds,
            )

            if not candidates:
                print(f"    No candidates for {qid}")
                continue

            candidates_df = pd.DataFrame(candidates)
            all_results.extend(candidates)

            # Stage 3: rank and select
            scored_df = score_candidates(candidates_df, prior, m0)

            # Add regime label for diversity selection
            if "regime_label" not in scored_df.columns:
                scored_df["regime_label"] = "unknown"

            selected_df = select_diverse(scored_df, n_select=N_SELECT, min_regimes=MIN_REGIMES)
            selected_df["selection_rank"] = range(1, len(selected_df) + 1)

            all_selected.extend(selected_df.to_dict("records"))

            # Manifest row
            regime_counts = candidates_df["regime_label"].value_counts()
            n_regimes = len(regime_counts)
            dominant_regime = regime_counts.index[0] if len(regime_counts) > 0 else "unknown"
            dominant_share = float(regime_counts.iloc[0] / len(candidates_df)) if len(candidates_df) > 0 else 0.0

            # Anchor baseline values
            anchor_truth = ANCHOR_TRUTH.get(anchor_date, {})

            manifest_rows.append({
                "question_id": qid,
                "question_text": question["question_text"],
                "anchor_date": anchor_date.date(),
                "n_candidates": len(candidates_df),
                "n_selected": len(selected_df),
                "regime_diversity": n_regimes,
                "dominant_regime": dominant_regime,
                "dominant_regime_share": dominant_share,
                "mean_pred_return": float(candidates_df["pred_return"].mean()),
                "std_pred_return": float(candidates_df["pred_return"].std()),
                "mean_entropy": float(candidates_df["entropy"].mean()),
                "anchor_pred_return": anchor_truth.get("pred_return", float("nan")),
                "anchor_w_ALT_GLD": anchor_truth.get("w_ALT_GLD", float("nan")),
                "anchor_w_EQ_US": anchor_truth.get("w_EQ_US", float("nan")),
            })

            # Regime transition rows for selected candidates
            for _, sel_row in selected_df.iterrows():
                m_sel = np.array([sel_row.get(col, m0[i]) for i, col in enumerate(MACRO_STATE_COLS)])
                trans = compute_regime_transition(m0, m_sel, regime_thresholds, scales)
                for shift in trans["key_shifts"]:
                    transition_rows.append({
                        "question_id": qid,
                        "anchor_date": anchor_date.date(),
                        "candidate_idx": sel_row.get("candidate_idx", -1),
                        "anchor_regime": trans["anchor_regime"],
                        "scenario_regime": trans["scenario_regime"],
                        "regime_changed": trans["regime_changed"],
                        "shift_variable": shift["variable"],
                        "shift_raw": shift["raw_shift"],
                        "shift_std": shift["std_shift"],
                        "implication": trans["implication"],
                    })

            n_regimes_selected = len(selected_df["regime_label"].unique()) if "regime_label" in selected_df.columns else 0
            print(
                f"    Selected: {len(selected_df)} candidates, "
                f"{n_regimes_selected} distinct regimes, "
                f"pred_return range=[{candidates_df['pred_return'].min():.3f}, "
                f"{candidates_df['pred_return'].max():.3f}]"
            )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Writing output files...")

    if not all_results:
        print("ERROR: No results to write. Check pipeline errors above.")
        return

    results_df = pd.DataFrame(all_results)
    results_path = REPORTS_DIR / "scenario_reset_results_v4.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Wrote: {results_path} ({len(results_df)} rows)")

    # Regime summary
    if "regime_label" in results_df.columns:
        regime_summary = (
            results_df.groupby(["question_id", "anchor_date", "regime_label"])
            .agg(
                n_candidates=("candidate_idx", "count"),
                mean_pred_return=("pred_return", "mean"),
                mean_gold_weight=("w_ALT_GLD", "mean"),
                mean_G_final=("G_final", "mean"),
            )
            .reset_index()
        )
        regime_sum_path = REPORTS_DIR / "scenario_reset_regime_summary_v4.csv"
        regime_summary.to_csv(regime_sum_path, index=False)
        print(f"  Wrote: {regime_sum_path}")

    # Selected cases
    if all_selected:
        selected_df = pd.DataFrame(all_selected)
        sel_path = REPORTS_DIR / "scenario_reset_selected_cases_v4.csv"
        selected_df.to_csv(sel_path, index=False)
        print(f"  Wrote: {sel_path} ({len(selected_df)} rows)")

    # Question manifest
    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        mfst_path = REPORTS_DIR / "scenario_reset_question_manifest_v4.csv"
        manifest_df.to_csv(mfst_path, index=False)
        print(f"  Wrote: {mfst_path}")

    # State shift summary
    key_shift_cols = ["infl_US", "ig_oas", "vix", "us_real10y", "short_rate_US", "term_slope_US"]
    shift_rows = []
    for _, row in results_df.iterrows():
        anchor_date_ts = pd.Timestamp(row["anchor_date"])
        m0_ref = np.array([
            fm[fm["month_end"] == anchor_date_ts][col].iloc[0]
            if not fm[fm["month_end"] == anchor_date_ts].empty else 0.0
            for col in MACRO_STATE_COLS
        ])
        for col in key_shift_cols:
            if col in row and col in MACRO_STATE_COLS:
                i = MACRO_STATE_COLS.index(col)
                raw_shift = float(row[col]) - float(m0_ref[i])
                std_shift = raw_shift / max(float(scales[i]), 1e-8)
                shift_rows.append({
                    "question_id": row["question_id"],
                    "anchor_date": row["anchor_date"],
                    "candidate_idx": row.get("candidate_idx", -1),
                    "variable": col,
                    "anchor_value": float(m0_ref[i]),
                    "scenario_value": float(row[col]),
                    "raw_shift": raw_shift,
                    "std_shift": std_shift,
                })

    if shift_rows:
        shift_df = pd.DataFrame(shift_rows)
        shift_path = REPORTS_DIR / "scenario_reset_state_shift_summary_v4.csv"
        shift_df.to_csv(shift_path, index=False)
        print(f"  Wrote: {shift_path}")

    # Portfolio response summary
    port_cols = [f"w_{s}" for s in SLEEVES_14]
    anchor_baselines = {}
    for ad in ANCHOR_DATES:
        truth = ANCHOR_TRUTH.get(ad, {})
        anchor_baselines[ad.date()] = truth

    port_rows = []
    for _, row in results_df.iterrows():
        ad_key = row["anchor_date"]
        truth = ANCHOR_TRUTH.get(pd.Timestamp(str(ad_key)), {})
        w_gold = float(row.get("w_ALT_GLD", 0.0))
        w_eq_us = float(row.get("w_EQ_US", 0.0))
        gold_shift = w_gold - truth.get("w_ALT_GLD", w_gold)
        eq_shift = w_eq_us - truth.get("w_EQ_US", w_eq_us)
        port_rows.append({
            "question_id": row["question_id"],
            "anchor_date": ad_key,
            "candidate_idx": row.get("candidate_idx", -1),
            "pred_return": float(row.get("pred_return", 0.0)),
            "pred_return_vs_anchor": float(row.get("pred_return", 0.0)) - truth.get("pred_return", 0.0),
            "w_ALT_GLD": w_gold,
            "w_ALT_GLD_vs_anchor": gold_shift,
            "w_EQ_US": w_eq_us,
            "w_EQ_US_vs_anchor": eq_shift,
            "entropy": float(row.get("entropy", 0.0)),
            "regime_label": row.get("regime_label", "unknown"),
        })

    if port_rows:
        port_df = pd.DataFrame(port_rows)
        port_path = REPORTS_DIR / "scenario_reset_portfolio_response_summary_v4.csv"
        port_df.to_csv(port_path, index=False)
        print(f"  Wrote: {port_path}")

    # Regime transition summary
    if transition_rows:
        trans_df = pd.DataFrame(transition_rows)
        trans_path = REPORTS_DIR / "scenario_reset_regime_transition_summary_v4.csv"
        trans_df.to_csv(trans_path, index=False)
        print(f"  Wrote: {trans_path}")

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if manifest_rows:
        mdf = pd.DataFrame(manifest_rows)
        for qid in mdf["question_id"].unique():
            q_rows = mdf[mdf["question_id"] == qid]
            print(f"\n{qid}:")
            print(f"  Mean regime diversity: {q_rows['regime_diversity'].mean():.1f}")
            print(f"  Mean dominant regime share: {q_rows['dominant_regime_share'].mean():.2%}")
            print(f"  pred_return range across anchors: "
                  f"[{q_rows['mean_pred_return'].min():.3f}, {q_rows['mean_pred_return'].max():.3f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
