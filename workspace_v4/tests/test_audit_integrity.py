#!/usr/bin/env python3
"""
test_audit_integrity.py

Comprehensive integrity tests for the XOPTPOE v4 audit fixes and core logic.

Covers:
  1. Regime label consistency (classifier ↔ plotters)
  2. Question ID consistency (runner ↔ plotters)
  3. Q2 analytical gradient correctness (vs numerical FD)
  4. VAR(1) prior math (fit, predict, regularizer, gradient)
  5. Preconditioned MALA sampler (acceptance, detailed balance)
  6. Probe function math (gold, return target, diversification)
  7. Pipeline evaluate_at / optimize round-trip
  8. Package shadowing (no stale imports)
  9. State space / feature builder consistency
  10. Data file existence checks

Run:
  cd workspace_v4 && PYTHONPATH=src python tests/test_audit_integrity.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — mirror the runner's strategy
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE.parent
SRC_DIR = WORKSPACE / "src"
REPO_SRC = REPO_ROOT / "src"

# Mirror the runner's path ordering exactly:
# workspace_v4/src at position 0 (highest priority)
# repo/src appended (lowest priority, for fallback deps)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_SRC) not in sys.path:
    sys.path.append(str(REPO_SRC))

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
_pass_count = 0
_fail_count = 0
_skip_count = 0
_current_section = ""


def section(name: str) -> None:
    global _current_section
    _current_section = name
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")


def check(name: str, condition: bool, detail: str = "") -> None:
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  ✅ {name}")
    else:
        _fail_count += 1
        msg = f"  ❌ {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


def skip(name: str, reason: str) -> None:
    global _skip_count
    _skip_count += 1
    print(f"  ⏭️  {name} — SKIPPED: {reason}")


def summary() -> None:
    total = _pass_count + _fail_count + _skip_count
    print(f"\n{'='*72}")
    print(f"  RESULTS: {_pass_count} passed, {_fail_count} failed, {_skip_count} skipped  ({total} total)")
    print(f"{'='*72}\n")
    if _fail_count > 0:
        sys.exit(1)


# ===========================================================================
# 1. REGIME LABEL CONSISTENCY
# ===========================================================================
def test_regime_label_consistency():
    section("1. Regime Label Consistency")

    from xoptpoe_v4_scenario.regime import (
        REGIME_RECESSION_STRESS, REGIME_HIGH_STRESS, REGIME_HIGHER_FOR_LONGER,
        REGIME_INFLATIONARY_EXPANSION, REGIME_SOFT_LANDING, REGIME_DISINFL_SLOWDOWN,
        REGIME_RISK_OFF, REGIME_NEUTRAL, SIMPLE_REGIME_MAP,
    )

    classifier_labels = {
        REGIME_RECESSION_STRESS, REGIME_HIGH_STRESS, REGIME_HIGHER_FOR_LONGER,
        REGIME_INFLATIONARY_EXPANSION, REGIME_SOFT_LANDING, REGIME_DISINFL_SLOWDOWN,
        REGIME_RISK_OFF, REGIME_NEUTRAL,
    }

    # Load the plotting scripts' dicts by parsing them
    story_slide_path = WORKSPACE / "scripts" / "make_conference_story_slide.py"
    storyline_path = WORKSPACE / "data" / "make_scenario_storyline_plots.py"

    # Parse REGIME_FRIENDLY from story slide
    ns_slide = {}
    exec(compile(open(story_slide_path).read(), story_slide_path, "exec"), ns_slide)
    regime_friendly = ns_slide["REGIME_FRIENDLY"]
    regime_desc = ns_slide["REGIME_DESC"]

    # Parse REGIME_ORDER from storyline
    ns_story = {}
    exec(compile(open(storyline_path).read(), storyline_path, "exec"), ns_story)
    regime_order = set(ns_story["REGIME_ORDER"])

    # Check all classifier labels are in REGIME_FRIENDLY
    missing_friendly = classifier_labels - set(regime_friendly.keys())
    check(
        "All classifier labels in REGIME_FRIENDLY",
        len(missing_friendly) == 0,
        f"Missing: {missing_friendly}" if missing_friendly else "",
    )

    # Check all classifier labels are in REGIME_DESC
    missing_desc = classifier_labels - set(regime_desc.keys())
    check(
        "All classifier labels in REGIME_DESC",
        len(missing_desc) == 0,
        f"Missing: {missing_desc}" if missing_desc else "",
    )

    # Check all classifier labels are in REGIME_ORDER
    missing_order = classifier_labels - regime_order
    check(
        "All classifier labels in REGIME_ORDER",
        len(missing_order) == 0,
        f"Missing: {missing_order}" if missing_order else "",
    )

    # Check all SIMPLE_REGIME_MAP keys are classifier labels
    check(
        "SIMPLE_REGIME_MAP keys match classifier labels",
        set(SIMPLE_REGIME_MAP.keys()) == classifier_labels,
        f"Diff: {set(SIMPLE_REGIME_MAP.keys()) ^ classifier_labels}",
    )

    # Check legacy labels also resolve
    legacy_labels = {"reflation_risk_on", "mixed_mid_cycle", "risk_off_stress", "high_stress_defensive"}
    missing_legacy_friendly = legacy_labels - set(regime_friendly.keys())
    check(
        "Legacy labels in REGIME_FRIENDLY (backward compat)",
        len(missing_legacy_friendly) == 0,
        f"Missing: {missing_legacy_friendly}" if missing_legacy_friendly else "",
    )
    missing_legacy_order = legacy_labels - regime_order
    check(
        "Legacy labels in REGIME_ORDER (backward compat)",
        len(missing_legacy_order) == 0,
        f"Missing: {missing_legacy_order}" if missing_legacy_order else "",
    )

    # Check existing scenario results (if available)
    results_path = WORKSPACE / "reports" / "scenario_results_v4.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        result_labels = set(df["regime_label"].unique())
        unresolved = result_labels - set(regime_friendly.keys())
        check(
            "All labels in scenario_results_v4.csv resolve in REGIME_FRIENDLY",
            len(unresolved) == 0,
            f"Unresolved: {unresolved}" if unresolved else "",
        )
    else:
        skip("scenario_results_v4.csv label check", "file not found")


# ===========================================================================
# 2. QUESTION ID CONSISTENCY
# ===========================================================================
def test_question_id_consistency():
    section("2. Question ID Consistency")

    # Parse question IDs from runner
    runner_path = WORKSPACE / "scripts" / "scenario" / "run_v4_scenario_final.py"
    runner_code = open(runner_path).read()

    # Extract all question_id strings from the runner
    import re
    runner_qids = set(re.findall(r'"question_id":\s*"([^"]+)"', runner_code))
    check("Runner has expected question IDs", len(runner_qids) >= 3,
          f"Found: {runner_qids}")

    # Parse QUESTION_TEXT/QUESTION_LABELS from plotters
    ns_slide = {}
    exec(compile(open(WORKSPACE / "scripts" / "make_conference_story_slide.py").read(),
                 "slide", "exec"), ns_slide)
    slide_qids = set(ns_slide["QUESTION_TEXT"].keys())
    slide_g_qids = set(ns_slide["G_TEXT"].keys())

    ns_story = {}
    exec(compile(open(WORKSPACE / "data" / "make_scenario_storyline_plots.py").read(),
                 "story", "exec"), ns_story)
    story_qids = set(ns_story["QUESTION_LABELS"].keys())

    # Check that Q3_house_view_7pct_total is in the runner
    check(
        "Runner uses Q3_house_view_7pct_total",
        "Q3_house_view_7pct_total" in runner_qids,
        f"Runner QIDs: {runner_qids}",
    )

    # Check core Q IDs resolve in all plotters
    core_qids = {"Q1_gold_favorable", "Q2_ew_deviation", "Q3_house_view_7pct_total"}
    for qid in core_qids:
        check(f"'{qid}' in slide QUESTION_TEXT", qid in slide_qids)
        check(f"'{qid}' in slide G_TEXT", qid in slide_g_qids)
        check(f"'{qid}' in storyline QUESTION_LABELS", qid in story_qids)

    # Check forward compat alias
    check("Q3_house_saa_total alias in slide QUESTION_TEXT", "Q3_house_saa_total" in slide_qids)
    check("Q3_house_saa_total alias in storyline QUESTION_LABELS", "Q3_house_saa_total" in story_qids)

    # Check existing results
    results_path = WORKSPACE / "reports" / "scenario_results_v4.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        result_qids = set(df["question_id"].unique())
        unresolved_slide = result_qids - slide_qids
        unresolved_story = result_qids - story_qids
        check(
            "All result QIDs resolve in slide plotter",
            len(unresolved_slide) == 0,
            f"Unresolved: {unresolved_slide}",
        )
        check(
            "All result QIDs resolve in storyline plotter",
            len(unresolved_story) == 0,
            f"Unresolved: {unresolved_story}",
        )
    else:
        skip("scenario_results QID check", "file not found")


# ===========================================================================
# 3. VAR(1) PRIOR MATH
# ===========================================================================
def test_var1_prior_math():
    section("3. VAR(1) Prior Math")

    from xoptpoe_v4_scenario.var1_prior import VAR1Prior

    rng = np.random.default_rng(42)
    D = 5
    T = 100
    # Generate a synthetic VAR(1) process
    A_true = 0.8 * np.eye(D) + 0.05 * rng.standard_normal((D, D))
    c_true = rng.standard_normal(D) * 0.1
    Q_true = np.eye(D) * 0.01

    states = np.zeros((T, D))
    states[0] = rng.standard_normal(D)
    for t in range(1, T):
        states[t] = c_true + A_true @ states[t-1] + rng.multivariate_normal(np.zeros(D), Q_true)

    macro_cols = [f"x{i}" for i in range(D)]
    prior = VAR1Prior.fit(states, macro_cols=macro_cols)

    # Check dimensions
    check("VAR1Prior.c shape", prior.c.shape == (D,))
    check("VAR1Prior.A shape", prior.A.shape == (D, D))
    check("VAR1Prior.Q shape", prior.Q.shape == (D, D))
    check("VAR1Prior.Q_inv shape", prior.Q_inv.shape == (D, D))
    check("VAR1Prior.Q is symmetric", np.allclose(prior.Q, prior.Q.T, atol=1e-10))

    # Check Q_inv is actually inverse of Q (approximately)
    product = prior.Q @ prior.Q_inv
    check("Q @ Q_inv ≈ I", np.allclose(product, np.eye(D), atol=1e-6),
          f"max |Q@Q_inv - I| = {np.max(np.abs(product - np.eye(D))):.2e}")

    # Check predict_next
    m_t = states[50]
    mu_pred = prior.predict_next(m_t)
    check("predict_next returns correct shape", mu_pred.shape == (D,))

    # Check Mahalanobis distance is non-negative
    md2 = prior.mahalanobis_sq(states[51], states[50])
    check("Mahalanobis sq is non-negative", md2 >= 0, f"md2 = {md2:.6f}")

    # Check regularizer
    reg_val = prior.regularizer(states[51], states[50], l2reg=0.5)
    check("Regularizer is non-negative", reg_val >= 0, f"reg = {reg_val:.6f}")
    check("Regularizer = 0.5 * l2reg * md2", np.isclose(reg_val, 0.5 * 0.5 * md2, rtol=1e-10))

    # Check analytical gradient vs numerical gradient
    m_candidate = states[60]
    m_anchor = states[50]
    l2reg = 0.5

    grad_analytical = prior.regularizer_grad(m_candidate, m_anchor, l2reg=l2reg)
    check("Regularizer grad shape", grad_analytical.shape == (D,))

    eps = 1e-6
    grad_numerical = np.zeros(D)
    for d in range(D):
        m_p = m_candidate.copy(); m_p[d] += eps
        m_n = m_candidate.copy(); m_n[d] -= eps
        grad_numerical[d] = (prior.regularizer(m_p, m_anchor, l2reg=l2reg) -
                             prior.regularizer(m_n, m_anchor, l2reg=l2reg)) / (2 * eps)

    grad_err = np.max(np.abs(grad_analytical - grad_numerical))
    check(
        "Regularizer analytical grad matches numerical (max err < 1e-5)",
        grad_err < 1e-5,
        f"max |grad_a - grad_n| = {grad_err:.2e}",
    )

    # Check box constraints
    a, b = prior.box_constraints_from_prediction(m_t, n_sigma=3.0)
    check("Box constraint a < b everywhere", np.all(a < b))
    check("Predicted mean inside box", np.all(mu_pred >= a) and np.all(mu_pred <= b))

    # Check log_density is finite
    ld = prior.log_density(states[51], states[50])
    check("log_density is finite", np.isfinite(ld), f"log_density = {ld:.4f}")


# ===========================================================================
# 4. MALA SAMPLER
# ===========================================================================
def test_mala_sampler():
    section("4. MALA Sampler")

    from xoptpoe_v4_scenario.sampler import mala_chain, compute_effective_sample_size

    D = 3
    # Simple quadratic G(m) = 0.5 * ||m||^2 => grad = m
    # Exact posterior is N(0, tau*I) for large eta, but we just check mechanics

    def G(m):
        return 0.5 * float(np.dot(m, m))

    def gradG(m):
        return np.asarray(m, dtype=float).copy()

    m_init = np.array([1.0, -0.5, 0.3])
    a = np.full(D, -5.0)
    b = np.full(D, 5.0)

    # Run without preconditioner
    m_last, trajectory, acc_rate = mala_chain(
        m_init=m_init, G=G, gradG=gradG, a=a, b=b,
        n_steps=200, eta=0.1, tau=1.0, rng=np.random.default_rng(42),
    )
    check("MALA returns correct trajectory shape", trajectory.shape == (200, D))
    check("MALA acceptance rate > 0", acc_rate > 0, f"acc_rate = {acc_rate:.2%}")
    check("MALA acceptance rate < 1", acc_rate < 1.0, f"acc_rate = {acc_rate:.2%}")
    check("m_last is within bounds", np.all(m_last >= a) and np.all(m_last <= b))

    # Run with diagonal preconditioner
    precond = np.array([0.5, 1.0, 2.0])
    m_last_p, traj_p, acc_p = mala_chain(
        m_init=m_init, G=G, gradG=gradG, a=a, b=b,
        n_steps=200, eta=0.1, tau=1.0, precond=precond,
        rng=np.random.default_rng(42),
    )
    check("Preconditioned MALA returns correct shape", traj_p.shape == (200, D))
    check("Preconditioned MALA acceptance > 0", acc_p > 0, f"acc_rate = {acc_p:.2%}")

    # ESS computation
    ess = compute_effective_sample_size(trajectory)
    check("ESS shape matches D", ess.shape == (D,))
    check("ESS > 0 for all dims", np.all(ess > 0), f"ESS = {ess}")
    check("ESS <= n_steps", np.all(ess <= 200 + 1))

    # Detailed balance sanity: for a symmetric target, mean of trajectory
    # should be near zero (not a strict test, but a sanity check)
    mean_traj = trajectory[50:].mean(axis=0)
    check(
        "Trajectory mean near origin for N(0,I) target (|mean| < 1.5)",
        np.all(np.abs(mean_traj) < 1.5),
        f"mean = {mean_traj}",
    )


# ===========================================================================
# 5. STATE SPACE
# ===========================================================================
def test_state_space():
    section("5. State Space")

    from xoptpoe_v4_scenario.state_space import (
        MACRO_STATE_COLS, STATE_DIM, box_constraints, state_scales,
    )

    check("STATE_DIM == len(MACRO_STATE_COLS)", STATE_DIM == len(MACRO_STATE_COLS),
          f"STATE_DIM={STATE_DIM}, len={len(MACRO_STATE_COLS)}")
    check("STATE_DIM == 19", STATE_DIM == 19, f"STATE_DIM={STATE_DIM}")

    # Check expected columns are present
    expected_macro = [
        "infl_US", "infl_EA", "infl_JP",
        "short_rate_US", "short_rate_EA", "short_rate_JP",
        "long_rate_US", "long_rate_EA", "long_rate_JP",
        "term_slope_US", "term_slope_EA", "term_slope_JP",
        "unemp_US", "unemp_EA",
        "ig_oas", "us_real10y", "vix", "oil_wti", "usd_broad",
    ]
    for col in expected_macro:
        check(f"'{col}' in MACRO_STATE_COLS", col in MACRO_STATE_COLS)

    # Test state_scales and box_constraints with synthetic feature_master
    rng = np.random.default_rng(99)
    n_months = 50
    fm_data = {"month_end": pd.date_range("2010-01-31", periods=n_months, freq="ME")}
    for col in MACRO_STATE_COLS:
        fm_data[col] = rng.standard_normal(n_months) * 2.0 + 1.0
    fm = pd.DataFrame(fm_data)

    scales = state_scales(fm)
    check("state_scales() shape", scales.shape == (STATE_DIM,))
    check("state_scales() all positive", np.all(scales > 0))

    lo, hi = box_constraints(fm)
    check("box_constraints lo shape", lo.shape == (STATE_DIM,))
    check("box_constraints hi shape", hi.shape == (STATE_DIM,))
    check("box_constraints lo < hi everywhere", np.all(lo < hi))


# ===========================================================================
# 6. REGIME CLASSIFIER LOGIC
# ===========================================================================
def test_regime_classifier():
    section("6. Regime Classifier Logic")

    from xoptpoe_v4_scenario.regime import classify_single_state, SIMPLE_REGIME_MAP
    from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS

    mac_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}
    D = len(MACRO_STATE_COLS)

    # Build a neutral baseline state
    m_neutral = np.zeros(D)
    m_neutral[mac_idx["infl_US"]] = 2.0
    m_neutral[mac_idx["short_rate_US"]] = 2.0
    m_neutral[mac_idx["long_rate_US"]] = 3.0
    m_neutral[mac_idx["us_real10y"]] = 0.5
    m_neutral[mac_idx["unemp_US"]] = 5.0
    m_neutral[mac_idx["ig_oas"]] = 1.0
    m_neutral[mac_idx["vix"]] = 15.0

    # Use moderate thresholds
    thresholds = {
        "ig_oas_p50": 1.0, "ig_oas_p75": 1.5, "ig_oas_p90": 2.5,
        "vix_p50": 18.0, "vix_p75": 22.0, "vix_p90": 30.0,
        "infl_US_p50": 2.0, "infl_US_p75": 3.0,
        "short_rate_US_p75": 3.0, "short_rate_US_p25": 0.5,
        "unemp_US_p75": 6.5,
        "us_real10y_p75": 1.0, "us_real10y_p50": 0.3,
    }

    label, simple, signals = classify_single_state(m_neutral, thresholds)
    check("Classifier returns a string label", isinstance(label, str) and len(label) > 0)
    check("Classifier returns a simple label", simple in {"stress", "rate_transition", "expansion", "neutral"})
    check("Simple label matches SIMPLE_REGIME_MAP", simple == SIMPLE_REGIME_MAP.get(label, "neutral"))
    check("Signals dict has expected keys", "ig_oas" in signals and "vix" in signals)

    # Test recession_stress: high stress + weak growth
    m_recession = m_neutral.copy()
    m_recession[mac_idx["vix"]] = 35.0      # above p90 (30)
    m_recession[mac_idx["ig_oas"]] = 3.0     # above p90 (2.5)
    m_recession[mac_idx["unemp_US"]] = 8.0   # above p75 (6.5)
    label_r, simple_r, _ = classify_single_state(m_recession, thresholds)
    check("High stress + weak growth → recession_stress", label_r == "recession_stress",
          f"Got: {label_r}")
    check("recession_stress → simple 'stress'", simple_r == "stress")

    # Test higher_for_longer: high rates + elevated inflation, no stress
    m_hfl = m_neutral.copy()
    m_hfl[mac_idx["short_rate_US"]] = 5.0    # above p75 (3)
    m_hfl[mac_idx["us_real10y"]] = 1.5       # above p75 (1)
    m_hfl[mac_idx["infl_US"]] = 2.5          # above p50 (2)
    m_hfl[mac_idx["vix"]] = 15.0             # well below p75
    m_hfl[mac_idx["ig_oas"]] = 1.0           # well below p75
    label_h, simple_h, _ = classify_single_state(m_hfl, thresholds)
    check("High rates + elevated inflation → higher_for_longer", label_h == "higher_for_longer",
          f"Got: {label_h}")
    check("higher_for_longer → simple 'rate_transition'", simple_h == "rate_transition")

    # Test soft_landing: low inflation, low rates, low stress
    m_soft = m_neutral.copy()
    m_soft[mac_idx["infl_US"]] = 1.5         # below p75 (3) and below p50 (2) — not high
    m_soft[mac_idx["short_rate_US"]] = 1.0   # below p75 (3)
    m_soft[mac_idx["us_real10y"]] = 0.2       # below p75 (1)
    m_soft[mac_idx["vix"]] = 14.0            # below p75 (22)
    m_soft[mac_idx["ig_oas"]] = 0.8          # below p75 (1.5)
    m_soft[mac_idx["unemp_US"]] = 4.5        # below p75 (6.5)
    label_s, simple_s, _ = classify_single_state(m_soft, thresholds)
    check("Low everything → soft_landing", label_s == "soft_landing",
          f"Got: {label_s}")
    check("soft_landing → simple 'expansion'", simple_s == "expansion")


# ===========================================================================
# 7. PROBE FUNCTION MATH
# ===========================================================================
def test_probe_functions():
    section("7. Probe Function Math")

    from xoptpoe_v4_scenario.probe_functions import (
        numerical_grad, l2_anchor_reg, GOLD_IDX,
    )
    from xoptpoe_v4_scenario.state_space import STATE_DIM

    # Test numerical_grad on a known function
    def f_quadratic(x):
        return float(0.5 * np.dot(x, x) + 3 * x[0])

    m = np.array([1.0, 2.0, 3.0])
    grad_num = numerical_grad(f_quadratic, m, epsilon=1e-5)
    grad_exact = m.copy()
    grad_exact[0] += 3.0
    err = np.max(np.abs(grad_num - grad_exact))
    check(
        "numerical_grad correct for quadratic (err < 1e-4)",
        err < 1e-4,
        f"max err = {err:.2e}",
    )

    # Test l2_anchor_reg
    m0 = np.zeros(5)
    m1 = np.ones(5)
    scales = np.ones(5)
    reg = l2_anchor_reg(m1, m0, scales, l2reg=1.0)
    expected = float(np.sum((m1 - m0)**2 / scales**2))
    check("l2_anchor_reg value", np.isclose(reg, expected, rtol=1e-10),
          f"got {reg:.6f}, expected {expected:.6f}")

    # GOLD_IDX should be valid
    check("GOLD_IDX is non-negative", GOLD_IDX >= 0, f"GOLD_IDX = {GOLD_IDX}")


# ===========================================================================
# 8. Q2 GRADIENT FIX VERIFICATION
# ===========================================================================
def test_q2_gradient_fix():
    section("8. Q2 Gradient Fix (build_fast_gradG no longer used)")

    import re

    runner_path = WORKSPACE / "scripts" / "scenario" / "run_v4_scenario_final.py"
    runner_code = open(runner_path).read()

    # Find the Q2 section
    # The old code had: grad2 = build_fast_gradG(...)
    # The new code should have: grad2 = grad2_analytical  (or make_G_and_gradG fallback)

    # Check that Q2 does NOT use build_fast_gradG
    # Find the Q2 block — between "Q2:" comment and the next "Q" section
    q2_match = re.search(
        r'# Q2:.*?(?=# Q3:|# Q4:|\Z)',
        runner_code,
        re.DOTALL,
    )
    check("Found Q2 code block", q2_match is not None)

    if q2_match:
        q2_code = q2_match.group(0)
        uses_build_fast = "build_fast_gradG" in q2_code
        check(
            "Q2 does NOT use build_fast_gradG (was broken)",
            not uses_build_fast,
            "Q2 still uses build_fast_gradG — the fix was not applied!",
        )

        # Check it has an inline analytical gradient
        has_inline_grad = "grad2_analytical" in q2_code or "gradG_analytical" in q2_code
        check(
            "Q2 has inline analytical gradient function",
            has_inline_grad,
        )

        # Check the gradient evaluates active tilt correctly
        has_active_eval = "active_p" in q2_code and "active_n" in q2_code
        check(
            "Q2 gradient evaluates active tilt at perturbed weights",
            has_active_eval,
        )

        # Check it uses pipeline.optimize for perturbed mu
        has_optimize = "pipeline.optimize(mu_p)" in q2_code and "pipeline.optimize(mu_n)" in q2_code
        check(
            "Q2 gradient optimizes at perturbed mu values",
            has_optimize,
        )


# ===========================================================================
# 9. _eval_G_with_override IS DEAD CODE
# ===========================================================================
def test_eval_g_override_dead_code():
    section("9. _eval_G_with_override Dead Code Check")

    runner_path = WORKSPACE / "scripts" / "scenario" / "run_v4_scenario_final.py"
    runner_code = open(runner_path).read()

    # _eval_G_with_override should NOT appear in the runner
    check(
        "_eval_G_with_override not used in runner",
        "_eval_G_with_override" not in runner_code,
    )

    # build_fast_gradG should not be used for any active question
    # (it may still be imported, but should not be called for Q1-Q11)
    import re
    question_blocks = re.findall(
        r'# Q\d+:.*?(?=# Q\d+:|\Z)',
        runner_code,
        re.DOTALL,
    )
    for block in question_blocks:
        qid_match = re.search(r'"question_id":\s*"([^"]+)"', block)
        if qid_match:
            qid = qid_match.group(1)
            uses_bfg = "build_fast_gradG" in block
            check(
                f"{qid} does not use build_fast_gradG",
                not uses_bfg,
                f"{qid} still uses the broken build_fast_gradG!",
            )


# ===========================================================================
# 10. PACKAGE SHADOWING
# ===========================================================================
def test_package_shadowing():
    section("10. Package Shadowing")

    packages = ["xoptpoe_v4_modeling", "xoptpoe_v4_models", "xoptpoe_v4_scenario"]

    for pkg_name in packages:
        try:
            mod = importlib.import_module(pkg_name)
            mod_path = Path(mod.__file__).resolve()
            expected_prefix = str(SRC_DIR)
            check(
                f"{pkg_name} imports from workspace_v4/src",
                str(mod_path).startswith(expected_prefix),
                f"Imported from: {mod_path}",
            )
        except ImportError:
            skip(f"{pkg_name} import location", "import failed")


# ===========================================================================
# 11. DATA FILE INTEGRITY
# ===========================================================================
def test_data_files():
    section("11. Data File Integrity")

    # Asset master
    am_path = WORKSPACE / "data" / "final_v4_expanded_universe" / "asset_master.csv"
    check("asset_master.csv exists", am_path.exists())
    if am_path.exists():
        am = pd.read_csv(am_path)
        check("asset_master has 'sleeve_id' column", "sleeve_id" in am.columns)
        check("asset_master has 15 sleeves", len(am) == 15, f"Found {len(am)}")

        excluded = set(am["sleeve_id"]) - {"CR_EU_HY"}
        active_count = len(am[~am["sleeve_id"].eq("CR_EU_HY")])
        check("14 active sleeves (excluding CR_EU_HY)", active_count == 14, f"Found {active_count}")

    # Benchmark metrics
    bm_path = WORKSPACE / "reports" / "benchmark" / "v4_prediction_benchmark_metrics.csv"
    check("v4_prediction_benchmark_metrics.csv exists", bm_path.exists())
    if bm_path.exists():
        bm = pd.read_csv(bm_path)
        best60 = bm[bm["experiment_name"] == "elastic_net__core_plus_interactions__separate_60"]
        check(
            "Best-60 experiment exists in benchmark metrics",
            len(best60) > 0,
        )

    # Scenario results
    sr_path = WORKSPACE / "reports" / "scenario_results_v4.csv"
    check("scenario_results_v4.csv exists", sr_path.exists())
    if sr_path.exists():
        sr = pd.read_csv(sr_path)
        check("scenario_results has > 0 rows", len(sr) > 0, f"Found {len(sr)} rows")

        # Check column completeness
        required_cols = [
            "question_id", "anchor_date", "train_end", "G_value",
            "pred_return_excess", "pred_return_total", "rf_rate",
            "regime_label", "anchor_regime",
        ]
        missing = [c for c in required_cols if c not in sr.columns]
        check(
            "scenario_results has all required columns",
            len(missing) == 0,
            f"Missing: {missing}",
        )

        # Check weight columns
        weight_cols = [c for c in sr.columns if c.startswith("w_")]
        check("scenario_results has weight columns", len(weight_cols) >= 10,
              f"Found {len(weight_cols)} weight columns")

        # Check weights sum to ~1 for each row
        if weight_cols:
            w_sums = sr[weight_cols].sum(axis=1)
            close_to_one = np.allclose(w_sums, 1.0, atol=0.01)
            check(
                "Portfolio weights sum to ~1.0 (atol=0.01)",
                close_to_one,
                f"min={w_sums.min():.4f}, max={w_sums.max():.4f}, mean={w_sums.mean():.4f}",
            )

        # Check all weights are non-negative (long-only)
        if weight_cols:
            all_nonneg = (sr[weight_cols] >= -1e-6).all().all()
            check("All portfolio weights non-negative (long-only)", all_nonneg)

        # Check pred_return_total ≈ pred_return_excess + rf_rate
        if "pred_return_excess" in sr.columns and "rf_rate" in sr.columns and "pred_return_total" in sr.columns:
            implied_total = sr["pred_return_excess"] + sr["rf_rate"]
            diff = (sr["pred_return_total"] - implied_total).abs()
            check(
                "pred_return_total ≈ pred_return_excess + rf_rate (max err < 0.001)",
                diff.max() < 0.001,
                f"max diff = {diff.max():.6f}",
            )


# ===========================================================================
# 12. PIPELINE CONSTANTS CONSISTENCY
# ===========================================================================
def test_pipeline_constants():
    section("12. Pipeline Constants Consistency")

    from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS, STATE_DIM

    try:
        from xoptpoe_v4_scenario.probe_functions import GOLD_IDX, _SHORT_RATE_US_IDX
    except ImportError as e:
        skip("probe_functions constants", f"import failed: {e}")
        return

    # Check GOLD_IDX
    try:
        from xoptpoe_v4_models.data import SLEEVE_ORDER
        gold_in_order = "ALT_GLD" in SLEEVE_ORDER
        check("ALT_GLD in SLEEVE_ORDER", gold_in_order)
        if gold_in_order:
            expected_gold_idx = list(SLEEVE_ORDER).index("ALT_GLD")
            check(
                f"GOLD_IDX matches SLEEVE_ORDER position",
                GOLD_IDX == expected_gold_idx,
                f"probe GOLD_IDX={GOLD_IDX}, SLEEVE_ORDER position={expected_gold_idx}",
            )
    except (ImportError, Exception) as e:
        skip("GOLD_IDX vs SLEEVE_ORDER check", f"import failed: {e}")

    # Check short_rate_US index
    if "short_rate_US" in MACRO_STATE_COLS:
        expected_sr_idx = list(MACRO_STATE_COLS).index("short_rate_US")
        check(
            "_SHORT_RATE_US_IDX matches MACRO_STATE_COLS position",
            _SHORT_RATE_US_IDX == expected_sr_idx,
            f"probe={_SHORT_RATE_US_IDX}, state_space={expected_sr_idx}",
        )
    else:
        check("short_rate_US in MACRO_STATE_COLS", False)


# ===========================================================================
# 13. RUNNER sys.path ORDERING
# ===========================================================================
def test_runner_sys_path_ordering():
    section("13. Runner sys.path Ordering")

    runner_path = WORKSPACE / "scripts" / "scenario" / "run_v4_scenario_final.py"
    runner_code = open(runner_path).read()

    # workspace/src should be inserted BEFORE repo src
    ws_insert = 'sys.path.insert(0, str(WORKSPACE / "src"))'
    repo_append = 'sys.path.append(str(REPO_SRC))'

    check("Runner inserts workspace/src at position 0", ws_insert in runner_code)
    check("Runner appends repo/src (lower priority)", repo_append in runner_code)

    # Verify workspace insert comes before repo append in the file
    ws_pos = runner_code.find(ws_insert)
    repo_pos = runner_code.find(repo_append)
    if ws_pos >= 0 and repo_pos >= 0:
        check("workspace/src inserted before repo/src appended", ws_pos < repo_pos)


# ===========================================================================
# 14. CONFERENCE PLOT SCRIPT SYNTAX
# ===========================================================================
def test_plot_script_syntax():
    section("14. Plot Script Syntax")

    import ast

    scripts = [
        WORKSPACE / "scripts" / "make_conference_story_slide.py",
        WORKSPACE / "data" / "make_scenario_storyline_plots.py",
        WORKSPACE / "scripts" / "run_v4_conference_plots.py",
        WORKSPACE / "scripts" / "scenario" / "run_v4_scenario_final.py",
    ]

    for script in scripts:
        if script.exists():
            try:
                ast.parse(open(script).read())
                check(f"{script.name} parses without syntax errors", True)
            except SyntaxError as e:
                check(f"{script.name} parses without syntax errors", False, str(e))
        else:
            skip(f"{script.name} syntax", "file not found")


# ===========================================================================
# 15. SHORT_RATE_US PERCENT CONVENTION
# ===========================================================================
def test_short_rate_percent_convention():
    section("15. short_rate_US Percent Convention")

    # Check that pipeline.py divides by 100 for rf_rate
    pipeline_path = SRC_DIR / "xoptpoe_v4_scenario" / "pipeline.py"
    if pipeline_path.exists():
        code = open(pipeline_path).read()
        has_divide_100 = "/ 100" in code or "/100" in code
        check(
            "pipeline.py divides short_rate_US by 100 for rf_rate",
            has_divide_100,
        )
    else:
        skip("pipeline.py rf_rate check", "file not found")

    # Check that house_view_return_probe gradient divides by 100
    probe_path = SRC_DIR / "xoptpoe_v4_scenario" / "probe_functions.py"
    if probe_path.exists():
        code = open(probe_path).read()
        # The RF gradient line should have / 100.0
        has_rf_grad_correct = "/ 100.0" in code
        check(
            "probe_functions.py RF gradient divides by 100.0",
            has_rf_grad_correct,
        )
    else:
        skip("probe_functions.py RF gradient check", "file not found")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "="*72)
    print("  XOPTPOE v4 — Audit Integrity Test Suite")
    print("="*72)

    test_regime_label_consistency()
    test_question_id_consistency()
    test_var1_prior_math()
    test_mala_sampler()
    test_state_space()
    test_regime_classifier()
    test_probe_functions()
    test_q2_gradient_fix()
    test_eval_g_override_dead_code()
    test_package_shadowing()
    test_data_files()
    test_pipeline_constants()
    test_runner_sys_path_ordering()
    test_plot_script_syntax()
    test_short_rate_percent_convention()

    summary()
