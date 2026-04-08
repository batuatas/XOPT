"""
probe_functions.py  —  v4

What is new in v4
-----------------
Kept the existing probes, and added cleaner story-driven probes:

1) less_gold_probe()
   - "What macro state makes the benchmark want less gold?"

2) more_diversification_probe()
   - wrapper for max_diversification_probe()
   - "What macro state spreads the portfolio out the most?"

3) classic_sixty_forty_probe()
   - cleaner 60/40 probe than the old sixty_forty_probe()
   - explicitly discourages alternatives from filling residual weight

4) excess_return_target_probe()
   - target excess return directly, e.g. 5%

5) max_total_equity_probe()
   - maximize total equity share, not just US equity

Notes
-----
- Existing probes are preserved for compatibility.
- house_view_return_probe() still targets TOTAL return, not excess return.
- short_rate_US is in percent units, so rf = short_rate_US / 100.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS, STATE_DIM

SLEEVES_14 = [
    "EQ_US", "EQ_EZ", "EQ_JP", "EQ_CN", "EQ_EM",
    "FI_UST", "FI_EU_GOVT",
    "CR_US_IG", "CR_EU_IG", "CR_US_HY",
    "RE_US", "LISTED_RE", "LISTED_INFRA",
    "ALT_GLD",
]
_SLEEVE_IDX = {s: i for i, s in enumerate(SLEEVES_14)}
GOLD_IDX: int = _SLEEVE_IDX["ALT_GLD"]
EQ_US_IDX: int = _SLEEVE_IDX["EQ_US"]
FI_UST_IDX: int = _SLEEVE_IDX["FI_UST"]
RE_US_IDX: int = _SLEEVE_IDX["RE_US"]

_EQ_SLEEVES = ["EQ_US", "EQ_EZ", "EQ_JP", "EQ_CN", "EQ_EM"]
_FICR_SLEEVES = ["FI_UST", "FI_EU_GOVT", "CR_US_IG", "CR_EU_IG", "CR_US_HY"]
_ALT_SLEEVES = ["ALT_GLD", "RE_US", "LISTED_RE", "LISTED_INFRA"]

_EQ_IDX = [_SLEEVE_IDX[s] for s in _EQ_SLEEVES]
_FICR_IDX = [_SLEEVE_IDX[s] for s in _FICR_SLEEVES]
_ALT_IDX = [_SLEEVE_IDX[s] for s in _ALT_SLEEVES]

EPS: float = 1e-10
_FD_EPS_W: float = 1e-4

# short_rate_US index in MACRO_STATE_COLS — for rf add-back in probes
try:
    _SHORT_RATE_US_IDX: int = list(MACRO_STATE_COLS).index("short_rate_US")
except ValueError:
    _SHORT_RATE_US_IDX = -1


def _rf_from_m(m: np.ndarray) -> float:
    """
    Extract rf rate (decimal) from macro state vector.
    short_rate_US is stored in PERCENT units -> divide by 100.
    Returns 0.0 if short_rate_US is not in MACRO_STATE_COLS.
    """
    if _SHORT_RATE_US_IDX >= 0:
        return float(m[_SHORT_RATE_US_IDX]) / 100.0
    return 0.0


# ---------------------------------------------------------------------------
# Gradient computation — full central differences (fallback)
# ---------------------------------------------------------------------------

def numerical_grad(
    G: Callable[[np.ndarray], float],
    m: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    grad = np.zeros_like(m)
    for i in range(len(m)):
        m_plus = m.copy()
        m_plus[i] += epsilon
        m_minus = m.copy()
        m_minus[i] -= epsilon
        grad[i] = (G(m_plus) - G(m_minus)) / (2.0 * epsilon)
    return grad


def make_G_and_gradG(
    G: Callable[[np.ndarray], float],
    epsilon: float = 1e-4,
) -> tuple[Callable, Callable]:
    def gradG(m: np.ndarray) -> np.ndarray:
        return numerical_grad(G, m, epsilon=epsilon)
    return G, gradG


# ---------------------------------------------------------------------------
# Fast gradient builder (analytical EN Jacobian path)
# ---------------------------------------------------------------------------

def build_fast_gradG(
    pipeline,
    G: Callable[[np.ndarray], float],
    m0: np.ndarray,
    prior,
    var1_l2reg: float,
    use_analytical: bool = True,
    fd_eps: float = 1e-4,
) -> Callable[[np.ndarray], np.ndarray]:
    if use_analytical and pipeline.has_analytical_grad():

        def gradG_fast(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            ev = pipeline.evaluate_at(m)
            mu_hat = ev["mu_hat"]

            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += fd_eps
                mu_n = mu_hat.copy()
                mu_n[k] -= fd_eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                G_p = _eval_G_with_override(G, pipeline, m, mu_p, w_p)
                G_n = _eval_G_with_override(G, pipeline, m, mu_n, w_n)
                dG_dmu[k] = (G_p - G_n) / (2.0 * fd_eps)

            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return gradG_fast
    else:
        def gradG_fd(m: np.ndarray) -> np.ndarray:
            return numerical_grad(G, m, epsilon=fd_eps)
        return gradG_fd


def _eval_G_with_override(G_full, pipeline, m, mu_override, w_override):
    return float(G_full(m))


# ---------------------------------------------------------------------------
# Regularizer helpers
# ---------------------------------------------------------------------------

def l2_anchor_reg(m, m0, scales, l2reg):
    diff = m - m0
    return float(l2reg * np.sum((diff / np.maximum(scales, 1e-10)) ** 2))


def var1_reg(m, m_anchor, prior, l2reg):
    return prior.regularizer(m, m_anchor, l2reg=l2reg)


# ---------------------------------------------------------------------------
# A. Benchmark return probe
# ---------------------------------------------------------------------------

def benchmark_return_probe(
    pipeline,
    m0: np.ndarray,
    realized_returns_60m: np.ndarray,
    b_target: float,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    ret = np.asarray(realized_returns_60m, dtype=float)

    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        port_ret = float(w @ ret)
        task = (port_ret - b_target) ** 2
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():
        gradG = build_fast_gradG(pipeline, G, m0, prior, var1_l2reg)
    else:
        _, gradG = make_G_and_gradG(G)
    return G, gradG


# ---------------------------------------------------------------------------
# B. Gold weight probes
# ---------------------------------------------------------------------------

def gold_weight_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
    sign: float = -1.0,
) -> tuple[Callable, Callable]:
    """
    sign = -1.0  -> maximize gold
    sign = +1.0  -> minimize gold
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        task = float(sign * w[GOLD_IDX])
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            dG_dmu = np.zeros(len(mu_hat))
            eps = _FD_EPS_W
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                dG_dmu[k] = sign * (w_p[GOLD_IDX] - w_n[GOLD_IDX]) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


def less_gold_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Story question:
      What macro state makes the benchmark want less gold?
    """
    return gold_weight_probe(
        pipeline=pipeline,
        m0=m0,
        scales=scales,
        l2reg=l2reg,
        prior=prior,
        var1_l2reg=var1_l2reg,
        sign=+1.0,
    )


def gold_transition_probe(
    pipeline,
    m_low_gold: np.ndarray,
    m_high_gold: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        task = float(-w[GOLD_IDX])
        if prior is not None:
            m_mid = 0.5 * (m_low_gold + m_high_gold)
            reg = var1_reg(m, m_mid, prior, l2reg=var1_l2reg)
        else:
            reg = (
                l2_anchor_reg(m, m_low_gold, scales, l2reg / 2.0)
                + l2_anchor_reg(m, m_high_gold, scales, l2reg / 2.0)
            )
        return task + reg

    _, gradG = make_G_and_gradG(G)
    return G, gradG


# ---------------------------------------------------------------------------
# C. Diversification probes
# ---------------------------------------------------------------------------

def diversification_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Original diversification probe:
    maximize entropy.
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        entropy = float(-np.sum(w * np.log(w + EPS)))
        task = -entropy
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                ent_p = float(-np.sum(w_p * np.log(w_p + EPS)))
                ent_n = float(-np.sum(w_n * np.log(w_n + EPS)))
                dG_dmu[k] = -(ent_p - ent_n) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


def more_diversification_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Cleaner story wrapper:
      What macro state spreads the benchmark out the most?
    """
    return max_diversification_probe(
        pipeline=pipeline,
        m0=m0,
        scales=scales,
        l2reg=l2reg,
        prior=prior,
        var1_l2reg=var1_l2reg,
    )


def equal_weight_excess_probe(
    pipeline,
    m0: np.ndarray,
    realized_returns_60m: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    ret = np.asarray(realized_returns_60m, dtype=float)
    n = len(ret)
    ew = np.ones(n, dtype=float) / n

    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        task = -(float(w @ ret) - float(ew @ ret))
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():
        gradG = build_fast_gradG(pipeline, G, m0, prior, var1_l2reg)
    else:
        _, gradG = make_G_and_gradG(G)
    return G, gradG


# ---------------------------------------------------------------------------
# D. Return target probes
# ---------------------------------------------------------------------------

def house_view_return_probe(
    pipeline,
    m0: np.ndarray,
    house_view_total: float,       # TOTAL return target
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Target TOTAL return = excess + rf.
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        ev = pipeline.evaluate_at(m)
        pred_total = ev["pred_return_total"]
        task = 5000.0 * (pred_total - house_view_total) ** 2
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            ev = pipeline.evaluate_at(m)
            mu_hat = ev["mu_hat"]
            pred_total = ev["pred_return_total"]

            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                pr_p = float(mu_p @ w_p) + ev["rf_rate"]
                pr_n = float(mu_n @ w_n) + ev["rf_rate"]
                dG_dmu[k] = 5000.0 * (pr_p - house_view_total) ** 2 - 5000.0 * (pr_n - house_view_total) ** 2
                dG_dmu[k] /= (2.0 * eps)

            grad_task = dG_dmu @ J_mu

            # rf depends on m through short_rate_US
            if _SHORT_RATE_US_IDX >= 0:
                grad_rf = np.zeros(STATE_DIM)
                grad_rf[_SHORT_RATE_US_IDX] = 2.0 * (pred_total - house_view_total) / 100.0
                grad_task = grad_task + grad_rf

            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


def excess_return_target_probe(
    pipeline,
    m0: np.ndarray,
    target_excess: float,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Directly target excess return, e.g. 5%.
    Story question:
      What macro state makes the benchmark predict about X% excess return?
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        ev = pipeline.evaluate_at(m)
        task = (ev["pred_return_excess"] - target_excess) ** 2
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            ev = pipeline.evaluate_at(m)
            mu_hat = ev["mu_hat"]

            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)

                pr_p = float(mu_p @ w_p)
                pr_n = float(mu_n @ w_n)
                dG_dmu[k] = ((pr_p - target_excess) ** 2 - (pr_n - target_excess) ** 2) / (2.0 * eps)

            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


# ---------------------------------------------------------------------------
# E. Stress regime probe
# ---------------------------------------------------------------------------

def stress_regime_probe(
    pipeline,
    m0: np.ndarray,
    realized_returns_60m: np.ndarray,
    scales: np.ndarray,
    nfci_proxy_idx: list[int],
    nfci_high_threshold: float,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    ret = np.asarray(realized_returns_60m, dtype=float)

    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        port_ret = float(w @ ret)
        task = -port_ret
        stress_val = float(np.mean([m[i] for i in nfci_proxy_idx]))
        stress_penalty = 0.5 * max(0.0, nfci_high_threshold - stress_val) ** 2
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + stress_penalty + reg

    _, gradG = make_G_and_gradG(G)
    return G, gradG


# ---------------------------------------------------------------------------
# F. Max diversification
# ---------------------------------------------------------------------------

def max_diversification_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Maximize portfolio entropy.
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        entropy = float(-np.sum(w[w > EPS] * np.log(w[w > EPS])))
        task = -entropy
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                ent_p = float(-np.sum(w_p[w_p > EPS] * np.log(w_p[w_p > EPS])))
                ent_n = float(-np.sum(w_n[w_n > EPS] * np.log(w_n[w_n > EPS])))
                dG_dmu[k] = -(ent_p - ent_n) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


# ---------------------------------------------------------------------------
# G. Max risk
# ---------------------------------------------------------------------------

def max_risk_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        ev = pipeline.evaluate_at(m)
        task = -ev["risk"]
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                risk_p = float(np.sqrt(max(w_p @ pipeline.sigma @ w_p, 0.0)))
                risk_n = float(np.sqrt(max(w_n @ pipeline.sigma @ w_n, 0.0)))
                dG_dmu[k] = -(risk_p - risk_n) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


# ---------------------------------------------------------------------------
# H. 60/40 probes
# ---------------------------------------------------------------------------

def sixty_forty_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    eq_target: float = 0.60,
    ficr_target: float = 0.40,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Original 60/40 probe:
    target total equity and total FI+credit, alternatives unconstrained.
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        w_eq = float(np.sum(w[_EQ_IDX]))
        w_ficr = float(np.sum(w[_FICR_IDX]))
        task = (w_eq - eq_target) ** 2 + (w_ficr - ficr_target) ** 2
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                eq_p = float(np.sum(w_p[_EQ_IDX]))
                eq_n = float(np.sum(w_n[_EQ_IDX]))
                ficr_p = float(np.sum(w_p[_FICR_IDX]))
                ficr_n = float(np.sum(w_n[_FICR_IDX]))
                task_p = (eq_p - eq_target) ** 2 + (ficr_p - ficr_target) ** 2
                task_n = (eq_n - eq_target) ** 2 + (ficr_n - ficr_target) ** 2
                dG_dmu[k] = (task_p - task_n) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


def classic_sixty_forty_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    eq_target: float = 0.60,
    ficr_target: float = 0.40,
    alt_target: float = 0.00,
    alt_penalty: float = 1.0,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Cleaner 60/40 story:
      - equities ≈ 60%
      - fixed income + credit ≈ 40%
      - alternatives/real assets ≈ 0%
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        w_eq = float(np.sum(w[_EQ_IDX]))
        w_ficr = float(np.sum(w[_FICR_IDX]))
        w_alt = float(np.sum(w[_ALT_IDX]))
        task = (
            (w_eq - eq_target) ** 2
            + (w_ficr - ficr_target) ** 2
            + alt_penalty * (w_alt - alt_target) ** 2
        )
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)

                eq_p = float(np.sum(w_p[_EQ_IDX]))
                eq_n = float(np.sum(w_n[_EQ_IDX]))
                ficr_p = float(np.sum(w_p[_FICR_IDX]))
                ficr_n = float(np.sum(w_n[_FICR_IDX]))
                alt_p = float(np.sum(w_p[_ALT_IDX]))
                alt_n = float(np.sum(w_n[_ALT_IDX]))

                task_p = (
                    (eq_p - eq_target) ** 2
                    + (ficr_p - ficr_target) ** 2
                    + alt_penalty * (alt_p - alt_target) ** 2
                )
                task_n = (
                    (eq_n - eq_target) ** 2
                    + (ficr_n - ficr_target) ** 2
                    + alt_penalty * (alt_n - alt_target) ** 2
                )
                dG_dmu[k] = (task_p - task_n) / (2.0 * eps)

            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


# ---------------------------------------------------------------------------
# I. Sharpe probe
# ---------------------------------------------------------------------------

def max_sharpe_total_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        ev = pipeline.evaluate_at(m)
        sharpe = ev["pred_return_total"] / max(ev["risk"], 1e-8)
        task = -sharpe
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            ev = pipeline.evaluate_at(m)
            mu_hat = ev["mu_hat"]
            rf = ev["rf_rate"]
            risk = max(ev["risk"], 1e-8)
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                ret_p = float(mu_p @ w_p) + rf
                ret_n = float(mu_n @ w_n) + rf
                risk_p = float(np.sqrt(max(w_p @ pipeline.sigma @ w_p, 0.0)))
                risk_n = float(np.sqrt(max(w_n @ pipeline.sigma @ w_n, 0.0)))
                sharpe_p = ret_p / max(risk_p, 1e-8)
                sharpe_n = ret_n / max(risk_n, 1e-8)
                dG_dmu[k] = -(sharpe_p - sharpe_n) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            if _SHORT_RATE_US_IDX >= 0:
                grad_rf = np.zeros(len(m))
                grad_rf[_SHORT_RATE_US_IDX] = -(1.0 / 100.0) / risk
                grad_task = grad_task + grad_rf
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


# ---------------------------------------------------------------------------
# J. Equity tilt probes
# ---------------------------------------------------------------------------

def max_equity_tilt_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Original Akif probe:
    maximize US equity allocation.
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        task = -float(w[EQ_US_IDX])
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                dG_dmu[k] = -(w_p[EQ_US_IDX] - w_n[EQ_US_IDX]) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


def max_total_equity_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    """
    Cleaner equity story:
      maximize total equity share across all equity sleeves.
    """
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        task = -float(np.sum(w[_EQ_IDX]))
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                eq_p = float(np.sum(w_p[_EQ_IDX]))
                eq_n = float(np.sum(w_n[_EQ_IDX]))
                dG_dmu[k] = -(eq_p - eq_n) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


# ---------------------------------------------------------------------------
# K. Narrative Event Probes
# ---------------------------------------------------------------------------

def flight_to_safety_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        task = -float(w[FI_UST_IDX] + w[GOLD_IDX])
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                task_p = float(w_p[FI_UST_IDX] + w_p[GOLD_IDX])
                task_n = float(w_n[FI_UST_IDX] + w_n[GOLD_IDX])
                dG_dmu[k] = -(task_p - task_n) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG


def real_asset_rotation_probe(
    pipeline,
    m0: np.ndarray,
    scales: np.ndarray,
    l2reg: float = 0.1,
    prior=None,
    var1_l2reg: float = 0.5,
) -> tuple[Callable, Callable]:
    def G(m: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        w = pipeline(m)
        task = -float(w[GOLD_IDX] + w[RE_US_IDX])
        reg = var1_reg(m, m0, prior, l2reg=var1_l2reg) if prior is not None \
            else l2_anchor_reg(m, m0, scales, l2reg)
        return task + reg

    if prior is not None and pipeline.has_analytical_grad():

        def gradG_analytical(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=float)
            J_mu = pipeline.analytical_grad_mu(m)
            mu_hat = pipeline.predict(pipeline._feature_builder(m))
            eps = _FD_EPS_W
            dG_dmu = np.zeros(len(mu_hat))
            for k in range(len(mu_hat)):
                mu_p = mu_hat.copy()
                mu_p[k] += eps
                mu_n = mu_hat.copy()
                mu_n[k] -= eps
                w_p = pipeline.optimize(mu_p)
                w_n = pipeline.optimize(mu_n)
                task_p = float(w_p[GOLD_IDX] + w_p[RE_US_IDX])
                task_n = float(w_n[GOLD_IDX] + w_n[RE_US_IDX])
                dG_dmu[k] = -(task_p - task_n) / (2.0 * eps)
            grad_task = dG_dmu @ J_mu
            grad_reg = prior.regularizer_grad(m, m0, l2reg=var1_l2reg)
            return grad_task + grad_reg

        return G, gradG_analytical
    else:
        _, gradG = make_G_and_gradG(G)
        return G, gradG