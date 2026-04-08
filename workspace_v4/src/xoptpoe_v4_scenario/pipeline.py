"""
pipeline.py  —  v3

Changes from v2
---------------
evaluate_at() now returns two additional keys:
  - rf_rate          : short_rate_US / 100  (decimal, e.g. 0.0442 = 4.42%/yr)
  - pred_return_total: pred_return_excess + rf_rate

short_rate_US is stored in PERCENT units in feature_master (confirmed).
Dividing by 100 converts to decimal, consistent with pred_return units.

This is the correct rf proxy for scenario generation:
  - annualized_rf_forward_return is NaN at all anchor dates (future data)
  - short_rate_US matches real-world fed funds exactly at anchor dates
  - It varies per scenario as MALA perturbs the macro state (correct)
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WORKSPACE_SRC = _HERE.parent
_REPO_SRC = _WORKSPACE_SRC.parent.parent / "src"

if str(_WORKSPACE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_SRC))
if str(_REPO_SRC) not in sys.path:
    sys.path.append(str(_REPO_SRC))

from xoptpoe_v4_models.optim_layers import (
    OptimizerConfig,
    RiskConfig,
    RobustOptimizerCache,
    build_sigma_map,
    estimate_ewma_covariance,
)
from xoptpoe_v4_scenario.state_space import (
    MACRO_STATE_COLS,
    STATE_DIM,
    INTERACTION_MAP,
    FastFeatureBuilder,
    build_feature_matrix,
    load_state,
)

# ---------------------------------------------------------------------------
# Locked benchmark constants
# ---------------------------------------------------------------------------
LAMBDA_RISK: float = 8.0
KAPPA: float = 0.10
OMEGA_TYPE: str = "identity"
BEST_60_EXPERIMENT: str = "elastic_net__core_plus_interactions__separate_60"

SLEEVES_14: tuple[str, ...] = (
    "EQ_US", "EQ_EZ", "EQ_JP", "EQ_CN", "EQ_EM",
    "FI_UST", "FI_EU_GOVT",
    "CR_US_IG", "CR_EU_IG", "CR_US_HY",
    "RE_US", "LISTED_RE", "LISTED_INFRA",
    "ALT_GLD",
)

# Index of short_rate_US in MACRO_STATE_COLS — used for rf add-back
# Resolved at module load time; safe because MACRO_STATE_COLS is frozen.
try:
    _SHORT_RATE_US_IDX: int = list(MACRO_STATE_COLS).index("short_rate_US")
except ValueError:
    _SHORT_RATE_US_IDX = -1  # fallback: rf_rate will be 0.0


@dataclass
class PreprocessorState:
    """Minimal fitted preprocessor state needed for inference."""
    feature_names: list[str]
    original_feature_names: list[str]
    indicator_feature_names: list[str]
    fill_values: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    _fill_arr: np.ndarray = field(default=None, repr=False)
    _mean_arr: np.ndarray = field(default=None, repr=False)
    _std_arr: np.ndarray = field(default=None, repr=False)
    _indicator_col_indices: list = field(default=None, repr=False)
    _output_col_order: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self._build_numpy_cache()

    def _build_numpy_cache(self):
        self._fill_arr = np.array(
            [self.fill_values[f] for f in self.original_feature_names], dtype=np.float64
        )
        self._mean_arr = np.array(
            [self.means[f] for f in self.original_feature_names], dtype=np.float64
        )
        self._std_arr = np.array(
            [self.stds[f] for f in self.original_feature_names], dtype=np.float64
        )
        orig_idx = {f: i for i, f in enumerate(self.original_feature_names)}
        self._indicator_col_indices = [orig_idx[f] for f in self.indicator_feature_names]
        assembled_cols = list(self.original_feature_names) + [
            f"{n}__missing" for n in self.indicator_feature_names
        ]
        assembled_idx = {f: i for i, f in enumerate(assembled_cols)}
        self._output_col_order = np.array(
            [assembled_idx[f] for f in self.feature_names], dtype=np.intp
        )

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        is_nan = np.isnan(X)
        X_filled = np.where(is_nan, self._fill_arr[np.newaxis, :], X)
        X_scaled = (X_filled - self._mean_arr) / self._std_arr
        if self._indicator_col_indices:
            indicators = is_nan[:, self._indicator_col_indices].astype(np.float64)
            assembled = np.concatenate([X_scaled, indicators], axis=1)
        else:
            assembled = X_scaled
        return assembled[:, self._output_col_order].astype(np.float32)

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        X_raw = X_df[self.original_feature_names].to_numpy(dtype=np.float64)
        return self.transform_numpy(X_raw)


class v4AllocationPipeline:
    """
    The locked v4 benchmark allocation pipeline.

    v3 changes
    ----------
    evaluate_at() returns two additional keys:
      rf_rate           : short_rate_US / 100  (decimal)
      pred_return_total : pred_return_excess + rf_rate

    short_rate_US is in PERCENT units in feature_master.
    Confirmed: 2022=4.15%, 2023=5.27%, 2024=4.42% match fed funds exactly.
    """

    def __init__(
        self,
        elastic_net_model,
        preprocessor: PreprocessorState,
        feature_columns: list[str],
        sigma: np.ndarray,
        anchor_date: pd.Timestamp,
        feature_matrix_base: pd.DataFrame,
        sleeve_order: Sequence[str] = SLEEVES_14,
    ):
        self.elastic_net = elastic_net_model
        self.preprocessor = preprocessor
        self.feature_columns = list(feature_columns)
        self.sigma = np.asarray(sigma, dtype=float)
        self.anchor_date = pd.Timestamp(anchor_date)
        self.feature_matrix_base = feature_matrix_base
        self.sleeve_order = list(sleeve_order)
        self.n_sleeves = len(sleeve_order)

        self._feature_builder = FastFeatureBuilder(feature_matrix_base, list(feature_columns))

        opt_config = OptimizerConfig(
            lambda_risk=LAMBDA_RISK,
            kappa=KAPPA,
            omega_type=OMEGA_TYPE,
        )
        self._opt_config = opt_config
        self._sigma_map = {self.anchor_date: self.sigma}
        self._opt_cache = RobustOptimizerCache(sigma_by_month=self._sigma_map)

        self._grad_mu_wrt_m: np.ndarray | None = None
        self._build_analytical_grad_cache()

    # ------------------------------------------------------------------
    # Analytical gradient cache (unchanged from v2)
    # ------------------------------------------------------------------

    def _build_analytical_grad_cache(self) -> None:
        try:
            en = self.elastic_net
            coef = np.asarray(en.coef_, dtype=float)
            prep = self.preprocessor
            feat_cols = self.feature_columns
            macro_col_idx = {col: i for i, col in enumerate(MACRO_STATE_COLS)}
            prep_name_to_idx = {name: j for j, name in enumerate(prep.feature_names)}
            base_row = self.feature_matrix_base.iloc[0]
            m_base = np.array(
                [float(base_row[col]) for col in MACRO_STATE_COLS], dtype=float
            )
            n_sleeves = self.n_sleeves
            n_state = STATE_DIM
            J = np.zeros((n_sleeves, n_state), dtype=float)
            X_base = self.feature_matrix_base[feat_cols].to_numpy(dtype=float)

            for j_feat, feat_name in enumerate(feat_cols):
                if feat_name not in prep_name_to_idx:
                    continue
                j_prep = prep_name_to_idx[feat_name]
                if j_prep >= len(coef):
                    continue
                coef_j = coef[j_prep]
                if coef_j == 0.0:
                    continue
                std_j = prep.stds.get(feat_name, 1.0)
                if std_j <= 1e-10:
                    continue
                if feat_name in macro_col_idx:
                    i_state = macro_col_idx[feat_name]
                    J[:, i_state] += coef_j / std_j
                elif feat_name in INTERACTION_MAP:
                    macro_col = INTERACTION_MAP[feat_name]
                    if macro_col not in macro_col_idx:
                        continue
                    i_state = macro_col_idx[macro_col]
                    base_macro_val = m_base[i_state]
                    if abs(base_macro_val) <= 1e-12:
                        continue
                    sleeve_vals = X_base[:, j_feat] / base_macro_val
                    J[:, i_state] += coef_j * sleeve_vals / std_j

            self._grad_mu_wrt_m = J

        except Exception as e:
            warnings.warn(
                f"analytical_grad_mu cache build failed ({e}); "
                "falling back to finite differences."
            )
            self._grad_mu_wrt_m = None

    def analytical_grad_mu(self, m: np.ndarray) -> np.ndarray:
        if self._grad_mu_wrt_m is None:
            raise RuntimeError("Analytical gradient cache not available.")
        return self._grad_mu_wrt_m

    def has_analytical_grad(self) -> bool:
        return self._grad_mu_wrt_m is not None

    # ------------------------------------------------------------------
    # Core pipeline methods
    # ------------------------------------------------------------------

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        X_preprocessed = self.preprocessor.transform_numpy(
            np.asarray(X_raw, dtype=np.float64)
        )
        return self.elastic_net.predict(X_preprocessed).astype(float)

    def optimize(self, mu_hat: np.ndarray) -> np.ndarray:
        return self._opt_cache.solve(
            month_end=self.anchor_date,
            mu=np.asarray(mu_hat, dtype=float),
            config=self._opt_config,
        )

    def __call__(self, m_perturbed: np.ndarray) -> np.ndarray:
        X_raw = self._feature_builder(m_perturbed)
        mu_hat = self.predict(X_raw)
        return self.optimize(mu_hat)

    def evaluate_at(self, m: np.ndarray) -> dict[str, object]:
        """
        Evaluate pipeline at macro state m.

        Returns
        -------
        dict with keys:
          w                  : portfolio weights (14,)
          mu_hat             : predicted sleeve excess returns (14,)
          pred_return        : mu_hat @ w  [EXCESS, decimal]
          pred_return_excess : alias for pred_return  [EXCESS, decimal]
          rf_rate            : short_rate_US / 100  [decimal]
          pred_return_total  : pred_return_excess + rf_rate  [TOTAL, decimal]
          risk               : portfolio volatility (annualized)
          entropy            : portfolio entropy
          sharpe_pred        : pred_return / risk  [uses EXCESS return]

        Units note
        ----------
        short_rate_US is stored in PERCENT in feature_master (e.g. 4.42 = 4.42%).
        Dividing by 100 converts to decimal, consistent with pred_return units.
        annualized_rf_forward_return is NaN at all anchor dates (future data)
        and cannot be used — short_rate_US is the correct contemporaneous proxy.
        """
        m = np.asarray(m, dtype=float)
        X_raw = self._feature_builder(m)
        mu_hat = self.predict(X_raw)
        w = self.optimize(mu_hat)

        pred_ret_excess = float(mu_hat @ w)
        risk = float(np.sqrt(max(w @ self.sigma @ w, 0.0)))
        entropy = float(-np.sum(w[w > 1e-10] * np.log(w[w > 1e-10])))

        # RF add-back
        # short_rate_US is in PERCENT units -> divide by 100
        if _SHORT_RATE_US_IDX >= 0:
            rf_rate = float(m[_SHORT_RATE_US_IDX]) / 100.0
        else:
            rf_rate = 0.0

        pred_ret_total = pred_ret_excess + rf_rate

        return {
            "w":                   w,
            "mu_hat":              mu_hat,
            "pred_return":         pred_ret_excess,   # backward compat alias
            "pred_return_excess":  pred_ret_excess,
            "rf_rate":             rf_rate,
            "pred_return_total":   pred_ret_total,
            "risk":                risk,
            "entropy":             entropy,
            "sharpe_pred":         pred_ret_excess / max(risk, 1e-8),
            "sharpe_pred_total":   pred_ret_total  / max(risk, 1e-8),
        }


# ---------------------------------------------------------------------------
# Refit helpers (unchanged)
# ---------------------------------------------------------------------------

def refit_elastic_net(
    modeling_panel: pd.DataFrame,
    feature_columns: list[str],
    feature_manifest: pd.DataFrame,
    train_end: pd.Timestamp,
    alpha: float,
    l1_ratio: float,
) -> tuple[object, PreprocessorState]:
    from sklearn.linear_model import ElasticNet

    TARGET_COL = "annualized_excess_forward_return"

    panel_60 = modeling_panel[
        (modeling_panel["horizon_months"] == 60) &
        (modeling_panel["baseline_trainable_flag"] == 1) &
        (modeling_panel["target_available_flag"] == 1)
    ].copy()
    panel_60["month_end"] = pd.to_datetime(panel_60["month_end"])
    train_df = panel_60[panel_60["month_end"] <= pd.Timestamp(train_end)].copy()
    if train_df.empty:
        raise ValueError(f"No training rows for train_end={train_end}")

    prep_state = _fit_preprocessor_state(train_df, feature_manifest, feature_columns)
    X_train_raw = train_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    X_train = prep_state.transform_numpy(X_train_raw)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)

    model = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=20000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, prep_state


def _fit_preprocessor_state(
    train_df: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    feature_columns: list[str],
) -> PreprocessorState:
    manifest = feature_manifest.set_index("feature_name")
    original = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")

    fill_values: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    indicator_features: list[str] = []

    for feat in feature_columns:
        if feat not in manifest.index:
            fill_values[feat] = 0.0
            means[feat] = 0.0
            stds[feat] = 1.0
            continue
        row = manifest.loc[feat]
        strategy = str(row.get("imputation_strategy_hint", ""))
        series = original[feat]

        if int(row.get("missing_indicator_recommended", 0)) == 1:
            indicator_features.append(feat)

        fill_value = (
            0.0
            if strategy == "zero_fill_with_indicator"
            else float(series.median()) if series.notna().any() else 0.0
        )
        fill_values[feat] = fill_value

        filled = series.fillna(fill_value)
        mean = float(filled.mean())
        std_val = float(filled.std(ddof=1)) if len(filled) > 1 else 1.0
        if not np.isfinite(std_val) or std_val <= 1e-8:
            std_val = 1.0
        means[feat] = mean
        stds[feat] = std_val

    indicator_feature_names = [f for f in feature_columns if f in indicator_features]
    output_names = list(feature_columns) + [
        f"{n}__missing" for n in indicator_feature_names
    ]

    return PreprocessorState(
        feature_names=output_names,
        original_feature_names=list(feature_columns),
        indicator_feature_names=indicator_feature_names,
        fill_values=fill_values,
        means=means,
        stds=stds,
    )



def benchmark_train_end(anchor_date: pd.Timestamp) -> pd.Timestamp:
    """Match the benchmark walk-forward cutoff used for the 60m target."""
    return pd.Timestamp(anchor_date) - pd.offsets.MonthEnd(60)


def load_locked_best60_params(
    metrics_path: str | Path,
    experiment_name: str = BEST_60_EXPERIMENT,
) -> dict[str, float]:
    """Load the selected Elastic Net hyperparameters used by the locked 60m benchmark."""
    metrics = pd.read_csv(metrics_path)
    row = metrics.loc[metrics["experiment_name"].eq(experiment_name)]
    if row.empty:
        raise KeyError(f"Missing experiment_name={experiment_name!r} in {metrics_path}")
    import ast
    params = ast.literal_eval(str(row["selected_params"].iloc[0]))
    alpha = float(params["alpha"])
    l1_ratio = float(params["l1_ratio"])
    return {"alpha": alpha, "l1_ratio": l1_ratio}


def build_benchmark_aligned_pipeline_at_date(
    anchor_date: pd.Timestamp,
    feature_master: pd.DataFrame,
    modeling_panel: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    feature_columns: list[str],
    excess_returns_monthly: pd.DataFrame,
    benchmark_metrics_path: str | Path,
    experiment_name: str = BEST_60_EXPERIMENT,
    train_end: pd.Timestamp | None = None,
    sleeve_order: Sequence[str] = SLEEVES_14,
    risk_config: RiskConfig | None = None,
) -> v4AllocationPipeline:
    """
    Rebuild the scenario pipeline so it matches the locked 60m benchmark path:
      1) use the selected Elastic Net hyperparameters from the prediction benchmark,
      2) refit per anchor with cutoff = anchor - 60 month-ends unless overridden,
      3) keep the allocator/risk layer unchanged.
    """
    anchor_date = pd.Timestamp(anchor_date)
    if train_end is None:
        train_end = benchmark_train_end(anchor_date)
    if risk_config is None:
        risk_config = RiskConfig()

    params = load_locked_best60_params(benchmark_metrics_path, experiment_name=experiment_name)
    en_model, preprocessor = refit_elastic_net(
        modeling_panel=modeling_panel,
        feature_columns=feature_columns,
        feature_manifest=feature_manifest,
        train_end=pd.Timestamp(train_end),
        alpha=float(params["alpha"]),
        l1_ratio=float(params["l1_ratio"]),
    )

    _, feature_matrix_base = load_state(
        anchor_date, feature_master, sleeve_order,
        modeling_panel=modeling_panel,
        horizon_months=60,
    )

    sigma = estimate_ewma_covariance(
        excess_history=excess_returns_monthly,
        month_end=anchor_date,
        config=risk_config,
    )

    return v4AllocationPipeline(
        elastic_net_model=en_model,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        sigma=sigma,
        anchor_date=anchor_date,
        feature_matrix_base=feature_matrix_base,
        sleeve_order=sleeve_order,
    )


def build_pipeline_at_date(
    anchor_date: pd.Timestamp,
    feature_master: pd.DataFrame,
    modeling_panel: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    feature_columns: list[str],
    excess_returns_monthly: pd.DataFrame,
    elastic_net_alpha: float,
    elastic_net_l1: float,
    train_end: pd.Timestamp | None = None,
    sleeve_order: Sequence[str] = SLEEVES_14,
    risk_config: RiskConfig | None = None,
) -> v4AllocationPipeline:
    anchor_date = pd.Timestamp(anchor_date)
    if train_end is None:
        train_end = anchor_date - pd.Timedelta(days=1)
    if risk_config is None:
        risk_config = RiskConfig()

    en_model, preprocessor = refit_elastic_net(
        modeling_panel=modeling_panel,
        feature_columns=feature_columns,
        feature_manifest=feature_manifest,
        train_end=train_end,
        alpha=elastic_net_alpha,
        l1_ratio=elastic_net_l1,
    )

    _, feature_matrix_base = load_state(
        anchor_date, feature_master, sleeve_order,
        modeling_panel=modeling_panel,
        horizon_months=60,
    )

    sigma = estimate_ewma_covariance(
        excess_history=excess_returns_monthly,
        month_end=anchor_date,
        config=risk_config,
    )

    return v4AllocationPipeline(
        elastic_net_model=en_model,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        sigma=sigma,
        anchor_date=anchor_date,
        feature_matrix_base=feature_matrix_base,
        sleeve_order=sleeve_order,
    )


# ---------------------------------------------------------------------------
# Pre-MALA alignment gate
# ---------------------------------------------------------------------------

_ANCHOR_2021: pd.Timestamp = pd.Timestamp("2021-12-31")
_GOLD_ZERO_TOL: float = 1e-3   # 0.1% — benchmark-aligned 2021 gold forecast is -5.9%


def validate_anchor_alignment(
    *,
    anchor: pd.Timestamp,
    pipeline: v4AllocationPipeline,
    m0: np.ndarray,
    expected_experiment: str,
    expected_train_end: pd.Timestamp,
    covariance_source: str,
    strict_2021: bool = True,
) -> dict:
    """
    Evaluate the benchmark-aligned anchor object before scenario generation.

    Prints a structured provenance block and returns a result dict.
    Must be called — and must return a passing result — before any MALA chain
    starts.

    Hard rules
    ----------
    - For 2021-12-31, expected_train_end must equal 2016-12-31.
    - For 2021-12-31 (strict_2021=True), w_ALT_GLD must be < 1e-3.
      If not, raises RuntimeError so MALA cannot start.
    - If covariance_source != 'benchmark_exact', prints
      *** APPROXIMATE BENCHMARK RECONSTRUCTION *** so output is never
      silently approximate.

    Returns
    -------
    dict with keys: anchor_date, experiment, train_end, covariance_source,
    pred_return_excess, rf_rate, pred_return_total, w_ALT_GLD, top_sleeve,
    top_weight, alignment_passed, covariance_exact, status.
    """
    anchor = pd.Timestamp(anchor)
    expected_train_end = pd.Timestamp(expected_train_end)
    m0 = np.asarray(m0, dtype=float)

    ev0 = pipeline.evaluate_at(m0)
    w0  = ev0["w"]

    gld_idx    = list(pipeline.sleeve_order).index("ALT_GLD")
    w_gld      = float(w0[gld_idx])
    top_idx    = int(np.argmax(w0))
    top_sleeve = pipeline.sleeve_order[top_idx]
    top_weight = float(w0[top_idx])

    pred_excess = float(ev0["pred_return_excess"])
    rf_rate     = float(ev0["rf_rate"])
    pred_total  = float(ev0["pred_return_total"])
    exact_cov   = (covariance_source == "benchmark_exact")

    # ── Structured provenance print ────────────────────────────────────────
    sep = "─" * 64
    print(sep)
    print(f"  [ALIGNMENT GATE] anchor_date   = {anchor.date()}")
    print(f"  experiment      = {expected_experiment}")
    print(f"  train_cutoff    = {expected_train_end.date()}")
    cov_tag = "" if exact_cov else "  *** APPROXIMATE BENCHMARK RECONSTRUCTION ***"
    print(f"  covariance      = {covariance_source}{cov_tag}")
    print(f"  pred_excess     = {pred_excess*100:.4f}%")
    print(f"  rf_rate         = {rf_rate*100:.3f}%")
    print(f"  pred_total      = {pred_total*100:.4f}%")
    print(f"  w_ALT_GLD       = {w_gld:.6f}  ({w_gld*100:.3f}%)")
    print(f"  top_sleeve      = {top_sleeve}  ({top_weight*100:.2f}%)")

    # ── Hard rules (2021-12-31 only) ───────────────────────────────────────
    passed   = True
    fail_msg = ""

    if anchor == _ANCHOR_2021:
        expected_cutoff_2021 = pd.Timestamp("2016-12-31")
        if expected_train_end != expected_cutoff_2021:
            fail_msg = (
                f"2021-12-31 train_end={expected_train_end.date()} "
                f"but benchmark requires {expected_cutoff_2021.date()} "
                "(anchor − 60 months). Scenario is probing the wrong model."
            )
            passed = False

        if passed and w_gld >= _GOLD_ZERO_TOL:
            fail_msg = (
                f"2021-12-31 w_ALT_GLD={w_gld:.4f} >= {_GOLD_ZERO_TOL}. "
                "Benchmark-aligned 2021 predictor ranks gold last "
                "(predicted excess ≈ −5.9%); expected weight ≈ 0. "
                "Scenario is probing the wrong anchor object."
            )
            passed = False

    status = "PASS" if passed else "FAIL"
    print(f"  alignment_gate  = {status}")
    if not passed:
        print(f"  FAIL REASON     = {fail_msg}")
    print(sep)

    if not passed and strict_2021 and anchor == _ANCHOR_2021:
        raise RuntimeError(
            f"[ALIGNMENT GATE FAILED] anchor=2021-12-31\n"
            f"  {fail_msg}\n"
            "MALA chains will not start until the pipeline is corrected."
        )

    return {
        "anchor_date":        anchor.date(),
        "experiment":         expected_experiment,
        "train_end":          expected_train_end.date(),
        "covariance_source":  covariance_source,
        "pred_return_excess": round(pred_excess, 6),
        "rf_rate":            round(rf_rate, 6),
        "pred_return_total":  round(pred_total, 6),
        "w_ALT_GLD":          round(w_gld, 6),
        "top_sleeve":         top_sleeve,
        "top_weight":         round(top_weight, 6),
        "alignment_passed":   passed,
        "covariance_exact":   exact_cov,
        "status":             status,
    }
