"""State-space definitions and anchor-row manipulation for v3 scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xoptpoe_v3_models.data import SLEEVE_ORDER


BASE_STATE_VARIABLES: tuple[str, ...] = (
    "infl_US",
    "unemp_US",
    "short_rate_US",
    "long_rate_US",
    "infl_EA",
    "unemp_EA",
    "short_rate_EA",
    "long_rate_EA",
    "infl_JP",
    "unemp_JP",
    "short_rate_JP",
    "long_rate_JP",
    "usd_broad",
    "vix",
    "us_real10y",
    "ig_oas",
    "oil_wti",
)

DERIVED_STATE_VARIABLES: tuple[str, ...] = (
    "term_slope_US",
    "term_slope_EA",
    "term_slope_JP",
    "infl_US_delta_1m",
    "unemp_US_delta_1m",
    "short_rate_US_delta_1m",
    "long_rate_US_delta_1m",
    "term_slope_US_delta_1m",
    "infl_EA_delta_1m",
    "unemp_EA_delta_1m",
    "short_rate_EA_delta_1m",
    "long_rate_EA_delta_1m",
    "term_slope_EA_delta_1m",
    "infl_JP_delta_1m",
    "unemp_JP_delta_1m",
    "short_rate_JP_delta_1m",
    "long_rate_JP_delta_1m",
    "term_slope_JP_delta_1m",
    "usd_broad_logchg_1m",
    "usd_broad_logchg_12m",
    "vix_delta_1m",
    "us_real10y_delta_1m",
    "ig_oas_delta_1m",
    "oil_wti_logchg_1m",
    "oil_wti_logchg_12m",
)

FIXED_CONTEXT_CANDIDATES: tuple[str, ...] = (
    "china_cli",
    "jp_pe_ratio",
    "cape_local",
    "cape_usa",
    "oecd_activity_proxy_local",
    "mom_12_1",
    "vol_12m",
    "em_minus_global_pe",
    "rel_mom_vs_treasury",
)


@dataclass(frozen=True)
class ScenarioStateSpec:
    """Scenario state definition for the first v3 pass."""

    base_variables: tuple[str, ...]
    derived_variables: tuple[str, ...]
    fixed_context_candidates: tuple[str, ...]
    scenario_horizons: tuple[int, ...] = (60, 120)


@dataclass(frozen=True)
class ScenarioAnchor:
    """Anchor-date context for scenario evaluation."""

    month_end: pd.Timestamp
    previous_month_end: pd.Timestamp
    lag12_month_end: pd.Timestamp
    scenario_horizons: tuple[int, ...]
    anchor_rows: pd.DataFrame
    current_base_state: np.ndarray
    previous_base_state: np.ndarray
    lag12_base_state: np.ndarray


def default_state_spec() -> ScenarioStateSpec:
    """Return the conservative interpretable first-pass state."""
    return ScenarioStateSpec(
        base_variables=BASE_STATE_VARIABLES,
        derived_variables=DERIVED_STATE_VARIABLES,
        fixed_context_candidates=FIXED_CONTEXT_CANDIDATES,
    )


def _month_level_state_rows(feature_master_monthly: pd.DataFrame) -> pd.DataFrame:
    keep = ["month_end", *BASE_STATE_VARIABLES]
    work = feature_master_monthly.loc[:, keep].copy()
    work["month_end"] = pd.to_datetime(work["month_end"])
    grouped = work.groupby("month_end", as_index=False).first().sort_values("month_end").reset_index(drop=True)
    for variable_name in BASE_STATE_VARIABLES:
        nunique = work.groupby("month_end")[variable_name].nunique(dropna=False)
        if int((nunique > 1).sum()) > 0:
            raise ValueError(f"Base state variable is not unique by month_end: {variable_name}")
    return grouped


def _extract_base_vector(rows: pd.DataFrame, variables: tuple[str, ...]) -> np.ndarray:
    if rows.empty:
        raise ValueError("Cannot extract state from an empty frame")
    first = rows.iloc[0]
    values = []
    for variable_name in variables:
        if variable_name not in rows.columns:
            raise ValueError(f"Anchor rows missing base variable: {variable_name}")
        nunique = rows[variable_name].nunique(dropna=False)
        if nunique != 1:
            raise ValueError(f"Base variable is not constant across anchor rows: {variable_name}")
        values.append(float(first[variable_name]))
    return np.asarray(values, dtype=float)


def build_anchor_context(
    modeling_panel_hstack: pd.DataFrame,
    feature_master_monthly: pd.DataFrame,
    *,
    month_end: pd.Timestamp,
    spec: ScenarioStateSpec,
) -> ScenarioAnchor:
    """Collect the anchor rows and their lagged monthly base state."""
    anchor_date = pd.Timestamp(month_end)
    panel = modeling_panel_hstack.copy()
    panel["month_end"] = pd.to_datetime(panel["month_end"])
    anchor_rows = panel.loc[
        panel["month_end"].eq(anchor_date) & panel["horizon_months"].isin(spec.scenario_horizons)
    ].copy()
    if anchor_rows.empty:
        raise ValueError(f"No active v3 stacked rows found for anchor date {anchor_date.date()}")

    expected_rows = len(SLEEVE_ORDER) * len(spec.scenario_horizons)
    if len(anchor_rows) != expected_rows:
        raise ValueError(
            f"Anchor date {anchor_date.date()} returned {len(anchor_rows)} rows; expected {expected_rows}"
        )

    dup_cnt = int(anchor_rows.duplicated(subset=["month_end", "sleeve_id", "horizon_months"]).sum())
    if dup_cnt > 0:
        raise ValueError(f"Anchor rows contain duplicate sleeve-horizon keys: {dup_cnt}")

    anchor_rows["sleeve_id"] = pd.Categorical(anchor_rows["sleeve_id"], categories=list(SLEEVE_ORDER), ordered=True)
    anchor_rows = (
        anchor_rows.sort_values(["horizon_months", "sleeve_id"])
        .reset_index(drop=True)
        .astype({"sleeve_id": str})
    )

    month_state = _month_level_state_rows(feature_master_monthly)
    if anchor_date not in set(month_state["month_end"]):
        raise ValueError(f"Anchor date {anchor_date.date()} is not present in feature_master_monthly")

    current_idx = month_state.index[month_state["month_end"].eq(anchor_date)][0]
    if current_idx < 12:
        raise ValueError(f"Anchor date {anchor_date.date()} does not have a full 12-month lag window")

    previous_row = month_state.iloc[current_idx - 1]
    lag12_row = month_state.iloc[current_idx - 12]
    current_row = month_state.iloc[current_idx]
    return ScenarioAnchor(
        month_end=anchor_date,
        previous_month_end=pd.Timestamp(previous_row["month_end"]),
        lag12_month_end=pd.Timestamp(lag12_row["month_end"]),
        scenario_horizons=spec.scenario_horizons,
        anchor_rows=anchor_rows,
        current_base_state=current_row[list(spec.base_variables)].to_numpy(dtype=float),
        previous_base_state=previous_row[list(spec.base_variables)].to_numpy(dtype=float),
        lag12_base_state=lag12_row[list(spec.base_variables)].to_numpy(dtype=float),
    )


def _safe_log_change(current: float, lagged: float) -> float:
    if not np.isfinite(current) or not np.isfinite(lagged) or current <= 0.0 or lagged <= 0.0:
        return float("nan")
    return float(np.log(current / lagged))


def _rebuild_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    formulas: dict[str, tuple[str, str]] = {
        "int_alternative_x_us_real10y": ("asset_group_dummy_alternative", "us_real10y"),
        "int_alternative_x_vix": ("asset_group_dummy_alternative", "vix"),
        "int_cape_local_x_us_real10y": ("cape_local", "us_real10y"),
        "int_cape_usa_x_us_real10y": ("cape_usa", "us_real10y"),
        "int_china_cli_x_ig_oas": ("china_cli", "ig_oas"),
        "int_equity_x_us_real10y": ("asset_group_dummy_equity", "us_real10y"),
        "int_equity_x_vix": ("asset_group_dummy_equity", "vix"),
        "int_fixed_income_x_us_real10y": ("asset_group_dummy_fixed_income", "us_real10y"),
        "int_fixed_income_x_vix": ("asset_group_dummy_fixed_income", "vix"),
        "int_ig_oas_x_cape_local": ("ig_oas", "cape_local"),
        "int_ig_oas_x_rel_mom_vs_treasury": ("ig_oas", "rel_mom_vs_treasury"),
        "int_log_horizon_x_cape_local": ("log_horizon_years", "cape_local"),
        "int_log_horizon_x_china_cli": ("log_horizon_years", "china_cli"),
        "int_log_horizon_x_jp_pe_ratio": ("log_horizon_years", "jp_pe_ratio"),
        "int_log_horizon_x_mom_12_1": ("log_horizon_years", "mom_12_1"),
        "int_log_horizon_x_us_real10y": ("log_horizon_years", "us_real10y"),
        "int_log_horizon_x_vix": ("log_horizon_years", "vix"),
        "int_log_horizon_x_vol_12m": ("log_horizon_years", "vol_12m"),
        "int_oecd_activity_proxy_local_x_local_term_slope": ("oecd_activity_proxy_local", "local_term_slope"),
        "int_real_asset_x_us_real10y": ("asset_group_dummy_real_asset", "us_real10y"),
        "int_real_asset_x_vix": ("asset_group_dummy_real_asset", "vix"),
        "int_us_real10y_x_mom_12_1": ("us_real10y", "mom_12_1"),
        "int_vix_x_em_minus_global_pe": ("vix", "em_minus_global_pe"),
        "int_vix_x_mom_12_1": ("vix", "mom_12_1"),
    }
    for target_name, (left_name, right_name) in formulas.items():
        if target_name in out.columns and left_name in out.columns and right_name in out.columns:
            out[target_name] = pd.to_numeric(out[left_name], errors="coerce") * pd.to_numeric(
                out[right_name], errors="coerce"
            )
    return out


def apply_state_vector(
    anchor: ScenarioAnchor,
    spec: ScenarioStateSpec,
    state_vector: np.ndarray,
) -> pd.DataFrame:
    """Overwrite the manipulated macro state and rebuild derived columns."""
    if len(state_vector) != len(spec.base_variables):
        raise ValueError(
            f"State vector length {len(state_vector)} does not match spec length {len(spec.base_variables)}"
        )

    out = anchor.anchor_rows.copy()
    current = {name: float(value) for name, value in zip(spec.base_variables, state_vector, strict=True)}
    previous = {
        name: float(value) for name, value in zip(spec.base_variables, anchor.previous_base_state, strict=True)
    }
    lag12 = {name: float(value) for name, value in zip(spec.base_variables, anchor.lag12_base_state, strict=True)}

    for variable_name, value in current.items():
        out[variable_name] = value

    out["term_slope_US"] = out["long_rate_US"] - out["short_rate_US"]
    out["term_slope_EA"] = out["long_rate_EA"] - out["short_rate_EA"]
    out["term_slope_JP"] = out["long_rate_JP"] - out["short_rate_JP"]

    for prefix in ("US", "EA", "JP"):
        out[f"infl_{prefix}_delta_1m"] = out[f"infl_{prefix}"] - previous[f"infl_{prefix}"]
        out[f"unemp_{prefix}_delta_1m"] = out[f"unemp_{prefix}"] - previous[f"unemp_{prefix}"]
        out[f"short_rate_{prefix}_delta_1m"] = out[f"short_rate_{prefix}"] - previous[f"short_rate_{prefix}"]
        out[f"long_rate_{prefix}_delta_1m"] = out[f"long_rate_{prefix}"] - previous[f"long_rate_{prefix}"]

    previous_slope_us = previous["long_rate_US"] - previous["short_rate_US"]
    previous_slope_ea = previous["long_rate_EA"] - previous["short_rate_EA"]
    previous_slope_jp = previous["long_rate_JP"] - previous["short_rate_JP"]
    out["term_slope_US_delta_1m"] = out["term_slope_US"] - previous_slope_us
    out["term_slope_EA_delta_1m"] = out["term_slope_EA"] - previous_slope_ea
    out["term_slope_JP_delta_1m"] = out["term_slope_JP"] - previous_slope_jp

    out["usd_broad_logchg_1m"] = _safe_log_change(current["usd_broad"], previous["usd_broad"])
    out["usd_broad_logchg_12m"] = _safe_log_change(current["usd_broad"], lag12["usd_broad"])
    out["vix_delta_1m"] = current["vix"] - previous["vix"]
    out["us_real10y_delta_1m"] = current["us_real10y"] - previous["us_real10y"]
    out["ig_oas_delta_1m"] = current["ig_oas"] - previous["ig_oas"]
    out["oil_wti_logchg_1m"] = _safe_log_change(current["oil_wti"], previous["oil_wti"])
    out["oil_wti_logchg_12m"] = _safe_log_change(current["oil_wti"], lag12["oil_wti"])

    out = _rebuild_interactions(out)
    return out


def build_state_manifest(feature_manifest: pd.DataFrame, spec: ScenarioStateSpec) -> pd.DataFrame:
    """Create the machine-readable state-manifest table for the first pass."""
    feature_names = set(feature_manifest["feature_name"].astype(str).tolist())
    rows: list[dict[str, object]] = []

    high_priority = {
        "usd_broad",
        "vix",
        "us_real10y",
        "ig_oas",
        "oil_wti",
        "infl_US",
        "short_rate_US",
        "long_rate_US",
    }

    for variable_name in spec.base_variables:
        rows.append(
            {
                "variable_name": variable_name,
                "state_block": "canonical_macro_base",
                "included_in_first_pass": 1,
                "manipulable_flag": 1,
                "interpretability_priority": "high" if variable_name in high_priority else "medium",
                "coverage_risk": "low",
                "notes": "Directly manipulated canonical macro state in the first v3 scenario pass.",
            }
        )

    for variable_name in spec.derived_variables:
        coverage_risk = "medium" if variable_name.endswith("12m") else "low"
        rows.append(
            {
                "variable_name": variable_name,
                "state_block": "derived_anchor_feature",
                "included_in_first_pass": 1,
                "manipulable_flag": 0,
                "interpretability_priority": "medium",
                "coverage_risk": coverage_risk,
                "notes": "Rebuilt from the manipulated base state and lagged actual anchor context.",
            }
        )

    for variable_name in spec.fixed_context_candidates:
        rows.append(
            {
                "variable_name": variable_name,
                "state_block": "held_fixed_enrichment",
                "included_in_first_pass": int(variable_name in feature_names),
                "manipulable_flag": 0,
                "interpretability_priority": "medium",
                "coverage_risk": "low" if variable_name in feature_names else "not_in_active_feature_set",
                "notes": "Held fixed at the anchor in the first pass to preserve interpretability.",
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["included_in_first_pass", "manipulable_flag", "variable_name"], ascending=[False, False, True]).reset_index(drop=True)
