#!/usr/bin/env python3
"""
reclassify_regimes_v4.py  —  v2

POST-HOC regime relabelling — does NOT re-run MALA.

What was broken in the original code
-------------------------------------
1. NFCI was loaded and percentiled but NEVER READ in score_regime_dimensions().
   Stress was computed from ig_oas + vix only.

2. NBER recession dates were hardcoded to only 6 post-1980 recessions,
   and add_recession_overlay() was never called in the runner scripts.

3. The full Recessiondating.md history (1857–2024) was completely ignored.

What this script does instead
------------------------------
1. NFCI is used as a THIRD stress signal alongside ig_oas and vix.
   NFCI > 0   = tighter than average (stress building)
   NFCI > 1   = significantly tight
   NFCI > 2   = crisis-level (2008 peak: 5.14, 2020 peak: 2.8)
   NFCI is interpolated from weekly to monthly and joined to scenarios
   via the anchor_date (MALA perturbs macro state but not calendar time).

2. Full NBER recession history (1857–2024) is loaded from Recessiondating.md
   and used to add:
     - in_recession flag for the anchor date
     - months_since_recession_end (cycle position)
     - recession_proximity label (in_recession / early_expansion /
       mid_expansion / late_expansion)
   These are ANCHOR-LEVEL context fields — they describe where we are
   in the cycle at the time of scenario generation, not the scenario itself.

3. All 19 macro state dims are used for per-bloc classification (US/EA/JP/Global).

Usage
-----
  # Q1–Q3 only
  python reclassify_regimes_v4.py \
      --input  workspace_v4/reports/scenario_results_v4.csv \
      --output workspace_v4/reports/scenario_results_v4_reclassified.csv \
      --nfci   workspace_v4/NFCI\ \(1\).csv

  # Q4–Q8 only (once akif run finishes)
  python reclassify_regimes_v4.py \
      --input  workspace_v4/reports/scenario_results_v4_akif.csv \
      --output workspace_v4/reports/scenario_results_v4_akif_reclassified.csv \
      --nfci   workspace_v4/NFCI\ \(1\).csv

  # Both combined
  python reclassify_regimes_v4.py \
      --input  workspace_v4/reports/scenario_results_v4.csv \
               workspace_v4/reports/scenario_results_v4_akif.csv \
      --output workspace_v4/reports/scenario_results_all_reclassified.csv \
      --nfci   workspace_v4/NFCI\ \(1\).csv

Runtime: ~10–20 seconds for 19,200 rows. No MALA, no pipeline, no model.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent
REPO_SRC  = WORKSPACE.parent.parent / "src"
for p in [str(WORKSPACE / "src"), str(REPO_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from xoptpoe_v4_scenario.state_space import MACRO_STATE_COLS

DATA_REFS = WORKSPACE / "data_refs"
NFCI_PATH = WORKSPACE / "NFCI (1).csv"

# ---------------------------------------------------------------------------
# Full NBER recession dates (from Recessiondating.md, 1857–2024)
# ---------------------------------------------------------------------------
NBER_RECESSIONS_FULL = [
    ("1857-06-01", "1858-12-01"), ("1860-10-01", "1861-06-01"),
    ("1865-04-01", "1867-12-01"), ("1869-06-01", "1870-12-01"),
    ("1873-10-01", "1879-03-01"), ("1882-03-01", "1885-05-01"),
    ("1887-03-01", "1888-04-01"), ("1890-07-01", "1891-05-01"),
    ("1893-01-01", "1894-06-01"), ("1895-12-01", "1897-06-01"),
    ("1899-06-01", "1900-12-01"), ("1902-09-01", "1904-08-01"),
    ("1907-05-01", "1908-06-01"), ("1910-01-01", "1912-01-01"),
    ("1913-01-01", "1914-12-01"), ("1918-08-01", "1919-03-01"),
    ("1920-01-01", "1921-07-01"), ("1923-05-01", "1924-07-01"),
    ("1926-10-01", "1927-11-01"), ("1929-08-01", "1933-03-01"),
    ("1937-05-01", "1938-06-01"), ("1945-02-01", "1945-10-01"),
    ("1948-11-01", "1949-10-01"), ("1953-07-01", "1954-05-01"),
    ("1957-08-01", "1958-04-01"), ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"), ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"), ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"), ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"), ("2020-02-01", "2020-04-01"),
]
_NBER_TS = [(pd.Timestamp(p), pd.Timestamp(t)) for p, t in NBER_RECESSIONS_FULL]


# ---------------------------------------------------------------------------
# NFCI loader
# ---------------------------------------------------------------------------

def load_nfci_monthly(nfci_path: Path) -> pd.Series:
    """
    Load NFCI CSV, resample weekly → monthly end-of-month.
    Returns a pd.Series indexed by month-end Timestamp.
    NFCI interpretation:
      > 0   tighter than historical average (stress building)
      > 1   significantly tight
      > 2   crisis-level (2008 peak=5.14, 2020 peak=2.8)
      < 0   looser than average (risk-on)
      < -1  very loose (historically rare: only 6 months in 55yr history)
    """
    nfci = pd.read_csv(nfci_path)
    nfci.columns = [c.strip() for c in nfci.columns]
    nfci["observation_date"] = pd.to_datetime(nfci["observation_date"])
    nfci["NFCI"] = pd.to_numeric(nfci["NFCI"], errors="coerce")
    nfci = nfci.dropna(subset=["NFCI"]).set_index("observation_date").sort_index()
    monthly = nfci["NFCI"].resample("ME").last().dropna()
    return monthly


# ---------------------------------------------------------------------------
# Recession context for a given date
# ---------------------------------------------------------------------------

def recession_context(date: pd.Timestamp) -> dict:
    """
    Compute recession context for a given calendar date using full NBER history.

    Returns:
      in_recession          : bool — date falls within a recession
      months_since_rec_end  : float — months since last recession trough
                              (0 if in recession, NaN if before first recession)
      recession_proximity   : str — in_recession / early_expansion (≤18m) /
                              mid_expansion (18–48m) / late_expansion (>48m)
      last_recession_peak   : str — date of last recession peak
      last_recession_trough : str — date of last recession trough
    """
    in_rec = any(peak <= date <= trough for peak, trough in _NBER_TS)

    past_troughs = [t for _, t in _NBER_TS if t < date]
    if not past_troughs:
        return {
            "in_recession": in_rec,
            "months_since_rec_end": float("nan"),
            "recession_proximity": "pre_nber_history",
            "last_recession_peak": "",
            "last_recession_trough": "",
        }

    last_trough = max(past_troughs)
    last_peak   = max(p for p, t in _NBER_TS if t == last_trough)
    months_since = (date - last_trough).days / 30.44

    if in_rec:
        proximity = "in_recession"
    elif months_since <= 18:
        proximity = "early_expansion"
    elif months_since <= 48:
        proximity = "mid_expansion"
    else:
        proximity = "late_expansion"

    return {
        "in_recession":           in_rec,
        "months_since_rec_end":   round(months_since, 1),
        "recession_proximity":    proximity,
        "last_recession_peak":    last_peak.strftime("%Y-%m-%d"),
        "last_recession_trough":  last_trough.strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def _load_thresholds(nfci_monthly: pd.Series | None = None) -> dict:
    print("Loading feature_master for threshold computation...")
    fm = pd.read_parquet(DATA_REFS / "feature_master_monthly.parquet")
    fm["month_end"] = pd.to_datetime(fm["month_end"])

    from xoptpoe_v4_scenario.regime import compute_regime_thresholds

    # Build a minimal nfci_df if we have NFCI data
    nfci_df = None
    if nfci_monthly is not None:
        nfci_df = pd.DataFrame({
            "month_end": nfci_monthly.index,
            "NFCI": nfci_monthly.values,
        })

    thresholds = compute_regime_thresholds(fm, nfci_df)

    # Add oil and usd percentiles (not in original compute_regime_thresholds)
    train_end = pd.Timestamp("2016-01-01")
    fm_one = fm.drop_duplicates(subset="month_end")
    fm_train = fm_one[fm_one["month_end"] < train_end]
    for col in ["oil_wti", "usd_broad", "infl_EA", "short_rate_EA",
                "unemp_EA", "term_slope_US", "term_slope_EA", "term_slope_JP"]:
        if col in fm_train.columns:
            vals = fm_train[col].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                thresholds[f"{col}_p25"] = float(np.percentile(vals, 25))
                thresholds[f"{col}_p50"] = float(np.percentile(vals, 50))
                thresholds[f"{col}_p75"] = float(np.percentile(vals, 75))
                thresholds[f"{col}_p90"] = float(np.percentile(vals, 90))

    # NFCI thresholds (pre-2016 training period)
    if nfci_monthly is not None:
        nfci_train = nfci_monthly[nfci_monthly.index < train_end]
        thresholds["nfci_p50"]  = float(nfci_train.quantile(0.50))
        thresholds["nfci_p75"]  = float(nfci_train.quantile(0.75))
        thresholds["nfci_p90"]  = float(nfci_train.quantile(0.90))
        thresholds["nfci_mean"] = float(nfci_train.mean())
        print(f"  NFCI thresholds (pre-2016): "
              f"p50={thresholds['nfci_p50']:.3f}, "
              f"p75={thresholds['nfci_p75']:.3f}, "
              f"p90={thresholds['nfci_p90']:.3f}")

    print(f"  Total thresholds: {len(thresholds)} keys")
    return thresholds


# ---------------------------------------------------------------------------
# Multi-bloc regime classifier — all 19 dims + NFCI
# ---------------------------------------------------------------------------

def _th(thresholds: dict, key: str, default: float) -> float:
    return thresholds.get(key, default)


def _classify_us_bloc(m: dict, th: dict) -> dict:
    infl    = m["infl_US"]
    short   = m["short_rate_US"]
    real10y = m["us_real10y"]
    unemp   = m["unemp_US"]
    slope   = m["term_slope_US"]

    growth = ("high" if unemp <= _th(th, "unemp_US_p25", 5.0) else
              "low"  if unemp >= _th(th, "unemp_US_p75", 7.0) else "neutral")

    inflation = ("low"  if infl <= _th(th, "infl_US_p25", 1.5) else
                 "high" if infl >= _th(th, "infl_US_p75", 3.0) else "neutral")

    policy_tight = (short  >= _th(th, "short_rate_US_p75", 3.5) or
                    real10y >= _th(th, "us_real10y_p75", 1.0))
    policy_easy  = (short  <= _th(th, "short_rate_US_p25", 0.5) and
                    real10y <= _th(th, "us_real10y_p25", -0.5))
    policy = "tight" if policy_tight else "easy" if policy_easy else "neutral"

    curve = ("inverted" if slope < 0 else
             "flat"     if slope <= _th(th, "term_slope_US_p25", 0.3) else
             "steep"    if slope >= _th(th, "term_slope_US_p75", 2.0) else "normal")

    return {"us_growth": growth, "us_inflation": inflation,
            "us_policy": policy, "us_curve": curve}


def _classify_ea_bloc(m: dict, th: dict) -> dict:
    infl  = m["infl_EA"]
    short = m["short_rate_EA"]
    unemp = m["unemp_EA"]
    slope = m["term_slope_EA"]

    growth = ("high" if unemp <= _th(th, "unemp_EA_p25", 7.5) else
              "low"  if unemp >= _th(th, "unemp_EA_p75", 10.0) else "neutral")

    inflation = ("low"  if infl <= _th(th, "infl_EA_p25", 1.0) else
                 "high" if infl >= _th(th, "infl_EA_p75", 2.5) else "neutral")

    policy_tight = short >= _th(th, "short_rate_EA_p75", 2.5)
    policy_easy  = short <= _th(th, "short_rate_EA_p25", 0.0)
    policy = "tight" if policy_tight else "easy" if policy_easy else "neutral"

    curve = ("inverted" if slope < 0 else
             "flat"     if slope < 0.5 else
             "steep"    if slope > 2.0 else "normal")

    return {"ea_growth": growth, "ea_inflation": inflation,
            "ea_policy": policy, "ea_curve": curve}


def _classify_jp_bloc(m: dict, th: dict) -> dict:
    infl  = m["infl_JP"]
    short = m["short_rate_JP"]
    slope = m["term_slope_JP"]

    # Japan-specific thresholds — deflation is the historical baseline
    inflation = ("deflation" if infl < 0.0 else
                 "low"       if infl < 1.0 else
                 "high"      if infl > 2.5 else "neutral")

    # NIRP and ZIRP are distinct JP policy states
    policy = ("nirp"    if short < 0.0 else
              "zirp"    if short < 0.25 else
              "tight"   if short > 1.0 else "neutral")

    curve = ("inverted" if slope < 0 else
             "flat"     if slope < 0.3 else
             "steep"    if slope > 1.5 else "normal")

    return {"jp_inflation": inflation, "jp_policy": policy, "jp_curve": curve}


def _classify_global_overlay(m: dict, th: dict, nfci_val: float | None) -> dict:
    """
    Global stress now uses THREE signals: ig_oas, vix, AND NFCI.
    NFCI is the broadest financial conditions index (105 sub-indicators).
    ig_oas and vix are credit and equity stress proxies.
    All three must agree for 'high' stress; any two for 'moderate'.
    """
    ig_oas = m["ig_oas"]
    vix    = m["vix"]
    oil    = m["oil_wti"]
    usd    = m["usd_broad"]

    # Individual stress signals
    ig_stress_high = ig_oas > _th(th, "ig_oas_p90", 2.5)
    ig_stress_mod  = ig_oas > _th(th, "ig_oas_p75", 1.5)
    vix_stress_high = vix   > _th(th, "vix_p90", 30.0)
    vix_stress_mod  = vix   > _th(th, "vix_p75", 22.0)

    # NFCI stress signal (if available)
    nfci_stress_high = False
    nfci_stress_mod  = False
    if nfci_val is not None and not np.isnan(nfci_val):
        nfci_stress_high = nfci_val > _th(th, "nfci_p90", 1.834)
        nfci_stress_mod  = nfci_val > _th(th, "nfci_p75", 0.374)

    # Combine: count how many signals are firing
    n_high = sum([ig_stress_high, vix_stress_high, nfci_stress_high])
    n_mod  = sum([ig_stress_mod,  vix_stress_mod,  nfci_stress_mod])

    # Require at least 2 signals to agree (more robust than any-one-fires)
    if n_high >= 2:
        stress = "high"
    elif n_mod >= 2:
        stress = "moderate"
    elif n_high >= 1 or n_mod >= 1:
        stress = "moderate"   # single signal = moderate, not high
    else:
        stress = "low"

    oil_regime = ("crash"  if oil < _th(th, "oil_wti_p25", 30.0) else
                  "spike"  if oil > _th(th, "oil_wti_p75", 75.0) else "neutral")

    usd_regime = ("strong" if usd > _th(th, "usd_broad_p75", 125.0) else
                  "weak"   if usd < _th(th, "usd_broad_p25", 110.0) else "neutral")

    return {
        "global_stress":  stress,
        "global_oil":     oil_regime,
        "global_usd":     usd_regime,
        "sig_ig_oas":     round(ig_oas, 3),
        "sig_vix":        round(vix, 2),
        "sig_oil_wti":    round(oil, 1),
        "sig_usd_broad":  round(usd, 1),
        "sig_nfci":       round(nfci_val, 3) if nfci_val is not None and not np.isnan(nfci_val) else float("nan"),
    }


def _classify_bloc_divergence(m: dict) -> dict:
    infl_spread  = m["infl_EA"]       - m["infl_US"]
    rate_spread  = m["short_rate_EA"] - m["short_rate_US"]
    slope_spread = m["term_slope_US"] - m["term_slope_EA"]

    infl_div  = ("ea_hotter"  if infl_spread  >  2.0 else
                 "us_hotter"  if infl_spread  < -2.0 else "aligned")
    rate_div  = ("ea_tighter" if rate_spread  >  1.5 else
                 "us_tighter" if rate_spread  < -1.5 else "aligned")
    curve_div = ("us_steeper" if slope_spread >  1.5 else
                 "ea_steeper" if slope_spread < -1.5 else "aligned")

    return {"bloc_infl_divergence": infl_div,
            "bloc_rate_divergence": rate_div,
            "bloc_curve_divergence": curve_div}


def _composite_label(us: dict, ea: dict, jp: dict,
                     glob: dict, div: dict) -> str:
    stress   = glob["global_stress"]
    growth   = us["us_growth"]
    infl     = us["us_inflation"]
    policy   = us["us_policy"]
    rate_div = div["bloc_rate_divergence"]
    infl_div = div["bloc_infl_divergence"]

    if stress == "high" and growth == "low":                           return "high_stress_defensive"
    if stress == "high":                                               return "risk_off_stress"
    if infl in ("high", "neutral") and policy == "tight":             return "higher_for_longer"
    if (rate_div != "aligned" or infl_div != "aligned") \
            and policy != "tight" and stress != "high":               return "bloc_divergence"
    if infl == "low" and policy == "easy" and growth in ("neutral", "high"): return "soft_landing"
    if infl in ("high", "neutral") and policy in ("neutral", "easy") and stress == "low": return "reflation_risk_on"
    if growth == "low" and infl == "low" and stress in ("low", "moderate"): return "disinflationary_slowdown"
    if stress == "moderate" and growth == "low":                       return "risk_off_stress"
    return "mixed_mid_cycle"


def score_regime_dimensions_v2(row: pd.Series,
                                thresholds: dict,
                                nfci_val: float | None = None) -> dict:
    """
    Classify a single already-sampled macro state row using all 19 dims + NFCI.

    nfci_val: the NFCI value at the anchor date (float or None).
              NFCI is an anchor-level signal — it doesn't vary per MALA sample
              because NFCI is not a dim in the macro state vector.
              It is used to calibrate the global stress classification.
    """
    m = {col: float(row[col]) for col in MACRO_STATE_COLS if col in row.index}

    us   = _classify_us_bloc(m, thresholds)
    ea   = _classify_ea_bloc(m, thresholds)
    jp   = _classify_jp_bloc(m, thresholds)
    glob = _classify_global_overlay(m, thresholds, nfci_val)
    div  = _classify_bloc_divergence(m)
    label = _composite_label(us, ea, jp, glob, div)

    stress    = glob["global_stress"]
    us_policy = us["us_policy"]
    fin_cond  = ("loose" if stress == "low" and us_policy == "easy" else
                 "tight" if stress in ("high", "moderate") or us_policy == "tight"
                 else "neutral")

    return {
        "regime_label":          label,
        # backward-compat aliases
        "dim_growth":            us["us_growth"],
        "dim_inflation":         us["us_inflation"],
        "dim_policy":            us["us_policy"],
        "dim_stress":            glob["global_stress"],
        "dim_fin_cond":          fin_cond,
        # US bloc
        "us_growth":             us["us_growth"],
        "us_inflation":          us["us_inflation"],
        "us_policy":             us["us_policy"],
        "us_curve":              us["us_curve"],
        # EA bloc
        "ea_growth":             ea["ea_growth"],
        "ea_inflation":          ea["ea_inflation"],
        "ea_policy":             ea["ea_policy"],
        "ea_curve":              ea["ea_curve"],
        # JP bloc
        "jp_inflation":          jp["jp_inflation"],
        "jp_policy":             jp["jp_policy"],
        "jp_curve":              jp["jp_curve"],
        # Global overlays
        "global_stress":         glob["global_stress"],
        "global_oil":            glob["global_oil"],
        "global_usd":            glob["global_usd"],
        # Cross-bloc divergence
        "bloc_infl_divergence":  div["bloc_infl_divergence"],
        "bloc_rate_divergence":  div["bloc_rate_divergence"],
        "bloc_curve_divergence": div["bloc_curve_divergence"],
        # Raw signals — all 19 dims
        "sig_ig_oas":            glob["sig_ig_oas"],
        "sig_vix":               glob["sig_vix"],
        "sig_nfci":              glob["sig_nfci"],
        "sig_oil_wti":           glob["sig_oil_wti"],
        "sig_usd_broad":         glob["sig_usd_broad"],
        "sig_infl_US":           round(m.get("infl_US", float("nan")), 3),
        "sig_infl_EA":           round(m.get("infl_EA", float("nan")), 3),
        "sig_infl_JP":           round(m.get("infl_JP", float("nan")), 3),
        "sig_short_rate_US":     round(m.get("short_rate_US", float("nan")), 3),
        "sig_short_rate_EA":     round(m.get("short_rate_EA", float("nan")), 3),
        "sig_short_rate_JP":     round(m.get("short_rate_JP", float("nan")), 3),
        "sig_us_real10y":        round(m.get("us_real10y", float("nan")), 3),
        "sig_unemp_US":          round(m.get("unemp_US", float("nan")), 3),
        "sig_unemp_EA":          round(m.get("unemp_EA", float("nan")), 3),
        "sig_term_slope_US":     round(m.get("term_slope_US", float("nan")), 3),
        "sig_term_slope_EA":     round(m.get("term_slope_EA", float("nan")), 3),
        "sig_term_slope_JP":     round(m.get("term_slope_JP", float("nan")), 3),
    }


# ---------------------------------------------------------------------------
# Bloc dims for transition tracking
# ---------------------------------------------------------------------------
BLOC_DIMS = [
    "us_growth", "us_inflation", "us_policy", "us_curve",
    "ea_growth", "ea_inflation", "ea_policy", "ea_curve",
    "jp_inflation", "jp_policy", "jp_curve",
    "global_stress", "global_oil", "global_usd",
    "bloc_infl_divergence", "bloc_rate_divergence", "bloc_curve_divergence",
]


def _reclassify_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (qid, anchor), grp in df.groupby(["question_id", "anchor_date"], sort=False):
        anchor_regime = grp["anchor_regime"].iloc[0]
        for idx, row in grp.iterrows():
            scenario_regime = row["regime_label"]
            transition = ("same_regime" if anchor_regime == scenario_regime
                          else f"{anchor_regime} -> {scenario_regime}")
            dim_changes = []
            for dim in BLOC_DIMS:
                anc_col = f"_anc_{dim}"
                if anc_col in row.index and dim in row.index:
                    if row[anc_col] != row[dim]:
                        dim_changes.append(f"{dim}: {row[anc_col]} -> {row[dim]}")
            rows.append({
                "idx": idx,
                "regime_transition": transition,
                "dim_changes": "; ".join(dim_changes) if dim_changes else "no_dimension_change",
            })
    trans_df = pd.DataFrame(rows).set_index("idx")
    df["regime_transition"] = trans_df["regime_transition"]
    df["dim_changes"]       = trans_df["dim_changes"]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reclassify(input_paths: list[Path],
               output_path: Path,
               nfci_path: Path | None = None) -> None:
    t0 = time.time()

    # ── Load input CSVs ───────────────────────────────────────────────────
    frames = []
    for p in input_paths:
        print(f"  Reading {p.name} ...")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    missing_macro = [c for c in MACRO_STATE_COLS if c not in df.columns]
    if missing_macro:
        raise ValueError(f"Input CSV missing macro state columns: {missing_macro}")

    # ── Load NFCI ─────────────────────────────────────────────────────────
    nfci_monthly = None
    nfci_path_used = nfci_path or NFCI_PATH
    if nfci_path_used.exists():
        print(f"\nLoading NFCI from {nfci_path_used.name} ...")
        nfci_monthly = load_nfci_monthly(nfci_path_used)
        print(f"  NFCI: {len(nfci_monthly)} monthly obs, "
              f"{nfci_monthly.index.min().date()} to {nfci_monthly.index.max().date()}")
    else:
        print(f"  WARNING: NFCI file not found at {nfci_path_used} — stress will use ig_oas+vix only")

    # ── Load thresholds ───────────────────────────────────────────────────
    thresholds = _load_thresholds(nfci_monthly)

    # ── Build anchor-level context (NFCI + recession) ─────────────────────
    print("\nBuilding anchor-level context (NFCI + NBER recession)...")
    anchor_context: dict[str, dict] = {}
    for anchor_str in sorted(df["anchor_date"].unique()):
        anchor_ts = pd.Timestamp(anchor_str)

        # NFCI at anchor date
        nfci_val = None
        if nfci_monthly is not None:
            closest_idx = np.argmin(np.abs(nfci_monthly.index - anchor_ts))
            nfci_val = float(nfci_monthly.iloc[closest_idx])

        # Recession context
        rec_ctx = recession_context(anchor_ts)

        anchor_context[anchor_str] = {
            "nfci_at_anchor":         round(nfci_val, 3) if nfci_val is not None else float("nan"),
            **rec_ctx,
        }
        nfci_str = f"{nfci_val:.3f}" if nfci_val is not None else "N/A"
        print(f"NFCI={nfci_str}, "
              f"recession_proximity={rec_ctx['recession_proximity']}, "
              f"months_since_rec_end={rec_ctx['months_since_rec_end']}")

    # ── Compute m0 anchor regime from feature_master ──────────────────────
    print("\nComputing anchor (m0) regimes from feature_master...")
    fm = pd.read_parquet(DATA_REFS / "feature_master_monthly.parquet")
    fm["month_end"] = pd.to_datetime(fm["month_end"])

    anchor_regime_map: dict[str, dict] = {}
    for anchor_str in sorted(df["anchor_date"].unique()):
        anchor_ts = pd.Timestamp(anchor_str)
        rows_fm = fm[fm["month_end"] == anchor_ts]
        if rows_fm.empty:
            available = pd.to_datetime(fm["month_end"].unique())
            closest = available[np.argmin(np.abs(available - anchor_ts))]
            rows_fm = fm[fm["month_end"] == closest]
        m0_series = pd.Series({col: float(rows_fm.iloc[0][col])
                                for col in MACRO_STATE_COLS if col in rows_fm.columns})
        nfci_val = anchor_context[anchor_str]["nfci_at_anchor"]
        anchor_regime_map[anchor_str] = score_regime_dimensions_v2(
            m0_series, thresholds, nfci_val=nfci_val
        )
        print(f"  {anchor_str}: anchor_regime = {anchor_regime_map[anchor_str]['regime_label']}")

    # ── Reclassify all rows ───────────────────────────────────────────────
    print(f"\nReclassifying {len(df):,} rows...")
    new_regime_records = []
    for i, (_, row) in enumerate(df.iterrows()):
        nfci_val = anchor_context.get(str(row["anchor_date"]), {}).get("nfci_at_anchor")
        rec = score_regime_dimensions_v2(row, thresholds, nfci_val=nfci_val)
        new_regime_records.append(rec)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{len(df):,} rows  ({time.time()-t0:.1f}s)")

    new_regime_df = pd.DataFrame(new_regime_records, index=df.index)

    # ── Attach anchor-level context columns ───────────────────────────────
    for col in ["nfci_at_anchor", "in_recession", "months_since_rec_end",
                "recession_proximity", "last_recession_peak", "last_recession_trough"]:
        df[col] = df["anchor_date"].map({k: v[col] for k, v in anchor_context.items()})

    df["anchor_regime"] = df["anchor_date"].map(
        {k: v["regime_label"] for k, v in anchor_regime_map.items()}
    )

    # Attach anchor bloc dims as temp cols for transition computation
    for dim in BLOC_DIMS:
        df[f"_anc_{dim}"] = df["anchor_date"].map(
            {k: v.get(dim, "") for k, v in anchor_regime_map.items()}
        )

    # Merge new regime cols
    for col in new_regime_df.columns:
        df[col] = new_regime_df[col].values

    # ── Recompute transitions ─────────────────────────────────────────────
    print("\nRecomputing regime transitions...")
    df = _reclassify_transitions(df)

    # Drop temp anchor bloc cols
    df = df.drop(columns=[f"_anc_{dim}" for dim in BLOC_DIMS], errors="ignore")

    # ── Save ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"   Output : {output_path}")
    print(f"   Rows   : {len(df):,}  |  Columns: {df.shape[1]}")

    print(f"\nNew regime_label distribution:")
    print(df["regime_label"].value_counts().to_string())

    print(f"\nRecession proximity at anchor dates:")
    print(df[["anchor_date","recession_proximity","months_since_rec_end","nfci_at_anchor"]]
          .drop_duplicates("anchor_date").sort_values("anchor_date").to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Post-hoc regime reclassification — all 19 macro dims + NFCI + NBER."
    )
    parser.add_argument("--input",  nargs="+", required=True,
                        help="One or more input scenario_results CSV paths")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--nfci",   default=None,
                        help="Path to NFCI CSV (default: workspace_v4/NFCI (1).csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)
    nfci_path   = Path(args.nfci) if args.nfci else None

    for p in input_paths:
        if not p.exists():
            print(f"ERROR: input file not found: {p}")
            sys.exit(1)

    reclassify(input_paths, output_path, nfci_path)
