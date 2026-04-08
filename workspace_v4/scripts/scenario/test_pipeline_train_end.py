#!/usr/bin/env python3
"""
Quick diagnostic: does build_pipeline_at_date actually use train_end?
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
REPO_SRC  = WORKSPACE.parent.parent / "src"

for p in [str(WORKSPACE / "src"), str(REPO_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from xoptpoe_v4_scenario.pipeline import build_pipeline_at_date, SLEEVES_14
from xoptpoe_v4_scenario.state_space import load_state

DATA_REFS = WORKSPACE / "data_refs"

# Load data
fm  = pd.read_parquet(DATA_REFS / "feature_master_monthly.parquet")
mp  = pd.read_parquet(DATA_REFS / "modeling_panel_hstack.parquet")
fsm = pd.read_csv(DATA_REFS / "feature_set_manifest.csv")
tp  = pd.read_parquet(DATA_REFS / "target_panel_long_horizon.parquet")

for df in [fm, mp, tp]:
    df["month_end"] = pd.to_datetime(df["month_end"])

feat_cols = fsm[fsm["include_core_plus_interactions"] == 1]["feature_name"].tolist()

excess_ret = fm.pivot_table(index="month_end", columns="sleeve_id",
                             values="ret_1m_lag", aggfunc="first")
avail = [s for s in SLEEVES_14 if s in excess_ret.columns]
excess_ret = excess_ret[avail].sort_index()

anchor = pd.Timestamp("2021-12-31")
m0, _ = load_state(anchor, fm, SLEEVES_14, modeling_panel=mp, horizon_months=60)

print("="*70)
print("TEST: Does build_pipeline_at_date use train_end parameter?")
print("="*70)
print()

# Test 1: train_end = 2021-02-28 (original, should give 8.1% gold)
print("Test 1: train_end = 2021-02-28 (original frozen model)")
try:
    p1 = build_pipeline_at_date(
        anchor_date=anchor,
        feature_master=fm,
        modeling_panel=mp,
        feature_manifest=fsm,
        feature_columns=feat_cols,
        excess_returns_monthly=excess_ret,
        elastic_net_alpha=0.005,
        elastic_net_l1=0.5,
        train_end=pd.Timestamp("2021-02-28"),
        sleeve_order=SLEEVES_14,
    )
    ev1 = p1.evaluate_at(m0)
    w1_gld = float(ev1["w"][SLEEVES_14.index("ALT_GLD")])
    print(f"  Result: w_ALT_GLD = {w1_gld*100:.1f}%")
except Exception as e:
    print(f"  ERROR: {e}")
    w1_gld = None

print()

# Test 2: train_end = 2021-11-30 (walk-forward, should give ~0% gold)
print("Test 2: train_end = 2021-11-30 (walk-forward model)")
try:
    p2 = build_pipeline_at_date(
        anchor_date=anchor,
        feature_master=fm,
        modeling_panel=mp,
        feature_manifest=fsm,
        feature_columns=feat_cols,
        excess_returns_monthly=excess_ret,
        elastic_net_alpha=0.005,
        elastic_net_l1=0.5,
        train_end=pd.Timestamp("2021-11-30"),
        sleeve_order=SLEEVES_14,
    )
    ev2 = p2.evaluate_at(m0)
    w2_gld = float(ev2["w"][SLEEVES_14.index("ALT_GLD")])
    print(f"  Result: w_ALT_GLD = {w2_gld*100:.1f}%")
except Exception as e:
    print(f"  ERROR: {e}")
    w2_gld = None

print()
print("="*70)
print("INTERPRETATION:")
print("="*70)

if w1_gld is not None and w2_gld is not None:
    if abs(w1_gld - w2_gld) < 0.001:
        print(f"❌ PROBLEM: Both train_end values give the SAME gold weight ({w1_gld*100:.1f}%)")
        print("   This means build_pipeline_at_date is IGNORING the train_end parameter.")
        print()
        print("   Possible causes:")
        print("   1. train_end parameter is not passed to the elastic net fit")
        print("   2. The pipeline is using a cached model")
        print("   3. build_pipeline_at_date doesn't support per-anchor training")
        print()
        print("   ACTION: Check the build_pipeline_at_date() source code.")
        print("   Look for where elastic net is fitted and ensure train_end is used.")
    else:
        print(f"✅ GOOD: Different train_end values give different results:")
        print(f"   train_end=2021-02-28: w_ALT_GLD = {w1_gld*100:.1f}%")
        print(f"   train_end=2021-11-30: w_ALT_GLD = {w2_gld*100:.1f}%")
        print()
        print(f"   Delta: {(w2_gld - w1_gld)*100:.1f}%")
        if w2_gld < w1_gld:
            print("   ✓ Gold weight DECREASED with more training data (expected)")
        else:
            print("   ⚠ Gold weight INCREASED with more training data (unexpected)")
else:
    print("Could not run both tests due to errors.")
