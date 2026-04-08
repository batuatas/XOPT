"""
check_rf_sources.py

Run this locally to confirm:
1. What rf columns exist in modeling_panel_hstack and target_panel_long_horizon
2. What short_rate_US actually is (units, series)
3. Whether annualized_rf_forward_return is available and in what units

python check_rf_sources.py
"""
import pandas as pd
import numpy as np

DATA = "/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs"

ANCHOR_DATES = ['2021-12-31', '2022-12-31', '2023-12-31', '2024-12-31']

# ── 1. feature_master_monthly ────────────────────────────────────────────────
print("=" * 60)
print("1. FEATURE MASTER — short_rate_US and rf-related columns")
print("=" * 60)

fm = pd.read_parquet(f"{DATA}/feature_master_monthly.parquet")
fm["month_end"] = pd.to_datetime(fm["month_end"])
fm_one = fm.drop_duplicates("month_end").sort_values("month_end")

# Find all rf/rate related columns
rf_cols_fm = [c for c in fm.columns if any(x in c.lower() for x in
              ['rf', 'risk_free', 'tb3', 'tbill', 't_bill', 'short_rate'])]
print(f"\nRF-related columns in feature_master: {rf_cols_fm}\n")

# Show short_rate_US at anchor dates
if "short_rate_US" in fm.columns:
    print("short_rate_US at anchor dates:")
    for d in ANCHOR_DATES:
        row = fm_one[fm_one["month_end"] == pd.Timestamp(d)]
        if not row.empty:
            val = float(row["short_rate_US"].iloc[0])
            print(f"  {d}: {val:.6f}")
    print()
    print("short_rate_US last 12 months:")
    print(fm_one[["month_end", "short_rate_US"]].tail(12).to_string(index=False))

# ── 2. target_panel_long_horizon ─────────────────────────────────────────────
print()
print("=" * 60)
print("2. TARGET PANEL — rf forward return columns")
print("=" * 60)

tp = pd.read_parquet(f"{DATA}/target_panel_long_horizon.parquet")
tp["month_end"] = pd.to_datetime(tp["month_end"])

rf_cols_tp = [c for c in tp.columns if any(x in c.lower() for x in
              ['rf', 'risk_free', 'tb3', 'tbill', 'total', 'excess'])]
print(f"\nReturn-related columns in target_panel: {rf_cols_tp}\n")

# Show sample values at 60m horizon for anchor dates
tp_60 = tp[tp["horizon_months"] == 60].copy()
key_cols = [c for c in ["sleeve_id", "month_end",
                         "annualized_rf_forward_return",
                         "annualized_total_forward_return",
                         "annualized_excess_forward_return"]
            if c in tp.columns]

print(f"Columns available: {key_cols}\n")

if "annualized_rf_forward_return" in tp.columns:
    print("annualized_rf_forward_return at anchor dates (EQ_US, 60m):")
    for d in ANCHOR_DATES:
        row = tp_60[(tp_60["month_end"] == pd.Timestamp(d)) &
                    (tp_60["sleeve_id"] == "EQ_US")]
        if not row.empty:
            rf_val  = row["annualized_rf_forward_return"].iloc[0]
            tot_val = row["annualized_total_forward_return"].iloc[0] \
                      if "annualized_total_forward_return" in row.columns else None
            exc_val = row["annualized_excess_forward_return"].iloc[0] \
                      if "annualized_excess_forward_return" in row.columns else None
            print(f"  {d}: rf={rf_val:.6f}  total={tot_val:.6f}  excess={exc_val:.6f}")
        else:
            print(f"  {d}: no row found")
    print()
    # Check units: rf should be ~0.05 (decimal) or ~5.0 (percent)
    rf_sample = tp_60["annualized_rf_forward_return"].dropna()
    print(f"annualized_rf_forward_return stats:")
    print(f"  min={rf_sample.min():.6f}  max={rf_sample.max():.6f}  "
          f"mean={rf_sample.mean():.6f}  median={rf_sample.median():.6f}")
    print(f"  -> If median ~0.02-0.05: DECIMAL units")
    print(f"  -> If median ~2-5:       PERCENT units")

# ── 3. modeling_panel_hstack ─────────────────────────────────────────────────
print()
print("=" * 60)
print("3. MODELING PANEL — rf columns")
print("=" * 60)

mp = pd.read_parquet(f"{DATA}/modeling_panel_hstack.parquet")
mp["month_end"] = pd.to_datetime(mp["month_end"])

rf_cols_mp = [c for c in mp.columns if any(x in c.lower() for x in
              ['rf', 'risk_free', 'tb3', 'tbill', 'total', 'excess', 'short_rate'])]
print(f"\nRF/return-related columns in modeling_panel: {rf_cols_mp}\n")

if "annualized_rf_forward_return" in mp.columns:
    mp_60 = mp[mp["horizon_months"] == 60]
    print("annualized_rf_forward_return at anchor dates (EQ_US, 60m):")
    for d in ANCHOR_DATES:
        row = mp_60[(mp_60["month_end"] == pd.Timestamp(d)) &
                    (mp_60["sleeve_id"] == "EQ_US")]
        if not row.empty:
            val = float(row["annualized_rf_forward_return"].iloc[0])
            print(f"  {d}: {val:.6f}")

# ── 4. Cross-check: short_rate_US vs annualized_rf ───────────────────────────
print()
print("=" * 60)
print("4. CROSS-CHECK: short_rate_US vs annualized_rf_forward_return")
print("=" * 60)

if "annualized_rf_forward_return" in tp.columns and "short_rate_US" in fm.columns:
    # Merge on month_end for EQ_US 60m
    tp_eq = tp_60[tp_60["sleeve_id"] == "EQ_US"][
        ["month_end", "annualized_rf_forward_return"]
    ].copy()
    fm_sr = fm_one[["month_end", "short_rate_US"]].copy()
    merged = tp_eq.merge(fm_sr, on="month_end", how="inner")
    merged = merged.dropna()

    print(f"\nCorrelation: {merged['annualized_rf_forward_return'].corr(merged['short_rate_US']):.4f}")
    print(f"\nSample (last 8 rows):")
    print(merged.tail(8).to_string(index=False))
    print()
    print("If short_rate_US is in % and rf_forward is in decimal:")
    merged["sr_decimal"] = merged["short_rate_US"] / 100.0
    print(f"  Correlation after /100: "
          f"{merged['annualized_rf_forward_return'].corr(merged['sr_decimal']):.4f}")
    print(f"  Mean diff (rf_fwd - sr/100): "
          f"{(merged['annualized_rf_forward_return'] - merged['sr_decimal']).mean():.6f}")
