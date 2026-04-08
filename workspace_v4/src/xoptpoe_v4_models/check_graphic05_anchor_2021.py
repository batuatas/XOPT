#!/usr/bin/env python3
"""
check_graphic05_anchor_2021.py  —  v2

Goal:
1. Reconstruct the exact 2021-12-31 benchmark anchor from the graphic05 path.
2. Print the sleeve predictions at the anchor.
3. Print the portfolio weights for every allocator-refinement config.
4. Confirm which config gives 0% gold (or near-zero gold).

Benchmark path reproduced:
- BEST_60_EXPERIMENT selected params from v4_prediction_benchmark_metrics.csv
- cutoff = anchor - MonthEnd(60) = 2016-12-31
- Train elastic_net on data with month_end <= cutoff, horizon=60
- Score at anchor date
- Covariance from ret_1m_lag in feature_master (fallback; exact path needs
  data/final_v4_expanded_universe/ which is absent in this workspace)

Run:
    cd workspace_v4
    PYTHONPATH=src python src/xoptpoe_v4_models/check_graphic05_anchor_2021.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure workspace_v4/src is importable when run as a standalone script
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent  # workspace_v4/src
for _p in [str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xoptpoe_v4_modeling.features import feature_columns_for_set
from xoptpoe_v4_models.data import SLEEVE_ORDER
from xoptpoe_v4_models.optim_layers import RiskConfig, RobustOptimizerCache, build_sigma_map
from xoptpoe_v4_scenario.pipeline import refit_elastic_net
from xoptpoe_v4_models.allocator_refinement import allocator_refinement_grid

BEST_60_EXPERIMENT = "elastic_net__core_plus_interactions__separate_60"


def normalize_long_only(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return np.repeat(1.0 / len(w), len(w))
    return w / s


def main() -> None:
    # __file__ is workspace_v4/src/xoptpoe_v4_models/check_graphic05_anchor_2021.py
    # parents[2] is workspace_v4/
    project_root = Path(__file__).resolve().parents[2]
    data_refs    = project_root / "data_refs"
    reports_root = project_root / "reports"

    anchor = pd.Timestamp("2021-12-31")
    cutoff = anchor - pd.offsets.MonthEnd(60)

    print("=" * 80)
    print(f"Reconstructing graphic05 benchmark anchor for {anchor.date()}")
    print(f"Benchmark train cutoff = {cutoff.date()}")
    print("=" * 80)

    # 1) Selected predictor params from benchmark report
    metrics_path = reports_root / "benchmark/v4_prediction_benchmark_metrics.csv"
    pred_metrics = pd.read_csv(metrics_path)
    row = pred_metrics.loc[pred_metrics["experiment_name"].eq(BEST_60_EXPERIMENT)]
    if row.empty:
        raise RuntimeError(
            f"Could not find {BEST_60_EXPERIMENT} in {metrics_path}"
        )
    params = ast.literal_eval(str(row["selected_params"].iloc[0]))

    print(f"BEST_60_EXPERIMENT = {BEST_60_EXPERIMENT}")
    print(f"selected_params    = {params}")
    print()

    # 2) Load modeling panel and feature manifest from data_refs
    full_panel = pd.read_parquet(data_refs / "modeling_panel_hstack.parquet")
    full_panel["month_end"] = pd.to_datetime(full_panel["month_end"])
    full_panel = full_panel.loc[full_panel["sleeve_id"].isin(SLEEVE_ORDER)].copy()

    feature_manifest = pd.read_csv(
        data_refs / "feature_set_manifest.csv",
        parse_dates=["first_valid_date", "last_valid_date"],
    )
    feature_columns = feature_columns_for_set(feature_manifest, "core_plus_interactions")

    train_pool = full_panel.loc[
        full_panel["horizon_months"].eq(60)
        & full_panel["baseline_trainable_flag"].eq(1)
        & full_panel["target_available_flag"].eq(1)
        & full_panel["annualized_excess_forward_return"].notna()
    ].copy()

    score_pool = full_panel.loc[full_panel["horizon_months"].eq(60)].copy()

    train_df = train_pool.loc[train_pool["month_end"].le(cutoff)].copy()
    score_df = score_pool.loc[score_pool["month_end"].eq(anchor)].copy()

    if score_df.empty:
        raise RuntimeError(f"No score rows found for anchor={anchor.date()}")
    if len(train_df) < len(SLEEVE_ORDER) * 12:
        raise RuntimeError(
            f"Not enough training rows before cutoff={cutoff.date()} "
            f"(got {len(train_df)}, need >= {len(SLEEVE_ORDER) * 12})"
        )

    ordered = score_df.set_index("sleeve_id").reindex(list(SLEEVE_ORDER)).reset_index()
    if ordered["sleeve_id"].isna().any():
        raise RuntimeError("Anchor score_df could not be aligned to SLEEVE_ORDER")

    print(f"Training rows: {len(train_df)}  |  Score rows: {len(score_df)}")
    print()

    # 3) Refit benchmark elastic net and score at anchor
    #    Uses pipeline.refit_elastic_net (same hyperparams, same preprocessor logic)
    en_model, preprocessor = refit_elastic_net(
        modeling_panel=train_pool,  # pass the full filtered pool; refit_elastic_net filters by cutoff
        feature_columns=feature_columns,
        feature_manifest=feature_manifest,
        train_end=cutoff,
        alpha=float(params["alpha"]),
        l1_ratio=float(params["l1_ratio"]),
    )

    X_score_raw = ordered[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    y_pred = en_model.predict(preprocessor.transform_numpy(X_score_raw))
    ordered = ordered.copy()
    ordered["y_pred"] = y_pred

    print("Predicted annualized excess returns at 2021-12-31:")
    pred_tbl = (
        ordered[["sleeve_id", "y_pred"]]
        .sort_values("y_pred", ascending=False)
        .reset_index(drop=True)
    )
    print(pred_tbl.to_string(index=False))
    print()

    gold_pred = float(pred_tbl.loc[pred_tbl["sleeve_id"].eq("ALT_GLD"), "y_pred"].iloc[0])
    gold_rank = int(pred_tbl.index[pred_tbl["sleeve_id"].eq("ALT_GLD")][0]) + 1
    print(f"ALT_GLD predicted excess return: {gold_pred:.6f}")
    print(f"ALT_GLD rank among sleeves      : {gold_rank}/{len(pred_tbl)}")
    print()

    # 4) Covariance: use ret_1m_lag from feature_master (fallback path)
    #    Note: exact benchmark uses realized excess returns from price files
    #    (data/final_v4_expanded_universe/target_raw_direct.csv + euro_synth +
    #    tb3ms_monthly.csv), which are absent in this workspace.
    #    ret_1m_lag is a close approximation but allocator weights may differ
    #    slightly from graphic05.
    print("Covariance source: ret_1m_lag from feature_master (fallback)")
    print("  (exact benchmark uses realized excess from price files; see comment)")
    print()

    fm = pd.read_parquet(data_refs / "feature_master_monthly.parquet")
    fm["month_end"] = pd.to_datetime(fm["month_end"])
    cov_panel = (
        fm[fm["sleeve_id"].isin(SLEEVE_ORDER)]
        .pivot_table(index="month_end", columns="sleeve_id", values="ret_1m_lag", aggfunc="first")
        .sort_index()
        .reindex(columns=list(SLEEVE_ORDER))
    )

    sigma_map = build_sigma_map(
        [anchor], excess_history=cov_panel, risk_config=RiskConfig()
    )
    optimizer_cache = RobustOptimizerCache(sigma_by_month=sigma_map)

    print("=" * 80)
    print("Allocator refinement grid at 2021-12-31")
    print("=" * 80)

    rows = []
    mu = ordered.set_index("sleeve_id").reindex(list(SLEEVE_ORDER))["y_pred"].to_numpy(dtype=float)

    for cfg in allocator_refinement_grid():
        w = normalize_long_only(optimizer_cache.solve(anchor, mu, cfg))
        w_series = pd.Series(w, index=list(SLEEVE_ORDER)).sort_values(ascending=False)

        rows.append({
            "config_label": f"lam{cfg.lambda_risk:g}_kap{cfg.kappa:g}_{cfg.omega_type}",
            "lambda_risk":  float(cfg.lambda_risk),
            "kappa":        float(cfg.kappa),
            "omega_type":   str(cfg.omega_type),
            "w_ALT_GLD":    float(w_series["ALT_GLD"]),
            "top_sleeve":   str(w_series.index[0]),
            "top_weight":   float(w_series.iloc[0]),
            "effective_n":  float(1.0 / np.square(w).sum()),
        })

        print(f"[{rows[-1]['config_label']}]")
        print(w_series.round(6).to_string())
        print("-" * 80)

    out = (
        pd.DataFrame(rows)
        .sort_values(["w_ALT_GLD", "effective_n", "config_label"])
        .reset_index(drop=True)
    )

    print()
    print("=" * 80)
    print("Summary sorted by gold weight  (covariance: ret_1m_lag fallback)")
    print("=" * 80)
    print(out.to_string(index=False))

    near_zero = out.loc[out["w_ALT_GLD"] <= 1e-4].copy()
    if near_zero.empty:
        print(
            "\nNo exact zero-gold config found in allocator_refinement_grid() "
            "with this covariance path."
        )
        print(
            "If the benchmark uses the same predictor but the exact realized-excess "
            "covariance, the result may differ. Provide "
            "data/final_v4_expanded_universe/target_raw_direct.csv for an exact check."
        )
    else:
        print("\nConfigs with ~0 gold (benchmark-aligned predictor, fallback cov):")
        print(
            near_zero[
                ["config_label", "w_ALT_GLD", "top_sleeve", "top_weight", "effective_n"]
            ].to_string(index=False)
        )

    # Print the locked benchmark config specifically
    locked_label = "lam8_kap0.1_identity"
    locked_row = out.loc[out["config_label"].eq(locked_label)]
    print()
    print("=" * 80)
    print(f"Locked benchmark config ({locked_label}):")
    if not locked_row.empty:
        r = locked_row.iloc[0]
        print(f"  w_ALT_GLD = {r['w_ALT_GLD']:.6f}")
        print(f"  top_sleeve = {r['top_sleeve']}  ({r['top_weight']:.4f})")
        print(f"  effective_n = {r['effective_n']:.2f}")
        gld_near_zero = r["w_ALT_GLD"] < 0.01
        print(f"  gold near-zero: {gld_near_zero}")
        if gld_near_zero:
            print("  => Matches graphic05 near-0% gold behavior.")
        else:
            print(
                f"  => Gold weight = {r['w_ALT_GLD']:.1%}. "
                "If this is >0, the covariance fallback may be shifting the solution. "
                "Provide exact price data for a definitive check."
            )
    else:
        print("  Config not found in grid (check LAMBDA_RISK / KAPPA naming).")


if __name__ == "__main__":
    main()
