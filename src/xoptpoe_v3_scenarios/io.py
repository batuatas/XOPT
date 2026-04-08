"""Filesystem and artifact loading helpers for the v3 scenario scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ScenarioPaths:
    """Active v3 filesystem layout for scenario-generation work."""

    project_root: Path
    data_root: Path
    modeling_root: Path
    reports_root: Path
    modeling_panel_hstack: Path
    feature_master_monthly: Path
    target_panel_long_horizon: Path
    modeling_panel_firstpass: Path
    split_manifest: Path
    feature_manifest: Path
    benchmark_manifest: Path
    benchmark_stack: Path
    prediction_metrics: Path
    prediction_by_sleeve: Path
    portfolio_metrics: Path
    portfolio_by_sleeve: Path
    portfolio_attribution: Path
    portfolio_returns: Path


def default_paths(project_root: Path) -> ScenarioPaths:
    """Resolve the active v3 path layout."""
    root = project_root.resolve()
    modeling_root = root / "data" / "modeling_v3"
    reports_root = root / "reports" / "v3_long_horizon_china"
    benchmark_manifest = modeling_root / "final_benchmark_manifest_v3.csv"
    if not benchmark_manifest.exists():
        alt = reports_root / "final_benchmark_manifest_v3.csv"
        benchmark_manifest = alt
    return ScenarioPaths(
        project_root=root,
        data_root=root / "data" / "final_v3_long_horizon_china",
        modeling_root=modeling_root,
        reports_root=reports_root,
        modeling_panel_hstack=root / "data" / "final_v3_long_horizon_china" / "modeling_panel_hstack.parquet",
        feature_master_monthly=root / "data" / "final_v3_long_horizon_china" / "feature_master_monthly.parquet",
        target_panel_long_horizon=root / "data" / "final_v3_long_horizon_china" / "target_panel_long_horizon.parquet",
        modeling_panel_firstpass=modeling_root / "modeling_panel_firstpass.parquet",
        split_manifest=modeling_root / "split_manifest.csv",
        feature_manifest=modeling_root / "feature_set_manifest.csv",
        benchmark_manifest=benchmark_manifest,
        benchmark_stack=reports_root / "final_benchmark_stack_v3.csv",
        prediction_metrics=reports_root / "prediction_benchmark_v3_metrics.csv",
        prediction_by_sleeve=reports_root / "prediction_benchmark_v3_by_sleeve.csv",
        portfolio_metrics=reports_root / "portfolio_benchmark_v3_metrics.csv",
        portfolio_by_sleeve=reports_root / "portfolio_benchmark_v3_by_sleeve.csv",
        portfolio_attribution=reports_root / "portfolio_benchmark_v3_attribution.csv",
        portfolio_returns=modeling_root / "portfolio_benchmark_v3_returns.parquet",
    )


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Thin wrapper for CSV loading with a clear path-based error."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def load_parquet(path: Path, **kwargs) -> pd.DataFrame:
    """Thin wrapper for parquet loading with a clear path-based error."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, **kwargs)


def load_active_artifacts(paths: ScenarioPaths) -> dict[str, pd.DataFrame]:
    """Load the active v3 artifacts needed by the scaffold."""
    return {
        "modeling_panel_hstack": load_parquet(paths.modeling_panel_hstack),
        "feature_master_monthly": load_parquet(paths.feature_master_monthly),
        "target_panel_long_horizon": load_parquet(paths.target_panel_long_horizon),
        "modeling_panel_firstpass": load_parquet(paths.modeling_panel_firstpass),
        "split_manifest": load_csv(paths.split_manifest, parse_dates=["start_month_end", "end_month_end"]),
        "feature_manifest": load_csv(paths.feature_manifest, parse_dates=["first_valid_date", "last_valid_date"]),
        "benchmark_manifest": load_csv(paths.benchmark_manifest),
        "benchmark_stack": load_csv(paths.benchmark_stack),
        "prediction_metrics": load_csv(paths.prediction_metrics),
        "prediction_by_sleeve": load_csv(paths.prediction_by_sleeve),
        "portfolio_metrics": load_csv(paths.portfolio_metrics),
        "portfolio_by_sleeve": load_csv(paths.portfolio_by_sleeve),
        "portfolio_attribution": load_csv(paths.portfolio_attribution),
        "portfolio_returns": load_parquet(paths.portfolio_returns),
    }
