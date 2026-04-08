"""I/O helpers for loading manifests and writing pipeline tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from xoptpoe_data.config import (
    ASSET_MASTER_SEED_FILE,
    MACRO_MANIFEST_FILE,
    SOURCE_MASTER_SEED_FILE,
    TARGET_MANIFEST_FILE,
    BuildPaths,
)


def load_csv(path: Path, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    """Load CSV with optional date parsing."""
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Persist dataframe as CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def write_json(payload: dict, path: Path) -> None:
    """Persist JSON payload with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    """Persist plain text content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_target_manifest(paths: BuildPaths) -> pd.DataFrame:
    """Load target series manifest."""
    return load_csv(paths.config_dir / TARGET_MANIFEST_FILE, parse_dates=["start_date_target"])


def load_macro_manifest(paths: BuildPaths) -> pd.DataFrame:
    """Load macro series manifest."""
    return load_csv(paths.config_dir / MACRO_MANIFEST_FILE)


def load_asset_master_seed(paths: BuildPaths) -> pd.DataFrame:
    """Load locked asset master seed."""
    return load_csv(paths.config_dir / ASSET_MASTER_SEED_FILE, parse_dates=["start_date_target"])


def load_source_master_seed(paths: BuildPaths) -> pd.DataFrame:
    """Load source master seed."""
    return load_csv(paths.config_dir / SOURCE_MASTER_SEED_FILE)


def validate_locked_asset_master(asset_master: pd.DataFrame) -> None:
    """Enforce locked sleeve constraints before build starts."""
    required_cols = {
        "sleeve_id",
        "ticker",
        "target_currency",
        "proxy_flag",
        "locked_flag",
        "start_date_target",
    }
    missing = required_cols - set(asset_master.columns)
    if missing:
        raise ValueError(f"asset_master_seed missing required columns: {sorted(missing)}")

    if asset_master["sleeve_id"].duplicated().any():
        raise ValueError("asset_master_seed has duplicate sleeve_id values")

    if asset_master["ticker"].duplicated().any():
        raise ValueError("asset_master_seed has duplicate ticker values")

    if (asset_master["target_currency"] != "USD").any():
        raise ValueError("asset_master_seed must have USD target_currency for all rows")

    if (asset_master["proxy_flag"] != 1).any():
        raise ValueError("asset_master_seed must have proxy_flag=1 for all rows")

    if len(asset_master) != 8:
        raise ValueError(f"asset_master_seed must have exactly 8 rows, found {len(asset_master)}")
