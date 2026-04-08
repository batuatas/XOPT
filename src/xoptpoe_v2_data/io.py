"""I/O helpers for the v2 long-horizon data build."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd



def load_csv(path: Path, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    """Load a CSV with optional date parsing."""
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)



def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write CSV, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)



def write_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write parquet, raising a clear error if no engine is available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=index)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support is required for v2_long_horizon outputs. Install pyarrow or fastparquet."
        ) from exc



def write_text(text: str, path: Path) -> None:
    """Write UTF-8 text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
