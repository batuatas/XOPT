"""I/O helpers for the XOPTPOE v2 modeling-preparation package."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd



def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet table from disk."""
    return pd.read_parquet(path)



def load_csv(path: Path, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    """Load a CSV with optional date parsing."""
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)



def write_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write parquet output, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)



def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write CSV output, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)



def write_text(text: str, path: Path) -> None:
    """Write UTF-8 text output, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
