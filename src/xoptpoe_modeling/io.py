"""I/O helpers for modeling-preparation artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_modeling_panel(path: Path) -> pd.DataFrame:
    """Load frozen modeling panel with month-end date parsing."""
    return pd.read_csv(path, parse_dates=["month_end"])


def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write CSV output, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def write_text(text: str, path: Path) -> None:
    """Write UTF-8 text output, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
