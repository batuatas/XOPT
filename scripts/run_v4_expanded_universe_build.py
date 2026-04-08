#!/usr/bin/env python3
"""Build the versioned v4 expanded-universe dataset."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v4_data import build_v4_expanded_universe_dataset


if __name__ == "__main__":
    build_v4_expanded_universe_dataset(PROJECT_ROOT)
