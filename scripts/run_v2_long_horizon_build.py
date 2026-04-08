#!/usr/bin/env python3
"""Entrypoint for the XOPTPOE v2 long-horizon data build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xoptpoe_v2_data.build import build_v2_long_horizon_dataset



def main() -> None:
    parser = argparse.ArgumentParser(description="Build the XOPTPOE v2 long-horizon dataset")
    parser.add_argument("--project-root", default=str(ROOT), help="Repository root path")
    args = parser.parse_args()
    build_v2_long_horizon_dataset(project_root=Path(args.project_root))


if __name__ == "__main__":
    main()
