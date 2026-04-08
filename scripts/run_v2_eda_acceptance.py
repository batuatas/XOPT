#!/usr/bin/env python3
"""Entrypoint for the XOPTPOE v2 long-horizon acceptance / EDA audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xoptpoe_v2_data.acceptance import build_v2_acceptance_audit



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the XOPTPOE v2_long_horizon acceptance / EDA audit")
    parser.add_argument("--project-root", default=str(ROOT), help="Repository root path")
    args = parser.parse_args()
    build_v2_acceptance_audit(project_root=Path(args.project_root))


if __name__ == "__main__":
    main()
