"""Configuration for the XOPTPOE v2 long-horizon data build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xoptpoe_data.config import LOCKED_SLEEVES, LOCAL_BLOCK_BY_SLEEVE, NO_LOCAL_BLOCK_SLEEVES

V2_VERSION = "v2_long_horizon"
LOCKED_HORIZONS = (60, 120, 180)

ASSET_CLASS_GROUP_BY_SLEEVE: dict[str, str] = {
    "EQ_US": "equity",
    "EQ_EZ": "equity",
    "EQ_JP": "equity",
    "EQ_EM": "equity",
    "FI_UST": "fixed_income",
    "FI_IG": "fixed_income",
    "ALT_GLD": "alternative",
    "RE_US": "real_asset",
}

EXPOSURE_REGION_BY_SLEEVE: dict[str, str] = {
    "EQ_US": "US",
    "EQ_EZ": "EURO_AREA",
    "EQ_JP": "JAPAN",
    "EQ_EM": "EM_GLOBAL",
    "FI_UST": "US",
    "FI_IG": "US",
    "ALT_GLD": "GLOBAL",
    "RE_US": "US",
}


@dataclass(frozen=True)
class V2Paths:
    """Filesystem layout for the v2 dataset build."""

    project_root: Path
    data_intermediate_dir: Path
    data_final_dir: Path
    data_final_v2_dir: Path
    data_external_dir: Path
    reports_dir: Path
    reports_v2_dir: Path

    def ensure_directories(self) -> None:
        for path in (self.data_final_v2_dir, self.reports_v2_dir):
            path.mkdir(parents=True, exist_ok=True)



def default_project_root() -> Path:
    """Infer the repository root from the installed package path."""
    return Path(__file__).resolve().parents[2]



def default_paths(project_root: Path | None = None) -> V2Paths:
    """Construct default v2 path configuration."""
    root = project_root or default_project_root()
    return V2Paths(
        project_root=root,
        data_intermediate_dir=root / "data" / "intermediate",
        data_final_dir=root / "data" / "final",
        data_final_v2_dir=root / "data" / "final_v2_long_horizon",
        data_external_dir=root / "data" / "external" / "akif_candidates",
        reports_dir=root / "reports",
        reports_v2_dir=root / "reports" / "v2_long_horizon",
    )
