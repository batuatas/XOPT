"""Configuration for the XOPTPOE v3 China-sleeve long-horizon build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xoptpoe_v2_data.config import LOCKED_HORIZONS

V3_VERSION = "v3_long_horizon_china"
CHINA_SLEEVE_ID = "EQ_CN"
CHINA_TICKER = "FXI"
CHINA_START_DATE = "2004-10-31"
NO_LOCAL_CORE_SLEEVES = frozenset({"EQ_EM", "ALT_GLD", CHINA_SLEEVE_ID})

ASSET_CLASS_GROUP_BY_SLEEVE_V3: dict[str, str] = {
    "EQ_US": "equity",
    "EQ_EZ": "equity",
    "EQ_JP": "equity",
    "EQ_EM": "equity",
    CHINA_SLEEVE_ID: "equity",
    "FI_UST": "fixed_income",
    "FI_IG": "fixed_income",
    "ALT_GLD": "alternative",
    "RE_US": "real_asset",
}

EXPOSURE_REGION_BY_SLEEVE_V3: dict[str, str] = {
    "EQ_US": "US",
    "EQ_EZ": "EURO_AREA",
    "EQ_JP": "JAPAN",
    "EQ_EM": "EM_GLOBAL",
    CHINA_SLEEVE_ID: "CHINA",
    "FI_UST": "US",
    "FI_IG": "US",
    "ALT_GLD": "GLOBAL",
    "RE_US": "US",
}

LOCAL_BLOCK_BY_SLEEVE_V3: dict[str, str] = {
    "EQ_US": "US",
    "EQ_EZ": "EURO_AREA",
    "EQ_JP": "JAPAN",
    CHINA_SLEEVE_ID: "CHINA",
    "FI_UST": "US",
    "FI_IG": "US",
    "RE_US": "US",
}

LOCKED_SLEEVES_V3: tuple[str, ...] = tuple(ASSET_CLASS_GROUP_BY_SLEEVE_V3.keys())


@dataclass(frozen=True)
class V3Paths:
    project_root: Path
    config_dir: Path
    data_intermediate_dir: Path
    data_final_dir: Path
    data_final_v2_dir: Path
    data_final_v3_dir: Path
    data_external_dir: Path
    reports_v3_dir: Path

    def ensure_directories(self) -> None:
        self.data_final_v3_dir.mkdir(parents=True, exist_ok=True)
        self.reports_v3_dir.mkdir(parents=True, exist_ok=True)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths(project_root: Path | None = None) -> V3Paths:
    root = project_root or default_project_root()
    return V3Paths(
        project_root=root,
        config_dir=root / "config",
        data_intermediate_dir=root / "data" / "intermediate",
        data_final_dir=root / "data" / "final",
        data_final_v2_dir=root / "data" / "final_v2_long_horizon",
        data_final_v3_dir=root / "data" / "final_v3_long_horizon_china",
        data_external_dir=root / "data" / "external" / "akif_candidates",
        reports_v3_dir=root / "reports" / "v3_long_horizon_china",
    )
