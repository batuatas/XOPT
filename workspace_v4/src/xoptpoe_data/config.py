"""Configuration and locked constants for XOPTPOE v1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LOCKED_SLEEVES: tuple[str, ...] = (
    "EQ_US",
    "EQ_EZ",
    "EQ_JP",
    "EQ_EM",
    "FI_UST",
    "FI_IG",
    "ALT_GLD",
    "RE_US",
)

LOCAL_BLOCKS: tuple[str, ...] = ("US", "EURO_AREA", "JAPAN")
GLOBAL_BLOCK: str = "GLOBAL"
NO_LOCAL_BLOCK_SLEEVES: frozenset[str] = frozenset({"EQ_EM", "ALT_GLD"})

LOCAL_BLOCK_BY_SLEEVE: dict[str, str] = {
    "EQ_US": "US",
    "EQ_EZ": "EURO_AREA",
    "EQ_JP": "JAPAN",
    "FI_UST": "US",
    "FI_IG": "US",
    "RE_US": "US",
}

TARGET_MANIFEST_FILE = "target_series_manifest.csv"
MACRO_MANIFEST_FILE = "macro_series_manifest.csv"
ASSET_MASTER_SEED_FILE = "asset_master_seed.csv"
SOURCE_MASTER_SEED_FILE = "source_master_seed.csv"

LAG_POLICY_TAG = "LOCKED_V1_OFFICIAL_MONTHLY_LAG1"


@dataclass(frozen=True)
class BuildPaths:
    """Filesystem paths used by the pipeline."""

    project_root: Path
    config_dir: Path
    schema_dir: Path
    src_dir: Path
    data_raw_dir: Path
    data_intermediate_dir: Path
    data_final_dir: Path
    reports_dir: Path

    def ensure_directories(self) -> None:
        """Create output directories if needed."""
        for path in (
            self.data_raw_dir,
            self.data_intermediate_dir,
            self.data_final_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BuildConfig:
    """Runtime configuration for a build."""

    paths: BuildPaths
    start_date: str = "2006-01-01"
    end_date: str | None = None
    lag_policy_tag: str = LAG_POLICY_TAG
    allow_macro_fallback: bool = False


def default_project_root() -> Path:
    """Infer repository root from package location."""
    return Path(__file__).resolve().parents[2]


def default_paths(project_root: Path | None = None) -> BuildPaths:
    """Construct default path set."""
    root = project_root or default_project_root()
    return BuildPaths(
        project_root=root,
        config_dir=root / "config",
        schema_dir=root / "schemas",
        src_dir=root / "src",
        data_raw_dir=root / "data" / "raw",
        data_intermediate_dir=root / "data" / "intermediate",
        data_final_dir=root / "data" / "final",
        reports_dir=root / "reports",
    )


def default_config(project_root: Path | None = None, end_date: str | None = None) -> BuildConfig:
    """Build default configuration instance."""
    return BuildConfig(paths=default_paths(project_root=project_root), end_date=end_date)
