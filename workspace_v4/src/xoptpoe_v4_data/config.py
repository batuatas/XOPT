"""Configuration for the XOPTPOE v4 expanded-universe first build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xoptpoe_v2_data.config import LOCKED_HORIZONS

V4_VERSION = "v4_expanded_universe"
BASE_FEATURE_START = "2006-01-31"
TARGET_FETCH_START = "2000-01-01"
FX_TICKER_EURUSD = "EURUSD=X"

TECH_COLS = (
    "ret_1m_lag",
    "ret_3m_lag",
    "ret_6m_lag",
    "ret_12m_lag",
    "mom_12_1",
    "vol_3m",
    "vol_12m",
    "maxdd_12m",
    "rel_mom_vs_treasury",
    "rel_mom_vs_us_equity",
    "rel_ret_1m_vs_treasury",
    "rel_ret_1m_vs_us_equity",
)

LOCAL_ALIAS_COLS = (
    "local_cpi_yoy",
    "local_unemp",
    "local_3m_rate",
    "local_10y_rate",
    "local_term_slope",
    "local_cpi_yoy_delta_1m",
    "local_unemp_delta_1m",
    "local_3m_rate_delta_1m",
    "local_10y_rate_delta_1m",
    "local_term_slope_delta_1m",
)

GLOBAL_REQUIRED_COLS = (
    "usd_broad_level",
    "usd_broad_logchg_1m",
    "vix_level",
    "us_real10y_level",
    "ig_oas_level",
    "oil_wti_level",
)

LOCKED_SLEEVES_V4 = (
    "EQ_US",
    "EQ_EZ",
    "EQ_JP",
    "EQ_CN",
    "EQ_EM",
    "FI_UST",
    "FI_EU_GOVT",
    "CR_US_IG",
    "CR_EU_IG",
    "CR_US_HY",
    "CR_EU_HY",
    "RE_US",
    "LISTED_RE",
    "LISTED_INFRA",
    "ALT_GLD",
)

EURO_FIXED_INCOME_SLEEVES = frozenset({"FI_EU_GOVT", "CR_EU_IG", "CR_EU_HY"})
NO_LOCAL_CORE_SLEEVES_V4 = frozenset({"EQ_EM", "EQ_CN", "LISTED_RE", "LISTED_INFRA", "ALT_GLD"})

ASSET_CLASS_GROUP_BY_SLEEVE_V4: dict[str, str] = {
    "EQ_US": "Equity",
    "EQ_EZ": "Equity",
    "EQ_JP": "Equity",
    "EQ_CN": "Equity",
    "EQ_EM": "Equity",
    "FI_UST": "Fixed Income",
    "FI_EU_GOVT": "Fixed Income",
    "CR_US_IG": "Credit",
    "CR_EU_IG": "Credit",
    "CR_US_HY": "Credit",
    "CR_EU_HY": "Credit",
    "RE_US": "Real Asset",
    "LISTED_RE": "Real Asset",
    "LISTED_INFRA": "Real Asset",
    "ALT_GLD": "Alternative",
}

EXPOSURE_REGION_BY_SLEEVE_V4: dict[str, str] = {
    "EQ_US": "US",
    "EQ_EZ": "EURO_AREA",
    "EQ_JP": "JAPAN",
    "EQ_CN": "CHINA",
    "EQ_EM": "EM_GLOBAL",
    "FI_UST": "US",
    "FI_EU_GOVT": "EURO_AREA",
    "CR_US_IG": "US",
    "CR_EU_IG": "EURO_AREA",
    "CR_US_HY": "US",
    "CR_EU_HY": "EURO_AREA",
    "RE_US": "US",
    "LISTED_RE": "EX_US_GLOBAL",
    "LISTED_INFRA": "GLOBAL",
    "ALT_GLD": "GLOBAL",
}

LOCAL_BLOCK_BY_SLEEVE_V4: dict[str, str] = {
    "EQ_US": "US",
    "EQ_EZ": "EURO_AREA",
    "EQ_JP": "JAPAN",
    "EQ_CN": "CHINA",
    "FI_UST": "US",
    "FI_EU_GOVT": "EURO_AREA",
    "CR_US_IG": "US",
    "CR_EU_IG": "EURO_AREA",
    "CR_US_HY": "US",
    "CR_EU_HY": "EURO_AREA",
    "RE_US": "US",
}

TEMPLATE_SLEEVE_BY_V4_SLEEVE: dict[str, str] = {
    "EQ_US": "EQ_US",
    "EQ_EZ": "EQ_EZ",
    "EQ_JP": "EQ_JP",
    "EQ_CN": "EQ_EM",
    "EQ_EM": "EQ_EM",
    "FI_UST": "FI_UST",
    "FI_EU_GOVT": "EQ_EZ",
    "CR_US_IG": "FI_IG",
    "CR_EU_IG": "EQ_EZ",
    "CR_US_HY": "FI_IG",
    "CR_EU_HY": "EQ_EZ",
    "RE_US": "RE_US",
    "LISTED_RE": "EQ_EM",
    "LISTED_INFRA": "EQ_EM",
    "ALT_GLD": "ALT_GLD",
}


@dataclass(frozen=True)
class V4Paths:
    project_root: Path
    config_dir: Path
    data_intermediate_dir: Path
    data_final_dir: Path
    data_final_v2_dir: Path
    data_final_v4_dir: Path
    data_modeling_v4_dir: Path
    data_external_dir: Path
    reports_v4_dir: Path

    def ensure_directories(self) -> None:
        for path in (self.data_final_v4_dir, self.data_modeling_v4_dir, self.reports_v4_dir):
            path.mkdir(parents=True, exist_ok=True)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths(project_root: Path | None = None) -> V4Paths:
    root = project_root or default_project_root()
    return V4Paths(
        project_root=root,
        config_dir=root / "config",
        data_intermediate_dir=root / "data" / "intermediate",
        data_final_dir=root / "data" / "final",
        data_final_v2_dir=root / "data" / "final_v2_long_horizon",
        data_final_v4_dir=root / "data" / "final_v4_expanded_universe",
        data_modeling_v4_dir=root / "data" / "modeling_v4",
        data_external_dir=root / "data" / "external" / "akif_candidates",
        reports_v4_dir=root / "reports" / "v4_expanded_universe",
    )
