from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt


SLEEVE_LABELS = {
    "EQ_US": "US Eq",
    "EQ_EZ": "Euro Eq",
    "EQ_JP": "Japan Eq",
    "EQ_CN": "China Eq",
    "EQ_EM": "EM Eq",
    "FI_UST": "UST",
    "FI_EU_GOVT": "Euro Gov",
    "CR_US_IG": "US IG",
    "CR_EU_IG": "Euro IG",
    "CR_US_HY": "US HY",
    "RE_US": "US RE",
    "LISTED_RE": "Listed RE xUS",
    "LISTED_INFRA": "Infra",
    "ALT_GLD": "Gold",
}

ASSET_CLASS_COLORS = {
    "Equity": "#295b84",
    "Fixed Income": "#5e8f72",
    "Credit": "#7aa88d",
    "Real Asset": "#b07d52",
    "Alternative": "#c9ad45",
}

SLEEVE_COLORS = {
    "EQ_US": "#295b84",
    "EQ_EZ": "#54779b",
    "EQ_JP": "#7297bc",
    "EQ_CN": "#3f6f97",
    "EQ_EM": "#8aa9c8",
    "FI_UST": "#5e8f72",
    "FI_EU_GOVT": "#84ac96",
    "CR_US_IG": "#4c8a84",
    "CR_EU_IG": "#6aa6a1",
    "CR_US_HY": "#34706d",
    "RE_US": "#9d6b4d",
    "LISTED_RE": "#ba8e64",
    "LISTED_INFRA": "#d0ac78",
    "ALT_GLD": "#c9ad45",
}

STRATEGY_COLORS = {
    "equal_weight": "#b8bfc5",
    "best_60_predictor": "#295b84",
    "best_60_tuned_robust": "#5e8f72",
}

STRATEGY_LABELS = {
    "equal_weight": "Equal Weight",
    "best_60_predictor": "Raw 5Y Benchmark",
    "best_60_tuned_robust": "Tuned 5Y Benchmark",
}

FOCUS_COLOR = "#295b84"
DIVERSIFIED_COLOR = "#5e8f72"
CHINA_COLOR = "#3f6f97"
NEUTRAL_DARK = "#49535b"
NEUTRAL_MID = "#9ca6ad"
NEUTRAL_LIGHT = "#dfe5e8"
GRID_COLOR = "#edf1f4"


def apply_conference_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.size": 14,
            "axes.titlesize": 22,
            "axes.labelsize": 15,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 10.5,
            "figure.titleweight": "bold",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.55,
            "grid.color": GRID_COLOR,
            "lines.linewidth": 2.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def sleeve_label(sleeve_id: str) -> str:
    return SLEEVE_LABELS.get(sleeve_id, sleeve_id)


def sleeve_color(sleeve_id: str) -> str:
    return SLEEVE_COLORS.get(sleeve_id, "#555555")


def strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


def strategy_color(strategy: str) -> str:
    return STRATEGY_COLORS.get(strategy, "#555555")


def save_figure(fig, *, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path
