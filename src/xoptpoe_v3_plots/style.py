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
    "FI_UST": "UST 7-10Y",
    "FI_IG": "IG Credit",
    "ALT_GLD": "Gold",
    "RE_US": "US REITs",
}

SLEEVE_COLORS = {
    "EQ_US": "#315d84",
    "EQ_EZ": "#8aa3bf",
    "EQ_JP": "#7fb3b3",
    "EQ_CN": "#b46a55",
    "EQ_EM": "#c89c6c",
    "FI_UST": "#74a887",
    "FI_IG": "#9bc9ab",
    "ALT_GLD": "#dfc868",
    "RE_US": "#b89a84",
}

STRATEGY_COLORS = {
    "equal_weight": "#9aa3a8",
    "best_60_predictor": "#1f4e79",
    "best_120_predictor": "#4f5b66",
    "combined_60_120_predictor": "#7a6f91",
    "combined_std_120tilt_top_k_capped": "#6b8f71",
    "best_shared_predictor": "#98a1a8",
    "pto_nn_signal": "#b4b9c3",
    "e2e_nn_signal": "#8fb3cc",
}

STRATEGY_LABELS = {
    "equal_weight": "Equal Weight",
    "best_60_predictor": "Robust 5Y Benchmark",
    "best_120_predictor": "Raw 10Y Ceiling",
    "combined_60_120_predictor": "Combined 5Y/10Y",
    "combined_std_120tilt_top_k_capped": "120-tilt Capped",
    "best_shared_predictor": "Shared Benchmark",
    "pto_nn_signal": "PTO",
    "e2e_nn_signal": "E2E",
}

FOCUS_COLOR = "#1f4e79"
CHINA_COLOR = "#b23a2f"
NEUTRAL_DARK = "#4f5b66"
NEUTRAL_MID = "#9aa3a8"
NEUTRAL_LIGHT = "#d9dfe3"
GRID_COLOR = "#edf1f4"


def apply_conference_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.size": 15,
            "axes.titlesize": 24,
            "axes.labelsize": 16,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
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


def strategy_color(strategy_label: str) -> str:
    return STRATEGY_COLORS.get(strategy_label, "#555555")


def strategy_label(strategy_label_value: str) -> str:
    return STRATEGY_LABELS.get(strategy_label_value, strategy_label_value)


def finalize_axes(ax, *, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_footnote(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=10.5, color="#555555")


def save_figure(fig, *, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path
