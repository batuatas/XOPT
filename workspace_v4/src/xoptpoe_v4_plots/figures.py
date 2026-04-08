from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from xoptpoe_v4_plots.io import BEST_60_EXPERIMENT, BEST_60_TUNED_LABEL, PlotContext
from xoptpoe_v4_plots.style import (
    ASSET_CLASS_COLORS,
    GRID_COLOR,
    NEUTRAL_DARK,
    NEUTRAL_LIGHT,
    NEUTRAL_MID,
    apply_conference_style,
    save_figure,
    sleeve_color,
    sleeve_label,
    strategy_color,
    strategy_label,
)


TITLE_SIZE = 22
SUBTITLE_SIZE = 11.6
BODY_SIZE = 12.6
SMALL_SIZE = 10.8
OFF_WHITE = "#f7f8f9"
SOFT_BLUE = "#eef3f8"
SOFT_GREEN = "#eef5f1"
SOFT_SAND = "#f6f0e8"
SOFT_LILAC = "#f1eef7"


@dataclass(frozen=True)
class FigureArtifact:
    stem: str
    title: str
    caption: str
    interpretation: str
    evidence_type: str
    deck_status: str
    png_path: Path
    pdf_path: Path


UNIVERSE_GROUPS: dict[str, list[str]] = {
    "Equity": ["EQ_US", "EQ_EZ", "EQ_JP", "EQ_CN", "EQ_EM"],
    "Fixed Income": ["FI_UST", "FI_EU_GOVT"],
    "Credit": ["CR_US_IG", "CR_EU_IG", "CR_US_HY"],
    "Real Asset": ["RE_US", "LISTED_RE", "LISTED_INFRA"],
    "Alternative": ["ALT_GLD"],
}


def _save(fig, ctx: PlotContext, stem: str) -> tuple[Path, Path]:
    return save_figure(fig, out_dir=ctx.paths.plots_dir, stem=stem)


def _ordered_sleeves(ctx: PlotContext) -> list[str]:
    return [s for s in ctx.tuned_weights["sleeve_id"].drop_duplicates().tolist() if pd.notna(s)]


def _prediction_metric(ctx: PlotContext) -> pd.Series:
    row = ctx.prediction_metrics.loc[ctx.prediction_metrics["experiment_name"].eq(BEST_60_EXPERIMENT)]
    if row.empty:
        raise KeyError(BEST_60_EXPERIMENT)
    return row.iloc[0]


def _pct_fmt(x: float, pos: int | None = None) -> str:
    return f"{100 * x:.0f}%"


def _rounded_panel(ax, x: float, y: float, w: float, h: float, fc: str = OFF_WHITE, radius: float = 0.03, ec: str = "none", lw: float = 0.0, alpha: float = 1.0) -> FancyBboxPatch:
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
    )
    ax.add_patch(panel)
    return panel


def _panel_title(ax, x: float, y: float, title: str, subtitle: str | None = None) -> None:
    ax.text(x, y, title, fontsize=TITLE_SIZE, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="top")
    if subtitle:
        ax.text(x, y - 0.05, subtitle, fontsize=SUBTITLE_SIZE, color=NEUTRAL_MID, ha="left", va="top")


def _sleeve_pill(ax, x: float, y: float, text: str, color: str, w: float = 0.125, h: float = 0.07) -> None:
    pill = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.03", linewidth=0, facecolor=color)
    ax.add_patch(pill)
    ax.text(x + w / 2, y + h / 2, text, fontsize=10.8, color="white", ha="center", va="center", fontweight="bold")


def _legend_handles_from_sleeves(sleeves: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7.5, markerfacecolor=sleeve_color(s), markeredgecolor="white", markeredgewidth=0.6, label=sleeve_label(s))
        for s in sleeves
    ]


def _end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(8, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10.8,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.94},
        clip_on=False,
    )


def graphic01_why_long_horizon_saa_v4(ctx: PlotContext) -> FigureArtifact:
    fig, ax = plt.subplots(figsize=(14.4, 8.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel_title(
        ax,
        0.06,
        0.93,
        "Why long-horizon strategic allocation?",
        "The problem is where the opportunity set may settle over years, then how that view maps into an investable benchmark.",
    )

    _rounded_panel(ax, 0.06, 0.16, 0.88, 0.58, fc=OFF_WHITE, radius=0.045)
    ax.text(0.10, 0.67, "5Y+ allocation questions", fontsize=29, color=NEUTRAL_DARK, fontweight="bold", ha="left", va="center")
    ax.text(0.10, 0.60, "Strategic allocators care about medium-term regime shifts, not just next-quarter market noise.", fontsize=13.4, color=NEUTRAL_MID, ha="left", va="center")

    stages = [
        ("Macro-financial state", "growth, inflation, rates,\nrisk appetite", SOFT_BLUE),
        ("AI prediction", "cross-sleeve 5Y expected\nexcess returns", SOFT_GREEN),
        ("Robust allocation", "long-only, fully invested,\ndiversified benchmark", SOFT_SAND),
        ("Scenario interpretation", "stress the benchmark under\ncoherent macro paths", SOFT_LILAC),
    ]
    xs = [0.10, 0.31, 0.52, 0.73]
    for x, (title, desc, fc) in zip(xs, stages):
        _rounded_panel(ax, x, 0.28, 0.16, 0.22, fc=fc, radius=0.04)
        ax.text(x + 0.02, 0.45, title, fontsize=15.0, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="center")
        ax.text(x + 0.02, 0.36, desc, fontsize=11.8, color=NEUTRAL_MID, ha="left", va="center")
    for x0, x1 in zip([0.26, 0.47, 0.68], [0.31, 0.52, 0.73]):
        ax.annotate("", xy=(x1, 0.39), xytext=(x0, 0.39), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=NEUTRAL_MID, mutation_scale=14))

    ax.text(0.10, 0.21, "Long-horizon SAA turns macro-financial state into a disciplined portfolio decision rather than a short-horizon trade call.", fontsize=13.0, color=NEUTRAL_DARK, ha="left", va="center")

    png, pdf = _save(fig, ctx, "graphic01_why_long_horizon_saa_v4")
    return FigureArtifact(
        stem="graphic01_why_long_horizon_saa_v4",
        title="Why long-horizon strategic allocation?",
        caption="Conceptual opener for the conference narrative: macro-financial state feeds AI prediction, robust allocation, and later scenario interpretation.",
        interpretation="Framing graphic for the broad audience: this is a strategic allocation workflow built around multi-year questions.",
        evidence_type="framing",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def graphic02_universe_and_target_v4(ctx: PlotContext) -> FigureArtifact:
    fig, ax = plt.subplots(figsize=(14.6, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel_title(
        ax,
        0.06,
        0.93,
        "Investable universe and prediction target",
        "The active benchmark spans 14 sleeves, while the prediction target stays deliberately simple: annualized 5Y excess return.",
    )

    _rounded_panel(ax, 0.06, 0.15, 0.58, 0.66, fc=OFF_WHITE, radius=0.045)
    _rounded_panel(ax, 0.68, 0.26, 0.26, 0.44, fc=SOFT_SAND, radius=0.045)

    layout = {
        "Equity": (0.09, 0.56, 0.52, 0.18),
        "Fixed Income": (0.09, 0.40, 0.25, 0.11),
        "Credit": (0.36, 0.40, 0.25, 0.11),
        "Real Asset": (0.09, 0.23, 0.40, 0.12),
        "Alternative": (0.51, 0.23, 0.10, 0.12),
    }
    for category, sleeves in UNIVERSE_GROUPS.items():
        x, y, w, h = layout[category]
        color = ASSET_CLASS_COLORS[category]
        _rounded_panel(ax, x, y, w, h, fc=color, radius=0.035)
        ax.text(x + 0.02, y + h - 0.035, category, fontsize=14.0, fontweight="bold", color="white", ha="left", va="top")
        cols = 3 if len(sleeves) >= 3 else 2
        for i, sleeve in enumerate(sleeves):
            row = i // cols
            col = i % cols
            sx = x + 0.02 + col * (w / cols)
            sy = y + h - 0.085 - row * 0.05
            ax.text(sx, sy, sleeve_label(sleeve), fontsize=11.6, color="white", ha="left", va="center")

    ax.text(0.71, 0.62, "Target", fontsize=17, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="center")
    ax.text(0.71, 0.53, "Annualized 5Y\nexcess return", fontsize=25, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="center")
    ax.text(0.71, 0.39, "60-month horizon\nby sleeve", fontsize=13.2, color=NEUTRAL_MID, ha="left", va="center")
    ax.annotate("", xy=(0.68, 0.48), xytext=(0.64, 0.48), arrowprops=dict(arrowstyle="-|>", lw=1.9, color=NEUTRAL_MID, mutation_scale=16))

    png, pdf = _save(fig, ctx, "graphic02_universe_and_target_v4")
    return FigureArtifact(
        stem="graphic02_universe_and_target_v4",
        title="Investable universe and prediction target",
        caption="Standalone conference graphic combining the active 14-sleeve investable universe with the locked 60-month excess-return target definition.",
        interpretation="Universe-plus-target slide: broad asset coverage on the left, single clear prediction target on the right.",
        evidence_type="setup",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def graphic03_features_to_ai_prediction_v4(ctx: PlotContext) -> FigureArtifact:
    fig, ax = plt.subplots(figsize=(14.6, 8.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel_title(
        ax,
        0.06,
        0.93,
        "From macro-financial state to AI prediction",
        "The benchmark predictor is an Elastic Net model that converts a broad state vector into sleeve-level 5Y return expectations.",
    )

    _rounded_panel(ax, 0.06, 0.18, 0.36, 0.60, fc=OFF_WHITE, radius=0.045)
    _rounded_panel(ax, 0.46, 0.28, 0.19, 0.40, fc=SOFT_GREEN, radius=0.05)
    _rounded_panel(ax, 0.70, 0.24, 0.24, 0.48, fc=SOFT_BLUE, radius=0.045)

    ax.text(0.09, 0.71, "Feature groups", fontsize=18, fontweight="bold", color=NEUTRAL_DARK, ha="left")
    feature_cards = [
        ("Global macro /\nmarket state", SOFT_BLUE),
        ("Regional macro\nblocks", SOFT_LILAC),
        ("Valuation /\ntechnical signals", SOFT_SAND),
        ("Interaction\nterms", SOFT_GREEN),
    ]
    y_positions = [0.58, 0.46, 0.34, 0.22]
    for (label, fc), y in zip(feature_cards, y_positions):
        _rounded_panel(ax, 0.10, y, 0.26, 0.085, fc=fc, radius=0.03)
        ax.text(0.12, y + 0.0425, label, fontsize=12.5, color=NEUTRAL_DARK, ha="left", va="center", fontweight="bold")

    ax.text(0.555, 0.58, "AI prediction", fontsize=18, fontweight="bold", color=NEUTRAL_DARK, ha="center")
    ax.text(0.555, 0.47, "Elastic Net\nbenchmark", fontsize=22, fontweight="bold", color=NEUTRAL_DARK, ha="center", va="center")
    ax.text(0.555, 0.34, "cross-sleeve\nsignal separation", fontsize=12.6, color=NEUTRAL_MID, ha="center", va="center")

    ax.text(0.73, 0.62, "Prediction output", fontsize=18, fontweight="bold", color=NEUTRAL_DARK, ha="left")
    ax.text(0.73, 0.50, "Annualized 5Y\nexcess return", fontsize=24, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="center")
    ax.text(0.73, 0.35, "one forecast per sleeve,\none month at a time", fontsize=12.8, color=NEUTRAL_MID, ha="left", va="center")

    ax.annotate("", xy=(0.46, 0.48), xytext=(0.42, 0.48), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=NEUTRAL_MID, mutation_scale=16))
    ax.annotate("", xy=(0.70, 0.48), xytext=(0.65, 0.48), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=NEUTRAL_MID, mutation_scale=16))

    png, pdf = _save(fig, ctx, "graphic03_features_to_ai_prediction_v4")
    return FigureArtifact(
        stem="graphic03_features_to_ai_prediction_v4",
        title="From macro-financial state to AI prediction",
        caption="Polished information-flow graphic showing broad feature groups feeding the locked Elastic Net benchmark and sleeve-level 5Y return output.",
        interpretation="Modeling pipeline graphic: broad information in, explicit AI benchmark in the middle, annualized 60-month excess return out.",
        evidence_type="setup",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def graphic04_prediction_evidence_v4(ctx: PlotContext) -> FigureArtifact:
    df = ctx.predictions_test.loc[
        ctx.predictions_test["experiment_name"].eq(BEST_60_EXPERIMENT)
        & ctx.predictions_test["horizon_months"].eq(60)
    ].copy()
    df["sleeve_id"] = df["sleeve_id"].astype(str)
    metric = _prediction_metric(ctx)
    x = df["y_true"].to_numpy(dtype=float)
    y = df["y_pred"].to_numpy(dtype=float)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.012

    fig = plt.figure(figsize=(14.6, 8.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.30], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_side.axis("off")

    for sleeve_id, chunk in df.groupby("sleeve_id"):
        ax.scatter(
            chunk["y_true"],
            chunk["y_pred"],
            s=54,
            alpha=0.78,
            color=sleeve_color(sleeve_id),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=NEUTRAL_MID, linewidth=1.4, linestyle=(0, (4, 4)), zorder=2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(color=GRID_COLOR, linewidth=0.7)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax.set_xlabel("Realized annualized 5Y excess return")
    ax.set_ylabel("Predicted annualized 5Y excess return")
    ax.set_title("The model produces cross-sleeve separation", loc="left", pad=16)

    _rounded_panel(ax_side, 0.05, 0.42, 0.88, 0.46, fc=OFF_WHITE, radius=0.04)
    ax_side.text(0.12, 0.80, "Test-set readout", fontsize=15.5, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="center")
    
    # Row 1 
    ax_side.text(0.12, 0.66, f"Correlation\n{metric['test_corr']:.2f}", fontsize=17, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="center")
    ax_side.text(0.55, 0.66, f"RMSE\n{100*metric['test_rmse']:.1f}%", fontsize=17, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="center")
    
    # Row 2
    ax_side.text(0.12, 0.52, f"Sign accuracy\n{metric['test_sign_accuracy']:.2f}", fontsize=17, fontweight="bold", color=NEUTRAL_DARK, ha="left", va="center")

    handles = _legend_handles_from_sleeves(_ordered_sleeves(ctx))
    ax_side.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.00), 
        frameon=False, ncol=2, handletextpad=0.4, labelspacing=0.85, borderaxespad=0
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.12, wspace=0.04)

    png, pdf = _save(fig, ctx, "graphic04_prediction_evidence_v4")
    return FigureArtifact(
        stem="graphic04_prediction_evidence_v4",
        title="The model produces cross-sleeve separation",
        caption="Finance-conference scatter showing predicted versus realized annualized 5Y excess returns with sleeve-colored points and a compact side readout.",
        interpretation="Prediction evidence graphic: the benchmark does not collapse to one view, it separates sleeves in economically meaningful ways.",
        evidence_type="prediction",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def graphic05_benchmark_behavior_v4(ctx: PlotContext) -> FigureArtifact:
    path = ctx.annual_rebalance_5y_path.path_frame.copy()
    weights = ctx.annual_rebalance_5y_path.weight_frame.copy()
    path = path.loc[(path["month_end"] >= pd.Timestamp("2015-01-31")) & (path["month_end"] <= pd.Timestamp("2025-12-31"))].copy()
    weights = weights.loc[(weights["anchor_month_end"] >= pd.Timestamp("2015-12-31")) & (weights["anchor_month_end"] <= pd.Timestamp("2025-12-31"))].copy()

    sleeve_order = _ordered_sleeves(ctx)
    yearly = weights.pivot(index="anchor_month_end", columns="sleeve_id", values="portfolio_weight").fillna(0.0).reindex(columns=sleeve_order)

    recent_weights = ctx.recent_5y_snapshot.weight_frame.copy()
    recent_weights["month_end"] = pd.to_datetime(recent_weights["month_end"])
    latest_recent = recent_weights["month_end"].max()
    latest_recent_ts = None
    if pd.notna(latest_recent) and latest_recent.year == 2026:
        latest_recent_ts = pd.Timestamp(latest_recent)
        latest_vector = (
            recent_weights.loc[recent_weights["month_end"].eq(latest_recent), ["sleeve_id", "portfolio_weight"]]
            .set_index("sleeve_id")["portfolio_weight"]
            .reindex(sleeve_order)
            .fillna(0.0)
        )
        yearly.loc[latest_recent_ts] = latest_vector.values

    fig = plt.figure(figsize=(14.8, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.9, 0.9], hspace=0.18)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    ax_top.set_title("The robust 5Y benchmark produces interpretable allocations", loc="left", pad=18)
    ax_top.text(0.0, 1.02, "Walk-forward refit at each year-end, then next-year hold", transform=ax_top.transAxes, fontsize=11.0, color=NEUTRAL_MID, ha="left", va="bottom")

    rebalance_points = pd.to_datetime(weights["anchor_month_end"].drop_duplicates().sort_values()) + pd.offsets.MonthEnd(1)
    for strategy in [BEST_60_TUNED_LABEL, "equal_weight"]:
        sub = path.loc[path["strategy_label"].eq(strategy)].copy()
        color = strategy_color(strategy)
        width = 3.0 if strategy == BEST_60_TUNED_LABEL else 2.1
        ax_top.plot(sub["month_end"], sub["wealth_index"], color=color, linewidth=width, zorder=3)
        marks = sub.loc[sub["month_end"].isin(rebalance_points)]
        ax_top.scatter(marks["month_end"], marks["wealth_index"], s=22 if strategy == BEST_60_TUNED_LABEL else 16, color=color, edgecolor="white", linewidth=0.5, zorder=4)
        if latest_recent_ts is not None and not sub.empty:
            last_x = pd.Timestamp(sub["month_end"].iloc[-1])
            last_y = float(sub["wealth_index"].iloc[-1])
            ax_top.plot([last_x, latest_recent_ts], [last_y, last_y], color=color, linewidth=1.5 if strategy == BEST_60_TUNED_LABEL else 1.1, linestyle=(0, (3, 3)), alpha=0.88, zorder=2)
            ax_top.scatter([latest_recent_ts], [last_y], s=18, color=color, edgecolor="white", linewidth=0.4, zorder=5)

    for point in rebalance_points:
        ax_top.axvline(point, color=NEUTRAL_LIGHT, linewidth=0.85, linestyle=(0, (2, 4)), zorder=1)

    ax_top.grid(color=GRID_COLOR, linewidth=0.7)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Wealth index")
    ax_top.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    best_last = path.loc[path["strategy_label"].eq(BEST_60_TUNED_LABEL)].iloc[-1]
    ew_last = path.loc[path["strategy_label"].eq("equal_weight")].iloc[-1]
    label_x = latest_recent_ts if latest_recent_ts is not None else pd.Timestamp(best_last["month_end"])
    _end_label(ax_top, label_x, best_last["wealth_index"], strategy_label(BEST_60_TUNED_LABEL), strategy_color(BEST_60_TUNED_LABEL), dy=8)
    _end_label(ax_top, label_x, ew_last["wealth_index"], strategy_label("equal_weight"), strategy_color("equal_weight"), dy=-10)

    bottoms = np.zeros(len(yearly.index), dtype=float)
    for sleeve_id in sleeve_order:
        vals = yearly[sleeve_id].to_numpy(dtype=float)
        ax_bot.bar(yearly.index, vals, width=190, bottom=bottoms, color=sleeve_color(sleeve_id), edgecolor="white", linewidth=0.55, align="center")
        bottoms += vals
    ax_bot.set_ylim(0.0, 1.0)
    ax_bot.set_ylabel("Weight")
    ax_bot.yaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax_bot.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax_bot.grid(axis="x", visible=False)
    ticks = list(yearly.index)
    labels = [("2026\n(Feb)" if latest_recent_ts is not None and pd.Timestamp(t) == latest_recent_ts else pd.Timestamp(t).strftime("%Y")) for t in ticks]
    ax_bot.set_xticks(ticks, labels)
    left_pad = yearly.index.min() - pd.Timedelta(days=230)
    right_pad = yearly.index.max() + pd.Timedelta(days=150)
    ax_bot.set_xlim(left_pad, right_pad)

    fig.legend(
        handles=_legend_handles_from_sleeves(sleeve_order),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=7,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.16, hspace=0.18)

    png, pdf = _save(fig, ctx, "graphic05_benchmark_behavior_v4")
    return FigureArtifact(
        stem="graphic05_benchmark_behavior_v4",
        title="The robust 5Y benchmark produces interpretable allocations",
        caption="Hero conference graphic pairing the realized walk-forward wealth path with yearly stacked allocations and the current 2026 snapshot.",
        interpretation="Hero benchmark graphic: the active 5Y benchmark beats equal weight over the realized path and remains readable as a diversified allocation object.",
        evidence_type="allocation behavior",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def build_v4_conference_figures(ctx: PlotContext) -> list[FigureArtifact]:
    apply_conference_style()
    return [
        graphic01_why_long_horizon_saa_v4(ctx),
        graphic02_universe_and_target_v4(ctx),
        graphic03_features_to_ai_prediction_v4(ctx),
        graphic04_prediction_evidence_v4(ctx),
        graphic05_benchmark_behavior_v4(ctx),
    ]
