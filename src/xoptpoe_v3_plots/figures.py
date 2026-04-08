from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from xoptpoe_v3_plots.io import PlotContext
from xoptpoe_v3_plots.style import (
    FOCUS_COLOR,
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


def _pct_fmt(x: float, pos: int | None = None) -> str:
    return f"{100 * x:.0f}%"


def _pct_text(x: float, decimals: int = 1) -> str:
    return f"{100 * x:.{decimals}f}%"


def _blend_with_white(color: str, weight: float = 0.45) -> tuple[float, float, float]:
    base = np.array(to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple((1.0 - weight) * base + weight * white)


def _selected_portfolio_metrics(ctx: PlotContext, labels: list[str]) -> pd.DataFrame:
    df = ctx.portfolio_metrics.copy()
    df = df.loc[df["split"].eq("test") & df["strategy_label"].isin(labels)].copy()
    df["strategy_label"] = pd.Categorical(df["strategy_label"], categories=labels, ordered=True)
    return df.sort_values("strategy_label").reset_index(drop=True)


def _prediction_panel(ctx: PlotContext, experiment_name: str, *, horizon: int) -> pd.DataFrame:
    df = ctx.predictions_test.copy()
    df = df.loc[df["experiment_name"].eq(experiment_name) & df["horizon_months"].eq(horizon)].copy()
    df["month_end"] = pd.to_datetime(df["month_end"])
    return df.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)


def _weight_panel(ctx: PlotContext, strategy_label_value: str, split: str = "test") -> pd.DataFrame:
    df = ctx.portfolio_attribution.copy()
    df = df.loc[df["split"].eq(split) & df["strategy_label"].eq(strategy_label_value)].copy()
    df["month_end"] = pd.to_datetime(df["month_end"])
    df["sleeve_id"] = pd.Categorical(df["sleeve_id"], categories=list(ctx.sleeve_order), ordered=True)
    return df.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)


def _metric_row(metrics: pd.DataFrame, experiment_name: str) -> pd.Series:
    row = metrics.loc[metrics["experiment_name"].eq(experiment_name)]
    if row.empty:
        raise KeyError(experiment_name)
    return row.iloc[0]


def _month_tick_positions(months: list[pd.Timestamp], *, step: int = 4) -> tuple[list[int], list[str]]:
    idx = list(range(0, len(months), step))
    labels = [months[i].strftime("%b\n%Y") for i in idx]
    return idx, labels


def _rank_spread_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for month_end, group in df.groupby("month_end", sort=True):
        ordered = group.sort_values("y_pred", ascending=False).reset_index(drop=True)
        top = ordered.head(3)
        bottom = ordered.tail(3)
        rank_ic = ordered[["y_true", "y_pred"]].corr(method="spearman").iloc[0, 1]
        sign_acc = ((ordered["y_true"] >= 0) == (ordered["y_pred"] >= 0)).mean()
        rows.append(
            {
                "month_end": pd.Timestamp(month_end),
                "top_bottom_spread": float(top["y_true"].mean() - bottom["y_true"].mean()),
                "top_avg_spread": float(top["y_true"].mean() - ordered["y_true"].mean()),
                "rank_ic": float(rank_ic),
                "sign_acc": float(sign_acc),
            }
        )
    out = pd.DataFrame(rows).sort_values("month_end").reset_index(drop=True)
    out["rank_ic_roll6"] = out["rank_ic"].rolling(6, min_periods=3).mean()
    out["spread_roll6"] = out["top_bottom_spread"].rolling(6, min_periods=3).mean()
    return out


def _recent_prediction_frame(ctx: PlotContext) -> pd.DataFrame:
    frame = ctx.recent_5y_snapshot.prediction_frame.copy()
    frame["month_end"] = pd.to_datetime(frame["month_end"])
    return frame.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)


def _recent_weight_frame(ctx: PlotContext) -> pd.DataFrame:
    frame = ctx.recent_5y_snapshot.weight_frame.copy()
    frame["month_end"] = pd.to_datetime(frame["month_end"])
    frame["portfolio_weight"] = frame["portfolio_weight"].clip(lower=0.0)
    frame["portfolio_weight"] = frame.groupby("month_end")["portfolio_weight"].transform(
        lambda s: s / s.sum() if float(s.sum()) > 0 else s
    )
    return frame.sort_values(["month_end", "sleeve_id"]).reset_index(drop=True)


def _style_axes(ax, *, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _figure_benchmark_60_overview(ctx: PlotContext) -> FigureArtifact:
    labels = ["best_60_predictor", "e2e_nn_signal", "pto_nn_signal", "equal_weight"]
    df = _selected_portfolio_metrics(ctx, labels)
    df["pretty"] = [strategy_label(v) for v in df["strategy_label"].astype(str)]
    color_map = {
        "best_60_predictor": FOCUS_COLOR,
        "e2e_nn_signal": _blend_with_white(FOCUS_COLOR, 0.55),
        "pto_nn_signal": NEUTRAL_MID,
        "equal_weight": NEUTRAL_LIGHT,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)
    specs = [("avg_return", "Average return"), ("sharpe", "Sharpe")]
    for ax, (col, xlabel) in zip(axes, specs, strict=True):
        vals = df[col].to_numpy(dtype=float)
        y = np.arange(len(df))
        colors = [color_map[v] for v in df["strategy_label"].astype(str)]
        ax.hlines(y, 0.0, vals, color=NEUTRAL_LIGHT, linewidth=2.0, zorder=1)
        ax.scatter(vals, y, s=140, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
        ax.set_yticks(y, df["pretty"])
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        if col == "avg_return":
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
            pad = 0.0025
            ax.set_xlim(0.0, max(vals) + 0.016)
        else:
            pad = 0.07
            ax.set_xlim(0.0, max(vals) + 0.45)
        for yi, val in zip(y, vals, strict=True):
            text = _pct_text(val) if col == "avg_return" else f"{val:.2f}"
            ax.text(val + pad, yi, text, va="center", ha="left", fontsize=11.5, color=NEUTRAL_DARK)
        _style_axes(ax, xlabel=xlabel)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.tight_layout(pad=1.5, w_pad=2.2)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="benchmark_60_overview")
    return FigureArtifact(
        stem="benchmark_60_overview",
        title="The 5Y benchmark improves on simple alternatives",
        caption="Historical validation comparison for the 60m carry-forward benchmark against equal weight and the two neural comparators.",
        interpretation="Compact benchmark comparison centered on the active 60m benchmark rather than the broader benchmark stack.",
        evidence_type="historical validation",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_prediction_scatter_60(ctx: PlotContext) -> FigureArtifact:
    df = _prediction_panel(ctx, ctx.active_60_prediction, horizon=60)
    metric = _metric_row(ctx.prediction_metrics, ctx.active_60_prediction)
    x = df["y_true"].to_numpy(dtype=float)
    y = df["y_pred"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    fit_x = np.linspace(float(min(x.min(), y.min())), float(max(x.max(), y.max())), 100)
    fit_y = intercept + slope * fit_x

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    ax.scatter(x, y, s=48, color=_blend_with_white(FOCUS_COLOR, 0.72), edgecolor="white", linewidth=0.6, alpha=0.95)
    ax.plot(fit_x, fit_y, color=FOCUS_COLOR, linewidth=2.6)
    ax.plot(fit_x, fit_x, color=NEUTRAL_MID, linewidth=1.1, linestyle=(0, (4, 4)))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax.grid(color=GRID_COLOR, linewidth=0.6)
    _style_axes(
        ax,
        xlabel="Realized annualized 5Y excess return",
        ylabel="Predicted annualized 5Y excess return",
    )
    summary = f"Corr {metric['test_corr']:.2f}\nRMSE {_pct_text(float(metric['test_rmse']))}\nSign {metric['test_sign_accuracy']:.2f}"
    ax.text(
        0.03,
        0.97,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        color=NEUTRAL_DARK,
        bbox={"facecolor": "white", "edgecolor": "#d8dfe4", "boxstyle": "round,pad=0.45", "alpha": 0.98},
    )
    fig.tight_layout(pad=1.4)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="prediction_scatter_60")
    return FigureArtifact(
        stem="prediction_scatter_60",
        title="5Y forecasts align with realized outcomes",
        caption="Historical test scatter for the active 60m prediction anchor.",
        interpretation="Main credibility figure for the 60m predictor: the signal is noisy, but it is not random.",
        evidence_type="historical validation",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_prediction_rank_spread_60(ctx: PlotContext) -> FigureArtifact:
    df = _prediction_panel(ctx, ctx.active_60_prediction, horizon=60)
    spread = _rank_spread_frame(df)
    colors = [FOCUS_COLOR if v >= 0 else NEUTRAL_MID for v in spread["top_bottom_spread"]]
    mean_spread = float(spread["top_bottom_spread"].mean())
    positive_share = float((spread["top_bottom_spread"] > 0).mean())

    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    ax.bar(spread["month_end"], spread["top_bottom_spread"], color=colors, width=23, edgecolor="white", linewidth=0.7)
    ax.axhline(0.0, color=NEUTRAL_MID, linewidth=1.0)
    ax.axhline(mean_spread, color=FOCUS_COLOR, linewidth=2.0, linestyle=(0, (5, 4)))
    months = list(spread["month_end"])
    idx, labels = _month_tick_positions(months, step=4)
    ax.set_xticks([months[i] for i in idx], labels)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    _style_axes(
        ax,
        xlabel="Historical test decision month",
        ylabel="Realized spread",
    )
    ax.text(
        0.01,
        0.96,
        f"Average spread {_pct_text(mean_spread)}\nPositive months {positive_share:.0%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.3,
        color=NEUTRAL_DARK,
    )
    fig.tight_layout(pad=1.4)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="prediction_rank_spread_60")
    return FigureArtifact(
        stem="prediction_rank_spread_60",
        title="Predicted leaders outperform predicted laggards",
        caption="Historical monthly top-minus-bottom realized spread for the active 60m predictor.",
        interpretation="Cross-sectional separation figure: the 60m predictor usually ranks better sleeves above worse sleeves.",
        evidence_type="historical validation",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_rolling_prediction_quality_60(ctx: PlotContext) -> FigureArtifact:
    df = _prediction_panel(ctx, ctx.active_60_prediction, horizon=60)
    spread = _rank_spread_frame(df)

    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    ax.plot(spread["month_end"], spread["rank_ic"], color=NEUTRAL_LIGHT, linewidth=1.8)
    ax.scatter(spread["month_end"], spread["rank_ic"], color=NEUTRAL_MID, s=26, zorder=3)
    ax.plot(spread["month_end"], spread["rank_ic_roll6"], color=FOCUS_COLOR, linewidth=2.7)
    ax.axhline(0.0, color=NEUTRAL_MID, linewidth=1.0, linestyle=(0, (4, 4)))
    months = list(spread["month_end"])
    idx, labels = _month_tick_positions(months, step=4)
    ax.set_xticks([months[i] for i in idx], labels)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    _style_axes(ax, xlabel="Historical test decision month", ylabel="Spearman rank IC")
    ax.text(
        0.01,
        0.95,
        f"Mean IC {spread['rank_ic'].mean():.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.3,
        color=NEUTRAL_DARK,
    )
    fig.tight_layout(pad=1.4)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="rolling_prediction_quality_60")
    return FigureArtifact(
        stem="rolling_prediction_quality_60",
        title="Prediction quality is persistent, not one-month noise",
        caption="Monthly and rolling rank-IC view for the active 60m predictor.",
        interpretation="Time-based quality check for the 60m signal; useful, but more appendix-like than the main rank-spread figure.",
        evidence_type="historical validation",
        deck_status="appendix",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_allocation_heatmap_best_60(ctx: PlotContext) -> FigureArtifact:
    weights = _weight_panel(ctx, "best_60_predictor")
    table = (
        weights.pivot(index="sleeve_id", columns="month_end", values="weight")
        .reindex(index=list(ctx.sleeve_order))
        .fillna(0.0)
    )
    months = list(table.columns)
    cmap = LinearSegmentedColormap.from_list(
        "conference_blue", ["#ffffff", "#e6eef5", "#9db8cd", FOCUS_COLOR]
    )

    fig, ax = plt.subplots(figsize=(14.0, 4.6))
    im = ax.imshow(
        table.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=max(0.30, float(np.nanquantile(table.to_numpy(dtype=float), 0.98))),
    )
    ax.set_yticks(range(len(ctx.sleeve_order)))
    ax.set_yticklabels([sleeve_label(s) for s in ctx.sleeve_order])
    idx, labels = _month_tick_positions(months, step=6)
    ax.set_xticks(idx, labels)
    ax.tick_params(axis="both", length=0)
    _style_axes(ax, xlabel="Historical test decision month", ylabel="Sleeve")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Weight")
    cbar.ax.tick_params(labelsize=10.5)
    fig.tight_layout(pad=1.4)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="allocation_heatmap_best_60")
    return FigureArtifact(
        stem="allocation_heatmap_best_60",
        title="The 5Y benchmark allocates across multiple sleeves",
        caption="Historical test-period weight heatmap for the active 60m portfolio benchmark.",
        interpretation="Allocation figure for the carry-forward 60m benchmark: broad enough to be interpretable, concentrated enough to matter.",
        evidence_type="allocation interpretation",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_portfolio_wealth_path_best_60(ctx: PlotContext) -> FigureArtifact:
    bundle = ctx.annual_rebalance_5y_path
    path = bundle.path_frame.copy()
    path["month_end"] = pd.to_datetime(path["month_end"])
    anchors = sorted(pd.to_datetime(bundle.weight_frame["anchor_month_end"]).unique())
    anchor_weights = bundle.weight_frame.copy()
    anchor_weights["anchor_month_end"] = pd.to_datetime(anchor_weights["anchor_month_end"])
    anchor_points = bundle.anchor_frame.copy()
    anchor_points["anchor_month_end"] = pd.to_datetime(anchor_points["anchor_month_end"])

    fig = plt.figure(figsize=(13.2, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.45], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    order = ["equal_weight", "best_60_predictor"]
    for strategy in order:
        sub = path.loc[path["strategy_label"].eq(strategy)].sort_values("month_end")
        color = strategy_color(strategy)
        linewidth = 2.7 if strategy == "best_60_predictor" else 2.0
        zorder = 4 if strategy == "best_60_predictor" else 3
        ax_top.plot(sub["month_end"], sub["wealth_index"], color=color, linewidth=linewidth, zorder=zorder)

        marks = anchor_points.loc[anchor_points["strategy_label"].eq(strategy)].sort_values("anchor_month_end")
        ax_top.scatter(
            marks["anchor_month_end"],
            marks["wealth_index"],
            s=24 if strategy == "best_60_predictor" else 18,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=zorder + 1,
        )

    for anchor in anchors:
        ax_top.axvline(pd.Timestamp(anchor), color=GRID_COLOR, linewidth=0.8, zorder=0)

    end_offsets = {"best_60_predictor": 9, "equal_weight": -9}
    for strategy in order:
        sub = path.loc[path["strategy_label"].eq(strategy)].sort_values("month_end")
        last = sub.iloc[-1]
        ax_top.annotate(
            strategy_label(strategy),
            xy=(last["month_end"], float(last["wealth_index"])),
            xytext=(8, end_offsets[strategy]),
            textcoords="offset points",
            ha="left",
            va="center",
            color=strategy_color(strategy),
            fontsize=12.5,
            fontweight="bold" if strategy == "best_60_predictor" else None,
        )

    ax_top.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax_top.grid(axis="x", visible=False)
    ax_top.set_xlim(path["month_end"].min(), path["month_end"].max() + pd.offsets.MonthEnd(2))
    ax_top.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_axes(ax_top, ylabel="Wealth index")
    ax_top.tick_params(axis="x", labelbottom=False)

    x = np.arange(len(anchors))
    holding_year_labels = [(pd.Timestamp(v) + pd.offsets.YearEnd(0)).year + 1 for v in anchors]
    bottoms = np.zeros(len(anchors))
    legend_handles: list[Patch] = []
    for sleeve_id in ctx.sleeve_order:
        vals = (
            anchor_weights.loc[anchor_weights["sleeve_id"].eq(sleeve_id)]
            .set_index("anchor_month_end")
            .reindex(anchors)["portfolio_weight"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        color = sleeve_color(sleeve_id)
        ax_bottom.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.68,
            color=color,
            edgecolor="white",
            linewidth=0.45,
        )
        bottoms = bottoms + vals
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=sleeve_label(sleeve_id)))

    ax_bottom.set_xticks(x, [str(v) for v in holding_year_labels])
    ax_bottom.set_ylim(0.0, 1.0)
    ax_bottom.yaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax_bottom.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax_bottom.grid(axis="x", visible=False)
    _style_axes(ax_bottom, xlabel="Holding year", ylabel="Weights")
    ax_bottom.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=5,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.45,
        fontsize=9.8,
    )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.98, bottom=0.20, hspace=0.08)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="portfolio_wealth_path_best_60")
    return FigureArtifact(
        stem="portfolio_wealth_path_best_60",
        title="Annual 5Y signals roll into a realized portfolio path",
        caption="Walk-forward annual-rebalance illustration: each December, the locked 60m benchmark is refit on then-observable history, the locked allocator sets weights once, and those weights are held through the next calendar year of realized monthly sleeve returns.",
        interpretation="Closest honest wealth-path view for the 5Y benchmark: annual 5Y forecasts map into fixed yearly allocations, then realized monthly returns trace the portfolio through calendar time.",
        evidence_type="historical validation",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_recent_prediction_snapshot_5y(ctx: PlotContext) -> FigureArtifact:
    frame = _recent_prediction_frame(ctx)
    anchors = sorted(frame["month_end"].dropna().unique())
    xmin = float(frame["predicted_annualized_excess_return"].min()) - 0.012
    xmax = float(frame["predicted_annualized_excess_return"].max()) + 0.012

    fig, axes = plt.subplots(1, len(anchors), figsize=(18.0, 6.0), sharex=True)
    if len(anchors) == 1:
        axes = [axes]
    for ax, month_end in zip(axes, anchors, strict=True):
        sub = frame.loc[frame["month_end"].eq(month_end)].copy()
        sub = sub.sort_values("predicted_annualized_excess_return", ascending=True).reset_index(drop=True)
        colors = [NEUTRAL_LIGHT] * len(sub)
        if len(colors) > 0:
            colors[-1] = FOCUS_COLOR
        ax.barh(
            [sleeve_label(s) for s in sub["sleeve_id"]],
            sub["predicted_annualized_excess_return"],
            color=colors,
            edgecolor="none",
            height=0.68,
        )
        ax.axvline(0.0, color=NEUTRAL_MID, linewidth=1.0)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
        ax.set_xlim(xmin, xmax)
        ax.text(
            0.5,
            1.02,
            pd.Timestamp(month_end).strftime("%Y"),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14.5,
            color=NEUTRAL_DARK,
            fontweight="bold",
        )
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
        _style_axes(ax)
    fig.supxlabel("Predicted annualized 5Y excess return")
    fig.tight_layout(pad=1.4, w_pad=1.2)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="recent_prediction_snapshot_5y")
    return FigureArtifact(
        stem="recent_prediction_snapshot_5y",
        title="Recent 5Y forecasts are differentiated across sleeves",
        caption="Forward-looking year-end 5Y forecast snapshots from the active 60m benchmark predictor.",
        interpretation="Forward-looking bridge into scenario work: current 5Y forecasts are selective rather than uniform.",
        evidence_type="recent forward-looking snapshot",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_recent_portfolio_snapshot_5y(ctx: PlotContext) -> FigureArtifact:
    frame = _recent_weight_frame(ctx)
    anchors = sorted(frame["month_end"].dropna().unique())
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    x = np.arange(len(anchors))
    bottoms = np.zeros(len(anchors))
    legend_handles: list[Patch] = []
    for sleeve_id in ctx.sleeve_order:
        vals = (
            frame.loc[frame["sleeve_id"].eq(sleeve_id)]
            .set_index("month_end")
            .reindex(anchors)["portfolio_weight"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        color = sleeve_color(sleeve_id)
        ax.bar(x, vals, bottom=bottoms, width=0.68, color=color, edgecolor="white", linewidth=0.5)
        bottoms = bottoms + vals
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=sleeve_label(sleeve_id)))
    ax.set_xticks(x, [pd.Timestamp(v).strftime("%Y") for v in anchors])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    _style_axes(ax, xlabel="Anchor year", ylabel="Portfolio weight")
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=5,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(pad=1.4)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="recent_portfolio_snapshot_5y")
    return FigureArtifact(
        stem="recent_portfolio_snapshot_5y",
        title="Recent 5Y forecasts map into diversified allocations",
        caption="Forward-looking model-implied 5Y allocations from the active 60m benchmark and the locked allocator settings.",
        interpretation="Main allocation snapshot for the scenario stage: forecasts translate into a diversified sleeve mix, not a single-theme bet.",
        evidence_type="recent forward-looking snapshot",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_active_contribution_best_60(ctx: PlotContext) -> FigureArtifact:
    df = ctx.portfolio_by_sleeve.copy()
    df = df.loc[df["strategy_label"].eq("best_60_predictor") & df["split"].eq("test")].copy()
    df = df.sort_values("avg_monthly_active_contribution_vs_equal_weight", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    colors = [sleeve_color(s) for s in df["sleeve_id"]]
    ax.barh(
        [sleeve_label(s) for s in df["sleeve_id"]],
        df["avg_monthly_active_contribution_vs_equal_weight"],
        color=colors,
        edgecolor="none",
        height=0.68,
    )
    ax.axvline(0.0, color=NEUTRAL_MID, linewidth=1.0)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_pct_fmt))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    for y, val in enumerate(df["avg_monthly_active_contribution_vs_equal_weight"].to_numpy(dtype=float)):
        pad = 0.00045 if val >= 0 else -0.00045
        ax.text(
            val + pad,
            y,
            _pct_text(val, decimals=2),
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=10.8,
            color=NEUTRAL_DARK,
        )
    _style_axes(ax, xlabel="Average active contribution vs equal weight", ylabel="Sleeve")
    fig.tight_layout(pad=1.4)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="active_contribution_best_60")
    return FigureArtifact(
        stem="active_contribution_best_60",
        title="Active behavior comes from a handful of sleeves",
        caption="Average monthly active contribution by sleeve for the carry-forward 60m portfolio benchmark.",
        interpretation="Allocation decomposition: the 60m benchmark is diversified, but not equal-weight-like; a few sleeves still drive most active behavior.",
        evidence_type="allocation interpretation",
        deck_status="main deck",
        png_path=png,
        pdf_path=pdf,
    )


def _figure_sleeve_wealth_paths_actual(ctx: PlotContext) -> FigureArtifact:
    wealth = ctx.sleeve_wealth.copy()
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    for sleeve_id in ctx.sleeve_order:
        sub = wealth.loc[wealth["sleeve_id"].eq(sleeve_id)]
        color = sleeve_color(sleeve_id)
        width = 2.4 if sleeve_id == "EQ_US" else 1.8
        alpha = 0.95 if sleeve_id in {"EQ_US", "ALT_GLD", "FI_UST"} else 0.80
        ax.plot(sub["month_end"], sub["wealth_index"], color=color, linewidth=width, alpha=alpha)
        ax.annotate(
            sleeve_label(sleeve_id),
            xy=(sub["month_end"].iloc[-1], float(sub["wealth_index"].iloc[-1])),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=10.8,
        )
    ax.set_yscale("log")
    ax.grid(color=GRID_COLOR, linewidth=0.55)
    ax.margins(x=0.08)
    _style_axes(ax, xlabel="Month", ylabel="Wealth index (log, rebased to 1)")
    fig.tight_layout(pad=1.4)
    png, pdf = save_figure(fig, out_dir=ctx.paths.plots_dir, stem="sleeve_wealth_paths_actual")
    return FigureArtifact(
        stem="sleeve_wealth_paths_actual",
        title="The sleeve universe spans distinct market histories",
        caption="Historical descriptive context for the nine active sleeves.",
        interpretation="Useful context slide if you want one market-backdrop visual before moving into model-implied forecasts.",
        evidence_type="historical validation",
        deck_status="appendix",
        png_path=png,
        pdf_path=pdf,
    )


def build_conference_figures(ctx: PlotContext) -> list[FigureArtifact]:
    apply_conference_style()
    return [
        _figure_benchmark_60_overview(ctx),
        _figure_prediction_scatter_60(ctx),
        _figure_prediction_rank_spread_60(ctx),
        _figure_rolling_prediction_quality_60(ctx),
        _figure_allocation_heatmap_best_60(ctx),
        _figure_portfolio_wealth_path_best_60(ctx),
        _figure_recent_prediction_snapshot_5y(ctx),
        _figure_recent_portfolio_snapshot_5y(ctx),
        _figure_active_contribution_best_60(ctx),
        _figure_sleeve_wealth_paths_actual(ctx),
    ]


def render_plot_index(ctx: PlotContext, artifacts: list[FigureArtifact]) -> str:
    main_deck = [art for art in artifacts if art.deck_status == "main deck"]
    appendix = [art for art in artifacts if art.deck_status == "appendix"]
    dropped_previous = [
        ("benchmark_story_compact", "drop", "Old raw-vs-robust framing; too benchmark-centric for the talk."),
        ("china_in_context", "drop", "China-as-headline framing is not part of the core story."),
        ("strategy_path_illustrative", "drop", "Overlapping-outcome cumulative path is not the cleanest conference figure."),
    ]

    lines = [
        "# XOPTPOE v3 Conference Plot Index",
        "",
        "- Active branch only: `v3_long_horizon_china`.",
        "- This figure set is presentation-oriented and centered on the active 60m benchmark.",
        "- Historical portfolio figures remain long-horizon SAA decision diagnostics, not a clean tradable monthly wealth backtest.",
        "- Titles belong on slides, not inside the plot canvas.",
        "",
        "| Filename | Intended slide title | Interpretation | Evidence type | Recommended for |",
        "| --- | --- | --- | --- | --- |",
    ]
    for art in artifacts:
        rel_png = art.png_path.relative_to(ctx.paths.project_root)
        lines.append(
            f"| `{rel_png}` | {art.title} | {art.interpretation} | {art.evidence_type} | {art.deck_status} |"
        )

    lines += [
        "",
        "## Best 5-7 Figures For The Main Deck",
        "",
    ]
    for i, art in enumerate(main_deck, start=1):
        rel_png = art.png_path.relative_to(ctx.paths.project_root)
        lines += [
            f"{i}. `{rel_png}`",
            f"   Slide title: {art.title}",
            f"   Why it matters: {art.interpretation}",
        ]

    lines += [
        "",
        "## Appendix Options",
        "",
    ]
    for art in appendix:
        rel_png = art.png_path.relative_to(ctx.paths.project_root)
        lines += [
            f"- `{rel_png}`",
            f"  Slide title: {art.title}",
            f"  Why it matters: {art.interpretation}",
        ]

    lines += [
        "",
        "## Dropped From The Core Deck",
        "",
    ]
    for stem, status, note in dropped_previous:
        lines.append(f"- `{stem}`: {status} | {note}")

    lines += [
        "",
        "## Figure Notes",
        "",
    ]
    for art in artifacts:
        rel_png = art.png_path.relative_to(ctx.paths.project_root)
        rel_pdf = art.pdf_path.relative_to(ctx.paths.project_root)
        lines += [
            f"## {art.stem}",
            f"- Intended slide title: {art.title}",
            f"- Interpretation: {art.interpretation}",
            f"- Evidence type: {art.evidence_type}",
            f"- Recommended for: {art.deck_status}",
            f"- Caption / note: {art.caption}",
            f"- PNG: `{rel_png}`",
            f"- PDF: `{rel_pdf}`",
            "",
        ]
    return "\n".join(lines)
