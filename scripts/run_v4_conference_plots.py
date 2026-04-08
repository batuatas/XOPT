#!/usr/bin/env python3
"""Build the v4 presentation / diagnostics plot package."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xoptpoe_v4_plots.figures import build_v4_conference_figures
from xoptpoe_v4_plots.io import BEST_60_EXPERIMENT, BEST_60_TUNED_LABEL, TUNED_OPTIMIZER_CONFIG, load_plot_context


def _write_index(ctx, artifacts) -> None:
    out = ctx.paths.reports_dir / "conference_plot_index_v4.md"
    main = [a for a in artifacts if a.deck_status == "main deck"]
    appendix = [a for a in artifacts if a.deck_status != "main deck"]
    retired = [
        "slide01_why_long_horizon_saa_v4.png",
        "slide02_universe_v4.png",
        "slide03_features_to_ai_prediction_v4.png",
        "slide04_prediction_evidence_v4.png",
        "slide05_prediction_to_allocation_v4.png",
        "slide06_benchmark_behavior_v4.png",
        "setup_overview_v4.png",
        "prediction_allocation_overview_v4.png",
        "universe_map_v4.png",
        "feature_target_pipeline_v4.png",
        "prediction_story_v4.png",
        "allocation_story_v4.png",
    ]
    lines = [
        "# XOPTPOE v4 Conference Plot Index",
        "",
        "- Active branch only: `v4_expanded_universe`.",
        "- This index reflects the redesigned pre-scenario conference graphic package, not the full internal diagnostics set.",
        "- Portfolio visuals remain long-horizon SAA decision diagnostics, not a clean tradable monthly backtest.",
        "",
        "## Active Benchmark Object",
        f"- Prediction object: `{BEST_60_EXPERIMENT}`",
        f"- Portfolio object: `{BEST_60_TUNED_LABEL}`",
        f"- Portfolio rule: robust allocator with lambda={TUNED_OPTIMIZER_CONFIG.lambda_risk:.1f}, kappa={TUNED_OPTIMIZER_CONFIG.kappa:.2f}, omega={TUNED_OPTIMIZER_CONFIG.omega_type}.",
        "",
        "## Final Main-Deck Recommendation",
        "",
        "| Order | Filename | Final slide title | One-line interpretation | Role in story | Deck use |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, art in enumerate(main, start=1):
        lines.append(
            f"| {i} | `reports/v4_expanded_universe/plots/{art.stem}.png` | {art.title} | {art.interpretation} | {art.evidence_type} | {art.deck_status} |"
        )
    lines.extend([
        "",
        "## Appendix Recommendation",
        "",
        "| Filename | Final slide title | One-line interpretation | Role in story | Deck use |",
        "| --- | --- | --- | --- | --- |",
    ])
    if appendix:
        for art in appendix:
            lines.append(
                f"| `reports/v4_expanded_universe/plots/{art.stem}.png` | {art.title} | {art.interpretation} | {art.evidence_type} | {art.deck_status} |"
            )
    else:
        lines.append("| None | None | No appendix-only figures in the active five-graphic package. | n/a | n/a |")
    lines.extend([
        "",
        "## Recommended 5-Graphic Main-Deck Package",
    ])
    for i, art in enumerate(main, start=1):
        lines.append(f"{i}. `{art.stem}.png`")
    lines.extend([
        "",
        "## Story Anchors",
        "- Framing: `graphic01_why_long_horizon_saa_v4.png`",
        "- Universe and target: `graphic02_universe_and_target_v4.png`",
        "- Information pipeline: `graphic03_features_to_ai_prediction_v4.png`",
        "- Prediction evidence: `graphic04_prediction_evidence_v4.png`",
        "- Hero benchmark behavior: `graphic05_benchmark_behavior_v4.png`",
        "",
        "## Retired Graphics To Discard From The Conference Deck",
    ])
    for stem in retired:
        lines.append(f"- `{stem}`")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ctx = load_plot_context(PROJECT_ROOT)
    artifacts = build_v4_conference_figures(ctx)
    _write_index(ctx, artifacts)
