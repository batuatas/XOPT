from __future__ import annotations

from pathlib import Path

from xoptpoe_v3_modeling.io import write_text
from xoptpoe_v3_plots.figures import build_conference_figures, render_plot_index
from xoptpoe_v3_plots.io import load_plot_context


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ctx = load_plot_context(project_root)
    artifacts = build_conference_figures(ctx)
    index_text = render_plot_index(ctx, artifacts)
    write_text(index_text, ctx.paths.reports_dir / "conference_plot_index.md")


if __name__ == "__main__":
    main()
