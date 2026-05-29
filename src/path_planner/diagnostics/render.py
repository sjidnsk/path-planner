from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from path_planner.core import CostGrid, PlanResult
from path_planner.postprocess import PostprocessResult


def render_diagnostics(
    grid: CostGrid,
    result: PlanResult,
    *,
    postprocess: PostprocessResult | None = None,
    png_path: str | Path,
    html_path: str | Path,
) -> None:
    png = Path(png_path)
    page = Path(html_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    page.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    _plot_cost_path(axes[0, 0], grid, result, postprocess)
    _plot_mask(axes[0, 1], grid)
    _plot_expanded(axes[1, 0], grid, result)
    _plot_metrics(axes[1, 1], result)
    fig.savefig(png, dpi=140)
    plt.close(fig)

    route = result.to_route_dict(grid.spec)
    if postprocess is not None:
        route["postprocess"] = postprocess.to_dict()
    summary = html.escape(json.dumps(route, ensure_ascii=False, indent=2))
    postprocess_summary = _html_postprocess_summary(postprocess)
    page.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html><head><meta charset="utf-8"><title>Path Planner Diagnostics</title></head><body>',
                "<h1>Path Planner Diagnostics</h1>",
                "<p>trajectory_kind: <strong>geometric_path</strong></p>",
                "<h2>Panels</h2>",
                "<ul>",
                "<li>Cost + Path</li>",
                "<li>Smoothed Path</li>",
                "<li>Blocked Cells</li>",
                "<li>Passable Mask</li>",
                "<li>Expanded Nodes</li>",
                "<li>Summary Metrics</li>",
                "</ul>",
                "<p>In Cost + Path, yellow cells are high cost; black cells are blocked by passable_mask.</p>",
                f'<img src="{html.escape(png.name)}" alt="diagnostics" style="max-width:100%;height:auto">',
                "<h2>Postprocess Summary</h2>",
                postprocess_summary,
                "<h2>Route JSON</h2>",
                f"<pre>{summary}</pre>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def _plot_cost_path(ax, grid: CostGrid, result: PlanResult, postprocess: PostprocessResult | None) -> None:
    ax.imshow(grid.cost, cmap="viridis", origin="upper")
    blocked = np.ma.masked_where(grid.passable_mask, np.ones(grid.spec.shape, dtype=float))
    ax.imshow(blocked, cmap=ListedColormap(["black"]), origin="upper", alpha=0.85)
    if result.path_cells:
        xs = [cell.x for cell in result.path_cells]
        ys = [cell.y for cell in result.path_cells]
        ax.plot(xs, ys, color="white", linewidth=2, label="Raw Path")
        ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], c=["lime", "red"], s=36)
    if postprocess is not None and postprocess.smoothed_path.cells:
        xs = [cell.x for cell in postprocess.smoothed_path.cells]
        ys = [cell.y for cell in postprocess.smoothed_path.cells]
        ax.plot(xs, ys, color="cyan", linewidth=1.5, linestyle="--", label="Smoothed Path")
    handles, labels = ax.get_legend_handles_labels()
    if np.any(~grid.passable_mask):
        handles.append(Patch(facecolor="black", alpha=0.85, label="Blocked Cells"))
        labels.append("Blocked Cells")
    if handles:
        ax.legend(handles, labels, loc="best")
    ax.set_title("Cost + Path")


def _plot_mask(ax, grid: CostGrid) -> None:
    ax.imshow(grid.passable_mask, cmap="gray", origin="upper")
    ax.set_title("Passable Mask")


def _plot_expanded(ax, grid: CostGrid, result: PlanResult) -> None:
    expanded = np.zeros(grid.spec.shape, dtype=float)
    for cell in result.diagnostics.expanded_cells:
        if grid.spec.in_bounds(cell):
            expanded[cell.y, cell.x] = 1.0
    ax.imshow(expanded, cmap="magma", origin="upper")
    ax.set_title("Expanded Nodes")


def _plot_metrics(ax, result: PlanResult) -> None:
    ax.axis("off")
    lines = [
        "Summary Metrics",
        f"success: {result.success}",
        f"failure_reason: {result.failure_reason.value if result.failure_reason else ''}",
        f"total_cost: {result.total_cost}",
        f"expanded_count: {result.expanded_count}",
        f"path_nodes: {len(result.path_cells)}",
        f"runtime_ms: {result.diagnostics.runtime_ms:.3f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace")


def _html_postprocess_summary(postprocess: PostprocessResult | None) -> str:
    if postprocess is None:
        return "<p>postprocess: not run</p>"
    report = postprocess.curvature_report
    return "\n".join(
        [
            "<ul>",
            f"<li>corridor_status: {html.escape(postprocess.corridor.status)}</li>",
            f"<li>smoothed_path_status: {html.escape(postprocess.smoothed_path.status)}</li>",
            f"<li>fallback_reason: {html.escape(str(postprocess.fallback_status.reason))}</li>",
            f"<li>curvature_report: {html.escape(report.summary)}</li>",
            "</ul>",
        ]
    )
