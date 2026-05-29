from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from path_planner.core import CostGrid, PlanResult
from path_planner.postprocess import PostprocessResult
from path_planner.postprocess.footprint import build_footprint_safe_mask


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
                "<li>Safety Corridor</li>",
                "<li>Vehicle-inflated Blocked Cells</li>",
                "<li>Smoothed Path</li>",
                "<li>Blocked Cells</li>",
                "<li>Passable Mask</li>",
                "<li>Expanded Nodes</li>",
                "<li>Summary Metrics</li>",
                "</ul>",
                "<p>In Cost + Path, dark/purple cells are lower cost and yellow cells are high cost; "
                "magenta cells mark the Safety Corridor; black cells are blocked by passable_mask; "
                "orange cells are vehicle-inflated blocked cells from the platform footprint; "
                "white line is raw A* path; cyan dashed line is smoothed path; "
                "orange X markers are curvature or turning-radius violations; green dot is start; red dot is goal.</p>",
                f'<img src="{html.escape(png.name)}" alt="diagnostics" style="max-width:100%;height:auto">',
                "<h2>Platform Constraints</h2>",
                _html_platform_summary(postprocess),
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
    if postprocess is not None and postprocess.corridor.sections:
        corridor = np.zeros(grid.spec.shape, dtype=float)
        for section in postprocess.corridor.sections:
            for cell in section.cells:
                if grid.spec.in_bounds(cell):
                    corridor[cell.y, cell.x] = 1.0
        corridor_mask = np.ma.masked_where(corridor == 0.0, corridor)
        ax.imshow(corridor_mask, cmap=ListedColormap(["magenta"]), origin="upper", alpha=0.28)
    if postprocess is not None and postprocess.platform_profile is not None:
        footprint = build_footprint_safe_mask(grid, postprocess.platform_profile)
        inflated_only = np.logical_and(grid.passable_mask, ~footprint.safe_mask)
        inflated = np.ma.masked_where(~inflated_only, np.ones(grid.spec.shape, dtype=float))
        ax.imshow(inflated, cmap=ListedColormap(["orange"]), origin="upper", alpha=0.45)
    blocked = np.ma.masked_where(grid.passable_mask, np.ones(grid.spec.shape, dtype=float))
    ax.imshow(blocked, cmap=ListedColormap(["black"]), origin="upper", alpha=0.9)
    if result.path_cells:
        xs = [cell.x for cell in result.path_cells]
        ys = [cell.y for cell in result.path_cells]
        ax.plot(xs, ys, color="white", linewidth=2, label="Raw Path")
        ax.scatter(xs[0], ys[0], c="lime", s=36, label="Start")
        ax.scatter(xs[-1], ys[-1], c="red", s=36, label="Goal")
    if postprocess is not None and postprocess.smoothed_path.cells:
        xs = [cell.x for cell in postprocess.smoothed_path.cells]
        ys = [cell.y for cell in postprocess.smoothed_path.cells]
        ax.plot(xs, ys, color="cyan", linewidth=1.5, linestyle="--", label="Smoothed Path")
    if postprocess is not None and postprocess.curvature_report.violation_indices:
        curvature_cells = (
            postprocess.smoothed_path.cells
            if postprocess.smoothed_path.status != "fallback"
            else postprocess.raw_path_cells
        )
        violation_cells = [
            curvature_cells[index]
            for index in postprocess.curvature_report.violation_indices
            if 0 <= index < len(curvature_cells)
        ]
        if violation_cells:
            ax.scatter(
                [cell.x for cell in violation_cells],
                [cell.y for cell in violation_cells],
                c="orange",
                marker="x",
                s=72,
                linewidths=2,
                label="Curvature Violation",
            )
    handles, labels = _cost_path_legend_handles(grid, result, postprocess)
    if handles:
        ax.legend(handles, labels, loc="best")
    ax.set_title("Cost + Path")


def _cost_path_legend_handles(
    grid: CostGrid,
    result: PlanResult,
    postprocess: PostprocessResult | None,
) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []
    if result.path_cells:
        handles.append(Line2D([0], [0], color="white", linewidth=2))
        labels.append("Raw Path")
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="lime", markersize=6))
        labels.append("Start")
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markersize=6))
        labels.append("Goal")
    if postprocess is not None and postprocess.smoothed_path.cells:
        handles.append(Line2D([0], [0], color="cyan", linewidth=1.5, linestyle="--"))
        labels.append("Smoothed Path")
    if postprocess is not None and postprocess.corridor.sections:
        handles.append(Patch(facecolor="magenta", alpha=0.28))
        labels.append("Safety Corridor")
    if postprocess is not None and postprocess.platform_profile is not None:
        footprint = build_footprint_safe_mask(grid, postprocess.platform_profile)
        if footprint.inflated_blocked_count > footprint.original_blocked_count:
            handles.append(Patch(facecolor="orange", alpha=0.45))
            labels.append("Vehicle-inflated Blocked Cells")
    if postprocess is not None and postprocess.curvature_report.violation_indices:
        handles.append(Line2D([0], [0], marker="x", color="orange", linestyle="none", markersize=8))
        labels.append("Curvature Violation")
    if np.any(~grid.passable_mask):
        handles.append(Patch(facecolor="black", alpha=0.9))
        labels.append("Blocked Cells")
    return handles, labels


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
    corridor = postprocess.corridor
    return "\n".join(
        [
            "<ul>",
            f"<li>corridor_status: {html.escape(postprocess.corridor.status)}</li>",
            f"<li>corridor_sections: {len(postprocess.corridor.sections)}</li>",
            f"<li>vehicle-inflated blocked cells: {corridor.inflated_blocked_count}</li>",
            f"<li>original blocked cells: {corridor.original_blocked_count}</li>",
            f"<li>footprint_radius_m: {html.escape(str(corridor.footprint_radius_m))}</li>",
            f"<li>smoothed_path_status: {html.escape(postprocess.smoothed_path.status)}</li>",
            f"<li>fallback_reason: {html.escape(str(postprocess.fallback_status.reason))}</li>",
            f"<li>curvature_report: {html.escape(report.summary)}</li>",
            f"<li>constraint_min_turning_radius: {html.escape(str(report.constraint_min_turning_radius))}</li>",
            f"<li>curvature_samples: {len(report.samples)}</li>",
            "</ul>",
            _html_warnings(postprocess),
            _html_curvature_table(postprocess),
        ]
    )


def _html_platform_summary(postprocess: PostprocessResult | None) -> str:
    if postprocess is None or postprocess.platform_profile is None:
        return "<p>platform_profile: not provided</p>"
    profile = postprocess.platform_profile
    rows = [
        "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
        "<tbody>",
        f"<tr><th>platform_key</th><td>{html.escape(profile.platform_key)}</td></tr>",
        f"<tr><th>platform_name</th><td>{html.escape(profile.platform_name)}</td></tr>",
        f"<tr><th>config_path</th><td>{html.escape(str(profile.config_path))}</td></tr>",
        f"<tr><th>body_length_m</th><td>{html.escape(str(profile.body_length_m))}</td></tr>",
        f"<tr><th>body_width_m</th><td>{html.escape(str(profile.body_width_m))}</td></tr>",
        f"<tr><th>footprint_radius_m</th><td>{html.escape(str(profile.footprint_radius_m))}</td></tr>",
        f"<tr><th>max_slope_deg</th><td>{html.escape(str(profile.max_slope_deg))}</td></tr>",
        f"<tr><th>max_obstacle_height_m</th><td>{html.escape(str(profile.max_obstacle_height_m))}</td></tr>",
        f"<tr><th>ground_clearance_m</th><td>{html.escape(str(profile.ground_clearance_m))}</td></tr>",
        f"<tr><th>effective_min_turning_radius_m</th><td>{html.escape(str(profile.effective_min_turning_radius_m))}</td></tr>",
        "</tbody>",
        "</table>",
    ]
    return "\n".join(rows)


def _html_warnings(postprocess: PostprocessResult) -> str:
    if not postprocess.constraint_warnings:
        return "<p>constraint_warnings: none</p>"
    rows = ["<h3>Constraint Warnings</h3>", "<ul>"]
    for warning in postprocess.constraint_warnings:
        rows.append(f"<li>{html.escape(warning)}</li>")
    rows.append("</ul>")
    return "\n".join(rows)


def _html_curvature_table(postprocess: PostprocessResult) -> str:
    rows = [
        "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
        "<thead><tr>",
        "<th>point_index</th>",
        "<th>turn_angle_deg</th>",
        "<th>curvature</th>",
        "<th>turning_radius</th>",
        "<th>violates</th>",
        "</tr></thead>",
        "<tbody>",
    ]
    for sample in postprocess.curvature_report.samples:
        rows.append(
            "<tr>"
            f"<td>{sample.point_index}</td>"
            f"<td>{sample.turn_angle_deg:.3f}</td>"
            f"<td>{sample.curvature:.6f}</td>"
            f"<td>{'' if sample.turning_radius is None else f'{sample.turning_radius:.3f}'}</td>"
            f"<td>{sample.violates}</td>"
            "</tr>"
        )
    if not postprocess.curvature_report.samples:
        rows.append("<tr><td colspan=\"5\">no intermediate path points</td></tr>")
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)
