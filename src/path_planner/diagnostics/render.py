from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from path_planner.core import CostGrid, PlanResult
from path_planner.optimization import (
    TrajectoryOptimizationResult,
    build_tracking_metric_comparison,
    merge_tracking_comparison,
)
from path_planner.postprocess import PostprocessResult
from path_planner.postprocess.footprint import build_footprint_safe_mask
from path_planner.tracking import TrackingSimulationResult

DIAGNOSTIC_COLORS = {
    "surface": "#f8fafc",
    "cost_low": "#f8fafc",
    "cost_high": "#64748b",
    "corridor": "#2563eb",
    "inflated": "#d97706",
    "blocked": "#020617",
    "raw_path": "#ffffff",
    "raw_path_outline": "#111827",
    "smoothed_path": "#06b6d4",
    "trackable_path": "#a855f7",
    "simulated_path": "#f97316",
    "optimized_path": "#22c55e",
    "start": "#16a34a",
    "goal": "#dc2626",
    "violation": "#dc2626",
    "expanded": "#2563eb",
}

COST_CMAP = LinearSegmentedColormap.from_list(
    "lunar_cost_neutral",
    [DIAGNOSTIC_COLORS["cost_low"], DIAGNOSTIC_COLORS["cost_high"]],
)


def render_diagnostics(
    grid: CostGrid,
    result: PlanResult,
    *,
    postprocess: PostprocessResult | None = None,
    tracking_simulation: TrackingSimulationResult | None = None,
    trajectory_optimization: TrajectoryOptimizationResult | None = None,
    optimized_tracking_simulation: TrackingSimulationResult | None = None,
    png_path: str | Path,
    html_path: str | Path,
) -> None:
    png = Path(png_path)
    page = Path(html_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    page.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)
    _plot_cost_path(axes[0, 0], grid, result, postprocess, tracking_simulation, trajectory_optimization)
    _plot_mask(axes[0, 1], grid)
    _plot_expanded(axes[1, 0], grid, result)
    _plot_metrics(axes[1, 1], result, postprocess)
    fig.savefig(png, dpi=140)
    plt.close(fig)

    route = result.to_route_dict(grid.spec)
    if postprocess is not None:
        route["postprocess"] = postprocess.to_dict()
    if tracking_simulation is not None:
        route["tracking_simulation_report"] = tracking_simulation.to_dict()
    if trajectory_optimization is not None:
        route["trajectory_optimization_report"] = merge_tracking_comparison(
            trajectory_optimization.to_dict(),
            tracking_simulation,
            optimized_tracking_simulation,
        )
    if optimized_tracking_simulation is not None:
        route["optimized_tracking_simulation_report"] = optimized_tracking_simulation.to_dict()
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
                "<li>Trackable Path</li>",
                "<li>Simulated Tracking Path</li>",
                "<li>Optimized Path</li>",
                "<li>Blocked Cells</li>",
                "<li>Passable Mask</li>",
                "<li>Expanded Nodes</li>",
                "<li>Summary Metrics</li>",
                "<li>Rover Footprint Scale</li>",
                "</ul>",
                "<p>In Cost + Path, Cost values use the grayscale colorbar labeled Cost; "
                "Cost + Path legend is placed outside the plot area; "
                "blue outlines mark the Safety Corridor; amber diagonal hatching marks vehicle-inflated blocked cells; "
                "black filled cells are blocked by passable_mask; white line with dark outline is raw A* path; "
                "cyan dashed line is smoothed path; purple markers/arrows are trackable waypoints; "
                "orange line is simulated tracking path; "
                "green line is optimized path; "
                "red X markers are curvature or turning-radius violations; "
                "red square markers are tracking-safety violations; "
                "green dot is start; red dot is goal.</p>",
                "<p>In Expanded Nodes, light cells are unexpanded passable space; "
                "blue filled cells are A* expanded nodes; black filled cells are blocked cells.</p>",
                f'<img src="{html.escape(png.name)}" alt="diagnostics" style="max-width:100%;height:auto">',
                _html_rover_scale_note(postprocess),
                "<h2>Search Constraints</h2>",
                _html_search_summary(result),
                "<h2>Platform Constraints</h2>",
                _html_platform_summary(postprocess),
                "<h2>Postprocess Summary</h2>",
                postprocess_summary,
                "<h2>Trackable Path</h2>",
                _html_trackable_path_summary(postprocess),
                "<h2>Tracking Safety Summary</h2>",
                _html_tracking_safety_summary(postprocess),
                "<h2>Tracking Simulation Summary</h2>",
                _html_tracking_simulation_summary(tracking_simulation),
                "<h2>Trajectory Optimization Summary</h2>",
                _html_trajectory_optimization_summary(
                    trajectory_optimization,
                    tracking_simulation,
                    optimized_tracking_simulation,
                ),
                "<h2>Route JSON</h2>",
                f"<pre>{summary}</pre>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def _plot_cost_path(
    ax,
    grid: CostGrid,
    result: PlanResult,
    postprocess: PostprocessResult | None,
    tracking_simulation: TrackingSimulationResult | None,
    trajectory_optimization: TrajectoryOptimizationResult | None,
) -> None:
    image = ax.imshow(grid.cost, cmap=COST_CMAP, origin="upper")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("Cost", rotation=90)
    if postprocess is not None and postprocess.corridor.sections:
        corridor = np.zeros(grid.spec.shape, dtype=float)
        for section in postprocess.corridor.sections:
            for cell in section.cells:
                if grid.spec.in_bounds(cell):
                    corridor[cell.y, cell.x] = 1.0
        _draw_cell_outline(
            ax,
            corridor.astype(bool),
            edgecolor=DIAGNOSTIC_COLORS["corridor"],
            linewidth=1.3,
            linestyle="--",
            zorder=2.0,
        )
    if postprocess is not None and postprocess.platform_profile is not None:
        footprint = build_footprint_safe_mask(grid, postprocess.platform_profile)
        inflated_only = np.logical_and(grid.passable_mask, ~footprint.safe_mask)
        _draw_cell_hatch(
            ax,
            inflated_only,
            edgecolor=DIAGNOSTIC_COLORS["inflated"],
            hatch="////",
            linewidth=0.7,
            zorder=2.5,
        )
    _draw_cell_fill(
        ax,
        ~grid.passable_mask,
        facecolor=DIAGNOSTIC_COLORS["blocked"],
        edgecolor="#ffffff",
        linewidth=0.25,
        zorder=3.0,
    )
    if result.path_cells:
        xs = [cell.x for cell in result.path_cells]
        ys = [cell.y for cell in result.path_cells]
        ax.plot(
            xs,
            ys,
            color=DIAGNOSTIC_COLORS["raw_path"],
            linewidth=2,
            label="Raw Path",
            path_effects=[
                path_effects.Stroke(linewidth=4, foreground=DIAGNOSTIC_COLORS["raw_path_outline"]),
                path_effects.Normal(),
            ],
        )
        ax.scatter(
            xs[0],
            ys[0],
            c=DIAGNOSTIC_COLORS["start"],
            edgecolors=DIAGNOSTIC_COLORS["raw_path"],
            linewidths=1,
            s=42,
            label="Start",
        )
        ax.scatter(
            xs[-1],
            ys[-1],
            c=DIAGNOSTIC_COLORS["goal"],
            edgecolors=DIAGNOSTIC_COLORS["raw_path"],
            linewidths=1,
            s=42,
            label="Goal",
        )
    if postprocess is not None and postprocess.smoothed_path.cells:
        xs = [cell.x for cell in postprocess.smoothed_path.cells]
        ys = [cell.y for cell in postprocess.smoothed_path.cells]
        ax.plot(
            xs,
            ys,
            color=DIAGNOSTIC_COLORS["smoothed_path"],
            linewidth=1.7,
            linestyle="--",
            label="Smoothed Path",
            path_effects=[
                path_effects.Stroke(linewidth=3, foreground=DIAGNOSTIC_COLORS["raw_path_outline"]),
                path_effects.Normal(),
            ],
        )
    if postprocess is not None and postprocess.trackable_path is not None and postprocess.trackable_path.waypoints:
        waypoints = postprocess.trackable_path.waypoints
        xs = [waypoint.cell.x for waypoint in waypoints]
        ys = [waypoint.cell.y for waypoint in waypoints]
        ax.scatter(
            xs,
            ys,
            c=DIAGNOSTIC_COLORS["trackable_path"],
            edgecolors=DIAGNOSTIC_COLORS["raw_path"],
            linewidths=0.7,
            s=28,
            marker="D",
            label="Trackable Waypoint",
            zorder=5.0,
        )
        ax.quiver(
            xs,
            ys,
            [np.cos(waypoint.heading_rad) for waypoint in waypoints],
            [np.sin(waypoint.heading_rad) for waypoint in waypoints],
            color=DIAGNOSTIC_COLORS["trackable_path"],
            angles="xy",
            scale_units="xy",
            scale=3.0,
            width=0.006,
            zorder=5.1,
        )
    if tracking_simulation is not None and tracking_simulation.states:
        xs = [_world_x_to_grid_x(grid, state.x) for state in tracking_simulation.states]
        ys = [_world_y_to_grid_y(grid, state.y) for state in tracking_simulation.states]
        ax.plot(
            xs,
            ys,
            color=DIAGNOSTIC_COLORS["simulated_path"],
            linewidth=1.8,
            label="Simulated Tracking Path",
            zorder=5.2,
            path_effects=[
                path_effects.Stroke(linewidth=3.0, foreground=DIAGNOSTIC_COLORS["raw_path_outline"]),
                path_effects.Normal(),
            ],
        )
    if trajectory_optimization is not None and trajectory_optimization.optimized_path:
        xs = [_world_x_to_grid_x(grid, point.x) for point in trajectory_optimization.optimized_path]
        ys = [_world_y_to_grid_y(grid, point.y) for point in trajectory_optimization.optimized_path]
        ax.plot(
            xs,
            ys,
            color=DIAGNOSTIC_COLORS["optimized_path"],
            linewidth=2.1,
            label="Optimized Path",
            zorder=5.25,
            path_effects=[
                path_effects.Stroke(linewidth=3.5, foreground=DIAGNOSTIC_COLORS["raw_path_outline"]),
                path_effects.Normal(),
            ],
        )
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
                c=DIAGNOSTIC_COLORS["violation"],
                marker="x",
                s=82,
                linewidths=2.2,
                label="Curvature Violation",
            )
    if (
        postprocess is not None
        and postprocess.trackable_path is not None
        and postprocess.tracking_safety_report is not None
        and postprocess.tracking_safety_report.violation_indices
    ):
        waypoints = postprocess.trackable_path.waypoints
        violation_waypoints = [
            waypoints[index]
            for index in postprocess.tracking_safety_report.violation_indices
            if 0 <= index < len(waypoints)
        ]
        if violation_waypoints:
            ax.scatter(
                [waypoint.cell.x for waypoint in violation_waypoints],
                [waypoint.cell.y for waypoint in violation_waypoints],
                c=DIAGNOSTIC_COLORS["violation"],
                marker="s",
                s=78,
                linewidths=1.2,
                edgecolors=DIAGNOSTIC_COLORS["raw_path"],
                label="Tracking Safety Violation",
                zorder=5.3,
            )
    if (
        tracking_simulation is not None
        and tracking_simulation.safety_report.violation_indices
        and tracking_simulation.states
    ):
        violation_states = [
            tracking_simulation.states[index]
            for index in tracking_simulation.safety_report.violation_indices
            if 0 <= index < len(tracking_simulation.states)
        ]
        if violation_states:
            ax.scatter(
                [_world_x_to_grid_x(grid, state.x) for state in violation_states],
                [_world_y_to_grid_y(grid, state.y) for state in violation_states],
                c=DIAGNOSTIC_COLORS["violation"],
                marker="P",
                s=72,
                linewidths=1.2,
                edgecolors=DIAGNOSTIC_COLORS["raw_path"],
                label="Tracking Simulation Violation",
                zorder=5.4,
            )
    handles, labels = _cost_path_legend_handles(grid, result, postprocess, tracking_simulation, trajectory_optimization)
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            fontsize=8,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            framealpha=0.94,
            borderaxespad=0.0,
        )
    ax.set_title("Cost + Path")


def _cost_path_legend_handles(
    grid: CostGrid,
    result: PlanResult,
    postprocess: PostprocessResult | None,
    tracking_simulation: TrackingSimulationResult | None,
    trajectory_optimization: TrajectoryOptimizationResult | None,
) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []
    handles.append(Patch(facecolor=DIAGNOSTIC_COLORS["cost_high"], alpha=0.45))
    labels.append("Cost Heatmap")
    if result.path_cells:
        handles.append(Line2D([0], [0], color=DIAGNOSTIC_COLORS["raw_path_outline"], linewidth=3))
        labels.append("Raw Path")
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=DIAGNOSTIC_COLORS["start"], markersize=6)
        )
        labels.append("Start")
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=DIAGNOSTIC_COLORS["goal"], markersize=6)
        )
        labels.append("Goal")
    if postprocess is not None and postprocess.smoothed_path.cells:
        handles.append(Line2D([0], [0], color=DIAGNOSTIC_COLORS["smoothed_path"], linewidth=1.7, linestyle="--"))
        labels.append("Smoothed Path")
    if postprocess is not None and postprocess.trackable_path is not None and postprocess.trackable_path.waypoints:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=DIAGNOSTIC_COLORS["trackable_path"],
                markersize=6,
            )
        )
        labels.append("Trackable Path")
    if tracking_simulation is not None and tracking_simulation.states:
        handles.append(Line2D([0], [0], color=DIAGNOSTIC_COLORS["simulated_path"], linewidth=1.8))
        labels.append("Simulated Tracking Path")
    if trajectory_optimization is not None and trajectory_optimization.optimized_path:
        handles.append(Line2D([0], [0], color=DIAGNOSTIC_COLORS["optimized_path"], linewidth=2.1))
        labels.append("Optimized Path")
    if postprocess is not None and postprocess.corridor.sections:
        handles.append(Patch(facecolor="none", edgecolor=DIAGNOSTIC_COLORS["corridor"], linestyle="--", linewidth=1.3))
        labels.append("Safety Corridor")
    if postprocess is not None and postprocess.platform_profile is not None:
        footprint = build_footprint_safe_mask(grid, postprocess.platform_profile)
        if footprint.inflated_blocked_count > footprint.original_blocked_count:
            handles.append(Patch(facecolor="none", edgecolor=DIAGNOSTIC_COLORS["inflated"], hatch="////", linewidth=0.7))
            labels.append("Vehicle-inflated Blocked Cells")
    if postprocess is not None and postprocess.curvature_report.violation_indices:
        handles.append(
            Line2D([0], [0], marker="x", color=DIAGNOSTIC_COLORS["violation"], linestyle="none", markersize=8)
        )
        labels.append("Curvature Violation")
    if (
        postprocess is not None
        and postprocess.tracking_safety_report is not None
        and postprocess.tracking_safety_report.violation_indices
    ):
        handles.append(
            Line2D([0], [0], marker="s", color=DIAGNOSTIC_COLORS["violation"], linestyle="none", markersize=7)
        )
        labels.append("Tracking Safety Violation")
    if (
        tracking_simulation is not None
        and tracking_simulation.safety_report.violation_indices
    ):
        handles.append(
            Line2D([0], [0], marker="P", color=DIAGNOSTIC_COLORS["violation"], linestyle="none", markersize=7)
        )
        labels.append("Tracking Simulation Violation")
    if np.any(~grid.passable_mask):
        handles.append(Patch(facecolor=DIAGNOSTIC_COLORS["blocked"], alpha=0.94))
        labels.append("Blocked Cells")
    return handles, labels


def _draw_cell_fill(ax, mask: np.ndarray, *, facecolor: str, edgecolor: str, linewidth: float, zorder: float) -> None:
    for y, x in np.argwhere(mask):
        ax.add_patch(
            Rectangle(
                (float(x) - 0.5, float(y) - 0.5),
                1.0,
                1.0,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder,
            )
        )


def _draw_cell_outline(
    ax,
    mask: np.ndarray,
    *,
    edgecolor: str,
    linewidth: float,
    linestyle: str,
    zorder: float,
) -> None:
    for y, x in np.argwhere(mask):
        ax.add_patch(
            Rectangle(
                (float(x) - 0.5, float(y) - 0.5),
                1.0,
                1.0,
                facecolor="none",
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
            )
        )


def _draw_cell_hatch(
    ax,
    mask: np.ndarray,
    *,
    edgecolor: str,
    hatch: str,
    linewidth: float,
    zorder: float,
) -> None:
    for y, x in np.argwhere(mask):
        ax.add_patch(
            Rectangle(
                (float(x) - 0.5, float(y) - 0.5),
                1.0,
                1.0,
                facecolor="none",
                edgecolor=edgecolor,
                hatch=hatch,
                linewidth=linewidth,
                zorder=zorder,
            )
        )


def _world_x_to_grid_x(grid: CostGrid, x: float) -> float:
    return (x - grid.spec.origin[0]) / grid.spec.resolution


def _world_y_to_grid_y(grid: CostGrid, y: float) -> float:
    return (y - grid.spec.origin[1]) / grid.spec.resolution


def _plot_mask(ax, grid: CostGrid) -> None:
    ax.imshow(
        grid.passable_mask,
        cmap=ListedColormap([DIAGNOSTIC_COLORS["blocked"], DIAGNOSTIC_COLORS["surface"]]),
        origin="upper",
    )
    ax.set_title("Passable Mask")


def _plot_expanded(ax, grid: CostGrid, result: PlanResult) -> None:
    categories = np.zeros(grid.spec.shape, dtype=int)
    categories[~grid.passable_mask] = 1
    for cell in result.diagnostics.expanded_cells:
        if grid.spec.in_bounds(cell) and grid.is_passable(cell):
            categories[cell.y, cell.x] = 2
    ax.imshow(
        categories,
        cmap=ListedColormap(
            [
                DIAGNOSTIC_COLORS["surface"],
                DIAGNOSTIC_COLORS["blocked"],
                DIAGNOSTIC_COLORS["expanded"],
            ]
        ),
        vmin=0,
        vmax=2,
        origin="upper",
        interpolation="nearest",
    )
    ax.legend(
        [
            Patch(facecolor=DIAGNOSTIC_COLORS["surface"], edgecolor="#cbd5e1"),
            Patch(facecolor=DIAGNOSTIC_COLORS["expanded"]),
            Patch(facecolor=DIAGNOSTIC_COLORS["blocked"]),
        ],
        ["Unexpanded Passable", "A* Expanded Node", "Blocked Cell"],
        loc="best",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        framealpha=0.92,
    )
    ax.set_title("Expanded Nodes")


def _plot_metrics(ax, result: PlanResult, postprocess: PostprocessResult | None) -> None:
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
    _draw_rover_footprint_scale(ax, postprocess)


def _draw_rover_footprint_scale(ax, postprocess: PostprocessResult | None) -> None:
    if postprocess is None or postprocess.platform_profile is None:
        return
    profile = postprocess.platform_profile
    if (
        profile.body_length_m is None
        or profile.body_width_m is None
        or profile.body_length_m <= 0.0
        or profile.body_width_m <= 0.0
    ):
        return

    body_length = profile.body_length_m
    body_width = profile.body_width_m
    footprint_radius = profile.footprint_radius_m
    box_width = 0.30
    box_height = box_width * min(body_width / body_length, 1.0)
    left = 0.60
    bottom = 0.28
    center = (left + box_width / 2.0, bottom + box_height / 2.0)
    circle_radius = (box_width**2 + box_height**2) ** 0.5 / 2.0

    ax.text(
        left,
        bottom + box_height + 0.16,
        "Rover Footprint Scale",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    ax.add_patch(
        Circle(
            center,
            circle_radius,
            transform=ax.transAxes,
            fill=False,
            linestyle="--",
            linewidth=1.4,
            edgecolor=DIAGNOSTIC_COLORS["inflated"],
        )
    )
    ax.add_patch(
        Rectangle(
            (left, bottom),
            box_width,
            box_height,
            transform=ax.transAxes,
            facecolor="#dbeafe",
            edgecolor=DIAGNOSTIC_COLORS["corridor"],
            linewidth=1.6,
            alpha=0.92,
        )
    )
    ax.annotate(
        "",
        xy=(left, bottom - 0.055),
        xytext=(left + box_width, bottom - 0.055),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "<->", "color": DIAGNOSTIC_COLORS["raw_path_outline"], "linewidth": 1.0},
    )
    ax.text(
        left + box_width / 2.0,
        bottom - 0.095,
        f"{body_length:.2f} m length",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
    )
    ax.annotate(
        "",
        xy=(left - 0.045, bottom),
        xytext=(left - 0.045, bottom + box_height),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "<->", "color": DIAGNOSTIC_COLORS["raw_path_outline"], "linewidth": 1.0},
    )
    ax.text(
        left - 0.075,
        bottom + box_height / 2.0,
        f"{body_width:.2f} m width",
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=7.5,
    )
    if footprint_radius is not None:
        ax.text(
            left,
            bottom - 0.17,
            f"footprint radius: {footprint_radius:.2f} m",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color=DIAGNOSTIC_COLORS["inflated"],
        )


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
            f"<li>trackable_waypoints: {0 if postprocess.trackable_path is None else len(postprocess.trackable_path.waypoints)}</li>",
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


def _html_trackable_path_summary(postprocess: PostprocessResult | None) -> str:
    if postprocess is None or postprocess.trackable_path is None:
        return "<p>trackable_path: not generated</p>"
    path = postprocess.trackable_path
    rows = [
        "<ul>",
        f"<li>source_path: {html.escape(path.source_path)}</li>",
        f"<li>waypoint_count: {len(path.waypoints)}</li>",
        f"<li>length_m: {path.length_m:.3f}</li>",
        f"<li>max_curvature: {path.max_curvature:.6f}</li>",
        f"<li>min_turning_radius_m: {html.escape(str(path.min_turning_radius_m))}</li>",
        f"<li>speed_profile: {html.escape(str([round(value, 4) for value in path.speed_profile]))}</li>",
        "</ul>",
        "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
        "<thead><tr>",
        "<th>index</th>",
        "<th>cell</th>",
        "<th>heading_rad</th>",
        "<th>turn_angle_deg</th>",
        "<th>curvature</th>",
        "<th>speed_mps</th>",
        "</tr></thead>",
        "<tbody>",
    ]
    for waypoint in path.waypoints:
        rows.append(
            "<tr>"
            f"<td>{waypoint.index}</td>"
            f"<td>{html.escape(str(waypoint.cell.to_list()))}</td>"
            f"<td>{waypoint.heading_rad:.3f}</td>"
            f"<td>{waypoint.turn_angle_deg:.3f}</td>"
            f"<td>{waypoint.curvature:.6f}</td>"
            f"<td>{waypoint.recommended_speed_mps:.4f}</td>"
            "</tr>"
        )
    if not path.waypoints:
        rows.append("<tr><td colspan=\"6\">no trackable waypoints</td></tr>")
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _html_tracking_safety_summary(postprocess: PostprocessResult | None) -> str:
    if postprocess is None or postprocess.tracking_safety_report is None:
        return "<p>tracking_safety_report: not generated</p>"
    report = postprocess.tracking_safety_report
    return "\n".join(
        [
            "<ul>",
            f"<li>is_safe: {report.is_safe}</li>",
            f"<li>tracking_error_bound_m: {report.tracking_error_bound_m}</li>",
            f"<li>checked_radius_m: {report.checked_radius_m}</li>",
            f"<li>min_clearance_m: {html.escape(str(report.min_clearance_m))}</li>",
            f"<li>violation_indices: {html.escape(str(list(report.violation_indices)))}</li>",
            f"<li>summary: {html.escape(report.summary)}</li>",
            "</ul>",
        ]
    )


def _html_tracking_simulation_summary(tracking_simulation: TrackingSimulationResult | None) -> str:
    if tracking_simulation is None:
        return "<p>tracking_simulation_report: not run</p>"
    metrics = tracking_simulation.metrics
    safety = tracking_simulation.safety_report
    return "\n".join(
        [
            "<ul>",
            f"<li>states: {len(tracking_simulation.states)}</li>",
            f"<li>is_safe: {safety.is_safe}</li>",
            f"<li>path_length_m: {metrics.path_length_m:.3f}</li>",
            f"<li>simulated_length_m: {metrics.simulated_length_m:.3f}</li>",
            f"<li>max_cross_track_error_m: {metrics.max_cross_track_error_m:.6f}</li>",
            f"<li>min_clearance_m: {html.escape(str(metrics.min_clearance_m))}</li>",
            f"<li>safety_violation_count: {metrics.safety_violation_count}</li>",
            f"<li>curvature_violation_count: {metrics.curvature_violation_count}</li>",
            f"<li>mean_speed_mps: {metrics.mean_speed_mps:.4f}</li>",
            f"<li>high_cost_exposure: {metrics.high_cost_exposure:.3f}</li>",
            f"<li>summary: {html.escape(safety.summary)}</li>",
            "</ul>",
        ]
    )


def _html_trajectory_optimization_summary(
    trajectory_optimization: TrajectoryOptimizationResult | None,
    tracking_simulation: TrackingSimulationResult | None,
    optimized_tracking_simulation: TrackingSimulationResult | None,
) -> str:
    if trajectory_optimization is None:
        return "<p>trajectory_optimization_report: not run</p>"
    metrics = trajectory_optimization.metrics
    fallback = trajectory_optimization.fallback_status
    comparison = build_tracking_metric_comparison(tracking_simulation, optimized_tracking_simulation)
    rows = [
        "<ul>",
        f"<li>solver_status: {html.escape(trajectory_optimization.solver_status)}</li>",
        f"<li>fallback_status: {html.escape(str(fallback.to_dict()))}</li>",
        f"<li>optimized_waypoints: {len(trajectory_optimization.optimized_path)}</li>",
        f"<li>reference_path_length_m: {metrics.reference_path_length_m:.3f}</li>",
        f"<li>optimized_path_length_m: {metrics.optimized_path_length_m:.3f}</li>",
        f"<li>length_delta_m: {metrics.length_delta_m:.3f}</li>",
        f"<li>reference_high_cost_exposure: {metrics.reference_high_cost_exposure:.3f}</li>",
        f"<li>optimized_high_cost_exposure: {metrics.optimized_high_cost_exposure:.3f}</li>",
        f"<li>high_cost_exposure_delta: {metrics.high_cost_exposure_delta:.3f}</li>",
        f"<li>curvature_violation_count: {metrics.curvature_violation_count}</li>",
        f"<li>objective_initial: {metrics.objective_initial:.6f}</li>",
        f"<li>objective_final: {metrics.objective_final:.6f}</li>",
        f"<li>solver_iterations: {metrics.solver_iterations}</li>",
        f"<li>is_within_corridor: {metrics.is_within_corridor}</li>",
        "</ul>",
        "<h3>baseline_vs_optimized</h3>",
    ]
    if comparison:
        rows.extend(["<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">", "<thead><tr>"])
        rows.extend(["<th>metric</th>", "<th>baseline</th>", "<th>optimized</th>", "<th>delta</th>"])
        rows.extend(["</tr></thead>", "<tbody>"])
        for metric, values in comparison.items():
            rows.append(
                "<tr>"
                f"<td>{html.escape(metric)}</td>"
                f"<td>{html.escape(str(values['baseline']))}</td>"
                f"<td>{html.escape(str(values['optimized']))}</td>"
                f"<td>{html.escape(str(values['delta']))}</td>"
                "</tr>"
            )
        rows.extend(["</tbody>", "</table>"])
    else:
        rows.append(f"<pre>{html.escape(json.dumps(metrics.baseline_vs_optimized, ensure_ascii=False, indent=2))}</pre>")
    return "\n".join(rows)


def _html_search_summary(result: PlanResult) -> str:
    diagnostics = result.diagnostics
    rows = [
        "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
        "<tbody>",
        f"<tr><th>search_mode</th><td>{html.escape(diagnostics.search_mode)}</td></tr>",
        f"<tr><th>passable_source</th><td>{html.escape(diagnostics.passable_source)}</td></tr>",
        f"<tr><th>platform_key</th><td>{html.escape(str(diagnostics.platform_key))}</td></tr>",
        f"<tr><th>footprint_radius_m</th><td>{html.escape(str(diagnostics.footprint_radius_m))}</td></tr>",
        f"<tr><th>original_blocked_count</th><td>{html.escape(str(diagnostics.original_blocked_count))}</td></tr>",
        f"<tr><th>inflated_blocked_count</th><td>{html.escape(str(diagnostics.inflated_blocked_count))}</td></tr>",
        f"<tr><th>terrain_layers</th><td>{html.escape(', '.join(diagnostics.terrain_layers))}</td></tr>",
        "</tbody>",
        "</table>",
    ]
    return "\n".join(rows)


def _html_rover_scale_note(postprocess: PostprocessResult | None) -> str:
    if postprocess is None or postprocess.platform_profile is None:
        return ""
    return (
        "<p><strong>Rover Footprint Scale:</strong> "
        "rover body length/width and footprint radius are drawn from platform_profile.</p>"
    )


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
