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
from path_planner.postprocess import PostprocessResult
from path_planner.postprocess.footprint import build_footprint_safe_mask

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
    png_path: str | Path,
    html_path: str | Path,
) -> None:
    png = Path(png_path)
    page = Path(html_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    page.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)
    _plot_cost_path(axes[0, 0], grid, result, postprocess)
    _plot_mask(axes[0, 1], grid)
    _plot_expanded(axes[1, 0], grid, result)
    _plot_metrics(axes[1, 1], result, postprocess)
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
                "<li>Rover Footprint Scale</li>",
                "</ul>",
                "<p>In Cost + Path, Cost values use the grayscale colorbar labeled Cost; "
                "Cost + Path legend is placed outside the plot area; "
                "blue outlines mark the Safety Corridor; amber diagonal hatching marks vehicle-inflated blocked cells; "
                "black filled cells are blocked by passable_mask; white line with dark outline is raw A* path; "
                "cyan dashed line is smoothed path; red X markers are curvature or turning-radius violations; "
                "green dot is start; red dot is goal.</p>",
                "<p>In Expanded Nodes, light cells are unexpanded passable space; "
                "blue filled cells are A* expanded nodes; black filled cells are blocked cells.</p>",
                f'<img src="{html.escape(png.name)}" alt="diagnostics" style="max-width:100%;height:auto">',
                _html_rover_scale_note(postprocess),
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
    handles, labels = _cost_path_legend_handles(grid, result, postprocess)
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
