from __future__ import annotations

from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint
from path_planner.platform import PlannerPlatformProfile
from path_planner.postprocess.footprint import build_footprint_safe_mask
from path_planner.postprocess.models import CorridorResult

from .models import ConvexRegion, ObstaclePrimitive, RegionEdge, RegionGraph, RegionGraphReport


def build_blocked_cell_obstacles(
    grid: CostGrid,
    *,
    platform_profile: PlannerPlatformProfile | None = None,
) -> tuple[ObstaclePrimitive, ...]:
    footprint = build_footprint_safe_mask(grid, platform_profile)
    obstacles: list[ObstaclePrimitive] = []
    for obstacle_id, (y, x) in enumerate(np.argwhere(~footprint.safe_mask)):
        cell = Cell(int(x), int(y))
        min_world, max_world = _cell_world_bounds(grid, cell, cell)
        obstacles.append(
            ObstaclePrimitive(
                obstacle_id=obstacle_id,
                source="blocked_cell_box",
                min_cell=cell,
                max_cell=cell,
                min_world=min_world,
                max_world=max_world,
            )
        )
    return tuple(obstacles)


def build_merged_blocked_cell_obstacles(
    grid: CostGrid,
    *,
    platform_profile: PlannerPlatformProfile | None = None,
) -> tuple[ObstaclePrimitive, ...]:
    footprint = build_footprint_safe_mask(grid, platform_profile)
    unsafe = ~footprint.safe_mask
    visited = np.zeros(unsafe.shape, dtype=bool)
    obstacles: list[ObstaclePrimitive] = []
    for y in range(grid.spec.height):
        for x in range(grid.spec.width):
            if visited[y, x] or not bool(unsafe[y, x]):
                continue
            max_x = x
            while max_x + 1 < grid.spec.width and bool(unsafe[y, max_x + 1]) and not bool(visited[y, max_x + 1]):
                max_x += 1
            max_y = y
            while max_y + 1 < grid.spec.height:
                next_y = max_y + 1
                if any(visited[next_y, xx] or not bool(unsafe[next_y, xx]) for xx in range(x, max_x + 1)):
                    break
                max_y = next_y
            for yy in range(y, max_y + 1):
                for xx in range(x, max_x + 1):
                    visited[yy, xx] = True
            min_cell = Cell(x, y)
            max_cell = Cell(max_x, max_y)
            min_world, max_world = _cell_world_bounds(grid, min_cell, max_cell)
            obstacles.append(
                ObstaclePrimitive(
                    obstacle_id=len(obstacles),
                    source="merged_blocked_rectangle",
                    min_cell=min_cell,
                    max_cell=max_cell,
                    min_world=min_world,
                    max_world=max_world,
                )
            )
    return tuple(obstacles)


def build_grid_box_regions(grid: CostGrid, corridor: CorridorResult) -> tuple[ConvexRegion, ...]:
    regions: list[ConvexRegion] = []
    for region_id, section in enumerate(corridor.sections):
        if not section.cells:
            min_cell = section.center
            max_cell = section.center
            validation_status = "empty"
        else:
            xs = [cell.x for cell in section.cells]
            ys = [cell.y for cell in section.cells]
            min_cell = Cell(min(xs), min(ys))
            max_cell = Cell(max(xs), max(ys))
            validation_status = "valid"
        min_world, max_world = _cell_world_bounds(grid, min_cell, max_cell)
        regions.append(
            ConvexRegion(
                region_id=region_id,
                source="grid_box",
                center_cell=section.center,
                min_cell=min_cell,
                max_cell=max_cell,
                min_world=min_world,
                max_world=max_world,
                cell_count=len(section.cells),
                validation_status=validation_status,
            )
        )
    return tuple(regions)


def build_region_edges(
    regions: tuple[ConvexRegion, ...],
    *,
    edge_source: str = "grid_adjacency",
    connection_kind: str = "overlap_or_touch",
) -> tuple[RegionEdge, ...]:
    edges: list[RegionEdge] = []
    for left_index, left in enumerate(regions):
        for right in regions[left_index + 1 :]:
            if left.overlaps_or_touches(right):
                edges.append(
                    RegionEdge(
                        edge_id=len(edges),
                        source=edge_source,
                        from_region_id=left.region_id,
                        to_region_id=right.region_id,
                        connection_kind=connection_kind,
                    )
                )
    return tuple(edges)


def build_region_graph_report(
    grid: CostGrid,
    corridor: CorridorResult,
    *,
    platform_profile: PlannerPlatformProfile | None = None,
    iris_region_report: Any | None = None,
) -> RegionGraphReport:
    obstacles = build_blocked_cell_obstacles(grid, platform_profile=platform_profile)
    iris_graph_fallback_reason = None
    if iris_region_report is not None:
        iris_candidate, iris_graph_fallback_reason = _try_build_iris_region_graph_report(
            iris_region_report=iris_region_report,
            obstacles=obstacles,
        )
        if iris_candidate is not None:
            return iris_candidate

    if corridor.status != "ok":
        return RegionGraphReport(
            status="failed",
            region_source="grid_box",
            obstacle_source="blocked_cell_box",
            graph=RegionGraph(regions=(), edges=(), obstacles=obstacles),
            failure_reason=corridor.failure_reason or "corridor_unavailable",
            fallback_used=True,
            motion_feasibility_status="not_evaluated",
            quality_metrics=_quality_metrics(
                requested_region_source="grid_box",
                graph_source="grid_box",
                regions=(),
                connected_component_count=0,
                start_goal_connected=False,
                fallback_used=True,
                fallback_reason=corridor.failure_reason or "corridor_unavailable",
            ),
        )
    regions = build_grid_box_regions(grid, corridor)
    if not regions:
        return RegionGraphReport(
            status="failed",
            region_source="grid_box",
            obstacle_source="blocked_cell_box",
            graph=RegionGraph(regions=(), edges=(), obstacles=obstacles),
            failure_reason="empty_region_graph",
            fallback_used=True,
            motion_feasibility_status="not_evaluated",
            quality_metrics=_quality_metrics(
                requested_region_source="grid_box",
                graph_source="grid_box",
                regions=(),
                connected_component_count=0,
                start_goal_connected=False,
                fallback_used=True,
                fallback_reason="empty_region_graph",
            ),
        )
    graph = _build_graph(regions, obstacles=obstacles)
    component_count = _connected_component_count(graph)
    start_goal_connected = _start_goal_connected(graph)
    fallback_reason = iris_graph_fallback_reason or _iris_fallback_reason(iris_region_report)
    return RegionGraphReport(
        status="ok",
        region_source="grid_box",
        obstacle_source="blocked_cell_box",
        graph=graph,
        failure_reason=fallback_reason,
        fallback_used=iris_region_report is not None,
        motion_feasibility_status="diagnostic_only",
        quality_metrics=_quality_metrics(
            requested_region_source="iris" if iris_region_report is not None else "grid_box",
            graph_source="grid_box",
            regions=regions,
            connected_component_count=component_count,
            start_goal_connected=start_goal_connected,
            fallback_used=iris_region_report is not None,
            fallback_reason=fallback_reason,
        ),
    )


def _try_build_iris_region_graph_report(
    *,
    iris_region_report: Any,
    obstacles: tuple[ObstaclePrimitive, ...],
) -> tuple[RegionGraphReport | None, str | None]:
    iris_regions = _regions_from_iris_report(iris_region_report)
    fallback_reason = _iris_fallback_reason(iris_region_report)
    if fallback_reason is not None:
        return None, fallback_reason
    if not iris_regions:
        return None, "iris_region_graph_fallback: empty_iris_graph_candidate"
    graph = _build_graph(
        iris_regions,
        obstacles=obstacles,
        edge_source="sampled_connectivity",
        connection_kind="sampled_overlap_or_touch",
    )
    component_count = _connected_component_count(graph)
    start_goal_connected = _start_goal_connected(graph)
    if not start_goal_connected:
        return None, (
            "iris_region_graph_fallback: start_goal_not_connected "
            f"(connected_component_count={component_count})"
        )
    return (
        RegionGraphReport(
            status="ok",
            region_source="iris",
            obstacle_source="blocked_cell_box",
            graph=graph,
            failure_reason=None,
            fallback_used=False,
            motion_feasibility_status="diagnostic_only",
            quality_metrics=_quality_metrics(
                requested_region_source="iris",
                graph_source="iris",
                regions=iris_regions,
                connected_component_count=component_count,
                start_goal_connected=start_goal_connected,
                fallback_used=False,
                fallback_reason=None,
            ),
        ),
        None,
    )


def _build_graph(
    regions: tuple[ConvexRegion, ...],
    *,
    obstacles: tuple[ObstaclePrimitive, ...],
    edge_source: str = "grid_adjacency",
    connection_kind: str = "overlap_or_touch",
) -> RegionGraph:
    return RegionGraph(
        regions=regions,
        edges=build_region_edges(regions, edge_source=edge_source, connection_kind=connection_kind),
        obstacles=obstacles,
    )


def _regions_from_iris_report(iris_region_report: Any) -> tuple[ConvexRegion, ...]:
    if getattr(iris_region_report, "status", None) != "ok":
        return ()
    regions: list[ConvexRegion] = []
    for region in getattr(iris_region_report, "regions", ()):
        if (
            getattr(region, "source", None) != "iris"
            or getattr(region, "validation_status", None) != "valid"
            or getattr(region, "failure_status", None) is not None
        ):
            return ()
        min_cell = getattr(region, "min_cell")
        max_cell = getattr(region, "max_cell")
        cell_count = (max_cell.x - min_cell.x + 1) * (max_cell.y - min_cell.y + 1)
        regions.append(
            ConvexRegion(
                region_id=getattr(region, "region_id"),
                source="iris",
                center_cell=getattr(region, "seed_cell"),
                min_cell=min_cell,
                max_cell=max_cell,
                min_world=getattr(region, "min_world"),
                max_world=getattr(region, "max_world"),
                cell_count=cell_count,
                validation_status=getattr(region, "validation_status"),
                fallback_reason=getattr(region, "fallback_reason"),
            )
        )
    return tuple(regions)


def _iris_fallback_reason(iris_region_report: Any | None) -> str | None:
    if iris_region_report is None:
        return None
    if getattr(iris_region_report, "status", None) != "ok":
        reason = getattr(iris_region_report, "failure_reason", None)
        return f"iris_region_graph_fallback: {reason or 'iris_region_report_not_ok'}"
    invalid_count = sum(
        1
        for region in getattr(iris_region_report, "regions", ())
        if (
            getattr(region, "source", None) != "iris"
            or getattr(region, "validation_status", None) != "valid"
            or getattr(region, "failure_status", None) is not None
        )
    )
    if invalid_count:
        return f"iris_region_graph_fallback: {invalid_count} invalid_or_fallback_iris_regions"
    if not getattr(iris_region_report, "regions", ()):
        return "iris_region_graph_fallback: empty_iris_region_report"
    return None


def _connected_component_count(graph: RegionGraph) -> int:
    if not graph.regions:
        return 0
    adjacency: dict[int, set[int]] = {region.region_id: set() for region in graph.regions}
    for edge in graph.edges:
        adjacency.setdefault(edge.from_region_id, set()).add(edge.to_region_id)
        adjacency.setdefault(edge.to_region_id, set()).add(edge.from_region_id)
    seen: set[int] = set()
    components = 0
    for region_id in adjacency:
        if region_id in seen:
            continue
        components += 1
        stack = [region_id]
        seen.add(region_id)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return components


def _start_goal_connected(graph: RegionGraph) -> bool:
    if not graph.regions:
        return False
    start_id = graph.regions[0].region_id
    goal_id = graph.regions[-1].region_id
    if start_id == goal_id:
        return True
    adjacency: dict[int, set[int]] = {region.region_id: set() for region in graph.regions}
    for edge in graph.edges:
        adjacency.setdefault(edge.from_region_id, set()).add(edge.to_region_id)
        adjacency.setdefault(edge.to_region_id, set()).add(edge.from_region_id)
    seen = {start_id}
    stack = [start_id]
    while stack:
        current = stack.pop()
        for neighbor in adjacency.get(current, set()):
            if neighbor == goal_id:
                return True
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return False


def _quality_metrics(
    *,
    requested_region_source: str,
    graph_source: str,
    regions: tuple[ConvexRegion, ...],
    connected_component_count: int,
    start_goal_connected: bool,
    fallback_used: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    iris_region_count = sum(1 for region in regions if region.source == "iris")
    grid_fallback_region_count = sum(1 for region in regions if region.source == "grid_box")
    invalid_region_count = sum(1 for region in regions if region.validation_status != "valid")
    total_regions = len(regions)
    return {
        "requested_region_source": requested_region_source,
        "graph_source": graph_source,
        "iris_region_count": iris_region_count,
        "grid_fallback_region_count": grid_fallback_region_count,
        "invalid_region_count": invalid_region_count,
        "fallback_ratio": 0.0 if total_regions == 0 else grid_fallback_region_count / total_regions,
        "connected_component_count": connected_component_count,
        "start_goal_connected": start_goal_connected,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def _cell_world_bounds(grid: CostGrid, min_cell: Cell, max_cell: Cell) -> tuple[WorldPoint, WorldPoint]:
    resolution = grid.spec.resolution
    min_world = grid.spec.cell_to_world(min_cell)
    max_corner = Cell(max_cell.x + 1, max_cell.y + 1)
    max_world = WorldPoint(
        grid.spec.origin[0] + max_corner.x * resolution,
        grid.spec.origin[1] + max_corner.y * resolution,
    )
    return min_world, max_world
