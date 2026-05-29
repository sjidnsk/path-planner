from __future__ import annotations

import math

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint
from path_planner.platform import PlannerPlatformProfile

from .models import TrackablePath, TrackableWaypoint, TrackingSafetyReport


def build_trackable_path(
    grid: CostGrid,
    path_cells: tuple[Cell, ...],
    *,
    source_path: str,
    max_speed_mps: float,
    min_speed_mps: float,
) -> TrackablePath:
    if max_speed_mps <= 0.0:
        raise ValueError("max_speed_mps must be positive")
    if min_speed_mps <= 0.0:
        raise ValueError("min_speed_mps must be positive")
    if min_speed_mps > max_speed_mps:
        raise ValueError("min_speed_mps cannot exceed max_speed_mps")

    world_points = tuple(grid.spec.cell_to_world(cell) for cell in path_cells)
    waypoints: list[TrackableWaypoint] = []
    for index, cell in enumerate(path_cells):
        heading = _heading_at(world_points, index)
        segment_length = 0.0 if index == 0 else _distance(world_points[index - 1], world_points[index])
        turn_angle = _turn_angle_deg(world_points, index)
        curvature = _curvature(world_points, index)
        turning_radius = 1.0 / curvature if curvature > 0.0 else None
        speed = _recommended_speed(
            grid,
            cell,
            curvature=curvature,
            max_speed_mps=max_speed_mps,
            min_speed_mps=min_speed_mps,
        )
        waypoints.append(
            TrackableWaypoint(
                index=index,
                cell=cell,
                world=world_points[index],
                heading_rad=heading,
                segment_length_m=segment_length,
                turn_angle_deg=turn_angle,
                curvature=curvature,
                turning_radius_m=turning_radius,
                recommended_speed_mps=speed,
            )
        )

    curvatures = [waypoint.curvature for waypoint in waypoints]
    positive_radii = [waypoint.turning_radius_m for waypoint in waypoints if waypoint.turning_radius_m is not None]
    return TrackablePath(
        source_path=source_path,
        waypoints=tuple(waypoints),
        length_m=sum(waypoint.segment_length_m for waypoint in waypoints),
        max_curvature=max(curvatures, default=0.0),
        min_turning_radius_m=min(positive_radii) if positive_radii else None,
    )


def evaluate_tracking_safety(
    grid: CostGrid,
    trackable_path: TrackablePath,
    *,
    platform_profile: PlannerPlatformProfile | None,
    tracking_error_bound_m: float,
) -> TrackingSafetyReport:
    if tracking_error_bound_m < 0.0:
        raise ValueError("tracking_error_bound_m must be nonnegative")
    footprint_radius_m = 0.0 if platform_profile is None or platform_profile.footprint_radius_m is None else platform_profile.footprint_radius_m
    checked_radius_m = footprint_radius_m + tracking_error_bound_m
    blocked_cells = tuple(Cell(int(x), int(y)) for y, x in np.argwhere(~grid.passable_mask))

    violation_indices: list[int] = []
    min_clearance = _min_clearance_to_blocked(grid, trackable_path, blocked_cells, footprint_radius_m)
    for index, waypoint in enumerate(trackable_path.waypoints):
        cells_to_check = [waypoint.cell]
        if index > 0:
            cells_to_check.extend(_cells_on_line(trackable_path.waypoints[index - 1].cell, waypoint.cell))
        if any(_cell_violates_tracking_radius(grid, cell, blocked_cells, checked_radius_m) for cell in cells_to_check):
            violation_indices.append(index)

    violations = tuple(sorted(set(violation_indices)))
    return TrackingSafetyReport(
        is_safe=not violations,
        tracking_error_bound_m=float(tracking_error_bound_m),
        checked_radius_m=float(checked_radius_m),
        min_clearance_m=min_clearance,
        violation_indices=violations,
        summary="tracking safety satisfied" if not violations else f"tracking safety violations: {len(violations)}",
    )


def _recommended_speed(
    grid: CostGrid,
    cell: Cell,
    *,
    curvature: float,
    max_speed_mps: float,
    min_speed_mps: float,
) -> float:
    cost = grid.cost_at(cell)
    cost_factor = 1.0 / max(1.0, cost)
    curvature_factor = 1.0 / (1.0 + 4.0 * curvature)
    speed = max_speed_mps * min(cost_factor, curvature_factor)
    return float(max(min_speed_mps, min(max_speed_mps, speed)))


def _heading_at(points: tuple[WorldPoint, ...], index: int) -> float:
    if len(points) < 2:
        return 0.0
    if index < len(points) - 1:
        start = points[index]
        goal = points[index + 1]
    else:
        start = points[index - 1]
        goal = points[index]
    return math.atan2(goal.y - start.y, goal.x - start.x)


def _turn_angle_deg(points: tuple[WorldPoint, ...], index: int) -> float:
    if index <= 0 or index >= len(points) - 1:
        return 0.0
    ab_x = points[index].x - points[index - 1].x
    ab_y = points[index].y - points[index - 1].y
    bc_x = points[index + 1].x - points[index].x
    bc_y = points[index + 1].y - points[index].y
    len_ab = math.hypot(ab_x, ab_y)
    len_bc = math.hypot(bc_x, bc_y)
    if len_ab == 0.0 or len_bc == 0.0:
        return 0.0
    cosine = (ab_x * bc_x + ab_y * bc_y) / (len_ab * len_bc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _curvature(points: tuple[WorldPoint, ...], index: int) -> float:
    if index <= 0 or index >= len(points) - 1:
        return 0.0
    a = points[index - 1]
    b = points[index]
    c = points[index + 1]
    side_ab = _distance(a, b)
    side_bc = _distance(b, c)
    side_ac = _distance(a, c)
    denominator = side_ab * side_bc * side_ac
    if denominator == 0.0:
        return 0.0
    doubled_area = abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
    if doubled_area == 0.0:
        return 0.0
    return 2.0 * doubled_area / denominator


def _distance(a: WorldPoint, b: WorldPoint) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def _min_clearance_to_blocked(
    grid: CostGrid,
    trackable_path: TrackablePath,
    blocked_cells: tuple[Cell, ...],
    footprint_radius_m: float,
) -> float | None:
    if not blocked_cells or not trackable_path.waypoints:
        return None
    min_distance = math.inf
    for waypoint in trackable_path.waypoints:
        for blocked in blocked_cells:
            distance = math.hypot(
                (waypoint.cell.x - blocked.x) * grid.spec.resolution,
                (waypoint.cell.y - blocked.y) * grid.spec.resolution,
            )
            min_distance = min(min_distance, distance)
    return float(max(0.0, min_distance - footprint_radius_m))


def _cell_violates_tracking_radius(
    grid: CostGrid,
    cell: Cell,
    blocked_cells: tuple[Cell, ...],
    checked_radius_m: float,
) -> bool:
    if not grid.spec.in_bounds(cell):
        return True
    if not blocked_cells:
        return False
    for blocked in blocked_cells:
        distance = math.hypot(
            (cell.x - blocked.x) * grid.spec.resolution,
            (cell.y - blocked.y) * grid.spec.resolution,
        )
        if distance < checked_radius_m:
            return True
    return False


def _cells_on_line(start: Cell, goal: Cell) -> tuple[Cell, ...]:
    cells: list[Cell] = []
    min_x = min(start.x, goal.x)
    max_x = max(start.x, goal.x)
    min_y = min(start.y, goal.y)
    max_y = max(start.y, goal.y)
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cell = Cell(x, y)
            if _segment_intersects_cell(start, goal, cell):
                cells.append(cell)
    return tuple(sorted(cells))


def _segment_intersects_cell(start: Cell, goal: Cell, cell: Cell) -> bool:
    x0, y0 = float(start.x), float(start.y)
    x1, y1 = float(goal.x), float(goal.y)
    dx = x1 - x0
    dy = y1 - y0
    left = cell.x - 0.5
    right = cell.x + 0.5
    bottom = cell.y - 0.5
    top = cell.y + 0.5

    t0 = 0.0
    t1 = 1.0
    for edge_delta, edge_distance in (
        (-dx, x0 - left),
        (dx, right - x0),
        (-dy, y0 - bottom),
        (dy, top - y0),
    ):
        if edge_delta == 0.0:
            if edge_distance < 0.0:
                return False
            continue
        ratio = edge_distance / edge_delta
        if edge_delta < 0.0:
            if ratio > t1:
                return False
            if ratio > t0:
                t0 = ratio
        else:
            if ratio < t0:
                return False
            if ratio < t1:
                t1 = ratio
    return True
