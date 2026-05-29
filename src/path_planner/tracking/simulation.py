from __future__ import annotations

import math

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint
from path_planner.platform import PlannerPlatformProfile
from path_planner.trajectory import TrackablePath

from .models import (
    TrackingSimulationConfig,
    TrackingSimulationMetrics,
    TrackingSimulationResult,
    TrackingSimulationSafetyReport,
    TrackingState,
)


def simulate_tracking(
    grid: CostGrid,
    trackable_path: TrackablePath,
    *,
    platform_profile: PlannerPlatformProfile | None,
    config: TrackingSimulationConfig,
) -> TrackingSimulationResult:
    if not trackable_path.waypoints:
        safety_report = TrackingSimulationSafetyReport(
            is_safe=False,
            min_clearance_m=None,
            violation_indices=(),
            summary="tracking simulation skipped: empty trackable path",
        )
        return TrackingSimulationResult(
            config=config,
            states=(),
            metrics=TrackingSimulationMetrics(
                path_length_m=0.0,
                simulated_length_m=0.0,
                max_cross_track_error_m=0.0,
                min_clearance_m=None,
                safety_violation_count=0,
                curvature_violation_count=0,
                mean_speed_mps=0.0,
                high_cost_exposure=0.0,
            ),
            safety_report=safety_report,
        )

    points = tuple(waypoint.world for waypoint in trackable_path.waypoints)
    cumulative = _cumulative_lengths(points)
    x = points[0].x
    y = points[0].y
    heading = trackable_path.waypoints[0].heading_rad
    states: list[TrackingState] = []
    max_steps = int(math.ceil(config.max_sim_time_s / config.time_step_s))

    for step in range(max_steps + 1):
        projection = _project_to_path(WorldPoint(x, y), points, cumulative)
        target_s = min(trackable_path.length_m, projection.path_s + config.lookahead_m)
        target = _point_at_s(points, cumulative, target_s)
        target_index = _target_waypoint_index(cumulative, target_s)
        speed = _speed_at_s(trackable_path, cumulative, target_s)
        cross_track_error = projection.cross_track_error_m
        states.append(
            TrackingState(
                time_s=step * config.time_step_s,
                x=x,
                y=y,
                heading_rad=heading,
                target_waypoint_index=target_index,
                speed_mps=speed,
                cross_track_error_m=cross_track_error,
            )
        )
        if _is_finished(WorldPoint(x, y), points[-1], projection.path_s, trackable_path.length_m):
            break

        desired_heading = math.atan2(target.y - y, target.x - x)
        alpha = _wrap_angle(desired_heading - heading)
        angular_velocity = 2.0 * speed * math.sin(alpha) / max(config.lookahead_m, 1e-9)
        heading = _wrap_angle(heading + angular_velocity * config.time_step_s)
        remaining_length = trackable_path.length_m - projection.path_s
        step_length = speed * config.time_step_s
        if remaining_length <= step_length:
            x = points[-1].x
            y = points[-1].y
        else:
            x += step_length * math.cos(heading)
            y += step_length * math.sin(heading)

    safety_report = _evaluate_simulated_safety(grid, states, platform_profile)
    metrics = _build_metrics(grid, trackable_path, states, platform_profile, safety_report)
    return TrackingSimulationResult(
        config=config,
        states=tuple(states),
        metrics=metrics,
        safety_report=safety_report,
    )


class _Projection:
    def __init__(self, path_s: float, cross_track_error_m: float) -> None:
        self.path_s = path_s
        self.cross_track_error_m = cross_track_error_m


def _project_to_path(point: WorldPoint, path: tuple[WorldPoint, ...], cumulative: tuple[float, ...]) -> _Projection:
    if len(path) == 1:
        return _Projection(0.0, math.hypot(point.x - path[0].x, point.y - path[0].y))

    best_s = 0.0
    best_distance = math.inf
    for index in range(len(path) - 1):
        start = path[index]
        end = path[index + 1]
        dx = end.x - start.x
        dy = end.y - start.y
        length_sq = dx * dx + dy * dy
        if length_sq == 0.0:
            ratio = 0.0
        else:
            ratio = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
            ratio = max(0.0, min(1.0, ratio))
        projected_x = start.x + ratio * dx
        projected_y = start.y + ratio * dy
        distance = math.hypot(point.x - projected_x, point.y - projected_y)
        if distance < best_distance:
            best_distance = distance
            best_s = cumulative[index] + ratio * math.sqrt(length_sq)
    return _Projection(best_s, best_distance)


def _cumulative_lengths(points: tuple[WorldPoint, ...]) -> tuple[float, ...]:
    values = [0.0]
    for start, end in zip(points[:-1], points[1:]):
        values.append(values[-1] + math.hypot(end.x - start.x, end.y - start.y))
    return tuple(values)


def _point_at_s(points: tuple[WorldPoint, ...], cumulative: tuple[float, ...], path_s: float) -> WorldPoint:
    if len(points) == 1:
        return points[0]
    clamped_s = max(0.0, min(cumulative[-1], path_s))
    for index in range(len(points) - 1):
        if cumulative[index] <= clamped_s <= cumulative[index + 1]:
            segment_length = cumulative[index + 1] - cumulative[index]
            if segment_length == 0.0:
                return points[index]
            ratio = (clamped_s - cumulative[index]) / segment_length
            return WorldPoint(
                points[index].x + ratio * (points[index + 1].x - points[index].x),
                points[index].y + ratio * (points[index + 1].y - points[index].y),
            )
    return points[-1]


def _target_waypoint_index(cumulative: tuple[float, ...], target_s: float) -> int:
    for index, value in enumerate(cumulative):
        if value >= target_s:
            return index
    return len(cumulative) - 1


def _speed_at_s(trackable_path: TrackablePath, cumulative: tuple[float, ...], target_s: float) -> float:
    index = _target_waypoint_index(cumulative, target_s)
    return trackable_path.waypoints[index].recommended_speed_mps


def _is_finished(point: WorldPoint, goal: WorldPoint, path_s: float, path_length: float) -> bool:
    return path_s >= path_length - 1e-6 and math.hypot(point.x - goal.x, point.y - goal.y) <= 0.05


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _evaluate_simulated_safety(
    grid: CostGrid,
    states: list[TrackingState],
    platform_profile: PlannerPlatformProfile | None,
) -> TrackingSimulationSafetyReport:
    footprint_radius = 0.0 if platform_profile is None or platform_profile.footprint_radius_m is None else platform_profile.footprint_radius_m
    blocked_cells = tuple(Cell(int(x), int(y)) for y, x in np.argwhere(~grid.passable_mask))
    violation_indices: list[int] = []
    min_clearance: float | None = None

    for index, state in enumerate(states):
        clearance = _clearance_to_blocked(grid, state.x, state.y, blocked_cells, footprint_radius)
        if clearance is not None:
            min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
        if not _world_in_bounds(grid, state.x, state.y) or (clearance is not None and clearance <= 0.0):
            violation_indices.append(index)

    violations = tuple(violation_indices)
    return TrackingSimulationSafetyReport(
        is_safe=not violations,
        min_clearance_m=min_clearance,
        violation_indices=violations,
        summary="tracking simulation safe" if not violations else f"tracking simulation safety violations: {len(violations)}",
    )


def _build_metrics(
    grid: CostGrid,
    trackable_path: TrackablePath,
    states: list[TrackingState],
    platform_profile: PlannerPlatformProfile | None,
    safety_report: TrackingSimulationSafetyReport,
) -> TrackingSimulationMetrics:
    simulated_length = 0.0
    high_cost_exposure = 0.0
    for previous, current in zip(states[:-1], states[1:]):
        segment_length = math.hypot(current.x - previous.x, current.y - previous.y)
        simulated_length += segment_length
        cell = grid.spec.world_to_cell(WorldPoint(current.x, current.y))
        if grid.spec.in_bounds(cell) and grid.cost_at(cell) >= 3.0:
            high_cost_exposure += segment_length

    speeds = [state.speed_mps for state in states]
    return TrackingSimulationMetrics(
        path_length_m=trackable_path.length_m,
        simulated_length_m=simulated_length,
        max_cross_track_error_m=max((state.cross_track_error_m for state in states), default=0.0),
        min_clearance_m=safety_report.min_clearance_m,
        safety_violation_count=len(safety_report.violation_indices),
        curvature_violation_count=_curvature_violation_count(trackable_path, platform_profile),
        mean_speed_mps=float(sum(speeds) / len(speeds)) if speeds else 0.0,
        high_cost_exposure=high_cost_exposure,
    )


def _curvature_violation_count(
    trackable_path: TrackablePath,
    platform_profile: PlannerPlatformProfile | None,
) -> int:
    min_turning_radius = None if platform_profile is None else platform_profile.effective_min_turning_radius_m
    if min_turning_radius is None or min_turning_radius <= 0.0:
        return 0
    return sum(
        1
        for waypoint in trackable_path.waypoints
        if waypoint.turning_radius_m is not None and waypoint.turning_radius_m < min_turning_radius
    )


def _clearance_to_blocked(
    grid: CostGrid,
    x: float,
    y: float,
    blocked_cells: tuple[Cell, ...],
    footprint_radius: float,
) -> float | None:
    if not blocked_cells:
        return None
    min_distance = math.inf
    for blocked in blocked_cells:
        blocked_world = grid.spec.cell_to_world(blocked)
        min_distance = min(min_distance, math.hypot(x - blocked_world.x, y - blocked_world.y))
    return float(max(0.0, min_distance - footprint_radius))


def _world_in_bounds(grid: CostGrid, x: float, y: float) -> bool:
    cell = grid.spec.world_to_cell(WorldPoint(x, y))
    return grid.spec.in_bounds(cell)
