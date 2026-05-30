from __future__ import annotations

import math

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint
from path_planner.platform import PlannerPlatformProfile
from path_planner.postprocess.models import CorridorResult, CorridorSection
from path_planner.trajectory import TrackablePath, TrackableWaypoint

from .models import (
    CorridorBox,
    TrajectoryOptimizationConfig,
    TrajectoryOptimizationFallbackStatus,
    TrajectoryOptimizationMetrics,
    TrajectoryOptimizationResult,
)
from .resampling import resample_world_points


HIGH_COST_EXPOSURE_TOLERANCE_M = 1e-3


def optimize_trajectory(
    grid: CostGrid,
    trackable_path: TrackablePath,
    corridor: CorridorResult,
    *,
    platform_profile: PlannerPlatformProfile | None,
    config: TrajectoryOptimizationConfig,
) -> TrajectoryOptimizationResult:
    reference_points = tuple(waypoint.world for waypoint in trackable_path.waypoints)
    if not reference_points:
        return _fallback_result(grid, trackable_path, corridor, config, "empty_trackable_path", platform_profile)
    if corridor.status != "ok" or not corridor.sections:
        return _fallback_result(
            grid,
            trackable_path,
            corridor,
            config,
            corridor.failure_reason or "corridor_unavailable",
            platform_profile,
        )

    boxes = _build_corridor_boxes(grid, trackable_path, corridor)
    if len(boxes) != len(reference_points) or any(not box.contains(point) for box, point in zip(boxes, reference_points)):
        return _fallback_result(grid, trackable_path, corridor, config, "reference_path_outside_corridor", platform_profile)

    reference = np.asarray([[point.x, point.y] for point in reference_points], dtype=float)
    points = reference.copy()
    objective_initial = _objective(grid, points, reference, config)
    solver_status = "converged"
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        previous = points.copy()
        for index in range(1, len(points) - 1):
            smooth_target = 0.5 * (points[index - 1] + points[index + 1])
            cost_target = _lowest_cost_point_in_box(grid, boxes[index], reference[index])
            numerator = (
                points[index]
                + config.weight_reference * reference[index]
                + config.weight_cost * cost_target
                + (config.weight_smoothness + config.weight_length + config.weight_curvature_proxy) * smooth_target
            )
            denominator = (
                1.0
                + config.weight_reference
                + config.weight_cost
                + config.weight_smoothness
                + config.weight_length
                + config.weight_curvature_proxy
            )
            candidate = numerator / max(denominator, 1e-12)
            candidate = candidate - config.weight_cost * 0.02 * grid.spec.resolution * _cost_gradient(grid, candidate)
            points[index] = _project_to_box(candidate, boxes[index])

        points[0] = reference[0]
        points[-1] = reference[-1]
        iterations = iteration
        if float(np.linalg.norm(points - previous)) <= config.convergence_tolerance:
            break
    else:
        solver_status = "max_iterations"

    points = _apply_high_cost_guard(grid, points, reference, config)
    optimized_points = tuple(WorldPoint(float(x), float(y)) for x, y in points)
    optimized_trackable = _build_trackable_path_from_world(grid, trackable_path, optimized_points)
    resampled_points = _build_resampled_points(grid, optimized_points, corridor, config)
    resampled_points = _apply_resampled_high_cost_guard(grid, resampled_points, reference_points, corridor, config)
    resampled_boxes = _build_corridor_boxes_for_points(grid, resampled_points, corridor)
    resampled_trackable = _build_trackable_path_from_world(grid, optimized_trackable, resampled_points)
    metrics = _build_metrics(
        grid,
        trackable_path,
        resampled_trackable,
        boxes,
        resampled_boxes,
        config,
        objective_initial,
        _objective(grid, points, reference, config),
        iterations,
        platform_profile,
    )
    warnings = _build_warnings(trackable_path, resampled_trackable, metrics)
    return TrajectoryOptimizationResult(
        config=config,
        solver_status=solver_status,
        fallback_status=TrajectoryOptimizationFallbackStatus(used_reference_path=False, reason=None),
        optimized_path=optimized_points,
        resampled_optimized_path=resampled_points,
        optimized_trackable_path=optimized_trackable,
        resampled_trackable_path=resampled_trackable,
        corridor_boxes=boxes,
        resampled_corridor_boxes=resampled_boxes,
        metrics=metrics,
        warnings=warnings,
    )


def _fallback_result(
    grid: CostGrid,
    trackable_path: TrackablePath,
    corridor: CorridorResult,
    config: TrajectoryOptimizationConfig,
    reason: str,
    platform_profile: PlannerPlatformProfile | None,
) -> TrajectoryOptimizationResult:
    reference_points = tuple(waypoint.world for waypoint in trackable_path.waypoints)
    optimized_trackable = trackable_path if trackable_path.waypoints else _build_trackable_path_from_world(grid, trackable_path, reference_points)
    boxes = _build_corridor_boxes(grid, trackable_path, corridor) if corridor.status == "ok" and corridor.sections else ()
    resampled_points = (
        _build_resampled_points(grid, reference_points, corridor, config)
        if corridor.status == "ok" and corridor.sections
        else reference_points
    )
    resampled_boxes = _build_corridor_boxes_for_points(grid, resampled_points, corridor) if corridor.status == "ok" and corridor.sections else ()
    resampled_trackable = _build_trackable_path_from_world(grid, optimized_trackable, resampled_points)
    reference = np.asarray([[point.x, point.y] for point in reference_points], dtype=float)
    objective = _objective(grid, reference, reference, config) if len(reference) else 0.0
    metrics = _build_metrics(
        grid,
        trackable_path,
        resampled_trackable,
        boxes,
        resampled_boxes,
        config,
        objective,
        objective,
        0,
        platform_profile,
    )
    return TrajectoryOptimizationResult(
        config=config,
        solver_status="fallback",
        fallback_status=TrajectoryOptimizationFallbackStatus(used_reference_path=True, reason=reason),
        optimized_path=reference_points,
        resampled_optimized_path=resampled_points,
        optimized_trackable_path=optimized_trackable,
        resampled_trackable_path=resampled_trackable,
        corridor_boxes=boxes,
        resampled_corridor_boxes=resampled_boxes,
        metrics=metrics,
        warnings=(reason,),
    )


def _build_corridor_boxes(
    grid: CostGrid,
    trackable_path: TrackablePath,
    corridor: CorridorResult,
) -> tuple[CorridorBox, ...]:
    boxes: list[CorridorBox] = []
    for point_index, waypoint in enumerate(trackable_path.waypoints):
        section_index, section = _nearest_section_for_point(grid, waypoint.world, corridor.sections)
        boxes.append(_section_to_box(grid, point_index, section_index, section))
    return tuple(boxes)


def _build_corridor_boxes_for_points(
    grid: CostGrid,
    points: tuple[WorldPoint, ...],
    corridor: CorridorResult,
) -> tuple[CorridorBox, ...]:
    boxes: list[CorridorBox] = []
    if corridor.status != "ok" or not corridor.sections:
        return ()
    for point_index, point in enumerate(points):
        section_index, section = _nearest_section_for_point(grid, point, corridor.sections)
        boxes.append(_section_to_box(grid, point_index, section_index, section))
    return tuple(boxes)


def _nearest_section_for_point(
    grid: CostGrid,
    point: WorldPoint,
    sections: tuple[CorridorSection, ...],
) -> tuple[int, CorridorSection]:
    cell = grid.spec.world_to_cell(point)
    candidates = [index for index, section in enumerate(sections) if _section_contains_point(grid, section, point)]
    if candidates:
        best_index = min(candidates, key=lambda index: (_section_distance_sq(sections[index], cell), -index))
        return best_index, sections[best_index]
    return _nearest_section(cell, sections)


def _nearest_section(cell: Cell, sections: tuple[CorridorSection, ...]) -> tuple[int, CorridorSection]:
    candidates = [index for index, section in enumerate(sections) if cell in section.cells]
    search_indices = candidates or list(range(len(sections)))
    best_index = min(search_indices, key=lambda index: (_section_distance_sq(sections[index], cell), -index))
    return best_index, sections[best_index]


def _section_distance_sq(section: CorridorSection, cell: Cell) -> int:
    return (section.center.x - cell.x) ** 2 + (section.center.y - cell.y) ** 2


def _section_contains_point(grid: CostGrid, section: CorridorSection, point: WorldPoint) -> bool:
    cells = section.cells or (section.center,)
    min_x = min(cell.x for cell in cells) * grid.spec.resolution + grid.spec.origin[0]
    max_x = max(cell.x for cell in cells) * grid.spec.resolution + grid.spec.origin[0]
    min_y = min(cell.y for cell in cells) * grid.spec.resolution + grid.spec.origin[1]
    max_y = max(cell.y for cell in cells) * grid.spec.resolution + grid.spec.origin[1]
    return min_x <= point.x <= max_x and min_y <= point.y <= max_y


def _section_to_box(grid: CostGrid, point_index: int, section_index: int, section: CorridorSection) -> CorridorBox:
    cells = section.cells or (section.center,)
    min_x = min(cell.x for cell in cells) * grid.spec.resolution + grid.spec.origin[0]
    max_x = max(cell.x for cell in cells) * grid.spec.resolution + grid.spec.origin[0]
    min_y = min(cell.y for cell in cells) * grid.spec.resolution + grid.spec.origin[1]
    max_y = max(cell.y for cell in cells) * grid.spec.resolution + grid.spec.origin[1]
    return CorridorBox(
        point_index=point_index,
        section_index=section_index,
        min_x=float(min_x),
        max_x=float(max_x),
        min_y=float(min_y),
        max_y=float(max_y),
    )


def _project_to_box(point: np.ndarray, box: CorridorBox) -> np.ndarray:
    return np.asarray(
        [
            min(max(float(point[0]), box.min_x), box.max_x),
            min(max(float(point[1]), box.min_y), box.max_y),
        ],
        dtype=float,
    )


def _build_resampled_points(
    grid: CostGrid,
    points: tuple[WorldPoint, ...],
    corridor: CorridorResult,
    config: TrajectoryOptimizationConfig,
) -> tuple[WorldPoint, ...]:
    if config.resample_spacing_m is None or len(points) <= 1:
        return points
    spacing = config.resample_spacing_m
    projected = _project_points_to_corridor(
        grid,
        resample_world_points(points, spacing_m=spacing),
        corridor,
        max_step_m=spacing,
    )
    best = projected
    internal_spacing = spacing
    for _ in range(8):
        if _max_spacing(projected) <= spacing + 1e-9:
            return projected
        if _max_spacing(projected) < _max_spacing(best):
            best = projected
        internal_spacing *= 0.5
        densified = _densify_world_points(projected, internal_spacing)
        projected = _project_points_to_corridor(grid, densified, corridor, max_step_m=spacing)
    return projected if _max_spacing(projected) <= _max_spacing(best) else best


def _project_points_to_corridor(
    grid: CostGrid,
    points: tuple[WorldPoint, ...],
    corridor: CorridorResult,
    *,
    max_step_m: float | None = None,
) -> tuple[WorldPoint, ...]:
    boxes = _build_corridor_boxes_for_points(grid, points, corridor)
    if len(boxes) != len(points):
        return points
    projected: list[WorldPoint] = []
    for index, (point, box) in enumerate(zip(points, boxes)):
        if index == 0 or index == len(points) - 1:
            projected.append(point)
            continue
        clipped = _project_to_box(np.asarray([point.x, point.y], dtype=float), box)
        candidate = WorldPoint(float(clipped[0]), float(clipped[1]))
        previous = projected[-1]
        if max_step_m is not None and _distance(previous, candidate) > max_step_m:
            distance = _distance(previous, candidate)
            ratio = max_step_m / max(distance, 1e-12)
            stepped = np.asarray(
                [
                    previous.x + ratio * (candidate.x - previous.x),
                    previous.y + ratio * (candidate.y - previous.y),
                ],
                dtype=float,
            )
            connected = _project_to_box(stepped, box)
            connected_candidate = WorldPoint(float(connected[0]), float(connected[1]))
            if _distance(previous, connected_candidate) < _distance(previous, candidate):
                candidate = connected_candidate
        projected.append(candidate)
    return tuple(projected)


def _densify_world_points(points: tuple[WorldPoint, ...], spacing_m: float) -> tuple[WorldPoint, ...]:
    if len(points) <= 1:
        return points
    densified: list[WorldPoint] = [points[0]]
    for start, goal in zip(points[:-1], points[1:]):
        distance = _distance(start, goal)
        steps = max(1, int(math.ceil(distance / spacing_m)))
        for step in range(1, steps):
            ratio = step / steps
            densified.append(
                WorldPoint(
                    start.x + ratio * (goal.x - start.x),
                    start.y + ratio * (goal.y - start.y),
                )
            )
        densified.append(goal)
    return tuple(densified)


def _max_spacing(points: tuple[WorldPoint, ...]) -> float:
    return max((_distance(previous, current) for previous, current in zip(points[:-1], points[1:])), default=0.0)


def _lowest_cost_point_in_box(grid: CostGrid, box: CorridorBox, reference_point: np.ndarray) -> np.ndarray:
    best_cell: Cell | None = None
    best_key: tuple[float, float] | None = None
    for y in range(grid.spec.height):
        for x in range(grid.spec.width):
            cell = Cell(x, y)
            point = grid.spec.cell_to_world(cell)
            if not box.contains(point) or not grid.is_passable(cell):
                continue
            distance_sq = (point.x - float(reference_point[0])) ** 2 + (point.y - float(reference_point[1])) ** 2
            key = (grid.cost_at(cell), distance_sq)
            if best_key is None or key < best_key:
                best_cell = cell
                best_key = key
    if best_cell is None:
        return reference_point
    point = grid.spec.cell_to_world(best_cell)
    return np.asarray([point.x, point.y], dtype=float)


def _cost_gradient(grid: CostGrid, point: np.ndarray) -> np.ndarray:
    cell = grid.spec.world_to_cell(WorldPoint(float(point[0]), float(point[1])))
    if not grid.spec.in_bounds(cell):
        return np.zeros(2, dtype=float)
    left = Cell(max(cell.x - 1, 0), cell.y)
    right = Cell(min(cell.x + 1, grid.spec.width - 1), cell.y)
    up = Cell(cell.x, max(cell.y - 1, 0))
    down = Cell(cell.x, min(cell.y + 1, grid.spec.height - 1))
    scale_x = max((right.x - left.x) * grid.spec.resolution, grid.spec.resolution)
    scale_y = max((down.y - up.y) * grid.spec.resolution, grid.spec.resolution)
    return np.asarray(
        [
            (grid.cost_at(right) - grid.cost_at(left)) / scale_x,
            (grid.cost_at(down) - grid.cost_at(up)) / scale_y,
        ],
        dtype=float,
    )


def _objective(
    grid: CostGrid,
    points: np.ndarray,
    reference: np.ndarray,
    config: TrajectoryOptimizationConfig,
) -> float:
    if len(points) == 0:
        return 0.0
    length = _array_path_length(points)
    smoothness = 0.0
    if len(points) > 2:
        second = points[:-2] - 2.0 * points[1:-1] + points[2:]
        smoothness = float(np.sum(second * second))
    reference_error = float(np.sum((points - reference) * (points - reference)))
    tracking_proxy = _tracking_error_proxy_for_array(points)
    spacing_cost = _spacing_uniformity_cost(points, config.resample_spacing_m)
    speed_smoothness = _speed_smoothness_proxy_for_array(points)
    high_cost = _high_cost_exposure(
        grid,
        tuple(WorldPoint(float(x), float(y)) for x, y in points),
        config.high_cost_threshold,
    )
    return float(
        config.weight_length * length
        + config.weight_smoothness * smoothness
        + config.weight_reference * reference_error
        + config.weight_curvature_proxy * smoothness
        + config.weight_tracking * tracking_proxy
        + config.weight_spacing * spacing_cost
        + config.weight_speed_smoothness * speed_smoothness
        + config.weight_cost * high_cost
    )


def _apply_high_cost_guard(
    grid: CostGrid,
    points: np.ndarray,
    reference: np.ndarray,
    config: TrajectoryOptimizationConfig,
) -> np.ndarray:
    reference_exposure = _high_cost_exposure(
        grid,
        tuple(WorldPoint(float(x), float(y)) for x, y in reference),
        config.high_cost_threshold,
    )
    current_exposure = _high_cost_exposure(
        grid,
        tuple(WorldPoint(float(x), float(y)) for x, y in points),
        config.high_cost_threshold,
    )
    if current_exposure <= reference_exposure:
        return points

    guarded = points.copy()
    for index in range(len(points) - 1):
        segment = (
            WorldPoint(float(guarded[index, 0]), float(guarded[index, 1])),
            WorldPoint(float(guarded[index + 1, 0]), float(guarded[index + 1, 1])),
        )
        reference_segment = (
            WorldPoint(float(reference[index, 0]), float(reference[index, 1])),
            WorldPoint(float(reference[index + 1, 0]), float(reference[index + 1, 1])),
        )
        if _high_cost_exposure(grid, segment, config.high_cost_threshold) <= 0.0:
            continue
        if _high_cost_exposure(grid, reference_segment, config.high_cost_threshold) > 0.0:
            continue
        if 0 < index < len(points) - 1:
            guarded[index] = reference[index]
        if 0 < index + 1 < len(points) - 1:
            guarded[index + 1] = reference[index + 1]

    guarded_exposure = _high_cost_exposure(
        grid,
        tuple(WorldPoint(float(x), float(y)) for x, y in guarded),
        config.high_cost_threshold,
    )
    return guarded if guarded_exposure <= reference_exposure else reference.copy()


def _apply_resampled_high_cost_guard(
    grid: CostGrid,
    candidate_points: tuple[WorldPoint, ...],
    reference_points: tuple[WorldPoint, ...],
    corridor: CorridorResult,
    config: TrajectoryOptimizationConfig,
) -> tuple[WorldPoint, ...]:
    reference_resampled = _build_resampled_points(grid, reference_points, corridor, config)
    reference_exposure = _high_cost_exposure(grid, reference_resampled, config.high_cost_threshold)
    candidate_exposure = _high_cost_exposure(grid, candidate_points, config.high_cost_threshold)
    if candidate_exposure <= reference_exposure + HIGH_COST_EXPOSURE_TOLERANCE_M:
        return candidate_points

    if len(candidate_points) == len(reference_resampled):
        for candidate_weight in (0.75, 0.5, 0.25, 0.1, 0.0):
            blended = _blend_world_points(candidate_points, reference_resampled, candidate_weight)
            blended = _project_points_to_corridor(grid, blended, corridor, max_step_m=config.resample_spacing_m)
            if config.resample_spacing_m is not None and _max_spacing(blended) > config.resample_spacing_m + 1e-9:
                continue
            if _high_cost_exposure(grid, blended, config.high_cost_threshold) <= reference_exposure + HIGH_COST_EXPOSURE_TOLERANCE_M:
                return blended

    return reference_resampled


def _blend_world_points(
    candidate_points: tuple[WorldPoint, ...],
    reference_points: tuple[WorldPoint, ...],
    candidate_weight: float,
) -> tuple[WorldPoint, ...]:
    reference_weight = 1.0 - candidate_weight
    return tuple(
        WorldPoint(
            candidate_weight * candidate.x + reference_weight * reference.x,
            candidate_weight * candidate.y + reference_weight * reference.y,
        )
        for candidate, reference in zip(candidate_points, reference_points)
    )


def _build_metrics(
    grid: CostGrid,
    reference: TrackablePath,
    optimized: TrackablePath,
    boxes: tuple[CorridorBox, ...],
    resampled_boxes: tuple[CorridorBox, ...],
    config: TrajectoryOptimizationConfig,
    objective_initial: float,
    objective_final: float,
    iterations: int,
    platform_profile: PlannerPlatformProfile | None,
) -> TrajectoryOptimizationMetrics:
    reference_points = tuple(waypoint.world for waypoint in reference.waypoints)
    optimized_points = tuple(waypoint.world for waypoint in optimized.waypoints)
    reference_metric_points = _build_metric_reference_points(reference_points, config)
    reference_exposure = _high_cost_exposure(grid, reference_metric_points, config.high_cost_threshold)
    optimized_exposure = _high_cost_exposure(grid, optimized_points, config.high_cost_threshold)
    curvature_violations = _curvature_violation_count(optimized, platform_profile)
    check_boxes = resampled_boxes if resampled_boxes else boxes
    within_corridor = len(check_boxes) == len(optimized_points) and all(
        box.contains(point) for box, point in zip(check_boxes, optimized_points)
    )
    reference_execution = _execution_metrics(reference)
    optimized_execution = _execution_metrics(optimized)
    baseline_vs_optimized = {
        "path_length_m": {
            "baseline": reference.length_m,
            "optimized": optimized.length_m,
            "delta": optimized.length_m - reference.length_m,
        },
        "high_cost_exposure": {
            "baseline": reference_exposure,
            "optimized": optimized_exposure,
            "delta": optimized_exposure - reference_exposure,
        },
        "curvature_violation_count": {
            "baseline": _curvature_violation_count(reference, platform_profile),
            "optimized": curvature_violations,
            "delta": curvature_violations - _curvature_violation_count(reference, platform_profile),
        },
        "heading_change_max_deg": _metric_delta(
            reference_execution["heading_change_max_deg"],
            optimized_execution["heading_change_max_deg"],
        ),
        "tracking_error_proxy": _metric_delta(
            reference_execution["tracking_error_proxy"],
            optimized_execution["tracking_error_proxy"],
        ),
        "waypoint_spacing_max_m": _metric_delta(
            reference_execution["waypoint_spacing_max_m"],
            optimized_execution["waypoint_spacing_max_m"],
        ),
        "waypoint_spacing_mean_m": _metric_delta(
            reference_execution["waypoint_spacing_mean_m"],
            optimized_execution["waypoint_spacing_mean_m"],
        ),
        "speed_smoothness_cost": _metric_delta(
            reference_execution["speed_smoothness_cost"],
            optimized_execution["speed_smoothness_cost"],
        ),
    }
    return TrajectoryOptimizationMetrics(
        reference_path_length_m=reference.length_m,
        optimized_path_length_m=optimized.length_m,
        length_delta_m=optimized.length_m - reference.length_m,
        reference_high_cost_exposure=reference_exposure,
        optimized_high_cost_exposure=optimized_exposure,
        high_cost_exposure_delta=optimized_exposure - reference_exposure,
        max_curvature=optimized.max_curvature,
        min_turning_radius_m=optimized.min_turning_radius_m,
        curvature_violation_count=curvature_violations,
        waypoint_spacing_mean_m=optimized_execution["waypoint_spacing_mean_m"],
        waypoint_spacing_max_m=optimized_execution["waypoint_spacing_max_m"],
        heading_change_max_deg=optimized_execution["heading_change_max_deg"],
        tracking_error_proxy=optimized_execution["tracking_error_proxy"],
        speed_smoothness_cost=optimized_execution["speed_smoothness_cost"],
        objective_initial=objective_initial,
        objective_final=objective_final,
        solver_iterations=iterations,
        is_within_corridor=within_corridor,
        baseline_vs_optimized=baseline_vs_optimized,
    )


def _build_metric_reference_points(
    reference_points: tuple[WorldPoint, ...],
    config: TrajectoryOptimizationConfig,
) -> tuple[WorldPoint, ...]:
    if config.resample_spacing_m is None or len(reference_points) <= 1:
        return reference_points
    return resample_world_points(reference_points, spacing_m=config.resample_spacing_m)


def _metric_delta(baseline: float, optimized: float) -> dict[str, float]:
    return {
        "baseline": baseline,
        "optimized": optimized,
        "delta": optimized - baseline,
    }


def _build_warnings(
    reference: TrackablePath,
    optimized: TrackablePath,
    metrics: TrajectoryOptimizationMetrics,
) -> tuple[str, ...]:
    warnings: list[str] = []
    reference_execution = _execution_metrics(reference)
    if metrics.optimized_high_cost_exposure > metrics.reference_high_cost_exposure + HIGH_COST_EXPOSURE_TOLERANCE_M:
        warnings.append("optimized_high_cost_exposure_worse_than_reference")
    if metrics.tracking_error_proxy >= reference_execution["tracking_error_proxy"] - 1e-9:
        warnings.append("tracking_error_proxy_not_improved")
    if optimized.max_curvature > reference.max_curvature + 1e-9:
        warnings.append("max_curvature_worse_than_reference")
    return tuple(warnings)


def _build_trackable_path_from_world(
    grid: CostGrid,
    reference: TrackablePath,
    points: tuple[WorldPoint, ...],
) -> TrackablePath:
    waypoints: list[TrackableWaypoint] = []
    for index, point in enumerate(points):
        curvature = _curvature(points, index)
        turning_radius = 1.0 / curvature if curvature > 0.0 else None
        reference_speed = _recommended_speed_for_point(grid, reference, point, curvature, index)
        waypoints.append(
            TrackableWaypoint(
                index=index,
                cell=grid.spec.world_to_cell(point),
                world=point,
                heading_rad=_heading_at(points, index),
                segment_length_m=0.0 if index == 0 else _distance(points[index - 1], point),
                turn_angle_deg=_turn_angle_deg(points, index),
                curvature=curvature,
                turning_radius_m=turning_radius,
                recommended_speed_mps=reference_speed,
            )
        )
    radii = [waypoint.turning_radius_m for waypoint in waypoints if waypoint.turning_radius_m is not None]
    return TrackablePath(
        source_path="optimized_path",
        waypoints=tuple(waypoints),
        length_m=sum(waypoint.segment_length_m for waypoint in waypoints),
        max_curvature=max((waypoint.curvature for waypoint in waypoints), default=0.0),
        min_turning_radius_m=min(radii) if radii else None,
    )


def _recommended_speed_for_point(
    grid: CostGrid,
    reference: TrackablePath,
    point: WorldPoint,
    curvature: float,
    index: int,
) -> float:
    speeds = reference.speed_profile
    if not speeds:
        return 0.05
    max_speed = max(speeds)
    min_speed = min(speeds)
    cell = grid.spec.world_to_cell(point)
    cost = grid.cost_at(cell) if grid.spec.in_bounds(cell) else max(1.0, max_speed / max(min_speed, 1e-9))
    cost_factor = 1.0 / max(1.0, cost)
    curvature_factor = 1.0 / (1.0 + 4.0 * curvature)
    reference_speed = speeds[min(index, len(speeds) - 1)]
    speed = min(max_speed, reference_speed, max_speed * min(cost_factor, curvature_factor))
    return float(max(min_speed, speed))


def _execution_metrics(trackable_path: TrackablePath) -> dict[str, float]:
    spacings = [waypoint.segment_length_m for waypoint in trackable_path.waypoints[1:]]
    heading_changes = [
        abs(_wrap_angle(current.heading_rad - previous.heading_rad))
        for previous, current in zip(trackable_path.waypoints[:-1], trackable_path.waypoints[1:])
    ]
    speeds = list(trackable_path.speed_profile)
    speed_deltas = [current - previous for previous, current in zip(speeds[:-1], speeds[1:])]
    tracking_proxy = (
        sum(change * change for change in heading_changes)
        + sum(waypoint.curvature * waypoint.curvature for waypoint in trackable_path.waypoints)
        + _spacing_uniformity_from_spacings(spacings)
    )
    return {
        "waypoint_spacing_mean_m": float(sum(spacings) / len(spacings)) if spacings else 0.0,
        "waypoint_spacing_max_m": float(max(spacings, default=0.0)),
        "heading_change_max_deg": float(math.degrees(max(heading_changes, default=0.0))),
        "tracking_error_proxy": float(tracking_proxy),
        "speed_smoothness_cost": float(sum(delta * delta for delta in speed_deltas)),
    }


def _tracking_error_proxy_for_array(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    headings = [
        math.atan2(float(points[index + 1, 1] - points[index, 1]), float(points[index + 1, 0] - points[index, 0]))
        for index in range(len(points) - 1)
    ]
    heading_changes = [
        abs(_wrap_angle(current - previous))
        for previous, current in zip(headings[:-1], headings[1:])
    ]
    return float(sum(change * change for change in heading_changes))


def _spacing_uniformity_cost(points: np.ndarray, target_spacing: float | None) -> float:
    if len(points) < 3:
        return 0.0
    spacings = [
        math.hypot(float(current[0] - previous[0]), float(current[1] - previous[1]))
        for previous, current in zip(points[:-1], points[1:])
    ]
    target = target_spacing if target_spacing is not None else sum(spacings) / len(spacings)
    return float(sum((spacing - target) * (spacing - target) for spacing in spacings))


def _spacing_uniformity_from_spacings(spacings: list[float]) -> float:
    if not spacings:
        return 0.0
    mean = sum(spacings) / len(spacings)
    return float(sum((spacing - mean) * (spacing - mean) for spacing in spacings))


def _speed_smoothness_proxy_for_array(points: np.ndarray) -> float:
    if len(points) < 4:
        return 0.0
    spacings = [
        math.hypot(float(current[0] - previous[0]), float(current[1] - previous[1]))
        for previous, current in zip(points[:-1], points[1:])
    ]
    deltas = [current - previous for previous, current in zip(spacings[:-1], spacings[1:])]
    return float(sum(delta * delta for delta in deltas))


def _array_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    deltas = points[1:] - points[:-1]
    return float(np.sum(np.sqrt(np.sum(deltas * deltas, axis=1))))


def _high_cost_exposure(grid: CostGrid, points: tuple[WorldPoint, ...], threshold: float) -> float:
    exposure = 0.0
    for previous, current in zip(points[:-1], points[1:]):
        segment_length = _distance(previous, current)
        midpoint = WorldPoint((previous.x + current.x) * 0.5, (previous.y + current.y) * 0.5)
        cell = grid.spec.world_to_cell(midpoint)
        if grid.spec.in_bounds(cell) and grid.cost_at(cell) >= threshold:
            exposure += segment_length
    return float(exposure)


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


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
