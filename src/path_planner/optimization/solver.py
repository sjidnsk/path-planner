from __future__ import annotations

import numpy as np

from path_planner.core import CostGrid, WorldPoint
from path_planner.platform import PlannerPlatformProfile
from path_planner.postprocess.models import CorridorResult
from path_planner.trajectory import TrackablePath

from .corridor_mapping import (
    build_corridor_boxes,
    build_corridor_boxes_for_points,
    build_resampled_points,
    lowest_cost_point_in_box,
    project_to_box,
)
from .guards import apply_high_cost_guard, apply_resampled_high_cost_guard
from .metrics import build_metrics, build_warnings
from .models import (
    TrajectoryOptimizationConfig,
    TrajectoryOptimizationFallbackStatus,
    TrajectoryOptimizationResult,
)
from .objective import cost_gradient, objective
from .rebuild import build_trackable_path_from_world


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

    boxes = build_corridor_boxes(grid, trackable_path, corridor)
    if len(boxes) != len(reference_points) or any(not box.contains(point) for box, point in zip(boxes, reference_points)):
        return _fallback_result(grid, trackable_path, corridor, config, "reference_path_outside_corridor", platform_profile)

    reference = np.asarray([[point.x, point.y] for point in reference_points], dtype=float)
    points = reference.copy()
    objective_initial = objective(grid, points, reference, config)
    solver_status = "converged"
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        previous = points.copy()
        for index in range(1, len(points) - 1):
            smooth_target = 0.5 * (points[index - 1] + points[index + 1])
            cost_target = lowest_cost_point_in_box(grid, boxes[index], reference[index])
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
            candidate = candidate - config.weight_cost * 0.02 * grid.spec.resolution * cost_gradient(grid, candidate)
            points[index] = project_to_box(candidate, boxes[index])

        points[0] = reference[0]
        points[-1] = reference[-1]
        iterations = iteration
        if float(np.linalg.norm(points - previous)) <= config.convergence_tolerance:
            break
    else:
        solver_status = "max_iterations"

    points = apply_high_cost_guard(grid, points, reference, config)
    optimized_points = tuple(WorldPoint(float(x), float(y)) for x, y in points)
    optimized_trackable = build_trackable_path_from_world(grid, trackable_path, optimized_points)
    resampled_points = build_resampled_points(grid, optimized_points, corridor, config)
    resampled_points = apply_resampled_high_cost_guard(grid, resampled_points, reference_points, corridor, config)
    resampled_boxes = build_corridor_boxes_for_points(grid, resampled_points, corridor)
    resampled_trackable = build_trackable_path_from_world(grid, optimized_trackable, resampled_points)
    metrics = build_metrics(
        grid,
        trackable_path,
        resampled_trackable,
        boxes,
        resampled_boxes,
        config,
        objective_initial,
        objective(grid, points, reference, config),
        iterations,
        platform_profile,
    )
    warnings = build_warnings(trackable_path, resampled_trackable, metrics)
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
    optimized_trackable = (
        trackable_path
        if trackable_path.waypoints
        else build_trackable_path_from_world(grid, trackable_path, reference_points)
    )
    boxes = build_corridor_boxes(grid, trackable_path, corridor) if corridor.status == "ok" and corridor.sections else ()
    resampled_points = (
        build_resampled_points(grid, reference_points, corridor, config)
        if corridor.status == "ok" and corridor.sections
        else reference_points
    )
    resampled_boxes = (
        build_corridor_boxes_for_points(grid, resampled_points, corridor)
        if corridor.status == "ok" and corridor.sections
        else ()
    )
    resampled_trackable = build_trackable_path_from_world(grid, optimized_trackable, resampled_points)
    reference = np.asarray([[point.x, point.y] for point in reference_points], dtype=float)
    current_objective = objective(grid, reference, reference, config) if len(reference) else 0.0
    metrics = build_metrics(
        grid,
        trackable_path,
        resampled_trackable,
        boxes,
        resampled_boxes,
        config,
        current_objective,
        current_objective,
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
