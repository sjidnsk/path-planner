from __future__ import annotations

import math

from path_planner.core import CostGrid, WorldPoint
from path_planner.platform import PlannerPlatformProfile
from path_planner.trajectory import TrackablePath

from .guards import HIGH_COST_EXPOSURE_TOLERANCE_M
from .geometry import wrap_angle
from .models import CorridorBox, TrajectoryOptimizationConfig, TrajectoryOptimizationMetrics
from .objective import high_cost_exposure
from .resampling import resample_world_points


def build_metrics(
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
    reference_exposure = high_cost_exposure(grid, reference_metric_points, config.high_cost_threshold)
    optimized_exposure = high_cost_exposure(grid, optimized_points, config.high_cost_threshold)
    curvature_violations = _curvature_violation_count(optimized, platform_profile)
    check_boxes = resampled_boxes if resampled_boxes else boxes
    within_corridor = len(check_boxes) == len(optimized_points) and all(
        box.contains(point) for box, point in zip(check_boxes, optimized_points)
    )
    reference_execution = execution_metrics(reference)
    optimized_execution = execution_metrics(optimized)
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


def build_warnings(
    reference: TrackablePath,
    optimized: TrackablePath,
    metrics: TrajectoryOptimizationMetrics,
) -> tuple[str, ...]:
    warnings: list[str] = []
    reference_execution = execution_metrics(reference)
    if metrics.optimized_high_cost_exposure > metrics.reference_high_cost_exposure + HIGH_COST_EXPOSURE_TOLERANCE_M:
        warnings.append("optimized_high_cost_exposure_worse_than_reference")
    if metrics.tracking_error_proxy >= reference_execution["tracking_error_proxy"] - 1e-9:
        warnings.append("tracking_error_proxy_not_improved")
    if optimized.max_curvature > reference.max_curvature + 1e-9:
        warnings.append("max_curvature_worse_than_reference")
    return tuple(warnings)


def execution_metrics(trackable_path: TrackablePath) -> dict[str, float]:
    spacings = [waypoint.segment_length_m for waypoint in trackable_path.waypoints[1:]]
    heading_changes = [
        abs(wrap_angle(current.heading_rad - previous.heading_rad))
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


def _spacing_uniformity_from_spacings(spacings: list[float]) -> float:
    if not spacings:
        return 0.0
    mean = sum(spacings) / len(spacings)
    return float(sum((spacing - mean) * (spacing - mean) for spacing in spacings))


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
