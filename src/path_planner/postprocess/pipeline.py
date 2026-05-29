from __future__ import annotations

from path_planner.core import CostGrid, PlanResult
from path_planner.platform import PlannerPlatformProfile
from path_planner.trajectory import build_trackable_path, evaluate_tracking_safety

from .corridor import build_corridor
from .curvature import check_curvature
from .models import CorridorResult, CurvatureReport, FallbackStatus, PostprocessResult, SmoothedPathResult
from .smoothing import smooth_path


def run_postprocess(
    grid: CostGrid,
    result: PlanResult,
    *,
    corridor_radius_cells: int = 1,
    max_curvature: float = 1.0,
    min_turning_radius: float | None = None,
    max_shortcut_cost: float | None = 3.0,
    platform_profile: PlannerPlatformProfile | None = None,
    max_speed_mps: float = 0.2,
    min_speed_mps: float = 0.05,
    tracking_error_bound_m: float = 0.0,
) -> PostprocessResult:
    constraint_warnings = platform_profile.constraint_warnings if platform_profile is not None else ()
    constraint_min_turning_radius = (
        min_turning_radius
        if min_turning_radius is not None
        else platform_profile.effective_min_turning_radius_m
        if platform_profile is not None
        else None
    )
    if not result.success:
        reason = result.failure_reason.value if result.failure_reason else "planning_failed"
        trackable = build_trackable_path(
            grid,
            result.path_cells,
            source_path="raw_path",
            max_speed_mps=max_speed_mps,
            min_speed_mps=min_speed_mps,
        )
        tracking_safety = evaluate_tracking_safety(
            grid,
            trackable,
            platform_profile=platform_profile,
            tracking_error_bound_m=tracking_error_bound_m,
        )
        return PostprocessResult(
            raw_path_cells=result.path_cells,
            raw_path_world=result.path_world,
            corridor=CorridorResult(
                status="failed",
                radius_cells=corridor_radius_cells,
                sections=(),
                failure_reason=reason,
            ),
            smoothed_path=SmoothedPathResult(
                status="fallback",
                cells=result.path_cells,
                world=result.path_world,
                fallback_reason=reason,
            ),
            curvature_report=CurvatureReport(
                is_feasible=False,
                max_curvature=0.0,
                min_turning_radius=None,
                constraint_min_turning_radius=constraint_min_turning_radius,
                violation_indices=(),
                summary=f"postprocess skipped: {reason}",
            ),
            fallback_status=FallbackStatus(used_raw_path=True, reason=reason),
            platform_profile=platform_profile,
            constraint_warnings=constraint_warnings,
            trackable_path=trackable,
            tracking_safety_report=tracking_safety,
        )

    corridor = build_corridor(
        grid,
        result.path_cells,
        radius_cells=corridor_radius_cells,
        platform_profile=platform_profile,
    )
    smoothed = smooth_path(grid, result.path_cells, max_shortcut_cost=max_shortcut_cost, platform_profile=platform_profile)
    curvature_points = smoothed.world if smoothed.status != "fallback" else result.path_world
    curvature = check_curvature(
        curvature_points,
        max_curvature=max_curvature,
        min_turning_radius=constraint_min_turning_radius,
    )
    source_path = "raw_path" if smoothed.status == "fallback" else "smoothed_path"
    source_cells = result.path_cells if source_path == "raw_path" else smoothed.cells
    trackable = build_trackable_path(
        grid,
        source_cells,
        source_path=source_path,
        max_speed_mps=max_speed_mps,
        min_speed_mps=min_speed_mps,
    )
    tracking_safety = evaluate_tracking_safety(
        grid,
        trackable,
        platform_profile=platform_profile,
        tracking_error_bound_m=tracking_error_bound_m,
    )

    fallback_reason = smoothed.fallback_reason or corridor.failure_reason
    return PostprocessResult(
        raw_path_cells=result.path_cells,
        raw_path_world=result.path_world,
        corridor=corridor,
        smoothed_path=smoothed,
        curvature_report=curvature,
        fallback_status=FallbackStatus(used_raw_path=fallback_reason is not None, reason=fallback_reason),
        platform_profile=platform_profile,
        constraint_warnings=constraint_warnings,
        trackable_path=trackable,
        tracking_safety_report=tracking_safety,
    )
