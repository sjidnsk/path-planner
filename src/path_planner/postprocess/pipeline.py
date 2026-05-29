from __future__ import annotations

from path_planner.core import CostGrid, PlanResult

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
) -> PostprocessResult:
    if not result.success:
        reason = result.failure_reason.value if result.failure_reason else "planning_failed"
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
                violation_indices=(),
                summary=f"postprocess skipped: {reason}",
            ),
            fallback_status=FallbackStatus(used_raw_path=True, reason=reason),
        )

    corridor = build_corridor(grid, result.path_cells, radius_cells=corridor_radius_cells)
    smoothed = smooth_path(grid, result.path_cells, max_shortcut_cost=max_shortcut_cost)
    curvature_points = smoothed.world if smoothed.status != "fallback" else result.path_world
    curvature = check_curvature(
        curvature_points,
        max_curvature=max_curvature,
        min_turning_radius=min_turning_radius,
    )

    fallback_reason = smoothed.fallback_reason or corridor.failure_reason
    return PostprocessResult(
        raw_path_cells=result.path_cells,
        raw_path_world=result.path_world,
        corridor=corridor,
        smoothed_path=smoothed,
        curvature_report=curvature,
        fallback_status=FallbackStatus(used_raw_path=fallback_reason is not None, reason=fallback_reason),
    )
