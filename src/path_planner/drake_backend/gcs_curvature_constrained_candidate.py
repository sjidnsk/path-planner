from __future__ import annotations

from math import hypot, isfinite

from path_planner.core import Cell, CostGrid, PlanResult, WorldPoint

from .gcs_motion_feasibility import (
    GCS_TRAJECTORY_SOURCE,
    build_gcs_motion_feasibility_report,
)
from .models import (
    ConvexRegionSequenceReport,
    GcsCurvatureConstrainedCandidateReport,
    GcsMotionFeasibilityReport,
    GcsTrajectoryReport,
)

REPAIR_STRATEGY_NONE = "none_required"
REPAIR_STRATEGY_NOT_ATTEMPTED = "not_attempted"
REPAIR_STRATEGY_SMOOTHING = "moving_average_smoothing"


def build_gcs_curvature_constrained_candidate_report(
    grid: CostGrid,
    result: PlanResult,
    convex_region_sequence_report: ConvexRegionSequenceReport | None,
    gcs_trajectory_report: GcsTrajectoryReport | None,
    gcs_motion_feasibility_report: GcsMotionFeasibilityReport | None,
    *,
    min_turning_radius_m: float | None = None,
    max_heading_change_deg: float = 120.0,
    max_curvature: float = 1.0,
    smoothing_passes: int = 4,
) -> GcsCurvatureConstrainedCandidateReport:
    if max_heading_change_deg < 0.0:
        raise ValueError("max_heading_change_deg must be nonnegative")
    if min_turning_radius_m is not None and min_turning_radius_m < 0.0:
        raise ValueError("min_turning_radius_m must be nonnegative")
    if max_curvature < 0.0:
        raise ValueError("max_curvature must be nonnegative")

    if gcs_trajectory_report is None:
        return _unavailable(
            "gcs_report_missing",
            status_before="diagnostic_only",
            status_after="diagnostic_only",
            repair_strategy=REPAIR_STRATEGY_NOT_ATTEMPTED,
            constraint_summary=_base_constraint_summary(
                max_curvature=max_curvature,
                min_turning_radius_m=min_turning_radius_m,
                max_heading_change_deg=max_heading_change_deg,
                repair_passes=0,
            ),
        )

    before = gcs_motion_feasibility_report
    if before is None:
        return _unavailable(
            "motion_report_missing",
            status_before="not_evaluated",
            status_after="not_evaluated",
            repair_strategy=REPAIR_STRATEGY_NOT_ATTEMPTED,
            sample_count=gcs_trajectory_report.sample_count,
            path_length=gcs_trajectory_report.path_length,
            constraint_summary=_base_constraint_summary(
                max_curvature=max_curvature,
                min_turning_radius_m=min_turning_radius_m,
                max_heading_change_deg=max_heading_change_deg,
                repair_passes=0,
            ),
        )

    if not gcs_trajectory_report.success:
        reason = "candidate_collision" if gcs_trajectory_report.collision_count > 0 else "gcs_trajectory_failed"
        return _unavailable(
            reason,
            status_before=before.feasibility_status,
            status_after=before.feasibility_status,
            repair_strategy=REPAIR_STRATEGY_NOT_ATTEMPTED,
            curvature_before=before.curvature_violation_count,
            curvature_after=before.curvature_violation_count,
            heading_before=before.heading_violation_count,
            heading_after=before.heading_violation_count,
            violation_indices_before=before.violation_indices,
            violation_indices_after=before.violation_indices,
            collision_count=gcs_trajectory_report.collision_count,
            sample_count=gcs_trajectory_report.sample_count,
            path_length=gcs_trajectory_report.path_length,
            constraint_summary=_constraint_summary(
                before,
                before,
                max_curvature=max_curvature,
                min_turning_radius_m=min_turning_radius_m,
                max_heading_change_deg=max_heading_change_deg,
                repair_passes=0,
            ),
        )

    points = tuple(gcs_trajectory_report.sampled_points)
    if before.feasibility_status == "diagnostic_only" or not before.evaluated:
        return _unavailable(
            before.fallback_reason or "motion_feasibility_not_evaluated",
            status_before=before.feasibility_status,
            status_after=before.feasibility_status,
            repair_strategy=REPAIR_STRATEGY_NOT_ATTEMPTED,
            curvature_before=before.curvature_violation_count,
            heading_before=before.heading_violation_count,
            violation_indices_before=before.violation_indices,
            sample_count=len(points),
            path_length=gcs_trajectory_report.path_length,
            constraint_summary=_constraint_summary(
                before,
                before,
                max_curvature=max_curvature,
                min_turning_radius_m=min_turning_radius_m,
                max_heading_change_deg=max_heading_change_deg,
                repair_passes=0,
            ),
        )

    if before.feasibility_status == "feasible":
        metrics = _path_metrics(grid, points)
        containment_violations = _region_containment_violation_count(convex_region_sequence_report, points)
        fallback_reason = _post_repair_blocker(
            collision_count=metrics.collision_count,
            containment_violations=containment_violations,
            before=before,
            after=before,
        )
        return GcsCurvatureConstrainedCandidateReport(
            attempted=True,
            available=fallback_reason is None,
            selected=fallback_reason is None,
            repair_success=False,
            source=GCS_TRAJECTORY_SOURCE,
            repair_strategy=REPAIR_STRATEGY_NONE,
            status_before=before.feasibility_status,
            status_after=before.feasibility_status,
            fallback_reason=fallback_reason,
            curvature_violation_count_before=before.curvature_violation_count,
            curvature_violation_count_after=before.curvature_violation_count,
            heading_violation_count_before=before.heading_violation_count,
            heading_violation_count_after=before.heading_violation_count,
            violation_indices_before=before.violation_indices,
            violation_indices_after=before.violation_indices,
            region_containment_violation_count=containment_violations,
            collision_count=metrics.collision_count,
            path_length=metrics.path_length,
            path_cost=metrics.path_cost,
            cost_delta_vs_baseline=_cost_delta_vs_baseline(grid, result, metrics.path_cost),
            constraint_summary=_constraint_summary(
                before,
                before,
                max_curvature=max_curvature,
                min_turning_radius_m=min_turning_radius_m,
                max_heading_change_deg=max_heading_change_deg,
                repair_passes=0,
            ),
        )

    if convex_region_sequence_report is None or not convex_region_sequence_report.regions:
        return _unavailable(
            "portal_overlap_insufficient",
            status_before=before.feasibility_status,
            status_after=before.feasibility_status,
            repair_strategy=REPAIR_STRATEGY_NOT_ATTEMPTED,
            curvature_before=before.curvature_violation_count,
            heading_before=before.heading_violation_count,
            violation_indices_before=before.violation_indices,
            sample_count=len(points),
            path_length=gcs_trajectory_report.path_length,
            constraint_summary=_constraint_summary(
                before,
                before,
                max_curvature=max_curvature,
                min_turning_radius_m=min_turning_radius_m,
                max_heading_change_deg=max_heading_change_deg,
                repair_passes=0,
            ),
        )

    repair_passes = max(int(smoothing_passes), 1)
    repaired_points = _smooth_points(points, passes=repair_passes)
    metrics = _path_metrics(grid, repaired_points)
    containment_violations = _region_containment_violation_count(convex_region_sequence_report, repaired_points)
    repaired_gcs_report = _gcs_report_for_points(gcs_trajectory_report, repaired_points)
    after = build_gcs_motion_feasibility_report(
        repaired_gcs_report,
        min_turning_radius_m=min_turning_radius_m,
        max_heading_change_deg=max_heading_change_deg,
        max_curvature=max_curvature,
    )
    fallback_reason = _post_repair_blocker(
        collision_count=metrics.collision_count,
        containment_violations=containment_violations,
        before=before,
        after=after,
    )
    selected = fallback_reason is None and after.feasibility_status == "feasible"
    return GcsCurvatureConstrainedCandidateReport(
        attempted=True,
        available=selected,
        selected=selected,
        repair_success=selected and _violation_count(after) < _violation_count(before),
        source=GCS_TRAJECTORY_SOURCE,
        repair_strategy=REPAIR_STRATEGY_SMOOTHING,
        status_before=before.feasibility_status,
        status_after=after.feasibility_status,
        fallback_reason=fallback_reason,
        curvature_violation_count_before=before.curvature_violation_count,
        curvature_violation_count_after=after.curvature_violation_count,
        heading_violation_count_before=before.heading_violation_count,
        heading_violation_count_after=after.heading_violation_count,
        violation_indices_before=before.violation_indices,
        violation_indices_after=after.violation_indices,
        region_containment_violation_count=containment_violations,
        collision_count=metrics.collision_count,
        path_length=metrics.path_length,
        path_cost=metrics.path_cost if metrics.collision_count == 0 else None,
        cost_delta_vs_baseline=(
            _cost_delta_vs_baseline(grid, result, metrics.path_cost) if metrics.collision_count == 0 else None
        ),
        constraint_summary=_constraint_summary(
            before,
            after,
            max_curvature=max_curvature,
            min_turning_radius_m=min_turning_radius_m,
            max_heading_change_deg=max_heading_change_deg,
            repair_passes=repair_passes,
        ),
    )


class _PathMetrics:
    def __init__(
        self,
        *,
        path_length: float,
        path_cost: float,
        collision_count: int,
    ) -> None:
        self.path_length = path_length
        self.path_cost = path_cost
        self.collision_count = collision_count


def _unavailable(
    reason: str,
    *,
    status_before: str,
    status_after: str,
    repair_strategy: str,
    curvature_before: int = 0,
    curvature_after: int = 0,
    heading_before: int = 0,
    heading_after: int = 0,
    violation_indices_before: tuple[int, ...] = (),
    violation_indices_after: tuple[int, ...] = (),
    collision_count: int = 0,
    sample_count: int = 0,
    path_length: float | None = None,
    constraint_summary: dict | None = None,
) -> GcsCurvatureConstrainedCandidateReport:
    summary = dict(constraint_summary or {})
    summary.setdefault("sample_count", int(sample_count))
    return GcsCurvatureConstrainedCandidateReport(
        attempted=True,
        available=False,
        selected=False,
        repair_success=False,
        source=GCS_TRAJECTORY_SOURCE,
        repair_strategy=repair_strategy,
        status_before=status_before,
        status_after=status_after,
        fallback_reason=reason,
        curvature_violation_count_before=curvature_before,
        curvature_violation_count_after=curvature_after,
        heading_violation_count_before=heading_before,
        heading_violation_count_after=heading_after,
        violation_indices_before=tuple(violation_indices_before),
        violation_indices_after=tuple(violation_indices_after),
        region_containment_violation_count=0,
        collision_count=collision_count,
        path_length=path_length,
        path_cost=None,
        cost_delta_vs_baseline=None,
        constraint_summary=summary,
    )


def _smooth_points(points: tuple[WorldPoint, ...], *, passes: int) -> tuple[WorldPoint, ...]:
    if len(points) < 3:
        return points
    current = tuple(points)
    for _ in range(passes):
        smoothed = [current[0]]
        for index in range(1, len(current) - 1):
            previous = current[index - 1]
            point = current[index]
            following = current[index + 1]
            smoothed.append(
                WorldPoint(
                    0.25 * previous.x + 0.5 * point.x + 0.25 * following.x,
                    0.25 * previous.y + 0.5 * point.y + 0.25 * following.y,
                )
            )
        smoothed.append(current[-1])
        current = tuple(smoothed)
    return current


def _gcs_report_for_points(
    base_report: GcsTrajectoryReport,
    points: tuple[WorldPoint, ...],
) -> GcsTrajectoryReport:
    return GcsTrajectoryReport(
        attempted=True,
        success=True,
        backend=base_report.backend,
        result_status="curvature_constrained_candidate",
        reason="curvature_constrained_candidate",
        sample_count=len(points),
        collision_count=0,
        path_length=_world_path_length(points),
        region_count=base_report.region_count,
        sampled_points=points,
    )


def _path_metrics(grid: CostGrid, points: tuple[WorldPoint, ...]) -> _PathMetrics:
    path_cost = 0.0
    collision_count = 0
    previous_cell: Cell | None = None
    for point in points:
        cell = grid.spec.world_to_cell(point)
        if not grid.is_passable(cell):
            collision_count += 1
            continue
        if cell == previous_cell:
            continue
        previous_cell = cell
        path_cost += grid.cost_at(cell)
    return _PathMetrics(
        path_length=_world_path_length(points),
        path_cost=float(path_cost),
        collision_count=collision_count,
    )


def _region_containment_violation_count(
    report: ConvexRegionSequenceReport | None,
    points: tuple[WorldPoint, ...],
) -> int:
    if report is None or not report.regions:
        return len(points)
    return sum(1 for point in points if not _point_in_any_region(report, point))


def _point_in_any_region(report: ConvexRegionSequenceReport, point: WorldPoint) -> bool:
    return any(_point_in_hpolyhedron(region.hpolyhedron_a, region.hpolyhedron_b, point) for region in report.regions)


def _point_in_hpolyhedron(
    a_matrix: tuple[tuple[float, ...], ...],
    b_vector: tuple[float, ...],
    point: WorldPoint,
    *,
    tolerance: float = 1e-9,
) -> bool:
    for row, bound in zip(a_matrix, b_vector):
        if row[0] * point.x + row[1] * point.y > bound + tolerance:
            return False
    return True


def _post_repair_blocker(
    *,
    collision_count: int,
    containment_violations: int,
    before: GcsMotionFeasibilityReport,
    after: GcsMotionFeasibilityReport,
) -> str | None:
    if collision_count > 0:
        return "candidate_collision"
    if containment_violations > 0:
        return "region_containment_failed"
    if after.feasibility_status == "feasible":
        return None
    if after.heading_violation_count > 0 and after.curvature_violation_count == 0:
        return "heading_transition_infeasible"
    if _violation_count(after) >= _violation_count(before):
        return "curvature_repair_not_better"
    return "corridor_too_narrow_for_turning_radius"


def _constraint_summary(
    before: GcsMotionFeasibilityReport,
    after: GcsMotionFeasibilityReport,
    *,
    max_curvature: float,
    min_turning_radius_m: float | None,
    max_heading_change_deg: float,
    repair_passes: int,
) -> dict:
    before_summary = dict(before.constraint_summary)
    after_summary = dict(after.constraint_summary)
    return {
        **_base_constraint_summary(
            max_curvature=max_curvature,
            min_turning_radius_m=min_turning_radius_m,
            max_heading_change_deg=max_heading_change_deg,
            repair_passes=repair_passes,
        ),
        "status_before": before.feasibility_status,
        "status_after": after.feasibility_status,
        "curvature_violation_indices_before": list(before.violation_indices),
        "curvature_violation_indices_after": list(after.violation_indices),
        "max_observed_curvature_before": before_summary.get("max_observed_curvature"),
        "max_observed_curvature_after": after_summary.get("max_observed_curvature"),
        "max_observed_heading_change_deg_before": before_summary.get("max_observed_heading_change_deg"),
        "max_observed_heading_change_deg_after": after_summary.get("max_observed_heading_change_deg"),
    }


def _base_constraint_summary(
    *,
    max_curvature: float,
    min_turning_radius_m: float | None,
    max_heading_change_deg: float,
    repair_passes: int,
) -> dict:
    return {
        "motion_model": "curvature_bounded",
        "max_curvature": float(max_curvature),
        "min_turning_radius_m": min_turning_radius_m,
        "max_heading_change_deg": float(max_heading_change_deg),
        "repair_passes": int(repair_passes),
        "repair_strategy": REPAIR_STRATEGY_SMOOTHING if repair_passes > 0 else REPAIR_STRATEGY_NOT_ATTEMPTED,
    }


def _violation_count(report: GcsMotionFeasibilityReport) -> int:
    return int(report.curvature_violation_count) + int(report.heading_violation_count)


def _world_path_length(points: tuple[WorldPoint, ...]) -> float:
    total = 0.0
    for first, second in zip(points[:-1], points[1:]):
        total += hypot(second.x - first.x, second.y - first.y)
    return float(total)


def _cost_delta_vs_baseline(grid: CostGrid, result: PlanResult, candidate_cost: float) -> float | None:
    baseline_cost = _finite_float(result.total_cost)
    if baseline_cost is None:
        baseline_cost = _cell_path_cost(grid, result.path_cells)
    if baseline_cost is None:
        return None
    return float(candidate_cost - baseline_cost)


def _cell_path_cost(grid: CostGrid, cells: tuple[Cell, ...]) -> float | None:
    if not cells:
        return None
    total = 0.0
    previous: Cell | None = None
    for cell in cells:
        if cell == previous:
            continue
        previous = cell
        if grid.is_passable(cell):
            total += grid.cost_at(cell)
    return float(total)


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return number
