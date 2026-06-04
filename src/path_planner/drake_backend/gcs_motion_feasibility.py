from __future__ import annotations

import math

from path_planner.core import WorldPoint
from path_planner.postprocess.curvature import check_curvature

from .models import GcsMotionFeasibilityReport, GcsTrajectoryReport

GCS_TRAJECTORY_SOURCE = "gcs_trajectory_sampled_points"
MOTION_MODEL_CURVATURE_BOUNDED = "curvature_bounded"


def build_gcs_motion_feasibility_report(
    gcs_trajectory_report: GcsTrajectoryReport | None,
    *,
    motion_model: str = MOTION_MODEL_CURVATURE_BOUNDED,
    min_turning_radius_m: float | None = None,
    max_heading_change_deg: float = 120.0,
    max_curvature: float = 1.0,
) -> GcsMotionFeasibilityReport:
    if max_heading_change_deg < 0.0:
        raise ValueError("max_heading_change_deg must be nonnegative")
    if min_turning_radius_m is not None and min_turning_radius_m < 0.0:
        raise ValueError("min_turning_radius_m must be nonnegative")
    if gcs_trajectory_report is None:
        return _diagnostic_only(
            "gcs_report_missing",
            motion_model=motion_model,
            min_turning_radius_m=min_turning_radius_m,
            max_heading_change_deg=max_heading_change_deg,
            max_curvature=max_curvature,
        )
    if not gcs_trajectory_report.success:
        return _diagnostic_only(
            "gcs_trajectory_failed",
            motion_model=motion_model,
            min_turning_radius_m=min_turning_radius_m,
            max_heading_change_deg=max_heading_change_deg,
            max_curvature=max_curvature,
            sample_count=gcs_trajectory_report.sample_count,
            path_length=gcs_trajectory_report.path_length,
        )

    points = gcs_trajectory_report.sampled_points
    if len(points) < 3:
        return _diagnostic_only(
            "insufficient_samples",
            motion_model=motion_model,
            min_turning_radius_m=min_turning_radius_m,
            max_heading_change_deg=max_heading_change_deg,
            max_curvature=max_curvature,
            sample_count=len(points),
            path_length=gcs_trajectory_report.path_length,
        )

    curvature = check_curvature(
        points,
        max_curvature=max_curvature,
        min_turning_radius=min_turning_radius_m,
    )
    heading_changes = _heading_changes_deg(points)
    heading_violation_indices = tuple(
        index
        for index, value in heading_changes.items()
        if value > max_heading_change_deg
    )
    curvature_violation_indices = tuple(curvature.violation_indices)
    violation_indices = tuple(sorted(set(curvature_violation_indices) | set(heading_violation_indices)))
    feasibility_status = "feasible" if not violation_indices else "infeasible"
    return GcsMotionFeasibilityReport(
        evaluated=True,
        trajectory_source=GCS_TRAJECTORY_SOURCE,
        motion_model=motion_model,
        feasibility_status=feasibility_status,
        fallback_reason=_constraint_fallback_reason(
            curvature_violation_indices=curvature_violation_indices,
            heading_violation_indices=heading_violation_indices,
        ),
        min_turning_radius_m=min_turning_radius_m,
        max_heading_change_deg=float(max_heading_change_deg),
        curvature_violation_count=len(curvature_violation_indices),
        heading_violation_count=len(heading_violation_indices),
        violation_indices=violation_indices,
        sample_count=len(points),
        path_length=_path_length(points, fallback=gcs_trajectory_report.path_length),
        constraint_summary={
            "motion_model": motion_model,
            "max_curvature": float(max_curvature),
            "min_turning_radius_m": min_turning_radius_m,
            "max_heading_change_deg": float(max_heading_change_deg),
            "max_observed_curvature": curvature.max_curvature,
            "min_observed_turning_radius_m": curvature.min_turning_radius,
            "max_observed_heading_change_deg": max(heading_changes.values(), default=0.0),
            "curvature_violation_indices": list(curvature_violation_indices),
            "heading_violation_indices": list(heading_violation_indices),
        },
    )


def _diagnostic_only(
    reason: str,
    *,
    motion_model: str,
    min_turning_radius_m: float | None,
    max_heading_change_deg: float,
    max_curvature: float,
    sample_count: int = 0,
    path_length: float = 0.0,
) -> GcsMotionFeasibilityReport:
    return GcsMotionFeasibilityReport(
        evaluated=False,
        trajectory_source=GCS_TRAJECTORY_SOURCE,
        motion_model=motion_model,
        feasibility_status="diagnostic_only",
        fallback_reason=reason,
        min_turning_radius_m=min_turning_radius_m,
        max_heading_change_deg=float(max_heading_change_deg),
        curvature_violation_count=0,
        heading_violation_count=0,
        violation_indices=(),
        sample_count=max(int(sample_count), 0),
        path_length=max(float(path_length), 0.0),
        constraint_summary={
            "motion_model": motion_model,
            "max_curvature": float(max_curvature),
            "min_turning_radius_m": min_turning_radius_m,
            "max_heading_change_deg": float(max_heading_change_deg),
            "max_observed_curvature": None,
            "min_observed_turning_radius_m": None,
            "max_observed_heading_change_deg": None,
            "curvature_violation_indices": [],
            "heading_violation_indices": [],
        },
    )


def _heading_changes_deg(points: tuple[WorldPoint, ...]) -> dict[int, float]:
    changes: dict[int, float] = {}
    for index in range(1, len(points) - 1):
        previous_heading = _heading(points[index - 1], points[index])
        next_heading = _heading(points[index], points[index + 1])
        changes[index] = abs(math.degrees(_angle_delta(previous_heading, next_heading)))
    return changes


def _heading(first: WorldPoint, second: WorldPoint) -> float:
    return math.atan2(second.y - first.y, second.x - first.x)


def _angle_delta(first: float, second: float) -> float:
    return math.atan2(math.sin(second - first), math.cos(second - first))


def _constraint_fallback_reason(
    *,
    curvature_violation_indices: tuple[int, ...],
    heading_violation_indices: tuple[int, ...],
) -> str | None:
    if curvature_violation_indices and heading_violation_indices:
        return "motion_constraint_violation"
    if curvature_violation_indices:
        return "curvature_constraint_violation"
    if heading_violation_indices:
        return "heading_constraint_violation"
    return None


def _path_length(points: tuple[WorldPoint, ...], *, fallback: float) -> float:
    total = 0.0
    for first, second in zip(points[:-1], points[1:]):
        total += math.hypot(second.x - first.x, second.y - first.y)
    return float(total if total > 0.0 else fallback)
