from __future__ import annotations

import math

from path_planner.core import WorldPoint

from .models import CurvatureReport, CurvatureSample


def check_curvature(
    points: tuple[WorldPoint, ...],
    *,
    max_curvature: float,
    min_turning_radius: float | None = None,
) -> CurvatureReport:
    if max_curvature < 0.0:
        raise ValueError("max_curvature must be nonnegative")
    if min_turning_radius is not None and min_turning_radius < 0.0:
        raise ValueError("min_turning_radius must be nonnegative")

    curvatures: list[float] = []
    violation_indices: list[int] = []
    samples: list[CurvatureSample] = []
    for index in range(1, len(points) - 1):
        curvature = _curvature(points[index - 1], points[index], points[index + 1])
        curvatures.append(curvature)
        turning_radius = 1.0 / curvature if curvature > 0.0 else None
        violates_curvature = curvature > max_curvature
        violates_radius = (
            min_turning_radius is not None
            and turning_radius is not None
            and turning_radius < min_turning_radius
        )
        violates = violates_curvature or violates_radius
        samples.append(
            CurvatureSample(
                point_index=index,
                turn_angle_deg=_turn_angle_deg(points[index - 1], points[index], points[index + 1]),
                curvature=float(curvature),
                turning_radius=turning_radius,
                violates=violates,
                violates_curvature=violates_curvature,
                violates_min_turning_radius=violates_radius,
            )
        )
        if violates:
            violation_indices.append(index)

    observed_max = max(curvatures, default=0.0)
    positive_radii = [1.0 / value for value in curvatures if value > 0.0]
    violations = tuple(violation_indices)
    return CurvatureReport(
        is_feasible=not violations,
        max_curvature=float(observed_max),
        min_turning_radius=min(positive_radii) if positive_radii else None,
        violation_indices=violations,
        summary="path satisfies curvature limit" if not violations else f"curvature violations: {len(violations)}",
        samples=tuple(samples),
    )


def _turn_angle_deg(a: WorldPoint, b: WorldPoint, c: WorldPoint) -> float:
    ab_x = b.x - a.x
    ab_y = b.y - a.y
    bc_x = c.x - b.x
    bc_y = c.y - b.y
    len_ab = math.hypot(ab_x, ab_y)
    len_bc = math.hypot(bc_x, bc_y)
    if len_ab == 0.0 or len_bc == 0.0:
        return 0.0
    cosine = (ab_x * bc_x + ab_y * bc_y) / (len_ab * len_bc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _curvature(a: WorldPoint, b: WorldPoint, c: WorldPoint) -> float:
    side_ab = math.hypot(b.x - a.x, b.y - a.y)
    side_bc = math.hypot(c.x - b.x, c.y - b.y)
    side_ac = math.hypot(c.x - a.x, c.y - a.y)
    denominator = side_ab * side_bc * side_ac
    if denominator == 0.0:
        return 0.0
    doubled_area = abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
    if doubled_area == 0.0:
        return 0.0
    return 2.0 * doubled_area / denominator
