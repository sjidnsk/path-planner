from __future__ import annotations

import math

from path_planner.core import WorldPoint

from .models import CurvatureReport


def check_curvature(points: tuple[WorldPoint, ...], *, max_curvature: float) -> CurvatureReport:
    if max_curvature < 0.0:
        raise ValueError("max_curvature must be nonnegative")

    curvatures: list[float] = []
    violation_indices: list[int] = []
    for index in range(1, len(points) - 1):
        curvature = _curvature(points[index - 1], points[index], points[index + 1])
        curvatures.append(curvature)
        if curvature > max_curvature:
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
    )


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
