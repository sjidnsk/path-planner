from __future__ import annotations

import math
from typing import Any

from path_planner.core import CostGrid, WorldPoint

from .models import ConvexRegionSequenceItem

DIRECTION_CONE_SCHEMA_VERSION = "gcs_direction_cone_constraint/v1"
GCS_COST_SCHEMA_VERSION = "gcs_cost_summary/v1"
DIRECTION_CONE_BACKEND_FALLBACK_REASON = "direction_cone_backend_constraint_not_supported"


def direction_cone_not_evaluated_summary(reason: str) -> dict[str, Any]:
    return {
        "schema_version": DIRECTION_CONE_SCHEMA_VERSION,
        "constraint_model": "direction_cone",
        "attempted": True,
        "evaluated": False,
        "status": "not_evaluated",
        "backend_enforced": False,
        "enforcing_backend": None,
        "fallback_reason": reason,
        "solver_constraint_count": 0,
        "max_allowed_direction_error_deg": None,
        "eta": None,
        "rho_min": None,
        "region_width_min_m": None,
        "parameter_count": 0,
        "violation_count": 0,
        "risk_flags": [],
        "parameters": [],
    }


def build_direction_cone_constraint_summary(
    points: tuple[WorldPoint, ...],
    regions: tuple[ConvexRegionSequenceItem, ...],
    *,
    max_allowed_direction_error_deg: float = 45.0,
) -> dict[str, Any]:
    if len(points) < 2:
        return direction_cone_not_evaluated_summary("insufficient_direction_cone_samples")
    eta = math.tan(math.radians(max_allowed_direction_error_deg))
    reference = _reference_points(points, regions)
    if len(reference) < 2:
        return direction_cone_not_evaluated_summary("insufficient_direction_cone_reference")

    parameters: list[dict[str, Any]] = []
    risk_flags: set[str] = set()
    violation_count = 0
    rho_values: list[float] = []
    width_values: list[float] = []

    segment_count = len(points) - 1
    for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        delta = (second.x - first.x, second.y - first.y)
        length = _norm(delta)
        tangent = _reference_tangent(reference, index, segment_count)
        normal = (-tangent[1], tangent[0])
        forward_distance = delta[0] * tangent[0] + delta[1] * tangent[1]
        rho = max(forward_distance, 0.0)
        rho_values.append(rho)
        region_width = _region_width(regions, index, segment_count)
        if region_width is not None:
            width_values.append(region_width)

        segment_flags: list[str] = []
        if length <= 0.0:
            segment_flags.append("zero_length_segment")
        else:
            segment_tangent = (delta[0] / length, delta[1] / length)
            direction_error = _direction_error_deg(segment_tangent, tangent)
            if direction_error > max_allowed_direction_error_deg:
                segment_flags.append("direction_cone_violation")
        if forward_distance <= 0.0:
            segment_flags.append("non_forward_segment")
        if region_width is not None and region_width <= 0.0:
            segment_flags.append("degenerate_region_width")

        if segment_flags:
            violation_count += 1
            risk_flags.update(segment_flags)
        parameters.append(
            {
                "edge_index": index,
                "tangent": [float(tangent[0]), float(tangent[1])],
                "normal": [float(normal[0]), float(normal[1])],
                "rho": float(rho),
                "eta": float(eta),
                "region_width_m": None if region_width is None else float(region_width),
                "direction_error_deg": float(direction_error) if length > 0.0 else None,
                "risk_flags": segment_flags,
            }
        )

    return {
        "schema_version": DIRECTION_CONE_SCHEMA_VERSION,
        "constraint_model": "direction_cone",
        "attempted": True,
        "evaluated": True,
        "status": "violated" if violation_count else "evaluated",
        "backend_enforced": False,
        "enforcing_backend": None,
        "fallback_reason": DIRECTION_CONE_BACKEND_FALLBACK_REASON,
        "solver_constraint_count": 0,
        "max_allowed_direction_error_deg": float(max_allowed_direction_error_deg),
        "eta": float(eta),
        "rho_min": float(min(rho_values)) if rho_values else None,
        "region_width_min_m": float(min(width_values)) if width_values else None,
        "parameter_count": len(parameters),
        "violation_count": int(violation_count),
        "risk_flags": sorted(risk_flags),
        "parameters": parameters,
    }


def mark_direction_cone_backend_enforced(
    summary: dict[str, Any],
    *,
    enforcing_backend: str,
    solver_constraint_count: int,
) -> dict[str, Any]:
    enforced = dict(summary)
    enforced["backend_enforced"] = True
    enforced["enforcing_backend"] = enforcing_backend
    enforced["fallback_reason"] = None
    enforced["solver_constraint_count"] = int(solver_constraint_count)
    enforced["status"] = "enforced" if int(enforced.get("violation_count") or 0) == 0 else "violated"
    return enforced


def empty_gcs_cost_summary() -> dict[str, Any]:
    return {
        "schema_version": GCS_COST_SCHEMA_VERSION,
        "path_length": 0.0,
        "terrain_path_cost": None,
        "high_cost_exposure": None,
        "energy_proxy": None,
        "smoothness_proxy": None,
    }


def build_gcs_cost_summary(
    grid: CostGrid,
    points: tuple[WorldPoint, ...],
    *,
    high_cost_threshold: float = 3.0,
) -> dict[str, Any]:
    path_length = _path_length(points)
    terrain_path_cost = 0.0
    high_cost_exposure = 0.0
    previous_cell = None
    for point in points:
        cell = grid.spec.world_to_cell(point)
        if previous_cell == cell:
            continue
        previous_cell = cell
        if not grid.is_passable(cell):
            continue
        cost = grid.cost_at(cell)
        terrain_path_cost += cost
        high_cost_exposure += max(cost - high_cost_threshold, 0.0)
    return {
        "schema_version": GCS_COST_SCHEMA_VERSION,
        "path_length": float(path_length),
        "terrain_path_cost": float(terrain_path_cost),
        "high_cost_exposure": float(high_cost_exposure),
        "energy_proxy": float(_energy_proxy(points)),
        "smoothness_proxy": float(_smoothness_proxy(points)),
    }


def _reference_points(
    points: tuple[WorldPoint, ...],
    regions: tuple[ConvexRegionSequenceItem, ...],
) -> tuple[WorldPoint, ...]:
    if len(regions) >= 2:
        return tuple(region.seed_world for region in regions)
    return points


def _reference_tangent(
    reference: tuple[WorldPoint, ...],
    segment_index: int,
    segment_count: int,
) -> tuple[float, float]:
    if len(reference) < 2:
        return (1.0, 0.0)
    reference_segment_count = len(reference) - 1
    if segment_count <= 1:
        ref_index = 0
    else:
        ref_index = min(
            int(round(segment_index * reference_segment_count / max(segment_count - 1, 1))),
            reference_segment_count - 1,
        )
    first = reference[ref_index]
    second = reference[ref_index + 1]
    delta = (second.x - first.x, second.y - first.y)
    length = _norm(delta)
    if length <= 0.0:
        return (1.0, 0.0)
    return (delta[0] / length, delta[1] / length)


def _region_width(
    regions: tuple[ConvexRegionSequenceItem, ...],
    segment_index: int,
    segment_count: int,
) -> float | None:
    if not regions:
        return None
    if segment_count <= 1:
        region_index = 0
    else:
        region_index = min(
            int(round(segment_index * (len(regions) - 1) / max(segment_count - 1, 1))),
            len(regions) - 1,
        )
    region = regions[region_index]
    return min(region.max_world.x - region.min_world.x, region.max_world.y - region.min_world.y)


def _direction_error_deg(segment_tangent: tuple[float, float], reference_tangent: tuple[float, float]) -> float:
    dot = segment_tangent[0] * reference_tangent[0] + segment_tangent[1] * reference_tangent[1]
    cross = segment_tangent[0] * reference_tangent[1] - segment_tangent[1] * reference_tangent[0]
    return abs(math.degrees(math.atan2(cross, dot)))


def _norm(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _path_length(points: tuple[WorldPoint, ...]) -> float:
    return sum(
        math.hypot(second.x - first.x, second.y - first.y)
        for first, second in zip(points[:-1], points[1:])
    )


def _energy_proxy(points: tuple[WorldPoint, ...]) -> float:
    return sum(
        (second.x - first.x) ** 2 + (second.y - first.y) ** 2
        for first, second in zip(points[:-1], points[1:])
    )


def _smoothness_proxy(points: tuple[WorldPoint, ...]) -> float:
    total = 0.0
    for before, current, after in zip(points[:-2], points[1:-1], points[2:]):
        first_heading = math.atan2(current.y - before.y, current.x - before.x)
        second_heading = math.atan2(after.y - current.y, after.x - current.x)
        delta = math.atan2(math.sin(second_heading - first_heading), math.cos(second_heading - first_heading))
        total += delta * delta
    return total
