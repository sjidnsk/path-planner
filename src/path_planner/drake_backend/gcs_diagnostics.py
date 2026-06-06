from __future__ import annotations

import math
from typing import Any

from path_planner.core import CostGrid, WorldPoint

from .models import ConvexRegionSequenceItem

DIRECTION_CONE_SCHEMA_VERSION = "gcs_direction_cone_constraint/v1"
GCS_COST_SCHEMA_VERSION = "gcs_cost_summary/v1"
DIRECTION_CONE_BACKEND_FALLBACK_REASON = "direction_cone_backend_constraint_not_supported"
DIRECTION_CONE_RHO_FLOOR_M = 1.0e-4
DIRECTION_CONE_SEED_RHO_RATIO = 0.05
DIRECTION_CONE_WIDTH_RHO_RATIO = 0.25


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
        "rho_lower_bound_min_m": None,
        "rho_source_counts": {},
        "region_width_min_m": None,
        "portal_width_min_m": None,
        "support_width_min_m": None,
        "constraint_tightness_min": None,
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
    rho_floor_m: float = DIRECTION_CONE_RHO_FLOOR_M,
    seed_rho_ratio: float = DIRECTION_CONE_SEED_RHO_RATIO,
    width_rho_ratio: float = DIRECTION_CONE_WIDTH_RHO_RATIO,
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
    rho_lower_bound_values: list[float] = []
    width_values: list[float] = []
    portal_width_values: list[float] = []
    support_width_values: list[float] = []
    constraint_tightness_values: list[float] = []
    rho_source_counts: dict[str, int] = {}

    segment_count = len(points) - 1
    reference_segment_count = len(reference) - 1
    for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        delta = (second.x - first.x, second.y - first.y)
        length = _norm(delta)
        reference_edge_index = _reference_edge_index(index, segment_count, reference_segment_count)
        edge_parameters = _edge_parameters_for_segment(
            regions,
            reference,
            reference_edge_index,
            max_allowed_direction_error_deg=max_allowed_direction_error_deg,
            rho_floor_m=rho_floor_m,
            seed_rho_ratio=seed_rho_ratio,
            width_rho_ratio=width_rho_ratio,
        )
        tangent = (edge_parameters["tangent"][0], edge_parameters["tangent"][1])
        normal = (-tangent[1], tangent[0])
        forward_distance = delta[0] * tangent[0] + delta[1] * tangent[1]
        rho = edge_parameters["rho_lower_bound_m"]
        rho_values.append(rho)
        rho_lower_bound_values.append(rho)
        rho_source = str(edge_parameters["rho_source"])
        rho_source_counts[rho_source] = rho_source_counts.get(rho_source, 0) + 1
        region_width = _region_width(regions, index, segment_count)
        if region_width is not None:
            width_values.append(region_width)
        portal_width = edge_parameters["portal_width_m"]
        if portal_width is not None:
            portal_width_values.append(portal_width)
        support_width = edge_parameters["support_width_m"]
        if support_width is not None:
            support_width_values.append(support_width)
        constraint_tightness = None
        if rho > 0.0:
            constraint_tightness = max(forward_distance, 0.0) / rho
            constraint_tightness_values.append(constraint_tightness)

        segment_flags: list[str] = list(edge_parameters["risk_flags"])
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
                "rho_lower_bound_m": float(rho),
                "rho_source": rho_source,
                "rho_seed_distance_m": float(edge_parameters["rho_seed_distance_m"]),
                "rho_portal_m": (
                    None
                    if edge_parameters["rho_portal_m"] is None
                    else float(edge_parameters["rho_portal_m"])
                ),
                "rho_support_m": (
                    None
                    if edge_parameters["rho_support_m"] is None
                    else float(edge_parameters["rho_support_m"])
                ),
                "eta": float(eta),
                "max_allowed_direction_error_deg": float(max_allowed_direction_error_deg),
                "region_width_m": None if region_width is None else float(region_width),
                "portal_width_m": None if portal_width is None else float(portal_width),
                "support_width_m": None if support_width is None else float(support_width),
                "constraint_tightness": (
                    None if constraint_tightness is None else float(constraint_tightness)
                ),
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
        "rho_lower_bound_min_m": (
            float(min(rho_lower_bound_values)) if rho_lower_bound_values else None
        ),
        "rho_source_counts": dict(sorted(rho_source_counts.items())),
        "region_width_min_m": float(min(width_values)) if width_values else None,
        "portal_width_min_m": float(min(portal_width_values)) if portal_width_values else None,
        "support_width_min_m": float(min(support_width_values)) if support_width_values else None,
        "constraint_tightness_min": (
            float(min(constraint_tightness_values)) if constraint_tightness_values else None
        ),
        "parameter_count": len(parameters),
        "violation_count": int(violation_count),
        "risk_flags": sorted(risk_flags),
        "parameters": parameters,
    }


def direction_cone_edge_parameters(
    first: ConvexRegionSequenceItem,
    second: ConvexRegionSequenceItem,
    *,
    max_allowed_direction_error_deg: float = 45.0,
    rho_floor_m: float = DIRECTION_CONE_RHO_FLOOR_M,
    seed_rho_ratio: float = DIRECTION_CONE_SEED_RHO_RATIO,
    width_rho_ratio: float = DIRECTION_CONE_WIDTH_RHO_RATIO,
) -> dict[str, Any]:
    seed_distance = _seed_distance(first.seed_world, second.seed_world)
    if seed_distance <= 0.0:
        tangent = (1.0, 0.0)
        normal = (0.0, 1.0)
    else:
        tangent = (
            (second.seed_world.x - first.seed_world.x) / seed_distance,
            (second.seed_world.y - first.seed_world.y) / seed_distance,
        )
        normal = (-tangent[1], tangent[0])

    eta = math.tan(math.radians(max_allowed_direction_error_deg))
    portal_width = _portal_width(first, second, normal)
    support_width = min(_support_width(first, normal), _support_width(second, normal))
    rho_seed = seed_rho_ratio * seed_distance
    rho_portal = None if portal_width is None else width_rho_ratio * portal_width
    rho_support = width_rho_ratio * support_width

    candidates = [("seed_distance", rho_seed)]
    if rho_portal is not None and rho_portal > 0.0:
        candidates.append(("portal_width", rho_portal))
    if rho_support > 0.0:
        candidates.append(("support_width", rho_support))
    rho_source, rho_candidate = min(candidates, key=lambda item: item[1])
    rho_lower_bound = max(rho_floor_m, rho_candidate)

    if rho_source == "seed_distance" and rho_portal is not None and rho_support > 0.0:
        rho_source = "seed_distance_portal_support_min"
    elif rho_source == "seed_distance" and rho_support > 0.0:
        rho_source = "seed_distance_support_min_no_portal_overlap"
    elif rho_source == "portal_width":
        rho_source = "portal_width_min"
    elif rho_source == "support_width":
        rho_source = "support_width_min"

    risk_flags: list[str] = []
    if seed_distance <= 0.0:
        risk_flags.append("degenerate_reference_segment")
    if portal_width == 0.0:
        risk_flags.append("degenerate_portal_width")
    elif portal_width is None:
        risk_flags.append("region_portal_gap")
    if support_width <= 0.0:
        risk_flags.append("degenerate_support_width")

    return {
        "tangent": [float(tangent[0]), float(tangent[1])],
        "normal": [float(normal[0]), float(normal[1])],
        "eta": float(eta),
        "seed_distance_m": float(seed_distance),
        "portal_width_m": None if portal_width is None else float(portal_width),
        "support_width_m": float(support_width),
        "rho_lower_bound_m": float(rho_lower_bound),
        "rho_source": rho_source,
        "rho_seed_distance_m": float(max(rho_floor_m, rho_seed)),
        "rho_portal_m": None if rho_portal is None else float(max(rho_floor_m, rho_portal)),
        "rho_support_m": float(max(rho_floor_m, rho_support)),
        "risk_flags": risk_flags,
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
        "terrain_cost_source": "not_evaluated",
        "high_cost_threshold": None,
        "sampled_cell_count": 0,
        "blocked_sample_count": 0,
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
    sampled_cell_count = 0
    blocked_sample_count = 0
    for point in points:
        cell = grid.spec.world_to_cell(point)
        if previous_cell == cell:
            continue
        previous_cell = cell
        if not grid.is_passable(cell):
            blocked_sample_count += 1
            continue
        sampled_cell_count += 1
        cost = grid.cost_at(cell)
        terrain_path_cost += cost
        high_cost_exposure += max(cost - high_cost_threshold, 0.0)
    return {
        "schema_version": GCS_COST_SCHEMA_VERSION,
        "path_length": float(path_length),
        "terrain_path_cost": float(terrain_path_cost),
        "high_cost_exposure": float(high_cost_exposure),
        "terrain_cost_source": "sampled_unique_passable_cells",
        "high_cost_threshold": float(high_cost_threshold),
        "sampled_cell_count": int(sampled_cell_count),
        "blocked_sample_count": int(blocked_sample_count),
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


def _reference_edge_index(segment_index: int, segment_count: int, reference_segment_count: int) -> int:
    if reference_segment_count <= 1 or segment_count <= 1:
        return 0
    return min(
        int(round(segment_index * reference_segment_count / max(segment_count - 1, 1))),
        reference_segment_count - 1,
    )


def _edge_parameters_for_segment(
    regions: tuple[ConvexRegionSequenceItem, ...],
    reference: tuple[WorldPoint, ...],
    reference_edge_index: int,
    *,
    max_allowed_direction_error_deg: float,
    rho_floor_m: float,
    seed_rho_ratio: float,
    width_rho_ratio: float,
) -> dict[str, Any]:
    if len(regions) >= 2 and reference_edge_index < len(regions) - 1:
        return direction_cone_edge_parameters(
            regions[reference_edge_index],
            regions[reference_edge_index + 1],
            max_allowed_direction_error_deg=max_allowed_direction_error_deg,
            rho_floor_m=rho_floor_m,
            seed_rho_ratio=seed_rho_ratio,
            width_rho_ratio=width_rho_ratio,
        )
    tangent = _reference_tangent(reference, reference_edge_index, max(len(reference) - 1, 1))
    seed_distance = _norm(
        (
            reference[reference_edge_index + 1].x - reference[reference_edge_index].x,
            reference[reference_edge_index + 1].y - reference[reference_edge_index].y,
        )
    )
    rho_lower_bound = max(rho_floor_m, seed_rho_ratio * seed_distance)
    return {
        "tangent": [float(tangent[0]), float(tangent[1])],
        "normal": [float(-tangent[1]), float(tangent[0])],
        "eta": float(math.tan(math.radians(max_allowed_direction_error_deg))),
        "seed_distance_m": float(seed_distance),
        "portal_width_m": None,
        "support_width_m": None,
        "rho_lower_bound_m": float(rho_lower_bound),
        "rho_source": "seed_distance_no_region_geometry",
        "rho_seed_distance_m": float(rho_lower_bound),
        "rho_portal_m": None,
        "rho_support_m": None,
        "risk_flags": [],
    }


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


def _portal_width(
    first: ConvexRegionSequenceItem,
    second: ConvexRegionSequenceItem,
    normal: tuple[float, float],
) -> float | None:
    overlap_x = min(first.max_world.x, second.max_world.x) - max(first.min_world.x, second.min_world.x)
    overlap_y = min(first.max_world.y, second.max_world.y) - max(first.min_world.y, second.min_world.y)
    if overlap_x < 0.0 or overlap_y < 0.0:
        return None
    return abs(normal[0]) * max(overlap_x, 0.0) + abs(normal[1]) * max(overlap_y, 0.0)


def _support_width(region: ConvexRegionSequenceItem, normal: tuple[float, float]) -> float:
    width_x = region.max_world.x - region.min_world.x
    width_y = region.max_world.y - region.min_world.y
    return abs(normal[0]) * width_x + abs(normal[1]) * width_y


def _direction_error_deg(segment_tangent: tuple[float, float], reference_tangent: tuple[float, float]) -> float:
    dot = segment_tangent[0] * reference_tangent[0] + segment_tangent[1] * reference_tangent[1]
    cross = segment_tangent[0] * reference_tangent[1] - segment_tangent[1] * reference_tangent[0]
    return abs(math.degrees(math.atan2(cross, dot)))


def _seed_distance(first: WorldPoint, second: WorldPoint) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


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
