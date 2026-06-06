from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint

from .gcs_diagnostics import (
    DIRECTION_CONE_RHO_FLOOR_M,
    DIRECTION_CONE_SEED_RHO_RATIO,
    DIRECTION_CONE_WIDTH_RHO_RATIO,
    build_direction_cone_constraint_summary,
    build_gcs_cost_summary,
    direction_cone_edge_parameters,
    direction_cone_not_evaluated_summary,
    empty_gcs_cost_summary,
    mark_direction_cone_backend_enforced,
)
from .gcs_trajectory import (
    _has_region_adjacency,
    _load_gcs_dependencies,
    _low_cost_anchor,
    _point_in_region,
    _region_matrices,
    _resample_polyline,
    _sample_collision_count,
    _sampled_path_length,
    _solution_status,
    _solution_success,
    _solver_exception_reason,
    _solver_result_reason,
    _validate_region_hpolyhedra,
)
from .models import ConvexRegionSequenceItem, ConvexRegionSequenceReport, GcsTrajectoryReport

PYDRAKE_CONTROL_POINT_BACKEND = "pydrake_control_point_direction_cone_program"
CONTROL_POINT_ENFORCING_BACKEND = "pydrake_control_point_mathematical_program"
CONTROL_POINT_PARAMETERIZATION = "control_point_derivative_proxy"
DERIVATIVE_PROXY = "successive_control_point_difference"
SEGMENT_LENGTH_OBJECTIVE_WEIGHT = 1.0
LOW_COST_ANCHOR_OBJECTIVE_WEIGHT = 0.1
CONTROL_POINT_TERRAIN_OBJECTIVE_WEIGHT = 0.05
CONTROL_POINT_SECOND_DIFFERENCE_OBJECTIVE_WEIGHT = 0.2
CONTROL_POINT_HIGH_COST_EXPOSURE_OBJECTIVE_WEIGHT = 0.0
CONTROL_POINT_TERRAIN_OBJECTIVE_SOURCE = "region_inverse_cost_weighted_passable_cell_centroid"
CONTROL_POINT_HIGH_COST_EXPOSURE_OBJECTIVE_SOURCE = "region_high_cost_exposure_proxy"
CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY = "proxy_not_continuous_field_integral"
OBJECTIVE_TERM_WEIGHTS = {
    "segment_length_quadratic": SEGMENT_LENGTH_OBJECTIVE_WEIGHT,
    "low_cost_anchor_quadratic": LOW_COST_ANCHOR_OBJECTIVE_WEIGHT,
    "control_point_terrain_anchor_quadratic": CONTROL_POINT_TERRAIN_OBJECTIVE_WEIGHT,
    "control_point_second_difference_quadratic": CONTROL_POINT_SECOND_DIFFERENCE_OBJECTIVE_WEIGHT,
}
HIGH_COST_EXPOSURE_OBJECTIVE_TERM = "control_point_high_cost_exposure_proxy_quadratic"
OBJECTIVE_TERMS = tuple(OBJECTIVE_TERM_WEIGHTS)


@dataclass(frozen=True)
class GcsControlPointSolverConfig:
    terrain_objective_weight: float = CONTROL_POINT_TERRAIN_OBJECTIVE_WEIGHT
    second_difference_weight: float = CONTROL_POINT_SECOND_DIFFERENCE_OBJECTIVE_WEIGHT
    high_cost_exposure_weight: float = CONTROL_POINT_HIGH_COST_EXPOSURE_OBJECTIVE_WEIGHT
    direction_cone_max_error_deg: float = 45.0
    direction_cone_rho_floor_m: float = DIRECTION_CONE_RHO_FLOOR_M
    direction_cone_seed_rho_ratio: float = DIRECTION_CONE_SEED_RHO_RATIO
    direction_cone_width_rho_ratio: float = DIRECTION_CONE_WIDTH_RHO_RATIO

    def __post_init__(self) -> None:
        if self.terrain_objective_weight < 0.0:
            raise ValueError("terrain_objective_weight must be non-negative")
        if self.second_difference_weight < 0.0:
            raise ValueError("second_difference_weight must be non-negative")
        if self.high_cost_exposure_weight < 0.0:
            raise ValueError("high_cost_exposure_weight must be non-negative")
        if not 0.0 < self.direction_cone_max_error_deg < 90.0:
            raise ValueError("direction_cone_max_error_deg must be between 0 and 90")
        if self.direction_cone_rho_floor_m < 0.0:
            raise ValueError("direction_cone_rho_floor_m must be non-negative")
        if self.direction_cone_seed_rho_ratio < 0.0:
            raise ValueError("direction_cone_seed_rho_ratio must be non-negative")
        if self.direction_cone_width_rho_ratio < 0.0:
            raise ValueError("direction_cone_width_rho_ratio must be non-negative")

    @property
    def objective_term_weights(self) -> dict[str, float]:
        weights = {
            "segment_length_quadratic": SEGMENT_LENGTH_OBJECTIVE_WEIGHT,
            "low_cost_anchor_quadratic": LOW_COST_ANCHOR_OBJECTIVE_WEIGHT,
            "control_point_terrain_anchor_quadratic": float(self.terrain_objective_weight),
            "control_point_second_difference_quadratic": float(self.second_difference_weight),
        }
        if self.high_cost_exposure_weight > 0.0:
            weights[HIGH_COST_EXPOSURE_OBJECTIVE_TERM] = float(self.high_cost_exposure_weight)
        return weights

    @property
    def objective_terms(self) -> tuple[str, ...]:
        terms = list(OBJECTIVE_TERMS)
        if self.high_cost_exposure_weight > 0.0:
            terms.append(HIGH_COST_EXPOSURE_OBJECTIVE_TERM)
        return tuple(terms)


@dataclass(frozen=True)
class _TerrainObjectiveAnchor:
    region_id: int
    point: WorldPoint
    nearest_cell: Cell | None
    cost: float | None
    min_region_cost: float | None
    mean_region_cost: float | None
    passable_cell_count: int
    high_cost_cell_count: int
    objective_weight: float
    high_cost_exposure_proxy_cost: float | None
    high_cost_exposure_objective_weight: float


def build_gcs_control_point_trajectory_report(
    grid: CostGrid,
    convex_region_sequence_report: ConvexRegionSequenceReport | None,
    *,
    sample_count: int = 25,
    config: GcsControlPointSolverConfig | None = None,
) -> GcsTrajectoryReport:
    config = GcsControlPointSolverConfig() if config is None else config
    if convex_region_sequence_report is None:
        return _not_attempted("convex_region_report_missing", "convex_region_report_missing", config=config)
    if not convex_region_sequence_report.gcs_ready:
        return _not_attempted(
            "convex_region_not_gcs_ready",
            convex_region_sequence_report.gcs_ready_reason,
            region_count=convex_region_sequence_report.region_count,
            config=config,
        )
    regions = convex_region_sequence_report.regions
    if not regions:
        return _not_attempted("convex_region_not_gcs_ready", "convex_region_sequence_empty", config=config)
    if not _has_region_adjacency(regions):
        return _not_attempted(
            "region_overlap_missing",
            "region_overlap_missing",
            region_count=convex_region_sequence_report.region_count,
            config=config,
        )

    try:
        deps = _load_gcs_dependencies()
    except Exception as exc:
        return _not_attempted(
            f"{type(exc).__name__}: {exc}",
            "pydrake_unavailable",
            region_count=convex_region_sequence_report.region_count,
            config=config,
        )

    sample_count = max(int(sample_count), 2)
    try:
        _validate_region_hpolyhedra(regions)
    except Exception as exc:
        return _attempted_failure(
            result_status=f"{type(exc).__name__}: {exc}",
            reason="invalid_hpolyhedron",
            region_count=convex_region_sequence_report.region_count,
            config=config,
        )

    try:
        sampled_points, result, solver_counts, terrain_anchors = _solve_control_point_path(
            deps,
            grid,
            regions,
            sample_count=sample_count,
            config=config,
        )
    except Exception as exc:
        return _attempted_failure(
            result_status=f"{type(exc).__name__}: {exc}",
            reason=_solver_exception_reason(exc),
            region_count=convex_region_sequence_report.region_count,
            config=config,
        )

    result_status = _solution_status(result)
    if not _solution_success(result):
        return _attempted_failure(
            result_status=result_status,
            reason=_solver_result_reason(result_status),
            region_count=convex_region_sequence_report.region_count,
            config=config,
        )

    collision_count = _sample_collision_count(grid, sampled_points)
    path_length = _sampled_path_length(sampled_points)
    constraint_summary = _control_point_constraint_summary(
        sampled_points,
        regions,
        solver_counts=solver_counts,
        terrain_objective_summary=_terrain_objective_summary(terrain_anchors, config=config),
        config=config,
    )
    cost_summary = _control_point_cost_summary(
        build_gcs_cost_summary(grid, sampled_points),
        terrain_objective_summary=_terrain_objective_summary(terrain_anchors, config=config),
    )
    if collision_count > 0:
        return GcsTrajectoryReport(
            attempted=True,
            success=False,
            backend=PYDRAKE_CONTROL_POINT_BACKEND,
            result_status="sampled_trajectory_collision",
            reason="sampled_trajectory_collision",
            sample_count=len(sampled_points),
            collision_count=collision_count,
            path_length=path_length,
            region_count=convex_region_sequence_report.region_count,
            sampled_points=sampled_points,
            constraint_summary=constraint_summary,
            cost_summary=cost_summary,
        )
    return GcsTrajectoryReport(
        attempted=True,
        success=True,
        backend=PYDRAKE_CONTROL_POINT_BACKEND,
        result_status=result_status,
        reason="control_point_direction_cone_solution_found",
        sample_count=len(sampled_points),
        collision_count=0,
        path_length=path_length,
        region_count=convex_region_sequence_report.region_count,
        sampled_points=sampled_points,
        constraint_summary=constraint_summary,
        cost_summary=cost_summary,
    )


def _not_attempted(
    result_status: str,
    reason: str,
    *,
    config: GcsControlPointSolverConfig,
    region_count: int = 0,
) -> GcsTrajectoryReport:
    return GcsTrajectoryReport(
        attempted=False,
        success=False,
        backend=PYDRAKE_CONTROL_POINT_BACKEND,
        result_status=result_status,
        reason=reason,
        sample_count=0,
        collision_count=0,
        path_length=0.0,
        region_count=region_count,
        constraint_summary=_not_evaluated_summary(reason, config=config),
        cost_summary=_not_evaluated_cost_summary(reason, config=config),
    )


def _attempted_failure(
    *,
    result_status: str,
    reason: str,
    region_count: int,
    config: GcsControlPointSolverConfig,
) -> GcsTrajectoryReport:
    return GcsTrajectoryReport(
        attempted=True,
        success=False,
        backend=PYDRAKE_CONTROL_POINT_BACKEND,
        result_status=result_status,
        reason=reason,
        sample_count=0,
        collision_count=0,
        path_length=0.0,
        region_count=region_count,
        constraint_summary=_not_evaluated_summary(reason, config=config),
        cost_summary=_not_evaluated_cost_summary(reason, config=config),
    )


def _solve_control_point_path(
    deps: dict[str, Any],
    grid: CostGrid,
    regions: tuple[ConvexRegionSequenceItem, ...],
    *,
    sample_count: int,
    config: GcsControlPointSolverConfig,
) -> tuple[tuple[WorldPoint, ...], Any, dict[str, int], tuple[_TerrainObjectiveAnchor, ...]]:
    if len(regions) < 2:
        raise ValueError("control-point direction_cone backend requires at least two regions")

    prog = deps["MathematicalProgram"]()
    control_points = prog.NewContinuousVariables(len(regions), 2, "cp")
    terrain_anchors = _terrain_objective_anchors(grid, regions, config=config)
    counts = {
        "solver_constraint_count": 0,
        "control_point_region_containment_count": 0,
        "start_goal_constraint_count": 0,
        "derivative_constraint_count": 0,
    }

    for index, region in enumerate(regions):
        a_matrix, b_vector = _region_matrices(region)
        for row, bound in zip(a_matrix, b_vector):
            prog.AddLinearConstraint(
                row[0] * control_points[index, 0] + row[1] * control_points[index, 1],
                -np.inf,
                bound,
            )
            counts["solver_constraint_count"] += 1
            counts["control_point_region_containment_count"] += 1

    first_seed = regions[0].seed_world
    last_seed = regions[-1].seed_world
    for variable, value in (
        (control_points[0, 0], first_seed.x),
        (control_points[0, 1], first_seed.y),
        (control_points[-1, 0], last_seed.x),
        (control_points[-1, 1], last_seed.y),
    ):
        prog.AddBoundingBoxConstraint(value, value, variable)
        counts["solver_constraint_count"] += 1
        counts["start_goal_constraint_count"] += 1

    for index, (first, second) in enumerate(zip(regions[:-1], regions[1:])):
        edge_parameters = direction_cone_edge_parameters(
            first,
            second,
            max_allowed_direction_error_deg=config.direction_cone_max_error_deg,
            rho_floor_m=config.direction_cone_rho_floor_m,
            seed_rho_ratio=config.direction_cone_seed_rho_ratio,
            width_rho_ratio=config.direction_cone_width_rho_ratio,
        )
        if edge_parameters["seed_distance_m"] <= 0.0:
            raise ValueError("direction_cone reference segment is degenerate")
        tangent = np.asarray(edge_parameters["tangent"], dtype=float)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        derivative_x = control_points[index + 1, 0] - control_points[index, 0]
        derivative_y = control_points[index + 1, 1] - control_points[index, 1]
        forward = tangent[0] * derivative_x + tangent[1] * derivative_y
        lateral = normal[0] * derivative_x + normal[1] * derivative_y
        eta = float(edge_parameters["eta"])
        rho = float(edge_parameters["rho_lower_bound_m"])

        prog.AddLinearConstraint(forward, rho, np.inf)
        prog.AddLinearConstraint(lateral - eta * forward, -np.inf, 0.0)
        prog.AddLinearConstraint(-lateral - eta * forward, -np.inf, 0.0)
        counts["solver_constraint_count"] += 3
        counts["derivative_constraint_count"] += 3

    objective = 0.0
    for index in range(len(regions) - 1):
        dx = control_points[index + 1, 0] - control_points[index, 0]
        dy = control_points[index + 1, 1] - control_points[index, 1]
        objective += dx * dx + dy * dy
    for index, region in enumerate(regions):
        anchor = _low_cost_anchor(grid, region)
        objective += LOW_COST_ANCHOR_OBJECTIVE_WEIGHT * (
            (control_points[index, 0] - anchor.x) ** 2
            + (control_points[index, 1] - anchor.y) ** 2
        )
    for index, anchor in enumerate(terrain_anchors):
        objective += anchor.objective_weight * (
            (control_points[index, 0] - anchor.point.x) ** 2
            + (control_points[index, 1] - anchor.point.y) ** 2
        )
        if anchor.high_cost_exposure_objective_weight > 0.0:
            objective += anchor.high_cost_exposure_objective_weight * (
                (control_points[index, 0] - anchor.point.x) ** 2
                + (control_points[index, 1] - anchor.point.y) ** 2
            )
    for index in range(1, len(regions) - 1):
        ddx = control_points[index + 1, 0] - 2.0 * control_points[index, 0] + control_points[index - 1, 0]
        ddy = control_points[index + 1, 1] - 2.0 * control_points[index, 1] + control_points[index - 1, 1]
        objective += config.second_difference_weight * (ddx * ddx + ddy * ddy)
    prog.AddQuadraticCost(objective)

    for index, region in enumerate(regions):
        prog.SetInitialGuess(control_points[index, 0], region.seed_world.x)
        prog.SetInitialGuess(control_points[index, 1], region.seed_world.y)

    result = deps["Solve"](prog)
    if not _solution_success(result):
        return (), result, counts, terrain_anchors

    solution = result.GetSolution(control_points)
    control_polyline = tuple(WorldPoint(float(point[0]), float(point[1])) for point in solution)
    return _resample_polyline(control_polyline, sample_count=sample_count), result, counts, terrain_anchors


def _control_point_constraint_summary(
    sampled_points: tuple[WorldPoint, ...],
    regions: tuple[ConvexRegionSequenceItem, ...],
    *,
    solver_counts: dict[str, int],
    terrain_objective_summary: dict[str, Any],
    config: GcsControlPointSolverConfig,
) -> dict[str, Any]:
    summary = mark_direction_cone_backend_enforced(
        build_direction_cone_constraint_summary(
            sampled_points,
            regions,
            max_allowed_direction_error_deg=config.direction_cone_max_error_deg,
            rho_floor_m=config.direction_cone_rho_floor_m,
            seed_rho_ratio=config.direction_cone_seed_rho_ratio,
            width_rho_ratio=config.direction_cone_width_rho_ratio,
        ),
        enforcing_backend=CONTROL_POINT_ENFORCING_BACKEND,
        solver_constraint_count=solver_counts["solver_constraint_count"],
    )
    summary.update(
        {
            "trajectory_parameterization": CONTROL_POINT_PARAMETERIZATION,
            "control_point_count": len(regions),
            "derivative_proxy": DERIVATIVE_PROXY,
            "derivative_constraint_count": solver_counts["derivative_constraint_count"],
            "control_point_region_containment_count": solver_counts[
                "control_point_region_containment_count"
            ],
            "start_goal_constraint_count": solver_counts["start_goal_constraint_count"],
            "objective_terms": list(config.objective_terms),
            "objective_term_weights": config.objective_term_weights,
            "terrain_objective_source": terrain_objective_summary["terrain_objective_source"],
            "terrain_objective_weight": terrain_objective_summary["terrain_objective_weight"],
            "terrain_objective_anchor_count": terrain_objective_summary[
                "terrain_objective_anchor_count"
            ],
            "terrain_objective_boundary": CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY,
        }
    )
    summary.update(_high_cost_exposure_constraint_fields(terrain_objective_summary, config=config))
    return summary


def _not_evaluated_summary(reason: str, *, config: GcsControlPointSolverConfig) -> dict[str, Any]:
    summary = direction_cone_not_evaluated_summary(reason)
    summary.update(
        {
            "trajectory_parameterization": CONTROL_POINT_PARAMETERIZATION,
            "control_point_count": 0,
            "derivative_proxy": DERIVATIVE_PROXY,
            "derivative_constraint_count": 0,
            "control_point_region_containment_count": 0,
            "start_goal_constraint_count": 0,
            "objective_terms": list(config.objective_terms),
            "objective_term_weights": config.objective_term_weights,
            "terrain_objective_source": "not_evaluated",
            "terrain_objective_weight": None,
            "terrain_objective_anchor_count": 0,
            "terrain_objective_boundary": CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY,
        }
    )
    summary.update(_not_evaluated_high_cost_exposure_fields(config=config))
    return summary


def _terrain_objective_anchors(
    grid: CostGrid,
    regions: tuple[ConvexRegionSequenceItem, ...],
    *,
    config: GcsControlPointSolverConfig,
    high_cost_threshold: float = 3.0,
) -> tuple[_TerrainObjectiveAnchor, ...]:
    anchors: list[_TerrainObjectiveAnchor] = []
    cost_floor = max(grid.min_passable_cost(), 1.0e-6)
    for region in regions:
        samples: list[tuple[Cell, WorldPoint, float]] = []
        for y in range(region.min_cell.y, region.max_cell.y + 1):
            for x in range(region.min_cell.x, region.max_cell.x + 1):
                cell = Cell(x, y)
                if not grid.spec.in_bounds(cell) or not grid.is_passable(cell):
                    continue
                point = grid.spec.cell_to_world(cell)
                if not _point_in_region(point, region):
                    continue
                samples.append((cell, point, grid.cost_at(cell)))
        if not samples:
            fallback_cost = grid.cost_at(region.seed_cell) if grid.is_passable(region.seed_cell) else None
            anchors.append(
                _TerrainObjectiveAnchor(
                    region_id=region.region_id,
                    point=region.seed_world,
                    nearest_cell=region.seed_cell if fallback_cost is not None else None,
                    cost=fallback_cost,
                    min_region_cost=fallback_cost,
                    mean_region_cost=fallback_cost,
                    passable_cell_count=0,
                    high_cost_cell_count=0,
                    objective_weight=config.terrain_objective_weight,
                    high_cost_exposure_proxy_cost=None,
                    high_cost_exposure_objective_weight=0.0,
                )
            )
            continue

        weights = [1.0 / max(cost, cost_floor) for _, _, cost in samples]
        weight_sum = float(sum(weights))
        anchor_point = WorldPoint(
            sum(point.x * weight for (_, point, _), weight in zip(samples, weights)) / weight_sum,
            sum(point.y * weight for (_, point, _), weight in zip(samples, weights)) / weight_sum,
        )
        nearest_cell, _, anchor_cost = min(
            samples,
            key=lambda item: (item[1].x - anchor_point.x) ** 2 + (item[1].y - anchor_point.y) ** 2,
        )
        costs = [cost for _, _, cost in samples]
        min_region_cost = min(costs)
        mean_region_cost = float(sum(costs) / len(costs))
        denominator = max(min_region_cost, cost_floor)
        cost_pressure = min(max(mean_region_cost / denominator, 1.0), 5.0)
        high_cost_exposure_values = [max(cost - high_cost_threshold, 0.0) for cost in costs]
        high_cost_proxy_cost = float(sum(high_cost_exposure_values) / len(high_cost_exposure_values))
        high_cost_pressure = min(max(high_cost_proxy_cost, 0.0), 5.0)
        high_cost_objective_weight = (
            float(config.high_cost_exposure_weight * high_cost_pressure)
            if config.high_cost_exposure_weight > 0.0 and high_cost_proxy_cost > 0.0
            else 0.0
        )
        anchors.append(
            _TerrainObjectiveAnchor(
                region_id=region.region_id,
                point=anchor_point,
                nearest_cell=nearest_cell,
                cost=float(anchor_cost),
                min_region_cost=float(min_region_cost),
                mean_region_cost=float(mean_region_cost),
                passable_cell_count=len(samples),
                high_cost_cell_count=sum(1 for cost in costs if cost > high_cost_threshold),
                objective_weight=float(config.terrain_objective_weight * cost_pressure),
                high_cost_exposure_proxy_cost=high_cost_proxy_cost,
                high_cost_exposure_objective_weight=high_cost_objective_weight,
            )
        )
    return tuple(anchors)


def _terrain_objective_summary(
    anchors: tuple[_TerrainObjectiveAnchor, ...],
    *,
    config: GcsControlPointSolverConfig,
) -> dict[str, Any]:
    costs = [anchor.cost for anchor in anchors if anchor.cost is not None]
    summary = {
        "terrain_objective_source": CONTROL_POINT_TERRAIN_OBJECTIVE_SOURCE,
        "terrain_objective_boundary": CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY,
        "terrain_objective_weight": float(config.terrain_objective_weight),
        "terrain_objective_anchor_count": len(anchors),
        "terrain_objective_passable_cell_count": sum(anchor.passable_cell_count for anchor in anchors),
        "terrain_objective_high_cost_cell_count": sum(anchor.high_cost_cell_count for anchor in anchors),
        "terrain_objective_high_cost_anchor_count": sum(
            1 for anchor in anchors if anchor.cost is not None and anchor.cost > 3.0
        ),
        "control_point_terrain_cost": float(sum(costs)) if costs else None,
        "control_point_terrain_cost_mean": float(sum(costs) / len(costs)) if costs else None,
        "terrain_objective_anchors": [_terrain_anchor_to_dict(anchor) for anchor in anchors],
    }
    if config.high_cost_exposure_weight > 0.0:
        proxy_costs = [
            anchor.high_cost_exposure_proxy_cost
            for anchor in anchors
            if anchor.high_cost_exposure_proxy_cost is not None
        ]
        summary.update(
            {
                "high_cost_exposure_objective_weight": float(config.high_cost_exposure_weight),
                "high_cost_exposure_proxy_cost": float(sum(proxy_costs)) if proxy_costs else None,
                "high_cost_exposure_proxy_cost_mean": (
                    float(sum(proxy_costs) / len(proxy_costs)) if proxy_costs else None
                ),
                "high_cost_exposure_proxy_source": CONTROL_POINT_HIGH_COST_EXPOSURE_OBJECTIVE_SOURCE,
                "high_cost_exposure_proxy_boundary": CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY,
                "high_cost_exposure_proxy_anchor_count": len(anchors),
                "high_cost_exposure_proxy_high_cost_anchor_count": sum(
                    1
                    for anchor in anchors
                    if anchor.high_cost_exposure_proxy_cost is not None
                    and anchor.high_cost_exposure_proxy_cost > 0.0
                ),
            }
        )
    return summary


def _terrain_anchor_to_dict(anchor: _TerrainObjectiveAnchor) -> dict[str, Any]:
    payload = {
        "region_id": anchor.region_id,
        "anchor_world": anchor.point.to_list(),
        "anchor_cell": None if anchor.nearest_cell is None else anchor.nearest_cell.to_list(),
        "anchor_cost": anchor.cost,
        "min_region_cost": anchor.min_region_cost,
        "mean_region_cost": anchor.mean_region_cost,
        "passable_cell_count": anchor.passable_cell_count,
        "high_cost_cell_count": anchor.high_cost_cell_count,
        "objective_weight": anchor.objective_weight,
    }
    if anchor.high_cost_exposure_objective_weight > 0.0:
        payload.update(
            {
                "high_cost_exposure_proxy_cost": anchor.high_cost_exposure_proxy_cost,
                "high_cost_exposure_objective_weight": anchor.high_cost_exposure_objective_weight,
            }
        )
    return payload


def _control_point_cost_summary(
    sampled_cost_summary: dict[str, Any],
    *,
    terrain_objective_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = dict(sampled_cost_summary)
    summary["sampled_terrain_cost"] = sampled_cost_summary.get("terrain_path_cost")
    summary.update(terrain_objective_summary)
    return summary


def _high_cost_exposure_constraint_fields(
    terrain_objective_summary: dict[str, Any],
    *,
    config: GcsControlPointSolverConfig,
) -> dict[str, Any]:
    if config.high_cost_exposure_weight <= 0.0:
        return {}
    return {
        "high_cost_exposure_objective_weight": terrain_objective_summary[
            "high_cost_exposure_objective_weight"
        ],
        "high_cost_exposure_proxy_source": terrain_objective_summary[
            "high_cost_exposure_proxy_source"
        ],
        "high_cost_exposure_proxy_anchor_count": terrain_objective_summary[
            "high_cost_exposure_proxy_anchor_count"
        ],
        "high_cost_exposure_proxy_high_cost_anchor_count": terrain_objective_summary[
            "high_cost_exposure_proxy_high_cost_anchor_count"
        ],
        "high_cost_exposure_proxy_boundary": CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY,
    }


def _not_evaluated_high_cost_exposure_fields(*, config: GcsControlPointSolverConfig) -> dict[str, Any]:
    if config.high_cost_exposure_weight <= 0.0:
        return {}
    return {
        "high_cost_exposure_objective_weight": float(config.high_cost_exposure_weight),
        "high_cost_exposure_proxy_source": "not_evaluated",
        "high_cost_exposure_proxy_anchor_count": 0,
        "high_cost_exposure_proxy_high_cost_anchor_count": 0,
        "high_cost_exposure_proxy_boundary": CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY,
    }


def _not_evaluated_cost_summary(reason: str, *, config: GcsControlPointSolverConfig) -> dict[str, Any]:
    summary = empty_gcs_cost_summary()
    summary.update(
        {
            "sampled_terrain_cost": None,
            "terrain_objective_source": "not_evaluated",
            "terrain_objective_boundary": CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY,
            "terrain_objective_weight": None,
            "terrain_objective_anchor_count": 0,
            "terrain_objective_passable_cell_count": 0,
            "terrain_objective_high_cost_cell_count": 0,
            "terrain_objective_high_cost_anchor_count": 0,
            "control_point_terrain_cost": None,
            "control_point_terrain_cost_mean": None,
            "terrain_objective_anchors": [],
            "terrain_objective_not_evaluated_reason": reason,
        }
    )
    if config.high_cost_exposure_weight > 0.0:
        summary.update(
            {
                "high_cost_exposure_objective_weight": float(config.high_cost_exposure_weight),
                "high_cost_exposure_proxy_cost": None,
                "high_cost_exposure_proxy_cost_mean": None,
                "high_cost_exposure_proxy_source": "not_evaluated",
                "high_cost_exposure_proxy_boundary": CONTROL_POINT_TERRAIN_OBJECTIVE_BOUNDARY,
                "high_cost_exposure_proxy_anchor_count": 0,
                "high_cost_exposure_proxy_high_cost_anchor_count": 0,
                "high_cost_exposure_proxy_not_evaluated_reason": reason,
            }
        )
    return summary
