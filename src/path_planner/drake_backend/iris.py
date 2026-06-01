from __future__ import annotations

from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint
from path_planner.platform import PlannerPlatformProfile
from path_planner.postprocess.footprint import build_footprint_safe_mask
from path_planner.postprocess.models import CorridorResult
from path_planner.regions import ConvexRegion, build_blocked_cell_obstacles, build_grid_box_regions

from .models import IrisRegion, IrisRegionReport

WORKSPACE_IRIS_BACKEND = "workspace_iris"
SEED_SOURCE_CORRIDOR_CENTERS = "postprocess_corridor_centers"
DOMAIN_SOURCE_CORRIDOR_GRID_BOX = "postprocess_corridor_grid_box"
OBSTACLE_SOURCE_BLOCKED_CELL_BOX = "blocked_cell_box"


def build_workspace_iris_region_report(
    grid: CostGrid,
    corridor: CorridorResult,
    *,
    platform_profile: PlannerPlatformProfile | None = None,
) -> IrisRegionReport:
    obstacles = build_blocked_cell_obstacles(grid, platform_profile=platform_profile)
    fallback_regions = _fallback_regions(grid, corridor)
    if corridor.status != "ok":
        return _fallback_report(
            regions=fallback_regions,
            obstacle_count=len(obstacles),
            failure_status="invalid_region_input",
            failure_reason=corridor.failure_reason or "corridor_unavailable",
            validation_status="not_evaluated",
        )
    if not corridor.sections:
        return _fallback_report(
            regions=fallback_regions,
            obstacle_count=len(obstacles),
            failure_status="invalid_region_input",
            failure_reason="empty_corridor",
            validation_status="not_evaluated",
        )

    try:
        geometry_optimization = _load_geometry_optimization()
    except ImportError as exc:
        return _fallback_report(
            regions=fallback_regions,
            obstacle_count=len(obstacles),
            failure_status="backend_unavailable",
            failure_reason=f"backend_unavailable: {exc}",
            validation_status="not_evaluated",
        )

    footprint = build_footprint_safe_mask(grid, platform_profile)
    try:
        drake_obstacles = [
            geometry_optimization.HPolyhedron.MakeBox(
                np.asarray(obstacle.min_world.to_list(), dtype=float),
                np.asarray(obstacle.max_world.to_list(), dtype=float),
            )
            for obstacle in obstacles
        ]
        regions: list[IrisRegion] = []
        fallback_used = False
        fallback_reasons: list[str] = []
        for fallback_region in fallback_regions:
            iris_region = _build_iris_region(
                geometry_optimization,
                grid,
                fallback_region,
                drake_obstacles,
                footprint.safe_mask,
            )
            if iris_region.failure_status is not None:
                fallback_used = True
                fallback_reasons.append(iris_region.fallback_reason or iris_region.failure_status)
            regions.append(iris_region)
    except Exception as exc:
        return _fallback_report(
            regions=fallback_regions,
            obstacle_count=len(obstacles),
            failure_status="solver_error",
            failure_reason=f"solver_error: {exc}",
            validation_status="not_evaluated",
        )

    if not regions:
        return _fallback_report(
            regions=fallback_regions,
            obstacle_count=len(obstacles),
            failure_status="infeasible",
            failure_reason="infeasible: no iris regions generated",
            validation_status="not_evaluated",
        )
    if fallback_used:
        return IrisRegionReport(
            backend=WORKSPACE_IRIS_BACKEND,
            status="fallback",
            seed_source=SEED_SOURCE_CORRIDOR_CENTERS,
            domain_source=DOMAIN_SOURCE_CORRIDOR_GRID_BOX,
            obstacle_source=OBSTACLE_SOURCE_BLOCKED_CELL_BOX,
            regions=tuple(regions),
            obstacle_count=len(obstacles),
            validation_status="fallback_used",
            failure_status="fallback_used",
            failure_reason="; ".join(fallback_reasons),
            fallback_used=True,
        )
    return IrisRegionReport(
        backend=WORKSPACE_IRIS_BACKEND,
        status="ok",
        seed_source=SEED_SOURCE_CORRIDOR_CENTERS,
        domain_source=DOMAIN_SOURCE_CORRIDOR_GRID_BOX,
        obstacle_source=OBSTACLE_SOURCE_BLOCKED_CELL_BOX,
        regions=tuple(regions),
        obstacle_count=len(obstacles),
        validation_status="valid",
        fallback_used=False,
    )


def _load_geometry_optimization() -> Any:
    from pydrake.geometry import optimization as geometry_optimization

    return geometry_optimization


def _fallback_report(
    *,
    regions: tuple[ConvexRegion, ...],
    obstacle_count: int,
    failure_status: str,
    failure_reason: str,
    validation_status: str,
) -> IrisRegionReport:
    return IrisRegionReport(
        backend=WORKSPACE_IRIS_BACKEND,
        status="fallback",
        seed_source=SEED_SOURCE_CORRIDOR_CENTERS,
        domain_source=DOMAIN_SOURCE_CORRIDOR_GRID_BOX,
        obstacle_source=OBSTACLE_SOURCE_BLOCKED_CELL_BOX,
        regions=tuple(
            _region_from_grid_box(
                region,
                validation_status=validation_status,
                failure_status=failure_status,
                fallback_reason=failure_reason,
            )
            for region in regions
        ),
        obstacle_count=obstacle_count,
        validation_status=validation_status,
        failure_status=failure_status,
        failure_reason=failure_reason,
        fallback_used=True,
    )


def _fallback_regions(grid: CostGrid, corridor: CorridorResult) -> tuple[ConvexRegion, ...]:
    if corridor.sections:
        return build_grid_box_regions(grid, corridor)
    return ()


def _build_iris_region(
    geometry_optimization: Any,
    grid: CostGrid,
    fallback_region: ConvexRegion,
    drake_obstacles: list[Any],
    safe_mask: np.ndarray,
) -> IrisRegion:
    sample = np.asarray(_cell_center_world(grid, fallback_region.center_cell).to_list(), dtype=float)
    if not bool(safe_mask[fallback_region.center_cell.y, fallback_region.center_cell.x]):
        return _region_from_grid_box(
            fallback_region,
            validation_status="not_evaluated",
            failure_status="invalid_region_input",
            fallback_reason="invalid_region_input: seed cell is not footprint-safe",
        )

    domain = geometry_optimization.HPolyhedron.MakeBox(
        np.asarray(fallback_region.min_world.to_list(), dtype=float),
        np.asarray(fallback_region.max_world.to_list(), dtype=float),
    )
    options = geometry_optimization.IrisOptions()
    options.require_sample_point_is_contained = True
    region = geometry_optimization.Iris(drake_obstacles, sample, domain, options)
    if bool(region.IsEmpty()):
        return _region_from_grid_box(
            fallback_region,
            validation_status="empty_region",
            failure_status="infeasible",
            fallback_reason="infeasible: iris returned an empty region",
        )
    validation = _validate_iris_region(region, grid, fallback_region, safe_mask)
    if validation is not None:
        return _region_from_grid_box(
            fallback_region,
            validation_status=validation,
            failure_status="fallback_used",
            fallback_reason=f"fallback_used: iris region validation failed ({validation})",
        )
    return _region_from_hpolyhedron(region, grid, fallback_region)


def _validate_iris_region(
    region: Any,
    grid: CostGrid,
    domain_region: ConvexRegion,
    safe_mask: np.ndarray,
) -> str | None:
    seed_sample = np.asarray(_cell_center_world(grid, domain_region.center_cell).to_list(), dtype=float)
    if not bool(region.PointInSet(seed_sample, 1e-8)):
        return "seed_not_contained"
    for y in range(domain_region.min_cell.y, domain_region.max_cell.y + 1):
        for x in range(domain_region.min_cell.x, domain_region.max_cell.x + 1):
            sample = np.asarray(_cell_center_world(grid, Cell(x, y)).to_list(), dtype=float)
            if bool(region.PointInSet(sample, 1e-8)) and not bool(safe_mask[y, x]):
                return "invalid_intersects_unsafe_cell"
    return None


def _region_from_hpolyhedron(region: Any, grid: CostGrid, domain_region: ConvexRegion) -> IrisRegion:
    included_cells: list[Cell] = []
    for y in range(domain_region.min_cell.y, domain_region.max_cell.y + 1):
        for x in range(domain_region.min_cell.x, domain_region.max_cell.x + 1):
            cell = Cell(x, y)
            sample = np.asarray(_cell_center_world(grid, cell).to_list(), dtype=float)
            if bool(region.PointInSet(sample, 1e-8)):
                included_cells.append(cell)
    if not included_cells:
        min_cell = domain_region.center_cell
        max_cell = domain_region.center_cell
    else:
        min_cell = Cell(min(cell.x for cell in included_cells), min(cell.y for cell in included_cells))
        max_cell = Cell(max(cell.x for cell in included_cells), max(cell.y for cell in included_cells))
    min_world, max_world = _cell_world_bounds(grid, min_cell, max_cell)
    return IrisRegion(
        region_id=domain_region.region_id,
        source="iris",
        seed_cell=domain_region.center_cell,
        seed_world=_cell_center_world(grid, domain_region.center_cell),
        min_cell=min_cell,
        max_cell=max_cell,
        min_world=min_world,
        max_world=max_world,
        domain_min_cell=domain_region.min_cell,
        domain_max_cell=domain_region.max_cell,
        domain_min_world=domain_region.min_world,
        domain_max_world=domain_region.max_world,
        hpolyhedron_a=_matrix_to_tuple(region.A()),
        hpolyhedron_b=_vector_to_tuple(region.b()),
        validation_status="valid",
    )


def _region_from_grid_box(
    region: ConvexRegion,
    *,
    validation_status: str,
    failure_status: str | None,
    fallback_reason: str | None,
) -> IrisRegion:
    return IrisRegion(
        region_id=region.region_id,
        source="grid_box",
        seed_cell=region.center_cell,
        seed_world=_world_bounds_center(region.min_world, region.max_world),
        min_cell=region.min_cell,
        max_cell=region.max_cell,
        min_world=region.min_world,
        max_world=region.max_world,
        domain_min_cell=region.min_cell,
        domain_max_cell=region.max_cell,
        domain_min_world=region.min_world,
        domain_max_world=region.max_world,
        hpolyhedron_a=_box_hpolyhedron_a(),
        hpolyhedron_b=_box_hpolyhedron_b(region.min_world, region.max_world),
        validation_status=validation_status,
        failure_status=failure_status,
        fallback_reason=fallback_reason,
    )


def _cell_center_world(grid: CostGrid, cell: Cell) -> WorldPoint:
    resolution = grid.spec.resolution
    return WorldPoint(
        grid.spec.origin[0] + (cell.x + 0.5) * resolution,
        grid.spec.origin[1] + (cell.y + 0.5) * resolution,
    )


def _cell_world_bounds(grid: CostGrid, min_cell: Cell, max_cell: Cell) -> tuple[WorldPoint, WorldPoint]:
    resolution = grid.spec.resolution
    min_world = grid.spec.cell_to_world(min_cell)
    max_world = WorldPoint(
        grid.spec.origin[0] + (max_cell.x + 1) * resolution,
        grid.spec.origin[1] + (max_cell.y + 1) * resolution,
    )
    return min_world, max_world


def _world_bounds_center(min_world: WorldPoint, max_world: WorldPoint) -> WorldPoint:
    return WorldPoint((min_world.x + max_world.x) * 0.5, (min_world.y + max_world.y) * 0.5)


def _box_hpolyhedron_a() -> tuple[tuple[float, ...], ...]:
    return ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))


def _box_hpolyhedron_b(min_world: WorldPoint, max_world: WorldPoint) -> tuple[float, ...]:
    return (max_world.x, -min_world.x, max_world.y, -min_world.y)


def _matrix_to_tuple(matrix: Any) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(matrix, dtype=float)
    return tuple(tuple(float(value) for value in row) for row in array)


def _vector_to_tuple(vector: Any) -> tuple[float, ...]:
    array = np.asarray(vector, dtype=float).reshape(-1)
    return tuple(float(value) for value in array)
