from __future__ import annotations

from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint
from path_planner.platform import PlannerPlatformProfile
from path_planner.postprocess.footprint import build_footprint_safe_mask
from path_planner.postprocess.models import CorridorResult
from path_planner.regions import (
    ConvexRegion,
    build_blocked_cell_obstacles,
    build_grid_box_regions,
    build_merged_blocked_cell_obstacles,
)

from .models import IrisRegion, IrisRegionReport

WORKSPACE_IRIS_BACKEND = "workspace_iris"
SEED_SOURCE_CORRIDOR_CENTERS = "postprocess_corridor_centers"
DOMAIN_SOURCE_CORRIDOR_GRID_BOX = "postprocess_corridor_grid_box"
DOMAIN_SOURCE_CORRIDOR_SAFE_COMPONENT_BOX = "postprocess_corridor_safe_component_box"
OBSTACLE_SOURCE_BLOCKED_CELL_BOX = "blocked_cell_box"
OBSTACLE_SOURCE_MERGED_BLOCKED_RECTANGLE = "merged_blocked_rectangle"


def build_workspace_iris_region_report(
    grid: CostGrid,
    corridor: CorridorResult,
    *,
    platform_profile: PlannerPlatformProfile | None = None,
) -> IrisRegionReport:
    blocked_obstacles = build_blocked_cell_obstacles(grid, platform_profile=platform_profile)
    merged_obstacles = build_merged_blocked_cell_obstacles(grid, platform_profile=platform_profile)
    if merged_obstacles and len(merged_obstacles) < len(blocked_obstacles):
        obstacles = merged_obstacles
        obstacle_source = OBSTACLE_SOURCE_MERGED_BLOCKED_RECTANGLE
    else:
        obstacles = blocked_obstacles
        obstacle_source = OBSTACLE_SOURCE_BLOCKED_CELL_BOX
    fallback_regions = _fallback_regions(grid, corridor)
    if corridor.status != "ok":
        return _fallback_report(
            regions=fallback_regions,
            obstacle_count=len(obstacles),
            obstacle_source=obstacle_source,
            failure_status="invalid_region_input",
            failure_reason=corridor.failure_reason or "corridor_unavailable",
            validation_status="not_evaluated",
        )
    if not corridor.sections:
        return _fallback_report(
            regions=fallback_regions,
            obstacle_count=len(obstacles),
            obstacle_source=obstacle_source,
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
            obstacle_source=obstacle_source,
            failure_status="backend_unavailable",
            failure_reason=f"backend_unavailable: {exc}",
            validation_status="not_evaluated",
        )

    footprint = build_footprint_safe_mask(grid, platform_profile)
    domain_regions = _domain_regions(grid, corridor, fallback_regions, footprint.safe_mask)
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
        for fallback_region, domain_region in zip(fallback_regions, domain_regions, strict=True):
            iris_region = _build_iris_region(
                geometry_optimization,
                grid,
                fallback_region,
                domain_region,
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
            obstacle_source=obstacle_source,
            failure_status="solver_error",
            failure_reason=f"solver_error: {exc}",
            validation_status="not_evaluated",
        )

    if not regions:
        return _fallback_report(
            regions=fallback_regions,
            obstacle_count=len(obstacles),
            obstacle_source=obstacle_source,
            failure_status="infeasible",
            failure_reason="infeasible: no iris regions generated",
            validation_status="not_evaluated",
        )
    if fallback_used:
        return IrisRegionReport(
            backend=WORKSPACE_IRIS_BACKEND,
            status="fallback",
            seed_source=SEED_SOURCE_CORRIDOR_CENTERS,
            domain_source=DOMAIN_SOURCE_CORRIDOR_SAFE_COMPONENT_BOX,
            obstacle_source=obstacle_source,
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
        domain_source=DOMAIN_SOURCE_CORRIDOR_SAFE_COMPONENT_BOX,
        obstacle_source=obstacle_source,
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
    obstacle_source: str,
    failure_status: str,
    failure_reason: str,
    validation_status: str,
) -> IrisRegionReport:
    return IrisRegionReport(
        backend=WORKSPACE_IRIS_BACKEND,
        status="fallback",
        seed_source=SEED_SOURCE_CORRIDOR_CENTERS,
        domain_source=DOMAIN_SOURCE_CORRIDOR_GRID_BOX,
        obstacle_source=obstacle_source,
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


def _domain_regions(
    grid: CostGrid,
    corridor: CorridorResult,
    fallback_regions: tuple[ConvexRegion, ...],
    safe_mask: np.ndarray,
) -> tuple[ConvexRegion, ...]:
    corridor_cells = {
        cell
        for section in corridor.sections
        for cell in section.cells
        if grid.spec.in_bounds(cell) and bool(safe_mask[cell.y, cell.x])
    }
    return tuple(_domain_region_for_seed(grid, region, corridor_cells) for region in fallback_regions)


def _domain_region_for_seed(
    grid: CostGrid,
    seed_region: ConvexRegion,
    corridor_cells: set[Cell],
) -> ConvexRegion:
    if seed_region.center_cell not in corridor_cells:
        return seed_region
    component = _connected_cells(seed_region.center_cell, corridor_cells)
    if not component:
        return seed_region
    min_cell = Cell(min(cell.x for cell in component), min(cell.y for cell in component))
    max_cell = Cell(max(cell.x for cell in component), max(cell.y for cell in component))
    min_world, max_world = _cell_world_bounds(grid, min_cell, max_cell)
    return ConvexRegion(
        region_id=seed_region.region_id,
        source="grid_box",
        center_cell=seed_region.center_cell,
        min_cell=min_cell,
        max_cell=max_cell,
        min_world=min_world,
        max_world=max_world,
        cell_count=len(component),
        validation_status="valid",
    )


def _connected_cells(seed: Cell, cells: set[Cell]) -> set[Cell]:
    seen = {seed}
    stack = [seed]
    while stack:
        current = stack.pop()
        for neighbor in (
            Cell(current.x + 1, current.y),
            Cell(current.x - 1, current.y),
            Cell(current.x, current.y + 1),
            Cell(current.x, current.y - 1),
        ):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _build_iris_region(
    geometry_optimization: Any,
    grid: CostGrid,
    fallback_region: ConvexRegion,
    domain_region: ConvexRegion,
    drake_obstacles: list[Any],
    safe_mask: np.ndarray,
) -> IrisRegion:
    sample = np.asarray(_cell_center_world(grid, fallback_region.center_cell).to_list(), dtype=float)
    if not bool(safe_mask[fallback_region.center_cell.y, fallback_region.center_cell.x]):
        return _region_from_grid_box(
            fallback_region,
            domain_region=domain_region,
            validation_status="not_evaluated",
            failure_status="invalid_region_input",
            fallback_reason="invalid_region_input: seed cell is not footprint-safe",
        )

    domain = geometry_optimization.HPolyhedron.MakeBox(
        np.asarray(domain_region.min_world.to_list(), dtype=float),
        np.asarray(domain_region.max_world.to_list(), dtype=float),
    )
    options = geometry_optimization.IrisOptions()
    options.require_sample_point_is_contained = True
    region = geometry_optimization.Iris(drake_obstacles, sample, domain, options)
    if bool(region.IsEmpty()):
        return _region_from_grid_box(
            fallback_region,
            domain_region=domain_region,
            validation_status="empty_region",
            failure_status="infeasible",
            fallback_reason="infeasible: iris returned an empty region",
        )
    validation = _validate_iris_region(region, grid, fallback_region, domain_region, safe_mask)
    if validation is not None:
        return _region_from_grid_box(
            fallback_region,
            domain_region=domain_region,
            validation_status=validation,
            failure_status="fallback_used",
            fallback_reason=f"fallback_used: iris region validation failed ({validation})",
        )
    return _region_from_hpolyhedron(region, grid, fallback_region, domain_region, safe_mask)


def _validate_iris_region(
    region: Any,
    grid: CostGrid,
    seed_region: ConvexRegion,
    domain_region: ConvexRegion,
    safe_mask: np.ndarray,
) -> str | None:
    seed_sample = np.asarray(_cell_center_world(grid, seed_region.center_cell).to_list(), dtype=float)
    if not bool(region.PointInSet(seed_sample, 1e-8)):
        return "seed_not_contained"
    for y in range(domain_region.min_cell.y, domain_region.max_cell.y + 1):
        for x in range(domain_region.min_cell.x, domain_region.max_cell.x + 1):
            sample = np.asarray(_cell_center_world(grid, Cell(x, y)).to_list(), dtype=float)
            if bool(region.PointInSet(sample, 1e-8)) and not bool(safe_mask[y, x]):
                return "invalid_intersects_unsafe_cell"
    return None


def _region_from_hpolyhedron(
    region: Any,
    grid: CostGrid,
    seed_region: ConvexRegion,
    domain_region: ConvexRegion,
    safe_mask: np.ndarray,
) -> IrisRegion:
    included_cells: set[Cell] = set()
    for y in range(domain_region.min_cell.y, domain_region.max_cell.y + 1):
        for x in range(domain_region.min_cell.x, domain_region.max_cell.x + 1):
            cell = Cell(x, y)
            sample = np.asarray(_cell_center_world(grid, cell).to_list(), dtype=float)
            if bool(region.PointInSet(sample, 1e-8)) and bool(safe_mask[y, x]):
                included_cells.add(cell)
    min_cell, max_cell = _largest_seeded_safe_rectangle(included_cells, seed_region.center_cell)
    min_world, max_world = _cell_world_bounds(grid, min_cell, max_cell)
    return IrisRegion(
        region_id=seed_region.region_id,
        source="iris",
        seed_cell=seed_region.center_cell,
        seed_world=_cell_center_world(grid, seed_region.center_cell),
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
    domain_region: ConvexRegion | None = None,
    validation_status: str,
    failure_status: str | None,
    fallback_reason: str | None,
) -> IrisRegion:
    domain = domain_region or region
    return IrisRegion(
        region_id=region.region_id,
        source="grid_box",
        seed_cell=region.center_cell,
        seed_world=_world_bounds_center(region.min_world, region.max_world),
        min_cell=region.min_cell,
        max_cell=region.max_cell,
        min_world=region.min_world,
        max_world=region.max_world,
        domain_min_cell=domain.min_cell,
        domain_max_cell=domain.max_cell,
        domain_min_world=domain.min_world,
        domain_max_world=domain.max_world,
        hpolyhedron_a=_box_hpolyhedron_a(),
        hpolyhedron_b=_box_hpolyhedron_b(region.min_world, region.max_world),
        validation_status=validation_status,
        failure_status=failure_status,
        fallback_reason=fallback_reason,
    )


def _largest_seeded_safe_rectangle(cells: set[Cell], seed: Cell) -> tuple[Cell, Cell]:
    if seed not in cells:
        return seed, seed
    xs = [cell.x for cell in cells]
    ys = [cell.y for cell in cells]
    best_min = seed
    best_max = seed
    best_area = 1
    best_extent = 0
    for min_y in range(min(ys), seed.y + 1):
        for max_y in range(seed.y, max(ys) + 1):
            for min_x in range(min(xs), seed.x + 1):
                for max_x in range(seed.x, max(xs) + 1):
                    area = (max_x - min_x + 1) * (max_y - min_y + 1)
                    if area < best_area:
                        continue
                    if not all(Cell(x, y) in cells for y in range(min_y, max_y + 1) for x in range(min_x, max_x + 1)):
                        continue
                    extent = (max_x - min_x) + (max_y - min_y)
                    if area > best_area or extent > best_extent:
                        best_min = Cell(min_x, min_y)
                        best_max = Cell(max_x, max_y)
                        best_area = area
                        best_extent = extent
    return best_min, best_max


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
