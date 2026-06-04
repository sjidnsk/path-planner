from __future__ import annotations

from typing import Any

from path_planner.core import Cell, CostGrid, PlanResult, WorldPoint
from path_planner.postprocess.models import CorridorResult, CorridorSection

from .models import ConvexRegionSequenceItem, ConvexRegionSequenceReport, IrisRegion, IrisRegionReport

FALLBACK_BOX_BACKEND = "fallback_box"
WORKSPACE_IRIS_BACKEND = "workspace_iris"


def build_convex_region_sequence_report(
    grid: CostGrid,
    result: PlanResult,
    corridor: CorridorResult,
    *,
    iris_region_report: IrisRegionReport | None = None,
) -> ConvexRegionSequenceReport:
    pydrake_available = _pydrake_available()
    if not result.success or not result.path_cells:
        return _empty_report(
            pydrake_available=pydrake_available,
            reason="path_unreachable",
        )
    if corridor.status != "ok":
        return _empty_report(
            pydrake_available=pydrake_available,
            reason=corridor.failure_reason or "corridor_unavailable",
        )

    backend = FALLBACK_BOX_BACKEND
    fallback_used = True
    if iris_region_report is not None and iris_region_report.status == "ok" and iris_region_report.regions:
        backend = WORKSPACE_IRIS_BACKEND
        fallback_used = bool(iris_region_report.fallback_used)
        regions = _sequence_from_iris_regions(grid, result.path_cells, iris_region_report.regions)
    else:
        regions = _fallback_box_sequence(grid, result.path_cells, corridor)
    if not regions and backend != FALLBACK_BOX_BACKEND:
        backend = FALLBACK_BOX_BACKEND
        fallback_used = True
        regions = _fallback_box_sequence(grid, result.path_cells, corridor)

    covered = {index for region in regions for index in region.covered_path_indices}
    expected = set(range(len(result.path_cells)))
    coverage_status = "covered" if covered == expected else "partial" if covered else "unavailable"
    start_contained = bool(regions and _cell_in_region(result.path_cells[0], regions[0]))
    goal_contained = bool(regions and _cell_in_region(result.path_cells[-1], regions[-1]))
    adjacent_overlap_count, portal_count = _adjacency_counts(regions)
    blocked_cell_violation_count = _blocked_cell_violation_count(grid, regions)
    gcs_ready, gcs_ready_reason = _gcs_ready_status(
        regions,
        coverage_status=coverage_status,
        start_contained=start_contained,
        goal_contained=goal_contained,
        adjacent_overlap_count=adjacent_overlap_count,
        portal_count=portal_count,
        blocked_cell_violation_count=blocked_cell_violation_count,
    )
    return ConvexRegionSequenceReport(
        backend=backend,
        fallback_used=fallback_used or backend == FALLBACK_BOX_BACKEND,
        coverage_status=coverage_status,
        start_contained=start_contained,
        goal_contained=goal_contained,
        adjacent_overlap_count=adjacent_overlap_count,
        portal_count=portal_count,
        blocked_cell_violation_count=blocked_cell_violation_count,
        gcs_ready=gcs_ready,
        gcs_ready_reason=gcs_ready_reason,
        regions=regions,
        pydrake_available=pydrake_available,
    )


def _empty_report(*, pydrake_available: bool, reason: str) -> ConvexRegionSequenceReport:
    return ConvexRegionSequenceReport(
        backend=FALLBACK_BOX_BACKEND,
        fallback_used=True,
        coverage_status="unavailable",
        start_contained=False,
        goal_contained=False,
        adjacent_overlap_count=0,
        portal_count=0,
        blocked_cell_violation_count=0,
        gcs_ready=False,
        gcs_ready_reason=reason,
        pydrake_available=pydrake_available,
    )


def _sequence_from_iris_regions(
    grid: CostGrid,
    path_cells: tuple[Cell, ...],
    iris_regions: tuple[IrisRegion, ...],
) -> tuple[ConvexRegionSequenceItem, ...]:
    sequence: list[ConvexRegionSequenceItem] = []
    for index, region in enumerate(iris_regions):
        source = "iris" if region.source == "iris" and region.failure_status is None else FALLBACK_BOX_BACKEND
        backend = WORKSPACE_IRIS_BACKEND if source == "iris" else FALLBACK_BOX_BACKEND
        sequence.append(
            ConvexRegionSequenceItem(
                region_id=index,
                backend=backend,
                source=source,
                seed_cell=region.seed_cell,
                seed_world=region.seed_world,
                min_cell=region.min_cell,
                max_cell=region.max_cell,
                min_world=region.min_world,
                max_world=region.max_world,
                hpolyhedron_a=region.hpolyhedron_a,
                hpolyhedron_b=region.hpolyhedron_b,
                covered_path_indices=_covered_path_indices(path_cells, region.min_cell, region.max_cell),
                validation_status=region.validation_status,
                fallback_reason=region.fallback_reason,
            )
        )
    return tuple(sequence)


def _fallback_box_sequence(
    grid: CostGrid,
    path_cells: tuple[Cell, ...],
    corridor: CorridorResult,
) -> tuple[ConvexRegionSequenceItem, ...]:
    sections_by_center = {section.center: section for section in corridor.sections}
    regions: list[ConvexRegionSequenceItem] = []
    for index, seed in enumerate(path_cells):
        section = sections_by_center.get(seed)
        candidates = _safe_section_cells(grid, section) if section is not None else {seed}
        min_cell, max_cell = _largest_seeded_safe_rectangle(candidates, seed)
        min_world, max_world = _cell_world_bounds(grid, min_cell, max_cell)
        regions.append(
            ConvexRegionSequenceItem(
                region_id=index,
                backend=FALLBACK_BOX_BACKEND,
                source=FALLBACK_BOX_BACKEND,
                seed_cell=seed,
                seed_world=_cell_center_world(grid, seed),
                min_cell=min_cell,
                max_cell=max_cell,
                min_world=min_world,
                max_world=max_world,
                hpolyhedron_a=_box_hpolyhedron_a(),
                hpolyhedron_b=_box_hpolyhedron_b(min_world, max_world),
                covered_path_indices=_covered_path_indices(path_cells, min_cell, max_cell),
                validation_status="valid",
                fallback_reason="fallback_box_not_drake_iris",
            )
        )
    return tuple(regions)


def _safe_section_cells(grid: CostGrid, section: CorridorSection) -> set[Cell]:
    return {
        cell
        for cell in section.cells
        if grid.spec.in_bounds(cell) and bool(grid.passable_mask[cell.y, cell.x])
    }


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
                    rectangle = {Cell(x, y) for y in range(min_y, max_y + 1) for x in range(min_x, max_x + 1)}
                    if not rectangle <= cells:
                        continue
                    area = len(rectangle)
                    extent = (max_x - min_x) + (max_y - min_y)
                    if area > best_area or (area == best_area and extent > best_extent):
                        best_min = Cell(min_x, min_y)
                        best_max = Cell(max_x, max_y)
                        best_area = area
                        best_extent = extent
    return best_min, best_max


def _covered_path_indices(path_cells: tuple[Cell, ...], min_cell: Cell, max_cell: Cell) -> tuple[int, ...]:
    return tuple(
        index
        for index, cell in enumerate(path_cells)
        if min_cell.x <= cell.x <= max_cell.x and min_cell.y <= cell.y <= max_cell.y
    )


def _cell_in_region(cell: Cell, region: ConvexRegionSequenceItem) -> bool:
    return region.min_cell.x <= cell.x <= region.max_cell.x and region.min_cell.y <= cell.y <= region.max_cell.y


def _adjacency_counts(regions: tuple[ConvexRegionSequenceItem, ...]) -> tuple[int, int]:
    overlap_count = 0
    portal_count = 0
    for first, second in zip(regions[:-1], regions[1:]):
        if _rectangles_overlap(first, second):
            overlap_count += 1
        elif _rectangles_touch(first, second):
            portal_count += 1
    return overlap_count, portal_count


def _rectangles_overlap(first: ConvexRegionSequenceItem, second: ConvexRegionSequenceItem) -> bool:
    return (
        max(first.min_cell.x, second.min_cell.x) <= min(first.max_cell.x, second.max_cell.x)
        and max(first.min_cell.y, second.min_cell.y) <= min(first.max_cell.y, second.max_cell.y)
    )


def _rectangles_touch(first: ConvexRegionSequenceItem, second: ConvexRegionSequenceItem) -> bool:
    return (
        first.min_cell.x <= second.max_cell.x + 1
        and second.min_cell.x <= first.max_cell.x + 1
        and first.min_cell.y <= second.max_cell.y + 1
        and second.min_cell.y <= first.max_cell.y + 1
    )


def _blocked_cell_violation_count(grid: CostGrid, regions: tuple[ConvexRegionSequenceItem, ...]) -> int:
    violations = 0
    for region in regions:
        for y in range(region.min_cell.y, region.max_cell.y + 1):
            for x in range(region.min_cell.x, region.max_cell.x + 1):
                if not grid.spec.in_bounds(Cell(x, y)) or not bool(grid.passable_mask[y, x]):
                    violations += 1
    return violations


def _gcs_ready_status(
    regions: tuple[ConvexRegionSequenceItem, ...],
    *,
    coverage_status: str,
    start_contained: bool,
    goal_contained: bool,
    adjacent_overlap_count: int,
    portal_count: int,
    blocked_cell_violation_count: int,
) -> tuple[bool, str]:
    if not regions:
        return False, "convex_region_sequence_empty"
    if coverage_status != "covered":
        return False, "path_coverage_incomplete"
    if not start_contained:
        return False, "start_not_contained"
    if not goal_contained:
        return False, "goal_not_contained"
    if blocked_cell_violation_count > 0:
        return False, "blocked_cell_violation"
    if adjacent_overlap_count + portal_count < len(regions) - 1:
        return False, "adjacent_region_gap"
    return True, "convex_region_sequence_ready"


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


def _box_hpolyhedron_a() -> tuple[tuple[float, ...], ...]:
    return ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))


def _box_hpolyhedron_b(min_world: WorldPoint, max_world: WorldPoint) -> tuple[float, ...]:
    return (max_world.x, -min_world.x, max_world.y, -min_world.y)


def _pydrake_available() -> bool:
    try:
        __import__("pydrake")
    except Exception:
        return False
    return True
