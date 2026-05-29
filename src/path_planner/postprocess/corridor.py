from __future__ import annotations

from path_planner.core import Cell, CostGrid
from path_planner.platform import PlannerPlatformProfile

from .footprint import build_footprint_safe_mask
from .models import CorridorResult, CorridorSection


def build_corridor(
    grid: CostGrid,
    path_cells: tuple[Cell, ...],
    *,
    radius_cells: int = 1,
    platform_profile: PlannerPlatformProfile | None = None,
) -> CorridorResult:
    if radius_cells < 0:
        raise ValueError("radius_cells must be nonnegative")
    footprint = build_footprint_safe_mask(grid, platform_profile)
    if not path_cells:
        return CorridorResult(
            status="failed",
            radius_cells=radius_cells,
            sections=(),
            failure_reason="empty_path",
            original_blocked_count=footprint.original_blocked_count,
            inflated_blocked_count=footprint.inflated_blocked_count,
            footprint_radius_m=footprint.footprint_radius_m,
        )

    sections: list[CorridorSection] = []
    for center in path_cells:
        if not grid.is_passable(center):
            return CorridorResult(
                status="failed",
                radius_cells=radius_cells,
                sections=(),
                failure_reason="path_cell_blocked",
                original_blocked_count=footprint.original_blocked_count,
                inflated_blocked_count=footprint.inflated_blocked_count,
                footprint_radius_m=footprint.footprint_radius_m,
            )
        if not bool(footprint.safe_mask[center.y, center.x]):
            return CorridorResult(
                status="failed",
                radius_cells=radius_cells,
                sections=(),
                failure_reason="path_cell_violates_platform_footprint",
                original_blocked_count=footprint.original_blocked_count,
                inflated_blocked_count=footprint.inflated_blocked_count,
                footprint_radius_m=footprint.footprint_radius_m,
            )
        cells = _safe_neighborhood(grid, center, radius_cells, footprint.safe_mask)
        sections.append(CorridorSection(center=center, cells=cells))

    return CorridorResult(
        status="ok",
        radius_cells=radius_cells,
        sections=tuple(sections),
        failure_reason=None,
        original_blocked_count=footprint.original_blocked_count,
        inflated_blocked_count=footprint.inflated_blocked_count,
        footprint_radius_m=footprint.footprint_radius_m,
    )


def _safe_neighborhood(grid: CostGrid, center: Cell, radius_cells: int, safe_mask) -> tuple[Cell, ...]:
    cells: list[Cell] = []
    for y in range(center.y - radius_cells, center.y + radius_cells + 1):
        for x in range(center.x - radius_cells, center.x + radius_cells + 1):
            cell = Cell(x, y)
            if grid.spec.in_bounds(cell) and bool(safe_mask[y, x]):
                cells.append(cell)
    return tuple(sorted(cells))
