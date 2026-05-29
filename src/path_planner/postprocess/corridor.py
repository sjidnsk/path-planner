from __future__ import annotations

from path_planner.core import Cell, CostGrid

from .models import CorridorResult, CorridorSection


def build_corridor(grid: CostGrid, path_cells: tuple[Cell, ...], *, radius_cells: int = 1) -> CorridorResult:
    if radius_cells < 0:
        raise ValueError("radius_cells must be nonnegative")
    if not path_cells:
        return CorridorResult(status="failed", radius_cells=radius_cells, sections=(), failure_reason="empty_path")

    sections: list[CorridorSection] = []
    for center in path_cells:
        if not grid.is_passable(center):
            return CorridorResult(
                status="failed",
                radius_cells=radius_cells,
                sections=(),
                failure_reason="path_cell_blocked",
            )
        cells = _passable_neighborhood(grid, center, radius_cells)
        sections.append(CorridorSection(center=center, cells=cells))

    return CorridorResult(status="ok", radius_cells=radius_cells, sections=tuple(sections), failure_reason=None)


def _passable_neighborhood(grid: CostGrid, center: Cell, radius_cells: int) -> tuple[Cell, ...]:
    cells: list[Cell] = []
    for y in range(center.y - radius_cells, center.y + radius_cells + 1):
        for x in range(center.x - radius_cells, center.x + radius_cells + 1):
            cell = Cell(x, y)
            if grid.is_passable(cell):
                cells.append(cell)
    return tuple(sorted(cells))
