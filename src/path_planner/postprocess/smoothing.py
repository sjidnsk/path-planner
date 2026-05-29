from __future__ import annotations

from path_planner.core import Cell, CostGrid

from .models import SmoothedPathResult


def has_line_of_sight(grid: CostGrid, start: Cell, goal: Cell) -> bool:
    return all(grid.is_passable(cell) for cell in _cells_on_line(start, goal))


def smooth_path(grid: CostGrid, path_cells: tuple[Cell, ...]) -> SmoothedPathResult:
    raw_world = tuple(grid.spec.cell_to_world(cell) for cell in path_cells)
    if not path_cells:
        return SmoothedPathResult(status="fallback", cells=path_cells, world=raw_world, fallback_reason="empty_path")
    if any(not grid.is_passable(cell) for cell in path_cells):
        return SmoothedPathResult(
            status="fallback",
            cells=path_cells,
            world=raw_world,
            fallback_reason="path_cell_blocked",
        )
    if len(path_cells) <= 2:
        return SmoothedPathResult(status="unchanged", cells=path_cells, world=raw_world, fallback_reason=None)

    smoothed: list[Cell] = [path_cells[0]]
    anchor_index = 0
    while anchor_index < len(path_cells) - 1:
        next_index = anchor_index + 1
        for candidate_index in range(len(path_cells) - 1, anchor_index, -1):
            if has_line_of_sight(grid, path_cells[anchor_index], path_cells[candidate_index]):
                next_index = candidate_index
                break
        smoothed.append(path_cells[next_index])
        anchor_index = next_index

    cells = tuple(smoothed)
    status = "shortcut" if len(cells) < len(path_cells) else "unchanged"
    return SmoothedPathResult(
        status=status,
        cells=cells,
        world=tuple(grid.spec.cell_to_world(cell) for cell in cells),
        fallback_reason=None,
    )


def _cells_on_line(start: Cell, goal: Cell) -> tuple[Cell, ...]:
    x0, y0 = start.x, start.y
    x1, y1 = goal.x, goal.y
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells: list[Cell] = []

    while True:
        cells.append(Cell(x0, y0))
        if x0 == x1 and y0 == y1:
            break
        doubled_error = 2 * error
        if doubled_error >= dy:
            error += dy
            x0 += sx
        if doubled_error <= dx:
            error += dx
            y0 += sy
    return tuple(cells)
