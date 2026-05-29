from __future__ import annotations

from path_planner.core import Cell, CostGrid
from path_planner.platform import PlannerPlatformProfile

from .footprint import build_footprint_safe_mask
from .models import SmoothedPathResult


def has_line_of_sight(
    grid: CostGrid,
    start: Cell,
    goal: Cell,
    *,
    max_cell_cost: float | None = None,
    platform_profile: PlannerPlatformProfile | None = None,
    platform_safe_mask=None,
) -> bool:
    if max_cell_cost is not None and max_cell_cost < 0.0:
        raise ValueError("max_cell_cost must be nonnegative")
    safe_mask = platform_safe_mask
    if safe_mask is None:
        safe_mask = build_footprint_safe_mask(grid, platform_profile).safe_mask
    for cell in _cells_on_line(start, goal):
        if not grid.spec.in_bounds(cell) or not bool(safe_mask[cell.y, cell.x]):
            return False
        if max_cell_cost is not None and grid.cost_at(cell) > max_cell_cost:
            return False
    return True


def smooth_path(
    grid: CostGrid,
    path_cells: tuple[Cell, ...],
    *,
    max_shortcut_cost: float | None = None,
    platform_profile: PlannerPlatformProfile | None = None,
) -> SmoothedPathResult:
    if max_shortcut_cost is not None and max_shortcut_cost < 0.0:
        raise ValueError("max_shortcut_cost must be nonnegative")
    footprint = build_footprint_safe_mask(grid, platform_profile)
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
    if any(not bool(footprint.safe_mask[cell.y, cell.x]) for cell in path_cells):
        return SmoothedPathResult(
            status="fallback",
            cells=path_cells,
            world=raw_world,
            fallback_reason="path_cell_violates_platform_footprint",
        )
    if len(path_cells) <= 2:
        return SmoothedPathResult(status="unchanged", cells=path_cells, world=raw_world, fallback_reason=None)

    smoothed: list[Cell] = [path_cells[0]]
    anchor_index = 0
    while anchor_index < len(path_cells) - 1:
        next_index = anchor_index + 1
        for candidate_index in range(len(path_cells) - 1, anchor_index, -1):
            if has_line_of_sight(
                grid,
                path_cells[anchor_index],
                path_cells[candidate_index],
                max_cell_cost=max_shortcut_cost,
                platform_safe_mask=footprint.safe_mask,
            ):
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
    cells: list[Cell] = []
    min_x = min(start.x, goal.x)
    max_x = max(start.x, goal.x)
    min_y = min(start.y, goal.y)
    max_y = max(start.y, goal.y)
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cell = Cell(x, y)
            if _segment_intersects_cell(start, goal, cell):
                cells.append(cell)
    return tuple(sorted(cells))


def _segment_intersects_cell(start: Cell, goal: Cell, cell: Cell) -> bool:
    x0, y0 = float(start.x), float(start.y)
    x1, y1 = float(goal.x), float(goal.y)
    dx = x1 - x0
    dy = y1 - y0
    left = cell.x - 0.5
    right = cell.x + 0.5
    bottom = cell.y - 0.5
    top = cell.y + 0.5

    t0 = 0.0
    t1 = 1.0
    for edge_delta, edge_distance in (
        (-dx, x0 - left),
        (dx, right - x0),
        (-dy, y0 - bottom),
        (dy, top - y0),
    ):
        if edge_delta == 0.0:
            if edge_distance < 0.0:
                return False
            continue
        ratio = edge_distance / edge_delta
        if edge_delta < 0.0:
            if ratio > t1:
                return False
            if ratio > t0:
                t0 = ratio
        else:
            if ratio < t0:
                return False
            if ratio < t1:
                t1 = ratio
    return True
