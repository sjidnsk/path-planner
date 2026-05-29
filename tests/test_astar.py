import numpy as np

from path_planner.core import Cell, CostGrid, FailureReason, GridSpec, PlanRequest
from path_planner.search import AStarPlanner


def grid_from(cost, mask=None):
    cost_array = np.asarray(cost, dtype=float)
    spec = GridSpec(width=cost_array.shape[1], height=cost_array.shape[0], resolution=1.0)
    passable = np.ones(cost_array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return CostGrid(spec=spec, cost=cost_array, passable_mask=passable)


def test_astar_finds_diagonal_path_on_empty_grid():
    grid = grid_from(np.ones((3, 3)))
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(2, 2)))

    assert result.success is True
    assert result.failure_reason is None
    assert result.path_cells == (Cell(0, 0), Cell(1, 1), Cell(2, 2))
    assert result.total_cost > 0.0
    assert result.expanded_count > 0


def test_astar_prefers_lower_weighted_route():
    cost = np.array(
        [
            [1.0, 20.0, 1.0],
            [1.0, 20.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    grid = grid_from(cost)
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(2, 0)))

    assert result.success is True
    assert Cell(1, 0) not in result.path_cells
    assert result.path_cells[-1] == Cell(2, 0)
