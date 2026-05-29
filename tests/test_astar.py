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


def test_astar_prevents_diagonal_corner_cutting():
    mask = np.array(
        [
            [True, False],
            [False, True],
        ]
    )
    grid = grid_from(np.ones((2, 2)), mask)
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(1, 1)))

    assert result.success is False
    assert result.failure_reason is FailureReason.UNREACHABLE


def test_astar_reports_blocked_start_and_goal():
    mask = np.array([[False, True], [True, False]])
    grid = grid_from(np.ones((2, 2)), mask)

    start_blocked = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(1, 0)))
    goal_blocked = AStarPlanner().plan(grid, PlanRequest(start=Cell(1, 0), goal=Cell(1, 1)))

    assert start_blocked.failure_reason is FailureReason.START_BLOCKED
    assert goal_blocked.failure_reason is FailureReason.GOAL_BLOCKED


def test_astar_reports_out_of_bounds():
    grid = grid_from(np.ones((2, 2)))

    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(-1, 0), goal=Cell(1, 1)))

    assert result.success is False
    assert result.failure_reason is FailureReason.START_OUT_OF_BOUNDS


def test_astar_reports_max_iterations():
    grid = grid_from(np.ones((5, 5)))
    result = AStarPlanner().plan(
        grid,
        PlanRequest(start=Cell(0, 0), goal=Cell(4, 4), max_iterations=1),
    )

    assert result.success is False
    assert result.failure_reason is FailureReason.MAX_ITERATIONS
    assert result.expanded_count == 1
