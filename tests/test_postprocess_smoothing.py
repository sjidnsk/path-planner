import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec
from path_planner.postprocess import has_line_of_sight, smooth_path


def make_grid(mask, resolution=1.0):
    passable = np.asarray(mask, dtype=bool)
    spec = GridSpec(width=passable.shape[1], height=passable.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=np.ones(passable.shape), passable_mask=passable)


def test_smooth_path_shortcuts_clear_path():
    grid = make_grid(np.ones((1, 4), dtype=bool), resolution=0.5)

    result = smooth_path(grid, (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0)))

    assert result.status == "shortcut"
    assert result.cells == (Cell(0, 0), Cell(3, 0))
    assert result.world[-1].to_list() == [1.5, 0.0]
    assert result.fallback_reason is None


def test_smooth_path_does_not_shortcut_through_obstacle():
    grid = make_grid(
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ]
    )

    result = smooth_path(grid, (Cell(0, 0), Cell(0, 1), Cell(0, 2), Cell(1, 2), Cell(2, 2)))

    assert has_line_of_sight(grid, Cell(0, 0), Cell(2, 2)) is False
    assert result.cells != (Cell(0, 0), Cell(2, 2))
    assert result.cells[0] == Cell(0, 0)
    assert result.cells[-1] == Cell(2, 2)


def test_smooth_path_falls_back_when_path_contains_blocked_cell():
    grid = make_grid([[True, True], [True, False]])
    raw_path = (Cell(0, 0), Cell(1, 1))

    result = smooth_path(grid, raw_path)

    assert result.status == "fallback"
    assert result.cells == raw_path
    assert result.fallback_reason == "path_cell_blocked"
