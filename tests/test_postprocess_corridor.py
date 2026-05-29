import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec
from path_planner.postprocess import build_corridor


def make_grid(mask):
    passable = np.asarray(mask, dtype=bool)
    spec = GridSpec(width=passable.shape[1], height=passable.shape[0], resolution=1.0)
    return CostGrid(spec=spec, cost=np.ones(passable.shape), passable_mask=passable)


def test_build_corridor_creates_sections_around_path_cells():
    grid = make_grid(np.ones((5, 5), dtype=bool))

    corridor = build_corridor(grid, (Cell(1, 1), Cell(2, 1), Cell(3, 1)), radius_cells=1)

    assert corridor.status == "ok"
    assert corridor.failure_reason is None
    assert len(corridor.sections) == 3
    assert corridor.sections[0].center == Cell(1, 1)
    assert Cell(0, 0) in corridor.sections[0].cells
    assert Cell(2, 2) in corridor.sections[0].cells


def test_build_corridor_clips_to_map_edges():
    grid = make_grid(np.ones((2, 2), dtype=bool))

    corridor = build_corridor(grid, (Cell(0, 0),), radius_cells=1)

    assert corridor.status == "ok"
    assert set(corridor.sections[0].cells) == {Cell(0, 0), Cell(1, 0), Cell(0, 1), Cell(1, 1)}
    assert corridor.sections[0].to_dict()["bounds"] == {"min": [0, 0], "max": [1, 1]}


def test_build_corridor_fails_when_path_cell_is_blocked():
    grid = make_grid([[True, True], [True, False]])

    corridor = build_corridor(grid, (Cell(1, 1),), radius_cells=1)

    assert corridor.status == "failed"
    assert corridor.failure_reason == "path_cell_blocked"
    assert corridor.sections == ()
