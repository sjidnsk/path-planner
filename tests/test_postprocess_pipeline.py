import numpy as np

from path_planner.adapters import route_result_to_json_dict
from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.postprocess import run_postprocess
from path_planner.search import AStarPlanner


def make_grid(shape=(1, 4)):
    spec = GridSpec(width=shape[1], height=shape[0], resolution=1.0)
    return CostGrid(spec=spec, cost=np.ones(shape), passable_mask=np.ones(shape, dtype=bool))


def test_run_postprocess_builds_phase2_payload_from_successful_plan():
    grid = make_grid()
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(3, 0)))

    postprocess = run_postprocess(grid, result, corridor_radius_cells=1, max_curvature=1.0)
    payload = postprocess.to_dict()

    assert payload["raw_path"]["cells"] == [[0, 0], [1, 0], [2, 0], [3, 0]]
    assert payload["corridor"]["status"] == "ok"
    assert payload["smoothed_path"]["status"] == "shortcut"
    assert payload["smoothed_path"]["cells"] == [[0, 0], [3, 0]]
    assert payload["curvature_report"]["is_feasible"] is True
    assert payload["fallback_status"] == {"used_raw_path": False, "reason": None}


def test_route_result_to_json_dict_preserves_phase1_fields_and_adds_postprocess():
    grid = make_grid()
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(3, 0)))
    postprocess = run_postprocess(grid, result, corridor_radius_cells=1, max_curvature=1.0)

    payload = route_result_to_json_dict(result, grid.spec, postprocess=postprocess)

    assert payload["reachable"] is True
    assert payload["geometric_path"]["cells"][0] == [0, 0]
    assert payload["failure_reason"] is None
    assert payload["postprocess"]["raw_path"]["cells"] == payload["geometric_path"]["cells"]
    assert "curvature_report" in payload["postprocess"]
