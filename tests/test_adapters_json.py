import json

import numpy as np

from path_planner.adapters import DevPlatformAdapter, load_plan_input, route_result_to_json_dict
from path_planner.core import Cell, FailureReason, GridSpec, PlanDiagnostics, PlanResult


def test_dev_platform_adapter_wraps_cost_and_mask_with_metadata():
    adapter = DevPlatformAdapter()

    grid = adapter.from_arrays(
        cost=np.array([[1.0, 2.0], [3.0, 4.0]]),
        passable_mask=np.array([[True, False], [True, True]]),
        resolution=0.5,
        origin=(1.0, 2.0),
        frame_id="moon",
        layers={"slope": np.array([[0.0, 5.0], [10.0, 20.0]])},
    )

    assert grid.spec == GridSpec(width=2, height=2, resolution=0.5, origin=(1.0, 2.0), frame_id="moon")
    assert grid.metadata["adapter"] == "dev-platform-constraints"
    assert grid.metadata["layers"] == ["slope"]


def test_load_plan_input_reads_internal_json_contract(tmp_path):
    payload = {
        "schema_version": "path-planner-request/v1",
        "grid": {"width": 2, "height": 2, "resolution": 1.0, "origin": [0.0, 0.0], "frame_id": "map"},
        "cost": [[1.0, 1.0], [1.0, 1.0]],
        "passable_mask": [[True, True], [True, True]],
        "start": [0, 0],
        "goal": [1, 1],
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    grid, request = load_plan_input(path)

    assert grid.spec.width == 2
    assert request.start == Cell(0, 0)
    assert request.goal == Cell(1, 1)


def test_route_result_to_json_dict_preserves_failure_reason():
    spec = GridSpec(width=2, height=2, resolution=1.0)
    result = PlanResult(
        success=False,
        path_cells=(),
        path_world=(),
        total_cost=float("inf"),
        expanded_count=0,
        failure_reason=FailureReason.UNREACHABLE,
        diagnostics=PlanDiagnostics(),
    )

    payload = route_result_to_json_dict(result, spec)

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["reachable"] is False
    assert payload["failure_reason"] == "unreachable"
