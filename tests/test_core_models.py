import math

import numpy as np
import pytest

from path_planner.core import (
    Cell,
    CostGrid,
    FailureReason,
    GridSpec,
    NeighborPolicy,
    PlanDiagnostics,
    PlanRequest,
    PlanResult,
    WorldPoint,
)


def test_grid_spec_converts_between_cell_and_world():
    spec = GridSpec(width=4, height=3, resolution=0.5, origin=(10.0, -2.0), frame_id="moon")

    assert spec.in_bounds(Cell(2, 1))
    assert not spec.in_bounds(Cell(4, 1))
    assert spec.cell_to_world(Cell(2, 1)) == WorldPoint(11.0, -1.5)
    assert spec.world_to_cell(WorldPoint(11.2, -1.2)) == Cell(2, 1)


def test_cost_grid_rejects_invalid_passable_costs():
    spec = GridSpec(width=2, height=2, resolution=1.0)
    cost = np.array([[1.0, math.inf], [1.0, 1.0]])
    mask = np.array([[True, True], [True, False]])

    with pytest.raises(ValueError, match="passable cells must have finite nonnegative cost"):
        CostGrid(spec=spec, cost=cost, passable_mask=mask)


def test_plan_result_route_dict_marks_geometric_path():
    spec = GridSpec(width=3, height=3, resolution=2.0)
    result = PlanResult(
        success=True,
        path_cells=(Cell(0, 0), Cell(1, 1)),
        path_world=(WorldPoint(0.0, 0.0), WorldPoint(2.0, 2.0)),
        total_cost=2.5,
        expanded_count=4,
        failure_reason=None,
        diagnostics=PlanDiagnostics(runtime_ms=1.0, max_frontier_size=2),
    )

    route = result.to_route_dict(spec)

    assert route["schema_version"] == "path-planner-route/v1"
    assert route["trajectory_kind"] == "geometric_path"
    assert route["reachable"] is True
    assert route["path_cost"] == 2.5
    assert route["geometric_path"]["cells"] == [[0, 0], [1, 1]]


def test_plan_request_defaults_to_8_neighbor_with_corner_cut_protection():
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 2))

    assert request.neighbor_policy is NeighborPolicy.EIGHT
    assert request.prevent_corner_cutting is True
    assert request.max_iterations == 100_000
    assert FailureReason.UNREACHABLE.value == "unreachable"
