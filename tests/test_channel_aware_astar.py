import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.search import AStarPlanner, ChannelAwareAStarConfig, ChannelAwareAStarPlanner


def make_grid(cost, passable=None):
    cost_array = np.asarray(cost, dtype=float)
    passable_mask = np.ones(cost_array.shape, dtype=bool) if passable is None else np.asarray(passable, dtype=bool)
    spec = GridSpec(width=cost_array.shape[1], height=cost_array.shape[0], resolution=1.0)
    return CostGrid(spec=spec, cost=cost_array, passable_mask=passable_mask)


def test_channel_aware_astar_selects_lower_risk_channel_when_opted_in():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 9.0, 9.0, 9.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 2), goal=Cell(4, 2))
    baseline = AStarPlanner().plan(grid, request)

    outcome = ChannelAwareAStarPlanner(
        ChannelAwareAStarConfig(
            neighborhood_radius_cells=1,
            neighborhood_mean_weight=3.0,
            neighborhood_max_weight=1.0,
            high_cost_exposure_weight=2.0,
            blocked_nearby_weight=0.0,
            clearance_weight=0.0,
            smoothness_weight=0.0,
            high_cost_threshold=4.0,
        )
    ).plan(grid, request, baseline_result=baseline)

    assert outcome.report.status == "selected"
    assert outcome.report.requested_backend == "channel_aware_astar"
    assert outcome.report.selected_backend == "channel_aware_astar"
    assert outcome.result.path_cells != baseline.path_cells
    assert any(cell.y == 1 for cell in outcome.result.path_cells)
    assert outcome.report.comparison["path_changed"] is True
    assert outcome.report.comparison["channel_cost_delta"] < 0.0
    assert outcome.report.channel_candidate["cost_terms"]["high_cost_exposure_proxy"] > 0.0


def test_channel_aware_astar_falls_back_with_machine_readable_reason_on_same_path():
    grid = make_grid(np.ones((3, 3)))
    request = PlanRequest(start=Cell(0, 1), goal=Cell(2, 1))
    baseline = AStarPlanner().plan(grid, request)

    outcome = ChannelAwareAStarPlanner().plan(grid, request, baseline_result=baseline)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    assert outcome.report.selected_backend == "astar"
    assert outcome.report.fallback_reason == "channel_candidate_same_as_baseline"
    payload = outcome.report.to_dict()
    assert payload["requested_backend"] == "channel_aware_astar"
    assert payload["config"]["neighborhood_radius_cells"] >= 1
    assert "center_cell_cost" in payload["channel_candidate"]["cost_terms"]
