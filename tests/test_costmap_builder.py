import numpy as np

from path_planner.core import GridSpec
from path_planner.costmap import CostmapWeights, PlatformLimits, SemanticLayers, build_cost_grid


def test_build_cost_grid_applies_hard_constraints_and_soft_costs():
    spec = GridSpec(width=3, height=2, resolution=1.0)
    layers = SemanticLayers(
        slope=np.array([[0.0, 10.0, 40.0], [5.0, 5.0, 5.0]]),
        roughness=np.array([[0.0, 0.5, 0.0], [0.2, 0.0, 0.0]]),
        illumination=np.array([[1.0, 0.5, 1.0], [0.0, 1.0, 1.0]]),
        confidence=np.array([[1.0, 0.5, 1.0], [0.5, 1.0, 1.0]]),
        obstacle=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        valid_mask=np.array([[True, True, True], [True, True, False]]),
    )

    grid = build_cost_grid(spec, layers, PlatformLimits(max_slope_deg=30.0), CostmapWeights())

    assert grid.passable_mask.tolist() == [[True, True, False], [True, False, False]]
    assert grid.cost[0, 0] == 1.0
    assert grid.cost[0, 1] > grid.cost[0, 0]
    assert grid.metadata["source"] == "semantic_layers"


def test_build_cost_grid_can_wrap_existing_cost_and_mask():
    spec = GridSpec(width=2, height=2, resolution=1.0)
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, False], [True, True]])

    grid = build_cost_grid(spec, SemanticLayers(cost=cost, passable_mask=mask))

    assert grid.cost.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert grid.passable_mask.tolist() == [[True, False], [True, True]]
    assert grid.metadata["source"] == "cost_passable_mask"
