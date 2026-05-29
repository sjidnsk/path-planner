import numpy as np

from path_planner.core import GridSpec
from path_planner.costmap import CostmapWeights, PlatformLimits, SemanticLayers, build_cost_grid
from path_planner.costmap.builder import platform_limits_from_profile
from path_planner.platform import PlannerPlatformProfile


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

    grid = build_cost_grid(
        spec,
        layers,
        PlatformLimits(max_slope_deg=30.0, obstacle_threshold=0.5),
        CostmapWeights(),
    )

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


def test_platform_limits_are_derived_from_planner_platform_profile():
    profile = PlannerPlatformProfile(
        platform_key="test",
        platform_name="Test Platform",
        config_path=None,
        safety_margin_m=0.0,
        body_length_m=1.0,
        body_width_m=0.5,
        footprint_radius_m=0.6,
        max_slope_deg=12.0,
        max_obstacle_height_m=0.2,
        ground_clearance_m=0.3,
        raw_min_turning_radius_m=0.0,
        effective_min_turning_radius_m=None,
        energy_model=None,
        parameter_sources={},
        constraint_sources={},
        constraint_warnings=(),
    )
    spec = GridSpec(width=3, height=1, resolution=1.0)
    layers = SemanticLayers(
        slope=np.array([[0.0, 13.0, 0.0]]),
        obstacle=np.array([[0.0, 0.0, 0.25]]),
    )

    limits = platform_limits_from_profile(profile)
    grid = build_cost_grid(spec, layers, limits)

    assert limits.max_slope_deg == 12.0
    assert limits.obstacle_threshold == 0.2
    assert grid.passable_mask.tolist() == [[True, False, False]]


def test_semantic_cost_grid_requires_platform_limits():
    spec = GridSpec(width=1, height=1, resolution=1.0)
    layers = SemanticLayers(slope=np.array([[0.0]]))

    try:
        build_cost_grid(spec, layers)
    except ValueError as exc:
        assert "platform limits are required" in str(exc)
    else:
        raise AssertionError("semantic layer costmap should require platform limits")
