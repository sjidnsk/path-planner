from importlib import import_module


def test_trajectory_optimization_solver_is_split_into_internal_modules():
    expected = {
        "path_planner.optimization.corridor_mapping": (
            "build_corridor_boxes",
            "build_corridor_boxes_for_points",
            "build_resampled_points",
            "lowest_cost_point_in_box",
            "project_points_to_corridor",
            "project_to_box",
        ),
        "path_planner.optimization.objective": (
            "cost_gradient",
            "high_cost_exposure",
            "objective",
        ),
        "path_planner.optimization.guards": (
            "apply_high_cost_guard",
            "apply_resampled_high_cost_guard",
        ),
        "path_planner.optimization.metrics": (
            "build_metrics",
            "build_warnings",
            "execution_metrics",
        ),
        "path_planner.optimization.rebuild": (
            "build_trackable_path_from_world",
        ),
    }

    for module_name, function_names in expected.items():
        module = import_module(module_name)
        for function_name in function_names:
            assert callable(getattr(module, function_name))
