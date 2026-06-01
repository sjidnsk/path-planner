import pytest
import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec
from path_planner.drake_backend import build_workspace_iris_region_report
from path_planner.postprocess import build_corridor
from path_planner.regions import build_region_graph_report


@pytest.mark.drake
def test_optional_pydrake_planning_apis_are_importable_when_available():
    pytest.importorskip("pydrake")

    from pydrake import planning, solvers, trajectories
    from pydrake.geometry import optimization as geometry_optimization

    for module, names in (
        (
            geometry_optimization,
            (
                "HPolyhedron",
                "Point",
                "Iris",
                "IrisOptions",
                "GraphOfConvexSets",
            ),
        ),
        (
            planning,
            (
                "GcsTrajectoryOptimization",
            ),
        ),
        (
            solvers,
            (
                "MathematicalProgram",
                "Solve",
            ),
        ),
        (
            trajectories,
            (
                "PiecewisePolynomial",
                "BsplineTrajectory",
            ),
        ),
    ):
        for name in names:
            assert hasattr(module, name), f"missing pydrake API: {module.__name__}.{name}"


@pytest.mark.drake
def test_workspace_iris_backend_generates_valid_regions_when_pydrake_is_available():
    pytest.importorskip("pydrake")

    mask = np.ones((4, 5), dtype=bool)
    grid = CostGrid(
        spec=GridSpec(width=5, height=4, resolution=1.0),
        cost=np.ones(mask.shape),
        passable_mask=mask,
    )
    corridor = build_corridor(grid, (Cell(1, 1), Cell(2, 1), Cell(3, 1)), radius_cells=1)

    report = build_workspace_iris_region_report(grid, corridor)
    payload = report.to_dict()

    assert payload["backend"] == "workspace_iris"
    assert payload["status"] == "ok"
    assert payload["fallback_used"] is False
    assert payload["validation_status"] == "valid"
    assert payload["region_count"] == len(corridor.sections)
    assert payload["regions"][0]["source"] == "iris"
    assert payload["regions"][0]["hpolyhedron"]["A"]
    assert payload["regions"][0]["hpolyhedron"]["b"]

    graph_report = build_region_graph_report(grid, corridor, iris_region_report=report)
    graph_payload = graph_report.to_dict()

    assert graph_payload["region_source"] == "iris"
    assert graph_payload["fallback_used"] is False
    assert graph_payload["quality_metrics"]["graph_source"] == "iris"
    assert graph_payload["quality_metrics"]["start_goal_connected"] is True
    assert graph_payload["graph"]["edges"]
