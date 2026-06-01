import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from path_planner.adapters import route_result_to_json_dict
from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.drake_backend import build_workspace_iris_region_report
from path_planner.postprocess import build_corridor
from path_planner.search import AStarPlanner


def make_grid(mask, resolution=1.0):
    passable = np.asarray(mask, dtype=bool)
    spec = GridSpec(width=passable.shape[1], height=passable.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=np.ones(passable.shape), passable_mask=passable)


def test_workspace_iris_backend_unavailable_returns_fallback_report(monkeypatch):
    import path_planner.drake_backend.iris as iris_backend

    grid = make_grid(
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ]
    )
    corridor = build_corridor(grid, (Cell(0, 1), Cell(1, 1), Cell(2, 1)), radius_cells=1)

    def unavailable():
        raise ImportError("simulated missing pydrake")

    monkeypatch.setattr(iris_backend, "_load_geometry_optimization", unavailable)

    report = build_workspace_iris_region_report(grid, corridor)
    payload = report.to_dict()

    assert payload["backend"] == "workspace_iris"
    assert payload["status"] == "fallback"
    assert payload["failure_status"] == "backend_unavailable"
    assert "simulated missing pydrake" in payload["failure_reason"]
    assert payload["fallback_used"] is True
    assert payload["seed_source"] == "postprocess_corridor_centers"
    assert payload["domain_source"] == "postprocess_corridor_grid_box"
    assert payload["obstacle_source"] == "blocked_cell_box"
    assert payload["region_count"] == len(corridor.sections)
    assert payload["regions"][0]["source"] == "grid_box"
    assert payload["regions"][0]["hpolyhedron"]["A"]
    assert payload["regions"][0]["hpolyhedron"]["b"]


def test_iris_region_report_is_optional_route_json_field():
    grid = make_grid(np.ones((3, 4), dtype=bool))
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    report = build_workspace_iris_region_report(grid, corridor)

    without_report = route_result_to_json_dict(plan, grid.spec)
    with_report = route_result_to_json_dict(plan, grid.spec, iris_region_report=report)

    assert "iris_region_report" not in without_report
    assert with_report["trajectory_kind"] == "geometric_path"
    assert with_report["reachable"] is True
    assert with_report["iris_region_report"]["backend"] == "workspace_iris"
    assert with_report["iris_region_report"]["region_count"] == len(corridor.sections)


def test_cli_drake_iris_regions_writes_optional_report_without_changing_route_semantics(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            "examples/demo_map_corridor.json",
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--drake-iris-regions",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    report = payload["iris_region_report"]
    graph_report = payload["region_graph_report"]

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert report["backend"] == "workspace_iris"
    assert report["status"] in {"ok", "fallback"}
    assert report["region_count"] > 0
    assert report["seed_source"] == "postprocess_corridor_centers"
    assert report["domain_source"] == "postprocess_corridor_grid_box"
    assert report["obstacle_source"] == "blocked_cell_box"
    assert graph_report["quality_metrics"]["requested_region_source"] == "iris"
    assert graph_report["quality_metrics"]["graph_source"] in {"iris", "grid_box"}
    assert graph_report["quality_metrics"]["start_goal_connected"] is True
    if report["status"] == "fallback":
        assert graph_report["region_source"] == "grid_box"
        assert graph_report["fallback_used"] is True
        assert "iris_region_graph_fallback" in graph_report["failure_reason"]
    html = (output_dir / "diagnostics.html").read_text(encoding="utf-8")
    assert "IRIS / Region Graph Summary" in html
    assert "not a GCS trajectory" in html
    assert "not an Ackermann/skid-steer feasibility proof" in html
    assert "iris_region_status" in completed.stdout


def test_pydrake_imports_are_confined_to_optional_backend_and_drake_tests():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    patterns = ("from " + "pydrake", "import " + "pydrake")
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in patterns):
            relative = path.relative_to(root).as_posix()
            if not relative.startswith("src/path_planner/drake_backend/"):
                offenders.append(relative)
    assert offenders == []
