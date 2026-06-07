import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from path_planner.adapters import load_plan_input
from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.diagnostics import render_diagnostics
from path_planner.platform import load_planner_platform_profile
from path_planner.postprocess import run_postprocess
from path_planner.search import AStarPlanner


def test_render_diagnostics_writes_png_and_html(tmp_path):
    spec = GridSpec(width=3, height=3, resolution=1.0)
    grid = CostGrid(spec=spec, cost=np.ones((3, 3)), passable_mask=np.ones((3, 3), dtype=bool))
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(2, 2)))
    postprocess = run_postprocess(grid, result, corridor_radius_cells=1, max_curvature=1.0)

    png_path = tmp_path / "diagnostics.png"
    html_path = tmp_path / "diagnostics.html"
    render_diagnostics(grid, result, postprocess=postprocess, png_path=png_path, html_path=html_path)

    assert png_path.exists()
    assert png_path.stat().st_size > 0
    html = html_path.read_text(encoding="utf-8")
    assert "Cost + Path" in html
    assert "Blocked Cells" in html
    assert "Safety Corridor" in html
    assert "Cost values use the grayscale colorbar labeled Cost" in html
    assert "Cost + Path legend is placed outside the plot area" in html
    assert "blue outlines mark the Safety Corridor" in html
    assert "amber diagonal hatching marks vehicle-inflated blocked cells" in html
    assert "black filled cells are blocked by passable_mask" in html
    assert "In Expanded Nodes, light cells are unexpanded passable space" in html
    assert "blue filled cells are A* expanded nodes" in html
    assert "red X markers are curvature or turning-radius violations" in html
    assert "green dot is start; red dot is goal" in html
    assert "Smoothed Path" in html
    assert "curvature_report" in html
    assert "Platform Constraints" in html
    assert "vehicle-inflated blocked cells" in html
    assert "turn_angle_deg" in html
    assert "turning_radius" in html
    assert "trajectory_kind" in html
    assert "geometric_path" in html


def test_cli_demo_writes_json_png_and_html(tmp_path):
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
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["diagnostics"]["search_mode"] == "platform_aware_astar"
    assert payload["diagnostics"]["passable_source"] == "inflated_passable_mask"
    assert payload["diagnostics"]["platform_key"] == "yutu2"
    assert payload["diagnostics"]["inflated_blocked_count"] > payload["diagnostics"]["original_blocked_count"]
    assert "postprocess" in payload
    assert "platform_profile" in payload["postprocess"]
    assert payload["postprocess"]["platform_profile"]["platform_key"] == "yutu2"
    assert "constraint_warnings" in payload["postprocess"]
    assert "corridor_report" in payload["postprocess"]
    assert payload["postprocess"]["raw_path"]["cells"] == payload["geometric_path"]["cells"]
    assert "curvature_report" in payload["postprocess"]
    assert "samples" in payload["postprocess"]["curvature_report"]
    assert "trackable_path" in payload["postprocess"]
    assert payload["postprocess"]["trackable_path"]["source_path"] == "smoothed_path"
    assert payload["postprocess"]["trackable_path"]["waypoints"]
    assert "speed_profile" in payload["postprocess"]["trackable_path"]
    assert "tracking_safety_report" in payload["postprocess"]
    assert "is_safe" in payload["postprocess"]["tracking_safety_report"]
    assert "planning_backend_report" not in payload
    assert "region_graph_report" in payload
    assert payload["region_graph_report"]["status"] == "ok"
    assert payload["region_graph_report"]["region_source"] == "grid_box"
    assert payload["region_graph_report"]["obstacle_source"] == "blocked_cell_box"
    assert payload["region_graph_report"]["motion_feasibility_status"] == "diagnostic_only"
    assert payload["region_graph_report"]["quality_metrics"]["graph_source"] == "grid_box"
    assert payload["region_graph_report"]["quality_metrics"]["start_goal_connected"] is True
    assert "iris_region_report" not in payload
    assert (output_dir / "diagnostics.png").exists()
    assert (output_dir / "diagnostics.html").exists()
    html = (output_dir / "diagnostics.html").read_text(encoding="utf-8")
    assert "Search Constraints" in html
    assert "platform_aware_astar" in html
    assert "inflated_passable_mask" in html
    assert "Trackable Path" in html
    assert "Tracking Safety Summary" in html
    assert "Rover Footprint Scale" in html
    assert "rover body length/width and footprint radius are drawn from platform_profile" in html
    assert "IRIS / Region Graph Summary" in html
    assert "2D workspace safe-region diagnostic" in html
    assert "reachable" in completed.stdout


def test_cli_region_graph_guided_backend_writes_additive_planning_report(tmp_path):
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
            "--planning-backend",
            "region_graph_guided",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    backend_report = payload["planning_backend_report"]

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert backend_report["requested_backend"] == "region_graph_guided"
    assert backend_report["status"] in {"selected", "fallback"}
    assert backend_report["selected_backend"] in {"astar", "region_graph_guided", "sampled_region_path"}
    assert "segment_count" in backend_report
    assert "comparison" in backend_report
    assert "region_graph_candidate" in backend_report
    assert "sampled_region_path_report" in backend_report
    assert backend_report["region_graph_candidate"]["region_graph_status"] == "ok"
    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["planning_backend"] == backend_report["selected_backend"]
    assert stdout_payload["planning_backend_fallback_reason"] == backend_report["fallback_reason"]


def test_cli_channel_aware_astar_backend_writes_additive_planning_report(tmp_path):
    input_json = tmp_path / "channel_request.json"
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    input_json.write_text(
        json.dumps(
            {
                "schema_version": "path-planner-request/v1",
                "grid": {"width": 5, "height": 5, "resolution": 1.0},
                "start": [0, 2],
                "goal": [4, 2],
                "cost": [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 9, 9, 9, 1],
                    [1, 1, 1, 1, 1],
                ],
                "passable_mask": [[True, True, True, True, True] for _ in range(5)],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--planning-backend",
            "channel_aware_astar",
            "--channel-aware-neighborhood-mean-weight",
            "3.0",
            "--channel-aware-neighborhood-max-weight",
            "1.0",
            "--channel-aware-high-cost-exposure-weight",
            "2.0",
            "--channel-aware-blocked-nearby-weight",
            "0.0",
            "--channel-aware-clearance-weight",
            "0.0",
            "--channel-aware-smoothness-weight",
            "0.0",
            "--channel-aware-high-cost-threshold",
            "4.0",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    backend_report = payload["planning_backend_report"]

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert backend_report["requested_backend"] == "channel_aware_astar"
    assert backend_report["status"] == "selected"
    assert backend_report["selected_backend"] == "channel_aware_astar"
    assert backend_report["comparison"]["path_changed"] is True
    assert backend_report["comparison"]["channel_cost_delta"] < 0.0
    assert backend_report["channel_candidate"]["cost_terms"]["high_cost_exposure_proxy"] > 0.0
    assert backend_report["execution_alignment"]["postprocess_seed"] == "selected_backend_result"
    assert backend_report["execution_alignment"]["default_route_replacement_verified"] is False
    assert backend_report["execution_alignment"]["verification_scope"] == "opt_in_audit_only"
    assert any(cell[1] == 1 for cell in payload["geometric_path"]["cells"])
    assert payload["postprocess"]["raw_path"]["cells"] == payload["geometric_path"]["cells"]
    assert any(cell[1] == 1 for cell in payload["postprocess"]["raw_path"]["cells"])
    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["planning_backend"] == "channel_aware_astar"
    assert stdout_payload["planning_backend_fallback_reason"] is None


def test_cli_demo_with_tracking_simulation_writes_report(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    subprocess.run(
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
            "--simulate-tracking",
            "--lookahead-m",
            "0.75",
            "--time-step-s",
            "0.2",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    report = payload["tracking_simulation_report"]

    assert "simulated_path" in report
    assert report["simulated_path"]
    assert "metrics" in report
    assert "safety_report" in report
    assert "max_cross_track_error_m" in report["metrics"]
    assert "min_clearance_m" in report["metrics"]
    assert "high_cost_exposure" in report["metrics"]
    assert report["config"]["max_sim_time_s"] == 600.0
    assert report["simulated_path"][-1]["target_waypoint_index"] == len(payload["postprocess"]["trackable_path"]["waypoints"]) - 1
    assert report["metrics"]["simulated_length_m"] >= report["metrics"]["path_length_m"] * 0.95
    html = (output_dir / "diagnostics.html").read_text(encoding="utf-8")
    assert "Tracking Simulation Summary" in html
    assert "Simulated Tracking Path" in html


def test_cli_demo_with_trajectory_optimization_writes_report(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    subprocess.run(
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
            "--simulate-tracking",
            "--optimize-trajectory",
            "--resample-spacing-m",
            "0.4",
            "--max-optimization-iter",
            "12",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    report = payload["trajectory_optimization_report"]

    assert "optimized_path" in report
    assert report["optimized_path"]
    assert "resampled_optimized_path" in report
    assert len(report["resampled_optimized_path"]) >= len(report["optimized_path"])
    assert "metrics" in report
    assert "solver_status" in report
    assert "fallback_status" in report
    assert "warnings" in report
    assert "optimized_tracking_simulation_report" in payload
    assert "baseline_vs_optimized" in report["metrics"]
    assert "tracking_error_proxy" in report["metrics"]
    assert "waypoint_spacing_max_m" in report["metrics"]
    assert report["metrics"]["waypoint_spacing_max_m"] <= 0.400001
    assert report["metrics"]["is_within_corridor"] is True
    assert report["metrics"]["optimized_high_cost_exposure"] <= report["metrics"]["reference_high_cost_exposure"] + 1e-3
    assert report["metrics"]["baseline_vs_optimized"]["high_cost_exposure"]["delta"] <= 1e-9
    assert len(report["resampled_corridor_boxes"]) == len(report["resampled_optimized_path"])
    for point, box in zip(report["resampled_optimized_path"], report["resampled_corridor_boxes"]):
        assert box["min"][0] <= point[0] <= box["max"][0]
        assert box["min"][1] <= point[1] <= box["max"][1]
    assert "heading_change_max_deg" in report["metrics"]["baseline_vs_optimized"]
    html = (output_dir / "diagnostics.html").read_text(encoding="utf-8")
    assert "Trajectory Optimization Summary" in html
    assert "Execution-Aware Optimization Summary" in html
    assert "Optimized Path" in html
    assert "Resampled Optimized Waypoints" in html
    assert "baseline_vs_optimized" in html


def test_corridor_demo_map_is_complex_and_keeps_corridor_feasible():
    grid, request = load_plan_input("examples/demo_map_corridor.json")
    result = AStarPlanner().plan(grid, request)
    postprocess = run_postprocess(
        grid,
        result,
        corridor_radius_cells=1,
        max_curvature=1.0,
        platform_profile=load_planner_platform_profile(platform="yutu2"),
    )

    assert grid.spec.frame_id == "demo_map_corridor_complex"
    assert grid.spec.width >= 18
    assert grid.spec.height >= 11
    assert len({cell.y for cell in result.path_cells}) >= 3
    assert len({cell.y for cell in postprocess.smoothed_path.cells}) >= 3
    assert postprocess.corridor.status == "ok"
    assert len(postprocess.corridor.sections) >= 18
    assert postprocess.corridor.original_blocked_count >= 20
    assert postprocess.corridor.inflated_blocked_count > postprocess.corridor.original_blocked_count
    assert postprocess.fallback_status.used_raw_path is False
