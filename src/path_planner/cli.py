from __future__ import annotations

import argparse
import json
from pathlib import Path

from path_planner.adapters import load_plan_input, route_result_to_json_dict
from path_planner.diagnostics import render_diagnostics
from path_planner.drake_backend import (
    build_convex_region_sequence_report,
    build_gcs_curvature_constrained_candidate_report,
    build_gcs_geometric_candidate_report,
    build_gcs_motion_feasibility_report,
    build_gcs_trajectory_report,
    build_workspace_iris_region_report,
)
from path_planner.optimization import TrajectoryOptimizationConfig, optimize_trajectory
from path_planner.platform import DEFAULT_PLATFORM_KEY, load_planner_platform_profile
from path_planner.postprocess import run_postprocess
from path_planner.regions import build_region_graph_report
from path_planner.search import ASTAR_BACKEND, REGION_GRAPH_GUIDED_BACKEND, AStarPlanner, RegionGraphGuidedPlanner
from path_planner.search import build_planning_grid
from path_planner.tracking import TrackingSimulationConfig, simulate_tracking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 path planner demo")
    parser.add_argument("--input", required=True, help="Path to path-planner-request/v1 JSON")
    parser.add_argument("--output-json", required=True, help="Path to write route JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostics.png and diagnostics.html")
    parser.add_argument("--corridor-radius-cells", type=int, default=1, help="Corridor radius in grid cells")
    parser.add_argument("--max-curvature", type=float, default=1.0, help="Maximum allowed discrete curvature")
    parser.add_argument(
        "--min-turning-radius",
        type=float,
        default=None,
        help="Override platform minimum turning radius in meters",
    )
    parser.add_argument("--platform", default=DEFAULT_PLATFORM_KEY, help="Platform key from dev-platform-constraints")
    parser.add_argument("--platform-config", default=None, help="Explicit platform config JSON path")
    parser.add_argument("--safety-margin-m", type=float, default=0.0, help="Extra footprint safety margin in meters")
    parser.add_argument(
        "--max-shortcut-cost",
        type=float,
        default=3.0,
        help="Maximum cell cost allowed inside a smoothing shortcut",
    )
    parser.add_argument(
        "--max-speed-mps",
        type=float,
        default=None,
        help="Maximum recommended trackable-path speed in m/s; defaults to platform speed_max when available",
    )
    parser.add_argument(
        "--min-speed-mps",
        type=float,
        default=0.01,
        help="Minimum recommended trackable-path speed in m/s",
    )
    parser.add_argument(
        "--tracking-error-bound-m",
        type=float,
        default=0.0,
        help="Lateral tracking error bound used for conservative safety-tube diagnostics",
    )
    parser.add_argument(
        "--simulate-tracking",
        action="store_true",
        help="Run a lightweight pure-pursuit tracking simulation for experiment baseline metrics",
    )
    parser.add_argument("--lookahead-m", type=float, default=0.75, help="Pure-pursuit lookahead distance in meters")
    parser.add_argument("--time-step-s", type=float, default=0.2, help="Tracking simulation time step in seconds")
    parser.add_argument("--max-sim-time-s", type=float, default=600.0, help="Maximum tracking simulation time in seconds")
    parser.add_argument(
        "--optimize-trajectory",
        action="store_true",
        help="Run fixed-corridor continuous trajectory optimization prototype",
    )
    parser.add_argument("--optimization-weight-smoothness", type=float, default=2.0)
    parser.add_argument("--optimization-weight-cost", type=float, default=2.0)
    parser.add_argument("--optimization-weight-reference", type=float, default=1.0)
    parser.add_argument("--optimization-weight-tracking", type=float, default=1.0)
    parser.add_argument("--optimization-weight-spacing", type=float, default=0.5)
    parser.add_argument("--optimization-weight-speed-smoothness", type=float, default=0.2)
    parser.add_argument("--resample-spacing-m", type=float, default=None)
    parser.add_argument("--max-optimization-iter", type=int, default=40)
    parser.add_argument(
        "--drake-iris-regions",
        action="store_true",
        help="Run optional 2D workspace Drake Iris region generation prototype",
    )
    parser.add_argument(
        "--gcs-trajectory-smoke",
        action="store_true",
        help="Run optional Drake GCS corridor trajectory smoke diagnostics",
    )
    parser.add_argument(
        "--gcs-geometric-candidate",
        action="store_true",
        help="Compare optional Drake GCS sampled trajectory as a geometric candidate",
    )
    parser.add_argument(
        "--gcs-motion-feasibility",
        action="store_true",
        help="Evaluate optional Drake GCS sampled trajectory against curvature/heading constraints",
    )
    parser.add_argument(
        "--gcs-curvature-constrained-candidate",
        action="store_true",
        help="Build an optional curvature-constrained GCS sampled candidate repair diagnostic",
    )
    parser.add_argument(
        "--max-heading-change-deg",
        type=float,
        default=120.0,
        help="Maximum heading change between sampled GCS trajectory segments for motion-feasibility diagnostics",
    )
    parser.add_argument(
        "--planning-backend",
        choices=(ASTAR_BACKEND, REGION_GRAPH_GUIDED_BACKEND),
        default=ASTAR_BACKEND,
        help="Planning backend to use; region_graph_guided is opt-in and falls back to A* when not better",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grid, request = load_plan_input(args.input)
    platform_profile = load_planner_platform_profile(
        platform=args.platform,
        config_path=args.platform_config,
        safety_margin_m=args.safety_margin_m,
        min_turning_radius_override_m=args.min_turning_radius,
    )
    planning_grid = build_planning_grid(grid, platform_profile=platform_profile)
    baseline_result = AStarPlanner().plan(planning_grid, request)
    result = baseline_result
    max_speed_mps = (
        args.max_speed_mps
        if args.max_speed_mps is not None
        else platform_profile.speed_max_mps
        if platform_profile.speed_max_mps is not None
        else 0.2
    )
    postprocess = run_postprocess(
        grid,
        baseline_result,
        corridor_radius_cells=args.corridor_radius_cells,
        max_curvature=args.max_curvature,
        max_shortcut_cost=args.max_shortcut_cost,
        platform_profile=platform_profile,
        max_speed_mps=max_speed_mps,
        min_speed_mps=args.min_speed_mps,
        tracking_error_bound_m=args.tracking_error_bound_m,
    )
    iris_region_report = None
    if args.drake_iris_regions:
        iris_region_report = build_workspace_iris_region_report(
            grid,
            postprocess.corridor,
            platform_profile=platform_profile,
        )
    region_graph_report = build_region_graph_report(
        grid,
        postprocess.corridor,
        platform_profile=platform_profile,
        iris_region_report=iris_region_report,
    )
    planning_backend_report = None
    if args.planning_backend == REGION_GRAPH_GUIDED_BACKEND:
        outcome = RegionGraphGuidedPlanner().plan(
            planning_grid,
            request,
            baseline_result=baseline_result,
            region_graph_report=region_graph_report,
        )
        planning_backend_report = outcome.report
        result = outcome.result
        if result is not baseline_result:
            postprocess = run_postprocess(
                grid,
                result,
                corridor_radius_cells=args.corridor_radius_cells,
                max_curvature=args.max_curvature,
                max_shortcut_cost=args.max_shortcut_cost,
                platform_profile=platform_profile,
                max_speed_mps=max_speed_mps,
                min_speed_mps=args.min_speed_mps,
                tracking_error_bound_m=args.tracking_error_bound_m,
            )
            iris_region_report = None
            if args.drake_iris_regions:
                iris_region_report = build_workspace_iris_region_report(
                    grid,
                    postprocess.corridor,
                    platform_profile=platform_profile,
                )
            region_graph_report = build_region_graph_report(
                grid,
                postprocess.corridor,
                platform_profile=platform_profile,
                iris_region_report=iris_region_report,
            )
    tracking_simulation = None
    if args.simulate_tracking and postprocess.trackable_path is not None:
        tracking_simulation = simulate_tracking(
            grid,
            postprocess.trackable_path,
            platform_profile=platform_profile,
            config=TrackingSimulationConfig(
                lookahead_m=args.lookahead_m,
                time_step_s=args.time_step_s,
                max_sim_time_s=args.max_sim_time_s,
            ),
        )
    trajectory_optimization = None
    optimized_tracking_simulation = None
    if args.optimize_trajectory and postprocess.trackable_path is not None:
        trajectory_optimization = optimize_trajectory(
            grid,
            postprocess.trackable_path,
            postprocess.corridor,
            platform_profile=platform_profile,
            config=TrajectoryOptimizationConfig(
                weight_smoothness=args.optimization_weight_smoothness,
                weight_cost=args.optimization_weight_cost,
                weight_reference=args.optimization_weight_reference,
                weight_tracking=args.optimization_weight_tracking,
                weight_spacing=args.optimization_weight_spacing,
                weight_speed_smoothness=args.optimization_weight_speed_smoothness,
                resample_spacing_m=args.resample_spacing_m,
                max_iterations=args.max_optimization_iter,
            ),
        )
        if args.simulate_tracking:
            optimized_tracking_simulation = simulate_tracking(
                grid,
                trajectory_optimization.resampled_trackable_path,
                platform_profile=platform_profile,
                config=TrackingSimulationConfig(
                    lookahead_m=args.lookahead_m,
                    time_step_s=args.time_step_s,
                    max_sim_time_s=args.max_sim_time_s,
                ),
            )

    convex_region_sequence_report = build_convex_region_sequence_report(
        grid,
        result,
        postprocess.corridor,
        iris_region_report=iris_region_report,
    )
    gcs_trajectory_report = None
    if (
        args.gcs_trajectory_smoke
        or args.gcs_geometric_candidate
        or args.gcs_motion_feasibility
        or args.gcs_curvature_constrained_candidate
    ):
        gcs_trajectory_report = build_gcs_trajectory_report(
            grid,
            convex_region_sequence_report,
        )
    gcs_candidate_report = None
    if args.gcs_geometric_candidate:
        gcs_candidate_report = build_gcs_geometric_candidate_report(
            grid,
            result,
            postprocess,
            gcs_trajectory_report,
        )
    gcs_motion_feasibility_report = None
    if args.gcs_motion_feasibility or args.gcs_curvature_constrained_candidate:
        gcs_motion_feasibility_report = build_gcs_motion_feasibility_report(
            gcs_trajectory_report,
            min_turning_radius_m=platform_profile.effective_min_turning_radius_m,
            max_heading_change_deg=args.max_heading_change_deg,
            max_curvature=args.max_curvature,
        )
    gcs_curvature_constrained_candidate_report = None
    if args.gcs_curvature_constrained_candidate:
        gcs_curvature_constrained_candidate_report = build_gcs_curvature_constrained_candidate_report(
            grid,
            result,
            convex_region_sequence_report,
            gcs_trajectory_report,
            gcs_motion_feasibility_report,
            min_turning_radius_m=platform_profile.effective_min_turning_radius_m,
            max_heading_change_deg=args.max_heading_change_deg,
            max_curvature=args.max_curvature,
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = route_result_to_json_dict(
        result,
        grid.spec,
        postprocess=postprocess,
        tracking_simulation=tracking_simulation,
        trajectory_optimization=trajectory_optimization,
        optimized_tracking_simulation=optimized_tracking_simulation,
        region_graph_report=region_graph_report,
        iris_region_report=iris_region_report,
        convex_region_sequence_report=convex_region_sequence_report,
        gcs_trajectory_report=gcs_trajectory_report,
        gcs_candidate_report=gcs_candidate_report,
        gcs_motion_feasibility_report=gcs_motion_feasibility_report,
        gcs_curvature_constrained_candidate_report=gcs_curvature_constrained_candidate_report,
        planning_backend_report=planning_backend_report,
    )
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_dir = Path(args.output_dir)
    render_diagnostics(
        grid,
        result,
        postprocess=postprocess,
        tracking_simulation=tracking_simulation,
        trajectory_optimization=trajectory_optimization,
        optimized_tracking_simulation=optimized_tracking_simulation,
        region_graph_report=region_graph_report,
        iris_region_report=iris_region_report,
        png_path=output_dir / "diagnostics.png",
        html_path=output_dir / "diagnostics.html",
    )

    print(
        json.dumps(
            {
                "reachable": result.success,
                "failure_reason": payload["failure_reason"],
                "platform": platform_profile.platform_key,
                "search_mode": result.diagnostics.search_mode,
                "trackable_waypoints": len(postprocess.trackable_path.waypoints) if postprocess.trackable_path else 0,
                "tracking_simulation": tracking_simulation is not None,
                "trajectory_optimization": trajectory_optimization is not None,
                "trajectory_optimization_status": (
                    trajectory_optimization.solver_status if trajectory_optimization is not None else None
                ),
                "region_graph_vertices": region_graph_report.vertex_count,
                "region_graph_edges": region_graph_report.edge_count,
                "region_graph_source": region_graph_report.region_source,
                "region_graph_fallback_used": region_graph_report.fallback_used,
                "region_graph_start_goal_connected": region_graph_report.quality_metrics.get(
                    "start_goal_connected"
                ),
                "iris_region_report": iris_region_report is not None,
                "iris_region_status": iris_region_report.status if iris_region_report is not None else None,
                "iris_region_count": iris_region_report.region_count if iris_region_report is not None else 0,
                "convex_region_backend": convex_region_sequence_report.backend,
                "convex_region_count": convex_region_sequence_report.region_count,
                "gcs_ready": convex_region_sequence_report.gcs_ready,
                "gcs_ready_reason": convex_region_sequence_report.gcs_ready_reason,
                "gcs_trajectory_smoke": gcs_trajectory_report is not None,
                "gcs_trajectory_attempted": (
                    gcs_trajectory_report.attempted if gcs_trajectory_report is not None else None
                ),
                "gcs_trajectory_success": (
                    gcs_trajectory_report.success if gcs_trajectory_report is not None else None
                ),
                "gcs_trajectory_reason": (
                    gcs_trajectory_report.reason if gcs_trajectory_report is not None else None
                ),
                "gcs_geometric_candidate": gcs_candidate_report is not None,
                "gcs_candidate_available": (
                    gcs_candidate_report.available if gcs_candidate_report is not None else None
                ),
                "gcs_candidate_selected": (
                    gcs_candidate_report.selected if gcs_candidate_report is not None else None
                ),
                "gcs_candidate_fallback_reason": (
                    gcs_candidate_report.fallback_reason if gcs_candidate_report is not None else None
                ),
                "gcs_motion_feasibility": gcs_motion_feasibility_report is not None,
                "gcs_motion_feasibility_evaluated": (
                    gcs_motion_feasibility_report.evaluated
                    if gcs_motion_feasibility_report is not None
                    else None
                ),
                "gcs_motion_feasibility_status": (
                    gcs_motion_feasibility_report.feasibility_status
                    if gcs_motion_feasibility_report is not None
                    else None
                ),
                "gcs_motion_feasibility_fallback_reason": (
                    gcs_motion_feasibility_report.fallback_reason
                    if gcs_motion_feasibility_report is not None
                    else None
                ),
                "gcs_curvature_constrained_candidate": (
                    gcs_curvature_constrained_candidate_report is not None
                ),
                "gcs_curvature_constrained_available": (
                    gcs_curvature_constrained_candidate_report.available
                    if gcs_curvature_constrained_candidate_report is not None
                    else None
                ),
                "gcs_curvature_constrained_selected": (
                    gcs_curvature_constrained_candidate_report.selected
                    if gcs_curvature_constrained_candidate_report is not None
                    else None
                ),
                "gcs_curvature_constrained_repair_success": (
                    gcs_curvature_constrained_candidate_report.repair_success
                    if gcs_curvature_constrained_candidate_report is not None
                    else None
                ),
                "gcs_curvature_constrained_fallback_reason": (
                    gcs_curvature_constrained_candidate_report.fallback_reason
                    if gcs_curvature_constrained_candidate_report is not None
                    else None
                ),
                "constraint_warnings": len(platform_profile.constraint_warnings),
                "planning_backend": (
                    planning_backend_report.selected_backend if planning_backend_report is not None else ASTAR_BACKEND
                ),
                "planning_backend_fallback_reason": (
                    planning_backend_report.fallback_reason if planning_backend_report is not None else None
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
