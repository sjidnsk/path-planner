from __future__ import annotations

import argparse
import json
from pathlib import Path

from path_planner.adapters import load_plan_input, route_result_to_json_dict
from path_planner.diagnostics import render_diagnostics
from path_planner.platform import DEFAULT_PLATFORM_KEY, load_planner_platform_profile
from path_planner.postprocess import run_postprocess
from path_planner.search import AStarPlanner, build_planning_grid
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
    result = AStarPlanner().plan(planning_grid, request)
    max_speed_mps = (
        args.max_speed_mps
        if args.max_speed_mps is not None
        else platform_profile.speed_max_mps
        if platform_profile.speed_max_mps is not None
        else 0.2
    )
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

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = route_result_to_json_dict(
        result,
        grid.spec,
        postprocess=postprocess,
        tracking_simulation=tracking_simulation,
    )
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_dir = Path(args.output_dir)
    render_diagnostics(
        grid,
        result,
        postprocess=postprocess,
        tracking_simulation=tracking_simulation,
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
                "constraint_warnings": len(platform_profile.constraint_warnings),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
