from __future__ import annotations

import argparse
import json
from pathlib import Path

from path_planner.adapters import load_plan_input, route_result_to_json_dict
from path_planner.diagnostics import render_diagnostics
from path_planner.platform import DEFAULT_PLATFORM_KEY, load_planner_platform_profile
from path_planner.postprocess import run_postprocess
from path_planner.search import AStarPlanner, build_planning_grid


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
    postprocess = run_postprocess(
        grid,
        result,
        corridor_radius_cells=args.corridor_radius_cells,
        max_curvature=args.max_curvature,
        max_shortcut_cost=args.max_shortcut_cost,
        platform_profile=platform_profile,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = route_result_to_json_dict(result, grid.spec, postprocess=postprocess)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_dir = Path(args.output_dir)
    render_diagnostics(
        grid,
        result,
        postprocess=postprocess,
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
                "constraint_warnings": len(platform_profile.constraint_warnings),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
