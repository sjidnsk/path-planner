from __future__ import annotations

import argparse
import json
from pathlib import Path

from path_planner.adapters import load_plan_input, route_result_to_json_dict
from path_planner.diagnostics import render_diagnostics
from path_planner.search import AStarPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 path planner demo")
    parser.add_argument("--input", required=True, help="Path to path-planner-request/v1 JSON")
    parser.add_argument("--output-json", required=True, help="Path to write route JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostics.png and diagnostics.html")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grid, request = load_plan_input(args.input)
    result = AStarPlanner().plan(grid, request)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = route_result_to_json_dict(result, grid.spec)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_dir = Path(args.output_dir)
    render_diagnostics(
        grid,
        result,
        png_path=output_dir / "diagnostics.png",
        html_path=output_dir / "diagnostics.html",
    )

    print(json.dumps({"reachable": result.success, "failure_reason": payload["failure_reason"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
