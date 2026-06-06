from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gcs_scenario_matrix import build_gcs_scenario_matrix_summary
from .gcs_trajectory import FORCE_PYDRAKE_UNAVAILABLE_ENV

GCS_CLI_SCENARIO_BATCH_SCHEMA_VERSION = "gcs_direction_cone_cli_scenario_batch/v1"


@dataclass(frozen=True)
class GcsCliScenario:
    case_id: str
    request: dict[str, Any]
    expected_outcome: str
    expected_decision_reason: str
    cli_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


def run_gcs_cli_scenario_batch(
    *,
    output_dir: str | Path,
    summary_json: str | Path | None = None,
    case_ids: tuple[str, ...] | list[str] | None = None,
    python_executable: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_json) if summary_json is not None else output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    platform_config_path = output_root / "batch-platform.json"
    platform_config_path.write_text(
        json.dumps(_batch_platform_config(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scenarios = _selected_scenarios(case_ids)
    matrix_cases: list[dict[str, Any]] = []
    case_metadata: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        case_dir = output_root / "cases" / scenario.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        request_path = case_dir / "request.json"
        route_path = case_dir / "route.json"
        stdout_path = case_dir / "stdout.json"
        stderr_path = case_dir / "stderr.txt"
        report_dir = case_dir / "diagnostics"
        request_path.write_text(
            json.dumps(scenario.request, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        env = _cli_environment(extra_env=extra_env)
        env.update(scenario.env)
        command = [
            python_executable or sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            str(request_path),
            "--output-json",
            str(route_path),
            "--output-dir",
            str(report_dir),
            "--platform",
            "batch",
            "--platform-config",
            str(platform_config_path),
            "--gcs-geometric-candidate",
            *scenario.cli_args,
        ]
        completed = subprocess.run(command, check=True, env=env, text=True, capture_output=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        route = json.loads(route_path.read_text(encoding="utf-8"))
        matrix_cases.append(
            {
                "case_id": scenario.case_id,
                "payload": route,
                "expected": {
                    "outcome": scenario.expected_outcome,
                    "decision_reason": scenario.expected_decision_reason,
                },
            }
        )
        case_metadata[scenario.case_id] = {
            "request_json_path": request_path.relative_to(output_root).as_posix(),
            "route_json_path": route_path.relative_to(output_root).as_posix(),
            "stdout_json_path": stdout_path.relative_to(output_root).as_posix(),
            "stderr_path": stderr_path.relative_to(output_root).as_posix(),
        }

    summary = build_gcs_scenario_matrix_summary(matrix_cases)
    summary["schema_version"] = GCS_CLI_SCENARIO_BATCH_SCHEMA_VERSION
    summary["cli_module"] = "path_planner.cli"
    for case in summary["cases"]:
        case.update(case_metadata[case["case_id"]])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GCS direction-cone CLI scenario batch evidence")
    parser.add_argument("--output-dir", required=True, help="Directory for per-case request/route/diagnostic output")
    parser.add_argument("--summary-json", default=None, help="Path to write the batch summary JSON")
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(sorted(_builtin_scenarios().keys())),
        help="Run one scenario case id; can be repeated. Defaults to all built-in cases.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_gcs_cli_scenario_batch(
        output_dir=args.output_dir,
        summary_json=args.summary_json,
        case_ids=tuple(args.case) if args.case else None,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _selected_scenarios(case_ids: tuple[str, ...] | list[str] | None) -> tuple[GcsCliScenario, ...]:
    scenarios = _builtin_scenarios()
    selected_ids = tuple(case_ids) if case_ids else tuple(scenarios)
    missing = [case_id for case_id in selected_ids if case_id not in scenarios]
    if missing:
        raise ValueError(f"unknown GCS CLI scenario case(s): {', '.join(missing)}")
    return tuple(scenarios[case_id] for case_id in selected_ids)


def _cli_environment(*, extra_env: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    src_dir = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(src_dir)
        if not existing_pythonpath
        else str(src_dir) + os.pathsep + existing_pythonpath
    )
    return env


def _builtin_scenarios() -> dict[str, GcsCliScenario]:
    return {
        "open_corridor_selected": GcsCliScenario(
            case_id="open_corridor_selected",
            request=_diagonal_open_request("open_corridor_selected", size=28),
            expected_outcome="selected",
            expected_decision_reason="gcs_candidate_quality_improved",
            cli_args=("--max-shortcut-cost", "0"),
        ),
        "cost_dominated_diagonal": GcsCliScenario(
            case_id="cost_dominated_diagonal",
            request=_diagonal_open_request("cost_dominated_diagonal", size=10),
            expected_outcome="blocked",
            expected_decision_reason="cost_dominated",
            cli_args=("--max-shortcut-cost", "0"),
        ),
        "duplicate_baseline": GcsCliScenario(
            case_id="duplicate_baseline",
            request=_request(
                "duplicate_baseline",
                cost=[
                    [1, 1, 1, 1],
                    [0, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                passable_mask=_full_mask(width=4, height=3),
                start=[0, 1],
                goal=[3, 1],
            ),
            expected_outcome="blocked",
            expected_decision_reason="path_duplicate_with_baseline",
            cli_args=("--max-shortcut-cost", "0"),
        ),
        "sampled_trajectory_collision": GcsCliScenario(
            case_id="sampled_trajectory_collision",
            request=_request(
                "sampled_trajectory_collision",
                cost=[
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                passable_mask=[
                    [True, True, True],
                    [True, False, True],
                    [True, True, True],
                ],
                start=[0, 1],
                goal=[2, 1],
            ),
            expected_outcome="blocked",
            expected_decision_reason="sampled_trajectory_collision",
        ),
        "motion_infeasible_turn": GcsCliScenario(
            case_id="motion_infeasible_turn",
            request=_request(
                "motion_infeasible_turn",
                cost=_filled_cost(width=8, height=4, value=1),
                passable_mask=_full_mask(width=8, height=4),
                start=[0, 3],
                goal=[7, 0],
            ),
            expected_outcome="blocked",
            expected_decision_reason="motion_infeasible",
            cli_args=(
                "--max-shortcut-cost",
                "0",
                "--gcs-motion-feasibility",
                "--max-heading-change-deg",
                "1",
            ),
        ),
        "degenerate_portal": GcsCliScenario(
            case_id="degenerate_portal",
            request=_request(
                "degenerate_portal",
                cost=_filled_cost(width=3, height=3, value=1),
                passable_mask=_full_mask(width=3, height=3),
                start=[0, 2],
                goal=[2, 0],
            ),
            expected_outcome="blocked",
            expected_decision_reason="degenerate_portal_width",
            cli_args=("--corridor-radius-cells", "0"),
        ),
        "pydrake_unavailable": GcsCliScenario(
            case_id="pydrake_unavailable",
            request=_request(
                "pydrake_unavailable",
                cost=[
                    [1, 1, 1, 1],
                    [0, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                passable_mask=_full_mask(width=4, height=3),
                start=[0, 1],
                goal=[3, 1],
            ),
            expected_outcome="blocked",
            expected_decision_reason="pydrake_unavailable",
            env={FORCE_PYDRAKE_UNAVAILABLE_ENV: "1"},
        ),
    }


def _diagonal_open_request(frame_id: str, *, size: int) -> dict[str, Any]:
    return _request(
        frame_id,
        cost=_filled_cost(width=size, height=size, value=1),
        passable_mask=_full_mask(width=size, height=size),
        start=[0, size - 1],
        goal=[size - 1, 0],
    )


def _request(
    frame_id: str,
    *,
    cost: list[list[float | int]],
    passable_mask: list[list[bool]],
    start: list[int],
    goal: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": "path-planner-request/v1",
        "grid": {
            "width": len(cost[0]),
            "height": len(cost),
            "resolution": 1.0,
            "origin": [0.0, 0.0],
            "frame_id": frame_id,
        },
        "cost": cost,
        "passable_mask": passable_mask,
        "start": start,
        "goal": goal,
    }


def _filled_cost(*, width: int, height: int, value: float | int) -> list[list[float | int]]:
    return [[value for _ in range(width)] for _ in range(height)]


def _full_mask(*, width: int, height: int) -> list[list[bool]]:
    return [[True for _ in range(width)] for _ in range(height)]


def _batch_platform_config() -> dict[str, Any]:
    return {
        "name": "Batch GCS Direction-Cone Test Platform",
        "parameters": {
            "max_slope_deg": _parameter(20.0, "deg"),
            "max_obstacle_height": _parameter(0.2, "m"),
            "ground_clearance": _parameter(0.18, "m"),
            "min_turning_radius": _parameter(0.0, "m"),
            "sensor_range": _parameter(5.0, "m"),
            "sensor_fov": _parameter(60.0, "deg"),
            "energy_model": {
                "value": {"base_cost": 1.0, "slope_cost": 1.0},
                "unit": "relative",
                "source_kind": "assumed",
                "note": "batch scenario default",
            },
        },
    }


def _parameter(value: float, unit: str) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "source_kind": "assumed",
        "note": "batch scenario default",
    }


if __name__ == "__main__":
    raise SystemExit(main())
