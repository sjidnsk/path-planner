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
GCS_MOTION_FEASIBILITY_CLI_BATCH_SCHEMA_VERSION = "gcs_motion_feasibility_cli_batch/v1"
GCS_CONTROL_POINT_TERRAIN_COST_CLI_BATCH_SCHEMA_VERSION = (
    "gcs_control_point_terrain_cost_cli_batch/v1"
)


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
    scenarios = _selected_scenarios(_builtin_scenarios(), case_ids)
    matrix_cases, case_metadata = _run_cli_route_cases(
        output_dir=output_dir,
        summary_json=summary_json,
        scenarios=scenarios,
        common_cli_args=("--gcs-geometric-candidate",),
        python_executable=python_executable,
        extra_env=extra_env,
    )
    summary = build_gcs_scenario_matrix_summary(matrix_cases)
    summary["schema_version"] = GCS_CLI_SCENARIO_BATCH_SCHEMA_VERSION
    summary["cli_module"] = "path_planner.cli"
    for case in summary["cases"]:
        case.update(case_metadata[case["case_id"]])
    _write_summary(summary_json, output_dir, summary)
    return summary


def run_gcs_motion_feasibility_cli_batch(
    *,
    output_dir: str | Path,
    summary_json: str | Path | None = None,
    case_ids: tuple[str, ...] | list[str] | None = None,
    python_executable: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    scenarios = _selected_scenarios(_motion_feasibility_scenarios(), case_ids)
    matrix_cases, case_metadata = _run_cli_route_cases(
        output_dir=output_dir,
        summary_json=summary_json,
        scenarios=scenarios,
        common_cli_args=("--gcs-geometric-candidate", "--gcs-motion-feasibility"),
        python_executable=python_executable,
        extra_env=extra_env,
    )
    summary = build_gcs_motion_feasibility_batch_summary(matrix_cases)
    summary["cli_module"] = "path_planner.cli"
    for case in summary["cases"]:
        case.update(case_metadata[case["case_id"]])
    _write_summary(summary_json, output_dir, summary)
    return summary


def run_gcs_control_point_terrain_cost_cli_batch(
    *,
    output_dir: str | Path,
    summary_json: str | Path | None = None,
    case_ids: tuple[str, ...] | list[str] | None = None,
    python_executable: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    scenarios = _selected_scenarios(_control_point_terrain_cost_scenarios(), case_ids)
    matrix_cases, case_metadata = _run_cli_route_cases(
        output_dir=output_dir,
        summary_json=summary_json,
        scenarios=scenarios,
        common_cli_args=("--gcs-control-point-candidate", "--gcs-motion-feasibility"),
        python_executable=python_executable,
        extra_env=extra_env,
    )
    summary = build_gcs_control_point_terrain_cost_batch_summary(matrix_cases)
    summary["cli_module"] = "path_planner.cli"
    for case in summary["cases"]:
        case.update(case_metadata[case["case_id"]])
    _write_summary(summary_json, output_dir, summary)
    return summary


def build_gcs_motion_feasibility_batch_summary(cases: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    rows = [_motion_feasibility_case_summary(case) for case in cases]
    return {
        "schema_version": GCS_MOTION_FEASIBILITY_CLI_BATCH_SCHEMA_VERSION,
        "case_count": len(rows),
        "feasible_count": sum(1 for row in rows if row["outcome"] == "feasible"),
        "infeasible_count": sum(1 for row in rows if row["outcome"] == "infeasible"),
        "diagnostic_only_count": sum(1 for row in rows if row["outcome"] == "diagnostic_only"),
        "candidate_selected_count": sum(1 for row in rows if row["candidate_selected"]),
        "candidate_blocked_count": sum(1 for row in rows if not row["candidate_selected"]),
        "decision_reason_counts": _counts(row["decision_reason"] for row in rows),
        "fallback_reason_counts": _counts(row["fallback_reason"] for row in rows),
        "expectation_failures": [
            {"case_id": row["case_id"], "mismatch_fields": row["mismatch_fields"]}
            for row in rows
            if row["mismatch_fields"]
        ],
        "cases": rows,
    }


def build_gcs_control_point_terrain_cost_batch_summary(
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    rows = [_control_point_terrain_cost_case_summary(case) for case in cases]
    return {
        "schema_version": GCS_CONTROL_POINT_TERRAIN_COST_CLI_BATCH_SCHEMA_VERSION,
        "case_count": len(rows),
        "selected_count": sum(1 for row in rows if row["outcome"] == "selected"),
        "blocked_count": sum(1 for row in rows if row["outcome"] == "blocked"),
        "terrain_objective_evaluated_count": sum(
            1
            for row in rows
            if row["terrain_objective_source"] not in {None, "not_evaluated"}
        ),
        "high_cost_exposure_blocked_count": sum(
            1
            for row in rows
            if row["outcome"] == "blocked" and (row["high_cost_exposure"] or 0.0) > 0.0
        ),
        "sampled_terrain_improved_count": sum(
            1
            for row in rows
            if row["cost_delta_vs_baseline"] is not None and row["cost_delta_vs_baseline"] < 0.0
        ),
        "decision_reason_counts": _counts(row["decision_reason"] for row in rows),
        "fallback_reason_counts": _counts(row["fallback_reason"] for row in rows),
        "expectation_failures": [
            {"case_id": row["case_id"], "mismatch_fields": row["mismatch_fields"]}
            for row in rows
            if row["mismatch_fields"]
        ],
        "cases": rows,
    }


def _run_cli_route_cases(
    *,
    output_dir: str | Path,
    summary_json: str | Path | None,
    scenarios: tuple[GcsCliScenario, ...],
    common_cli_args: tuple[str, ...],
    python_executable: str | None,
    extra_env: dict[str, str] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_json) if summary_json is not None else output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    platform_config_path = output_root / "batch-platform.json"
    platform_config_path.write_text(
        json.dumps(_batch_platform_config(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
            *common_cli_args,
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
    return matrix_cases, case_metadata


def _write_summary(summary_json: str | Path | None, output_dir: str | Path, summary: dict[str, Any]) -> None:
    output_root = Path(output_dir)
    summary_path = Path(summary_json) if summary_json is not None else output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GCS CLI scenario batch evidence")
    parser.add_argument(
        "--batch-kind",
        choices=("direction-cone", "motion-feasibility", "control-point-terrain-cost"),
        default="direction-cone",
        help="Batch evidence kind to run. Defaults to the direction-cone batch for backward compatibility.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for per-case request/route/diagnostic output")
    parser.add_argument("--summary-json", default=None, help="Path to write the batch summary JSON")
    parser.add_argument(
        "--case",
        action="append",
        help="Run one scenario case id; can be repeated. Defaults to all built-in cases.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.batch_kind == "motion-feasibility":
            summary = run_gcs_motion_feasibility_cli_batch(
                output_dir=args.output_dir,
                summary_json=args.summary_json,
                case_ids=tuple(args.case) if args.case else None,
            )
        elif args.batch_kind == "control-point-terrain-cost":
            summary = run_gcs_control_point_terrain_cost_cli_batch(
                output_dir=args.output_dir,
                summary_json=args.summary_json,
                case_ids=tuple(args.case) if args.case else None,
            )
        else:
            summary = run_gcs_cli_scenario_batch(
                output_dir=args.output_dir,
                summary_json=args.summary_json,
                case_ids=tuple(args.case) if args.case else None,
            )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _selected_scenarios(
    scenarios: dict[str, GcsCliScenario],
    case_ids: tuple[str, ...] | list[str] | None,
) -> tuple[GcsCliScenario, ...]:
    selected_ids = tuple(case_ids) if case_ids else tuple(scenarios)
    missing = [case_id for case_id in selected_ids if case_id not in scenarios]
    if missing:
        raise ValueError(f"unknown GCS CLI scenario case(s): {', '.join(missing)}")
    return tuple(scenarios[case_id] for case_id in selected_ids)


def _motion_feasibility_case_summary(case: dict[str, Any]) -> dict[str, Any]:
    payload = dict(case.get("payload") or {})
    expected = dict(case.get("expected") or {})
    motion_summary = payload.get("gcs_motion_feasibility_constraint_summary") or {}
    if not isinstance(motion_summary, dict):
        motion_summary = {}
    outcome = str(payload.get("gcs_motion_feasibility_feasibility_status") or "not_evaluated")
    decision_reason = _motion_decision_reason(payload, outcome)
    fallback_reason = None if outcome == "feasible" else decision_reason
    candidate_selected = bool(payload.get("gcs_candidate_selected", False))
    candidate_fallback_reason = payload.get("gcs_candidate_fallback_reason")
    row = {
        "case_id": str(case.get("case_id") or "unnamed"),
        "outcome": outcome,
        "decision_reason": decision_reason,
        "fallback_reason": fallback_reason,
        "trajectory_attempted": payload.get("gcs_trajectory_attempted"),
        "trajectory_success": payload.get("gcs_trajectory_success"),
        "trajectory_reason": payload.get("gcs_trajectory_reason"),
        "motion_evaluated": payload.get("gcs_motion_feasibility_evaluated"),
        "motion_status": outcome,
        "motion_fallback_reason": payload.get("gcs_motion_feasibility_fallback_reason"),
        "motion_model": payload.get("gcs_motion_feasibility_motion_model"),
        "min_turning_radius_m": payload.get("gcs_motion_feasibility_min_turning_radius_m"),
        "max_heading_change_deg": payload.get("gcs_motion_feasibility_max_heading_change_deg"),
        "curvature_violation_count": int(payload.get("gcs_motion_feasibility_curvature_violation_count") or 0),
        "heading_violation_count": int(payload.get("gcs_motion_feasibility_heading_violation_count") or 0),
        "violation_indices": list(payload.get("gcs_motion_feasibility_violation_indices") or []),
        "sample_count": int(payload.get("gcs_motion_feasibility_sample_count") or 0),
        "path_length": payload.get("gcs_motion_feasibility_path_length"),
        "max_observed_curvature": motion_summary.get("max_observed_curvature"),
        "min_observed_turning_radius_m": motion_summary.get("min_observed_turning_radius_m"),
        "max_observed_heading_change_deg": motion_summary.get("max_observed_heading_change_deg"),
        "candidate_selected": candidate_selected,
        "candidate_available": payload.get("gcs_candidate_available"),
        "candidate_fallback_reason": candidate_fallback_reason,
        "motion_gate_blocked_candidate": (
            outcome == "infeasible"
            and candidate_fallback_reason == "motion_infeasible"
            and payload.get("gcs_trajectory_success") is True
        ),
    }
    row["mismatch_fields"] = _motion_mismatch_fields(row, expected)
    return row


def _motion_decision_reason(payload: dict[str, Any], outcome: str) -> str | None:
    if payload.get("gcs_trajectory_reason") == "pydrake_unavailable":
        return "pydrake_unavailable"
    if outcome == "feasible":
        return "motion_feasible"
    reason = (
        payload.get("gcs_motion_feasibility_fallback_reason")
        or payload.get("gcs_candidate_fallback_reason")
        or payload.get("gcs_trajectory_reason")
    )
    return str(reason) if reason is not None else None


def _motion_mismatch_fields(row: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    expected_outcome = expected.get("outcome")
    if expected_outcome is not None and row.get("outcome") != expected_outcome:
        mismatches.append("outcome")
    expected_reason = expected.get("decision_reason")
    if expected_reason is not None and row.get("decision_reason") != expected_reason:
        mismatches.append("decision_reason")
    return mismatches


def _control_point_terrain_cost_case_summary(case: dict[str, Any]) -> dict[str, Any]:
    payload = dict(case.get("payload") or {})
    expected = dict(case.get("expected") or {})
    trajectory_cost = _mapping(payload.get("gcs_trajectory_cost_summary"))
    candidate_cost = _mapping(payload.get("gcs_candidate_cost_summary"))
    selected = bool(payload.get("gcs_candidate_selected", False))
    outcome = "selected" if selected else "blocked"
    decision_reason = _control_point_terrain_decision_reason(payload, candidate_cost)
    fallback_reason = None if selected else decision_reason
    row = {
        "case_id": str(case.get("case_id") or "unnamed"),
        "outcome": outcome,
        "selected": selected,
        "fallback_reason": fallback_reason,
        "decision_reason": decision_reason,
        "trajectory_backend": payload.get("gcs_trajectory_backend"),
        "trajectory_attempted": payload.get("gcs_trajectory_attempted"),
        "trajectory_success": payload.get("gcs_trajectory_success"),
        "trajectory_reason": payload.get("gcs_trajectory_reason"),
        "candidate_available": payload.get("gcs_candidate_available"),
        "candidate_selected": selected,
        "candidate_fallback_reason": payload.get("gcs_candidate_fallback_reason"),
        "candidate_decision": candidate_cost.get("candidate_decision"),
        "cost_delta_vs_baseline": payload.get(
            "gcs_candidate_cost_delta_vs_baseline",
            candidate_cost.get("cost_delta_vs_baseline"),
        ),
        "sampled_terrain_cost": trajectory_cost.get("sampled_terrain_cost"),
        "control_point_terrain_cost": trajectory_cost.get("control_point_terrain_cost"),
        "terrain_objective_source": trajectory_cost.get("terrain_objective_source"),
        "terrain_objective_weight": trajectory_cost.get("terrain_objective_weight"),
        "terrain_objective_anchor_count": trajectory_cost.get("terrain_objective_anchor_count"),
        "terrain_objective_boundary": trajectory_cost.get("terrain_objective_boundary"),
        "high_cost_exposure": payload.get(
            "gcs_candidate_high_cost_exposure",
            candidate_cost.get("high_cost_exposure"),
        ),
        "baseline_high_cost_exposure": candidate_cost.get("baseline_high_cost_exposure"),
        "high_cost_exposure_delta_vs_baseline": candidate_cost.get(
            "high_cost_exposure_delta_vs_baseline"
        ),
        "motion_status": payload.get("gcs_motion_feasibility_feasibility_status"),
        "motion_fallback_reason": payload.get("gcs_motion_feasibility_fallback_reason"),
    }
    row["mismatch_fields"] = _motion_mismatch_fields(row, expected)
    return row


def _control_point_terrain_decision_reason(
    payload: dict[str, Any],
    candidate_cost: dict[str, Any],
) -> str | None:
    if payload.get("gcs_trajectory_reason") == "pydrake_unavailable":
        return "pydrake_unavailable"
    cost_reason = candidate_cost.get("decision_reason")
    if cost_reason == "cost_not_evaluated":
        cost_reason = None
    reason = (
        cost_reason
        or payload.get("gcs_candidate_selection_reason")
        or payload.get("gcs_candidate_fallback_reason")
        or payload.get("gcs_trajectory_reason")
    )
    return str(reason) if reason is not None else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


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
        "direction_cone_obstacle_detour": GcsCliScenario(
            case_id="direction_cone_obstacle_detour",
            request=_request(
                "direction_cone_obstacle_detour",
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
            expected_decision_reason="direction_cone_constraint_violation",
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


def _motion_feasibility_scenarios() -> dict[str, GcsCliScenario]:
    diagonal_turn_request = _request(
        "motion_feasibility_turn",
        cost=_filled_cost(width=8, height=4, value=1),
        passable_mask=_full_mask(width=8, height=4),
        start=[0, 3],
        goal=[7, 0],
    )
    straight_request = _request(
        "straight_feasible",
        cost=[
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        passable_mask=_full_mask(width=4, height=3),
        start=[0, 1],
        goal=[3, 1],
    )
    return {
        "straight_feasible": GcsCliScenario(
            case_id="straight_feasible",
            request=straight_request,
            expected_outcome="feasible",
            expected_decision_reason="motion_feasible",
            cli_args=("--max-shortcut-cost", "0", "--max-heading-change-deg", "120"),
        ),
        "gentle_turn_feasible": GcsCliScenario(
            case_id="gentle_turn_feasible",
            request=_diagonal_open_request("gentle_turn_feasible", size=28),
            expected_outcome="feasible",
            expected_decision_reason="motion_feasible",
            cli_args=("--max-shortcut-cost", "0", "--max-heading-change-deg", "120"),
        ),
        "sharp_turn_blocked": GcsCliScenario(
            case_id="sharp_turn_blocked",
            request=diagonal_turn_request,
            expected_outcome="infeasible",
            expected_decision_reason="heading_constraint_violation",
            cli_args=("--max-shortcut-cost", "0", "--max-heading-change-deg", "1"),
        ),
        "tight_radius_blocked": GcsCliScenario(
            case_id="tight_radius_blocked",
            request=diagonal_turn_request,
            expected_outcome="infeasible",
            expected_decision_reason="curvature_constraint_violation",
            cli_args=(
                "--max-shortcut-cost",
                "0",
                "--max-heading-change-deg",
                "120",
                "--min-turning-radius",
                "20",
            ),
        ),
        "direction_cone_selected_but_motion_blocked": GcsCliScenario(
            case_id="direction_cone_selected_but_motion_blocked",
            request=_diagonal_open_request("direction_cone_selected_but_motion_blocked", size=28),
            expected_outcome="infeasible",
            expected_decision_reason="heading_constraint_violation",
            cli_args=("--max-shortcut-cost", "0", "--max-heading-change-deg", "1"),
        ),
        "pydrake_unavailable": GcsCliScenario(
            case_id="pydrake_unavailable",
            request=straight_request,
            expected_outcome="diagnostic_only",
            expected_decision_reason="pydrake_unavailable",
            env={FORCE_PYDRAKE_UNAVAILABLE_ENV: "1"},
        ),
    }


def _control_point_terrain_cost_scenarios() -> dict[str, GcsCliScenario]:
    turn_request = _request(
        "control_point_motion_infeasible",
        cost=_filled_cost(width=8, height=4, value=1),
        passable_mask=_full_mask(width=8, height=4),
        start=[0, 3],
        goal=[7, 0],
    )
    high_cost_exposure_request = _request(
        "control_point_high_cost_exposure_blocked",
        cost=[
            [1, 10, 1, 1, 1],
            [1, 1, 10, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 10, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        passable_mask=_full_mask(width=5, height=5),
        start=[0, 4],
        goal=[4, 0],
    )
    straight_request = _request(
        "control_point_pydrake_unavailable",
        cost=[
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        passable_mask=_full_mask(width=4, height=3),
        start=[0, 1],
        goal=[3, 1],
    )
    return {
        "control_point_low_cost_selected": GcsCliScenario(
            case_id="control_point_low_cost_selected",
            request=_diagonal_open_request("control_point_low_cost_selected", size=28),
            expected_outcome="selected",
            expected_decision_reason="gcs_candidate_quality_improved",
            cli_args=("--max-shortcut-cost", "0"),
        ),
        "control_point_cost_dominated": GcsCliScenario(
            case_id="control_point_cost_dominated",
            request=_diagonal_open_request("control_point_cost_dominated", size=10),
            expected_outcome="blocked",
            expected_decision_reason="cost_dominated",
            cli_args=("--max-shortcut-cost", "0"),
        ),
        "control_point_high_cost_exposure_blocked": GcsCliScenario(
            case_id="control_point_high_cost_exposure_blocked",
            request=high_cost_exposure_request,
            expected_outcome="blocked",
            expected_decision_reason="cost_dominated",
            cli_args=("--max-shortcut-cost", "0"),
        ),
        "control_point_motion_infeasible": GcsCliScenario(
            case_id="control_point_motion_infeasible",
            request=turn_request,
            expected_outcome="blocked",
            expected_decision_reason="motion_infeasible",
            cli_args=("--max-shortcut-cost", "0", "--max-heading-change-deg", "1"),
        ),
        "control_point_pydrake_unavailable": GcsCliScenario(
            case_id="control_point_pydrake_unavailable",
            request=straight_request,
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
