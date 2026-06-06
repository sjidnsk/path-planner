from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

GCS_SCENARIO_MATRIX_SCHEMA_VERSION = "gcs_direction_cone_scenario_matrix/v1"


def build_gcs_scenario_matrix_summary(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [_case_summary(case) for case in cases]
    selected_count = sum(1 for row in rows if row["outcome"] == "selected")
    blocked_count = sum(1 for row in rows if row["outcome"] == "blocked")
    expectation_failures = [
        {
            "case_id": row["case_id"],
            "mismatch_fields": row["mismatch_fields"],
        }
        for row in rows
        if row["mismatch_fields"]
    ]
    return {
        "schema_version": GCS_SCENARIO_MATRIX_SCHEMA_VERSION,
        "case_count": len(rows),
        "selected_count": selected_count,
        "blocked_count": blocked_count,
        "decision_reason_counts": _counts(row["decision_reason"] for row in rows),
        "fallback_reason_counts": _counts(row["fallback_reason"] for row in rows),
        "expectation_failures": expectation_failures,
        "cases": rows,
    }


def _case_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(case.get("payload") or {})
    expected = dict(case.get("expected") or {})
    direction_cone = _direction_cone_summary(payload)
    candidate_cost = dict(payload.get("gcs_candidate_cost_summary") or {})
    selected = bool(payload.get("gcs_candidate_selected", False))
    fallback_reason = _fallback_reason(payload, direction_cone)
    decision_reason = _decision_reason(payload, candidate_cost, fallback_reason)
    outcome = "selected" if selected else "blocked"
    row = {
        "case_id": str(case.get("case_id") or "unnamed"),
        "outcome": outcome,
        "selected": selected,
        "fallback_reason": fallback_reason,
        "decision_reason": decision_reason,
        "trajectory_attempted": payload.get("gcs_trajectory_attempted"),
        "trajectory_success": payload.get("gcs_trajectory_success"),
        "trajectory_reason": payload.get("gcs_trajectory_reason"),
        "direction_cone_status": direction_cone.get("status"),
        "direction_cone_backend_enforced": direction_cone.get("backend_enforced"),
        "direction_cone_violation_count": direction_cone.get("violation_count"),
        "direction_cone_risk_flags": list(direction_cone.get("risk_flags") or []),
        "rho_source_counts": dict(direction_cone.get("rho_source_counts") or {}),
        "portal_width_min_m": direction_cone.get("portal_width_min_m"),
        "support_width_min_m": direction_cone.get("support_width_min_m"),
        "constraint_tightness_min": direction_cone.get("constraint_tightness_min"),
        "candidate_decision": candidate_cost.get("candidate_decision"),
        "quality_gate": dict(candidate_cost.get("quality_gate") or {}),
        "cost_delta_vs_baseline": payload.get(
            "gcs_candidate_cost_delta_vs_baseline",
            candidate_cost.get("cost_delta_vs_baseline"),
        ),
        "high_cost_exposure": payload.get(
            "gcs_candidate_high_cost_exposure",
            candidate_cost.get("high_cost_exposure"),
        ),
    }
    row["mismatch_fields"] = _mismatch_fields(row, expected)
    return row


def _direction_cone_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate_constraints = payload.get("gcs_candidate_constraint_summary") or {}
    direction_cone = candidate_constraints.get("direction_cone") if isinstance(candidate_constraints, Mapping) else None
    if isinstance(direction_cone, Mapping) and direction_cone:
        return dict(direction_cone)
    trajectory_summary = payload.get("gcs_trajectory_constraint_summary") or {}
    return dict(trajectory_summary) if isinstance(trajectory_summary, Mapping) else {}


def _fallback_reason(payload: Mapping[str, Any], direction_cone: Mapping[str, Any]) -> str | None:
    if payload.get("gcs_candidate_selected") is True:
        return None
    trajectory_reason = payload.get("gcs_trajectory_reason")
    if trajectory_reason == "pydrake_unavailable":
        return str(trajectory_reason)
    direct_reason = payload.get("gcs_candidate_fallback_reason") or trajectory_reason
    if direct_reason:
        return str(direct_reason)
    risk_flags = direction_cone.get("risk_flags") or []
    if risk_flags:
        return str(risk_flags[0])
    return None


def _decision_reason(
    payload: Mapping[str, Any],
    candidate_cost: Mapping[str, Any],
    fallback_reason: str | None,
) -> str | None:
    direction_cone = _direction_cone_summary(payload)
    risk_flags = direction_cone.get("risk_flags") or []
    if fallback_reason == "direction_cone_constraint_violation" and "degenerate_portal_width" in risk_flags:
        return "degenerate_portal_width"
    cost_reason = candidate_cost.get("decision_reason")
    if cost_reason == "cost_not_evaluated":
        cost_reason = None
    reason = cost_reason or payload.get("gcs_candidate_selection_reason") or fallback_reason
    return str(reason) if reason is not None else None


def _mismatch_fields(row: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    mismatches: list[str] = []
    expected_outcome = expected.get("outcome")
    if expected_outcome is not None and row.get("outcome") != expected_outcome:
        mismatches.append("outcome")
    expected_reason = expected.get("decision_reason")
    if expected_reason is not None and row.get("decision_reason") != expected_reason:
        mismatches.append("decision_reason")
    return mismatches


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
