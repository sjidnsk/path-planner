from __future__ import annotations

from path_planner.v2 import FailureCategoryV2, PlatformKindV2
from path_planner.v2.timing import execute_timed_request_v2


def _callbacks(*, outcome: object = "route", validation: object = "l2-evidence") -> dict[str, object]:
    return {
        "decode_request": lambda payload: payload,
        "build_platform_stack": lambda request: ("stack", request),
        "plan_request": lambda request, stack: outcome,
        "revalidate_success_route": lambda request, planned, stack: validation,
        "assemble_result": lambda planned, validation, timing: {
            "planned": planned,
            "validation": validation,
            "timing": timing,
        },
    }


def test_timed_executor_uses_five_nonoverlapping_perf_counter_ns_intervals() -> None:
    ticks = iter((10, 20, 30, 40, 50, 60, 70, 80, 90, 100))

    result = execute_timed_request_v2(
        "request", clock_ns=lambda: next(ticks), **_callbacks()
    )

    assert result.timing.as_dict() == {
        "input_validation_ns": 10,
        "platform_instantiation_ns": 10,
        "search_ns": 10,
        "complete_route_validation_ns": 10,
        "result_assembly_ns": 10,
        "total_ns": 50,
    }


def test_phase_sum_equals_total_ns_exactly() -> None:
    ticks = iter((0, 3, 3, 8, 8, 21, 21, 34, 34, 55))
    result = execute_timed_request_v2(
        "request", clock_ns=lambda: next(ticks), **_callbacks()
    )

    assert result.timing.total_ns == sum(result.timing.phase_values())


def test_timeout_outcome_cannot_encode_a_success_route() -> None:
    timeout = {
        "category": FailureCategoryV2.TIMEOUT,
        "platform_kind": PlatformKindV2.WHEEL,
        "route": "must-not-survive",
    }
    result = execute_timed_request_v2(
        "request", **_callbacks(outcome=timeout, validation="checked-failure")
    )

    assert result.outcome["category"] is FailureCategoryV2.TIMEOUT
    assert result.assembled["planned"]["route"] is None


def test_timing_wrapper_does_not_change_plan_v2_semantic_digest() -> None:
    planned = {"semantic_digest": "a" * 64, "route": ("a", "b")}
    result = execute_timed_request_v2("request", **_callbacks(outcome=planned))

    assert result.outcome is planned
    assert result.outcome["semantic_digest"] == "a" * 64
