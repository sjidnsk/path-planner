from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

from path_planner.v2.contracts import (
    FailureCategoryV2,
    FailureEvidenceV2,
    PlanningFailureV2,
    PlanningSuccessV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)


@dataclass(frozen=True, slots=True)
class TimingBreakdownV2:
    input_validation_ns: int
    platform_instantiation_ns: int
    search_ns: int
    complete_route_validation_ns: int
    result_assembly_ns: int
    total_ns: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.phase_values()):
            raise ValueError("timing phases must be nonnegative integers")
        if self.total_ns != sum(self.phase_values()):
            raise ValueError("total_ns must equal the exact sum of timing phases")

    def phase_values(self) -> tuple[int, int, int, int, int]:
        return (
            self.input_validation_ns,
            self.platform_instantiation_ns,
            self.search_ns,
            self.complete_route_validation_ns,
            self.result_assembly_ns,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_validation_ns": self.input_validation_ns,
            "platform_instantiation_ns": self.platform_instantiation_ns,
            "search_ns": self.search_ns,
            "complete_route_validation_ns": self.complete_route_validation_ns,
            "result_assembly_ns": self.result_assembly_ns,
            "total_ns": self.total_ns,
        }


@dataclass(frozen=True, slots=True)
class TimedRequestResultV2:
    outcome: object
    assembled: object
    timing: TimingBreakdownV2


def _interval(clock_ns: Callable[[], int], callback: Callable[[], Any]) -> tuple[Any, int]:
    started = clock_ns()
    value = callback()
    finished = clock_ns()
    if type(started) is not int or type(finished) is not int or finished < started:
        raise ValueError("clock_ns must return nondecreasing exact integers")
    return value, finished - started


def _revalidation_failure(success: PlanningSuccessV2, evidence: object) -> PlanningFailureV2:
    details = (("validation_type", type(evidence).__name__),)
    return PlanningFailureV2(
        request_id=success.request_id,
        platform_kind=success.platform_kind,
        category=FailureCategoryV2.VALIDATION_FAILED,
        reason_code="complete_route_l2_revalidation_failed",
        evidence=FailureEvidenceV2(
            stage="complete_route_validation",
            checks=("complete_route_l2_revalidation_failed",),
            details=details,
        ),
        search_telemetry=success.search_telemetry,
    )


def _cannot_encode_timeout_success(outcome: object) -> object:
    if type(outcome) is PlanningFailureV2:
        return outcome
    if isinstance(outcome, Mapping) and outcome.get("category") is FailureCategoryV2.TIMEOUT:
        clean = dict(outcome)
        clean["route"] = None
        return clean
    return outcome


def execute_timed_request_v2(
    encoded_request: object,
    *,
    decode_request: Callable[[object], object],
    build_platform_stack: Callable[[object], object],
    plan_request: Callable[[object, object], object],
    revalidate_success_route: Callable[[object, object, object], object],
    assemble_result: Callable[[object, object, TimingBreakdownV2], object],
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> TimedRequestResultV2:
    """Run five sequential, non-overlapping in-worker timing phases.

    Queue waiting and input-file reads are deliberately outside this primitive:
    callers hand it already-loaded ``encoded_request`` data from their worker.
    """
    if not all(
        callable(callback)
        for callback in (
            decode_request,
            build_platform_stack,
            plan_request,
            revalidate_success_route,
            assemble_result,
            clock_ns,
        )
    ):
        raise TypeError("all timed request callbacks and clock_ns must be callable")
    try:
        request, input_ns = _interval(clock_ns, lambda: decode_request(encoded_request))
        stack, stack_ns = _interval(clock_ns, lambda: build_platform_stack(request))
        outcome, search_ns = _interval(clock_ns, lambda: plan_request(request, stack))
        outcome = _cannot_encode_timeout_success(outcome)
        validation, validation_ns = _interval(
            clock_ns,
            lambda: revalidate_success_route(request, outcome, stack),
        )
        if type(outcome) is PlanningSuccessV2 and not (
            type(validation) is ValidationEvidenceV2
            and validation.passed is True
            and validation.level is ValidationLevelV2.L2
        ):
            outcome = _revalidation_failure(outcome, validation)
        provisional = TimingBreakdownV2(
            input_ns, stack_ns, search_ns, validation_ns, 0,
            input_ns + stack_ns + search_ns + validation_ns,
        )
        assembled, assembly_ns = _interval(
            clock_ns,
            lambda: assemble_result(outcome, validation, provisional),
        )
        timing = TimingBreakdownV2(
            input_ns, stack_ns, search_ns, validation_ns, assembly_ns,
            input_ns + stack_ns + search_ns + validation_ns + assembly_ns,
        )
        return TimedRequestResultV2(outcome=outcome, assembled=assembled, timing=timing)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        # Return a row-shaped failure instead of silently dropping a formal call.
        failure = {
            "category": FailureCategoryV2.INTERNAL_ERROR,
            "reason_code": "timed_executor_exception",
            "route": None,
            "exception_type": type(exc).__name__,
        }
        zero = TimingBreakdownV2(0, 0, 0, 0, 0, 0)
        return TimedRequestResultV2(outcome=failure, assembled=failure, timing=zero)
