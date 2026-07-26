from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter_ns

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
    timing_measurement_valid: bool = True

    def __post_init__(self) -> None:
        if type(self.total_ns) is not int:
            raise TypeError("total_ns must be an exact integer")
        if any(type(value) is not int or value < 0 for value in self.phase_values()):
            raise ValueError("timing phases must be nonnegative integers")
        if self.total_ns != sum(self.phase_values()):
            raise ValueError("total_ns must equal the exact sum of timing phases")
        if type(self.timing_measurement_valid) is not bool:
            raise TypeError("timing_measurement_valid must be exact bool")

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


_MISSING_CATEGORY = object()
_FAILURE_CATEGORY_VALUES = frozenset(category.value for category in FailureCategoryV2)
_PHASE_NAMES = (
    "input_validation",
    "platform_instantiation",
    "search",
    "complete_route_validation",
    "result_assembly",
)


def _normalize_outcome(outcome: object) -> object:
    if type(outcome) is PlanningFailureV2:
        return outcome
    if not isinstance(outcome, Mapping):
        return outcome
    category = outcome.get("category", _MISSING_CATEGORY)
    if category is _MISSING_CATEGORY:
        return outcome
    recognized = (
        type(category) is FailureCategoryV2
        or type(category) is str
        and category in _FAILURE_CATEGORY_VALUES
    )
    if recognized:
        clean = dict(outcome)
        clean["route"] = None
        return clean
    clean = dict(outcome)
    clean["category"] = FailureCategoryV2.INTERNAL_ERROR
    clean["reason_code"] = "unknown_failure_category"
    clean["route"] = None
    return clean


def _validated_outcome(
    outcome: object,
    validation: object,
) -> object:
    if type(outcome) is PlanningSuccessV2 and not (
        type(validation) is ValidationEvidenceV2
        and validation.passed is True
        and validation.level is ValidationLevelV2.L2
    ):
        return _revalidation_failure(outcome, validation)
    return outcome


def _read_clock(clock_ns: Callable[[], int]) -> int:
    value = clock_ns()
    if type(value) is not int:
        raise TypeError("clock_ns must return exact integers")
    return value


def _timing(
    durations: list[int],
    *,
    measurement_valid: bool,
) -> TimingBreakdownV2:
    return TimingBreakdownV2(
        input_validation_ns=durations[0],
        platform_instantiation_ns=durations[1],
        search_ns=durations[2],
        complete_route_validation_ns=durations[3],
        result_assembly_ns=durations[4],
        total_ns=sum(durations),
        timing_measurement_valid=measurement_valid,
    )


def _internal_failure_result(
    *,
    exception_stage: str,
    exception: Exception,
    durations: list[int],
    timing_measurement_valid: bool,
) -> TimedRequestResultV2:
    failure = {
        "category": FailureCategoryV2.INTERNAL_ERROR,
        "reason_code": "timed_executor_exception",
        "route": None,
        "exception_stage": exception_stage,
        "exception_type": type(exception).__name__,
        "timing_measurement_valid": timing_measurement_valid,
    }
    return TimedRequestResultV2(
        outcome=failure,
        assembled=failure,
        timing=_timing(
            durations,
            measurement_valid=timing_measurement_valid,
        ),
    )


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
    durations = [0, 0, 0, 0, 0]
    try:
        boundary = _read_clock(clock_ns)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        return _internal_failure_result(
            exception_stage=_PHASE_NAMES[0],
            exception=exc,
            durations=durations,
            timing_measurement_valid=False,
        )

    request: object = None
    stack: object = None
    outcome: object = None
    validation: object = None
    assembled: object = None

    def run_phase(index: int, callback: Callable[[], object]) -> tuple[object, int] | TimedRequestResultV2:
        nonlocal boundary
        try:
            value = callback()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            measurement_valid = True
            try:
                finished = _read_clock(clock_ns)
                if finished < boundary:
                    raise ValueError("clock_ns must be nondecreasing")
                durations[index] = finished - boundary
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except Exception:
                measurement_valid = False
            return _internal_failure_result(
                exception_stage=_PHASE_NAMES[index],
                exception=exc,
                durations=durations,
                timing_measurement_valid=measurement_valid,
            )
        try:
            finished = _read_clock(clock_ns)
            if finished < boundary:
                raise ValueError("clock_ns must be nondecreasing")
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:
            return _internal_failure_result(
                exception_stage=_PHASE_NAMES[index],
                exception=exc,
                durations=durations,
                timing_measurement_valid=False,
            )
        durations[index] = finished - boundary
        boundary = finished
        return value, finished

    phase_result = run_phase(0, lambda: decode_request(encoded_request))
    if type(phase_result) is TimedRequestResultV2:
        return phase_result
    request = phase_result[0]

    phase_result = run_phase(1, lambda: build_platform_stack(request))
    if type(phase_result) is TimedRequestResultV2:
        return phase_result
    stack = phase_result[0]

    def search_phase() -> object:
        return _normalize_outcome(plan_request(request, stack))

    phase_result = run_phase(2, search_phase)
    if type(phase_result) is TimedRequestResultV2:
        return phase_result
    outcome = phase_result[0]

    def validation_phase() -> tuple[object, object]:
        checked = revalidate_success_route(request, outcome, stack)
        return _validated_outcome(outcome, checked), checked

    phase_result = run_phase(3, validation_phase)
    if type(phase_result) is TimedRequestResultV2:
        return phase_result
    outcome, validation = phase_result[0]

    def assembly_phase() -> object:
        provisional = _timing(durations, measurement_valid=True)
        return assemble_result(outcome, validation, provisional)

    phase_result = run_phase(4, assembly_phase)
    if type(phase_result) is TimedRequestResultV2:
        return phase_result
    assembled = phase_result[0]
    return TimedRequestResultV2(
        outcome=outcome,
        assembled=assembled,
        timing=_timing(durations, measurement_valid=True),
    )
