from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import pytest

from path_planner.v2 import (
    CacheEvidenceV2,
    CostBreakdownV2,
    FailureCategoryV2,
    FailureEvidenceV2,
    ObservationProjectionV2,
    PlanningFailureV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    RoutePrimitiveV2,
    SearchTelemetryV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.timing import TimingBreakdownV2, execute_timed_request_v2


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


@dataclass
class _AdvancingClock:
    now: int = 0
    reads: int = 0

    def __call__(self) -> int:
        self.reads += 1
        return self.now

    def advance(self, amount: int) -> None:
        self.now += amount


def test_timed_executor_uses_five_nonoverlapping_perf_counter_ns_intervals() -> None:
    clock = _AdvancingClock()

    def phase(value: object, amount: int):
        def callback(*_args: object) -> object:
            clock.advance(amount)
            return value

        return callback

    result = execute_timed_request_v2(
        "request",
        decode_request=phase("decoded", 2),
        build_platform_stack=phase("stack", 3),
        plan_request=phase({"semantic_digest": "a" * 64}, 5),
        revalidate_success_route=phase("validation", 7),
        assemble_result=phase("assembled", 11),
        clock_ns=clock,
    )

    assert result.timing.as_dict() == {
        "input_validation_ns": 2,
        "platform_instantiation_ns": 3,
        "search_ns": 5,
        "complete_route_validation_ns": 7,
        "result_assembly_ns": 11,
        "total_ns": 28,
    }
    assert clock.reads == 6


def test_phase_sum_equals_total_ns_exactly() -> None:
    ticks = iter((0, 3, 8, 21, 34, 55))
    result = execute_timed_request_v2(
        "request", clock_ns=lambda: next(ticks), **_callbacks()
    )

    assert result.timing.total_ns == sum(result.timing.phase_values())


def test_phase_sum_rejects_bool_total_even_when_bool_compares_equal() -> None:
    with pytest.raises(TypeError, match="total_ns"):
        TimingBreakdownV2(1, 0, 0, 0, 0, True)


@pytest.mark.parametrize("category", tuple(FailureCategoryV2))
@pytest.mark.parametrize("as_string", (False, True))
@pytest.mark.parametrize("include_route", (False, True))
def test_structured_failure_mapping_cannot_encode_a_success_route(
    category: FailureCategoryV2,
    as_string: bool,
    include_route: bool,
) -> None:
    failure = {"category": category.value if as_string else category}
    if include_route:
        failure["route"] = "must-not-survive"
    result = execute_timed_request_v2(
        "request", **_callbacks(outcome=failure, validation="checked-failure")
    )

    assert result.outcome["category"] == failure["category"]
    assert result.assembled["planned"]["route"] is None


def test_unknown_nonempty_failure_category_fails_closed_without_route() -> None:
    result = execute_timed_request_v2(
        "request",
        **_callbacks(
            outcome={"category": "future-failure", "route": "must-not-survive"},
            validation="checked-failure",
        ),
    )

    assert result.outcome["category"] is FailureCategoryV2.INTERNAL_ERROR
    assert result.outcome["reason_code"] == "unknown_failure_category"
    assert result.outcome["route"] is None


def _telemetry() -> SearchTelemetryV2:
    return SearchTelemetryV2(
        expanded_states=1,
        generated_primitives=1,
        rejected_l0=0,
        rejected_l1=0,
        rejected_l2=0,
        elapsed_s=0.1,
        timed_out=False,
        accelerator_used=False,
        ackermann_feasible_claimed=False,
        termination_reason="complete",
    )


def _success() -> PlanningSuccessV2:
    start = PoseStateV2(0.0, 0.0, 0.0)
    goal = PoseStateV2(1.0, 0.0, 0.0)
    primitive = RoutePrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=start,
        end_state=goal,
        duration_s=1.0,
        distance_m=1.0,
        energy_cost=0.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
    )
    return PlanningSuccessV2(
        request_id="timed-success",
        platform_kind=PlatformKindV2.WHEEL,
        route=TypedRouteV2(
            platform_kind=PlatformKindV2.WHEEL,
            primitives=(primitive,),
            total_cost=1.0,
        ),
        observation_projection=ObservationProjectionV2(
            source="test",
            sample_states=(start, goal),
            expected_new_observed_cells=0.0,
            expected_information_gain=0.0,
        ),
        cost_breakdown=CostBreakdownV2(
            distance_cost=1.0,
            risk_cost=0.0,
            energy_cost=0.0,
            time_cost=0.0,
            total_cost=1.0,
        ),
        validation_evidence=ValidationEvidenceV2(
            validator_id="initial-l2",
            level=ValidationLevelV2.L2,
            passed=True,
            checks=("initial_l2_passed",),
        ),
        search_telemetry=_telemetry(),
        cache_evidence=CacheEvidenceV2(
            cache_namespace="test",
            cache_key="key",
            hit=False,
        ),
    )


@pytest.mark.parametrize(
    ("evidence", "stays_success"),
    (
        (
            ValidationEvidenceV2(
                validator_id="complete-route-l2",
                level=ValidationLevelV2.L2,
                passed=True,
                checks=("complete_route_l2_passed",),
            ),
            True,
        ),
        (
            ValidationEvidenceV2(
                validator_id="complete-route-l1",
                level=ValidationLevelV2.L1,
                passed=True,
                checks=("only_l1_passed",),
            ),
            False,
        ),
        (
            ValidationEvidenceV2(
                validator_id="complete-route-l2",
                level=ValidationLevelV2.L2,
                passed=False,
                checks=("complete_route_l2_failed",),
            ),
            False,
        ),
    ),
)
def test_planning_success_requires_complete_l2_revalidation(
    evidence: ValidationEvidenceV2,
    stays_success: bool,
) -> None:
    success = _success()
    result = execute_timed_request_v2(
        "request", **_callbacks(outcome=success, validation=evidence)
    )

    assert isinstance(result.outcome, PlanningSuccessV2) is stays_success
    if stays_success:
        assert result.outcome is success
    else:
        assert isinstance(result.outcome, PlanningFailureV2)
        assert result.outcome.category is FailureCategoryV2.VALIDATION_FAILED


def test_planning_failure_is_preserved_without_route() -> None:
    failure = PlanningFailureV2(
        request_id="timed-failure",
        platform_kind=PlatformKindV2.WHEEL,
        category=FailureCategoryV2.NO_COMPLETE_ROUTE,
        reason_code="no_route",
        evidence=FailureEvidenceV2(
            stage="search",
            checks=("no_route",),
            details=(),
        ),
        search_telemetry=_telemetry(),
    )
    result = execute_timed_request_v2(
        "request", **_callbacks(outcome=failure, validation="checked-failure")
    )

    assert result.outcome is failure
    assert not hasattr(result.outcome, "route")


class _SlowFailureMapping(Mapping[str, object]):
    def __init__(self, clock: _AdvancingClock) -> None:
        self._clock = clock
        self._data = {
            "category": FailureCategoryV2.TIMEOUT.value,
            "route": "must-not-survive",
        }

    def __getitem__(self, key: str) -> object:
        self._clock.advance(2)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        self._clock.advance(3)
        return iter(self._data)

    def __len__(self) -> int:
        self._clock.advance(5)
        return len(self._data)


def test_mapping_normalization_and_assembly_are_inside_named_phases() -> None:
    clock = _AdvancingClock()
    slow_mapping = _SlowFailureMapping(clock)

    def plan(*_args: object) -> object:
        clock.advance(7)
        return slow_mapping

    def revalidate(*_args: object) -> str:
        clock.advance(11)
        return "checked-failure"

    def assemble(*_args: object) -> str:
        clock.advance(13)
        return "assembled"

    result = execute_timed_request_v2(
        "request",
        decode_request=lambda payload: payload,
        build_platform_stack=lambda _request: "stack",
        plan_request=plan,
        revalidate_success_route=revalidate,
        assemble_result=assemble,
        clock_ns=clock,
    )

    assert result.timing.search_ns > 7
    assert result.timing.complete_route_validation_ns == 11
    assert result.timing.result_assembly_ns == 13
    assert result.timing.total_ns == sum(result.timing.phase_values())


def test_l2_decision_is_inside_complete_route_validation_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _AdvancingClock()
    evidence = ValidationEvidenceV2(
        validator_id="complete-route-l2",
        level=ValidationLevelV2.L2,
        passed=True,
        checks=("complete_route_l2_passed",),
    )
    success = _success()
    passed_descriptor = ValidationEvidenceV2.passed

    def slow_passed(instance: ValidationEvidenceV2) -> bool:
        clock.advance(17)
        return passed_descriptor.__get__(instance, ValidationEvidenceV2)

    monkeypatch.setattr(
        ValidationEvidenceV2,
        "passed",
        property(slow_passed),
    )
    result = execute_timed_request_v2(
        "request",
        clock_ns=clock,
        **_callbacks(outcome=success, validation=evidence),
    )

    assert result.timing.complete_route_validation_ns == 17
    assert isinstance(result.outcome, PlanningSuccessV2)


@pytest.mark.parametrize(
    "exception_stage",
    (
        "input_validation",
        "platform_instantiation",
        "search",
        "complete_route_validation",
        "result_assembly",
    ),
)
def test_callback_exception_preserves_partial_timing(
    exception_stage: str,
) -> None:
    clock = _AdvancingClock()
    phase_names = (
        "input_validation",
        "platform_instantiation",
        "search",
        "complete_route_validation",
        "result_assembly",
    )
    durations = (2, 3, 5, 7, 11)

    def callback(stage: str, value: object):
        def run(*_args: object) -> object:
            clock.advance(durations[phase_names.index(stage)])
            if stage == exception_stage:
                raise LookupError("boom")
            return value

        return run

    result = execute_timed_request_v2(
        "request",
        decode_request=callback("input_validation", "decoded"),
        build_platform_stack=callback("platform_instantiation", "stack"),
        plan_request=callback("search", {"semantic_digest": "a" * 64}),
        revalidate_success_route=callback(
            "complete_route_validation", "validation"
        ),
        assemble_result=callback("result_assembly", "assembled"),
        clock_ns=clock,
    )

    failed_index = phase_names.index(exception_stage)
    assert result.timing.phase_values() == tuple(
        duration if index <= failed_index else 0
        for index, duration in enumerate(durations)
    )
    assert result.timing.total_ns == sum(result.timing.phase_values())
    assert result.outcome["category"] is FailureCategoryV2.INTERNAL_ERROR
    assert result.outcome["exception_stage"] == exception_stage
    assert result.outcome["exception_type"] == "LookupError"
    assert result.outcome["timing_measurement_valid"] is True


@pytest.mark.parametrize("fatal", (KeyboardInterrupt, SystemExit, MemoryError))
def test_fatal_callback_exception_propagates(fatal: type[BaseException]) -> None:
    def stop(_payload: object) -> object:
        raise fatal()

    with pytest.raises(fatal):
        execute_timed_request_v2(
            "request", decode_request=stop, **{
                key: value
                for key, value in _callbacks().items()
                if key != "decode_request"
            }
        )


@pytest.mark.parametrize(
    "clock_values",
    (
        iter((0, "not-an-int")),
        iter((10, 5)),
    ),
)
def test_invalid_or_backwards_clock_fails_closed(clock_values: Iterator[object]) -> None:
    result = execute_timed_request_v2(
        "request",
        clock_ns=lambda: next(clock_values),
        **_callbacks(),
    )

    assert result.outcome["category"] is FailureCategoryV2.INTERNAL_ERROR
    assert result.outcome["exception_stage"] == "input_validation"
    assert result.outcome["timing_measurement_valid"] is False
    assert result.timing.timing_measurement_valid is False


def test_clock_exception_fails_closed_as_invalid_timing() -> None:
    calls = 0

    def clock() -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("clock failed")
        return 0

    result = execute_timed_request_v2(
        "request", clock_ns=clock, **_callbacks()
    )

    assert result.outcome["exception_stage"] == "input_validation"
    assert result.outcome["exception_type"] == "RuntimeError"
    assert result.outcome["timing_measurement_valid"] is False


def test_callback_exception_with_unreadable_end_boundary_marks_timing_invalid() -> None:
    calls = 0

    def clock() -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("end boundary unavailable")
        return 0

    def fail(_payload: object) -> object:
        raise LookupError("decode failed")

    result = execute_timed_request_v2(
        "request",
        clock_ns=clock,
        decode_request=fail,
        **{
            key: value
            for key, value in _callbacks().items()
            if key != "decode_request"
        },
    )

    assert calls == 2
    assert result.outcome["exception_type"] == "LookupError"
    assert result.outcome["timing_measurement_valid"] is False


def test_timing_wrapper_does_not_change_plan_v2_semantic_digest() -> None:
    planned = {"semantic_digest": "a" * 64, "route": ("a", "b")}
    result = execute_timed_request_v2("request", **_callbacks(outcome=planned))

    assert result.outcome is planned
    assert result.outcome["semantic_digest"] == "a" * 64
