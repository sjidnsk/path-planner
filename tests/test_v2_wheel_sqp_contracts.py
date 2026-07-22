from dataclasses import replace

import pytest

import path_planner.v2 as v2
from path_planner.v2.contracts import (
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    SearchTelemetryV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.wheel_sqp_contracts import (
    L2ReserveModelV1,
    WheelCorridorV2,
    WheelKinematicRouteV2,
    WheelKinematicSegmentV2,
    WheelL2CounterexampleV2,
    WheelSQPCandidateV2,
    WheelSQPInitialGuessV2,
    WheelSQPModeV2,
    WheelSQPOptimizationResultV2,
    WheelSQPResourceLedgerV1,
    WheelSQPSearchTelemetryV2,
    WheelSQPStatusV2,
    WheelSQPValidationEvidenceV2,
    WheelSQPWorkLedgerV1,
    WheelSQPWorkLimitError,
    WheelTrajectoryL2ResultV2,
)
from path_planner.v2.wheel_sqp_serialization import (
    CanonicalWheelCandidateV1,
    CanonicalWheelSegmentV1,
)


REQUEST_HASH = "c" * 64
PROFILE_HASH = "d" * 64
SNAPSHOT_HASH = "e" * 64


def _segment() -> WheelKinematicSegmentV2:
    start = PoseStateV2(0.0, 0.0, 0.0)
    end = PoseStateV2(1.0, 0.0, 0.0)
    return WheelKinematicSegmentV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=start,
        end_state=end,
        duration_s=1.0,
        distance_m=1.0,
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        v_mps=1.0,
        omega_radps=0.0,
        reverse=False,
        turn_in_place=False,
        relative_energy=1.0,
        samples=(start, end),
        segment_hash="f" * 64,
    )


def _route(segment: WheelKinematicSegmentV2) -> WheelKinematicRouteV2:
    return WheelKinematicRouteV2(
        platform_kind=PlatformKindV2.WHEEL,
        primitives=(segment,),
        total_cost=1.0,
        route_hash="a" * 64,
        source_candidate_hash="b" * 64,
        request_hash=REQUEST_HASH,
        profile_hash=PROFILE_HASH,
        terrain_snapshot_hash=SNAPSHOT_HASH,
        capability_revision="wheel_kinematic_corridor_sqp/v1",
        solver_contract_id="wheel_kinematic_direct_multiple_shooting_sqp/v1",
        canonicalization_id="wheel_kinematic_decimal12_half_even/v1",
        validator_contract_id="wheel_kinematic_continuous_rectangle_sweep_l2/v1",
    )


def test_wheel_kinematic_segment_and_route_require_exact_hashes_and_l2() -> None:
    segment = _segment()
    route = _route(segment)

    assert type(segment) is WheelKinematicSegmentV2
    assert type(route) is WheelKinematicRouteV2
    assert isinstance(route, TypedRouteV2)
    assert segment.kind is PrimitiveKindV2.WHEEL_MOTION
    assert segment.energy_cost == segment.relative_energy
    assert segment.validation_level is ValidationLevelV2.L2
    assert len(segment.segment_hash) == 64
    assert len(route.route_hash) == 64
    assert len(route.source_candidate_hash) == 64
    assert route.request_hash == REQUEST_HASH
    assert route.profile_hash == PROFILE_HASH
    assert route.terrain_snapshot_hash == SNAPSHOT_HASH
    with pytest.raises(ValueError, match="reverse"):
        replace(segment, reverse=not segment.reverse)


def test_wheel_sqp_evidence_and_telemetry_are_base_compatible_but_exactly_typed() -> None:
    evidence = WheelSQPValidationEvidenceV2(
        validator_id="wheel_kinematic_continuous_rectangle_sweep_l2/v1",
        level=ValidationLevelV2.L2,
        passed=True,
        checks=("ok",),
        route_hash="a" * 64,
        candidate_hash="b" * 64,
        request_hash=REQUEST_HASH,
        profile_hash=PROFILE_HASH,
        terrain_snapshot_hash=SNAPSHOT_HASH,
        solver_contract_id="wheel_kinematic_direct_multiple_shooting_sqp/v1",
        checked_interval_count=1,
        checked_cell_count=2,
        repair_applied=False,
    )
    telemetry = WheelSQPSearchTelemetryV2(
        expanded_states=1,
        generated_primitives=2,
        rejected_l0=0,
        rejected_l1=0,
        rejected_l2=0,
        elapsed_s=0.1,
        timed_out=False,
        accelerator_used=False,
        ackermann_feasible_claimed=False,
        termination_reason="complete",
        corridor_count=3,
        selected_corridor_index=0,
        corridor_expansions=4,
        sqp_iterations=5,
        sqp_function_evaluations=6,
        l2_interval_count=7,
        l2_cell_count=8,
        repair_attempted=False,
        decision_hash="9" * 64,
    )

    assert isinstance(evidence, ValidationEvidenceV2)
    assert isinstance(telemetry, SearchTelemetryV2)
    assert type(evidence) is WheelSQPValidationEvidenceV2
    assert type(telemetry) is WheelSQPSearchTelemetryV2
    assert evidence.route_hash == "a" * 64
    assert evidence.candidate_hash == "b" * 64
    assert telemetry.corridor_count == 3
    assert canonical_json_bytes(
        ValidationEvidenceV2("base/v1", ValidationLevelV2.L2, True, ("ok",))
    ) == b'{"checks":["ok"],"level":"L2","passed":true,"validator_id":"base/v1"}'


def test_wheel_sqp_contract_surface_is_opt_in_v2_only() -> None:
    assert v2.WheelKinematicSegmentV2 is WheelKinematicSegmentV2
    assert v2.WheelSQPModeV2 is WheelSQPModeV2
    assert v2.L2ReserveModelV1 is L2ReserveModelV1
    assert tuple(WheelSQPModeV2) == (
        WheelSQPModeV2.FORWARD,
        WheelSQPModeV2.REVERSE,
        WheelSQPModeV2.TURN_LEFT,
        WheelSQPModeV2.TURN_RIGHT,
        WheelSQPModeV2.STOP,
    )
    assert WheelSQPStatusV2.FEASIBLE.value == "feasible"
    assert "validation_level" not in CanonicalWheelCandidateV1.__dataclass_fields__
    assert "validation_level" not in CanonicalWheelSegmentV1.__dataclass_fields__
    assert not issubclass(CanonicalWheelSegmentV1, WheelKinematicSegmentV2)
    assert all(
        value is not None
        for value in (
            WheelCorridorV2,
            WheelSQPInitialGuessV2,
            WheelSQPCandidateV2,
            WheelSQPOptimizationResultV2,
            WheelL2CounterexampleV2,
            WheelTrajectoryL2ResultV2,
            WheelSQPResourceLedgerV1,
        )
    )


def _deadline(remaining_s: float = 1.0) -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, remaining_s, lambda: 0.0)


def _l2_evidence(*, passed: bool = True) -> WheelSQPValidationEvidenceV2:
    return WheelSQPValidationEvidenceV2(
        validator_id="wheel_kinematic_continuous_rectangle_sweep_l2/v1",
        level=ValidationLevelV2.L2,
        passed=passed,
        checks=("checked",),
        route_hash="a" * 64,
        candidate_hash="b" * 64,
        request_hash=REQUEST_HASH,
        profile_hash=PROFILE_HASH,
        terrain_snapshot_hash=SNAPSHOT_HASH,
        solver_contract_id="wheel_kinematic_direct_multiple_shooting_sqp/v1",
        checked_interval_count=1,
        checked_cell_count=1,
        repair_applied=False,
    )


def _reserve_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "segment_count": 1,
        "broadphase_cell_bound": 2,
        "interval_record_bound": 3,
        "encoded_state_bound": 4,
        "encoded_scalar_bound": 5,
        "max_segments": 10,
        "max_l2_candidate_cells": 20,
        "max_l2_interval_records": 30,
        "max_encoded_state_bound": 40,
        "max_encoded_scalar_bound": 50,
    }
    values.update(overrides)
    return values


def test_l2_reserve_uses_all_caller_supplied_caps_before_formula() -> None:
    reserve = L2ReserveModelV1().reserve_s(
        **_reserve_kwargs(
            segment_count=99,
            broadphase_cell_bound=98,
            interval_record_bound=97,
            encoded_state_bound=96,
            encoded_scalar_bound=95,
            max_segments=2,
            max_l2_candidate_cells=3,
            max_l2_interval_records=4,
            max_encoded_state_bound=5,
            max_encoded_scalar_bound=6,
        )
    )

    assert reserve == (
        0.015
        + 0.00025 * 2
        + 0.000002 * 3
        + 0.000001 * 4
        + 0.000002 * 5
        + 0.000001 * 6
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"segment_count": -1},
        {"max_encoded_state_bound": True},
        {"encoded_scalar_bound": float("inf")},
        {"encoded_state_bound": 10**400, "max_encoded_state_bound": 10**400},
    ],
)
def test_l2_reserve_assess_fails_closed_for_invalid_or_overflow_bounds(overrides) -> None:
    ledger = L2ReserveModelV1().assess(_deadline(), **_reserve_kwargs(**overrides))

    assert type(ledger) is WheelSQPResourceLedgerV1
    assert ledger.accepted is False
    assert ledger.reserve_s == 1.0
    assert ledger.remaining_s == 1.0
    assert ledger.reason_code == "wheel_sqp_resource_budget_exceeded"


def test_l2_reserve_assess_rejects_the_strict_deadline_boundary() -> None:
    ledger = L2ReserveModelV1().assess(
        _deadline(0.015),
        **_reserve_kwargs(
            segment_count=0,
            broadphase_cell_bound=0,
            interval_record_bound=0,
            encoded_state_bound=0,
            encoded_scalar_bound=0,
        ),
    )

    assert ledger.accepted is False
    assert ledger.reserve_s == ledger.remaining_s == 0.015
    assert ledger.reason_code == "wheel_sqp_resource_budget_exceeded"


def test_passed_l2_result_requires_passing_exact_wheel_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        WheelTrajectoryL2ResultV2(True, None, None)
    with pytest.raises(ValueError, match="evidence"):
        WheelTrajectoryL2ResultV2(True, _l2_evidence(passed=False), None)
    with pytest.raises(ValueError, match="counterexample"):
        WheelTrajectoryL2ResultV2(False, None, None)

    result = WheelTrajectoryL2ResultV2(True, _l2_evidence(), None)

    assert result.passed is True
    assert result.evidence is not None


def test_wheel_sqp_validation_evidence_rejects_noncontract_validator() -> None:
    with pytest.raises(ValueError, match="validator_id"):
        replace(_l2_evidence(), validator_id="fake/v1")


def test_shared_work_ledger_exposes_read_only_cumulative_counters() -> None:
    budget = ResourceBudgetV2(
        max_expanded_states=3,
        max_route_states=4,
        max_memory_bytes=128,
    )
    deadline = _deadline()
    ledger = WheelSQPWorkLedgerV1(budget, deadline)

    ledger.charge_expansion()
    ledger.charge_memory(17)
    ledger.charge_route_states(2)

    assert ledger.resource_budget is budget
    assert ledger.deadline is deadline
    assert ledger.expanded_states == 1
    assert ledger.accounted_bytes == 17
    assert ledger.route_states == 2
    with pytest.raises(AttributeError):
        ledger.expanded_states = 0
    with pytest.raises(AttributeError):
        ledger.accounted_bytes = 0
    with pytest.raises(AttributeError):
        ledger.route_states = 0


@pytest.mark.parametrize(
    ("charge", "reason"),
    [
        (lambda ledger: ledger.charge_expansion(), "planning_deadline_expired"),
        (lambda ledger: ledger.charge_memory(1), "planning_deadline_expired"),
        (lambda ledger: ledger.charge_route_states(1), "planning_deadline_expired"),
    ],
)
def test_work_ledger_deadline_precedes_every_resource_limit(charge, reason) -> None:
    budget = ResourceBudgetV2(
        max_expanded_states=0,
        max_route_states=0,
        max_memory_bytes=1,
    )
    expired = PlanningDeadlineV2(0.0, 1.0, lambda: 1.0)
    ledger = WheelSQPWorkLedgerV1(budget, expired)

    with pytest.raises(WheelSQPWorkLimitError) as caught:
        charge(ledger)

    assert caught.value.reason_code == reason
    assert ledger.expanded_states == 0
    assert ledger.accounted_bytes == 0
    assert ledger.route_states == 0


@pytest.mark.parametrize(
    ("ledger", "charge", "reason"),
    [
        (
            WheelSQPWorkLedgerV1(ResourceBudgetV2(0, 1, 1), _deadline()),
            lambda item: item.charge_expansion(),
            "wheel_sqp_corridor_budget_exceeded",
        ),
        (
            WheelSQPWorkLedgerV1(ResourceBudgetV2(1, 1, 1), _deadline()),
            lambda item: item.charge_memory(2),
            "wheel_sqp_resource_budget_exceeded",
        ),
        (
            WheelSQPWorkLedgerV1(ResourceBudgetV2(1, 0, 1), _deadline()),
            lambda item: item.charge_route_states(1),
            "wheel_sqp_resource_budget_exceeded",
        ),
    ],
)
def test_work_ledger_raises_typed_stable_limit_reasons(ledger, charge, reason) -> None:
    with pytest.raises(WheelSQPWorkLimitError) as caught:
        charge(ledger)

    assert caught.value.reason_code == reason
