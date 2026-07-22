from dataclasses import replace

import pytest

import path_planner.v2 as v2
from path_planner.v2.contracts import (
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    SearchTelemetryV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.serialization import canonical_json_bytes
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
    WheelTrajectoryL2ResultV2,
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
