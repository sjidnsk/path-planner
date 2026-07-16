from dataclasses import FrozenInstanceError, replace
from math import nextafter, pi

import numpy as np
import pytest

import path_planner.v2 as v2
import path_planner.v2.validation as validation_module
from path_planner.core import Cell
from path_planner.search import MotionPrimitive, Pose2D, replay_motion_primitive
from path_planner.search.hybrid_astar import MAX_REPLAY_STEPS
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    RoutePrimitiveV2,
    TypedRouteV2,
    ValidationLevelV2,
)
from path_planner.v2.geometry import (
    conservative_wheel_pose_cells,
    conservative_wheel_sweep_cells,
)
from path_planner.v2.profiles import PlatformProfileV2, WheelProfileV2
from path_planner.v2.providers.wheel import WheelMotionPrimitiveV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)
from path_planner.v2.validation import (
    WHEEL_ROUTE_VALIDATOR_ID_V2,
    WHEEL_TRANSITION_VALIDATOR_ID_V2,
    WheelValidationResultV2,
    validate_route_l2,
    validate_wheel_transition_l2,
)


def _snapshot(
    *,
    width: int = 12,
    height: int = 6,
    default_slope: float = 0.0,
    slope_updates: tuple[tuple[Cell, float], ...] = (),
    unknown_cells: tuple[Cell, ...] = (),
    hard_cells: tuple[Cell, ...] = (),
    not_traversable_cells: tuple[Cell, ...] = (),
) -> TerrainSnapshotV2:
    shape = (height, width)
    slope = np.full(shape, default_slope, dtype=np.float64)
    traversable = np.ones(shape, dtype=bool)
    hard = np.zeros(shape, dtype=bool)
    observed = np.ones(shape, dtype=bool)
    for cell, value in slope_updates:
        slope[cell.y, cell.x] = value
    for cell in unknown_cells:
        observed[cell.y, cell.x] = False
    for cell in hard_cells:
        hard[cell.y, cell.x] = True
        traversable[cell.y, cell.x] = False
    for cell in not_traversable_cells:
        traversable[cell.y, cell.x] = False
    return TerrainSnapshotV2(
        geometry=FineGridGeometryV2(width=width, height=height, frame_id="moon"),
        elevation_m=np.zeros(shape),
        slope_deg=slope,
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=observed,
        confidence=np.ones(shape),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="route-validation-fixture",
            source_hash="route-validation-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )


def _wheel_profile(
    *,
    profile_id: str = "wheel-route/v1",
    duration_s: float = 1.0,
    integration_dt_s: float = 0.25,
    position_tolerance_m: float = 0.0,
    heading_tolerance_rad: float = 0.0,
    **overrides,
) -> WheelProfileV2:
    profile = PlatformProfileV2(
        profile_id=profile_id,
        platform_kind=PlatformKindV2.WHEEL,
        capability_revision="wheel-route-capability/v1",
        simulation_proxy=False,
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=position_tolerance_m,
        goal_heading_tolerance_rad=heading_tolerance_rad,
    )
    values = {
        "profile": profile,
        "primitive_duration_s": duration_s,
        "integration_dt_s": integration_dt_s,
    }
    values.update(overrides)
    return WheelProfileV2(**values)


def _typed_motion(
    profile: WheelProfileV2,
    start: Pose2D,
    control: MotionPrimitive,
    *,
    replay_dt_s: float | None = None,
    validation_level: ValidationLevelV2 = ValidationLevelV2.L0,
) -> tuple[WheelMotionPrimitiveV2, object]:
    replay = replay_motion_primitive(
        start,
        control,
        profile.integration_dt_s if replay_dt_s is None else replay_dt_s,
    )
    samples = tuple(
        PoseStateV2(sample.x_m, sample.y_m, sample.theta_rad)
        for sample in replay.samples
    )
    primitive = WheelMotionPrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=samples[0],
        end_state=samples[-1],
        duration_s=control.duration_s,
        distance_m=replay.distance_m,
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=validation_level,
        control_name=control.name,
        samples=samples,
        v_mps=control.v_mps,
        omega_radps=control.omega_radps,
        reverse=control.reverse,
        turn_in_place=control.turn_in_place,
    )
    return primitive, replay


def _hold(state: PoseStateV2) -> WheelMotionPrimitiveV2:
    return WheelMotionPrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=state,
        end_state=state,
        duration_s=0.0,
        distance_m=0.0,
        energy_cost=0.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        control_name="hold",
        samples=(state,),
        v_mps=0.0,
        omega_radps=0.0,
        reverse=False,
        turn_in_place=False,
    )


def _route(*primitives: WheelMotionPrimitiveV2, complete: bool = True) -> TypedRouteV2:
    return TypedRouteV2(
        platform_kind=PlatformKindV2.WHEEL,
        primitives=primitives,
        total_cost=0.0,
        is_complete=complete,
    )


def _request(
    snapshot: TerrainSnapshotV2,
    profile: WheelProfileV2,
    start: PoseStateV2,
    goal: PoseStateV2,
    *,
    max_route_states: int = 10_000,
    profile_id: str | None = None,
) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="route-validation-request",
        platform_profile_id=profile.profile.profile_id if profile_id is None else profile_id,
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(max_route_states=max_route_states),
        timeout_s=10.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=7,
    )


def _deadline(clock=None) -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, 1.0, (lambda: 0.0) if clock is None else clock)


def _straight_fixture(*, duration_s: float = 2.0):
    profile = _wheel_profile(duration_s=duration_s)
    start = Pose2D(1.25, 1.25, 0.0)
    control = MotionPrimitive("forward", 1.0, 0.0, duration_s)
    primitive, replay = _typed_motion(profile, start, control)
    return profile, start, control, primitive, replay


def test_wheel_validation_contract_is_public_frozen_and_slotted() -> None:
    assert v2.WheelValidationResultV2 is WheelValidationResultV2
    assert v2.validate_wheel_transition_l2 is validate_wheel_transition_l2
    assert v2.validate_route_l2 is validate_route_l2

    result = WheelValidationResultV2(
        evidence=v2.ValidationEvidenceV2(
            validator_id="path-planner-v2-wheel-route-l2/v1",
            level=v2.ValidationLevelV2.L2,
            passed=True,
            checks=("route_l2_valid",),
        ),
        reason_code="route_l2_valid",
        timed_out=False,
        failed_cell=None,
        failed_primitive_index=None,
        checked_cell_count=1,
    )

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.reason_code = "tampered"


def test_wheel_validation_result_rejects_forged_evidence_or_checks() -> None:
    def evidence(*, passed: bool, checks: tuple[str, ...]):
        return v2.ValidationEvidenceV2(
            validator_id="path-planner-v2-wheel-route-l2/v1",
            level=v2.ValidationLevelV2.L2,
            passed=passed,
            checks=checks,
        )

    with pytest.raises(ValueError, match="passed.*reason_code"):
        WheelValidationResultV2(
            evidence=evidence(passed=True, checks=("terrain_unknown",)),
            reason_code="terrain_unknown",
            timed_out=False,
            failed_cell=None,
            failed_primitive_index=0,
            checked_cell_count=1,
        )
    with pytest.raises(ValueError, match="passed.*reason_code"):
        WheelValidationResultV2(
            evidence=evidence(passed=False, checks=("route_l2_valid",)),
            reason_code="route_l2_valid",
            timed_out=False,
            failed_cell=None,
            failed_primitive_index=None,
            checked_cell_count=1,
        )
    with pytest.raises(ValueError, match="checks"):
        WheelValidationResultV2(
            evidence=evidence(passed=False, checks=("other",)),
            reason_code="terrain_unknown",
            timed_out=False,
            failed_cell=None,
            failed_primitive_index=0,
            checked_cell_count=1,
        )


@pytest.mark.parametrize(
    ("validator_id", "reason_code", "passed", "failed_cell", "failed_index"),
    [
        ("unknown-validator/v1", "terrain_unknown", False, None, 0),
        (
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            "transition_l2_valid",
            True,
            None,
            None,
        ),
        (
            WHEEL_TRANSITION_VALIDATOR_ID_V2,
            "route_l2_valid",
            True,
            None,
            None,
        ),
        (WHEEL_ROUTE_VALIDATOR_ID_V2, "unknown_reason", False, None, 0),
        (
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            "planning_deadline_expired",
            False,
            Cell(0, 0),
            0,
        ),
        (
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            "terrain_unknown",
            False,
            Cell(0, 0),
            None,
        ),
    ],
)
def test_wheel_validation_result_rejects_invalid_validator_reason_or_metadata(
    validator_id,
    reason_code,
    passed,
    failed_cell,
    failed_index,
) -> None:
    evidence = v2.ValidationEvidenceV2(
        validator_id=validator_id,
        level=ValidationLevelV2.L2,
        passed=passed,
        checks=(reason_code,),
    )

    with pytest.raises(ValueError):
        WheelValidationResultV2(
            evidence=evidence,
            reason_code=reason_code,
            timed_out=reason_code == "planning_deadline_expired",
            failed_cell=failed_cell,
            failed_primitive_index=failed_index,
            checked_cell_count=1,
        )


@pytest.mark.parametrize("entry_point", ["transition", "route"])
def test_entry_points_reaudit_tampered_wheel_profile_without_overflow_leak(
    entry_point,
) -> None:
    profile, _, _, primitive, transition = _straight_fixture(duration_s=1.0)
    object.__setattr__(profile, "integration_dt_s", 10**400)
    snapshot = _snapshot()
    anchor = FineSafetyAnchorV2(snapshot)

    if entry_point == "transition":
        result = validate_wheel_transition_l2(
            transition,
            anchor,
            profile,
            _deadline(),
        )
    else:
        result = validate_route_l2(
            _route(primitive),
            _request(snapshot, profile, primitive.start_state, primitive.end_state),
            anchor,
            profile,
            _deadline(),
        )

    assert result.reason_code == "wheel_profile_contract_mismatch"
    assert result.evidence.passed is False
    assert result.checked_cell_count == 0


def test_route_reaudits_huge_request_goal_before_tolerance_arithmetic() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    request = _request(
        snapshot,
        profile,
        primitive.start_state,
        primitive.end_state,
    )
    object.__setattr__(request.goal_state, "x_m", 10**400)

    result = validate_route_l2(
        _route(primitive),
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "route_goal_contract_mismatch"
    assert result.evidence.passed is False
    assert result.checked_cell_count > 0


def test_route_reaudits_huge_request_start_before_route_comparison() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    request_start = PoseStateV2(
        primitive.start_state.x_m,
        primitive.start_state.y_m,
        primitive.start_state.heading_rad,
    )
    request = _request(
        snapshot,
        profile,
        request_start,
        primitive.end_state,
    )
    object.__setattr__(request.start_state, "x_m", 10**400)

    result = validate_route_l2(
        _route(primitive),
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "route_start_contract_mismatch"
    assert result.evidence.passed is False
    assert result.checked_cell_count > 0


@pytest.mark.parametrize("tamper_kind", ["budget_type", "max_type", "max_negative"])
def test_route_reaudits_request_route_state_budget_before_validation(
    tamper_kind,
) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    request = _request(
        snapshot,
        profile,
        primitive.start_state,
        primitive.end_state,
    )
    if tamper_kind == "budget_type":
        object.__setattr__(request, "resource_budget", object())
    elif tamper_kind == "max_type":
        object.__setattr__(request.resource_budget, "max_route_states", True)
    else:
        object.__setattr__(request.resource_budget, "max_route_states", -1)

    result = validate_route_l2(
        _route(primitive),
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "planning_request_contract_mismatch"
    assert result.evidence.passed is False
    assert result.checked_cell_count == 0


def test_terrain_failure_beats_huge_request_goal_contract_mismatch() -> None:
    profile, start, control, primitive, _ = _straight_fixture(duration_s=1.0)
    base = _snapshot()
    contacted = conservative_wheel_sweep_cells(
        start,
        control,
        base.geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    blocked = contacted[len(contacted) // 2]
    snapshot = _snapshot(hard_cells=(blocked,))
    request = _request(
        snapshot,
        profile,
        primitive.start_state,
        primitive.end_state,
    )
    object.__setattr__(request.goal_state, "x_m", 10**400)

    result = validate_route_l2(
        _route(primitive),
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "terrain_hard_obstacle"
    assert result.failed_cell == blocked
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == len(contacted)


@pytest.mark.parametrize("tamper_kind", ["cell", "level", "passed_reason"])
def test_route_rejects_tampered_anchor_query_contract(
    monkeypatch,
    tamper_kind,
) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    anchor = FineSafetyAnchorV2(snapshot)
    original_query = FineSafetyAnchorV2.query
    requested_cells: list[Cell] = []

    def tampered_query(self, cell, max_slope_deg=30.0):
        requested_cells.append(cell)
        query = original_query(self, cell, max_slope_deg)
        if tamper_kind == "cell":
            object.__setattr__(query, "cell", Cell(cell.x + 1, cell.y))
        elif tamper_kind == "level":
            object.__setattr__(query, "validation_level", ValidationLevelV2.L1)
        else:
            object.__setattr__(query, "passed", False)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", tampered_query)
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        anchor,
        profile,
        _deadline(),
    )

    assert result.reason_code == "terrain_query_contract_mismatch"
    assert result.checked_cell_count == 1
    assert result.failed_cell is None
    assert result.failed_primitive_index == 0


def test_transition_l2_is_public_and_queries_every_swept_cell_at_exact_30(
    monkeypatch,
) -> None:
    profile, start, control, _, transition = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot(default_slope=30.0)
    anchor = FineSafetyAnchorV2(snapshot)
    thresholds: list[float] = []
    original_query = FineSafetyAnchorV2.query

    def recording_query(self, cell, max_slope_deg=30.0):
        thresholds.append(max_slope_deg)
        return original_query(self, cell, max_slope_deg)

    monkeypatch.setattr(FineSafetyAnchorV2, "query", recording_query)
    result = validate_wheel_transition_l2(transition, anchor, profile, _deadline())
    expected_cells = conservative_wheel_sweep_cells(
        start,
        control,
        snapshot.geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )

    assert result.evidence.passed is True
    assert result.evidence.level is ValidationLevelV2.L2
    assert result.reason_code == "transition_l2_valid"
    assert result.checked_cell_count == len(expected_cells)
    assert thresholds == [30.0] * len(expected_cells)


def test_transition_raw_start_tampering_fails_after_safe_independent_replay() -> None:
    profile, _, _, _, transition = _straight_fixture(duration_s=1.0)
    object.__setattr__(
        transition,
        "start",
        Pose2D(nextafter(transition.start.x_m, float("inf")), 1.25, 0.0),
    )
    snapshot = _snapshot()

    result = validate_wheel_transition_l2(
        transition,
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_replay_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count > 0


def test_deadline_during_public_transition_replay_is_timeout_not_validation_failure() -> None:
    profile, _, _, _, transition = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()

    class ReplayClock:
        calls = 0

        def __call__(self):
            self.calls += 1
            return 0.0 if self.calls < 3 else 1.0

    result = validate_wheel_transition_l2(
        transition,
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(ReplayClock()),
    )

    assert result.timed_out is True
    assert result.reason_code == "planning_deadline_expired"
    assert result.evidence.passed is False
    assert result.checked_cell_count == 0


def test_nonhold_zero_motion_transition_is_rejected_structurally() -> None:
    profile = _wheel_profile()
    start = Pose2D(1.25, 1.25, 0.0)
    transition = replay_motion_primitive(
        start,
        MotionPrimitive("idle", 0.0, 0.0, 1.0),
        profile.integration_dt_s,
    )
    snapshot = _snapshot()

    result = validate_wheel_transition_l2(
        transition,
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_zero_motion_mismatch"
    assert result.evidence.passed is False
    assert result.checked_cell_count > 0


def test_route_rejects_blocked_intermediate_sweep_even_when_endpoints_are_safe() -> None:
    profile, start, control, primitive, replay = _straight_fixture()
    geometry = _snapshot().geometry
    blocked = Cell(4, 2)
    assert blocked in conservative_wheel_sweep_cells(
        start,
        control,
        geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    assert blocked not in conservative_wheel_pose_cells(
        start,
        geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    assert blocked not in conservative_wheel_pose_cells(
        replay.end,
        geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    snapshot = _snapshot(hard_cells=(blocked,))
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.evidence.passed is False
    assert result.reason_code == "terrain_hard_obstacle"
    assert result.failed_cell == blocked
    assert result.failed_primitive_index == 0


def test_unknown_precedes_hard_obstacle_in_the_same_contacted_cell() -> None:
    profile, start, control, primitive, _ = _straight_fixture()
    cell = Cell(4, 2)
    snapshot = _snapshot(unknown_cells=(cell,), hard_cells=(cell,))
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    expected_count = len(
        conservative_wheel_sweep_cells(
            start,
            control,
            snapshot.geometry,
            body_length_m=profile.body_length_m,
            body_width_m=profile.body_width_m,
            safety_margin_m=profile.footprint_safety_margin_m,
        )
    )

    assert result.reason_code == "terrain_unknown"
    assert result.failed_cell == cell
    assert result.checked_cell_count == expected_count


def test_slope_closed_boundary_passes_and_nextafter_above_fails() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    all_equal = _snapshot(default_slope=30.0)
    equal_result = validate_route_l2(
        _route(primitive),
        _request(all_equal, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(all_equal),
        profile,
        _deadline(),
    )
    contacted = conservative_wheel_sweep_cells(
        Pose2D(
            primitive.start_state.x_m,
            primitive.start_state.y_m,
            primitive.start_state.heading_rad,
        ),
        MotionPrimitive(
            primitive.control_name,
            primitive.v_mps,
            primitive.omega_radps,
            primitive.duration_s,
            reverse=primitive.reverse,
            turn_in_place=primitive.turn_in_place,
        ),
        all_equal.geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    failed_cell = contacted[len(contacted) // 2]
    above = _snapshot(
        default_slope=30.0,
        slope_updates=((failed_cell, nextafter(30.0, float("inf"))),),
    )
    above_result = validate_route_l2(
        _route(primitive),
        _request(above, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(above),
        profile,
        _deadline(),
    )

    assert equal_result.evidence.passed is True
    assert above_result.reason_code == "terrain_slope_exceeded"
    assert above_result.failed_cell == failed_cell


def test_replay_tampering_fails_structurally_but_still_scans_safe_terrain() -> None:
    profile, start, control, primitive, _ = _straight_fixture(duration_s=1.0)
    tampered = replace(
        primitive.samples[1],
        x_m=nextafter(primitive.samples[1].x_m, float("inf")),
    )
    object.__setattr__(
        primitive,
        "samples",
        (primitive.samples[0], tampered, *primitive.samples[2:]),
    )
    snapshot = _snapshot()
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    expected_count = len(
        conservative_wheel_sweep_cells(
            start,
            control,
            snapshot.geometry,
            body_length_m=profile.body_length_m,
            body_width_m=profile.body_width_m,
            safety_margin_m=profile.footprint_safety_margin_m,
        )
    )

    assert result.reason_code == "primitive_replay_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == expected_count


def test_snapshot_identity_mismatch_fails_closed_without_anchor_query(monkeypatch) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    request_snapshot = _snapshot()
    anchor = FineSafetyAnchorV2(_snapshot())

    def forbidden_query(*_args, **_kwargs):
        raise AssertionError("mismatched anchor must never be queried")

    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden_query)
    result = validate_route_l2(
        _route(primitive),
        _request(
            request_snapshot,
            profile,
            primitive.start_state,
            primitive.end_state,
        ),
        anchor,
        profile,
        _deadline(),
    )

    assert result.reason_code == "terrain_snapshot_identity_mismatch"
    assert result.checked_cell_count == 0


def test_public_query_snapshot_hash_mismatch_fails_closed() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    anchor = FineSafetyAnchorV2(snapshot)
    object.__setattr__(anchor, "_snapshot_hash", "0" * 64)

    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        anchor,
        profile,
        _deadline(),
    )

    assert result.reason_code == "terrain_snapshot_hash_mismatch"
    assert result.checked_cell_count == 1


@pytest.mark.parametrize(
    ("profile", "control", "replay_dt_s", "reason_code"),
    [
        (
            _wheel_profile(reverse_enabled=False),
            MotionPrimitive("reverse", -0.5, 0.0, 1.0, reverse=True),
            None,
            "wheel_reverse_disabled",
        ),
        (
            _wheel_profile(turn_in_place_enabled=False),
            MotionPrimitive("turn", 0.0, 0.5, 1.0, turn_in_place=True),
            None,
            "wheel_turn_in_place_disabled",
        ),
        (
            _wheel_profile(max_speed_mps=1.0),
            MotionPrimitive("too_fast", nextafter(1.0, float("inf")), 0.0, 1.0),
            None,
            "wheel_speed_limit_exceeded",
        ),
        (
            _wheel_profile(min_turning_radius_m=1.0, max_angular_speed_radps=1.0),
            MotionPrimitive("tight", 0.5, 1.0, 1.0),
            None,
            "wheel_min_turning_radius_violated",
        ),
        (
            _wheel_profile(integration_dt_s=0.25),
            MotionPrimitive("dt_mismatch", 0.5, 0.25, 1.0),
            0.5,
            "primitive_replay_mismatch",
        ),
        (
            _wheel_profile(max_angular_speed_radps=0.5),
            MotionPrimitive(
                "too_angular",
                0.5,
                nextafter(0.5, float("inf")),
                1.0,
            ),
            None,
            "wheel_angular_speed_limit_exceeded",
        ),
        (
            _wheel_profile(duration_s=1.0),
            MotionPrimitive("wrong_duration", 0.5, 0.0, 0.5),
            None,
            "primitive_duration_mismatch",
        ),
    ],
)
def test_profile_replay_capability_speed_and_radius_mismatches_fail_structurally(
    profile,
    control,
    replay_dt_s,
    reason_code,
) -> None:
    primitive, _ = _typed_motion(
        profile,
        Pose2D(1.25, 1.25, 0.0),
        control,
        replay_dt_s=replay_dt_s,
    )
    snapshot = _snapshot()
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == reason_code
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count > 0


def test_route_structure_profile_start_connectivity_and_resource_fail_closed() -> None:
    profile, _, _, first, first_replay = _straight_fixture(duration_s=1.0)
    second, _ = _typed_motion(
        profile,
        first_replay.end,
        MotionPrimitive("forward", 1.0, 0.0, 1.0),
    )
    snapshot = _snapshot()
    connected = _route(first, second)

    incomplete = replace(connected, is_complete=False)
    incomplete_result = validate_route_l2(
        incomplete,
        _request(snapshot, profile, first.start_state, second.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    profile_result = validate_route_l2(
        connected,
        _request(
            snapshot,
            profile,
            first.start_state,
            second.end_state,
            profile_id="other/v1",
        ),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    start_result = validate_route_l2(
        connected,
        _request(
            snapshot,
            profile,
            PoseStateV2(9.0, 9.0, 0.0),
            second.end_state,
        ),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    object.__setattr__(second, "start_state", PoseStateV2(2.5, 1.25, 0.0))
    disconnected_result = validate_route_l2(
        connected,
        _request(snapshot, profile, first.start_state, second.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    resource_result = validate_route_l2(
        connected,
        _request(
            snapshot,
            profile,
            first.start_state,
            second.end_state,
            max_route_states=1,
        ),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert incomplete_result.reason_code == "route_incomplete"
    assert profile_result.reason_code == "wheel_profile_identity_mismatch"
    assert start_result.reason_code == "route_start_mismatch"
    assert disconnected_result.reason_code == "route_connectivity_mismatch"
    assert resource_result.reason_code == "route_state_budget_exceeded"
    assert resource_result.checked_cell_count == 0


def test_exact_wheel_platform_and_primitive_type_are_reaudited() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    route = _route(primitive)
    object.__setattr__(route, "platform_kind", PlatformKindV2.LEGGED)
    platform_result = validate_route_l2(
        route,
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    base_primitive = RoutePrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=primitive.start_state,
        end_state=primitive.end_state,
        duration_s=primitive.duration_s,
        distance_m=primitive.distance_m,
        energy_cost=primitive.energy_cost,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
    )
    object.__setattr__(route, "platform_kind", PlatformKindV2.WHEEL)
    object.__setattr__(route, "primitives", (base_primitive,))
    primitive_result = validate_route_l2(
        route,
        _request(
            snapshot,
            profile,
            base_primitive.start_state,
            base_primitive.end_state,
        ),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert platform_result.reason_code == "wheel_platform_identity_mismatch"
    assert platform_result.checked_cell_count > 0
    assert primitive_result.reason_code == "wheel_primitive_type_mismatch"
    assert primitive_result.checked_cell_count == 0


def test_hold_is_independently_swept_and_route_state_budget_counts_actual_samples() -> None:
    profile = _wheel_profile()
    state = PoseStateV2(1.25, 1.25, 0.0)
    hold = _hold(state)
    base = _snapshot()
    contacted = conservative_wheel_pose_cells(
        Pose2D(state.x_m, state.y_m, state.heading_rad),
        base.geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    blocked = contacted[len(contacted) // 2]
    snapshot = _snapshot(hard_cells=(blocked,))
    terrain_result = validate_route_l2(
        _route(hold),
        _request(snapshot, profile, state, state, max_route_states=1),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    resource_result = validate_route_l2(
        _route(hold),
        _request(snapshot, profile, state, state, max_route_states=0),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert terrain_result.reason_code == "terrain_hard_obstacle"
    assert terrain_result.checked_cell_count == len(contacted)
    assert resource_result.reason_code == "route_state_budget_exceeded"
    assert resource_result.checked_cell_count == 0


def test_tampered_hold_numeric_field_fails_stably_and_still_scans_terrain() -> None:
    profile = _wheel_profile()
    state = PoseStateV2(1.25, 1.25, 0.0)
    hold = _hold(state)
    object.__setattr__(hold, "observation_contribution", "bad")
    snapshot = _snapshot()

    result = validate_route_l2(
        _route(hold),
        _request(snapshot, profile, state, state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_hold_contract_mismatch"
    assert result.evidence.passed is False
    assert result.checked_cell_count > 0


def test_route_state_budget_uses_larger_of_declared_and_replayed_samples() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    route = _route(primitive)
    object.__setattr__(primitive, "samples", (primitive.start_state,) * 20)
    snapshot = _snapshot()

    result = validate_route_l2(
        route,
        _request(
            snapshot,
            profile,
            primitive.start_state,
            primitive.end_state,
            max_route_states=10,
        ),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "route_state_budget_exceeded"
    assert result.checked_cell_count == 0


def test_route_replay_contract_error_is_stable_and_still_scans_when_safe() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    object.__setattr__(
        profile,
        "integration_dt_s",
        nextafter(0.0, float("inf")),
    )
    snapshot = _snapshot()

    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_replay_mismatch"
    assert result.checked_cell_count > 0


def test_deadline_expiring_after_snapshot_hash_preempts_empty_validation_path() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()

    class HashClock:
        calls = 0

        def __call__(self):
            self.calls += 1
            return 0.0 if self.calls == 1 else 1.0

    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(HashClock()),
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.timed_out is True
    assert result.checked_cell_count == 0


def test_wrap_safe_goal_tolerance_equality_passes_and_nextafter_outside_fails() -> None:
    tolerance = 0.2
    profile = _wheel_profile(
        position_tolerance_m=0.125,
        heading_tolerance_rad=tolerance,
    )
    snapshot = _snapshot()
    boundary_state = PoseStateV2(1.375, 1.25, tolerance)
    boundary_route = _route(_hold(boundary_state))
    goal = PoseStateV2(1.25, 1.25, 0.0)
    boundary_result = validate_route_l2(
        boundary_route,
        _request(snapshot, profile, boundary_state, goal),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    outside_state = PoseStateV2(
        1.375,
        1.25,
        nextafter(tolerance, float("inf")),
    )
    outside_route = _route(_hold(outside_state))
    original_end = outside_route.primitives[-1].end_state
    outside_result = validate_route_l2(
        outside_route,
        _request(snapshot, profile, outside_state, goal),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert boundary_result.evidence.passed is True
    assert boundary_result.reason_code == "route_l2_valid"
    assert outside_result.reason_code == "route_goal_tolerance_exceeded"
    assert outside_route.primitives[-1].end_state is original_end


def test_global_priority_later_oob_beats_earlier_slope_with_stable_metadata() -> None:
    profile = _wheel_profile(duration_s=1.0)
    first, first_replay = _typed_motion(
        profile,
        Pose2D(0.75, 1.25, 0.0),
        MotionPrimitive("first", 1.0, 0.0, 1.0),
    )
    second, _ = _typed_motion(
        profile,
        first_replay.end,
        MotionPrimitive("second", 1.0, 0.0, 1.0),
    )
    base = _snapshot(width=6)
    first_cells = conservative_wheel_sweep_cells(
        Pose2D(0.75, 1.25, 0.0),
        MotionPrimitive("first", 1.0, 0.0, 1.0),
        base.geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    slope_cell = next(cell for cell in first_cells if base.geometry.in_bounds(cell))
    snapshot = _snapshot(
        width=6,
        slope_updates=((slope_cell, nextafter(30.0, float("inf"))),),
    )
    second_cells = conservative_wheel_sweep_cells(
        first_replay.end,
        MotionPrimitive("second", 1.0, 0.0, 1.0),
        snapshot.geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    expected_oob = next(cell for cell in second_cells if not snapshot.geometry.in_bounds(cell))
    result = validate_route_l2(
        _route(first, second),
        _request(snapshot, profile, first.start_state, second.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "terrain_out_of_bounds"
    assert result.failed_primitive_index == 1
    assert result.failed_cell == expected_oob
    assert result.checked_cell_count == len(first_cells) + len(second_cells)


def test_deadline_during_query_returns_timeout_without_passing_evidence(monkeypatch) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    anchor = FineSafetyAnchorV2(snapshot)

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    original_query = FineSafetyAnchorV2.query

    def expire_after_query(self, cell, max_slope_deg=30.0):
        query = original_query(self, cell, max_slope_deg)
        clock.now = 1.0
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", expire_after_query)
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        anchor,
        profile,
        _deadline(clock),
    )

    assert result.timed_out is True
    assert result.reason_code == "planning_deadline_expired"
    assert result.evidence.passed is False
    assert result.checked_cell_count == 1


@pytest.mark.parametrize("hard_obstacle", [False, True])
def test_extreme_finite_goal_headings_are_reduced_before_subtraction_and_keep_terrain_priority(
    hard_obstacle,
) -> None:
    max_finite = float.fromhex("0x1.fffffffffffffp+1023")
    profile = _wheel_profile(heading_tolerance_rad=pi)
    state = PoseStateV2(1.25, 1.25, max_finite)
    goal = PoseStateV2(1.25, 1.25, -max_finite)
    hold = _hold(state)
    base = _snapshot()
    contacted = conservative_wheel_pose_cells(
        Pose2D(state.x_m, state.y_m, state.heading_rad),
        base.geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    snapshot = _snapshot(hard_cells=(contacted[0],) if hard_obstacle else ())

    result = validate_route_l2(
        _route(hold),
        _request(snapshot, profile, state, goal),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == (
        "terrain_hard_obstacle" if hard_obstacle else "route_l2_valid"
    )
    assert result.evidence.passed is (not hard_obstacle)


class _AttributeErrorClock:
    def __call__(self):
        raise AttributeError("tampered clock")


@pytest.mark.parametrize("entry_point", ["route", "transition"])
@pytest.mark.parametrize(
    "clock",
    [
        lambda: True,
        lambda: float("nan"),
        lambda: float("inf"),
        lambda: 10**400,
        _AttributeErrorClock(),
    ],
    ids=["bool", "nan", "inf", "huge", "attribute-error"],
)
def test_tampered_exact_deadline_clock_fails_with_stable_contract_reason(
    entry_point,
    clock,
) -> None:
    profile, _, _, primitive, transition = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    deadline = _deadline()
    object.__setattr__(deadline, "_monotonic_clock", clock)

    if entry_point == "route":
        result = validate_route_l2(
            _route(primitive),
            _request(snapshot, profile, primitive.start_state, primitive.end_state),
            FineSafetyAnchorV2(snapshot),
            profile,
            deadline,
        )
    else:
        result = validate_wheel_transition_l2(
            transition,
            FineSafetyAnchorV2(snapshot),
            profile,
            deadline,
        )

    assert result.reason_code == "planning_deadline_contract_mismatch"
    assert result.timed_out is False
    assert result.checked_cell_count == 0


@pytest.mark.parametrize("entry_point", ["route", "transition"])
@pytest.mark.parametrize("tamper_kind", ["geometry", "layer-shape", "confidence-range"])
def test_tampered_exact_snapshot_fails_closed_without_hash_or_index_exception(
    entry_point,
    tamper_kind,
) -> None:
    profile, _, _, primitive, transition = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    anchor = FineSafetyAnchorV2(snapshot)
    request = _request(snapshot, profile, primitive.start_state, primitive.end_state)
    if tamper_kind == "geometry":
        object.__setattr__(snapshot, "geometry", object())
    elif tamper_kind == "layer-shape":
        object.__setattr__(snapshot, "slope_deg", np.zeros((1, 1), dtype=np.float64))
    else:
        confidence = np.full(snapshot.confidence.shape, 2.0, dtype=np.float64)
        confidence.setflags(write=False)
        object.__setattr__(snapshot, "confidence", confidence)

    if entry_point == "route":
        result = validate_route_l2(
            _route(primitive), request, anchor, profile, _deadline()
        )
    else:
        result = validate_wheel_transition_l2(
            transition, anchor, profile, _deadline()
        )

    assert result.reason_code == "terrain_snapshot_hash_mismatch"
    assert result.checked_cell_count == 0


@pytest.mark.parametrize(
    "error_type",
    [TypeError, ValueError, OverflowError, AttributeError, IndexError],
)
def test_anchor_query_expected_exceptions_become_stable_contract_failure(
    monkeypatch,
    error_type,
) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()

    def broken_query(*_args, **_kwargs):
        raise error_type("tampered query")

    monkeypatch.setattr(FineSafetyAnchorV2, "query", broken_query)
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "terrain_query_contract_mismatch"
    assert result.checked_cell_count == 1
    assert result.failed_cell is None
    assert result.failed_primitive_index == 0


def test_snapshot_mutation_during_query_is_detected_after_cell_scan(monkeypatch) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    anchor = FineSafetyAnchorV2(snapshot)
    original_query = FineSafetyAnchorV2.query
    calls = 0

    def mutate_after_query(self, cell, max_slope_deg=30.0):
        nonlocal calls
        query = original_query(self, cell, max_slope_deg)
        calls += 1
        if calls == 1:
            elevation = np.ones(snapshot.elevation_m.shape, dtype=np.float64)
            elevation.setflags(write=False)
            object.__setattr__(snapshot, "elevation_m", elevation)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", mutate_after_query)
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        anchor,
        profile,
        _deadline(),
    )

    assert result.reason_code == "terrain_snapshot_hash_mismatch"
    assert result.checked_cell_count > 0
    assert result.failed_cell is None


def test_snapshot_provenance_scan_observes_deadline_without_rebuilding(
    monkeypatch,
) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    anchor = FineSafetyAnchorV2(snapshot)
    object.__setattr__(
        snapshot.provenance,
        "details",
        (("a", 1), ("b", 2), ("c", 3)),
    )

    class DetailClock:
        calls = 0

        def __call__(self):
            self.calls += 1
            return 0.0 if self.calls < 4 else 1.0

    original_replace = validation_module.replace

    def forbid_provenance_replace(value, *args, **kwargs):
        if type(value) is TerrainProvenanceV2:
            raise AssertionError("provenance must be reaudited without rebuilding")
        return original_replace(value, *args, **kwargs)

    monkeypatch.setattr(validation_module, "replace", forbid_provenance_replace)
    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        anchor,
        profile,
        _deadline(DetailClock()),
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.checked_cell_count == 0


class _ExplodingEquality:
    def __eq__(self, _other):
        raise AssertionError("untrusted equality executed")

    def __ne__(self, _other):
        raise AssertionError("untrusted inequality executed")


def test_route_prescan_never_executes_untrusted_state_equality() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    request = _request(snapshot, profile, primitive.start_state, primitive.end_state)
    object.__setattr__(primitive, "start_state", _ExplodingEquality())

    result = validate_route_l2(
        _route(primitive),
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_structure_mismatch"
    assert result.failed_primitive_index == 0


def test_declared_sample_budget_is_rejected_before_replay(monkeypatch) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    object.__setattr__(primitive, "samples", (primitive.start_state,) * 2)
    snapshot = _snapshot()

    def forbidden_replay(*_args, **_kwargs):
        raise AssertionError("replay must not run after declared budget is exhausted")

    monkeypatch.setattr(validation_module, "replay_motion_primitive", forbidden_replay)
    result = validate_route_l2(
        _route(primitive),
        _request(
            snapshot,
            profile,
            primitive.start_state,
            primitive.end_state,
            max_route_states=1,
        ),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "route_state_budget_exceeded"
    assert result.checked_cell_count == 0


def test_declared_samples_above_public_replay_cap_fail_as_resource_error() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    object.__setattr__(
        primitive,
        "samples",
        (primitive.start_state,) * (MAX_REPLAY_STEPS + 2),
    )
    snapshot = _snapshot()

    result = validate_route_l2(
        _route(primitive),
        _request(
            snapshot,
            profile,
            primitive.start_state,
            primitive.end_state,
            max_route_states=MAX_REPLAY_STEPS + 2,
        ),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "route_state_budget_exceeded"
    assert result.checked_cell_count == 0


@pytest.mark.parametrize(
    ("reason_code", "failed_cell", "failed_index", "checked_count"),
    [
        ("terrain_unknown", None, 0, 1),
        ("terrain_unknown", Cell(0, 0), 0, 0),
        ("terrain_unknown", Cell(True, 0), 0, 1),
        ("primitive_structure_mismatch", Cell(0, 0), 0, 1),
    ],
)
def test_validation_result_rejects_inconsistent_terrain_failure_metadata(
    reason_code,
    failed_cell,
    failed_index,
    checked_count,
) -> None:
    evidence = v2.ValidationEvidenceV2(
        validator_id=WHEEL_ROUTE_VALIDATOR_ID_V2,
        level=ValidationLevelV2.L2,
        passed=False,
        checks=(reason_code,),
    )

    with pytest.raises(ValueError):
        WheelValidationResultV2(
            evidence=evidence,
            reason_code=reason_code,
            timed_out=False,
            failed_cell=failed_cell,
            failed_primitive_index=failed_index,
            checked_cell_count=checked_count,
        )


class _EvilTuple(tuple):
    def __len__(self):
        raise RuntimeError("untrusted tuple length executed")

    def __iter__(self):
        raise RuntimeError("untrusted tuple iteration executed")

    def __eq__(self, _other):
        raise RuntimeError("untrusted tuple equality executed")

    def __ne__(self, _other):
        raise RuntimeError("untrusted tuple inequality executed")


def test_nonexact_samples_tuple_never_executes_untrusted_tuple_protocol() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    object.__setattr__(primitive, "samples", _EvilTuple(primitive.samples))

    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, primitive.start_state, primitive.end_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_replay_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count > 0


def test_nonhold_missing_start_coordinate_fails_without_attribute_leak() -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    request_start = replace(primitive.start_state)
    request_goal = replace(primitive.end_state)
    object.__delattr__(primitive.start_state, "x_m")

    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, request_start, request_goal),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_structure_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0


def test_hold_missing_start_coordinate_fails_without_attribute_leak() -> None:
    profile = _wheel_profile()
    state = PoseStateV2(1.25, 1.25, 0.0)
    request_state = replace(state)
    hold = _hold(state)
    snapshot = _snapshot()
    object.__delattr__(hold.start_state, "x_m")

    result = validate_route_l2(
        _route(hold),
        _request(snapshot, profile, request_state, request_state),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_hold_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0


@pytest.mark.parametrize("field_name", ["samples", "control_name"])
def test_missing_nonhold_field_is_sanitized_before_primitive_audit(field_name) -> None:
    profile, _, _, primitive, _ = _straight_fixture(duration_s=1.0)
    snapshot = _snapshot()
    request_start = replace(primitive.start_state)
    request_goal = replace(primitive.end_state)
    object.__delattr__(primitive, field_name)

    result = validate_route_l2(
        _route(primitive),
        _request(snapshot, profile, request_start, request_goal),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    assert result.reason_code == "primitive_structure_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count > 0
