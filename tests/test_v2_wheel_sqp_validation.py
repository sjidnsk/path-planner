from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json
from math import cos, hypot, inf, nextafter, pi, sin
from types import SimpleNamespace

import numpy as np
import pytest

import path_planner.v2.wheel_sqp_validation as wheel_sqp_validation_module
from path_planner.core import Cell
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.profiles import PlatformProfileV2, WheelKinematicSQPProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
    snapshot_hash,
)
from path_planner.v2.wheel_kinematics import (
    integrate_wheel_segment_v2,
    wheel_pose_at_elapsed_v2,
)
from path_planner.v2.wheel_sqp_contracts import (
    L2ReserveModelV1,
    WheelSQPResourceLedgerV1,
    WheelSQPWorkLimitError,
    WheelSQPWorkLedgerV1,
    WheelTrajectoryL2ResultV2,
    _make_wheel_sqp_resource_estimate_v1,
    _make_wheel_trajectory_l2_result_v2,
)
from path_planner.v2.wheel_sqp_serialization import (
    canonicalize_unwrapped_pose_v2,
    decode_wheel_candidate_v2,
    encode_wheel_candidate_v2,
    encode_wheel_route_v2,
    materialize_canonical_wheel_candidate_v2,
    project_wheel_route_to_candidate_v1,
)
from path_planner.v2.wheel_sqp_validation import (
    oriented_rectangle_cell_separation_v2,
    promote_wheel_candidate_to_route_v2,
    prove_wheel_segment_sweep_v2,
    validate_wheel_sqp_candidate_l2,
)


PROFILE = WheelKinematicSQPProfileV2(
    profile=PlatformProfileV2(
        profile_id="scout-mini-wheel-kinematic-sqp/v1",
        platform_kind=PlatformKindV2.WHEEL,
        capability_revision="wheel_kinematic_corridor_sqp/v1",
        simulation_proxy=False,
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=0.25,
        goal_heading_tolerance_rad=0.08726646259971647,
    )
)


def _snapshot() -> TerrainSnapshotV2:
    geometry = FineGridGeometryV2(8, 8, origin=(-2.0, -2.0))
    return TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=np.ones(geometry.shape, dtype=np.bool_),
        hard_obstacle_mask=np.zeros(geometry.shape, dtype=np.bool_),
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task7-fixture",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )


SNAPSHOT = _snapshot()
REQUEST = PlanningRequestV2(
    request_id="wheel-sqp-task7",
    platform_profile_id=PROFILE.profile.profile_id,
    start_state=PoseStateV2(0.0, 0.0, 0.0),
    goal_state=PoseStateV2(1.0, 0.0, 0.0),
    terrain_snapshot=SNAPSHOT,
    objective_profile=ObjectiveProfileV2(),
    resource_budget=ResourceBudgetV2(100_000, 100_000, 0),
    timeout_s=2.0,
    accelerator_policy=AcceleratorPolicyV2.DISABLED,
    determinism_seed=31,
)


def test_closed_time_pose_helper_preserves_both_exact_segment_endpoints() -> None:
    start = PoseStateV2(1.0, 2.0, 0.25)
    end = wheel_pose_at_elapsed_v2(start, 0.6, 0.2, 1.5, 1.5)

    assert wheel_pose_at_elapsed_v2(start, 0.6, 0.2, 1.5, 0.0) is start
    assert end.heading_rad == pytest.approx(0.25 + 0.2 * 1.5, abs=1.0e-15)


def test_task7_independent_validator_surface_is_present() -> None:
    assert all(
        callable(value)
        for value in (
            oriented_rectangle_cell_separation_v2,
            prove_wheel_segment_sweep_v2,
            validate_wheel_sqp_candidate_l2,
            promote_wheel_candidate_to_route_v2,
        )
    )
    assert pi > 0.0


def _task7_arc_segment(
    start_heading: float,
    v_mps: float,
    omega_radps: float,
    duration_s: float = 2.0,
):
    start = PoseStateV2(0.0, 0.0, start_heading)
    end = integrate_wheel_segment_v2(start, v_mps, omega_radps, duration_s)
    request = replace(
        REQUEST,
        request_id=f"task7-arc-{start_heading}-{v_mps}-{omega_radps}",
        start_state=start,
        goal_state=end,
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=start,
                    end_state=end,
                    v_mps=v_mps,
                    omega_radps=omega_radps,
                    duration_s=duration_s,
                ),
            )
        ),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(request.terrain_snapshot),
    )
    return candidate.segments[0]


@pytest.mark.parametrize(
    ("start_heading", "omega_radps", "v_mps", "critical_heading"),
    (
        (-pi / 4.0, pi / 4.0, 0.2, 0.0),
        (7.0 * pi / 4.0, pi / 4.0, 0.2, 2.0 * pi),
        (pi / 4.0, -pi / 4.0, 0.2, 0.0),
        (-pi / 4.0, pi / 4.0, -0.2, 0.0),
    ),
)
def test_arc_center_bounds_use_analytic_quadrant_extrema(
    start_heading: float,
    omega_radps: float,
    v_mps: float,
    critical_heading: float,
) -> None:
    segment = _task7_arc_segment(start_heading, v_mps, omega_radps)
    elapsed = (critical_heading - segment.start_state.heading_rad) / segment.omega_radps
    critical = wheel_pose_at_elapsed_v2(
        segment.start_state,
        segment.v_mps,
        segment.omega_radps,
        segment.duration_s,
        elapsed,
    )
    raw_end = integrate_wheel_segment_v2(
        segment.start_state,
        segment.v_mps,
        segment.omega_radps,
        segment.duration_s,
    )
    points = (segment.start_state, raw_end, segment.end_state, critical)

    bounds = wheel_sqp_validation_module._wheel_segment_center_bounds_v2(segment)

    assert bounds == (
        nextafter(min(point.x_m for point in points), -inf),
        nextafter(max(point.x_m for point in points), inf),
        nextafter(min(point.y_m for point in points), -inf),
        nextafter(max(point.y_m for point in points), inf),
    )
    assert bounds[3] - bounds[2] < 2.0 * abs(segment.v_mps) * segment.duration_s


def test_full_circle_center_bounds_use_circle_extrema_without_iteration() -> None:
    segment = _task7_arc_segment(0.25, 0.2, pi / 30.0, 61.0)
    ratio = segment.v_mps / segment.omega_radps
    center_x = segment.start_state.x_m - ratio * sin(segment.start_state.heading_rad)
    center_y = segment.start_state.y_m + ratio * cos(segment.start_state.heading_rad)
    radius = abs(ratio)

    assert wheel_sqp_validation_module._wheel_segment_center_bounds_v2(segment) == (
        nextafter(center_x - radius, -inf),
        nextafter(center_x + radius, inf),
        nextafter(center_y - radius, -inf),
        nextafter(center_y + radius, inf),
    )


@pytest.mark.parametrize(
    ("lower_x", "expected_lo"),
    (
        (nextafter(2.0, -inf), 3),
        (2.0, 3),
        (nextafter(2.0, inf), 4),
    ),
)
def test_closed_aabb_lower_gridline_and_adjacent_ulps_use_closed_cell_formula(
    lower_x: float,
    expected_lo: int,
) -> None:
    geometry = FineGridGeometryV2(16, 16, origin=(0.0, 0.0))
    cell_range = wheel_sqp_validation_module._wheel_closed_aabb_cell_range_v2(
        lower_x,
        2.25,
        1.1,
        1.2,
        geometry,
    )

    assert cell_range.column_lo == expected_lo
    assert cell_range.column_hi == 4


def test_candidate_cell_cap_precedes_cell_materialization_terrain_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, request, deadline, ledger = _candidate_for_control_sequence(
        ((0.0, 0.0, 0.05), (0.0, 0.0, 0.05)),
        "task7-count-first-cap",
    )
    repeated = wheel_sqp_validation_module._WheelClosedCellRangeV2(
        column_lo=0,
        column_hi=500_000,
        row_lo=0,
        row_hi=0,
    )
    range_calls: list[object] = []

    def range_spy(segment, geometry, profile):
        range_calls.append(segment)
        return repeated

    def forbidden(*args, **kwargs):
        raise AssertionError("cap must precede enumeration, terrain, and controls")

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_wheel_segment_closed_aabb_cell_range_v2",
        range_spy,
    )
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_iter_wheel_closed_cell_range_v2",
        forbidden,
    )
    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden)
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "wheel_segment_center_control_slew_v1",
        forbidden,
    )

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(request.terrain_snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert len(range_calls) == 2
    assert result.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert result.counterexample is None
    assert result.checked_cell_count == 2 * (repeated.cell_count + 4)


def test_broadphase_expiry_precedes_candidate_cell_cap_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    clock = [0.0]
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock[0])
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    oversized = wheel_sqp_validation_module._WheelClosedCellRangeV2(
        column_lo=0,
        column_hi=PROFILE.max_l2_candidate_cells,
        row_lo=0,
        row_hi=0,
    )

    def expiring_broadphase(*_args):
        clock[0] = 2.0
        return oversized

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_wheel_segment_closed_aabb_cell_range_v2",
        expiring_broadphase,
    )

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.counterexample is None


def test_candidate_decode_inflation_is_rejected_before_json_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    payload = json.loads(encode_wheel_candidate_v2(candidate))
    payload["segments"][0]["samples"] = (
        payload["segments"][0]["samples"] * 1_000
    )
    inflated_bytes = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    estimate = ledger.latest_attempt_estimate(REQUEST, PROFILE)
    assert estimate.codec_bytes < len(inflated_bytes) < ledger.effective_memory_limit_bytes

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("inflated candidate must be bounded before full JSON decode")

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "decode_wheel_candidate_v2",
        forbidden_decode,
    )

    result = validate_wheel_sqp_candidate_l2(
        inflated_bytes,
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert result.candidate_hash is None


def test_candidate_decode_preflight_rejects_escaped_pose_keys_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    payload = json.loads(encode_wheel_candidate_v2(candidate))
    payload["segments"][0]["samples"] = payload["segments"][0]["samples"] * 3
    escaped_bytes = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8").replace(b'"heading_rad"', b'"\\u0068eading_rad"')
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    estimate = ledger.latest_attempt_estimate(REQUEST, PROFILE)
    assert len(escaped_bytes) < estimate.codec_bytes

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("escaped candidate keys must be bounded before decode")

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "decode_wheel_candidate_v2",
        forbidden_decode,
    )

    result = validate_wheel_sqp_candidate_l2(
        escaped_bytes,
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "wheel_sqp_identity_mismatch"
    assert result.candidate_hash is None


def test_candidate_decode_preflight_bounds_compact_container_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    payload = json.loads(encode_wheel_candidate_v2(candidate))
    payload["segments"][0]["samples"] = [{} for _ in range(200)]
    compact_bytes = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    estimate = ledger.latest_attempt_estimate(REQUEST, PROFILE)
    assert b"\\" not in compact_bytes
    assert len(compact_bytes) < estimate.codec_bytes

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("container cardinality must be bounded before decode")

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "decode_wheel_candidate_v2",
        forbidden_decode,
    )

    result = validate_wheel_sqp_candidate_l2(
        compact_bytes,
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert result.candidate_hash is None


def test_candidate_decode_preflight_expiry_precedes_resource_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    candidate_bytes = encode_wheel_candidate_v2(candidate)
    clock = [0.0]
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock[0])
    ledger = _admitted_ledger(candidate, REQUEST, deadline)

    def expiring_rejection(*_args):
        clock[0] = 2.0
        return False

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("expired decode preflight must not enter full decoder")

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_wheel_candidate_decode_within_estimate_v2",
        expiring_rejection,
    )
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "decode_wheel_candidate_v2",
        forbidden_decode,
    )

    result = validate_wheel_sqp_candidate_l2(
        candidate_bytes,
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.candidate_hash is None


def test_unrepresentable_half_meter_cells_at_huge_origin_fail_numeric_before_terrain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = float(2**52)
    geometry = FineGridGeometryV2(8, 8, origin=(origin, origin))
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=np.ones(geometry.shape, dtype=np.bool_),
        hard_obstacle_mask=np.zeros(geometry.shape, dtype=np.bool_),
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="task7-huge-origin",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    start = PoseStateV2(origin + 1.0, origin + 1.0, 0.0)
    goal = integrate_wheel_segment_v2(start, 0.5, 0.0, 2.0)
    request = replace(
        REQUEST,
        request_id="task7-huge-origin",
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=start,
                    end_state=goal,
                    v_mps=0.5,
                    omega_radps=0.0,
                    duration_s=2.0,
                ),
            )
        ),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)

    def forbidden_query(*args, **kwargs):
        raise AssertionError("numeric geometry failure must precede terrain query")

    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden_query)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "wheel_sqp_numeric_contract_failed"
    assert result.checked_cell_count == 0
    assert result.counterexample is None


def test_passed_l2_receipt_is_factory_only_and_has_no_public_route_authority() -> None:
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    receipt = _make_wheel_trajectory_l2_result_v2(
        passed=True,
        reason_code=None,
        route=None,
        evidence=None,
        counterexample=None,
        actual_start=REQUEST.start_state,
        actual_goal=PoseStateV2(0.9, 0.0, 0.01),
        goal_position_error_m=0.1,
        goal_heading_error_rad=0.01,
        checked_cell_count=8,
        checked_interval_count=16,
        candidate_hash="a" * 64,
        request_hash="b" * 64,
        profile_hash="c" * 64,
        terrain_snapshot_hash="d" * 64,
        capability_revision="wheel_kinematic_corridor_sqp/v1",
        solver_contract_id="wheel_kinematic_direct_multiple_shooting_sqp/v1",
        canonicalization_id="wheel_kinematic_decimal12_half_even/v1",
        control_slew_id="wheel_segment_center_control_slew/v1",
        observation_source_id="wheel_kinematic_derived_samples/v1",
        validator_contract_id="wheel_kinematic_continuous_rectangle_sweep_l2/v1",
        request=REQUEST,
        profile=PROFILE,
        deadline=deadline,
        input_bytes_hash="e" * 64,
    )

    assert receipt.passed is True
    assert receipt.reason_code is None
    assert receipt.route is receipt.evidence is receipt.counterexample is None
    with pytest.raises(TypeError):
        WheelTrajectoryL2ResultV2()
    with pytest.raises(TypeError):
        replace(receipt, candidate_hash="f" * 64)


def _straight_candidate_and_receipt(
    deadline: PlanningDeadlineV2 | None = None,
) -> tuple[object, WheelTrajectoryL2ResultV2]:
    actual_goal = PoseStateV2(0.9, 0.0, 0.0)
    solved = SimpleNamespace(
        segments=(
            SimpleNamespace(
                start_state=REQUEST.start_state,
                end_state=actual_goal,
                v_mps=0.45,
                omega_radps=0.0,
                duration_s=2.0,
            ),
        )
    )
    terrain_hash = snapshot_hash(SNAPSHOT)
    candidate = materialize_canonical_wheel_candidate_v2(
        solved,
        request=REQUEST,
        profile=PROFILE,
        terrain_snapshot_hash=terrain_hash,
    )
    candidate_bytes = encode_wheel_candidate_v2(candidate)
    decoded = decode_wheel_candidate_v2(candidate_bytes)
    if deadline is None:
        deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    receipt = _make_wheel_trajectory_l2_result_v2(
        passed=True,
        reason_code=None,
        route=None,
        evidence=None,
        counterexample=None,
        actual_start=decoded.start_state,
        actual_goal=decoded.actual_endpoint,
        goal_position_error_m=0.1,
        goal_heading_error_rad=0.0,
        checked_cell_count=8,
        checked_interval_count=16,
        candidate_hash=decoded.candidate_hash,
        request_hash=decoded.request_hash,
        profile_hash=decoded.profile_hash,
        terrain_snapshot_hash=decoded.terrain_snapshot_hash,
        capability_revision=decoded.capability_revision,
        solver_contract_id=decoded.solver_contract_id,
        canonicalization_id=decoded.canonicalization_id,
        control_slew_id=decoded.control_slew_id,
        observation_source_id=decoded.observation_source_id,
        validator_contract_id="wheel_kinematic_continuous_rectangle_sweep_l2/v1",
        request=REQUEST,
        profile=PROFILE,
        deadline=deadline,
        input_bytes_hash=sha256(candidate_bytes).hexdigest(),
    )
    return decoded, receipt


def _admitted_ledger(
    candidate: object,
    request: PlanningRequestV2,
    deadline: PlanningDeadlineV2,
) -> WheelSQPWorkLedgerV1:
    segment_count = len(candidate.segments)
    sample_count = sum(len(segment.samples) for segment in candidate.segments)
    broadphase = min(
        PROFILE.max_l2_candidate_cells,
        segment_count * (request.terrain_snapshot.geometry.width * request.terrain_snapshot.geometry.height + 4),
    )
    interval_factor = (1 << 25) - 1
    interval_count = min(PROFILE.max_l2_interval_records, broadphase * interval_factor)
    encoded_states = 3 + 4 * segment_count + 2 * sample_count
    encoded_scalars = 14 + 24 * segment_count + 6 * sample_count
    variable_count = 6 * segment_count
    constraint_count = 12 * segment_count - 1
    decision_bytes = 8 * variable_count
    jacobian_bytes = 8 * constraint_count * variable_count
    solver_bytes = PROFILE.solver_memory_reservation_bytes
    l2_queue_bytes = 96 * interval_count
    codec_bytes = 64 * encoded_states + 16 * encoded_scalars
    required_bytes = (
        decision_bytes + jacobian_bytes + solver_bytes + l2_queue_bytes + codec_bytes
    )
    reserve = L2ReserveModelV1().assess(
        deadline,
        segment_count=segment_count,
        broadphase_cell_bound=broadphase,
        interval_record_bound=interval_count,
        encoded_state_bound=encoded_states,
        encoded_scalar_bound=encoded_scalars,
        max_segments=PROFILE.max_segments,
        max_l2_candidate_cells=PROFILE.max_l2_candidate_cells,
        max_l2_interval_records=PROFILE.max_l2_interval_records,
        max_encoded_state_bound=encoded_states,
        max_encoded_scalar_bound=encoded_scalars,
    )
    assert reserve.accepted
    ledger = WheelSQPWorkLedgerV1(request.resource_budget, deadline)
    ledger.reserve_attempt(required_bytes, encoded_states)
    estimate = _make_wheel_sqp_resource_estimate_v1(
        segment_count=segment_count,
        variable_count=variable_count,
        equality_count=3 * segment_count,
        base_inequality_count=4 * segment_count - 1,
        terrain_inequality_count=5 * segment_count,
        total_constraint_count=constraint_count,
        broadphase_cell_bound=broadphase,
        interval_record_bound=interval_count,
        encoded_state_bound=encoded_states,
        encoded_scalar_bound=encoded_scalars,
        decision_bytes=decision_bytes,
        jacobian_bytes=jacobian_bytes,
        solver_bytes=solver_bytes,
        l2_queue_bytes=l2_queue_bytes,
        codec_bytes=codec_bytes,
        required_bytes=required_bytes,
        post_solver_reserve=reserve,
    )
    ledger._bind_latest_attempt_estimate(estimate, request, PROFILE)
    return ledger


def _rehash_candidate_json(payload: dict[str, object]) -> bytes:
    segments = payload["segments"]
    assert type(segments) is list
    for segment in segments:
        assert type(segment) is dict
        segment["segment_hash"] = sha256(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in segment.items()
                    if key != "segment_hash"
                }
            )
        ).hexdigest()
    payload["candidate_hash"] = sha256(
        canonical_json_bytes(
            {
                key: value
                for key, value in payload.items()
                if key != "candidate_hash"
            }
        )
    ).hexdigest()
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_promotion_atomically_returns_strict_route_and_matching_evidence() -> None:
    candidate, receipt = _straight_candidate_and_receipt()

    promotion = promote_wheel_candidate_to_route_v2(candidate, receipt)

    assert promotion.route.source_candidate_hash == candidate.candidate_hash
    assert promotion.evidence.candidate_hash == candidate.candidate_hash
    assert promotion.evidence.route_hash == promotion.route.route_hash
    assert promotion.evidence.repair_applied is False
    assert encode_wheel_route_v2(promotion.route)
    projected = project_wheel_route_to_candidate_v1(
        promotion.route,
        REQUEST,
        PROFILE,
        candidate.terrain_snapshot_hash,
    )
    assert projected.candidate_hash == candidate.candidate_hash


def test_promotion_digest_expiry_precedes_candidate_reseal_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock[0])
    candidate, receipt = _straight_candidate_and_receipt(deadline)

    class _ExpiringDigest:
        def hexdigest(self) -> str:
            clock[0] = 2.0
            return "f" * 64

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "sha256",
        lambda _payload: _ExpiringDigest(),
    )

    with pytest.raises(WheelSQPWorkLimitError) as caught:
        promote_wheel_candidate_to_route_v2(candidate, receipt)

    assert caught.value.reason_code == "planning_deadline_expired"


def test_promotion_entry_expiry_returns_typed_timeout() -> None:
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 2.0)
    candidate, receipt = _straight_candidate_and_receipt(deadline)

    with pytest.raises(WheelSQPWorkLimitError) as caught:
        promote_wheel_candidate_to_route_v2(candidate, receipt)

    assert caught.value.reason_code == "planning_deadline_expired"


@pytest.mark.parametrize("phase", ("route_encode", "candidate_projection"))
def test_promotion_phase_completion_expiry_returns_typed_timeout(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    clock = [0.0]
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock[0])
    candidate, receipt = _straight_candidate_and_receipt(deadline)
    target = (
        "encode_wheel_route_v2"
        if phase == "route_encode"
        else "project_wheel_route_to_candidate_v1"
    )
    original = getattr(wheel_sqp_validation_module, target)

    def expire_after_bounded_phase(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 2.0
        return result

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        target,
        expire_after_bounded_phase,
    )

    with pytest.raises(WheelSQPWorkLimitError) as caught:
        promote_wheel_candidate_to_route_v2(candidate, receipt)

    assert caught.value.reason_code == "planning_deadline_expired"


def test_promotion_rejects_projection_candidate_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, receipt = _straight_candidate_and_receipt()
    original = wheel_sqp_validation_module.project_wheel_route_to_candidate_v1

    def mismatching_projection(*args, **kwargs):
        projected = original(*args, **kwargs)
        return SimpleNamespace(candidate_hash="f" * 64, projected=projected)

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "project_wheel_route_to_candidate_v1",
        mismatching_projection,
    )

    with pytest.raises(
        ValueError, match="public route projection changed candidate hash"
    ):
        promote_wheel_candidate_to_route_v2(candidate, receipt)


def test_promotion_rejects_candidate_receipt_reseal_mismatch_before_route_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt = _straight_candidate_and_receipt()
    other_goal = PoseStateV2(0.8, 0.0, 0.0)
    other = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=REQUEST.start_state,
                    end_state=other_goal,
                    v_mps=0.4,
                    omega_radps=0.0,
                    duration_s=2.0,
                ),
            )
        ),
        request=REQUEST,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(SNAPSHOT),
    )

    def forbidden_route_build(*_args, **_kwargs):
        raise AssertionError("route build must follow candidate/receipt reseal binding")

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "encode_wheel_route_v2",
        forbidden_route_build,
    )

    with pytest.raises(ValueError, match="receipt input bytes do not match candidate"):
        promote_wheel_candidate_to_route_v2(other, receipt)


@pytest.mark.parametrize(
    "target",
    ("encode_wheel_route_v2", "project_wheel_route_to_candidate_v1"),
)
@pytest.mark.parametrize("fault_type", (KeyboardInterrupt, SystemExit, MemoryError))
def test_promotion_propagates_critical_faults_without_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    fault_type: type[BaseException],
) -> None:
    candidate, receipt = _straight_candidate_and_receipt()
    fault = fault_type("task7-critical-promotion-fault")

    def raise_fault(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(wheel_sqp_validation_module, target, raise_fault)

    with pytest.raises(fault_type) as caught:
        promote_wheel_candidate_to_route_v2(candidate, receipt)

    assert caught.value is fault


def test_l2_decodes_fresh_candidate_and_preserves_real_terminal_pose() -> None:
    candidate, _ = _straight_candidate_and_receipt()
    candidate_bytes = encode_wheel_candidate_v2(candidate)
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)

    result = validate_wheel_sqp_candidate_l2(
        candidate_bytes,
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is True
    assert result.actual_start == REQUEST.start_state
    assert result.actual_goal == candidate.actual_endpoint
    assert result.actual_goal != REQUEST.goal_state
    assert result.goal_position_error_m == pytest.approx(0.1, abs=1.0e-12)
    assert result.route is result.evidence is None


def test_sat_contact_and_explicit_separation_epsilon_are_unsafe() -> None:
    geometry = FineGridGeometryV2(4, 4, origin=(0.0, 0.0))
    cell = Cell(0, 0)
    contact_x = 0.25 + PROFILE.body_length_m / 2.0 + 0.25

    contact_gap = oriented_rectangle_cell_separation_v2(
        PoseStateV2(contact_x, 0.25, 0.0), cell, geometry, PROFILE
    )
    epsilon_gap = oriented_rectangle_cell_separation_v2(
        PoseStateV2(
            contact_x + PROFILE.continuous_separation_epsilon_m,
            0.25,
            0.0,
        ),
        cell,
        geometry,
        PROFILE,
    )
    clear_gap = oriented_rectangle_cell_separation_v2(
        PoseStateV2(
            contact_x + 2.0 * PROFILE.continuous_separation_epsilon_m,
            0.25,
            0.0,
        ),
        cell,
        geometry,
        PROFILE,
    )

    assert contact_gap <= PROFILE.continuous_separation_epsilon_m
    assert epsilon_gap <= PROFILE.continuous_separation_epsilon_m
    assert clear_gap > PROFILE.continuous_separation_epsilon_m


def test_oob_halfspace_margins_are_stable_and_contact_band_is_unsafe() -> None:
    geometry = FineGridGeometryV2(4, 4, origin=(0.0, 0.0))
    half_length = PROFILE.body_length_m / 2.0
    epsilon = PROFILE.continuous_separation_epsilon_m

    ordered = wheel_sqp_validation_module._oriented_rectangle_boundary_separations_v2(
        PoseStateV2(0.8, 0.9, 0.0), geometry, PROFILE
    )
    contact = wheel_sqp_validation_module._oriented_rectangle_boundary_separations_v2(
        PoseStateV2(half_length, 1.0, 0.0), geometry, PROFILE
    )
    epsilon_clearance = (
        wheel_sqp_validation_module._oriented_rectangle_boundary_separations_v2(
            PoseStateV2(nextafter(half_length + epsilon, -inf), 1.0, 0.0),
            geometry,
            PROFILE,
        )
    )
    clear = wheel_sqp_validation_module._oriented_rectangle_boundary_separations_v2(
        PoseStateV2(half_length + 2.0 * epsilon, 1.0, 0.0), geometry, PROFILE
    )

    assert ordered == pytest.approx((0.494, 0.894, 0.61, 0.81), abs=2.0e-15)
    assert contact[0] <= epsilon
    assert epsilon_clearance[0] <= epsilon
    assert clear[0] > epsilon


def test_continuous_sweep_rejects_hard_cell_between_safe_endpoints() -> None:
    geometry = FineGridGeometryV2(8, 8, origin=(-2.0, -2.0))
    hard = np.zeros(geometry.shape, dtype=np.bool_)
    hard[4, 4] = True
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    traversable[4, 4] = False
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task7-mid-sweep",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    start = PoseStateV2(-1.0, 0.0, 0.0)
    goal = PoseStateV2(1.0, 0.0, 0.0)
    request = replace(
        REQUEST,
        request_id="wheel-sqp-task7-mid-sweep",
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
    )
    solved = SimpleNamespace(
        segments=(
            SimpleNamespace(
                start_state=start,
                end_state=integrate_wheel_segment_v2(start, 1.0, 0.0, 2.0),
                v_mps=1.0,
                omega_radps=0.0,
                duration_s=2.0,
            ),
        )
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        solved,
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is False
    assert result.reason_code == "wheel_sqp_candidate_l2_rejected"
    assert result.counterexample is not None
    assert result.counterexample.cell == Cell(4, 4)
    assert result.counterexample.terrain_reason == "terrain_hard_obstacle"
    assert result.counterexample.repairable is True


def _task7_mid_sweep_terrain_result(
    terrain_case: str,
    *,
    slope_value: float = 0.0,
):
    geometry = FineGridGeometryV2(8, 8, origin=(-2.0, -2.0))
    observed = np.ones(geometry.shape, dtype=np.bool_)
    hard = np.zeros(geometry.shape, dtype=np.bool_)
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    slope = np.zeros(geometry.shape)
    target = (4, 4)
    if terrain_case in ("unknown", "all"):
        observed[target] = False
    if terrain_case in ("hard", "all"):
        hard[target] = True
        traversable[target] = False
    if terrain_case in ("not_traversable", "all"):
        traversable[target] = False
    if terrain_case in ("slope", "all"):
        slope[target] = slope_value
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=slope,
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=observed,
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id=f"task7-mid-sweep-{terrain_case}-{slope_value}",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    start = PoseStateV2(-1.0, 0.0, 0.0)
    goal = PoseStateV2(1.0, 0.0, 0.0)
    request = replace(
        REQUEST,
        request_id=f"task7-mid-sweep-{terrain_case}-{slope_value}",
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=start,
                    end_state=integrate_wheel_segment_v2(start, 1.0, 0.0, 2.0),
                    v_mps=1.0,
                    omega_radps=0.0,
                    duration_s=2.0,
                ),
            )
        ),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    return validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(snapshot),
        PROFILE,
        deadline,
        _admitted_ledger(candidate, request, deadline),
    )


@pytest.mark.parametrize(
    ("terrain_case", "terrain_reason"),
    (
        ("unknown", "terrain_unknown"),
        ("hard", "terrain_hard_obstacle"),
        ("not_traversable", "terrain_not_traversable"),
        ("slope", "terrain_slope_exceeded"),
        ("all", "terrain_unknown"),
    ),
)
def test_mid_sweep_terrain_reason_and_precedence_are_stable(
    terrain_case: str,
    terrain_reason: str,
) -> None:
    result = _task7_mid_sweep_terrain_result(
        terrain_case,
        slope_value=nextafter(30.0, inf),
    )

    assert result.reason_code == "wheel_sqp_candidate_l2_rejected"
    assert result.counterexample is not None
    assert result.counterexample.cell == Cell(4, 4)
    assert result.counterexample.terrain_reason == terrain_reason
    assert result.counterexample.repairable is True


def test_mid_sweep_slope_boundary_is_closed() -> None:
    at_limit = _task7_mid_sweep_terrain_result("slope", slope_value=30.0)
    above_limit = _task7_mid_sweep_terrain_result(
        "slope", slope_value=nextafter(30.0, inf)
    )

    assert at_limit.passed is True
    assert above_limit.passed is False
    assert above_limit.counterexample is not None
    assert above_limit.counterexample.terrain_reason == "terrain_slope_exceeded"
    assert above_limit.counterexample.repairable is True


def test_lipschitz_proves_broadphase_invalid_cell_far_from_footprint() -> None:
    geometry = FineGridGeometryV2(8, 8, origin=(-1.7, -1.7))
    hard = np.zeros(geometry.shape, dtype=np.bool_)
    hard[4, 4] = True
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    traversable[4, 4] = False
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task7-lipschitz-safe",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    pose = PoseStateV2(0.0, 0.0, 0.0)
    request = replace(
        REQUEST,
        request_id="wheel-sqp-task7-lipschitz-safe",
        start_state=pose,
        goal_state=pose,
        terrain_snapshot=snapshot,
    )
    solved = SimpleNamespace(
        segments=(
            SimpleNamespace(
                start_state=pose,
                end_state=pose,
                v_mps=0.0,
                omega_radps=0.0,
                duration_s=1.0,
            ),
        )
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        solved,
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is True
    assert result.counterexample is None
    assert result.checked_interval_count >= 1


def _task7_direct_sweep_context(clock: list[float] | None = None):
    candidate, _ = _straight_candidate_and_receipt()
    if clock is None:
        clock = [0.0]
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock[0])
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    estimate = ledger.latest_attempt_estimate(REQUEST, PROFILE)
    return candidate, deadline, ledger, estimate, clock


def _task7_prove_direct_sweep(
    candidate,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
    estimate,
    *,
    interval_record_limit: int,
):
    return prove_wheel_segment_sweep_v2(
        candidate.segments[0],
        0,
        candidate.candidate_hash,
        (),
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
        estimate,
        base_codec_delta_bytes=0,
        created_record_offset=0,
        interval_record_limit=interval_record_limit,
        codec_reseal_tail_s=0.0,
    )


def _assert_task7_unrepairable_interval_failure(result, reason_code: str) -> None:
    assert result.passed is False
    assert result.reason_code == reason_code
    assert result.counterexample is not None
    assert result.counterexample.segment_index == 0
    assert result.counterexample.cell == Cell(-1, 0)
    assert result.counterexample.terrain_reason == "terrain_out_of_bounds"
    assert result.counterexample.repairable is False


def test_interval_depth_cap_fails_closed_with_unrepairable_dyadic_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, deadline, ledger, estimate, _ = _task7_direct_sweep_context()
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_wheel_sweep_source_gap_v2",
        lambda *_args: 2.0e-9,
    )

    result = _task7_prove_direct_sweep(
        candidate,
        deadline,
        ledger,
        estimate,
        interval_record_limit=PROFILE.max_l2_interval_records,
    )

    _assert_task7_unrepairable_interval_failure(
        result, "wheel_sqp_candidate_l2_rejected"
    )
    assert result.counterexample.time_fraction_lo == 0.0
    assert result.counterexample.time_fraction_hi == 1.0 / (1 << 24)


def test_interval_record_cap_fails_closed_before_child_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, deadline, ledger, estimate, _ = _task7_direct_sweep_context()
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_wheel_sweep_source_gap_v2",
        lambda *_args: 0.1,
    )

    result = _task7_prove_direct_sweep(
        candidate,
        deadline,
        ledger,
        estimate,
        interval_record_limit=4,
    )

    _assert_task7_unrepairable_interval_failure(
        result, "wheel_sqp_candidate_l2_rejected"
    )
    assert (
        result.counterexample.time_fraction_lo,
        result.counterexample.time_fraction_mid,
        result.counterexample.time_fraction_hi,
    ) == (0.0, 0.25, 0.5)


def test_interval_memory_admission_fails_closed_before_child_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, deadline, ledger, estimate, _ = _task7_direct_sweep_context()
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_wheel_sweep_source_gap_v2",
        lambda *_args: 0.1,
    )
    original_admit = WheelSQPWorkLedgerV1.admit_validation_delta
    calls = [0]

    def fail_fifth_admission(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 5:
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        return original_admit(self, *args, **kwargs)

    monkeypatch.setattr(
        WheelSQPWorkLedgerV1,
        "admit_validation_delta",
        fail_fifth_admission,
    )

    result = _task7_prove_direct_sweep(
        candidate,
        deadline,
        ledger,
        estimate,
        interval_record_limit=PROFILE.max_l2_interval_records,
    )

    _assert_task7_unrepairable_interval_failure(
        result, "wheel_sqp_resource_budget_exceeded"
    )
    assert calls[0] == 5


def test_interval_deadline_fails_closed_after_bounded_midpoint_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    candidate, deadline, ledger, estimate, _ = _task7_direct_sweep_context(clock)
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_wheel_sweep_source_gap_v2",
        lambda *_args: 0.1,
    )
    original_integrate = wheel_sqp_validation_module.integrate_wheel_segment_v2
    calls = [0]

    def expiring_integrate(*args, **kwargs):
        result = original_integrate(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            clock[0] = 2.0
        return result

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "integrate_wheel_segment_v2",
        expiring_integrate,
    )

    result = _task7_prove_direct_sweep(
        candidate,
        deadline,
        ledger,
        estimate,
        interval_record_limit=PROFILE.max_l2_interval_records,
    )

    _assert_task7_unrepairable_interval_failure(
        result, "planning_deadline_expired"
    )
    assert calls[0] == 2


@pytest.mark.parametrize("fault_type", (KeyboardInterrupt, SystemExit, MemoryError))
def test_interval_gap_propagates_critical_faults(
    monkeypatch: pytest.MonkeyPatch,
    fault_type: type[BaseException],
) -> None:
    candidate, deadline, ledger, estimate, _ = _task7_direct_sweep_context()
    fault = fault_type("task7-critical-interval-fault")

    def raise_fault(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "_wheel_sweep_source_gap_v2",
        raise_fault,
    )

    with pytest.raises(fault_type) as caught:
        _task7_prove_direct_sweep(
            candidate,
            deadline,
            ledger,
            estimate,
            interval_record_limit=PROFILE.max_l2_interval_records,
        )

    assert caught.value is fault


def test_dyadic_pq_finds_non_midpoint_collision_and_repeats_identically() -> None:
    geometry = FineGridGeometryV2(12, 8, origin=(-3.0, -2.0))
    hard = np.zeros(geometry.shape, dtype=np.bool_)
    hard[4, 3] = True
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    traversable[4, 3] = False
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task7-quarter-collision",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    start = PoseStateV2(-2.0, 0.0, 0.0)
    goal = PoseStateV2(2.0, 0.0, 0.0)
    request = replace(
        REQUEST,
        request_id="wheel-sqp-task7-quarter-collision",
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
    )
    solved = SimpleNamespace(
        segments=(
            SimpleNamespace(
                start_state=start,
                end_state=goal,
                v_mps=1.0,
                omega_radps=0.0,
                duration_s=4.0,
            ),
        )
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        solved,
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    candidate_bytes = encode_wheel_candidate_v2(candidate)
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)
    anchor = FineSafetyAnchorV2(snapshot)

    first = validate_wheel_sqp_candidate_l2(
        candidate_bytes, request, anchor, PROFILE, deadline, ledger
    )
    second = validate_wheel_sqp_candidate_l2(
        candidate_bytes, request, anchor, PROFILE, deadline, ledger
    )

    assert first.passed is second.passed is False
    assert first.counterexample is not None
    assert first.counterexample == second.counterexample
    assert first.counterexample.cell == Cell(3, 4)
    assert first.counterexample.time_fraction_mid == 0.25
    assert first.counterexample.repairable is True


def _candidate_for_control_sequence(
    controls: tuple[tuple[float, float, float], ...],
    request_id: str,
) -> tuple[object, PlanningRequestV2, PlanningDeadlineV2, WheelSQPWorkLedgerV1]:
    current = REQUEST.start_state
    raw_segments: list[SimpleNamespace] = []
    for v_mps, omega_radps, duration_s in controls:
        end = integrate_wheel_segment_v2(
            current, v_mps, omega_radps, duration_s
        )
        raw_segments.append(
            SimpleNamespace(
                start_state=current,
                end_state=end,
                v_mps=v_mps,
                omega_radps=omega_radps,
                duration_s=duration_s,
            )
        )
        current = end
    request = replace(
        REQUEST,
        request_id=request_id,
        goal_state=current,
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(segments=tuple(raw_segments)),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(request.terrain_snapshot),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)
    return candidate, request, deadline, ledger


@pytest.mark.parametrize(
    ("name", "accepted_controls", "rejected_controls", "limit_name", "slew_index"),
    (
        (
            "acceleration",
            ((0.0, 0.0, 1.0), (0.5, 0.0, 1.0)),
            ((0.0, 0.0, 1.0), (0.500000000001, 0.0, 1.0)),
            "max_linear_accel_mps2",
            0,
        ),
        (
            "deceleration",
            ((0.75, 0.0, 1.0), (0.0, 0.0, 1.0)),
            ((0.750000000001, 0.0, 1.0), (0.0, 0.0, 1.0)),
            "max_linear_decel_mps2",
            1,
        ),
        (
            "angular",
            ((0.0, -0.392699081699, 0.5), (0.0, 0.392699081698, 0.5)),
            ((0.0, -0.392699081699, 0.5), (0.0, 0.392699081699, 0.5)),
            "max_angular_accel_radps2",
            2,
        ),
    ),
)
def test_control_slew_closed_boundary_and_next_canonical_increment(
    name: str,
    accepted_controls: tuple[tuple[float, float, float], ...],
    rejected_controls: tuple[tuple[float, float, float], ...],
    limit_name: str,
    slew_index: int,
) -> None:
    accepted, accepted_request, accepted_deadline, accepted_ledger = (
        _candidate_for_control_sequence(
            accepted_controls,
            f"wheel-sqp-{name}-closed",
        )
    )
    rejected, rejected_request, rejected_deadline, rejected_ledger = (
        _candidate_for_control_sequence(
            rejected_controls,
            f"wheel-sqp-{name}-next-canonical",
        )
    )
    accepted_slew = wheel_sqp_validation_module.wheel_segment_center_control_slew_v1(
        *accepted_controls[0][:2],
        accepted_controls[0][2],
        *accepted_controls[1][:2],
        accepted_controls[1][2],
    )[slew_index]
    rejected_slew = wheel_sqp_validation_module.wheel_segment_center_control_slew_v1(
        *rejected_controls[0][:2],
        rejected_controls[0][2],
        *rejected_controls[1][:2],
        rejected_controls[1][2],
    )[slew_index]
    limit = getattr(PROFILE, limit_name)

    assert accepted_slew <= limit < rejected_slew
    canonical_control_delta = sum(
        (
            abs(Decimal(str(rejected)) - Decimal(str(accepted)))
            for rejected_segment, accepted_segment in zip(
                rejected_controls,
                accepted_controls,
                strict=True,
            )
            for rejected, accepted in zip(
                rejected_segment[:2],
                accepted_segment[:2],
                strict=True,
            )
        ),
        Decimal(0),
    )
    assert canonical_control_delta == Decimal("0.000000000001")

    accepted_result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(accepted),
        accepted_request,
        FineSafetyAnchorV2(accepted_request.terrain_snapshot),
        PROFILE,
        accepted_deadline,
        accepted_ledger,
    )
    rejected_result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(rejected),
        rejected_request,
        FineSafetyAnchorV2(rejected_request.terrain_snapshot),
        PROFILE,
        rejected_deadline,
        rejected_ledger,
    )

    assert accepted_result.passed is True
    assert rejected_result.passed is False
    assert rejected_result.reason_code == "wheel_sqp_candidate_l2_rejected"


def test_single_segment_heading_change_closed_boundary_and_next_canonical_increment() -> None:
    accepted_duration = 3.141592653589
    rejected_duration = 3.141592653590
    omega_radps = 0.5
    accepted, accepted_request, accepted_deadline, accepted_ledger = (
        _candidate_for_control_sequence(
            ((0.0, omega_radps, accepted_duration),),
            "wheel-sqp-heading-change-closed",
        )
    )
    rejected, rejected_request, rejected_deadline, rejected_ledger = (
        _candidate_for_control_sequence(
            ((0.0, omega_radps, rejected_duration),),
            "wheel-sqp-heading-change-next-canonical",
        )
    )

    assert (
        Decimal(str(rejected_duration)) - Decimal(str(accepted_duration))
        == Decimal("0.000000000001")
    )
    assert (
        abs(omega_radps * accepted_duration)
        <= PROFILE.max_segment_heading_change_rad
        < abs(omega_radps * rejected_duration)
    )

    accepted_result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(accepted),
        accepted_request,
        FineSafetyAnchorV2(accepted_request.terrain_snapshot),
        PROFILE,
        accepted_deadline,
        accepted_ledger,
    )
    rejected_result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(rejected),
        rejected_request,
        FineSafetyAnchorV2(rejected_request.terrain_snapshot),
        PROFILE,
        rejected_deadline,
        rejected_ledger,
    )

    assert accepted_result.passed is True
    assert rejected_result.passed is False
    assert rejected_result.reason_code == "wheel_sqp_candidate_l2_rejected"


def test_control_slew_expiry_precedes_same_step_kinematic_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, request, _, _ = _candidate_for_control_sequence(
        ((0.0, 0.0, 1.0), (0.1, 0.0, 1.0)),
        "wheel-sqp-control-step-expired",
    )
    clock = [0.0]
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock[0])
    ledger = _admitted_ledger(candidate, request, deadline)

    def expiring_invalid_slew(*_args):
        clock[0] = 2.0
        return (inf, 0.0, 0.0)

    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "wheel_segment_center_control_slew_v1",
        expiring_invalid_slew,
    )

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(request.terrain_snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "planning_deadline_expired"


def _stationary_candidate_for_goal(
    start: PoseStateV2,
    goal: PoseStateV2,
    request_id: str,
) -> tuple[object, PlanningRequestV2, PlanningDeadlineV2, WheelSQPWorkLedgerV1]:
    request = replace(
        REQUEST,
        request_id=request_id,
        start_state=start,
        goal_state=goal,
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=start,
                    end_state=start,
                    v_mps=0.0,
                    omega_radps=0.0,
                    duration_s=0.05,
                ),
            )
        ),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(request.terrain_snapshot),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    return candidate, request, deadline, _admitted_ledger(candidate, request, deadline)


def test_goal_position_tolerance_is_closed_and_rejects_nextafter() -> None:
    tolerance = PROFILE.profile.goal_position_tolerance_m
    accepted, accepted_request, accepted_deadline, accepted_ledger = (
        _stationary_candidate_for_goal(
            PoseStateV2(0.0, 0.0, 0.0),
            PoseStateV2(tolerance, 0.0, 0.0),
            "wheel-sqp-goal-position-closed",
        )
    )
    rejected_error = nextafter(tolerance, inf)
    rejected, rejected_request, rejected_deadline, rejected_ledger = (
        _stationary_candidate_for_goal(
            PoseStateV2(0.0, 0.0, 0.0),
            PoseStateV2(rejected_error, 0.0, 0.0),
            "wheel-sqp-goal-position-nextafter",
        )
    )

    assert hypot(
        accepted.actual_endpoint.x_m - accepted_request.goal_state.x_m,
        accepted.actual_endpoint.y_m - accepted_request.goal_state.y_m,
    ) == tolerance
    assert hypot(
        rejected.actual_endpoint.x_m - rejected_request.goal_state.x_m,
        rejected.actual_endpoint.y_m - rejected_request.goal_state.y_m,
    ) == rejected_error

    accepted_result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(accepted),
        accepted_request,
        FineSafetyAnchorV2(accepted_request.terrain_snapshot),
        PROFILE,
        accepted_deadline,
        accepted_ledger,
    )
    rejected_result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(rejected),
        rejected_request,
        FineSafetyAnchorV2(rejected_request.terrain_snapshot),
        PROFILE,
        rejected_deadline,
        rejected_ledger,
    )

    assert accepted_result.passed is True
    assert rejected_result.passed is False
    assert rejected_result.reason_code == "wheel_sqp_goal_tolerance_exceeded"


def test_goal_heading_tolerance_wrap_is_closed_and_rejects_adjacent_float() -> None:
    start_heading = 3.1
    tolerance = PROFILE.profile.goal_heading_tolerance_rad
    boundary_heading = start_heading - 2.0 * pi + tolerance
    accepted_goal_heading = nextafter(boundary_heading, inf)
    rejected_goal_heading = nextafter(accepted_goal_heading, inf)
    accepted, accepted_request, accepted_deadline, accepted_ledger = (
        _stationary_candidate_for_goal(
            PoseStateV2(0.0, 0.0, start_heading),
            PoseStateV2(0.0, 0.0, accepted_goal_heading),
            "wheel-sqp-goal-heading-wrap-closed",
        )
    )
    rejected, rejected_request, rejected_deadline, rejected_ledger = (
        _stationary_candidate_for_goal(
            PoseStateV2(0.0, 0.0, start_heading),
            PoseStateV2(0.0, 0.0, rejected_goal_heading),
            "wheel-sqp-goal-heading-wrap-nextafter",
        )
    )
    accepted_error = abs(
        wheel_sqp_validation_module.canonicalize_wheel_heading_v2(
            accepted.actual_endpoint.heading_rad
            - accepted_request.goal_state.heading_rad
        )
    )
    rejected_error = abs(
        wheel_sqp_validation_module.canonicalize_wheel_heading_v2(
            rejected.actual_endpoint.heading_rad
            - rejected_request.goal_state.heading_rad
        )
    )

    assert accepted.actual_endpoint.heading_rad > 0.0
    assert accepted_goal_heading < 0.0
    assert nextafter(accepted_goal_heading, inf) == rejected_goal_heading
    assert accepted_error <= tolerance < rejected_error

    accepted_result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(accepted),
        accepted_request,
        FineSafetyAnchorV2(accepted_request.terrain_snapshot),
        PROFILE,
        accepted_deadline,
        accepted_ledger,
    )
    rejected_result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(rejected),
        rejected_request,
        FineSafetyAnchorV2(rejected_request.terrain_snapshot),
        PROFILE,
        rejected_deadline,
        rejected_ledger,
    )

    assert accepted_result.passed is True
    assert rejected_result.passed is False
    assert rejected_result.reason_code == "wheel_sqp_goal_tolerance_exceeded"


@pytest.mark.parametrize(
    ("identity_name", "payload_field", "forged_value"),
    (
        (
            "request_hash",
            "request_hash",
            "c" * 64,
        ),
        (
            "solver_id",
            "solver_contract_id",
            "forged_wheel_solver/v1",
        ),
        (
            "snapshot_hash",
            "terrain_snapshot_hash",
            "a" * 64,
        ),
        (
            "profile_hash",
            "profile_hash",
            "b" * 64,
        ),
        (
            "capability",
            "capability_revision",
            "forged_wheel_capability/v1",
        ),
        (
            "canonicalization",
            "canonicalization_id",
            "forged_wheel_canonicalization/v1",
        ),
        (
            "control_slew",
            "control_slew_id",
            "forged_wheel_control_slew/v1",
        ),
        (
            "observation",
            "observation_source_id",
            "forged_wheel_observation/v1",
        ),
    ),
)
def test_l2_identity_mutations_have_stable_identity_mismatch_taxonomy(
    identity_name: str,
    payload_field: str,
    forged_value: str,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    payload = json.loads(encode_wheel_candidate_v2(candidate))
    payload[payload_field] = forged_value
    candidate_bytes = _rehash_candidate_json(payload)
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)

    result = validate_wheel_sqp_candidate_l2(
        candidate_bytes,
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert identity_name
    assert result.passed is False
    assert result.reason_code == "wheel_sqp_identity_mismatch"
    assert result.route is result.evidence is result.counterexample is None


def test_invalid_cell_order_and_duplicates_do_not_change_first_counterexample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = SNAPSHOT.geometry
    hard = np.zeros(geometry.shape, dtype=np.bool_)
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    first_cell = Cell(3, 4)
    second_cell = Cell(4, 4)
    for cell in (first_cell, second_cell):
        hard[cell.y, cell.x] = True
        traversable[cell.y, cell.x] = False
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-invalid-cell-order",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    request = replace(
        REQUEST,
        request_id="wheel-sqp-invalid-cell-order",
        terrain_snapshot=snapshot,
    )
    actual_goal = PoseStateV2(0.9, 0.0, 0.0)
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=request.start_state,
                    end_state=actual_goal,
                    v_mps=0.45,
                    omega_radps=0.0,
                    duration_s=2.0,
                ),
            )
        ),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    candidate_bytes = encode_wheel_candidate_v2(candidate)
    orders = (
        (second_cell, first_cell, second_cell, first_cell),
        (first_cell, second_cell, first_cell, second_cell, second_cell),
    )
    counterexamples = []
    for order in orders:
        monkeypatch.setattr(
            wheel_sqp_validation_module,
            "_iter_wheel_closed_cell_range_v2",
            lambda _cell_range, order=order: iter(order),
        )
        deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
        result = validate_wheel_sqp_candidate_l2(
            candidate_bytes,
            request,
            FineSafetyAnchorV2(snapshot),
            PROFILE,
            deadline,
            _admitted_ledger(candidate, request, deadline),
        )
        assert result.reason_code == "wheel_sqp_candidate_l2_rejected"
        assert result.counterexample is not None
        counterexamples.append(result.counterexample)

    assert counterexamples[0] == counterexamples[1]
    assert counterexamples[0].cell == first_cell
    assert counterexamples[0].terrain_reason == "terrain_hard_obstacle"


@pytest.mark.parametrize("turn_omega", (0.1, -0.1))
def test_direction_change_requires_dedicated_stop_across_turn_mode(
    turn_omega: float,
) -> None:
    invalid, invalid_request, invalid_deadline, invalid_ledger = (
        _candidate_for_control_sequence(
            ((0.1, 0.0, 1.0), (0.0, turn_omega, 1.0), (-0.1, 0.0, 1.0)),
            f"wheel-sqp-turn-sign-{turn_omega}",
        )
    )
    valid, valid_request, valid_deadline, valid_ledger = (
        _candidate_for_control_sequence(
            (
                (0.1, 0.0, 1.0),
                (0.0, turn_omega, 1.0),
                (0.0, 0.0, 1.0),
                (-0.1, 0.0, 1.0),
            ),
            f"wheel-sqp-turn-stop-sign-{turn_omega}",
        )
    )

    rejected = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(invalid),
        invalid_request,
        FineSafetyAnchorV2(invalid_request.terrain_snapshot),
        PROFILE,
        invalid_deadline,
        invalid_ledger,
    )
    accepted = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(valid),
        valid_request,
        FineSafetyAnchorV2(valid_request.terrain_snapshot),
        PROFILE,
        valid_deadline,
        valid_ledger,
    )

    assert rejected.passed is False
    assert rejected.reason_code == "wheel_sqp_candidate_l2_rejected"
    assert accepted.passed is True


def test_actual_l2_reserve_is_assessed_before_any_terrain_query(monkeypatch) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    candidate_bytes = encode_wheel_candidate_v2(candidate)
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    anchor = FineSafetyAnchorV2(SNAPSHOT)
    events: list[tuple[str, object]] = []
    original_assess = L2ReserveModelV1.assess
    original_query = FineSafetyAnchorV2.query

    def assess_spy(self, checked_deadline, **kwargs):
        events.append(("assess", kwargs))
        return original_assess(self, checked_deadline, **kwargs)

    def query_spy(self, cell, max_slope_deg=30.0):
        events.append(("query", cell))
        return original_query(self, cell, max_slope_deg)

    monkeypatch.setattr(L2ReserveModelV1, "assess", assess_spy)
    monkeypatch.setattr(FineSafetyAnchorV2, "query", query_spy)

    result = validate_wheel_sqp_candidate_l2(
        candidate_bytes, REQUEST, anchor, PROFILE, deadline, ledger
    )

    assert result.passed is True
    assert events[0][0] == "assess"
    actual = events[0][1]
    sample_count = sum(len(segment.samples) for segment in candidate.segments)
    encoded_states = 3 + 4 * len(candidate.segments) + 2 * sample_count
    encoded_scalars = 14 + 24 * len(candidate.segments) + 6 * sample_count
    expected_intervals = min(
        PROFILE.max_l2_interval_records,
        result.checked_cell_count
        * ((1 << (PROFILE.max_l2_subdivision_depth + 1)) - 1),
    )
    assert actual == {
        "segment_count": len(candidate.segments),
        "broadphase_cell_bound": result.checked_cell_count,
        "interval_record_bound": expected_intervals,
        "encoded_state_bound": encoded_states,
        "encoded_scalar_bound": encoded_scalars,
        "max_segments": PROFILE.max_segments,
        "max_l2_candidate_cells": PROFILE.max_l2_candidate_cells,
        "max_l2_interval_records": PROFILE.max_l2_interval_records,
        "max_encoded_state_bound": encoded_states,
        "max_encoded_scalar_bound": encoded_scalars,
    }


def test_actual_l2_reserve_equal_to_remaining_rejects_before_terrain(monkeypatch) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    assess_calls: list[dict[str, int]] = []

    def equal_reserve(self, checked_deadline, **kwargs):
        assess_calls.append(kwargs)
        remaining = checked_deadline.remaining_s
        return WheelSQPResourceLedgerV1(
            remaining,
            remaining,
            False,
            "wheel_sqp_resource_budget_exceeded",
        )

    def forbidden_query(*args, **kwargs):
        raise AssertionError("terrain query must follow accepted actual reserve")

    monkeypatch.setattr(L2ReserveModelV1, "assess", equal_reserve)
    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden_query)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert assess_calls
    assert result.passed is False
    assert result.reason_code == "wheel_sqp_resource_budget_exceeded"


@pytest.mark.parametrize("fault_type", (KeyboardInterrupt, SystemExit, MemoryError))
def test_l2_terrain_query_propagates_critical_faults(
    monkeypatch: pytest.MonkeyPatch,
    fault_type: type[BaseException],
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    fault = fault_type("task7-critical-validator-fault")

    def raise_fault(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(FineSafetyAnchorV2, "query", raise_fault)

    with pytest.raises(fault_type) as caught:
        validate_wheel_sqp_candidate_l2(
            encode_wheel_candidate_v2(candidate),
            REQUEST,
            FineSafetyAnchorV2(SNAPSHOT),
            PROFILE,
            deadline,
            ledger,
        )

    assert caught.value is fault


def _task7_risk_candidate_and_request():
    request = replace(
        REQUEST,
        request_id="task7-risk-objective",
        objective_profile=ObjectiveProfileV2(
            distance_weight=0.0,
            risk_weight=1.0,
            energy_weight=0.5,
            time_weight=0.5,
        ),
    )
    actual_goal = PoseStateV2(0.9, 0.0, 0.0)
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=request.start_state,
                    end_state=actual_goal,
                    v_mps=0.45,
                    omega_radps=0.0,
                    duration_s=2.0,
                ),
            )
        ),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(request.terrain_snapshot),
    )
    return candidate, request


def test_nonzero_risk_is_rejected_after_actual_resources_but_before_terrain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, request = _task7_risk_candidate_and_request()
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)

    def forbidden_query(*args, **kwargs):
        raise AssertionError("unsupported risk must precede terrain query")

    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden_query)
    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(request.terrain_snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "wheel_sqp_objective_unsupported"
    assert result.route is result.evidence is result.counterexample is None


def test_entry_expired_precedes_hash_decode_malformed_and_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, risk_request = _task7_risk_candidate_and_request()
    clock = SimpleNamespace(now=0.0)
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock.now)
    ledger = _admitted_ledger(candidate, risk_request, deadline)
    encoded = encode_wheel_candidate_v2(candidate)
    clock.now = deadline.deadline_monotonic_s

    def forbidden_hash(*args, **kwargs):
        raise AssertionError("expired entry must precede input hashing and decode")

    monkeypatch.setattr(wheel_sqp_validation_module, "sha256", forbidden_hash)
    for candidate_bytes in (b"{", encoded):
        result = validate_wheel_sqp_candidate_l2(
            candidate_bytes,
            risk_request,
            FineSafetyAnchorV2(risk_request.terrain_snapshot),
            PROFILE,
            deadline,
            ledger,
        )
    assert result.reason_code == "planning_deadline_expired"
    assert result.candidate_hash is None


def test_latest_estimate_expiry_precedes_same_step_identity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    clock = [0.0]
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock[0])
    ledger = _admitted_ledger(candidate, REQUEST, deadline)

    def expiring_estimate(*_args, **_kwargs):
        clock[0] = 2.0
        raise ValueError("same-step forged estimate")

    monkeypatch.setattr(
        WheelSQPWorkLedgerV1,
        "latest_attempt_estimate",
        expiring_estimate,
    )

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.reason_code == "planning_deadline_expired"


@pytest.mark.parametrize(
    ("remaining_kind", "expected_reason"),
    (
        ("tail", "wheel_sqp_resource_budget_exceeded"),
        ("zero", "planning_deadline_expired"),
    ),
)
def test_exact_codec_reseal_tail_is_kept_after_full_actual_reserve(
    monkeypatch: pytest.MonkeyPatch,
    remaining_kind: str,
    expected_reason: str,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    clock = SimpleNamespace(now=0.0)
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: clock.now)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    sample_count = sum(len(segment.samples) for segment in candidate.segments)
    encoded_states = 3 + 4 * len(candidate.segments) + 2 * sample_count
    encoded_scalars = 14 + 24 * len(candidate.segments) + 6 * sample_count
    tail = L2ReserveModelV1().reserve_s(
        segment_count=len(candidate.segments),
        broadphase_cell_bound=0,
        interval_record_bound=0,
        encoded_state_bound=encoded_states,
        encoded_scalar_bound=encoded_scalars,
        max_segments=PROFILE.max_segments,
        max_l2_candidate_cells=PROFILE.max_l2_candidate_cells,
        max_l2_interval_records=PROFILE.max_l2_interval_records,
        max_encoded_state_bound=encoded_states,
        max_encoded_scalar_bound=encoded_scalars,
    )
    original_assess = L2ReserveModelV1.assess
    full_assessments = []

    def assess_then_flip(self, checked_deadline, **kwargs):
        assessed = original_assess(self, checked_deadline, **kwargs)
        full_assessments.append(assessed)
        remaining = tail if remaining_kind == "tail" else 0.0
        clock.now = checked_deadline.deadline_monotonic_s - remaining
        if remaining_kind == "tail":
            clock.now = nextafter(clock.now, inf)
        return assessed

    def forbidden(*args, **kwargs):
        raise AssertionError("tail rejection must precede controls and terrain")

    monkeypatch.setattr(L2ReserveModelV1, "assess", assess_then_flip)
    monkeypatch.setattr(wheel_sqp_validation_module, "_expected_mode_v2", forbidden)
    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert full_assessments and full_assessments[0].accepted is True
    assert result.reason_code == expected_reason
    assert result.counterexample is None


def test_terrain_roots_are_preadmitted_before_invalid_source_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    events: list[tuple[str, object]] = []
    original_admit = WheelSQPWorkLedgerV1.admit_validation_delta
    original_query = FineSafetyAnchorV2.query
    returned_invalid = False

    def admit_spy(self, estimate, candidate_hash, *, memory_bytes, route_states):
        events.append(("admit", memory_bytes))
        return original_admit(
            self,
            estimate,
            candidate_hash,
            memory_bytes=memory_bytes,
            route_states=route_states,
        )

    def query_spy(self, cell, max_slope_deg=30.0):
        nonlocal returned_invalid
        result = original_query(self, cell, max_slope_deg)
        if not returned_invalid:
            returned_invalid = True
            events.append(("query_invalid", cell))
            return SimpleNamespace(
                passed=False,
                reason_code="terrain_hard_obstacle",
            )
        events.append(("query_safe", cell))
        return result

    def proof_spy(*args, **kwargs):
        invalid_cells = args[3]
        events.append(("proof", invalid_cells))
        return wheel_sqp_validation_module._WheelSegmentSweepResultV2(
            True,
            None,
            None,
            0,
            5,
        )

    monkeypatch.setattr(
        WheelSQPWorkLedgerV1,
        "admit_validation_delta",
        admit_spy,
    )
    monkeypatch.setattr(FineSafetyAnchorV2, "query", query_spy)
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "prove_wheel_segment_sweep_v2",
        proof_spy,
    )

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is True
    base_codec_bytes = events[0][1]
    assert events[1] == ("admit", base_codec_bytes + 416 * 4)
    invalid_query_index = next(
        index for index, event in enumerate(events) if event[0] == "query_invalid"
    )
    assert events[invalid_query_index + 1] == (
        "admit",
        base_codec_bytes + 416 * 5,
    )
    proof_index = next(index for index, event in enumerate(events) if event[0] == "proof")
    assert invalid_query_index + 1 < proof_index
    assert events[proof_index][1] == ((events[invalid_query_index][1], "terrain_hard_obstacle"),)


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    (
        (
            WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded"),
            "wheel_sqp_resource_budget_exceeded",
        ),
        (OverflowError("root bytes overflow"), "wheel_sqp_numeric_contract_failed"),
        (ValueError("root estimate identity"), "wheel_sqp_identity_mismatch"),
    ),
)
def test_boundary_root_preadmission_failure_stops_before_terrain_and_proof(
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
    expected_reason: str,
) -> None:
    candidate, _ = _straight_candidate_and_receipt()
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, REQUEST, deadline)
    original_admit = WheelSQPWorkLedgerV1.admit_validation_delta
    call_count = 0

    def fail_second_admission(
        self, estimate, candidate_hash, *, memory_bytes, route_states
    ):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise fault
        return original_admit(
            self,
            estimate,
            candidate_hash,
            memory_bytes=memory_bytes,
            route_states=route_states,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("failed root admission must stop before terrain and proof")

    monkeypatch.setattr(
        WheelSQPWorkLedgerV1,
        "admit_validation_delta",
        fail_second_admission,
    )
    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden)
    monkeypatch.setattr(
        wheel_sqp_validation_module,
        "prove_wheel_segment_sweep_v2",
        forbidden,
    )

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        REQUEST,
        FineSafetyAnchorV2(SNAPSHOT),
        PROFILE,
        deadline,
        ledger,
    )

    assert call_count == 2
    assert result.reason_code == expected_reason
    assert result.counterexample is None


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("node_disconnect", "wheel_sqp_candidate_l2_rejected"),
        ("wrong_replay_end", "wheel_sqp_candidate_l2_rejected"),
        ("nonpositive_dt", "wheel_sqp_numeric_contract_failed"),
        ("cost", "wheel_sqp_identity_mismatch"),
    ),
)
def test_l2_maps_strict_codec_semantic_taxonomy(
    mutation: str,
    expected_reason: str,
) -> None:
    if mutation == "node_disconnect":
        candidate, request, deadline, ledger = _candidate_for_control_sequence(
            ((0.2, 0.0, 1.0), (0.2, 0.0, 1.0)),
            "wheel-sqp-codec-node-disconnect",
        )
    else:
        candidate, _ = _straight_candidate_and_receipt()
        request = REQUEST
        deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
        ledger = _admitted_ledger(candidate, request, deadline)
    payload = json.loads(encode_wheel_candidate_v2(candidate))

    if mutation == "node_disconnect":
        payload["segments"][1]["start_state"]["x_m"] = round(
            payload["segments"][1]["start_state"]["x_m"] + 0.01, 12
        )
        payload["segments"][1]["samples"][0]["x_m"] = payload["segments"][1][
            "start_state"
        ]["x_m"]
    elif mutation == "wrong_replay_end":
        drifted_x = round(payload["segments"][0]["end_state"]["x_m"] + 0.01, 12)
        payload["segments"][0]["end_state"]["x_m"] = drifted_x
        payload["segments"][0]["samples"][-1]["x_m"] = drifted_x
        payload["actual_endpoint"]["x_m"] = drifted_x
    elif mutation == "nonpositive_dt":
        payload["segments"][0]["duration_s"] = 0.0
    elif mutation == "cost":
        payload["total_cost"] += 1.0
    else:  # pragma: no cover - parameter set is closed
        raise AssertionError(mutation)

    result = validate_wheel_sqp_candidate_l2(
        _rehash_candidate_json(payload),
        request,
        FineSafetyAnchorV2(request.terrain_snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is False
    assert result.reason_code == expected_reason
    assert result.route is result.evidence is result.counterexample is None


def test_continuous_footprint_oob_uses_nonrepairable_boundary_counterexample() -> None:
    geometry = FineGridGeometryV2(4, 4, origin=(0.0, 0.0))
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=np.ones(geometry.shape, dtype=np.bool_),
        hard_obstacle_mask=np.zeros(geometry.shape, dtype=np.bool_),
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task7-oob",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    start = PoseStateV2(0.3, 1.0, 0.0)
    goal = PoseStateV2(1.0, 1.0, 0.0)
    request = replace(
        REQUEST,
        request_id="wheel-sqp-task7-oob",
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
    )
    solved = SimpleNamespace(
        segments=(
            SimpleNamespace(
                start_state=start,
                end_state=goal,
                v_mps=0.35,
                omega_radps=0.0,
                duration_s=2.0,
            ),
        )
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        solved,
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is False
    assert result.reason_code == "wheel_sqp_candidate_l2_rejected"
    assert result.counterexample is not None
    assert result.counterexample.terrain_reason == "terrain_out_of_bounds"
    assert result.counterexample.repairable is False
    assert result.counterexample.cell == Cell(-1, 0)


def _task7_seam_snapshot(
    geometry: FineGridGeometryV2,
    *,
    source_id: str,
    hard_cell: Cell | None = None,
) -> TerrainSnapshotV2:
    hard = np.zeros(geometry.shape, dtype=np.bool_)
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    if hard_cell is not None:
        hard[hard_cell.y, hard_cell.x] = True
        traversable[hard_cell.y, hard_cell.x] = False
    return TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id=source_id,
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )


def _task7_assert_seam_only(
    segment,
    raw_gap: float,
    declared_gap: float,
) -> None:
    raw_endpoint = integrate_wheel_segment_v2(
        segment.start_state,
        segment.v_mps,
        segment.omega_radps,
        segment.duration_s,
    )
    footprint_radius = hypot(
        PROFILE.body_length_m / 2.0 + PROFILE.footprint_safety_margin_m,
        PROFILE.body_width_m / 2.0 + PROFILE.footprint_safety_margin_m,
    )
    connector = nextafter(
        hypot(
            segment.end_state.x_m - raw_endpoint.x_m,
            segment.end_state.y_m - raw_endpoint.y_m,
        )
        + footprint_radius
        * abs(segment.end_state.heading_rad - raw_endpoint.heading_rad),
        inf,
    )
    seam_rhs = nextafter(
        PROFILE.continuous_separation_epsilon_m + connector, inf
    )
    assert raw_gap > PROFILE.continuous_separation_epsilon_m
    assert declared_gap > PROFILE.continuous_separation_epsilon_m
    assert connector > 0.0
    assert min(raw_gap, declared_gap) <= seam_rhs


def test_decimal12_in_bounds_seam_requires_independent_clearance_proof() -> None:
    geometry = FineGridGeometryV2(8, 8, origin=(-2.0, -2.0))
    hard_cell = Cell(4, 4)
    snapshot = _task7_seam_snapshot(
        geometry,
        source_id="wheel-sqp-task7-in-bounds-seam",
        hard_cell=hard_cell,
    )
    contact_x = 0.25 + PROFILE.body_length_m / 2.0 + 0.25
    raw_pose = PoseStateV2(
        contact_x
        + PROFILE.continuous_separation_epsilon_m
        + 1.0e-12
        - 0.499e-12,
        0.25 + 0.499e-12,
        0.499e-12,
    )
    declared_pose = canonicalize_unwrapped_pose_v2(raw_pose)
    request = replace(
        REQUEST,
        request_id="wheel-sqp-task7-in-bounds-seam",
        start_state=raw_pose,
        goal_state=declared_pose,
        terrain_snapshot=snapshot,
    )
    solved = SimpleNamespace(
        segments=(
            SimpleNamespace(
                start_state=raw_pose,
                end_state=raw_pose,
                v_mps=0.0,
                omega_radps=0.0,
                duration_s=0.05,
            ),
        )
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        solved,
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    segment = candidate.segments[0]
    raw_endpoint = integrate_wheel_segment_v2(
        segment.start_state,
        segment.v_mps,
        segment.omega_radps,
        segment.duration_s,
    )
    _task7_assert_seam_only(
        segment,
        oriented_rectangle_cell_separation_v2(
            raw_endpoint, hard_cell, geometry, PROFILE
        ),
        oriented_rectangle_cell_separation_v2(
            segment.end_state, hard_cell, geometry, PROFILE
        ),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is False
    assert result.reason_code == "wheel_sqp_candidate_l2_rejected"
    assert result.counterexample is not None
    assert result.counterexample.cell == hard_cell
    assert result.counterexample.terrain_reason == "terrain_hard_obstacle"
    assert (
        result.counterexample.time_fraction_lo,
        result.counterexample.time_fraction_mid,
        result.counterexample.time_fraction_hi,
    ) == (1.0, 1.0, 1.0)
    assert result.counterexample.repairable is False


def test_decimal12_left_oob_seam_requires_independent_clearance_proof() -> None:
    geometry = FineGridGeometryV2(4, 4, origin=(0.0, 0.0))
    snapshot = _task7_seam_snapshot(
        geometry,
        source_id="wheel-sqp-task7-left-oob-seam",
    )
    raw_pose = PoseStateV2(
        PROFILE.body_length_m / 2.0
        + PROFILE.continuous_separation_epsilon_m
        + 1.0e-12
        - 0.499e-12,
        1.0 + 0.499e-12,
        0.499e-12,
    )
    declared_pose = canonicalize_unwrapped_pose_v2(raw_pose)
    request = replace(
        REQUEST,
        request_id="wheel-sqp-task7-left-oob-seam",
        start_state=raw_pose,
        goal_state=declared_pose,
        terrain_snapshot=snapshot,
    )
    solved = SimpleNamespace(
        segments=(
            SimpleNamespace(
                start_state=raw_pose,
                end_state=raw_pose,
                v_mps=0.0,
                omega_radps=0.0,
                duration_s=0.05,
            ),
        )
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        solved,
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    segment = candidate.segments[0]
    raw_endpoint = integrate_wheel_segment_v2(
        segment.start_state,
        segment.v_mps,
        segment.omega_radps,
        segment.duration_s,
    )
    _task7_assert_seam_only(
        segment,
        wheel_sqp_validation_module._oriented_rectangle_boundary_separations_v2(
            raw_endpoint, geometry, PROFILE
        )[0],
        wheel_sqp_validation_module._oriented_rectangle_boundary_separations_v2(
            segment.end_state, geometry, PROFILE
        )[0],
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is False
    assert result.reason_code == "wheel_sqp_candidate_l2_rejected"
    assert result.counterexample is not None
    assert result.counterexample.cell == Cell(-1, 0)
    assert result.counterexample.terrain_reason == "terrain_out_of_bounds"
    assert (
        result.counterexample.time_fraction_lo,
        result.counterexample.time_fraction_mid,
        result.counterexample.time_fraction_hi,
    ) == (1.0, 1.0, 1.0)
    assert result.counterexample.repairable is False


def test_arc_midpoint_bottom_oob_uses_continuous_halfspace_proof() -> None:
    geometry = FineGridGeometryV2(4, 4, origin=(0.0, 0.0))
    snapshot = _task7_seam_snapshot(
        geometry,
        source_id="wheel-sqp-task7-arc-bottom-oob",
    )
    start = PoseStateV2(0.55, 0.45, -pi / 4.0)
    raw_goal = integrate_wheel_segment_v2(start, 0.5, pi / 4.0, 2.0)
    request = replace(
        REQUEST,
        request_id="wheel-sqp-task7-arc-bottom-oob",
        start_state=start,
        goal_state=raw_goal,
        terrain_snapshot=snapshot,
    )
    solved = SimpleNamespace(
        segments=(
            SimpleNamespace(
                start_state=start,
                end_state=raw_goal,
                v_mps=0.5,
                omega_radps=pi / 4.0,
                duration_s=2.0,
            ),
        )
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        solved,
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=snapshot_hash(snapshot),
    )
    deadline = PlanningDeadlineV2(0.0, 2.0, lambda: 0.0)
    ledger = _admitted_ledger(candidate, request, deadline)

    result = validate_wheel_sqp_candidate_l2(
        encode_wheel_candidate_v2(candidate),
        request,
        FineSafetyAnchorV2(snapshot),
        PROFILE,
        deadline,
        ledger,
    )

    assert result.passed is False
    assert result.reason_code == "wheel_sqp_candidate_l2_rejected"
    assert result.counterexample is not None
    assert result.counterexample.cell == Cell(-3, 0)
    assert result.counterexample.terrain_reason == "terrain_out_of_bounds"
    assert result.counterexample.time_fraction_mid == 0.5
    assert result.counterexample.repairable is False
