from dataclasses import dataclass, replace
import json
from math import copysign, pi

import numpy as np
import pytest

from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    ValidationLevelV2,
)
from path_planner.v2.profiles import PlatformProfileV2, WheelKinematicSQPProfileV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import FineGridGeometryV2, TerrainProvenanceV2, TerrainSnapshotV2
from path_planner.v2.wheel_kinematics import integrate_wheel_segment_v2
from path_planner.v2.wheel_sqp_contracts import WheelKinematicRouteV2, WheelKinematicSegmentV2
from path_planner.v2.wheel_sqp_serialization import (
    CanonicalWheelCandidateV1,
    WheelSQPCodecError,
    canonicalize_wheel_heading_v2,
    canonicalize_wheel_scalar_v2,
    decode_wheel_candidate_v2,
    decode_wheel_route_v2,
    encode_wheel_candidate_v2,
    encode_wheel_route_v2,
    materialize_canonical_wheel_candidate_v2,
    project_wheel_route_to_candidate_v1,
    wheel_candidate_hash_v2,
    wheel_route_hash_v2,
    wheel_segment_hash_v2,
)


SNAPSHOT_HASH = "e" * 64


def _profile() -> WheelKinematicSQPProfileV2:
    return WheelKinematicSQPProfileV2(
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


PROFILE = _profile()


def _snapshot() -> TerrainSnapshotV2:
    shape = (2, 4)
    return TerrainSnapshotV2(
        geometry=FineGridGeometryV2(width=4, height=2),
        elevation_m=np.zeros(shape),
        slope_deg=np.zeros(shape),
        traversable_mask=np.ones(shape, dtype=bool),
        hard_obstacle_mask=np.zeros(shape, dtype=bool),
        observed_mask=np.ones(shape, dtype=bool),
        confidence=np.ones(shape),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="wheel-sqp-task2-fixture",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )


def make_request(*, start: PoseStateV2 | None = None) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="wheel-sqp-task2",
        platform_profile_id=PROFILE.profile.profile_id,
        start_state=start or PoseStateV2(0.0, 0.0, 0.0),
        goal_state=PoseStateV2(1.5, 0.5, 0.4),
        terrain_snapshot=_snapshot(),
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(),
        timeout_s=2.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=19,
    )


REQUEST = make_request()


@dataclass(frozen=True)
class RawSegment:
    start_state: PoseStateV2
    end_state: PoseStateV2
    v_mps: float
    omega_radps: float
    duration_s: float


@dataclass(frozen=True)
class RawCandidate:
    segments: tuple[RawSegment, ...]


def _raw_candidate(request: PlanningRequestV2 = REQUEST) -> RawCandidate:
    first_end = integrate_wheel_segment_v2(request.start_state, 0.6, 0.2, 1.0)
    second_end = integrate_wheel_segment_v2(first_end, 0.0, -0.3, 0.5)
    return RawCandidate(
        segments=(
            RawSegment(request.start_state, first_end, 0.6, 0.2, 1.0),
            RawSegment(first_end, second_end, 0.0, -0.3, 0.5),
        )
    )


def make_canonical_candidate(
    request: PlanningRequestV2 = REQUEST,
) -> CanonicalWheelCandidateV1:
    return materialize_canonical_wheel_candidate_v2(
        _raw_candidate(request),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash=SNAPSHOT_HASH,
    )


def make_l2_public_route_fixture() -> WheelKinematicRouteV2:
    candidate = make_canonical_candidate()
    primitives = []
    for private in candidate.segments:
        public = WheelKinematicSegmentV2(
            kind=PrimitiveKindV2.WHEEL_MOTION,
            start_state=private.start_state,
            end_state=private.end_state,
            duration_s=private.duration_s,
            distance_m=private.distance_m,
            energy_cost=private.relative_energy,
            observation_contribution=0.0,
            validation_level=ValidationLevelV2.L2,
            v_mps=private.v_mps,
            omega_radps=private.omega_radps,
            reverse=private.v_mps < 0.0,
            turn_in_place=private.v_mps == 0.0 and private.omega_radps != 0.0,
            relative_energy=private.relative_energy,
            samples=private.samples,
            segment_hash="0" * 64,
        )
        primitives.append(replace(public, segment_hash=wheel_segment_hash_v2(public)))
    route = WheelKinematicRouteV2(
        platform_kind=PlatformKindV2.WHEEL,
        primitives=tuple(primitives),
        total_cost=candidate.total_cost,
        route_hash="0" * 64,
        source_candidate_hash=candidate.candidate_hash,
        request_hash=candidate.request_hash,
        profile_hash=candidate.profile_hash,
        terrain_snapshot_hash=candidate.terrain_snapshot_hash,
        capability_revision=candidate.capability_revision,
        solver_contract_id=candidate.solver_contract_id,
        canonicalization_id=candidate.canonicalization_id,
        validator_contract_id="wheel_kinematic_continuous_rectangle_sweep_l2/v1",
    )
    return replace(route, route_hash=wheel_route_hash_v2(route))


def _dump(payload: object, *, allow_nan: bool = False) -> bytes:
    return json.dumps(
        payload,
        allow_nan=allow_nan,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_canonicalization_is_half_even_decimal12_wrap_safe_and_zero_normalized() -> None:
    assert canonicalize_wheel_scalar_v2(-0.0) == 0.0
    assert copysign(1.0, canonicalize_wheel_scalar_v2(-0.0)) == 1.0
    assert canonicalize_wheel_scalar_v2(1.2345678901235) == 1.234567890124
    assert canonicalize_wheel_heading_v2(2.0 * pi) == 0.0
    assert canonicalize_wheel_heading_v2(-pi) == -pi


def test_materialization_preserves_exact_start_and_builds_unwrapped_canonical_seams() -> None:
    request = make_request(start=PoseStateV2(0.12345678901234567, 0.75, 2.0 * pi))
    candidate = make_canonical_candidate(request)
    assert candidate.start_state == request.start_state
    assert candidate.start_state.heading_rad == 2.0 * pi
    assert candidate.segments[0].start_state is request.start_state
    assert candidate.segments[0].end_state == candidate.segments[1].start_state
    assert candidate.actual_endpoint == candidate.segments[-1].end_state
    assert not hasattr(candidate, "validation_level")
    assert all(not hasattr(segment, "validation_level") for segment in candidate.segments)


def test_materialization_rejects_raw_residual_before_canonical_replay_can_replace_it() -> None:
    raw = _raw_candidate()
    bad = replace(raw.segments[0], end_state=replace(raw.segments[0].end_state, x_m=1.0))
    with pytest.raises(ValueError, match="residual"):
        materialize_canonical_wheel_candidate_v2(
            replace(raw, segments=(bad, raw.segments[1])),
            request=REQUEST,
            profile=PROFILE,
            terrain_snapshot_hash=SNAPSHOT_HASH,
        )


def test_materialization_rejects_request_profile_identity_mismatch() -> None:
    request = replace(REQUEST, platform_profile_id="some-other-profile/v1")
    with pytest.raises(ValueError, match="profile"):
        materialize_canonical_wheel_candidate_v2(
            _raw_candidate(request),
            request=request,
            profile=PROFILE,
            terrain_snapshot_hash=SNAPSHOT_HASH,
        )


def test_candidate_codec_round_trip_is_byte_stable_and_rejects_untrusted_variants() -> None:
    candidate = make_canonical_candidate()
    encoded = encode_wheel_candidate_v2(candidate)
    decoded = decode_wheel_candidate_v2(encoded)
    assert type(decoded) is CanonicalWheelCandidateV1
    assert decoded is not candidate
    assert encode_wheel_candidate_v2(decoded) == encoded
    assert decoded.candidate_hash == wheel_candidate_hash_v2(decoded)

    payload = json.loads(encoded)
    unknown = dict(payload, unknown=True)
    duplicate = encoded.replace(
        b"{",
        b'{"candidate_hash":"' + candidate.candidate_hash.encode() + b'",',
        1,
    )
    boolean_number = dict(payload, total_cost=True)
    nonfinite = dict(payload, total_cost=float("nan"))
    solver_drift = dict(
        payload,
        solver_contract_id="wheel_kinematic_direct_multiple_shooting_sqp/v2",
    )
    segment_drift = json.loads(encoded)
    segment_drift["segments"][0]["segment_hash"] = "f" * 64
    variants = (
        _dump(unknown),
        duplicate,
        _dump(boolean_number),
        _dump(nonfinite, allow_nan=True),
        _dump(solver_drift),
        _dump(segment_drift),
        b"\xef\xbb\xbf" + encoded,
        b"\xff" + encoded,
    )
    for mutated in variants:
        with pytest.raises(WheelSQPCodecError):
            decode_wheel_candidate_v2(mutated)


def test_candidate_decoder_rederives_segment_totals_even_when_attacker_rehashes() -> None:
    candidate = make_canonical_candidate()
    drifted = replace(
        candidate,
        total_distance_m=candidate.total_distance_m + 1.0,
        candidate_hash="0" * 64,
    )
    drifted = replace(drifted, candidate_hash=wheel_candidate_hash_v2(drifted))
    with pytest.raises(WheelSQPCodecError, match="total_distance_m"):
        decode_wheel_candidate_v2(_dump(json.loads(canonical_json_bytes(drifted))))


def test_public_route_codec_is_byte_stable_and_accepts_only_exact_l2_route() -> None:
    route = make_l2_public_route_fixture()
    encoded = encode_wheel_route_v2(route)
    decoded = decode_wheel_route_v2(encoded)
    assert type(decoded) is WheelKinematicRouteV2
    assert decoded is not route
    assert encode_wheel_route_v2(decoded) == encoded
    assert decoded.route_hash == wheel_route_hash_v2(decoded)
    projected = project_wheel_route_to_candidate_v1(decoded, REQUEST, PROFILE, SNAPSHOT_HASH)
    assert wheel_candidate_hash_v2(projected) == decoded.source_candidate_hash
    assert all(p.segment_hash == wheel_segment_hash_v2(p) for p in decoded.primitives)

    payload = json.loads(encoded)
    payload["primitives"][0]["validation_level"] = "L1"
    with pytest.raises(WheelSQPCodecError):
        decode_wheel_route_v2(_dump(payload))


def test_hashes_exclude_only_their_own_digest_and_bind_semantic_payloads() -> None:
    candidate = make_canonical_candidate()
    segment = candidate.segments[0]
    assert (
        wheel_segment_hash_v2(replace(segment, segment_hash="a" * 64))
        == segment.segment_hash
    )
    changed_samples = list(segment.samples)
    changed_samples[1] = replace(
        changed_samples[1],
        x_m=changed_samples[1].x_m + 1.0e-12,
    )
    assert (
        wheel_segment_hash_v2(replace(segment, samples=tuple(changed_samples)))
        != segment.segment_hash
    )
    assert (
        wheel_candidate_hash_v2(replace(candidate, candidate_hash="a" * 64))
        == candidate.candidate_hash
    )
    assert (
        wheel_candidate_hash_v2(
            replace(candidate, total_cost=candidate.total_cost + 1.0)
        )
        != candidate.candidate_hash
    )

    route = make_l2_public_route_fixture()
    assert wheel_route_hash_v2(replace(route, route_hash="a" * 64)) == route.route_hash
    for field in (
        "source_candidate_hash",
        "request_hash",
        "profile_hash",
        "terrain_snapshot_hash",
    ):
        changed = replace(route, **{field: "1" * 64})
        assert wheel_route_hash_v2(changed) != route.route_hash
    assert (
        wheel_route_hash_v2(replace(route, total_cost=route.total_cost + 1.0))
        != route.route_hash
    )


def test_projection_rejects_public_route_semantic_drift_instead_of_granting_candidate_trust() -> None:
    route = make_l2_public_route_fixture()
    first = route.primitives[0]
    drifted = replace(
        first,
        relative_energy=first.relative_energy + 0.1,
        energy_cost=first.energy_cost + 0.1,
    )
    drifted = replace(drifted, segment_hash=wheel_segment_hash_v2(drifted))
    changed_route = replace(
        route,
        primitives=(drifted, *route.primitives[1:]),
        route_hash="0" * 64,
    )
    changed_route = replace(changed_route, route_hash=wheel_route_hash_v2(changed_route))
    with pytest.raises(WheelSQPCodecError, match="energy"):
        project_wheel_route_to_candidate_v1(changed_route, REQUEST, PROFILE, SNAPSHOT_HASH)
