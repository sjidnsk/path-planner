from __future__ import annotations

from dataclasses import dataclass, fields, replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from hashlib import sha256
import json
from math import isfinite, pi
from numbers import Real
from typing import Any

from path_planner.v2.contracts import (
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ValidationLevelV2,
)
from path_planner.v2.profiles import (
    PlatformProfileV2,
    WheelKinematicSQPProfileV2,
)
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.wheel_kinematics import (
    integrate_wheel_segment_v2,
    sample_wheel_segment_v2,
    wheel_relative_energy_v1,
)
from path_planner.v2.wheel_sqp_contracts import (
    WHEEL_KINEMATIC_CANONICALIZATION_V2,
    WHEEL_KINEMATIC_CONTROL_SLEW_V2,
    WHEEL_KINEMATIC_L2_VALIDATOR_V2,
    WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2,
    WHEEL_KINEMATIC_SEGMENT_SCHEMA_V2,
    WHEEL_KINEMATIC_SOLVER_CONTRACT_V2,
    WheelKinematicRouteV2,
    WheelKinematicSegmentV2,
    WheelSQPModeV2,
)


_CAPABILITY_REVISION = "wheel_kinematic_corridor_sqp/v1"
_ZERO_HASH = "0" * 64
_DECIMAL_QUANTUM = Decimal("1e-12")
_CANDIDATE_KEYS = frozenset(
    {
        "actual_endpoint",
        "candidate_hash",
        "canonicalization_id",
        "capability_revision",
        "control_slew_id",
        "observation_source_id",
        "profile_hash",
        "request_hash",
        "requested_goal_state",
        "segments",
        "solver_contract_id",
        "start_state",
        "terrain_snapshot_hash",
        "total_cost",
        "total_distance_m",
        "total_duration_s",
        "total_relative_energy",
    }
)
_PRIVATE_SEGMENT_KEYS = frozenset(
    {
        "distance_m",
        "duration_s",
        "end_state",
        "mode",
        "omega_radps",
        "relative_energy",
        "samples",
        "segment_hash",
        "start_state",
        "v_mps",
    }
)
_POSE_KEYS = frozenset({"heading_rad", "x_m", "y_m"})


class WheelSQPCodecError(ValueError):
    pass


def _exact_hash(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact str")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_id(value: object, name: str, expected: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact str")
    if value != expected:
        raise ValueError(f"{name} must be {expected}")
    return value


def _exact_float(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be exact float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _exact_pose(value: object, name: str) -> PoseStateV2:
    if type(value) is not PoseStateV2:
        raise TypeError(f"{name} must be exact PoseStateV2")
    return value


def _mode_for(v_mps: float, omega_radps: float) -> WheelSQPModeV2:
    if v_mps > 0.0:
        return WheelSQPModeV2.FORWARD
    if v_mps < 0.0:
        return WheelSQPModeV2.REVERSE
    if omega_radps > 0.0:
        return WheelSQPModeV2.TURN_LEFT
    if omega_radps < 0.0:
        return WheelSQPModeV2.TURN_RIGHT
    return WheelSQPModeV2.STOP


@dataclass(frozen=True, slots=True)
class CanonicalWheelSegmentV1:
    start_state: PoseStateV2
    end_state: PoseStateV2
    v_mps: float
    omega_radps: float
    duration_s: float
    mode: WheelSQPModeV2
    distance_m: float
    relative_energy: float
    samples: tuple[PoseStateV2, ...]
    segment_hash: str

    def __post_init__(self) -> None:
        _exact_pose(self.start_state, "start_state")
        _exact_pose(self.end_state, "end_state")
        v = _exact_float(self.v_mps, "v_mps")
        omega = _exact_float(self.omega_radps, "omega_radps")
        _exact_float(self.duration_s, "duration_s", positive=True)
        if type(self.mode) is not WheelSQPModeV2 or self.mode is not _mode_for(
            v,
            omega,
        ):
            raise ValueError("mode must exactly match canonical controls")
        _exact_float(self.distance_m, "distance_m", nonnegative=True)
        _exact_float(self.relative_energy, "relative_energy", nonnegative=True)
        if type(self.samples) is not tuple or not self.samples:
            raise TypeError("samples must be a nonempty exact tuple")
        if any(type(sample) is not PoseStateV2 for sample in self.samples):
            raise TypeError("samples must contain exact PoseStateV2 values")
        if self.samples[0] != self.start_state or self.samples[-1] != self.end_state:
            raise ValueError("samples must begin and end at segment states")
        _exact_hash(self.segment_hash, "segment_hash")


@dataclass(frozen=True, slots=True)
class CanonicalWheelCandidateV1:
    start_state: PoseStateV2
    requested_goal_state: PoseStateV2
    actual_endpoint: PoseStateV2
    segments: tuple[CanonicalWheelSegmentV1, ...]
    total_distance_m: float
    total_relative_energy: float
    total_duration_s: float
    total_cost: float
    request_hash: str
    profile_hash: str
    terrain_snapshot_hash: str
    capability_revision: str
    solver_contract_id: str
    canonicalization_id: str
    control_slew_id: str
    observation_source_id: str
    candidate_hash: str

    def __post_init__(self) -> None:
        _exact_pose(self.start_state, "start_state")
        _exact_pose(self.requested_goal_state, "requested_goal_state")
        _exact_pose(self.actual_endpoint, "actual_endpoint")
        if type(self.segments) is not tuple or not self.segments:
            raise TypeError("segments must be a nonempty exact tuple")
        if any(
            type(segment) is not CanonicalWheelSegmentV1
            for segment in self.segments
        ):
            raise TypeError("segments must contain exact CanonicalWheelSegmentV1 values")
        if self.segments[0].start_state != self.start_state:
            raise ValueError("candidate start must match first segment")
        if self.segments[-1].end_state != self.actual_endpoint:
            raise ValueError("actual_endpoint must match final segment")
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if previous.end_state != current.start_state:
                raise ValueError("candidate segment endpoints must be connected")
        for name in (
            "total_distance_m",
            "total_relative_energy",
            "total_duration_s",
            "total_cost",
        ):
            _exact_float(getattr(self, name), name, nonnegative=True)
        for name in (
            "request_hash",
            "profile_hash",
            "terrain_snapshot_hash",
            "candidate_hash",
        ):
            _exact_hash(getattr(self, name), name)
        _exact_id(self.capability_revision, "capability_revision", _CAPABILITY_REVISION)
        _exact_id(
            self.solver_contract_id,
            "solver_contract_id",
            WHEEL_KINEMATIC_SOLVER_CONTRACT_V2,
        )
        _exact_id(
            self.canonicalization_id,
            "canonicalization_id",
            WHEEL_KINEMATIC_CANONICALIZATION_V2,
        )
        _exact_id(
            self.control_slew_id,
            "control_slew_id",
            WHEEL_KINEMATIC_CONTROL_SLEW_V2,
        )
        _exact_id(
            self.observation_source_id,
            "observation_source_id",
            WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2,
        )


def canonicalize_wheel_scalar_v2(value: float) -> float:
    normalized = _finite(value, "value")
    try:
        with localcontext() as context:
            context.prec = 400
            decimal_value = Decimal(format(normalized, ".17g"))
            rounded = decimal_value.quantize(_DECIMAL_QUANTUM, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        raise ValueError("value cannot be canonicalized to decimal12") from exc
    result = float(rounded)
    if not isfinite(result):
        raise ValueError("canonical value must be finite")
    return 0.0 if result == 0.0 else result


def canonicalize_positive_duration_v2(value: float) -> float:
    result = canonicalize_wheel_scalar_v2(value)
    if result <= 0.0:
        raise ValueError("duration_s must remain positive after canonicalization")
    return result


def canonicalize_wheel_heading_v2(value: float) -> float:
    normalized = _finite(value, "heading_rad")
    wrapped = (normalized + pi) % (2.0 * pi) - pi
    return 0.0 if wrapped == 0.0 else wrapped


def canonicalize_unwrapped_pose_v2(value: PoseStateV2) -> PoseStateV2:
    value = _exact_pose(value, "pose")
    return PoseStateV2(
        canonicalize_wheel_scalar_v2(value.x_m),
        canonicalize_wheel_scalar_v2(value.y_m),
        canonicalize_wheel_scalar_v2(value.heading_rad),
    )


def _pose_residual(left: PoseStateV2, right: PoseStateV2) -> float:
    return max(
        abs(left.x_m - right.x_m),
        abs(left.y_m - right.y_m),
        abs(canonicalize_wheel_heading_v2(left.heading_rad - right.heading_rad)),
    )


def _require_pose_residual(left: PoseStateV2, right: PoseStateV2, tolerance: float) -> None:
    if _pose_residual(left, right) > tolerance:
        raise ValueError("wheel segment pose residual exceeds hard_constraint_tolerance")


def _hash_without(value: object, own_field: str) -> str:
    payload = {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name != own_field
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def wheel_segment_hash_v2(segment: CanonicalWheelSegmentV1 | WheelKinematicSegmentV2) -> str:
    if type(segment) not in (CanonicalWheelSegmentV1, WheelKinematicSegmentV2):
        raise TypeError("segment must be an exact canonical or public wheel segment")
    return _hash_without(segment, "segment_hash")


def wheel_candidate_hash_v2(candidate: CanonicalWheelCandidateV1) -> str:
    if type(candidate) is not CanonicalWheelCandidateV1:
        raise TypeError("candidate must be exact CanonicalWheelCandidateV1")
    return _hash_without(candidate, "candidate_hash")


def wheel_route_hash_v2(route: WheelKinematicRouteV2) -> str:
    if type(route) is not WheelKinematicRouteV2:
        raise TypeError("route must be exact WheelKinematicRouteV2")
    return _hash_without(route, "route_hash")


def _request_hash_v2(request: PlanningRequestV2, terrain_snapshot_hash: str) -> str:
    payload = {
        "accelerator_policy": request.accelerator_policy,
        "determinism_seed": request.determinism_seed,
        "goal_state": request.goal_state,
        "objective_profile": request.objective_profile,
        "platform_profile_id": request.platform_profile_id,
        "request_id": request.request_id,
        "resource_budget": request.resource_budget,
        "start_state": request.start_state,
        "terrain_snapshot_hash": terrain_snapshot_hash,
        "timeout_s": request.timeout_s,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _profile_hash_v2(profile: WheelKinematicSQPProfileV2) -> str:
    return sha256(canonical_json_bytes(profile)).hexdigest()


def _canonical_samples(
    start: PoseStateV2,
    v_mps: float,
    omega_radps: float,
    duration_s: float,
    profile: WheelKinematicSQPProfileV2,
    end: PoseStateV2,
) -> tuple[PoseStateV2, ...]:
    derived = sample_wheel_segment_v2(start, v_mps, omega_radps, duration_s, profile)
    middle = tuple(canonicalize_unwrapped_pose_v2(sample) for sample in derived[1:-1])
    return (start, *middle, end)


def _segment_from_values(
    *,
    start: PoseStateV2,
    end: PoseStateV2,
    v_mps: float,
    omega_radps: float,
    duration_s: float,
    profile: WheelKinematicSQPProfileV2,
) -> CanonicalWheelSegmentV1:
    segment = CanonicalWheelSegmentV1(
        start_state=start,
        end_state=end,
        v_mps=v_mps,
        omega_radps=omega_radps,
        duration_s=duration_s,
        mode=_mode_for(v_mps, omega_radps),
        distance_m=canonicalize_wheel_scalar_v2(abs(v_mps) * duration_s),
        relative_energy=canonicalize_wheel_scalar_v2(
            wheel_relative_energy_v1(v_mps, omega_radps, duration_s, profile)
        ),
        samples=_canonical_samples(start, v_mps, omega_radps, duration_s, profile, end),
        segment_hash=_ZERO_HASH,
    )
    return replace(segment, segment_hash=wheel_segment_hash_v2(segment))


def _weighted_cost(
    segments: tuple[CanonicalWheelSegmentV1, ...], request: PlanningRequestV2
) -> tuple[float, float, float, float]:
    distance = canonicalize_wheel_scalar_v2(sum(segment.distance_m for segment in segments))
    energy = canonicalize_wheel_scalar_v2(sum(segment.relative_energy for segment in segments))
    duration = canonicalize_wheel_scalar_v2(sum(segment.duration_s for segment in segments))
    objective = request.objective_profile
    total = canonicalize_wheel_scalar_v2(
        objective.distance_weight * distance
        + objective.energy_weight * energy
        + objective.time_weight * duration
    )
    return distance, energy, duration, total


def materialize_canonical_wheel_candidate_v2(
    candidate: object,
    *,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    terrain_snapshot_hash: str,
) -> CanonicalWheelCandidateV1:
    if type(request) is not PlanningRequestV2:
        raise TypeError("request must be exact PlanningRequestV2")
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
    if request.platform_profile_id != profile.profile.profile_id:
        raise ValueError("request and wheel SQP profile identity mismatch")
    snapshot_hash = _exact_hash(terrain_snapshot_hash, "terrain_snapshot_hash")
    raw_segments = getattr(candidate, "segments", None)
    if type(raw_segments) is not tuple or not raw_segments:
        raise TypeError("candidate.segments must be a nonempty exact tuple")

    current = request.start_state
    raw_current = request.start_state
    canonical_segments: list[CanonicalWheelSegmentV1] = []
    for raw in raw_segments:
        raw_start = _exact_pose(getattr(raw, "start_state", None), "raw.start_state")
        raw_end = _exact_pose(getattr(raw, "end_state", None), "raw.end_state")
        raw_v = _finite(getattr(raw, "v_mps", None), "raw.v_mps")
        raw_omega = _finite(getattr(raw, "omega_radps", None), "raw.omega_radps")
        raw_dt = _finite(getattr(raw, "duration_s", None), "raw.duration_s")
        if raw_dt <= 0.0:
            raise ValueError("raw.duration_s must be positive")
        _require_pose_residual(raw_start, raw_current, profile.hard_constraint_tolerance)
        raw_replay = integrate_wheel_segment_v2(raw_current, raw_v, raw_omega, raw_dt)
        _require_pose_residual(raw_end, raw_replay, profile.hard_constraint_tolerance)

        v = canonicalize_wheel_scalar_v2(raw_v)
        omega = canonicalize_wheel_scalar_v2(raw_omega)
        duration = canonicalize_positive_duration_v2(raw_dt)
        declared_end = canonicalize_unwrapped_pose_v2(raw_end)
        replay_end = canonicalize_unwrapped_pose_v2(
            integrate_wheel_segment_v2(current, v, omega, duration)
        )
        _require_pose_residual(declared_end, replay_end, profile.hard_constraint_tolerance)
        canonical_segments.append(
            _segment_from_values(
                start=current,
                end=replay_end,
                v_mps=v,
                omega_radps=omega,
                duration_s=duration,
                profile=profile,
            )
        )
        current = replay_end
        raw_current = raw_end

    segments = tuple(canonical_segments)
    distance, energy, duration, total = _weighted_cost(segments, request)
    result = CanonicalWheelCandidateV1(
        start_state=request.start_state,
        requested_goal_state=request.goal_state,
        actual_endpoint=current,
        segments=segments,
        total_distance_m=distance,
        total_relative_energy=energy,
        total_duration_s=duration,
        total_cost=total,
        request_hash=_request_hash_v2(request, snapshot_hash),
        profile_hash=_profile_hash_v2(profile),
        terrain_snapshot_hash=snapshot_hash,
        capability_revision=_CAPABILITY_REVISION,
        solver_contract_id=WHEEL_KINEMATIC_SOLVER_CONTRACT_V2,
        canonicalization_id=WHEEL_KINEMATIC_CANONICALIZATION_V2,
        control_slew_id=WHEEL_KINEMATIC_CONTROL_SLEW_V2,
        observation_source_id=WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2,
        candidate_hash=_ZERO_HASH,
    )
    return replace(result, candidate_hash=wheel_candidate_hash_v2(result))


_CODEC_PROFILE = WheelKinematicSQPProfileV2(
    profile=PlatformProfileV2(
        profile_id="scout-mini-wheel-kinematic-sqp/v1",
        platform_kind=PlatformKindV2.WHEEL,
        capability_revision=_CAPABILITY_REVISION,
        simulation_proxy=False,
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=0.25,
        goal_heading_tolerance_rad=0.08726646259971647,
    )
)


def _validate_private_segment(segment: CanonicalWheelSegmentV1) -> None:
    if canonicalize_wheel_scalar_v2(segment.v_mps) != segment.v_mps:
        raise ValueError("v_mps is not canonical")
    if canonicalize_wheel_scalar_v2(segment.omega_radps) != segment.omega_radps:
        raise ValueError("omega_radps is not canonical")
    if canonicalize_positive_duration_v2(segment.duration_s) != segment.duration_s:
        raise ValueError("duration_s is not canonical")
    replay = canonicalize_unwrapped_pose_v2(
        integrate_wheel_segment_v2(
            segment.start_state,
            segment.v_mps,
            segment.omega_radps,
            segment.duration_s,
        )
    )
    if replay != segment.end_state:
        raise ValueError("segment endpoint does not match analytic replay")
    expected = _segment_from_values(
        start=segment.start_state,
        end=replay,
        v_mps=segment.v_mps,
        omega_radps=segment.omega_radps,
        duration_s=segment.duration_s,
        profile=_CODEC_PROFILE,
    )
    for name in ("mode", "distance_m", "relative_energy", "samples"):
        if getattr(segment, name) != getattr(expected, name):
            raise ValueError(f"segment {name} does not match derived value")
    if wheel_segment_hash_v2(segment) != segment.segment_hash:
        raise ValueError("segment_hash does not match segment payload")


def _validate_candidate(candidate: CanonicalWheelCandidateV1) -> None:
    for segment in candidate.segments:
        _validate_private_segment(segment)
    expected_totals = {
        "total_distance_m": canonicalize_wheel_scalar_v2(
            sum(segment.distance_m for segment in candidate.segments)
        ),
        "total_relative_energy": canonicalize_wheel_scalar_v2(
            sum(segment.relative_energy for segment in candidate.segments)
        ),
        "total_duration_s": canonicalize_wheel_scalar_v2(
            sum(segment.duration_s for segment in candidate.segments)
        ),
    }
    for name, expected in expected_totals.items():
        if getattr(candidate, name) != expected:
            raise ValueError(f"{name} does not match canonical segment total")
    if candidate.candidate_hash != wheel_candidate_hash_v2(candidate):
        raise ValueError("candidate_hash does not match candidate payload")


def _validate_public_segment(segment: WheelKinematicSegmentV2) -> None:
    if segment.validation_level is not ValidationLevelV2.L2:
        raise ValueError("public wheel segment must already be exact L2")
    if canonicalize_wheel_scalar_v2(segment.v_mps) != segment.v_mps:
        raise ValueError("public v_mps is not canonical")
    if canonicalize_wheel_scalar_v2(segment.omega_radps) != segment.omega_radps:
        raise ValueError("public omega_radps is not canonical")
    if canonicalize_positive_duration_v2(segment.duration_s) != segment.duration_s:
        raise ValueError("public duration_s is not canonical")
    end = canonicalize_unwrapped_pose_v2(
        integrate_wheel_segment_v2(
            segment.start_state,
            segment.v_mps,
            segment.omega_radps,
            segment.duration_s,
        )
    )
    expected = _segment_from_values(
        start=segment.start_state,
        end=end,
        v_mps=segment.v_mps,
        omega_radps=segment.omega_radps,
        duration_s=segment.duration_s,
        profile=_CODEC_PROFILE,
    )
    if segment.end_state != end:
        raise ValueError("public segment endpoint does not match analytic replay")
    if segment.distance_m != expected.distance_m:
        raise ValueError("public segment distance does not match derived value")
    if (
        segment.relative_energy != expected.relative_energy
        or segment.energy_cost != expected.relative_energy
    ):
        raise ValueError("public segment energy does not match derived value")
    if segment.samples != expected.samples:
        raise ValueError("public segment samples do not match derived value")
    if segment.segment_hash != wheel_segment_hash_v2(segment):
        raise ValueError("segment_hash does not match public segment payload")


def _validate_route(route: WheelKinematicRouteV2) -> None:
    if route.is_complete is not True:
        raise ValueError("public wheel route must be complete")
    for segment in route.primitives:
        _validate_public_segment(segment)
    if route.route_hash != wheel_route_hash_v2(route):
        raise ValueError("route_hash does not match route payload")


def encode_wheel_candidate_v2(candidate: CanonicalWheelCandidateV1) -> bytes:
    if type(candidate) is not CanonicalWheelCandidateV1:
        raise WheelSQPCodecError("candidate must be exact CanonicalWheelCandidateV1")
    try:
        _validate_candidate(candidate)
        return canonical_json_bytes(candidate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WheelSQPCodecError(str(exc)) from exc


def encode_wheel_route_v2(route: WheelKinematicRouteV2) -> bytes:
    if type(route) is not WheelKinematicRouteV2:
        raise WheelSQPCodecError("route must be exact WheelKinematicRouteV2")
    try:
        _validate_route(route)
        return canonical_json_bytes(route)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WheelSQPCodecError(str(exc)) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WheelSQPCodecError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise WheelSQPCodecError(f"non-finite JSON constant: {value}")


def _load_strict(encoded: bytes) -> dict[str, Any]:
    if type(encoded) is not bytes:
        raise WheelSQPCodecError("encoded payload must be exact bytes")
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise WheelSQPCodecError("UTF-8 BOM is forbidden")
    try:
        text = encoded.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WheelSQPCodecError("payload must be strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise WheelSQPCodecError("top-level payload must be an object")
    return value


def _keys(value: object, expected: frozenset[str], name: str) -> dict[str, Any]:
    if (
        type(value) is not dict
        or frozenset(value) != expected
        or len(value) != len(expected)
    ):
        raise WheelSQPCodecError(f"{name} must contain the exact key set")
    return value


def _json_float(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise WheelSQPCodecError(f"{name} must be an exact finite JSON float")
    return value


def _json_str(value: object, name: str) -> str:
    if type(value) is not str:
        raise WheelSQPCodecError(f"{name} must be exact JSON string")
    return value


def _decode_pose(value: object, name: str) -> PoseStateV2:
    payload = _keys(value, _POSE_KEYS, name)
    return PoseStateV2(
        _json_float(payload["x_m"], f"{name}.x_m"),
        _json_float(payload["y_m"], f"{name}.y_m"),
        _json_float(payload["heading_rad"], f"{name}.heading_rad"),
    )


def _decode_pose_list(value: object, name: str) -> tuple[PoseStateV2, ...]:
    if type(value) is not list or not value:
        raise WheelSQPCodecError(f"{name} must be a nonempty exact JSON array")
    return tuple(
        _decode_pose(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _decode_private_segment(value: object, index: int) -> CanonicalWheelSegmentV1:
    payload = _keys(value, _PRIVATE_SEGMENT_KEYS, f"segments[{index}]")
    try:
        mode = WheelSQPModeV2(_json_str(payload["mode"], "mode"))
    except ValueError as exc:
        raise WheelSQPCodecError("mode is not a contract value") from exc
    return CanonicalWheelSegmentV1(
        start_state=_decode_pose(payload["start_state"], "start_state"),
        end_state=_decode_pose(payload["end_state"], "end_state"),
        v_mps=_json_float(payload["v_mps"], "v_mps"),
        omega_radps=_json_float(payload["omega_radps"], "omega_radps"),
        duration_s=_json_float(payload["duration_s"], "duration_s"),
        mode=mode,
        distance_m=_json_float(payload["distance_m"], "distance_m"),
        relative_energy=_json_float(payload["relative_energy"], "relative_energy"),
        samples=_decode_pose_list(payload["samples"], "samples"),
        segment_hash=_json_str(payload["segment_hash"], "segment_hash"),
    )


def decode_wheel_candidate_v2(encoded: bytes) -> CanonicalWheelCandidateV1:
    try:
        payload = _keys(_load_strict(encoded), _CANDIDATE_KEYS, "candidate")
        raw_segments = payload["segments"]
        if type(raw_segments) is not list or not raw_segments:
            raise WheelSQPCodecError("segments must be a nonempty exact JSON array")
        candidate = CanonicalWheelCandidateV1(
            start_state=_decode_pose(payload["start_state"], "start_state"),
            requested_goal_state=_decode_pose(
                payload["requested_goal_state"],
                "requested_goal_state",
            ),
            actual_endpoint=_decode_pose(payload["actual_endpoint"], "actual_endpoint"),
            segments=tuple(
                _decode_private_segment(value, index)
                for index, value in enumerate(raw_segments)
            ),
            total_distance_m=_json_float(
                payload["total_distance_m"], "total_distance_m"
            ),
            total_relative_energy=_json_float(
                payload["total_relative_energy"], "total_relative_energy"
            ),
            total_duration_s=_json_float(payload["total_duration_s"], "total_duration_s"),
            total_cost=_json_float(payload["total_cost"], "total_cost"),
            request_hash=_json_str(payload["request_hash"], "request_hash"),
            profile_hash=_json_str(payload["profile_hash"], "profile_hash"),
            terrain_snapshot_hash=_json_str(
                payload["terrain_snapshot_hash"], "terrain_snapshot_hash"
            ),
            capability_revision=_json_str(
                payload["capability_revision"], "capability_revision"
            ),
            solver_contract_id=_json_str(
                payload["solver_contract_id"], "solver_contract_id"
            ),
            canonicalization_id=_json_str(
                payload["canonicalization_id"], "canonicalization_id"
            ),
            control_slew_id=_json_str(payload["control_slew_id"], "control_slew_id"),
            observation_source_id=_json_str(
                payload["observation_source_id"], "observation_source_id"
            ),
            candidate_hash=_json_str(payload["candidate_hash"], "candidate_hash"),
        )
        _validate_candidate(candidate)
        if canonical_json_bytes(candidate) != encoded:
            raise WheelSQPCodecError("candidate encoding is not canonical byte form")
        return candidate
    except WheelSQPCodecError:
        raise
    except (TypeError, ValueError, OverflowError, KeyError) as exc:
        raise WheelSQPCodecError(str(exc)) from exc


_PUBLIC_SEGMENT_KEYS = frozenset(field.name for field in fields(WheelKinematicSegmentV2))
_ROUTE_KEYS = frozenset(field.name for field in fields(WheelKinematicRouteV2))


def _decode_public_segment(value: object, index: int) -> WheelKinematicSegmentV2:
    payload = _keys(value, _PUBLIC_SEGMENT_KEYS, f"primitives[{index}]")
    try:
        kind = PrimitiveKindV2(_json_str(payload["kind"], "kind"))
        level = ValidationLevelV2(
            _json_str(payload["validation_level"], "validation_level")
        )
    except ValueError as exc:
        raise WheelSQPCodecError("public segment enum value is invalid") from exc
    reverse = payload["reverse"]
    turn = payload["turn_in_place"]
    if type(reverse) is not bool or type(turn) is not bool:
        raise WheelSQPCodecError("public segment flags must be exact JSON bool")
    return WheelKinematicSegmentV2(
        kind=kind,
        start_state=_decode_pose(payload["start_state"], "start_state"),
        end_state=_decode_pose(payload["end_state"], "end_state"),
        duration_s=_json_float(payload["duration_s"], "duration_s"),
        distance_m=_json_float(payload["distance_m"], "distance_m"),
        energy_cost=_json_float(payload["energy_cost"], "energy_cost"),
        observation_contribution=_json_float(
            payload["observation_contribution"],
            "observation_contribution",
        ),
        validation_level=level,
        v_mps=_json_float(payload["v_mps"], "v_mps"),
        omega_radps=_json_float(payload["omega_radps"], "omega_radps"),
        reverse=reverse,
        turn_in_place=turn,
        relative_energy=_json_float(payload["relative_energy"], "relative_energy"),
        samples=_decode_pose_list(payload["samples"], "samples"),
        segment_hash=_json_str(payload["segment_hash"], "segment_hash"),
        segment_schema_id=_json_str(payload["segment_schema_id"], "segment_schema_id"),
    )


def decode_wheel_route_v2(encoded: bytes) -> WheelKinematicRouteV2:
    try:
        payload = _keys(_load_strict(encoded), _ROUTE_KEYS, "route")
        raw_primitives = payload["primitives"]
        if type(raw_primitives) is not list or not raw_primitives:
            raise WheelSQPCodecError("primitives must be a nonempty exact JSON array")
        try:
            platform_kind = PlatformKindV2(
                _json_str(payload["platform_kind"], "platform_kind")
            )
        except ValueError as exc:
            raise WheelSQPCodecError("platform_kind is not a contract value") from exc
        if type(payload["is_complete"]) is not bool:
            raise WheelSQPCodecError("is_complete must be exact JSON bool")
        route = WheelKinematicRouteV2(
            platform_kind=platform_kind,
            primitives=tuple(
                _decode_public_segment(value, index)
                for index, value in enumerate(raw_primitives)
            ),
            total_cost=_json_float(payload["total_cost"], "total_cost"),
            is_complete=payload["is_complete"],
            route_hash=_json_str(payload["route_hash"], "route_hash"),
            source_candidate_hash=_json_str(
                payload["source_candidate_hash"], "source_candidate_hash"
            ),
            request_hash=_json_str(payload["request_hash"], "request_hash"),
            profile_hash=_json_str(payload["profile_hash"], "profile_hash"),
            terrain_snapshot_hash=_json_str(
                payload["terrain_snapshot_hash"], "terrain_snapshot_hash"
            ),
            capability_revision=_json_str(
                payload["capability_revision"], "capability_revision"
            ),
            solver_contract_id=_json_str(payload["solver_contract_id"], "solver_contract_id"),
            canonicalization_id=_json_str(
                payload["canonicalization_id"], "canonicalization_id"
            ),
            validator_contract_id=_json_str(
                payload["validator_contract_id"], "validator_contract_id"
            ),
        )
        _validate_route(route)
        if canonical_json_bytes(route) != encoded:
            raise WheelSQPCodecError("route encoding is not canonical byte form")
        return route
    except WheelSQPCodecError:
        raise
    except (TypeError, ValueError, OverflowError, KeyError) as exc:
        raise WheelSQPCodecError(str(exc)) from exc


def project_wheel_route_to_candidate_v1(
    route: WheelKinematicRouteV2,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    terrain_snapshot_hash: str,
) -> CanonicalWheelCandidateV1:
    try:
        if type(route) is not WheelKinematicRouteV2:
            raise TypeError("route must be exact WheelKinematicRouteV2")
        if type(request) is not PlanningRequestV2:
            raise TypeError("request must be exact PlanningRequestV2")
        if type(profile) is not WheelKinematicSQPProfileV2:
            raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
        snapshot_hash = _exact_hash(terrain_snapshot_hash, "terrain_snapshot_hash")
        _validate_route(route)
        expected_identities = {
            "request_hash": _request_hash_v2(request, snapshot_hash),
            "profile_hash": _profile_hash_v2(profile),
            "terrain_snapshot_hash": snapshot_hash,
            "capability_revision": _CAPABILITY_REVISION,
            "solver_contract_id": WHEEL_KINEMATIC_SOLVER_CONTRACT_V2,
            "canonicalization_id": WHEEL_KINEMATIC_CANONICALIZATION_V2,
            "validator_contract_id": WHEEL_KINEMATIC_L2_VALIDATOR_V2,
        }
        for name, expected in expected_identities.items():
            if getattr(route, name) != expected:
                raise ValueError(f"route {name} identity mismatch")
        if route.primitives[0].start_state != request.start_state:
            raise ValueError("route start does not equal exact request start")
        segments = tuple(
            _segment_from_values(
                start=segment.start_state,
                end=segment.end_state,
                v_mps=segment.v_mps,
                omega_radps=segment.omega_radps,
                duration_s=segment.duration_s,
                profile=profile,
            )
            for segment in route.primitives
        )
        distance, energy, duration, total = _weighted_cost(segments, request)
        if route.total_cost != total:
            raise ValueError("route total cost does not match request-relative cost")
        candidate = CanonicalWheelCandidateV1(
            start_state=request.start_state,
            requested_goal_state=request.goal_state,
            actual_endpoint=segments[-1].end_state,
            segments=segments,
            total_distance_m=distance,
            total_relative_energy=energy,
            total_duration_s=duration,
            total_cost=total,
            request_hash=route.request_hash,
            profile_hash=route.profile_hash,
            terrain_snapshot_hash=route.terrain_snapshot_hash,
            capability_revision=route.capability_revision,
            solver_contract_id=route.solver_contract_id,
            canonicalization_id=route.canonicalization_id,
            control_slew_id=WHEEL_KINEMATIC_CONTROL_SLEW_V2,
            observation_source_id=WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2,
            candidate_hash=_ZERO_HASH,
        )
        candidate = replace(candidate, candidate_hash=wheel_candidate_hash_v2(candidate))
        if candidate.candidate_hash != route.source_candidate_hash:
            raise ValueError("source_candidate_hash does not match private semantic projection")
        return candidate
    except WheelSQPCodecError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise WheelSQPCodecError(str(exc)) from exc
