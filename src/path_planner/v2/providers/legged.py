from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from math import cos, copysign, hypot, isclose, isfinite, pi, sin

from path_planner.core import Cell, WorldPoint
from path_planner.search.hybrid_astar import MAX_REPLAY_STEPS
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    CacheEvidenceV2,
    CostBreakdownV2,
    FailureCategoryV2,
    FailureEvidenceV2,
    ObjectiveProfileV2,
    ObservationProjectionV2,
    PlanningFailureV2,
    PlanningOutcomeV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    RoutePrimitiveV2,
    ResourceBudgetV2,
    SearchTelemetryV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.oracles.legged import (
    LEGGED_CRAWL_SEQUENCE_V2,
    LEGGED_FOOT_STORAGE_ORDER_V2,
    LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2,
    LeggedFootContactV2,
    LeggedStepCandidateV2,
    LeggedValidationResultV2,
    LegIdV2,
    validate_legged_step_l2,
)
from path_planner.v2.profiles import LeggedProfileV2, PlatformProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.search import SearchQueueEntryV2, StableSearchQueueV2
from path_planner.v2.terrain import (
    FineSafetyAnchorV2,
    TerrainSnapshotV2,
    snapshot_hash,
)


LEGGED_SEARCH_STATE_SCHEMA_V2 = "path-planner-v2-legged-search-state/v1"
LEGGED_STEP_PRIMITIVE_SCHEMA_V2 = "path-planner-v2-legged-step/v1"
LEGGED_RESOURCE_PROXY_ID_V2 = "legged_static_crawl_relative_resource/v1"
LEGGED_CAPABILITY_LEVEL_V2 = "simulation_proxy"
LEGGED_LOCAL_FOOTHOLD_OFFSETS_V2 = (
    (0.0, 0.0),
    (-0.25, 0.0),
    (0.0, -0.25),
    (0.0, 0.25),
    (0.25, 0.0),
    (-0.25, -0.25),
    (-0.25, 0.25),
    (0.25, -0.25),
    (0.25, 0.25),
    (-0.50, 0.0),
    (0.0, -0.50),
    (0.0, 0.50),
    (0.50, 0.0),
)
LEGGED_SEARCH_MEMORY_ACCOUNTING_ID_V2 = "legged_search_fixed_record_512b/v1"
LEGGED_SEARCH_RECORD_BYTES_V2 = 512

_LEGGED_OBSERVATION_SOURCE_V2 = (
    "legged_route_body_samples_gain_not_computed/v1"
)
_LEGGED_CACHE_NAMESPACE_DISABLED_V2 = "path-planner-v2-legged-cache-disabled/v1"
_LEGGED_CACHE_KEY_DISABLED_V2 = "legged-cache-disabled/v1"
_TRUSTED_VALIDATE_LEGGED_STEP_L2_V2 = validate_legged_step_l2
_TRUSTED_SNAPSHOT_HASH_V2 = snapshot_hash

_TWO_PI = 2.0 * pi
_RESOURCE_REL_TOL = 1e-12
_RESOURCE_ABS_TOL = 1e-12


def _raise_contract_value_error(name: str, error: Exception) -> None:
    if isinstance(error, (TypeError, ValueError, MemoryError)):
        raise error
    raise ValueError(f"{name} contract evaluation failed") from None


def _call_contract_helper_v2(
    name: str,
    helper: object,
    *args: object,
    **kwargs: object,
) -> object:
    try:
        return helper(*args, **kwargs)  # type: ignore[operator]
    except Exception as error:
        _raise_contract_value_error(name, error)


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and copysign(1.0, value) < 0.0


def _audit_float(
    value: object,
    name: str,
    *,
    allow_signed_zero: bool,
    nonnegative: bool = False,
) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if _is_negative_zero(value) and not allow_signed_zero:
        raise ValueError(f"{name} must use canonical positive zero")
    return 0.0 if value == 0.0 else value


def _preaudit_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _audit_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    return value


def _audit_exact_string(value: object, expected: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if value != expected:
        raise ValueError(f"{name} must be {expected}")
    return value


def _preaudit_pose(value: object, name: str) -> None:
    if type(value) is not PoseStateV2:
        raise TypeError(f"{name} must be exact PoseStateV2")
    _preaudit_float(value.x_m, f"{name}.x_m")
    _preaudit_float(value.y_m, f"{name}.y_m")
    _preaudit_float(value.heading_rad, f"{name}.heading_rad")


def _preaudit_point(value: object, name: str) -> None:
    if type(value) is not WorldPoint:
        raise TypeError(f"{name} must be exact WorldPoint")
    _preaudit_float(value.x, f"{name}.x")
    _preaudit_float(value.y, f"{name}.y")


def _preaudit_contacts(value: object, name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) != 4:
        raise ValueError(f"{name} must contain exactly four contacts")
    for index, contact in enumerate(value):
        if type(contact) is not LeggedFootContactV2:
            raise TypeError(f"{name}[{index}] must be exact LeggedFootContactV2")
        if type(contact.leg_id) is not LegIdV2:
            raise TypeError(f"{name}[{index}].leg_id must be exact LegIdV2")
        _preaudit_point(contact.foothold, f"{name}[{index}].foothold")


def _preaudit_search_state(value: object, name: str) -> None:
    if type(value) is not LeggedSearchStateV2:
        raise TypeError(f"{name} must be exact LeggedSearchStateV2")
    _preaudit_pose(value.body_state, f"{name}.body_state")
    _preaudit_contacts(value.foot_contacts, f"{name}.foot_contacts")
    if type(value.sequence_phase) is not int:
        raise TypeError(f"{name}.sequence_phase must be an exact int")
    if type(value.schema_version) is not str:
        raise TypeError(f"{name}.schema_version must be an exact built-in str")


def _canonical_heading_v2(value: float) -> float:
    if -pi <= value <= pi:
        return 0.0 if value == 0.0 else value
    try:
        wrapped = (value + pi) % _TWO_PI - pi
    except Exception as error:
        _raise_contract_value_error("heading", error)
    if wrapped == -pi and value > 0.0:
        return pi
    return 0.0 if wrapped == 0.0 else wrapped


def _audit_pose_public(
    value: object,
    name: str,
    *,
    wrap_heading: bool,
) -> PoseStateV2:
    _preaudit_pose(value, name)
    raw_x = value.x_m
    raw_y = value.y_m
    raw_heading = value.heading_rad
    x_m = _audit_float(raw_x, f"{name}.x_m", allow_signed_zero=True)
    y_m = _audit_float(raw_y, f"{name}.y_m", allow_signed_zero=True)
    heading = _audit_float(
        raw_heading,
        f"{name}.heading_rad",
        allow_signed_zero=True,
    )
    if wrap_heading:
        heading = _canonical_heading_v2(heading)
    return PoseStateV2(x_m, y_m, heading)


def _audit_pose_canonical(value: object, name: str) -> PoseStateV2:
    _preaudit_pose(value, name)
    x_m = _audit_float(value.x_m, f"{name}.x_m", allow_signed_zero=False)
    y_m = _audit_float(value.y_m, f"{name}.y_m", allow_signed_zero=False)
    heading = _audit_float(
        value.heading_rad,
        f"{name}.heading_rad",
        allow_signed_zero=False,
    )
    if not -pi <= heading <= pi:
        raise ValueError(f"{name}.heading_rad must already be canonical")
    return value


def _audit_point_public(value: object, name: str) -> WorldPoint:
    _preaudit_point(value, name)
    x_m = _audit_float(value.x, f"{name}.x", allow_signed_zero=True)
    y_m = _audit_float(value.y, f"{name}.y", allow_signed_zero=True)
    return WorldPoint(x_m, y_m)


def _audit_point_canonical(value: object, name: str) -> WorldPoint:
    _preaudit_point(value, name)
    x_m = _audit_float(value.x, f"{name}.x", allow_signed_zero=False)
    y_m = _audit_float(value.y, f"{name}.y", allow_signed_zero=False)
    return value


def _audit_contacts_canonical(
    value: object,
    name: str,
) -> tuple[LeggedFootContactV2, ...]:
    _preaudit_contacts(value, name)
    for index, (contact, expected_leg) in enumerate(
        zip(value, LEGGED_FOOT_STORAGE_ORDER_V2, strict=True)
    ):
        if contact.leg_id is not expected_leg:
            raise ValueError(f"{name} must follow exact storage order")
        _audit_point_canonical(
            contact.foothold,
            f"{name}[{index}].foothold",
        )
    return value


def _audit_phase(value: object, name: str) -> int:
    phase = _audit_exact_int(value, name)
    if not 0 <= phase <= 3:
        raise ValueError(f"{name} must be in [0, 3]")
    return phase


@dataclass(frozen=True, slots=True)
class LeggedSearchStateV2:
    body_state: PoseStateV2
    foot_contacts: tuple[LeggedFootContactV2, ...]
    sequence_phase: int
    schema_version: str = LEGGED_SEARCH_STATE_SCHEMA_V2

    def __post_init__(self) -> None:
        raw_body = self.body_state
        raw_contacts = self.foot_contacts
        raw_phase = self.sequence_phase
        raw_schema = self.schema_version

        _preaudit_pose(raw_body, "body_state")
        _preaudit_contacts(raw_contacts, "foot_contacts")
        if type(raw_phase) is not int:
            raise TypeError("sequence_phase must be an exact int")
        if type(raw_schema) is not str:
            raise TypeError("schema_version must be an exact built-in str")

        body = _audit_pose_public(raw_body, "body_state", wrap_heading=True)
        audited_contacts = _audit_contacts_canonical(raw_contacts, "foot_contacts")
        phase = _audit_phase(raw_phase, "sequence_phase")
        schema = _audit_exact_string(
            raw_schema,
            LEGGED_SEARCH_STATE_SCHEMA_V2,
            "schema_version",
        )
        contacts = tuple(
            LeggedFootContactV2(
                contact.leg_id,
                WorldPoint(contact.foothold.x, contact.foothold.y),
            )
            for contact in audited_contacts
        )
        object.__setattr__(self, "body_state", body)
        object.__setattr__(self, "foot_contacts", contacts)
        object.__setattr__(self, "sequence_phase", phase)
        object.__setattr__(self, "schema_version", schema)


def _audit_search_state_canonical(
    value: object,
    name: str,
) -> LeggedSearchStateV2:
    _preaudit_search_state(value, name)
    body = _audit_pose_canonical(value.body_state, f"{name}.body_state")
    contacts = _audit_contacts_canonical(
        value.foot_contacts,
        f"{name}.foot_contacts",
    )
    phase = _audit_phase(value.sequence_phase, f"{name}.sequence_phase")
    schema = _audit_exact_string(
        value.schema_version,
        LEGGED_SEARCH_STATE_SCHEMA_V2,
        f"{name}.schema_version",
    )
    return value


def _trig_components_v2(heading_rad: float) -> tuple[float, float]:
    return cos(heading_rad), sin(heading_rad)


def nominal_legged_search_state_v2(
    body_state: PoseStateV2,
) -> LeggedSearchStateV2:
    canonical_body = _audit_pose_public(
        body_state,
        "body_state",
        wrap_heading=True,
    )
    try:
        cosine, sine = _trig_components_v2(canonical_body.heading_rad)
    except Exception as error:
        _raise_contract_value_error("nominal stance trigonometry", error)
    cosine = _audit_float(
        cosine,
        "nominal stance cosine",
        allow_signed_zero=True,
    )
    sine = _audit_float(
        sine,
        "nominal stance sine",
        allow_signed_zero=True,
    )
    local_contacts = (
        (LegIdV2.FRONT_LEFT, 0.35, 0.25),
        (LegIdV2.FRONT_RIGHT, 0.35, -0.25),
        (LegIdV2.REAR_LEFT, -0.35, 0.25),
        (LegIdV2.REAR_RIGHT, -0.35, -0.25),
    )
    contacts: list[LeggedFootContactV2] = []
    try:
        for leg_id, local_x, local_y in local_contacts:
            world_x = canonical_body.x_m + cosine * local_x - sine * local_y
            world_y = canonical_body.y_m + sine * local_x + cosine * local_y
            world_x = _audit_float(
                world_x,
                "nominal foothold x",
                allow_signed_zero=True,
            )
            world_y = _audit_float(
                world_y,
                "nominal foothold y",
                allow_signed_zero=True,
            )
            contacts.append(
                LeggedFootContactV2(leg_id, WorldPoint(world_x, world_y))
            )
    except Exception as error:
        _raise_contract_value_error("nominal stance geometry", error)
    return LeggedSearchStateV2(canonical_body, tuple(contacts), 0)


def _float_bits_v2(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), "big", signed=False)


def _independent_float_bits_v2(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), "big", signed=False)


def _canonical_float_word_v2(
    value: object,
    name: str,
    *,
    nonnegative: bool = False,
) -> int:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if _is_negative_zero(value):
        raise ValueError(f"{name} must use canonical positive zero")
    return _independent_float_bits_v2(value)


def _canonical_pose_snapshot_v2(value: object, name: str) -> tuple[int, int, int]:
    if type(value) is not PoseStateV2:
        raise TypeError(f"{name} must be exact PoseStateV2")
    snapshot = (
        _canonical_float_word_v2(value.x_m, f"{name}.x_m"),
        _canonical_float_word_v2(value.y_m, f"{name}.y_m"),
        _canonical_float_word_v2(value.heading_rad, f"{name}.heading_rad"),
    )
    if not -pi <= value.heading_rad <= pi:
        raise ValueError(f"{name}.heading_rad must already be canonical")
    return snapshot


def _canonical_point_snapshot_v2(value: object, name: str) -> tuple[int, int]:
    if type(value) is not WorldPoint:
        raise TypeError(f"{name} must be exact WorldPoint")
    return (
        _canonical_float_word_v2(value.x, f"{name}.x"),
        _canonical_float_word_v2(value.y, f"{name}.y"),
    )


def _canonical_contacts_snapshot_v2(
    value: object,
    name: str,
) -> tuple[tuple[LegIdV2, int, int], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) != 4:
        raise ValueError(f"{name} must contain exactly four contacts")
    snapshot: list[tuple[LegIdV2, int, int]] = []
    for index, expected_leg in enumerate(LEGGED_FOOT_STORAGE_ORDER_V2):
        contact = value[index]
        if type(contact) is not LeggedFootContactV2:
            raise TypeError(f"{name}[{index}] must be exact LeggedFootContactV2")
        if type(contact.leg_id) is not LegIdV2:
            raise TypeError(f"{name}[{index}].leg_id must be exact LegIdV2")
        if contact.leg_id is not expected_leg:
            raise ValueError(f"{name} must follow exact storage order")
        point = _canonical_point_snapshot_v2(
            contact.foothold,
            f"{name}[{index}].foothold",
        )
        snapshot.append((expected_leg, *point))
    return tuple(snapshot)


def _canonical_search_state_snapshot_v2(
    value: object,
    name: str,
) -> tuple[object, ...]:
    if type(value) is not LeggedSearchStateV2:
        raise TypeError(f"{name} must be exact LeggedSearchStateV2")
    if type(value.sequence_phase) is not int:
        raise TypeError(f"{name}.sequence_phase must be an exact int")
    if not 0 <= value.sequence_phase <= 3:
        raise ValueError(f"{name}.sequence_phase must be in [0, 3]")
    if type(value.schema_version) is not str:
        raise TypeError(f"{name}.schema_version must be an exact built-in str")
    if value.schema_version != LEGGED_SEARCH_STATE_SCHEMA_V2:
        raise ValueError(f"{name}.schema_version must match the frozen schema")
    return (
        value.schema_version,
        value.sequence_phase,
        _canonical_pose_snapshot_v2(value.body_state, f"{name}.body_state"),
        _canonical_contacts_snapshot_v2(
            value.foot_contacts,
            f"{name}.foot_contacts",
        ),
    )


def legged_state_key_v2(state: LeggedSearchStateV2) -> tuple[int, ...]:
    before = _call_contract_helper_v2(
        "legged state pre-audit snapshot",
        _canonical_search_state_snapshot_v2,
        state,
        "state",
    )
    audited = _call_contract_helper_v2(
        "legged state audit",
        _audit_search_state_canonical,
        state,
        "state",
    )
    if type(audited) is not LeggedSearchStateV2 or audited is not state:
        raise ValueError("legged state audit must return exact LeggedSearchStateV2")
    after = _call_contract_helper_v2(
        "legged state post-audit snapshot",
        _canonical_search_state_snapshot_v2,
        state,
        "state",
    )
    if type(before) is not tuple or type(after) is not tuple or after != before:
        raise ValueError("legged state audit must not mutate canonical payload")
    floats = (
        audited.body_state.x_m,
        audited.body_state.y_m,
        audited.body_state.heading_rad,
        *(
            coordinate
            for contact in audited.foot_contacts
            for coordinate in (contact.foothold.x, contact.foothold.y)
        ),
    )
    try:
        words_list: list[int] = []
        for value in floats:
            word = _float_bits_v2(value)
            independent_word = _independent_float_bits_v2(value)
            if (
                type(word) is not int
                or not 0 <= word < 1 << 64
                or word != independent_word
            ):
                raise ValueError("float bit helper returned noncanonical word")
            words_list.append(word)
        words = tuple(words_list)
    except Exception as error:
        _raise_contract_value_error("legged state key", error)
    return (1, audited.sequence_phase, *words)


def _hypot_v2(x_value: float, y_value: float) -> float:
    return hypot(x_value, y_value)


def _resource_values_v2(
    start: LeggedSearchStateV2,
    lift: PoseStateV2,
    end: LeggedSearchStateV2,
    moving_leg: LegIdV2,
    target: WorldPoint,
) -> tuple[float, float, float]:
    moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(moving_leg)
    source = start.foot_contacts[moving_index].foothold
    try:
        foot_travel = _hypot_v2(target.x - source.x, target.y - source.y)
        first_body = _hypot_v2(
            lift.x_m - start.body_state.x_m,
            lift.y_m - start.body_state.y_m,
        )
        second_body = _hypot_v2(
            end.body_state.x_m - lift.x_m,
            end.body_state.y_m - lift.y_m,
        )
        distance = first_body + second_body
        energy = distance + foot_travel
    except Exception as error:
        _raise_contract_value_error("legged relative resource", error)
    foot_travel = _audit_float(
        foot_travel,
        "derived foot_travel_m",
        allow_signed_zero=True,
        nonnegative=True,
    )
    distance = _audit_float(
        distance,
        "derived distance_m",
        allow_signed_zero=True,
        nonnegative=True,
    )
    energy = _audit_float(
        energy,
        "derived energy_cost",
        allow_signed_zero=True,
        nonnegative=True,
    )
    return foot_travel, distance, energy


def _validate_step_relations(
    *,
    start_state: PoseStateV2,
    end_state: PoseStateV2,
    start_legged_state: LeggedSearchStateV2,
    lift_body_state: PoseStateV2,
    end_legged_state: LeggedSearchStateV2,
    moving_leg: LegIdV2,
    target_foothold: WorldPoint,
) -> tuple[float, float, float]:
    if start_state != start_legged_state.body_state:
        raise ValueError("start_state must exactly match start_legged_state.body_state")
    if end_state != end_legged_state.body_state:
        raise ValueError("end_state must exactly match end_legged_state.body_state")
    expected_leg = LEGGED_CRAWL_SEQUENCE_V2[start_legged_state.sequence_phase]
    if moving_leg is not expected_leg:
        raise ValueError("moving_leg must match the crawl sequence phase")
    if end_legged_state.sequence_phase != (start_legged_state.sequence_phase + 1) % 4:
        raise ValueError("end sequence phase must advance by one")

    moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(moving_leg)
    for index, (start_contact, end_contact) in enumerate(
        zip(
            start_legged_state.foot_contacts,
            end_legged_state.foot_contacts,
            strict=True,
        )
    ):
        if index == moving_index:
            if end_contact.foothold != target_foothold:
                raise ValueError("moving contact must exactly equal target_foothold")
        elif end_contact != start_contact:
            raise ValueError("nonmoving contacts must remain exactly unchanged")
    return _resource_values_v2(
        start_legged_state,
        lift_body_state,
        end_legged_state,
        moving_leg,
        target_foothold,
    )


def _resource_matches(raw: float, expected: float) -> bool:
    return isclose(
        raw,
        expected,
        rel_tol=_RESOURCE_REL_TOL,
        abs_tol=_RESOURCE_ABS_TOL,
    )


@dataclass(frozen=True, slots=True)
class LeggedStepPrimitiveV2(RoutePrimitiveV2):
    start_legged_state: LeggedSearchStateV2
    lift_body_state: PoseStateV2
    end_legged_state: LeggedSearchStateV2
    moving_leg: LegIdV2
    target_foothold: WorldPoint
    foot_travel_m: float
    capability: str = LEGGED_CAPABILITY_LEVEL_V2
    resource_proxy_id: str = LEGGED_RESOURCE_PROXY_ID_V2
    primitive_schema_version: str = LEGGED_STEP_PRIMITIVE_SCHEMA_V2

    def __post_init__(self) -> None:
        raw_resource_evidence = (
            self.foot_travel_m,
            self.distance_m,
            self.energy_cost,
        )
        result = _call_contract_helper_v2(
            "legged step primitive construction",
            self._initialize_canonical_payload,
        )
        if result is not None:
            raise ValueError("legged step primitive initializer must return None")
        seal_result = _call_contract_helper_v2(
            "legged step primitive postcondition",
            _seal_primitive_canonical_postcondition_v2,
            self,
            raw_resource_evidence,
        )
        if seal_result is not None:
            raise ValueError("legged step primitive postcondition must return None")

    def _initialize_canonical_payload(self) -> None:
        raw_kind = self.kind
        raw_start_state = self.start_state
        raw_end_state = self.end_state
        raw_duration = self.duration_s
        raw_distance = self.distance_m
        raw_energy = self.energy_cost
        raw_observation = self.observation_contribution
        raw_validation = self.validation_level
        raw_start_legged = self.start_legged_state
        raw_lift = self.lift_body_state
        raw_end_legged = self.end_legged_state
        raw_moving_leg = self.moving_leg
        raw_target = self.target_foothold
        raw_foot_travel = self.foot_travel_m
        raw_capability = self.capability
        raw_resource = self.resource_proxy_id
        raw_schema = self.primitive_schema_version

        # Complete the exact raw audit before any semantic comparison,
        # arithmetic, payload rebuilding, or base-class normalization.
        if type(raw_kind) is not PrimitiveKindV2:
            raise TypeError("kind must be exact PrimitiveKindV2")
        _preaudit_pose(raw_start_state, "start_state")
        _preaudit_pose(raw_end_state, "end_state")
        _preaudit_float(raw_duration, "duration_s")
        _preaudit_float(raw_distance, "distance_m")
        _preaudit_float(raw_energy, "energy_cost")
        _preaudit_float(raw_observation, "observation_contribution")
        if type(raw_validation) is not ValidationLevelV2:
            raise TypeError("validation_level must be exact ValidationLevelV2")
        _preaudit_search_state(raw_start_legged, "start_legged_state")
        _preaudit_pose(raw_lift, "lift_body_state")
        _preaudit_search_state(raw_end_legged, "end_legged_state")
        if type(raw_moving_leg) is not LegIdV2:
            raise TypeError("moving_leg must be exact LegIdV2")
        _preaudit_point(raw_target, "target_foothold")
        _preaudit_float(raw_foot_travel, "foot_travel_m")
        if type(raw_capability) is not str:
            raise TypeError("capability must be an exact built-in str")
        if type(raw_resource) is not str:
            raise TypeError("resource_proxy_id must be an exact built-in str")
        if type(raw_schema) is not str:
            raise TypeError("primitive_schema_version must be an exact built-in str")

        if type(raw_kind) is not PrimitiveKindV2:
            raise TypeError("kind must be exact PrimitiveKindV2")
        if raw_kind is not PrimitiveKindV2.LEG_STEP:
            raise ValueError("kind must be PrimitiveKindV2.LEG_STEP")
        start_state = _audit_pose_public(
            raw_start_state,
            "start_state",
            wrap_heading=False,
        )
        end_state = _audit_pose_public(
            raw_end_state,
            "end_state",
            wrap_heading=False,
        )
        duration = _audit_float(
            raw_duration,
            "duration_s",
            allow_signed_zero=True,
            nonnegative=True,
        )
        distance = _audit_float(
            raw_distance,
            "distance_m",
            allow_signed_zero=True,
            nonnegative=True,
        )
        energy = _audit_float(
            raw_energy,
            "energy_cost",
            allow_signed_zero=True,
            nonnegative=True,
        )
        observation = _audit_float(
            raw_observation,
            "observation_contribution",
            allow_signed_zero=True,
            nonnegative=True,
        )
        if type(raw_validation) is not ValidationLevelV2:
            raise TypeError("validation_level must be exact ValidationLevelV2")
        if raw_validation is not ValidationLevelV2.L2:
            raise ValueError("validation_level must be ValidationLevelV2.L2")
        start_legged = _audit_search_state_canonical(
            raw_start_legged,
            "start_legged_state",
        )
        lift = _audit_pose_public(
            raw_lift,
            "lift_body_state",
            wrap_heading=True,
        )
        end_legged = _audit_search_state_canonical(
            raw_end_legged,
            "end_legged_state",
        )
        if type(raw_moving_leg) is not LegIdV2:
            raise TypeError("moving_leg must be exact LegIdV2")
        target = _audit_point_public(raw_target, "target_foothold")
        foot_travel = _audit_float(
            raw_foot_travel,
            "foot_travel_m",
            allow_signed_zero=True,
            nonnegative=True,
        )
        capability = _audit_exact_string(
            raw_capability,
            LEGGED_CAPABILITY_LEVEL_V2,
            "capability",
        )
        resource = _audit_exact_string(
            raw_resource,
            LEGGED_RESOURCE_PROXY_ID_V2,
            "resource_proxy_id",
        )
        schema = _audit_exact_string(
            raw_schema,
            LEGGED_STEP_PRIMITIVE_SCHEMA_V2,
            "primitive_schema_version",
        )

        if duration != 1.0:
            raise ValueError("duration_s must be exactly 1.0")
        if observation != 0.0:
            raise ValueError("observation_contribution must be exactly 0.0")

        object.__setattr__(self, "kind", raw_kind)
        object.__setattr__(self, "start_state", start_state)
        object.__setattr__(self, "end_state", end_state)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "energy_cost", energy)
        object.__setattr__(self, "observation_contribution", observation)
        object.__setattr__(self, "validation_level", raw_validation)
        object.__setattr__(self, "start_legged_state", start_legged)
        object.__setattr__(self, "lift_body_state", lift)
        object.__setattr__(self, "end_legged_state", end_legged)
        object.__setattr__(self, "moving_leg", raw_moving_leg)
        object.__setattr__(self, "target_foothold", target)
        object.__setattr__(self, "foot_travel_m", foot_travel)
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "resource_proxy_id", resource)
        object.__setattr__(self, "primitive_schema_version", schema)
        RoutePrimitiveV2.__post_init__(self)

        expected_resources = _call_contract_helper_v2(
            "legged step relation validation",
            _validate_step_relations,
            start_state=start_state,
            end_state=end_state,
            start_legged_state=start_legged,
            lift_body_state=lift,
            end_legged_state=end_legged,
            moving_leg=raw_moving_leg,
            target_foothold=target,
        )
        if (
            type(expected_resources) is not tuple
            or len(expected_resources) != 3
            or any(type(value) is not float for value in expected_resources)
        ):
            raise ValueError("legged step relation validation returned invalid resources")
        expected_foot, expected_distance, expected_energy = (
            _audit_float(
                expected_resources[0],
                "relation foot_travel_m",
                allow_signed_zero=False,
                nonnegative=True,
            ),
            _audit_float(
                expected_resources[1],
                "relation distance_m",
                allow_signed_zero=False,
                nonnegative=True,
            ),
            _audit_float(
                expected_resources[2],
                "relation energy_cost",
                allow_signed_zero=False,
                nonnegative=True,
            ),
        )
        for raw_value, expected_value, name in (
            (foot_travel, expected_foot, "foot_travel_m"),
            (distance, expected_distance, "distance_m"),
            (energy, expected_energy, "energy_cost"),
        ):
            matches = _call_contract_helper_v2(
                "legged resource comparison",
                _resource_matches,
                raw_value,
                expected_value,
            )
            if type(matches) is not bool:
                raise ValueError("legged resource comparison must return exact bool")
            if not matches:
                raise ValueError(f"{name} must match the relative resource proxy")
        object.__setattr__(self, "foot_travel_m", expected_foot)
        object.__setattr__(self, "distance_m", expected_distance)
        object.__setattr__(self, "energy_cost", expected_energy)

    def as_oracle_candidate(self) -> LeggedStepCandidateV2:
        pre_seal = _call_contract_helper_v2(
            "legged primitive pre-audit postcondition",
            _seal_primitive_canonical_postcondition_v2,
            self,
        )
        if pre_seal is not None:
            raise ValueError("legged primitive postcondition must return None")
        audited = _call_contract_helper_v2(
            "legged primitive audit",
            _audit_primitive_canonical,
            self,
        )
        if audited is not self:
            raise ValueError("legged primitive audit must preserve object identity")
        post_seal = _call_contract_helper_v2(
            "legged primitive post-audit postcondition",
            _seal_primitive_canonical_postcondition_v2,
            self,
        )
        if post_seal is not None:
            raise ValueError("legged primitive postcondition must return None")
        candidate = _call_contract_helper_v2(
            "oracle candidate",
            _candidate_from_payload_v2,
            self.start_legged_state,
            self.lift_body_state,
            self.end_legged_state,
            self.moving_leg,
            self.target_foothold,
        )
        if type(candidate) is not LeggedStepCandidateV2:
            raise ValueError("oracle candidate helper returned invalid payload")
        candidate_seal = _call_contract_helper_v2(
            "oracle candidate postcondition",
            _seal_candidate_postcondition_v2,
            candidate,
            self,
        )
        if candidate_seal is not None:
            raise ValueError("oracle candidate postcondition must return None")
        return candidate


def _audit_primitive_canonical(value: object) -> LeggedStepPrimitiveV2:
    if type(value) is not LeggedStepPrimitiveV2:
        raise TypeError("primitive must be exact LeggedStepPrimitiveV2")
    if type(value.kind) is not PrimitiveKindV2 or value.kind is not PrimitiveKindV2.LEG_STEP:
        raise ValueError("primitive kind drifted from LEG_STEP")
    start_state = _audit_pose_canonical(value.start_state, "start_state")
    end_state = _audit_pose_canonical(value.end_state, "end_state")
    duration = _audit_float(
        value.duration_s,
        "duration_s",
        allow_signed_zero=False,
        nonnegative=True,
    )
    distance = _audit_float(
        value.distance_m,
        "distance_m",
        allow_signed_zero=False,
        nonnegative=True,
    )
    energy = _audit_float(
        value.energy_cost,
        "energy_cost",
        allow_signed_zero=False,
        nonnegative=True,
    )
    observation = _audit_float(
        value.observation_contribution,
        "observation_contribution",
        allow_signed_zero=False,
        nonnegative=True,
    )
    if duration != 1.0 or observation != 0.0:
        raise ValueError("primitive fixed base fields drifted")
    if (
        type(value.validation_level) is not ValidationLevelV2
        or value.validation_level is not ValidationLevelV2.L2
    ):
        raise ValueError("primitive validation level drifted from L2")
    start_legged = _audit_search_state_canonical(
        value.start_legged_state,
        "start_legged_state",
    )
    lift = _audit_pose_canonical(value.lift_body_state, "lift_body_state")
    end_legged = _audit_search_state_canonical(
        value.end_legged_state,
        "end_legged_state",
    )
    if type(value.moving_leg) is not LegIdV2:
        raise TypeError("moving_leg must be exact LegIdV2")
    target = _audit_point_canonical(value.target_foothold, "target_foothold")
    foot = _audit_float(
        value.foot_travel_m,
        "foot_travel_m",
        allow_signed_zero=False,
        nonnegative=True,
    )
    _audit_exact_string(
        value.capability,
        LEGGED_CAPABILITY_LEVEL_V2,
        "capability",
    )
    _audit_exact_string(
        value.resource_proxy_id,
        LEGGED_RESOURCE_PROXY_ID_V2,
        "resource_proxy_id",
    )
    _audit_exact_string(
        value.primitive_schema_version,
        LEGGED_STEP_PRIMITIVE_SCHEMA_V2,
        "primitive_schema_version",
    )
    expected_foot, expected_distance, expected_energy = _validate_step_relations(
        start_state=start_state,
        end_state=end_state,
        start_legged_state=start_legged,
        lift_body_state=lift,
        end_legged_state=end_legged,
        moving_leg=value.moving_leg,
        target_foothold=target,
    )
    if (foot, distance, energy) != (
        expected_foot,
        expected_distance,
        expected_energy,
    ):
        raise ValueError("primitive resource payload drifted")
    return value


def _seal_primitive_canonical_postcondition_v2(
    value: object,
    raw_resource_evidence: object | None = None,
) -> None:
    if type(value) is not LeggedStepPrimitiveV2:
        raise TypeError("primitive must be exact LeggedStepPrimitiveV2")
    if type(value.kind) is not PrimitiveKindV2:
        raise TypeError("kind must be exact PrimitiveKindV2")
    if value.kind is not PrimitiveKindV2.LEG_STEP:
        raise ValueError("kind must be PrimitiveKindV2.LEG_STEP")

    start_pose = _canonical_pose_snapshot_v2(value.start_state, "start_state")
    end_pose = _canonical_pose_snapshot_v2(value.end_state, "end_state")
    lift_pose = _canonical_pose_snapshot_v2(
        value.lift_body_state,
        "lift_body_state",
    )
    _canonical_float_word_v2(value.duration_s, "duration_s", nonnegative=True)
    distance_word = _canonical_float_word_v2(
        value.distance_m,
        "distance_m",
        nonnegative=True,
    )
    energy_word = _canonical_float_word_v2(
        value.energy_cost,
        "energy_cost",
        nonnegative=True,
    )
    _canonical_float_word_v2(
        value.observation_contribution,
        "observation_contribution",
        nonnegative=True,
    )
    foot_word = _canonical_float_word_v2(
        value.foot_travel_m,
        "foot_travel_m",
        nonnegative=True,
    )
    if value.duration_s != 1.0:
        raise ValueError("duration_s must be exactly 1.0")
    if value.observation_contribution != 0.0:
        raise ValueError("observation_contribution must be exactly 0.0")
    if type(value.validation_level) is not ValidationLevelV2:
        raise TypeError("validation_level must be exact ValidationLevelV2")
    if value.validation_level is not ValidationLevelV2.L2:
        raise ValueError("validation_level must be ValidationLevelV2.L2")

    _canonical_search_state_snapshot_v2(
        value.start_legged_state,
        "start_legged_state",
    )
    _canonical_search_state_snapshot_v2(
        value.end_legged_state,
        "end_legged_state",
    )
    start_body = _canonical_pose_snapshot_v2(
        value.start_legged_state.body_state,
        "start_legged_state.body_state",
    )
    end_body = _canonical_pose_snapshot_v2(
        value.end_legged_state.body_state,
        "end_legged_state.body_state",
    )
    if start_pose != start_body:
        raise ValueError("start_state must bitwise match start legged body")
    if end_pose != end_body:
        raise ValueError("end_state must bitwise match end legged body")

    if type(value.moving_leg) is not LegIdV2:
        raise TypeError("moving_leg must be exact LegIdV2")
    start_phase = value.start_legged_state.sequence_phase
    end_phase = value.end_legged_state.sequence_phase
    if value.moving_leg is not LEGGED_CRAWL_SEQUENCE_V2[start_phase]:
        raise ValueError("moving_leg must match the crawl sequence phase")
    if end_phase != (start_phase + 1) % 4:
        raise ValueError("end sequence phase must advance by one")

    target = _canonical_point_snapshot_v2(value.target_foothold, "target_foothold")
    start_contacts = _canonical_contacts_snapshot_v2(
        value.start_legged_state.foot_contacts,
        "start_legged_state.foot_contacts",
    )
    end_contacts = _canonical_contacts_snapshot_v2(
        value.end_legged_state.foot_contacts,
        "end_legged_state.foot_contacts",
    )
    moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(value.moving_leg)
    for index, (start_contact, end_contact) in enumerate(
        zip(start_contacts, end_contacts, strict=True)
    ):
        if index == moving_index:
            if end_contact[1:] != target:
                raise ValueError("moving contact must bitwise match target foothold")
        elif end_contact != start_contact:
            raise ValueError("nonmoving contacts must remain bitwise unchanged")

    source = value.start_legged_state.foot_contacts[moving_index].foothold
    target_point = value.target_foothold
    foot_expected = hypot(target_point.x - source.x, target_point.y - source.y)
    first_body = hypot(
        value.lift_body_state.x_m - value.start_state.x_m,
        value.lift_body_state.y_m - value.start_state.y_m,
    )
    second_body = hypot(
        value.end_state.x_m - value.lift_body_state.x_m,
        value.end_state.y_m - value.lift_body_state.y_m,
    )
    distance_expected = first_body + second_body
    energy_expected = distance_expected + foot_expected
    expected_words = (
        _canonical_float_word_v2(
            foot_expected,
            "independent foot_travel_m",
            nonnegative=True,
        ),
        _canonical_float_word_v2(
            distance_expected,
            "independent distance_m",
            nonnegative=True,
        ),
        _canonical_float_word_v2(
            energy_expected,
            "independent energy_cost",
            nonnegative=True,
        ),
    )
    if raw_resource_evidence is not None:
        if type(raw_resource_evidence) is not tuple:
            raise TypeError("raw resource evidence must be an exact tuple")
        if len(raw_resource_evidence) != 3:
            raise ValueError("raw resource evidence must contain exactly three values")
        raw_values: list[float] = []
        for raw_value, name in zip(
            raw_resource_evidence,
            ("raw foot_travel_m", "raw distance_m", "raw energy_cost"),
            strict=True,
        ):
            if type(raw_value) is not float:
                raise TypeError(f"{name} must be an exact built-in float")
            if not isfinite(raw_value):
                raise ValueError(f"{name} must be finite")
            if raw_value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            raw_values.append(raw_value)
        for raw_value, expected_value, name in zip(
            raw_values,
            (foot_expected, distance_expected, energy_expected),
            ("foot_travel_m", "distance_m", "energy_cost"),
            strict=True,
        ):
            if not isclose(
                raw_value,
                expected_value,
                rel_tol=_RESOURCE_REL_TOL,
                abs_tol=_RESOURCE_ABS_TOL,
            ):
                raise ValueError(
                    f"raw {name} must match the direct relative resource proxy"
                )
    if (foot_word, distance_word, energy_word) != expected_words:
        raise ValueError("primitive resources must bitwise match direct recomputation")

    for actual, expected, name in (
        (value.capability, LEGGED_CAPABILITY_LEVEL_V2, "capability"),
        (value.resource_proxy_id, LEGGED_RESOURCE_PROXY_ID_V2, "resource_proxy_id"),
        (
            value.primitive_schema_version,
            LEGGED_STEP_PRIMITIVE_SCHEMA_V2,
            "primitive_schema_version",
        ),
    ):
        if type(actual) is not str:
            raise TypeError(f"{name} must be an exact built-in str")
        if actual != expected:
            raise ValueError(f"{name} must match the frozen value")


def _seal_candidate_postcondition_v2(
    candidate: object,
    primitive: object,
) -> None:
    if type(candidate) is not LeggedStepCandidateV2:
        raise TypeError("candidate must be exact LeggedStepCandidateV2")
    if type(primitive) is not LeggedStepPrimitiveV2:
        raise TypeError("primitive must be exact LeggedStepPrimitiveV2")

    candidate_start = _canonical_pose_snapshot_v2(
        candidate.start_body_state,
        "candidate.start_body_state",
    )
    candidate_lift = _canonical_pose_snapshot_v2(
        candidate.lift_body_state,
        "candidate.lift_body_state",
    )
    candidate_end = _canonical_pose_snapshot_v2(
        candidate.end_body_state,
        "candidate.end_body_state",
    )
    candidate_contacts = _canonical_contacts_snapshot_v2(
        candidate.foot_contacts,
        "candidate.foot_contacts",
    )
    if type(candidate.moving_leg) is not LegIdV2:
        raise TypeError("candidate.moving_leg must be exact LegIdV2")
    if type(candidate.sequence_phase) is not int:
        raise TypeError("candidate.sequence_phase must be an exact int")
    if not 0 <= candidate.sequence_phase <= 3:
        raise ValueError("candidate.sequence_phase must be in [0, 3]")
    candidate_target = _canonical_point_snapshot_v2(
        candidate.target_foothold,
        "candidate.target_foothold",
    )

    primitive_start = _canonical_pose_snapshot_v2(
        primitive.start_legged_state.body_state,
        "primitive.start_legged_state.body_state",
    )
    primitive_lift = _canonical_pose_snapshot_v2(
        primitive.lift_body_state,
        "primitive.lift_body_state",
    )
    primitive_end = _canonical_pose_snapshot_v2(
        primitive.end_legged_state.body_state,
        "primitive.end_legged_state.body_state",
    )
    primitive_contacts = _canonical_contacts_snapshot_v2(
        primitive.start_legged_state.foot_contacts,
        "primitive.start_legged_state.foot_contacts",
    )
    primitive_target = _canonical_point_snapshot_v2(
        primitive.target_foothold,
        "primitive.target_foothold",
    )
    if (
        candidate_start != primitive_start
        or candidate_lift != primitive_lift
        or candidate_end != primitive_end
        or candidate_contacts != primitive_contacts
        or candidate.moving_leg is not primitive.moving_leg
        or candidate.sequence_phase != primitive.start_legged_state.sequence_phase
        or candidate_target != primitive_target
    ):
        raise ValueError("oracle candidate must bitwise match primitive payload")


def _candidate_from_payload_v2(
    start: LeggedSearchStateV2,
    lift: PoseStateV2,
    end: LeggedSearchStateV2,
    moving_leg: LegIdV2,
    target: WorldPoint,
) -> LeggedStepCandidateV2:
    return LeggedStepCandidateV2(
        start_body_state=start.body_state,
        lift_body_state=lift,
        end_body_state=end.body_state,
        foot_contacts=start.foot_contacts,
        moving_leg=moving_leg,
        sequence_phase=start.sequence_phase,
        target_foothold=target,
    )


_LEGGED_STEP_FAILURE_REASONS_V2 = frozenset(
    {
        "legged_step_structure_mismatch",
        "legged_foothold_grid_misaligned",
        "legged_foothold_unknown",
        "legged_foothold_hard_obstacle",
        "legged_foothold_not_traversable",
        "legged_foothold_slope_exceeded",
        "legged_step_length_exceeded",
        "legged_step_height_exceeded",
        "legged_support_margin_insufficient",
        "legged_body_sweep_unknown",
        "legged_body_sweep_collision",
        "legged_foot_sequence_invalid",
    }
)
_LEGGED_GLOBAL_A2_FAILURE_REASONS_V2 = frozenset(
    {
        "planning_deadline_contract_mismatch",
        "terrain_snapshot_hash_mismatch",
        "terrain_query_contract_mismatch",
        "legged_profile_contract_mismatch",
    }
)


class _LeggedAuthorityFailure(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class _LeggedDeadlineExpired(RuntimeError):
    pass


def _provider_float_word_v2(
    value: object,
    name: str,
    *,
    nonnegative: bool = False,
    canonical_zero: bool = False,
) -> int:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if canonical_zero and _is_negative_zero(value):
        raise ValueError(f"{name} must use canonical positive zero")
    return int.from_bytes(struct.pack(">d", value), "big", signed=False)


def _provider_positive_zero_v2(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _provider_nonnegative_float_v2(value: object, name: str) -> float:
    _provider_float_word_v2(
        value,
        name,
        nonnegative=True,
        canonical_zero=True,
    )
    return value  # type: ignore[return-value]


def _provider_pose_token_v2(
    value: object,
    name: str,
    *,
    canonical: bool,
) -> tuple[int, int, int]:
    if type(value) is not PoseStateV2:
        raise TypeError(f"{name} must be exact PoseStateV2")
    token = (
        _provider_float_word_v2(value.x_m, f"{name}.x_m", canonical_zero=canonical),
        _provider_float_word_v2(value.y_m, f"{name}.y_m", canonical_zero=canonical),
        _provider_float_word_v2(
            value.heading_rad,
            f"{name}.heading_rad",
            canonical_zero=canonical,
        ),
    )
    if canonical and not -pi <= value.heading_rad <= pi:
        raise ValueError(f"{name}.heading_rad must already be canonical")
    return token


def _provider_request_token_v2(request: object) -> tuple[object, ...]:
    if type(request) is not PlanningRequestV2:
        raise TypeError("request must be exact PlanningRequestV2")
    if type(request.request_id) is not str or not request.request_id:
        raise ValueError("request_id must be exact nonempty str")
    if type(request.platform_profile_id) is not str or not request.platform_profile_id:
        raise ValueError("platform_profile_id must be exact nonempty str")
    objective = request.objective_profile
    resource = request.resource_budget
    if type(objective) is not ObjectiveProfileV2:
        raise TypeError("objective_profile must be exact ObjectiveProfileV2")
    if type(resource) is not ResourceBudgetV2:
        raise TypeError("resource_budget must be exact ResourceBudgetV2")
    weights = tuple(
        _provider_float_word_v2(
            getattr(objective, name),
            f"objective_profile.{name}",
            nonnegative=True,
            canonical_zero=True,
        )
        for name in (
            "distance_weight",
            "risk_weight",
            "energy_weight",
            "time_weight",
        )
    )
    limits: list[int] = []
    for name in (
        "max_expanded_states",
        "max_route_states",
        "max_memory_bytes",
    ):
        value = getattr(resource, name)
        if type(value) is not int or value < 0:
            raise TypeError(f"resource_budget.{name} must be exact nonnegative int")
        limits.append(value)
    if type(request.accelerator_policy) is not AcceleratorPolicyV2:
        raise TypeError("accelerator_policy must be exact AcceleratorPolicyV2")
    if type(request.determinism_seed) is not int:
        raise TypeError("determinism_seed must be exact int")
    if type(request.terrain_snapshot) is not TerrainSnapshotV2:
        raise TypeError("terrain_snapshot must be exact TerrainSnapshotV2")
    return (
        id(request),
        request.request_id,
        request.platform_profile_id,
        _provider_pose_token_v2(request.start_state, "request.start_state", canonical=False),
        _provider_pose_token_v2(request.goal_state, "request.goal_state", canonical=False),
        id(request.terrain_snapshot),
        weights,
        tuple(limits),
        _provider_float_word_v2(
            request.timeout_s,
            "request.timeout_s",
            nonnegative=True,
            canonical_zero=True,
        ),
        request.accelerator_policy,
        request.determinism_seed,
    )


def _provider_profile_token_v2(profile: object) -> tuple[object, ...]:
    if type(profile) is not LeggedProfileV2:
        raise TypeError("legged_profile must be exact LeggedProfileV2")
    base = profile.profile
    if type(base) is not PlatformProfileV2:
        raise TypeError("legged_profile.profile must be exact PlatformProfileV2")
    for name in ("profile_id", "capability_revision", "schema_version"):
        value = getattr(base, name)
        if type(value) is not str or not value:
            raise ValueError(f"profile.{name} must be exact nonempty str")
    if type(base.platform_kind) is not PlatformKindV2:
        raise TypeError("profile.platform_kind must be exact PlatformKindV2")
    if type(base.simulation_proxy) is not bool:
        raise TypeError("profile.simulation_proxy must be exact bool")
    base_floats = tuple(
        _provider_float_word_v2(
            getattr(base, name),
            f"profile.{name}",
            nonnegative=True,
            canonical_zero=True,
        )
        for name in (
            "max_traversable_slope_deg",
            "goal_position_tolerance_m",
            "goal_heading_tolerance_rad",
        )
    )
    legged_floats = tuple(
        _provider_float_word_v2(
            getattr(profile, name),
            f"legged_profile.{name}",
            nonnegative=True,
            canonical_zero=True,
        )
        for name in (
            "body_length_m",
            "body_width_m",
            "nominal_foot_rectangle_length_m",
            "nominal_foot_rectangle_width_m",
            "max_step_length_m",
            "max_step_height_m",
            "max_foothold_slope_deg",
            "min_support_margin_m",
            "local_foothold_grid_spacing_m",
        )
    )
    return (
        id(profile),
        id(base),
        base.profile_id,
        base.platform_kind,
        base.capability_revision,
        base.simulation_proxy,
        base.schema_version,
        base_floats,
        legged_floats,
    )


def _provider_snapshot_digest_v2(snapshot: object) -> str:
    digest = _TRUSTED_SNAPSHOT_HASH_V2(snapshot)
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("snapshot hash must be lowercase SHA-256")
    return digest


class _LeggedAuthorityGuardV2:
    def __init__(
        self,
        request: PlanningRequestV2,
        anchor: FineSafetyAnchorV2,
        profile: LeggedProfileV2,
        deadline: PlanningDeadlineV2,
    ) -> None:
        self.request = request
        self.anchor = anchor
        self.profile = profile
        self.deadline = deadline
        self.last_now = 0.0
        self._deadline_token = self._read_deadline_token()
        try:
            self._request_token = _provider_request_token_v2(request)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise _LeggedAuthorityFailure("planning_request_contract_mismatch") from None
        try:
            self._profile_token = _provider_profile_token_v2(profile)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise _LeggedAuthorityFailure("legged_profile_contract_mismatch") from None
        if request.platform_profile_id != profile.profile.profile_id:
            raise _LeggedAuthorityFailure("planning_request_contract_mismatch")
        if anchor.snapshot is not request.terrain_snapshot:
            raise _LeggedAuthorityFailure("terrain_snapshot_identity_mismatch")
        self._snapshot_identity = id(request.terrain_snapshot)
        try:
            self._snapshot_digest = _provider_snapshot_digest_v2(
                request.terrain_snapshot
            )
            cached = anchor._snapshot_hash
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise _LeggedAuthorityFailure("terrain_snapshot_hash_mismatch") from None
        if type(cached) is not str or cached != self._snapshot_digest:
            raise _LeggedAuthorityFailure("terrain_snapshot_hash_mismatch")

    def _read_deadline_token(self) -> tuple[str, str, int]:
        try:
            started = self.deadline.started_monotonic_s
            cutoff = self.deadline.deadline_monotonic_s
            clock = self.deadline._monotonic_clock
            if type(started) is not float or not isfinite(started):
                raise ValueError
            if type(cutoff) is not float or not isfinite(cutoff) or cutoff < started:
                raise ValueError
            if not callable(clock):
                raise TypeError
            return started.hex(), cutoff.hex(), id(clock)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise _LeggedAuthorityFailure("planning_deadline_contract_mismatch") from None

    @property
    def started(self) -> float:
        return float.fromhex(self._deadline_token[0])

    @property
    def cutoff(self) -> float:
        return float.fromhex(self._deadline_token[1])

    def _seal_deadline(self) -> None:
        if self._read_deadline_token() != self._deadline_token:
            raise _LeggedAuthorityFailure("planning_deadline_contract_mismatch")

    def _seal_authority(self) -> None:
        try:
            request_token = _provider_request_token_v2(self.request)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise _LeggedAuthorityFailure("planning_request_contract_mismatch") from None
        if request_token != self._request_token:
            raise _LeggedAuthorityFailure("planning_request_contract_mismatch")
        try:
            profile_token = _provider_profile_token_v2(self.profile)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise _LeggedAuthorityFailure("legged_profile_contract_mismatch") from None
        if profile_token != self._profile_token:
            raise _LeggedAuthorityFailure("legged_profile_contract_mismatch")
        if (
            self.anchor.snapshot is not self.request.terrain_snapshot
            or id(self.request.terrain_snapshot) != self._snapshot_identity
        ):
            raise _LeggedAuthorityFailure("terrain_snapshot_identity_mismatch")
        try:
            digest = _provider_snapshot_digest_v2(self.request.terrain_snapshot)
            cached = self.anchor._snapshot_hash
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            raise _LeggedAuthorityFailure("terrain_snapshot_hash_mismatch") from None
        if digest != self._snapshot_digest or cached != self._snapshot_digest:
            raise _LeggedAuthorityFailure("terrain_snapshot_hash_mismatch")

    def seal_without_clock(self) -> None:
        self._seal_deadline()
        self._seal_authority()

    def check(self) -> float:
        self.seal_without_clock()
        clock_failed = False
        try:
            now = self.deadline._monotonic_clock()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            clock_failed = True
            now = None
        self._seal_deadline()
        self._seal_authority()
        if clock_failed:
            raise _LeggedAuthorityFailure("planning_deadline_contract_mismatch")
        try:
            _provider_float_word_v2(now, "monotonic clock result")
        except Exception:
            raise _LeggedAuthorityFailure("planning_deadline_contract_mismatch") from None
        self.last_now = now
        if now >= self.cutoff:
            raise _LeggedDeadlineExpired
        return now


@dataclass(frozen=True, slots=True)
class _LeggedNodeRecordV2:
    candidate_id: str
    state: LeggedSearchStateV2
    state_key: tuple[int, ...]
    depth: int
    distance_cost: float
    risk_cost: float
    energy_cost: float
    time_cost: float
    path_cost: float
    parent_candidate_id: str | None
    incoming_primitive: LeggedStepPrimitiveV2 | None


def _provider_category_v2(reason_code: str, stage: str) -> FailureCategoryV2:
    if reason_code == "planning_deadline_expired":
        return FailureCategoryV2.TIMEOUT
    if reason_code in {
        "legged_risk_objective_unsupported",
        "legged_accelerator_required_unsupported",
        "legged_hold_oracle_unavailable",
    }:
        return FailureCategoryV2.UNSUPPORTED_CAPABILITY
    if reason_code == "legged_goal_heading_unreachable":
        return FailureCategoryV2.GOAL_POSE_UNREACHABLE
    if reason_code in {
        "legged_expansion_budget_exhausted",
        "legged_route_state_budget_exhausted",
        "legged_search_memory_budget_exceeded",
        "route_state_budget_exceeded",
    }:
        return FailureCategoryV2.RESOURCE_LIMIT
    if reason_code == "legged_no_complete_route":
        return FailureCategoryV2.NO_COMPLETE_ROUTE
    if reason_code in {
        "legged_initial_state_contract_mismatch",
        "legged_candidate_generation_contract_mismatch",
        "legged_search_cost_contract_mismatch",
        "legged_step_oracle_contract_mismatch",
        "legged_route_oracle_contract_mismatch",
        "planning_deadline_contract_mismatch",
    }:
        return FailureCategoryV2.INTERNAL_ERROR
    if reason_code in {"route_goal_contract_mismatch", "route_goal_tolerance_exceeded"}:
        return FailureCategoryV2.GOAL_POSE_UNREACHABLE
    return FailureCategoryV2.VALIDATION_FAILED


def _provider_failure_v2(
    *,
    request_id: str,
    reason_code: str,
    stage: str,
    expanded_states: int,
    generated_primitives: int,
    rejected_l2: int,
    elapsed_s: float,
    details: tuple[tuple[str, str | int | float | bool | None], ...] = (),
    category: FailureCategoryV2 | None = None,
) -> PlanningFailureV2:
    timed_out = reason_code == "planning_deadline_expired"
    return PlanningFailureV2(
        request_id=request_id,
        platform_kind=PlatformKindV2.LEGGED,
        category=(
            _provider_category_v2(reason_code, stage)
            if category is None
            else category
        ),
        reason_code=reason_code,
        evidence=FailureEvidenceV2(
            stage=stage,
            checks=(reason_code,),
            details=details,
        ),
        search_telemetry=SearchTelemetryV2(
            expanded_states=expanded_states,
            generated_primitives=generated_primitives,
            rejected_l0=0,
            rejected_l1=0,
            rejected_l2=rejected_l2,
            elapsed_s=elapsed_s,
            timed_out=timed_out,
            accelerator_used=False,
            ackermann_feasible_claimed=False,
            termination_reason=reason_code,
        ),
    )


def _provider_exact_heading_equal_v2(left: float, right: float) -> bool:
    return (
        math.remainder(
            math.remainder(left, math.tau) - math.remainder(right, math.tau),
            math.tau,
        )
        == 0.0
    )


def _provider_canonical_heading_v2(value: float) -> float:
    if -pi <= value <= pi:
        return _provider_positive_zero_v2(value)
    wrapped = (value + pi) % (2.0 * pi) - pi
    if wrapped == -pi and value > 0.0:
        return pi
    return _provider_positive_zero_v2(wrapped)


def _provider_initial_state_v2(request: PlanningRequestV2) -> LeggedSearchStateV2:
    state = nominal_legged_search_state_v2(request.start_state)
    if type(state) is not LeggedSearchStateV2:
        raise TypeError("nominal state must be exact LeggedSearchStateV2")
    canonical_heading = _provider_canonical_heading_v2(request.start_state.heading_rad)
    expected_body = PoseStateV2(
        _provider_positive_zero_v2(request.start_state.x_m),
        _provider_positive_zero_v2(request.start_state.y_m),
        canonical_heading,
    )
    cosine = cos(canonical_heading)
    sine = sin(canonical_heading)
    expected_points = (
        (0.35, 0.25),
        (0.35, -0.25),
        (-0.35, 0.25),
        (-0.35, -0.25),
    )
    expected_contacts = tuple(
        LeggedFootContactV2(
            leg,
            WorldPoint(
                _provider_positive_zero_v2(
                    expected_body.x_m + cosine * local_x - sine * local_y
                ),
                _provider_positive_zero_v2(
                    expected_body.y_m + sine * local_x + cosine * local_y
                ),
            ),
        )
        for leg, (local_x, local_y) in zip(
            LEGGED_FOOT_STORAGE_ORDER_V2,
            expected_points,
            strict=True,
        )
    )
    expected = LeggedSearchStateV2(expected_body, expected_contacts, 0)
    if legged_state_key_v2(state) != legged_state_key_v2(expected):
        raise ValueError("nominal state postcondition mismatch")
    return state


def _provider_candidate_v2(
    state: LeggedSearchStateV2,
    fixed_yaw: float,
    offset: tuple[float, float],
) -> tuple[LeggedStepPrimitiveV2, LeggedStepCandidateV2]:
    phase = state.sequence_phase
    moving_leg = LEGGED_CRAWL_SEQUENCE_V2[phase]
    moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(moving_leg)
    source = state.foot_contacts[moving_index].foothold
    cosine = cos(fixed_yaw)
    sine = sin(fixed_yaw)
    for value, name in ((cosine, "cosine"), (sine, "sine")):
        _provider_float_word_v2(value, name)
    local_x, local_y = offset
    target_x = source.x + cosine * local_x - sine * local_y
    target_y = source.y + sine * local_x + cosine * local_y
    target = WorldPoint(
        _provider_positive_zero_v2(target_x),
        _provider_positive_zero_v2(target_y),
    )
    nonmoving = tuple(
        contact.foothold
        for index, contact in enumerate(state.foot_contacts)
        if index != moving_index
    )
    lift = PoseStateV2(
        _provider_positive_zero_v2(math.fsum(point.x for point in nonmoving) / 3.0),
        _provider_positive_zero_v2(math.fsum(point.y for point in nonmoving) / 3.0),
        fixed_yaw,
    )
    end_contacts = list(state.foot_contacts)
    end_contacts[moving_index] = LeggedFootContactV2(moving_leg, target)
    end_body = PoseStateV2(
        _provider_positive_zero_v2(
            math.fsum(contact.foothold.x for contact in end_contacts) / 4.0
        ),
        _provider_positive_zero_v2(
            math.fsum(contact.foothold.y for contact in end_contacts) / 4.0
        ),
        fixed_yaw,
    )
    end_state = LeggedSearchStateV2(
        end_body,
        tuple(end_contacts),
        (phase + 1) % 4,
    )
    resource_values = _call_contract_helper_v2(
        "legged provider relative resource",
        _resource_values_v2,
        state,
        lift,
        end_state,
        moving_leg,
        target,
    )
    if type(resource_values) is not tuple or len(resource_values) != 3:
        raise ValueError("legged resource helper must return exactly three values")
    foot_travel, distance, energy = resource_values
    for value, name in (
        (foot_travel, "foot_travel_m"),
        (distance, "distance_m"),
        (energy, "energy_cost"),
    ):
        _provider_float_word_v2(
            value,
            name,
            nonnegative=True,
            canonical_zero=True,
        )
    primitive = LeggedStepPrimitiveV2(
        kind=PrimitiveKindV2.LEG_STEP,
        start_state=state.body_state,
        end_state=end_body,
        duration_s=1.0,
        distance_m=distance,
        energy_cost=energy,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        start_legged_state=state,
        lift_body_state=lift,
        end_legged_state=end_state,
        moving_leg=moving_leg,
        target_foothold=target,
        foot_travel_m=foot_travel,
    )
    candidate = primitive.as_oracle_candidate()
    _seal_candidate_postcondition_v2(candidate, primitive)
    return primitive, candidate


def _provider_a2_token_v2(result: object) -> tuple[object, ...]:
    if type(result) is not LeggedValidationResultV2:
        raise TypeError("Search A2 must return exact LeggedValidationResultV2")
    evidence = result.evidence
    if type(evidence) is not ValidationEvidenceV2:
        raise TypeError("Search A2 evidence must be exact ValidationEvidenceV2")
    cell = result.failed_cell
    if cell is not None:
        if type(cell) is not Cell or type(cell.x) is not int or type(cell.y) is not int:
            raise TypeError("failed_cell must be exact Cell")
        cell_token: tuple[int, int] | None = (cell.x, cell.y)
    else:
        cell_token = None
    if result.failed_leg is not None and type(result.failed_leg) is not LegIdV2:
        raise TypeError("failed_leg must be exact LegIdV2")
    margin_word = (
        None
        if result.minimum_support_margin_m is None
        else _provider_float_word_v2(
            result.minimum_support_margin_m,
            "minimum_support_margin_m",
            nonnegative=True,
            canonical_zero=True,
        )
    )
    return (
        evidence.validator_id,
        evidence.level,
        evidence.passed,
        evidence.checks,
        result.reason_code,
        result.timed_out,
        cell_token,
        result.failed_leg,
        result.checked_cell_count,
        margin_word,
        result.minimum_support_margin_m,
    )


def _provider_audit_a2_v2(result: object) -> tuple[object, ...]:
    first = _provider_a2_token_v2(result)
    cell = None if first[6] is None else Cell(first[6][0], first[6][1])
    rebuilt = LeggedValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=first[0],
            level=first[1],
            passed=first[2],
            checks=first[3],
        ),
        reason_code=first[4],
        timed_out=first[5],
        failed_cell=cell,
        failed_leg=first[7],
        checked_cell_count=first[8],
        minimum_support_margin_m=first[10],
    )
    if _provider_a2_token_v2(rebuilt) != first or _provider_a2_token_v2(result) != first:
        raise ValueError("Search A2 result drifted during audit")
    return first


def _provider_cost_component_v2(
    parent: float,
    weight: float,
    resource: float,
    name: str,
) -> float:
    for value, value_name in (
        (parent, f"{name} parent"),
        (weight, f"{name} weight"),
        (resource, f"{name} resource"),
    ):
        _provider_nonnegative_float_v2(value, value_name)
    product = weight * resource
    _provider_nonnegative_float_v2(product, f"{name} product")
    total = parent + product
    return _provider_nonnegative_float_v2(total, name)


def _provider_child_costs_v2(
    parent: _LeggedNodeRecordV2,
    primitive: LeggedStepPrimitiveV2,
    request: PlanningRequestV2,
) -> tuple[float, float, float, float, float]:
    objective = request.objective_profile
    weight_token = tuple(
        _provider_float_word_v2(
            getattr(objective, name),
            f"objective.{name}",
            nonnegative=True,
            canonical_zero=True,
        )
        for name in (
            "distance_weight",
            "risk_weight",
            "energy_weight",
            "time_weight",
        )
    )
    resource_token = (
        _provider_float_word_v2(
            primitive.distance_m,
            "primitive.distance_m",
            nonnegative=True,
            canonical_zero=True,
        ),
        _provider_float_word_v2(
            primitive.energy_cost,
            "primitive.energy_cost",
            nonnegative=True,
            canonical_zero=True,
        ),
        _provider_float_word_v2(
            primitive.duration_s,
            "primitive.duration_s",
            nonnegative=True,
            canonical_zero=True,
        ),
    )
    distance = _provider_cost_component_v2(
        parent.distance_cost,
        objective.distance_weight,
        primitive.distance_m,
        "distance_cost",
    )
    risk = 0.0
    energy = _provider_cost_component_v2(
        parent.energy_cost,
        objective.energy_weight,
        primitive.energy_cost,
        "energy_cost",
    )
    time = _provider_cost_component_v2(
        parent.time_cost,
        objective.time_weight,
        primitive.duration_s,
        "time_cost",
    )
    total = sum((distance, risk, energy, time))
    _provider_nonnegative_float_v2(total, "path_cost")
    if weight_token != tuple(
        _provider_float_word_v2(
            getattr(objective, name),
            f"objective.{name}",
            nonnegative=True,
            canonical_zero=True,
        )
        for name in (
            "distance_weight",
            "risk_weight",
            "energy_weight",
            "time_weight",
        )
    ) or resource_token != (
        _provider_float_word_v2(
            primitive.distance_m,
            "primitive.distance_m",
            nonnegative=True,
            canonical_zero=True,
        ),
        _provider_float_word_v2(
            primitive.energy_cost,
            "primitive.energy_cost",
            nonnegative=True,
            canonical_zero=True,
        ),
        _provider_float_word_v2(
            primitive.duration_s,
            "primitive.duration_s",
            nonnegative=True,
            canonical_zero=True,
        ),
    ):
        raise ValueError("cost authority drifted during calculation")
    return distance, risk, energy, time, total


def _provider_final_token_v2(
    result: object,
    result_type: type,
) -> tuple[object, ...]:
    if type(result) is not result_type:
        raise TypeError("final validator returned wrong result type")
    evidence = result.evidence
    if type(evidence) is not ValidationEvidenceV2:
        raise TypeError("final evidence must be exact ValidationEvidenceV2")
    cell = result.failed_cell
    cell_token = None
    if cell is not None:
        if type(cell) is not Cell or type(cell.x) is not int or type(cell.y) is not int:
            raise TypeError("final failed_cell must be exact Cell")
        cell_token = (cell.x, cell.y)
    margin_word = (
        None
        if result.minimum_support_margin_m is None
        else _provider_float_word_v2(
            result.minimum_support_margin_m,
            "final minimum_support_margin_m",
            nonnegative=True,
            canonical_zero=True,
        )
    )
    return (
        evidence.validator_id,
        evidence.level,
        evidence.passed,
        evidence.checks,
        result.reason_code,
        result.timed_out,
        cell_token,
        result.failed_leg,
        result.failed_primitive_index,
        result.checked_cell_count,
        margin_word,
        result.minimum_support_margin_m,
        result.validated_route_hash,
    )


def _provider_audit_final_v2(
    result: object,
    result_type: type,
) -> tuple[object, ...]:
    first = _provider_final_token_v2(result, result_type)
    cell = None if first[6] is None else Cell(first[6][0], first[6][1])
    rebuilt = result_type(
        evidence=ValidationEvidenceV2(first[0], first[1], first[2], first[3]),
        reason_code=first[4],
        timed_out=first[5],
        failed_cell=cell,
        failed_leg=first[7],
        failed_primitive_index=first[8],
        checked_cell_count=first[9],
        minimum_support_margin_m=first[11],
        validated_route_hash=first[12],
    )
    if (
        _provider_final_token_v2(rebuilt, result_type) != first
        or _provider_final_token_v2(result, result_type) != first
    ):
        raise ValueError("final result drifted during audit")
    return first


@dataclass(frozen=True, slots=True)
class LeggedPrimitiveProviderV2:
    legged_profile: LeggedProfileV2

    def __post_init__(self) -> None:
        if type(self.legged_profile) is not LeggedProfileV2:
            raise TypeError("legged_profile must be exact LeggedProfileV2")

    @property
    def profile(self) -> PlatformProfileV2:
        return self.legged_profile.profile

    def plan(
        self,
        request: PlanningRequestV2,
        anchor: FineSafetyAnchorV2,
        deadline: PlanningDeadlineV2,
    ) -> PlanningOutcomeV2:
        if type(request) is not PlanningRequestV2:
            raise TypeError("request must be exact PlanningRequestV2")
        if type(anchor) is not FineSafetyAnchorV2:
            raise TypeError("anchor must be exact FineSafetyAnchorV2")
        if type(deadline) is not PlanningDeadlineV2:
            raise TypeError("deadline must be exact PlanningDeadlineV2")
        if type(self.legged_profile) is not LeggedProfileV2:
            raise TypeError("legged_profile must be exact LeggedProfileV2")

        expanded_states = 0
        generated_primitives = 0
        rejected_l2 = 0
        request_id = (
            request.request_id
            if type(request.request_id) is str and request.request_id
            else "legged-provider-invalid-request"
        )
        guard: _LeggedAuthorityGuardV2 | None = None

        def elapsed() -> float:
            if guard is None:
                return 0.0
            value = max(0.0, guard.last_now - guard.started)
            if type(value) is not float or not isfinite(value):
                return 0.0
            return _provider_positive_zero_v2(value)

        def fail(
            reason_code: str,
            stage: str,
            details: tuple[tuple[str, str | int | float | bool | None], ...] = (),
            category: FailureCategoryV2 | None = None,
        ) -> PlanningFailureV2:
            return _provider_failure_v2(
                request_id=request_id,
                reason_code=reason_code,
                stage=stage,
                expanded_states=expanded_states,
                generated_primitives=generated_primitives,
                rejected_l2=rejected_l2,
                elapsed_s=elapsed(),
                details=details,
                category=category,
            )

        def checkpoint(stage: str) -> PlanningFailureV2 | None:
            assert guard is not None
            try:
                guard.check()
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except _LeggedDeadlineExpired:
                return fail("planning_deadline_expired", stage)
            except _LeggedAuthorityFailure as error:
                return fail(error.reason_code, stage)
            return None

        try:
            guard = _LeggedAuthorityGuardV2(
                request,
                anchor,
                self.legged_profile,
                deadline,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except _LeggedAuthorityFailure as error:
            return fail(error.reason_code, "provider_entry")
        stopped = checkpoint("provider_entry")
        if stopped is not None:
            return stopped

        try:
            risk_weight = request.objective_profile.risk_weight
            if type(risk_weight) is not float or risk_weight != 0.0:
                return fail("legged_risk_objective_unsupported", "capability_check")
            if request.accelerator_policy is AcceleratorPolicyV2.REQUIRED:
                return fail(
                    "legged_accelerator_required_unsupported",
                    "capability_check",
                )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return fail("planning_request_contract_mismatch", "capability_check")

        try:
            initial_state = _provider_initial_state_v2(request)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return fail("legged_initial_state_contract_mismatch", "initial_state")
        fixed_yaw = initial_state.body_state.heading_rad
        try:
            if not _provider_exact_heading_equal_v2(
                request.goal_state.heading_rad,
                fixed_yaw,
            ):
                return fail("legged_goal_heading_unreachable", "goal_semantics")
            physical_hold = (
                request.start_state.x_m == request.goal_state.x_m
                and request.start_state.y_m == request.goal_state.y_m
                and _provider_exact_heading_equal_v2(
                    request.start_state.heading_rad,
                    request.goal_state.heading_rad,
                )
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return fail("planning_request_contract_mismatch", "goal_semantics")
        if physical_hold:
            return fail("legged_hold_oracle_unavailable", "capability_check")

        budget = request.resource_budget
        max_expanded = budget.max_expanded_states
        max_route_states = budget.max_route_states
        max_memory_bytes = budget.max_memory_bytes
        effective_cap = min(max_route_states, MAX_REPLAY_STEPS + 1)
        if max_expanded == 0:
            return fail(
                "legged_expansion_budget_exhausted",
                "resource_check",
                (("attempted_expanded_states", 1), ("max_expanded_states", 0)),
            )
        if effective_cap <= 1:
            return fail(
                "legged_route_state_budget_exhausted",
                "resource_check",
                (
                    ("attempted_route_states", 2),
                    ("effective_max_route_states", effective_cap),
                    ("requested_max_route_states", max_route_states),
                ),
            )
        stopped = checkpoint("preflight")
        if stopped is not None:
            return stopped

        admitted_record_count = 0
        prospective_count = 1
        if (
            max_memory_bytes > 0
            and prospective_count * LEGGED_SEARCH_RECORD_BYTES_V2 > max_memory_bytes
        ):
            return fail(
                "legged_search_memory_budget_exceeded",
                "resource_check",
                (
                    ("accounting_id", LEGGED_SEARCH_MEMORY_ACCOUNTING_ID_V2),
                    ("attempted_record_count", prospective_count),
                    ("max_memory_bytes", max_memory_bytes),
                    ("record_bytes", LEGGED_SEARCH_RECORD_BYTES_V2),
                    ("retained_record_count", admitted_record_count),
                ),
            )

        try:
            start_key = legged_state_key_v2(initial_state)
            start_id = "legged-node-00000000000000000000"
            start_node = _LeggedNodeRecordV2(
                candidate_id=start_id,
                state=initial_state,
                state_key=start_key,
                depth=0,
                distance_cost=0.0,
                risk_cost=0.0,
                energy_cost=0.0,
                time_cost=0.0,
                path_cost=0.0,
                parent_candidate_id=None,
                incoming_primitive=None,
            )
            queue = StableSearchQueueV2()
            queue.extend(
                (
                    SearchQueueEntryV2(
                        candidate_id=start_id,
                        state_key=start_key,
                        primitive_key="start",
                        path_cost=0.0,
                        anchor_heuristic=0.0,
                        auxiliary_heuristics=(),
                    ),
                )
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return fail("legged_initial_state_contract_mismatch", "initial_state")
        admitted_record_count = 1
        nodes = {start_id: start_node}
        best_by_state: dict[tuple[int, ...], tuple[float, str]] = {
            start_key: (0.0, start_id)
        }
        closed: set[tuple[int, ...]] = set()
        serial = 1
        route_budget_blocked = False
        route_budget_attempted_states: int | None = None
        goal_node: _LeggedNodeRecordV2 | None = None

        while len(queue) > 0:
            stopped = checkpoint("search_pop")
            if stopped is not None:
                return stopped
            try:
                entry = queue.pop_anchor()
                node = nodes[entry.candidate_id]
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except Exception:
                return fail("legged_search_cost_contract_mismatch", "search_cost")
            best = best_by_state.get(entry.state_key)
            if best is None or best != (entry.path_cost, entry.candidate_id):
                continue
            if entry.state_key in closed:
                continue
            body = node.state.body_state
            try:
                at_goal = (
                    body.x_m == request.goal_state.x_m
                    and body.y_m == request.goal_state.y_m
                    and _provider_exact_heading_equal_v2(
                        body.heading_rad,
                        request.goal_state.heading_rad,
                    )
                )
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except Exception:
                return fail("planning_request_contract_mismatch", "search")
            if at_goal:
                goal_node = node
                break
            if expanded_states >= max_expanded:
                return fail(
                    "legged_expansion_budget_exhausted",
                    "search",
                    (
                        ("attempted_expanded_states", expanded_states + 1),
                        ("max_expanded_states", max_expanded),
                    ),
                )
            closed.add(entry.state_key)
            expanded_states += 1
            phase = node.state.sequence_phase

            for offset_index, offset in enumerate(LEGGED_LOCAL_FOOTHOLD_OFFSETS_V2):
                try:
                    primitive, candidate = _provider_candidate_v2(
                        node.state,
                        fixed_yaw,
                        offset,
                    )
                    candidate_token = (
                        _canonical_pose_snapshot_v2(candidate.start_body_state, "candidate.start"),
                        _canonical_pose_snapshot_v2(candidate.lift_body_state, "candidate.lift"),
                        _canonical_pose_snapshot_v2(candidate.end_body_state, "candidate.end"),
                        _canonical_contacts_snapshot_v2(candidate.foot_contacts, "candidate.contacts"),
                        candidate.moving_leg,
                        candidate.sequence_phase,
                        _canonical_point_snapshot_v2(candidate.target_foothold, "candidate.target"),
                    )
                except (KeyboardInterrupt, SystemExit, MemoryError):
                    raise
                except Exception:
                    return fail(
                        "legged_candidate_generation_contract_mismatch",
                        "candidate_generation",
                    )

                stopped = checkpoint("search_edge_validation")
                if stopped is not None:
                    return stopped
                child_id = f"legged-node-{serial:020d}"
                serial += 1
                generated_primitives += 1
                try:
                    a2_result = _TRUSTED_VALIDATE_LEGGED_STEP_L2_V2(
                        candidate,
                        anchor,
                        self.legged_profile,
                        deadline,
                    )
                except (KeyboardInterrupt, SystemExit, MemoryError):
                    raise
                except Exception:
                    stopped = checkpoint("search_edge_validation")
                    if stopped is not None:
                        return stopped
                    return fail(
                        "legged_step_oracle_contract_mismatch",
                        "search_edge_validation",
                    )
                stopped = checkpoint("search_edge_validation")
                if stopped is not None:
                    return stopped
                try:
                    _seal_primitive_canonical_postcondition_v2(primitive)
                    audited_primitive = _audit_primitive_canonical(primitive)
                    if audited_primitive is not primitive:
                        raise ValueError("primitive audit changed object identity")
                    _seal_primitive_canonical_postcondition_v2(primitive)
                    _seal_candidate_postcondition_v2(candidate, primitive)
                    if candidate_token != (
                        _canonical_pose_snapshot_v2(candidate.start_body_state, "candidate.start"),
                        _canonical_pose_snapshot_v2(candidate.lift_body_state, "candidate.lift"),
                        _canonical_pose_snapshot_v2(candidate.end_body_state, "candidate.end"),
                        _canonical_contacts_snapshot_v2(candidate.foot_contacts, "candidate.contacts"),
                        candidate.moving_leg,
                        candidate.sequence_phase,
                        _canonical_point_snapshot_v2(candidate.target_foothold, "candidate.target"),
                    ):
                        raise ValueError("candidate mutated during Search A2")
                    a2_token = _provider_audit_a2_v2(a2_result)
                except (KeyboardInterrupt, SystemExit, MemoryError):
                    raise
                except Exception:
                    return fail(
                        "legged_step_oracle_contract_mismatch",
                        "search_edge_validation",
                    )
                a2_reason = a2_token[4]
                if a2_reason in _LEGGED_STEP_FAILURE_REASONS_V2:
                    rejected_l2 += 1
                    continue
                if a2_reason == "planning_deadline_expired":
                    return fail("planning_deadline_expired", "search_edge_validation")
                if a2_reason in _LEGGED_GLOBAL_A2_FAILURE_REASONS_V2:
                    return fail(
                        a2_reason,
                        "search_edge_validation",
                        category=FailureCategoryV2.VALIDATION_FAILED,
                    )
                if (
                    a2_reason != "legged_step_l2_valid"
                    or a2_token[0] != LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2
                    or a2_token[1] is not ValidationLevelV2.L2
                    or a2_token[2] is not True
                    or a2_token[3] != ("legged_step_l2_valid",)
                ):
                    return fail(
                        "legged_step_oracle_contract_mismatch",
                        "search_edge_validation",
                    )

                try:
                    costs = _provider_child_costs_v2(node, primitive, request)
                    child_key = legged_state_key_v2(primitive.end_legged_state)
                except (KeyboardInterrupt, SystemExit, MemoryError):
                    raise
                except Exception:
                    return fail(
                        "legged_search_cost_contract_mismatch",
                        "search_cost",
                    )
                existing = best_by_state.get(child_key)
                if existing is not None and not costs[4] < existing[0]:
                    continue
                child_depth = node.depth + 1
                if child_depth + 1 > effective_cap:
                    route_budget_blocked = True
                    if route_budget_attempted_states is None:
                        route_budget_attempted_states = child_depth + 1
                    continue
                prospective_count = admitted_record_count + 1
                if (
                    max_memory_bytes > 0
                    and prospective_count * LEGGED_SEARCH_RECORD_BYTES_V2
                    > max_memory_bytes
                ):
                    return fail(
                        "legged_search_memory_budget_exceeded",
                        "search",
                        (
                            ("accounting_id", LEGGED_SEARCH_MEMORY_ACCOUNTING_ID_V2),
                            ("attempted_record_count", prospective_count),
                            ("max_memory_bytes", max_memory_bytes),
                            ("record_bytes", LEGGED_SEARCH_RECORD_BYTES_V2),
                            ("retained_record_count", admitted_record_count),
                        ),
                    )
                child = _LeggedNodeRecordV2(
                    candidate_id=child_id,
                    state=primitive.end_legged_state,
                    state_key=child_key,
                    depth=child_depth,
                    distance_cost=costs[0],
                    risk_cost=costs[1],
                    energy_cost=costs[2],
                    time_cost=costs[3],
                    path_cost=costs[4],
                    parent_candidate_id=node.candidate_id,
                    incoming_primitive=primitive,
                )
                try:
                    queue.extend(
                        (
                            SearchQueueEntryV2(
                                candidate_id=child_id,
                                state_key=child_key,
                                primitive_key=f"{phase}:{offset_index:02d}",
                                path_cost=child.path_cost,
                                anchor_heuristic=0.0,
                                auxiliary_heuristics=(),
                            ),
                        )
                    )
                except (KeyboardInterrupt, SystemExit, MemoryError):
                    raise
                except Exception:
                    return fail(
                        "legged_search_cost_contract_mismatch",
                        "search_cost",
                    )
                admitted_record_count = prospective_count
                nodes[child_id] = child
                best_by_state[child_key] = (child.path_cost, child_id)

        if goal_node is None:
            if route_budget_blocked:
                return fail(
                    "legged_route_state_budget_exhausted",
                    "search",
                    (
                        (
                            "attempted_route_states",
                            route_budget_attempted_states,
                        ),
                        ("effective_max_route_states", effective_cap),
                        ("requested_max_route_states", max_route_states),
                    ),
                )
            return fail("legged_no_complete_route", "search")

        reverse_primitives: list[LeggedStepPrimitiveV2] = []
        cursor = goal_node
        try:
            while cursor.parent_candidate_id is not None:
                if type(cursor.incoming_primitive) is not LeggedStepPrimitiveV2:
                    raise ValueError("route lineage missing incoming primitive")
                reverse_primitives.append(cursor.incoming_primitive)
                cursor = nodes[cursor.parent_candidate_id]
            primitives = tuple(reversed(reverse_primitives))
            if not primitives:
                raise ValueError("legged route must be nonempty")
            route = TypedRouteV2(
                platform_kind=PlatformKindV2.LEGGED,
                primitives=primitives,
                total_cost=goal_node.path_cost,
                is_complete=True,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return fail(
                "legged_candidate_generation_contract_mismatch",
                "candidate_generation",
            )

        stopped = checkpoint("route_validation")
        if stopped is not None:
            return stopped
        from path_planner.v2.validation import (
            _TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2,
            _TRUSTED_LEGGED_ROUTE_RESULT_TYPE_V2,
            _TRUSTED_VALIDATE_LEGGED_ROUTE_L2_V2,
        )

        trusted_route_digest = _TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2
        trusted_result_type = _TRUSTED_LEGGED_ROUTE_RESULT_TYPE_V2
        trusted_validate_route = _TRUSTED_VALIDATE_LEGGED_ROUTE_L2_V2
        try:
            pre_digest = trusted_route_digest(route)
            final_result = trusted_validate_route(
                route,
                request,
                anchor,
                self.legged_profile,
                deadline,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            stopped = checkpoint("route_validation")
            if stopped is not None:
                return stopped
            return fail("legged_route_oracle_contract_mismatch", "route_validation")
        stopped = checkpoint("route_validation")
        if stopped is not None:
            return stopped
        try:
            post_digest = trusted_route_digest(route)
            final_token = _provider_audit_final_v2(
                final_result,
                trusted_result_type,
            )
            if pre_digest != post_digest:
                raise ValueError("route digest drifted during final validation")
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return fail("legged_route_oracle_contract_mismatch", "route_validation")

        final_reason = final_token[4]
        if final_reason != "legged_route_l2_valid":
            final_category = (
                FailureCategoryV2.TIMEOUT
                if final_reason == "planning_deadline_expired"
                else FailureCategoryV2.RESOURCE_LIMIT
                if final_reason == "route_state_budget_exceeded"
                else FailureCategoryV2.GOAL_POSE_UNREACHABLE
                if final_reason
                in {"route_goal_contract_mismatch", "route_goal_tolerance_exceeded"}
                else FailureCategoryV2.INTERNAL_ERROR
                if final_reason == "legged_step_oracle_contract_mismatch"
                else FailureCategoryV2.VALIDATION_FAILED
            )
            return fail(
                final_reason,
                "route_validation",
                category=final_category,
            )
        if (
            final_token[0] != "path-planner-v2-legged-route-l2/v1"
            or final_token[1] is not ValidationLevelV2.L2
            or final_token[2] is not True
            or final_token[3] != ("legged_route_l2_valid",)
            or type(final_token[12]) is not str
            or final_token[12] != pre_digest
        ):
            return fail("legged_route_oracle_contract_mismatch", "route_validation")

        stopped = checkpoint("success_return")
        if stopped is not None:
            return stopped
        try:
            guard.seal_without_clock()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except _LeggedAuthorityFailure as error:
            return fail(error.reason_code, "success_return")
        try:
            final_digest = trusted_route_digest(route)
            final_replay_token = _provider_audit_final_v2(
                final_result,
                trusted_result_type,
            )
            if final_digest != pre_digest or final_replay_token != final_token:
                raise ValueError("final route authority drifted before return")
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            return fail("legged_route_oracle_contract_mismatch", "route_validation")
        costs = CostBreakdownV2(
            distance_cost=goal_node.distance_cost,
            risk_cost=goal_node.risk_cost,
            energy_cost=goal_node.energy_cost,
            time_cost=goal_node.time_cost,
            total_cost=goal_node.path_cost,
        )
        samples: list[PoseStateV2] = [primitives[0].start_state]
        for primitive in primitives:
            samples.extend((primitive.lift_body_state, primitive.end_state))
        return PlanningSuccessV2(
            request_id=request_id,
            platform_kind=PlatformKindV2.LEGGED,
            route=route,
            observation_projection=ObservationProjectionV2(
                source=_LEGGED_OBSERVATION_SOURCE_V2,
                sample_states=tuple(samples),
                expected_new_observed_cells=0.0,
                expected_information_gain=0.0,
            ),
            cost_breakdown=costs,
            validation_evidence=final_result.evidence,
            search_telemetry=SearchTelemetryV2(
                expanded_states=expanded_states,
                generated_primitives=generated_primitives,
                rejected_l0=0,
                rejected_l1=0,
                rejected_l2=rejected_l2,
                elapsed_s=elapsed(),
                timed_out=False,
                accelerator_used=False,
                ackermann_feasible_claimed=False,
                termination_reason="legged_route_l2_valid",
            ),
            cache_evidence=CacheEvidenceV2(
                cache_namespace=_LEGGED_CACHE_NAMESPACE_DISABLED_V2,
                cache_key=_LEGGED_CACHE_KEY_DISABLED_V2,
                hit=False,
            ),
        )


__all__ = [
    "LEGGED_CAPABILITY_LEVEL_V2",
    "LEGGED_LOCAL_FOOTHOLD_OFFSETS_V2",
    "LEGGED_RESOURCE_PROXY_ID_V2",
    "LEGGED_SEARCH_MEMORY_ACCOUNTING_ID_V2",
    "LEGGED_SEARCH_RECORD_BYTES_V2",
    "LEGGED_SEARCH_STATE_SCHEMA_V2",
    "LEGGED_STEP_PRIMITIVE_SCHEMA_V2",
    "LeggedPrimitiveProviderV2",
    "LeggedSearchStateV2",
    "LeggedStepPrimitiveV2",
    "legged_state_key_v2",
    "nominal_legged_search_state_v2",
]
