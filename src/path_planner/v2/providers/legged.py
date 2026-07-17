from __future__ import annotations

import struct
from dataclasses import dataclass
from math import cos, copysign, hypot, isclose, isfinite, pi, sin

from path_planner.core import WorldPoint
from path_planner.v2.contracts import (
    PoseStateV2,
    PrimitiveKindV2,
    RoutePrimitiveV2,
    ValidationLevelV2,
)
from path_planner.v2.oracles.legged import (
    LEGGED_CRAWL_SEQUENCE_V2,
    LEGGED_FOOT_STORAGE_ORDER_V2,
    LeggedFootContactV2,
    LeggedStepCandidateV2,
    LegIdV2,
)


LEGGED_SEARCH_STATE_SCHEMA_V2 = "path-planner-v2-legged-search-state/v1"
LEGGED_STEP_PRIMITIVE_SCHEMA_V2 = "path-planner-v2-legged-step/v1"
LEGGED_RESOURCE_PROXY_ID_V2 = "legged_static_crawl_relative_resource/v1"
LEGGED_CAPABILITY_LEVEL_V2 = "simulation_proxy"

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


__all__ = [
    "LEGGED_CAPABILITY_LEVEL_V2",
    "LEGGED_RESOURCE_PROXY_ID_V2",
    "LEGGED_SEARCH_STATE_SCHEMA_V2",
    "LEGGED_STEP_PRIMITIVE_SCHEMA_V2",
    "LeggedSearchStateV2",
    "LeggedStepPrimitiveV2",
    "legged_state_key_v2",
    "nominal_legged_search_state_v2",
]
