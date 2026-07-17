from __future__ import annotations

import os
import struct
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from math import isfinite, pi

import numpy as np
import pytest

import path_planner.v2 as v2
import path_planner.v2.providers as provider_exports
import path_planner.v2.providers.legged as legged_module
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
from path_planner.v2.providers.legged import (
    LEGGED_CAPABILITY_LEVEL_V2,
    LEGGED_RESOURCE_PROXY_ID_V2,
    LEGGED_SEARCH_STATE_SCHEMA_V2,
    LEGGED_STEP_PRIMITIVE_SCHEMA_V2,
    LeggedSearchStateV2,
    LeggedStepPrimitiveV2,
    legged_state_key_v2,
    nominal_legged_search_state_v2,
)
from path_planner.v2.serialization import canonical_json_bytes


class FloatSubclass(float):
    pass


class IntSubclass(int):
    pass


class StringSubclass(str):
    pass


class TupleSubclass(tuple):
    pass


class PoseSubclass(PoseStateV2):
    pass


class PointSubclass(WorldPoint):
    pass


class ContactSubclass(LeggedFootContactV2):
    pass


class SearchStateSubclass(LeggedSearchStateV2):
    pass


def _contacts(
    *,
    fl: WorldPoint = WorldPoint(0.35, 0.25),
    fr: WorldPoint = WorldPoint(0.35, -0.25),
    rl: WorldPoint = WorldPoint(-0.35, 0.25),
    rr: WorldPoint = WorldPoint(-0.35, -0.25),
) -> tuple[LeggedFootContactV2, ...]:
    return (
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, fl),
        LeggedFootContactV2(LegIdV2.FRONT_RIGHT, fr),
        LeggedFootContactV2(LegIdV2.REAR_LEFT, rl),
        LeggedFootContactV2(LegIdV2.REAR_RIGHT, rr),
    )


def _state(
    body: PoseStateV2 = PoseStateV2(0.0, 0.0, 0.0),
    contacts: tuple[LeggedFootContactV2, ...] | None = None,
    phase: int = 0,
    schema: str = LEGGED_SEARCH_STATE_SCHEMA_V2,
) -> LeggedSearchStateV2:
    return LeggedSearchStateV2(
        body_state=body,
        foot_contacts=_contacts() if contacts is None else contacts,
        sequence_phase=phase,
        schema_version=schema,
    )


def _primitive_values() -> dict[str, object]:
    start = _state()
    target = WorldPoint(0.60, 0.25)
    end_contacts = _contacts(fl=target)
    end = _state(PoseStateV2(0.0, -0.10, 0.0), end_contacts, 1)
    return {
        "kind": PrimitiveKindV2.LEG_STEP,
        "start_state": start.body_state,
        "end_state": end.body_state,
        "duration_s": 1.0,
        "distance_m": 0.10,
        "energy_cost": 0.35,
        "observation_contribution": 0.0,
        "validation_level": ValidationLevelV2.L2,
        "start_legged_state": start,
        "lift_body_state": PoseStateV2(0.0, -0.10, 0.0),
        "end_legged_state": end,
        "moving_leg": LegIdV2.FRONT_LEFT,
        "target_foothold": target,
        "foot_travel_m": 0.25,
        "capability": LEGGED_CAPABILITY_LEVEL_V2,
        "resource_proxy_id": LEGGED_RESOURCE_PROXY_ID_V2,
        "primitive_schema_version": LEGGED_STEP_PRIMITIVE_SCHEMA_V2,
    }


def _primitive(**overrides: object) -> LeggedStepPrimitiveV2:
    values = _primitive_values()
    values.update(overrides)
    return LeggedStepPrimitiveV2(**values)


def _zero_primitive(**overrides: object) -> LeggedStepPrimitiveV2:
    start = _state(PoseStateV2(-0.0, -0.0, -0.0))
    end = _state(PoseStateV2(-0.0, -0.0, -0.0), start.foot_contacts, 1)
    values: dict[str, object] = {
        "kind": PrimitiveKindV2.LEG_STEP,
        "start_state": PoseStateV2(-0.0, -0.0, -0.0),
        "end_state": PoseStateV2(-0.0, -0.0, -0.0),
        "duration_s": 1.0,
        "distance_m": -0.0,
        "energy_cost": -0.0,
        "observation_contribution": -0.0,
        "validation_level": ValidationLevelV2.L2,
        "start_legged_state": start,
        "lift_body_state": PoseStateV2(-0.0, -0.0, -0.0),
        "end_legged_state": end,
        "moving_leg": LegIdV2.FRONT_LEFT,
        "target_foothold": start.foot_contacts[0].foothold,
        "foot_travel_m": -0.0,
        "capability": LEGGED_CAPABILITY_LEVEL_V2,
        "resource_proxy_id": LEGGED_RESOURCE_PROXY_ID_V2,
        "primitive_schema_version": LEGGED_STEP_PRIMITIVE_SCHEMA_V2,
    }
    values.update(overrides)
    return LeggedStepPrimitiveV2(**values)


def _forge_pose(field: str, value: object) -> PoseStateV2:
    pose = PoseStateV2(1.0, 2.0, 0.25)
    object.__setattr__(pose, field, value)
    return pose


def _float_bits(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), "big", signed=False)


def test_public_legged_provider_constants_are_frozen_and_exported() -> None:
    expected = {
        "LEGGED_SEARCH_STATE_SCHEMA_V2": "path-planner-v2-legged-search-state/v1",
        "LEGGED_STEP_PRIMITIVE_SCHEMA_V2": "path-planner-v2-legged-step/v1",
        "LEGGED_RESOURCE_PROXY_ID_V2": "legged_static_crawl_relative_resource/v1",
        "LEGGED_CAPABILITY_LEVEL_V2": "simulation_proxy",
    }
    for name, value in expected.items():
        assert globals()[name] == value
        assert getattr(provider_exports, name) == value
        assert getattr(v2, name) == value


def test_public_types_functions_and_dataclass_field_order_are_frozen() -> None:
    names = (
        "LeggedSearchStateV2",
        "LeggedStepPrimitiveV2",
        "nominal_legged_search_state_v2",
        "legged_state_key_v2",
    )
    for name in names:
        assert getattr(provider_exports, name) is globals()[name]
        assert getattr(v2, name) is globals()[name]
    assert [field.name for field in fields(LeggedSearchStateV2)] == [
        "body_state",
        "foot_contacts",
        "sequence_phase",
        "schema_version",
    ]
    assert [field.name for field in fields(LeggedStepPrimitiveV2)] == [
        "kind",
        "start_state",
        "end_state",
        "duration_s",
        "distance_m",
        "energy_cost",
        "observation_contribution",
        "validation_level",
        "start_legged_state",
        "lift_body_state",
        "end_legged_state",
        "moving_leg",
        "target_foothold",
        "foot_travel_m",
        "capability",
        "resource_proxy_id",
        "primitive_schema_version",
    ]
    state = _state()
    primitive = _primitive()
    assert isinstance(primitive, RoutePrimitiveV2)
    assert not hasattr(state, "__dict__")
    assert not hasattr(primitive, "__dict__")
    with pytest.raises(FrozenInstanceError):
        state.sequence_phase = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        primitive.foot_travel_m = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("left", "right", "expected_heading"),
    [
        (0.0, 2.0 * pi, 0.0),
        (pi, 3.0 * pi, pi),
        (-pi, -3.0 * pi, -pi),
    ],
)
def test_state_canonical_heading_equivalence_and_signed_zero(
    left: float,
    right: float,
    expected_heading: float,
) -> None:
    first = _state(PoseStateV2(-0.0, -0.0, left))
    second = _state(PoseStateV2(0.0, 0.0, right))
    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert legged_state_key_v2(first) == legged_state_key_v2(second)
    assert first.body_state.heading_rad == expected_heading
    assert not np.signbit(first.body_state.x_m)
    assert not np.signbit(first.body_state.y_m)
    assert not np.signbit(first.body_state.heading_rad) if expected_heading == 0.0 else True


def test_state_preserves_positive_and_negative_half_turn_sweep_ties() -> None:
    positive = _state(PoseStateV2(0.0, 0.0, pi))
    negative = _state(PoseStateV2(0.0, 0.0, -pi))
    assert positive.body_state.heading_rad == pi
    assert negative.body_state.heading_rad == -pi
    assert positive != negative
    assert legged_state_key_v2(positive) != legged_state_key_v2(negative)


@pytest.mark.parametrize("field", ["x_m", "y_m", "heading_rad"])
@pytest.mark.parametrize(
    "bad",
    [1, True, np.float64(1.0), FloatSubclass(1.0), float("nan"), float("inf")],
)
def test_state_rejects_nonexact_or_nonfinite_body_fields(
    field: str,
    bad: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _state(_forge_pose(field, bad))


def test_state_rejects_body_and_contact_container_aliases() -> None:
    with pytest.raises(TypeError):
        _state(PoseSubclass(0.0, 0.0, 0.0))
    contacts = _contacts()
    invalid = (
        list(contacts),
        TupleSubclass(contacts),
        contacts[:3],
        contacts + (contacts[-1],),
        (contacts[1], contacts[0], contacts[2], contacts[3]),
        (contacts[0], contacts[0], contacts[2], contacts[3]),
        (ContactSubclass(contacts[0].leg_id, contacts[0].foothold), *contacts[1:]),
    )
    for value in invalid:
        with pytest.raises((TypeError, ValueError)):
            LeggedSearchStateV2(PoseStateV2(0.0, 0.0, 0.0), value, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("target", "field", "bad"),
    [
        ("leg", "leg_id", "front_left"),
        ("point", "foothold", PointSubclass(0.35, 0.25)),
        ("x", "x", 1),
        ("x", "x", FloatSubclass(0.35)),
        ("x", "x", float("nan")),
        ("x", "x", -0.0),
        ("y", "y", np.float64(0.25)),
        ("y", "y", float("inf")),
        ("y", "y", -0.0),
    ],
)
def test_state_rejects_contact_postconstruction_drift(
    target: str,
    field: str,
    bad: object,
) -> None:
    contacts = list(_contacts())
    contact = contacts[0]
    if target in {"leg", "point"}:
        object.__setattr__(contact, field, bad)
    else:
        object.__setattr__(contact.foothold, field, bad)
    with pytest.raises((TypeError, ValueError)):
        _state(contacts=tuple(contacts))


@pytest.mark.parametrize("phase", [True, -1, 4, 1.0, IntSubclass(1)])
def test_state_rejects_invalid_phase(phase: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        LeggedSearchStateV2(
            PoseStateV2(0.0, 0.0, 0.0),
            _contacts(),
            phase,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "schema",
    ["wrong", StringSubclass(LEGGED_SEARCH_STATE_SCHEMA_V2), 1],
)
def test_state_rejects_invalid_schema(schema: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        LeggedSearchStateV2(
            PoseStateV2(0.0, 0.0, 0.0),
            _contacts(),
            0,
            schema,  # type: ignore[arg-type]
        )


def test_nominal_stance_rotates_canonical_body_local_contacts() -> None:
    state = nominal_legged_search_state_v2(PoseStateV2(1.0, 2.0, pi / 2.0))
    assert state.sequence_phase == 0
    assert state.body_state == PoseStateV2(1.0, 2.0, pi / 2.0)
    expected = (
        (LegIdV2.FRONT_LEFT, 0.75, 2.35),
        (LegIdV2.FRONT_RIGHT, 1.25, 2.35),
        (LegIdV2.REAR_LEFT, 0.75, 1.65),
        (LegIdV2.REAR_RIGHT, 1.25, 1.65),
    )
    for contact, (leg, x_m, y_m) in zip(state.foot_contacts, expected, strict=True):
        assert contact.leg_id is leg
        assert contact.foothold.x == pytest.approx(x_m)
        assert contact.foothold.y == pytest.approx(y_m)


@pytest.mark.parametrize(
    ("left", "right"),
    [(0.0, 2.0 * pi), (pi, 3.0 * pi), (-pi, -3.0 * pi)],
)
def test_nominal_uses_canonical_heading_before_trig(
    left: float,
    right: float,
) -> None:
    first = nominal_legged_search_state_v2(PoseStateV2(1.0, 2.0, left))
    second = nominal_legged_search_state_v2(PoseStateV2(1.0, 2.0, right))
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert legged_state_key_v2(first) == legged_state_key_v2(second)


def test_nominal_preserves_half_turn_tie_and_candidate_bytes() -> None:
    positive = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, pi))
    negative = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, -pi))
    assert legged_state_key_v2(positive) != legged_state_key_v2(negative)
    from_zero = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    from_full_turn = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 2.0 * pi))
    candidates = []
    for state in (from_zero, from_full_turn):
        candidates.append(
            LeggedStepCandidateV2(
                start_body_state=state.body_state,
                lift_body_state=PoseStateV2(0.0, -0.1, 0.0),
                end_body_state=PoseStateV2(0.0, -0.1, 0.0),
                foot_contacts=state.foot_contacts,
                moving_leg=LegIdV2.FRONT_LEFT,
                sequence_phase=0,
                target_foothold=WorldPoint(0.6, 0.25),
            )
        )
    assert canonical_json_bytes(candidates[0]) == canonical_json_bytes(candidates[1])


@pytest.mark.parametrize("field", ["x_m", "y_m", "heading_rad"])
@pytest.mark.parametrize(
    "bad",
    [1, True, np.float64(1.0), FloatSubclass(1.0), float("nan"), float("inf")],
)
def test_nominal_rejects_nested_drift_before_trig(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad: object,
) -> None:
    monkeypatch.setattr(
        legged_module,
        "_trig_components_v2",
        lambda _heading: (_ for _ in ()).throw(AssertionError("trig called")),
    )
    with pytest.raises((TypeError, ValueError)):
        nominal_legged_search_state_v2(_forge_pose(field, bad))


def test_nominal_rejects_top_level_subclass_and_canonicalizes_signed_zero() -> None:
    with pytest.raises(TypeError):
        nominal_legged_search_state_v2(PoseSubclass(0.0, 0.0, 0.0))
    state = nominal_legged_search_state_v2(PoseStateV2(-0.0, -0.0, -0.0))
    for value in (
        state.body_state.x_m,
        state.body_state.y_m,
        state.body_state.heading_rad,
        *(
            coordinate
            for contact in state.foot_contacts
            for coordinate in (contact.foothold.x, contact.foothold.y)
        ),
    ):
        if value == 0.0:
            assert not np.signbit(value)


def test_nominal_max_finite_is_finite_and_deterministic() -> None:
    body = PoseStateV2(sys.float_info.max, -sys.float_info.max, 0.0)
    first = nominal_legged_search_state_v2(body)
    second = nominal_legged_search_state_v2(body)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert all(
        isfinite(value)
        for contact in first.foot_contacts
        for value in (contact.foothold.x, contact.foothold.y)
    )


@pytest.mark.parametrize("mode", ["raise", "nonfinite"])
def test_nominal_converts_ordinary_trig_faults_to_value_error(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    def fault(_heading: float) -> tuple[float, float]:
        if mode == "raise":
            raise RuntimeError("ordinary trig fault")
        return float("inf"), 0.0

    monkeypatch.setattr(legged_module, "_trig_components_v2", fault)
    with pytest.raises(ValueError):
        nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), MemoryError()])
def test_nominal_propagates_critical_baseexceptions(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    monkeypatch.setattr(
        legged_module,
        "_trig_components_v2",
        lambda _heading: (_ for _ in ()).throw(error),
    )
    with pytest.raises(type(error)):
        nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))


def test_key_is_full_binary64_payload_with_fixed_builtin_int_layout() -> None:
    state = _state(PoseStateV2(1.25, -2.5, 0.375), phase=2)
    key = legged_state_key_v2(state)
    expected = (
        1,
        2,
        _float_bits(state.body_state.x_m),
        _float_bits(state.body_state.y_m),
        _float_bits(state.body_state.heading_rad),
        *(
            word
            for contact in state.foot_contacts
            for word in (_float_bits(contact.foothold.x), _float_bits(contact.foothold.y))
        ),
    )
    assert key == expected
    assert type(key) is tuple
    assert len(key) == 13
    assert all(type(word) is int for word in key)
    assert key[4] == _float_bits(state.body_state.heading_rad)


def test_key_is_sensitive_to_every_body_contact_and_phase_coordinate() -> None:
    base = _state(PoseStateV2(1.0, 2.0, 0.25))
    base_key = legged_state_key_v2(base)
    variants = [
        _state(PoseStateV2(1.01, 2.0, 0.25)),
        _state(PoseStateV2(1.0, 2.01, 0.25)),
        _state(PoseStateV2(1.0, 2.0, 0.26)),
        _state(PoseStateV2(1.0, 2.0, 0.25), phase=1),
    ]
    for index in range(4):
        for axis in ("x", "y"):
            contacts = list(base.foot_contacts)
            point = contacts[index].foothold
            changed = WorldPoint(
                point.x + (0.01 if axis == "x" else 0.0),
                point.y + (0.01 if axis == "y" else 0.0),
            )
            contacts[index] = LeggedFootContactV2(contacts[index].leg_id, changed)
            variants.append(_state(base.body_state, tuple(contacts)))
    assert all(legged_state_key_v2(variant) != base_key for variant in variants)


def test_key_rejects_top_level_and_all_nested_postconstruction_drift() -> None:
    with pytest.raises(TypeError):
        legged_state_key_v2(SearchStateSubclass(PoseStateV2(0.0, 0.0, 0.0), _contacts(), 0))

    mutations: list[tuple[str, object]] = [
        ("body", _forge_pose("x_m", -0.0)),
        ("phase", True),
        ("phase", IntSubclass(1)),
        ("schema", StringSubclass(LEGGED_SEARCH_STATE_SCHEMA_V2)),
    ]
    for kind, value in mutations:
        state = _state()
        if kind == "body":
            object.__setattr__(state, "body_state", value)
        elif kind == "phase":
            object.__setattr__(state, "sequence_phase", value)
        else:
            object.__setattr__(state, "schema_version", value)
        with pytest.raises((TypeError, ValueError)):
            legged_state_key_v2(state)

    for index in range(4):
        for axis in ("x", "y"):
            state = _state()
            object.__setattr__(state.foot_contacts[index].foothold, axis, -0.0)
            with pytest.raises((TypeError, ValueError)):
                legged_state_key_v2(state)


def test_key_is_independent_of_pythonhashseed() -> None:
    code = (
        "from path_planner.v2.contracts import PoseStateV2;"
        "from path_planner.v2.providers.legged import "
        "nominal_legged_search_state_v2,legged_state_key_v2;"
        "print(legged_state_key_v2(nominal_legged_search_state_v2(PoseStateV2(1.0,2.0,0.25))))"
    )
    outputs = []
    for seed in ("1", "999"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (os.path.abspath("src"), env.get("PYTHONPATH", "")))
        )
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                cwd=os.getcwd(),
                env=env,
                text=True,
            )
        )
    assert outputs[0] == outputs[1]


def test_key_converts_ordinary_pack_fault_and_propagates_baseexceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    monkeypatch.setattr(
        legged_module,
        "_float_bits_v2",
        lambda _value: (_ for _ in ()).throw(RuntimeError("pack fault")),
    )
    with pytest.raises(ValueError):
        legged_state_key_v2(state)
    for error in (KeyboardInterrupt(), SystemExit(), MemoryError()):
        monkeypatch.setattr(
            legged_module,
            "_float_bits_v2",
            lambda _value, error=error: (_ for _ in ()).throw(error),
        )
        with pytest.raises(type(error)):
            legged_state_key_v2(state)


@pytest.mark.parametrize(
    "field",
    [
        "duration_s",
        "distance_m",
        "energy_cost",
        "observation_contribution",
        "foot_travel_m",
    ],
)
@pytest.mark.parametrize("bad", [1, True, np.float64(0.1), FloatSubclass(0.1)])
def test_primitive_audits_raw_numeric_types_before_base_normalization(
    field: str,
    bad: object,
) -> None:
    with pytest.raises(TypeError):
        _primitive(**{field: bad})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("kind", PrimitiveKindV2.WHEEL_MOTION),
        ("kind", "leg_step"),
        ("validation_level", ValidationLevelV2.L1),
        ("validation_level", "L2"),
        ("duration_s", 2.0),
        ("observation_contribution", 0.1),
        ("capability", "physical"),
        ("capability", StringSubclass(LEGGED_CAPABILITY_LEVEL_V2)),
        ("resource_proxy_id", "joules/v1"),
        ("primitive_schema_version", "wrong"),
    ],
)
def test_primitive_rejects_frozen_base_and_resource_contract_drift(
    field: str,
    bad: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _primitive(**{field: bad})


@pytest.mark.parametrize("pose_field", ["start_state", "end_state", "lift_body_state"])
@pytest.mark.parametrize("nested", [1, True, np.float64(0.0), FloatSubclass(0.0), float("nan")])
def test_primitive_rejects_raw_pose_nested_type_drift(
    pose_field: str,
    nested: object,
) -> None:
    values = _primitive_values()
    pose = values[pose_field]
    assert type(pose) is PoseStateV2
    object.__setattr__(pose, "x_m", nested)
    with pytest.raises((TypeError, ValueError)):
        LeggedStepPrimitiveV2(**values)


def test_primitive_rejects_pose_subclasses_and_wrap_equivalent_endpoints() -> None:
    with pytest.raises(TypeError):
        _primitive(start_state=PoseSubclass(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        _primitive(start_state=PoseStateV2(0.0, 0.0, 2.0 * pi))
    with pytest.raises(ValueError):
        _primitive(end_state=PoseStateV2(0.0, -0.10, 2.0 * pi))


def test_primitive_enforces_sequence_phase_leg_and_contact_replacement() -> None:
    with pytest.raises(ValueError):
        _primitive(moving_leg=LegIdV2.REAR_RIGHT)

    values = _primitive_values()
    end = values["end_legged_state"]
    assert type(end) is LeggedSearchStateV2
    wrong_phase = _state(end.body_state, end.foot_contacts, 2)
    with pytest.raises(ValueError):
        _primitive(end_legged_state=wrong_phase)

    changed_nonmoving = list(end.foot_contacts)
    changed_nonmoving[1] = LeggedFootContactV2(
        LegIdV2.FRONT_RIGHT,
        WorldPoint(0.40, -0.25),
    )
    bad_end = _state(end.body_state, tuple(changed_nonmoving), 1)
    with pytest.raises(ValueError):
        _primitive(end_legged_state=bad_end)

    mismatched_target = WorldPoint(0.61, 0.25)
    with pytest.raises(ValueError):
        _primitive(target_foothold=mismatched_target)


def test_primitive_canonicalizes_public_signed_zero_without_claiming_safety() -> None:
    primitive = _zero_primitive()
    for value in (
        primitive.start_state.x_m,
        primitive.start_state.y_m,
        primitive.start_state.heading_rad,
        primitive.end_state.x_m,
        primitive.end_state.y_m,
        primitive.end_state.heading_rad,
        primitive.lift_body_state.x_m,
        primitive.lift_body_state.y_m,
        primitive.lift_body_state.heading_rad,
        primitive.distance_m,
        primitive.energy_cost,
        primitive.observation_contribution,
        primitive.foot_travel_m,
    ):
        assert value == 0.0
        assert not np.signbit(value)
    assert primitive.capability == "simulation_proxy"
    assert "joule" not in primitive.resource_proxy_id.lower()


@pytest.mark.parametrize("field", ["distance_m", "energy_cost", "foot_travel_m"])
def test_primitive_rejects_negative_even_inside_tolerance(field: str) -> None:
    for negative in (-5e-324, -5e-13):
        with pytest.raises(ValueError):
            _primitive(**{field: negative})


def test_primitive_resource_values_use_recomputed_canonical_bytes() -> None:
    exact = _primitive()
    inside = _primitive(
        distance_m=0.10 + 5e-13,
        foot_travel_m=0.25 - 5e-13,
        energy_cost=0.35 + 5e-13,
    )
    assert inside.distance_m == 0.10
    assert inside.foot_travel_m == 0.25
    assert inside.energy_cost == 0.35
    assert canonical_json_bytes(inside) == canonical_json_bytes(exact)
    for field, value in (
        ("distance_m", 0.10 + 3e-12),
        ("foot_travel_m", 0.25 + 3e-12),
        ("energy_cost", 0.35 + 3e-12),
    ):
        with pytest.raises(ValueError):
            _primitive(**{field: value})


def test_lift_state_is_part_of_canonical_serialization_and_hash() -> None:
    first = _primitive()
    values = _primitive_values()
    values["lift_body_state"] = PoseStateV2(0.0, -0.05, 0.0)
    values["distance_m"] = 0.10
    second = LeggedStepPrimitiveV2(**values)
    assert canonical_json_bytes(first) != canonical_json_bytes(second)
    assert hash(first) != hash(second)


def test_as_oracle_candidate_is_exact_lossless_payload() -> None:
    primitive = _primitive()
    candidate = primitive.as_oracle_candidate()
    expected = LeggedStepCandidateV2(
        start_body_state=primitive.start_legged_state.body_state,
        lift_body_state=primitive.lift_body_state,
        end_body_state=primitive.end_legged_state.body_state,
        foot_contacts=primitive.start_legged_state.foot_contacts,
        moving_leg=primitive.moving_leg,
        sequence_phase=primitive.start_legged_state.sequence_phase,
        target_foothold=primitive.target_foothold,
    )
    assert type(candidate) is LeggedStepCandidateV2
    assert candidate == expected
    assert canonical_json_bytes(candidate) == canonical_json_bytes(expected)


@pytest.mark.parametrize(
    ("target", "field", "bad"),
    [
        ("primitive", "distance_m", -0.0),
        ("primitive", "capability", StringSubclass(LEGGED_CAPABILITY_LEVEL_V2)),
        ("base_pose", "x_m", 0),
        ("lift_pose", "y_m", -0.0),
        ("target", "x", -0.0),
        ("state", "sequence_phase", True),
        ("contact", "x", -0.0),
    ],
)
def test_as_oracle_candidate_rejects_postconstruction_drift(
    target: str,
    field: str,
    bad: object,
) -> None:
    primitive = _primitive()
    if target == "primitive":
        object.__setattr__(primitive, field, bad)
    elif target == "base_pose":
        object.__setattr__(primitive.start_state, field, bad)
    elif target == "lift_pose":
        object.__setattr__(primitive.lift_body_state, field, bad)
    elif target == "target":
        object.__setattr__(primitive.target_foothold, field, bad)
    elif target == "state":
        object.__setattr__(primitive.start_legged_state, field, bad)
    else:
        object.__setattr__(primitive.start_legged_state.foot_contacts[0].foothold, field, bad)
    with pytest.raises((TypeError, ValueError)):
        primitive.as_oracle_candidate()


def test_primitive_constructor_fault_boundary_converts_only_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        legged_module,
        "_hypot_v2",
        lambda *_values: (_ for _ in ()).throw(RuntimeError("hypot fault")),
    )
    with pytest.raises(ValueError):
        _primitive()
    for error in (KeyboardInterrupt(), SystemExit(), MemoryError()):
        monkeypatch.setattr(
            legged_module,
            "_hypot_v2",
            lambda *_values, error=error: (_ for _ in ()).throw(error),
        )
        with pytest.raises(type(error)):
            _primitive()


def test_as_oracle_candidate_fault_boundary_converts_only_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive = _primitive()
    monkeypatch.setattr(
        legged_module,
        "_candidate_from_payload_v2",
        lambda *_values: (_ for _ in ()).throw(RuntimeError("candidate fault")),
    )
    with pytest.raises(ValueError):
        primitive.as_oracle_candidate()
    for error in (KeyboardInterrupt(), SystemExit(), MemoryError()):
        monkeypatch.setattr(
            legged_module,
            "_candidate_from_payload_v2",
            lambda *_values, error=error: (_ for _ in ()).throw(error),
        )
        with pytest.raises(type(error)):
            primitive.as_oracle_candidate()


@pytest.mark.parametrize(
    ("raw_heading", "expected_heading"),
    [
        (2.0 * pi, 0.0),
        (-3.0 * pi, -pi),
        (pi, pi),
        (-pi, -pi),
        (
            sys.float_info.max,
            (sys.float_info.max + pi) % (2.0 * pi) - pi,
        ),
        (
            -sys.float_info.max,
            (-sys.float_info.max + pi) % (2.0 * pi) - pi,
        ),
    ],
)
def test_primitive_canonicalizes_lift_heading_before_candidate_conversion(
    raw_heading: float,
    expected_heading: float,
) -> None:
    primitive = _primitive(
        lift_body_state=PoseStateV2(0.0, -0.10, raw_heading),
    )
    candidate = primitive.as_oracle_candidate()
    assert primitive.lift_body_state.heading_rad == expected_heading
    assert candidate.lift_body_state == primitive.lift_body_state
    assert canonical_json_bytes(candidate.lift_body_state) == canonical_json_bytes(
        primitive.lift_body_state
    )


@pytest.mark.parametrize(
    ("helper_name", "entrypoint"),
    [
        ("_audit_search_state_canonical", lambda: legged_state_key_v2(_state())),
        ("_validate_step_relations", _primitive),
        ("_resource_matches", _primitive),
        ("_audit_primitive_canonical", lambda: _primitive().as_oracle_candidate()),
    ],
)
def test_public_boundaries_convert_internal_runtime_faults_to_value_error(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    entrypoint: object,
) -> None:
    monkeypatch.setattr(
        legged_module,
        helper_name,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError(f"{helper_name} fault")
        ),
    )
    with pytest.raises(ValueError):
        entrypoint()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("helper_name", "entrypoint"),
    [
        ("_audit_search_state_canonical", lambda: legged_state_key_v2(_state())),
        ("_validate_step_relations", _primitive),
        ("_resource_matches", _primitive),
        ("_audit_primitive_canonical", lambda: _primitive().as_oracle_candidate()),
    ],
)
@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), MemoryError()])
def test_public_boundaries_propagate_internal_critical_faults(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    entrypoint: object,
    error: BaseException,
) -> None:
    monkeypatch.setattr(
        legged_module,
        helper_name,
        lambda *_args, error=error, **_kwargs: (_ for _ in ()).throw(error),
    )
    with pytest.raises(type(error)):
        entrypoint()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("helper_name", "entrypoint"),
    [
        ("_audit_search_state_canonical", lambda: legged_state_key_v2(_state())),
        ("_validate_step_relations", _primitive),
        ("_resource_matches", _primitive),
        ("_audit_primitive_canonical", lambda: _primitive().as_oracle_candidate()),
    ],
)
@pytest.mark.parametrize("error", [TypeError("typed contract"), ValueError("value contract")])
def test_public_boundaries_preserve_contract_error_types(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    entrypoint: object,
    error: Exception,
) -> None:
    monkeypatch.setattr(
        legged_module,
        helper_name,
        lambda *_args, error=error, **_kwargs: (_ for _ in ()).throw(error),
    )
    with pytest.raises(type(error), match=str(error)):
        entrypoint()  # type: ignore[operator]


def test_key_rejects_in_place_mutation_by_audit_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(PoseStateV2(1.0, 2.0, 0.25))
    real_audit = legged_module._audit_search_state_canonical

    def audit_then_mutate(value: object, name: str) -> LeggedSearchStateV2:
        audited = real_audit(value, name)
        object.__setattr__(audited.body_state, "x_m", -0.0)
        return audited

    monkeypatch.setattr(
        legged_module,
        "_audit_search_state_canonical",
        audit_then_mutate,
    )
    with pytest.raises(ValueError):
        legged_state_key_v2(state)


@pytest.mark.parametrize(
    "bad_word",
    [True, IntSubclass(1), -1, 1 << 64, 0],
)
def test_key_rejects_noncanonical_or_mismatched_bit_words(
    monkeypatch: pytest.MonkeyPatch,
    bad_word: object,
) -> None:
    monkeypatch.setattr(
        legged_module,
        "_float_bits_v2",
        lambda _value: bad_word,
    )
    with pytest.raises(ValueError):
        legged_state_key_v2(_state(PoseStateV2(1.0, 2.0, 0.25)))


def test_constructor_rejects_raw_payload_when_initializer_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        LeggedStepPrimitiveV2,
        "_initialize_canonical_payload",
        lambda _self: None,
    )
    values = _primitive_values()
    values["duration_s"] = 1
    with pytest.raises(TypeError):
        LeggedStepPrimitiveV2(**values)


@pytest.mark.parametrize("fault", ["moving_leg", "end_phase"])
def test_constructor_independently_rejects_relation_helper_bypass(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    values = _primitive_values()
    if fault == "moving_leg":
        values["moving_leg"] = LegIdV2.REAR_RIGHT
    else:
        end = values["end_legged_state"]
        assert type(end) is LeggedSearchStateV2
        values["end_legged_state"] = _state(
            end.body_state,
            end.foot_contacts,
            2,
        )
    monkeypatch.setattr(
        legged_module,
        "_validate_step_relations",
        lambda **_kwargs: (0.25, 0.10, 0.35),
    )
    with pytest.raises(ValueError):
        LeggedStepPrimitiveV2(**values)


def test_constructor_rejects_negative_resources_returned_by_relation_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        legged_module,
        "_validate_step_relations",
        lambda **_kwargs: (-5e-324, -5e-324, -5e-324),
    )
    with pytest.raises(ValueError):
        _zero_primitive()


@pytest.mark.parametrize("fault", ["start_negative_zero", "phase_bool"])
def test_candidate_rejects_exact_type_with_forged_nested_payload(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    real_builder = legged_module._candidate_from_payload_v2

    def build_then_forge(*args: object) -> LeggedStepCandidateV2:
        candidate = real_builder(*args)
        if fault == "start_negative_zero":
            object.__setattr__(candidate.start_body_state, "x_m", -0.0)
        else:
            object.__setattr__(candidate, "sequence_phase", False)
        return candidate

    monkeypatch.setattr(
        legged_module,
        "_candidate_from_payload_v2",
        build_then_forge,
    )
    with pytest.raises((TypeError, ValueError)):
        _primitive().as_oracle_candidate()
