from __future__ import annotations

import os
import struct
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
from math import isfinite, nextafter, pi

import numpy as np
import pytest

import path_planner.v2 as v2
import path_planner.v2.providers as provider_exports
import path_planner.v2.providers.legged as legged_module
import path_planner.v2.validation as validation_module
from path_planner.core import Cell, WorldPoint
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
)
from path_planner.v2.profiles import LeggedProfileV2, PlatformProfileV2
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
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
    snapshot_hash,
)


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


def test_public_legged_route_l2_validation_surface_is_frozen() -> None:
    assert validation_module.LEGGED_ROUTE_VALIDATOR_ID_V2 == (
        "path-planner-v2-legged-route-l2/v1"
    )
    assert v2.LEGGED_ROUTE_VALIDATOR_ID_V2 == (
        "path-planner-v2-legged-route-l2/v1"
    )
    assert v2.LeggedRouteValidationResultV2 is (
        validation_module.LeggedRouteValidationResultV2
    )
    assert v2.validate_legged_route_l2 is validation_module.validate_legged_route_l2
    assert tuple(
        field.name for field in fields(validation_module.LeggedRouteValidationResultV2)
    ) == (
        "evidence",
        "reason_code",
        "timed_out",
        "failed_cell",
        "failed_leg",
        "failed_primitive_index",
        "checked_cell_count",
        "minimum_support_margin_m",
        "validated_route_hash",
    )


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"distance_m": 9.0},
        {"foot_travel_m": 8.0},
        {"energy_cost": 17.0},
        {"distance_m": 9.0, "foot_travel_m": 8.0, "energy_cost": 17.0},
    ],
)
def test_constructor_preserves_raw_resource_evidence_when_matcher_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, float],
) -> None:
    monkeypatch.setattr(
        legged_module,
        "_resource_matches",
        lambda *_args: True,
    )
    with pytest.raises(ValueError):
        _primitive(**overrides)


def _route_profile() -> LeggedProfileV2:
    return LeggedProfileV2(
        profile=PlatformProfileV2(
            profile_id="legged-static-crawl/v1",
            platform_kind=PlatformKindV2.LEGGED,
            capability_revision="simulation_proxy_static_crawl/v1",
            simulation_proxy=True,
            max_traversable_slope_deg=30.0,
        )
    )


def _route_snapshot() -> TerrainSnapshotV2:
    shape = (20, 20)
    return TerrainSnapshotV2(
        geometry=FineGridGeometryV2(
            width=20,
            height=20,
            origin=(-5.0, -5.0),
            frame_id="moon",
        ),
        elevation_m=np.zeros(shape, dtype=np.float64),
        slope_deg=np.zeros(shape, dtype=np.float64),
        traversable_mask=np.ones(shape, dtype=bool),
        hard_obstacle_mask=np.zeros(shape, dtype=bool),
        observed_mask=np.ones(shape, dtype=bool),
        confidence=np.ones(shape, dtype=np.float64),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="legged-route-fixture",
            source_hash="legged-route-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )


def _route_step(
    start: LeggedSearchStateV2,
    *,
    target: WorldPoint,
    end_body: PoseStateV2,
) -> LeggedStepPrimitiveV2:
    moving_leg = LEGGED_CRAWL_SEQUENCE_V2[start.sequence_phase]
    moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(moving_leg)
    contacts = list(start.foot_contacts)
    contacts[moving_index] = LeggedFootContactV2(moving_leg, target)
    end = LeggedSearchStateV2(
        body_state=end_body,
        foot_contacts=tuple(contacts),
        sequence_phase=(start.sequence_phase + 1) % 4,
    )
    source = start.foot_contacts[moving_index].foothold
    foot_travel = float(np.hypot(target.x - source.x, target.y - source.y))
    distance = float(
        np.hypot(
            end.body_state.x_m - start.body_state.x_m,
            end.body_state.y_m - start.body_state.y_m,
        )
    )
    return LeggedStepPrimitiveV2(
        kind=PrimitiveKindV2.LEG_STEP,
        start_state=start.body_state,
        end_state=end.body_state,
        duration_s=1.0,
        distance_m=distance,
        energy_cost=distance + foot_travel,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        start_legged_state=start,
        lift_body_state=end.body_state,
        end_legged_state=end,
        moving_leg=moving_leg,
        target_foothold=target,
        foot_travel_m=foot_travel,
    )


def _route_primitives() -> tuple[LeggedStepPrimitiveV2, LeggedStepPrimitiveV2]:
    first = _route_step(
        _state(),
        target=WorldPoint(0.60, 0.25),
        end_body=PoseStateV2(0.0, -0.10, 0.0),
    )
    second = _route_step(
        first.end_legged_state,
        target=WorldPoint(-0.35, -0.50),
        end_body=PoseStateV2(0.0, -0.20, 0.0),
    )
    return first, second


def _legged_route(
    *primitives: RoutePrimitiveV2,
    complete: bool = True,
    platform: PlatformKindV2 = PlatformKindV2.LEGGED,
) -> TypedRouteV2:
    chosen = _route_primitives() if not primitives else primitives
    return TypedRouteV2(
        platform_kind=platform,
        primitives=tuple(chosen),
        total_cost=0.0,
        is_complete=complete,
    )


def _route_request(
    snapshot: TerrainSnapshotV2,
    profile: LeggedProfileV2,
    route: TypedRouteV2,
    *,
    start: PoseStateV2 | None = None,
    goal: PoseStateV2 | None = None,
    max_route_states: int = 10_000,
    profile_id: str | None = None,
) -> PlanningRequestV2:
    first = route.primitives[0]
    last = route.primitives[-1]
    return PlanningRequestV2(
        request_id="legged-route-request",
        platform_profile_id=(
            profile.profile.profile_id if profile_id is None else profile_id
        ),
        start_state=first.start_state if start is None else start,
        goal_state=last.end_state if goal is None else goal,
        terrain_snapshot=snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(max_route_states=max_route_states),
        timeout_s=10.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=7,
    )


def _route_deadline(clock=None) -> PlanningDeadlineV2:
    return PlanningDeadlineV2(
        0.0,
        100.0,
        (lambda: 0.0) if clock is None else clock,
    )


def _a2_result(
    reason: str = "legged_step_l2_valid",
    *,
    checked: int = 3,
    cell: Cell | None = None,
    leg: LegIdV2 | None = None,
    margin: float | None = 0.08,
) -> LeggedValidationResultV2:
    return LeggedValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=reason == "legged_step_l2_valid",
            checks=(reason,),
        ),
        reason_code=reason,
        timed_out=reason == "planning_deadline_expired",
        failed_cell=cell,
        failed_leg=leg,
        checked_cell_count=checked,
        minimum_support_margin_m=margin,
    )


def _mutate_a2_failure_to_pass(result: LeggedValidationResultV2) -> None:
    object.__setattr__(result.evidence, "passed", True)
    object.__setattr__(result.evidence, "checks", ("legged_step_l2_valid",))
    object.__setattr__(result, "reason_code", "legged_step_l2_valid")
    object.__setattr__(result, "failed_cell", None)


def _route_call(
    route: TypedRouteV2 | None = None,
    *,
    request: PlanningRequestV2 | None = None,
    anchor: FineSafetyAnchorV2 | None = None,
    profile: LeggedProfileV2 | None = None,
    deadline: PlanningDeadlineV2 | None = None,
):
    chosen_route = _legged_route() if route is None else route
    chosen_profile = _route_profile() if profile is None else profile
    chosen_anchor = FineSafetyAnchorV2(_route_snapshot()) if anchor is None else anchor
    chosen_request = (
        _route_request(chosen_anchor.snapshot, chosen_profile, chosen_route)
        if request is None
        else request
    )
    return validation_module.validate_legged_route_l2(
        chosen_route,
        chosen_request,
        chosen_anchor,
        chosen_profile,
        _route_deadline() if deadline is None else deadline,
    )


def test_legged_route_result_is_frozen_slotted_and_enforces_pass_contract() -> None:
    digest = "a" * 64
    result = validation_module.LeggedRouteValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=validation_module.LEGGED_ROUTE_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=True,
            checks=("legged_route_l2_valid",),
        ),
        reason_code="legged_route_l2_valid",
        timed_out=False,
        failed_cell=None,
        failed_leg=None,
        failed_primitive_index=None,
        checked_cell_count=1,
        minimum_support_margin_m=0.05,
        validated_route_hash=digest,
    )
    assert result.validated_route_hash == digest
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.reason_code = "route_incomplete"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("reason", "index", "cell", "leg", "margin"),
    [
        ("legged_route_l2_valid", 0, None, None, 0.05),
        ("route_start_mismatch", None, None, None, None),
        ("route_start_mismatch", 1, None, None, None),
        ("route_connectivity_mismatch", None, None, None, None),
        ("planning_deadline_expired", None, Cell(0, 0), None, None),
        ("route_incomplete", None, None, LegIdV2.FRONT_LEFT, None),
        ("route_goal_tolerance_exceeded", 1, Cell(0, 0), None, None),
    ],
)
def test_legged_route_result_rejects_reason_metadata_mismatches(
    reason: str,
    index: int | None,
    cell: Cell | None,
    leg: LegIdV2 | None,
    margin: float | None,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        validation_module.LeggedRouteValidationResultV2(
            evidence=ValidationEvidenceV2(
                validator_id=validation_module.LEGGED_ROUTE_VALIDATOR_ID_V2,
                level=ValidationLevelV2.L2,
                passed=reason == "legged_route_l2_valid",
                checks=(reason,),
            ),
            reason_code=reason,
            timed_out=reason == "planning_deadline_expired",
            failed_cell=cell,
            failed_leg=leg,
            failed_primitive_index=index,
            checked_cell_count=1,
            minimum_support_margin_m=margin,
            validated_route_hash="a" * 64,
        )


def test_legged_route_result_budget_failure_requires_zero_checked_count() -> None:
    with pytest.raises(ValueError):
        validation_module.LeggedRouteValidationResultV2(
            evidence=ValidationEvidenceV2(
                validator_id=validation_module.LEGGED_ROUTE_VALIDATOR_ID_V2,
                level=ValidationLevelV2.L2,
                passed=False,
                checks=("route_state_budget_exceeded",),
            ),
            reason_code="route_state_budget_exceeded",
            timed_out=False,
            failed_cell=None,
            failed_leg=None,
            failed_primitive_index=0,
            checked_cell_count=1,
            minimum_support_margin_m=None,
            validated_route_hash="a" * 64,
        )


def test_legged_route_happy_path_replays_every_step_and_aggregates_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _legged_route()
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    calls: list[LeggedStepCandidateV2] = []

    def exact_authority(candidate, _anchor, _profile, _deadline):
        calls.append(candidate)
        return _a2_result(
            checked=5 + len(calls),
            margin=(0.08, 0.07)[len(calls) - 1],
        )

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        exact_authority,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert len(calls) == 2
    assert result.reason_code == "legged_route_l2_valid"
    assert result.checked_cell_count == 13
    assert result.minimum_support_margin_m == 0.07
    assert result.failed_primitive_index is None
    assert result.validated_route_hash == sha256(canonical_json_bytes(route)).hexdigest()


def test_legged_route_default_path_replays_the_real_pinned_a2_oracle() -> None:
    route = _single_heading_route(0.0)
    result = _route_call(route)
    assert result.reason_code == "legged_route_l2_valid"
    assert result.checked_cell_count > 0
    assert result.minimum_support_margin_m is not None
    assert result.minimum_support_margin_m >= 0.05
    assert result.validated_route_hash == sha256(canonical_json_bytes(route)).hexdigest()


@pytest.mark.parametrize(
    ("reason", "cell", "leg", "margin"),
    [
        ("legged_step_structure_mismatch", None, None, None),
        ("legged_foothold_grid_misaligned", None, LegIdV2.FRONT_LEFT, None),
        ("legged_foothold_unknown", Cell(1, 2), LegIdV2.FRONT_LEFT, None),
        ("legged_foothold_hard_obstacle", Cell(1, 2), LegIdV2.FRONT_LEFT, None),
        ("legged_foothold_not_traversable", Cell(1, 2), LegIdV2.FRONT_LEFT, None),
        ("legged_foothold_slope_exceeded", Cell(1, 2), LegIdV2.FRONT_LEFT, None),
        ("legged_step_length_exceeded", None, LegIdV2.FRONT_LEFT, None),
        ("legged_step_height_exceeded", None, LegIdV2.FRONT_LEFT, None),
        ("legged_support_margin_insufficient", None, None, None),
        ("legged_body_sweep_unknown", Cell(1, 2), None, 0.05),
        ("legged_body_sweep_collision", Cell(1, 2), None, 0.05),
        ("legged_foot_sequence_invalid", None, LegIdV2.FRONT_LEFT, 0.05),
    ],
)
def test_legged_route_preserves_every_a2_step_failure(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    cell: Cell | None,
    leg: LegIdV2 | None,
    margin: float | None,
) -> None:
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(
            reason,
            checked=2,
            cell=cell,
            leg=leg,
            margin=margin,
        ),
    )
    result = _route_call()
    assert result.reason_code == reason
    assert result.failed_primitive_index == 0
    assert result.failed_cell == cell
    assert result.failed_leg is leg
    assert result.minimum_support_margin_m == margin
    assert result.checked_cell_count == 4


def test_legged_route_a2_reason_priority_precedes_index_and_structural_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        (
            _a2_result(
                "legged_body_sweep_collision",
                checked=3,
                cell=Cell(2, 2),
                margin=0.05,
            ),
            _a2_result(
                "legged_foothold_unknown",
                checked=4,
                cell=Cell(1, 1),
                leg=LegIdV2.REAR_RIGHT,
                margin=None,
            ),
        )
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: next(results),
    )
    result = _route_call(_legged_route(complete=False))
    assert result.reason_code == "legged_foothold_unknown"
    assert result.failed_primitive_index == 1
    assert result.failed_cell == Cell(1, 1)
    assert result.failed_leg is LegIdV2.REAR_RIGHT
    assert result.checked_cell_count == 7


def _single_heading_route(heading: float) -> TypedRouteV2:
    start = _state(PoseStateV2(0.0, 0.0, heading))
    primitive = _route_step(
        start,
        target=WorldPoint(0.60, 0.25),
        end_body=PoseStateV2(0.0, -0.10, heading),
    )
    return _legged_route(primitive)


@pytest.mark.parametrize(
    ("stored", "requested", "expected"),
    [
        (0.0, 2.0 * pi, "legged_route_l2_valid"),
        (-pi, -3.0 * pi, "legged_route_l2_valid"),
        (-pi, pi, "route_start_mismatch"),
        (pi, -pi, "route_start_mismatch"),
    ],
)
def test_legged_route_start_uses_a2_tie_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
    stored: float,
    requested: float,
    expected: str,
) -> None:
    route = _single_heading_route(stored)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(
        snapshot,
        profile,
        route,
        start=PoseStateV2(0.0, 0.0, requested),
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert result.reason_code == expected
    assert result.failed_primitive_index == (0 if expected != "legged_route_l2_valid" else None)


def test_legged_route_start_position_is_exact_without_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _legged_route()
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(
        snapshot,
        profile,
        route,
        start=PoseStateV2(nextafter(0.0, 1.0), 0.0, 0.0),
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert result.reason_code == "route_start_mismatch"
    assert result.failed_primitive_index == 0


@pytest.mark.parametrize(
    ("stored", "requested", "expected"),
    [
        (0.0, 2.0 * pi, "legged_route_l2_valid"),
        (-pi, pi, "legged_route_l2_valid"),
        (0.0, nextafter(0.0, 1.0), "route_goal_tolerance_exceeded"),
        (0.0, 1.7976931348623157e308, "route_goal_tolerance_exceeded"),
    ],
)
def test_legged_route_goal_uses_physical_heading_equivalence(
    monkeypatch: pytest.MonkeyPatch,
    stored: float,
    requested: float,
    expected: str,
) -> None:
    route = _single_heading_route(stored)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(
        snapshot,
        profile,
        route,
        goal=PoseStateV2(0.0, -0.10, requested),
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert result.reason_code == expected
    assert result.failed_primitive_index == (
        0 if expected == "route_goal_tolerance_exceeded" else None
    )


@pytest.mark.parametrize("drift", ["contact", "phase"])
def test_legged_route_connectivity_uses_full_legged_state(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    first, second = _route_primitives()
    if drift == "contact":
        changed = list(first.end_legged_state.foot_contacts)
        changed[1] = LeggedFootContactV2(
            LegIdV2.FRONT_RIGHT,
            WorldPoint(0.60, -0.25),
        )
        second_start = LeggedSearchStateV2(
            first.end_state,
            tuple(changed),
            first.end_legged_state.sequence_phase,
        )
    else:
        second_start = LeggedSearchStateV2(
            first.end_state,
            first.end_legged_state.foot_contacts,
            2,
        )
    moving_leg = LEGGED_CRAWL_SEQUENCE_V2[second_start.sequence_phase]
    moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(moving_leg)
    source = second_start.foot_contacts[moving_index].foothold
    replacement = WorldPoint(source.x + 0.25, source.y)
    disconnected = _route_step(
        second_start,
        target=replacement,
        end_body=second.end_state,
    )
    route = _legged_route(first, disconnected)
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = _route_call(route)
    assert result.reason_code == "route_connectivity_mismatch"
    assert result.failed_primitive_index == 1


@pytest.mark.parametrize(
    ("limit", "expected_index"),
    [(0, 0), (1, 0), (2, 1)],
)
def test_legged_route_budget_is_prechecked_before_any_a2(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
    expected_index: int,
) -> None:
    route = _legged_route()
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(
        snapshot,
        profile,
        route,
        max_route_states=limit,
    )

    def forbidden(*_args):
        raise AssertionError("A2 must not run after route budget precheck")

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        forbidden,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert result.reason_code == "route_state_budget_exceeded"
    assert result.failed_primitive_index == expected_index
    assert result.checked_cell_count == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_cost", "bad"),
        ("is_complete", "bad"),
        ("platform_kind", "bad"),
    ],
)
def test_legged_route_budget_precedes_nonempty_top_level_structure_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(
        snapshot,
        profile,
        route,
        max_route_states=0,
    )
    object.__setattr__(route, field, value)
    calls = 0

    def a2_double(*_args):
        nonlocal calls
        calls += 1
        return _a2_result()

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        a2_double,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert calls == 0
    assert result.reason_code == "route_state_budget_exceeded"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0


def test_legged_route_budget_hard_cap_equal_and_first_overflow_are_exact() -> None:
    assert validation_module._legged_route_budget_overflow_index_v2(
        MAX_REPLAY_STEPS,
        MAX_REPLAY_STEPS + 1,
    ) is None
    assert validation_module._legged_route_budget_overflow_index_v2(
        MAX_REPLAY_STEPS + 1,
        MAX_REPLAY_STEPS + 2,
    ) == MAX_REPLAY_STEPS


def test_legged_route_skips_unconvertible_primitive_but_later_safety_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _route_primitives()
    base_only = RoutePrimitiveV2(
        kind=PrimitiveKindV2.LEG_STEP,
        start_state=first.start_state,
        end_state=first.end_state,
        duration_s=first.duration_s,
        distance_m=first.distance_m,
        energy_cost=first.energy_cost,
        observation_contribution=first.observation_contribution,
        validation_level=ValidationLevelV2.L2,
    )
    route = _legged_route(base_only, second)
    calls = 0

    def safety(*_args):
        nonlocal calls
        calls += 1
        return _a2_result(
            "legged_body_sweep_collision",
            checked=4,
            cell=Cell(3, 3),
            margin=0.05,
        )

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        safety,
    )
    result = _route_call(route)
    assert calls == 1
    assert result.reason_code == "legged_body_sweep_collision"
    assert result.failed_primitive_index == 1
    assert result.checked_cell_count == 4
    assert result.validated_route_hash is None


@pytest.mark.parametrize("fault", ["wrong_type", "forged_result", "exception"])
def test_legged_route_rejects_a2_authority_contract_faults(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    if fault == "wrong_type":
        authority = lambda *_args: object()
    elif fault == "forged_result":
        forged = _a2_result()
        object.__setattr__(forged.evidence, "checks", ("legged_body_sweep_collision",))
        authority = lambda *_args: forged
    else:
        def authority(*_args):
            raise RuntimeError("ordinary A2 fault")

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        authority,
    )
    result = _route_call()
    assert result.reason_code == "legged_step_oracle_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0
    assert result.failed_cell is None
    assert result.failed_leg is None
    assert result.minimum_support_margin_m is None


def test_legged_route_reseals_authority_after_an_ordinary_a2_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)

    def mutate_then_fail(*_args):
        object.__setattr__(request, "request_id", "drifted-by-a2")
        raise RuntimeError("ordinary A2 fault after persistent drift")

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        mutate_then_fail,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert result.reason_code == "planning_request_contract_mismatch"
    assert result.failed_primitive_index is None


@pytest.mark.parametrize("fault_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_legged_route_propagates_critical_a2_faults(
    monkeypatch: pytest.MonkeyPatch,
    fault_type: type[BaseException],
) -> None:
    def authority(*_args):
        raise fault_type()

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        authority,
    )
    with pytest.raises(fault_type):
        _route_call()


def test_legged_route_public_authority_rebindings_do_not_change_pinned_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    monkeypatch.setattr(
        validation_module,
        "validate_legged_step_l2",
        lambda *_args: (_ for _ in ()).throw(AssertionError("public A2 rebound")),
    )
    monkeypatch.setattr(
        validation_module,
        "canonical_json_bytes",
        lambda *_args: (_ for _ in ()).throw(AssertionError("public serializer rebound")),
    )
    monkeypatch.setattr(
        validation_module,
        "snapshot_hash",
        lambda *_args: (_ for _ in ()).throw(AssertionError("public hasher rebound")),
    )
    monkeypatch.setattr(
        validation_module,
        "sha256",
        lambda *_args: (_ for _ in ()).throw(AssertionError("public SHA rebound")),
    )
    result = _route_call(_single_heading_route(0.0))
    assert result.reason_code == "legged_route_l2_valid"


def test_legged_route_candidate_must_bind_to_the_same_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _route_primitives()
    wrong = second.as_oracle_candidate()
    monkeypatch.setattr(
        LeggedStepPrimitiveV2,
        "as_oracle_candidate",
        lambda _self: wrong,
    )

    def forbidden(*_args):
        raise AssertionError("mismatched candidate must not reach A2")

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        forbidden,
    )
    result = _route_call(_legged_route(first))
    assert result.reason_code == "legged_primitive_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0
    assert result.validated_route_hash is None


@pytest.mark.parametrize(
    "fault",
    ["moving_end_contact", "end_phase", "distance", "foot_travel", "energy"],
)
def test_legged_route_independently_reaudits_primitive_relations_and_resources(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    primitive = _single_heading_route(0.0).primitives[0]
    candidate = primitive.as_oracle_candidate()
    route = _legged_route(primitive)
    if fault == "moving_end_contact":
        moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(primitive.moving_leg)
        object.__setattr__(
            primitive.end_legged_state.foot_contacts[moving_index],
            "foothold",
            WorldPoint(4.0, 4.0),
        )
    elif fault == "end_phase":
        object.__setattr__(primitive.end_legged_state, "sequence_phase", 3)
    elif fault == "distance":
        object.__setattr__(primitive, "distance_m", 9.0)
    elif fault == "foot_travel":
        object.__setattr__(primitive, "foot_travel_m", 9.0)
    else:
        object.__setattr__(primitive, "energy_cost", 9.0)
    monkeypatch.setattr(
        LeggedStepPrimitiveV2,
        "as_oracle_candidate",
        lambda _self: candidate,
    )

    def forbidden(*_args):
        raise AssertionError("forged primitive must not reach A2")

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        forbidden,
    )
    result = _route_call(route)
    assert result.reason_code == "legged_primitive_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0
    assert result.validated_route_hash is None


def test_legged_route_global_a2_failure_is_immediate_and_unindexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def authority(*_args):
        nonlocal calls
        calls += 1
        return _a2_result(
            "terrain_query_contract_mismatch",
            checked=2,
            margin=None,
        )

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        authority,
    )
    result = _route_call()
    assert calls == 1
    assert result.reason_code == "terrain_query_contract_mismatch"
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == 2


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("started_monotonic_s", 1.0, "planning_deadline_contract_mismatch"),
        ("deadline_monotonic_s", 0.0, "planning_deadline_contract_mismatch"),
        ("_monotonic_clock", lambda: 0.0, "planning_deadline_contract_mismatch"),
    ],
)
def test_legged_route_deadline_authority_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    expected: str,
) -> None:
    holder: dict[str, object] = {}
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            object.__setattr__(holder["deadline"], field, value)
        return 0.0

    deadline = _route_deadline(clock)
    holder["deadline"] = deadline
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = _route_call(_single_heading_route(0.0), deadline=deadline)
    assert result.reason_code == expected
    assert result.failed_primitive_index is None


@pytest.mark.parametrize("clock_value", [float("nan"), float("inf"), 1])
def test_legged_route_clock_requires_exact_finite_float(
    clock_value: object,
) -> None:
    result = _route_call(
        _single_heading_route(0.0),
        deadline=_route_deadline(lambda: clock_value),
    )
    assert result.reason_code == "planning_deadline_contract_mismatch"


@pytest.mark.parametrize("drift", ["cutoff", "callback"])
def test_legged_route_clock_result_helper_cannot_hide_deadline_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        if drift == "cutoff" and clock_calls == 4:
            return 150.0
        return 0.0

    deadline = _route_deadline(clock)
    real_word = validation_module._legged_float_word_v2
    result_audits = 0

    def audit_then_drift(value, name, **kwargs):
        nonlocal result_audits
        word = real_word(value, name, **kwargs)
        if name == "deadline clock result":
            result_audits += 1
            if result_audits == 4:
                if drift == "cutoff":
                    object.__setattr__(deadline, "deadline_monotonic_s", 200.0)
                else:
                    object.__setattr__(deadline, "_monotonic_clock", lambda: 0.0)
        return word

    monkeypatch.setattr(validation_module, "_legged_float_word_v2", audit_then_drift)
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = _route_call(_single_heading_route(0.0), deadline=deadline)
    assert clock_calls == 4
    assert result_audits == 4
    assert result.reason_code == "planning_deadline_contract_mismatch"


@pytest.mark.parametrize("fault_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_legged_route_clock_result_audit_propagates_critical_faults(
    monkeypatch: pytest.MonkeyPatch,
    fault_type: type[BaseException],
) -> None:
    real_word = validation_module._legged_float_word_v2

    def fault(value, name, **kwargs):
        if name == "deadline clock result":
            raise fault_type()
        return real_word(value, name, **kwargs)

    monkeypatch.setattr(validation_module, "_legged_float_word_v2", fault)
    with pytest.raises(fault_type):
        _route_call(_single_heading_route(0.0))


@pytest.mark.parametrize("drift", ["route", "request", "profile"])
def test_legged_route_deadline_reason_reseals_only_route_hash_without_callback(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()

    def expire_with_drift() -> float:
        if drift == "route":
            object.__setattr__(route, "total_cost", 1.0)
        elif drift == "request":
            object.__setattr__(request, "request_id", "expired-request-drift")
        else:
            object.__setattr__(profile, "max_step_length_m", 0.25)
        return 100.0

    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(expire_with_drift),
    )
    assert result.reason_code == "planning_deadline_expired"
    assert result.validated_route_hash == (None if drift == "route" else expected_hash)


def test_legged_route_final_deadline_callback_is_last_external_clock_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = _route_call(
        _single_heading_route(0.0),
        deadline=_route_deadline(clock),
    )
    assert result.reason_code == "legged_route_l2_valid"
    assert calls == 4


@pytest.mark.parametrize(
    ("target", "field", "value", "expected"),
    [
        ("request", "request_id", "drifted", "planning_request_contract_mismatch"),
        ("request", "start_state", PoseStateV2(1.0, 0.0, 0.0), "route_start_contract_mismatch"),
        ("request", "goal_state", PoseStateV2(1.0, 0.0, 0.0), "route_goal_contract_mismatch"),
        ("profile", "max_step_length_m", 0.25, "legged_profile_contract_mismatch"),
        ("route", "total_cost", 1.0, "route_structure_mismatch"),
    ],
)
def test_legged_route_callback_mutation_is_caught_by_detached_tokens(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    objects = {"request": request, "profile": profile, "route": route}
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        if calls == 4:
            object.__setattr__(objects[target], field, value)
        return 0.0

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(clock),
    )
    assert result.reason_code == expected


@pytest.mark.parametrize(
    ("helper_name", "target", "field", "value", "expected"),
    [
        (
            "_legged_rebuild_route_token_v2",
            "route",
            "total_cost",
            1.0,
            "route_structure_mismatch",
        ),
        (
            "_legged_request_other_token_v2",
            "request",
            "request_id",
            "last-helper-drift",
            "planning_request_contract_mismatch",
        ),
        (
            "_legged_profile_token_v2",
            "profile",
            "max_step_length_m",
            0.25,
            "legged_profile_contract_mismatch",
        ),
    ],
)
def test_legged_route_final_helper_call_cannot_leave_persistent_drift(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    target: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    objects = {"route": route, "request": request, "profile": profile}
    real_helper = getattr(validation_module, helper_name)
    calls = 0

    def drift_after_last_audit(*args, **kwargs):
        nonlocal calls
        calls += 1
        token = real_helper(*args, **kwargs)
        if calls == 6:
            object.__setattr__(objects[target], field, value)
        return token

    monkeypatch.setattr(validation_module, helper_name, drift_after_last_audit)
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert calls == 6
    assert result.reason_code == expected


def test_legged_route_direct_postcondition_catches_final_low_level_float_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    real_word = validation_module._legged_float_word_v2
    profile_audits = 0

    def drift_after_final_profile_float(value, name, **kwargs):
        nonlocal profile_audits
        word = real_word(value, name, **kwargs)
        if name == "legged_profile.local_foothold_grid_spacing_m":
            profile_audits += 1
            if profile_audits == 5:
                object.__setattr__(request, "request_id", "low-level-final-drift")
        return word

    monkeypatch.setattr(
        validation_module,
        "_legged_float_word_v2",
        drift_after_final_profile_float,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert profile_audits == 5
    assert result.reason_code == "planning_request_contract_mismatch"


def test_legged_route_allows_equal_value_immutable_nested_request_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            object.__setattr__(
                request,
                "objective_profile",
                replace(request.objective_profile),
            )
        return 0.0

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(clock),
    )
    assert result.reason_code == "legged_route_l2_valid"


@pytest.mark.parametrize("field", ["geometry", "provenance"])
def test_legged_route_allows_equal_value_nested_snapshot_metadata_replacement(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    anchor = FineSafetyAnchorV2(snapshot)
    expected_snapshot_hash = snapshot_hash(snapshot)
    replacement = replace(getattr(snapshot, field))
    assert type(replacement) is type(getattr(snapshot, field))
    assert replacement == getattr(snapshot, field)
    assert replacement is not getattr(snapshot, field)
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            object.__setattr__(snapshot, field, replacement)
        return 0.0

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        anchor,
        profile,
        _route_deadline(clock),
    )
    assert snapshot_hash(snapshot) == expected_snapshot_hash
    assert result.reason_code == "legged_route_l2_valid"


@pytest.mark.parametrize(
    "field",
    [
        "elevation_m",
        "slope_deg",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence",
    ],
)
def test_legged_route_allows_bit_equal_canonical_snapshot_layer_replacement(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    anchor = FineSafetyAnchorV2(snapshot)
    expected_snapshot_hash = snapshot_hash(snapshot)
    original = getattr(snapshot, field)
    replacement = np.frombuffer(
        original.tobytes(order="C"),
        dtype=original.dtype,
    ).reshape(original.shape)
    assert type(replacement) is np.ndarray
    assert replacement is not original
    assert replacement.dtype == original.dtype
    assert replacement.shape == original.shape
    assert replacement.flags.c_contiguous
    assert not replacement.flags.writeable
    assert replacement.tobytes(order="C") == original.tobytes(order="C")
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            object.__setattr__(snapshot, field, replacement)
        return 0.0

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        anchor,
        profile,
        _route_deadline(clock),
    )
    assert snapshot_hash(snapshot) == expected_snapshot_hash
    assert result.reason_code == "legged_route_l2_valid"


def test_legged_route_serializer_failure_and_byte_drift_have_stable_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_CANONICAL_JSON_BYTES_V2",
        lambda _route: "not-bytes",
    )
    invalid = _route_call(_single_heading_route(0.0))
    assert invalid.reason_code == "route_hash_contract_mismatch"
    assert invalid.validated_route_hash is None

    real = canonical_json_bytes
    calls = 0

    def drifting(route):
        nonlocal calls
        calls += 1
        payload = real(route)
        return payload if calls == 1 else payload + b" "

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_CANONICAL_JSON_BYTES_V2",
        drifting,
    )
    drift = _route_call(_single_heading_route(0.0))
    assert drift.reason_code == "route_structure_mismatch"
    assert drift.validated_route_hash is None


def test_legged_route_final_route_bytes_helper_cannot_hide_route_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    real_route_bytes = validation_module._legged_trusted_route_bytes_v2
    calls = 0

    def drift_after_final_route_bytes(value):
        nonlocal calls
        payload = real_route_bytes(value)
        calls += 1
        if calls == 16:
            object.__setattr__(route, "total_cost", 1.0)
        return payload

    monkeypatch.setattr(
        validation_module,
        "_legged_trusted_route_bytes_v2",
        drift_after_final_route_bytes,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = _route_call(route)
    assert calls == 16
    assert result.reason_code == "route_structure_mismatch"
    assert result.validated_route_hash is None


def test_legged_route_final_snapshot_digest_helper_cannot_hide_provenance_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    real_digest = validation_module._legged_trusted_snapshot_digest_v2
    calls = 0

    def drift_after_final_snapshot_digest(value):
        nonlocal calls
        digest = real_digest(value)
        calls += 1
        if calls == 11:
            object.__setattr__(
                snapshot,
                "provenance",
                replace(snapshot.provenance, source_id="final-digest-drift"),
            )
        return digest

    monkeypatch.setattr(
        validation_module,
        "_legged_trusted_snapshot_digest_v2",
        drift_after_final_snapshot_digest,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert calls == 11
    assert result.reason_code == "terrain_snapshot_hash_mismatch"


def test_legged_route_direct_seal_pins_original_route_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    real_serializer = validation_module._TRUSTED_CANONICAL_JSON_BYTES_V2
    calls = 0

    def mutate_after_old_final_call(value):
        nonlocal calls
        payload = real_serializer(value)
        calls += 1
        if calls == 26:
            object.__setattr__(request, "request_id", "bottom-serializer-drift")
        return payload

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_CANONICAL_JSON_BYTES_V2",
        mutate_after_old_final_call,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert calls == 16
    assert request.request_id == "legged-route-request"
    assert result.reason_code == "legged_route_l2_valid"


def test_legged_route_direct_seal_pins_original_route_hasher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    real_sha256 = validation_module._TRUSTED_ROUTE_SHA256_V2
    calls = 0

    class HashProxy:
        def __init__(self, hasher, mutate: bool) -> None:
            self._hasher = hasher
            self._mutate = mutate

        def hexdigest(self) -> str:
            digest = self._hasher.hexdigest()
            if self._mutate:
                object.__setattr__(request, "request_id", "bottom-route-sha-drift")
            return digest

    def mutate_after_old_final_call(payload):
        nonlocal calls
        calls += 1
        return HashProxy(real_sha256(payload), calls == 11)

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_ROUTE_SHA256_V2",
        mutate_after_old_final_call,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert calls == 6
    assert request.request_id == "legged-route-request"
    assert result.reason_code == "legged_route_l2_valid"


def test_legged_route_direct_seal_pins_original_snapshot_hasher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    real_snapshot_hash = validation_module._TRUSTED_SNAPSHOT_HASH_V2
    calls = 0

    def mutate_after_old_final_call(value):
        nonlocal calls
        digest = real_snapshot_hash(value)
        calls += 1
        if calls == 16:
            object.__setattr__(request, "request_id", "bottom-snapshot-hash-drift")
        return digest

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_SNAPSHOT_HASH_V2",
        mutate_after_old_final_call,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert calls == 11
    assert request.request_id == "legged-route-request"
    assert result.reason_code == "legged_route_l2_valid"


def test_legged_route_freezes_audited_a2_failure_before_final_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_cell = Cell(2, 3)
    a2_result = _a2_result(
        "legged_body_sweep_collision",
        cell=failed_cell,
        margin=0.08,
    )
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 4:
            object.__setattr__(
                a2_result,
                "reason_code",
                "legged_body_sweep_unknown",
            )
            object.__setattr__(
                a2_result.evidence,
                "checks",
                ("legged_body_sweep_unknown",),
            )
            object.__setattr__(failed_cell, "x", 9)
        return 0.0

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: a2_result,
    )
    result = _route_call(
        _single_heading_route(0.0),
        deadline=_route_deadline(clock),
    )
    assert clock_calls == 4
    assert result.reason_code == "legged_body_sweep_collision"
    assert result.failed_cell == Cell(2, 3)


def test_legged_route_detects_a2_failure_drift_in_post_a2_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a2_result = _a2_result(
        "legged_body_sweep_collision",
        cell=Cell(2, 3),
        margin=0.08,
    )
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 3:
            _mutate_a2_failure_to_pass(a2_result)
        return 0.0

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: a2_result,
    )
    result = _route_call(
        _single_heading_route(0.0),
        deadline=_route_deadline(clock),
    )
    assert clock_calls == 3
    assert result.reason_code == "legged_step_oracle_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0


def test_legged_route_detects_a2_drift_on_public_result_token_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a2_result = _a2_result(
        "legged_body_sweep_collision",
        cell=Cell(2, 3),
        margin=0.08,
    )
    real_token = validation_module._legged_a2_result_token_v2
    calls = 0

    def mutate_on_entry(result):
        nonlocal calls
        calls += 1
        if calls == 1:
            _mutate_a2_failure_to_pass(result)
        return real_token(result)

    monkeypatch.setattr(
        validation_module,
        "_legged_a2_result_token_v2",
        mutate_on_entry,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: a2_result,
    )
    result = _route_call(_single_heading_route(0.0))
    assert calls == 1
    assert result.reason_code == "legged_step_oracle_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0


def test_legged_route_rejects_invalid_private_route_hasher_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_ROUTE_SHA256_V2",
        lambda _payload: object(),
    )
    result = _route_call(_single_heading_route(0.0))
    assert result.reason_code == "route_hash_contract_mismatch"
    assert result.validated_route_hash is None


def test_legged_route_rejects_wrong_top_level_types() -> None:
    route = _legged_route()
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    anchor = FineSafetyAnchorV2(snapshot)
    deadline = _route_deadline()
    values = [route, request, anchor, profile, deadline]
    for index in range(5):
        wrong = list(values)
        wrong[index] = object()
        with pytest.raises(TypeError):
            validation_module.validate_legged_route_l2(*wrong)
