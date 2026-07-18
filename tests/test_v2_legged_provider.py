from __future__ import annotations

import os
import json
import struct
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
from math import copysign, fsum, hypot, isfinite, nextafter, pi

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
    FailureCategoryV2,
    ObjectiveProfileV2,
    PlanningFailureV2,
    PlanningRequestV2,
    PlanningSuccessV2,
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
    LeggedPrimitiveProviderV2,
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


@pytest.mark.parametrize(
    ("helper_kind", "target", "expected_reason", "keeps_hash"),
    [
        ("primitive_token", "route", "route_structure_mismatch", False),
        ("conversion", "route", "route_structure_mismatch", False),
        ("candidate_match", "route", "route_structure_mismatch", False),
        (
            "primitive_token",
            "request",
            "planning_request_contract_mismatch",
            True,
        ),
        ("conversion", "request", "planning_request_contract_mismatch", True),
        (
            "candidate_match",
            "request",
            "planning_request_contract_mismatch",
            True,
        ),
        (
            "primitive_token",
            "snapshot_anchor",
            "terrain_snapshot_hash_mismatch",
            True,
        ),
        (
            "conversion",
            "snapshot_anchor",
            "terrain_snapshot_hash_mismatch",
            True,
        ),
        (
            "candidate_match",
            "snapshot_anchor",
            "terrain_snapshot_hash_mismatch",
            True,
        ),
        (
            "primitive_token",
            "profile",
            "legged_profile_contract_mismatch",
            True,
        ),
        (
            "primitive_token",
            "deadline",
            "planning_deadline_contract_mismatch",
            True,
        ),
    ],
)
def test_legged_route_pins_entry_authority_before_dynamic_primitive_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    helper_kind: str,
    target: str,
    expected_reason: str,
    keeps_hash: bool,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    anchor = FineSafetyAnchorV2(snapshot)
    deadline = _route_deadline()
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    mutated = False

    def mutate_authority() -> None:
        nonlocal mutated
        if mutated:
            return
        mutated = True
        if target == "route":
            object.__setattr__(route, "total_cost", 1.0)
        elif target == "request":
            object.__setattr__(request, "request_id", "pre-pin-request-drift")
        elif target == "snapshot_anchor":
            object.__setattr__(snapshot.provenance, "source_id", "pre-pin-drift")
            object.__setattr__(anchor, "_snapshot_hash", snapshot_hash(snapshot))
        elif target == "profile":
            object.__setattr__(profile, "max_step_length_m", 0.25)
        else:
            object.__setattr__(deadline, "deadline_monotonic_s", 99.0)

    if helper_kind == "primitive_token":
        real_helper = validation_module._legged_primitive_token_v2

        def primitive_token(*args, **kwargs):
            token = real_helper(*args, **kwargs)
            mutate_authority()
            return token

        monkeypatch.setattr(
            validation_module,
            "_legged_primitive_token_v2",
            primitive_token,
        )
    elif helper_kind == "conversion":
        real_conversion = LeggedStepPrimitiveV2.as_oracle_candidate

        def conversion(primitive):
            candidate = real_conversion(primitive)
            mutate_authority()
            return candidate

        monkeypatch.setattr(
            LeggedStepPrimitiveV2,
            "as_oracle_candidate",
            conversion,
        )
    else:
        real_match = validation_module._legged_candidate_matches_primitive_v2

        def candidate_match(*args, **kwargs):
            matches = real_match(*args, **kwargs)
            mutate_authority()
            return matches

        monkeypatch.setattr(
            validation_module,
            "_legged_candidate_matches_primitive_v2",
            candidate_match,
        )

    a2_calls = 0

    def a2_double(*_args):
        nonlocal a2_calls
        a2_calls += 1
        return _a2_result()

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        a2_double,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        anchor,
        profile,
        deadline,
    )
    assert mutated is True
    assert a2_calls == 0
    assert result.reason_code == expected_reason
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == 0
    assert result.validated_route_hash == (expected_hash if keeps_hash else None)


@pytest.mark.parametrize("helper_name", ["_legged_float_word_v2", "_legged_profile_token_v2"])
def test_legged_route_pins_entry_before_early_dynamic_audits(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    route = _single_heading_route(0.0)
    real_helper = getattr(validation_module, helper_name)
    mutated = False

    def mutate_after_return(*args, **kwargs):
        nonlocal mutated
        result = real_helper(*args, **kwargs)
        if not mutated:
            mutated = True
            object.__setattr__(route, "total_cost", 1.0)
        return result

    monkeypatch.setattr(validation_module, helper_name, mutate_after_return)
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = _route_call(route)
    assert mutated is True
    assert result.reason_code == "route_structure_mismatch"
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == 0
    assert result.validated_route_hash is None


def test_legged_route_entry_pin_covers_a_later_unreviewed_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _route_primitives()
    route = _legged_route(first, second)
    replacement = _route_step(
        second.start_legged_state,
        target=WorldPoint(-0.10, -0.50),
        end_body=second.end_state,
    )
    real_conversion = LeggedStepPrimitiveV2.as_oracle_candidate
    mutated = False

    def mutate_second_after_first(primitive):
        nonlocal mutated
        candidate = real_conversion(primitive)
        if not mutated:
            mutated = True
            for field_name in (
                "end_legged_state",
                "target_foothold",
                "foot_travel_m",
                "energy_cost",
            ):
                object.__setattr__(second, field_name, getattr(replacement, field_name))
        return candidate

    monkeypatch.setattr(
        LeggedStepPrimitiveV2,
        "as_oracle_candidate",
        mutate_second_after_first,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = _route_call(route)
    assert mutated is True
    assert result.reason_code == "route_structure_mismatch"
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == 0
    assert result.validated_route_hash is None


def test_legged_route_candidate_must_bind_to_the_same_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _route_primitives()
    route = _legged_route(first)
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
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
    result = _route_call(route)
    assert result.reason_code == "legged_primitive_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0
    assert result.validated_route_hash == expected_hash


def test_legged_route_direct_candidate_binding_rejects_a_lying_public_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _route_primitives()
    route = _legged_route(first)
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    wrong = second.as_oracle_candidate()
    monkeypatch.setattr(
        LeggedStepPrimitiveV2,
        "as_oracle_candidate",
        lambda _self: wrong,
    )
    monkeypatch.setattr(
        validation_module,
        "_legged_candidate_matches_primitive_v2",
        lambda *_args: True,
    )
    a2_calls = 0

    def forbidden(*_args):
        nonlocal a2_calls
        a2_calls += 1
        return _a2_result()

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        forbidden,
    )
    result = _route_call(route)
    assert a2_calls == 0
    assert result.reason_code == "legged_primitive_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0
    assert result.validated_route_hash == expected_hash


def test_legged_route_rechecks_bound_candidate_after_pre_a2_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    real_conversion = LeggedStepPrimitiveV2.as_oracle_candidate
    candidate_box: dict[str, LeggedStepCandidateV2] = {}
    clock_calls = 0
    a2_calls = 0

    def capture_candidate(primitive):
        candidate = real_conversion(primitive)
        candidate_box["value"] = candidate
        return candidate

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 2:
            candidate = candidate_box["value"]
            object.__setattr__(
                candidate,
                "sequence_phase",
                (candidate.sequence_phase + 1) % 4,
            )
        return 0.0

    def forbidden(*_args):
        nonlocal a2_calls
        a2_calls += 1
        return _a2_result()

    monkeypatch.setattr(
        LeggedStepPrimitiveV2,
        "as_oracle_candidate",
        capture_candidate,
    )
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
        _route_deadline(clock),
    )
    assert clock_calls == 3
    assert a2_calls == 0
    assert result.reason_code == "legged_primitive_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0
    assert result.validated_route_hash == expected_hash


@pytest.mark.parametrize(
    (
        "first_reason",
        "expected_reason",
        "expected_index",
        "expected_cell",
        "expected_margin",
    ),
    [
        (
            "legged_body_sweep_collision",
            "legged_body_sweep_collision",
            0,
            Cell(2, 3),
            0.08,
        ),
        (
            "legged_step_l2_valid",
            "legged_primitive_contract_mismatch",
            1,
            None,
            None,
        ),
    ],
)
def test_legged_route_aggregates_later_pre_a2_candidate_drift(
    monkeypatch: pytest.MonkeyPatch,
    first_reason: str,
    expected_reason: str,
    expected_index: int,
    expected_cell: Cell | None,
    expected_margin: float | None,
) -> None:
    first, second = _route_primitives()
    route = _legged_route(first, second)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    real_conversion = LeggedStepPrimitiveV2.as_oracle_candidate
    candidates: list[LeggedStepCandidateV2] = []
    clock_calls = 0
    a2_calls = 0

    def capture_candidate(primitive):
        candidate = real_conversion(primitive)
        candidates.append(candidate)
        return candidate

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 4:
            candidate = candidates[1]
            object.__setattr__(
                candidate,
                "sequence_phase",
                (candidate.sequence_phase + 1) % 4,
            )
        return 0.0

    def first_only_a2(*_args):
        nonlocal a2_calls
        a2_calls += 1
        if a2_calls > 1:
            raise AssertionError("drifted second candidate must not reach A2")
        return _a2_result(
            first_reason,
            checked=5,
            cell=(
                Cell(2, 3)
                if first_reason == "legged_body_sweep_collision"
                else None
            ),
            margin=0.08,
        )

    monkeypatch.setattr(
        LeggedStepPrimitiveV2,
        "as_oracle_candidate",
        capture_candidate,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        first_only_a2,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(clock),
    )
    assert a2_calls == 1
    assert result.reason_code == expected_reason
    assert result.failed_primitive_index == expected_index
    assert result.failed_cell == expected_cell
    assert result.failed_leg is None
    assert result.checked_cell_count == 5
    assert result.minimum_support_margin_m == expected_margin
    assert result.validated_route_hash == expected_hash


@pytest.mark.parametrize("mutation_stage", ["a2", "public_result"])
def test_legged_route_rechecks_candidate_after_a2_and_result_audit(
    monkeypatch: pytest.MonkeyPatch,
    mutation_stage: str,
) -> None:
    route = _single_heading_route(0.0)
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    real_conversion = LeggedStepPrimitiveV2.as_oracle_candidate
    real_result_token = validation_module._legged_a2_result_token_v2
    candidate_box: dict[str, LeggedStepCandidateV2] = {}

    def capture_candidate(primitive):
        candidate = real_conversion(primitive)
        candidate_box["value"] = candidate
        return candidate

    def mutate_candidate() -> None:
        candidate = candidate_box["value"]
        object.__setattr__(
            candidate,
            "sequence_phase",
            (candidate.sequence_phase + 1) % 4,
        )

    def a2_double(*_args):
        if mutation_stage == "a2":
            mutate_candidate()
        return _a2_result(checked=5)

    def result_token(result):
        token = real_result_token(result)
        if mutation_stage == "public_result":
            mutate_candidate()
        return token

    monkeypatch.setattr(
        LeggedStepPrimitiveV2,
        "as_oracle_candidate",
        capture_candidate,
    )
    monkeypatch.setattr(
        validation_module,
        "_legged_a2_result_token_v2",
        result_token,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        a2_double,
    )
    result = _route_call(route)
    assert result.reason_code == "legged_step_oracle_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 5
    assert result.validated_route_hash == expected_hash


def test_legged_route_deadline_drift_precedes_early_dynamic_audit_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    deadline = _route_deadline()
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    clock_calls = 0
    a2_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0

    object.__setattr__(deadline, "_monotonic_clock", clock)

    def mutate_deadline_then_fail(*_args, **_kwargs):
        object.__setattr__(deadline, "deadline_monotonic_s", 99.0)
        raise RuntimeError("ordinary early profile audit failure")

    def forbidden(*_args):
        nonlocal a2_calls
        a2_calls += 1
        return _a2_result()

    monkeypatch.setattr(
        validation_module,
        "_legged_profile_token_v2",
        mutate_deadline_then_fail,
    )
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
        deadline,
    )
    assert clock_calls == 0
    assert a2_calls == 0
    assert result.reason_code == "planning_deadline_contract_mismatch"
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == 0
    assert result.validated_route_hash == expected_hash


@pytest.mark.parametrize("invalid_source", ["snapshot_identity", "profile"])
def test_legged_route_invalid_entry_deadline_precedes_baseline_audits(
    monkeypatch: pytest.MonkeyPatch,
    invalid_source: str,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    anchor = FineSafetyAnchorV2(snapshot)
    deadline = _route_deadline()
    object.__setattr__(deadline, "deadline_monotonic_s", 0)
    if invalid_source == "snapshot_identity":
        anchor = FineSafetyAnchorV2(_route_snapshot())
    else:
        object.__setattr__(profile, "max_step_length_m", 0.25)
    clock_calls = 0
    a2_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0

    object.__setattr__(deadline, "_monotonic_clock", clock)

    def forbidden(*_args):
        nonlocal a2_calls
        a2_calls += 1
        return _a2_result()

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        forbidden,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        anchor,
        profile,
        deadline,
    )
    assert clock_calls == 0
    assert a2_calls == 0
    assert result.reason_code == "planning_deadline_contract_mismatch"
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == 0
    assert result.validated_route_hash is None


@pytest.mark.parametrize(
    ("drift_call", "expected_clock_calls", "expected_a2_calls", "expected_checked"),
    [(2, 1, 0, 0), (5, 3, 1, 5)],
)
def test_legged_route_dynamic_seal_failure_rechecks_direct_route_authority(
    monkeypatch: pytest.MonkeyPatch,
    drift_call: int,
    expected_clock_calls: int,
    expected_a2_calls: int,
    expected_checked: int,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    real_profile_token = validation_module._legged_profile_token_v2
    profile_calls = 0
    clock_calls = 0
    a2_calls = 0

    def drift_route_then_fail(*args, **kwargs):
        nonlocal profile_calls
        profile_calls += 1
        token = real_profile_token(*args, **kwargs)
        if profile_calls == drift_call:
            object.__setattr__(route, "total_cost", 1.0)
            raise RuntimeError("dynamic profile audit left route drift")
        return token

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0

    def a2_double(*_args):
        nonlocal a2_calls
        a2_calls += 1
        return _a2_result(checked=5)

    monkeypatch.setattr(
        validation_module,
        "_legged_profile_token_v2",
        drift_route_then_fail,
    )
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
        _route_deadline(clock),
    )
    assert profile_calls == drift_call
    assert clock_calls == expected_clock_calls
    assert a2_calls == expected_a2_calls
    assert result.reason_code == "route_structure_mismatch"
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == expected_checked
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


@pytest.mark.parametrize(
    ("target", "mode", "expected_reason", "keeps_hash", "expected_checked"),
    [
        ("route", "raise", "route_structure_mismatch", False, 0),
        ("request", "raise", "planning_request_contract_mismatch", True, 0),
        ("route", "mismatch", "route_structure_mismatch", False, 0),
        ("request", "mismatch", "planning_request_contract_mismatch", True, 0),
        ("route", "pass", "route_structure_mismatch", False, 5),
        ("request", "pass", "planning_request_contract_mismatch", True, 5),
        ("route", "step", "route_structure_mismatch", False, 5),
        ("request", "step", "planning_request_contract_mismatch", True, 5),
        ("route", "global", "route_structure_mismatch", False, 5),
        ("request", "global", "planning_request_contract_mismatch", True, 5),
    ],
)
def test_legged_route_post_a2_public_token_authority_drift_has_priority(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mode: str,
    expected_reason: str,
    keeps_hash: bool,
    expected_checked: int,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    if mode == "global":
        a2_result = _a2_result(
            "terrain_query_contract_mismatch",
            checked=5,
            margin=None,
        )
    elif mode == "step":
        a2_result = _a2_result(
            "legged_body_sweep_collision",
            checked=5,
            cell=Cell(2, 3),
            margin=0.08,
        )
    else:
        a2_result = _a2_result(checked=5)
    real_token = validation_module._legged_a2_result_token_v2
    calls = 0

    def public_token(result):
        nonlocal calls
        calls += 1
        if target == "route":
            object.__setattr__(route, "total_cost", 1.0)
        else:
            object.__setattr__(request, "request_id", "post-a2-token-drift")
        if mode == "raise":
            raise RuntimeError("ordinary public A2 token fault")
        if mode == "mismatch":
            return ("public-token-mismatch",)
        return real_token(result)

    monkeypatch.setattr(
        validation_module,
        "_legged_a2_result_token_v2",
        public_token,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: a2_result,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        _route_deadline(),
    )
    assert calls == 1
    assert result.reason_code == expected_reason
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == expected_checked
    assert result.validated_route_hash == (expected_hash if keeps_hash else None)


def test_legged_route_post_a2_public_token_deadline_drift_has_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    snapshot = _route_snapshot()
    profile = _route_profile()
    request = _route_request(snapshot, profile, route)
    deadline = _route_deadline()
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    a2_result = _a2_result(
        "terrain_query_contract_mismatch",
        checked=5,
        margin=None,
    )
    real_token = validation_module._legged_a2_result_token_v2

    def drift_deadline(result):
        object.__setattr__(deadline, "deadline_monotonic_s", 99.0)
        return real_token(result)

    monkeypatch.setattr(
        validation_module,
        "_legged_a2_result_token_v2",
        drift_deadline,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: a2_result,
    )
    result = validation_module.validate_legged_route_l2(
        route,
        request,
        FineSafetyAnchorV2(snapshot),
        profile,
        deadline,
    )
    assert result.reason_code == "planning_deadline_contract_mismatch"
    assert result.failed_primitive_index is None
    assert result.checked_cell_count == 5
    assert result.validated_route_hash == expected_hash


def test_legged_route_public_token_fault_without_authority_drift_stays_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _single_heading_route(0.0)
    expected_hash = sha256(canonical_json_bytes(route)).hexdigest()
    monkeypatch.setattr(
        validation_module,
        "_legged_a2_result_token_v2",
        lambda _result: (_ for _ in ()).throw(RuntimeError("ordinary token fault")),
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_STEP_VALIDATOR_V2",
        lambda *_args: _a2_result(),
    )
    result = _route_call(route)
    assert result.reason_code == "legged_step_oracle_contract_mismatch"
    assert result.failed_primitive_index == 0
    assert result.checked_cell_count == 0
    assert result.validated_route_hash == expected_hash


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


# Task 9A3 provider behavior.  These tests use the documented private authority
# seams so the search, geometry, accounting, and final replay remain real.
_PROVIDER_OFFSETS = (
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


def _provider_request(
    *,
    start: PoseStateV2 = PoseStateV2(0.0, 0.0, 0.0),
    goal: PoseStateV2 = PoseStateV2(-0.0625, 0.0, 0.0),
    objective: ObjectiveProfileV2 | None = None,
    budget: ResourceBudgetV2 | None = None,
    accelerator: AcceleratorPolicyV2 = AcceleratorPolicyV2.DISABLED,
) -> tuple[PlanningRequestV2, FineSafetyAnchorV2, LeggedProfileV2]:
    profile = _route_profile()
    snapshot = _route_snapshot()
    request = PlanningRequestV2(
        request_id="legged-provider-request",
        platform_profile_id=profile.profile.profile_id,
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
        objective_profile=objective or ObjectiveProfileV2(),
        resource_budget=budget or ResourceBudgetV2(),
        timeout_s=10.0,
        accelerator_policy=accelerator,
        determinism_seed=7,
    )
    return request, FineSafetyAnchorV2(snapshot), profile


def _provider_pass_result(
    candidate: LeggedStepCandidateV2,
    *_args,
) -> LeggedValidationResultV2:
    if candidate.target_foothold == WorldPoint(0.35 - 0.25, 0.25):
        return _a2_result()
    return _a2_result(
        "legged_step_length_exceeded",
        checked=1,
        leg=candidate.moving_leg,
        margin=None,
    )


def _install_provider_authorities(
    monkeypatch: pytest.MonkeyPatch,
    *,
    a2=_provider_pass_result,
    final_reason: str = "legged_route_l2_valid",
    result_sink: list[object] | None = None,
) -> list[TypedRouteV2]:
    final_calls: list[TypedRouteV2] = []

    def final(route, *_args):
        final_calls.append(route)
        route_hash = validation_module._TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2(
            route
        )
        result = validation_module.LeggedRouteValidationResultV2(
            evidence=ValidationEvidenceV2(
                validator_id=validation_module.LEGGED_ROUTE_VALIDATOR_ID_V2,
                level=ValidationLevelV2.L2,
                passed=final_reason == "legged_route_l2_valid",
                checks=(final_reason,),
            ),
            reason_code=final_reason,
            timed_out=final_reason == "planning_deadline_expired",
            failed_cell=None,
            failed_leg=None,
            failed_primitive_index=None,
            checked_cell_count=(3 if final_reason == "legged_route_l2_valid" else 0),
            minimum_support_margin_m=(
                0.08 if final_reason == "legged_route_l2_valid" else None
            ),
            validated_route_hash=route_hash,
        )
        if result_sink is not None:
            result_sink.append(result)
        return result

    monkeypatch.setattr(
        legged_module,
        "_TRUSTED_VALIDATE_LEGGED_STEP_L2_V2",
        a2,
        raising=False,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_VALIDATE_LEGGED_ROUTE_L2_V2",
        final,
        raising=False,
    )
    return final_calls


def _graph_step_v2(
    start: LeggedSearchStateV2,
    *,
    target: WorldPoint,
    end_body: PoseStateV2,
    lift_body: PoseStateV2 | None = None,
) -> LeggedStepPrimitiveV2:
    moving_leg = LEGGED_CRAWL_SEQUENCE_V2[start.sequence_phase]
    moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(moving_leg)
    contacts = list(start.foot_contacts)
    contacts[moving_index] = LeggedFootContactV2(moving_leg, target)
    end = LeggedSearchStateV2(
        end_body,
        tuple(contacts),
        (start.sequence_phase + 1) % 4,
    )
    lift = start.body_state if lift_body is None else lift_body
    foot, distance, energy = legged_module._resource_values_v2(
        start,
        lift,
        end,
        moving_leg,
        target,
    )
    return LeggedStepPrimitiveV2(
        kind=PrimitiveKindV2.LEG_STEP,
        start_state=start.body_state,
        end_state=end_body,
        duration_s=1.0,
        distance_m=distance,
        energy_cost=energy,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        start_legged_state=start,
        lift_body_state=lift,
        end_legged_state=end,
        moving_leg=moving_leg,
        target_foothold=target,
        foot_travel_m=foot,
    )


def _install_graph_v2(
    monkeypatch: pytest.MonkeyPatch,
    edges: tuple[
        tuple[LeggedSearchStateV2, tuple[float, float], LeggedStepPrimitiveV2],
        ...,
    ],
) -> None:
    real_candidate = legged_module._provider_candidate_v2
    edge_by_key = {
        (legged_state_key_v2(start), offset): primitive
        for start, offset, primitive in edges
    }
    admitted = tuple(primitive.as_oracle_candidate() for _, _, primitive in edges)

    def graph_candidate(state, fixed_yaw, offset):
        primitive = edge_by_key.get((legged_state_key_v2(state), offset))
        if primitive is None:
            return real_candidate(state, fixed_yaw, offset)
        assert primitive.start_legged_state == state
        return primitive, primitive.as_oracle_candidate()

    def graph_a2(candidate, *_args):
        if any(candidate == expected for expected in admitted):
            return _a2_result()
        return _a2_result(
            "legged_step_length_exceeded",
            checked=1,
            leg=candidate.moving_leg,
            margin=None,
        )

    monkeypatch.setattr(legged_module, "_provider_candidate_v2", graph_candidate)
    _install_provider_authorities(monkeypatch, a2=graph_a2)


def _queue_entry_token_v2(entry: object) -> tuple[object, ...]:
    assert type(entry) is legged_module.SearchQueueEntryV2
    return (
        entry.candidate_id,
        entry.primitive_key,
        entry.state_key,
        entry.path_cost.hex(),
        entry.anchor_heuristic.hex(),
        entry.auxiliary_heuristics,
        (
            (entry.path_cost + entry.anchor_heuristic).hex(),
            entry.path_cost.hex(),
            entry.state_key,
            entry.primitive_key,
            entry.candidate_id,
        ),
    )


def _install_queue_trace_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[object, ...]]]:
    events: list[tuple[str, tuple[object, ...]]] = []
    real_queue = legged_module.StableSearchQueueV2

    class TracingQueue(real_queue):
        def extend(self, entries):
            events.extend(
                ("push", _queue_entry_token_v2(entry)) for entry in entries
            )
            return super().extend(entries)

        def pop_anchor(self):
            entry = super().pop_anchor()
            events.append(("pop", _queue_entry_token_v2(entry)))
            return entry

    monkeypatch.setattr(legged_module, "StableSearchQueueV2", TracingQueue)
    return events


def _install_scripted_queue_v2(
    monkeypatch: pytest.MonkeyPatch,
    candidate_order: tuple[str, ...],
) -> list[tuple[str, tuple[object, ...]]]:
    events: list[tuple[str, tuple[object, ...]]] = []

    class ScriptedQueue:
        def __init__(self) -> None:
            self.entries: dict[str, object] = {}
            self.next_index = 0

        def __len__(self) -> int:
            return len(self.entries)

        def extend(self, entries) -> None:
            assert type(entries) is tuple
            for entry in entries:
                token = _queue_entry_token_v2(entry)
                events.append(("push", token))
                assert entry.candidate_id not in self.entries
                self.entries[entry.candidate_id] = entry

        def pop_anchor(self):
            candidate_id = candidate_order[self.next_index]
            self.next_index += 1
            entry = self.entries.pop(candidate_id)
            events.append(("pop", _queue_entry_token_v2(entry)))
            return entry

    monkeypatch.setattr(legged_module, "StableSearchQueueV2", ScriptedQueue)
    return events


def _lineage_graph_v2() -> tuple[
    LeggedSearchStateV2,
    tuple[
        tuple[LeggedSearchStateV2, tuple[float, float], LeggedStepPrimitiveV2],
        ...,
    ],
    tuple[LeggedStepPrimitiveV2, ...],
]:
    start = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    expensive_s = _graph_step_v2(
        start,
        target=WorldPoint(0.10, 0.25),
        end_body=PoseStateV2(-0.20, 0.0, 0.0),
        lift_body=PoseStateV2(10.0, 0.0, 0.0),
    )
    branch_1 = _graph_step_v2(
        start,
        target=WorldPoint(0.20, 0.25),
        end_body=PoseStateV2(-0.05, 0.20, 0.0),
    )
    branch_2 = _graph_step_v2(
        branch_1.end_legged_state,
        target=branch_1.end_legged_state.foot_contacts[3].foothold,
        end_body=PoseStateV2(-0.05, 0.30, 0.0),
    )
    branch_3 = _graph_step_v2(
        branch_2.end_legged_state,
        target=branch_2.end_legged_state.foot_contacts[1].foothold,
        end_body=PoseStateV2(-0.05, 0.40, 0.0),
    )
    branch_4 = _graph_step_v2(
        branch_3.end_legged_state,
        target=branch_3.end_legged_state.foot_contacts[2].foothold,
        end_body=PoseStateV2(-0.05, 0.50, 0.0),
    )
    improved_s = _graph_step_v2(
        branch_4.end_legged_state,
        target=expensive_s.target_foothold,
        end_body=expensive_s.end_state,
        lift_body=PoseStateV2(-0.10, 0.25, 0.0),
    )
    child_c = _graph_step_v2(
        expensive_s.end_legged_state,
        target=WorldPoint(0.0, -0.25),
        end_body=PoseStateV2(-0.40, 0.0, 0.0),
        lift_body=PoseStateV2(-0.30, 0.0, 0.0),
    )
    assert improved_s.end_legged_state == expensive_s.end_legged_state
    edges = (
        (start, _PROVIDER_OFFSETS[0], expensive_s),
        (start, _PROVIDER_OFFSETS[1], branch_1),
        (expensive_s.end_legged_state, _PROVIDER_OFFSETS[0], child_c),
        (branch_1.end_legged_state, _PROVIDER_OFFSETS[0], branch_2),
        (branch_2.end_legged_state, _PROVIDER_OFFSETS[0], branch_3),
        (branch_3.end_legged_state, _PROVIDER_OFFSETS[0], branch_4),
        (branch_4.end_legged_state, _PROVIDER_OFFSETS[0], improved_s),
    )
    return (
        start,
        edges,
        (
            expensive_s,
            branch_1,
            branch_2,
            branch_3,
            branch_4,
            improved_s,
            child_c,
        ),
    )


def test_legged_provider_public_surface_and_exact_profile_are_frozen() -> None:
    provider_type = getattr(legged_module, "LeggedPrimitiveProviderV2")
    assert getattr(legged_module, "LEGGED_LOCAL_FOOTHOLD_OFFSETS_V2") == (
        _PROVIDER_OFFSETS
    )
    assert all(
        type(offset) is tuple
        and len(offset) == 2
        and all(type(value) is float for value in offset)
        for offset in _PROVIDER_OFFSETS
    )
    assert getattr(legged_module, "LEGGED_SEARCH_MEMORY_ACCOUNTING_ID_V2") == (
        "legged_search_fixed_record_512b/v1"
    )
    assert getattr(legged_module, "LEGGED_SEARCH_RECORD_BYTES_V2") == 512
    for name in (
        "LEGGED_LOCAL_FOOTHOLD_OFFSETS_V2",
        "LEGGED_SEARCH_MEMORY_ACCOUNTING_ID_V2",
        "LEGGED_SEARCH_RECORD_BYTES_V2",
        "LeggedPrimitiveProviderV2",
    ):
        assert getattr(provider_exports, name) is getattr(legged_module, name)
        assert getattr(v2, name) is getattr(legged_module, name)
    profile = _route_profile()
    provider = provider_type(profile)
    assert provider.profile is profile.profile
    assert isinstance(provider, provider_exports.PrimitiveProviderV2)
    with pytest.raises(FrozenInstanceError):
        provider.legged_profile = profile  # type: ignore[misc]
    with pytest.raises(TypeError):
        provider_type(object())


def test_legged_provider_generates_exact_geometry_costs_and_final_replay_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_calls = _install_provider_authorities(monkeypatch)
    request, anchor, profile = _provider_request()
    provider = getattr(legged_module, "LeggedPrimitiveProviderV2")(profile)
    outcome = provider.plan(request, anchor, _route_deadline())
    assert type(outcome) is PlanningSuccessV2
    assert len(final_calls) == 1
    primitive = outcome.route.primitives[0]
    assert type(primitive) is LeggedStepPrimitiveV2
    assert primitive.start_legged_state == nominal_legged_search_state_v2(
        request.start_state
    )
    assert primitive.moving_leg is LegIdV2.FRONT_LEFT
    assert primitive.target_foothold == WorldPoint(0.35 - 0.25, 0.25)
    assert primitive.lift_body_state == PoseStateV2(
        fsum((0.35, -0.35, -0.35)) / 3.0,
        fsum((-0.25, 0.25, -0.25)) / 3.0,
        0.0,
    )
    assert primitive.end_state == PoseStateV2(-0.0625, 0.0, 0.0)
    assert primitive.end_legged_state.sequence_phase == 1
    assert primitive.end_legged_state.foot_contacts[0].foothold == WorldPoint(
        0.35 - 0.25,
        0.25,
    )
    assert primitive.end_legged_state.foot_contacts[1:] == (
        primitive.start_legged_state.foot_contacts[1:]
    )
    distance = hypot(primitive.lift_body_state.x_m, primitive.lift_body_state.y_m)
    distance += hypot(
        primitive.end_state.x_m - primitive.lift_body_state.x_m,
        primitive.end_state.y_m - primitive.lift_body_state.y_m,
    )
    assert primitive.distance_m.hex() == distance.hex()
    assert primitive.energy_cost.hex() == (distance + 0.25).hex()
    assert outcome.cost_breakdown.distance_cost.hex() == 0.0.hex()
    assert outcome.cost_breakdown.risk_cost.hex() == 0.0.hex()
    assert outcome.cost_breakdown.energy_cost.hex() == (
        0.5 * primitive.energy_cost
    ).hex()
    assert outcome.cost_breakdown.time_cost.hex() == 0.5.hex()
    assert outcome.cost_breakdown.total_cost.hex() == outcome.route.total_cost.hex()
    assert outcome.observation_projection.sample_states == (
        primitive.start_state,
        primitive.lift_body_state,
        primitive.end_state,
    )
    assert outcome.observation_projection.source == (
        "legged_route_body_samples_gain_not_computed/v1"
    )


def test_legged_provider_success_payload_and_search_telemetry_are_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_provider_authorities(monkeypatch)
    request, anchor, profile = _provider_request()
    outcome = getattr(legged_module, "LeggedPrimitiveProviderV2")(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningSuccessV2
    assert outcome.validation_evidence.validator_id == (
        "path-planner-v2-legged-route-l2/v1"
    )
    assert outcome.validation_evidence.checks == ("legged_route_l2_valid",)
    assert outcome.search_telemetry.expanded_states == 1
    assert outcome.search_telemetry.generated_primitives == 13
    assert outcome.search_telemetry.rejected_l2 == 12
    assert outcome.search_telemetry.termination_reason == "legged_route_l2_valid"
    assert outcome.search_telemetry.accelerator_used is False
    assert outcome.search_telemetry.ackermann_feasible_claimed is False
    assert outcome.cache_evidence.cache_namespace == (
        "path-planner-v2-legged-cache-disabled/v1"
    )
    assert outcome.cache_evidence.cache_key == "legged-cache-disabled/v1"
    assert outcome.cache_evidence.hit is False


@pytest.mark.parametrize(
    ("request_kwargs", "reason", "category", "stage", "details"),
    [
        (
            {"objective": ObjectiveProfileV2(risk_weight=0.25)},
            "legged_risk_objective_unsupported",
            FailureCategoryV2.UNSUPPORTED_CAPABILITY,
            "capability_check",
            (),
        ),
        (
            {"accelerator": AcceleratorPolicyV2.REQUIRED},
            "legged_accelerator_required_unsupported",
            FailureCategoryV2.UNSUPPORTED_CAPABILITY,
            "capability_check",
            (),
        ),
        (
            {"goal": PoseStateV2(-0.0625, 0.0, 0.25)},
            "legged_goal_heading_unreachable",
            FailureCategoryV2.GOAL_POSE_UNREACHABLE,
            "goal_semantics",
            (),
        ),
        (
            {"goal": PoseStateV2(0.0, 0.0, 2.0 * pi)},
            "legged_hold_oracle_unavailable",
            FailureCategoryV2.UNSUPPORTED_CAPABILITY,
            "capability_check",
            (),
        ),
        (
            {"budget": ResourceBudgetV2(max_expanded_states=0)},
            "legged_expansion_budget_exhausted",
            FailureCategoryV2.RESOURCE_LIMIT,
            "resource_check",
            (("attempted_expanded_states", 1), ("max_expanded_states", 0)),
        ),
        (
            {"budget": ResourceBudgetV2(max_route_states=1)},
            "legged_route_state_budget_exhausted",
            FailureCategoryV2.RESOURCE_LIMIT,
            "resource_check",
            (
                ("attempted_route_states", 2),
                ("effective_max_route_states", 1),
                ("requested_max_route_states", 1),
            ),
        ),
        (
            {"budget": ResourceBudgetV2(max_memory_bytes=511)},
            "legged_search_memory_budget_exceeded",
            FailureCategoryV2.RESOURCE_LIMIT,
            "resource_check",
            (
                ("accounting_id", "legged_search_fixed_record_512b/v1"),
                ("attempted_record_count", 1),
                ("max_memory_bytes", 511),
                ("record_bytes", 512),
                ("retained_record_count", 0),
            ),
        ),
    ],
)
def test_legged_provider_preflight_failures_are_exact_and_do_not_call_a2(
    monkeypatch: pytest.MonkeyPatch,
    request_kwargs,
    reason,
    category,
    stage,
    details,
) -> None:
    a2_calls = []
    monkeypatch.setattr(
        legged_module,
        "_TRUSTED_VALIDATE_LEGGED_STEP_L2_V2",
        lambda *args: a2_calls.append(args),
        raising=False,
    )
    request, anchor, profile = _provider_request(**request_kwargs)
    outcome = getattr(legged_module, "LeggedPrimitiveProviderV2")(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == reason
    assert outcome.category is category
    assert outcome.evidence.stage == stage
    assert outcome.evidence.checks == (reason,)
    assert outcome.evidence.details == details
    assert outcome.search_telemetry.termination_reason == reason
    assert a2_calls == []


def test_legged_provider_all_step_failures_return_no_complete_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(candidate, *_args):
        return _a2_result(
            "legged_step_length_exceeded",
            checked=1,
            leg=candidate.moving_leg,
            margin=None,
        )

    _install_provider_authorities(monkeypatch, a2=reject)
    request, anchor, profile = _provider_request()
    outcome = getattr(legged_module, "LeggedPrimitiveProviderV2")(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "legged_no_complete_route"
    assert outcome.category is FailureCategoryV2.NO_COMPLETE_ROUTE
    assert outcome.evidence.stage == "search"
    assert outcome.search_telemetry.expanded_states == 1
    assert outcome.search_telemetry.generated_primitives == 13
    assert outcome.search_telemetry.rejected_l2 == 13


@pytest.mark.parametrize(
    ("a2_value", "reason", "category"),
    [
        (object(), "legged_step_oracle_contract_mismatch", FailureCategoryV2.INTERNAL_ERROR),
        (
            _a2_result("terrain_snapshot_hash_mismatch", checked=0, margin=None),
            "terrain_snapshot_hash_mismatch",
            FailureCategoryV2.VALIDATION_FAILED,
        ),
    ],
)
def test_legged_provider_a2_contract_and_global_failures_are_not_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    a2_value,
    reason,
    category,
) -> None:
    _install_provider_authorities(monkeypatch, a2=lambda *_args: a2_value)
    request, anchor, profile = _provider_request()
    outcome = getattr(legged_module, "LeggedPrimitiveProviderV2")(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == reason
    assert outcome.category is category
    assert outcome.evidence.stage == "search_edge_validation"
    assert outcome.search_telemetry.generated_primitives == 1
    assert outcome.search_telemetry.rejected_l2 == 0


def test_legged_provider_final_replay_nonpass_denies_found_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_provider_authorities(
        monkeypatch,
        final_reason="terrain_snapshot_hash_mismatch",
    )
    request, anchor, profile = _provider_request()
    outcome = getattr(legged_module, "LeggedPrimitiveProviderV2")(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "terrain_snapshot_hash_mismatch"
    assert outcome.category is FailureCategoryV2.VALIDATION_FAILED
    assert outcome.evidence.stage == "route_validation"


def test_legged_provider_entry_deadline_uses_exact_cutoff_and_stage() -> None:
    request, anchor, profile = _provider_request()
    deadline = PlanningDeadlineV2(0.0, 1.0, lambda: 1.0)
    outcome = getattr(legged_module, "LeggedPrimitiveProviderV2")(profile).plan(
        request,
        anchor,
        deadline,
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "planning_deadline_expired"
    assert outcome.category is FailureCategoryV2.TIMEOUT
    assert outcome.evidence.stage == "provider_entry"
    assert outcome.search_telemetry.timed_out is True
    assert outcome.search_telemetry.elapsed_s == 1.0


def _mutate_final_payload(
    kind: str,
    route: TypedRouteV2,
    result: object,
) -> None:
    primitive = route.primitives[0]
    assert type(primitive) is LeggedStepPrimitiveV2
    if kind == "target":
        target = primitive.target_foothold
        object.__setattr__(
            primitive,
            "target_foothold",
            WorldPoint(target.x + 0.25, target.y),
        )
    elif kind == "full_state":
        object.__setattr__(primitive.end_legged_state, "sequence_phase", 3)
    elif kind == "resource":
        object.__setattr__(
            primitive,
            "energy_cost",
            nextafter(primitive.energy_cost, float("inf")),
        )
    elif kind == "route":
        object.__setattr__(
            route,
            "total_cost",
            nextafter(route.total_cost, float("inf")),
        )
    elif kind == "evidence":
        object.__setattr__(result.evidence, "checks", ("forged-final-check",))
    else:  # pragma: no cover - test helper contract
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "mutation_kind",
    ["target", "full_state", "resource", "route", "evidence"],
)
def test_legged_provider_success_return_clock_cannot_mutate_final_authority(
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    final_results: list[object] = []
    final_calls = _install_provider_authorities(
        monkeypatch,
        result_sink=final_results,
    )
    request, anchor, profile = _provider_request()
    clocks_after_final = 0

    def clock() -> float:
        nonlocal clocks_after_final
        if final_calls:
            clocks_after_final += 1
            if clocks_after_final == 2:
                _mutate_final_payload(
                    mutation_kind,
                    final_calls[0],
                    final_results[0],
                )
        return 0.0

    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        PlanningDeadlineV2(0.0, 100.0, clock),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "legged_route_oracle_contract_mismatch"
    assert outcome.category is FailureCategoryV2.INTERNAL_ERROR
    assert outcome.evidence.stage == "route_validation"
    assert clocks_after_final == 2


def test_legged_provider_success_return_authority_drift_keeps_success_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_calls = _install_provider_authorities(monkeypatch)
    request, anchor, profile = _provider_request()
    clocks_after_final = 0

    def clock() -> float:
        nonlocal clocks_after_final
        if final_calls:
            clocks_after_final += 1
            if clocks_after_final == 2:
                object.__setattr__(request, "determinism_seed", 8)
        return 0.0

    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        PlanningDeadlineV2(0.0, 100.0, clock),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "planning_request_contract_mismatch"
    assert outcome.category is FailureCategoryV2.VALIDATION_FAILED
    assert outcome.evidence.stage == "success_return"
    assert clocks_after_final == 2


@pytest.mark.parametrize("critical", [KeyboardInterrupt, SystemExit, MemoryError])
def test_legged_provider_final_no_clock_reseal_propagates_critical_faults(
    monkeypatch: pytest.MonkeyPatch,
    critical,
) -> None:
    _install_provider_authorities(monkeypatch)
    request, anchor, profile = _provider_request()
    real_digest = validation_module._TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2
    calls = 0

    def digest(route):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise critical("critical final reseal")
        return real_digest(route)

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2",
        digest,
    )
    with pytest.raises(critical):
        LeggedPrimitiveProviderV2(profile).plan(
            request,
            anchor,
            _route_deadline(),
        )


def _mutate_provider_authority(
    kind: str,
    request: PlanningRequestV2,
    anchor: FineSafetyAnchorV2,
    profile: LeggedProfileV2,
) -> str:
    if kind == "request":
        object.__setattr__(request, "determinism_seed", 8)
        return "planning_request_contract_mismatch"
    if kind == "profile":
        object.__setattr__(profile, "max_step_length_m", 0.25)
        return "legged_profile_contract_mismatch"
    if kind == "anchor":
        object.__setattr__(anchor, "snapshot", _route_snapshot())
        return "terrain_snapshot_identity_mismatch"
    if kind == "snapshot_hash":
        replacement = np.array(
            request.terrain_snapshot.elevation_m,
            dtype=np.float64,
            copy=True,
        )
        replacement[0, 0] = 1.0
        object.__setattr__(
            request.terrain_snapshot,
            "elevation_m",
            replacement,
        )
        return "terrain_snapshot_hash_mismatch"
    raise AssertionError(kind)  # pragma: no cover


@pytest.mark.parametrize(
    ("authority_kind", "bad_clock_kind"),
    [
        (authority_kind, bad_clock_kind)
        for authority_kind in ("request", "profile", "anchor", "snapshot_hash")
        for bad_clock_kind in ("raise", "wrong_type", "nonfinite")
    ],
)
def test_legged_provider_bad_clock_reseals_authority_before_clock_contract(
    authority_kind: str,
    bad_clock_kind: str,
) -> None:
    request, anchor, profile = _provider_request()
    expected_reason = ""

    def clock():
        nonlocal expected_reason
        expected_reason = _mutate_provider_authority(
            authority_kind,
            request,
            anchor,
            profile,
        )
        if bad_clock_kind == "raise":
            raise RuntimeError("ordinary clock fault")
        if bad_clock_kind == "wrong_type":
            return object()
        return float("nan")

    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        PlanningDeadlineV2(0.0, 100.0, clock),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == expected_reason
    assert outcome.evidence.stage == "provider_entry"


@pytest.mark.parametrize("critical", [KeyboardInterrupt, SystemExit, MemoryError])
def test_legged_provider_clock_critical_faults_propagate_unchanged(critical) -> None:
    request, anchor, profile = _provider_request()

    def clock():
        raise critical("critical clock fault")

    with pytest.raises(critical):
        LeggedPrimitiveProviderV2(profile).plan(
            request,
            anchor,
            PlanningDeadlineV2(0.0, 100.0, clock),
        )


@pytest.mark.parametrize("mutation_kind", ["target", "full_state", "resource"])
def test_legged_provider_reaudits_primitive_after_search_a2(
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    active_primitives: list[LeggedStepPrimitiveV2] = []
    real_as_candidate = LeggedStepPrimitiveV2.as_oracle_candidate

    def capture(primitive):
        active_primitives.append(primitive)
        return real_as_candidate(primitive)

    monkeypatch.setattr(LeggedStepPrimitiveV2, "as_oracle_candidate", capture)

    def a2(candidate, *_args):
        if candidate.target_foothold == WorldPoint(0.35 - 0.25, 0.25):
            primitive = active_primitives[-1]
            if mutation_kind == "target":
                object.__setattr__(
                    primitive,
                    "target_foothold",
                    WorldPoint(primitive.target_foothold.x + 0.25, 0.25),
                )
            elif mutation_kind == "full_state":
                object.__setattr__(primitive.end_legged_state, "sequence_phase", 3)
            else:
                object.__setattr__(
                    primitive,
                    "energy_cost",
                    nextafter(primitive.energy_cost, float("inf")),
                )
            return _a2_result()
        return _a2_result(
            "legged_step_length_exceeded",
            checked=1,
            leg=candidate.moving_leg,
            margin=None,
        )

    _install_provider_authorities(monkeypatch, a2=a2)
    request, anchor, profile = _provider_request()
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "legged_step_oracle_contract_mismatch"
    assert outcome.evidence.stage == "search_edge_validation"
    assert outcome.search_telemetry.generated_primitives == 2
    assert outcome.search_telemetry.rejected_l2 == 1


@pytest.mark.parametrize("critical", [KeyboardInterrupt, SystemExit, MemoryError])
def test_legged_provider_post_a2_primitive_reaudit_propagates_critical(
    monkeypatch: pytest.MonkeyPatch,
    critical,
) -> None:
    after_a2 = False
    real_audit = legged_module._audit_primitive_canonical

    def audit(primitive):
        if after_a2:
            raise critical("critical primitive reaudit")
        return real_audit(primitive)

    def a2(candidate, *_args):
        nonlocal after_a2
        if candidate.target_foothold == WorldPoint(0.35 - 0.25, 0.25):
            after_a2 = True
            return _a2_result()
        return _a2_result(
            "legged_step_length_exceeded",
            checked=1,
            leg=candidate.moving_leg,
            margin=None,
        )

    monkeypatch.setattr(legged_module, "_audit_primitive_canonical", audit)
    _install_provider_authorities(monkeypatch, a2=a2)
    request, anchor, profile = _provider_request()
    with pytest.raises(critical):
        LeggedPrimitiveProviderV2(profile).plan(
            request,
            anchor,
            _route_deadline(),
        )


def _zero_offset_only_a2(*, novel_at_phase_three: bool):
    def validate(candidate, *_args):
        moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(candidate.moving_leg)
        source = candidate.foot_contacts[moving_index].foothold
        dx = candidate.target_foothold.x - source.x
        dy = candidate.target_foothold.y - source.y
        expected_dx = -0.25 if novel_at_phase_three and candidate.sequence_phase == 3 else 0.0
        if dx == expected_dx and dy == 0.0:
            return _a2_result()
        return _a2_result(
            "legged_step_length_exceeded",
            checked=1,
            leg=candidate.moving_leg,
            margin=None,
        )

    return validate


@pytest.mark.parametrize(
    ("novel_at_phase_three", "expected_reason"),
    [
        (False, "legged_no_complete_route"),
        (True, "legged_route_state_budget_exhausted"),
    ],
)
def test_legged_provider_route_cap_only_reports_nondominated_novel_child(
    monkeypatch: pytest.MonkeyPatch,
    novel_at_phase_three: bool,
    expected_reason: str,
) -> None:
    _install_provider_authorities(
        monkeypatch,
        a2=_zero_offset_only_a2(novel_at_phase_three=novel_at_phase_three),
    )
    request, anchor, profile = _provider_request(
        goal=PoseStateV2(2.0, 2.0, 0.0),
        budget=ResourceBudgetV2(max_route_states=4),
    )
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == expected_reason
    assert outcome.evidence.stage == "search"
    if novel_at_phase_three:
        assert outcome.evidence.details == (
            ("attempted_route_states", 5),
            ("effective_max_route_states", 4),
            ("requested_max_route_states", 4),
        )
    else:
        assert outcome.evidence.details == ()


@pytest.mark.parametrize(
    ("max_memory_bytes", "expected_type", "expected_reason"),
    [
        (512, PlanningFailureV2, "legged_search_memory_budget_exceeded"),
        (1024, PlanningSuccessV2, "legged_route_l2_valid"),
        (0, PlanningSuccessV2, "legged_route_l2_valid"),
    ],
)
def test_legged_provider_runtime_memory_boundary_counts_only_admitted_records(
    monkeypatch: pytest.MonkeyPatch,
    max_memory_bytes: int,
    expected_type: type,
    expected_reason: str,
) -> None:
    _install_provider_authorities(monkeypatch)
    request, anchor, profile = _provider_request(
        budget=ResourceBudgetV2(max_memory_bytes=max_memory_bytes),
    )
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is expected_type
    assert outcome.search_telemetry.termination_reason == expected_reason
    if max_memory_bytes == 512:
        assert outcome.evidence.stage == "search"
        assert outcome.evidence.details == (
            ("accounting_id", "legged_search_fixed_record_512b/v1"),
            ("attempted_record_count", 2),
            ("max_memory_bytes", 512),
            ("record_bytes", 512),
            ("retained_record_count", 1),
        )


def test_legged_provider_multiedge_lineage_cost_fold_and_goal_pop_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    first, _ = legged_module._provider_candidate_v2(initial, 0.0, (-0.25, 0.0))
    second, _ = legged_module._provider_candidate_v2(
        first.end_legged_state,
        0.0,
        (-0.25, 0.0),
    )

    def only_negative_x_offset(candidate, *_args):
        moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(candidate.moving_leg)
        source = candidate.foot_contacts[moving_index].foothold
        if candidate.target_foothold == WorldPoint(source.x - 0.25, source.y):
            return _a2_result()
        return _a2_result(
            "legged_step_length_exceeded",
            checked=1,
            leg=candidate.moving_leg,
            margin=None,
        )

    _install_provider_authorities(monkeypatch, a2=only_negative_x_offset)
    request, anchor, profile = _provider_request(goal=second.end_state)
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningSuccessV2
    assert outcome.route.primitives == (first, second)
    assert outcome.search_telemetry.expanded_states == 2
    assert outcome.route.primitives[0].end_legged_state == (
        outcome.route.primitives[1].start_legged_state
    )
    expected_energy = fsum(
        0.5 * primitive.energy_cost for primitive in outcome.route.primitives
    )
    expected_time = fsum(
        0.5 * primitive.duration_s for primitive in outcome.route.primitives
    )
    assert outcome.cost_breakdown.energy_cost.hex() == expected_energy.hex()
    assert outcome.cost_breakdown.time_cost.hex() == expected_time.hex()
    assert outcome.route.total_cost.hex() == fsum((expected_energy, expected_time)).hex()
    assert outcome.observation_projection.sample_states == (
        first.start_state,
        first.lift_body_state,
        first.end_state,
        second.lift_body_state,
        second.end_state,
    )


@pytest.mark.parametrize(
    ("reason", "failed_index", "expected_category"),
    [
        ("planning_deadline_expired", None, FailureCategoryV2.TIMEOUT),
        ("route_state_budget_exceeded", 0, FailureCategoryV2.RESOURCE_LIMIT),
        ("route_goal_contract_mismatch", None, FailureCategoryV2.GOAL_POSE_UNREACHABLE),
        ("route_goal_tolerance_exceeded", 0, FailureCategoryV2.GOAL_POSE_UNREACHABLE),
        ("legged_step_oracle_contract_mismatch", 0, FailureCategoryV2.INTERNAL_ERROR),
        ("terrain_snapshot_hash_mismatch", None, FailureCategoryV2.VALIDATION_FAILED),
        ("route_incomplete", None, FailureCategoryV2.VALIDATION_FAILED),
        ("route_connectivity_mismatch", 0, FailureCategoryV2.VALIDATION_FAILED),
    ],
)
def test_legged_provider_final_replay_reason_mapping_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    failed_index: int | None,
    expected_category: FailureCategoryV2,
) -> None:
    _install_provider_authorities(monkeypatch)

    def final(route, *_args):
        digest = validation_module._TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2(route)
        return validation_module._legged_route_result_v2(
            reason,
            failed_primitive_index=failed_index,
            checked_cell_count=0,
            validated_route_hash=digest,
        )

    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_VALIDATE_LEGGED_ROUTE_L2_V2",
        final,
    )
    request, anchor, profile = _provider_request()
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == reason
    assert outcome.category is expected_category
    assert outcome.evidence.stage == "route_validation"


def test_legged_provider_cross_branch_equal_cost_queue_order_and_goal_pop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    branch_x, _ = legged_module._provider_candidate_v2(
        initial,
        0.0,
        (-0.25, 0.0),
    )
    branch_y, _ = legged_module._provider_candidate_v2(
        initial,
        0.0,
        (0.0, -0.25),
    )
    assert branch_x.duration_s.hex() == branch_y.duration_s.hex()
    expected = min(
        (branch_x, branch_y),
        key=lambda primitive: legged_state_key_v2(primitive.end_legged_state),
    )
    admitted_targets = {branch_x.target_foothold, branch_y.target_foothold}

    def two_equal_branches(candidate, *_args):
        if candidate.target_foothold in admitted_targets:
            return _a2_result()
        return _a2_result(
            "legged_step_length_exceeded",
            checked=1,
            leg=candidate.moving_leg,
            margin=None,
        )

    queue_events: list[tuple[str, object]] = []
    real_queue = legged_module.StableSearchQueueV2

    class TracingQueue(real_queue):
        def extend(self, entries):
            queue_events.extend(("push", entry) for entry in entries)
            return super().extend(entries)

        def pop_anchor(self):
            entry = super().pop_anchor()
            queue_events.append(("pop", entry))
            return entry

    monkeypatch.setattr(legged_module, "StableSearchQueueV2", TracingQueue)
    _install_provider_authorities(monkeypatch, a2=two_equal_branches)
    request, anchor, profile = _provider_request(
        goal=expected.end_state,
        objective=ObjectiveProfileV2(
            distance_weight=0.0,
            risk_weight=0.0,
            energy_weight=0.0,
            time_weight=1.0,
        ),
    )
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningSuccessV2
    assert outcome.route.primitives == (expected,)
    popped = [entry for action, entry in queue_events if action == "pop"]
    pushed = [entry for action, entry in queue_events if action == "push"]
    assert [entry.candidate_id for entry in popped] == [
        "legged-node-00000000000000000000",
        next(
            entry.candidate_id
            for entry in pushed
            if entry.state_key == legged_state_key_v2(expected.end_legged_state)
        ),
    ]
    assert popped[1].path_cost.hex() == outcome.route.total_cost.hex()
    assert outcome.search_telemetry.expanded_states == 1


def test_legged_provider_equal_cost_no_replace_and_stale_closed_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    better, _ = legged_module._provider_candidate_v2(
        initial,
        0.0,
        (-0.25, 0.0),
    )
    worse_lift = PoseStateV2(
        better.lift_body_state.x_m + 1.0,
        better.lift_body_state.y_m,
        better.lift_body_state.heading_rad,
    )
    worse_distance = hypot(
        worse_lift.x_m - better.start_state.x_m,
        worse_lift.y_m - better.start_state.y_m,
    ) + hypot(
        better.end_state.x_m - worse_lift.x_m,
        better.end_state.y_m - worse_lift.y_m,
    )
    worse = LeggedStepPrimitiveV2(
        kind=PrimitiveKindV2.LEG_STEP,
        start_state=better.start_state,
        end_state=better.end_state,
        duration_s=better.duration_s,
        distance_m=worse_distance,
        energy_cost=worse_distance + better.foot_travel_m,
        observation_contribution=better.observation_contribution,
        validation_level=ValidationLevelV2.L2,
        start_legged_state=better.start_legged_state,
        lift_body_state=worse_lift,
        end_legged_state=better.end_legged_state,
        moving_leg=better.moving_leg,
        target_foothold=better.target_foothold,
        foot_travel_m=better.foot_travel_m,
    )
    real_candidate = legged_module._provider_candidate_v2

    def graph_candidate(state, fixed_yaw, offset):
        if state.sequence_phase == 0:
            if offset == _PROVIDER_OFFSETS[0]:
                return worse, worse.as_oracle_candidate()
            if offset in (_PROVIDER_OFFSETS[1], _PROVIDER_OFFSETS[2]):
                return better, better.as_oracle_candidate()
        return real_candidate(state, fixed_yaw, offset)

    def phase_zero_only(candidate, *_args):
        if (
            candidate.sequence_phase == 0
            and candidate.target_foothold == better.target_foothold
        ):
            return _a2_result()
        return _a2_result(
            "legged_step_length_exceeded",
            checked=1,
            leg=candidate.moving_leg,
            margin=None,
        )

    queue_events: list[tuple[str, object]] = []
    real_queue = legged_module.StableSearchQueueV2

    class TracingQueue(real_queue):
        def extend(self, entries):
            queue_events.extend(("push", entry) for entry in entries)
            return super().extend(entries)

        def pop_anchor(self):
            entry = super().pop_anchor()
            queue_events.append(("pop", entry))
            return entry

    monkeypatch.setattr(legged_module, "_provider_candidate_v2", graph_candidate)
    monkeypatch.setattr(legged_module, "StableSearchQueueV2", TracingQueue)
    _install_provider_authorities(monkeypatch, a2=phase_zero_only)
    request, anchor, profile = _provider_request(
        goal=PoseStateV2(2.0, 2.0, 0.0),
    )
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "legged_no_complete_route"
    pushed = [entry for action, entry in queue_events if action == "push"]
    popped = [entry for action, entry in queue_events if action == "pop"]
    assert len(pushed) == 3  # start, worse, better; equal best is not reinserted
    assert [entry.candidate_id for entry in popped] == [
        "legged-node-00000000000000000000",
        "legged-node-00000000000000000002",
        "legged-node-00000000000000000001",
    ]
    assert popped[1].path_cost < popped[2].path_cost
    assert outcome.search_telemetry.expanded_states == 2
    assert outcome.search_telemetry.generated_primitives == 26


@pytest.mark.parametrize(
    ("clock_index", "expected_stage", "expected_a2_calls", "expected_final_calls"),
    [
        (1, "provider_entry", 0, 0),
        (2, "preflight", 0, 0),
        (3, "search_pop", 0, 0),
        (4, "search_edge_validation", 0, 0),
        (5, "search_edge_validation", 1, 0),
        (31, "route_validation", 13, 0),
        (32, "route_validation", 13, 1),
        (33, "success_return", 13, 1),
    ],
)
def test_legged_provider_deadline_stage_and_oracle_boundary_matrix(
    monkeypatch: pytest.MonkeyPatch,
    clock_index: int,
    expected_stage: str,
    expected_a2_calls: int,
    expected_final_calls: int,
) -> None:
    a2_calls: list[LeggedStepCandidateV2] = []

    def counted_a2(candidate, *args):
        a2_calls.append(candidate)
        return _provider_pass_result(candidate, *args)

    final_calls = _install_provider_authorities(monkeypatch, a2=counted_a2)
    request, anchor, profile = _provider_request()
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 100.0 if calls == clock_index else 0.0

    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        PlanningDeadlineV2(0.0, 100.0, clock),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "planning_deadline_expired"
    assert outcome.evidence.stage == expected_stage
    assert len(a2_calls) == expected_a2_calls
    assert len(final_calls) == expected_final_calls
    assert calls == clock_index


def test_legged_provider_fingerprint_is_repeated_and_pythonhashseed_stable() -> None:
    script = r'''
import json
import runpy

g = runpy.run_path("tests/test_v2_legged_provider.py")
lm = g["legged_module"]
vm = g["validation_module"]
lm._TRUSTED_VALIDATE_LEGGED_STEP_L2_V2 = g["_provider_pass_result"]
queue_events = []
real_queue = lm.StableSearchQueueV2

class TracingQueue(real_queue):
    def extend(self, entries):
        queue_events.extend(
            ["push", g["_queue_entry_token_v2"](entry)] for entry in entries
        )
        return super().extend(entries)

    def pop_anchor(self):
        entry = super().pop_anchor()
        queue_events.append(["pop", g["_queue_entry_token_v2"](entry)])
        return entry

lm.StableSearchQueueV2 = TracingQueue

def final(route, *_args):
    digest = vm._TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2(route)
    return vm._legged_route_result_v2(
        "legged_route_l2_valid",
        checked_cell_count=3,
        minimum_support_margin_m=0.08,
        validated_route_hash=digest,
    )

vm._TRUSTED_VALIDATE_LEGGED_ROUTE_L2_V2 = final
request, anchor, profile = g["_provider_request"]()
outcome = g["LeggedPrimitiveProviderV2"](profile).plan(
    request,
    anchor,
    g["_route_deadline"](),
)
primitive_tokens = []
candidate_keys = []
for primitive in outcome.route.primitives:
    primitive_tokens.append(
        vm._TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2(
            g["TypedRouteV2"](
                g["PlatformKindV2"].LEGGED,
                (primitive,),
                primitive.energy_cost,
            )
        )
    )
    candidate_keys.append(
        [
            list(g["legged_state_key_v2"](primitive.start_legged_state)),
            list(g["legged_state_key_v2"](primitive.end_legged_state)),
        ]
    )
telemetry = outcome.search_telemetry
print(json.dumps({
    "route_digest": vm._TRUSTED_LEGGED_ROUTE_DIGEST_AUTHORITY_V2(outcome.route),
    "primitive_tokens": primitive_tokens,
    "candidate_keys": candidate_keys,
    "queue_events": queue_events,
    "cost_words": [
        outcome.cost_breakdown.distance_cost.hex(),
        outcome.cost_breakdown.risk_cost.hex(),
        outcome.cost_breakdown.energy_cost.hex(),
        outcome.cost_breakdown.time_cost.hex(),
        outcome.cost_breakdown.total_cost.hex(),
        outcome.route.total_cost.hex(),
    ],
    "decision_telemetry": [
        telemetry.expanded_states,
        telemetry.generated_primitives,
        telemetry.rejected_l2,
        telemetry.timed_out,
        telemetry.accelerator_used,
        telemetry.ackermann_feasible_claimed,
        telemetry.termination_reason,
    ],
}, sort_keys=True, separators=(",", ":")))
'''
    outputs: list[str] = []
    for seed in ("1", "999"):
        for _ in range(2):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            env["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (os.path.join(os.getcwd(), "src"), env.get("PYTHONPATH")),
                )
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.getcwd(),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(completed.stdout.strip())
    assert len(set(outputs)) == 1
    fingerprint = json.loads(outputs[0])
    assert fingerprint["decision_telemetry"] == [1, 13, 12, False, False, False, "legged_route_l2_valid"]
    assert len(fingerprint["primitive_tokens"]) == 1
    assert len(fingerprint["candidate_keys"]) == 1
    assert [
        [event[0], event[1][0], event[1][1]]
        for event in fingerprint["queue_events"]
    ] == [
        ["push", "legged-node-00000000000000000000", "start"],
        ["pop", "legged-node-00000000000000000000", "start"],
        ["push", "legged-node-00000000000000000002", "0:01"],
        ["pop", "legged-node-00000000000000000002", "0:01"],
    ]
    for _, token in fingerprint["queue_events"]:
        assert token[4] == 0.0.hex()
        assert token[5] == []
        assert token[6] == [
            token[3],
            token[3],
            token[2],
            token[1],
            token[0],
        ]


def test_legged_provider_goal_is_accepted_only_when_cheapest_goal_is_popped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    goal_body = PoseStateV2(-0.5, 0.0, 0.0)
    direct = _graph_step_v2(
        start,
        target=WorldPoint(0.0, 0.25),
        end_body=goal_body,
        lift_body=PoseStateV2(10.0, 0.0, 0.0),
    )
    first = _graph_step_v2(
        start,
        target=WorldPoint(0.10, 0.25),
        end_body=PoseStateV2(-0.10, 0.10, 0.0),
    )
    second = _graph_step_v2(
        first.end_legged_state,
        target=first.end_legged_state.foot_contacts[1].foothold,
        end_body=goal_body,
        lift_body=PoseStateV2(-0.30, 0.05, 0.0),
    )
    _install_graph_v2(
        monkeypatch,
        (
            (start, _PROVIDER_OFFSETS[0], direct),
            (start, _PROVIDER_OFFSETS[1], first),
            (first.end_legged_state, _PROVIDER_OFFSETS[0], second),
        ),
    )
    events = _install_queue_trace_v2(monkeypatch)
    request, anchor, profile = _provider_request(goal=goal_body)
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningSuccessV2
    assert outcome.route.primitives == (first, second)
    direct_cost = fsum((0.5 * direct.energy_cost, 0.5 * direct.duration_s))
    cheap_cost = fsum(
        component
        for primitive in (first, second)
        for component in (0.5 * primitive.energy_cost, 0.5 * primitive.duration_s)
    )
    assert cheap_cost < direct_cost
    assert outcome.route.total_cost.hex() == cheap_cost.hex()
    pushed = [token for action, token in events if action == "push"]
    popped = [token for action, token in events if action == "pop"]
    assert [token[0] for token in pushed] == [
        "legged-node-00000000000000000000",
        "legged-node-00000000000000000001",
        "legged-node-00000000000000000002",
        "legged-node-00000000000000000014",
    ]
    assert [token[0] for token in popped] == [
        "legged-node-00000000000000000000",
        "legged-node-00000000000000000002",
        "legged-node-00000000000000000014",
    ]
    expected_push_tokens = (
        (
            "legged-node-00000000000000000000",
            "start",
            legged_state_key_v2(start),
            0.0.hex(),
        ),
        (
            "legged-node-00000000000000000001",
            "0:00",
            legged_state_key_v2(direct.end_legged_state),
            direct_cost.hex(),
        ),
        (
            "legged-node-00000000000000000002",
            "0:01",
            legged_state_key_v2(first.end_legged_state),
            fsum((0.5 * first.energy_cost, 0.5)).hex(),
        ),
        (
            "legged-node-00000000000000000014",
            "1:00",
            legged_state_key_v2(second.end_legged_state),
            cheap_cost.hex(),
        ),
    )
    assert tuple(token[:4] for token in pushed) == expected_push_tokens
    for token in (*pushed, *popped):
        assert token[4] == 0.0.hex()
        assert token[5] == ()
        assert token[6] == (
            token[3],
            token[3],
            token[2],
            token[1],
            token[0],
        )
    assert outcome.search_telemetry.expanded_states == 2
    assert outcome.search_telemetry.generated_primitives == 26
    assert outcome.search_telemetry.rejected_l2 == 23


def test_legged_provider_lineage_survives_later_closed_state_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, edges, primitives = _lineage_graph_v2()
    (
        expensive_s,
        branch_1,
        branch_2,
        branch_3,
        branch_4,
        improved_s,
        child_c,
    ) = primitives
    _install_graph_v2(monkeypatch, edges)
    events = _install_scripted_queue_v2(
        monkeypatch,
        (
            "legged-node-00000000000000000000",
            "legged-node-00000000000000000001",
            "legged-node-00000000000000000002",
            "legged-node-00000000000000000027",
            "legged-node-00000000000000000040",
            "legged-node-00000000000000000053",
            "legged-node-00000000000000000066",
            "legged-node-00000000000000000014",
        ),
    )
    request, anchor, profile = _provider_request(goal=child_c.end_state)
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningSuccessV2
    assert outcome.route.primitives == (expensive_s, child_c)
    assert outcome.route.primitives[0].end_legged_state == (
        outcome.route.primitives[1].start_legged_state
    )
    assert outcome.route.primitives != (
        branch_1,
        branch_2,
        branch_3,
        branch_4,
        improved_s,
        child_c,
    )
    popped = [token for action, token in events if action == "pop"]
    assert [token[0] for token in popped] == [
        "legged-node-00000000000000000000",
        "legged-node-00000000000000000001",
        "legged-node-00000000000000000002",
        "legged-node-00000000000000000027",
        "legged-node-00000000000000000040",
        "legged-node-00000000000000000053",
        "legged-node-00000000000000000066",
        "legged-node-00000000000000000014",
    ]
    expensive_entry = next(token for token in popped if token[0].endswith("01"))
    improved_entry = next(token for token in popped if token[0].endswith("66"))
    assert expensive_entry[2] == improved_entry[2]
    assert float.fromhex(improved_entry[3]) < float.fromhex(expensive_entry[3])
    assert outcome.search_telemetry.expanded_states == 6
    assert outcome.search_telemetry.generated_primitives == 78
    assert outcome.search_telemetry.rejected_l2 == 71


@pytest.mark.parametrize(
    ("max_expanded_states", "expected_type", "expected_reason"),
    [
        (6, PlanningSuccessV2, "legged_route_l2_valid"),
        (5, PlanningFailureV2, "legged_expansion_budget_exhausted"),
    ],
)
def test_legged_provider_runtime_expansion_budget_is_exact_n_vs_n_plus_one(
    monkeypatch: pytest.MonkeyPatch,
    max_expanded_states: int,
    expected_type: type,
    expected_reason: str,
) -> None:
    _, edges, primitives = _lineage_graph_v2()
    child_c = primitives[-1]
    _install_graph_v2(monkeypatch, edges)
    order = (
        "legged-node-00000000000000000000",
        "legged-node-00000000000000000001",
        "legged-node-00000000000000000002",
        "legged-node-00000000000000000027",
        "legged-node-00000000000000000040",
        "legged-node-00000000000000000053",
        "legged-node-00000000000000000066",
        "legged-node-00000000000000000014",
    )
    events = _install_scripted_queue_v2(monkeypatch, order)
    request, anchor, profile = _provider_request(
        goal=child_c.end_state,
        budget=ResourceBudgetV2(max_expanded_states=max_expanded_states),
    )
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is expected_type
    assert outcome.search_telemetry.termination_reason == expected_reason
    if max_expanded_states == 6:
        assert outcome.search_telemetry.expanded_states == 6
        assert [
            token[0] for action, token in events if action == "pop"
        ][-2:] == [
            "legged-node-00000000000000000066",
            "legged-node-00000000000000000014",
        ]
    else:
        assert outcome.evidence.stage == "search"
        assert outcome.evidence.details == (
            ("attempted_expanded_states", 6),
            ("max_expanded_states", 5),
        )
        assert outcome.search_telemetry.expanded_states == 5
        assert outcome.search_telemetry.generated_primitives == 65


@pytest.mark.parametrize(
    ("max_memory_bytes", "attempted", "retained", "expanded", "generated"),
    [
        (1024, 3, 2, 1, 2),
        (1536, 4, 3, 2, 14),
    ],
)
def test_legged_provider_memory_retains_improved_and_stale_records(
    monkeypatch: pytest.MonkeyPatch,
    max_memory_bytes: int,
    attempted: int,
    retained: int,
    expanded: int,
    generated: int,
) -> None:
    start = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    better = _graph_step_v2(
        start,
        target=WorldPoint(0.10, 0.25),
        end_body=PoseStateV2(-0.20, 0.0, 0.0),
        lift_body=PoseStateV2(-0.10, 0.0, 0.0),
    )
    worse = _graph_step_v2(
        start,
        target=better.target_foothold,
        end_body=better.end_state,
        lift_body=PoseStateV2(10.0, 0.0, 0.0),
    )
    assert worse.end_legged_state == better.end_legged_state
    assert worse.energy_cost > better.energy_cost
    child = _graph_step_v2(
        better.end_legged_state,
        target=better.end_legged_state.foot_contacts[3].foothold,
        end_body=PoseStateV2(-0.30, 0.0, 0.0),
        lift_body=PoseStateV2(-0.25, 0.0, 0.0),
    )
    _install_graph_v2(
        monkeypatch,
        (
            (start, _PROVIDER_OFFSETS[0], worse),
            (start, _PROVIDER_OFFSETS[1], better),
            (better.end_legged_state, _PROVIDER_OFFSETS[0], child),
        ),
    )
    request, anchor, profile = _provider_request(
        goal=PoseStateV2(2.0, 2.0, 0.0),
        budget=ResourceBudgetV2(max_memory_bytes=max_memory_bytes),
    )
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "legged_search_memory_budget_exceeded"
    assert outcome.evidence.stage == "search"
    assert outcome.evidence.details == (
        ("accounting_id", "legged_search_fixed_record_512b/v1"),
        ("attempted_record_count", attempted),
        ("max_memory_bytes", max_memory_bytes),
        ("record_bytes", 512),
        ("retained_record_count", retained),
    )
    assert outcome.search_telemetry.expanded_states == expanded
    assert outcome.search_telemetry.generated_primitives == generated


def test_legged_provider_observation_keeps_adjacent_duplicate_body_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    first = _graph_step_v2(
        start,
        target=WorldPoint(0.10, 0.25),
        end_body=PoseStateV2(-0.20, 0.0, 0.0),
        lift_body=PoseStateV2(-0.10, 0.0, 0.0),
    )
    second = _graph_step_v2(
        first.end_legged_state,
        target=first.end_legged_state.foot_contacts[3].foothold,
        end_body=PoseStateV2(-0.40, 0.0, 0.0),
        lift_body=first.end_state,
    )
    _install_graph_v2(
        monkeypatch,
        (
            (start, _PROVIDER_OFFSETS[0], first),
            (first.end_legged_state, _PROVIDER_OFFSETS[0], second),
        ),
    )
    request, anchor, profile = _provider_request(goal=second.end_state)
    outcome = LeggedPrimitiveProviderV2(profile).plan(
        request,
        anchor,
        _route_deadline(),
    )
    assert type(outcome) is PlanningSuccessV2
    samples = outcome.observation_projection.sample_states
    assert len(samples) == 5
    assert samples == (
        first.start_state,
        first.lift_body_state,
        first.end_state,
        second.lift_body_state,
        second.end_state,
    )
    assert samples[2] == samples[3]


def test_provider_candidate_reuses_9a1_resource_helper_exactly_once_for_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))
    real_resource_values = legged_module._resource_values_v2
    calls: list[tuple[tuple[object, ...], tuple[float, float, float]]] = []

    def counted(*args):
        result = real_resource_values(*args)
        calls.append((args, result))
        return result

    monkeypatch.setattr(legged_module, "_resource_values_v2", counted)
    primitive, candidate = legged_module._provider_candidate_v2(
        start,
        0.0,
        (-0.25, 0.0),
    )
    assert type(primitive) is LeggedStepPrimitiveV2
    assert type(candidate) is LeggedStepCandidateV2
    assert len(calls) == 3
    assert all(call_args == calls[0][0] for call_args, _ in calls)
    foot, distance, energy = calls[0][1]
    assert primitive.foot_travel_m.hex() == foot.hex()
    assert primitive.distance_m.hex() == distance.hex()
    assert primitive.energy_cost.hex() == energy.hex()


def test_provider_candidate_resource_helper_ordinary_fault_is_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))

    def faulty(*_args):
        raise RuntimeError("ordinary resource helper fault")

    monkeypatch.setattr(legged_module, "_resource_values_v2", faulty)
    with pytest.raises(ValueError):
        legged_module._provider_candidate_v2(start, 0.0, (-0.25, 0.0))


@pytest.mark.parametrize("critical", [KeyboardInterrupt, SystemExit, MemoryError])
def test_provider_candidate_resource_helper_critical_fault_propagates(
    monkeypatch: pytest.MonkeyPatch,
    critical,
) -> None:
    start = nominal_legged_search_state_v2(PoseStateV2(0.0, 0.0, 0.0))

    def faulty(*_args):
        raise critical("critical resource helper fault")

    monkeypatch.setattr(legged_module, "_resource_values_v2", faulty)
    with pytest.raises(critical):
        legged_module._provider_candidate_v2(start, 0.0, (-0.25, 0.0))
