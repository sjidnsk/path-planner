from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import nextafter, pi

import numpy as np
import pytest

import path_planner.v2 as v2
import path_planner.v2.oracles as oracle_exports
import path_planner.v2.oracles.legged as legged_module
from path_planner.core import Cell, WorldPoint
from path_planner.v2.contracts import (
    PlatformKindV2,
    PoseStateV2,
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
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    SafetyQueryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


def _profile() -> LeggedProfileV2:
    return LeggedProfileV2(
        profile=PlatformProfileV2(
            profile_id="legged-static-crawl/v1",
            platform_kind=PlatformKindV2.LEGGED,
            capability_revision="simulation_proxy_static_crawl/v1",
            simulation_proxy=True,
            max_traversable_slope_deg=30.0,
        )
    )


def _anchor(
    *,
    origin: tuple[float, float] = (-5.0, -5.0),
    unknown: tuple[Cell, ...] = (),
    hard: tuple[Cell, ...] = (),
    not_traversable: tuple[Cell, ...] = (),
    slopes: tuple[tuple[Cell, float], ...] = (),
    elevations: tuple[tuple[Cell, float], ...] = (),
) -> FineSafetyAnchorV2:
    shape = (20, 20)
    observed = np.ones(shape, dtype=bool)
    traversable = np.ones(shape, dtype=bool)
    hard_mask = np.zeros(shape, dtype=bool)
    slope = np.zeros(shape, dtype=np.float64)
    elevation = np.zeros(shape, dtype=np.float64)
    for cell in unknown:
        observed[cell.y, cell.x] = False
    for cell in hard:
        hard_mask[cell.y, cell.x] = True
        traversable[cell.y, cell.x] = False
    for cell in not_traversable:
        traversable[cell.y, cell.x] = False
    for cell, value in slopes:
        slope[cell.y, cell.x] = value
    for cell, value in elevations:
        elevation[cell.y, cell.x] = value
    snapshot = TerrainSnapshotV2(
        geometry=FineGridGeometryV2(
            width=20,
            height=20,
            origin=origin,
            frame_id="moon",
        ),
        elevation_m=elevation,
        slope_deg=slope,
        traversable_mask=traversable,
        hard_obstacle_mask=hard_mask,
        observed_mask=observed,
        confidence=np.ones(shape, dtype=np.float64),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="legged-oracle-fixture",
            source_hash="legged-oracle-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )
    return FineSafetyAnchorV2(snapshot)


def _contacts() -> tuple[LeggedFootContactV2, ...]:
    return (
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, WorldPoint(0.35, 0.25)),
        LeggedFootContactV2(LegIdV2.FRONT_RIGHT, WorldPoint(0.35, -0.25)),
        LeggedFootContactV2(LegIdV2.REAR_LEFT, WorldPoint(-0.35, 0.25)),
        LeggedFootContactV2(LegIdV2.REAR_RIGHT, WorldPoint(-0.35, -0.25)),
    )


def _candidate(**overrides: object) -> LeggedStepCandidateV2:
    values: dict[str, object] = {
        "start_body_state": PoseStateV2(0.0, 0.0, 0.0),
        "lift_body_state": PoseStateV2(0.0, -0.10, 0.0),
        "end_body_state": PoseStateV2(0.0, -0.10, 0.0),
        "foot_contacts": _contacts(),
        "moving_leg": LegIdV2.FRONT_LEFT,
        "sequence_phase": 0,
        "target_foothold": WorldPoint(0.60, 0.25),
    }
    values.update(overrides)
    return LeggedStepCandidateV2(**values)


def _deadline(now: float = 0.0, cutoff: float = 100.0) -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, cutoff, lambda: now)


def test_public_legged_surface_and_fixed_orders_are_exact() -> None:
    assert LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2 == (
        "path-planner-v2-legged-static-stability/v1"
    )
    assert LEGGED_FOOT_STORAGE_ORDER_V2 == (
        LegIdV2.FRONT_LEFT,
        LegIdV2.FRONT_RIGHT,
        LegIdV2.REAR_LEFT,
        LegIdV2.REAR_RIGHT,
    )
    assert LEGGED_CRAWL_SEQUENCE_V2 == (
        LegIdV2.FRONT_LEFT,
        LegIdV2.REAR_RIGHT,
        LegIdV2.FRONT_RIGHT,
        LegIdV2.REAR_LEFT,
    )
    for name in (
        "LegIdV2",
        "LeggedFootContactV2",
        "LeggedStepCandidateV2",
        "LeggedValidationResultV2",
        "validate_legged_step_l2",
    ):
        assert getattr(v2, name) is getattr(oracle_exports, name)


def test_contact_and_candidate_are_frozen_and_canonicalize_signed_zero() -> None:
    contact = LeggedFootContactV2(
        LegIdV2.FRONT_LEFT,
        WorldPoint(-0.0, -0.0),
    )
    assert contact.foothold == WorldPoint(0.0, 0.0)
    assert np.signbit(contact.foothold.x) is np.False_
    candidate = _candidate(
        start_body_state=PoseStateV2(-0.0, -0.0, -0.0),
    )
    assert candidate.start_body_state == PoseStateV2(0.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        contact.foothold = WorldPoint(1.0, 1.0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        candidate.sequence_phase = 1  # type: ignore[misc]


@pytest.mark.parametrize("phase", [True, -1, 4, 1.0])
def test_candidate_rejects_non_exact_or_out_of_range_phase(phase: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _candidate(sequence_phase=phase)


def test_candidate_rejects_contact_aliases_duplicates_and_wrong_order() -> None:
    with pytest.raises(TypeError):
        _candidate(foot_contacts=list(_contacts()))
    with pytest.raises(ValueError):
        _candidate(foot_contacts=_contacts()[:-1])
    duplicate = (*_contacts()[:3], _contacts()[2])
    with pytest.raises(ValueError):
        _candidate(foot_contacts=duplicate)
    with pytest.raises(ValueError):
        _candidate(foot_contacts=tuple(reversed(_contacts())))
    with pytest.raises(TypeError):
        LeggedFootContactV2("front_left", WorldPoint(0.0, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, WorldPoint(float("nan"), 0.0))


def test_oracle_reaudits_forged_nested_state_and_rejects_top_level_subclasses() -> None:
    forged = _candidate()
    object.__setattr__(forged.foot_contacts[0], "foothold", WorldPoint(float("nan"), 0.0))
    assert validate_legged_step_l2(
        forged, _anchor(), _profile(), _deadline()
    ).reason_code == "legged_step_structure_mismatch"

    class DerivedCandidate(LeggedStepCandidateV2):
        pass

    values = _candidate()
    derived = DerivedCandidate(
        values.start_body_state,
        values.lift_body_state,
        values.end_body_state,
        values.foot_contacts,
        values.moving_leg,
        values.sequence_phase,
        values.target_foothold,
    )
    with pytest.raises(TypeError):
        validate_legged_step_l2(derived, _anchor(), _profile(), _deadline())


def test_oracle_reaudit_rejects_type_and_signed_zero_contract_drift() -> None:
    forged_state = _candidate()
    object.__setattr__(forged_state.start_body_state, "x_m", 0)
    assert validate_legged_step_l2(
        forged_state, _anchor(), _profile(), _deadline()
    ).reason_code == "legged_step_structure_mismatch"

    forged_zero = _candidate()
    object.__setattr__(forged_zero.start_body_state, "x_m", -0.0)
    assert validate_legged_step_l2(
        forged_zero, _anchor(), _profile(), _deadline()
    ).reason_code == "legged_step_structure_mismatch"

    forged_contact = _candidate()
    object.__setattr__(forged_contact.foot_contacts[0].foothold, "x", np.float64(0.35))
    assert validate_legged_step_l2(
        forged_contact, _anchor(), _profile(), _deadline()
    ).reason_code == "legged_step_structure_mismatch"

    forged_profile = _profile()
    object.__setattr__(forged_profile, "body_length_m", np.float64(0.60))
    assert validate_legged_step_l2(
        _candidate(), _anchor(), forged_profile, _deadline()
    ).reason_code == "legged_profile_contract_mismatch"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_traversable_slope_deg", 30),
        ("goal_heading_tolerance_rad", 0),
        ("goal_position_tolerance_m", False),
        ("profile_id", ""),
        ("schema_version", "wrong-schema/v1"),
        ("platform_kind", "legged"),
        ("simulation_proxy", 1),
        ("max_traversable_slope_deg", float("nan")),
    ],
)
def test_profile_reaudit_rejects_raw_nested_base_contract_drift(
    field: str,
    value: object,
) -> None:
    profile = _profile()
    object.__setattr__(profile.profile, field, value)
    result = validate_legged_step_l2(
        _candidate(), _anchor(), profile, _deadline()
    )
    assert result.reason_code == "legged_profile_contract_mismatch"


def test_profile_reaudit_rejects_raw_nested_string_subclass() -> None:
    class DerivedStr(str):
        pass

    profile = _profile()
    object.__setattr__(profile.profile, "profile_id", DerivedStr("legged-static-crawl/v1"))
    assert validate_legged_step_l2(
        _candidate(), _anchor(), profile, _deadline()
    ).reason_code == "legged_profile_contract_mismatch"


def test_happy_path_returns_authoritative_l2_evidence() -> None:
    result = validate_legged_step_l2(
        _candidate(),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert type(result) is LeggedValidationResultV2
    assert result.evidence.validator_id == LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2
    assert result.evidence.level is ValidationLevelV2.L2
    assert result.evidence.passed is True
    assert result.evidence.checks == ("legged_step_l2_valid",)
    assert result.reason_code == "legged_step_l2_valid"
    assert result.timed_out is False
    assert result.failed_cell is None
    assert result.failed_leg is None
    assert result.checked_cell_count > 0
    assert result.minimum_support_margin_m is not None
    assert result.minimum_support_margin_m >= 0.05


@pytest.mark.parametrize(
    ("anchor", "reason", "failed_cell"),
    [
        (_anchor(unknown=(Cell(11, 10),)), "legged_foothold_unknown", Cell(11, 10)),
        (
            _anchor(hard=(Cell(11, 10),)),
            "legged_foothold_hard_obstacle",
            Cell(11, 10),
        ),
        (
            _anchor(not_traversable=(Cell(11, 10),)),
            "legged_foothold_not_traversable",
            Cell(11, 10),
        ),
        (
            _anchor(slopes=((Cell(11, 10), 25.000000000000004),)),
            "legged_foothold_slope_exceeded",
            Cell(11, 10),
        ),
    ],
)
def test_foothold_terrain_failures_have_stable_leg_and_cell(
    anchor: FineSafetyAnchorV2,
    reason: str,
    failed_cell: Cell,
) -> None:
    result = validate_legged_step_l2(_candidate(), anchor, _profile(), _deadline())
    assert result.reason_code == reason
    assert result.failed_cell == failed_cell
    assert result.failed_leg is LegIdV2.FRONT_LEFT
    assert result.checked_cell_count > 0
    assert result.minimum_support_margin_m is None


def test_grid_length_height_support_and_sequence_failures_are_typed() -> None:
    grid = validate_legged_step_l2(
        _candidate(target_foothold=WorldPoint(0.61, 0.25)),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert (grid.reason_code, grid.failed_leg) == (
        "legged_foothold_grid_misaligned",
        LegIdV2.FRONT_LEFT,
    )

    length = validate_legged_step_l2(
        _candidate(target_foothold=WorldPoint(1.10, 0.25)),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert (length.reason_code, length.failed_leg) == (
        "legged_step_length_exceeded",
        LegIdV2.FRONT_LEFT,
    )

    height = validate_legged_step_l2(
        _candidate(),
        _anchor(elevations=((Cell(11, 10), 0.25000000000000006),)),
        _profile(),
        _deadline(),
    )
    assert (height.reason_code, height.failed_leg) == (
        "legged_step_height_exceeded",
        LegIdV2.FRONT_LEFT,
    )

    support = validate_legged_step_l2(
        _candidate(
            lift_body_state=PoseStateV2(0.0, 0.0, 0.0),
            end_body_state=PoseStateV2(0.0, 0.0, 0.0),
        ),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert support.reason_code == "legged_support_margin_insufficient"
    assert support.failed_cell is None
    assert support.failed_leg is None
    assert support.minimum_support_margin_m is None

    sequence = validate_legged_step_l2(
        _candidate(sequence_phase=1),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert sequence.reason_code == "legged_foot_sequence_invalid"
    assert sequence.failed_leg is LegIdV2.FRONT_LEFT
    assert sequence.minimum_support_margin_m is not None


def _wide_contacts() -> tuple[LeggedFootContactV2, ...]:
    return (
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, WorldPoint(0.75, 0.75)),
        LeggedFootContactV2(LegIdV2.FRONT_RIGHT, WorldPoint(0.75, -0.75)),
        LeggedFootContactV2(LegIdV2.REAR_LEFT, WorldPoint(-0.75, 0.75)),
        LeggedFootContactV2(LegIdV2.REAR_RIGHT, WorldPoint(-0.75, -0.75)),
    )


def _boundary_contacts(point: WorldPoint) -> tuple[LeggedFootContactV2, ...]:
    return (
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, point),
        LeggedFootContactV2(LegIdV2.FRONT_RIGHT, WorldPoint(2.25, -2.25)),
        LeggedFootContactV2(LegIdV2.REAR_LEFT, WorldPoint(-2.25, 2.25)),
        LeggedFootContactV2(LegIdV2.REAR_RIGHT, WorldPoint(-2.25, -2.25)),
    )


@pytest.mark.parametrize(
    ("anchor", "reason"),
    [
        (_anchor(unknown=(Cell(9, 9),)), "legged_body_sweep_unknown"),
        (_anchor(hard=(Cell(9, 9),)), "legged_body_sweep_collision"),
        (
            _anchor(slopes=((Cell(9, 9), 30.000000000000004),)),
            "legged_body_sweep_collision",
        ),
    ],
)
def test_body_sweep_failures_preserve_support_margin(
    anchor: FineSafetyAnchorV2,
    reason: str,
) -> None:
    result = validate_legged_step_l2(
        _candidate(
            foot_contacts=_wide_contacts(),
            lift_body_state=PoseStateV2(0.0, -0.20, 0.0),
            end_body_state=PoseStateV2(0.0, -0.20, 0.0),
            target_foothold=WorldPoint(1.0, 0.75),
        ),
        anchor,
        _profile(),
        _deadline(),
    )
    assert result.reason_code == reason
    assert result.failed_cell == Cell(9, 9)
    assert result.failed_leg is None
    assert result.minimum_support_margin_m is not None


@pytest.mark.parametrize("end_heading", [pi, -pi])
def test_body_sweep_catches_pure_rotation_interior_collision(
    end_heading: float,
) -> None:
    center = WorldPoint(-0.225, -0.17500000000000002)
    obstacle = Cell(9, 8)
    geometry = _anchor().snapshot.geometry
    endpoint_cells = set(
        v2.oriented_rectangle_cells(center, 0.0, 0.60, 0.40, geometry)
    ) | set(
        v2.oriented_rectangle_cells(center, end_heading, 0.60, 0.40, geometry)
    )
    assert obstacle not in endpoint_cells
    result = validate_legged_step_l2(
        _candidate(
            foot_contacts=_wide_contacts(),
            start_body_state=PoseStateV2(center.x, center.y, 0.0),
            lift_body_state=PoseStateV2(center.x, center.y, end_heading),
            end_body_state=PoseStateV2(center.x, center.y, end_heading),
            target_foothold=WorldPoint(1.0, 0.75),
        ),
        _anchor(hard=(obstacle,)),
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "legged_body_sweep_collision"
    assert result.failed_cell == obstacle


@pytest.mark.parametrize(
    ("obstacle", "fraction"),
    [(Cell(9, 6), 0.25), (Cell(10, 6), 0.75)],
)
def test_body_sweep_catches_quarter_interval_thin_obstacle(
    obstacle: Cell,
    fraction: float,
) -> None:
    contacts = (
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, WorldPoint(3.0, 3.0)),
        LeggedFootContactV2(LegIdV2.FRONT_RIGHT, WorldPoint(3.0, -3.0)),
        LeggedFootContactV2(LegIdV2.REAR_LEFT, WorldPoint(-3.0, 3.0)),
        LeggedFootContactV2(LegIdV2.REAR_RIGHT, WorldPoint(-3.0, -3.0)),
    )
    start = PoseStateV2(-1.0, -1.5, 0.0)
    end = PoseStateV2(1.0, -1.5, 0.0)
    geometry = _anchor().snapshot.geometry
    endpoint_cells = set(
        v2.oriented_rectangle_cells(
            WorldPoint(start.x_m, start.y_m),
            start.heading_rad,
            0.60,
            0.40,
            geometry,
        )
    ) | set(
        v2.oriented_rectangle_cells(
            WorldPoint(end.x_m, end.y_m),
            end.heading_rad,
            0.60,
            0.40,
            geometry,
        )
    )
    interval_pose = WorldPoint(-1.0 + 2.0 * fraction, -1.5)
    assert obstacle not in endpoint_cells
    assert obstacle in set(
        v2.oriented_rectangle_cells(interval_pose, 0.0, 0.60, 0.40, geometry)
    )

    result = validate_legged_step_l2(
        _candidate(
            start_body_state=start,
            lift_body_state=end,
            end_body_state=end,
            foot_contacts=contacts,
            target_foothold=WorldPoint(3.5, 3.0),
        ),
        _anchor(hard=(obstacle,)),
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "legged_body_sweep_collision"
    assert result.failed_cell == obstacle


@pytest.mark.parametrize("end_heading", [pi, -pi])
def test_body_sweep_translation_and_pi_tie_catches_mid_interval_collision(
    end_heading: float,
) -> None:
    start = PoseStateV2(-0.225, -0.175, 0.0)
    end = PoseStateV2(0.025, -0.175, end_heading)
    obstacle = Cell(9, 8)
    geometry = _anchor().snapshot.geometry
    endpoint_cells = set(
        v2.oriented_rectangle_cells(
            WorldPoint(start.x_m, start.y_m),
            start.heading_rad,
            0.60,
            0.40,
            geometry,
        )
    ) | set(
        v2.oriented_rectangle_cells(
            WorldPoint(end.x_m, end.y_m),
            end.heading_rad,
            0.60,
            0.40,
            geometry,
        )
    )
    assert obstacle not in endpoint_cells

    result = validate_legged_step_l2(
        _candidate(
            foot_contacts=_wide_contacts(),
            start_body_state=start,
            lift_body_state=end,
            end_body_state=end,
            target_foothold=WorldPoint(1.0, 0.75),
        ),
        _anchor(hard=(obstacle,)),
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "legged_body_sweep_collision"
    assert result.failed_cell == obstacle


@pytest.mark.parametrize("end_heading", [pi, -pi])
@pytest.mark.parametrize("fraction", [0.25, 0.75])
def test_body_sweep_pi_tie_catches_quarter_interval_oob(
    end_heading: float,
    fraction: float,
) -> None:
    contacts = (
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, WorldPoint(-0.75, 4.1)),
        LeggedFootContactV2(LegIdV2.FRONT_RIGHT, WorldPoint(0.75, 4.1)),
        LeggedFootContactV2(LegIdV2.REAR_LEFT, WorldPoint(-0.75, 4.9)),
        LeggedFootContactV2(LegIdV2.REAR_RIGHT, WorldPoint(0.75, 4.9)),
    )
    start = PoseStateV2(0.0, 4.59, 0.0)
    end = PoseStateV2(0.0, 4.59, end_heading)
    geometry = _anchor().snapshot.geometry
    center = WorldPoint(start.x_m, start.y_m)
    for safe_heading in (start.heading_rad, end_heading / 2.0, end_heading):
        safe_cells = v2.oriented_rectangle_cells(
            center,
            safe_heading,
            0.60,
            0.40,
            geometry,
        )
        assert all(geometry.in_bounds(cell) for cell in safe_cells)
    interval_cells = v2.oriented_rectangle_cells(
        center,
        end_heading * fraction,
        0.60,
        0.40,
        geometry,
    )
    assert any(not geometry.in_bounds(cell) for cell in interval_cells)

    result = validate_legged_step_l2(
        _candidate(
            foot_contacts=contacts,
            start_body_state=start,
            lift_body_state=end,
            end_body_state=end,
            target_foothold=WorldPoint(-0.25, 4.1),
        ),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "legged_body_sweep_unknown"
    assert result.failed_cell is not None
    assert not geometry.in_bounds(result.failed_cell)


def test_body_sweep_oob_is_unknown_and_exact_body_slope_30_passes() -> None:
    edge_contacts = (
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, WorldPoint(-4.4, -4.4)),
        LeggedFootContactV2(LegIdV2.FRONT_RIGHT, WorldPoint(-4.4, -4.9)),
        LeggedFootContactV2(LegIdV2.REAR_LEFT, WorldPoint(-4.9, -4.4)),
        LeggedFootContactV2(LegIdV2.REAR_RIGHT, WorldPoint(-4.9, -4.9)),
    )
    oob = validate_legged_step_l2(
        _candidate(
            foot_contacts=edge_contacts,
            start_body_state=PoseStateV2(-4.65, -4.65, 0.0),
            lift_body_state=PoseStateV2(-4.70, -4.70, 0.0),
            end_body_state=PoseStateV2(-4.70, -4.70, 0.0),
            target_foothold=WorldPoint(-4.15, -4.4),
        ),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert oob.reason_code == "legged_body_sweep_unknown"
    assert oob.failed_cell is not None
    assert oob.failed_cell.x < 0 or oob.failed_cell.y < 0

    exact_slope = validate_legged_step_l2(
        _candidate(
            foot_contacts=_wide_contacts(),
            lift_body_state=PoseStateV2(0.0, -0.20, 0.0),
            end_body_state=PoseStateV2(0.0, -0.20, 0.0),
            target_foothold=WorldPoint(1.0, 0.75),
        ),
        _anchor(slopes=((Cell(9, 9), 30.0),)),
        _profile(),
        _deadline(),
    )
    assert exact_slope.reason_code == "legged_step_l2_valid"


@pytest.mark.parametrize(
    ("moving_leg", "phase", "lift_y"),
    [
        (LegIdV2.FRONT_LEFT, 0, -0.10),
        (LegIdV2.REAR_RIGHT, 1, 0.10),
        (LegIdV2.FRONT_RIGHT, 2, 0.10),
        (LegIdV2.REAR_LEFT, 3, -0.10),
    ],
)
def test_all_four_swing_legs_require_pre_shift_then_pass(
    moving_leg: LegIdV2,
    phase: int,
    lift_y: float,
) -> None:
    source = next(
        contact.foothold for contact in _contacts() if contact.leg_id is moving_leg
    )
    common = {
        "moving_leg": moving_leg,
        "sequence_phase": phase,
        "target_foothold": WorldPoint(source.x + 0.25, source.y),
    }
    no_shift = validate_legged_step_l2(
        _candidate(
            **common,
            lift_body_state=PoseStateV2(0.0, 0.0, 0.0),
            end_body_state=PoseStateV2(0.0, 0.0, 0.0),
        ),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert no_shift.reason_code == "legged_support_margin_insufficient"
    shifted = validate_legged_step_l2(
        _candidate(
            **common,
            lift_body_state=PoseStateV2(0.0, lift_y, 0.0),
            end_body_state=PoseStateV2(0.0, lift_y, 0.0),
        ),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert shifted.reason_code == "legged_step_l2_valid"


def test_deadline_profile_structure_snapshot_and_query_contract_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), _deadline(now=1.0, cutoff=1.0)
    )
    assert expired.reason_code == "planning_deadline_expired"
    assert expired.timed_out is True

    bad_deadline = _deadline()
    object.__setattr__(bad_deadline, "deadline_monotonic_s", 100)
    assert validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), bad_deadline
    ).reason_code == "planning_deadline_contract_mismatch"

    bad_profile = _profile()
    object.__setattr__(bad_profile, "max_step_length_m", 1.0)
    assert validate_legged_step_l2(
        _candidate(), _anchor(), bad_profile, _deadline()
    ).reason_code == "legged_profile_contract_mismatch"

    bad_candidate = _candidate()
    object.__setattr__(bad_candidate, "target_foothold", WorldPoint(float("nan"), 0.0))
    assert validate_legged_step_l2(
        bad_candidate, _anchor(), _profile(), _deadline()
    ).reason_code == "legged_step_structure_mismatch"

    bad_anchor = _anchor()
    object.__setattr__(bad_anchor, "_snapshot_hash", "0" * 64)
    assert validate_legged_step_l2(
        _candidate(), bad_anchor, _profile(), _deadline()
    ).reason_code == "terrain_snapshot_hash_mismatch"

    monkeypatch.setattr(FineSafetyAnchorV2, "query", lambda *_args: object())
    assert validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), _deadline()
    ).reason_code == "terrain_query_contract_mismatch"


def test_query_local_semantics_reject_safe_lies_for_contact_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FineSafetyAnchorV2.query
    contact_anchor = _anchor(unknown=(Cell(11, 10),))

    def lie_about_contact(self, cell, threshold=30.0):
        query = original(self, cell, threshold)
        if cell == Cell(11, 10) and threshold == 25.0:
            return SafetyQueryV2(
                cell=cell,
                passed=True,
                reason_code="terrain_safe",
                slope_deg=0.0,
                confidence=1.0,
                snapshot_hash=query.snapshot_hash,
            )
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", lie_about_contact)
    assert validate_legged_step_l2(
        _candidate(), contact_anchor, _profile(), _deadline()
    ).reason_code == "terrain_query_contract_mismatch"

    body_anchor = _anchor(hard=(Cell(9, 9),))

    def lie_about_body(self, cell, threshold=30.0):
        query = original(self, cell, threshold)
        if cell == Cell(9, 9) and threshold == 30.0:
            return SafetyQueryV2(
                cell=cell,
                passed=True,
                reason_code="terrain_safe",
                slope_deg=0.0,
                confidence=1.0,
                snapshot_hash=query.snapshot_hash,
            )
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", lie_about_body)
    assert validate_legged_step_l2(
        _candidate(
            foot_contacts=_wide_contacts(),
            lift_body_state=PoseStateV2(0.0, -0.20, 0.0),
            end_body_state=PoseStateV2(0.0, -0.20, 0.0),
            target_foothold=WorldPoint(1.0, 0.75),
        ),
        body_anchor,
        _profile(),
        _deadline(),
    ).reason_code == "terrain_query_contract_mismatch"


def test_contact_query_order_deduplicates_pairs_and_preserves_threshold_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Cell, float]] = []
    original = FineSafetyAnchorV2.query

    def record(self, cell, threshold=30.0):
        calls.append((cell, threshold))
        return original(self, cell, threshold)

    monkeypatch.setattr(FineSafetyAnchorV2, "query", record)
    result = validate_legged_step_l2(_candidate(), _anchor(), _profile(), _deadline())
    assert result.reason_code == "legged_step_l2_valid"
    contact_calls = [cell for cell, threshold in calls if threshold == 25.0]
    body_calls = [cell for cell, threshold in calls if threshold == 30.0]
    assert len(contact_calls) == len(set(contact_calls))
    assert body_calls == sorted(set(body_calls), key=lambda cell: (cell.y, cell.x))
    assert result.checked_cell_count == len(calls)
    assert any(cell in set(contact_calls) for cell in body_calls)


def test_fine_boundary_contact_queries_all_closed_square_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FineSafetyAnchorV2.query
    calls: list[tuple[Cell, float]] = []

    def record(self, cell, threshold=30.0):
        calls.append((cell, threshold))
        return original(self, cell, threshold)

    contacts = list(_contacts())
    contacts[0] = LeggedFootContactV2(LegIdV2.FRONT_LEFT, WorldPoint(0.5, 0.5))
    monkeypatch.setattr(FineSafetyAnchorV2, "query", record)
    result = validate_legged_step_l2(
        _candidate(
            foot_contacts=tuple(contacts),
            target_foothold=WorldPoint(0.75, 0.5),
        ),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "legged_step_l2_valid"
    contact_cells = {cell for cell, threshold in calls if threshold == 25.0}
    assert {Cell(10, 10), Cell(11, 10), Cell(10, 11), Cell(11, 11)} <= contact_cells


@pytest.mark.parametrize(
    ("point", "expected_cells"),
    [
        (WorldPoint(0.25, 0.25), {Cell(10, 10)}),
        (WorldPoint(0.5, 0.25), {Cell(10, 10), Cell(11, 10)}),
        (
            WorldPoint(0.5, 0.5),
            {Cell(10, 10), Cell(11, 10), Cell(10, 11), Cell(11, 11)},
        ),
        (
            WorldPoint(nextafter(0.5, 0.0), 0.25),
            {Cell(10, 10), Cell(11, 10)},
        ),
        (
            WorldPoint(nextafter(0.5, 1.0), 0.25),
            {Cell(10, 10), Cell(11, 10)},
        ),
        (WorldPoint(0.5 - 0.5e-12, 0.25), {Cell(10, 10), Cell(11, 10)}),
        (WorldPoint(0.5 + 0.5e-12, 0.25), {Cell(10, 10), Cell(11, 10)}),
        (WorldPoint(0.5 - 2.0e-12, 0.25), {Cell(10, 10)}),
        (WorldPoint(0.5 + 2.0e-12, 0.25), {Cell(11, 10)}),
        (
            WorldPoint(nextafter(0.5, 0.0), nextafter(0.5, 1.0)),
            {Cell(10, 10), Cell(11, 10), Cell(10, 11), Cell(11, 11)},
        ),
        (
            WorldPoint(0.5 - 0.5e-12, 0.5 + 0.5e-12),
            {Cell(10, 10), Cell(11, 10), Cell(10, 11), Cell(11, 11)},
        ),
        (WorldPoint(0.5 - 2.0e-12, 0.5 + 2.0e-12), {Cell(10, 11)}),
    ],
)
def test_public_oracle_contact_cell_boundary_matrix(
    monkeypatch: pytest.MonkeyPatch,
    point: WorldPoint,
    expected_cells: set[Cell],
) -> None:
    original = FineSafetyAnchorV2.query
    calls: list[tuple[Cell, float]] = []

    def record(self, cell, threshold=30.0):
        calls.append((cell, threshold))
        return original(self, cell, threshold)

    monkeypatch.setattr(FineSafetyAnchorV2, "query", record)
    result = validate_legged_step_l2(
        _candidate(
            foot_contacts=_boundary_contacts(point),
            target_foothold=point,
        ),
        _anchor(),
        _profile(),
        _deadline(),
    )
    neighborhood = {
        Cell(x, y)
        for y in range(9, 13)
        for x in range(9, 13)
    }
    observed_cells = {
        cell
        for cell, threshold in calls
        if threshold == 25.0 and cell in neighborhood
    }
    assert result.reason_code == "legged_step_l2_valid"
    assert observed_cells == expected_cells
    assert len(observed_cells) in {1, 2, 4}


def test_exact_closed_boundaries_pass_and_next_values_fail() -> None:
    assert validate_legged_step_l2(
        _candidate(target_foothold=WorldPoint(0.85, 0.25)),
        _anchor(elevations=((Cell(11, 10), 0.25),)),
        _profile(),
        _deadline(),
    ).reason_code == "legged_step_l2_valid"
    assert validate_legged_step_l2(
        _candidate(),
        _anchor(slopes=((Cell(11, 10), 25.0),)),
        _profile(),
        _deadline(),
    ).reason_code == "legged_step_l2_valid"

    support_contacts = (
        LeggedFootContactV2(LegIdV2.FRONT_LEFT, WorldPoint(1.0, 0.0)),
        LeggedFootContactV2(LegIdV2.FRONT_RIGHT, WorldPoint(0.05, -1.0)),
        LeggedFootContactV2(LegIdV2.REAR_LEFT, WorldPoint(0.05, 1.0)),
        LeggedFootContactV2(LegIdV2.REAR_RIGHT, WorldPoint(-1.0, 0.0)),
    )
    exact = _candidate(
        foot_contacts=support_contacts,
        lift_body_state=PoseStateV2(0.0, 0.0, 0.0),
        end_body_state=PoseStateV2(0.0, 0.0, 0.0),
        target_foothold=WorldPoint(1.25, 0.0),
    )
    assert validate_legged_step_l2(
        exact, _anchor(), _profile(), _deadline()
    ).reason_code == "legged_step_l2_valid"
    below_contacts = list(support_contacts)
    below_contacts[1] = LeggedFootContactV2(
        LegIdV2.FRONT_RIGHT,
        WorldPoint(nextafter(0.05, 0.0), -1.0),
    )
    below_contacts[2] = LeggedFootContactV2(
        LegIdV2.REAR_LEFT,
        WorldPoint(nextafter(0.05, 0.0), 1.0),
    )
    below = replace(exact, foot_contacts=tuple(below_contacts))
    assert validate_legged_step_l2(
        below, _anchor(), _profile(), _deadline()
    ).reason_code == "legged_support_margin_insufficient"


@pytest.mark.parametrize("heading", [pi / 2.0, 5.0 * pi / 2.0, -3.0 * pi / 2.0])
def test_local_foothold_lattice_is_rotation_and_wrap_consistent(heading: float) -> None:
    result = validate_legged_step_l2(
        _candidate(
            start_body_state=PoseStateV2(0.0, 0.0, heading),
            lift_body_state=PoseStateV2(0.0, -0.10, heading),
            end_body_state=PoseStateV2(0.0, -0.10, heading),
            target_foothold=WorldPoint(0.35, 0.50),
        ),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "legged_step_l2_valid"


@pytest.mark.parametrize(
    ("offset", "reason"),
    [
        (0.5e-12, "legged_step_l2_valid"),
        (2.0e-12, "legged_foothold_grid_misaligned"),
    ],
)
def test_local_foothold_lattice_uses_fixed_absolute_tolerance(
    offset: float,
    reason: str,
) -> None:
    result = validate_legged_step_l2(
        _candidate(target_foothold=WorldPoint(0.60 + offset, 0.25)),
        _anchor(),
        _profile(),
        _deadline(),
    )
    assert result.reason_code == reason


def test_failure_precedence_is_independent_of_terrain_iteration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixed = _anchor(
        unknown=(Cell(11, 10),),
        hard=(Cell(10, 10),),
    )
    result = validate_legged_step_l2(
        _candidate(target_foothold=WorldPoint(0.61, 0.25)),
        mixed,
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "legged_foothold_grid_misaligned"

    original = FineSafetyAnchorV2.query

    def malformed(self, cell, threshold=30.0):
        if threshold == 25.0:
            return object()
        return original(self, cell, threshold)

    monkeypatch.setattr(FineSafetyAnchorV2, "query", malformed)
    result = validate_legged_step_l2(
        _candidate(target_foothold=WorldPoint(0.61, 0.25)),
        mixed,
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "terrain_query_contract_mismatch"


def test_same_reason_foothold_and_body_ties_are_stable() -> None:
    foothold_anchor = _anchor(
        hard=(Cell(11, 10), Cell(10, 10), Cell(10, 9)),
    )
    for _ in range(3):
        result = validate_legged_step_l2(
            _candidate(), foothold_anchor, _profile(), _deadline()
        )
        assert result.reason_code == "legged_foothold_hard_obstacle"
        assert result.failed_leg is LegIdV2.FRONT_LEFT
        assert result.failed_cell == Cell(10, 10)

    body_anchor = _anchor(
        hard=(Cell(10, 9), Cell(9, 10), Cell(9, 9)),
    )
    body_candidate = _candidate(
        foot_contacts=_wide_contacts(),
        lift_body_state=PoseStateV2(0.0, -0.20, 0.0),
        end_body_state=PoseStateV2(0.0, -0.20, 0.0),
        target_foothold=WorldPoint(1.0, 0.75),
    )
    for _ in range(3):
        result = validate_legged_step_l2(
            body_candidate, body_anchor, _profile(), _deadline()
        )
        assert result.reason_code == "legged_body_sweep_collision"
        assert result.failed_leg is None
        assert result.failed_cell == Cell(9, 9)


def test_valid_but_foreign_query_hash_precedes_local_semantic_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FineSafetyAnchorV2.query

    def foreign(self, cell, threshold=30.0):
        query = original(self, cell, threshold)
        if threshold == 25.0:
            return SafetyQueryV2(
                cell=cell,
                passed=True,
                reason_code="terrain_safe",
                slope_deg=1.0,
                confidence=1.0,
                snapshot_hash="0" * 64,
            )
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", foreign)
    result = validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), _deadline()
    )
    assert result.reason_code == "terrain_snapshot_hash_mismatch"


def test_deadline_uses_pinned_cutoff_and_propagates_critical_base_exceptions() -> None:
    clock_calls = 0
    holder: dict[str, PlanningDeadlineV2] = {}

    def restoring_clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        deadline = holder["deadline"]
        object.__setattr__(deadline, "deadline_monotonic_s", 0.0)
        object.__setattr__(deadline, "deadline_monotonic_s", 100.0)
        return 0.0

    deadline = PlanningDeadlineV2(0.0, 100.0, restoring_clock)
    holder["deadline"] = deadline
    assert validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), deadline
    ).reason_code == "legged_step_l2_valid"
    assert clock_calls > 1

    def interrupted() -> float:
        raise MemoryError("critical")

    with pytest.raises(MemoryError, match="critical"):
        validate_legged_step_l2(
            _candidate(),
            _anchor(),
            _profile(),
            PlanningDeadlineV2(0.0, 100.0, interrupted),
        )


def test_query_identity_drift_is_contract_mismatch_and_memory_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FineSafetyAnchorV2.query
    replacement = _anchor().snapshot.slope_deg

    def drift(self, cell, threshold=30.0):
        query = original(self, cell, threshold)
        object.__setattr__(self.snapshot, "slope_deg", replacement)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", drift)
    assert validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), _deadline()
    ).reason_code == "terrain_query_contract_mismatch"

    def exhausted(*_args, **_kwargs):
        raise MemoryError("query critical")

    monkeypatch.setattr(FineSafetyAnchorV2, "query", exhausted)
    with pytest.raises(MemoryError, match="query critical"):
        validate_legged_step_l2(
            _candidate(), _anchor(), _profile(), _deadline()
        )


def test_query_detects_signed_zero_geometry_metadata_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FineSafetyAnchorV2.query
    contacts = tuple(
        LeggedFootContactV2(
            contact.leg_id,
            WorldPoint(contact.foothold.x + 2.0, contact.foothold.y + 2.0),
        )
        for contact in _contacts()
    )

    def drift(self, cell, threshold=30.0):
        query = original(self, cell, threshold)
        object.__setattr__(self.snapshot.geometry, "origin", (-0.0, 0.0))
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", drift)
    result = validate_legged_step_l2(
        _candidate(
            start_body_state=PoseStateV2(2.0, 2.0, 0.0),
            lift_body_state=PoseStateV2(2.0, 1.90, 0.0),
            end_body_state=PoseStateV2(2.0, 1.90, 0.0),
            foot_contacts=contacts,
            target_foothold=WorldPoint(2.60, 2.25),
        ),
        _anchor(origin=(0.0, 0.0)),
        _profile(),
        _deadline(),
    )
    assert result.reason_code == "terrain_query_contract_mismatch"


def _one_cell_anchor() -> FineSafetyAnchorV2:
    shape = (1, 1)
    return FineSafetyAnchorV2(
        TerrainSnapshotV2(
            geometry=FineGridGeometryV2(
                width=1,
                height=1,
                origin=(-0.25, -0.25),
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
                source_id="one-cell-fixture",
                source_hash="one-cell-fixture-hash",
                physical_obstacle_cells_written=False,
            ),
        )
    )


def test_equal_by_value_one_cell_layer_replacement_is_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = _one_cell_anchor()
    replacement = _one_cell_anchor().snapshot.slope_deg
    assert replacement is not anchor.snapshot.slope_deg
    original = FineSafetyAnchorV2.query
    changed = False

    def replace_layer(self, cell, threshold=30.0):
        nonlocal changed
        query = original(self, cell, threshold)
        if not changed:
            changed = True
            object.__setattr__(self.snapshot, "slope_deg", replacement)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", replace_layer)
    result = validate_legged_step_l2(
        _candidate(), anchor, _profile(), _deadline()
    )
    assert result.reason_code == "terrain_query_contract_mismatch"


def test_equal_hash_string_replacement_is_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = _one_cell_anchor()
    original_hash = anchor._snapshot_hash
    replacement = ("!" + original_hash)[1:]
    assert type(replacement) is str
    assert replacement == original_hash
    assert replacement is not original_hash
    original = FineSafetyAnchorV2.query
    changed = False

    def replace_hash(self, cell, threshold=30.0):
        nonlocal changed
        query = original(self, cell, threshold)
        if not changed:
            changed = True
            object.__setattr__(self, "_snapshot_hash", replacement)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", replace_hash)
    result = validate_legged_step_l2(
        _candidate(), anchor, _profile(), _deadline()
    )
    assert result.reason_code == "terrain_query_contract_mismatch"


def test_body_and_contact_calculation_use_canonical_geometry_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = _anchor()
    authority_geometry = anchor.snapshot.geometry
    seen: list[FineGridGeometryV2] = []
    original = legged_module.oriented_rectangle_cells

    def record_geometry(center, theta, length, width, geometry):
        seen.append(geometry)
        return original(center, theta, length, width, geometry)

    monkeypatch.setattr(legged_module, "oriented_rectangle_cells", record_geometry)
    result = validate_legged_step_l2(
        _candidate(), anchor, _profile(), _deadline()
    )
    assert result.reason_code == "legged_step_l2_valid"
    assert seen
    assert all(type(geometry) is FineGridGeometryV2 for geometry in seen)
    assert all(geometry is not authority_geometry for geometry in seen)


def test_contact_cells_remain_bound_to_canonical_geometry_during_restored_drift() -> None:
    obstacle = Cell(11, 11)
    candidate = _candidate(
        foot_contacts=_wide_contacts(),
        lift_body_state=PoseStateV2(0.0, -0.20, 0.0),
        end_body_state=PoseStateV2(0.0, -0.20, 0.0),
        target_foothold=WorldPoint(1.25, 0.75),
    )
    baseline = validate_legged_step_l2(
        candidate,
        _anchor(hard=(obstacle,)),
        _profile(),
        _deadline(),
    )
    assert baseline.reason_code == "legged_foothold_hard_obstacle"
    assert baseline.failed_leg is LegIdV2.FRONT_LEFT
    assert baseline.failed_cell == obstacle

    anchor = _anchor(hard=(obstacle,))
    authority_geometry = anchor.snapshot.geometry
    original_origin = authority_geometry.origin
    calls = 0

    def restoring_clock() -> float:
        nonlocal calls
        calls += 1
        if calls == 4:
            object.__setattr__(authority_geometry, "origin", (-4.0, -4.0))
        elif calls == 5:
            object.__setattr__(authority_geometry, "origin", original_origin)
        return 0.0

    result = validate_legged_step_l2(
        candidate,
        anchor,
        _profile(),
        PlanningDeadlineV2(0.0, 100.0, restoring_clock),
    )
    assert calls >= 5
    assert authority_geometry.origin is original_origin
    assert result.reason_code == "legged_foothold_hard_obstacle"
    assert result.failed_leg is LegIdV2.FRONT_LEFT
    assert result.failed_cell == obstacle


def test_deadline_signed_zero_and_persistent_clock_mutation_are_contract_failures() -> None:
    signed_holder: dict[str, PlanningDeadlineV2] = {}

    def signed_zero_clock() -> float:
        object.__setattr__(signed_holder["deadline"], "started_monotonic_s", -0.0)
        return 0.0

    signed_zero = PlanningDeadlineV2(0.0, 100.0, signed_zero_clock)
    signed_holder["deadline"] = signed_zero
    assert validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), signed_zero
    ).reason_code == "planning_deadline_contract_mismatch"

    calls = 0
    holder: dict[str, PlanningDeadlineV2] = {}

    def mutating_clock() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            object.__setattr__(holder["deadline"], "deadline_monotonic_s", 200.0)
        return 0.0

    deadline = PlanningDeadlineV2(0.0, 100.0, mutating_clock)
    holder["deadline"] = deadline
    assert validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), deadline
    ).reason_code == "planning_deadline_contract_mismatch"


def test_late_deadline_has_no_partial_metadata_and_huge_sweep_fails_before_allocation() -> None:
    calls = 0

    def late_clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls < 20 else 100.0

    timed_out = validate_legged_step_l2(
        _candidate(),
        _anchor(),
        _profile(),
        PlanningDeadlineV2(0.0, 100.0, late_clock),
    )
    assert timed_out.reason_code == "planning_deadline_expired"
    assert timed_out.timed_out is True
    assert timed_out.failed_cell is None
    assert timed_out.failed_leg is None
    assert timed_out.minimum_support_margin_m is None

    huge = _candidate(end_body_state=PoseStateV2(1.0e308, 0.0, 0.0))
    structure = validate_legged_step_l2(
        huge, _anchor(), _profile(), _deadline()
    )
    assert structure.reason_code == "legged_step_structure_mismatch"
    assert structure.checked_cell_count == 0

    structure_clock_calls = 0

    def expire_on_structure_return() -> float:
        nonlocal structure_clock_calls
        structure_clock_calls += 1
        return 0.0 if structure_clock_calls == 1 else 100.0

    timeout_precedes_structure = validate_legged_step_l2(
        huge,
        _anchor(),
        _profile(),
        PlanningDeadlineV2(0.0, 100.0, expire_on_structure_return),
    )
    assert timeout_precedes_structure.reason_code == "planning_deadline_expired"
    assert timeout_precedes_structure.timed_out is True


def _evidence(reason: str, passed: bool = False) -> ValidationEvidenceV2:
    return ValidationEvidenceV2(
        validator_id=LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2,
        level=ValidationLevelV2.L2,
        passed=passed,
        checks=(reason,),
    )


def test_result_constructor_rejects_cross_reason_metadata() -> None:
    with pytest.raises(ValueError):
        LeggedValidationResultV2(
            evidence=_evidence("planning_deadline_expired"),
            reason_code="planning_deadline_expired",
            timed_out=True,
            failed_cell=Cell(1, 1),
            failed_leg=None,
            checked_cell_count=1,
            minimum_support_margin_m=None,
        )
    with pytest.raises(ValueError):
        LeggedValidationResultV2(
            evidence=_evidence("legged_body_sweep_collision"),
            reason_code="legged_body_sweep_collision",
            timed_out=False,
            failed_cell=Cell(1, 1),
            failed_leg=LegIdV2.FRONT_LEFT,
            checked_cell_count=1,
            minimum_support_margin_m=0.05,
        )
    with pytest.raises(ValueError):
        LeggedValidationResultV2(
            evidence=_evidence("legged_step_l2_valid", passed=True),
            reason_code="legged_step_l2_valid",
            timed_out=False,
            failed_cell=None,
            failed_leg=None,
            checked_cell_count=1,
            minimum_support_margin_m=nextafter(0.05, 0.0),
        )
    with pytest.raises(ValueError):
        LeggedValidationResultV2(
            evidence=_evidence("legged_step_l2_valid", passed=True),
            reason_code="legged_step_l2_valid",
            timed_out=False,
            failed_cell=None,
            failed_leg=None,
            checked_cell_count=0,
            minimum_support_margin_m=0.05,
        )


def test_result_reaudits_exact_nested_evidence_types() -> None:
    class DerivedTuple(tuple):
        pass

    class DerivedStr(str):
        pass

    for forged_checks in (
        DerivedTuple(("legged_step_l2_valid",)),
        (DerivedStr("legged_step_l2_valid"),),
    ):
        evidence = _evidence("legged_step_l2_valid", passed=True)
        object.__setattr__(evidence, "checks", forged_checks)
        with pytest.raises((TypeError, ValueError)):
            LeggedValidationResultV2(
                evidence=evidence,
                reason_code="legged_step_l2_valid",
                timed_out=False,
                failed_cell=None,
                failed_leg=None,
                checked_cell_count=1,
                minimum_support_margin_m=0.05,
            )

    for field, value in (
        ("validator_id", DerivedStr(LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2)),
        ("level", "L2"),
        ("passed", 1),
    ):
        evidence = _evidence("legged_step_l2_valid", passed=True)
        object.__setattr__(evidence, field, value)
        with pytest.raises((TypeError, ValueError)):
            LeggedValidationResultV2(
                evidence=evidence,
                reason_code="legged_step_l2_valid",
                timed_out=False,
                failed_cell=None,
                failed_leg=None,
                checked_cell_count=1,
                minimum_support_margin_m=0.05,
            )


@pytest.mark.parametrize(
    ("reason", "passed", "failed_cell", "failed_leg", "margin"),
    [
        ("legged_foothold_grid_misaligned", False, None, LegIdV2.FRONT_LEFT, None),
        ("legged_foothold_unknown", False, Cell(1, 1), LegIdV2.FRONT_LEFT, None),
        ("legged_step_length_exceeded", False, None, LegIdV2.FRONT_LEFT, None),
        ("legged_step_height_exceeded", False, None, LegIdV2.FRONT_LEFT, None),
        ("legged_support_margin_insufficient", False, None, None, None),
        ("legged_body_sweep_collision", False, Cell(1, 1), None, 0.05),
        ("legged_foot_sequence_invalid", False, None, LegIdV2.FRONT_LEFT, 0.05),
        ("legged_step_l2_valid", True, None, None, 0.05),
    ],
)
def test_post_contact_results_require_positive_checked_count(
    reason: str,
    passed: bool,
    failed_cell: Cell | None,
    failed_leg: LegIdV2 | None,
    margin: float | None,
) -> None:
    with pytest.raises(ValueError):
        LeggedValidationResultV2(
            evidence=_evidence(reason, passed=passed),
            reason_code=reason,
            timed_out=False,
            failed_cell=failed_cell,
            failed_leg=failed_leg,
            checked_cell_count=0,
            minimum_support_margin_m=margin,
        )


def test_repeated_validations_have_canonical_bytes() -> None:
    first = validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), _deadline()
    )
    second = validate_legged_step_l2(
        _candidate(), _anchor(), _profile(), _deadline()
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
