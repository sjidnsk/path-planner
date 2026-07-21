from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
from importlib import import_module
from inspect import Parameter, signature
from math import copysign, cos, fsum, nextafter, pi, sin, sqrt
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import path_planner.v2 as v2_exports
import path_planner.v2.ballistics as ballistics_module
import path_planner.v2.hopper_authority as authority_module
import path_planner.v2.oracles as oracle_exports
from path_planner.core import Cell, WorldPoint
from path_planner.v2.ballistics import (
    BallisticSampleV2,
    BallisticStartV2,
    LandingCellMassV2,
    sample_ballistic_arc,
)
from path_planner.v2.contracts import (
    FailureCategoryV2,
    PlatformKindV2,
    PoseStateV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.profiles import HopperProfileV2, PlatformProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


_EXPECTED_COUNTER_IDS = (
    "hopper_arc_interval_count/v1",
    "hopper_arc_candidate_width/v1",
    "hopper_arc_candidate_height/v1",
    "hopper_arc_interval_cartesian_product/v1",
    "hopper_arc_distinct_cell_count/v1",
    "hopper_arc_interval_cell_visit_count/v1",
)

_ORACLE_ONLY_PARAMETER_SET_ID = "hopper_gate5b_algorithm_fixture/v1"
_ORACLE_ONLY_PROFILE_ID = "hopper-lunar-ballistic-gate5b-fixture/v1"
_ORACLE_ONLY_STOP_ID = (
    "hopper_same_height_nominal_recenter_capture_le_3mps_"
    "stop_simulation_proxy/v1"
)
_ORACLE_ONLY_ENERGY_ID = "hopper_launch_speed_squared_relative_energy/v1"
_SECTION2_FIXTURE_IDENTITIES = frozenset(
    {
        "hopper_gate5b_algorithm_fixture/v1",
        "hopper-lunar-ballistic-gate5b-fixture/v1",
        "hopper_same_height_nominal_recenter_capture_le_3mps_stop_simulation_proxy/v1",
        "hopper_launch_speed_squared_relative_energy/v1",
    }
)

_EXPECTED_REASONS = (
    "planning_deadline_contract_mismatch",
    "hopper_authority_contract_mismatch",
    "hopper_profile_contract_mismatch",
    "terrain_snapshot_identity_mismatch",
    "terrain_snapshot_hash_mismatch",
    "hopper_terrain_geometry_contract_mismatch",
    "planning_deadline_expired",
    "terrain_query_contract_mismatch",
    "hopper_numeric_contract_mismatch",
    "hopper_replay_work_budget_exceeded",
    "hopper_launch_unknown",
    "hopper_launch_unsafe",
    "hopper_arc_boundary_violation",
    "hopper_arc_unknown",
    "hopper_arc_clearance_violation",
    "hopper_landing_probability_below_threshold",
    "hopper_landing_zone_unknown",
    "hopper_landing_zone_unsafe",
    "hopper_landing_slope_exceeded",
    "hopper_landing_height_unreachable",
    "hopper_landing_theta_unreachable",
    "hopper_stop_condition_failed",
    "hopper_jump_l2_valid",
)


def _module():
    return import_module("path_planner.v2.oracles.hopper")


def _profile(**overrides: object) -> HopperProfileV2:
    values: dict[str, object] = {
        "profile": PlatformProfileV2(
            profile_id=_ORACLE_ONLY_PROFILE_ID,
            platform_kind=PlatformKindV2.HOPPER,
            capability_revision="simulation_proxy_lunar_ballistic/v1",
            simulation_proxy=True,
            max_traversable_slope_deg=30.0,
            goal_position_tolerance_m=0.0,
            goal_heading_tolerance_rad=0.0,
        ),
        "body_envelope_radius_m": 0.25,
        "launch_reference_height_m": 0.50,
        "arc_clearance_margin_m": 0.10,
        "landing_footprint_radius_m": 0.30,
        "stop_condition": _ORACLE_ONLY_STOP_ID,
        "energy_model": _ORACLE_ONLY_ENERGY_ID,
    }
    values.update(overrides)
    return HopperProfileV2(**values)


def _anchor(
    *,
    width: int = 64,
    height: int = 64,
    origin: tuple[float, float] = (-8.0, -8.0),
    unknown: tuple[Cell, ...] = (),
    hard: tuple[Cell, ...] = (),
    not_traversable: tuple[Cell, ...] = (),
    slopes: tuple[tuple[Cell, float], ...] = (),
    elevations: tuple[tuple[Cell, float], ...] = (),
) -> FineSafetyAnchorV2:
    shape = (height, width)
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
            width=width,
            height=height,
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
            source_id="gate5b-hopper-oracle-fixture",
            source_hash="gate5b-hopper-oracle-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )
    return FineSafetyAnchorV2(snapshot)


def _deadline(now: float = 0.0, cutoff: float = 100.0) -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, cutoff, lambda: now)


def _candidate(**overrides: object):
    module = _module()
    values: dict[str, object] = {
        "start_state": PoseStateV2(0.25, 0.25, 0.0),
        "support_height_m": 0.0,
        "hopper_profile": _profile(),
        "parameter_set_id": _ORACLE_ONLY_PARAMETER_SET_ID,
        "speed_index": 1,
        "elevation_index": 1,
        "azimuth_index": 0,
        "schema_version": "hopper-jump-candidate/v1",
    }
    values.update(overrides)
    return module.HopperJumpCandidateV2(**values)


def _validate(*, candidate=None, anchor=None, deadline=None):
    module = _module()
    return module.validate_hopper_jump_l2(
        _candidate() if candidate is None else candidate,
        _anchor() if anchor is None else anchor,
        _deadline() if deadline is None else deadline,
    )


def _zero_counts() -> tuple[tuple[str, int], ...]:
    return tuple((counter_id, 0) for counter_id in _EXPECTED_COUNTER_IDS)


def _cell_for(anchor: FineSafetyAnchorV2, x_m: float, y_m: float) -> Cell:
    return anchor.snapshot.geometry.world_to_cell(WorldPoint(x_m, y_m))


def _fraction(value: float | Fraction) -> Fraction:
    if type(value) is Fraction:
        return value
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _reference_candidate_bounds(
    lower: float | Fraction,
    upper: float | Fraction,
    origin: float | Fraction,
    resolution: float | Fraction,
) -> tuple[int, int]:
    lower_ratio = (_fraction(lower) - _fraction(origin)) / _fraction(resolution)
    upper_ratio = (_fraction(upper) - _fraction(origin)) / _fraction(resolution)
    lower_ceil = -((-lower_ratio.numerator) // lower_ratio.denominator)
    return lower_ceil - 1, upper_ratio.numerator // upper_ratio.denominator


def _reference_disk_segment_square_relation(
    x0: float | Fraction,
    y0: float | Fraction,
    dx: float | Fraction,
    dy: float | Fraction,
    body_radius: float | Fraction,
    clearance_margin: float | Fraction,
    x_lower: float | Fraction,
    x_upper: float | Fraction,
    y_lower: float | Fraction,
    y_upper: float | Fraction,
) -> str:
    x0_q, y0_q = _fraction(x0), _fraction(y0)
    dx_q, dy_q = _fraction(dx), _fraction(dy)
    radius = _fraction(body_radius) + _fraction(clearance_margin)
    radius_squared = radius * radius
    x_lower_q, x_upper_q = _fraction(x_lower), _fraction(x_upper)
    y_lower_q, y_upper_q = _fraction(y_lower), _fraction(y_upper)
    partition = {Fraction(0), Fraction(1)}
    for start, delta, lower, upper in (
        (x0_q, dx_q, x_lower_q, x_upper_q),
        (y0_q, dy_q, y_lower_q, y_upper_q),
    ):
        if delta != 0:
            for boundary in (lower, upper):
                value = (boundary - start) / delta
                if 0 <= value <= 1:
                    partition.add(value)

    def axis_distance(
        start: Fraction,
        delta: Fraction,
        lower: Fraction,
        upper: Fraction,
        at: Fraction,
    ) -> tuple[Fraction, Fraction]:
        point = start + delta * at
        if point < lower:
            return -delta, lower - start
        if point > upper:
            return delta, start - upper
        return Fraction(0), Fraction(0)

    singleton = False
    ordered = sorted(partition)
    for left, right in zip(ordered, ordered[1:]):
        midpoint = (left + right) / 2
        ax, bx = axis_distance(x0_q, dx_q, x_lower_q, x_upper_q, midpoint)
        ay, by = axis_distance(y0_q, dy_q, y_lower_q, y_upper_q, midpoint)
        quadratic = ax * ax + ay * ay
        linear = 2 * (ax * bx + ay * by)
        constant = bx * bx + by * by
        candidates = {left, right}
        if quadratic != 0:
            vertex = -linear / (2 * quadratic)
            if left <= vertex <= right:
                candidates.add(vertex)
        values = tuple(
            quadratic * value * value + linear * value + constant
            for value in candidates
        )
        minimum = min(values)
        if minimum < radius_squared:
            return "interval"
        if quadratic == 0 and linear == 0 and constant == radius_squared:
            return "interval"
        if minimum == radius_squared:
            singleton = True
    return "singleton" if singleton else "empty"


def _reference_parabola_clearance(
    launch_z: float,
    apex_height: float,
    terrain_elevation: float,
    body_radius: float,
    clearance_margin: float,
    parameter_numerator: int,
    parameter_denominator: int,
) -> bool:
    parameter = Fraction(parameter_numerator, parameter_denominator)
    height = _fraction(launch_z) + (
        4 * _fraction(apex_height) * parameter * (1 - parameter)
    )
    required = (
        _fraction(terrain_elevation)
        + _fraction(body_radius)
        + _fraction(clearance_margin)
    )
    return height >= required


def _reference_default_arc_counts() -> tuple[tuple[str, int], ...]:
    origin_x = origin_y = _fraction(-8.0)
    resolution = _fraction(0.5)
    x0 = y0 = _fraction(0.25)
    speed = 2.0
    elevation = pi / 4.0
    gravity = 1.62
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / gravity
    flight_time = 2.0 * vertical_time_scale
    dx = _fraction((horizontal_speed * 1.0) * flight_time)
    dy = Fraction(0)
    radius = _fraction(0.25) + _fraction(0.10)
    min_x, max_x = _reference_candidate_bounds(
        min(x0, x0 + dx) - radius,
        max(x0, x0 + dx) + radius,
        origin_x,
        resolution,
    )
    min_y, max_y = _reference_candidate_bounds(
        min(y0, y0 + dy) - radius,
        max(y0, y0 + dy) + radius,
        origin_y,
        resolution,
    )
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    overlapping: list[tuple[int, int]] = []
    for cell_y in range(min_y, max_y + 1):
        y_lower = origin_y + cell_y * resolution
        for cell_x in range(min_x, max_x + 1):
            x_lower = origin_x + cell_x * resolution
            relation = _reference_disk_segment_square_relation(
                x0,
                y0,
                dx,
                dy,
                _fraction(0.25),
                _fraction(0.10),
                x_lower,
                x_lower + resolution,
                y_lower,
                y_lower + resolution,
            )
            if relation != "empty":
                overlapping.append((cell_x, cell_y))
    return (
        (_EXPECTED_COUNTER_IDS[0], 1),
        (_EXPECTED_COUNTER_IDS[1], width),
        (_EXPECTED_COUNTER_IDS[2], height),
        (_EXPECTED_COUNTER_IDS[3], width * height),
        (_EXPECTED_COUNTER_IDS[4], len(set(overlapping))),
        (_EXPECTED_COUNTER_IDS[5], width * height),
    )


def _reference_qsqrt_sign(
    rational: Fraction,
    radical_coefficient: Fraction,
    radicand: Fraction,
) -> int:
    assert radicand >= 0
    if radical_coefficient == 0 or radicand == 0:
        return (rational > 0) - (rational < 0)
    if rational == 0:
        return (radical_coefficient > 0) - (radical_coefficient < 0)
    if (rational > 0) == (radical_coefficient > 0):
        return 1 if rational > 0 else -1
    rational_squared = rational * rational
    radical_squared = radical_coefficient * radical_coefficient * radicand
    if rational_squared == radical_squared:
        return 0
    if rational > 0:
        return 1 if rational_squared > radical_squared else -1
    return -1 if rational_squared > radical_squared else 1


def _reference_diagonal_clearance_bracket() -> tuple[float, float]:
    speed = 2.0
    elevation = pi / 4.0
    gravity = 1.62
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / gravity
    flight_time = 2.0 * vertical_time_scale
    dx = _fraction((horizontal_speed * 1.0) * flight_time)
    apex_height = _fraction(vertical_time_scale * (0.5 * vertical_speed))
    x0 = y0 = _fraction(0.25)
    x_lower = _fraction(1.0)
    y_lower = _fraction(0.5)
    radius = _fraction(0.25) + _fraction(0.10)
    radicand = radius * radius - (y_lower - y0) * (y_lower - y0)
    assert radicand > 0

    # The first closed overlap endpoint with cell [1,1.5] x [0.5,1.0] is
    # u = a + b*sqrt(radicand).  Lift the full parabola at that endpoint
    # into q + k*sqrt(radicand), then bracket it with adjacent binary64 words.
    a = (x_lower - x0) / dx
    b = -Fraction(1, 1) / dx
    constant = a - a * a - b * b * radicand
    radical = b - 2 * a * b
    threshold_q = _fraction(0.50) - radius + 4 * apex_height * constant
    threshold_k = 4 * apex_height * radical

    def clearance_sign(elevation_m: float) -> int:
        return _reference_qsqrt_sign(
            threshold_q - _fraction(elevation_m),
            threshold_k,
            radicand,
        )

    probe = float(threshold_q) + float(threshold_k) * sqrt(float(radicand))
    for _ in range(8):
        probe_sign = clearance_sign(probe)
        if probe_sign >= 0:
            lower = probe
            upper = nextafter(probe, float("inf"))
            if clearance_sign(upper) < 0:
                break
            probe = upper
        else:
            upper = probe
            lower = nextafter(probe, float("-inf"))
            if clearance_sign(lower) >= 0:
                break
            probe = lower
    else:
        raise AssertionError("failed to bracket exact Qsqrt clearance threshold")
    assert nextafter(lower, float("inf")) == upper
    assert clearance_sign(lower) >= 0
    assert clearance_sign(upper) < 0
    return lower, upper


def _exact_overlap_boundary_cases() -> tuple[
    tuple[tuple[float, ...], str], ...
]:
    minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
    large = float(2**50)
    enormous = float.fromhex("0x1.0000000000000p+900")
    enormous_ulp = nextafter(enormous, float("inf")) - enormous
    return (
        (
            (
                0.0,
                0.0,
                0.0,
                0.0,
                minimum_subnormal,
                0.0,
                minimum_subnormal,
                2.0 * minimum_subnormal,
                -minimum_subnormal,
                minimum_subnormal,
            ),
            "interval",
        ),
        (
            (
                0.0,
                0.0,
                0.0,
                0.0,
                minimum_subnormal,
                minimum_subnormal,
                3.0 * minimum_subnormal,
                4.0 * minimum_subnormal,
                -minimum_subnormal,
                minimum_subnormal,
            ),
            "empty",
        ),
        (
            (
                large + 0.25,
                0.0,
                0.0,
                0.0,
                0.25,
                0.0,
                large + 0.50,
                large + 1.0,
                -0.50,
                0.50,
            ),
            "interval",
        ),
        (
            (
                large + 0.25,
                0.0,
                0.0,
                0.0,
                nextafter(0.25, 0.0),
                0.0,
                large + 0.50,
                large + 1.0,
                -0.50,
                0.50,
            ),
            "empty",
        ),
        (
            (
                enormous,
                0.0,
                0.0,
                0.0,
                enormous_ulp,
                0.0,
                enormous + enormous_ulp,
                enormous + 2.0 * enormous_ulp,
                -enormous_ulp,
                enormous_ulp,
            ),
            "interval",
        ),
        (
            (
                enormous,
                0.0,
                0.0,
                0.0,
                nextafter(enormous_ulp, 0.0),
                0.0,
                enormous + enormous_ulp,
                enormous + 2.0 * enormous_ulp,
                -enormous_ulp,
                enormous_ulp,
            ),
            "empty",
        ),
    )


def _assert_result_envelope(
    result,
    *,
    reason_code: str,
    category: FailureCategoryV2 | None,
    stage: str,
    timed_out: bool = False,
    details: tuple[tuple[str, object], ...] | None = (),
) -> None:
    module = _module()
    assert type(result) is module.HopperValidationResultV2
    assert result.reason_code == reason_code
    assert result.category is category
    assert result.stage == stage
    if details is not None:
        assert result.details == details
    assert result.timed_out is timed_out
    assert result.evidence == ValidationEvidenceV2(
        validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
        level=ValidationLevelV2.L2,
        passed=reason_code == "hopper_jump_l2_valid",
        checks=(reason_code,),
    )
    if reason_code == "hopper_jump_l2_valid":
        assert type(result.selected_landing_mass) is float
        assert result.selected_landing_mass >= 0.99
    elif stage not in ("landing_probability", "landing_validation", "stop_validation"):
        assert result.selected_landing_mass is None


def _assert_contract_details(result, stage: str) -> None:
    assert type(result.details) is tuple
    assert tuple(key for key, _ in result.details) == ("actual", "expected", "phase")
    assert result.details[2] == ("phase", stage)
    for item in result.details:
        assert type(item) is tuple and len(item) == 2
        key, value = item
        assert type(key) is str and key
        assert type(value) in (type(None), bool, int, float, str)
        if type(value) is float:
            assert value == value and value not in (float("inf"), float("-inf"))


def test_hopper_oracle_candidate_freezes_exact_fields_slots_types_and_schema() -> None:
    module = _module()

    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    class StrSubclass(str):
        pass

    class PoseStateSubclass(PoseStateV2):
        pass

    class PlatformProfileSubclass(PlatformProfileV2):
        pass

    base_platform_profile = _profile().profile
    nested_profile_subclass = PlatformProfileSubclass(
        **{
            field.name: getattr(base_platform_profile, field.name)
            for field in fields(PlatformProfileV2)
        }
    )
    forged_nested_profile = _profile()
    object.__setattr__(forged_nested_profile, "profile", nested_profile_subclass)
    candidate = _candidate(
        start_state=PoseStateV2(-0.0, -0.0, -0.0),
        support_height_m=-0.0,
    )
    assert tuple(field.name for field in fields(module.HopperJumpCandidateV2)) == (
        "start_state",
        "support_height_m",
        "hopper_profile",
        "parameter_set_id",
        "speed_index",
        "elevation_index",
        "azimuth_index",
        "schema_version",
    )
    assert not hasattr(candidate, "__dict__")
    assert candidate.start_state == PoseStateV2(0.0, 0.0, 0.0)
    assert candidate.support_height_m == 0.0
    assert copysign(1.0, candidate.support_height_m) == 1.0
    assert candidate.schema_version == "hopper-jump-candidate/v1"
    assert {
        candidate.parameter_set_id,
        candidate.hopper_profile.profile.profile_id,
        candidate.hopper_profile.stop_condition,
        candidate.hopper_profile.energy_model,
    } == _SECTION2_FIXTURE_IDENTITIES
    assert "formal" not in candidate.parameter_set_id
    forbidden = {
        "dt_s",
        "speed_mps",
        "elevation_rad",
        "azimuth_rad",
        "x_direction",
        "y_direction",
        "samples",
        "landing_endpoint",
    }
    assert forbidden.isdisjoint(field.name for field in fields(candidate))
    with pytest.raises(FrozenInstanceError):
        candidate.speed_index = 0  # type: ignore[misc]

    for name, invalid in (
        ("start_state", object()),
        ("start_state", PoseStateSubclass(0.25, 0.25, 0.0)),
        ("support_height_m", True),
        ("support_height_m", 0),
        ("support_height_m", FloatSubclass(0.0)),
        ("support_height_m", float("nan")),
        ("support_height_m", float("inf")),
        ("hopper_profile", object()),
        ("hopper_profile", forged_nested_profile),
        ("parameter_set_id", ""),
        ("parameter_set_id", b"fixture"),
        ("parameter_set_id", StrSubclass(_ORACLE_ONLY_PARAMETER_SET_ID)),
        ("speed_index", True),
        ("speed_index", IntSubclass(1)),
        ("speed_index", -1),
        ("speed_index", 4),
        ("elevation_index", True),
        ("elevation_index", IntSubclass(1)),
        ("elevation_index", -1),
        ("elevation_index", 3),
        ("azimuth_index", True),
        ("azimuth_index", IntSubclass(0)),
        ("azimuth_index", -1),
        ("azimuth_index", 16),
        ("schema_version", StrSubclass("hopper-jump-candidate/v1")),
        ("schema_version", "hopper-jump-candidate/v2"),
    ):
        with pytest.raises((TypeError, ValueError)):
            _candidate(**{name: invalid})


def test_hopper_oracle_result_freezes_exact_fields_reason_stage_evidence_and_schema() -> None:
    module = _module()

    class IntSubclass(int):
        pass

    class StrSubclass(str):
        pass

    class TupleSubclass(tuple):
        pass

    assert module.HOPPER_JUMP_VALIDATOR_ID_V2 == "path-planner-v2-hopper-jump-l2/v1"
    assert module.HOPPER_JUMP_REASON_CODES_V1 == _EXPECTED_REASONS
    assert module.HOPPER_REPLAY_COUNTER_IDS_V1 == _EXPECTED_COUNTER_IDS
    assert tuple(field.name for field in fields(module.HopperValidationResultV2)) == (
        "evidence",
        "reason_code",
        "category",
        "stage",
        "details",
        "timed_out",
        "failed_cell",
        "segment_index",
        "replay_work_counts",
        "selected_landing_mass",
        "schema_version",
    )
    result = module.HopperValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=True,
            checks=("hopper_jump_l2_valid",),
        ),
        reason_code="hopper_jump_l2_valid",
        category=None,
        stage="stop_validation",
        details=(),
        timed_out=False,
        failed_cell=None,
        segment_index=None,
        replay_work_counts=_reference_default_arc_counts(),
        selected_landing_mass=0.99,
        schema_version="hopper-jump-validation-result/v1",
    )
    assert not hasattr(result, "__dict__")
    assert result.selected_landing_mass == 0.99
    assert result.replay_work_counts == _reference_default_arc_counts()
    assert result.replay_work_counts == (
        (_EXPECTED_COUNTER_IDS[0], 1),
        (_EXPECTED_COUNTER_IDS[1], 8),
        (_EXPECTED_COUNTER_IDS[2], 3),
        (_EXPECTED_COUNTER_IDS[3], 24),
        (_EXPECTED_COUNTER_IDS[4], 20),
        (_EXPECTED_COUNTER_IDS[5], 24),
    )
    assert result.evidence.checks == (result.reason_code,)
    assert result.evidence.passed is True
    with pytest.raises(FrozenInstanceError):
        result.reason_code = "hopper_arc_unknown"  # type: ignore[misc]

    resource_reason = "hopper_replay_work_budget_exceeded"
    resource_result = module.HopperValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=False,
            checks=(resource_reason,),
        ),
        reason_code=resource_reason,
        category=FailureCategoryV2.RESOURCE_LIMIT,
        stage="replay_work",
        details=(
            ("attempted_work_units", 100_001),
            ("max_work_units", 100_000),
            ("phase", _EXPECTED_COUNTER_IDS[0]),
        ),
        timed_out=False,
        failed_cell=None,
        segment_index=None,
        replay_work_counts=_zero_counts(),
        selected_landing_mass=None,
        schema_version="hopper-jump-validation-result/v1",
    )
    assert resource_result.details[2] == ("phase", _EXPECTED_COUNTER_IDS[0])

    semantic_reason = "hopper_arc_unknown"
    semantic_cell = Cell(2, 3)
    semantic_result = module.HopperValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=False,
            checks=(semantic_reason,),
        ),
        reason_code=semantic_reason,
        category=FailureCategoryV2.VALIDATION_FAILED,
        stage="arc_validation",
        details=(),
        timed_out=False,
        failed_cell=semantic_cell,
        segment_index=0,
        replay_work_counts=tuple(
            (counter_id, 1) for counter_id in _EXPECTED_COUNTER_IDS
        ),
        selected_landing_mass=None,
        schema_version="hopper-jump-validation-result/v1",
    )
    assert semantic_result.failed_cell == semantic_cell
    with pytest.raises((TypeError, ValueError)):
        replace(semantic_result, failed_cell=None)
    with pytest.raises((TypeError, ValueError)):
        replace(semantic_result, segment_index=None)

    invalid_values = (
        ("reason_code", "not-a-hopper-reason"),
        ("reason_code", StrSubclass("hopper_jump_l2_valid")),
        ("category", FailureCategoryV2.TIMEOUT),
        ("stage", StrSubclass("arc_validation")),
        ("details", []),
        ("details", TupleSubclass(())),
        ("details", (("phase", "replay_work"), ("actual", "late"))),
        ("details", (("actual", float("nan")),)),
        ("timed_out", 1),
        ("failed_cell", object()),
        ("segment_index", True),
        ("replay_work_counts", _zero_counts()[:-1]),
        ("replay_work_counts", tuple((name, True) for name in _EXPECTED_COUNTER_IDS)),
        (
            "replay_work_counts",
            tuple((name, 100_001) for name in _EXPECTED_COUNTER_IDS),
        ),
        ("selected_landing_mass", -0.01),
        ("selected_landing_mass", 1.01),
        ("selected_landing_mass", float("nan")),
        ("schema_version", StrSubclass("hopper-jump-validation-result/v1")),
        ("schema_version", "hopper-jump-validation-result/v2"),
    )
    for name, invalid in invalid_values:
        with pytest.raises((TypeError, ValueError)):
            replace(result, **{name: invalid})

    for malformed_evidence in (
        ValidationEvidenceV2(
            validator_id="wrong-validator/v1",
            level=ValidationLevelV2.L2,
            passed=True,
            checks=(result.reason_code,),
        ),
        ValidationEvidenceV2(
            validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L1,
            passed=True,
            checks=(result.reason_code,),
        ),
        ValidationEvidenceV2(
            validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=False,
            checks=(result.reason_code,),
        ),
        ValidationEvidenceV2(
            validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=True,
            checks=("hopper_arc_unknown",),
        ),
        ValidationEvidenceV2(
            validator_id=StrSubclass(module.HOPPER_JUMP_VALIDATOR_ID_V2),
            level=ValidationLevelV2.L2,
            passed=True,
            checks=(result.reason_code,),
        ),
        ValidationEvidenceV2(
            validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=True,
            checks=TupleSubclass((result.reason_code,)),
        ),
        ValidationEvidenceV2(
            validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=True,
            checks=(StrSubclass(result.reason_code),),
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(result, evidence=malformed_evidence)

    derived_resource_values = (
        TupleSubclass(resource_result.details),
        tuple(TupleSubclass(item) for item in resource_result.details),
        (
            (StrSubclass("attempted_work_units"), 100_001),
            ("max_work_units", 100_000),
            ("phase", _EXPECTED_COUNTER_IDS[0]),
        ),
        (
            ("attempted_work_units", IntSubclass(100_001)),
            ("max_work_units", 100_000),
            ("phase", _EXPECTED_COUNTER_IDS[0]),
        ),
    )
    for derived_details in derived_resource_values:
        with pytest.raises((TypeError, ValueError)):
            replace(resource_result, details=derived_details)

    derived_counter_values = (
        TupleSubclass(result.replay_work_counts),
        tuple(TupleSubclass(item) for item in result.replay_work_counts),
        tuple(
            (StrSubclass(counter_id), value)
            for counter_id, value in result.replay_work_counts
        ),
        tuple(
            (counter_id, IntSubclass(value))
            for counter_id, value in result.replay_work_counts
        ),
    )
    for derived_counts in derived_counter_values:
        with pytest.raises((TypeError, ValueError)):
            replace(result, replay_work_counts=derived_counts)

    timeout_reason = "planning_deadline_expired"
    with pytest.raises((TypeError, ValueError)):
        replace(
            result,
            evidence=ValidationEvidenceV2(
                validator_id=module.HOPPER_JUMP_VALIDATOR_ID_V2,
                level=ValidationLevelV2.L2,
                passed=False,
                checks=(timeout_reason,),
            ),
            reason_code=timeout_reason,
            category=FailureCategoryV2.TIMEOUT,
            timed_out=False,
        )


def test_hopper_oracle_entrypoint_freezes_exact_signature_and_11b6_exports() -> None:
    module = _module()
    function = module.validate_hopper_jump_l2
    parameters = tuple(signature(function).parameters.values())
    assert tuple(parameter.name for parameter in parameters) == (
        "candidate",
        "anchor",
        "deadline",
    )
    assert all(
        parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is Parameter.empty
        for parameter in parameters
    )
    assert tuple(parameter.annotation for parameter in parameters) == (
        "HopperJumpCandidateV2",
        "FineSafetyAnchorV2",
        "PlanningDeadlineV2",
    )
    assert signature(function).return_annotation == "HopperValidationResultV2"
    for name in (
        "HopperJumpCandidateV2",
        "HopperValidationResultV2",
        "validate_hopper_jump_l2",
    ):
        implementation = getattr(module, name)
        assert implementation is not None
        assert name in getattr(oracle_exports, "__all__", ())
        assert name in getattr(v2_exports, "__all__", ())
        assert getattr(oracle_exports, name) is implementation
        assert getattr(v2_exports, name) is implementation


def test_hopper_oracle_rejects_forged_candidate_profile_and_indices_before_helpers(monkeypatch) -> None:
    module = _module()
    calls: list[str] = []

    def forbidden_ballistic(*_args, **_kwargs):
        calls.append("ballistic")
        raise AssertionError("candidate audit must precede ballistic replay")

    def forbidden_query(*_args, **_kwargs):
        calls.append("query")
        raise AssertionError("candidate audit must precede terrain query")

    monkeypatch.setattr(module, "_call_captured_ballistic_helper_v2", forbidden_ballistic)
    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden_query)

    class CandidateSubclass(module.HopperJumpCandidateV2):
        pass

    base_candidate = _candidate()
    derived_candidate = CandidateSubclass(
        **{
            field.name: getattr(base_candidate, field.name)
            for field in fields(module.HopperJumpCandidateV2)
        }
    )
    result = _validate(candidate=derived_candidate)
    assert result.reason_code == "hopper_numeric_contract_mismatch"
    assert calls == []

    forged_state = PoseStateV2(0.25, 0.25, 0.0)
    object.__setattr__(forged_state, "x_m", float("nan"))
    candidate = _candidate()
    object.__setattr__(candidate, "start_state", forged_state)
    result = _validate(candidate=candidate)
    assert result.reason_code == "hopper_numeric_contract_mismatch"
    assert calls == []

    forged_profile = _profile()
    object.__setattr__(forged_profile, "azimuth_direction_count", 15)
    candidate = _candidate()
    object.__setattr__(candidate, "hopper_profile", forged_profile)
    result = _validate(candidate=candidate)
    assert result.reason_code == "hopper_profile_contract_mismatch"
    assert calls == []

    for name, invalid in (
        ("speed_index", True),
        ("elevation_index", 3),
        ("azimuth_index", -1),
    ):
        candidate = _candidate()
        object.__setattr__(candidate, name, invalid)
        result = _validate(candidate=candidate)
        assert result.reason_code == "hopper_numeric_contract_mismatch"
        assert calls == []


def test_hopper_oracle_pre_audits_deadline_profile_snapshot_geometry_and_resource_authority(monkeypatch) -> None:
    module = _module()

    def assert_internal(
        result,
        reason_code: str,
        stage: str,
        *,
        zero_counts: bool = True,
    ) -> None:
        _assert_result_envelope(
            result,
            reason_code=reason_code,
            category=FailureCategoryV2.INTERNAL_ERROR,
            stage=stage,
            details=None,
        )
        _assert_contract_details(result, stage)
        assert result.failed_cell is None
        assert result.segment_index is None
        if zero_counts:
            assert result.replay_work_counts == _zero_counts()

    deadline = _deadline()
    object.__setattr__(deadline, "deadline_monotonic_s", float("nan"))
    assert_internal(
        _validate(deadline=deadline),
        "planning_deadline_contract_mismatch",
        "launch_validation",
    )

    profile = _profile()
    object.__setattr__(profile.profile, "profile_id", "")
    candidate = _candidate()
    object.__setattr__(candidate, "hopper_profile", profile)
    assert_internal(
        _validate(candidate=candidate),
        "hopper_profile_contract_mismatch",
        "launch_validation",
    )

    anchor = _anchor()
    changed_elevation = np.array(anchor.snapshot.elevation_m, copy=True)
    changed_elevation[0, 0] = 1.0
    object.__setattr__(anchor.snapshot, "elevation_m", changed_elevation)
    assert_internal(
        _validate(anchor=anchor),
        "terrain_snapshot_hash_mismatch",
        "launch_validation",
    )

    anchor = _anchor()
    object.__setattr__(anchor.snapshot.geometry, "resolution_m", 1.0)
    assert_internal(
        _validate(anchor=anchor),
        "hopper_terrain_geometry_contract_mismatch",
        "launch_validation",
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(FineSafetyAnchorV2, "query", lambda *_args, **_kwargs: object())
        assert_internal(
            _validate(),
            "terrain_query_contract_mismatch",
            "launch_validation",
        )

    anchor = _anchor()
    original_call = module._call_captured_ballistic_helper_v2
    with monkeypatch.context() as scoped:

        def snapshot_identity_drift(*args, **kwargs):
            samples = original_call(*args, **kwargs)
            object.__setattr__(anchor, "snapshot", _anchor().snapshot)
            return samples

        scoped.setattr(
            module,
            "_call_captured_ballistic_helper_v2",
            snapshot_identity_drift,
        )
        assert_internal(
            _validate(anchor=anchor),
            "terrain_snapshot_identity_mismatch",
            "arc_candidate_enumeration",
        )

    target_arc_cell = _cell_for(_anchor(), 1.25, 0.25)
    original_query = FineSafetyAnchorV2.query
    with monkeypatch.context() as scoped:

        def malformed_arc_query(self, cell, *args, **kwargs):
            if cell == target_arc_cell:
                return object()
            return original_query(self, cell, *args, **kwargs)

        scoped.setattr(FineSafetyAnchorV2, "query", malformed_arc_query)
        assert_internal(
            _validate(),
            "terrain_query_contract_mismatch",
            "arc_validation",
            zero_counts=False,
        )

    monkeypatch.setattr(authority_module, "HOPPER_MAX_REPLAY_STEPS_V2", 99_999)
    assert_internal(
        _validate(),
        "hopper_authority_contract_mismatch",
        "launch_validation",
    )


def test_hopper_oracle_reseals_normal_ordinary_and_critical_helper_completion(monkeypatch) -> None:
    module = _module()
    trusted_binding = ballistics_module._sample_hopper_ballistic_arc_capped_v2

    def rebound(*_args, **_kwargs):
        raise AssertionError("rebound specialized helper must never dispatch")

    with monkeypatch.context() as scoped:
        original_call = module._call_captured_ballistic_helper_v2

        def normal_drift(*args, **kwargs):
            result = original_call(*args, **kwargs)
            scoped.setattr(
                ballistics_module,
                "_sample_hopper_ballistic_arc_capped_v2",
                rebound,
            )
            return result

        scoped.setattr(module, "_call_captured_ballistic_helper_v2", normal_drift)
        normal_result = _validate()
        _assert_result_envelope(
            normal_result,
            reason_code="hopper_authority_contract_mismatch",
            category=FailureCategoryV2.INTERNAL_ERROR,
            stage="arc_candidate_enumeration",
            details=None,
        )
        _assert_contract_details(normal_result, "arc_candidate_enumeration")

    assert ballistics_module._sample_hopper_ballistic_arc_capped_v2 is trusted_binding
    with monkeypatch.context() as scoped:

        def ordinary_failure(*_args, **_kwargs):
            raise RuntimeError("ordinary helper failure")

        scoped.setattr(module, "_call_captured_ballistic_helper_v2", ordinary_failure)
        ordinary_result = _validate()
        _assert_result_envelope(
            ordinary_result,
            reason_code="hopper_numeric_contract_mismatch",
            category=FailureCategoryV2.INTERNAL_ERROR,
            stage="arc_candidate_enumeration",
            details=None,
        )
        _assert_contract_details(ordinary_result, "arc_candidate_enumeration")

    with monkeypatch.context() as scoped:

        def ordinary_drift(*_args, **_kwargs):
            scoped.setattr(
                ballistics_module,
                "_sample_hopper_ballistic_arc_capped_v2",
                rebound,
            )
            raise RuntimeError("ordinary helper failure")

        scoped.setattr(module, "_call_captured_ballistic_helper_v2", ordinary_drift)
        drift_result = _validate()
        _assert_result_envelope(
            drift_result,
            reason_code="hopper_authority_contract_mismatch",
            category=FailureCategoryV2.INTERNAL_ERROR,
            stage="arc_candidate_enumeration",
            details=None,
        )
        _assert_contract_details(drift_result, "arc_candidate_enumeration")

    for critical in (KeyboardInterrupt, MemoryError, SystemExit):
        with monkeypatch.context() as scoped:

            def critical_failure(*_args, _critical=critical, **_kwargs):
                scoped.setattr(
                    ballistics_module,
                    "_sample_hopper_ballistic_arc_capped_v2",
                    rebound,
                )
                raise _critical("critical helper failure")

            scoped.setattr(module, "_call_captured_ballistic_helper_v2", critical_failure)
            with pytest.raises(critical, match="critical helper failure"):
                _validate()


def test_hopper_oracle_requires_captured_a2_two_sample_flight_time_partition(monkeypatch) -> None:
    module = _module()
    observed: list[tuple[object, ...]] = []
    admissions: list[tuple[str, int]] = []
    ledger_events: list[tuple[object, ...]] = []
    original = module._call_captured_ballistic_helper_v2
    original_admit = module._admit_replay_work_v2
    original_insert = module._insert_distinct_cell_v2

    def tracking_call(*args, **kwargs):
        samples = original(*args, **kwargs)
        observed.append((*args, kwargs, samples))
        return samples

    def tracking_admit(counts, counter_id, attempted_value):
        admissions.append((counter_id, attempted_value))
        ledger_events.append(("admit", counter_id, attempted_value))
        return original_admit(counts, counter_id, attempted_value)

    def tracking_insert(*args, **kwargs):
        ledger_events.append(("insert",))
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(module, "_call_captured_ballistic_helper_v2", tracking_call)
    monkeypatch.setattr(module, "_admit_replay_work_v2", tracking_admit)
    monkeypatch.setattr(module, "_insert_distinct_cell_v2", tracking_insert)
    result = _validate()
    _assert_result_envelope(
        result,
        reason_code="hopper_jump_l2_valid",
        category=None,
        stage="stop_validation",
    )
    expected_counts = _reference_default_arc_counts()
    assert expected_counts == (
        (_EXPECTED_COUNTER_IDS[0], 1),
        (_EXPECTED_COUNTER_IDS[1], 8),
        (_EXPECTED_COUNTER_IDS[2], 3),
        (_EXPECTED_COUNTER_IDS[3], 24),
        (_EXPECTED_COUNTER_IDS[4], 20),
        (_EXPECTED_COUNTER_IDS[5], 24),
    )
    assert result.replay_work_counts == expected_counts
    assert admissions[:5] == [
        expected_counts[0],
        expected_counts[1],
        expected_counts[2],
        expected_counts[3],
        expected_counts[5],
    ]
    assert admissions[5:] == [
        (_EXPECTED_COUNTER_IDS[4], count) for count in range(1, 21)
    ]
    assert ledger_events[:5] == [
        ("admit", counter_id, value)
        for counter_id, value in admissions[:5]
    ]
    assert ledger_events[5:] == [
        item
        for count in range(1, 21)
        for item in (
            ("admit", _EXPECTED_COUNTER_IDS[4], count),
            ("insert",),
        )
    ]
    assert tuple(
        next(value for observed_id, value in reversed(admissions) if observed_id == counter_id)
        for counter_id in _EXPECTED_COUNTER_IDS
    ) == tuple(value for _, value in expected_counts)
    assert len(observed) == 1
    authority, start, speed, elevation, azimuth_index, gravity, kwargs, samples = observed[0]
    assert authority is authority_module.HOPPER_RESOURCE_AUTHORITY_V2
    assert type(start) is BallisticStartV2
    assert (speed, elevation, azimuth_index, gravity) == (2.0, pi / 4.0, 0, 1.62)
    assert kwargs == {}
    assert type(samples) is tuple and len(samples) == 2
    flight_time = 2.0 * ((speed * sin(elevation)) / gravity)
    assert tuple(sample.time_s.hex() for sample in samples) == (
        0.0.hex(),
        flight_time.hex(),
    )
    assert module._partition_ballistic_samples_v2(samples, flight_time) == (
        (0.0, flight_time),
    )


def test_hopper_oracle_rejects_a2_direction_endpoint_and_sample_postcondition_drift_globally(monkeypatch) -> None:
    module = _module()
    candidate = _candidate()
    profile = candidate.hopper_profile
    start = BallisticStartV2(
        candidate.start_state.x_m,
        candidate.start_state.y_m,
        candidate.support_height_m + profile.launch_reference_height_m,
    )
    canonical = authority_module._call_captured_ballistic_helper_v2(
        authority_module.HOPPER_RESOURCE_AUTHORITY_V2,
        start,
        profile.launch_speeds_mps[candidate.speed_index],
        profile.launch_elevations_rad[candidate.elevation_index],
        candidate.azimuth_index,
        profile.gravity_mps2,
    )
    variants = (
        canonical[:1],
        canonical + (canonical[-1],),
        (
            canonical[0],
            replace(canonical[-1], time_s=nextafter(canonical[-1].time_s, 0.0)),
        ),
        (
            canonical[0],
            replace(canonical[-1], x_m=nextafter(canonical[-1].x_m, float("inf"))),
        ),
        (canonical[-1], canonical[0]),
    )
    terrain_calls = 0
    original_query = FineSafetyAnchorV2.query

    def tracking_query(self, *args, **kwargs):
        nonlocal terrain_calls
        terrain_calls += 1
        return original_query(self, *args, **kwargs)

    monkeypatch.setattr(FineSafetyAnchorV2, "query", tracking_query)
    for forged in variants:
        terrain_calls = 0
        monkeypatch.setattr(
            module,
            "_call_captured_ballistic_helper_v2",
            lambda *_args, _forged=forged, **_kwargs: _forged,
        )
        result = _validate(candidate=candidate)
        _assert_result_envelope(
            result,
            reason_code="hopper_numeric_contract_mismatch",
            category=FailureCategoryV2.INTERNAL_ERROR,
            stage="arc_candidate_enumeration",
            details=None,
        )
        _assert_contract_details(result, "arc_candidate_enumeration")
        assert result.replay_work_counts == _zero_counts()
        assert terrain_calls == 0


def test_hopper_launch_footprint_requires_closed_in_bounds_observed_safe_same_height_cells() -> None:
    module = _module()
    base_anchor = _anchor()
    launch_cell = _cell_for(base_anchor, 0.25, 0.25)

    passing = _validate(anchor=base_anchor)
    assert passing.reason_code == "hopper_jump_l2_valid"
    assert passing.category is None
    assert passing.evidence.passed is True
    assert passing.details == ()
    assert passing.stage == "stop_validation"
    assert type(passing.selected_landing_mass) is float
    assert passing.selected_landing_mass >= 0.99
    alternate_oracle_profile = _profile(
        stop_condition="hopper_oracle_unbound_stop_proxy/v1",
        energy_model="hopper_oracle_unbound_energy_proxy/v1",
    )
    assert _validate(
        candidate=_candidate(
            hopper_profile=alternate_oracle_profile,
            parameter_set_id="hopper_oracle_unbound_parameter_set/v1",
        ),
        anchor=base_anchor,
    ).reason_code == "hopper_authority_contract_mismatch"
    launch_unknown = _validate(anchor=_anchor(unknown=(launch_cell,)))
    assert launch_unknown.reason_code == "hopper_launch_unknown"
    assert launch_unknown.category is FailureCategoryV2.VALIDATION_FAILED
    assert launch_unknown.stage == "launch_validation"
    assert launch_unknown.failed_cell == launch_cell
    assert _validate(anchor=_anchor(hard=(launch_cell,))).reason_code == "hopper_launch_unsafe"
    assert (
        _validate(anchor=_anchor(not_traversable=(launch_cell,))).reason_code
        == "hopper_launch_unsafe"
    )
    assert (
        _validate(anchor=_anchor(elevations=((launch_cell, 0.1),))).reason_code
        == "hopper_launch_unsafe"
    )

    boundary_anchor = _anchor(width=16, height=16, origin=(0.0, 0.0))
    boundary_candidate = _candidate(start_state=PoseStateV2(0.0, 0.0, 0.0))
    result = _validate(candidate=boundary_candidate, anchor=boundary_anchor)
    assert result.reason_code == "hopper_launch_unsafe"
    assert result.stage == "launch_validation"
    assert result.failed_cell is not None
    assert module.HOPPER_JUMP_REASON_CODES_V1 == _EXPECTED_REASONS


def test_hopper_launch_footprint_enforces_exact_radius_reference_height_and_slope_boundary() -> None:
    anchor = _anchor()
    launch_cell = _cell_for(anchor, 0.25, 0.25)
    equality_profile = _profile(
        body_envelope_radius_m=0.25,
        arc_clearance_margin_m=0.25,
        launch_reference_height_m=0.50,
    )
    assert _validate(candidate=_candidate(hopper_profile=equality_profile)).reason_code == (
        "hopper_jump_l2_valid"
    )
    below_profile = _profile(
        body_envelope_radius_m=0.25,
        arc_clearance_margin_m=0.25,
        launch_reference_height_m=nextafter(0.50, 0.0),
    )
    assert _validate(candidate=_candidate(hopper_profile=below_profile)).reason_code == (
        "hopper_launch_unsafe"
    )
    assert (
        _validate(anchor=_anchor(slopes=((launch_cell, 30.0),))).reason_code
        == "hopper_jump_l2_valid"
    )
    assert (
        _validate(
            anchor=_anchor(slopes=((launch_cell, nextafter(30.0, float("inf"))),))
        ).reason_code
        == "hopper_launch_unsafe"
    )


def test_hopper_landing_requires_unconditioned_probability_before_terrain(
    monkeypatch,
) -> None:
    module = _module()
    original_query = FineSafetyAnchorV2.query
    landing_queries = 0

    def tracking_query(self, cell, max_slope_deg=30.0):
        nonlocal landing_queries
        if max_slope_deg == 15.0:
            landing_queries += 1
        return original_query(self, cell, max_slope_deg)

    monkeypatch.setattr(FineSafetyAnchorV2, "query", tracking_query)
    low_prefix = (LandingCellMassV2(Cell(21, 16), 0.98, True),)
    monkeypatch.setattr(
        module,
        "_call_captured_landing_helper_v2",
        lambda *_args, **_kwargs: low_prefix,
    )
    low = _validate()
    assert low.reason_code == "hopper_landing_probability_below_threshold"
    assert low.stage == "landing_probability"
    assert low.selected_landing_mass == fsum(
        item.probability_mass for item in low_prefix
    )
    assert landing_queries == 0

    oob_prefix = (LandingCellMassV2(Cell(-1, 16), 0.99, False),)
    monkeypatch.setattr(
        module,
        "_call_captured_landing_helper_v2",
        lambda *_args, **_kwargs: oob_prefix,
    )
    oob = _validate()
    assert oob.reason_code == "hopper_landing_zone_unsafe"
    assert oob.stage == "landing_validation"
    assert oob.selected_landing_mass == 0.99
    assert oob.failed_cell == Cell(-1, 16)
    assert oob.segment_index == 0


def test_hopper_landing_checks_full_cell_square_and_nominal_mean_footprints(
    monkeypatch,
) -> None:
    module = _module()
    selected = (LandingCellMassV2(Cell(21, 16), 0.99, True),)
    monkeypatch.setattr(
        module,
        "_call_captured_landing_helper_v2",
        lambda *_args, **_kwargs: selected,
    )
    diagonal = Cell(22, 17)
    dilated = _validate(anchor=_anchor(slopes=((diagonal, 16.0),)))
    assert dilated.reason_code == "hopper_landing_slope_exceeded"
    assert dilated.failed_cell == diagonal
    assert dilated.segment_index == 0

    remote = (LandingCellMassV2(Cell(30, 16), 0.99, True),)
    monkeypatch.setattr(
        module,
        "_call_captured_landing_helper_v2",
        lambda *_args, **_kwargs: remote,
    )
    nominal_mean_cell = Cell(21, 16)
    nominal = _validate(anchor=_anchor(slopes=((nominal_mean_cell, 16.0),)))
    assert nominal.reason_code == "hopper_landing_slope_exceeded"
    assert nominal.failed_cell == nominal_mean_cell
    assert nominal.segment_index is None


def test_hopper_landing_reason_mapping_and_same_height_boundary(monkeypatch) -> None:
    module = _module()
    selected_cell = Cell(30, 16)
    selected = (LandingCellMassV2(selected_cell, 0.99, True),)
    monkeypatch.setattr(
        module,
        "_call_captured_landing_helper_v2",
        lambda *_args, **_kwargs: selected,
    )
    cases = (
        (_anchor(unknown=(selected_cell,)), "hopper_landing_zone_unknown"),
        (_anchor(hard=(selected_cell,)), "hopper_landing_zone_unsafe"),
        (
            _anchor(slopes=((selected_cell, nextafter(15.0, float("inf"))),)),
            "hopper_landing_slope_exceeded",
        ),
        (
            _anchor(elevations=((selected_cell, nextafter(0.0, float("inf"))),)),
            "hopper_landing_height_unreachable",
        ),
    )
    for anchor, expected in cases:
        result = _validate(anchor=anchor)
        assert result.reason_code == expected
        assert result.stage == "landing_validation"
        assert result.failed_cell == selected_cell
        assert result.segment_index == 0
        assert result.selected_landing_mass == 0.99

    equality = _validate(anchor=_anchor(slopes=((selected_cell, 15.0),)))
    assert equality.reason_code == "hopper_jump_l2_valid"
    assert equality.stage == "stop_validation"


def test_hopper_fixture_fixed_yaw_and_closed_stop_seam() -> None:
    module = _module()
    record = authority_module.HOPPER_GATE5B_ALGORITHM_FIXTURE_V1
    candidate = _candidate(
        start_state=PoseStateV2(0.25, 0.25, 1.25),
        speed_index=3,
        azimuth_index=4,
    )
    pose = module._nominal_landing_pose_v2(candidate, 2.0, 3.0)
    assert pose == PoseStateV2(2.0, 3.0, 1.25)
    assert record.evidence_class == "test_fixture"
    assert record.simulation_proxy is True
    assert record.formal_evidence_eligible is False
    assert record.stop_evaluator(3.0) is True
    result = _validate(candidate=candidate)
    assert result.reason_code == "hopper_jump_l2_valid"
    assert result.stage == "stop_validation"
    assert type(result.selected_landing_mass) is float
    assert result.selected_landing_mass >= 0.99


def test_hopper_arc_detects_interior_unknown_and_clearance_with_safe_endpoints() -> None:
    anchor = _anchor()
    interior = _cell_for(anchor, 1.25, 0.25)
    unknown = _validate(anchor=_anchor(unknown=(interior,)))
    assert unknown.reason_code == "hopper_arc_unknown"
    assert unknown.category is FailureCategoryV2.VALIDATION_FAILED
    assert unknown.stage == "arc_validation"
    assert unknown.segment_index == 0
    assert unknown.failed_cell == interior
    # The full parabola clears this cell even though a straight 3-D line between
    # the two ground-height samples would not; samples partition but never replace it.
    assert _validate(anchor=_anchor(elevations=((interior, 0.20),))).reason_code == (
        "hopper_jump_l2_valid"
    )
    collision = _anchor(elevations=((interior, 0.80),))
    assert _validate(anchor=collision).reason_code == "hopper_arc_clearance_violation"
    assert _validate(anchor=anchor).reason_code == "hopper_jump_l2_valid"


def test_hopper_arc_treats_hard_obstacles_as_unbounded_columns_at_any_altitude() -> None:
    anchor = _anchor()
    interior = _cell_for(anchor, 1.25, 0.25)
    result = _validate(anchor=_anchor(hard=(interior,)))
    assert result.reason_code == "hopper_arc_clearance_violation"
    assert result.stage == "arc_validation"
    assert result.failed_cell == interior


def test_hopper_arc_freezes_closed_edge_corner_tangency_and_product_proxy() -> None:
    module = _module()
    relation = module._exact_disk_segment_square_relation_v2

    numeric_hook_calls: list[str] = []

    class FloatSubclass(float):
        def as_integer_ratio(self):
            numeric_hook_calls.append("as_integer_ratio")
            return super().as_integer_ratio()

    class IntSubclass(int):
        def bit_length(self):
            numeric_hook_calls.append("bit_length")
            return super().bit_length()

    fixtures = (
        (0.0, 0.0, 2.0, 0.0, 0.5, 0.5, 1.0, 2.0, 1.0, 2.0),
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.5, 1.0, 1.5),
        (0.0, 0.0, 1.0, 0.0, 0.125, 0.125, 2.0, 2.5, 2.0, 2.5),
        (0.0, 0.0, 2.0, 0.0, 0.5, 0.25, 1.0, 1.5, 0.5, 1.0),
    )
    for fixture in fixtures:
        assert relation(*fixture) == _reference_disk_segment_square_relation(*fixture)
    for position in range(len(fixtures[0])):
        forged = list(fixtures[0])
        forged[position] = FloatSubclass(forged[position])
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            relation(*forged)
        assert numeric_hook_calls == []
    classify = module._classify_exact_quadratic_overlap_v2
    assert classify(0, 0, -1, 0, 1, 1, 1) == "interval"
    assert classify(0, 0, 0, 0, 1, 1, 1) == "interval"
    assert classify(0, 0, 1, 0, 1, 1, 1) == "empty"
    with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
        classify(0, 1, 0, 0, 1, 1, 1)
    exact_fixture = (0, 0, -1, 0, 1, 1, 1)
    for position in range(len(exact_fixture)):
        forged = list(exact_fixture)
        forged[position] = IntSubclass(forged[position])
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            classify(*forged)
        assert numeric_hook_calls == []

    # A true 3-D sphere would clear this diagonal separation because
    # 0.30^2 + 0.30^2 > 0.35^2.  The approved separable product proxy is
    # intentionally conservative: horizontal overlap plus vertical gap < rho fails.
    assert 18 * 4 > 49
    assert relation(
        0.0,
        0.0,
        0.0,
        0.0,
        0.175,
        0.175,
        0.30,
        0.80,
        -0.25,
        0.25,
    ) == (
        "interval"
    )
    assert module._exact_parabola_clearance_at_rational_v2(
        0.30,
        0.0,
        0.0,
        0.175,
        0.175,
        0,
        1,
    ) is False
    assert module._exact_parabola_clearance_at_rational_v2(
        0.5,
        1.0,
        0.25,
        0.25,
        0.25,
        1,
        2,
    ) is True


def test_hopper_arc_candidate_bounds_include_incident_cells_without_false_broadphase_oob() -> None:
    module = _module()

    numeric_hook_calls: list[str] = []

    class FloatSubclass(float):
        def as_integer_ratio(self):
            numeric_hook_calls.append("as_integer_ratio")
            return super().as_integer_ratio()

    large = float(2**50)
    for fixture in (
        (0.0, 0.0, 0.0, 0.5),
        (-0.25, 0.25, 0.0, 0.5),
        (10.0, nextafter(10.0, float("inf")), -0.5, 0.5),
        (large + 0.25, large + 0.50, large, 0.5),
    ):
        assert module._exact_candidate_index_bounds_v2(*fixture) == (
            _reference_candidate_bounds(*fixture)
        )
    bounds_fixture = [0.0, 1.0, 0.0, 0.5]
    for position in range(len(bounds_fixture)):
        forged = list(bounds_fixture)
        forged[position] = FloatSubclass(forged[position])
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            module._exact_candidate_index_bounds_v2(*forged)
        assert numeric_hook_calls == []
    for expected_width, upper in (
        (100_000, 49_999.0),
        (100_001, 49_999.5),
    ):
        lower_index, upper_index = module._exact_candidate_index_bounds_v2(
            0.0,
            upper,
            0.0,
            0.5,
        )
        assert upper_index - lower_index + 1 == expected_width
    assert module._exact_disk_segment_square_relation_v2(
        0.26,
        0.26,
        0.0,
        0.0,
        0.25,
        0.10,
        -0.5,
        0.0,
        -0.5,
        0.0,
    ) == "empty"

    for fixture, expected in _exact_overlap_boundary_cases():
        assert _reference_disk_segment_square_relation(*fixture) == expected
        assert module._exact_disk_segment_square_relation_v2(*fixture) == expected

    boundary_anchor = _anchor(width=16, height=16, origin=(0.0, 0.0))
    boundary_candidate = _candidate(
        start_state=PoseStateV2(2.0, 2.0, 0.0),
        speed_index=3,
        elevation_index=1,
        azimuth_index=10,
    )
    boundary = _validate(candidate=boundary_candidate, anchor=boundary_anchor)
    assert boundary.reason_code == "hopper_arc_boundary_violation"
    assert boundary.stage == "arc_validation"
    profile = boundary_candidate.hopper_profile
    speed = profile.launch_speeds_mps[boundary_candidate.speed_index]
    elevation = profile.launch_elevations_rad[boundary_candidate.elevation_index]
    _, x_direction, y_direction = module._canonical_hopper_action_v2(
        boundary_candidate.azimuth_index
    )
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / profile.gravity_mps2
    flight_time = 2.0 * vertical_time_scale
    dx = (horizontal_speed * x_direction) * flight_time
    dy = (horizontal_speed * y_direction) * flight_time
    radius = profile.body_envelope_radius_m + profile.arc_clearance_margin_m
    min_x, max_x = _reference_candidate_bounds(
        min(boundary_candidate.start_state.x_m, boundary_candidate.start_state.x_m + dx)
        - radius,
        max(boundary_candidate.start_state.x_m, boundary_candidate.start_state.x_m + dx)
        + radius,
        0.0,
        0.5,
    )
    min_y, max_y = _reference_candidate_bounds(
        min(boundary_candidate.start_state.y_m, boundary_candidate.start_state.y_m + dy)
        - radius,
        max(boundary_candidate.start_state.y_m, boundary_candidate.start_state.y_m + dy)
        + radius,
        0.0,
        0.5,
    )
    oob_candidates: list[tuple[Cell, str]] = []
    for cell_y in range(min_y, max_y + 1):
        for cell_x in range(min_x, max_x + 1):
            if 0 <= cell_x < 16 and 0 <= cell_y < 16:
                continue
            relation = _reference_disk_segment_square_relation(
                boundary_candidate.start_state.x_m,
                boundary_candidate.start_state.y_m,
                dx,
                dy,
                profile.body_envelope_radius_m,
                profile.arc_clearance_margin_m,
                cell_x * 0.5,
                (cell_x + 1) * 0.5,
                cell_y * 0.5,
                (cell_y + 1) * 0.5,
            )
            oob_candidates.append((Cell(cell_x, cell_y), relation))
    first_overlap_index = next(
        index
        for index, (_cell, relation) in enumerate(oob_candidates)
        if relation != "empty"
    )
    assert boundary.failed_cell == oob_candidates[first_overlap_index][0]


def test_hopper_arc_freezes_exact_qsqrt_roots_and_adjacent_float_brackets() -> None:
    module = _module()
    sign = module._exact_qsqrt_sign_v2

    numeric_hook_calls: list[str] = []

    class IntSubclass(int):
        def bit_length(self):
            numeric_hook_calls.append("bit_length")
            return super().bit_length()

        def __format__(self, format_spec):
            numeric_hook_calls.append("__format__")
            return super().__format__(format_spec)

        def __bool__(self):
            numeric_hook_calls.append("__bool__")
            return super().__bool__()

        def __eq__(self, other):
            numeric_hook_calls.append("__eq__")
            return super().__eq__(other)

        def __ne__(self, other):
            numeric_hook_calls.append("__ne__")
            return super().__ne__(other)

        def __lt__(self, other):
            numeric_hook_calls.append("__lt__")
            return super().__lt__(other)

        def __le__(self, other):
            numeric_hook_calls.append("__le__")
            return super().__le__(other)

        def __gt__(self, other):
            numeric_hook_calls.append("__gt__")
            return super().__gt__(other)

        def __ge__(self, other):
            numeric_hook_calls.append("__ge__")
            return super().__ge__(other)

    class FloatSubclass(float):
        def as_integer_ratio(self):
            numeric_hook_calls.append("as_integer_ratio")
            return super().as_integer_ratio()

    assert sign(-1, 1, 2) == 1
    assert sign(-2, 1, 4) == 0
    assert sign(1, -1, 2) == -1
    assert sign(0, 1, 2) == 1
    assert sign(0, -1, 2) == -1
    for position in range(3):
        forged = [1, 1, 2]
        forged[position] = IntSubclass(forged[position])
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            sign(*forged)
        assert numeric_hook_calls == []
    sqrt_two_lower = float.fromhex("0x1.6a09e667f3bccp+0")
    sqrt_two_upper = nextafter(sqrt_two_lower, float("inf"))
    assert module._compare_binary64_to_qsqrt_v2(sqrt_two_lower, 0, 1, 2, 1) == -1
    assert module._compare_binary64_to_qsqrt_v2(sqrt_two_upper, 0, 1, 2, 1) == 1
    with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
        module._compare_binary64_to_qsqrt_v2(
            FloatSubclass(sqrt_two_lower),
            0,
            1,
            2,
            1,
        )
    assert numeric_hook_calls == []
    compare_fixture = [sqrt_two_lower, 0, 1, 2, 1]
    for position in range(1, len(compare_fixture)):
        forged = list(compare_fixture)
        forged[position] = IntSubclass(forged[position])
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            module._compare_binary64_to_qsqrt_v2(*forged)
        assert numeric_hook_calls == []

    speed = 2.0
    elevation = pi / 4.0
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / 1.62
    flight_time = 2.0 * vertical_time_scale
    horizontal_dx = (horizontal_speed * 1.0) * flight_time
    irrational_corner_fixture = (
        0.25,
        0.25,
        horizontal_dx,
        0.0,
        0.25,
        0.10,
        1.0,
        1.5,
        0.5,
        1.0,
    )
    assert _reference_disk_segment_square_relation(*irrational_corner_fixture) == (
        "interval"
    )
    assert module._exact_disk_segment_square_relation_v2(
        *irrational_corner_fixture
    ) == "interval"
    anchor = _anchor()
    diagonal_cell = _cell_for(anchor, 1.25, 0.75)
    integrated = _validate(anchor=_anchor(elevations=((diagonal_cell, 0.60),)))
    assert integrated.reason_code == "hopper_arc_clearance_violation"
    assert integrated.failed_cell == diagonal_cell

    safe_elevation, unsafe_elevation = _reference_diagonal_clearance_bracket()
    assert nextafter(safe_elevation, float("inf")) == unsafe_elevation
    exact_safe = _validate(
        anchor=_anchor(elevations=((diagonal_cell, safe_elevation),))
    )
    exact_unsafe = _validate(
        anchor=_anchor(elevations=((diagonal_cell, unsafe_elevation),))
    )
    assert exact_safe.reason_code == "hopper_jump_l2_valid"
    assert exact_unsafe.reason_code == "hopper_arc_clearance_violation"
    assert exact_unsafe.failed_cell == diagonal_cell
    assert exact_unsafe.segment_index == 0


def test_hopper_arc_freezes_exact_vertical_endpoint_minimum_equality_and_one_ulp_failure() -> None:
    module = _module()
    check = module._exact_parabola_clearance_at_rational_v2

    numeric_hook_calls: list[str] = []

    class IntSubclass(int):
        def bit_length(self):
            numeric_hook_calls.append("bit_length")
            return super().bit_length()

    class FloatSubclass(float):
        def as_integer_ratio(self):
            numeric_hook_calls.append("as_integer_ratio")
            return super().as_integer_ratio()

    fixtures = (
        (0.5, 1.0, 0.0, 0.25, 0.25, 0, 1),
        (nextafter(0.5, 0.0), 1.0, 0.0, 0.25, 0.25, 0, 1),
        (0.5, 1.0, 1.0, 0.25, 0.25, 1, 2),
        (0.5, nextafter(1.0, 0.0), 1.0, 0.25, 0.25, 1, 2),
        (0.25 + 0.10, 0.0, 0.0, 0.25, 0.10, 0, 1),
    )
    for fixture in fixtures:
        assert check(*fixture) is _reference_parabola_clearance(*fixture)
    for position in range(5):
        forged = [0.5, 1.0, 0.0, 0.25, 0.25, 1, 2]
        forged[position] = FloatSubclass(forged[position])
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            check(*forged)
        assert numeric_hook_calls == []
    with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
        check(0.5, 1.0, 0.0, 0.25, 0.25, IntSubclass(1), 2)
    assert numeric_hook_calls == []
    for position in (5, 6):
        forged = [0.5, 1.0, 0.0, 0.25, 0.25, 1, 2]
        forged[position] = IntSubclass(forged[position])
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            check(*forged)
        assert numeric_hook_calls == []


def test_hopper_arc_preserves_cardinal_near_cardinal_nonzero_and_large_origin_geometry() -> None:
    module = _module()
    action = module._canonical_hopper_action_v2
    expected_cardinals = {
        0: (1.0, 0.0),
        4: (0.0, 1.0),
        8: (-1.0, 0.0),
        12: (0.0, -1.0),
    }
    for index in range(16):
        azimuth, x_direction, y_direction = action(index)
        assert azimuth.hex() == (2.0 * pi * index / 16.0).hex()
        if index in expected_cardinals:
            assert (x_direction, y_direction) == expected_cardinals[index]
            for component, expected in zip(
                (x_direction, y_direction), expected_cardinals[index], strict=True
            ):
                if expected == 0.0:
                    assert copysign(1.0, component) == 1.0
        else:
            assert x_direction.hex() == cos(azimuth).hex()
            assert y_direction.hex() == sin(azimuth).hex()
            assert x_direction != 0.0
            assert y_direction != 0.0

    speed = 2.0
    elevation = pi / 4.0
    gravity = 1.62
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / gravity
    flight_time = 2.0 * vertical_time_scale
    near_cardinal_endpoints: list[BallisticSampleV2] = []
    for azimuth in (
        nextafter(pi / 2.0, float("-inf")),
        nextafter(pi / 2.0, float("inf")),
    ):
        samples = sample_ballistic_arc(
            BallisticStartV2(0.0, 0.0, 0.50),
            speed,
            elevation,
            azimuth,
            gravity,
            flight_time,
        )
        endpoint = samples[-1]
        expected_x = (horizontal_speed * cos(azimuth)) * flight_time
        expected_y = (horizontal_speed * sin(azimuth)) * flight_time
        assert endpoint.x_m.hex() == expected_x.hex()
        assert endpoint.y_m.hex() == expected_y.hex()
        assert endpoint.x_m != 0.0
        near_cardinal_endpoints.append(endpoint)
    assert near_cardinal_endpoints[0].x_m > 0.0
    assert near_cardinal_endpoints[1].x_m < 0.0

    for origin in (0.0, 1.0, 10.0, 1_000_000.0):
        anchor = _anchor(origin=(origin - 8.0, origin - 8.0))
        start = PoseStateV2(origin + 0.25, origin + 0.25, 0.0)
        for speed_index in range(4):
            for elevation_index in range(3):
                for azimuth_index in range(16):
                    result = _validate(
                        candidate=_candidate(
                            start_state=start,
                            speed_index=speed_index,
                            elevation_index=elevation_index,
                            azimuth_index=azimuth_index,
                        ),
                        anchor=anchor,
                    )
                    assert result.reason_code == "hopper_jump_l2_valid"


def test_hopper_arc_private_alternate_partitions_are_equivalent_with_repair_sample_sentinel(monkeypatch) -> None:
    module = _module()
    candidate = _candidate()
    profile = candidate.hopper_profile
    speed = profile.launch_speeds_mps[candidate.speed_index]
    elevation = profile.launch_elevations_rad[candidate.elevation_index]
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / profile.gravity_mps2
    flight_time = 2.0 * vertical_time_scale
    start = BallisticStartV2(
        candidate.start_state.x_m,
        candidate.start_state.y_m,
        candidate.support_height_m + profile.launch_reference_height_m,
    )
    repair_calls = 0
    original_repair = ballistics_module._bounded_local_repair

    def tracking_repair(*args, **kwargs):
        nonlocal repair_calls
        repair_calls += 1
        return original_repair(*args, **kwargs)

    monkeypatch.setattr(ballistics_module, "_bounded_local_repair", tracking_repair)
    canonical_samples = authority_module._call_captured_ballistic_helper_v2(
        authority_module.HOPPER_RESOURCE_AUTHORITY_V2,
        start,
        speed,
        elevation,
        candidate.azimuth_index,
        profile.gravity_mps2,
    )
    fine_samples = sample_ballistic_arc(
        start,
        speed,
        elevation,
        0.0,
        profile.gravity_mps2,
        0.1,
    )
    below_samples = sample_ballistic_arc(
        start,
        speed,
        elevation,
        0.0,
        profile.gravity_mps2,
        nextafter(flight_time, 0.0),
    )
    above_samples = sample_ballistic_arc(
        start,
        speed,
        elevation,
        0.0,
        profile.gravity_mps2,
        nextafter(flight_time, float("inf")),
    )
    assert repair_calls > 0
    sample_sets = (canonical_samples, fine_samples, below_samples, above_samples)
    for samples in sample_sets:
        partitions = module._partition_ballistic_samples_v2(samples, flight_time)
        assert partitions == tuple(
            (left.time_s, right.time_s)
            for left, right in zip(samples, samples[1:])
        )
        assert partitions[0][0] == 0.0
        assert partitions[-1][1] == flight_time
        assert all(left < right for left, right in partitions)
        assert all(
            left[1] == right[0]
            for left, right in zip(partitions, partitions[1:])
        )
        assert len(partitions) == len(samples) - 1

    base_anchor = _anchor()
    interior = _cell_for(base_anchor, 1.25, 0.25)
    for anchor in (
        base_anchor,
        _anchor(unknown=(interior,)),
        _anchor(elevations=((interior, 0.80),)),
    ):
        operational = _validate(candidate=candidate, anchor=anchor)
        direct_results = tuple(
            module._validate_hopper_arc_partitions_v2(
                candidate,
                anchor,
                _deadline(),
                samples,
            )
            for samples in sample_sets
        )
        assert all(result.reason_code == operational.reason_code for result in direct_results)
        assert all(result.failed_cell == operational.failed_cell for result in direct_results)
        assert all(
            result.replay_work_counts[0]
            == (_EXPECTED_COUNTER_IDS[0], len(samples) - 1)
            for result, samples in zip(direct_results, sample_sets, strict=True)
        )


def test_hopper_arc_private_alternate_partition_work_cap_fails_closed() -> None:
    module = _module()
    oversized = tuple(
        BallisticSampleV2(index / 100_001, 0.0, 0.0, 0.0)
        for index in range(100_002)
    )
    with pytest.raises(ValueError, match="hopper_replay_work_budget_exceeded"):
        module._partition_ballistic_samples_v2(oversized, 1.0)


def _assert_integrated_replay_precheck(monkeypatch, counter_id: str) -> None:
    module = _module()
    original_admit = module._admit_replay_work_v2
    original_bounds = module._exact_candidate_index_bounds_v2
    original_relation = module._exact_disk_segment_square_relation_v2
    original_insert = module._insert_distinct_cell_v2
    original_query = FineSafetyAnchorV2.query
    target_seen = False
    admitted_ids: list[str] = []
    later_work: list[str] = []
    phase_events: list[str] = []
    insertion_calls: list[object] = []

    def forced_admit(counts, active_counter_id, attempted_value):
        nonlocal target_seen
        admitted_ids.append(active_counter_id)
        phase_events.append(f"admit:{active_counter_id}")
        if active_counter_id == counter_id:
            target_seen = True
            return original_admit(counts, active_counter_id, 100_001)
        return original_admit(counts, active_counter_id, attempted_value)

    def tracking_bounds(*args, **kwargs):
        if _EXPECTED_COUNTER_IDS[0] in admitted_ids:
            phase_events.append("candidate_bounds")
        if target_seen:
            later_work.append("candidate_bounds")
        return original_bounds(*args, **kwargs)

    def tracking_relation(*args, **kwargs):
        if _EXPECTED_COUNTER_IDS[0] in admitted_ids:
            phase_events.append("exact_refinement")
        if target_seen:
            later_work.append("exact_refinement")
        return original_relation(*args, **kwargs)

    def tracking_query(self, *args, **kwargs):
        if _EXPECTED_COUNTER_IDS[0] in admitted_ids:
            phase_events.append("terrain_query")
        if target_seen:
            later_work.append("terrain_query")
        return original_query(self, *args, **kwargs)

    def tracking_insert(*args, **kwargs):
        insertion_calls.append(args[-1] if args else None)
        phase_events.append("ledger_insert")
        if target_seen:
            later_work.append("ledger_insert")
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(module, "_admit_replay_work_v2", forced_admit)
    monkeypatch.setattr(module, "_exact_candidate_index_bounds_v2", tracking_bounds)
    monkeypatch.setattr(module, "_exact_disk_segment_square_relation_v2", tracking_relation)
    monkeypatch.setattr(module, "_insert_distinct_cell_v2", tracking_insert)
    monkeypatch.setattr(FineSafetyAnchorV2, "query", tracking_query)
    result = _validate()
    expected_details = (
        ("attempted_work_units", 100_001),
        ("max_work_units", 100_000),
        ("phase", counter_id),
    )
    _assert_result_envelope(
        result,
        reason_code="hopper_replay_work_budget_exceeded",
        category=FailureCategoryV2.RESOURCE_LIMIT,
        stage="replay_work",
        details=expected_details,
    )
    admitted_prefix_counts = {
        _EXPECTED_COUNTER_IDS[0]: _zero_counts(),
        _EXPECTED_COUNTER_IDS[1]: (
            (_EXPECTED_COUNTER_IDS[0], 1),
            *_zero_counts()[1:],
        ),
        _EXPECTED_COUNTER_IDS[2]: (
            (_EXPECTED_COUNTER_IDS[0], 1),
            (_EXPECTED_COUNTER_IDS[1], 8),
            *_zero_counts()[2:],
        ),
        _EXPECTED_COUNTER_IDS[3]: (
            (_EXPECTED_COUNTER_IDS[0], 1),
            (_EXPECTED_COUNTER_IDS[1], 8),
            (_EXPECTED_COUNTER_IDS[2], 3),
            *_zero_counts()[3:],
        ),
        _EXPECTED_COUNTER_IDS[5]: (
            (_EXPECTED_COUNTER_IDS[0], 1),
            (_EXPECTED_COUNTER_IDS[1], 8),
            (_EXPECTED_COUNTER_IDS[2], 3),
            (_EXPECTED_COUNTER_IDS[3], 24),
            *_zero_counts()[4:],
        ),
        _EXPECTED_COUNTER_IDS[4]: (
            (_EXPECTED_COUNTER_IDS[0], 1),
            (_EXPECTED_COUNTER_IDS[1], 8),
            (_EXPECTED_COUNTER_IDS[2], 3),
            (_EXPECTED_COUNTER_IDS[3], 24),
            (_EXPECTED_COUNTER_IDS[4], 0),
            (_EXPECTED_COUNTER_IDS[5], 24),
        ),
    }
    assert result.replay_work_counts == admitted_prefix_counts[counter_id]
    assert target_seen is True
    assert admitted_ids[-1] == counter_id
    assert later_work == []
    assert insertion_calls == []
    expected_prefixes = {
        _EXPECTED_COUNTER_IDS[0]: (_EXPECTED_COUNTER_IDS[0],),
        _EXPECTED_COUNTER_IDS[1]: _EXPECTED_COUNTER_IDS[:2],
        _EXPECTED_COUNTER_IDS[2]: _EXPECTED_COUNTER_IDS[:3],
        _EXPECTED_COUNTER_IDS[3]: _EXPECTED_COUNTER_IDS[:4],
        _EXPECTED_COUNTER_IDS[5]: (*_EXPECTED_COUNTER_IDS[:4], _EXPECTED_COUNTER_IDS[5]),
        _EXPECTED_COUNTER_IDS[4]: (
            *_EXPECTED_COUNTER_IDS[:4],
            _EXPECTED_COUNTER_IDS[5],
            _EXPECTED_COUNTER_IDS[4],
        ),
    }
    assert tuple(admitted_ids) == expected_prefixes[counter_id]
    target_event = f"admit:{counter_id}"
    target_position = phase_events.index(target_event)
    prefix_events = phase_events[:target_position]
    if counter_id == _EXPECTED_COUNTER_IDS[0]:
        assert prefix_events == []
    elif counter_id == _EXPECTED_COUNTER_IDS[4]:
        assert "terrain_query" not in prefix_events
    else:
        assert "exact_refinement" not in prefix_events
        assert "terrain_query" not in prefix_events


def test_hopper_replay_cap_interval_count_is_prechecked_before_interval_work(monkeypatch) -> None:
    _assert_integrated_replay_precheck(monkeypatch, _EXPECTED_COUNTER_IDS[0])


def test_hopper_replay_cap_candidate_width_is_prechecked_before_enumeration(monkeypatch) -> None:
    _assert_integrated_replay_precheck(monkeypatch, _EXPECTED_COUNTER_IDS[1])


def test_hopper_replay_cap_candidate_height_is_prechecked_before_enumeration(monkeypatch) -> None:
    _assert_integrated_replay_precheck(monkeypatch, _EXPECTED_COUNTER_IDS[2])


def test_hopper_replay_cap_interval_cartesian_product_is_prechecked_before_product(monkeypatch) -> None:
    _assert_integrated_replay_precheck(monkeypatch, _EXPECTED_COUNTER_IDS[3])


def test_hopper_replay_cap_distinct_cell_count_is_prechecked_before_ledger_insertion(monkeypatch) -> None:
    _assert_integrated_replay_precheck(monkeypatch, _EXPECTED_COUNTER_IDS[4])


def test_hopper_replay_cap_interval_cell_visit_count_is_prechecked_before_cell_visit(monkeypatch) -> None:
    _assert_integrated_replay_precheck(monkeypatch, _EXPECTED_COUNTER_IDS[5])


def test_hopper_exact_integer_bit_ceiling_precedes_allocation_and_is_not_memory_failure(monkeypatch) -> None:
    module = _module()
    arena = authority_module._HopperExactIntegerArenaV2(
        authority_module.HOPPER_RESOURCE_AUTHORITY_V2
    )
    calls: list[str] = []
    assert arena.consume_admitted_integers(
        result_bit_bounds=(262_144,),
        operation=lambda: (0,),
        consumer=lambda values: values[0],
    ) == 0
    with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
        arena.consume_admitted_integers(
            result_bit_bounds=(262_145,),
            operation=lambda: calls.append("operation") or (0,),
            consumer=lambda values: values[0],
    )
    assert calls == []

    created_arenas: list[object] = []
    trusted_arena_type = authority_module._HopperExactIntegerArenaV2

    class TrackingArena:
        global_live_integer_slots = 0
        global_peak_live_integer_slots = 0

        def __init__(self, authority):
            self.inner = trusted_arena_type(authority)
            self.consume_calls = 0
            self.operation_calls = 0
            self.consumer_calls = 0
            self.consume_records: list[tuple[int, tuple[int, ...]]] = []
            created_arenas.append(self)

        @property
        def peak_live_integer_slots(self):
            return self.inner.peak_live_integer_slots

        @property
        def live_integer_slots(self):
            return self.inner.live_integer_slots

        def consume_admitted_integers(self, *args, **kwargs):
            self.consume_calls += 1
            assert args == ()
            bounds = tuple(kwargs["result_bit_bounds"])
            operation = kwargs["operation"]
            consumer = kwargs["consumer"]
            acquired = False

            def tracked_operation():
                nonlocal acquired
                attempted = type(self).global_live_integer_slots + len(bounds)
                assert attempted <= 128
                self.consume_records.append(
                    (type(self).global_live_integer_slots, bounds)
                )
                type(self).global_live_integer_slots = attempted
                type(self).global_peak_live_integer_slots = max(
                    type(self).global_peak_live_integer_slots,
                    attempted,
                )
                acquired = True
                self.operation_calls += 1
                return operation()

            def tracked_consumer(values):
                assert acquired
                self.consumer_calls += 1
                return consumer(values)

            forwarded = dict(kwargs)
            forwarded["operation"] = tracked_operation
            forwarded["consumer"] = tracked_consumer
            try:
                return self.inner.consume_admitted_integers(**forwarded)
            finally:
                if acquired:
                    type(self).global_live_integer_slots -= len(bounds)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def reset_tracking() -> None:
        assert TrackingArena.global_live_integer_slots == 0
        TrackingArena.global_peak_live_integer_slots = 0
        created_arenas.clear()

    def assert_tracking_closed(label: str) -> None:
        assert created_arenas, label
        assert sum(item.consume_calls for item in created_arenas) > 0, label
        assert all(
            item.consume_calls == item.operation_calls == item.consumer_calls
            for item in created_arenas
        ), label
        assert TrackingArena.global_live_integer_slots == 0, label
        assert 0 < TrackingArena.global_peak_live_integer_slots <= 128, label
        assert all(
            before + len(bounds) <= 128
            and all(0 < bound <= 262_144 for bound in bounds)
            for item in created_arenas
            for before, bounds in item.consume_records
        ), label
        assert all(item.live_integer_slots == 0 for item in created_arenas), label

    monkeypatch.setattr(module, "_HopperExactIntegerArenaV2", TrackingArena)
    reset_tracking()
    assert _validate().reason_code == "hopper_jump_l2_valid"
    assert_tracking_closed("top_level_default")

    candidate = _candidate()
    profile = candidate.hopper_profile
    speed = profile.launch_speeds_mps[candidate.speed_index]
    elevation = profile.launch_elevations_rad[candidate.elevation_index]
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / profile.gravity_mps2
    flight_time = 2.0 * vertical_time_scale
    start = BallisticStartV2(
        candidate.start_state.x_m,
        candidate.start_state.y_m,
        candidate.support_height_m + profile.launch_reference_height_m,
    )
    samples = authority_module._call_captured_ballistic_helper_v2(
        authority_module.HOPPER_RESOURCE_AUTHORITY_V2,
        start,
        speed,
        elevation,
        candidate.azimuth_index,
        profile.gravity_mps2,
    )
    exact_seams = (
        (
            "partition",
            lambda: module._partition_ballistic_samples_v2(samples, flight_time),
        ),
        (
            "candidate_bounds",
            lambda: module._exact_candidate_index_bounds_v2(0.0, 1.0, -8.0, 0.5),
        ),
        (
            "disk_segment_square",
            lambda: module._exact_disk_segment_square_relation_v2(
                0.25,
                0.25,
                2.0,
                0.0,
                0.25,
                0.10,
                1.0,
                1.5,
                0.0,
                0.5,
            ),
        ),
        (
            "quadratic_overlap",
            lambda: module._classify_exact_quadratic_overlap_v2(
                1,
                0,
                -1,
                0,
                1,
                1,
                1,
            ),
        ),
        ("qsqrt_sign", lambda: module._exact_qsqrt_sign_v2(-1, 1, 2)),
        (
            "binary64_qsqrt_compare",
            lambda: module._compare_binary64_to_qsqrt_v2(
                float.fromhex("0x1.6a09e667f3bccp+0"),
                0,
                1,
                2,
                1,
            ),
        ),
        (
            "rational_parabola",
            lambda: module._exact_parabola_clearance_at_rational_v2(
                0.5,
                1.0,
                0.0,
                0.25,
                0.25,
                1,
                2,
            ),
        ),
        (
            "direct_arc_core",
            lambda: module._validate_hopper_arc_partitions_v2(
                candidate,
                _anchor(),
                _deadline(),
                samples,
            ),
        ),
    )
    for seam_name, invoke in exact_seams:
        reset_tracking()
        invoke()
        assert_tracking_closed(seam_name)

    for case_index, (fixture, expected) in enumerate(
        _exact_overlap_boundary_cases()
    ):
        label = f"exact_overlap_boundary_{case_index}"
        reset_tracking()
        assert module._exact_disk_segment_square_relation_v2(*fixture) == expected
        assert_tracking_closed(label)

    near_cap = 1 << (authority_module.HOPPER_EXACT_MAX_INTEGER_BITS_V2 - 1)
    combined_sub_cap = (
        1 << (authority_module.HOPPER_EXACT_MAX_INTEGER_BITS_V2 // 2)
    ) + 1
    near_cap_invocations = (
        lambda: module._classify_exact_quadratic_overlap_v2(
            near_cap,
            near_cap,
            near_cap,
            0,
            1,
            1,
            1,
        ),
        lambda: module._exact_qsqrt_sign_v2(near_cap, -near_cap, near_cap),
        lambda: module._exact_qsqrt_sign_v2(
            combined_sub_cap,
            -combined_sub_cap,
            2,
        ),
        lambda: module._compare_binary64_to_qsqrt_v2(
            1.0,
            near_cap,
            -near_cap,
            near_cap,
            1,
        ),
        lambda: module._exact_parabola_clearance_at_rational_v2(
            0.5,
            1.0,
            0.0,
            0.25,
            0.25,
            near_cap,
            near_cap - 1,
        ),
    )
    for invoke in near_cap_invocations:
        reset_tracking()
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            invoke()
        assert sum(item.operation_calls for item in created_arenas) == 0
        assert sum(item.consumer_calls for item in created_arenas) == 0
        assert TrackingArena.global_live_integer_slots == 0

    reset_tracking()
    diagonal_cell = _cell_for(_anchor(), 1.25, 0.75)
    _, unsafe_elevation = _reference_diagonal_clearance_bracket()
    exact_boundary = _validate(
        anchor=_anchor(elevations=((diagonal_cell, unsafe_elevation),))
    )
    assert exact_boundary.reason_code == "hopper_arc_clearance_violation"
    assert_tracking_closed("top_level_qsqrt_boundary")

    class PoisonToken:
        @staticmethod
        def _fail(*_args, **_kwargs):
            raise ValueError("hopper_numeric_contract_mismatch")

        __bool__ = _fail
        __iter__ = _fail
        __len__ = _fail
        __getitem__ = _fail
        __index__ = _fail
        __int__ = _fail
        __float__ = _fail
        __eq__ = _fail
        __ne__ = _fail
        __lt__ = _fail
        __le__ = _fail
        __gt__ = _fail
        __ge__ = _fail
        __add__ = _fail
        __radd__ = _fail
        __sub__ = _fail
        __rsub__ = _fail
        __mul__ = _fail
        __rmul__ = _fail
        __truediv__ = _fail
        __rtruediv__ = _fail
        __floordiv__ = _fail
        __rfloordiv__ = _fail
        __mod__ = _fail
        __rmod__ = _fail
        __pow__ = _fail
        __rpow__ = _fail

    poison_token = PoisonToken()

    class PoisonArena:
        def __init__(self, authority):
            authority_module._require_canonical_resource_authority_v2(authority)

        @property
        def live_integer_slots(self):
            return 0

        @property
        def peak_live_integer_slots(self):
            return 0

        def consume_admitted_integers(self, *args, **kwargs):
            assert args == ()
            assert set(kwargs) == {"result_bit_bounds", "operation", "consumer"}
            return poison_token

    monkeypatch.setattr(module, "_HopperExactIntegerArenaV2", PoisonArena)
    for seam_name, invoke in exact_seams[:-1]:
        try:
            poisoned = invoke()
        except ValueError as exc:
            assert str(exc) == "hopper_numeric_contract_mismatch", seam_name
        else:
            assert poisoned is poison_token, seam_name
    for fixture, _expected in _exact_overlap_boundary_cases():
        try:
            poisoned = module._exact_disk_segment_square_relation_v2(*fixture)
        except ValueError as exc:
            assert str(exc) == "hopper_numeric_contract_mismatch"
        else:
            assert poisoned is poison_token

    poisoned_direct = exact_seams[-1][1]()
    assert poisoned_direct.reason_code == "hopper_numeric_contract_mismatch"
    poisoned_top_level = _validate()
    assert poisoned_top_level.reason_code == "hopper_numeric_contract_mismatch"


def test_hopper_exact_integer_arena_allows_128_live_slots_and_rejects_slot_129() -> None:
    _module()
    arena = authority_module._HopperExactIntegerArenaV2(
        authority_module.HOPPER_RESOURCE_AUTHORITY_V2
    )
    observations: list[tuple[int, int]] = []

    def outer_consumer(values: tuple[int, ...]) -> int:
        observations.append((arena.live_integer_slots, len(values)))
        with pytest.raises(ValueError, match="hopper_numeric_contract_mismatch"):
            arena.consume_admitted_integers(
                result_bit_bounds=(1,),
                operation=lambda: (_ for _ in ()).throw(
                    AssertionError("slot 129 must fail before operation")
                ),
                consumer=lambda inner: inner[0],
            )
        return len(values)

    assert arena.consume_admitted_integers(
        result_bit_bounds=(1,) * 128,
        operation=lambda: (0,) * 128,
        consumer=outer_consumer,
    ) == 128
    assert observations == [(128, 128)]
    assert arena.peak_live_integer_slots == 128
    assert arena.live_integer_slots == 0


def test_hopper_oracle_stable_winner_and_no_later_call_follow_total_order(monkeypatch) -> None:
    module = _module()
    anchor = _anchor()
    interior = _cell_for(anchor, 1.25, 0.25)
    query_calls = 0
    ballistic_calls = 0
    original_query = FineSafetyAnchorV2.query
    original_ballistic = module._call_captured_ballistic_helper_v2

    def tracking_query(self, *args, **kwargs):
        nonlocal query_calls
        query_calls += 1
        return original_query(self, *args, **kwargs)

    def tracking_ballistic(*args, **kwargs):
        nonlocal ballistic_calls
        ballistic_calls += 1
        return original_ballistic(*args, **kwargs)

    monkeypatch.setattr(FineSafetyAnchorV2, "query", tracking_query)
    monkeypatch.setattr(module, "_call_captured_ballistic_helper_v2", tracking_ballistic)
    expired = _deadline(now=101.0, cutoff=100.0)
    result = _validate(anchor=_anchor(unknown=(interior,), hard=(Cell(interior.x + 1, interior.y),)), deadline=expired)
    _assert_result_envelope(
        result,
        reason_code="planning_deadline_expired",
        category=FailureCategoryV2.TIMEOUT,
        stage="launch_validation",
        timed_out=True,
    )
    assert query_calls == 0
    assert ballistic_calls == 0

    expiry = {"active": False}

    def triggered_clock() -> float:
        return 101.0 if expiry["active"] else 0.0

    with monkeypatch.context() as scoped:
        terrain_after_helper = 0

        def expire_after_helper(*args, **kwargs):
            result = original_ballistic(*args, **kwargs)
            expiry["active"] = True
            return result

        def forbidden_post_helper_query(self, *args, **kwargs):
            nonlocal terrain_after_helper
            terrain_after_helper += 1
            return original_query(self, *args, **kwargs)

        scoped.setattr(
            module,
            "_call_captured_ballistic_helper_v2",
            expire_after_helper,
        )
        scoped.setattr(FineSafetyAnchorV2, "query", forbidden_post_helper_query)
        result = _validate(
            deadline=PlanningDeadlineV2(0.0, 100.0, triggered_clock)
        )
        _assert_result_envelope(
            result,
            reason_code="planning_deadline_expired",
            category=FailureCategoryV2.TIMEOUT,
            stage="arc_candidate_enumeration",
            timed_out=True,
        )
        assert result.replay_work_counts == _zero_counts()
        assert terrain_after_helper == 0

    expiry["active"] = False
    with monkeypatch.context() as scoped:
        original_admit = module._admit_replay_work_v2
        original_bounds = module._exact_candidate_index_bounds_v2
        candidate_bound_calls = 0

        def expire_after_interval_admission(counts, counter_id, attempted_value):
            result = original_admit(counts, counter_id, attempted_value)
            if counter_id == _EXPECTED_COUNTER_IDS[0]:
                expiry["active"] = True
            return result

        def forbidden_post_interval_bounds(*args, **kwargs):
            nonlocal candidate_bound_calls
            candidate_bound_calls += 1
            return original_bounds(*args, **kwargs)

        scoped.setattr(
            module,
            "_admit_replay_work_v2",
            expire_after_interval_admission,
        )
        scoped.setattr(
            module,
            "_exact_candidate_index_bounds_v2",
            forbidden_post_interval_bounds,
        )
        result = _validate(
            deadline=PlanningDeadlineV2(0.0, 100.0, triggered_clock)
        )
        _assert_result_envelope(
            result,
            reason_code="planning_deadline_expired",
            category=FailureCategoryV2.TIMEOUT,
            stage="replay_work",
            timed_out=True,
        )
        assert result.replay_work_counts == (
            (_EXPECTED_COUNTER_IDS[0], 1),
            *_zero_counts()[1:],
        )
        assert candidate_bound_calls == 0

    expiry["active"] = False
    with monkeypatch.context() as scoped:
        arc_query_cells: list[Cell] = []

        def expire_on_interior_query(self, cell, *args, **kwargs):
            result = original_query(self, cell, *args, **kwargs)
            arc_query_cells.append(cell)
            if cell == interior:
                expiry["active"] = True
            return result

        scoped.setattr(FineSafetyAnchorV2, "query", expire_on_interior_query)
        result = _validate(
            deadline=PlanningDeadlineV2(0.0, 100.0, triggered_clock)
        )
        _assert_result_envelope(
            result,
            reason_code="planning_deadline_expired",
            category=FailureCategoryV2.TIMEOUT,
            stage="arc_validation",
            timed_out=True,
        )
        assert arc_query_cells[-1] == interior
        assert arc_query_cells.count(interior) == 1

    query_calls = 0
    ballistic_calls = 0
    with monkeypatch.context() as scoped:
        scoped.setattr(authority_module, "HOPPER_MAX_REPLAY_STEPS_V2", 99_999)
        result = _validate(deadline=expired)
        _assert_result_envelope(
            result,
            reason_code="hopper_authority_contract_mismatch",
            category=FailureCategoryV2.INTERNAL_ERROR,
            stage="launch_validation",
            details=None,
        )
        _assert_contract_details(result, "launch_validation")
        assert query_calls == 0
        assert ballistic_calls == 0

    result = _validate(
        anchor=_anchor(
            unknown=(interior,),
            hard=(Cell(interior.x + 1, interior.y),),
            elevations=((Cell(interior.x + 2, interior.y), 1.0),),
        )
    )
    _assert_result_envelope(
        result,
        reason_code="hopper_arc_unknown",
        category=FailureCategoryV2.VALIDATION_FAILED,
        stage="arc_validation",
    )
    assert result.failed_cell == interior
    assert module.HOPPER_JUMP_REASON_CODES_V1 == _EXPECTED_REASONS


def test_hopper_oracle_repeated_runs_preserve_reason_cells_counters_and_evidence_bytes() -> None:
    anchor = _anchor()
    interior = _cell_for(anchor, 1.25, 0.25)
    results = tuple(_validate(anchor=_anchor(unknown=(interior,))) for _ in range(4))
    assert all(result == results[0] for result in results)
    assert all(result.failed_cell == interior for result in results)
    assert len({canonical_json_bytes(result) for result in results}) == 1

    test_path = Path(__file__).resolve()
    source_root = Path(_module().__file__).resolve().parents[3]
    script = (
        "import runpy\n"
        f"ns = runpy.run_path({str(test_path)!r})\n"
        "anchor = ns['_anchor']()\n"
        "cell = ns['_cell_for'](anchor, 1.25, 0.25)\n"
        "other = ns['Cell'](cell.x + 1, cell.y)\n"
        "result = ns['_validate'](anchor=ns['_anchor'](unknown=(other, cell)))\n"
        "print(ns['canonical_json_bytes'](result).hex())\n"
    )
    process_outputs: list[str] = []
    for seed in ("1", "777"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(source_root)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30.0,
        )
        process_outputs.append(completed.stdout.strip())
    assert process_outputs[0] == process_outputs[1]


def test_hopper_oracle_source_keeps_basic_11b2_boundary() -> None:
    # Completion-first basic boundary: keep the Hopper oracle private to 11B2,
    # require the captured specialized helper and exact-arena admission, and do
    # not enforce implementation-shape/taint rules here.  The detailed static
    # audit below is intentionally outside the active acceptance path.
    module = _module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "path_planner.v2.providers" not in source
    assert "path_planner.v2.api" not in source
    assert "sample_ballistic_arc_capped_v2" not in source
    assert "_call_captured_ballistic_helper_v2" in source
    assert "consume_admitted_integers" in source
    return

    module = _module()
    source_path = Path(module.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_names = {
        "Fraction",
        "Decimal",
        "sympy",
        "sample_ballistic_arc",
        "sample_ballistic_arc_capped_v2",
        "run_admitted",
        "sqrt",
        "isqrt",
        "gcd",
        "getsizeof",
        "from_bytes",
        "getattr",
        "globals",
        "locals",
        "make_dataclass",
        "new_class",
        "setattr",
        "vars",
        "eval",
        "exec",
        "compile",
        "__import__",
        "__build_class__",
        "__call__",
        "__class__",
        "__bases__",
        "__dict__",
        "__delattr__",
        "__mro__",
        "__new__",
        "__subclasses__",
        "_authority",
        "_live_integer_slots",
        "_peak_live_integer_slots",
        "mro",
        "tracemalloc",
    }
    observed_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    observed_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert forbidden_names.isdisjoint(observed_names | observed_attributes)
    assert not any(
        isinstance(node, (ast.FormattedValue, ast.JoinedStr))
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, (ast.AsyncWith, ast.With))
        for node in ast.walk(tree)
    )

    forbidden_import_fragments = (
        "provider",
        "validation",
        "hopper_api",
        "api",
        "oracles.legged",
        "geometry",
        "fractions",
        "decimal",
        "sympy",
    )
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert all(
        fragment not in imported
        for imported in imported_modules
        for fragment in forbidden_import_fragments
    )
    assert set(imported_modules) <= {
        "__future__",
        "collections.abc",
        "dataclasses",
        "math",
        "numpy",
        "typing",
        "path_planner.core",
        "path_planner.v2.ballistics",
        "path_planner.v2.contracts",
        "path_planner.v2.hopper_authority",
        "path_planner.v2.profiles",
        "path_planner.v2.runtime",
        "path_planner.v2.terrain",
    }
    assert not any(
        isinstance(node, ast.ClassDef) and (node.bases or node.keywords)
        for node in ast.walk(tree)
    )
    assert all(
        len(node.args) == 1 and not node.keywords
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type"
    )

    consume_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "consume_admitted_integers"
    ]
    assert consume_calls

    arena_imports = tuple(
        alias
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "path_planner.v2.hopper_authority"
        for alias in node.names
        if alias.name == "_HopperExactIntegerArenaV2"
        and (alias.asname or alias.name) == "_HopperExactIntegerArenaV2"
    )
    assert len(arena_imports) == 1
    authority_imports = tuple(
        alias
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "path_planner.v2.hopper_authority"
        for alias in node.names
        if alias.name == "HOPPER_RESOURCE_AUTHORITY_V2"
        and (alias.asname or alias.name) == "HOPPER_RESOURCE_AUTHORITY_V2"
    )
    assert len(authority_imports) == 1
    assert not any(
        (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id
            in {
                "_HopperExactIntegerArenaV2",
                "HOPPER_RESOURCE_AUTHORITY_V2",
            }
        )
        or (
            isinstance(node, ast.arg)
            and node.arg
            in {
                "_HopperExactIntegerArenaV2",
                "HOPPER_RESOURCE_AUTHORITY_V2",
            }
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name
            in {
                "_HopperExactIntegerArenaV2",
                "HOPPER_RESOURCE_AUTHORITY_V2",
            }
        )
        or (
            isinstance(node, ast.alias)
            and (node.asname or node.name.rsplit(".", 1)[-1])
            in {
                "_HopperExactIntegerArenaV2",
                "HOPPER_RESOURCE_AUTHORITY_V2",
            }
            and node not in {*arena_imports, *authority_imports}
        )
        or (
            isinstance(node, (ast.Global, ast.Nonlocal))
            and bool(
                set(node.names)
                & {
                    "_HopperExactIntegerArenaV2",
                    "HOPPER_RESOURCE_AUTHORITY_V2",
                }
            )
        )
        or (
            isinstance(node, ast.ExceptHandler)
            and node.name
            in {
                "_HopperExactIntegerArenaV2",
                "HOPPER_RESOURCE_AUTHORITY_V2",
            }
        )
        or (
            isinstance(node, (ast.MatchAs, ast.MatchStar))
            and node.name
            in {
                "_HopperExactIntegerArenaV2",
                "HOPPER_RESOURCE_AUTHORITY_V2",
            }
        )
        or (
            isinstance(node, ast.MatchMapping)
            and node.rest
            in {
                "_HopperExactIntegerArenaV2",
                "HOPPER_RESOURCE_AUTHORITY_V2",
            }
        )
        for node in ast.walk(tree)
    )

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == "type"
        ):
            continue
        parent = parents.get(node)
        assert isinstance(parent, ast.Call) and parent.func is node
        assert len(parent.args) == 1 and not parent.keywords
        assert isinstance(parents.get(parent), ast.Compare)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__setattr__"
        ):
            continue
        assert isinstance(node.func.value, ast.Name)
        assert node.func.value.id == "object"
        assert len(node.args) == 3 and not node.keywords
        field_name = node.args[1]
        assert isinstance(field_name, ast.Constant)
        assert type(field_name.value) is str
        assert field_name.value not in {
            "_authority",
            "_live_integer_slots",
            "_peak_live_integer_slots",
            "as_integer_ratio",
            "bit_length",
        }

    function_nodes = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )

    def lexical_scope(node: ast.AST):
        current = parents.get(node)
        while current is not None:
            if isinstance(
                current,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
            ):
                return current
            current = parents.get(current)
        return tree

    def resolve_function(name_node: ast.Name):
        scope = lexical_scope(name_node)
        while True:
            candidates = tuple(
                function
                for function in function_nodes
                if function.name == name_node.id
                and lexical_scope(function) is scope
            )
            if candidates:
                assert len(candidates) == 1
                return candidates[0]
            if scope is tree:
                return None
            scope = lexical_scope(scope)

    def lexical_body_nodes(root: ast.AST) -> set[ast.AST]:
        observed: set[ast.AST] = {root}
        pending = list(ast.iter_child_nodes(root))
        while pending:
            node = pending.pop()
            observed.add(node)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
            ):
                continue
            pending.extend(ast.iter_child_nodes(node))
        return observed

    def source_position(node: ast.AST) -> tuple[int, int]:
        return node.lineno, node.col_offset

    def unique_reaching_assignment(name: str, use: ast.AST) -> ast.AST:
        scope = lexical_scope(use)
        while True:
            body = lexical_body_nodes(scope) if scope is not tree else set(tree.body)
            assignments = tuple(
                node
                for node in body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and name
                in {
                    descendant.id
                    for target in (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else (node.target,)
                    )
                    for descendant in ast.walk(target)
                    if isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Store)
                }
            )
            stores = tuple(
                node
                for node in body
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id == name
            )
            special_bindings = tuple(
                node
                for node in body
                if (
                    isinstance(node, ast.arg)
                    and node.arg == name
                )
                or (
                    isinstance(node, ast.alias)
                    and (node.asname or node.name.rsplit(".", 1)[-1]) == name
                )
                or (
                    isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    )
                    and node is not scope
                    and node.name == name
                )
                or (
                    isinstance(node, ast.ExceptHandler)
                    and node.name == name
                )
                or (
                    isinstance(node, ast.MatchAs)
                    and node.name == name
                )
                or (
                    isinstance(node, ast.MatchStar)
                    and node.name == name
                )
                or (
                    isinstance(node, ast.MatchMapping)
                    and node.rest == name
                )
                or (
                    isinstance(node, (ast.Global, ast.Nonlocal))
                    and name in node.names
                )
            )
            if assignments or stores or special_bindings:
                assert not special_bindings
                assert len(assignments) == 1
                definition = assignments[0]
                targets = (
                    definition.targets
                    if isinstance(definition, ast.Assign)
                    else (definition.target,)
                )
                assert len(targets) == 1
                assert isinstance(targets[0], ast.Name)
                assert targets[0].id == name
                assert stores == (targets[0],)
                assert source_position(definition) < source_position(use)
                control_ancestors = (
                    ast.AsyncFor,
                    ast.AsyncWith,
                    ast.For,
                    ast.If,
                    ast.Match,
                    ast.Try,
                    ast.While,
                    ast.With,
                    ast.comprehension,
                )
                ancestor = definition
                while ancestor is not scope:
                    ancestor = parents[ancestor]
                    assert not isinstance(ancestor, control_ancestors)
                return definition
            assert scope is not tree
            scope = lexical_scope(scope)

    for consume_call in consume_calls:
        assert isinstance(consume_call.func, ast.Attribute)
        receiver = consume_call.func.value
        assert isinstance(receiver, ast.Name)
        definition = unique_reaching_assignment(receiver.id, consume_call)
        definition_value = definition.value
        assert isinstance(definition_value, ast.Call)
        assert isinstance(definition_value.func, ast.Name)
        assert definition_value.func.id == "_HopperExactIntegerArenaV2"
        assert len(definition_value.args) == 1
        assert not definition_value.keywords
        authority_argument = definition_value.args[0]
        assert isinstance(authority_argument, ast.Name)
        assert authority_argument.id == "HOPPER_RESOURCE_AUTHORITY_V2"

    operation_callbacks: set[ast.AST] = set()
    consumer_callbacks: set[ast.AST] = set()
    operation_consume_calls: dict[ast.AST, ast.Call] = {}
    named_callback_uses: list[tuple[ast.AST, ast.Name]] = []

    def resolve_callback(value: ast.AST, expected_parameters: int) -> ast.AST:
        if isinstance(value, ast.Lambda):
            callback = value
        else:
            assert isinstance(value, ast.Name)
            callback = resolve_function(value)
            assert callback is not None
            assert lexical_scope(callback) is lexical_scope(value)
            named_callback_uses.append((callback, value))
        arguments = callback.args
        positional = (*arguments.posonlyargs, *arguments.args)
        assert len(positional) == expected_parameters
        assert not arguments.kwonlyargs
        assert arguments.vararg is None
        assert arguments.kwarg is None
        assert not arguments.defaults
        assert not arguments.kw_defaults
        return callback

    for consume_call in consume_calls:
        assert not consume_call.args
        assert not isinstance(parents.get(consume_call), ast.Expr)
        keyword_map = {keyword.arg: keyword.value for keyword in consume_call.keywords}
        assert None not in keyword_map
        assert set(keyword_map) == {
            "result_bit_bounds",
            "operation",
            "consumer",
        }
        operation_callback = resolve_callback(keyword_map["operation"], 0)
        assert operation_callback not in operation_consume_calls
        operation_callbacks.add(operation_callback)
        operation_consume_calls[operation_callback] = consume_call
        consumer_callbacks.add(resolve_callback(keyword_map["consumer"], 1))

    for callback, use in named_callback_uses:
        loads = tuple(
            node
            for node in ast.walk(lexical_scope(use))
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and resolve_function(node) is callback
        )
        assert loads == (use,)

    callback_nodes = operation_callbacks | consumer_callbacks
    forbidden_callback_nodes = (
        ast.AugAssign,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
    )
    mutation_methods = {"add", "append", "extend", "insert", "setdefault", "update"}
    for callback in callback_nodes:
        body = lexical_body_nodes(callback)
        assert not any(
            isinstance(node, forbidden_callback_nodes) for node in body
        )
        assert not any(
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
            for node in body
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in mutation_methods
            for node in body
        )

    def expand_callback_callgraph(
        callbacks: set[ast.AST],
    ) -> tuple[set[ast.AST], set[ast.AST]]:
        owned_functions = {
            callback
            for callback in callbacks
            if isinstance(callback, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        owned_nodes = set().union(
            *(lexical_body_nodes(callback) for callback in callbacks)
        )
        changed = True
        while changed:
            changed = False
            for node in tuple(owned_nodes):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                ):
                    continue
                target = resolve_function(node.func)
                if target is None or target in callback_nodes | owned_functions:
                    continue
                owned_functions.add(target)
                owned_nodes.update(lexical_body_nodes(target))
                changed = True
        return owned_nodes, owned_functions

    operation_nodes, operation_owned_functions = expand_callback_callgraph(
        operation_callbacks
    )
    consumer_nodes, consumer_owned_functions = expand_callback_callgraph(
        consumer_callbacks
    )
    admitted_nodes = operation_nodes | consumer_nodes
    for node in operation_nodes | consumer_nodes:
        assert not isinstance(node, forbidden_callback_nodes)
        assert not (
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
        )
        assert not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in mutation_methods
        )
    for function in operation_owned_functions | consumer_owned_functions:
        call_sites = tuple(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and resolve_function(node.func) is function
        )
        assert call_sites
        assert all(call_site in admitted_nodes for call_site in call_sites)

    # bit_length() is the safe pre-allocation inspection used to derive a
    # conservative bound; unlike the sources below, it does not allocate the
    # exact numerator/denominator or a prospective arithmetic result.
    exact_source_attributes = {"as_integer_ratio"}
    exact_source_functions = {"divmod", "int", "pow"}
    risky_binary_operators = (
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in exact_source_attributes
        ):
            assert node in operation_nodes
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in exact_source_functions
        ):
            assert node in operation_nodes
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            risky_binary_operators,
        ):
            assert node in operation_nodes

    forbidden_operation_accumulators = {
        "accumulate",
        "prod",
        "reduce",
        "sum",
    }
    forbidden_numeric_hooks = {
        "__abs__",
        "__add__",
        "__aenter__",
        "__aexit__",
        "__bool__",
        "__divmod__",
        "__enter__",
        "__exit__",
        "__format__",
        "__floordiv__",
        "__getattr__",
        "__getattribute__",
        "__index__",
        "__int__",
        "__invert__",
        "__lshift__",
        "__mod__",
        "__mul__",
        "__neg__",
        "__or__",
        "__pos__",
        "__pow__",
        "__radd__",
        "__rdivmod__",
        "__rfloordiv__",
        "__rlshift__",
        "__rmod__",
        "__rmul__",
        "__ror__",
        "__round__",
        "__rpow__",
        "__rrshift__",
        "__rshift__",
        "__rsub__",
        "__rxor__",
        "__sub__",
        "__trunc__",
        "__xor__",
    }
    assert forbidden_numeric_hooks.isdisjoint(observed_attributes)
    builtin_exact_source_names = exact_source_functions
    for source_name in (
        builtin_exact_source_names
        | exact_source_attributes
        | forbidden_numeric_hooks
        | {
            "ValueError",
            "bit_length",
            "float",
            "max",
            "min",
            "object",
            "type",
        }
    ):
        assert not any(
            (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id == source_name
            )
            or (
                isinstance(node, ast.arg)
                and node.arg == source_name
            )
            or (
                isinstance(node, ast.alias)
                and (node.asname or node.name.rsplit(".", 1)[-1]) == source_name
            )
            or (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == source_name
            )
            or (
                isinstance(node, ast.ExceptHandler)
                and node.name == source_name
            )
            or (
                isinstance(node, (ast.MatchAs, ast.MatchStar))
                and node.name == source_name
            )
            or (
                isinstance(node, ast.MatchMapping)
                and node.rest == source_name
            )
            for node in ast.walk(tree)
        )
    assert not any(
        (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name
            in {"as_integer_ratio", "bit_length"} | forbidden_numeric_hooks
        )
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr
            in {"as_integer_ratio", "bit_length"} | forbidden_numeric_hooks
        )
        for node in ast.walk(tree)
    )

    def expression_dependencies(
        expression: ast.AST,
        scope: ast.AST,
        use: ast.AST,
    ) -> set[ast.AST]:
        observed = set(ast.walk(expression))
        changed = True
        scope_body = lexical_body_nodes(scope) if scope is not tree else set(ast.walk(tree))
        while changed:
            changed = False
            loaded = {
                node.id
                for node in observed
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            for node in scope_body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                if source_position(node) >= source_position(use):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                names = {
                    target.id for target in targets if isinstance(target, ast.Name)
                }
                if not (names & loaded):
                    continue
                for name in names & loaded:
                    reaching_definitions = tuple(
                        candidate
                        for candidate in scope_body
                        if isinstance(candidate, (ast.Assign, ast.AnnAssign))
                        and source_position(candidate) < source_position(use)
                        and name
                        in {
                            target.id
                            for target in (
                                candidate.targets
                                if isinstance(candidate, ast.Assign)
                                else (candidate.target,)
                            )
                            if isinstance(target, ast.Name)
                        }
                    )
                    assert reaching_definitions == (node,)
                before = len(observed)
                observed.update(ast.walk(node.value))
                changed = changed or len(observed) != before
        return observed

    for operation in operation_callbacks:
        owned_nodes, owned_functions = expand_callback_callgraph({operation})
        expected_owned_functions = (
            {operation}
            if isinstance(operation, (ast.FunctionDef, ast.AsyncFunctionDef))
            else set()
        )
        assert owned_functions == expected_owned_functions
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            and node is not operation
            for node in owned_nodes
        )
        assert not any(
            isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
            for node in owned_nodes
        )
        assert not any(
            isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id in forbidden_operation_accumulators
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in forbidden_operation_accumulators
                )
            )
            for node in owned_nodes
        )
        for node in owned_nodes:
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                assert node.func.id in builtin_exact_source_names
            else:
                assert isinstance(node.func, ast.Attribute)
                assert node.func.attr in exact_source_attributes
                assert isinstance(node.func.value, ast.Name)
                enclosing_scope = lexical_scope(operation)
                assert enclosing_scope is not tree
                enclosing_parameters = {
                    argument.arg
                    for argument in (
                        *enclosing_scope.args.posonlyargs,
                        *enclosing_scope.args.args,
                        *enclosing_scope.args.kwonlyargs,
                    )
                }
                assert node.func.value.id in enclosing_parameters
        allocating_nodes = tuple(
            node
            for node in owned_nodes
            if (
                isinstance(node, (ast.BinOp, ast.UnaryOp))
                or (
                    isinstance(node, ast.Call)
                    and (
                        (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr in exact_source_attributes
                        )
                        or (
                            isinstance(node.func, ast.Name)
                            and node.func.id in exact_source_functions
                        )
                    )
                )
            )
        )
        assert len(allocating_nodes) == 1
        allocating_node = allocating_nodes[0]
        assert not any(
            isinstance(descendant, (ast.BinOp, ast.UnaryOp))
            and descendant is not allocating_node
            for descendant in ast.walk(allocating_node)
        )
        returns = tuple(
            node for node in owned_nodes if isinstance(node, ast.Return)
        )
        if isinstance(operation, ast.Lambda):
            returned = operation.body
        else:
            assert len(returns) == 1
            assert returns[0].value is not None
            returned = returns[0].value
        direct_result_names = {
            target.id
            for node in owned_nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is allocating_node
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
        }
        returned_elements = (
            tuple(returned.elts)
            if isinstance(returned, ast.Tuple)
            else (returned,)
        )
        assert any(
            element is allocating_node
            or (
                isinstance(element, ast.Name)
                and element.id in direct_result_names
            )
            for element in returned_elements
        )
        consume_call = operation_consume_calls[operation]
        keyword_map = {
            keyword.arg: keyword.value for keyword in consume_call.keywords
        }
        bound_expression = keyword_map["result_bit_bounds"]
        assert isinstance(bound_expression, ast.Tuple)
        assert bound_expression.elts
        assert not any(isinstance(element, ast.Starred) for element in bound_expression.elts)
        if (
            isinstance(allocating_node, ast.Call)
            and isinstance(allocating_node.func, ast.Attribute)
            and allocating_node.func.attr == "as_integer_ratio"
        ):
            enclosing_scope = lexical_scope(operation)
            assert enclosing_scope is not tree
            receiver = allocating_node.func.value
            assert isinstance(receiver, ast.Name)
            assert not any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id == receiver.id
                for node in lexical_body_nodes(enclosing_scope)
            )
            assert not any(
                (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    and node.id == receiver.id
                )
                or (isinstance(node, ast.arg) and node.arg == receiver.id)
                for node in owned_nodes
            )
            assert len(bound_expression.elts) == 2
            assert all(
                isinstance(element, ast.Constant)
                and type(element.value) is int
                and 1_075 <= element.value <= 262_144
                for element in bound_expression.elts
            )
        else:
            dependency_nodes = set().union(
                *(
                    expression_dependencies(
                        element,
                        lexical_scope(consume_call),
                        consume_call,
                    )
                    for element in bound_expression.elts
                )
            )
            bit_length_calls = tuple(
                node
                for node in dependency_nodes
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "bit_length"
            )
            assert bit_length_calls

            scope = lexical_scope(consume_call)
            scope_body = (
                lexical_body_nodes(scope)
                if scope is not tree
                else set(ast.walk(tree))
            )
            definitions: dict[str, list[ast.AST]] = {}
            for node in scope_body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                if source_position(node) >= source_position(consume_call):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    definitions.setdefault(target.id, []).append(node.value)

            control_flow_ancestors = (
                ast.AsyncFor,
                ast.AsyncWith,
                ast.For,
                ast.If,
                ast.IfExp,
                ast.Match,
                ast.Try,
                ast.While,
                ast.With,
                ast.comprehension,
            )
            for bit_length_call in bit_length_calls:
                assert lexical_scope(bit_length_call) is scope
                assert source_position(bit_length_call) < source_position(consume_call)
                receiver = bit_length_call.func.value
                assert isinstance(receiver, (ast.Name, ast.Subscript))
                receiver_names = {
                    node.id
                    for node in ast.walk(receiver)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                }
                assert receiver_names
                assert not any(
                    (
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, (ast.Store, ast.Del))
                        and node.id in receiver_names
                    )
                    or (
                        isinstance(node, ast.arg)
                        and node.arg in receiver_names
                    )
                    for node in owned_nodes
                )

                ancestor = bit_length_call
                metadata_assignment = None
                while ancestor is not scope:
                    ancestor = parents[ancestor]
                    assert not isinstance(ancestor, control_flow_ancestors)
                    if metadata_assignment is None and isinstance(
                        ancestor,
                        (ast.Assign, ast.AnnAssign),
                    ):
                        metadata_assignment = ancestor
                assert metadata_assignment is not None
                assert lexical_scope(metadata_assignment) is scope

                consume_ancestor = consume_call
                while consume_ancestor is not scope:
                    consume_ancestor = parents[consume_ancestor]
                    assert not isinstance(
                        consume_ancestor,
                        control_flow_ancestors,
                    )

                for later in scope_body:
                    if not (
                        source_position(bit_length_call)
                        < source_position(later)
                        < source_position(consume_call)
                    ):
                        continue
                    assert not (
                        isinstance(later, ast.Name)
                        and isinstance(later.ctx, (ast.Store, ast.Del))
                        and later.id in receiver_names
                    )
                    assert not (
                        isinstance(later, (ast.Attribute, ast.Subscript))
                        and isinstance(later.ctx, (ast.Store, ast.Del))
                    )
                    if not isinstance(later, ast.Call):
                        continue
                    if later is consume_call or (
                        isinstance(later.func, ast.Attribute)
                        and later.func.attr == "bit_length"
                    ):
                        continue
                    raise AssertionError(
                        "metadata receiver version window contains a call"
                    )

            def evaluate_bound(
                expression: ast.AST,
                bit_lengths: dict[str, int],
            ) -> int:
                if isinstance(expression, ast.Constant):
                    assert type(expression.value) is int
                    return expression.value
                if isinstance(expression, ast.Name):
                    if expression.id == "HOPPER_EXACT_MAX_INTEGER_BITS_V2":
                        return 262_144
                    assert expression.id in definitions
                    assert len(definitions[expression.id]) == 1
                    return evaluate_bound(
                        definitions[expression.id][0],
                        bit_lengths,
                    )
                if isinstance(expression, ast.Call):
                    if (
                        isinstance(expression.func, ast.Attribute)
                        and expression.func.attr == "bit_length"
                    ):
                        assert not expression.args and not expression.keywords
                        key = ast.dump(
                            expression.func.value,
                            include_attributes=False,
                        )
                        assert key in bit_lengths
                        return bit_lengths[key]
                    assert isinstance(expression.func, ast.Name)
                    assert expression.func.id in {"max", "min"}
                    assert not expression.keywords and expression.args
                    values = tuple(
                        evaluate_bound(argument, bit_lengths)
                        for argument in expression.args
                    )
                    return (
                        max(values)
                        if expression.func.id == "max"
                        else min(values)
                    )
                if isinstance(expression, ast.UnaryOp):
                    value = evaluate_bound(expression.operand, bit_lengths)
                    if isinstance(expression.op, ast.UAdd):
                        return value
                    assert isinstance(expression.op, ast.USub)
                    return -value
                assert isinstance(expression, ast.BinOp)
                left = evaluate_bound(expression.left, bit_lengths)
                right = evaluate_bound(expression.right, bit_lengths)
                if isinstance(expression.op, ast.Add):
                    return left + right
                if isinstance(expression.op, ast.Sub):
                    return left - right
                if isinstance(expression.op, ast.Mult):
                    return left * right
                assert isinstance(expression.op, ast.FloorDiv)
                assert right != 0
                return left // right

            def operand_key(expression: ast.AST) -> tuple[str | None, int]:
                if isinstance(expression, ast.Constant):
                    assert type(expression.value) is int
                    return None, max(1, abs(expression.value).bit_length())
                return ast.dump(expression, include_attributes=False), 1

            def ordered_symbol(kind: str, *values: object) -> tuple[object, ...]:
                return (kind, *sorted(values, key=repr))

            def bit_symbol(expression: ast.AST) -> tuple[object, ...]:
                key, fixed_bits = operand_key(expression)
                return (
                    ("constant", fixed_bits)
                    if key is None
                    else ("bit_length", key)
                )

            def normalize_bound(expression: ast.AST) -> tuple[object, ...]:
                if isinstance(expression, ast.Constant):
                    assert type(expression.value) is int
                    return ("constant", expression.value)
                if isinstance(expression, ast.Name):
                    if expression.id == "HOPPER_EXACT_MAX_INTEGER_BITS_V2":
                        return ("constant", 262_144)
                    assert expression.id in definitions
                    assert len(definitions[expression.id]) == 1
                    stores = tuple(
                        node
                        for node in scope_body
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, (ast.Store, ast.Del))
                        and node.id == expression.id
                    )
                    assert len(stores) == 1
                    assert not any(
                        isinstance(node, ast.arg) and node.arg == expression.id
                        for node in scope_body
                    )
                    assert not any(
                        isinstance(node, ast.alias)
                        and (node.asname or node.name.rsplit(".", 1)[-1])
                        == expression.id
                        for node in scope_body
                    )
                    return normalize_bound(definitions[expression.id][0])
                if isinstance(expression, ast.Call):
                    if isinstance(expression.func, ast.Attribute):
                        assert expression.func.attr == "bit_length"
                        assert not expression.args and not expression.keywords
                        return bit_symbol(expression.func.value)
                    assert isinstance(expression.func, ast.Name)
                    assert expression.func.id == "max"
                    assert len(expression.args) == 2
                    assert not expression.keywords
                    return ordered_symbol(
                        "max",
                        *(normalize_bound(argument) for argument in expression.args),
                    )
                assert isinstance(expression, ast.BinOp)
                assert isinstance(expression.op, ast.Add)
                return ordered_symbol(
                    "add",
                    normalize_bound(expression.left),
                    normalize_bound(expression.right),
                )

            if isinstance(allocating_node, ast.UnaryOp):
                operand_expressions = (allocating_node.operand,)
                expected_bounds = lambda bits: (bits[0] + 1,)
            elif isinstance(allocating_node, ast.BinOp):
                assert isinstance(
                    allocating_node.op,
                    (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod),
                )
                operand_expressions = (
                    allocating_node.left,
                    allocating_node.right,
                )
                if isinstance(allocating_node.op, (ast.Add, ast.Sub)):
                    expected_bounds = lambda bits: (max(bits) + 1,)
                elif isinstance(allocating_node.op, ast.Mult):
                    expected_bounds = lambda bits: (bits[0] + bits[1],)
                elif isinstance(allocating_node.op, ast.FloorDiv):
                    expected_bounds = lambda bits: (bits[0],)
                else:
                    expected_bounds = lambda bits: (bits[1],)
            else:
                assert isinstance(allocating_node, ast.Call)
                assert isinstance(allocating_node.func, ast.Name)
                assert allocating_node.func.id == "divmod"
                assert len(allocating_node.args) == 2
                assert not allocating_node.keywords
                operand_expressions = tuple(allocating_node.args)
                expected_bounds = lambda bits: (bits[0], bits[1])

            operand_symbols = tuple(
                bit_symbol(expression) for expression in operand_expressions
            )
            if isinstance(allocating_node, ast.UnaryOp):
                expected_symbols = (
                    ordered_symbol(
                        "add",
                        operand_symbols[0],
                        ("constant", 1),
                    ),
                )
            elif isinstance(allocating_node, ast.BinOp):
                if isinstance(allocating_node.op, (ast.Add, ast.Sub)):
                    expected_symbols = (
                        ordered_symbol(
                            "add",
                            ordered_symbol("max", *operand_symbols),
                            ("constant", 1),
                        ),
                    )
                elif isinstance(allocating_node.op, ast.Mult):
                    expected_symbols = (ordered_symbol("add", *operand_symbols),)
                elif isinstance(allocating_node.op, ast.FloorDiv):
                    expected_symbols = (operand_symbols[0],)
                else:
                    expected_symbols = (operand_symbols[1],)
            else:
                expected_symbols = operand_symbols
            declared_symbols = tuple(
                normalize_bound(element) for element in bound_expression.elts
            )
            assert declared_symbols == expected_symbols

            operand_specs = tuple(operand_key(item) for item in operand_expressions)
            variable_keys = tuple(
                key for key, _fixed_bits in operand_specs if key is not None
            )
            bit_cases: list[dict[str, int]] = []
            for common_bits in (1, 64, 131_073, 262_144):
                bit_cases.append({key: common_bits for key in variable_keys})
            for high_key in variable_keys:
                bit_cases.append(
                    {
                        key: (131_073 if key == high_key else 1)
                        for key in variable_keys
                    }
                )
            for bit_lengths in bit_cases:
                operand_bits = tuple(
                    fixed_bits if key is None else bit_lengths[key]
                    for key, fixed_bits in operand_specs
                )
                declared = tuple(
                    evaluate_bound(element, bit_lengths)
                    for element in bound_expression.elts
                )
                expected = expected_bounds(operand_bits)
                assert len(declared) == len(expected)
                assert all(
                    type(value) is int and value >= required and value > 0
                    for value, required in zip(declared, expected, strict=True)
                )

    def assigned_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return set().union(*(assigned_names(item) for item in target.elts))
        return set()

    def local_assignment_target(target: ast.AST) -> bool:
        if isinstance(target, ast.Name):
            return True
        if isinstance(target, (ast.Tuple, ast.List)):
            return bool(target.elts) and all(
                local_assignment_target(item) for item in target.elts
            )
        return False

    exact_integer_parameter_positions = {
        "_classify_exact_quadratic_overlap_v2": tuple(range(7)),
        "_exact_qsqrt_sign_v2": tuple(range(3)),
        "_compare_binary64_to_qsqrt_v2": (1, 2, 3, 4),
        "_exact_parabola_clearance_at_rational_v2": (5, 6),
    }
    exact_float_parameter_positions = {
        "_exact_disk_segment_square_relation_v2": tuple(range(10)),
        "_exact_candidate_index_bounds_v2": tuple(range(4)),
        "_compare_binary64_to_qsqrt_v2": (0,),
        "_exact_parabola_clearance_at_rational_v2": tuple(range(5)),
    }

    def exact_type_rejection_guard_name(
        statement: ast.AST,
        expected_type_name: str,
    ) -> str | None:
        if not isinstance(statement, ast.If) or statement.orelse:
            return None
        test = statement.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == expected_type_name
            and isinstance(test.left, ast.Call)
            and isinstance(test.left.func, ast.Name)
            and test.left.func.id == "type"
            and len(test.left.args) == 1
            and not test.left.keywords
            and isinstance(test.left.args[0], ast.Name)
        ):
            return None
        if len(statement.body) != 1 or not isinstance(statement.body[0], ast.Raise):
            return None
        raised = statement.body[0].exc
        if not (
            isinstance(raised, ast.Call)
            and isinstance(raised.func, ast.Name)
            and raised.func.id == "ValueError"
            and len(raised.args) == 1
            and not raised.keywords
            and isinstance(raised.args[0], ast.Constant)
            and raised.args[0].value == "hopper_numeric_contract_mismatch"
        ):
            return None
        return test.left.args[0].id

    def loads_any_name(node: ast.AST, names: set[str]) -> bool:
        return any(
            isinstance(descendant, ast.Name)
            and isinstance(descendant.ctx, ast.Load)
            and descendant.id in names
            for descendant in ast.walk(node)
        )

    def lexical_loads_any_name(node: ast.AST, names: set[str]) -> bool:
        return any(
            isinstance(descendant, ast.Name)
            and isinstance(descendant.ctx, ast.Load)
            and descendant.id in names
            for descendant in lexical_body_nodes(node)
        )

    indirect_binding_nodes = (
        ast.AsyncFor,
        ast.For,
        ast.Match,
        ast.NamedExpr,
        ast.comprehension,
    )
    top_level_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for expected_type_name, parameter_positions in (
        ("int", exact_integer_parameter_positions),
        ("float", exact_float_parameter_positions),
    ):
        for function_name, positions in parameter_positions.items():
            function = top_level_functions[function_name]
            positional = (*function.args.posonlyargs, *function.args.args)
            protected_parameters = {
                positional[position].arg for position in positions
            }
            guards = {
                name: statement
                for statement in function.body
                if (
                    name := exact_type_rejection_guard_name(
                        statement,
                        expected_type_name,
                    )
                )
                is not None
            }
            assert set(guards) == protected_parameters
            for parameter_name, guard in guards.items():
                guard_nodes = set(ast.walk(guard))
                for load in lexical_body_nodes(function):
                    if not (
                        isinstance(load, ast.Name)
                        and isinstance(load.ctx, ast.Load)
                        and load.id == parameter_name
                        and load not in guard_nodes
                    ):
                        continue
                    assert source_position(guard) < source_position(load)
    for function_name, positions in exact_integer_parameter_positions.items():
        function = top_level_functions[function_name]
        positional = (*function.args.posonlyargs, *function.args.args)
        tainted = {positional[position].arg for position in positions}
        body = lexical_body_nodes(function)
        changed = True
        while changed:
            changed = False
            for node in body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                loaded_names = {
                    descendant.id
                    for descendant in ast.walk(value)
                    if isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                }
                if not (loaded_names & tainted):
                    continue
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "bit_length"
                ):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                assert all(local_assignment_target(target) for target in targets)
                for target in targets:
                    new_names = assigned_names(target)
                    if not new_names <= tainted:
                        tainted.update(new_names)
                        changed = True
        assert not any(
            isinstance(node, indirect_binding_nodes)
            and loads_any_name(node, tainted)
            for node in body
        )
        assert not any(
            isinstance(node, ast.AugAssign)
            and (
                any(
                    isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, (ast.Store, ast.Del))
                    and descendant.id in tainted
                    for descendant in ast.walk(node.target)
                )
                or loads_any_name(node.value, tainted)
            )
            for node in body
        )
        for nested in ast.walk(function):
            if nested is function or not isinstance(
                nested,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
            ):
                continue
            if lexical_loads_any_name(nested, tainted):
                assert nested in operation_callbacks
        for node in body:
            if isinstance(node, ast.BinOp):
                loaded_names = {
                    descendant.id
                    for descendant in ast.walk(node)
                    if isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                }
                if loaded_names & tainted:
                    assert node in operation_nodes
            if isinstance(node, ast.UnaryOp):
                loaded_names = {
                    descendant.id
                    for descendant in ast.walk(node)
                    if isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                }
                if loaded_names & tainted:
                    assert node in operation_nodes
            if isinstance(node, ast.Call) and node not in operation_nodes:
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {"bit_length", "consume_admitted_integers"}
                ):
                    continue
                loaded_names = {
                    descendant.id
                    for descendant in ast.walk(node)
                    if isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                }
                if loaded_names & tainted:
                    assert isinstance(node.func, ast.Name)
                    assert node.func.id in {"isinstance", "type"}

    def callback_parameter_name(callback: ast.AST) -> str:
        arguments = callback.args
        return (*arguments.posonlyargs, *arguments.args)[0].arg

    for consumer in consumer_callbacks:
        body = lexical_body_nodes(consumer)
        all_consumer_nodes = set(ast.walk(consumer))
        lease_aliases = {callback_parameter_name(consumer)}
        tainted_names = set(lease_aliases)
        changed = True
        while changed:
            changed = False
            for node in body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                loaded_names = {
                    descendant.id
                    for descendant in ast.walk(value)
                    if isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                }
                if not (loaded_names & tainted_names):
                    continue
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "bit_length"
                ):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                assert all(local_assignment_target(target) for target in targets)
                for target in targets:
                    new_names = assigned_names(target)
                    if not new_names <= tainted_names:
                        tainted_names.update(new_names)
                        changed = True
                    if (
                        isinstance(target, ast.Name)
                        and isinstance(value, ast.Name)
                        and value.id in lease_aliases
                    ):
                        lease_aliases.add(target.id)
        alias_loads = tuple(
            node
            for node in all_consumer_nodes
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in tainted_names
        )
        assert alias_loads
        assert not any(
            isinstance(node, indirect_binding_nodes)
            and lexical_scope(node) is consumer
            and loads_any_name(node, tainted_names)
            for node in all_consumer_nodes
        )
        for load in alias_loads:
            nested_scope = lexical_scope(load)
            if nested_scope is not consumer:
                assert nested_scope in operation_callbacks
        for node in all_consumer_nodes:
            if not isinstance(node, (ast.BinOp, ast.UnaryOp)):
                continue
            loaded_names = {
                descendant.id
                for descendant in ast.walk(node)
                if isinstance(descendant, ast.Name)
                and isinstance(descendant.ctx, ast.Load)
            }
            if loaded_names & tainted_names:
                assert node in operation_nodes
        for node in all_consumer_nodes:
            if not isinstance(node, ast.Call) or node in operation_nodes:
                continue
            loaded_names = {
                descendant.id
                for descendant in ast.walk(node)
                if isinstance(descendant, ast.Name)
                and isinstance(descendant.ctx, ast.Load)
            }
            if not (loaded_names & tainted_names):
                continue
            assert (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {"bit_length", "consume_admitted_integers"}
            ) or (
                isinstance(node.func, ast.Name)
                and node.func.id in {"isinstance", "type"}
            )
        for node in body:
            if isinstance(node, ast.Return) and node.value is not None:
                assert not any(
                    isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                    and descendant.id in lease_aliases
                    and not isinstance(parents.get(descendant), ast.Subscript)
                    for descendant in ast.walk(node.value)
                )
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if value is None or not any(
                    isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                    and descendant.id in tainted_names
                    for descendant in ast.walk(value)
                ):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                assert all(isinstance(target, ast.Name) for target in targets)

    arena_backed_functions = {
        function
        for function in function_nodes
        if any(call in lexical_body_nodes(function) for call in consume_calls)
    }
    changed = True
    while changed:
        changed = False
        for function in function_nodes:
            if function in arena_backed_functions:
                continue
            body = lexical_body_nodes(function)
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and resolve_function(node.func) in arena_backed_functions
                for node in body
            ):
                arena_backed_functions.add(function)
                changed = True

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and resolve_function(node.func) in arena_backed_functions
        ):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Return):
            continue
        assert isinstance(parent, (ast.Assign, ast.AnnAssign))
        assert parent.value is node

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and resolve_function(node) in arena_backed_functions
        ):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        assert (
            isinstance(parent, ast.keyword)
            and parent.arg == "consumer"
            and resolve_function(node) in consumer_callbacks
            and isinstance(parents.get(parent), ast.Call)
            and parents[parent] in consume_calls
        )

    for consume_call in consume_calls:
        parent = parents.get(consume_call)
        if isinstance(parent, ast.Lambda):
            assert parent in consumer_callbacks
        else:
            assert isinstance(parent, ast.Return)

    function_bodies = {
        function: lexical_body_nodes(function) for function in function_nodes
    }
    escaped_names_by_function = {function: set() for function in function_nodes}
    for function, body in function_bodies.items():
        for node in body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and resolve_function(value.func) in arena_backed_functions
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            assert all(local_assignment_target(target) for target in targets)
            escaped_names_by_function[function].update(
                set().union(*(assigned_names(target) for target in targets))
            )

    changed = True
    while changed:
        changed = False
        for function, body in function_bodies.items():
            escaped_exact_names = escaped_names_by_function[function]
            for node in body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                loads_escaped = any(
                    isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                    and descendant.id in escaped_exact_names
                    for descendant in ast.walk(value)
                )
                is_bound_metadata = (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "bit_length"
                )
                if not loads_escaped or is_bound_metadata:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                assert all(local_assignment_target(target) for target in targets)
                new_names = set().union(*(assigned_names(target) for target in targets))
                if not new_names <= escaped_exact_names:
                    escaped_exact_names.update(new_names)
                    changed = True

            for node in body:
                if not isinstance(node, ast.NamedExpr):
                    continue
                if not any(
                    isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                    and descendant.id in escaped_exact_names
                    for descendant in ast.walk(node.value)
                ):
                    continue
                assert isinstance(node.target, ast.Name)
                if node.target.id not in escaped_exact_names:
                    escaped_exact_names.add(node.target.id)
                    changed = True

            for node in body:
                if isinstance(node, (ast.For, ast.AsyncFor)) and loads_any_name(
                    node.iter,
                    escaped_exact_names,
                ):
                    assert (
                        isinstance(node.iter, ast.Call)
                        and isinstance(node.iter.func, ast.Name)
                        and node.iter.func.id == "range"
                    )
                    assert local_assignment_target(node.target)
                    target_names = assigned_names(node.target)
                    if not target_names <= escaped_exact_names:
                        escaped_exact_names.update(target_names)
                        changed = True
                if isinstance(node, (ast.comprehension, ast.Match)) and loads_any_name(
                    node,
                    escaped_exact_names,
                ):
                    raise AssertionError("escaped exact value used by indirect binding")

            positional_parameters_cache: dict[ast.AST, tuple[ast.arg, ...]] = {}
            for node in body:
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                ):
                    continue
                target = resolve_function(node.func)
                if target is None:
                    continue
                positional_parameters = positional_parameters_cache.setdefault(
                    target,
                    (*target.args.posonlyargs, *target.args.args),
                )
                argument_pairs: list[tuple[ast.AST, ast.arg | None]] = []
                for index, argument in enumerate(node.args):
                    if isinstance(argument, ast.Starred):
                        parameter = target.args.vararg
                        argument = argument.value
                    elif index < len(positional_parameters):
                        parameter = positional_parameters[index]
                    else:
                        parameter = target.args.vararg
                    argument_pairs.append((argument, parameter))
                parameter_by_name = {
                    parameter.arg: parameter
                    for parameter in (
                        *positional_parameters,
                        *target.args.kwonlyargs,
                    )
                }
                for keyword in node.keywords:
                    parameter = (
                        target.args.kwarg
                        if keyword.arg is None
                        else parameter_by_name.get(keyword.arg)
                    )
                    argument_pairs.append((keyword.value, parameter))
                for argument, parameter in argument_pairs:
                    if not any(
                        isinstance(descendant, ast.Name)
                        and isinstance(descendant.ctx, ast.Load)
                        and descendant.id in escaped_exact_names
                        for descendant in ast.walk(argument)
                    ):
                        continue
                    assert target not in arena_backed_functions
                    assert parameter is not None
                    target_names = escaped_names_by_function[target]
                    if parameter.arg not in target_names:
                        target_names.add(parameter.arg)
                        changed = True

            for target in function_nodes:
                scope = lexical_scope(target)
                while scope is not tree and scope is not function:
                    scope = lexical_scope(scope)
                if scope is not function:
                    continue
                target_body = lexical_body_nodes(target)
                locally_bound = {
                    argument.arg
                    for argument in (
                        *target.args.posonlyargs,
                        *target.args.args,
                        *target.args.kwonlyargs,
                    )
                }
                if target.args.vararg is not None:
                    locally_bound.add(target.args.vararg.arg)
                if target.args.kwarg is not None:
                    locally_bound.add(target.args.kwarg.arg)
                locally_bound.update(
                    name.id
                    for name in target_body
                    if isinstance(name, ast.Name)
                    and isinstance(name.ctx, (ast.Store, ast.Del))
                )
                captured = {
                    name.id
                    for name in target_body
                    if isinstance(name, ast.Name)
                    and isinstance(name.ctx, ast.Load)
                    and name.id in escaped_exact_names
                    and name.id not in locally_bound
                }
                defaults = (
                    *target.args.defaults,
                    *(default for default in target.args.kw_defaults if default is not None),
                )
                assert not any(
                    loads_any_name(default, escaped_exact_names)
                    for default in defaults
                )
                target_names = escaped_names_by_function[target]
                if not captured <= target_names:
                    target_names.update(captured)
                    changed = True

    safe_escaped_call_names = {"Cell", "isinstance", "len", "range", "type"}
    safe_escaped_call_attributes = {"bit_length"}
    assert sum(
        1
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "path_planner.core"
        for alias in node.names
        if alias.name == "Cell" and (alias.asname or alias.name) == "Cell"
    ) == 1
    assert not any(
        (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id == "Cell"
        )
        or (isinstance(node, ast.arg) and node.arg == "Cell")
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == "Cell"
        )
        for node in ast.walk(tree)
    )
    for builtin_name in safe_escaped_call_names - {"Cell"}:
        assert not any(
            (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id == builtin_name
            )
            or (isinstance(node, ast.arg) and node.arg == builtin_name)
            or (
                isinstance(node, ast.alias)
                and (node.asname or node.name.rsplit(".", 1)[-1]) == builtin_name
            )
            or (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == builtin_name
            )
            for node in ast.walk(tree)
        )
    for function, body in function_bodies.items():
        escaped_exact_names = escaped_names_by_function[function]
        if not escaped_exact_names:
            continue
        assert not any(
            isinstance(node, ast.Lambda)
            and lexical_scope(node) is function
            and any(
                isinstance(descendant, ast.Name)
                and isinstance(descendant.ctx, ast.Load)
                and descendant.id in escaped_exact_names
                for descendant in ast.walk(node)
            )
            for node in ast.walk(function)
        )
        assert not any(
            isinstance(node, (ast.Global, ast.Nonlocal))
            for node in body
        )
        for operation in operation_callbacks:
            scope = lexical_scope(operation)
            while scope is not tree and scope is not function:
                scope = lexical_scope(scope)
            if scope is not function:
                continue
            assert not any(
                isinstance(descendant, ast.Name)
                and isinstance(descendant.ctx, ast.Load)
                and descendant.id in escaped_exact_names
                for descendant in ast.walk(operation)
            )
        for node in body:
            if isinstance(node, ast.AugAssign):
                assert not (
                    assigned_names(node.target) & escaped_exact_names
                )
            loaded_names = {
                descendant.id
                for descendant in ast.walk(node)
                if isinstance(descendant, ast.Name)
                and isinstance(descendant.ctx, ast.Load)
            }
            if not (loaded_names & escaped_exact_names):
                continue
            assert not isinstance(
                node,
                (ast.AugAssign, ast.Await, ast.Yield, ast.YieldFrom),
            )
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else (node.target,)
                )
                assert all(local_assignment_target(target) for target in targets)
            if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                assert node in operation_nodes
            if isinstance(node, ast.Call) and node not in operation_nodes:
                target = (
                    resolve_function(node.func)
                    if isinstance(node.func, ast.Name)
                    else None
                )
                if target is not None:
                    continue
                if isinstance(node.func, ast.Name):
                    assert node.func.id in safe_escaped_call_names
                else:
                    assert isinstance(node.func, ast.Attribute)
                    assert node.func.attr in safe_escaped_call_attributes

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "as_integer_ratio"
        for node in operation_nodes
    )
