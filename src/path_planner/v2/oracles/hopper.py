from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, floor, fsum, isfinite, nextafter, pi, sin

import numpy as np

from path_planner.core import Cell, WorldPoint
from path_planner.v2.ballistics import (
    BallisticSampleV2,
    BallisticStartV2,
    LandingCellMassV2,
)
from path_planner.v2.contracts import (
    FailureCategoryV2,
    PlatformKindV2,
    PoseStateV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.hopper_authority import (
    HOPPER_EXACT_MAX_INTEGER_BITS_V2,
    HOPPER_MAX_REPLAY_STEPS_V2,
    HOPPER_RESOURCE_AUTHORITY_V2,
    _call_captured_ballistic_helper_v2,
    _call_captured_landing_helper_v2,
    _hopper_parameter_set_in_memory_token_v2,
    _HopperExactIntegerArenaV2,
    _lookup_hopper_parameter_set_v2,
    _require_canonical_resource_authority_v2,
    HopperParameterSetRecordV2,
)
from path_planner.v2.profiles import HopperProfileV2, PlatformProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    SafetyQueryV2,
    TerrainSnapshotV2,
    snapshot_hash,
)


HOPPER_JUMP_VALIDATOR_ID_V2 = "path-planner-v2-hopper-jump-l2/v1"
HOPPER_REPLAY_COUNTER_IDS_V1 = (
    "hopper_arc_interval_count/v1",
    "hopper_arc_candidate_width/v1",
    "hopper_arc_candidate_height/v1",
    "hopper_arc_interval_cartesian_product/v1",
    "hopper_arc_distinct_cell_count/v1",
    "hopper_arc_interval_cell_visit_count/v1",
)
HOPPER_JUMP_REASON_CODES_V1 = (
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

_CANDIDATE_SCHEMA = "hopper-jump-candidate/v1"
_RESULT_SCHEMA = "hopper-jump-validation-result/v1"
_STAGES = (
    "launch_validation",
    "arc_candidate_enumeration",
    "replay_work",
    "arc_validation",
    "landing_probability",
    "landing_validation",
    "stop_validation",
)


def _copy_platform_profile_v2(value: object) -> PlatformProfileV2:
    if type(value) is not PlatformProfileV2:
        raise TypeError("hopper profile platform must be exact PlatformProfileV2")
    return PlatformProfileV2(
        profile_id=value.profile_id,
        platform_kind=value.platform_kind,
        capability_revision=value.capability_revision,
        simulation_proxy=value.simulation_proxy,
        max_traversable_slope_deg=value.max_traversable_slope_deg,
        goal_position_tolerance_m=value.goal_position_tolerance_m,
        goal_heading_tolerance_rad=value.goal_heading_tolerance_rad,
        schema_version=value.schema_version,
    )


def _copy_hopper_profile_v2(value: object) -> HopperProfileV2:
    if type(value) is not HopperProfileV2:
        raise TypeError("hopper_profile must be exact HopperProfileV2")
    platform = _copy_platform_profile_v2(value.profile)
    if platform.platform_kind is not PlatformKindV2.HOPPER:
        raise ValueError("hopper profile platform kind mismatch")
    return HopperProfileV2(
        profile=platform,
        gravity_mps2=value.gravity_mps2,
        launch_speeds_mps=value.launch_speeds_mps,
        launch_elevations_rad=value.launch_elevations_rad,
        azimuth_direction_count=value.azimuth_direction_count,
        landing_sigma_range_scale=value.landing_sigma_range_scale,
        landing_sigma_offset_m=value.landing_sigma_offset_m,
        max_landing_slope_deg=value.max_landing_slope_deg,
        landing_probability_threshold=value.landing_probability_threshold,
        midcourse_correction_enabled=value.midcourse_correction_enabled,
        inflight_observation_enabled=value.inflight_observation_enabled,
        body_envelope_radius_m=value.body_envelope_radius_m,
        launch_reference_height_m=value.launch_reference_height_m,
        arc_clearance_margin_m=value.arc_clearance_margin_m,
        landing_footprint_radius_m=value.landing_footprint_radius_m,
        stop_condition=value.stop_condition,
        energy_model=value.energy_model,
    )


@dataclass(frozen=True, slots=True)
class HopperJumpCandidateV2:
    start_state: PoseStateV2
    support_height_m: float
    hopper_profile: HopperProfileV2
    parameter_set_id: str
    speed_index: int
    elevation_index: int
    azimuth_index: int
    schema_version: str = _CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.start_state) is not PoseStateV2:
            raise TypeError("start_state must be exact PoseStateV2")
        start = PoseStateV2(
            self.start_state.x_m,
            self.start_state.y_m,
            self.start_state.heading_rad,
        )
        object.__setattr__(self, "start_state", start)
        if type(self.support_height_m) is not float or not isfinite(self.support_height_m):
            raise TypeError("support_height_m must be exact finite float")
        object.__setattr__(
            self,
            "support_height_m",
            0.0 if self.support_height_m == 0.0 else self.support_height_m,
        )
        object.__setattr__(self, "hopper_profile", _copy_hopper_profile_v2(self.hopper_profile))
        if type(self.parameter_set_id) is not str or not self.parameter_set_id.strip():
            raise TypeError("parameter_set_id must be exact nonempty str")
        for value, upper in (
            (self.speed_index, 4),
            (self.elevation_index, 3),
            (self.azimuth_index, 16),
        ):
            if type(value) is not int:
                raise TypeError("candidate indices must be exact int")
            if not 0 <= value < upper:
                raise ValueError("candidate index out of range")
        if type(self.schema_version) is not str or self.schema_version != _CANDIDATE_SCHEMA:
            raise ValueError("candidate schema mismatch")


def _validate_detail_scalar_v2(value: object) -> None:
    if type(value) not in (type(None), bool, int, float, str):
        raise TypeError("detail value must be a scalar")
    if type(value) is float and not isfinite(value):
        raise ValueError("detail float must be finite")


@dataclass(frozen=True, slots=True)
class HopperValidationResultV2:
    evidence: ValidationEvidenceV2
    reason_code: str
    category: FailureCategoryV2 | None
    stage: str
    details: tuple[tuple[str, object], ...]
    timed_out: bool
    failed_cell: Cell | None
    segment_index: int | None
    replay_work_counts: tuple[tuple[str, int], ...]
    selected_landing_mass: float | None
    schema_version: str = _RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.reason_code) is not str or self.reason_code not in HOPPER_JUMP_REASON_CODES_V1:
            raise ValueError("reason_code must be a frozen Hopper reason")
        if self.category is not None and type(self.category) is not FailureCategoryV2:
            raise TypeError("category must be exact FailureCategoryV2 or None")
        if type(self.stage) is not str or self.stage not in _STAGES:
            raise ValueError("stage must be a frozen Hopper stage")
        if type(self.details) is not tuple:
            raise TypeError("details must be exact tuple")
        keys = []
        for item in self.details:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("detail entries must be exact pairs")
            key, value = item
            if type(key) is not str or not key:
                raise TypeError("detail keys must be exact nonempty str")
            _validate_detail_scalar_v2(value)
            keys.append(key)
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("detail keys must be unique and sorted")
        if type(self.timed_out) is not bool:
            raise TypeError("timed_out must be exact bool")
        if self.failed_cell is not None and type(self.failed_cell) is not Cell:
            raise TypeError("failed_cell must be exact Cell or None")
        if self.segment_index is not None and (
            type(self.segment_index) is not int or self.segment_index < 0
        ):
            raise TypeError("segment_index must be exact nonnegative int or None")
        if type(self.replay_work_counts) is not tuple or len(self.replay_work_counts) != 6:
            raise TypeError("replay_work_counts must be the six exact counters")
        for position, item in enumerate(self.replay_work_counts):
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("replay counter must be exact pair")
            counter_id, count = item
            if type(counter_id) is not str or counter_id != HOPPER_REPLAY_COUNTER_IDS_V1[position]:
                raise ValueError("replay counter id mismatch")
            if type(count) is not int or not 0 <= count <= HOPPER_MAX_REPLAY_STEPS_V2:
                raise ValueError("replay counter value mismatch")
        if self.selected_landing_mass is not None:
            if (
                type(self.selected_landing_mass) is not float
                or not isfinite(self.selected_landing_mass)
                or not 0.0 <= self.selected_landing_mass <= 1.0
            ):
                raise ValueError("selected_landing_mass must be canonical probability or None")
        if type(self.schema_version) is not str or self.schema_version != _RESULT_SCHEMA:
            raise ValueError("result schema mismatch")
        if type(self.evidence) is not ValidationEvidenceV2:
            raise TypeError("evidence must be exact ValidationEvidenceV2")
        passed = self.reason_code == "hopper_jump_l2_valid"
        if (
            type(self.evidence.validator_id) is not str
            or self.evidence.validator_id != HOPPER_JUMP_VALIDATOR_ID_V2
            or self.evidence.level is not ValidationLevelV2.L2
            or type(self.evidence.passed) is not bool
            or self.evidence.passed is not passed
            or type(self.evidence.checks) is not tuple
            or self.evidence.checks != (self.reason_code,)
            or any(type(check) is not str for check in self.evidence.checks)
        ):
            raise ValueError("evidence contract mismatch")
        expected_category = _category_for_reason_v2(self.reason_code)
        if self.category is not expected_category:
            raise ValueError("category does not match reason")
        if self.timed_out is not (self.reason_code == "planning_deadline_expired"):
            raise ValueError("timed_out does not match reason")
        arc_semantic = self.reason_code in (
            "hopper_arc_boundary_violation",
            "hopper_arc_unknown",
            "hopper_arc_clearance_violation",
        )
        launch_semantic = self.reason_code in ("hopper_launch_unknown", "hopper_launch_unsafe")
        landing_cell_semantic = self.reason_code in (
            "hopper_landing_zone_unknown",
            "hopper_landing_zone_unsafe",
            "hopper_landing_slope_exceeded",
            "hopper_landing_height_unreachable",
        )
        if arc_semantic and (self.failed_cell is None or self.segment_index is None):
            raise ValueError("arc semantic failure requires cell and segment")
        if launch_semantic and (self.failed_cell is None or self.segment_index is not None):
            raise ValueError("launch semantic failure requires only cell")
        if landing_cell_semantic and self.failed_cell is None:
            raise ValueError("landing semantic failure requires a cell")
        if not arc_semantic and not launch_semantic and not landing_cell_semantic and (
            self.failed_cell is not None or self.segment_index is not None
        ):
            raise ValueError("nonsemantic result cannot carry failure location")


def _category_for_reason_v2(reason_code: str) -> FailureCategoryV2 | None:
    if reason_code == "hopper_jump_l2_valid":
        return None
    if reason_code == "planning_deadline_expired":
        return FailureCategoryV2.TIMEOUT
    if reason_code == "hopper_replay_work_budget_exceeded":
        return FailureCategoryV2.RESOURCE_LIMIT
    if reason_code in (
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
    ):
        return FailureCategoryV2.VALIDATION_FAILED
    return FailureCategoryV2.INTERNAL_ERROR


def _zero_counts_v2() -> tuple[tuple[str, int], ...]:
    return tuple((counter_id, 0) for counter_id in HOPPER_REPLAY_COUNTER_IDS_V1)


def _result_v2(
    reason_code: str,
    stage: str,
    counts: list[int] | tuple[tuple[str, int], ...],
    *,
    details: tuple[tuple[str, object], ...] = (),
    failed_cell: Cell | None = None,
    segment_index: int | None = None,
    selected_landing_mass: float | None = None,
) -> HopperValidationResultV2:
    if type(counts) is list:
        frozen_counts = tuple(
            (counter_id, counts[index])
            for index, counter_id in enumerate(HOPPER_REPLAY_COUNTER_IDS_V1)
        )
    else:
        frozen_counts = counts
    passed = reason_code == "hopper_jump_l2_valid"
    return HopperValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=HOPPER_JUMP_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=passed,
            checks=(reason_code,),
        ),
        reason_code=reason_code,
        category=_category_for_reason_v2(reason_code),
        stage=stage,
        details=details,
        timed_out=reason_code == "planning_deadline_expired",
        failed_cell=failed_cell,
        segment_index=segment_index,
        replay_work_counts=frozen_counts,
        selected_landing_mass=selected_landing_mass,
        schema_version=_RESULT_SCHEMA,
    )


def _contract_result_v2(reason: str, stage: str, counts: list[int]) -> HopperValidationResultV2:
    return _result_v2(
        reason,
        stage,
        counts,
        details=(("actual", reason), ("expected", "canonical"), ("phase", stage)),
    )


def _int_add_v2(left: int, right: int) -> int:
    if type(left) is not int or type(right) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    left_bits = left.bit_length()
    right_bits = right.bit_length()
    arena = _HopperExactIntegerArenaV2(HOPPER_RESOURCE_AUTHORITY_V2)
    return arena.consume_admitted_integers(
        result_bit_bounds=(max(left_bits, right_bits) + 1,),
        operation=lambda: (left + right,),
        consumer=lambda values: values[0],
    )


def _int_sub_v2(left: int, right: int) -> int:
    if type(left) is not int or type(right) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    left_bits = left.bit_length()
    right_bits = right.bit_length()
    arena = _HopperExactIntegerArenaV2(HOPPER_RESOURCE_AUTHORITY_V2)
    return arena.consume_admitted_integers(
        result_bit_bounds=(max(left_bits, right_bits) + 1,),
        operation=lambda: (left - right,),
        consumer=lambda values: values[0],
    )


def _int_mul_v2(left: int, right: int) -> int:
    if type(left) is not int or type(right) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if left == 0 or right == 0:
        return 0
    left_bits = left.bit_length()
    right_bits = right.bit_length()
    arena = _HopperExactIntegerArenaV2(HOPPER_RESOURCE_AUTHORITY_V2)
    return arena.consume_admitted_integers(
        result_bit_bounds=(left_bits + right_bits,),
        operation=lambda: (left * right,),
        consumer=lambda values: values[0],
    )


def _int_floor_div_v2(left: int, right: int) -> int:
    if type(left) is not int or type(right) is not int or right == 0:
        raise ValueError("hopper_numeric_contract_mismatch")
    if left == 0:
        return 0
    left_bits = left.bit_length()
    arena = _HopperExactIntegerArenaV2(HOPPER_RESOURCE_AUTHORITY_V2)
    return arena.consume_admitted_integers(
        result_bit_bounds=(left_bits,),
        operation=lambda: (left // right,),
        consumer=lambda values: values[0],
    )


def _int_divmod_v2(left: int, right: int) -> tuple[int, int]:
    if type(left) is not int or type(right) is not int or right == 0:
        raise ValueError("hopper_numeric_contract_mismatch")
    if left == 0:
        return 0, 0
    left_bits = left.bit_length()
    right_bits = right.bit_length()
    arena = _HopperExactIntegerArenaV2(HOPPER_RESOURCE_AUTHORITY_V2)
    return arena.consume_admitted_integers(
        result_bit_bounds=(left_bits, right_bits),
        operation=lambda: divmod(left, right),
        consumer=lambda values: (values[0], values[1]),
    )


def _int_neg_v2(value: int) -> int:
    if type(value) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if value == 0:
        return 0
    value_bits = value.bit_length()
    arena = _HopperExactIntegerArenaV2(HOPPER_RESOURCE_AUTHORITY_V2)
    return arena.consume_admitted_integers(
        result_bit_bounds=(value_bits + 1,),
        operation=lambda: (-value,),
        consumer=lambda values: values[0],
    )


def _float_ratio_v2(value: float) -> tuple[int, int]:
    arena = _HopperExactIntegerArenaV2(HOPPER_RESOURCE_AUTHORITY_V2)
    return arena.consume_admitted_integers(
        result_bit_bounds=(1_075, 1_075),
        operation=lambda: value.as_integer_ratio(),
        consumer=lambda values: (values[0], values[1]),
    )


def _rat_add_v2(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    left_cross = _int_mul_v2(left[0], right[1])
    right_cross = _int_mul_v2(right[0], left[1])
    numerator = _int_add_v2(left_cross, right_cross)
    denominator = _int_mul_v2(left[1], right[1])
    return numerator, denominator


def _rat_sub_v2(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    left_cross = _int_mul_v2(left[0], right[1])
    right_cross = _int_mul_v2(right[0], left[1])
    numerator = _int_sub_v2(left_cross, right_cross)
    denominator = _int_mul_v2(left[1], right[1])
    return numerator, denominator


def _rat_mul_v2(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    numerator = _int_mul_v2(left[0], right[0])
    denominator = _int_mul_v2(left[1], right[1])
    return numerator, denominator


def _rat_div_v2(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    if right[0] == 0:
        raise ValueError("hopper_numeric_contract_mismatch")
    numerator = _int_mul_v2(left[0], right[1])
    denominator = _int_mul_v2(left[1], right[0])
    if denominator < 0:
        numerator = _int_neg_v2(numerator)
        denominator = _int_neg_v2(denominator)
    return numerator, denominator


def _rat_cmp_v2(left: tuple[int, int], right: tuple[int, int]) -> int:
    left_cross = _int_mul_v2(left[0], right[1])
    right_cross = _int_mul_v2(right[0], left[1])
    return (left_cross > right_cross) - (left_cross < right_cross)


def _rat_floor_v2(value: tuple[int, int]) -> int:
    return _int_floor_div_v2(value[0], value[1])


def _rat_ceil_v2(value: tuple[int, int]) -> int:
    quotient_remainder = _int_divmod_v2(value[0], value[1])
    if quotient_remainder[1] == 0:
        return quotient_remainder[0]
    result = _int_add_v2(quotient_remainder[0], 1)
    return result


def _exact_qsqrt_sign_v2(
    rational: int,
    radical_coefficient: int,
    radicand: int,
) -> int:
    if type(rational) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(radical_coefficient) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(radicand) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if max(
        rational.bit_length(),
        radical_coefficient.bit_length(),
        radicand.bit_length(),
    ) > HOPPER_EXACT_MAX_INTEGER_BITS_V2 // 2:
        raise ValueError("hopper_numeric_contract_mismatch")
    if radicand < 0:
        raise ValueError("hopper_numeric_contract_mismatch")
    probe = _int_add_v2(rational, 0)
    if radical_coefficient == 0 or radicand == 0:
        return (probe > 0) - (probe < 0)
    if rational == 0:
        return (radical_coefficient > 0) - (radical_coefficient < 0)
    if (rational > 0) == (radical_coefficient > 0):
        return 1 if rational > 0 else -1
    rational_squared = _int_mul_v2(rational, rational)
    coefficient_squared = _int_mul_v2(radical_coefficient, radical_coefficient)
    radical_squared = _int_mul_v2(coefficient_squared, radicand)
    if rational_squared == radical_squared:
        return 0
    if rational > 0:
        return 1 if rational_squared > radical_squared else -1
    return -1 if rational_squared > radical_squared else 1


def _compare_binary64_to_qsqrt_v2(
    value: float,
    rational: int,
    radical_coefficient: int,
    radicand: int,
    denominator: int,
) -> int:
    if type(value) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(rational) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(radical_coefficient) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(radicand) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(denominator) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if max(
        rational.bit_length(),
        radical_coefficient.bit_length(),
        radicand.bit_length(),
        denominator.bit_length(),
    ) > HOPPER_EXACT_MAX_INTEGER_BITS_V2 // 2:
        raise ValueError("hopper_numeric_contract_mismatch")
    if not isfinite(value) or radicand < 0 or denominator <= 0:
        raise ValueError("hopper_numeric_contract_mismatch")
    ratio = _float_ratio_v2(value)
    scaled_value = _int_mul_v2(ratio[0], denominator)
    scaled_rational = _int_mul_v2(rational, ratio[1])
    q_value = _int_sub_v2(scaled_value, scaled_rational)
    neg_coefficient = _int_neg_v2(radical_coefficient)
    k_value = _int_mul_v2(neg_coefficient, ratio[1])
    result = _exact_qsqrt_sign_v2(q_value, k_value, radicand)
    return result


def _exact_parabola_clearance_at_rational_v2(
    launch_z: float,
    apex_height: float,
    terrain_elevation: float,
    body_radius: float,
    clearance_margin: float,
    parameter_numerator: int,
    parameter_denominator: int,
) -> bool:
    if type(launch_z) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(apex_height) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(terrain_elevation) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(body_radius) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(clearance_margin) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(parameter_numerator) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(parameter_denominator) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if max(
        parameter_numerator.bit_length(),
        parameter_denominator.bit_length(),
    ) > HOPPER_EXACT_MAX_INTEGER_BITS_V2 // 2:
        raise ValueError("hopper_numeric_contract_mismatch")
    if parameter_denominator <= 0:
        raise ValueError("hopper_numeric_contract_mismatch")
    values = (
        launch_z,
        apex_height,
        terrain_elevation,
        body_radius,
        clearance_margin,
    )
    if any(not isfinite(item) for item in values):
        raise ValueError("hopper_numeric_contract_mismatch")
    launch = _float_ratio_v2(launch_z)
    apex = _float_ratio_v2(apex_height)
    terrain = _float_ratio_v2(terrain_elevation)
    body = _float_ratio_v2(body_radius)
    margin = _float_ratio_v2(clearance_margin)
    parameter = (parameter_numerator, parameter_denominator)
    one_minus = _rat_sub_v2((1, 1), parameter)
    product = _rat_mul_v2(parameter, one_minus)
    four_apex = _rat_mul_v2((4, 1), apex)
    rise = _rat_mul_v2(four_apex, product)
    height = _rat_add_v2(launch, rise)
    required = _rat_add_v2(terrain, body)
    required = _rat_add_v2(required, margin)
    comparison = _rat_cmp_v2(height, required)
    return comparison >= 0


def _exact_candidate_index_bounds_v2(
    lower: float,
    upper: float,
    origin: float,
    resolution: float,
) -> tuple[int, int]:
    if type(lower) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(upper) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(origin) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(resolution) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if not all(isfinite(item) for item in (lower, upper, origin, resolution)):
        raise ValueError("hopper_numeric_contract_mismatch")
    if lower > upper or resolution <= 0.0:
        raise ValueError("hopper_numeric_contract_mismatch")
    lower_ratio = _rat_sub_v2(_float_ratio_v2(lower), _float_ratio_v2(origin))
    lower_ratio = _rat_div_v2(lower_ratio, _float_ratio_v2(resolution))
    upper_ratio = _rat_sub_v2(_float_ratio_v2(upper), _float_ratio_v2(origin))
    upper_ratio = _rat_div_v2(upper_ratio, _float_ratio_v2(resolution))
    lower_ceil = _rat_ceil_v2(lower_ratio)
    lower_bound = _int_sub_v2(lower_ceil, 1)
    upper_bound = _rat_floor_v2(upper_ratio)
    return lower_bound, upper_bound


def _fast_candidate_index_bounds_v2(
    lower: float,
    upper: float,
    origin: float,
    resolution: float,
) -> tuple[int, int]:
    return (
        ceil((lower - origin) / resolution) - 1,
        floor((upper - origin) / resolution),
    )


def _classify_exact_quadratic_overlap_v2(
    quadratic: int,
    linear: int,
    constant: int,
    lower_numerator: int,
    lower_denominator: int,
    upper_numerator: int,
    upper_denominator: int,
) -> str:
    if type(quadratic) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(linear) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(constant) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(lower_numerator) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(lower_denominator) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(upper_numerator) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(upper_denominator) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if max(
        quadratic.bit_length(),
        linear.bit_length(),
        constant.bit_length(),
        lower_numerator.bit_length(),
        lower_denominator.bit_length(),
        upper_numerator.bit_length(),
        upper_denominator.bit_length(),
    ) > HOPPER_EXACT_MAX_INTEGER_BITS_V2 // 2:
        raise ValueError("hopper_numeric_contract_mismatch")
    if quadratic < 0 or lower_denominator <= 0 or upper_denominator <= 0:
        raise ValueError("hopper_numeric_contract_mismatch")
    lower_upper = _int_mul_v2(lower_numerator, upper_denominator)
    upper_lower = _int_mul_v2(upper_numerator, lower_denominator)
    if lower_upper > upper_lower:
        raise ValueError("hopper_numeric_contract_mismatch")
    if quadratic == 0:
        if linear != 0:
            raise ValueError("hopper_numeric_contract_mismatch")
        return "interval" if constant <= 0 else "empty"
    left_n2 = _int_mul_v2(lower_numerator, lower_numerator)
    left_q = _int_mul_v2(quadratic, left_n2)
    left_ld = _int_mul_v2(linear, lower_numerator)
    left_ld = _int_mul_v2(left_ld, lower_denominator)
    left_d2 = _int_mul_v2(lower_denominator, lower_denominator)
    left_c = _int_mul_v2(constant, left_d2)
    left_value = _int_add_v2(left_q, left_ld)
    left_value = _int_add_v2(left_value, left_c)
    right_n2 = _int_mul_v2(upper_numerator, upper_numerator)
    right_q = _int_mul_v2(quadratic, right_n2)
    right_ld = _int_mul_v2(linear, upper_numerator)
    right_ld = _int_mul_v2(right_ld, upper_denominator)
    right_d2 = _int_mul_v2(upper_denominator, upper_denominator)
    right_c = _int_mul_v2(constant, right_d2)
    right_value = _int_add_v2(right_q, right_ld)
    right_value = _int_add_v2(right_value, right_c)
    if left_value < 0 or right_value < 0:
        return "interval"
    if left_value == 0 or right_value == 0:
        return "singleton"
    vertex_numerator = _int_neg_v2(linear)
    vertex_denominator = _int_mul_v2(2, quadratic)
    left_cmp = _rat_cmp_v2(
        (vertex_numerator, vertex_denominator),
        (lower_numerator, lower_denominator),
    )
    right_cmp = _rat_cmp_v2(
        (vertex_numerator, vertex_denominator),
        (upper_numerator, upper_denominator),
    )
    if left_cmp < 0 or right_cmp > 0:
        return "empty"
    discriminant_left = _int_mul_v2(linear, linear)
    discriminant_right = _int_mul_v2(quadratic, constant)
    discriminant_right = _int_mul_v2(4, discriminant_right)
    discriminant = _int_sub_v2(discriminant_left, discriminant_right)
    if discriminant > 0:
        return "interval"
    if discriminant == 0:
        return "singleton"
    return "empty"


def _axis_distance_coefficients_v2(
    start: tuple[int, int],
    delta: tuple[int, int],
    lower: tuple[int, int],
    upper: tuple[int, int],
    at: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    offset = _rat_mul_v2(delta, at)
    point = _rat_add_v2(start, offset)
    below = _rat_cmp_v2(point, lower)
    above = _rat_cmp_v2(point, upper)
    if below < 0:
        negative_delta = (_int_neg_v2(delta[0]), delta[1])
        intercept = _rat_sub_v2(lower, start)
        return negative_delta, intercept
    if above > 0:
        intercept = _rat_sub_v2(start, upper)
        return delta, intercept
    return (0, 1), (0, 1)


def _evaluate_distance_polynomial_v2(
    quadratic: tuple[int, int],
    linear: tuple[int, int],
    constant: tuple[int, int],
    at: tuple[int, int],
) -> tuple[int, int]:
    squared = _rat_mul_v2(at, at)
    first = _rat_mul_v2(quadratic, squared)
    second = _rat_mul_v2(linear, at)
    value = _rat_add_v2(first, second)
    value = _rat_add_v2(value, constant)
    return value


def _exact_disk_segment_square_relation_v2(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    body_radius: float,
    clearance_margin: float,
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
) -> str:
    if type(x0) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(y0) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(dx) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(dy) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(body_radius) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(clearance_margin) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(x_lower) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(x_upper) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(y_lower) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(y_upper) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    values = (x0, y0, dx, dy, body_radius, clearance_margin, x_lower, x_upper, y_lower, y_upper)
    if any(not isfinite(item) for item in values):
        raise ValueError("hopper_numeric_contract_mismatch")
    if body_radius < 0.0 or clearance_margin < 0.0 or x_lower > x_upper or y_lower > y_upper:
        raise ValueError("hopper_numeric_contract_mismatch")
    starts = (_float_ratio_v2(x0), _float_ratio_v2(y0))
    deltas = (_float_ratio_v2(dx), _float_ratio_v2(dy))
    lowers = (_float_ratio_v2(x_lower), _float_ratio_v2(y_lower))
    uppers = (_float_ratio_v2(x_upper), _float_ratio_v2(y_upper))
    body = _float_ratio_v2(body_radius)
    margin = _float_ratio_v2(clearance_margin)
    radius = _rat_add_v2(body, margin)
    radius_squared = _rat_mul_v2(radius, radius)
    partition = [(0, 1), (1, 1)]
    for axis in range(2):
        if deltas[axis][0] == 0:
            continue
        for boundary in (lowers[axis], uppers[axis]):
            crossing = _rat_sub_v2(boundary, starts[axis])
            crossing = _rat_div_v2(crossing, deltas[axis])
            lower_cmp = _rat_cmp_v2(crossing, (0, 1))
            upper_cmp = _rat_cmp_v2(crossing, (1, 1))
            if lower_cmp < 0 or upper_cmp > 0:
                continue
            if all(_rat_cmp_v2(crossing, existing) != 0 for existing in partition):
                position = 0
                while position < len(partition) and _rat_cmp_v2(partition[position], crossing) < 0:
                    position += 1
                partition.insert(position, crossing)
    singleton = False
    for left, right in zip(partition, partition[1:]):
        midpoint_sum = _rat_add_v2(left, right)
        midpoint = _rat_div_v2(midpoint_sum, (2, 1))
        ax, bx = _axis_distance_coefficients_v2(starts[0], deltas[0], lowers[0], uppers[0], midpoint)
        ay, by = _axis_distance_coefficients_v2(starts[1], deltas[1], lowers[1], uppers[1], midpoint)
        ax_squared = _rat_mul_v2(ax, ax)
        ay_squared = _rat_mul_v2(ay, ay)
        quadratic = _rat_add_v2(ax_squared, ay_squared)
        axbx = _rat_mul_v2(ax, bx)
        ayby = _rat_mul_v2(ay, by)
        linear = _rat_add_v2(axbx, ayby)
        linear = _rat_mul_v2((2, 1), linear)
        bx_squared = _rat_mul_v2(bx, bx)
        by_squared = _rat_mul_v2(by, by)
        constant = _rat_add_v2(bx_squared, by_squared)
        constant = _rat_sub_v2(constant, radius_squared)
        candidates = [left, right]
        if quadratic[0] != 0:
            negative_linear = (_int_neg_v2(linear[0]), linear[1])
            double_quadratic = _rat_mul_v2((2, 1), quadratic)
            vertex = _rat_div_v2(negative_linear, double_quadratic)
            left_cmp = _rat_cmp_v2(vertex, left)
            right_cmp = _rat_cmp_v2(vertex, right)
            if left_cmp >= 0 and right_cmp <= 0:
                candidates.append(vertex)
        minimum = _evaluate_distance_polynomial_v2(quadratic, linear, constant, candidates[0])
        for candidate in candidates[1:]:
            value = _evaluate_distance_polynomial_v2(quadratic, linear, constant, candidate)
            comparison = _rat_cmp_v2(value, minimum)
            if comparison < 0:
                minimum = value
        zero_comparison = _rat_cmp_v2(minimum, (0, 1))
        if zero_comparison < 0:
            return "interval"
        if quadratic[0] == 0 and linear[0] == 0 and constant[0] == 0:
            return "interval"
        if zero_comparison == 0:
            singleton = True
    return "singleton" if singleton else "empty"


def _fast_disk_segment_square_relation_v2(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    body_radius: float,
    clearance_margin: float,
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
) -> str:
    radius = body_radius + clearance_margin
    radius_squared = radius * radius
    partition = [0.0, 1.0]
    for start, delta, lower, upper in (
        (x0, dx, x_lower, x_upper),
        (y0, dy, y_lower, y_upper),
    ):
        if delta == 0.0:
            continue
        for boundary in (lower, upper):
            crossing = (boundary - start) / delta
            if 0.0 <= crossing <= 1.0 and crossing not in partition:
                partition.append(crossing)
    partition.sort()
    singleton = False
    for left, right in zip(partition, partition[1:]):
        midpoint = 0.5 * (left + right)
        coefficients = []
        for start, delta, lower, upper in (
            (x0, dx, x_lower, x_upper),
            (y0, dy, y_lower, y_upper),
        ):
            point = start + delta * midpoint
            if point < lower:
                coefficients.append((-delta, lower - start))
            elif point > upper:
                coefficients.append((delta, start - upper))
            else:
                coefficients.append((0.0, 0.0))
        ax, bx = coefficients[0]
        ay, by = coefficients[1]
        quadratic = ax * ax + ay * ay
        linear = 2.0 * (ax * bx + ay * by)
        constant = bx * bx + by * by - radius_squared
        candidates = [left, right]
        if quadratic != 0.0:
            vertex = -linear / (2.0 * quadratic)
            if left <= vertex <= right:
                candidates.append(vertex)
        minimum = min(
            quadratic * value * value + linear * value + constant
            for value in candidates
        )
        if minimum < 0.0:
            return "interval"
        if quadratic == 0.0 and linear == 0.0 and constant == 0.0:
            return "interval"
        if minimum == 0.0:
            singleton = True
    return "singleton" if singleton else "empty"


def _exact_parabola_clearance_at_qsqrt_v2(
    launch_z: float,
    apex_height: float,
    terrain_elevation: float,
    body_radius: float,
    clearance_margin: float,
    root_rational: int,
    root_radical_coefficient: int,
    root_radicand: int,
    root_denominator: int,
) -> bool:
    launch = _float_ratio_v2(launch_z)
    apex = _float_ratio_v2(apex_height)
    terrain = _float_ratio_v2(terrain_elevation)
    body = _float_ratio_v2(body_radius)
    margin = _float_ratio_v2(clearance_margin)
    required = _rat_add_v2(terrain, body)
    required = _rat_add_v2(required, margin)
    base = _rat_sub_v2(launch, required)
    rational_times_denominator = _int_mul_v2(
        root_rational,
        root_denominator,
    )
    rational_squared = _int_mul_v2(root_rational, root_rational)
    coefficient_squared = _int_mul_v2(
        root_radical_coefficient,
        root_radical_coefficient,
    )
    radical_square = _int_mul_v2(coefficient_squared, root_radicand)
    constant_numerator = _int_sub_v2(
        rational_times_denominator,
        rational_squared,
    )
    constant_numerator = _int_sub_v2(constant_numerator, radical_square)
    coefficient_times_denominator = _int_mul_v2(
        root_radical_coefficient,
        root_denominator,
    )
    twice_rational = _int_mul_v2(2, root_rational)
    twice_product = _int_mul_v2(
        twice_rational,
        root_radical_coefficient,
    )
    radical_numerator = _int_sub_v2(
        coefficient_times_denominator,
        twice_product,
    )
    denominator_squared = _int_mul_v2(root_denominator, root_denominator)
    four_apex = _rat_mul_v2((4, 1), apex)
    constant_rise = _rat_mul_v2(
        four_apex,
        (constant_numerator, denominator_squared),
    )
    radical_rise = _rat_mul_v2(
        four_apex,
        (radical_numerator, denominator_squared),
    )
    constant_clearance = _rat_add_v2(base, constant_rise)
    scaled_constant = _int_mul_v2(
        constant_clearance[0],
        radical_rise[1],
    )
    scaled_radical = _int_mul_v2(
        radical_rise[0],
        constant_clearance[1],
    )
    sign = _exact_qsqrt_sign_v2(
        scaled_constant,
        scaled_radical,
        root_radicand,
    )
    return sign >= 0


def _exact_cell_overlap_clearance_v2(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    body_radius: float,
    clearance_margin: float,
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
    launch_z: float,
    apex_height: float,
    terrain_elevation: float,
) -> bool:
    rounded_required = terrain_elevation + body_radius + clearance_margin
    conservative_required = nextafter(
        nextafter(rounded_required, float("inf")),
        float("inf"),
    )
    if conservative_required < launch_z:
        return True
    globally_safe_at_ground_endpoints = _exact_parabola_clearance_at_rational_v2(
        launch_z,
        apex_height,
        terrain_elevation,
        body_radius,
        clearance_margin,
        0,
        1,
    )
    if globally_safe_at_ground_endpoints:
        return True
    starts = (_float_ratio_v2(x0), _float_ratio_v2(y0))
    deltas = (_float_ratio_v2(dx), _float_ratio_v2(dy))
    lowers = (_float_ratio_v2(x_lower), _float_ratio_v2(y_lower))
    uppers = (_float_ratio_v2(x_upper), _float_ratio_v2(y_upper))
    body = _float_ratio_v2(body_radius)
    margin = _float_ratio_v2(clearance_margin)
    radius = _rat_add_v2(body, margin)
    radius_squared = _rat_mul_v2(radius, radius)
    partition = [(0, 1), (1, 1)]
    for axis in range(2):
        if deltas[axis][0] == 0:
            continue
        for boundary in (lowers[axis], uppers[axis]):
            crossing = _rat_sub_v2(boundary, starts[axis])
            crossing = _rat_div_v2(crossing, deltas[axis])
            lower_comparison = _rat_cmp_v2(crossing, (0, 1))
            upper_comparison = _rat_cmp_v2(crossing, (1, 1))
            if lower_comparison < 0 or upper_comparison > 0:
                continue
            if all(
                _rat_cmp_v2(crossing, existing) != 0
                for existing in partition
            ):
                position = 0
                while (
                    position < len(partition)
                    and _rat_cmp_v2(partition[position], crossing) < 0
                ):
                    position += 1
                partition.insert(position, crossing)
    for left, right in zip(partition, partition[1:]):
        midpoint_sum = _rat_add_v2(left, right)
        midpoint = _rat_div_v2(midpoint_sum, (2, 1))
        ax, bx = _axis_distance_coefficients_v2(
            starts[0],
            deltas[0],
            lowers[0],
            uppers[0],
            midpoint,
        )
        ay, by = _axis_distance_coefficients_v2(
            starts[1],
            deltas[1],
            lowers[1],
            uppers[1],
            midpoint,
        )
        ax_squared = _rat_mul_v2(ax, ax)
        ay_squared = _rat_mul_v2(ay, ay)
        quadratic = _rat_add_v2(ax_squared, ay_squared)
        axbx = _rat_mul_v2(ax, bx)
        ayby = _rat_mul_v2(ay, by)
        linear = _rat_add_v2(axbx, ayby)
        linear = _rat_mul_v2((2, 1), linear)
        bx_squared = _rat_mul_v2(bx, bx)
        by_squared = _rat_mul_v2(by, by)
        constant = _rat_add_v2(bx_squared, by_squared)
        constant = _rat_sub_v2(constant, radius_squared)
        left_value = _evaluate_distance_polynomial_v2(
            quadratic,
            linear,
            constant,
            left,
        )
        right_value = _evaluate_distance_polynomial_v2(
            quadratic,
            linear,
            constant,
            right,
        )
        left_sign = _rat_cmp_v2(left_value, (0, 1))
        right_sign = _rat_cmp_v2(right_value, (0, 1))
        if quadratic[0] == 0:
            if linear[0] != 0:
                raise ValueError("hopper_numeric_contract_mismatch")
            if constant[0] > 0:
                continue
            left_safe = _exact_parabola_clearance_at_rational_v2(
                launch_z,
                apex_height,
                terrain_elevation,
                body_radius,
                clearance_margin,
                left[0],
                left[1],
            )
            right_safe = _exact_parabola_clearance_at_rational_v2(
                launch_z,
                apex_height,
                terrain_elevation,
                body_radius,
                clearance_margin,
                right[0],
                right[1],
            )
            if not left_safe or not right_safe:
                return False
            continue
        q_denominator = quadratic[1]
        l_denominator = linear[1]
        c_denominator = constant[1]
        first_scale = _int_mul_v2(l_denominator, c_denominator)
        integer_quadratic = _int_mul_v2(quadratic[0], first_scale)
        second_scale = _int_mul_v2(q_denominator, c_denominator)
        integer_linear = _int_mul_v2(linear[0], second_scale)
        third_scale = _int_mul_v2(q_denominator, l_denominator)
        integer_constant = _int_mul_v2(constant[0], third_scale)
        linear_squared = _int_mul_v2(integer_linear, integer_linear)
        quadratic_constant = _int_mul_v2(
            integer_quadratic,
            integer_constant,
        )
        four_quadratic_constant = _int_mul_v2(4, quadratic_constant)
        discriminant = _int_sub_v2(
            linear_squared,
            four_quadratic_constant,
        )
        if discriminant < 0:
            continue
        negative_linear = _int_neg_v2(integer_linear)
        root_denominator = _int_mul_v2(2, integer_quadratic)
        if left_sign <= 0:
            left_safe = _exact_parabola_clearance_at_rational_v2(
                launch_z,
                apex_height,
                terrain_elevation,
                body_radius,
                clearance_margin,
                left[0],
                left[1],
            )
        else:
            left_safe = _exact_parabola_clearance_at_qsqrt_v2(
                launch_z,
                apex_height,
                terrain_elevation,
                body_radius,
                clearance_margin,
                negative_linear,
                -1,
                discriminant,
                root_denominator,
            )
        if right_sign <= 0:
            right_safe = _exact_parabola_clearance_at_rational_v2(
                launch_z,
                apex_height,
                terrain_elevation,
                body_radius,
                clearance_margin,
                right[0],
                right[1],
            )
        else:
            right_safe = _exact_parabola_clearance_at_qsqrt_v2(
                launch_z,
                apex_height,
                terrain_elevation,
                body_radius,
                clearance_margin,
                negative_linear,
                1,
                discriminant,
                root_denominator,
            )
        if not left_safe or not right_safe:
            return False
    return True


def _partition_ballistic_samples_v2(
    samples: tuple[BallisticSampleV2, ...],
    flight_time: float,
) -> tuple[tuple[float, float], ...]:
    if type(samples) is not tuple or type(flight_time) is not float or not isfinite(flight_time):
        raise ValueError("hopper_numeric_contract_mismatch")
    flight_ratio = _float_ratio_v2(flight_time)
    if flight_ratio[1] <= 0:
        raise ValueError("hopper_numeric_contract_mismatch")
    interval_count = len(samples) - 1
    if interval_count < 1:
        raise ValueError("hopper_numeric_contract_mismatch")
    if interval_count > HOPPER_MAX_REPLAY_STEPS_V2:
        raise ValueError("hopper_replay_work_budget_exceeded")
    partitions = []
    for index, sample in enumerate(samples):
        if type(sample) is not BallisticSampleV2:
            raise ValueError("hopper_numeric_contract_mismatch")
        audited = BallisticSampleV2(sample.time_s, sample.x_m, sample.y_m, sample.z_m)
        if audited != sample:
            raise ValueError("hopper_numeric_contract_mismatch")
        if index == 0 and sample.time_s != 0.0:
            raise ValueError("hopper_numeric_contract_mismatch")
        if index == len(samples) - 1 and sample.time_s != flight_time:
            raise ValueError("hopper_numeric_contract_mismatch")
        if index and samples[index - 1].time_s >= sample.time_s:
            raise ValueError("hopper_numeric_contract_mismatch")
        if index:
            partitions.append((samples[index - 1].time_s, sample.time_s))
    return tuple(partitions)


def _canonical_hopper_action_v2(azimuth_index: int) -> tuple[float, float, float]:
    if type(azimuth_index) is not int or not 0 <= azimuth_index < 16:
        raise ValueError("hopper_numeric_contract_mismatch")
    azimuth = 2.0 * pi * azimuth_index / 16.0
    if azimuth_index == 0:
        return azimuth, 1.0, 0.0
    if azimuth_index == 4:
        return azimuth, 0.0, 1.0
    if azimuth_index == 8:
        return azimuth, -1.0, 0.0
    if azimuth_index == 12:
        return azimuth, 0.0, -1.0
    return azimuth, cos(azimuth), sin(azimuth)


def _admit_replay_work_v2(counts: list[int], counter_id: str, attempted_value: int) -> None:
    if type(counts) is not list or counter_id not in HOPPER_REPLAY_COUNTER_IDS_V1 or type(attempted_value) is not int:
        raise ValueError("hopper_numeric_contract_mismatch")
    if attempted_value < 0 or attempted_value > HOPPER_MAX_REPLAY_STEPS_V2:
        raise ValueError("hopper_replay_work_budget_exceeded", counter_id, attempted_value)
    counts[HOPPER_REPLAY_COUNTER_IDS_V1.index(counter_id)] = attempted_value


def _insert_distinct_cell_v2(cells: set[Cell], cell: Cell) -> None:
    if type(cells) is not set or type(cell) is not Cell:
        raise ValueError("hopper_numeric_contract_mismatch")
    cells.add(Cell(cell.x, cell.y))


def _resource_result_from_error_v2(error: ValueError, counts: list[int]) -> HopperValidationResultV2:
    if len(error.args) == 3 and error.args[0] == "hopper_replay_work_budget_exceeded":
        counter_id = error.args[1]
        attempted = error.args[2]
    else:
        counter_id = HOPPER_REPLAY_COUNTER_IDS_V1[0]
        attempted = HOPPER_MAX_REPLAY_STEPS_V2 + 1
    return _result_v2(
        "hopper_replay_work_budget_exceeded",
        "replay_work",
        counts,
        details=(
            ("attempted_work_units", attempted),
            ("max_work_units", HOPPER_MAX_REPLAY_STEPS_V2),
            ("phase", counter_id),
        ),
    )


def _deadline_expired_v2(deadline: PlanningDeadlineV2) -> bool:
    try:
        return deadline.expired
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        raise ValueError("planning_deadline_contract_mismatch") from None


def _query_is_exact_v2(query: object, cell: Cell, anchor_hash: str) -> bool:
    return (
        type(query) is SafetyQueryV2
        and query.cell == cell
        and query.snapshot_hash == anchor_hash
        and query.validation_level is ValidationLevelV2.L2
        and type(query.passed) is bool
        and type(query.reason_code) is str
    )


def _launch_footprint_v2(
    candidate: HopperJumpCandidateV2,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
    counts: list[int],
) -> tuple[HopperValidationResultV2 | None, float]:
    profile = candidate.hopper_profile
    radius = profile.body_envelope_radius_m + profile.arc_clearance_margin_m
    x0 = candidate.start_state.x_m
    y0 = candidate.start_state.y_m
    geometry = anchor.snapshot.geometry
    min_x, max_x = _fast_candidate_index_bounds_v2(
        x0 - radius, x0 + radius, geometry.origin[0], geometry.resolution_m
    )
    min_y, max_y = _fast_candidate_index_bounds_v2(
        y0 - radius, y0 + radius, geometry.origin[1], geometry.resolution_m
    )
    common_elevation = None
    first_required_cell = None
    for cell_y in range(min_y, max_y + 1):
        for cell_x in range(min_x, max_x + 1):
            x_lower = geometry.origin[0] + cell_x * geometry.resolution_m
            y_lower = geometry.origin[1] + cell_y * geometry.resolution_m
            relation = _fast_disk_segment_square_relation_v2(
                x0, y0, 0.0, 0.0,
                profile.body_envelope_radius_m,
                profile.arc_clearance_margin_m,
                x_lower, x_lower + geometry.resolution_m,
                y_lower, y_lower + geometry.resolution_m,
            )
            if relation == "empty":
                continue
            cell = Cell(cell_x, cell_y)
            if first_required_cell is None:
                first_required_cell = cell
            if _deadline_expired_v2(deadline):
                return _result_v2("planning_deadline_expired", "launch_validation", counts), 0.0
            query = anchor.query(cell, profile.profile.max_traversable_slope_deg)
            if not _query_is_exact_v2(query, cell, anchor._snapshot_hash):
                return _contract_result_v2("terrain_query_contract_mismatch", "launch_validation", counts), 0.0
            if query.reason_code == "terrain_unknown":
                return _result_v2("hopper_launch_unknown", "launch_validation", counts, failed_cell=cell), 0.0
            if not query.passed:
                return _result_v2("hopper_launch_unsafe", "launch_validation", counts, failed_cell=cell), 0.0
            elevation = float(anchor.snapshot.elevation_m[cell_y, cell_x])
            if common_elevation is None:
                common_elevation = elevation
            elif elevation != common_elevation:
                return _result_v2("hopper_launch_unsafe", "launch_validation", counts, failed_cell=cell), 0.0
    if common_elevation is None:
        return _contract_result_v2("hopper_numeric_contract_mismatch", "launch_validation", counts), 0.0
    if common_elevation != candidate.support_height_m:
        return _result_v2(
            "hopper_launch_unsafe",
            "launch_validation",
            counts,
            failed_cell=first_required_cell,
        ), 0.0
    if profile.launch_reference_height_m < radius:
        return _result_v2(
            "hopper_launch_unsafe",
            "launch_validation",
            counts,
            failed_cell=Cell(min_x, min_y),
        ), 0.0
    return None, common_elevation


def _validate_hopper_arc_partitions_v2(
    candidate: HopperJumpCandidateV2,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
    samples: tuple[BallisticSampleV2, ...],
) -> HopperValidationResultV2:
    counts = [0, 0, 0, 0, 0, 0]
    profile = candidate.hopper_profile
    speed = profile.launch_speeds_mps[candidate.speed_index]
    elevation = profile.launch_elevations_rad[candidate.elevation_index]
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / profile.gravity_mps2
    flight_time = 2.0 * vertical_time_scale
    _, x_direction, y_direction = _canonical_hopper_action_v2(candidate.azimuth_index)
    dx = (horizontal_speed * x_direction) * flight_time
    dy = (horizontal_speed * y_direction) * flight_time
    apex_height = vertical_time_scale * (0.5 * vertical_speed)
    launch_z = candidate.support_height_m + profile.launch_reference_height_m
    try:
        partitions = _partition_ballistic_samples_v2(samples, flight_time)
        _admit_replay_work_v2(counts, HOPPER_REPLAY_COUNTER_IDS_V1[0], len(partitions))
    except ValueError as error:
        if error.args and error.args[0] == "hopper_replay_work_budget_exceeded":
            return _resource_result_from_error_v2(error, counts)
        return _contract_result_v2("hopper_numeric_contract_mismatch", "arc_candidate_enumeration", counts)
    distinct_cells: set[Cell] = set()
    geometry = anchor.snapshot.geometry
    radius = profile.body_envelope_radius_m + profile.arc_clearance_margin_m
    for segment_index, _partition in enumerate(partitions):
        if _deadline_expired_v2(deadline):
            return _result_v2("planning_deadline_expired", "replay_work", counts)
        min_x, max_x = _exact_candidate_index_bounds_v2(
            min(candidate.start_state.x_m, candidate.start_state.x_m + dx) - radius,
            max(candidate.start_state.x_m, candidate.start_state.x_m + dx) + radius,
            geometry.origin[0], geometry.resolution_m,
        )
        min_y, max_y = _exact_candidate_index_bounds_v2(
            min(candidate.start_state.y_m, candidate.start_state.y_m + dy) - radius,
            max(candidate.start_state.y_m, candidate.start_state.y_m + dy) + radius,
            geometry.origin[1], geometry.resolution_m,
        )
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        product = width * height
        try:
            _admit_replay_work_v2(counts, HOPPER_REPLAY_COUNTER_IDS_V1[1], counts[1] + width)
            _admit_replay_work_v2(counts, HOPPER_REPLAY_COUNTER_IDS_V1[2], counts[2] + height)
            _admit_replay_work_v2(counts, HOPPER_REPLAY_COUNTER_IDS_V1[3], counts[3] + product)
            _admit_replay_work_v2(counts, HOPPER_REPLAY_COUNTER_IDS_V1[5], counts[5] + product)
        except ValueError as error:
            return _resource_result_from_error_v2(error, counts)
        for cell_y in range(min_y, max_y + 1):
            for cell_x in range(min_x, max_x + 1):
                x_lower = geometry.origin[0] + cell_x * geometry.resolution_m
                y_lower = geometry.origin[1] + cell_y * geometry.resolution_m
                relation = _fast_disk_segment_square_relation_v2(
                    candidate.start_state.x_m,
                    candidate.start_state.y_m,
                    dx,
                    dy,
                    profile.body_envelope_radius_m,
                    profile.arc_clearance_margin_m,
                    x_lower,
                    x_lower + geometry.resolution_m,
                    y_lower,
                    y_lower + geometry.resolution_m,
                )
                if relation == "empty":
                    continue
                cell = Cell(cell_x, cell_y)
                if cell not in distinct_cells:
                    try:
                        _admit_replay_work_v2(
                            counts,
                            HOPPER_REPLAY_COUNTER_IDS_V1[4],
                            counts[4] + 1,
                        )
                    except ValueError as error:
                        return _resource_result_from_error_v2(error, counts)
                    _insert_distinct_cell_v2(distinct_cells, cell)
                if _deadline_expired_v2(deadline):
                    return _result_v2("planning_deadline_expired", "arc_validation", counts)
                query = anchor.query(cell, profile.profile.max_traversable_slope_deg)
                if not _query_is_exact_v2(query, cell, anchor._snapshot_hash):
                    return _contract_result_v2("terrain_query_contract_mismatch", "arc_validation", counts)
                if query.reason_code == "terrain_out_of_bounds":
                    return _result_v2("hopper_arc_boundary_violation", "arc_validation", counts, failed_cell=cell, segment_index=segment_index)
                if query.reason_code == "terrain_unknown":
                    return _result_v2("hopper_arc_unknown", "arc_validation", counts, failed_cell=cell, segment_index=segment_index)
                if query.reason_code == "terrain_hard_obstacle":
                    return _result_v2("hopper_arc_clearance_violation", "arc_validation", counts, failed_cell=cell, segment_index=segment_index)
                terrain_elevation = float(anchor.snapshot.elevation_m[cell_y, cell_x])
                clearance_passed = _exact_cell_overlap_clearance_v2(
                    candidate.start_state.x_m,
                    candidate.start_state.y_m,
                    dx,
                    dy,
                    profile.body_envelope_radius_m,
                    profile.arc_clearance_margin_m,
                    x_lower,
                    x_lower + geometry.resolution_m,
                    y_lower,
                    y_lower + geometry.resolution_m,
                    launch_z,
                    apex_height,
                    terrain_elevation,
                )
                if not clearance_passed:
                    return _result_v2("hopper_arc_clearance_violation", "arc_validation", counts, failed_cell=cell, segment_index=segment_index)
    return _result_v2("hopper_jump_l2_valid", "arc_validation", counts)


def _nominal_landing_pose_v2(
    candidate: HopperJumpCandidateV2,
    mean_x_m: float,
    mean_y_m: float,
) -> PoseStateV2:
    if type(candidate) is not HopperJumpCandidateV2:
        raise ValueError("hopper_numeric_contract_mismatch")
    if type(mean_x_m) is not float or type(mean_y_m) is not float:
        raise ValueError("hopper_numeric_contract_mismatch")
    return PoseStateV2(mean_x_m, mean_y_m, candidate.start_state.heading_rad)


def _candidate_parameter_record_v2(
    candidate: HopperJumpCandidateV2,
) -> HopperParameterSetRecordV2:
    record = _lookup_hopper_parameter_set_v2(candidate.parameter_set_id)
    if type(record) is not HopperParameterSetRecordV2:
        raise ValueError("hopper_authority_contract_mismatch")
    _hopper_parameter_set_in_memory_token_v2(record)
    profile = candidate.hopper_profile
    if (
        profile.profile.profile_id != record.base_profile_id
        or profile.stop_condition != record.stop_condition
        or profile.energy_model != record.energy_model
    ):
        raise ValueError("hopper_profile_contract_mismatch")
    return record


def _closed_square_dilation_intersects_cell_v2(
    source_left: float,
    source_right: float,
    source_bottom: float,
    source_top: float,
    radius_m: float,
    cell_left: float,
    cell_right: float,
    cell_bottom: float,
    cell_top: float,
) -> bool:
    dx = max(source_left - cell_right, cell_left - source_right, 0.0)
    dy = max(source_bottom - cell_top, cell_bottom - source_top, 0.0)
    return dx * dx + dy * dy <= radius_m * radius_m


def _landing_failure_from_query_v2(
    query: SafetyQueryV2,
    elevation_m: float | None,
    support_height_m: float,
) -> tuple[int, str] | None:
    if query.reason_code == "terrain_unknown":
        return 40, "hopper_landing_zone_unknown"
    if query.reason_code in (
        "terrain_out_of_bounds",
        "terrain_hard_obstacle",
        "terrain_not_traversable",
    ):
        return 41, "hopper_landing_zone_unsafe"
    if query.reason_code == "terrain_slope_exceeded":
        return 42, "hopper_landing_slope_exceeded"
    if not query.passed:
        return 41, "hopper_landing_zone_unsafe"
    if elevation_m != support_height_m:
        return 43, "hopper_landing_height_unreachable"
    return None


def _validate_hopper_landing_and_stop_v2(
    candidate: HopperJumpCandidateV2,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
    counts: list[int],
    record: HopperParameterSetRecordV2,
    mean_x_m: float,
    mean_y_m: float,
    horizontal_range_m: float,
    speed_mps: float,
) -> HopperValidationResultV2:
    profile = candidate.hopper_profile
    sigma_m = horizontal_range_m * profile.landing_sigma_range_scale
    sigma_m = sigma_m + profile.landing_sigma_offset_m
    try:
        prefix = _call_captured_landing_helper_v2(
            HOPPER_RESOURCE_AUTHORITY_V2,
            WorldPoint(mean_x_m, mean_y_m),
            sigma_m,
            profile.landing_probability_threshold,
            anchor.snapshot.geometry,
        )
        _require_canonical_resource_authority_v2(HOPPER_RESOURCE_AUTHORITY_V2)
        _hopper_parameter_set_in_memory_token_v2(record)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return _contract_result_v2(
            "hopper_numeric_contract_mismatch", "landing_probability", counts
        )
    if _deadline_expired_v2(deadline):
        return _result_v2("planning_deadline_expired", "landing_probability", counts)
    if type(prefix) is not tuple or not prefix:
        return _contract_result_v2(
            "hopper_numeric_contract_mismatch", "landing_probability", counts
        )
    if any(type(item) is not LandingCellMassV2 for item in prefix):
        return _contract_result_v2(
            "hopper_numeric_contract_mismatch", "landing_probability", counts
        )
    selected_mass = fsum(item.probability_mass for item in prefix)
    if not isfinite(selected_mass) or not 0.0 <= selected_mass <= 1.0:
        return _contract_result_v2(
            "hopper_numeric_contract_mismatch", "landing_probability", counts
        )
    if selected_mass < profile.landing_probability_threshold:
        return _result_v2(
            "hopper_landing_probability_below_threshold",
            "landing_probability",
            counts,
            selected_landing_mass=selected_mass,
        )

    geometry = anchor.snapshot.geometry
    radius = profile.landing_footprint_radius_m
    failures: list[
        tuple[int, tuple[int, int], int, int, int, str, Cell, int | None]
    ] = []

    def check_cell(cell: Cell, prefix_rank: int | None) -> HopperValidationResultV2 | None:
        if _deadline_expired_v2(deadline):
            return _result_v2(
                "planning_deadline_expired",
                "landing_validation",
                counts,
                selected_landing_mass=selected_mass,
            )
        query = anchor.query(cell, profile.max_landing_slope_deg)
        if not _query_is_exact_v2(query, cell, anchor._snapshot_hash):
            return _contract_result_v2(
                "terrain_query_contract_mismatch", "landing_validation", counts
            )
        elevation = None
        if geometry.in_bounds(cell):
            elevation = float(anchor.snapshot.elevation_m[cell.y, cell.x])
        failure = _landing_failure_from_query_v2(
            query, elevation, candidate.support_height_m
        )
        if failure is not None:
            rank, reason = failure
            optional_rank = (0, prefix_rank) if prefix_rank is not None else (1, 0)
            failures.append(
                (rank, optional_rank, 1, cell.y, cell.x, reason, cell, prefix_rank)
            )
        return None

    for prefix_rank, item in enumerate(prefix):
        if not item.in_bounds:
            failures.append(
                (
                    41,
                    (0, prefix_rank),
                    0,
                    item.cell.y,
                    item.cell.x,
                    "hopper_landing_zone_unsafe",
                    item.cell,
                    prefix_rank,
                )
            )
        source_left = geometry.origin[0] + item.cell.x * geometry.resolution_m
        source_right = source_left + geometry.resolution_m
        source_bottom = geometry.origin[1] + item.cell.y * geometry.resolution_m
        source_top = source_bottom + geometry.resolution_m
        min_x, max_x = _fast_candidate_index_bounds_v2(
            source_left - radius,
            source_right + radius,
            geometry.origin[0],
            geometry.resolution_m,
        )
        min_y, max_y = _fast_candidate_index_bounds_v2(
            source_bottom - radius,
            source_top + radius,
            geometry.origin[1],
            geometry.resolution_m,
        )
        for cell_y in range(min_y, max_y + 1):
            for cell_x in range(min_x, max_x + 1):
                cell_left = geometry.origin[0] + cell_x * geometry.resolution_m
                cell_bottom = geometry.origin[1] + cell_y * geometry.resolution_m
                if not _closed_square_dilation_intersects_cell_v2(
                    source_left,
                    source_right,
                    source_bottom,
                    source_top,
                    radius,
                    cell_left,
                    cell_left + geometry.resolution_m,
                    cell_bottom,
                    cell_bottom + geometry.resolution_m,
                ):
                    continue
                result = check_cell(Cell(cell_x, cell_y), prefix_rank)
                if result is not None:
                    return result

    min_x, max_x = _fast_candidate_index_bounds_v2(
        mean_x_m - radius,
        mean_x_m + radius,
        geometry.origin[0],
        geometry.resolution_m,
    )
    min_y, max_y = _fast_candidate_index_bounds_v2(
        mean_y_m - radius,
        mean_y_m + radius,
        geometry.origin[1],
        geometry.resolution_m,
    )
    for cell_y in range(min_y, max_y + 1):
        for cell_x in range(min_x, max_x + 1):
            cell_left = geometry.origin[0] + cell_x * geometry.resolution_m
            cell_bottom = geometry.origin[1] + cell_y * geometry.resolution_m
            relation = _fast_disk_segment_square_relation_v2(
                mean_x_m,
                mean_y_m,
                0.0,
                0.0,
                radius,
                0.0,
                cell_left,
                cell_left + geometry.resolution_m,
                cell_bottom,
                cell_bottom + geometry.resolution_m,
            )
            if relation == "empty":
                continue
            result = check_cell(Cell(cell_x, cell_y), None)
            if result is not None:
                return result

    if failures:
        _rank, _optional, _source, _y, _x, reason, cell, prefix_rank = min(
            failures
        )
        return _result_v2(
            reason,
            "landing_validation",
            counts,
            failed_cell=cell,
            segment_index=prefix_rank,
            selected_landing_mass=selected_mass,
        )

    _nominal_landing_pose_v2(candidate, mean_x_m, mean_y_m)
    try:
        stopped = record.stop_evaluator(speed_mps)
        _hopper_parameter_set_in_memory_token_v2(record)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return _contract_result_v2(
            "hopper_numeric_contract_mismatch", "stop_validation", counts
        )
    if type(stopped) is not bool:
        return _contract_result_v2(
            "hopper_numeric_contract_mismatch", "stop_validation", counts
        )
    if not stopped:
        return _result_v2(
            "hopper_stop_condition_failed",
            "stop_validation",
            counts,
            selected_landing_mass=selected_mass,
        )
    return _result_v2(
        "hopper_jump_l2_valid",
        "stop_validation",
        counts,
        selected_landing_mass=selected_mass,
    )


def _audit_candidate_for_call_v2(value: object) -> HopperJumpCandidateV2:
    if type(value) is not HopperJumpCandidateV2:
        raise ValueError("hopper_numeric_contract_mismatch")
    try:
        return HopperJumpCandidateV2(
            start_state=value.start_state,
            support_height_m=value.support_height_m,
            hopper_profile=value.hopper_profile,
            parameter_set_id=value.parameter_set_id,
            speed_index=value.speed_index,
            elevation_index=value.elevation_index,
            azimuth_index=value.azimuth_index,
            schema_version=value.schema_version,
        )
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception as error:
        if type(value.hopper_profile) is HopperProfileV2:
            try:
                _copy_hopper_profile_v2(value.hopper_profile)
            except Exception:
                raise ValueError("hopper_profile_contract_mismatch") from None
        raise ValueError("hopper_numeric_contract_mismatch") from error


def validate_hopper_jump_l2(
    candidate: HopperJumpCandidateV2,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
) -> HopperValidationResultV2:
    counts = [0, 0, 0, 0, 0, 0]
    try:
        _require_canonical_resource_authority_v2(HOPPER_RESOURCE_AUTHORITY_V2)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return _contract_result_v2("hopper_authority_contract_mismatch", "launch_validation", counts)
    if type(deadline) is not PlanningDeadlineV2:
        return _contract_result_v2("planning_deadline_contract_mismatch", "launch_validation", counts)
    try:
        if (
            type(deadline.started_monotonic_s) is not float
            or type(deadline.deadline_monotonic_s) is not float
            or not isfinite(deadline.started_monotonic_s)
            or not isfinite(deadline.deadline_monotonic_s)
            or deadline.deadline_monotonic_s < deadline.started_monotonic_s
        ):
            raise ValueError("planning_deadline_contract_mismatch")
    except Exception:
        return _contract_result_v2("planning_deadline_contract_mismatch", "launch_validation", counts)
    try:
        audited_candidate = _audit_candidate_for_call_v2(candidate)
    except ValueError as error:
        reason = "hopper_profile_contract_mismatch" if str(error) == "hopper_profile_contract_mismatch" else "hopper_numeric_contract_mismatch"
        return _contract_result_v2(reason, "launch_validation", counts)
    try:
        parameter_record = _candidate_parameter_record_v2(audited_candidate)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except ValueError as error:
        reason = (
            "hopper_profile_contract_mismatch"
            if str(error) == "hopper_profile_contract_mismatch"
            else "hopper_authority_contract_mismatch"
        )
        return _contract_result_v2(reason, "launch_validation", counts)
    if type(anchor) is not FineSafetyAnchorV2 or type(anchor.snapshot) is not TerrainSnapshotV2:
        return _contract_result_v2("terrain_snapshot_identity_mismatch", "launch_validation", counts)
    snapshot = anchor.snapshot
    if type(snapshot.geometry) is not FineGridGeometryV2:
        return _contract_result_v2("hopper_terrain_geometry_contract_mismatch", "launch_validation", counts)
    geometry = snapshot.geometry
    if (
        type(geometry.width) is not int
        or type(geometry.height) is not int
        or type(geometry.origin) is not tuple
        or len(geometry.origin) != 2
        or type(geometry.origin[0]) is not float
        or type(geometry.origin[1]) is not float
        or type(geometry.resolution_m) is not float
        or geometry.resolution_m != 0.5
    ):
        return _contract_result_v2("hopper_terrain_geometry_contract_mismatch", "launch_validation", counts)
    try:
        current_hash = snapshot_hash(snapshot)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return _contract_result_v2("terrain_snapshot_hash_mismatch", "launch_validation", counts)
    if current_hash != anchor._snapshot_hash:
        return _contract_result_v2("terrain_snapshot_hash_mismatch", "launch_validation", counts)
    try:
        expired = _deadline_expired_v2(deadline)
    except ValueError:
        return _contract_result_v2("planning_deadline_contract_mismatch", "launch_validation", counts)
    if expired:
        return _result_v2("planning_deadline_expired", "launch_validation", counts)
    profile = audited_candidate.hopper_profile
    speed = profile.launch_speeds_mps[audited_candidate.speed_index]
    elevation = profile.launch_elevations_rad[audited_candidate.elevation_index]
    _, x_direction, y_direction = _canonical_hopper_action_v2(audited_candidate.azimuth_index)
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    vertical_time_scale = vertical_speed / profile.gravity_mps2
    flight_time = 2.0 * vertical_time_scale
    dx = (horizontal_speed * x_direction) * flight_time
    dy = (horizontal_speed * y_direction) * flight_time
    start_z = audited_candidate.support_height_m + profile.launch_reference_height_m
    start = BallisticStartV2(audited_candidate.start_state.x_m, audited_candidate.start_state.y_m, start_z)
    try:
        samples = _call_captured_ballistic_helper_v2(
            HOPPER_RESOURCE_AUTHORITY_V2,
            start,
            speed,
            elevation,
            audited_candidate.azimuth_index,
            profile.gravity_mps2,
        )
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        try:
            _require_canonical_resource_authority_v2(HOPPER_RESOURCE_AUTHORITY_V2)
        except Exception:
            return _contract_result_v2("hopper_authority_contract_mismatch", "arc_candidate_enumeration", counts)
        return _contract_result_v2("hopper_numeric_contract_mismatch", "arc_candidate_enumeration", counts)
    try:
        _require_canonical_resource_authority_v2(HOPPER_RESOURCE_AUTHORITY_V2)
    except Exception:
        return _contract_result_v2("hopper_authority_contract_mismatch", "arc_candidate_enumeration", counts)
    if anchor.snapshot is not snapshot:
        return _contract_result_v2("terrain_snapshot_identity_mismatch", "arc_candidate_enumeration", counts)
    if snapshot_hash(snapshot) != current_hash:
        return _contract_result_v2("terrain_snapshot_hash_mismatch", "arc_candidate_enumeration", counts)
    try:
        if _deadline_expired_v2(deadline):
            return _result_v2("planning_deadline_expired", "arc_candidate_enumeration", counts)
    except ValueError:
        return _contract_result_v2("planning_deadline_contract_mismatch", "arc_candidate_enumeration", counts)
    landing_x = audited_candidate.start_state.x_m + dx
    landing_y = audited_candidate.start_state.y_m + dy
    valid_samples = (
        type(samples) is tuple
        and len(samples) == 2
        and type(samples[0]) is BallisticSampleV2
        and type(samples[1]) is BallisticSampleV2
        and samples[0] == BallisticSampleV2(0.0, start.x_m, start.y_m, start.z_m)
        and samples[1] == BallisticSampleV2(flight_time, landing_x, landing_y, start.z_m)
    )
    if not valid_samples:
        return _contract_result_v2("hopper_numeric_contract_mismatch", "arc_candidate_enumeration", counts)
    try:
        launch_result, _launch_elevation = _launch_footprint_v2(
            audited_candidate,
            anchor,
            deadline,
            counts,
        )
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return _contract_result_v2(
            "hopper_numeric_contract_mismatch",
            "launch_validation",
            counts,
        )
    if launch_result is not None:
        return launch_result
    arc_result = _validate_hopper_arc_partitions_v2(
        audited_candidate, anchor, deadline, samples
    )
    if arc_result.reason_code != "hopper_jump_l2_valid":
        return arc_result
    landing_counts = [count for _counter_id, count in arc_result.replay_work_counts]
    horizontal_range = horizontal_speed * flight_time
    return _validate_hopper_landing_and_stop_v2(
        audited_candidate,
        anchor,
        deadline,
        landing_counts,
        parameter_record,
        landing_x,
        landing_y,
        horizontal_range,
        speed,
    )
