from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from heapq import heappop, heappush
from math import ceil, cos, floor, hypot, inf, isfinite, nextafter, pi, sin

from path_planner.core import Cell
from path_planner.v2.contracts import (
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ValidationLevelV2,
)
from path_planner.v2.profiles import WheelKinematicSQPProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    snapshot_hash,
)
from path_planner.v2.wheel_kinematics import (
    integrate_wheel_segment_v2,
    wheel_pose_at_elapsed_v2,
    wheel_relative_energy_v1,
    wheel_segment_center_control_slew_v1,
)
from path_planner.v2.wheel_sqp_contracts import (
    WHEEL_KINEMATIC_CANONICALIZATION_V2,
    WHEEL_KINEMATIC_CONTROL_SLEW_V2,
    WHEEL_KINEMATIC_L2_VALIDATOR_V2,
    WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2,
    WHEEL_KINEMATIC_SEGMENT_SCHEMA_V2,
    WHEEL_KINEMATIC_SOLVER_CONTRACT_V2,
    L2ReserveModelV1,
    WheelKinematicRouteV2,
    WheelKinematicSegmentV2,
    WheelL2CounterexampleV2,
    WheelSQPModeV2,
    WheelSQPPromotionV2,
    WheelSQPResourceEstimateV1,
    WheelSQPValidationEvidenceV2,
    WheelSQPWorkLimitError,
    WheelSQPWorkLedgerV1,
    WheelTrajectoryL2ResultV2,
    _WHEEL_SQP_L2_RESULT_AUTHORITY,
    _make_wheel_trajectory_l2_result_v2,
)
from path_planner.v2.wheel_sqp_serialization import (
    CanonicalWheelSegmentV1,
    CanonicalWheelCandidateV1,
    WheelSQPCodecError,
    canonicalize_unwrapped_pose_v2,
    canonicalize_wheel_heading_v2,
    canonicalize_wheel_scalar_v2,
    decode_wheel_candidate_v2,
    encode_wheel_candidate_v2,
    encode_wheel_route_v2,
    project_wheel_route_to_candidate_v1,
    wheel_route_hash_v2,
    wheel_segment_hash_v2,
    wheel_sqp_profile_hash_v2,
    wheel_sqp_request_hash_v2,
)


_ZERO_HASH = "0" * 64


def _raise_if_deadline_expired(deadline: object) -> None:
    if deadline.expired:  # type: ignore[attr-defined]
        raise WheelSQPWorkLimitError("planning_deadline_expired")


def _wheel_l2_codec_reseal_tail_s_v2(
    profile: WheelKinematicSQPProfileV2,
    *,
    segment_count: int,
    encoded_state_count: int,
    encoded_scalar_count: int,
) -> float:
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
    for name, value in (
        ("segment_count", segment_count),
        ("encoded_state_count", encoded_state_count),
        ("encoded_scalar_count", encoded_scalar_count),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"{name} must be an exact nonnegative int")
    return L2ReserveModelV1().reserve_s(
        segment_count=segment_count,
        broadphase_cell_bound=0,
        interval_record_bound=0,
        encoded_state_bound=encoded_state_count,
        encoded_scalar_bound=encoded_scalar_count,
        max_segments=profile.max_segments,
        max_l2_candidate_cells=profile.max_l2_candidate_cells,
        max_l2_interval_records=profile.max_l2_interval_records,
        max_encoded_state_bound=encoded_state_count,
        max_encoded_scalar_bound=encoded_scalar_count,
    )


def _wheel_l2_tail_limit_reason_v2(
    deadline: PlanningDeadlineV2,
    codec_reseal_tail_s: float,
) -> str | None:
    if type(deadline) is not PlanningDeadlineV2:
        raise TypeError("deadline must be exact PlanningDeadlineV2")
    if (
        type(codec_reseal_tail_s) is not float
        or not isfinite(codec_reseal_tail_s)
        or codec_reseal_tail_s < 0.0
    ):
        raise TypeError("codec_reseal_tail_s must be an exact finite nonnegative float")
    remaining = deadline.remaining_s
    if remaining <= 0.0:
        return "planning_deadline_expired"
    if remaining <= codec_reseal_tail_s:
        return "wheel_sqp_resource_budget_exceeded"
    return None


def _wheel_candidate_decode_within_estimate_v2(
    candidate_bytes: bytes,
    estimate: WheelSQPResourceEstimateV1,
) -> bool:
    """Bound wire size and canonical segment/sample markers before JSON allocation."""

    if type(candidate_bytes) is not bytes:
        raise TypeError("candidate_bytes must be exact bytes")
    if type(estimate) is not WheelSQPResourceEstimateV1:
        raise TypeError("estimate must be exact WheelSQPResourceEstimateV1")
    if len(candidate_bytes) > estimate.codec_bytes:
        return False
    container_limit = estimate.encoded_state_bound + estimate.segment_count + 2
    separator_limit = (
        estimate.encoded_state_bound + estimate.encoded_scalar_bound
    )
    if (
        candidate_bytes.count(b"{") + candidate_bytes.count(b"[")
        > container_limit
        or candidate_bytes.count(b"}") + candidate_bytes.count(b"]")
        > container_limit
        or candidate_bytes.count(b",") > separator_limit
        or candidate_bytes.count(b":") > separator_limit
    ):
        return False
    segment_count = candidate_bytes.count(b'"segment_hash"')
    sample_array_count = candidate_bytes.count(b'"samples"')
    pose_count = candidate_bytes.count(b'"heading_rad"')
    if (
        segment_count > estimate.segment_count
        or sample_array_count > estimate.segment_count
    ):
        return False

    fixed_pose_count = 3 + 2 * segment_count
    if pose_count < fixed_pose_count:
        return True
    sample_count = pose_count - fixed_pose_count
    encoded_state_count = 3 + 4 * segment_count + 2 * sample_count
    encoded_scalar_count = 14 + 24 * segment_count + 6 * sample_count
    return (
        encoded_state_count <= estimate.encoded_state_bound
        and encoded_scalar_count <= estimate.encoded_scalar_bound
    )


def oriented_rectangle_cell_separation_v2(
    pose: PoseStateV2,
    cell: Cell,
    geometry: FineGridGeometryV2,
    profile: WheelKinematicSQPProfileV2,
) -> float:
    if type(pose) is not PoseStateV2:
        raise TypeError("pose must be exact PoseStateV2")
    if type(cell) is not Cell:
        raise TypeError("cell must be exact Cell")
    if type(geometry) is not FineGridGeometryV2:
        raise TypeError("geometry must be exact FineGridGeometryV2")
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
    if not geometry.in_bounds(cell):
        raise ValueError("cell must be inside the fine grid")

    center = geometry.cell_center(cell)
    forward_x = cos(pose.heading_rad)
    forward_y = sin(pose.heading_rad)
    lateral_x = -forward_y
    lateral_y = forward_x
    delta_x = pose.x_m - center.x
    delta_y = pose.y_m - center.y
    half_length = profile.body_length_m / 2.0 + profile.footprint_safety_margin_m
    half_width = profile.body_width_m / 2.0 + profile.footprint_safety_margin_m
    cell_half_width = geometry.resolution_m / 2.0
    axes = (
        (1.0, 0.0),
        (0.0, 1.0),
        (forward_x, forward_y),
        (lateral_x, lateral_y),
    )
    gaps: list[float] = []
    for axis_x, axis_y in axes:
        center_projection = abs(delta_x * axis_x + delta_y * axis_y)
        support = (
            half_length * abs(forward_x * axis_x + forward_y * axis_y)
            + half_width * abs(lateral_x * axis_x + lateral_y * axis_y)
            + cell_half_width * (abs(axis_x) + abs(axis_y))
        )
        gap = center_projection - support
        if not isfinite(gap):
            raise ValueError("SAT gap must be finite")
        gaps.append(gap)
    return nextafter(max(gaps), -inf)


def _oriented_rectangle_boundary_separations_v2(
    pose: PoseStateV2,
    geometry: FineGridGeometryV2,
    profile: WheelKinematicSQPProfileV2,
) -> tuple[float, float, float, float]:
    if type(pose) is not PoseStateV2:
        raise TypeError("pose must be exact PoseStateV2")
    if type(geometry) is not FineGridGeometryV2:
        raise TypeError("geometry must be exact FineGridGeometryV2")
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")

    lower_x, lower_y = geometry.origin
    upper_x = lower_x + geometry.width * geometry.resolution_m
    upper_y = lower_y + geometry.height * geometry.resolution_m
    if not all(isfinite(value) for value in (lower_x, lower_y, upper_x, upper_y)):
        raise ValueError("derived map boundary must be finite")
    if not (upper_x > lower_x and upper_y > lower_y):
        raise ValueError("derived map boundary must be strictly ordered")

    heading_cos = abs(cos(pose.heading_rad))
    heading_sin = abs(sin(pose.heading_rad))
    half_length = profile.body_length_m / 2.0 + profile.footprint_safety_margin_m
    half_width = profile.body_width_m / 2.0 + profile.footprint_safety_margin_m
    support_x = half_length * heading_cos + half_width * heading_sin
    support_y = half_length * heading_sin + half_width * heading_cos
    margins = (
        pose.x_m - support_x - lower_x,
        upper_x - pose.x_m - support_x,
        pose.y_m - support_y - lower_y,
        upper_y - pose.y_m - support_y,
    )
    if not all(isfinite(value) for value in margins):
        raise ValueError("boundary separation margin must be finite")
    return tuple(nextafter(value, -inf) for value in margins)


_INVALID_REASON_RANK = {
    "terrain_unknown": 0,
    "terrain_hard_obstacle": 1,
    "terrain_not_traversable": 2,
    "terrain_slope_exceeded": 3,
}

_OOB_BOUNDARIES = (
    ("left", Cell(-1, 0)),
    ("right", Cell(-2, 0)),
    ("bottom", Cell(-3, 0)),
    ("top", Cell(-4, 0)),
)


@dataclass(frozen=True, slots=True)
class _WheelSweepSourceV2:
    cell: Cell
    terrain_reason: str
    reason_rank: int
    order_primary: int
    order_secondary: int
    boundary_index: int


def _ordered_wheel_sweep_sources_v2(
    invalid_cells: tuple[tuple[Cell, str], ...],
) -> tuple[_WheelSweepSourceV2, ...]:
    sources = [
        _WheelSweepSourceV2(
            cell=pseudo_cell,
            terrain_reason="terrain_out_of_bounds",
            reason_rank=0,
            order_primary=boundary_index,
            order_secondary=0,
            boundary_index=boundary_index,
        )
        for boundary_index, (_, pseudo_cell) in enumerate(_OOB_BOUNDARIES)
    ]
    sources.extend(
        _WheelSweepSourceV2(
            cell=cell,
            terrain_reason=reason,
            reason_rank=1 + _INVALID_REASON_RANK[reason],
            order_primary=cell.y,
            order_secondary=cell.x,
            boundary_index=-1,
        )
        for cell, reason in sorted(
            invalid_cells,
            key=lambda item: (
                _INVALID_REASON_RANK[item[1]],
                item[0].y,
                item[0].x,
            ),
        )
    )
    return tuple(sources)


def _wheel_sweep_source_gap_v2(
    pose: PoseStateV2,
    source: _WheelSweepSourceV2,
    geometry: FineGridGeometryV2,
    profile: WheelKinematicSQPProfileV2,
) -> float:
    if source.boundary_index >= 0:
        return _oriented_rectangle_boundary_separations_v2(
            pose,
            geometry,
            profile,
        )[source.boundary_index]
    return oriented_rectangle_cell_separation_v2(
        pose,
        source.cell,
        geometry,
        profile,
    )


@dataclass(frozen=True, slots=True)
class _WheelClosedCellRangeV2:
    column_lo: int
    column_hi: int
    row_lo: int
    row_hi: int

    def __post_init__(self) -> None:
        for name in ("column_lo", "column_hi", "row_lo", "row_hi"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact nonnegative int")
        if self.column_hi < self.column_lo or self.row_hi < self.row_lo:
            raise ValueError("closed cell range must be nonempty and ordered")

    @property
    def cell_count(self) -> int:
        return (self.column_hi - self.column_lo + 1) * (
            self.row_hi - self.row_lo + 1
        )


def _wheel_segment_center_bounds_v2(
    segment: CanonicalWheelSegmentV1,
) -> tuple[float, float, float, float]:
    if type(segment) is not CanonicalWheelSegmentV1:
        raise TypeError("segment must be exact CanonicalWheelSegmentV1")

    raw_end = integrate_wheel_segment_v2(
        segment.start_state,
        segment.v_mps,
        segment.omega_radps,
        segment.duration_s,
    )
    center_x = [segment.start_state.x_m, raw_end.x_m, segment.end_state.x_m]
    center_y = [segment.start_state.y_m, raw_end.y_m, segment.end_state.y_m]
    if segment.omega_radps != 0.0 and segment.v_mps != 0.0:
        theta_start = segment.start_state.heading_rad
        theta_end = theta_start + segment.omega_radps * segment.duration_s
        span = abs(theta_end - theta_start)
        if not all(isfinite(value) for value in (theta_start, theta_end, span)):
            raise ValueError("wheel arc heading interval must be finite")
        if span >= 2.0 * pi:
            ratio = segment.v_mps / segment.omega_radps
            circle_x = theta_start
            circle_y = theta_start
            circle_x = segment.start_state.x_m - ratio * sin(circle_x)
            circle_y = segment.start_state.y_m + ratio * cos(circle_y)
            radius = abs(ratio)
            if not all(isfinite(value) for value in (circle_x, circle_y, radius)):
                raise ValueError("wheel arc circle must be finite")
            center_x.extend((circle_x - radius, circle_x + radius))
            center_y.extend((circle_y - radius, circle_y + radius))
        else:
            theta_lo = min(theta_start, theta_end)
            theta_hi = max(theta_start, theta_end)
            for offset, axis in ((pi / 2.0, "x"), (0.0, "y")):
                first = ceil((theta_lo - offset) / pi)
                last = floor((theta_hi - offset) / pi)
                for index in range(first, last + 1):
                    critical_heading = offset + index * pi
                    elapsed = (
                        critical_heading - theta_start
                    ) / segment.omega_radps
                    if not (0.0 <= elapsed <= segment.duration_s):
                        continue
                    pose = wheel_pose_at_elapsed_v2(
                        segment.start_state,
                        segment.v_mps,
                        segment.omega_radps,
                        segment.duration_s,
                        elapsed,
                    )
                    if axis == "x":
                        center_x.append(pose.x_m)
                    else:
                        center_y.append(pose.y_m)

    bounds = (
        nextafter(min(center_x), -inf),
        nextafter(max(center_x), inf),
        nextafter(min(center_y), -inf),
        nextafter(max(center_y), inf),
    )
    if not all(isfinite(value) for value in bounds):
        raise ValueError("wheel segment center bounds must be finite")
    return bounds


def _wheel_closed_aabb_cell_range_v2(
    lower_x: float,
    upper_x: float,
    lower_y: float,
    upper_y: float,
    geometry: FineGridGeometryV2,
) -> _WheelClosedCellRangeV2 | None:
    if type(geometry) is not FineGridGeometryV2:
        raise TypeError("geometry must be exact FineGridGeometryV2")
    bounds = (lower_x, upper_x, lower_y, upper_y)
    if not all(type(value) is float and isfinite(value) for value in bounds):
        raise TypeError("closed AABB bounds must be exact finite floats")
    if lower_x > upper_x or lower_y > upper_y:
        raise ValueError("closed AABB bounds must be ordered")

    origin_x, origin_y = geometry.origin
    resolution = geometry.resolution_m
    map_upper_x = origin_x + geometry.width * resolution
    map_upper_y = origin_y + geometry.height * resolution
    first_edge_x = origin_x + resolution
    first_edge_y = origin_y + resolution
    first_center_x = origin_x + 0.5 * resolution
    first_center_y = origin_y + 0.5 * resolution
    last_lower_x = origin_x + (geometry.width - 1) * resolution
    last_lower_y = origin_y + (geometry.height - 1) * resolution
    last_center_x = origin_x + (geometry.width - 0.5) * resolution
    last_center_y = origin_y + (geometry.height - 0.5) * resolution
    derived = (
        origin_x,
        origin_y,
        map_upper_x,
        map_upper_y,
        first_edge_x,
        first_edge_y,
        first_center_x,
        first_center_y,
        last_lower_x,
        last_lower_y,
        last_center_x,
        last_center_y,
    )
    if not all(isfinite(value) for value in derived):
        raise ValueError("derived fine-grid edges and centers must be finite")
    if not (
        origin_x < first_center_x < first_edge_x
        and origin_y < first_center_y < first_edge_y
        and last_lower_x < last_center_x < map_upper_x
        and last_lower_y < last_center_y < map_upper_y
    ):
        raise ValueError("fine-grid edges and centers must be representable")

    column_lo = ceil((lower_x - origin_x) / resolution) - 1
    column_hi = floor((upper_x - origin_x) / resolution)
    row_lo = ceil((lower_y - origin_y) / resolution) - 1
    row_hi = floor((upper_y - origin_y) / resolution)
    column_lo = max(0, column_lo)
    column_hi = min(geometry.width - 1, column_hi)
    row_lo = max(0, row_lo)
    row_hi = min(geometry.height - 1, row_hi)
    if column_lo > column_hi or row_lo > row_hi:
        return None
    return _WheelClosedCellRangeV2(column_lo, column_hi, row_lo, row_hi)


def _wheel_segment_closed_aabb_cell_range_v2(
    segment: CanonicalWheelSegmentV1,
    geometry: FineGridGeometryV2,
    profile: WheelKinematicSQPProfileV2,
) -> _WheelClosedCellRangeV2 | None:
    if type(segment) is not CanonicalWheelSegmentV1:
        raise TypeError("segment must be exact CanonicalWheelSegmentV1")
    if type(geometry) is not FineGridGeometryV2:
        raise TypeError("geometry must be exact FineGridGeometryV2")
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")

    center_lower_x, center_upper_x, center_lower_y, center_upper_y = (
        _wheel_segment_center_bounds_v2(segment)
    )

    half_length = profile.body_length_m / 2.0 + profile.footprint_safety_margin_m
    half_width = profile.body_width_m / 2.0 + profile.footprint_safety_margin_m
    radius = hypot(half_length, half_width)
    lower_x = nextafter(center_lower_x - radius, -inf)
    upper_x = nextafter(center_upper_x + radius, inf)
    lower_y = nextafter(center_lower_y - radius, -inf)
    upper_y = nextafter(center_upper_y + radius, inf)
    if not all(isfinite(value) for value in (lower_x, upper_x, lower_y, upper_y)):
        raise ValueError("wheel segment broadphase AABB must be finite")
    return _wheel_closed_aabb_cell_range_v2(
        lower_x, upper_x, lower_y, upper_y, geometry
    )


def _iter_wheel_closed_cell_range_v2(
    cell_range: _WheelClosedCellRangeV2,
):
    if type(cell_range) is not _WheelClosedCellRangeV2:
        raise TypeError("cell_range must be exact _WheelClosedCellRangeV2")
    for row in range(cell_range.row_lo, cell_range.row_hi + 1):
        for column in range(cell_range.column_lo, cell_range.column_hi + 1):
            yield Cell(column, row)


@dataclass(frozen=True, slots=True)
class _WheelSegmentSweepResultV2:
    passed: bool
    reason_code: str | None
    counterexample: WheelL2CounterexampleV2 | None
    checked_interval_count: int
    created_record_count: int


def prove_wheel_segment_sweep_v2(
    segment: CanonicalWheelSegmentV1,
    segment_index: int,
    candidate_hash: str,
    invalid_cells: tuple[tuple[Cell, str], ...],
    anchor: FineSafetyAnchorV2,
    profile: WheelKinematicSQPProfileV2,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
    estimate: WheelSQPResourceEstimateV1,
    *,
    base_codec_delta_bytes: int,
    created_record_offset: int,
    interval_record_limit: int,
    codec_reseal_tail_s: float,
) -> _WheelSegmentSweepResultV2:
    if type(segment) is not CanonicalWheelSegmentV1:
        raise TypeError("segment must be exact CanonicalWheelSegmentV1")
    if type(segment_index) is not int or segment_index < 0:
        raise TypeError("segment_index must be an exact nonnegative int")
    if (
        type(candidate_hash) is not str
        or len(candidate_hash) != 64
        or any(character not in "0123456789abcdef" for character in candidate_hash)
    ):
        raise ValueError("candidate_hash must be a lowercase SHA-256 digest")
    if type(invalid_cells) is not tuple:
        raise TypeError("invalid_cells must be an exact tuple")
    if type(anchor) is not FineSafetyAnchorV2:
        raise TypeError("anchor must be exact FineSafetyAnchorV2")
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
    if type(deadline) is not PlanningDeadlineV2:
        raise TypeError("deadline must be exact PlanningDeadlineV2")
    if type(ledger) is not WheelSQPWorkLedgerV1:
        raise TypeError("ledger must be exact WheelSQPWorkLedgerV1")
    if type(estimate) is not WheelSQPResourceEstimateV1:
        raise TypeError("estimate must be exact WheelSQPResourceEstimateV1")
    for name, value in (
        ("base_codec_delta_bytes", base_codec_delta_bytes),
        ("created_record_offset", created_record_offset),
        ("interval_record_limit", interval_record_limit),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"{name} must be an exact nonnegative int")
    if interval_record_limit > profile.max_l2_interval_records:
        raise ValueError("interval_record_limit exceeds profile cap")
    if (
        type(codec_reseal_tail_s) is not float
        or not isfinite(codec_reseal_tail_s)
        or codec_reseal_tail_s < 0.0
    ):
        raise TypeError("codec_reseal_tail_s must be an exact finite nonnegative float")
    for item in invalid_cells:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not Cell
            or item[1] not in _INVALID_REASON_RANK
        ):
            raise TypeError("invalid_cells entries must be exact (Cell, reason) tuples")
    snapshot_digest = snapshot_hash(anchor.snapshot)
    epsilon = profile.continuous_separation_epsilon_m
    geometry = anchor.snapshot.geometry
    sources = _ordered_wheel_sweep_sources_v2(invalid_cells)

    def limit_reason() -> str | None:
        return _wheel_l2_tail_limit_reason_v2(deadline, codec_reseal_tail_s)

    def counterexample(
        cell: Cell,
        terrain_reason: str,
        lo: float,
        mid: float,
        hi: float,
        *,
        repairable: bool,
    ) -> WheelL2CounterexampleV2:
        return WheelL2CounterexampleV2(
            segment_index=segment_index,
            candidate_segment_hash=segment.segment_hash,
            time_fraction_lo=lo,
            time_fraction_mid=mid,
            time_fraction_hi=hi,
            cell=cell,
            terrain_reason=terrain_reason,
            snapshot_hash=snapshot_digest,
            candidate_hash=candidate_hash,
            repairable=repairable,
        )

    reason_code = limit_reason()
    if reason_code is not None:
        source = sources[0]
        return _WheelSegmentSweepResultV2(
            False,
            reason_code,
            counterexample(
                source.cell,
                source.terrain_reason,
                0.0,
                0.5,
                1.0,
                repairable=False,
            ),
            0,
            0,
        )
    try:
        raw_end = integrate_wheel_segment_v2(
            segment.start_state,
            segment.v_mps,
            segment.omega_radps,
            segment.duration_s,
        )
    except (TypeError, ValueError, OverflowError):
        source = sources[0]
        return _WheelSegmentSweepResultV2(
            False,
            "wheel_sqp_numeric_contract_failed",
            counterexample(
                source.cell,
                source.terrain_reason,
                0.0,
                0.5,
                1.0,
                repairable=False,
            ),
            0,
            0,
        )
    reason_code = limit_reason()
    if reason_code is not None:
        source = sources[0]
        return _WheelSegmentSweepResultV2(
            False,
            reason_code,
            counterexample(
                source.cell,
                source.terrain_reason,
                0.0,
                0.5,
                1.0,
                repairable=False,
            ),
            0,
            0,
        )
    endpoint_poses = (
        (0.0, segment.start_state),
        (1.0, raw_end),
        (1.0, segment.end_state),
    )
    raw_gaps = [0.0] * len(sources)
    declared_gaps = [0.0] * len(sources)
    for endpoint_rank, (fraction, pose) in enumerate(endpoint_poses):
        reason_code = limit_reason()
        if reason_code is not None:
            source = sources[0]
            return _WheelSegmentSweepResultV2(
                False,
                reason_code,
                counterexample(
                    source.cell,
                    source.terrain_reason,
                    0.0,
                    0.5,
                    1.0,
                    repairable=False,
                ),
                0,
                0,
            )
        for source_index, source in enumerate(sources):
            reason_code = limit_reason()
            if reason_code is not None:
                return _WheelSegmentSweepResultV2(
                    False,
                    reason_code,
                    counterexample(
                        source.cell,
                        source.terrain_reason,
                        fraction,
                        fraction,
                        fraction,
                        repairable=False,
                    ),
                    0,
                    0,
                )
            try:
                gap = _wheel_sweep_source_gap_v2(
                    pose,
                    source,
                    geometry,
                    profile,
                )
            except (TypeError, ValueError, OverflowError):
                return _WheelSegmentSweepResultV2(
                    False,
                    "wheel_sqp_numeric_contract_failed",
                    counterexample(
                        source.cell,
                        source.terrain_reason,
                        fraction,
                        fraction,
                        fraction,
                        repairable=False,
                    ),
                    0,
                    0,
                )
            reason_code = limit_reason()
            if reason_code is not None:
                return _WheelSegmentSweepResultV2(
                    False,
                    reason_code,
                    counterexample(
                        source.cell,
                        source.terrain_reason,
                        fraction,
                        fraction,
                        fraction,
                        repairable=False,
                    ),
                    0,
                    0,
                )
            if endpoint_rank == 1:
                raw_gaps[source_index] = gap
            elif endpoint_rank == 2:
                declared_gaps[source_index] = gap
            if gap <= epsilon:
                return _WheelSegmentSweepResultV2(
                    False,
                    "wheel_sqp_candidate_l2_rejected",
                    counterexample(
                        source.cell,
                        source.terrain_reason,
                        fraction,
                        fraction,
                        fraction,
                        repairable=source.boundary_index < 0,
                    ),
                    0,
                    0,
                )

    half_length = profile.body_length_m / 2.0 + profile.footprint_safety_margin_m
    half_width = profile.body_width_m / 2.0 + profile.footprint_safety_margin_m
    try:
        delta_x = segment.end_state.x_m - raw_end.x_m
        delta_y = segment.end_state.y_m - raw_end.y_m
        delta_heading = segment.end_state.heading_rad - raw_end.heading_rad
        footprint_radius = hypot(half_length, half_width)
        translation_bound = hypot(delta_x, delta_y)
        rotation_bound = footprint_radius * abs(delta_heading)
        connector_bound = nextafter(translation_bound + rotation_bound, inf)
        seam_rhs = nextafter(epsilon + connector_bound, inf)
        if not all(
            isfinite(value)
            for value in (
                delta_x,
                delta_y,
                delta_heading,
                footprint_radius,
                translation_bound,
                rotation_bound,
                connector_bound,
                seam_rhs,
            )
        ):
            raise ValueError("canonical seam bound must be finite")
    except (TypeError, ValueError, OverflowError):
        source = sources[0]
        return _WheelSegmentSweepResultV2(
            False,
            "wheel_sqp_numeric_contract_failed",
            counterexample(
                source.cell,
                source.terrain_reason,
                1.0,
                1.0,
                1.0,
                repairable=False,
            ),
            0,
            0,
        )
    for source_index, source in enumerate(sources):
        reason_code = limit_reason()
        if reason_code is not None:
            return _WheelSegmentSweepResultV2(
                False,
                reason_code,
                counterexample(
                    source.cell,
                    source.terrain_reason,
                    1.0,
                    1.0,
                    1.0,
                    repairable=False,
                ),
                0,
                0,
            )
        if min(raw_gaps[source_index], declared_gaps[source_index]) <= seam_rhs:
            return _WheelSegmentSweepResultV2(
                False,
                "wheel_sqp_candidate_l2_rejected",
                counterexample(
                    source.cell,
                    source.terrain_reason,
                    1.0,
                    1.0,
                    1.0,
                    repairable=False,
                ),
                0,
                0,
            )

    tick_scale = 1 << profile.max_l2_subdivision_depth
    heap: list[tuple[int, int, int, int, int, int, int]] = []
    created = 0
    checked = 0

    def fail_for_record(
        reason_code: str,
        source: _WheelSweepSourceV2,
        lo_tick: int,
        hi_tick: int,
    ) -> _WheelSegmentSweepResultV2:
        mid_tick = (lo_tick + hi_tick) // 2
        return _WheelSegmentSweepResultV2(
            False,
            reason_code,
            counterexample(
                source.cell,
                source.terrain_reason,
                lo_tick / tick_scale,
                mid_tick / tick_scale,
                hi_tick / tick_scale,
                repairable=False,
            ),
            checked,
            created,
        )

    def admit_record(
        source: _WheelSweepSourceV2,
        lo_tick: int,
        hi_tick: int,
    ) -> _WheelSegmentSweepResultV2 | None:
        nonlocal created
        prospective_global_count = created_record_offset + created + 1
        reason_code = limit_reason()
        if reason_code is not None:
            return fail_for_record(reason_code, source, lo_tick, hi_tick)
        if prospective_global_count > interval_record_limit:
            return fail_for_record(
                "wheel_sqp_candidate_l2_rejected",
                source,
                lo_tick,
                hi_tick,
            )
        try:
            ledger.admit_validation_delta(
                estimate,
                candidate_hash,
                memory_bytes=(
                    base_codec_delta_bytes + 416 * prospective_global_count
                ),
                route_states=0,
            )
        except WheelSQPWorkLimitError as exc:
            return fail_for_record(
                exc.reason_code, source, lo_tick, hi_tick
            )
        except (TypeError, ValueError, OverflowError):
            return fail_for_record(
                "wheel_sqp_numeric_contract_failed",
                source,
                lo_tick,
                hi_tick,
            )
        reason_code = limit_reason()
        if reason_code is not None:
            return fail_for_record(reason_code, source, lo_tick, hi_tick)
        created += 1
        return None

    for source_index, source in enumerate(sources):
        admission_failure = admit_record(source, 0, tick_scale)
        if admission_failure is not None:
            return admission_failure
        heappush(
            heap,
            (
                0,
                source.reason_rank,
                source.order_primary,
                source.order_secondary,
                tick_scale,
                0,
                source_index,
            ),
        )

    while heap:
        (
            lo_tick,
            _,
            _,
            _,
            hi_tick,
            depth,
            source_index,
        ) = heappop(heap)
        source = sources[source_index]
        checked += 1
        reason_code = limit_reason()
        if reason_code is not None:
            return fail_for_record(reason_code, source, lo_tick, hi_tick)
        mid_tick = (lo_tick + hi_tick) // 2
        if mid_tick in (lo_tick, hi_tick):
            return fail_for_record(
                "wheel_sqp_candidate_l2_rejected",
                source,
                lo_tick,
                hi_tick,
            )
        try:
            midpoint = integrate_wheel_segment_v2(
                segment.start_state,
                segment.v_mps,
                segment.omega_radps,
                segment.duration_s * (mid_tick / tick_scale),
            )
            gap = _wheel_sweep_source_gap_v2(
                midpoint,
                source,
                geometry,
                profile,
            )
            reason_code = limit_reason()
            if reason_code is not None:
                return fail_for_record(reason_code, source, lo_tick, hi_tick)
            if gap <= epsilon:
                return _WheelSegmentSweepResultV2(
                    False,
                    "wheel_sqp_candidate_l2_rejected",
                    counterexample(
                        source.cell,
                        source.terrain_reason,
                        lo_tick / tick_scale,
                        mid_tick / tick_scale,
                        hi_tick / tick_scale,
                        repairable=source.boundary_index < 0,
                    ),
                    checked,
                    created,
                )
            half_time = (
                segment.duration_s * ((hi_tick - lo_tick) / tick_scale) * 0.5
            )
            if source.boundary_index >= 0:
                distance_upper = 0.0
                lipschitz_upper = nextafter(
                    abs(segment.v_mps)
                    + abs(segment.omega_radps) * (half_length + half_width),
                    inf,
                )
            else:
                cell_center = geometry.cell_center(source.cell)
                distance_upper = nextafter(
                    hypot(
                        midpoint.x_m - cell_center.x,
                        midpoint.y_m - cell_center.y,
                    )
                    + abs(segment.v_mps) * half_time,
                    inf,
                )
                lipschitz_upper = nextafter(
                    abs(segment.v_mps)
                    + abs(segment.omega_radps)
                    * (
                        distance_upper
                        + half_length
                        + half_width
                        + geometry.resolution_m
                    ),
                    inf,
                )
            proof_rhs = nextafter(
                lipschitz_upper * half_time + epsilon,
                inf,
            )
            if not all(
                isfinite(value)
                for value in (distance_upper, lipschitz_upper, proof_rhs)
            ):
                return fail_for_record(
                    "wheel_sqp_numeric_contract_failed",
                    source,
                    lo_tick,
                    hi_tick,
                )
            if gap > proof_rhs:
                continue
        except (TypeError, ValueError, OverflowError):
            return fail_for_record(
                "wheel_sqp_numeric_contract_failed",
                source,
                lo_tick,
                hi_tick,
            )
        if depth >= profile.max_l2_subdivision_depth:
            return fail_for_record(
                "wheel_sqp_candidate_l2_rejected",
                source,
                lo_tick,
                hi_tick,
            )
        for child_lo, child_hi in ((lo_tick, mid_tick), (mid_tick, hi_tick)):
            admission_failure = admit_record(source, child_lo, child_hi)
            if admission_failure is not None:
                return admission_failure
            heappush(
                heap,
                (
                    child_lo,
                    source.reason_rank,
                    source.order_primary,
                    source.order_secondary,
                    child_hi,
                    depth + 1,
                    source_index,
                ),
            )
    reason_code = limit_reason()
    if reason_code is not None:
        return fail_for_record(reason_code, sources[0], 0, tick_scale)
    return _WheelSegmentSweepResultV2(True, None, None, checked, created)


def _l2_receipt_v2(
    *,
    passed: bool,
    reason_code: str | None,
    candidate: CanonicalWheelCandidateV1 | None,
    counterexample: WheelL2CounterexampleV2 | None,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    deadline: PlanningDeadlineV2,
    input_bytes_hash: str,
    checked_cell_count: int = 0,
    checked_interval_count: int = 0,
    goal_position_error_m: float | None = None,
    goal_heading_error_rad: float | None = None,
) -> WheelTrajectoryL2ResultV2:
    return _make_wheel_trajectory_l2_result_v2(
        passed=passed,
        reason_code=reason_code,
        route=None,
        evidence=None,
        counterexample=counterexample,
        actual_start=None if candidate is None else candidate.start_state,
        actual_goal=None if candidate is None else candidate.actual_endpoint,
        goal_position_error_m=goal_position_error_m,
        goal_heading_error_rad=goal_heading_error_rad,
        checked_cell_count=checked_cell_count,
        checked_interval_count=checked_interval_count,
        candidate_hash=None if candidate is None else candidate.candidate_hash,
        request_hash=None if candidate is None else candidate.request_hash,
        profile_hash=None if candidate is None else candidate.profile_hash,
        terrain_snapshot_hash=(
            None if candidate is None else candidate.terrain_snapshot_hash
        ),
        capability_revision=(
            None if candidate is None else candidate.capability_revision
        ),
        solver_contract_id=(
            None if candidate is None else candidate.solver_contract_id
        ),
        canonicalization_id=(
            None if candidate is None else candidate.canonicalization_id
        ),
        control_slew_id=None if candidate is None else candidate.control_slew_id,
        observation_source_id=(
            None if candidate is None else candidate.observation_source_id
        ),
        validator_contract_id=(
            None if candidate is None else WHEEL_KINEMATIC_L2_VALIDATOR_V2
        ),
        request=request,
        profile=profile,
        deadline=deadline,
        input_bytes_hash=input_bytes_hash,
    )


def _failed_l2_receipt_v2(
    reason_code: str,
    candidate: CanonicalWheelCandidateV1 | None,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    deadline: PlanningDeadlineV2,
    input_bytes_hash: str,
    *,
    counterexample: WheelL2CounterexampleV2 | None = None,
    checked_cell_count: int = 0,
    checked_interval_count: int = 0,
    goal_position_error_m: float | None = None,
    goal_heading_error_rad: float | None = None,
) -> WheelTrajectoryL2ResultV2:
    return _l2_receipt_v2(
        passed=False,
        reason_code=reason_code,
        candidate=candidate,
        counterexample=counterexample,
        request=request,
        profile=profile,
        deadline=deadline,
        input_bytes_hash=input_bytes_hash,
        checked_cell_count=checked_cell_count,
        checked_interval_count=checked_interval_count,
        goal_position_error_m=goal_position_error_m,
        goal_heading_error_rad=goal_heading_error_rad,
    )


def _expected_mode_v2(v_mps: float, omega_radps: float) -> WheelSQPModeV2:
    if v_mps > 0.0:
        return WheelSQPModeV2.FORWARD
    if v_mps < 0.0:
        return WheelSQPModeV2.REVERSE
    if omega_radps > 0.0:
        return WheelSQPModeV2.TURN_LEFT
    if omega_radps < 0.0:
        return WheelSQPModeV2.TURN_RIGHT
    return WheelSQPModeV2.STOP


def _candidate_identity_matches_v2(
    candidate: CanonicalWheelCandidateV1,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    terrain_snapshot_hash: str,
) -> bool:
    return (
        candidate.request_hash
        == wheel_sqp_request_hash_v2(request, terrain_snapshot_hash)
        and candidate.profile_hash == wheel_sqp_profile_hash_v2(profile)
        and candidate.terrain_snapshot_hash == terrain_snapshot_hash
        and candidate.capability_revision == profile.profile.capability_revision
        and candidate.solver_contract_id == WHEEL_KINEMATIC_SOLVER_CONTRACT_V2
        and candidate.canonicalization_id == WHEEL_KINEMATIC_CANONICALIZATION_V2
        and candidate.control_slew_id == WHEEL_KINEMATIC_CONTROL_SLEW_V2
        and candidate.observation_source_id == WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2
        and candidate.start_state == request.start_state
        and candidate.requested_goal_state == request.goal_state
    )


def validate_wheel_sqp_candidate_l2(
    candidate_bytes: bytes,
    request: PlanningRequestV2,
    anchor: FineSafetyAnchorV2,
    profile: WheelKinematicSQPProfileV2,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
) -> WheelTrajectoryL2ResultV2:
    if type(candidate_bytes) is not bytes:
        raise TypeError("candidate_bytes must be exact bytes")
    if type(request) is not PlanningRequestV2:
        raise TypeError("request must be exact PlanningRequestV2")
    if type(anchor) is not FineSafetyAnchorV2:
        raise TypeError("anchor must be exact FineSafetyAnchorV2")
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
    if type(deadline) is not PlanningDeadlineV2:
        raise TypeError("deadline must be exact PlanningDeadlineV2")
    if type(ledger) is not WheelSQPWorkLedgerV1:
        raise TypeError("ledger must be exact WheelSQPWorkLedgerV1")
    if ledger.deadline is not deadline or ledger.resource_budget is not request.resource_budget:
        raise ValueError("wheel SQP validation resource authority mismatch")

    if deadline.expired:
        return _failed_l2_receipt_v2(
            "planning_deadline_expired",
            None,
            request,
            profile,
            deadline,
            _ZERO_HASH,
        )
    input_bytes_hash = sha256(candidate_bytes).hexdigest()
    if deadline.expired:
        return _failed_l2_receipt_v2(
            "planning_deadline_expired",
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    if len(candidate_bytes) > ledger.effective_memory_limit_bytes:
        return _failed_l2_receipt_v2(
            "wheel_sqp_resource_budget_exceeded",
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    try:
        estimate = ledger.latest_attempt_estimate(request, profile)
    except WheelSQPWorkLimitError as exc:
        reason_code = (
            "planning_deadline_expired" if deadline.expired else exc.reason_code
        )
        return _failed_l2_receipt_v2(
            reason_code,
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    except (TypeError, ValueError):
        reason_code = (
            "planning_deadline_expired"
            if deadline.expired
            else "wheel_sqp_identity_mismatch"
        )
        return _failed_l2_receipt_v2(
            reason_code,
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    if deadline.expired:
        return _failed_l2_receipt_v2(
            "planning_deadline_expired",
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    # This fixed ASCII schema's canonical encoder never emits JSON escapes.
    has_noncanonical_escape = b"\\" in candidate_bytes
    if deadline.expired:
        return _failed_l2_receipt_v2(
            "planning_deadline_expired",
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    if has_noncanonical_escape:
        return _failed_l2_receipt_v2(
            "wheel_sqp_identity_mismatch",
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    decode_within_estimate = _wheel_candidate_decode_within_estimate_v2(
        candidate_bytes,
        estimate,
    )
    if deadline.expired:
        return _failed_l2_receipt_v2(
            "planning_deadline_expired",
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    if not decode_within_estimate:
        return _failed_l2_receipt_v2(
            "wheel_sqp_resource_budget_exceeded",
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    try:
        candidate = decode_wheel_candidate_v2(candidate_bytes)
    except WheelSQPCodecError as exc:
        if deadline.expired:
            return _failed_l2_receipt_v2(
                "planning_deadline_expired",
                None,
                request,
                profile,
                deadline,
                input_bytes_hash,
            )
        return _failed_l2_receipt_v2(
            exc.reason_code,
            None,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    if deadline.expired:
        return _failed_l2_receipt_v2(
            "planning_deadline_expired",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )

    terrain_digest = snapshot_hash(request.terrain_snapshot)
    identity_mismatch = (
        anchor.snapshot is not request.terrain_snapshot
        or request.platform_profile_id != profile.profile.profile_id
        or not _candidate_identity_matches_v2(
            candidate, request, profile, terrain_digest
        )
    )
    if deadline.expired:
        return _failed_l2_receipt_v2(
            "planning_deadline_expired",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    if identity_mismatch:
        return _failed_l2_receipt_v2(
            "wheel_sqp_identity_mismatch",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )

    if estimate.segment_count != len(candidate.segments):
        return _failed_l2_receipt_v2(
            "wheel_sqp_identity_mismatch",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )

    sample_count = sum(len(segment.samples) for segment in candidate.segments)
    encoded_state_count = 3 + 4 * len(candidate.segments) + 2 * sample_count
    encoded_scalar_count = 14 + 24 * len(candidate.segments) + 6 * sample_count
    actual_codec_bytes = 64 * encoded_state_count + 16 * encoded_scalar_count
    base_codec_delta_bytes = max(0, actual_codec_bytes - estimate.codec_bytes)
    codec_reseal_tail_s = _wheel_l2_codec_reseal_tail_s_v2(
        profile,
        segment_count=len(candidate.segments),
        encoded_state_count=encoded_state_count,
        encoded_scalar_count=encoded_scalar_count,
    )

    def tail_failure(
        *,
        checked_cell_count: int = 0,
        checked_interval_count: int = 0,
        counterexample: WheelL2CounterexampleV2 | None = None,
        goal_position_error_m: float | None = None,
        goal_heading_error_rad: float | None = None,
    ) -> WheelTrajectoryL2ResultV2 | None:
        reason_code = _wheel_l2_tail_limit_reason_v2(
            deadline, codec_reseal_tail_s
        )
        if reason_code is None:
            return None
        return _failed_l2_receipt_v2(
            reason_code,
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
            checked_cell_count=checked_cell_count,
            checked_interval_count=checked_interval_count,
            counterexample=counterexample,
            goal_position_error_m=goal_position_error_m,
            goal_heading_error_rad=goal_heading_error_rad,
        )

    try:
        ledger.admit_validation_delta(
            estimate,
            candidate.candidate_hash,
            memory_bytes=base_codec_delta_bytes,
            route_states=max(0, encoded_state_count - estimate.encoded_state_bound),
        )
    except WheelSQPWorkLimitError as exc:
        return _failed_l2_receipt_v2(
            exc.reason_code,
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )
    except (TypeError, ValueError):
        return _failed_l2_receipt_v2(
            "wheel_sqp_identity_mismatch",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )

    broadphase_ranges: list[_WheelClosedCellRangeV2 | None] = []
    broadphase_pair_count = 0
    try:
        for segment in candidate.segments:
            if deadline.expired:
                return _failed_l2_receipt_v2(
                    "planning_deadline_expired",
                    candidate,
                    request,
                    profile,
                    deadline,
                    input_bytes_hash,
                    checked_cell_count=broadphase_pair_count,
                )
            cell_range = _wheel_segment_closed_aabb_cell_range_v2(
                segment, anchor.snapshot.geometry, profile
            )
            if deadline.expired:
                return _failed_l2_receipt_v2(
                    "planning_deadline_expired",
                    candidate,
                    request,
                    profile,
                    deadline,
                    input_bytes_hash,
                    checked_cell_count=broadphase_pair_count,
                )
            broadphase_ranges.append(cell_range)
            broadphase_pair_count += len(_OOB_BOUNDARIES) + (
                0 if cell_range is None else cell_range.cell_count
            )
    except (TypeError, ValueError, OverflowError):
        if deadline.expired:
            return _failed_l2_receipt_v2(
                "planning_deadline_expired",
                candidate,
                request,
                profile,
                deadline,
                input_bytes_hash,
                checked_cell_count=broadphase_pair_count,
            )
        return _failed_l2_receipt_v2(
            "wheel_sqp_numeric_contract_failed",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
            checked_cell_count=0,
        )
    if broadphase_pair_count > profile.max_l2_candidate_cells:
        return _failed_l2_receipt_v2(
            "wheel_sqp_resource_budget_exceeded",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
            checked_cell_count=broadphase_pair_count,
        )

    interval_factor = (1 << (profile.max_l2_subdivision_depth + 1)) - 1
    actual_interval_record_bound = min(
        profile.max_l2_interval_records,
        broadphase_pair_count * interval_factor,
    )
    actual_reserve = L2ReserveModelV1().assess(
        deadline,
        segment_count=len(candidate.segments),
        broadphase_cell_bound=broadphase_pair_count,
        interval_record_bound=actual_interval_record_bound,
        encoded_state_bound=encoded_state_count,
        encoded_scalar_bound=encoded_scalar_count,
        max_segments=profile.max_segments,
        max_l2_candidate_cells=profile.max_l2_candidate_cells,
        max_l2_interval_records=profile.max_l2_interval_records,
        max_encoded_state_bound=encoded_state_count,
        max_encoded_scalar_bound=encoded_scalar_count,
    )
    if not actual_reserve.accepted:
        assert actual_reserve.reason_code is not None
        return _failed_l2_receipt_v2(
            actual_reserve.reason_code,
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
            checked_cell_count=broadphase_pair_count,
        )

    stopped = tail_failure(checked_cell_count=broadphase_pair_count)
    if stopped is not None:
        return stopped
    if request.objective_profile.risk_weight != 0.0:
        return _failed_l2_receipt_v2(
            "wheel_sqp_objective_unsupported",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
            checked_cell_count=broadphase_pair_count,
        )

    current = request.start_state
    previous = None
    total_distance = 0.0
    total_energy = 0.0
    total_duration = 0.0
    previous_translation_sign = 0
    stop_since_translation = False
    try:
        for segment in candidate.segments:
            stopped = tail_failure(checked_cell_count=broadphase_pair_count)
            if stopped is not None:
                return stopped
            if segment.start_state != current:
                return _failed_l2_receipt_v2(
                    "wheel_sqp_candidate_l2_rejected",
                    candidate,
                    request,
                    profile,
                    deadline,
                    input_bytes_hash,
                )
            if (
                segment.mode is not _expected_mode_v2(
                    segment.v_mps, segment.omega_radps
                )
                or not (
                    profile.min_segment_duration_s
                    <= segment.duration_s
                    <= profile.max_segment_duration_s
                )
                or abs(segment.v_mps) > profile.max_speed_mps
                or abs(segment.omega_radps) > profile.max_angular_speed_radps
                or abs(segment.omega_radps * segment.duration_s)
                > profile.max_segment_heading_change_rad
                or (segment.v_mps < 0.0 and not profile.reverse_enabled)
                or (
                    segment.v_mps == 0.0
                    and segment.omega_radps != 0.0
                    and not profile.turn_in_place_enabled
                )
                or 0.0 < abs(segment.v_mps) < profile.min_nonzero_control
                or 0.0 < abs(segment.omega_radps) < profile.min_nonzero_control
            ):
                return _failed_l2_receipt_v2(
                    "wheel_sqp_candidate_l2_rejected",
                    candidate,
                    request,
                    profile,
                    deadline,
                    input_bytes_hash,
                )
            if segment.mode is WheelSQPModeV2.STOP:
                stop_since_translation = True
            elif segment.mode in (WheelSQPModeV2.FORWARD, WheelSQPModeV2.REVERSE):
                current_translation_sign = (
                    1 if segment.mode is WheelSQPModeV2.FORWARD else -1
                )
                if (
                    previous_translation_sign != 0
                    and current_translation_sign != previous_translation_sign
                    and not stop_since_translation
                ):
                    return _failed_l2_receipt_v2(
                        "wheel_sqp_candidate_l2_rejected",
                        candidate,
                        request,
                        profile,
                        deadline,
                        input_bytes_hash,
                    )
                previous_translation_sign = current_translation_sign
                stop_since_translation = False
            if previous is not None:
                accel, decel, angular_accel = wheel_segment_center_control_slew_v1(
                    previous.v_mps,
                    previous.omega_radps,
                    previous.duration_s,
                    segment.v_mps,
                    segment.omega_radps,
                    segment.duration_s,
                )
                stopped = tail_failure(checked_cell_count=broadphase_pair_count)
                if stopped is not None:
                    return stopped
                if (
                    accel > profile.max_linear_accel_mps2
                    or decel > profile.max_linear_decel_mps2
                    or angular_accel > profile.max_angular_accel_radps2
                ):
                    return _failed_l2_receipt_v2(
                        "wheel_sqp_candidate_l2_rejected",
                        candidate,
                        request,
                        profile,
                        deadline,
                        input_bytes_hash,
                    )
            raw_endpoint = integrate_wheel_segment_v2(
                current,
                segment.v_mps,
                segment.omega_radps,
                segment.duration_s,
            )
            stopped = tail_failure(checked_cell_count=broadphase_pair_count)
            if stopped is not None:
                return stopped
            if canonicalize_unwrapped_pose_v2(raw_endpoint) != segment.end_state:
                return _failed_l2_receipt_v2(
                    "wheel_sqp_candidate_l2_rejected",
                    candidate,
                    request,
                    profile,
                    deadline,
                    input_bytes_hash,
                )
            expected_distance = canonicalize_wheel_scalar_v2(
                abs(segment.v_mps) * segment.duration_s
            )
            expected_energy = canonicalize_wheel_scalar_v2(
                wheel_relative_energy_v1(
                    segment.v_mps,
                    segment.omega_radps,
                    segment.duration_s,
                    profile,
                )
            )
            if (
                segment.distance_m != expected_distance
                or segment.relative_energy != expected_energy
            ):
                return _failed_l2_receipt_v2(
                    "wheel_sqp_candidate_l2_rejected",
                    candidate,
                    request,
                    profile,
                    deadline,
                    input_bytes_hash,
                )
            total_distance += segment.distance_m
            total_energy += segment.relative_energy
            total_duration += segment.duration_s
            current = segment.end_state
            previous = segment
            stopped = tail_failure(checked_cell_count=broadphase_pair_count)
            if stopped is not None:
                return stopped
    except (TypeError, ValueError, OverflowError):
        return _failed_l2_receipt_v2(
            "wheel_sqp_numeric_contract_failed",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )

    canonical_distance = canonicalize_wheel_scalar_v2(total_distance)
    canonical_energy = canonicalize_wheel_scalar_v2(total_energy)
    canonical_duration = canonicalize_wheel_scalar_v2(total_duration)
    objective = request.objective_profile
    canonical_cost = canonicalize_wheel_scalar_v2(
        objective.distance_weight * canonical_distance
        + objective.energy_weight * canonical_energy
        + objective.time_weight * canonical_duration
    )
    if (
        candidate.actual_endpoint != current
        or candidate.total_distance_m != canonical_distance
        or candidate.total_relative_energy != canonical_energy
        or candidate.total_duration_s != canonical_duration
        or candidate.total_cost != canonical_cost
    ):
        return _failed_l2_receipt_v2(
            "wheel_sqp_identity_mismatch",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
        )

    position_error = hypot(
        current.x_m - request.goal_state.x_m,
        current.y_m - request.goal_state.y_m,
    )
    heading_error = abs(
        canonicalize_wheel_heading_v2(
            current.heading_rad - request.goal_state.heading_rad
        )
    )
    if (
        position_error > profile.profile.goal_position_tolerance_m
        or heading_error > profile.profile.goal_heading_tolerance_rad
    ):
        return _failed_l2_receipt_v2(
            "wheel_sqp_goal_tolerance_exceeded",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
            goal_position_error_m=position_error,
            goal_heading_error_rad=heading_error,
        )

    checked_intervals = 0
    created_records = 0

    def preadmit_terrain_roots(prospective_root_count: int) -> str | None:
        try:
            ledger.admit_validation_delta(
                estimate,
                candidate.candidate_hash,
                memory_bytes=(
                    base_codec_delta_bytes
                    + 416 * (created_records + prospective_root_count)
                ),
                route_states=0,
            )
        except WheelSQPWorkLimitError as exc:
            return exc.reason_code
        except OverflowError:
            return "wheel_sqp_numeric_contract_failed"
        except (TypeError, ValueError):
            return "wheel_sqp_identity_mismatch"
        return None

    for segment_index, (segment, cell_range) in enumerate(
        zip(candidate.segments, broadphase_ranges, strict=True)
    ):
        stopped = tail_failure(
            checked_cell_count=broadphase_pair_count,
            checked_interval_count=checked_intervals,
            goal_position_error_m=position_error,
            goal_heading_error_rad=heading_error,
        )
        if stopped is not None:
            return stopped
        prospective_root_count = len(_OOB_BOUNDARIES)
        root_failure_reason = preadmit_terrain_roots(prospective_root_count)
        if root_failure_reason is not None:
            return _failed_l2_receipt_v2(
                root_failure_reason,
                candidate,
                request,
                profile,
                deadline,
                input_bytes_hash,
                checked_cell_count=broadphase_pair_count,
                checked_interval_count=checked_intervals,
                goal_position_error_m=position_error,
                goal_heading_error_rad=heading_error,
            )
        stopped = tail_failure(
            checked_cell_count=broadphase_pair_count,
            checked_interval_count=checked_intervals,
            goal_position_error_m=position_error,
            goal_heading_error_rad=heading_error,
        )
        if stopped is not None:
            return stopped
        # The full pair count is capped before the first terrain-mask read.
        invalid_cells: list[tuple[Cell, str]] = []
        if cell_range is not None:
            for cell in _iter_wheel_closed_cell_range_v2(cell_range):
                stopped = tail_failure(
                    checked_cell_count=broadphase_pair_count,
                    checked_interval_count=checked_intervals,
                    goal_position_error_m=position_error,
                    goal_heading_error_rad=heading_error,
                )
                if stopped is not None:
                    return stopped
                query = anchor.query(cell, profile.profile.max_traversable_slope_deg)
                stopped = tail_failure(
                    checked_cell_count=broadphase_pair_count,
                    checked_interval_count=checked_intervals,
                    goal_position_error_m=position_error,
                    goal_heading_error_rad=heading_error,
                )
                if stopped is not None:
                    return stopped
                if not query.passed:
                    prospective_root_count += 1
                    root_failure_reason = preadmit_terrain_roots(
                        prospective_root_count
                    )
                    if root_failure_reason is not None:
                        return _failed_l2_receipt_v2(
                            root_failure_reason,
                            candidate,
                            request,
                            profile,
                            deadline,
                            input_bytes_hash,
                            checked_cell_count=broadphase_pair_count,
                            checked_interval_count=checked_intervals,
                            goal_position_error_m=position_error,
                            goal_heading_error_rad=heading_error,
                        )
                    stopped = tail_failure(
                        checked_cell_count=broadphase_pair_count,
                        checked_interval_count=checked_intervals,
                        goal_position_error_m=position_error,
                        goal_heading_error_rad=heading_error,
                    )
                    if stopped is not None:
                        return stopped
                    invalid_cells.append((cell, query.reason_code))
        sweep = prove_wheel_segment_sweep_v2(
            segment,
            segment_index,
            candidate.candidate_hash,
            tuple(invalid_cells),
            anchor,
            profile,
            deadline,
            ledger,
            estimate,
            base_codec_delta_bytes=base_codec_delta_bytes,
            created_record_offset=created_records,
            interval_record_limit=actual_interval_record_bound,
            codec_reseal_tail_s=codec_reseal_tail_s,
        )
        checked_intervals += sweep.checked_interval_count
        created_records += sweep.created_record_count
        if not sweep.passed:
            assert sweep.reason_code is not None
            return _failed_l2_receipt_v2(
                sweep.reason_code,
                candidate,
                request,
                profile,
                deadline,
                input_bytes_hash,
                counterexample=sweep.counterexample,
                checked_cell_count=broadphase_pair_count,
                checked_interval_count=checked_intervals,
                goal_position_error_m=position_error,
                goal_heading_error_rad=heading_error,
            )
        stopped = tail_failure(
            checked_cell_count=broadphase_pair_count,
            checked_interval_count=checked_intervals,
            goal_position_error_m=position_error,
            goal_heading_error_rad=heading_error,
        )
        if stopped is not None:
            return stopped

    stopped = tail_failure(
        checked_cell_count=broadphase_pair_count,
        checked_interval_count=checked_intervals,
    )
    if stopped is not None:
        return stopped
    resealed_bytes = encode_wheel_candidate_v2(candidate)
    stopped = tail_failure(
        checked_cell_count=broadphase_pair_count,
        checked_interval_count=checked_intervals,
    )
    if stopped is not None:
        return stopped
    resealed_hash = sha256(candidate_bytes).hexdigest()
    stopped = tail_failure(
        checked_cell_count=broadphase_pair_count,
        checked_interval_count=checked_intervals,
    )
    if stopped is not None:
        return stopped
    identity_matches = _candidate_identity_matches_v2(
        candidate,
        request,
        profile,
        snapshot_hash(request.terrain_snapshot),
    )
    stopped = tail_failure(
        checked_cell_count=broadphase_pair_count,
        checked_interval_count=checked_intervals,
    )
    if stopped is not None:
        return stopped
    if (
        resealed_bytes != candidate_bytes
        or resealed_hash != input_bytes_hash
        or not identity_matches
    ):
        return _failed_l2_receipt_v2(
            "wheel_sqp_identity_mismatch",
            candidate,
            request,
            profile,
            deadline,
            input_bytes_hash,
            checked_cell_count=broadphase_pair_count,
            checked_interval_count=checked_intervals,
        )
    stopped = tail_failure(
        checked_cell_count=broadphase_pair_count,
        checked_interval_count=checked_intervals,
    )
    if stopped is not None:
        return stopped
    return _l2_receipt_v2(
        passed=True,
        reason_code=None,
        candidate=candidate,
        counterexample=None,
        request=request,
        profile=profile,
        deadline=deadline,
        input_bytes_hash=input_bytes_hash,
        checked_cell_count=broadphase_pair_count,
        checked_interval_count=checked_intervals,
        goal_position_error_m=position_error,
        goal_heading_error_rad=heading_error,
    )


def promote_wheel_candidate_to_route_v2(
    candidate: CanonicalWheelCandidateV1,
    receipt: WheelTrajectoryL2ResultV2,
    *,
    repair_applied: bool = False,
) -> WheelSQPPromotionV2:
    if type(candidate) is not CanonicalWheelCandidateV1:
        raise TypeError("candidate must be exact CanonicalWheelCandidateV1")
    if type(receipt) is not WheelTrajectoryL2ResultV2:
        raise TypeError("receipt must be exact WheelTrajectoryL2ResultV2")
    if type(repair_applied) is not bool:
        raise TypeError("repair_applied must be exact bool")
    if receipt._promotion_authority is not _WHEEL_SQP_L2_RESULT_AUTHORITY:
        raise TypeError("receipt lacks validator promotion authority")
    if receipt.passed is not True:
        raise ValueError("promotion requires a passed L2 receipt")
    if receipt.route is not None or receipt.evidence is not None:
        raise ValueError("promotion requires a pre-promotion receipt")

    _raise_if_deadline_expired(receipt._deadline)
    candidate_bytes = encode_wheel_candidate_v2(candidate)
    _raise_if_deadline_expired(receipt._deadline)
    candidate_bytes_hash = sha256(candidate_bytes).hexdigest()
    _raise_if_deadline_expired(receipt._deadline)
    if candidate_bytes_hash != receipt._input_bytes_hash:
        raise ValueError("receipt input bytes do not match candidate")
    expected_identities = {
        "candidate_hash": candidate.candidate_hash,
        "request_hash": candidate.request_hash,
        "profile_hash": candidate.profile_hash,
        "terrain_snapshot_hash": candidate.terrain_snapshot_hash,
        "capability_revision": candidate.capability_revision,
        "solver_contract_id": candidate.solver_contract_id,
        "canonicalization_id": candidate.canonicalization_id,
        "control_slew_id": candidate.control_slew_id,
        "observation_source_id": candidate.observation_source_id,
        "validator_contract_id": WHEEL_KINEMATIC_L2_VALIDATOR_V2,
    }
    for name, expected in expected_identities.items():
        if getattr(receipt, name) != expected:
            raise ValueError(f"receipt {name} does not match candidate")
    if receipt.actual_start != candidate.start_state:
        raise ValueError("receipt actual start does not match candidate")
    if receipt.actual_goal != candidate.actual_endpoint:
        raise ValueError("receipt actual goal does not match candidate")

    primitives: list[WheelKinematicSegmentV2] = []
    for segment in candidate.segments:
        _raise_if_deadline_expired(receipt._deadline)
        public_segment = WheelKinematicSegmentV2(
            kind=PrimitiveKindV2.WHEEL_MOTION,
            start_state=segment.start_state,
            end_state=segment.end_state,
            duration_s=segment.duration_s,
            distance_m=segment.distance_m,
            energy_cost=segment.relative_energy,
            observation_contribution=0.0,
            validation_level=ValidationLevelV2.L2,
            v_mps=segment.v_mps,
            omega_radps=segment.omega_radps,
            reverse=segment.v_mps < 0.0,
            turn_in_place=segment.v_mps == 0.0 and segment.omega_radps != 0.0,
            relative_energy=segment.relative_energy,
            samples=segment.samples,
            segment_hash=_ZERO_HASH,
            segment_schema_id=WHEEL_KINEMATIC_SEGMENT_SCHEMA_V2,
        )
        primitives.append(
            replace(
                public_segment,
                segment_hash=wheel_segment_hash_v2(public_segment),
            )
        )

    route = WheelKinematicRouteV2(
        platform_kind=PlatformKindV2.WHEEL,
        primitives=tuple(primitives),
        total_cost=candidate.total_cost,
        is_complete=True,
        route_hash=_ZERO_HASH,
        source_candidate_hash=candidate.candidate_hash,
        request_hash=candidate.request_hash,
        profile_hash=candidate.profile_hash,
        terrain_snapshot_hash=candidate.terrain_snapshot_hash,
        capability_revision=candidate.capability_revision,
        solver_contract_id=candidate.solver_contract_id,
        canonicalization_id=candidate.canonicalization_id,
        validator_contract_id=WHEEL_KINEMATIC_L2_VALIDATOR_V2,
    )
    route = replace(route, route_hash=wheel_route_hash_v2(route))
    _raise_if_deadline_expired(receipt._deadline)
    encode_wheel_route_v2(route)
    _raise_if_deadline_expired(receipt._deadline)
    projected = project_wheel_route_to_candidate_v1(
        route,
        receipt._request,
        receipt._profile,
        candidate.terrain_snapshot_hash,
    )
    _raise_if_deadline_expired(receipt._deadline)
    if projected.candidate_hash != candidate.candidate_hash:
        raise ValueError("public route projection changed candidate hash")

    evidence = WheelSQPValidationEvidenceV2(
        validator_id=WHEEL_KINEMATIC_L2_VALIDATOR_V2,
        level=ValidationLevelV2.L2,
        passed=True,
        checks=(
            "strict_candidate_replay",
            "continuous_rectangle_sweep",
            "goal_tolerance",
            "strict_route_reseal",
        ),
        route_hash=route.route_hash,
        candidate_hash=candidate.candidate_hash,
        request_hash=candidate.request_hash,
        profile_hash=candidate.profile_hash,
        terrain_snapshot_hash=candidate.terrain_snapshot_hash,
        solver_contract_id=candidate.solver_contract_id,
        checked_interval_count=receipt.checked_interval_count,
        checked_cell_count=receipt.checked_cell_count,
        repair_applied=repair_applied,
    )
    promotion = WheelSQPPromotionV2(route=route, evidence=evidence)
    if not (
        candidate.candidate_hash
        == receipt.candidate_hash
        == promotion.route.source_candidate_hash
        == promotion.evidence.candidate_hash
    ):
        raise ValueError("promotion candidate hashes are not identical")
    _raise_if_deadline_expired(receipt._deadline)
    return promotion
