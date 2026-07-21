from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from path_planner.v2.contracts import (
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    RoutePrimitiveV2,
    TypedRouteV2,
    ValidationLevelV2,
)
from path_planner.v2.observation import (
    OBSERVATION_PROJECTION_SOURCE_V2,
    OBSERVATION_ROUTE_SAMPLE_STEP_M_V2,
    OBSERVED_TERRAIN_SOURCE_KIND_V2,
    ObservedTerrainInputV2,
    project_route_observation_v2,
)
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
    snapshot_hash,
)


@dataclass(frozen=True, slots=True)
class _SampledPrimitive(RoutePrimitiveV2):
    samples: tuple[PoseStateV2, ...]

    def __post_init__(self) -> None:
        RoutePrimitiveV2.__post_init__(self)
        if self.samples[0] != self.start_state or self.samples[-1] != self.end_state:
            raise ValueError("sample endpoints must match the primitive")


@dataclass(frozen=True, slots=True)
class _LeggedPrimitive(RoutePrimitiveV2):
    lift_body_state: PoseStateV2


def _primitive(
    kind: PrimitiveKindV2,
    start: PoseStateV2,
    end: PoseStateV2,
) -> RoutePrimitiveV2:
    return RoutePrimitiveV2(
        kind=kind,
        start_state=start,
        end_state=end,
        duration_s=1.0,
        distance_m=float(np.hypot(end.x_m - start.x_m, end.y_m - start.y_m)),
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
    )


def _sampled_primitive(samples: tuple[PoseStateV2, ...]) -> _SampledPrimitive:
    return _SampledPrimitive(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=samples[0],
        end_state=samples[-1],
        duration_s=1.0,
        distance_m=sum(
            float(np.hypot(right.x_m - left.x_m, right.y_m - left.y_m))
            for left, right in zip(samples[:-1], samples[1:], strict=True)
        ),
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        samples=samples,
    )


def _snapshot(
    *,
    width: int = 9,
    height: int = 3,
    observed_mask: np.ndarray | None = None,
    hidden_value: float = 7.0,
    source_kind: str = "truth-kind-a/v1",
    source_id: str = "truth-source-a",
    source_hash: str = "truth-hash-a",
    detail: str = "truth-detail-a",
    physical_obstacle_cells_written: bool = False,
) -> TerrainSnapshotV2:
    geometry = FineGridGeometryV2(width=width, height=height)
    observed = (
        np.ones(geometry.shape, dtype=bool)
        if observed_mask is None
        else np.array(observed_mask, dtype=bool, copy=True)
    )
    elevation = np.zeros(geometry.shape, dtype=np.float64)
    slope = np.zeros(geometry.shape, dtype=np.float64)
    traversable = np.array(observed, dtype=bool, copy=True)
    obstacle = np.zeros(geometry.shape, dtype=bool)
    confidence = np.where(observed, 1.0, hidden_value / 10.0).astype(np.float64)
    elevation[~observed] = hidden_value
    slope[~observed] = hidden_value
    obstacle[~observed] = hidden_value > 8.0
    traversable[~observed] = False
    return TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=elevation,
        slope_deg=slope,
        traversable_mask=traversable,
        hard_obstacle_mask=obstacle,
        observed_mask=observed,
        confidence=confidence,
        provenance=TerrainProvenanceV2(
            source_kind=source_kind,
            source_id=source_id,
            source_hash=source_hash,
            physical_obstacle_cells_written=physical_obstacle_cells_written,
            details=(("truth_detail", detail),),
        ),
    )


def _observed(snapshot: TerrainSnapshotV2, start: PoseStateV2) -> ObservedTerrainInputV2:
    return ObservedTerrainInputV2(
        request_id="ppo-request-001",
        start_state=start,
        terrain_snapshot=snapshot,
    )


def test_observed_input_masks_unknown_values_and_rebuilds_truth_independent_provenance() -> None:
    observed_mask = np.zeros((1, 5), dtype=bool)
    observed_mask[0, :2] = True
    start = PoseStateV2(0.25, 0.25, 0.0)

    left = _observed(
        _snapshot(
            width=5,
            height=1,
            observed_mask=observed_mask,
            hidden_value=7.0,
            source_kind="truth-kind-a/v1",
            source_id="truth-source-a",
            source_hash="truth-hash-a",
            detail="truth-detail-a",
            physical_obstacle_cells_written=False,
        ),
        start,
    )
    right = _observed(
        _snapshot(
            width=5,
            height=1,
            observed_mask=observed_mask,
            hidden_value=9.0,
            source_kind="truth-kind-b/v9",
            source_id="truth-source-b",
            source_hash="truth-hash-b",
            detail="truth-detail-b",
            physical_obstacle_cells_written=True,
        ),
        start,
    )

    assert snapshot_hash(left.terrain_snapshot) == snapshot_hash(right.terrain_snapshot)
    assert left.terrain_snapshot.provenance == right.terrain_snapshot.provenance
    assert left.terrain_snapshot.provenance.source_kind == OBSERVED_TERRAIN_SOURCE_KIND_V2
    assert left.terrain_snapshot.provenance.physical_obstacle_cells_written is False
    unknown = ~left.terrain_snapshot.observed_mask
    assert np.all(left.terrain_snapshot.elevation_m[unknown] == 0.0)
    assert np.all(left.terrain_snapshot.slope_deg[unknown] == 0.0)
    assert np.all(left.terrain_snapshot.confidence[unknown] == 0.0)
    assert not np.any(left.terrain_snapshot.traversable_mask[unknown])
    assert not np.any(left.terrain_snapshot.hard_obstacle_mask[unknown])


def test_wheel_projection_uses_fixed_xy_arc_length_tangents_and_exact_endpoint_theta() -> None:
    start = PoseStateV2(0.25, 0.75, 0.3)
    middle = PoseStateV2(1.25, 0.75, 0.4)
    end = PoseStateV2(2.25, 0.75, 0.5)
    route = TypedRouteV2(
        platform_kind=PlatformKindV2.WHEEL,
        primitives=(
            _sampled_primitive((start, PoseStateV2(0.75, 0.75, 0.35), middle)),
            _sampled_primitive((middle, PoseStateV2(1.75, 0.75, 0.45), end)),
        ),
        total_cost=2.0,
    )
    endpoint_theta = 3.0 * pi

    projection = project_route_observation_v2(
        route,
        _observed(_snapshot(), start),
        endpoint_theta_rad=endpoint_theta,
    )

    assert OBSERVATION_ROUTE_SAMPLE_STEP_M_V2 == 1.0
    assert projection.source == OBSERVATION_PROJECTION_SOURCE_V2
    assert projection.sample_states == (
        PoseStateV2(0.25, 0.75, 0.3),
        PoseStateV2(1.25, 0.75, 0.4),
        PoseStateV2(2.25, 0.75, endpoint_theta),
    )


def test_legged_projection_resamples_start_lift_end_and_uses_outgoing_tangent_at_seam() -> None:
    start = PoseStateV2(0.25, 0.25, 0.1)
    lift = PoseStateV2(1.25, 0.25, 0.2)
    end = PoseStateV2(1.25, 1.25, 0.3)
    primitive = _LeggedPrimitive(
        kind=PrimitiveKindV2.LEG_STEP,
        start_state=start,
        end_state=end,
        duration_s=1.0,
        distance_m=2.0,
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        lift_body_state=lift,
    )
    route = TypedRouteV2(PlatformKindV2.LEGGED, (primitive,), 1.0)

    projection = project_route_observation_v2(
        route,
        _observed(_snapshot(), start),
        endpoint_theta_rad=-0.75,
    )

    assert projection.sample_states == (
        PoseStateV2(0.25, 0.25, 0.1),
        PoseStateV2(1.25, 0.25, 0.2),
        PoseStateV2(1.25, 1.25, -0.75),
    )


def test_wheel_reverse_motion_preserves_route_heading_instead_of_xy_tangent() -> None:
    start = PoseStateV2(2.25, 0.75, 0.0)
    middle = PoseStateV2(1.25, 0.75, 0.0)
    end = PoseStateV2(0.25, 0.75, 0.0)
    route = TypedRouteV2(
        PlatformKindV2.WHEEL,
        (_sampled_primitive((start, middle, end)),),
        2.0,
    )

    projection = project_route_observation_v2(
        route,
        _observed(_snapshot(), start),
        endpoint_theta_rad=0.0,
    )

    assert tuple(state.heading_rad for state in projection.sample_states) == (0.0, 0.0, 0.0)


def test_zero_xy_turn_preserves_route_start_then_exact_target_theta() -> None:
    start = PoseStateV2(1.25, 0.75, -0.5)
    turned = PoseStateV2(1.25, 0.75, 1.0)
    route = TypedRouteV2(
        PlatformKindV2.WHEEL,
        (_sampled_primitive((start, turned)),),
        1.0,
    )

    projection = project_route_observation_v2(
        route,
        _observed(_snapshot(), start),
        endpoint_theta_rad=1.25,
    )

    assert projection.sample_states == (
        start,
        PoseStateV2(turned.x_m, turned.y_m, 1.25),
    )


def test_route_heading_interpolation_takes_the_wrap_safe_short_arc() -> None:
    start = PoseStateV2(0.25, 0.75, pi - 0.1)
    end = PoseStateV2(2.25, 0.75, -pi + 0.1)
    route = TypedRouteV2(
        PlatformKindV2.WHEEL,
        (_sampled_primitive((start, end)),),
        2.0,
    )

    projection = project_route_observation_v2(
        route,
        _observed(_snapshot(), start),
        endpoint_theta_rad=end.heading_rad,
    )

    assert projection.sample_states[0].heading_rad == start.heading_rad
    assert projection.sample_states[1].heading_rad == pi
    assert projection.sample_states[-1].heading_rad == end.heading_rad


def test_hopper_projection_contains_launch_and_each_landing_only() -> None:
    launch = PoseStateV2(0.25, 0.25, 0.1)
    landing_one = PoseStateV2(2.25, 0.25, 0.1)
    landing_two = PoseStateV2(4.25, 0.25, 0.1)
    route = TypedRouteV2(
        PlatformKindV2.HOPPER,
        (
            _primitive(PrimitiveKindV2.BALLISTIC_JUMP, launch, landing_one),
            _primitive(PrimitiveKindV2.BALLISTIC_JUMP, landing_one, landing_two),
        ),
        2.0,
    )

    projection = project_route_observation_v2(
        route,
        _observed(_snapshot(), launch),
        endpoint_theta_rad=-1.25,
    )

    assert projection.sample_states == (
        launch,
        landing_one,
        PoseStateV2(landing_two.x_m, landing_two.y_m, -1.25),
    )


def test_los_counts_the_first_unknown_then_stops_without_reading_hidden_truth() -> None:
    observed_mask = np.zeros((1, 9), dtype=bool)
    observed_mask[0, :2] = True
    start = PoseStateV2(0.25, 0.25, 0.0)
    route = TypedRouteV2(
        PlatformKindV2.WHEEL,
        (_sampled_primitive((start,)),),
        0.0,
    )
    left = _observed(
        _snapshot(
            width=9,
            height=1,
            observed_mask=observed_mask,
            hidden_value=7.0,
            source_hash="truth-a",
            detail="a",
        ),
        start,
    )
    right = _observed(
        _snapshot(
            width=9,
            height=1,
            observed_mask=observed_mask,
            hidden_value=9.0,
            source_hash="truth-b",
            detail="b",
        ),
        start,
    )

    left_projection = project_route_observation_v2(
        route,
        left,
        endpoint_theta_rad=0.0,
    )
    right_projection = project_route_observation_v2(
        route,
        right,
        endpoint_theta_rad=0.0,
    )

    assert left_projection == right_projection
    assert left_projection.expected_new_observed_cells == 1.0
    assert left_projection.expected_information_gain == 0.0
