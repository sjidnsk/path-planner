from __future__ import annotations

from dataclasses import replace
import json
from math import atan2, inf, nextafter, pi
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import path_planner.v2.wheel_sqp_initialization as wheel_initialization
from path_planner.core import Cell
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    ResourceBudgetV2,
    RoutePrimitiveV2,
)
from path_planner.v2.profiles import PlatformProfileV2, WheelKinematicSQPProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)
from path_planner.v2.wheel_corridors import (
    WheelCorridorGraphV1,
    build_blocked_components_v1,
    topology_signature_v1,
    wheel_corridor_path_hash_v1,
)
from path_planner.v2.wheel_kinematics import integrate_wheel_segment_v2
from path_planner.v2.wheel_sqp_contracts import (
    WheelCorridorV2,
    WheelKinematicSegmentV2,
    WheelSQPInitialGuessV2,
    WheelSQPModeV2,
    WheelSQPWorkLedgerV1,
    WheelTopologySignatureV1,
)
from path_planner.v2.wheel_sqp_initialization import (
    WheelSQPInitializationError,
    initialize_wheel_trajectory_v2,
    select_wheel_modes_v2,
    simplify_wheel_corridor_v2,
)


PROFILE = WheelKinematicSQPProfileV2(
    profile=PlatformProfileV2(
        profile_id="scout-mini-wheel-kinematic-sqp/v1",
        platform_kind=PlatformKindV2.WHEEL,
        capability_revision="wheel_kinematic_corridor_sqp/v1",
        simulation_proxy=False,
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=0.25,
        goal_heading_tolerance_rad=0.08726646259971647,
    )
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _snapshot(
    width: int,
    height: int,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    unknown: set[Cell] = frozenset(),
    hard: set[Cell] = frozenset(),
    not_traversable: set[Cell] = frozenset(),
    slope: dict[Cell, float] | None = None,
    confidence: dict[Cell, float] | None = None,
) -> TerrainSnapshotV2:
    geometry = FineGridGeometryV2(width, height, origin=origin)
    observed = np.ones(geometry.shape, dtype=np.bool_)
    hard_mask = np.zeros(geometry.shape, dtype=np.bool_)
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    slope_deg = np.zeros(geometry.shape, dtype=np.float64)
    confidence_layer = np.ones(geometry.shape, dtype=np.float64)
    for cell in unknown:
        observed[cell.y, cell.x] = False
    for cell in hard:
        hard_mask[cell.y, cell.x] = True
        traversable[cell.y, cell.x] = False
    for cell in not_traversable:
        traversable[cell.y, cell.x] = False
    for cell, value in (slope or {}).items():
        slope_deg[cell.y, cell.x] = value
    for cell, value in (confidence or {}).items():
        confidence_layer[cell.y, cell.x] = value
    return TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape, dtype=np.float64),
        slope_deg=slope_deg,
        traversable_mask=traversable,
        hard_obstacle_mask=hard_mask,
        observed_mask=observed,
        confidence=confidence_layer,
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task4-fixture",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )


def _budget(
    *,
    max_expanded_states: int = 100_000,
    max_route_states: int = 100_000,
    max_memory_bytes: int = 0,
) -> ResourceBudgetV2:
    return ResourceBudgetV2(
        max_expanded_states=max_expanded_states,
        max_route_states=max_route_states,
        max_memory_bytes=max_memory_bytes,
    )


def _request(
    snapshot: TerrainSnapshotV2,
    start: PoseStateV2,
    goal: PoseStateV2,
    *,
    budget: ResourceBudgetV2,
    objective: ObjectiveProfileV2 | None = None,
    seed: int = 19,
) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="wheel-sqp-task4",
        platform_profile_id=PROFILE.profile.profile_id,
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
        objective_profile=objective or ObjectiveProfileV2(),
        resource_budget=budget,
        timeout_s=2.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=seed,
    )


def _corridor_from_cells(
    snapshot: TerrainSnapshotV2,
    cells: tuple[Cell, ...],
    ledger: WheelSQPWorkLedgerV1,
) -> WheelCorridorV2:
    graph = WheelCorridorGraphV1.from_snapshot(snapshot, 30.0, ledger=ledger)
    components = build_blocked_components_v1(graph, ledger=ledger)
    signature = topology_signature_v1(cells, components, ledger=ledger)
    digest = wheel_corridor_path_hash_v1(
        graph.snapshot_identity,
        cells,
        ledger=ledger,
    )
    guide_cost = sum(
        graph.edge_cost(left, right)
        for left, right in zip(cells, cells[1:])
    )
    path_length = sum(
        graph.edge_length_m(left, right)
        for left, right in zip(cells, cells[1:])
    )
    ledger.charge_route_states(len(cells))
    return WheelCorridorV2(
        corridor_index=0,
        cells=cells,
        corridor_hash=digest,
        guide_cost=float(guide_cost),
        path_length_m=float(path_length),
        topology_signature=signature,
    )


def _case(
    snapshot: TerrainSnapshotV2,
    cells: tuple[Cell, ...],
    start: PoseStateV2,
    goal: PoseStateV2,
    *,
    budget: ResourceBudgetV2 | None = None,
    request_budget: ResourceBudgetV2 | None = None,
    objective: ObjectiveProfileV2 | None = None,
    seed: int = 19,
) -> tuple[
    WheelCorridorV2,
    PlanningRequestV2,
    WheelSQPWorkLedgerV1,
    PlanningDeadlineV2,
    _Clock,
]:
    ledger_budget = budget or _budget()
    clock = _Clock()
    deadline = PlanningDeadlineV2(0.0, 1.0, clock)
    ledger = WheelSQPWorkLedgerV1(ledger_budget, deadline)
    corridor = _corridor_from_cells(snapshot, cells, ledger)
    request = _request(
        snapshot,
        start,
        goal,
        budget=request_budget or ledger_budget,
        objective=objective,
        seed=seed,
    )
    return corridor, request, ledger, deadline, clock


def _leave_memory_bytes(ledger: WheelSQPWorkLedgerV1, remaining: int) -> None:
    amount = ledger.effective_memory_limit_bytes - ledger.accounted_bytes - remaining
    assert amount >= 0
    ledger.charge_memory(amount)


def _line_cells(last_x: int, *, y: int = 0) -> tuple[Cell, ...]:
    return tuple(Cell(x, y) for x in range(last_x + 1))


def _assert_connected(guess: WheelSQPInitialGuessV2) -> None:
    assert guess.segments[0].start_state == guess.start_state
    assert guess.segments[-1].end_state == guess.actual_endpoint
    for left, right in zip(guess.segments, guess.segments[1:]):
        assert left.end_state == right.start_state
    current = guess.start_state
    for segment in guess.segments:
        assert segment.start_state == current
        current = integrate_wheel_segment_v2(
            current,
            segment.v_mps,
            segment.omega_radps,
            segment.duration_s,
        )
        assert segment.end_state == current
    assert current == guess.actual_endpoint


def test_initializer_uses_real_endpoints_and_internal_connected_segments() -> None:
    snapshot = _snapshot(12, 8)
    cells = (
        Cell(1, 1),
        Cell(2, 1),
        Cell(3, 1),
        Cell(4, 2),
        Cell(5, 3),
        Cell(6, 4),
        Cell(7, 5),
        Cell(8, 5),
        Cell(9, 5),
        Cell(10, 6),
    )
    start = PoseStateV2(0.61, 0.74, 0.2)
    goal = PoseStateV2(5.38, 3.11, -0.4)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        cells,
        start,
        goal,
    )

    points = simplify_wheel_corridor_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )
    guess = initialize_wheel_trajectory_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert points[0] == (start.x_m, start.y_m)
    assert points[-1] == (goal.x_m, goal.y_m)
    start_center = snapshot.geometry.cell_center(cells[0])
    goal_center = snapshot.geometry.cell_center(cells[-1])
    assert points[0] != (start_center.x, start_center.y)
    assert points[-1] != (goal_center.x, goal_center.y)
    assert type(guess) is WheelSQPInitialGuessV2
    assert guess.start_state is request.start_state
    assert guess.requested_goal is request.goal_state
    assert guess.segments[0].start_state is request.start_state
    assert len(guess.initial_guess_hash) == 64
    assert all(not isinstance(segment, RoutePrimitiveV2) for segment in guess.segments)
    assert all(type(segment) is not WheelKinematicSegmentV2 for segment in guess.segments)
    assert all(not hasattr(segment, "validation_level") for segment in guess.segments)
    _assert_connected(guess)


def test_profile_capabilities_are_sealed_forward_reverse_and_turn() -> None:
    assert PROFILE.reverse_enabled is True
    assert PROFILE.turn_in_place_enabled is True


def test_same_pose_and_heading_produces_one_exact_positive_stop() -> None:
    snapshot = _snapshot(1, 1)
    pose = PoseStateV2(0.21, 0.32, 0.7)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        (Cell(0, 0),),
        pose,
        pose,
    )

    guess = initialize_wheel_trajectory_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert len(guess.segments) == 1
    stop = guess.segments[0]
    assert stop.mode is WheelSQPModeV2.STOP
    assert (stop.v_mps, stop.omega_radps, stop.duration_s) == (0.0, 0.0, 0.05)
    assert stop.start_state is request.start_state
    assert stop.end_state == request.start_state


def test_wrap_equivalent_heading_is_exact_hold_not_a_hidden_turn() -> None:
    snapshot = _snapshot(1, 1)
    start = PoseStateV2(0.25, 0.25, 0.0)
    goal = PoseStateV2(0.25, 0.25, 2.0 * pi)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        (Cell(0, 0),),
        start,
        goal,
    )

    guess = initialize_wheel_trajectory_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert len(guess.segments) == 1
    assert guess.segments[0].mode is WheelSQPModeV2.STOP
    assert guess.actual_endpoint == request.start_state
    assert guess.requested_goal is request.goal_state


@pytest.mark.parametrize(
    ("goal_heading", "expected_count"),
    [
        (pi / 2.0, 1),
        (nextafter(pi / 2.0, inf), 2),
        (pi, 2),
    ],
)
def test_turns_split_at_exact_quarter_turn_and_positive_pi_ties_left(
    goal_heading: float,
    expected_count: int,
) -> None:
    snapshot = _snapshot(1, 1)
    start = PoseStateV2(0.25, 0.25, 0.0)
    goal = PoseStateV2(0.25, 0.25, goal_heading)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        (Cell(0, 0),),
        start,
        goal,
    )

    guess = initialize_wheel_trajectory_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert len(guess.segments) == expected_count
    assert all(
        abs(segment.omega_radps) * segment.duration_s
        <= PROFILE.max_segment_heading_change_rad
        for segment in guess.segments
    )
    assert all(segment.mode is WheelSQPModeV2.TURN_LEFT for segment in guess.segments)
    _assert_connected(guess)


def test_reverse_is_selected_for_rear_goal_without_direct_sign_change() -> None:
    snapshot = _snapshot(5, 1)
    cells = tuple(reversed(_line_cells(4)))
    start = PoseStateV2(2.25, 0.25, 0.0)
    goal = PoseStateV2(0.25, 0.25, 0.0)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        cells,
        start,
        goal,
    )

    guess = initialize_wheel_trajectory_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert WheelSQPModeV2.REVERSE in tuple(segment.mode for segment in guess.segments)
    for left, right in zip(guess.segments, guess.segments[1:]):
        assert left.v_mps * right.v_mps >= 0.0


def test_mode_dp_uses_global_exact_tie_key_with_forward_rank_first() -> None:
    snapshot = _snapshot(5, 1)
    start = PoseStateV2(0.25, 0.25, pi / 2.0)
    goal = PoseStateV2(2.25, 0.25, pi / 2.0)
    _, request, ledger, deadline, _ = _case(
        snapshot,
        _line_cells(4),
        start,
        goal,
        objective=ObjectiveProfileV2(
            distance_weight=0.0,
            risk_weight=0.0,
            energy_weight=0.0,
            time_weight=1.0,
        ),
    )

    modes = select_wheel_modes_v2(
        ((0.25, 0.25), (1.25, 0.25), (2.25, 0.25)),
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert modes == (WheelSQPModeV2.FORWARD, WheelSQPModeV2.FORWARD)


def test_mode_dp_includes_terminal_turn_before_applying_forward_tie_rank() -> None:
    snapshot = _snapshot(5, 1)
    start = PoseStateV2(2.25, 0.25, 0.0)
    goal = PoseStateV2(0.25, 0.25, pi)
    _, request, ledger, deadline, _ = _case(
        snapshot,
        tuple(reversed(_line_cells(4))),
        start,
        goal,
        objective=ObjectiveProfileV2(
            distance_weight=0.0,
            risk_weight=0.0,
            energy_weight=0.0,
            time_weight=1.0,
        ),
    )

    modes = select_wheel_modes_v2(
        ((2.25, 0.25), (0.25, 0.25)),
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert modes == (WheelSQPModeV2.FORWARD,)


def test_direction_switch_inserts_one_exact_stop_between_translation_signs() -> None:
    snapshot = _snapshot(4, 3, not_traversable={Cell(1, 1), Cell(2, 1)})
    cells = (
        Cell(0, 2),
        Cell(1, 2),
        Cell(2, 2),
        Cell(3, 2),
        Cell(3, 1),
        Cell(3, 0),
        Cell(2, 0),
        Cell(1, 0),
        Cell(0, 0),
    )
    start = PoseStateV2(0.25, 1.25, 0.0)
    goal = PoseStateV2(0.25, 0.25, 0.0)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        cells,
        start,
        goal,
        objective=ObjectiveProfileV2(
            distance_weight=0.0,
            risk_weight=0.0,
            energy_weight=0.5,
            time_weight=0.5,
        ),
    )

    points = simplify_wheel_corridor_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )
    assert points == (
        (0.25, 1.25),
        (1.75, 1.25),
        (1.75, 0.25),
        (0.25, 0.25),
    )

    guess = initialize_wheel_trajectory_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    translation_indices = [
        index for index, segment in enumerate(guess.segments) if segment.v_mps != 0.0
    ]
    switches = [
        (left, right)
        for left, right in zip(translation_indices, translation_indices[1:])
        if guess.segments[left].v_mps * guess.segments[right].v_mps < 0.0
    ]
    assert switches
    for left, right in switches:
        stops = [
            segment
            for segment in guess.segments[left + 1 : right]
            if segment.mode is WheelSQPModeV2.STOP
        ]
        assert len(stops) == 1
        assert (stops[0].v_mps, stops[0].omega_radps, stops[0].duration_s) == (
            0.0,
            0.0,
            0.05,
        )


def test_slew_repair_reaches_fixed_point_without_changing_leg_geometry() -> None:
    snapshot = _snapshot(1, 1)
    angle = 0.04
    distance = 0.025
    start = PoseStateV2(0.25, 0.25, 0.0)
    goal_x = start.x_m + distance * np.cos(angle)
    goal_y = start.y_m + distance * np.sin(angle)
    goal = PoseStateV2(
        goal_x,
        goal_y,
        atan2(goal_y - start.y_m, goal_x - start.x_m),
    )
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        (Cell(0, 0),),
        start,
        goal,
    )

    guess = initialize_wheel_trajectory_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert any(segment.duration_s > 0.05 for segment in guess.segments)
    for left, right in zip(guess.segments, guess.segments[1:]):
        tau = 0.5 * (left.duration_s + right.duration_s)
        speed_up = max(0.0, abs(right.v_mps) - abs(left.v_mps)) / tau
        slow_down = max(0.0, abs(left.v_mps) - abs(right.v_mps)) / tau
        angular_slew = abs(right.omega_radps - left.omega_radps) / tau
        assert speed_up <= PROFILE.max_linear_accel_mps2
        assert slow_down <= PROFILE.max_linear_decel_mps2
        assert angular_slew <= PROFILE.max_angular_accel_radps2
    assert guess.actual_endpoint.x_m == pytest.approx(goal.x_m, abs=1.0e-14)
    assert guess.actual_endpoint.y_m == pytest.approx(goal.y_m, abs=1.0e-14)
    assert guess.actual_endpoint.heading_rad == pytest.approx(goal.heading_rad, abs=1.0e-14)


def test_simplification_rejects_shortcut_through_invalid_center_cell() -> None:
    snapshot = _snapshot(3, 3, hard={Cell(1, 1)})
    cells = (
        Cell(0, 1),
        Cell(0, 0),
        Cell(1, 0),
        Cell(2, 0),
        Cell(2, 1),
    )
    start = PoseStateV2(0.25, 0.75, 0.0)
    goal = PoseStateV2(1.25, 0.75, 0.0)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        cells,
        start,
        goal,
    )

    points = simplify_wheel_corridor_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert len(points) > 2
    assert points[0] == (0.25, 0.75)
    assert points[-1] == (1.25, 0.75)


def test_supercover_includes_cell_touched_only_at_exact_grid_vertex() -> None:
    snapshot = _snapshot(3, 3, hard={Cell(1, 0)})
    cells = (
        Cell(0, 0),
        Cell(0, 1),
        Cell(1, 1),
        Cell(2, 1),
        Cell(2, 2),
    )
    start = PoseStateV2(0.25, 0.25, 0.0)
    goal = PoseStateV2(1.25, 1.25, pi / 4.0)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        cells,
        start,
        goal,
    )

    points = simplify_wheel_corridor_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert points != ((start.x_m, start.y_m), (goal.x_m, goal.y_m))
    assert points[0] == (start.x_m, start.y_m)
    assert points[-1] == (goal.x_m, goal.y_m)


def test_exact_outer_boundary_touch_fails_closed_for_single_cell_corridor() -> None:
    snapshot = _snapshot(1, 1)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        (Cell(0, 0),),
        PoseStateV2(0.0, 0.25, 0.0),
        PoseStateV2(0.25, 0.25, 0.0),
    )

    with pytest.raises(WheelSQPInitializationError) as caught:
        simplify_wheel_corridor_v2(corridor, request, PROFILE, ledger, deadline)

    assert caught.value.reason_code == "wheel_sqp_initialization_failed"


def test_nextafter_inside_outer_boundary_remains_initializable() -> None:
    snapshot = _snapshot(1, 1)
    start_x = nextafter(0.0, inf)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        (Cell(0, 0),),
        PoseStateV2(start_x, 0.25, 0.0),
        PoseStateV2(0.25, 0.25, 0.0),
    )

    points = simplify_wheel_corridor_v2(corridor, request, PROFILE, ledger, deadline)

    assert points == ((start_x, 0.25), (0.25, 0.25))


def test_nominal_long_leg_split_is_exactly_sixty_meters() -> None:
    snapshot = _snapshot(242, 1)
    cells = _line_cells(241)
    start = PoseStateV2(0.25, 0.25, 0.0)
    goal = PoseStateV2(120.75, 0.25, 0.0)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        cells,
        start,
        goal,
    )

    points = simplify_wheel_corridor_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert points == (
        (0.25, 0.25),
        (0.25 + 120.5 / 3.0, 0.25),
        (0.25 + 2.0 * 120.5 / 3.0, 0.25),
        (120.75, 0.25),
    )


@pytest.mark.parametrize(
    ("translation_m", "heading_rad"),
    [
        (nextafter(PROFILE.min_nonzero_control * PROFILE.min_segment_duration_s, 0.0), 0.0),
        (0.0, nextafter(PROFILE.min_nonzero_control * PROFILE.min_segment_duration_s, 0.0)),
    ],
)
def test_tiny_nonzero_motion_is_not_silently_erased(
    translation_m: float,
    heading_rad: float,
) -> None:
    snapshot = _snapshot(1, 1)
    start = PoseStateV2(0.25, 0.25, 0.0)
    goal = PoseStateV2(0.25 + translation_m, 0.25, heading_rad)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        (Cell(0, 0),),
        start,
        goal,
    )

    with pytest.raises(WheelSQPInitializationError) as caught:
        initialize_wheel_trajectory_v2(
            corridor,
            request,
            PROFILE,
            ledger,
            deadline,
        )

    assert caught.value.reason_code == "wheel_sqp_initialization_failed"


def test_exact_segment_cap_accepts_48_and_rejects_49() -> None:
    accepted_snapshot = _snapshot(5761, 1)
    accepted_cells = _line_cells(5760)
    accepted = _case(
        accepted_snapshot,
        accepted_cells,
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2880.25, 0.25, 0.0),
    )
    accepted_guess = initialize_wheel_trajectory_v2(
        accepted[0],
        accepted[1],
        PROFILE,
        accepted[2],
        accepted[3],
    )
    assert len(accepted_guess.segments) == 48

    rejected_snapshot = _snapshot(5762, 1)
    rejected_cells = _line_cells(5761)
    rejected = _case(
        rejected_snapshot,
        rejected_cells,
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2880.75, 0.25, 0.0),
    )
    with pytest.raises(WheelSQPInitializationError) as caught:
        initialize_wheel_trajectory_v2(
            rejected[0],
            rejected[1],
            PROFILE,
            rejected[2],
            rejected[3],
        )
    assert caught.value.reason_code == "wheel_sqp_initialization_failed"


def test_forged_corridor_hash_and_topology_fail_identity_without_partial_output() -> None:
    snapshot = _snapshot(5, 1)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )

    for forged in (
        replace(corridor, corridor_hash="f" * 64),
        replace(corridor, topology_signature=WheelTopologySignatureV1(())),
    ):
        if forged == corridor:
            continue
        with pytest.raises(WheelSQPInitializationError) as caught:
            simplify_wheel_corridor_v2(
                forged,
                request,
                PROFILE,
                ledger,
                deadline,
            )
        assert caught.value.reason_code == "wheel_sqp_identity_mismatch"


def test_budget_copy_is_accepted_but_value_drift_and_foreign_deadline_are_rejected() -> None:
    snapshot = _snapshot(5, 1)
    original = _budget()
    copied = ResourceBudgetV2(
        original.max_expanded_states,
        original.max_route_states,
        original.max_memory_bytes,
    )
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
        budget=original,
        request_budget=copied,
    )
    assert simplify_wheel_corridor_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    drifted_request = replace(
        request,
        resource_budget=replace(copied, max_route_states=copied.max_route_states - 1),
    )
    with pytest.raises(WheelSQPInitializationError) as caught:
        simplify_wheel_corridor_v2(
            corridor,
            drifted_request,
            PROFILE,
            ledger,
            deadline,
        )
    assert caught.value.reason_code == "wheel_sqp_identity_mismatch"

    foreign_deadline = PlanningDeadlineV2(0.0, 1.0, _Clock())
    with pytest.raises(WheelSQPInitializationError) as caught:
        simplify_wheel_corridor_v2(
            corridor,
            request,
            PROFILE,
            ledger,
            foreign_deadline,
        )
    assert caught.value.reason_code == "wheel_sqp_identity_mismatch"


def test_risk_objective_is_rejected_with_stable_typed_reason() -> None:
    snapshot = _snapshot(5, 1)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
        objective=ObjectiveProfileV2(
            distance_weight=0.0,
            risk_weight=1.0,
            energy_weight=0.0,
            time_weight=0.0,
        ),
    )

    with pytest.raises(WheelSQPInitializationError) as caught:
        initialize_wheel_trajectory_v2(
            corridor,
            request,
            PROFILE,
            ledger,
            deadline,
        )

    assert caught.value.reason_code == "wheel_sqp_objective_unsupported"


def test_deadline_precedes_resource_and_ordinary_initialization_failure() -> None:
    snapshot = _snapshot(5, 1)
    corridor, request, ledger, deadline, clock = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    clock.now = 1.0

    with pytest.raises(WheelSQPInitializationError) as caught:
        initialize_wheel_trajectory_v2(
            replace(corridor, corridor_hash="f" * 64),
            request,
            PROFILE,
            ledger,
            deadline,
        )

    assert caught.value.reason_code == "planning_deadline_expired"


def test_shared_resource_limit_is_typed_and_timeout_keeps_precedence() -> None:
    snapshot = _snapshot(5, 1)
    tight_budget = _budget(max_memory_bytes=400)
    corridor, request, ledger, deadline, clock = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
        budget=tight_budget,
    )

    with pytest.raises(WheelSQPInitializationError) as caught:
        initialize_wheel_trajectory_v2(
            corridor,
            request,
            PROFILE,
            ledger,
            deadline,
        )
    assert caught.value.reason_code == "wheel_sqp_resource_budget_exceeded"

    timed = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
        budget=tight_budget,
    )
    timed[4].now = 1.0
    with pytest.raises(WheelSQPInitializationError) as caught:
        initialize_wheel_trajectory_v2(
            timed[0],
            timed[1],
            PROFILE,
            timed[2],
            timed[3],
        )
    assert caught.value.reason_code == "planning_deadline_expired"


def test_deadline_flip_at_digest_boundary_returns_no_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(5, 1)
    corridor, request, ledger, deadline, clock = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    original_snapshot_hash = wheel_initialization.snapshot_hash

    def expiring_snapshot_hash(value: TerrainSnapshotV2) -> str:
        digest = original_snapshot_hash(value)
        clock.now = 1.0
        return digest

    monkeypatch.setattr(wheel_initialization, "snapshot_hash", expiring_snapshot_hash)
    with pytest.raises(WheelSQPInitializationError) as caught:
        initialize_wheel_trajectory_v2(
            corridor,
            request,
            PROFILE,
            ledger,
            deadline,
        )
    assert caught.value.reason_code == "planning_deadline_expired"


def test_deadline_flip_after_split_return_returns_no_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(5, 1)
    corridor, request, ledger, deadline, clock = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    original_split = wheel_initialization._split_long_legs

    def expiring_split(*args: object, **kwargs: object) -> object:
        result = original_split(*args, **kwargs)
        clock.now = 1.0
        return result

    monkeypatch.setattr(wheel_initialization, "_split_long_legs", expiring_split)
    with pytest.raises(WheelSQPInitializationError) as caught:
        simplify_wheel_corridor_v2(corridor, request, PROFILE, ledger, deadline)

    assert caught.value.reason_code == "planning_deadline_expired"


def test_deadline_flip_during_last_terminal_energy_returns_no_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(5, 1)
    _, request, ledger, deadline, clock = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    original_energy = wheel_initialization.wheel_relative_energy_v1
    authority_calls = 0

    def expiring_energy(*args: object, **kwargs: object) -> float:
        nonlocal authority_calls
        authority_calls += 1
        result = original_energy(*args, **kwargs)
        if authority_calls == 6:
            clock.now = 1.0
        return result

    monkeypatch.setattr(wheel_initialization, "wheel_relative_energy_v1", expiring_energy)
    with pytest.raises(WheelSQPInitializationError) as caught:
        select_wheel_modes_v2(
            ((0.25, 0.25), (2.25, 0.25)),
            request,
            PROFILE,
            ledger,
            deadline,
        )

    assert authority_calls == 6
    assert caught.value.reason_code == "planning_deadline_expired"


def test_deadline_flip_during_final_guess_construction_returns_no_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(5, 1)
    corridor, request, ledger, deadline, clock = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    original_guess_type = wheel_initialization.WheelSQPInitialGuessV2

    def expiring_guess(**kwargs: object) -> WheelSQPInitialGuessV2:
        clock.now = 1.0
        return original_guess_type(**kwargs)

    monkeypatch.setattr(wheel_initialization, "WheelSQPInitialGuessV2", expiring_guess)
    with pytest.raises(WheelSQPInitializationError) as caught:
        initialize_wheel_trajectory_v2(corridor, request, PROFILE, ledger, deadline)

    assert caught.value.reason_code == "planning_deadline_expired"


def test_slew_bisection_checks_deadline_before_next_authority_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    deadline = PlanningDeadlineV2(0.0, 1.0, clock)
    ledger = WheelSQPWorkLedgerV1(_budget(), deadline)
    controls = (
        wheel_initialization._Control(0.0, 0.0, 0.05, 0.0, 0.0),
        wheel_initialization._Control(0.5, 0.0, 0.05, 0.025, 0.0),
    )
    original_authority = wheel_initialization.wheel_segment_center_control_slew_v1
    authority_calls = 0

    def expiring_authority(*args: object, **kwargs: object) -> tuple[float, float, float]:
        nonlocal authority_calls
        authority_calls += 1
        if authority_calls == 7:
            raise AssertionError("slew authority called after deadline expiry")
        result = original_authority(*args, **kwargs)
        if authority_calls == 6:
            clock.now = 1.0
        return result

    monkeypatch.setattr(
        wheel_initialization,
        "wheel_segment_center_control_slew_v1",
        expiring_authority,
    )
    with pytest.raises(WheelSQPInitializationError) as caught:
        wheel_initialization._repair_control_slew(controls, PROFILE, ledger)

    assert authority_calls == 6
    assert caught.value.reason_code == "planning_deadline_expired"


def test_raw_points_are_admitted_before_builder_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(1, 1)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        (Cell(0, 0),),
        PoseStateV2(0.1, 0.25, 0.0),
        PoseStateV2(0.25, 0.25, 0.0),
    )
    graph = WheelCorridorGraphV1.from_snapshot(snapshot, 30.0, ledger=ledger)
    components = build_blocked_components_v1(graph, ledger=ledger)
    monkeypatch.setattr(
        wheel_initialization,
        "_rebuild_corridor_authorities",
        lambda *args, **kwargs: (graph, components),
    )

    def forbidden_raw_builder(*args: object, **kwargs: object) -> object:
        raise AssertionError("raw point builder ran before memory admission")

    monkeypatch.setattr(
        wheel_initialization,
        "_raw_corridor_points",
        forbidden_raw_builder,
    )
    _leave_memory_bytes(ledger, 1)
    with pytest.raises(WheelSQPInitializationError) as caught:
        simplify_wheel_corridor_v2(corridor, request, PROFILE, ledger, deadline)

    assert caught.value.reason_code == "wheel_sqp_resource_budget_exceeded"


def test_controls_are_admitted_before_control_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(5, 1)
    _, request, ledger, _, _ = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )

    def forbidden_control(*args: object, **kwargs: object) -> object:
        raise AssertionError("control constructed before memory admission")

    monkeypatch.setattr(wheel_initialization, "_Control", forbidden_control)
    _leave_memory_bytes(ledger, 1)
    with pytest.raises(WheelSQPInitializationError) as caught:
        wheel_initialization._schedule(
            ((0.25, 0.25), (2.25, 0.25)),
            (WheelSQPModeV2.FORWARD,),
            request,
            PROFILE,
            ledger,
        )

    assert caught.value.reason_code == "wheel_sqp_resource_budget_exceeded"


def test_digest_bytes_are_admitted_before_canonical_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(5, 1)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    guess = initialize_wheel_trajectory_v2(corridor, request, PROFILE, ledger, deadline)

    def forbidden_payload(*args: object, **kwargs: object) -> object:
        raise AssertionError("digest payload was built before memory admission")

    def forbidden_encoder(*args: object, **kwargs: object) -> object:
        raise AssertionError("canonical encoder ran before memory admission")

    monkeypatch.setattr(
        wheel_initialization,
        "_initial_guess_payload",
        forbidden_payload,
        raising=False,
    )
    monkeypatch.setattr(wheel_initialization, "canonical_json_bytes", forbidden_encoder)
    _leave_memory_bytes(ledger, 1)
    with pytest.raises(WheelSQPInitializationError) as caught:
        wheel_initialization._initial_guess_digest(
            corridor,
            request,
            PROFILE,
            guess.segments,
            ledger,
        )

    assert caught.value.reason_code == "wheel_sqp_resource_budget_exceeded"


def test_initializer_calls_the_single_task2_relative_energy_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(5, 1)
    corridor, request, ledger, deadline, _ = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    calls: list[tuple[float, float, float]] = []
    authority = wheel_initialization.wheel_relative_energy_v1

    def recording_authority(v_mps, omega_radps, duration_s, profile):
        calls.append((v_mps, omega_radps, duration_s))
        return authority(v_mps, omega_radps, duration_s, profile)

    monkeypatch.setattr(
        wheel_initialization,
        "wheel_relative_energy_v1",
        recording_authority,
    )
    guess = initialize_wheel_trajectory_v2(
        corridor,
        request,
        PROFILE,
        ledger,
        deadline,
    )

    assert calls
    assert len(calls) >= len(guess.segments)


def test_digest_excludes_ledger_counters_but_binds_resource_and_snapshot_identity() -> None:
    base_snapshot = _snapshot(5, 1)
    first = _case(
        base_snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    second = _case(
        base_snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    second[2].charge_memory(13)
    first_guess = initialize_wheel_trajectory_v2(
        first[0], first[1], PROFILE, first[2], first[3]
    )
    second_guess = initialize_wheel_trajectory_v2(
        second[0], second[1], PROFILE, second[2], second[3]
    )
    assert first_guess.initial_guess_hash == second_guess.initial_guess_hash

    resource_bound = _case(
        base_snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
        budget=_budget(max_memory_bytes=10_000_000),
    )
    resource_guess = initialize_wheel_trajectory_v2(
        resource_bound[0],
        resource_bound[1],
        PROFILE,
        resource_bound[2],
        resource_bound[3],
    )
    assert resource_guess.modes == first_guess.modes
    assert resource_guess.initial_guess_hash != first_guess.initial_guess_hash

    confidence_snapshot = _snapshot(5, 1, confidence={Cell(2, 0): 0.25})
    confidence_bound = _case(
        confidence_snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
    )
    confidence_guess = initialize_wheel_trajectory_v2(
        confidence_bound[0],
        confidence_bound[1],
        PROFILE,
        confidence_bound[2],
        confidence_bound[3],
    )
    assert confidence_guess.modes == first_guess.modes
    assert confidence_guess.initial_guess_hash != first_guess.initial_guess_hash


def test_seed_changes_digest_but_not_geometry_or_selected_modes() -> None:
    snapshot = _snapshot(5, 1)
    first = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
        seed=1,
    )
    second = _case(
        snapshot,
        _line_cells(4),
        PoseStateV2(0.25, 0.25, 0.0),
        PoseStateV2(2.25, 0.25, 0.0),
        seed=2,
    )
    first_guess = initialize_wheel_trajectory_v2(
        first[0], first[1], PROFILE, first[2], first[3]
    )
    second_guess = initialize_wheel_trajectory_v2(
        second[0], second[1], PROFILE, second[2], second[3]
    )

    assert tuple(segment.mode for segment in first_guess.segments) == tuple(
        segment.mode for segment in second_guess.segments
    )
    assert tuple(
        (segment.v_mps, segment.omega_radps, segment.duration_s)
        for segment in first_guess.segments
    ) == tuple(
        (segment.v_mps, segment.omega_radps, segment.duration_s)
        for segment in second_guess.segments
    )
    assert first_guess.initial_guess_hash != second_guess.initial_guess_hash


def test_hash_seed_subprocesses_have_identical_modes_bytes_and_digest() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = r'''
import json
import numpy as np
from path_planner.core import Cell
from path_planner.v2.contracts import AcceleratorPolicyV2, ObjectiveProfileV2, PlanningRequestV2, PlatformKindV2, PoseStateV2, ResourceBudgetV2
from path_planner.v2.profiles import PlatformProfileV2, WheelKinematicSQPProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import FineGridGeometryV2, TerrainProvenanceV2, TerrainSnapshotV2
from path_planner.v2.wheel_corridors import WheelCorridorGraphV1, build_blocked_components_v1, topology_signature_v1, wheel_corridor_path_hash_v1
from path_planner.v2.wheel_sqp_contracts import WheelCorridorV2, WheelSQPWorkLedgerV1
from path_planner.v2.wheel_sqp_initialization import initialize_wheel_trajectory_v2

geometry = FineGridGeometryV2(5, 1)
snapshot = TerrainSnapshotV2(
    geometry,
    np.zeros(geometry.shape), np.zeros(geometry.shape),
    np.ones(geometry.shape, dtype=bool), np.zeros(geometry.shape, dtype=bool),
    np.ones(geometry.shape, dtype=bool), np.ones(geometry.shape),
    TerrainProvenanceV2("measured_terrain/v1", "task4", "fixture", True),
)
budget = ResourceBudgetV2(100000, 100000, 0)
deadline = PlanningDeadlineV2(0.0, 1.0, lambda: 0.0)
ledger = WheelSQPWorkLedgerV1(budget, deadline)
graph = WheelCorridorGraphV1.from_snapshot(snapshot, 30.0, ledger=ledger)
components = build_blocked_components_v1(graph, ledger=ledger)
cells = tuple(Cell(x, 0) for x in range(5))
signature = topology_signature_v1(cells, components, ledger=ledger)
digest = wheel_corridor_path_hash_v1(graph.snapshot_identity, cells, ledger=ledger)
ledger.charge_route_states(len(cells))
guide_cost = sum(graph.edge_cost(left, right) for left, right in zip(cells, cells[1:]))
path_length = sum(graph.edge_length_m(left, right) for left, right in zip(cells, cells[1:]))
corridor = WheelCorridorV2(0, cells, digest, guide_cost, path_length, signature)
profile = WheelKinematicSQPProfileV2(profile=PlatformProfileV2(
    "scout-mini-wheel-kinematic-sqp/v1", PlatformKindV2.WHEEL,
    "wheel_kinematic_corridor_sqp/v1", False, 30.0, 0.25,
    0.08726646259971647,
))
request = PlanningRequestV2(
    "task4", profile.profile.profile_id,
    PoseStateV2(0.25, 0.25, 0.0), PoseStateV2(2.25, 0.25, 0.0),
    snapshot, ObjectiveProfileV2(), budget, 2.0,
    AcceleratorPolicyV2.DISABLED, 19,
)
guess = initialize_wheel_trajectory_v2(corridor, request, profile, ledger, deadline)
print(json.dumps({
    "digest": guess.initial_guess_hash,
    "segments": [[s.mode.value, s.v_mps, s.omega_radps, s.duration_s,
                  [s.start_state.x_m, s.start_state.y_m, s.start_state.heading_rad],
                  [s.end_state.x_m, s.end_state.y_m, s.end_state.heading_rad]]
                 for s in guess.segments],
}, sort_keys=True, separators=(",", ":")))
'''
    outputs = []
    for seed in ("0", "1"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(repo_root / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())

    assert json.loads(outputs[0]) == json.loads(outputs[1])
    assert outputs[0] == outputs[1]
