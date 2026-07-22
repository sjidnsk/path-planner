from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import hypot, inf, nextafter, pi
from types import SimpleNamespace

import numpy as np
import pytest

import path_planner.v2.wheel_sqp_solver as solver
import path_planner.v2.wheel_corridors as wheel_corridors
from path_planner.core import Cell
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.profiles import PlatformProfileV2, WheelKinematicSQPProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
    snapshot_hash,
)
from path_planner.v2.wheel_corridors import (
    WheelSQPTerrainGuideV1,
    wheel_corridor_path_hash_v1,
    wheel_sqp_corridor_broadphase_cell_bound_v1,
)
from path_planner.v2.wheel_kinematics import integrate_wheel_segment_v2
from path_planner.v2.wheel_sqp_contracts import (
    WheelCorridorV2,
    WheelSQPInitialGuessV2,
    WheelSQPModeV2,
    WheelSQPResourceEstimateV1,
    WheelSQPResourceLedgerV1,
    WheelSQPWorkLedgerV1,
    WheelSQPWorkLimitError,
    WheelTopologySignatureV1,
    _WheelSQPInitialSegmentV1,
)
from path_planner.v2.wheel_sqp_initialization import _initial_guess_payload
from path_planner.v2.wheel_sqp_solver import (
    WheelSQPProblemV1,
    audit_wheel_sqp_candidate_v2,
    build_wheel_sqp_terrain_constraints_v2,
    estimate_wheel_sqp_attempt_resources_v2,
    solve_wheel_sqp_v2,
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
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _snapshot(
    *,
    width: int = 10,
    height: int = 6,
    origin: tuple[float, float] = (-1.0, -1.0),
    unknown: set[Cell] = frozenset(),
    hard: set[Cell] = frozenset(),
    not_traversable: set[Cell] = frozenset(),
    slope: dict[Cell, float] | None = None,
) -> TerrainSnapshotV2:
    geometry = FineGridGeometryV2(width, height, origin=origin)
    observed = np.ones(geometry.shape, dtype=np.bool_)
    hard_mask = np.zeros(geometry.shape, dtype=np.bool_)
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    slope_deg = np.zeros(geometry.shape, dtype=np.float64)
    for cell in unknown:
        observed[cell.y, cell.x] = False
    for cell in hard:
        hard_mask[cell.y, cell.x] = True
        traversable[cell.y, cell.x] = False
    for cell in not_traversable:
        traversable[cell.y, cell.x] = False
    for cell, value in (slope or {}).items():
        slope_deg[cell.y, cell.x] = value
    return TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape, dtype=np.float64),
        slope_deg=slope_deg,
        traversable_mask=traversable,
        hard_obstacle_mask=hard_mask,
        observed_mask=observed,
        confidence=np.ones(geometry.shape, dtype=np.float64),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task6-fixture",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )


def _guide(snapshot: TerrainSnapshotV2) -> WheelSQPTerrainGuideV1:
    deadline = PlanningDeadlineV2(0.0, 5.0, lambda: 0.0)
    ledger = WheelSQPWorkLedgerV1(ResourceBudgetV2(100_000, 100_000, 0), deadline)
    return WheelSQPTerrainGuideV1.from_snapshot(
        snapshot,
        30.0,
        ledger=ledger,
    )


def _problem(
    snapshot: TerrainSnapshotV2,
    *,
    controls: tuple[tuple[float, float, float, WheelSQPModeV2], ...] = (
        (0.5, 0.0, 2.0, WheelSQPModeV2.FORWARD),
    ),
    start: PoseStateV2 = PoseStateV2(0.0, 0.0, 0.0),
    cells: tuple[Cell, ...] = (Cell(2, 2), Cell(3, 2), Cell(4, 2)),
    budget: ResourceBudgetV2 = ResourceBudgetV2(100_000, 100_000, 0),
    deadline_end: float = 20.0,
    objective: ObjectiveProfileV2 | None = None,
    terrain_guide: WheelSQPTerrainGuideV1 | None = None,
) -> tuple[WheelSQPProblemV1, PlanningDeadlineV2, WheelSQPWorkLedgerV1, _Clock]:
    clock = _Clock()
    deadline = PlanningDeadlineV2(0.0, deadline_end, clock)
    ledger = WheelSQPWorkLedgerV1(budget, deadline)
    guide = terrain_guide or WheelSQPTerrainGuideV1.from_snapshot(
        snapshot, 30.0, ledger=ledger
    )
    segments: list[_WheelSQPInitialSegmentV1] = []
    current = start
    for v_mps, omega_radps, duration_s, mode in controls:
        end = integrate_wheel_segment_v2(current, v_mps, omega_radps, duration_s)
        segment = _WheelSQPInitialSegmentV1(
            start_state=current,
            end_state=end,
            v_mps=v_mps,
            omega_radps=omega_radps,
            duration_s=duration_s,
            mode=mode,
            distance_m=abs(v_mps) * duration_s,
            relative_energy=solver.wheel_relative_energy_v1(
                v_mps, omega_radps, duration_s, PROFILE
            ),
        )
        segments.append(segment)
        current = end
    request = PlanningRequestV2(
        request_id="wheel-sqp-task6",
        platform_profile_id=PROFILE.profile.profile_id,
        start_state=start,
        goal_state=current,
        terrain_snapshot=snapshot,
        objective_profile=objective or ObjectiveProfileV2(),
        resource_budget=budget,
        timeout_s=deadline_end,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=29,
    )
    corridor_hash = wheel_corridor_path_hash_v1(snapshot_hash(snapshot), cells)
    corridor = WheelCorridorV2(
        corridor_index=0,
        cells=cells,
        corridor_hash=corridor_hash,
        guide_cost=1.0,
        path_length_m=1.0,
        topology_signature=WheelTopologySignatureV1(()),
    )
    segment_tuple = tuple(segments)
    initial_hash = sha256(
        canonical_json_bytes(
            _initial_guess_payload(
                corridor,
                request,
                PROFILE,
                segment_tuple,
                snapshot_hash(snapshot),
            )
        )
    ).hexdigest()
    guess = WheelSQPInitialGuessV2(
        corridor_hash=corridor_hash,
        start_state=start,
        requested_goal=current,
        segments=segment_tuple,
        actual_endpoint=current,
        initial_guess_hash=initial_hash,
    )
    old_receipt = WheelSQPResourceLedgerV1(0.01, deadline_end, True, None)
    problem = WheelSQPProblemV1(
        corridor=corridor,
        initial_guess=guess,
        request=request,
        profile=PROFILE,
        post_solver_reserve=old_receipt,
        terrain_guide=guide,
    )
    return problem, deadline, ledger, clock


def test_optimizer_invalid_records_use_stable_reason_precedence_and_slope_boundary() -> None:
    both = Cell(5, 1)
    snapshot = _snapshot(
        unknown={Cell(1, 1), both},
        hard={Cell(2, 1), both},
        not_traversable={Cell(3, 1), both},
        slope={Cell(4, 1): nextafter(30.0, inf), Cell(6, 1): 30.0, both: 31.0},
    )
    guide = _guide(snapshot)

    assert guide.invalid_cells == (
        Cell(1, 1), Cell(2, 1), Cell(3, 1), Cell(4, 1), both
    )
    assert tuple(record.reason_code for record in guide.invalid_records) == (
        "terrain_unknown",
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_slope_exceeded",
        "terrain_unknown",
    )
    assert Cell(6, 1) not in guide.invalid_cells


def test_signed_clearance_has_deterministic_outside_inside_and_face_tie_gradients() -> None:
    guide = _guide(_snapshot(width=8, height=8, hard={Cell(3, 3)}))
    outside = guide.nearest_signed_clearance(0.0, 0.75)
    assert outside.signed_distance_m == pytest.approx(0.5, abs=1.0e-15)
    assert (outside.gradient_x, outside.gradient_y) == (-1.0, 0.0)
    inside_center = guide.nearest_signed_clearance(0.75, 0.75)
    assert inside_center.signed_distance_m == pytest.approx(-0.25, abs=1.0e-15)
    assert (inside_center.gradient_x, inside_center.gradient_y) == (-1.0, 0.0)
    assert inside_center.reason_code == "terrain_hard_obstacle"


def test_map_exterior_signed_clearance_points_toward_interior() -> None:
    guide = _guide(_snapshot(width=2, height=2, origin=(0.0, 0.0)))
    inside = guide.nearest_signed_clearance(0.5, 0.5)
    assert inside.signed_distance_m == pytest.approx(0.5, abs=0.0)
    assert (inside.gradient_x, inside.gradient_y) == (1.0, 0.0)
    outside = guide.nearest_signed_clearance(-0.25, 0.5)
    assert outside.signed_distance_m == pytest.approx(-0.25, abs=0.0)
    assert (outside.gradient_x, outside.gradient_y) == (1.0, 0.0)
    assert outside.reason_code == "terrain_out_of_bounds"


def test_five_analytic_fraction_rows_detect_mid_segment_obstacle_and_match_jacobian() -> None:
    snapshot = _snapshot(hard={Cell(3, 2)})
    problem, _, _, _ = _problem(
        snapshot,
        start=PoseStateV2(0.0, 0.1, 0.0),
        controls=((1.0, 0.0, 1.0, WheelSQPModeV2.FORWARD),),
    )
    vector = problem.initial_vector
    terrain = build_wheel_sqp_terrain_constraints_v2(problem, vector)
    assert terrain.values.shape == (5,)
    assert terrain.jacobian.shape == (5, 6)
    assert terrain.values[2] < 0.0
    step = 1.0e-6
    for column in range(vector.size):
        lower = vector.copy()
        upper = vector.copy()
        lower[column] -= step
        upper[column] += step
        expected = (
            problem.terrain_inequalities(upper)
            - problem.terrain_inequalities(lower)
        ) / (2.0 * step)
        assert terrain.jacobian[:, column] == pytest.approx(
            expected, rel=3.0e-5, abs=3.0e-7
        )


def test_resource_estimate_uses_frozen_counts_and_exact_byte_formula() -> None:
    problem, deadline, ledger, _ = _problem(_snapshot())
    estimate = estimate_wheel_sqp_attempt_resources_v2(problem, deadline, ledger)
    assert type(estimate) is WheelSQPResourceEstimateV1
    assert estimate.segment_count == 1
    assert estimate.variable_count == 6
    assert estimate.equality_count == 3
    assert estimate.base_inequality_count == 3
    assert estimate.terrain_inequality_count == 5
    assert estimate.total_constraint_count == 11
    assert estimate.decision_bytes == 48
    assert estimate.jacobian_bytes == 8 * 11 * 6
    assert estimate.required_bytes == (
        estimate.decision_bytes
        + estimate.jacobian_bytes
        + estimate.solver_bytes
        + estimate.l2_queue_bytes
        + estimate.codec_bytes
    )
    assert estimate.post_solver_reserve.accepted is True


def test_resource_admission_atomically_charges_memory_and_route_states() -> None:
    problem, deadline, ledger, _ = _problem(_snapshot())
    before = (ledger.accounted_bytes, ledger.route_states)
    estimate = estimate_wheel_sqp_attempt_resources_v2(problem, deadline, ledger)
    assert ledger.accounted_bytes == before[0] + estimate.required_bytes
    assert ledger.route_states == before[1] + estimate.encoded_state_bound


def test_strict_reserve_rejects_before_backend_load(monkeypatch: pytest.MonkeyPatch) -> None:
    problem, deadline, ledger, clock = _problem(_snapshot(), deadline_end=1.0)
    clock.now = nextafter(1.0, 0.0)
    monkeypatch.setattr(
        solver,
        "_load_scipy_optimize_v1",
        lambda: (_ for _ in ()).throw(AssertionError("backend loaded before admission")),
    )
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code in {
        "planning_deadline_expired",
        "wheel_sqp_resource_budget_exceeded",
    }
    assert result.candidate is None


def test_rho_zero_never_calls_positive_duration_authorities_with_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, _, _, _ = _problem(_snapshot())
    original_integrate = solver.integrate_wheel_segment_v2
    original_jacobian = solver.wheel_segment_jacobian_v2
    durations: list[float] = []

    def checked_integrate(start, v, omega, duration):
        assert duration > 0.0
        durations.append(duration)
        return original_integrate(start, v, omega, duration)

    def checked_jacobian(start, v, omega, duration):
        assert duration > 0.0
        return original_jacobian(start, v, omega, duration)

    monkeypatch.setattr(solver, "integrate_wheel_segment_v2", checked_integrate)
    monkeypatch.setattr(solver, "wheel_segment_jacobian_v2", checked_jacobian)
    build_wheel_sqp_terrain_constraints_v2(problem, problem.initial_vector)
    assert durations == [0.5, 1.0, 1.5, 2.0]


def test_later_segment_terrain_jacobian_binds_previous_node_and_rho_duration() -> None:
    problem, _, _, _ = _problem(
        _snapshot(width=20, height=20, origin=(-5.0, -5.0)),
        start=PoseStateV2(0.0, 0.0, 0.2),
        controls=(
            (0.3, 0.1, 1.0, WheelSQPModeV2.FORWARD),
            (0.3, 0.1, 1.0, WheelSQPModeV2.FORWARD),
        ),
        cells=(Cell(10, 10), Cell(11, 10)),
    )
    vector = problem.initial_vector
    actual = problem.terrain_jacobian(vector)
    step = 1.0e-6
    for column in (0, 1, 2, 9, 10, 11):
        lower = vector.copy()
        upper = vector.copy()
        lower[column] -= step
        upper[column] += step
        expected = (
            problem.terrain_inequalities(upper)[5:]
            - problem.terrain_inequalities(lower)[5:]
        ) / (2.0 * step)
        assert actual[5:, column] == pytest.approx(
            expected,
            rel=3.0e-5,
            abs=3.0e-7,
        )


def test_task5_prefix_and_task6_total_constraint_counts_are_frozen() -> None:
    problem, deadline, ledger, _ = _problem(
        _snapshot(),
        controls=(
            (0.25, 0.0, 2.0, WheelSQPModeV2.FORWARD),
            (0.25, 0.0, 2.0, WheelSQPModeV2.FORWARD),
        ),
    )
    vector = problem.initial_vector
    assert problem.dynamics_residual(vector).shape == (6,)
    assert problem.inequalities(vector).shape == (17,)
    assert problem.inequality_jacobian(vector).shape == (17, 12)
    estimate = estimate_wheel_sqp_attempt_resources_v2(problem, deadline, ledger)
    assert estimate.base_inequality_count == 7
    assert estimate.terrain_inequality_count == 10
    assert estimate.total_constraint_count == 23


def test_broadphase_cap_uses_cap_plus_one_rejection_without_cell_enumeration() -> None:
    deadline = PlanningDeadlineV2(0.0, 5.0, lambda: 0.0)
    ledger = WheelSQPWorkLedgerV1(ResourceBudgetV2(1, 1, 0), deadline)
    corridor = WheelCorridorV2(
        corridor_index=0,
        cells=(Cell(0, 0),),
        corridor_hash="a" * 64,
        guide_cost=0.0,
        path_length_m=0.0,
        topology_signature=WheelTopologySignatureV1(()),
    )
    exact = wheel_sqp_corridor_broadphase_cell_bound_v1(
        corridor,
        FineGridGeometryV2(1000, 1000),
        1.0e9,
        ledger=ledger,
    )
    assert exact == 1_000_000
    with pytest.raises(WheelSQPWorkLimitError) as caught:
        wheel_sqp_corridor_broadphase_cell_bound_v1(
            corridor,
            FineGridGeometryV2(1001, 1000),
            1.0e9,
            ledger=ledger,
        )
    assert caught.value.reason_code == "wheel_sqp_resource_budget_exceeded"


def test_closed_broadphase_aabb_includes_exact_lower_gridline_contacts() -> None:
    deadline = PlanningDeadlineV2(0.0, 5.0, lambda: 0.0)
    ledger = WheelSQPWorkLedgerV1(ResourceBudgetV2(1, 1, 0), deadline)
    corridor = WheelCorridorV2(
        corridor_index=0,
        cells=(Cell(1, 1),),
        corridor_hash="b" * 64,
        guide_cost=0.0,
        path_length_m=0.0,
        topology_signature=WheelTopologySignatureV1(()),
    )
    geometry = FineGridGeometryV2(4, 4)
    below = 0.25
    for _ in range(3):
        below = nextafter(below, 0.0)

    assert wheel_sqp_corridor_broadphase_cell_bound_v1(
        corridor, geometry, below, ledger=ledger
    ) == 1
    assert wheel_sqp_corridor_broadphase_cell_bound_v1(
        corridor, geometry, 0.25, ledger=ledger
    ) == 9
    assert wheel_sqp_corridor_broadphase_cell_bound_v1(
        corridor, geometry, nextafter(0.25, inf), ledger=ledger
    ) == 9


def test_interval_saturation_and_sampling_ceil_preserve_exact_nextafter() -> None:
    assert solver._interval_record_bound_v1(0, 262_144) == 0
    assert solver._interval_record_bound_v1(1, 262_144) == 262_144
    assert solver._ceil_sampling_intervals_v1(1.0, 0.25) == 4
    assert solver._ceil_sampling_intervals_v1(nextafter(1.0, inf), 0.25) == 5


def test_atomic_reservation_exact_caps_and_failure_leave_no_half_charge() -> None:
    deadline = PlanningDeadlineV2(0.0, 5.0, lambda: 0.0)
    exact = WheelSQPWorkLedgerV1(ResourceBudgetV2(1, 10, 100), deadline)
    exact.reserve_attempt(100, 10)
    assert (exact.accounted_bytes, exact.route_states) == (100, 10)

    rejected = WheelSQPWorkLedgerV1(ResourceBudgetV2(1, 10, 100), deadline)
    with pytest.raises(WheelSQPWorkLimitError):
        rejected.reserve_attempt(101, 10)
    assert (rejected.accounted_bytes, rejected.route_states) == (0, 0)
    with pytest.raises(WheelSQPWorkLimitError):
        rejected.reserve_attempt(100, 11)
    assert (rejected.accounted_bytes, rejected.route_states) == (0, 0)


def test_attempt_route_and_memory_exact_boundaries() -> None:
    probe, probe_deadline, probe_ledger, _ = _problem(_snapshot())
    before_memory = probe_ledger.accounted_bytes
    estimate = estimate_wheel_sqp_attempt_resources_v2(
        probe,
        probe_deadline,
        probe_ledger,
    )
    exact_memory = before_memory + estimate.required_bytes
    exact_states = estimate.encoded_state_bound

    exact_budget = ResourceBudgetV2(100_000, exact_states, exact_memory)
    exact_problem, exact_deadline, exact_ledger, _ = _problem(
        _snapshot(), budget=exact_budget
    )
    exact_estimate = estimate_wheel_sqp_attempt_resources_v2(
        exact_problem, exact_deadline, exact_ledger
    )
    assert exact_ledger.accounted_bytes == exact_memory
    assert exact_ledger.route_states == exact_estimate.encoded_state_bound

    route_budget = ResourceBudgetV2(100_000, exact_states - 1, exact_memory)
    route_problem, route_deadline, route_ledger, _ = _problem(
        _snapshot(), budget=route_budget
    )
    before = (route_ledger.accounted_bytes, route_ledger.route_states)
    with pytest.raises(WheelSQPWorkLimitError):
        estimate_wheel_sqp_attempt_resources_v2(
            route_problem, route_deadline, route_ledger
        )
    assert (route_ledger.accounted_bytes, route_ledger.route_states) == before

    memory_budget = ResourceBudgetV2(100_000, exact_states, exact_memory - 1)
    memory_problem, memory_deadline, memory_ledger, _ = _problem(
        _snapshot(), budget=memory_budget
    )
    before = (memory_ledger.accounted_bytes, memory_ledger.route_states)
    with pytest.raises(WheelSQPWorkLimitError):
        estimate_wheel_sqp_attempt_resources_v2(
            memory_problem, memory_deadline, memory_ledger
        )
    assert (memory_ledger.accounted_bytes, memory_ledger.route_states) == before


def test_rejected_admission_precedes_numpy_allocation_and_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = ResourceBudgetV2(100_000, 100_000, 1)
    problem, deadline, ledger, _ = _problem(_snapshot(), budget=budget)
    monkeypatch.setattr(
        solver.np,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("numpy allocation before admission")
        ),
    )
    monkeypatch.setattr(
        solver,
        "_load_scipy_optimize_v1",
        lambda: (_ for _ in ()).throw(AssertionError("backend before admission")),
    )
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert result.candidate is None


def test_audit_marks_only_terminal_failure_and_never_terrain_failure_as_incumbent_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_problem, _, _, _ = _problem(_snapshot())
    terminal_only = open_problem.initial_vector.copy()
    terminal_only[3] = 0.7
    terminal_only[0:3] = np.array((1.4, 0.0, 0.0), dtype=np.float64)
    terminal_audit = audit_wheel_sqp_candidate_v2(open_problem, terminal_only)
    assert terminal_audit.reason_code == "wheel_sqp_infeasible"
    assert terminal_audit.terminal_only_failure is True

    terrain_problem, deadline, ledger, _ = _problem(
        _snapshot(hard={Cell(5, 2)})
    )
    terrain_bad = terrain_problem.initial_vector.copy()
    terrain_bad[3] = 1.0
    terrain_bad[0:3] = np.array((2.0, 0.0, 0.0), dtype=np.float64)
    terrain_audit = audit_wheel_sqp_candidate_v2(terrain_problem, terrain_bad)
    assert terrain_audit.reason_code == "wheel_sqp_infeasible"
    assert terrain_audit.terminal_only_failure is False

    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
    monkeypatch.setattr(
        solver,
        "_run_slsqp_v1",
        lambda **kwargs: SimpleNamespace(x=terrain_bad, success=True, nit=1),
    )
    result = solve_wheel_sqp_v2(terrain_problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_infeasible"
    assert result.candidate is None


def test_n48_resource_counts_hit_the_exact_frozen_boundary() -> None:
    controls = tuple(
        (0.0, 0.0, 0.05, WheelSQPModeV2.STOP)
        for _ in range(48)
    )
    problem, deadline, ledger, _ = _problem(_snapshot(), controls=controls)
    estimate = estimate_wheel_sqp_attempt_resources_v2(problem, deadline, ledger)
    assert estimate.segment_count == 48
    assert estimate.variable_count == 288
    assert estimate.equality_count == 144
    assert estimate.base_inequality_count == 191
    assert estimate.terrain_inequality_count == 240
    assert estimate.total_constraint_count == 575


def test_strict_reserve_equality_and_adjacent_binary64_values_have_exact_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe, probe_deadline, probe_ledger, _ = _problem(_snapshot())
    reserve = estimate_wheel_sqp_attempt_resources_v2(
        probe, probe_deadline, probe_ledger
    ).post_solver_reserve.reserve_s

    for remaining in (nextafter(reserve, 0.0), reserve):
        problem, deadline, ledger, _ = _problem(
            _snapshot(), deadline_end=remaining
        )
        result = solve_wheel_sqp_v2(problem, deadline, ledger)
        assert result.reason_code == "wheel_sqp_resource_budget_exceeded"
        assert result.candidate is None

    admitted, admitted_deadline, admitted_ledger, _ = _problem(
        _snapshot(), deadline_end=nextafter(reserve, inf)
    )
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: None)
    result = solve_wheel_sqp_v2(admitted, admitted_deadline, admitted_ledger)
    assert result.reason_code == "wheel_sqp_backend_unavailable"

    expired, expired_deadline, expired_ledger, clock = _problem(_snapshot())
    clock.now = expired_deadline.deadline_monotonic_s
    result = solve_wheel_sqp_v2(expired, expired_deadline, expired_ledger)
    assert result.reason_code == "planning_deadline_expired"


def test_resource_estimate_rejects_forged_component_bytes() -> None:
    problem, deadline, ledger, _ = _problem(_snapshot())
    estimate = estimate_wheel_sqp_attempt_resources_v2(problem, deadline, ledger)
    with pytest.raises(TypeError):
        replace(
            estimate,
            decision_bytes=estimate.decision_bytes + 1,
            required_bytes=estimate.required_bytes + 1,
        )
    with pytest.raises(TypeError):
        replace(
            estimate,
            solver_bytes=estimate.solver_bytes + 1,
            required_bytes=estimate.required_bytes + 1,
        )


def test_terrain_guide_is_factory_only_and_binds_exact_snapshot_identity() -> None:
    snapshot = _snapshot()
    guide = _guide(snapshot)
    with pytest.raises(TypeError):
        WheelSQPTerrainGuideV1(
            terrain_snapshot_hash=guide.terrain_snapshot_hash,
            snapshot=snapshot,
            invalid_count=0,
            _records=b"",
            max_slope_deg=30.0,
        )
    with pytest.raises(TypeError):
        replace(guide, invalid_count=0, _records=b"")

    equal_but_distinct = _snapshot()
    assert snapshot_hash(equal_but_distinct) == snapshot_hash(snapshot)
    foreign_guide = _guide(equal_but_distinct)
    with pytest.raises(ValueError, match="exact request snapshot"):
        _problem(snapshot, terrain_guide=foreign_guide)


def test_terrain_guide_rechecks_deadline_after_snapshot_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot()
    clock = _Clock()
    deadline = PlanningDeadlineV2(0.0, 5.0, clock)
    ledger = WheelSQPWorkLedgerV1(ResourceBudgetV2(100_000, 100_000, 0), deadline)
    authority = wheel_corridors.snapshot_hash

    def expiring_hash(value: TerrainSnapshotV2) -> str:
        result = authority(value)
        clock.now = 5.0
        return result

    monkeypatch.setattr(wheel_corridors, "snapshot_hash", expiring_hash)
    with pytest.raises(WheelSQPWorkLimitError) as caught:
        WheelSQPTerrainGuideV1.from_snapshot(snapshot, 30.0, ledger=ledger)
    assert caught.value.reason_code == "planning_deadline_expired"


def test_resource_estimate_is_factory_only_and_replace_cannot_forge_bounds() -> None:
    problem, deadline, ledger, _ = _problem(_snapshot())
    estimate = estimate_wheel_sqp_attempt_resources_v2(problem, deadline, ledger)
    forged_required = (
        estimate.decision_bytes + estimate.jacobian_bytes + estimate.solver_bytes
    )
    with pytest.raises(TypeError):
        replace(
            estimate,
            broadphase_cell_bound=0,
            interval_record_bound=0,
            encoded_state_bound=0,
            encoded_scalar_bound=0,
            l2_queue_bytes=0,
            codec_bytes=0,
            required_bytes=forged_required,
        )
    with pytest.raises(TypeError):
        WheelSQPResourceEstimateV1(
            segment_count=1,
            variable_count=6,
            equality_count=3,
            base_inequality_count=3,
            terrain_inequality_count=5,
            total_constraint_count=11,
            broadphase_cell_bound=0,
            interval_record_bound=0,
            encoded_state_bound=0,
            encoded_scalar_bound=0,
            decision_bytes=48,
            jacobian_bytes=528,
            solver_bytes=16_777_216,
            l2_queue_bytes=0,
            codec_bytes=0,
            required_bytes=16_777_792,
            post_solver_reserve=estimate.post_solver_reserve,
        )


def test_expired_deadline_precedes_unsupported_risk_and_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, clock = _problem(
        _snapshot(),
        objective=ObjectiveProfileV2(risk_weight=1.0),
    )
    clock.now = deadline.deadline_monotonic_s
    monkeypatch.setattr(
        solver,
        "_load_scipy_optimize_v1",
        lambda: (_ for _ in ()).throw(AssertionError("backend loaded")),
    )
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "planning_deadline_expired"
    assert result.candidate is None


def test_value_only_terrain_path_does_not_build_segment_jacobians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, _, _, _ = _problem(_snapshot())
    monkeypatch.setattr(
        solver,
        "wheel_segment_jacobian_v2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("value-only terrain path built a Jacobian")
        ),
    )
    assert problem.terrain_inequalities(problem.initial_vector).shape == (5,)
