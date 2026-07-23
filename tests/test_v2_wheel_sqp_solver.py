from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import cos, inf, nextafter, pi, sin
from types import SimpleNamespace

import numpy as np
import pytest

import path_planner.v2.wheel_sqp_solver as solver
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
)
from path_planner.v2.wheel_kinematics import (
    integrate_wheel_segment_v2,
    wheel_relative_energy_jacobian_v1,
)
from path_planner.v2.wheel_sqp_contracts import (
    WheelCorridorV2,
    WheelSQPInitialGuessV2,
    WheelSQPModeV2,
    WheelSQPRepairConstraintV1,
    WheelSQPResourceLedgerV1,
    WheelSQPStatusV2,
    WheelSQPWorkLedgerV1,
    WheelTopologySignatureV1,
    _WheelSQPInitialSegmentV1,
)
from path_planner.v2.wheel_sqp_initialization import _initial_guess_payload
from path_planner.v2.wheel_sqp_serialization import (
    CanonicalWheelCandidateV1,
    materialize_canonical_wheel_candidate_v2,
)
from path_planner.v2.wheel_sqp_solver import (
    WheelSQPLayoutV1,
    WheelSQPProblemV1,
    audit_wheel_sqp_candidate_v2,
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
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _snapshot() -> TerrainSnapshotV2:
    geometry = FineGridGeometryV2(8, 6, origin=(-1.0, -1.0))
    return TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape, dtype=np.float64),
        slope_deg=np.zeros(geometry.shape, dtype=np.float64),
        traversable_mask=np.ones(geometry.shape, dtype=np.bool_),
        hard_obstacle_mask=np.zeros(geometry.shape, dtype=np.bool_),
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape, dtype=np.float64),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task5-fixture",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )


def _initial_segment(
    start: PoseStateV2,
    *,
    v: float,
    omega: float,
    dt: float,
    mode: WheelSQPModeV2,
) -> _WheelSQPInitialSegmentV1:
    end = integrate_wheel_segment_v2(start, v, omega, dt)
    return _WheelSQPInitialSegmentV1(
        start_state=start,
        end_state=end,
        v_mps=v,
        omega_radps=omega,
        duration_s=dt,
        mode=mode,
        distance_m=abs(v) * dt,
        relative_energy=solver.wheel_relative_energy_v1(v, omega, dt, PROFILE),
    )


def _problem(
    controls: tuple[tuple[float, float, float, WheelSQPModeV2], ...] = (
        (0.5, 0.0, 2.0, WheelSQPModeV2.FORWARD),
    ),
    *,
    goal: PoseStateV2 | None = None,
    objective: ObjectiveProfileV2 | None = None,
) -> tuple[WheelSQPProblemV1, PlanningDeadlineV2, WheelSQPWorkLedgerV1, _Clock]:
    terrain = _snapshot()
    budget = ResourceBudgetV2(100_000, 100_000, 0)
    clock = _Clock()
    deadline = PlanningDeadlineV2(0.0, 5.0, clock)
    ledger = WheelSQPWorkLedgerV1(budget, deadline)
    terrain_guide = WheelSQPTerrainGuideV1.from_snapshot(
        terrain,
        30.0,
        ledger=ledger,
    )
    start = PoseStateV2(0.0, 0.0, 0.0)
    segments: list[_WheelSQPInitialSegmentV1] = []
    current = start
    for v, omega, dt, mode in controls:
        segment = _initial_segment(current, v=v, omega=omega, dt=dt, mode=mode)
        segments.append(segment)
        current = segment.end_state
    requested_goal = goal or current
    request = PlanningRequestV2(
        request_id="wheel-sqp-task5",
        platform_profile_id=PROFILE.profile.profile_id,
        start_state=start,
        goal_state=requested_goal,
        terrain_snapshot=terrain,
        objective_profile=objective or ObjectiveProfileV2(),
        resource_budget=budget,
        timeout_s=5.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=23,
    )
    cells = (Cell(0, 0), Cell(1, 0), Cell(2, 0))
    corridor_hash = wheel_corridor_path_hash_v1(snapshot_hash(terrain), cells)
    corridor = WheelCorridorV2(
        corridor_index=0,
        cells=cells,
        corridor_hash=corridor_hash,
        guide_cost=2.0,
        path_length_m=2.0,
        topology_signature=WheelTopologySignatureV1(()),
    )
    segment_tuple = tuple(segments)
    payload = _initial_guess_payload(
        corridor,
        request,
        PROFILE,
        segment_tuple,
        snapshot_hash(terrain),
    )
    guess = WheelSQPInitialGuessV2(
        corridor_hash=corridor_hash,
        start_state=start,
        requested_goal=requested_goal,
        segments=segment_tuple,
        actual_endpoint=current,
        initial_guess_hash=sha256(canonical_json_bytes(payload)).hexdigest(),
    )
    reserve = WheelSQPResourceLedgerV1(
        reserve_s=0.1,
        remaining_s=5.0,
        accepted=True,
        reason_code=None,
    )
    problem = WheelSQPProblemV1(
        corridor=corridor,
        initial_guess=guess,
        request=request,
        profile=PROFILE,
        post_solver_reserve=reserve,
        terrain_guide=terrain_guide,
    )
    return problem, deadline, ledger, clock


def _repair(
    problem: WheelSQPProblemV1,
    *,
    face: str = "left",
    segment_index: int = 0,
    rhos: tuple[float, float, float] = (0.25, 0.5, 0.75),
    bounds: tuple[float, float, float, float] = (1.0, 1.5, 0.0, 0.5),
) -> WheelSQPRepairConstraintV1:
    return WheelSQPRepairConstraintV1(
        source_candidate_hash="a" * 64,
        source_segment_hash="b" * 64,
        source_snapshot_hash=snapshot_hash(problem.request.terrain_snapshot),
        segment_index=segment_index,
        cell=Cell(4, 2),
        face=face,
        rho_lo=rhos[0],
        rho_mid=rhos[1],
        rho_hi=rhos[2],
        cell_left_x_m=bounds[0],
        cell_right_x_m=bounds[1],
        cell_bottom_y_m=bounds[2],
        cell_top_y_m=bounds[3],
        clearance_m=1.0e-4,
    )


def test_problem_accepts_only_exact_repair_with_matching_solver_identity() -> None:
    problem, _, _, _ = _problem()
    repair = _repair(problem)

    assert replace(problem, repair=repair).repair is repair
    with pytest.raises(TypeError, match="repair must be exact"):
        replace(problem, repair=object())
    with pytest.raises(ValueError, match="repair segment_index"):
        replace(problem, repair=replace(repair, segment_index=1))
    with pytest.raises(ValueError, match="repair snapshot"):
        replace(problem, repair=replace(repair, source_snapshot_hash="f" * 64))


@pytest.mark.parametrize("face", ("left", "right", "bottom", "top"))
def test_repair_rows_append_exact_fixed_face_margins_after_legacy_prefix(
    face: str,
) -> None:
    legacy, _, _, _ = _problem(
        controls=((0.4, 0.2, 1.0, WheelSQPModeV2.FORWARD),)
    )
    repair = _repair(legacy, face=face)
    problem = replace(legacy, repair=repair)
    vector = problem.initial_vector

    legacy_values = legacy.inequalities(vector)
    legacy_jacobian = legacy.inequality_jacobian(vector)
    actual_values = problem.inequalities(vector)
    actual_jacobian = problem.inequality_jacobian(vector)

    assert actual_values.shape == (legacy_values.size + 3,)
    assert actual_jacobian.shape == (legacy_jacobian.shape[0] + 3, vector.size)
    assert actual_values[:-3].tobytes() == legacy_values.tobytes()
    assert actual_jacobian[:-3].tobytes() == legacy_jacobian.tobytes()

    half_length = PROFILE.body_length_m / 2.0 + PROFILE.footprint_safety_margin_m
    half_width = PROFILE.body_width_m / 2.0 + PROFILE.footprint_safety_margin_m
    expected: list[float] = []
    for rho in repair.time_fractions:
        pose = integrate_wheel_segment_v2(
            problem.request.start_state,
            vector[3],
            vector[4],
            rho * vector[5],
        )
        support_x = half_length * abs(cos(pose.heading_rad)) + half_width * abs(
            sin(pose.heading_rad)
        )
        support_y = half_length * abs(sin(pose.heading_rad)) + half_width * abs(
            cos(pose.heading_rad)
        )
        expected.append(
            {
                "left": repair.cell_left_x_m
                - pose.x_m
                - support_x
                - repair.clearance_m,
                "right": pose.x_m
                - repair.cell_right_x_m
                - support_x
                - repair.clearance_m,
                "bottom": repair.cell_bottom_y_m
                - pose.y_m
                - support_y
                - repair.clearance_m,
                "top": pose.y_m
                - repair.cell_top_y_m
                - support_y
                - repair.clearance_m,
            }[face]
        )
    assert actual_values[-3:] == pytest.approx(expected, rel=0.0, abs=1.0e-15)


@pytest.mark.parametrize("face", ("left", "right", "bottom", "top"))
def test_repair_jacobian_matches_central_difference_away_from_support_kinks(
    face: str,
) -> None:
    legacy, _, _, _ = _problem(
        controls=((0.4, 0.2, 1.0, WheelSQPModeV2.FORWARD),)
    )
    problem = replace(legacy, repair=_repair(legacy, face=face))
    vector = problem.initial_vector
    actual = problem.inequality_jacobian(vector)[-3:]
    step = 1.0e-6

    for column in range(vector.size):
        lower = vector.copy()
        upper = vector.copy()
        lower[column] -= step
        upper[column] += step
        expected = (
            problem.inequalities(upper)[-3:] - problem.inequalities(lower)[-3:]
        ) / (2.0 * step)
        assert actual[:, column] == pytest.approx(
            expected,
            rel=3.0e-6,
            abs=3.0e-8,
        )


def test_later_segment_positive_rho_repair_jacobian_matches_all_columns() -> None:
    legacy, _, _, _ = _problem(
        controls=(
            (0.4, 0.2, 1.0, WheelSQPModeV2.FORWARD),
            (0.5, 0.1, 1.0, WheelSQPModeV2.FORWARD),
        )
    )
    problem = replace(
        legacy,
        repair=_repair(legacy, segment_index=1, rhos=(0.25, 0.5, 0.75)),
    )
    vector = problem.initial_vector
    actual = problem.repair_jacobian(vector)
    step = 1.0e-6

    for column in range(vector.size):
        lower = vector.copy()
        upper = vector.copy()
        lower[column] -= step
        upper[column] += step
        expected = (
            problem.repair_inequalities(upper)
            - problem.repair_inequalities(lower)
        ) / (2.0 * step)
        assert actual[:, column] == pytest.approx(
            expected,
            rel=3.0e-6,
            abs=3.0e-8,
        )


def test_first_segment_rho_zero_repair_jacobian_row_is_all_zero() -> None:
    legacy, _, _, _ = _problem(
        controls=((0.4, 0.2, 1.0, WheelSQPModeV2.FORWARD),)
    )
    problem = replace(
        legacy,
        repair=_repair(legacy, segment_index=0, rhos=(0.0, 0.25, 0.75)),
    )

    first_row = problem.repair_jacobian(problem.initial_vector)[0]

    assert np.array_equal(first_row, np.zeros(problem.layout.variable_count))
    assert first_row.tobytes() == bytes(first_row.nbytes)


def test_later_segment_rho_zero_repair_row_only_uses_previous_declared_state() -> None:
    legacy, _, _, _ = _problem(
        controls=(
            (0.4, 0.2, 1.0, WheelSQPModeV2.FORWARD),
            (0.5, 0.1, 1.0, WheelSQPModeV2.FORWARD),
        )
    )
    problem = replace(
        legacy,
        repair=_repair(legacy, segment_index=1, rhos=(0.0, 0.25, 0.75)),
    )
    vector = problem.initial_vector
    start = problem.layout.unpack(vector).states[0]
    half_length = PROFILE.body_length_m / 2.0 + PROFILE.footprint_safety_margin_m
    half_width = PROFILE.body_width_m / 2.0 + PROFILE.footprint_safety_margin_m
    support_theta = (
        -half_length * sin(start.heading_rad)
        + half_width * cos(start.heading_rad)
    )
    expected = np.zeros(vector.size, dtype=np.float64)
    expected[0] = -1.0
    expected[2] = -support_theta

    assert problem.inequality_jacobian(vector)[-3] == pytest.approx(
        expected,
        rel=0.0,
        abs=1.0e-15,
    )


def test_negative_repair_margin_is_infeasible_and_never_terminal_only() -> None:
    legacy, _, _, _ = _problem(goal=PoseStateV2(2.0, 0.0, 0.0))
    problem = replace(
        legacy,
        repair=_repair(
            legacy,
            face="left",
            rhos=(0.0, 0.5, 1.0),
            bounds=(0.0, 0.5, 0.0, 0.5),
        ),
    )

    audit = audit_wheel_sqp_candidate_v2(problem, problem.initial_vector)

    assert audit.passed is False
    assert audit.reason_code == "wheel_sqp_infeasible"
    assert audit.terminal_only_failure is False


def test_layout_has_six_physical_values_per_segment_and_excludes_fixed_start() -> None:
    start = PoseStateV2(0.1, 0.2, -0.3)
    q1 = PoseStateV2(1.0, 2.0, 3.0)
    q2 = PoseStateV2(4.0, 5.0, 6.0)
    layout = WheelSQPLayoutV1(segment_count=2, start_state=start)

    vector = layout.pack(
        states=(q1, q2),
        controls=((0.5, 0.0, 1.0), (0.0, pi / 4.0, 2.0)),
    )

    assert vector.dtype == np.float64
    assert vector.flags.c_contiguous
    assert vector.tolist() == [
        1.0, 2.0, 3.0, 0.5, 0.0, 1.0,
        4.0, 5.0, 6.0, 0.0, pi / 4.0, 2.0,
    ]
    assert layout.variable_count == 12
    assert layout.start_state is start
    unpacked = layout.unpack(vector)
    assert unpacked.states == (q1, q2)
    assert unpacked.controls == ((0.5, 0.0, 1.0), (0.0, pi / 4.0, 2.0))
    assert np.array_equal(layout.pack(unpacked.states, unpacked.controls), vector)
    assert layout.residual_scales.tolist() == [1.0, 1.0, pi, 1.0, 1.0, pi]


def test_layout_rejects_more_than_48_segments_before_any_array_allocation() -> None:
    start = PoseStateV2(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="48"):
        WheelSQPLayoutV1(segment_count=49, start_state=start)
    with pytest.raises(ValueError, match="48"):
        WheelSQPLayoutV1(segment_count=10**100, start_state=start)


def test_audit_accepts_exact_terminal_boundary_and_rejects_one_ulp_outside() -> None:
    base, _, _, _ = _problem(goal=PoseStateV2(1.25, 0.0, 0.0))
    vector = base.initial_vector
    assert audit_wheel_sqp_candidate_v2(base, vector).passed

    outside, _, _, _ = _problem(
        goal=PoseStateV2(nextafter(1.25, inf), 0.0, 0.0)
    )
    audit = audit_wheel_sqp_candidate_v2(outside, outside.initial_vector)
    assert not audit.passed
    assert audit.reason_code == "wheel_sqp_infeasible"


def test_heading_terminal_boundary_preserves_one_ulp_and_wrap_semantics() -> None:
    boundary, _, _, _ = _problem(
        controls=(
            (
                0.0,
                PROFILE.profile.goal_heading_tolerance_rad,
                1.0,
                WheelSQPModeV2.TURN_LEFT,
            ),
        ),
        goal=PoseStateV2(0.0, 0.0, 0.0),
    )
    assert audit_wheel_sqp_candidate_v2(boundary, boundary.initial_vector).passed

    outside = boundary.initial_vector
    outside_omega = nextafter(PROFILE.profile.goal_heading_tolerance_rad, inf)
    replay = integrate_wheel_segment_v2(
        boundary.request.start_state,
        0.0,
        outside_omega,
        1.0,
    )
    outside[:3] = (replay.x_m, replay.y_m, replay.heading_rad)
    outside[4] = outside_omega
    audit = audit_wheel_sqp_candidate_v2(boundary, outside)
    assert not audit.passed
    assert audit.reason_code == "wheel_sqp_infeasible"

    wrapped, _, _, _ = _problem(
        controls=(
            (
                0.0,
                PROFILE.profile.goal_heading_tolerance_rad,
                1.0,
                WheelSQPModeV2.TURN_LEFT,
            ),
        ),
        goal=PoseStateV2(0.0, 0.0, 2.0 * pi),
    )
    assert audit_wheel_sqp_candidate_v2(wrapped, wrapped.initial_vector).passed


def test_dynamics_and_objective_jacobians_match_central_difference() -> None:
    problem, _, _, _ = _problem(
        controls=((0.6, 0.2, 1.0, WheelSQPModeV2.FORWARD),)
    )
    vector = problem.initial_vector
    dynamics_jacobian = problem.dynamics_jacobian(vector)
    objective_jacobian = problem.objective_jacobian(vector)
    step = 1.0e-6
    for column in range(vector.size):
        lower = vector.copy()
        upper = vector.copy()
        lower[column] -= step
        upper[column] += step
        expected_dynamics = (
            problem.dynamics_residual(upper) - problem.dynamics_residual(lower)
        ) / (2.0 * step)
        expected_objective = (
            problem.objective(upper) - problem.objective(lower)
        ) / (2.0 * step)
        assert dynamics_jacobian[:, column] == pytest.approx(
            expected_dynamics, rel=2.0e-6, abs=2.0e-8
        )
        assert objective_jacobian[column] == pytest.approx(
            expected_objective, rel=3.0e-6, abs=3.0e-8
        )


def test_relative_energy_jacobian_is_the_single_normalized_energy_authority() -> None:
    actual = wheel_relative_energy_jacobian_v1(-0.5, 0.2, 2.0, PROFILE)
    assert actual == pytest.approx((-2.5, 0.4, 0.715), abs=1.0e-15)


def test_problem_rejects_adjacent_translation_sign_flip_without_stop() -> None:
    with pytest.raises(ValueError, match="dedicated STOP"):
        _problem(
            controls=(
                (0.5, 0.0, 1.0, WheelSQPModeV2.FORWARD),
                (-0.5, 0.0, 1.0, WheelSQPModeV2.REVERSE),
            )
        )


def test_problem_rejects_translation_sign_flip_across_turns_without_stop() -> None:
    with pytest.raises(ValueError, match="dedicated STOP"):
        _problem(
            controls=(
                (0.5, 0.0, 1.0, WheelSQPModeV2.FORWARD),
                (0.0, pi / 4.0, 1.0, WheelSQPModeV2.TURN_LEFT),
                (-0.5, 0.0, 1.0, WheelSQPModeV2.REVERSE),
            )
        )


def test_problem_reseals_initial_guess_hash_before_solver_identity() -> None:
    problem, _, _, _ = _problem()
    arbitrary_hash = replace(
        problem.initial_guess,
        initial_guess_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="initial guess hash"):
        replace(problem, initial_guess=arbitrary_hash)

    changed_segment = replace(
        problem.initial_guess.segments[0],
        relative_energy=problem.initial_guess.segments[0].relative_energy + 1.0,
    )
    stale_hash = replace(
        problem.initial_guess,
        segments=(changed_segment,),
    )
    with pytest.raises(ValueError, match="initial guess hash"):
        replace(problem, initial_guess=stale_hash)


def test_risk_is_rejected_before_lazy_backend_load(monkeypatch: pytest.MonkeyPatch) -> None:
    problem, deadline, ledger, _ = _problem(
        objective=ObjectiveProfileV2(risk_weight=1.0)
    )

    def forbidden_loader() -> object:
        raise AssertionError("backend loader must not run")

    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", forbidden_loader)
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_objective_unsupported"
    assert result.candidate is None


def test_missing_backend_is_typed_and_has_no_partial_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: None)
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_backend_unavailable"
    assert result.candidate is None


@pytest.mark.parametrize("loader_result", ("missing", "throw"))
def test_deadline_expiry_during_backend_load_wins_over_backend_reason(
    monkeypatch: pytest.MonkeyPatch,
    loader_result: str,
) -> None:
    problem, deadline, ledger, clock = _problem()

    def slow_loader() -> object | None:
        clock.now = 5.0
        if loader_result == "throw":
            raise RuntimeError("late loader failure")
        return None

    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", slow_loader)
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "planning_deadline_expired"
    assert result.candidate is None


def test_raw_success_false_does_not_veto_independently_feasible_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
    monkeypatch.setattr(
        solver,
        "_run_slsqp_v1",
        lambda **kwargs: SimpleNamespace(
            x=problem.initial_vector.copy(), nit=2, success=False, status=9, message="ignored"
        ),
    )

    result = solve_wheel_sqp_v2(problem, deadline, ledger)

    assert result.status is WheelSQPStatusV2.FEASIBLE
    assert result.reason_code is None
    assert result.candidate is not None
    assert not isinstance(result.candidate, RoutePrimitiveV2)
    assert not hasattr(result.candidate, "validation_level")
    assert all(not isinstance(segment, RoutePrimitiveV2) for segment in result.candidate.segments)


def test_raw_success_true_cannot_bypass_independent_dynamics_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    bad = problem.initial_vector.copy()
    bad[0] += 0.01
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
    monkeypatch.setattr(
        solver,
        "_run_slsqp_v1",
        lambda **kwargs: SimpleNamespace(
            x=bad, nit=1, success=True, status=0, message="fake success"
        ),
    )

    result = solve_wheel_sqp_v2(problem, deadline, ledger)

    assert result.reason_code == "wheel_sqp_numeric_contract_failed"
    assert result.candidate is None


def test_terminal_infeasible_raw_uses_only_the_independently_audited_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    raw = problem.layout.pack(
        states=(integrate_wheel_segment_v2(problem.request.start_state, 0.7, 0.0, 1.0),),
        controls=((0.7, 0.0, 1.0),),
    )
    assert audit_wheel_sqp_candidate_v2(problem, raw).reason_code == "wheel_sqp_infeasible"
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
    monkeypatch.setattr(
        solver,
        "_run_slsqp_v1",
        lambda **kwargs: SimpleNamespace(x=raw, success=True, nit=1),
    )
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.candidate is not None
    incumbent_vector = problem.layout.pack(
        tuple(segment.end_state for segment in result.candidate.segments),
        tuple(
            (segment.v_mps, segment.omega_radps, segment.duration_s)
            for segment in result.candidate.segments
        ),
    )
    assert audit_wheel_sqp_candidate_v2(problem, incumbent_vector).passed
    assert np.array_equal(incumbent_vector, problem.initial_vector)


def test_late_backend_return_is_deadline_expired_even_with_feasible_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, clock = _problem()
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())

    def late_return(**kwargs: object) -> SimpleNamespace:
        clock.now = 5.0
        return SimpleNamespace(x=problem.initial_vector.copy(), nit=1, success=True)

    monkeypatch.setattr(solver, "_run_slsqp_v1", late_return)
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "planning_deadline_expired"
    assert result.candidate is None


def test_unexpected_backend_exception_is_typed_but_memory_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
    monkeypatch.setattr(
        solver, "_run_slsqp_v1", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_internal_error"
    assert result.candidate is None

    problem, deadline, ledger, _ = _problem()
    monkeypatch.setattr(
        solver, "_run_slsqp_v1", lambda **kwargs: (_ for _ in ()).throw(MemoryError())
    )
    with pytest.raises(MemoryError):
        solve_wheel_sqp_v2(problem, deadline, ledger)


def test_heading_change_and_slew_use_exact_closed_boundaries() -> None:
    turn_problem, _, _, _ = _problem(
        controls=((0.0, pi / 4.0, 2.0, WheelSQPModeV2.TURN_LEFT),)
    )
    assert audit_wheel_sqp_candidate_v2(
        turn_problem, turn_problem.initial_vector
    ).passed
    turn_outside = turn_problem.initial_vector
    turn_outside[4] = nextafter(pi / 4.0, inf)
    replay = integrate_wheel_segment_v2(
        turn_problem.request.start_state,
        float(turn_outside[3]),
        float(turn_outside[4]),
        float(turn_outside[5]),
    )
    turn_outside[:3] = (replay.x_m, replay.y_m, replay.heading_rad)
    assert not audit_wheel_sqp_candidate_v2(turn_problem, turn_outside).passed

    slew_problem, _, _, _ = _problem(
        controls=(
            (0.5, 0.0, 1.0, WheelSQPModeV2.FORWARD),
            (1.0, 0.0, 1.0, WheelSQPModeV2.FORWARD),
        )
    )
    assert audit_wheel_sqp_candidate_v2(
        slew_problem, slew_problem.initial_vector
    ).passed
    slew_outside = slew_problem.initial_vector
    shorter = nextafter(1.0, 0.0)
    slew_outside[5] = shorter
    slew_outside[11] = shorter
    first = integrate_wheel_segment_v2(
        slew_problem.request.start_state,
        float(slew_outside[3]),
        float(slew_outside[4]),
        shorter,
    )
    second = integrate_wheel_segment_v2(
        first,
        float(slew_outside[9]),
        float(slew_outside[10]),
        shorter,
    )
    slew_outside[:3] = (first.x_m, first.y_m, first.heading_rad)
    slew_outside[6:9] = (second.x_m, second.y_m, second.heading_rad)
    assert not audit_wheel_sqp_candidate_v2(slew_problem, slew_outside).passed


def test_inequality_jacobian_matches_central_difference_away_from_kinks() -> None:
    problem, _, _, _ = _problem(
        controls=(
            (0.4, 0.1, 1.0, WheelSQPModeV2.FORWARD),
            (0.6, 0.2, 1.0, WheelSQPModeV2.FORWARD),
        )
    )
    vector = problem.initial_vector
    actual = problem.inequality_jacobian(vector)
    step = 1.0e-6
    for column in range(vector.size):
        lower = vector.copy()
        upper = vector.copy()
        lower[column] -= step
        upper[column] += step
        expected = (
            problem.inequalities(upper) - problem.inequalities(lower)
        ) / (2.0 * step)
        assert actual[:, column] == pytest.approx(
            expected,
            rel=3.0e-6,
            abs=3.0e-8,
        )


def test_function_evaluation_cap_aborts_inside_backend_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())

    def exhaust(**kwargs: object) -> None:
        objective_fn = kwargs["objective_fn"]
        vector = problem.initial_vector
        for _ in range(problem.profile.max_sqp_function_evaluations + 1):
            objective_fn(vector)

    monkeypatch.setattr(solver, "_run_slsqp_v1", exhaust)
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert result.function_evaluation_count == 4096
    assert result.candidate is None


def test_strict_reserve_rejects_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, clock = _problem()
    clock.now = 4.9

    def forbidden_loader() -> object:
        raise AssertionError("backend must not load without the strict tail reserve")

    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", forbidden_loader)
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert result.candidate is None


def test_nonfinite_raw_vector_is_numeric_failure_without_partial_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    bad = problem.initial_vector
    bad[0] = np.nan
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
    monkeypatch.setattr(
        solver,
        "_run_slsqp_v1",
        lambda **kwargs: SimpleNamespace(x=bad, nit=1, success=True),
    )
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_numeric_contract_failed"
    assert result.candidate is None


def test_raw_status_does_not_enter_solver_audit_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashes: list[str] = []
    for success, status, message in (
        (True, 0, "ok"),
        (False, 9, "iteration limit"),
    ):
        problem, deadline, ledger, _ = _problem()
        monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
        monkeypatch.setattr(
            solver,
            "_run_slsqp_v1",
            lambda **kwargs: SimpleNamespace(
                x=problem.initial_vector,
                nit=1,
                success=success,
                status=status,
                message=message,
            ),
        )
        result = solve_wheel_sqp_v2(problem, deadline, ledger)
        assert result.candidate is not None
        hashes.append(result.candidate.candidate_hash)
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("raw_nit", (-1, 0, 41, 10**100, None))
def test_raw_iteration_fields_have_no_decision_or_telemetry_authority(
    monkeypatch: pytest.MonkeyPatch,
    raw_nit: object,
) -> None:
    problem, deadline, ledger, _ = _problem()
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
    monkeypatch.setattr(
        solver,
        "_run_slsqp_v1",
        lambda **kwargs: SimpleNamespace(
            x=problem.initial_vector,
            nit=raw_nit,
            nfev=10**100,
            njev=-1,
            success=True,
        ),
    )
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.status is WheelSQPStatusV2.FEASIBLE
    assert result.iteration_count == 0
    assert result.candidate is not None


def test_callback_iteration_cap_is_the_only_iteration_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())

    def exhaust(**kwargs: object) -> None:
        callback = kwargs["callback"]
        vector = problem.initial_vector
        for _ in range(problem.profile.max_sqp_iterations + 1):
            callback(vector)

    monkeypatch.setattr(solver, "_run_slsqp_v1", exhaust)
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert result.iteration_count == 40
    assert result.candidate is None


def test_backend_receives_exact_unbuffered_terminal_margins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem(goal=PoseStateV2(1.25, 0.0, 0.0))
    captured: list[np.ndarray] = []
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())

    def inspect(**kwargs: object) -> SimpleNamespace:
        captured.append(kwargs["inequality_fn"](problem.initial_vector))
        return SimpleNamespace(x=problem.initial_vector, success=True, nit=999)

    monkeypatch.setattr(solver, "_run_slsqp_v1", inspect)
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.candidate is not None
    assert captured[0][0] == 0.0
    assert captured[0][1] > 0.0


def test_private_solver_candidate_materializes_only_into_task2_canonical_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, deadline, ledger, _ = _problem()
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: object())
    monkeypatch.setattr(
        solver,
        "_run_slsqp_v1",
        lambda **kwargs: SimpleNamespace(
            x=problem.initial_vector,
            nit=1,
            success=True,
        ),
    )
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.candidate is not None
    canonical = materialize_canonical_wheel_candidate_v2(
        result.candidate,
        request=problem.request,
        profile=problem.profile,
        terrain_snapshot_hash=snapshot_hash(problem.request.terrain_snapshot),
    )
    assert type(canonical) is CanonicalWheelCandidateV1
    assert canonical.candidate_hash != result.candidate.candidate_hash


def test_real_scipy_backend_returns_an_independently_audited_candidate() -> None:
    assert solver._load_scipy_optimize_v1() is not None
    problem, deadline, ledger, _ = _problem()
    result = solve_wheel_sqp_v2(problem, deadline, ledger)
    assert result.status is WheelSQPStatusV2.FEASIBLE
    assert result.reason_code is None
    assert result.candidate is not None
    vector = problem.layout.pack(
        tuple(segment.end_state for segment in result.candidate.segments),
        tuple(
            (segment.v_mps, segment.omega_radps, segment.duration_s)
            for segment in result.candidate.segments
        ),
    )
    assert audit_wheel_sqp_candidate_v2(problem, vector).passed
