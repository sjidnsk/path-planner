from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import inf, nextafter, pi
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
from path_planner.v2.wheel_corridors import wheel_corridor_path_hash_v1
from path_planner.v2.wheel_kinematics import (
    integrate_wheel_segment_v2,
    wheel_relative_energy_jacobian_v1,
)
from path_planner.v2.wheel_sqp_contracts import (
    WheelCorridorV2,
    WheelSQPInitialGuessV2,
    WheelSQPModeV2,
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
    geometry = FineGridGeometryV2(5, 3)
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
    )
    clock = _Clock()
    deadline = PlanningDeadlineV2(0.0, 5.0, clock)
    ledger = WheelSQPWorkLedgerV1(budget, deadline)
    return problem, deadline, ledger, clock


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
