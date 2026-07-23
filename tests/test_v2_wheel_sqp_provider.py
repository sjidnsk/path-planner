from dataclasses import FrozenInstanceError, replace
import inspect
from math import pi
from types import SimpleNamespace

import numpy as np
import pytest

import path_planner.v2.wheel_sqp_contracts as wheel_sqp_contracts_module
import path_planner.v2.wheel_sqp_validation as wheel_sqp_validation_module
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
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.wheel_sqp_contracts import (
    WheelL2CounterexampleV2,
    WheelSQPWorkLedgerV1,
)
from path_planner.v2.wheel_sqp_serialization import (
    encode_wheel_candidate_v2,
    materialize_canonical_wheel_candidate_v2,
    wheel_candidate_hash_v2,
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
GEOMETRY = FineGridGeometryV2(6, 5, origin=(13.25, -7.75))
CELL = Cell(2, 3)


def _snapshot() -> TerrainSnapshotV2:
    return TerrainSnapshotV2(
        geometry=GEOMETRY,
        elevation_m=np.zeros(GEOMETRY.shape),
        slope_deg=np.zeros(GEOMETRY.shape),
        traversable_mask=np.ones(GEOMETRY.shape, dtype=np.bool_),
        hard_obstacle_mask=np.zeros(GEOMETRY.shape, dtype=np.bool_),
        observed_mask=np.ones(GEOMETRY.shape, dtype=np.bool_),
        confidence=np.ones(GEOMETRY.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task8-repair-fixture",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _stationary_authorities(
    snapshot: TerrainSnapshotV2,
) -> tuple[FineSafetyAnchorV2, PlanningDeadlineV2, WheelSQPWorkLedgerV1]:
    deadline = PlanningDeadlineV2(0.0, 2.0, _Clock())
    ledger = WheelSQPWorkLedgerV1(ResourceBudgetV2(), deadline)
    return FineSafetyAnchorV2(snapshot), deadline, ledger


def _provider_request(
    snapshot: TerrainSnapshotV2,
    *,
    start: PoseStateV2,
    goal: PoseStateV2,
    objective: ObjectiveProfileV2 | None = None,
    accelerator_policy: AcceleratorPolicyV2 = AcceleratorPolicyV2.DISABLED,
) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="wheel-sqp-task8-provider",
        platform_profile_id=PROFILE.profile.profile_id,
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
        objective_profile=objective or ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(),
        timeout_s=5.0,
        accelerator_policy=accelerator_policy,
        determinism_seed=47,
    )


def _snapshot_with_invalid_cells(
    *,
    hard: tuple[Cell, ...] = (),
    unknown: tuple[Cell, ...] = (),
) -> TerrainSnapshotV2:
    hard_mask = np.zeros(GEOMETRY.shape, dtype=np.bool_)
    traversable_mask = np.ones(GEOMETRY.shape, dtype=np.bool_)
    observed_mask = np.ones(GEOMETRY.shape, dtype=np.bool_)
    for cell in hard:
        hard_mask[cell.y, cell.x] = True
        traversable_mask[cell.y, cell.x] = False
    for cell in unknown:
        observed_mask[cell.y, cell.x] = False
    return TerrainSnapshotV2(
        geometry=GEOMETRY,
        elevation_m=np.zeros(GEOMETRY.shape),
        slope_deg=np.zeros(GEOMETRY.shape),
        traversable_mask=traversable_mask,
        hard_obstacle_mask=hard_mask,
        observed_mask=observed_mask,
        confidence=np.ones(GEOMETRY.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task8-stationary-fixture",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )


def test_stationary_full_rectangle_accepts_all_safe_terrain() -> None:
    snapshot = _snapshot()
    anchor, deadline, ledger = _stationary_authorities(snapshot)
    center = GEOMETRY.cell_center(CELL)

    reason = wheel_sqp_validation_module.validate_wheel_stationary_footprint_v2(
        PoseStateV2(center.x, center.y, 0.0),
        anchor,
        PROFILE,
        deadline,
        ledger,
    )

    assert reason is None


def test_stationary_full_rectangle_rejects_neighbor_obstacle_while_center_is_safe() -> None:
    snapshot = _snapshot_with_invalid_cells(hard=(Cell(CELL.x + 1, CELL.y),))
    anchor, deadline, ledger = _stationary_authorities(snapshot)
    center = GEOMETRY.cell_center(CELL)

    reason = wheel_sqp_validation_module.validate_wheel_stationary_footprint_v2(
        PoseStateV2(center.x, center.y, 0.0),
        anchor,
        PROFILE,
        deadline,
        ledger,
    )

    assert reason == "terrain_hard_obstacle"


def test_stationary_map_boundary_contact_is_out_of_bounds() -> None:
    snapshot = _snapshot()
    anchor, deadline, ledger = _stationary_authorities(snapshot)
    half_length = PROFILE.body_length_m / 2.0
    pose = PoseStateV2(
        GEOMETRY.origin[0] + half_length,
        GEOMETRY.cell_center(CELL).y,
        0.0,
    )

    reason = wheel_sqp_validation_module.validate_wheel_stationary_footprint_v2(
        pose,
        anchor,
        PROFILE,
        deadline,
        ledger,
    )

    assert reason == "terrain_out_of_bounds"


def test_stationary_invalid_reason_priority_matches_l2() -> None:
    snapshot = _snapshot_with_invalid_cells(
        hard=(Cell(CELL.x + 1, CELL.y),),
        unknown=(Cell(CELL.x - 1, CELL.y),),
    )
    anchor, deadline, ledger = _stationary_authorities(snapshot)
    center = GEOMETRY.cell_center(CELL)

    reason = wheel_sqp_validation_module.validate_wheel_stationary_footprint_v2(
        PoseStateV2(center.x, center.y, 0.0),
        anchor,
        PROFILE,
        deadline,
        ledger,
    )

    assert reason == "terrain_unknown"


def _candidate_bytes(*, offset_x_m: float = 0.0, offset_y_m: float = 0.0):
    center = GEOMETRY.cell_center(CELL)
    pose = PoseStateV2(
        center.x + offset_x_m,
        center.y + offset_y_m,
        pi / 4.0,
    )
    request = PlanningRequestV2(
        request_id="wheel-sqp-task8-repair",
        platform_profile_id=PROFILE.profile.profile_id,
        start_state=pose,
        goal_state=pose,
        terrain_snapshot=_snapshot(),
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(),
        timeout_s=2.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=41,
    )
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=pose,
                    end_state=pose,
                    v_mps=0.0,
                    omega_radps=0.0,
                    duration_s=0.5,
                ),
            )
        ),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash="d" * 64,
    )
    return candidate, encode_wheel_candidate_v2(candidate)


def _counterexample(
    *,
    repairable: bool = True,
    cell: Cell = CELL,
    terrain_reason: str = "terrain_hard_obstacle",
    offset_x_m: float = 0.0,
    offset_y_m: float = 0.0,
):
    candidate, candidate_bytes = _candidate_bytes(
        offset_x_m=offset_x_m,
        offset_y_m=offset_y_m,
    )
    return (
        WheelL2CounterexampleV2(
            segment_index=0,
            candidate_segment_hash=candidate.segments[0].segment_hash,
            time_fraction_lo=0.0,
            time_fraction_mid=0.5,
            time_fraction_hi=1.0,
            cell=cell,
            terrain_reason=terrain_reason,
            snapshot_hash=candidate.terrain_snapshot_hash,
            candidate_hash=candidate.candidate_hash,
            repairable=repairable,
        ),
        candidate,
        candidate_bytes,
    )


def test_known_center_collision_chooses_left_on_exact_four_way_tie() -> None:
    counterexample, candidate, candidate_bytes = _counterexample()

    repair = wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
        counterexample,
        candidate_bytes,
        PROFILE,
        geometry=GEOMETRY,
    )

    assert repair.source_candidate_hash == candidate.candidate_hash
    assert repair.source_segment_hash == candidate.segments[0].segment_hash
    assert repair.source_snapshot_hash == candidate.terrain_snapshot_hash
    assert repair.segment_index == 0
    assert repair.cell == CELL
    assert repair.face == "left"
    assert repair.time_fractions == (0.0, 0.5, 1.0)
    assert repair.clearance_m == 1.0e-4


def test_repair_freezes_nonzero_origin_world_cell_bounds_and_is_immutable() -> None:
    counterexample, _, candidate_bytes = _counterexample()

    repair = wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
        counterexample,
        candidate_bytes,
        PROFILE,
        geometry=GEOMETRY,
    )

    assert repair.cell_bounds_m == (14.25, 14.75, -6.25, -5.75)
    with pytest.raises(FrozenInstanceError):
        repair.face = "right"


def test_repair_accepts_nonbinary_origin_cell_extent_from_factory() -> None:
    geometry = FineGridGeometryV2(4, 4, origin=(0.1, 0.1))
    cell = Cell(1, 1)
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape),
        slope_deg=np.zeros(geometry.shape),
        traversable_mask=np.ones(geometry.shape, dtype=np.bool_),
        hard_obstacle_mask=np.zeros(geometry.shape, dtype=np.bool_),
        observed_mask=np.ones(geometry.shape, dtype=np.bool_),
        confidence=np.ones(geometry.shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-sqp-task8-nonbinary-origin",
            source_hash="fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    center = geometry.cell_center(cell)
    pose = PoseStateV2(center.x, center.y, 0.0)
    request = _provider_request(snapshot, start=pose, goal=pose)
    candidate = materialize_canonical_wheel_candidate_v2(
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start_state=pose,
                    end_state=pose,
                    v_mps=0.0,
                    omega_radps=0.0,
                    duration_s=0.5,
                ),
            )
        ),
        request=request,
        profile=PROFILE,
        terrain_snapshot_hash="d" * 64,
    )
    counterexample = WheelL2CounterexampleV2(
        segment_index=0,
        candidate_segment_hash=candidate.segments[0].segment_hash,
        time_fraction_lo=0.0,
        time_fraction_mid=0.5,
        time_fraction_hi=1.0,
        cell=cell,
        terrain_reason="terrain_hard_obstacle",
        snapshot_hash=candidate.terrain_snapshot_hash,
        candidate_hash=candidate.candidate_hash,
        repairable=True,
    )

    repair = wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
        counterexample,
        encode_wheel_candidate_v2(candidate),
        PROFILE,
        geometry=geometry,
    )

    assert repair.cell_bounds_m == (0.6, 1.1, 0.6, 1.1)


def test_repair_contract_rejects_non_fine_grid_cell_extent() -> None:
    counterexample, _, candidate_bytes = _counterexample()
    repair = wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
        counterexample,
        candidate_bytes,
        PROFILE,
        geometry=GEOMETRY,
    )

    with pytest.raises(ValueError, match="fine-grid resolution"):
        replace(
            repair,
            cell_right_x_m=repair.cell_right_x_m + GEOMETRY.resolution_m,
        )


@pytest.mark.parametrize(
    ("offset_x_m", "offset_y_m", "expected_face"),
    (
        (-0.1, 0.0, "left"),
        (0.1, 0.0, "right"),
        (0.0, -0.1, "bottom"),
        (0.0, 0.1, "top"),
    ),
)
def test_repair_selects_maximum_oriented_support_margin(
    offset_x_m: float,
    offset_y_m: float,
    expected_face: str,
) -> None:
    counterexample, _, candidate_bytes = _counterexample(
        offset_x_m=offset_x_m,
        offset_y_m=offset_y_m,
    )

    repair = wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
        counterexample,
        candidate_bytes,
        PROFILE,
        geometry=GEOMETRY,
    )

    assert repair.face == expected_face


def test_nonrepairable_counterexample_cannot_create_repair_authority() -> None:
    counterexample, _, candidate_bytes = _counterexample(repairable=False)

    with pytest.raises(wheel_sqp_contracts_module.WheelSQPRepairNotAllowed):
        wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
            counterexample,
            candidate_bytes,
            PROFILE,
            geometry=GEOMETRY,
        )


@pytest.mark.parametrize(
    "terrain_reason",
    ("terrain_unknown",),
)
def test_unknown_terrain_cannot_create_repair_authority(terrain_reason: str) -> None:
    counterexample, _, candidate_bytes = _counterexample(
        terrain_reason=terrain_reason,
    )

    with pytest.raises(wheel_sqp_contracts_module.WheelSQPRepairNotAllowed):
        wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
            counterexample,
            candidate_bytes,
            PROFILE,
            geometry=GEOMETRY,
        )


def test_out_of_bounds_cell_cannot_create_repair_authority() -> None:
    counterexample, _, candidate_bytes = _counterexample(cell=Cell(-1, 3))

    with pytest.raises(wheel_sqp_contracts_module.WheelSQPRepairNotAllowed):
        wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
            counterexample,
            candidate_bytes,
            PROFILE,
            geometry=GEOMETRY,
        )


def test_in_bounds_noncolliding_cell_cannot_create_repair_authority() -> None:
    counterexample, _, candidate_bytes = _counterexample(cell=Cell(5, 0))

    with pytest.raises(wheel_sqp_contracts_module.WheelSQPRepairNotAllowed):
        wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
            counterexample,
            candidate_bytes,
            PROFILE,
            geometry=GEOMETRY,
        )


@pytest.mark.parametrize(
    "mutation",
    ("candidate_hash", "snapshot_hash", "segment_hash", "segment_index"),
)
def test_repair_binds_all_counterexample_lineage(mutation: str) -> None:
    counterexample, _, candidate_bytes = _counterexample()
    changes = {
        "candidate_hash": {"candidate_hash": "a" * 64},
        "snapshot_hash": {"snapshot_hash": "a" * 64},
        "segment_hash": {"candidate_segment_hash": "a" * 64},
        "segment_index": {"segment_index": 1},
    }
    forged = replace(counterexample, **changes[mutation])

    with pytest.raises(wheel_sqp_contracts_module.WheelSQPRepairNotAllowed):
        wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
            forged,
            candidate_bytes,
            PROFILE,
            geometry=GEOMETRY,
        )


def test_repair_strictly_decodes_canonical_candidate_bytes() -> None:
    counterexample, _, candidate_bytes = _counterexample()

    with pytest.raises(wheel_sqp_contracts_module.WheelSQPRepairNotAllowed):
        wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
            counterexample,
            candidate_bytes + b"\n",
            PROFILE,
            geometry=GEOMETRY,
        )


def test_repair_binds_candidate_profile_hash() -> None:
    counterexample, candidate, _ = _counterexample()
    forged = replace(candidate, profile_hash="a" * 64, candidate_hash="0" * 64)
    forged = replace(forged, candidate_hash=wheel_candidate_hash_v2(forged))
    forged_bytes = encode_wheel_candidate_v2(forged)
    forged_counterexample = replace(
        counterexample,
        candidate_hash=forged.candidate_hash,
    )

    with pytest.raises(wheel_sqp_contracts_module.WheelSQPRepairNotAllowed):
        wheel_sqp_validation_module.repair_constraint_from_counterexample_v2(
            forged_counterexample,
            forged_bytes,
            PROFILE,
            geometry=GEOMETRY,
        )


def test_provider_is_frozen_and_stores_only_the_exact_profile() -> None:
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2

    provider = WheelKinematicSQPProviderV2(PROFILE)

    assert provider.wheel_sqp_profile is PROFILE
    assert provider.profile is PROFILE.profile
    with pytest.raises(FrozenInstanceError):
        provider.wheel_sqp_profile = PROFILE
    with pytest.raises(TypeError):
        WheelKinematicSQPProviderV2(object())


def test_provider_source_contains_no_old_wheel_or_hybrid_fallback() -> None:
    import path_planner.v2.providers.wheel_sqp as provider_module

    source = inspect.getsource(provider_module)
    forbidden = "Hybrid" + "AStarPlanner"
    old_provider = "providers" + ".wheel"
    assert forbidden not in source
    assert old_provider not in source


def test_exact_hold_runs_stop_sqp_and_l2_without_corridor_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import path_planner.v2.providers.wheel_sqp as provider_module
    from path_planner.v2.contracts import PlanningSuccessV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2
    from path_planner.v2.wheel_sqp_contracts import WheelKinematicRouteV2

    snapshot = _snapshot()
    center = GEOMETRY.cell_center(CELL)
    pose = PoseStateV2(center.x, center.y, 0.0)
    request = PlanningRequestV2(
        request_id="wheel-sqp-task8-exact-hold",
        platform_profile_id=PROFILE.profile.profile_id,
        start_state=pose,
        goal_state=pose,
        terrain_snapshot=snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(),
        timeout_s=5.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=43,
    )
    deadline = PlanningDeadlineV2(0.0, 5.0, _Clock())

    def forbidden_generator(*args, **kwargs):
        raise AssertionError("exact hold must not call corridor generation")

    monkeypatch.setattr(
        provider_module,
        "generate_wheel_corridors_v2",
        forbidden_generator,
    )

    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        deadline,
    )

    assert type(outcome) is PlanningSuccessV2
    assert type(outcome.route) is WheelKinematicRouteV2
    assert len(outcome.route.primitives) == 1
    stop = outcome.route.primitives[0]
    assert (stop.v_mps, stop.omega_radps, stop.duration_s) == (0.0, 0.0, 0.05)
    assert stop.reverse is False
    assert stop.turn_in_place is False
    assert outcome.search_telemetry.corridor_count == 0
    assert outcome.search_telemetry.selected_corridor_index is None
    assert outcome.search_telemetry.repair_attempted is False
    assert outcome.validation_evidence.repair_applied is False


def test_provider_preflight_maps_backend_objective_and_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import path_planner.v2.providers.wheel_sqp as provider_module
    from path_planner.v2.contracts import FailureCategoryV2, PlanningFailureV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2

    snapshot = _snapshot()
    center = GEOMETRY.cell_center(CELL)
    pose = PoseStateV2(center.x, center.y, 0.0)
    provider = WheelKinematicSQPProviderV2(PROFILE)
    anchor = FineSafetyAnchorV2(snapshot)

    monkeypatch.setattr(
        provider_module.wheel_sqp_solver,
        "_load_scipy_optimize_v1",
        lambda: None,
    )
    missing = provider.plan(
        _provider_request(snapshot, start=pose, goal=pose),
        anchor,
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )
    assert type(missing) is PlanningFailureV2
    assert missing.reason_code == "wheel_sqp_backend_unavailable"
    assert missing.category is FailureCategoryV2.UNSUPPORTED_CAPABILITY
    assert not hasattr(missing, "route")

    monkeypatch.setattr(
        provider_module.wheel_sqp_solver,
        "_load_scipy_optimize_v1",
        lambda: object(),
    )
    unsupported_objective = provider.plan(
        _provider_request(
            snapshot,
            start=pose,
            goal=pose,
            objective=ObjectiveProfileV2(
                distance_weight=0.0,
                risk_weight=1.0,
                energy_weight=0.0,
                time_weight=0.0,
            ),
        ),
        anchor,
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )
    assert unsupported_objective.reason_code == "wheel_sqp_objective_unsupported"
    assert unsupported_objective.category is FailureCategoryV2.UNSUPPORTED_CAPABILITY

    required = provider.plan(
        _provider_request(
            snapshot,
            start=pose,
            goal=pose,
            accelerator_policy=AcceleratorPolicyV2.REQUIRED,
        ),
        anchor,
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )
    assert required.reason_code == "wheel_sqp_accelerator_required_unsupported"
    assert required.category is FailureCategoryV2.UNSUPPORTED_CAPABILITY


def test_provider_maps_additional_full_footprint_start_and_goal_failures() -> None:
    from path_planner.v2.contracts import FailureCategoryV2, PlanningFailureV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2

    obstacle = Cell(CELL.x + 1, CELL.y)
    snapshot = _snapshot_with_invalid_cells(hard=(obstacle,))
    goal_center = GEOMETRY.cell_center(CELL)
    goal = PoseStateV2(goal_center.x, goal_center.y, 0.0)
    safe_cell = Cell(1, 1)
    safe_center = GEOMETRY.cell_center(safe_cell)
    safe = PoseStateV2(safe_center.x, safe_center.y, 0.0)
    provider = WheelKinematicSQPProviderV2(PROFILE)
    anchor = FineSafetyAnchorV2(snapshot)

    unsafe_start = provider.plan(
        _provider_request(snapshot, start=goal, goal=goal),
        anchor,
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )
    assert type(unsafe_start) is PlanningFailureV2
    assert unsafe_start.reason_code == "wheel_sqp_start_invalid"
    assert unsafe_start.category is FailureCategoryV2.UNSAFE_START
    assert unsafe_start.evidence.details == (
        ("terrain_reason", "terrain_hard_obstacle"),
    )

    unsafe_goal = provider.plan(
        _provider_request(snapshot, start=safe, goal=goal),
        anchor,
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )
    assert type(unsafe_goal) is PlanningFailureV2
    assert unsafe_goal.reason_code == "wheel_sqp_goal_invalid"
    assert unsafe_goal.category is FailureCategoryV2.UNSAFE_GOAL


def test_provider_snapshot_identity_and_initial_deadline_have_stable_precedence() -> None:
    from path_planner.v2.contracts import FailureCategoryV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2

    snapshot = _snapshot()
    center = GEOMETRY.cell_center(CELL)
    pose = PoseStateV2(center.x, center.y, 0.0)
    request = _provider_request(snapshot, start=pose, goal=pose)
    provider = WheelKinematicSQPProviderV2(PROFILE)

    mismatch = provider.plan(
        request,
        FineSafetyAnchorV2(_snapshot()),
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )
    assert mismatch.reason_code == "wheel_sqp_identity_mismatch"
    assert mismatch.category is FailureCategoryV2.INTERNAL_ERROR

    expired = provider.plan(
        request,
        FineSafetyAnchorV2(snapshot),
        PlanningDeadlineV2(0.0, 0.0, _Clock()),
    )
    assert expired.reason_code == "planning_deadline_expired"
    assert expired.category is FailureCategoryV2.TIMEOUT


def test_provider_repeated_calls_keep_no_state_and_repeat_decision_hash() -> None:
    from path_planner.v2.contracts import PlanningSuccessV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2

    snapshot = _snapshot()
    center = GEOMETRY.cell_center(CELL)
    pose = PoseStateV2(center.x, center.y, 0.0)
    request = _provider_request(snapshot, start=pose, goal=pose)
    provider = WheelKinematicSQPProviderV2(PROFILE)
    anchor = FineSafetyAnchorV2(snapshot)

    first = provider.plan(
        request,
        anchor,
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )
    second = provider.plan(
        request,
        anchor,
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )

    assert type(first) is PlanningSuccessV2
    assert type(second) is PlanningSuccessV2
    assert first.search_telemetry.decision_hash == second.search_telemetry.decision_hash
    assert first.route.route_hash == second.route.route_hash


def _orchestration_fixture(monkeypatch: pytest.MonkeyPatch):
    import path_planner.v2.providers.wheel_sqp as provider_module
    from path_planner.v2.terrain import snapshot_hash
    from path_planner.v2.wheel_corridors import WheelCorridorGenerationResultV2
    from path_planner.v2.wheel_sqp_contracts import (
        WheelCorridorV2,
        WheelTopologySignatureV1,
    )
    from path_planner.v2.wheel_corridors import wheel_corridor_path_hash_v1

    snapshot = _snapshot()
    start_cell = Cell(1, 1)
    goal_cell = Cell(4, 3)
    start_center = GEOMETRY.cell_center(start_cell)
    goal_center = GEOMETRY.cell_center(goal_cell)
    request = _provider_request(
        snapshot,
        start=PoseStateV2(start_center.x, start_center.y, 0.0),
        goal=PoseStateV2(goal_center.x, goal_center.y, 0.0),
    )
    digest = snapshot_hash(snapshot)
    paths = (
        (start_cell, goal_cell),
        (start_cell, Cell(2, 1), goal_cell),
        (start_cell, Cell(1, 2), goal_cell),
    )
    corridors = tuple(
        WheelCorridorV2(
            corridor_index=index,
            cells=cells,
            corridor_hash=wheel_corridor_path_hash_v1(digest, cells),
            guide_cost=float(index + 1),
            path_length_m=float(index + 1),
            topology_signature=WheelTopologySignatureV1(()),
        )
        for index, cells in enumerate(paths)
    )
    generated = WheelCorridorGenerationResultV2(
        corridors=corridors,
        reason_code="wheel_sqp_corridors_ready",
        expanded_states=7,
        terrain_snapshot_hash=digest,
    )
    monkeypatch.setattr(
        provider_module,
        "generate_wheel_corridors_v2",
        lambda *args, **kwargs: generated,
    )
    return provider_module, snapshot, request, corridors


def _fake_attempt(
    provider_module,
    *,
    passed: bool,
    counterexample: WheelL2CounterexampleV2 | None = None,
    reason_code: str | None = None,
):
    initial = SimpleNamespace(
        segments=(object(),),
        modes=(SimpleNamespace(value="forward"),),
    )
    optimization = SimpleNamespace(
        iteration_count=2,
        function_evaluation_count=3,
        reason_code=None,
        status=SimpleNamespace(value="feasible"),
    )
    receipt = SimpleNamespace(
        passed=passed,
        reason_code=None if passed else reason_code or "wheel_sqp_continuous_collision",
        counterexample=counterexample,
        checked_cell_count=5,
        checked_interval_count=11,
    )
    return provider_module._AttemptOutcomeV1(
        initial_guess=initial,
        optimization=optimization,
        candidate=object(),
        candidate_bytes=b"canonical-candidate",
        l2=receipt,
        reason_code=None,
    )


def _repair_for(snapshot_digest: str) -> wheel_sqp_contracts_module.WheelSQPRepairConstraintV1:
    left = GEOMETRY.origin[0] + CELL.x * GEOMETRY.resolution_m
    bottom = GEOMETRY.origin[1] + CELL.y * GEOMETRY.resolution_m
    return wheel_sqp_contracts_module.WheelSQPRepairConstraintV1(
        source_candidate_hash="a" * 64,
        source_segment_hash="b" * 64,
        source_snapshot_hash=snapshot_digest,
        segment_index=0,
        cell=CELL,
        face="left",
        rho_lo=0.0,
        rho_mid=0.5,
        rho_hi=1.0,
        cell_left_x_m=left,
        cell_right_x_m=left + GEOMETRY.resolution_m,
        cell_bottom_y_m=bottom,
        cell_top_y_m=bottom + GEOMETRY.resolution_m,
        clearance_m=1.0e-4,
    )


def test_first_full_l2_route_returns_without_touching_later_corridors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2

    provider_module, snapshot, request, _ = _orchestration_fixture(monkeypatch)
    calls: list[tuple[int, bool]] = []
    sentinel = object()

    def fake_run(corridor, *args, repair=None, **kwargs):
        calls.append((corridor.corridor_index, repair is not None))
        if corridor.corridor_index != 0 or repair is not None:
            raise AssertionError("later corridor or repair must not run")
        return _fake_attempt(provider_module, passed=True)

    monkeypatch.setattr(provider_module, "_run_attempt", fake_run)
    monkeypatch.setattr(provider_module, "_build_success", lambda *a, **k: sentinel)

    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )

    assert outcome is sentinel
    assert calls == [(0, False)]


def test_one_repair_is_run_once_and_can_return_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2
    from path_planner.v2.terrain import snapshot_hash

    provider_module, snapshot, request, _ = _orchestration_fixture(monkeypatch)
    digest = snapshot_hash(snapshot)
    counterexample = WheelL2CounterexampleV2(
        segment_index=0,
        candidate_segment_hash="b" * 64,
        time_fraction_lo=0.0,
        time_fraction_mid=0.5,
        time_fraction_hi=1.0,
        cell=CELL,
        terrain_reason="terrain_hard_obstacle",
        snapshot_hash=digest,
        candidate_hash="a" * 64,
        repairable=True,
    )
    repair = _repair_for(digest)
    calls: list[tuple[int, bool]] = []
    success_args: list[bool] = []
    sentinel = object()

    def fake_run(corridor, *args, repair=None, **kwargs):
        calls.append((corridor.corridor_index, repair is not None))
        return _fake_attempt(
            provider_module,
            passed=repair is not None,
            counterexample=None if repair is not None else counterexample,
        )

    def fake_success(*args, repair_applied=False, **kwargs):
        success_args.append(repair_applied)
        return sentinel

    monkeypatch.setattr(provider_module, "_run_attempt", fake_run)
    monkeypatch.setattr(
        provider_module,
        "repair_constraint_from_counterexample_v2",
        lambda *args, **kwargs: repair,
    )
    monkeypatch.setattr(provider_module, "_build_success", fake_success)

    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )

    assert outcome is sentinel
    assert calls == [(0, False), (0, True)]
    assert success_args == [True]


def test_failed_repair_moves_to_next_corridor_without_second_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2
    from path_planner.v2.terrain import snapshot_hash

    provider_module, snapshot, request, _ = _orchestration_fixture(monkeypatch)
    digest = snapshot_hash(snapshot)
    counterexample = WheelL2CounterexampleV2(
        segment_index=0,
        candidate_segment_hash="b" * 64,
        time_fraction_lo=0.0,
        time_fraction_mid=0.5,
        time_fraction_hi=1.0,
        cell=CELL,
        terrain_reason="terrain_hard_obstacle",
        snapshot_hash=digest,
        candidate_hash="a" * 64,
        repairable=True,
    )
    repair = _repair_for(digest)
    calls: list[tuple[int, bool]] = []
    sentinel = object()

    def fake_run(corridor, *args, repair=None, **kwargs):
        calls.append((corridor.corridor_index, repair is not None))
        if corridor.corridor_index == 1 and repair is None:
            return _fake_attempt(provider_module, passed=True)
        return _fake_attempt(
            provider_module,
            passed=False,
            counterexample=counterexample,
        )

    monkeypatch.setattr(provider_module, "_run_attempt", fake_run)
    monkeypatch.setattr(
        provider_module,
        "repair_constraint_from_counterexample_v2",
        lambda *args, **kwargs: repair,
    )
    monkeypatch.setattr(provider_module, "_build_success", lambda *a, **k: sentinel)

    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )

    assert outcome is sentinel
    assert calls == [(0, False), (0, True), (1, False)]


def test_final_failure_uses_frozen_precedence_not_last_corridor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from path_planner.v2.contracts import FailureCategoryV2, PlanningFailureV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2

    provider_module, snapshot, request, _ = _orchestration_fixture(monkeypatch)
    reasons = {
        0: "wheel_sqp_infeasible",
        1: "wheel_sqp_resource_budget_exceeded",
        2: "wheel_sqp_initialization_failed",
    }

    def fake_run(corridor, *args, **kwargs):
        return provider_module._AttemptOutcomeV1(
            initial_guess=None,
            optimization=None,
            candidate=None,
            candidate_bytes=None,
            l2=None,
            reason_code=reasons[corridor.corridor_index],
        )

    monkeypatch.setattr(provider_module, "_run_attempt", fake_run)
    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert outcome.category is FailureCategoryV2.RESOURCE_LIMIT
    assert outcome.search_telemetry.decision_hash is not None


@pytest.mark.parametrize(
    "source",
    ("original_attempt", "original_l2", "repair_attempt", "repair_l2"),
)
def test_component_deadline_reason_stops_before_next_corridor_even_if_clock_is_live(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    from path_planner.v2.contracts import FailureCategoryV2, PlanningFailureV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2
    from path_planner.v2.terrain import snapshot_hash

    provider_module, snapshot, request, _ = _orchestration_fixture(monkeypatch)
    digest = snapshot_hash(snapshot)
    counterexample = WheelL2CounterexampleV2(
        segment_index=0,
        candidate_segment_hash="b" * 64,
        time_fraction_lo=0.0,
        time_fraction_mid=0.5,
        time_fraction_hi=1.0,
        cell=CELL,
        terrain_reason="terrain_hard_obstacle",
        snapshot_hash=digest,
        candidate_hash="a" * 64,
        repairable=True,
    )
    repair = _repair_for(digest)
    calls: list[tuple[int, bool]] = []

    def fake_run(corridor, *args, repair=None, **kwargs):
        is_repair = repair is not None
        calls.append((corridor.corridor_index, is_repair))
        target_repair = source.startswith("repair_")
        if target_repair and not is_repair:
            return _fake_attempt(
                provider_module,
                passed=False,
                counterexample=counterexample,
            )
        if source.endswith("attempt"):
            return provider_module._AttemptOutcomeV1(
                initial_guess=None,
                optimization=None,
                candidate=None,
                candidate_bytes=None,
                l2=None,
                reason_code="planning_deadline_expired",
            )
        return _fake_attempt(
            provider_module,
            passed=False,
            reason_code="planning_deadline_expired",
        )

    monkeypatch.setattr(provider_module, "_run_attempt", fake_run)
    monkeypatch.setattr(
        provider_module,
        "repair_constraint_from_counterexample_v2",
        lambda *args, **kwargs: repair,
    )
    deadline = PlanningDeadlineV2(0.0, 5.0, _Clock())
    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        deadline,
    )

    assert deadline.expired is False
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "planning_deadline_expired"
    assert outcome.category is FailureCategoryV2.TIMEOUT
    assert calls == (
        [(0, False), (0, True)]
        if source.startswith("repair_")
        else [(0, False)]
    )


def test_nonrepairable_l2_never_enters_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from path_planner.v2.contracts import PlanningFailureV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2
    from path_planner.v2.terrain import snapshot_hash

    provider_module, snapshot, request, _ = _orchestration_fixture(monkeypatch)
    digest = snapshot_hash(snapshot)
    counterexample = WheelL2CounterexampleV2(
        segment_index=0,
        candidate_segment_hash="b" * 64,
        time_fraction_lo=0.0,
        time_fraction_mid=0.5,
        time_fraction_hi=1.0,
        cell=CELL,
        terrain_reason="terrain_unknown",
        snapshot_hash=digest,
        candidate_hash="a" * 64,
        repairable=False,
    )
    calls: list[tuple[int, bool]] = []

    def fake_run(corridor, *args, repair=None, **kwargs):
        calls.append((corridor.corridor_index, repair is not None))
        return _fake_attempt(
            provider_module,
            passed=False,
            counterexample=counterexample,
        )

    monkeypatch.setattr(provider_module, "_run_attempt", fake_run)
    monkeypatch.setattr(
        provider_module,
        "repair_constraint_from_counterexample_v2",
        lambda *a, **k: pytest.fail("nonrepairable L2 must not create repair"),
    )
    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_candidate_l2_rejected"
    assert calls == [(0, False), (1, False), (2, False)]
    assert outcome.search_telemetry.repair_attempted is False


def test_unexpected_exception_exposes_only_its_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from path_planner.v2.contracts import FailureCategoryV2, PlanningFailureV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2

    provider_module, snapshot, request, _ = _orchestration_fixture(monkeypatch)

    def explode(*args, **kwargs):
        raise RuntimeError("sensitive implementation detail")

    monkeypatch.setattr(provider_module, "_run_attempt", explode)
    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_internal_error"
    assert outcome.category is FailureCategoryV2.INTERNAL_ERROR
    assert outcome.evidence.details == (("exception_type", "RuntimeError"),)
    assert "sensitive" not in repr(outcome)


def test_l2_success_keeps_real_endpoint_within_tolerance_without_goal_snapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import path_planner.v2.providers.wheel_sqp as provider_module
    from path_planner.v2.contracts import PlanningSuccessV2
    from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2
    from path_planner.v2.wheel_kinematics import wheel_relative_energy_v1
    from path_planner.v2.wheel_sqp_contracts import (
        WheelSQPCandidateV2,
        WheelSQPOptimizationResultV2,
        WheelSQPModeV2,
        WheelSQPStatusV2,
        _WheelSQPExactSegmentV1,
    )

    snapshot = _snapshot()
    center = GEOMETRY.cell_center(CELL)
    start = PoseStateV2(center.x, center.y, 0.0)
    requested_goal = PoseStateV2(center.x + 0.1, center.y, 0.0)
    request = _provider_request(snapshot, start=start, goal=requested_goal)

    def stop_candidate(problem, deadline, ledger):
        provider_module.wheel_sqp_solver.estimate_wheel_sqp_attempt_resources_v2(
            problem,
            deadline,
            ledger,
        )
        duration = 0.05
        segment = _WheelSQPExactSegmentV1(
            start_state=start,
            end_state=start,
            v_mps=0.0,
            omega_radps=0.0,
            duration_s=duration,
            mode=WheelSQPModeV2.STOP,
            distance_m=0.0,
            relative_energy=wheel_relative_energy_v1(
                0.0,
                0.0,
                duration,
                PROFILE,
            ),
        )
        candidate = WheelSQPCandidateV2(
            candidate_hash="a" * 64,
            corridor_hash=problem.corridor.corridor_hash,
            initial_guess_hash=problem.initial_guess.initial_guess_hash,
            segments=(segment,),
            objective_value=0.0,
            status=WheelSQPStatusV2.FEASIBLE,
        )
        return WheelSQPOptimizationResultV2(
            status=WheelSQPStatusV2.FEASIBLE,
            candidate=candidate,
            iteration_count=0,
            function_evaluation_count=0,
            reason_code=None,
        )

    monkeypatch.setattr(
        provider_module.wheel_sqp_solver,
        "solve_wheel_sqp_v2",
        stop_candidate,
    )
    outcome = WheelKinematicSQPProviderV2(PROFILE).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        PlanningDeadlineV2(0.0, 5.0, _Clock()),
    )

    assert type(outcome) is PlanningSuccessV2
    assert outcome.route.primitives[-1].end_state == start
    assert outcome.route.primitives[-1].end_state != requested_goal
    assert outcome.validation_evidence.passed is True
