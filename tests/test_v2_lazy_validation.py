from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from math import nan

import numpy as np
import pytest

import path_planner.v2 as v2
import path_planner.v2.validation as validation_module
from path_planner.core import Cell, WorldPoint
from path_planner.search import MotionPrimitive, Pose2D, replay_motion_primitive
from path_planner.v2.cache import ValidationCacheV2
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.geometry import conservative_wheel_sweep_cells
from path_planner.v2.profiles import PlatformProfileV2, WheelProfileV2
from path_planner.v2.providers.wheel import WheelMotionPrimitiveV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)
from path_planner.v2.validation import RouteValidationResultV2, validate_route


def _snapshot(
    *,
    width: int = 12,
    height: int = 6,
    unknown_cells: tuple[Cell, ...] = (),
    hard_cells: tuple[Cell, ...] = (),
) -> TerrainSnapshotV2:
    shape = (height, width)
    traversable = np.ones(shape, dtype=bool)
    hard = np.zeros(shape, dtype=bool)
    observed = np.ones(shape, dtype=bool)
    for cell in unknown_cells:
        observed[cell.y, cell.x] = False
    for cell in hard_cells:
        hard[cell.y, cell.x] = True
        traversable[cell.y, cell.x] = False
    return TerrainSnapshotV2(
        geometry=FineGridGeometryV2(width=width, height=height, frame_id="moon"),
        elevation_m=np.zeros(shape),
        slope_deg=np.zeros(shape),
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=observed,
        confidence=np.ones(shape),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="lazy-validation-fixture",
            source_hash="lazy-validation-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )


def _profile() -> WheelProfileV2:
    return WheelProfileV2(
        profile=PlatformProfileV2(
            profile_id="wheel-lazy/v1",
            platform_kind=PlatformKindV2.WHEEL,
            capability_revision="wheel-lazy-capability/v1",
            simulation_proxy=False,
            max_traversable_slope_deg=30.0,
        ),
        primitive_duration_s=1.0,
        integration_dt_s=0.25,
    )


def _primitive(
    profile: WheelProfileV2,
    *,
    start: Pose2D | None = None,
) -> WheelMotionPrimitiveV2:
    start = start or Pose2D(1.25, 1.25, 0.0)
    control = MotionPrimitive("forward", 1.0, 0.0, 1.0)
    replay = replay_motion_primitive(start, control, profile.integration_dt_s)
    samples = tuple(
        PoseStateV2(sample.x_m, sample.y_m, sample.theta_rad)
        for sample in replay.samples
    )
    return WheelMotionPrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=samples[0],
        end_state=samples[-1],
        duration_s=control.duration_s,
        distance_m=replay.distance_m,
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L0,
        control_name=control.name,
        samples=samples,
        v_mps=control.v_mps,
        omega_radps=control.omega_radps,
        reverse=False,
        turn_in_place=False,
    )


def _route(primitive: WheelMotionPrimitiveV2, *, complete: bool = True) -> TypedRouteV2:
    return TypedRouteV2(
        platform_kind=PlatformKindV2.WHEEL,
        primitives=(primitive,),
        total_cost=1.0,
        is_complete=complete,
    )


def _request(
    snapshot: TerrainSnapshotV2,
    profile: WheelProfileV2,
    primitive: WheelMotionPrimitiveV2,
) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="lazy-validation-request",
        platform_profile_id=profile.profile.profile_id,
        start_state=primitive.start_state,
        goal_state=primitive.end_state,
        terrain_snapshot=snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(),
        timeout_s=1.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=11,
    )


def _deadline() -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, 1.0, lambda: 0.0)


def _sample_cells(
    primitive: WheelMotionPrimitiveV2,
    geometry: FineGridGeometryV2,
) -> frozenset[Cell]:
    return frozenset(
        geometry.world_to_cell(WorldPoint(sample.x_m, sample.y_m))
        for sample in primitive.samples
    )


def _body_only_cell(
    primitive: WheelMotionPrimitiveV2,
    profile: WheelProfileV2,
    geometry: FineGridGeometryV2,
) -> Cell:
    sweep = conservative_wheel_sweep_cells(
        Pose2D(
            primitive.start_state.x_m,
            primitive.start_state.y_m,
            primitive.start_state.heading_rad,
        ),
        MotionPrimitive(
            primitive.control_name,
            primitive.v_mps,
            primitive.omega_radps,
            primitive.duration_s,
            reverse=primitive.reverse,
            turn_in_place=primitive.turn_in_place,
        ),
        geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    return next(cell for cell in sweep if cell not in _sample_cells(primitive, geometry))


def _decision(result: RouteValidationResultV2):
    return (
        result.route,
        result.success,
        result.reason_code,
        result.stage_evidence,
        result.l2_result,
    )


def test_route_pipeline_contract_is_public_frozen_and_slotted() -> None:
    assert v2.RouteValidationResultV2 is RouteValidationResultV2
    assert v2.validate_route is validate_route
    profile = _profile()
    snapshot = _snapshot()
    result = validate_route(
        _route(_primitive(profile)),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L0,
    )

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.reason_code = "forged"


def test_route_pipeline_result_rejects_a_stage_after_prior_rejection() -> None:
    profile = _profile()
    route = _route(_primitive(profile))

    with pytest.raises(ValueError, match="rejected earlier stage"):
        RouteValidationResultV2(
            route=route,
            success=False,
            reason_code="route_requires_l2_validation",
            stage_evidence=(
                ValidationEvidenceV2(
                    validator_id=validation_module.WHEEL_ROUTE_L0_VALIDATOR_ID_V2,
                    level=ValidationLevelV2.L0,
                    passed=False,
                    checks=("route_incomplete",),
                ),
                ValidationEvidenceV2(
                    validator_id=validation_module.WHEEL_ROUTE_L1_VALIDATOR_ID_V2,
                    level=ValidationLevelV2.L1,
                    passed=True,
                    checks=("route_l1_valid",),
                ),
            ),
            l2_result=None,
            cache_hits=0,
            cache_misses=0,
        )


@pytest.mark.parametrize(
    ("validator_id", "checks"),
    [
        ("fake-l0-validator/v1", ("route_l0_valid",)),
        (validation_module.WHEEL_ROUTE_L0_VALIDATOR_ID_V2, ("fake_l0_pass",)),
    ],
)
def test_route_pipeline_result_rejects_forged_l0_identity_or_pass_reason(
    validator_id,
    checks,
) -> None:
    profile = _profile()
    route = _route(_primitive(profile))

    with pytest.raises(ValueError, match="L0 validator"):
        RouteValidationResultV2(
            route=route,
            success=False,
            reason_code="route_requires_l2_validation",
            stage_evidence=(
                ValidationEvidenceV2(
                    validator_id=validator_id,
                    level=ValidationLevelV2.L0,
                    passed=True,
                    checks=checks,
                ),
            ),
            l2_result=None,
            cache_hits=0,
            cache_misses=0,
        )


def test_route_pipeline_result_rejects_fake_l1_and_transition_l2_identity() -> None:
    profile = _profile()
    primitive = _primitive(profile)
    route_l2 = replace(
        _route(primitive),
        primitives=(replace(primitive, validation_level=ValidationLevelV2.L2),),
    )
    l0 = ValidationEvidenceV2(
        validator_id=validation_module.WHEEL_ROUTE_L0_VALIDATOR_ID_V2,
        level=ValidationLevelV2.L0,
        passed=True,
        checks=("route_l0_valid",),
    )
    fake_l1 = ValidationEvidenceV2(
        validator_id="fake-l1-validator/v1",
        level=ValidationLevelV2.L1,
        passed=True,
        checks=("route_l1_valid",),
    )
    with pytest.raises(ValueError, match="L1 validator"):
        RouteValidationResultV2(
            route=route_l2,
            success=False,
            reason_code="route_requires_l2_validation",
            stage_evidence=(l0, fake_l1),
            l2_result=None,
            cache_hits=0,
            cache_misses=0,
        )

    l1 = replace(
        fake_l1,
        validator_id=validation_module.WHEEL_ROUTE_L1_VALIDATOR_ID_V2,
    )
    transition_evidence = ValidationEvidenceV2(
        validator_id=validation_module.WHEEL_TRANSITION_VALIDATOR_ID_V2,
        level=ValidationLevelV2.L2,
        passed=True,
        checks=("transition_l2_valid",),
    )
    transition_result = validation_module.WheelValidationResultV2(
        evidence=transition_evidence,
        reason_code="transition_l2_valid",
        timed_out=False,
        failed_cell=None,
        failed_primitive_index=None,
        checked_cell_count=1,
    )
    with pytest.raises(ValueError, match="route L2 validator"):
        RouteValidationResultV2(
            route=route_l2,
            success=True,
            reason_code="transition_l2_valid",
            stage_evidence=(l0, l1, transition_evidence),
            l2_result=transition_result,
            cache_hits=0,
            cache_misses=0,
        )


def test_route_pipeline_result_rejects_l2_stage_without_l2_result() -> None:
    profile = _profile()
    primitive = _primitive(profile)
    route_l2 = replace(
        _route(primitive),
        primitives=(replace(primitive, validation_level=ValidationLevelV2.L2),),
    )
    evidence = (
        ValidationEvidenceV2(
            validator_id=validation_module.WHEEL_ROUTE_L0_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L0,
            passed=True,
            checks=("route_l0_valid",),
        ),
        ValidationEvidenceV2(
            validator_id=validation_module.WHEEL_ROUTE_L1_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L1,
            passed=True,
            checks=("route_l1_valid",),
        ),
        ValidationEvidenceV2(
            validator_id=validation_module.WHEEL_ROUTE_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=True,
            checks=("route_l2_valid",),
        ),
    )

    with pytest.raises(ValueError, match="L2 stage requires l2_result"):
        RouteValidationResultV2(
            route=route_l2,
            success=False,
            reason_code="route_requires_l2_validation",
            stage_evidence=evidence,
            l2_result=None,
            cache_hits=0,
            cache_misses=0,
        )

@pytest.mark.parametrize("level", [ValidationLevelV2.L0, ValidationLevelV2.L1])
def test_l0_and_l1_can_never_return_success(level) -> None:
    profile = _profile()
    snapshot = _snapshot()
    result = validate_route(
        route=_route(_primitive(profile)),
        anchor=FineSafetyAnchorV2(snapshot),
        profile=profile,
        max_level=level,
    )

    assert result.success is False
    assert result.reason_code == "route_requires_l2_validation"
    assert tuple(evidence.level for evidence in result.stage_evidence) == (
        (ValidationLevelV2.L0,)
        if level is ValidationLevelV2.L0
        else (ValidationLevelV2.L0, ValidationLevelV2.L1)
    )
    assert all(evidence.passed for evidence in result.stage_evidence)
    assert result.l2_result is None


def test_l0_rejects_incomplete_route_and_stops_before_l1_or_l2(monkeypatch) -> None:
    profile = _profile()
    snapshot = _snapshot()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("later validation must not run")

    monkeypatch.setattr(validation_module, "_validate_route_l1", forbidden)
    monkeypatch.setattr(validation_module, "validate_route_l2", forbidden)
    result = validate_route(
        _route(_primitive(profile), complete=False),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L2,
        request=_request(snapshot, profile, _primitive(profile)),
        deadline=_deadline(),
    )

    assert result.success is False
    assert result.reason_code == "route_incomplete"
    assert tuple(item.level for item in result.stage_evidence) == (ValidationLevelV2.L0,)
    assert result.l2_result is None


def test_l0_rejects_obvious_fine_grid_boundary_violation() -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile, start=Pose2D(-0.25, 1.25, 0.0))

    result = validate_route(
        _route(primitive),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L0,
    )

    assert result.reason_code == "terrain_out_of_bounds"
    assert result.stage_evidence[0].passed is False


def test_l1_rejects_unknown_observed_sample_cell_and_stops_before_l2(monkeypatch) -> None:
    profile = _profile()
    base = _snapshot()
    primitive = _primitive(profile)
    unknown = base.geometry.world_to_cell(
        WorldPoint(primitive.samples[-1].x_m, primitive.samples[-1].y_m)
    )
    snapshot = _snapshot(unknown_cells=(unknown,))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("L2 must not run after L1 rejection")

    monkeypatch.setattr(validation_module, "validate_route_l2", forbidden)
    result = validate_route(
        _route(primitive),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L2,
        request=_request(snapshot, profile, primitive),
        deadline=_deadline(),
    )

    assert result.success is False
    assert result.reason_code == "terrain_unknown"
    assert tuple(item.level for item in result.stage_evidence) == (
        ValidationLevelV2.L0,
        ValidationLevelV2.L1,
    )
    assert result.stage_evidence[-1].passed is False
    assert result.l2_result is None


def test_l1_is_not_body_sweep_authority() -> None:
    profile = _profile()
    base = _snapshot()
    primitive = _primitive(profile)
    blocked = _body_only_cell(primitive, profile, base.geometry)
    snapshot = _snapshot(hard_cells=(blocked,))

    result = validate_route(
        _route(primitive),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L1,
    )

    assert result.success is False
    assert result.reason_code == "route_requires_l2_validation"
    assert result.stage_evidence[-1].passed is True


@pytest.mark.parametrize("missing", ["request", "deadline"])
def test_missing_l2_authority_fails_closed(missing) -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile)
    kwargs = {"request": _request(snapshot, profile, primitive), "deadline": _deadline()}
    kwargs[missing] = None

    result = validate_route(
        _route(primitive),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L2,
        **kwargs,
    )

    assert result.success is False
    assert result.reason_code == "route_l2_authority_unavailable"
    assert tuple(item.level for item in result.stage_evidence) == (
        ValidationLevelV2.L0,
        ValidationLevelV2.L1,
    )
    assert result.l2_result is None


def test_real_wheel_l2_pass_is_the_only_success_and_preserves_evidence() -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile)

    result = validate_route(
        _route(primitive),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L2,
        request=_request(snapshot, profile, primitive),
        deadline=_deadline(),
    )

    assert result.success is True
    assert result.reason_code == "route_l2_valid"
    assert result.l2_result is not None
    assert result.l2_result.evidence.passed is True
    assert tuple(item.level for item in result.stage_evidence) == (
        ValidationLevelV2.L0,
        ValidationLevelV2.L1,
        ValidationLevelV2.L2,
    )
    assert result.stage_evidence[-1] == result.l2_result.evidence
    assert result.route.is_complete is True
    assert all(
        primitive.validation_level is ValidationLevelV2.L2
        for primitive in result.route.primitives
    )
    assert result.l2_result.validated_route_hash == sha256(
        v2.canonical_json_bytes(result.route)
    ).hexdigest()


def test_route_pipeline_result_rejects_success_with_non_l2_route() -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile)
    route = _route(primitive)
    l2 = validation_module.validate_route_l2(
        route,
        _request(snapshot, profile, primitive),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )

    with pytest.raises(ValueError, match="route primitives must all be L2"):
        RouteValidationResultV2(
            route=route,
            success=True,
            reason_code=l2.reason_code,
            stage_evidence=(
                ValidationEvidenceV2(
                    validator_id=validation_module.WHEEL_ROUTE_L0_VALIDATOR_ID_V2,
                    level=ValidationLevelV2.L0,
                    passed=True,
                    checks=("route_l0_valid",),
                ),
                ValidationEvidenceV2(
                    validator_id=validation_module.WHEEL_ROUTE_L1_VALIDATOR_ID_V2,
                    level=ValidationLevelV2.L1,
                    passed=True,
                    checks=("route_l1_valid",),
                ),
                l2.evidence,
            ),
            l2_result=l2,
            cache_hits=0,
            cache_misses=0,
        )


def test_route_l2_evidence_is_bound_to_the_exact_validated_route_identity() -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive_a = _primitive(profile)
    route_a = replace(
        _route(primitive_a),
        primitives=(replace(primitive_a, validation_level=ValidationLevelV2.L2),),
    )
    l2_a = validation_module.validate_route_l2(
        route_a,
        _request(snapshot, profile, primitive_a),
        FineSafetyAnchorV2(snapshot),
        profile,
        _deadline(),
    )
    primitive_b = _primitive(profile, start=Pose2D(1.75, 1.25, 0.0))
    route_b = replace(
        _route(primitive_b),
        primitives=(replace(primitive_b, validation_level=ValidationLevelV2.L2),),
    )
    stages = (
        ValidationEvidenceV2(
            validator_id=validation_module.WHEEL_ROUTE_L0_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L0,
            passed=True,
            checks=("route_l0_valid",),
        ),
        ValidationEvidenceV2(
            validator_id=validation_module.WHEEL_ROUTE_L1_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L1,
            passed=True,
            checks=("route_l1_valid",),
        ),
        l2_a.evidence,
    )

    with pytest.raises(ValueError, match="validated route identity"):
        RouteValidationResultV2(
            route=route_b,
            success=True,
            reason_code=l2_a.reason_code,
            stage_evidence=stages,
            l2_result=l2_a,
            cache_hits=0,
            cache_misses=0,
        )

    route_a_copy = replace(
        route_a,
        primitives=tuple(replace(item) for item in route_a.primitives),
    )
    same_content = RouteValidationResultV2(
        route=route_a_copy,
        success=True,
        reason_code=l2_a.reason_code,
        stage_evidence=stages,
        l2_result=l2_a,
        cache_hits=0,
        cache_misses=0,
    )
    assert same_content.success is True


def test_transition_validation_result_forbids_route_identity_hash() -> None:
    transition_evidence = ValidationEvidenceV2(
        validator_id=validation_module.WHEEL_TRANSITION_VALIDATOR_ID_V2,
        level=ValidationLevelV2.L2,
        passed=True,
        checks=("transition_l2_valid",),
    )

    with pytest.raises(ValueError, match="transition.*route hash"):
        validation_module.WheelValidationResultV2(
            evidence=transition_evidence,
            reason_code="transition_l2_valid",
            timed_out=False,
            failed_cell=None,
            failed_primitive_index=None,
            checked_cell_count=1,
            validated_route_hash="a" * 64,
        )


def test_unexpected_l2_runtime_error_propagates(monkeypatch) -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile)

    def explode(*_args, **_kwargs):
        raise RuntimeError("unexpected-l2-sentinel")

    monkeypatch.setattr(validation_module, "validate_route_l2", explode)
    with pytest.raises(RuntimeError, match="unexpected-l2-sentinel"):
        validate_route(
            _route(primitive),
            FineSafetyAnchorV2(snapshot),
            profile,
            ValidationLevelV2.L2,
            request=_request(snapshot, profile, primitive),
            deadline=_deadline(),
        )


def test_wrong_l2_return_type_is_rejected(monkeypatch) -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile)
    monkeypatch.setattr(validation_module, "validate_route_l2", lambda *_args: object())

    with pytest.raises(TypeError, match="exact WheelValidationResultV2"):
        validate_route(
            _route(primitive),
            FineSafetyAnchorV2(snapshot),
            profile,
            ValidationLevelV2.L2,
            request=_request(snapshot, profile, primitive),
            deadline=_deadline(),
        )


def test_real_wheel_l2_rejection_is_never_downgraded() -> None:
    profile = _profile()
    base = _snapshot()
    primitive = _primitive(profile)
    blocked = _body_only_cell(primitive, profile, base.geometry)
    snapshot = _snapshot(hard_cells=(blocked,))

    result = validate_route(
        _route(primitive),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L2,
        request=_request(snapshot, profile, primitive),
        deadline=_deadline(),
    )

    assert result.success is False
    assert result.reason_code == "terrain_hard_obstacle"
    assert result.l2_result is not None
    assert result.l2_result.reason_code == result.reason_code
    assert result.stage_evidence[-1] == result.l2_result.evidence


def test_cache_on_off_decisions_match_and_second_run_has_real_hits() -> None:
    profile = _profile()
    snapshot = _snapshot()
    route = _route(_primitive(profile))
    anchor = FineSafetyAnchorV2(snapshot)
    cache = ValidationCacheV2()

    cache_off = validate_route(route, anchor, profile, ValidationLevelV2.L1)
    first = validate_route(route, anchor, profile, ValidationLevelV2.L1, cache=cache)
    second = validate_route(route, anchor, profile, ValidationLevelV2.L1, cache=cache)

    assert _decision(cache_off) == _decision(first) == _decision(second)
    assert cache_off.cache_hits == cache_off.cache_misses == 0
    assert first.cache_hits == 0
    assert first.cache_misses == 1
    assert second.cache_hits == 1
    assert second.cache_misses == 0


def test_preloaded_wrong_l1_evidence_cannot_change_the_decision() -> None:
    profile = _profile()
    base = _snapshot()
    primitive = _primitive(profile)
    unknown = base.geometry.world_to_cell(
        WorldPoint(primitive.samples[-1].x_m, primitive.samples[-1].y_m)
    )
    snapshot = _snapshot(unknown_cells=(unknown,))
    route = _route(primitive)
    anchor = FineSafetyAnchorV2(snapshot)
    cache = ValidationCacheV2()
    key = validation_module._lazy_cache_key(
        route,
        anchor,
        profile,
        ValidationLevelV2.L1,
    )
    assert key is not None
    cache.put(
        key,
        ValidationEvidenceV2(
            validator_id=validation_module.WHEEL_ROUTE_L1_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L1,
            passed=True,
            checks=("route_l1_valid",),
        ),
    )

    cache_off = validate_route(route, anchor, profile, ValidationLevelV2.L1)
    cache_on = validate_route(
        route,
        anchor,
        profile,
        ValidationLevelV2.L1,
        cache=cache,
    )

    assert cache_on.cache_hits == 0
    assert cache_on.cache_misses == 1
    assert _decision(cache_on) == _decision(cache_off)
    assert cache_on.reason_code == "terrain_unknown"

    second = validate_route(
        route,
        anchor,
        profile,
        ValidationLevelV2.L1,
        cache=cache,
    )
    assert second.cache_hits == 1
    assert second.cache_misses == 0
    assert _decision(second) == _decision(cache_off)


def test_second_run_skips_the_verified_cached_l1_validator(monkeypatch) -> None:
    profile = _profile()
    snapshot = _snapshot()
    route = _route(_primitive(profile))
    anchor = FineSafetyAnchorV2(snapshot)
    cache = ValidationCacheV2()
    first = validate_route(
        route,
        anchor,
        profile,
        ValidationLevelV2.L1,
        cache=cache,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("verified L1 cache hit must skip fresh validation")

    monkeypatch.setattr(validation_module, "_validate_route_l1", forbidden)
    second = validate_route(
        route,
        anchor,
        profile,
        ValidationLevelV2.L1,
        cache=cache,
    )

    assert first.cache_misses == 1
    assert second.cache_hits == 1
    assert _decision(second) == _decision(first)


def test_shared_cache_counters_do_not_pollute_per_call_telemetry(monkeypatch) -> None:
    profile = _profile()
    snapshot = _snapshot()
    route = _route(_primitive(profile))
    anchor = FineSafetyAnchorV2(snapshot)
    cache = ValidationCacheV2()
    original = validation_module._validate_route_l1

    def polluting(*args, **kwargs):
        assert cache.get(object()) is None
        return original(*args, **kwargs)

    monkeypatch.setattr(validation_module, "_validate_route_l1", polluting)
    result = validate_route(
        route,
        anchor,
        profile,
        ValidationLevelV2.L1,
        cache=cache,
    )

    assert cache.miss_count > result.cache_misses
    assert result.cache_hits == 0
    assert result.cache_misses == 1


def test_whole_route_l2_is_never_reused_from_primitive_cache(monkeypatch) -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile)
    route = _route(primitive)
    request = _request(snapshot, profile, primitive)
    anchor = FineSafetyAnchorV2(snapshot)
    cache = ValidationCacheV2()
    original = validation_module.validate_route_l2
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(validation_module, "validate_route_l2", counted)
    first = validate_route(
        route,
        anchor,
        profile,
        ValidationLevelV2.L2,
        request=request,
        deadline=_deadline(),
        cache=cache,
    )
    second = validate_route(
        route,
        anchor,
        profile,
        ValidationLevelV2.L2,
        request=request,
        deadline=_deadline(),
        cache=cache,
    )

    assert calls == 2
    assert _decision(first) == _decision(second)
    assert second.cache_hits == 1


@pytest.mark.parametrize("value", [True, nan, 10**400])
def test_forged_numeric_primitive_values_fail_without_exception_leak(value) -> None:
    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile)
    object.__setattr__(primitive, "v_mps", value)

    result = validate_route(
        _route(primitive),
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L1,
    )

    assert result.success is False
    assert result.reason_code == "primitive_structure_mismatch"


def test_nonexact_route_and_level_are_rejected_without_subclass_protocol() -> None:
    class RouteSubclass(TypedRouteV2):
        pass

    profile = _profile()
    snapshot = _snapshot()
    primitive = _primitive(profile)
    route = RouteSubclass(
        platform_kind=PlatformKindV2.WHEEL,
        primitives=(primitive,),
        total_cost=1.0,
        is_complete=True,
    )

    with pytest.raises(TypeError, match="route must be exact TypedRouteV2"):
        validate_route(
            route,
            FineSafetyAnchorV2(snapshot),
            profile,
            ValidationLevelV2.L0,
        )
    with pytest.raises(TypeError, match="max_level must be exact ValidationLevelV2"):
        validate_route(
            _route(primitive),
            FineSafetyAnchorV2(snapshot),
            profile,
            "L1",
        )


def test_profile_and_primitive_drift_do_not_share_cached_evidence() -> None:
    profile = _profile()
    snapshot = _snapshot()
    route = _route(_primitive(profile))
    cache = ValidationCacheV2()
    first = validate_route(
        route,
        FineSafetyAnchorV2(snapshot),
        profile,
        ValidationLevelV2.L1,
        cache=cache,
    )
    changed_profile = replace(profile, max_speed_mps=0.5)
    changed = validate_route(
        route,
        FineSafetyAnchorV2(snapshot),
        changed_profile,
        ValidationLevelV2.L1,
        cache=cache,
    )

    assert first.cache_misses == 1
    assert changed.cache_hits == 0
    assert changed.cache_misses == 1
    assert changed.reason_code == "wheel_speed_limit_exceeded"
