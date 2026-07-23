from dataclasses import replace

import numpy as np
import pytest

import path_planner
import path_planner.v2 as v2
import path_planner.v2.providers.wheel_sqp as provider_module
import path_planner.v2.wheel_sqp_api as wheel_sqp_api
import path_planner.v2.wheel_sqp_serialization as serialization_module
import path_planner.v2.wheel_sqp_validation as validation_module
from path_planner.core import Cell
from path_planner.v2.api import plan_v2
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    FailureCategoryV2,
    PlanningFailureV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    ResourceBudgetV2,
    TypedRouteV2,
)
from path_planner.v2.observation import (
    ObservedTerrainInputV2,
    project_route_observation_v2,
)
from path_planner.v2.profiles import (
    PlatformProfileRegistryV2,
    PlatformProfileV2,
    WheelKinematicSQPProfileV2,
)
from path_planner.v2.providers.wheel_sqp import WheelKinematicSQPProviderV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)
from path_planner.v2.wheel_sqp_contracts import (
    WheelKinematicRouteV2,
    WheelKinematicSegmentV2,
)
from path_planner.v2.wheel_sqp_serialization import (
    wheel_route_hash_v2,
    wheel_segment_hash_v2,
)


def _api_fixture():
    profile = WheelKinematicSQPProfileV2(
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
    provider = WheelKinematicSQPProviderV2(profile)
    geometry = FineGridGeometryV2(6, 5, origin=(13.25, -7.75))
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
            source_id="wheel-sqp-task9-api-fixture",
            source_hash="wheel-sqp-task9-api-fixture-hash",
            physical_obstacle_cells_written=True,
        ),
    )
    center = geometry.cell_center(Cell(2, 3))
    pose = PoseStateV2(center.x, center.y, 0.0)
    request = PlanningRequestV2(
        request_id="wheel-sqp-task9-api",
        platform_profile_id=provider.profile.profile_id,
        start_state=pose,
        goal_state=pose,
        terrain_snapshot=snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(),
        timeout_s=2.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=20260722,
    )
    registry = PlatformProfileRegistryV2((provider.profile,))
    return request, registry, provider


def test_new_capability_requires_explicit_profile_and_provider_registration() -> None:
    request, registry, provider = _api_fixture()

    missing = plan_v2(request, registry=registry, providers={})
    assert type(missing) is PlanningFailureV2
    assert missing.reason_code == "primitive_provider_unregistered"

    success = plan_v2(
        request,
        registry=registry,
        providers={provider.profile.profile_id: provider},
    )
    assert type(success) is PlanningSuccessV2
    assert type(success.route) is WheelKinematicRouteV2
    assert all(type(item) is WheelKinematicSegmentV2 for item in success.route.primitives)


def test_new_surface_is_exported_only_from_v2() -> None:
    assert "WheelKinematicSQPProviderV2" not in path_planner.__dict__
    assert v2.WheelKinematicSQPProviderV2 is WheelKinematicSQPProviderV2
    assert v2.WHEEL_KINEMATIC_CORRIDOR_SQP_CAPABILITY_V2 == (
        "wheel_kinematic_corridor_sqp/v1"
    )


def _execute(request, registry, provider, *, clock=None):
    kwargs = {} if clock is None else {"monotonic_clock": clock}
    return plan_v2(
        request,
        registry=registry,
        providers={provider.profile.profile_id: provider},
        **kwargs,
    )


def _patch_trusted_provider_engine(monkeypatch, replacement) -> None:
    monkeypatch.setattr(provider_module, "_plan_wheel_sqp_v2", replacement)
    monkeypatch.setattr(wheel_sqp_api, "_TRUSTED_PROVIDER_ENGINE", replacement)


def test_new_capability_uses_the_dedicated_dispatcher(monkeypatch) -> None:
    request, registry, provider = _api_fixture()
    sentinel = object()
    calls = []

    def dispatch(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(wheel_sqp_api, "dispatch_wheel_sqp_provider_v2", dispatch)
    outcome = _execute(request, registry, provider)

    assert outcome is sentinel
    assert calls and calls[0][0] is request and calls[0][2] is provider


def test_wrong_provider_type_is_rejected_before_its_plan_runs() -> None:
    request, registry, provider = _api_fixture()

    class ForgedProvider:
        profile = provider.profile
        wheel_sqp_profile = provider.wheel_sqp_profile

        def __init__(self) -> None:
            self.called = False

        def plan(self, *args):
            self.called = True
            raise AssertionError("forged provider must not run")

    forged = ForgedProvider()
    outcome = plan_v2(
        request,
        registry=registry,
        providers={provider.profile.profile_id: forged},
    )

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"
    assert forged.called is False


def test_old_wheel_provider_is_never_used_as_a_fallback(monkeypatch) -> None:
    request, registry, provider = _api_fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("old wheel provider fallback is forbidden")

    monkeypatch.setattr(v2.WheelPrimitiveProviderV2, "plan", forbidden)
    outcome = _execute(request, registry, provider)
    assert type(outcome) is PlanningSuccessV2


def test_trusted_l2_runs_exactly_once_and_api_does_not_repeat_the_sweep(
    monkeypatch,
) -> None:
    request, registry, provider = _api_fixture()
    original = validation_module.validate_wheel_sqp_candidate_l2
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(provider_module, "validate_wheel_sqp_candidate_l2", counted)
    monkeypatch.setattr(validation_module, "validate_wheel_sqp_candidate_l2", counted)
    monkeypatch.setattr(wheel_sqp_api, "_TRUSTED_L2_VALIDATOR", counted)

    outcome = _execute(request, registry, provider)
    assert type(outcome) is PlanningSuccessV2
    assert calls == 1


def test_source_callable_drift_is_rejected_before_provider_execution(
    monkeypatch,
) -> None:
    request, registry, provider = _api_fixture()

    monkeypatch.setattr(
        provider_module,
        "validate_wheel_sqp_candidate_l2",
        lambda *args, **kwargs: pytest.fail("drifted L2 must not run"),
    )
    outcome = _execute(request, registry, provider)

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"


def test_provider_engine_rebinding_is_rejected_before_execution(monkeypatch) -> None:
    request, registry, provider = _api_fixture()

    monkeypatch.setattr(
        provider_module,
        "_plan_wheel_sqp_v2",
        lambda *args, **kwargs: pytest.fail("rebound provider engine must not run"),
    )
    outcome = _execute(request, registry, provider)

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"


@pytest.mark.parametrize("helper", ("_decision_hash", "_telemetry"))
def test_decision_authority_helper_rebinding_is_rejected_before_execution(
    monkeypatch,
    helper: str,
) -> None:
    request, registry, provider = _api_fixture()
    monkeypatch.setattr(
        provider_module,
        helper,
        lambda *args, **kwargs: pytest.fail("rebound decision helper must not run"),
    )

    outcome = _execute(request, registry, provider)

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"


@pytest.mark.parametrize("drift", ("request_id", "profile", "snapshot"))
def test_provider_state_drift_is_rejected_and_keeps_original_request_id(
    monkeypatch,
    drift: str,
) -> None:
    request, registry, provider = _api_fixture()
    original_request_id = request.request_id
    original = provider_module._plan_wheel_sqp_v2

    def drift_after_result(current_provider, current_request, anchor, deadline):
        outcome = original(current_provider, current_request, anchor, deadline)
        if drift == "request_id":
            object.__setattr__(current_request, "request_id", "forged-request-id")
        elif drift == "profile":
            object.__setattr__(
                current_provider.wheel_sqp_profile,
                "max_speed_mps",
                0.5,
            )
        else:
            provenance = replace(
                current_request.terrain_snapshot.provenance,
                source_hash="forged-snapshot-hash",
            )
            object.__setattr__(
                current_request.terrain_snapshot,
                "provenance",
                provenance,
            )
        return outcome

    _patch_trusted_provider_engine(monkeypatch, drift_after_result)
    outcome = _execute(request, registry, provider)

    assert type(outcome) is PlanningFailureV2
    assert outcome.request_id == original_request_id
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"


@pytest.mark.parametrize(
    "forgery",
    (
        "wrong_route_type",
        "wrong_segment_type",
        "segment_hash",
        "route_hash",
        "source_candidate_hash",
        "route_request_hash",
        "evidence_candidate_hash",
        "evidence_route_hash",
        "cost",
        "observation",
        "telemetry",
        "decision_hash",
    ),
)
def test_api_rejects_forged_wheel_sqp_success(
    monkeypatch,
    forgery: str,
) -> None:
    request, registry, provider = _api_fixture()
    original = provider_module._plan_wheel_sqp_v2

    def forged_result(current_provider, current_request, anchor, deadline):
        outcome = original(current_provider, current_request, anchor, deadline)
        assert type(outcome) is PlanningSuccessV2
        route = outcome.route
        segment = route.primitives[0]
        if forgery == "wrong_route_type":
            object.__setattr__(outcome, "route", object())
        elif forgery == "wrong_segment_type":
            object.__setattr__(route, "primitives", (object(),))
        elif forgery == "segment_hash":
            object.__setattr__(segment, "segment_hash", "0" * 64)
        elif forgery == "route_hash":
            object.__setattr__(route, "route_hash", "0" * 64)
        elif forgery == "source_candidate_hash":
            object.__setattr__(route, "source_candidate_hash", "0" * 64)
        elif forgery == "route_request_hash":
            object.__setattr__(route, "request_hash", "0" * 64)
        elif forgery == "evidence_candidate_hash":
            object.__setattr__(outcome.validation_evidence, "candidate_hash", "0" * 64)
        elif forgery == "evidence_route_hash":
            object.__setattr__(outcome.validation_evidence, "route_hash", "0" * 64)
        elif forgery == "cost":
            object.__setattr__(outcome.cost_breakdown, "distance_cost", 1.0)
        elif forgery == "observation":
            object.__setattr__(outcome.observation_projection, "source", "forged/v1")
        elif forgery == "telemetry":
            object.__setattr__(outcome.search_telemetry, "ackermann_feasible_claimed", True)
        else:
            object.__setattr__(outcome.search_telemetry, "decision_hash", "0" * 64)
        return outcome

    _patch_trusted_provider_engine(monkeypatch, forged_result)
    outcome = _execute(request, registry, provider)

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"


def test_hostile_wrong_provider_property_is_converted_to_identity_failure() -> None:
    request, registry, provider = _api_fixture()

    class HostileProvider:
        profile = provider.profile

        @property
        def wheel_sqp_profile(self):
            raise RuntimeError("hostile property")

        def plan(self, *args):
            raise AssertionError("hostile provider must not run")

    outcome = plan_v2(
        request,
        registry=registry,
        providers={provider.profile.profile_id: HostileProvider()},
    )

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"


def test_provider_property_cannot_mutate_capability_to_escape_sqp_dispatch() -> None:
    request, registry, trusted_provider = _api_fixture()
    genuine = _execute(request, registry, trusted_provider)
    assert type(genuine) is PlanningSuccessV2

    class EscapingProvider:
        def __init__(self) -> None:
            self.called = False

        @property
        def profile(self):
            object.__setattr__(
                trusted_provider.profile,
                "capability_revision",
                "wheel-provider-capability/v1",
            )
            return trusted_provider.profile

        def plan(self, *args):
            self.called = True
            return genuine

    escaping = EscapingProvider()
    outcome = plan_v2(
        request,
        registry=registry,
        providers={request.platform_profile_id: escaping},
    )

    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"
    assert escaping.called is False


def test_api_rejects_forged_failure_category(monkeypatch) -> None:
    request, registry, provider = _api_fixture()
    request = replace(
        request,
        objective_profile=replace(request.objective_profile, risk_weight=1.0),
    )
    original = provider_module._plan_wheel_sqp_v2

    def forged_failure(*args, **kwargs):
        outcome = original(*args, **kwargs)
        assert type(outcome) is PlanningFailureV2
        object.__setattr__(outcome, "category", FailureCategoryV2.INTERNAL_ERROR)
        return outcome

    _patch_trusted_provider_engine(monkeypatch, forged_failure)
    outcome = _execute(request, registry, provider)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"


def test_genuine_provider_failure_keeps_its_audited_reason_and_decision_binding() -> None:
    request, registry, provider = _api_fixture()
    request = replace(
        request,
        objective_profile=replace(request.objective_profile, risk_weight=1.0),
    )

    outcome = _execute(request, registry, provider)

    assert type(outcome) is PlanningFailureV2
    assert outcome.category is FailureCategoryV2.UNSUPPORTED_CAPABILITY
    assert outcome.reason_code == "wheel_sqp_objective_unsupported"
    assert outcome.search_telemetry.decision_hash is None


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_deadline_after_provider_completion_replaces_success_with_timeout(
    monkeypatch,
) -> None:
    request, registry, provider = _api_fixture()
    clock = _Clock()
    original = provider_module._plan_wheel_sqp_v2

    def expire_after_result(*args, **kwargs):
        outcome = original(*args, **kwargs)
        clock.now = 3.0
        return outcome

    _patch_trusted_provider_engine(monkeypatch, expire_after_result)
    outcome = _execute(request, registry, provider, clock=clock)

    assert type(outcome) is PlanningFailureV2
    assert outcome.category is FailureCategoryV2.TIMEOUT
    assert outcome.reason_code == "planning_deadline_expired"


def test_deadline_during_codec_reseal_replaces_success_with_timeout(
    monkeypatch,
) -> None:
    request, registry, provider = _api_fixture()
    clock = _Clock()
    original = serialization_module.encode_wheel_route_v2
    calls = 0

    def expire_on_api_encode(route):
        nonlocal calls
        calls += 1
        encoded = original(route)
        if calls == 1:
            clock.now = 3.0
        return encoded

    monkeypatch.setattr(
        serialization_module,
        "encode_wheel_route_v2",
        expire_on_api_encode,
    )
    monkeypatch.setattr(
        wheel_sqp_api,
        "_TRUSTED_ROUTE_ENCODER",
        expire_on_api_encode,
    )
    outcome = _execute(request, registry, provider, clock=clock)

    assert calls >= 1
    assert type(outcome) is PlanningFailureV2
    assert outcome.category is FailureCategoryV2.TIMEOUT
    assert outcome.reason_code == "planning_deadline_expired"


def test_observation_projection_accepts_exact_additive_route_with_equal_samples() -> None:
    request, registry, provider = _api_fixture()
    outcome = _execute(request, registry, provider)
    assert type(outcome) is PlanningSuccessV2
    route = outcome.route
    observed = ObservedTerrainInputV2(
        request_id=request.request_id,
        start_state=request.start_state,
        terrain_snapshot=request.terrain_snapshot,
    )
    base_carrier = TypedRouteV2(
        platform_kind=route.platform_kind,
        primitives=route.primitives,
        total_cost=route.total_cost,
        is_complete=True,
    )

    additive = project_route_observation_v2(
        route,
        observed,
        endpoint_theta_rad=request.goal_state.heading_rad,
    )
    base = project_route_observation_v2(
        base_carrier,
        observed,
        endpoint_theta_rad=request.goal_state.heading_rad,
    )
    assert additive == base


def test_sample_drift_changes_both_segment_and_route_hashes() -> None:
    request, registry, provider = _api_fixture()
    outcome = _execute(request, registry, provider)
    assert type(outcome) is PlanningSuccessV2
    route = outcome.route
    segment = route.primitives[0]
    original_segment_hash = wheel_segment_hash_v2(segment)
    original_route_hash = wheel_route_hash_v2(route)
    changed = PoseStateV2(
        segment.end_state.x_m + 0.01,
        segment.end_state.y_m,
        segment.end_state.heading_rad,
    )
    object.__setattr__(segment, "samples", (segment.samples[0], changed))

    assert wheel_segment_hash_v2(segment) != original_segment_hash
    assert wheel_route_hash_v2(route) != original_route_hash
