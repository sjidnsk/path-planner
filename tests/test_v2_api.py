import path_planner
import numpy as np
import pytest

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.search import AStarPlanner
from path_planner.v2.api import plan_v2
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    CacheEvidenceV2,
    CostBreakdownV2,
    FailureCategoryV2,
    FailureEvidenceV2,
    ObjectiveProfileV2,
    ObservationProjectionV2,
    PlanningFailureV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    RoutePrimitiveV2,
    SearchTelemetryV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.profiles import PlatformProfileRegistryV2, PlatformProfileV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


def _profile(
    profile_id: str = "wheel-safe/v1",
    *,
    platform_kind: PlatformKindV2 = PlatformKindV2.WHEEL,
    capability_revision: str = "wheel-capability/v1",
    max_slope: float = 30.0,
) -> PlatformProfileV2:
    return PlatformProfileV2(
        profile_id=profile_id,
        platform_kind=platform_kind,
        capability_revision=capability_revision,
        simulation_proxy=False,
        max_traversable_slope_deg=max_slope,
    )


def _snapshot(
    *,
    slope: np.ndarray | None = None,
    traversable: np.ndarray | None = None,
    hard: np.ndarray | None = None,
    observed: np.ndarray | None = None,
) -> TerrainSnapshotV2:
    shape = (3, 3)
    return TerrainSnapshotV2(
        geometry=FineGridGeometryV2(width=3, height=3, frame_id="moon"),
        elevation_m=np.zeros(shape),
        slope_deg=np.zeros(shape) if slope is None else slope,
        traversable_mask=np.ones(shape, dtype=bool) if traversable is None else traversable,
        hard_obstacle_mask=np.zeros(shape, dtype=bool) if hard is None else hard,
        observed_mask=np.ones(shape, dtype=bool) if observed is None else observed,
        confidence=np.ones(shape),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="api-fixture",
            source_hash="api-fixture-hash",
            physical_obstacle_cells_written=False,
            details=(('scenario', 'gate-1c'),),
        ),
    )


def _request(
    *,
    profile_id: str = "wheel-safe/v1",
    snapshot: object | None = None,
    start: PoseStateV2 | None = None,
    goal: PoseStateV2 | None = None,
) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="request-api-001",
        platform_profile_id=profile_id,
        start_state=start or PoseStateV2(0.25, 0.25, 0.0),
        goal_state=goal or PoseStateV2(1.25, 1.25, 0.0),
        terrain_snapshot=_snapshot() if snapshot is None else snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(),
        timeout_s=5.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=7,
    )


def _telemetry(reason: str = "provider_completed") -> SearchTelemetryV2:
    return SearchTelemetryV2(
        expanded_states=1,
        generated_primitives=1,
        rejected_l0=0,
        rejected_l1=0,
        rejected_l2=0,
        elapsed_s=0.0,
        timed_out=False,
        accelerator_used=False,
        ackermann_feasible_claimed=False,
        termination_reason=reason,
    )


def _success(
    request: PlanningRequestV2,
    *,
    request_id: str | None = None,
    platform_kind: PlatformKindV2 = PlatformKindV2.WHEEL,
    start: PoseStateV2 | None = None,
    end: PoseStateV2 | None = None,
) -> PlanningSuccessV2:
    primitive_kind = {
        PlatformKindV2.WHEEL: PrimitiveKindV2.WHEEL_MOTION,
        PlatformKindV2.LEGGED: PrimitiveKindV2.LEG_STEP,
        PlatformKindV2.HOPPER: PrimitiveKindV2.BALLISTIC_JUMP,
    }[platform_kind]
    primitive = RoutePrimitiveV2(
        kind=primitive_kind,
        start_state=start or request.start_state,
        end_state=end or request.goal_state,
        duration_s=1.0,
        distance_m=1.0,
        energy_cost=0.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
    )
    return PlanningSuccessV2(
        request_id=request_id or request.request_id,
        platform_kind=platform_kind,
        route=TypedRouteV2(platform_kind, (primitive,), 1.0),
        observation_projection=ObservationProjectionV2(
            source="api-test/v1",
            sample_states=(primitive.start_state, primitive.end_state),
            expected_new_observed_cells=0.0,
            expected_information_gain=0.0,
        ),
        cost_breakdown=CostBreakdownV2(1.0, 0.0, 0.0, 0.0, 1.0),
        validation_evidence=ValidationEvidenceV2(
            "api-test-validator/v1",
            ValidationLevelV2.L2,
            True,
            ("terrain_safe",),
        ),
        search_telemetry=_telemetry(),
        cache_evidence=CacheEvidenceV2("api-test", request.request_id, False),
    )


def _failure(
    request: PlanningRequestV2,
    *,
    platform_kind: PlatformKindV2 | None,
) -> PlanningFailureV2:
    if platform_kind is None:
        category = FailureCategoryV2.UNSUPPORTED_CAPABILITY
        reason = "platform_profile_unresolved"
        stage = "profile_resolution"
    else:
        category = FailureCategoryV2.NO_COMPLETE_ROUTE
        reason = "provider_no_route"
        stage = "provider_search"
    return PlanningFailureV2(
        request_id=request.request_id,
        platform_kind=platform_kind,
        category=category,
        reason_code=reason,
        evidence=FailureEvidenceV2(stage, (reason,), ()),
        search_telemetry=_telemetry(reason),
    )


class _ProviderSpy:
    def __init__(self, profile: PlatformProfileV2, outcome_or_factory) -> None:
        self.profile = profile
        self.outcome_or_factory = outcome_or_factory
        self.calls: list[tuple[PlanningRequestV2, FineSafetyAnchorV2]] = []

    def plan(self, request: PlanningRequestV2, anchor: FineSafetyAnchorV2):
        self.calls.append((request, anchor))
        if callable(self.outcome_or_factory):
            return self.outcome_or_factory(request)
        return self.outcome_or_factory


def test_unknown_profile_is_the_only_unresolved_shape_and_is_byte_stable() -> None:
    request = _request(profile_id="unknown/v1")
    registry = PlatformProfileRegistryV2((_profile(),))

    first = plan_v2(request, registry=registry, providers={})
    second = plan_v2(request, registry=registry, providers={})

    assert isinstance(first, PlanningFailureV2)
    assert first.platform_kind is None
    assert first.category is FailureCategoryV2.UNSUPPORTED_CAPABILITY
    assert first.reason_code == "platform_profile_unresolved"
    assert first.evidence.stage == "profile_resolution"
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_single_request_calls_only_the_exact_registered_provider_once() -> None:
    request = _request()
    target = _profile()
    other = _profile("legged-safe/v1", platform_kind=PlatformKindV2.LEGGED)
    target_provider = _ProviderSpy(target, lambda value: _success(value))
    other_provider = _ProviderSpy(other, AssertionError("wrong provider called"))

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((other, target)),
        providers={target.profile_id: target_provider, other.profile_id: other_provider},
    )

    assert isinstance(outcome, PlanningSuccessV2)
    assert len(target_provider.calls) == 1
    assert target_provider.calls[0][0] is request
    assert isinstance(target_provider.calls[0][1], FineSafetyAnchorV2)
    assert other_provider.calls == []


@pytest.mark.parametrize("providers", [{}, {"wrong-key/v1": object()}])
def test_missing_or_wrongly_keyed_provider_fails_closed_without_v1_fallback(
    providers,
    monkeypatch,
) -> None:
    request = _request()
    profile = _profile()

    def forbidden_v1(*args, **kwargs):
        raise AssertionError("v1 fallback called")

    monkeypatch.setattr(AStarPlanner, "plan", forbidden_v1)
    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers=providers,
    )

    assert isinstance(outcome, PlanningFailureV2)
    assert outcome.platform_kind is PlatformKindV2.WHEEL
    assert outcome.category is FailureCategoryV2.UNSUPPORTED_CAPABILITY
    assert outcome.reason_code == "primitive_provider_unregistered"


def test_provider_profile_must_exactly_match_registry_profile() -> None:
    request = _request()
    registered = _profile()
    mismatched = _profile(capability_revision="wheel-capability/v2")
    provider = _ProviderSpy(mismatched, lambda value: _success(value))

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((registered,)),
        providers={registered.profile_id: provider},
    )

    assert outcome.category is FailureCategoryV2.UNSUPPORTED_CAPABILITY
    assert outcome.reason_code == "primitive_provider_profile_mismatch"
    assert provider.calls == []


@pytest.mark.parametrize(
    ("endpoint", "reason", "category"),
    [
        ("start_out", "terrain_out_of_bounds", FailureCategoryV2.UNSAFE_START),
        ("start_unknown", "terrain_unknown", FailureCategoryV2.UNSAFE_START),
        ("start_hard", "terrain_hard_obstacle", FailureCategoryV2.UNSAFE_START),
        ("start_slope", "terrain_slope_exceeded", FailureCategoryV2.UNSAFE_START),
        ("goal_out", "terrain_out_of_bounds", FailureCategoryV2.UNSAFE_GOAL),
        ("goal_unknown", "terrain_unknown", FailureCategoryV2.UNSAFE_GOAL),
    ],
)
def test_preflight_fails_before_provider_and_preserves_anchor_reason(
    endpoint,
    reason,
    category,
) -> None:
    observed = np.ones((3, 3), dtype=bool)
    hard = np.zeros((3, 3), dtype=bool)
    traversable = np.ones((3, 3), dtype=bool)
    slope = np.zeros((3, 3))
    start = PoseStateV2(0.25, 0.25, 0.0)
    goal = PoseStateV2(1.25, 1.25, 0.0)
    if endpoint == "start_out":
        start = PoseStateV2(-0.25, 0.25, 0.0)
    elif endpoint == "start_unknown":
        observed[0, 0] = False
    elif endpoint == "start_hard":
        hard[0, 0] = True
        traversable[0, 0] = False
    elif endpoint == "start_slope":
        slope[0, 0] = 31.0
    elif endpoint == "goal_out":
        goal = PoseStateV2(1.75, 1.25, 0.0)
    elif endpoint == "goal_unknown":
        observed[2, 2] = False
    snapshot = _snapshot(
        slope=slope,
        traversable=traversable,
        hard=hard,
        observed=observed,
    )
    request = _request(snapshot=snapshot, start=start, goal=goal)
    profile = _profile()
    provider = _ProviderSpy(profile, lambda value: _success(value))

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
    )

    assert outcome.category is category
    assert outcome.reason_code == reason
    assert outcome.evidence.stage in {"start_safety", "goal_safety"}
    assert provider.calls == []


def test_exact_terrain_snapshot_is_required_as_typed_invalid_request() -> None:
    request = _request(snapshot=object())
    profile = _profile()

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={},
    )

    assert outcome.category is FailureCategoryV2.INVALID_REQUEST
    assert outcome.reason_code == "terrain_snapshot_invalid"
    assert outcome.platform_kind is PlatformKindV2.WHEEL


def test_provider_exception_records_only_stable_exception_type() -> None:
    request = _request()
    profile = _profile()

    def fail(_request):
        raise LookupError("unstable message contents")

    provider = _ProviderSpy(profile, fail)
    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
    )

    assert outcome.category is FailureCategoryV2.INTERNAL_ERROR
    assert outcome.reason_code == "primitive_provider_exception"
    assert outcome.evidence.details == (("exception_type", "LookupError"),)
    assert "unstable message contents" not in canonical_json_bytes(outcome).decode("utf-8")


@pytest.mark.parametrize(
    "outcome_factory",
    [
        lambda request: object(),
        lambda request: _success(request, request_id="other-request"),
        lambda request: _success(request, platform_kind=PlatformKindV2.LEGGED),
        lambda request: _success(request, start=PoseStateV2(0.75, 0.25, 0.0)),
        lambda request: _success(request, end=PoseStateV2(0.75, 1.25, 0.0)),
        lambda request: _failure(request, platform_kind=None),
        lambda request: _failure(request, platform_kind=PlatformKindV2.LEGGED),
    ],
)
def test_invalid_provider_outcomes_are_replaced_by_stable_postcondition_failure(
    outcome_factory,
) -> None:
    request = _request()
    profile = _profile()
    provider = _ProviderSpy(profile, outcome_factory)

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
    )

    assert outcome.category is FailureCategoryV2.INTERNAL_ERROR
    assert outcome.reason_code == "primitive_provider_outcome_invalid"
    assert outcome.evidence.stage == "provider_postcondition"


def test_valid_known_platform_failure_is_preserved() -> None:
    request = _request()
    profile = _profile()
    expected = _failure(request, platform_kind=PlatformKindV2.WHEEL)
    provider = _ProviderSpy(profile, expected)

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
    )

    assert outcome is expected
    assert len(provider.calls) == 1


def test_api_argument_type_errors_do_not_become_typed_request_failures() -> None:
    request = _request()
    registry = PlatformProfileRegistryV2((_profile(),))

    with pytest.raises(TypeError, match="request"):
        plan_v2(object(), registry=registry, providers={})
    with pytest.raises(TypeError, match="registry"):
        plan_v2(request, registry=object(), providers={})
    with pytest.raises(TypeError, match="providers"):
        plan_v2(request, registry=registry, providers=[])


def test_plan_v2_remains_opt_in_and_v1_route_schema_is_unchanged() -> None:
    spec = GridSpec(width=2, height=2, resolution=0.5)
    grid = CostGrid(
        spec=spec,
        cost=np.ones((2, 2)),
        passable_mask=np.ones((2, 2), dtype=bool),
    )
    route = AStarPlanner().plan(
        grid,
        PlanRequest(start=Cell(0, 0), goal=Cell(1, 1)),
    ).to_route_dict(spec)

    assert "plan_v2" not in path_planner.__dict__
    assert route["schema_version"] == "path-planner-route/v1"
