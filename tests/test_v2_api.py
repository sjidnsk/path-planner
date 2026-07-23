from dataclasses import FrozenInstanceError
from math import fsum, hypot, nextafter, pi

import path_planner
import path_planner.v2 as v2
import path_planner.v2.api as api_module
import numpy as np
import pytest

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest, WorldPoint
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
from path_planner.v2.oracles.legged import (
    LEGGED_FOOT_STORAGE_ORDER_V2,
    LeggedFootContactV2,
    LegIdV2,
)
from path_planner.v2.providers.legged import (
    LeggedSearchStateV2,
    LeggedStepPrimitiveV2,
    nominal_legged_search_state_v2,
)
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
    goal_position_tolerance_m: float = 0.0,
    goal_heading_tolerance_rad: float = 0.0,
) -> PlatformProfileV2:
    return PlatformProfileV2(
        profile_id=profile_id,
        platform_kind=platform_kind,
        capability_revision=capability_revision,
        simulation_proxy=False,
        max_traversable_slope_deg=max_slope,
        goal_position_tolerance_m=goal_position_tolerance_m,
        goal_heading_tolerance_rad=goal_heading_tolerance_rad,
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
    timeout_s: float = 5.0,
) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="request-api-001",
        platform_profile_id=profile_id,
        start_state=start or PoseStateV2(0.25, 0.25, 0.0),
        goal_state=goal or PoseStateV2(1.25, 1.25, 0.0),
        terrain_snapshot=_snapshot() if snapshot is None else snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(),
        timeout_s=timeout_s,
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
        self.calls: list[tuple[PlanningRequestV2, FineSafetyAnchorV2, object]] = []

    def plan(self, request: PlanningRequestV2, anchor: FineSafetyAnchorV2, deadline):
        self.calls.append((request, anchor, deadline))
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
    deadline = target_provider.calls[0][2]
    assert isinstance(deadline, v2.PlanningDeadlineV2)
    assert deadline.deadline_monotonic_s - deadline.started_monotonic_s == 2.0
    assert other_provider.calls == []


def test_old_wheel_capability_never_enters_wheel_sqp_dispatch(monkeypatch) -> None:
    import path_planner.v2.wheel_sqp_api as wheel_sqp_api

    request = _request()
    profile = _profile()
    provider = _ProviderSpy(profile, lambda value: _success(value))

    def forbidden(*args, **kwargs):
        raise AssertionError("old wheel capability must not enter wheel SQP dispatch")

    monkeypatch.setattr(wheel_sqp_api, "dispatch_wheel_sqp_provider_v2", forbidden)
    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
    )

    assert type(outcome) is PlanningSuccessV2
    assert len(provider.calls) == 1


class _FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self) -> float:
        self._last = next(self._values, self._last)
        return self._last


def test_planning_deadline_is_immutable_slotted_and_uses_shared_clock() -> None:
    clock = _FakeClock(10.0)
    deadline = v2.PlanningDeadlineV2(10.0, 12.0, clock)

    assert not hasattr(deadline, "__dict__")
    assert "clock" not in repr(deadline)
    assert deadline == v2.PlanningDeadlineV2(10.0, 12.0, _FakeClock(99.0))
    assert deadline.elapsed_s == 0.0
    assert deadline.remaining_s == 2.0
    assert deadline.expired is False
    clock.now = 12.5
    assert deadline.elapsed_s == 2.5
    assert deadline.remaining_s == 0.0
    assert deadline.expired is True
    with pytest.raises(FrozenInstanceError):
        deadline.deadline_monotonic_s = 13.0


@pytest.mark.parametrize(
    ("request_timeout_s", "effective_timeout_s"),
    [(5.0, 2.0), (0.25, 0.25)],
)
def test_provider_receives_exact_shared_deadline_with_effective_timeout(
    request_timeout_s,
    effective_timeout_s,
) -> None:
    clock = _FakeClock(10.0)
    request = _request(timeout_s=request_timeout_s)
    profile = _profile()
    provider = _ProviderSpy(profile, lambda value: _success(value))

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
        monotonic_clock=clock,
    )

    assert isinstance(outcome, PlanningSuccessV2)
    assert len(provider.calls) == 1
    called_request, anchor, deadline = provider.calls[0]
    assert called_request is request
    assert isinstance(anchor, FineSafetyAnchorV2)
    assert deadline.started_monotonic_s == 10.0
    assert deadline.deadline_monotonic_s == 10.0 + effective_timeout_s


def test_expired_deadline_before_provider_dispatch_returns_typed_timeout() -> None:
    clock = _FakeClock(10.0)
    request = _request(timeout_s=0.0)
    profile = _profile()
    provider = _ProviderSpy(profile, lambda value: _success(value))

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
        monotonic_clock=clock,
    )

    assert isinstance(outcome, PlanningFailureV2)
    assert outcome.category is FailureCategoryV2.TIMEOUT
    assert outcome.reason_code == "planning_deadline_expired"
    assert outcome.evidence.stage == "provider_dispatch"
    assert outcome.search_telemetry.timed_out is True
    assert outcome.search_telemetry.elapsed_s == 0.0
    assert provider.calls == []


def test_late_provider_success_is_replaced_by_typed_timeout() -> None:
    clock = _FakeClock(10.0)
    request = _request(timeout_s=0.25)
    profile = _profile()

    def late_success(value):
        clock.now = 10.5
        return _success(value)

    provider = _ProviderSpy(profile, late_success)
    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
        monotonic_clock=clock,
    )

    assert isinstance(outcome, PlanningFailureV2)
    assert outcome.category is FailureCategoryV2.TIMEOUT
    assert outcome.reason_code == "planning_deadline_expired"
    assert outcome.evidence.stage == "provider_completion"
    assert outcome.search_telemetry.timed_out is True
    assert outcome.search_telemetry.elapsed_s == 0.5
    assert len(provider.calls) == 1


def test_success_expiring_during_postcondition_is_replaced_by_timeout() -> None:
    clock = _SequenceClock(10.0, 10.0, 10.0, 10.5, 10.5)
    request = _request(timeout_s=0.25)
    profile = _profile()
    provider = _ProviderSpy(profile, lambda value: _success(value))

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
        monotonic_clock=clock,
    )

    assert isinstance(outcome, PlanningFailureV2)
    assert outcome.category is FailureCategoryV2.TIMEOUT
    assert outcome.evidence.stage == "provider_postcondition"
    assert outcome.search_telemetry.elapsed_s == 0.5


def test_plan_v2_rejects_noncallable_clock_before_first_call() -> None:
    request = _request()
    registry = PlatformProfileRegistryV2((_profile(),))

    with pytest.raises(TypeError, match="monotonic_clock must be callable"):
        plan_v2(request, registry=registry, providers={}, monotonic_clock=None)


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


@pytest.mark.parametrize(
    ("profile", "goal", "actual_end"),
    [
        (
            _profile(goal_position_tolerance_m=0.25),
            PoseStateV2(1.25, 1.25, 0.0),
            PoseStateV2(1.5, 1.25, 0.0),
        ),
        (
            _profile(goal_heading_tolerance_rad=0.5),
            PoseStateV2(1.25, 1.25, 2.0 * pi - 0.25),
            PoseStateV2(1.25, 1.25, 0.25),
        ),
    ],
)
def test_goal_tolerance_closed_boundary_passes_without_endpoint_mutation(
    profile,
    goal,
    actual_end,
) -> None:
    request = _request(goal=goal)
    provider = _ProviderSpy(profile, lambda value: _success(value, end=actual_end))

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
    )

    assert isinstance(outcome, PlanningSuccessV2)
    assert outcome.route.primitives[-1].end_state is actual_end


@pytest.mark.parametrize(
    ("profile", "goal", "actual_end"),
    [
        (
            _profile(goal_position_tolerance_m=0.25),
            PoseStateV2(1.25, 1.25, 0.0),
            PoseStateV2(nextafter(1.5, float("inf")), 1.25, 0.0),
        ),
        (
            _profile(goal_heading_tolerance_rad=nextafter(0.5, 0.0)),
            PoseStateV2(1.25, 1.25, 2.0 * pi - 0.25),
            PoseStateV2(1.25, 1.25, 0.25),
        ),
    ],
)
def test_goal_tolerance_just_outside_is_rejected(profile, goal, actual_end) -> None:
    request = _request(goal=goal)
    provider = _ProviderSpy(profile, lambda value: _success(value, end=actual_end))

    outcome = plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: provider},
    )

    assert isinstance(outcome, PlanningFailureV2)
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


def test_gate3_accelerator_contracts_are_public_only_under_v2_namespace() -> None:
    expected = {
        "HIERARCHY_SCALES_V2",
        "ConservativeHierarchyV2",
        "HierarchyCellHintV2",
        "HierarchyContractErrorV2",
        "HierarchyHintStatusV2",
        "SearchQueueEntryV2",
        "StableSearchQueueV2",
    }

    assert expected <= set(v2.__all__)
    assert all(name in v2.__dict__ for name in expected)
    assert all(name not in path_planner.__dict__ for name in expected)


def _api_legged_profile() -> PlatformProfileV2:
    return PlatformProfileV2(
        profile_id="legged-static-crawl/v1",
        platform_kind=PlatformKindV2.LEGGED,
        capability_revision="simulation_proxy_static_crawl/v1",
        simulation_proxy=True,
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=0.0,
        goal_heading_tolerance_rad=0.0,
    )


def _api_legged_request(*, heading: float = 0.0) -> PlanningRequestV2:
    start = PoseStateV2(1.25, 1.25, heading)
    return _request(
        profile_id="legged-static-crawl/v1",
        start=start,
        goal=PoseStateV2(1.1875, 1.25, heading),
    )


def _api_legged_success(request: PlanningRequestV2) -> PlanningSuccessV2:
    start = nominal_legged_search_state_v2(request.start_state)
    moving_leg = LegIdV2.FRONT_LEFT
    moving_index = LEGGED_FOOT_STORAGE_ORDER_V2.index(moving_leg)
    source = start.foot_contacts[moving_index].foothold
    target = type(source)(source.x - 0.25, source.y)
    nonmoving = tuple(
        contact.foothold
        for index, contact in enumerate(start.foot_contacts)
        if index != moving_index
    )
    lift = PoseStateV2(
        fsum(point.x for point in nonmoving) / 3.0,
        fsum(point.y for point in nonmoving) / 3.0,
        start.body_state.heading_rad,
    )
    contacts = list(start.foot_contacts)
    contacts[moving_index] = LeggedFootContactV2(moving_leg, target)
    end_body = PoseStateV2(
        fsum(contact.foothold.x for contact in contacts) / 4.0,
        fsum(contact.foothold.y for contact in contacts) / 4.0,
        start.body_state.heading_rad,
    )
    end = LeggedSearchStateV2(end_body, tuple(contacts), 1)
    foot_travel = hypot(target.x - source.x, target.y - source.y)
    distance = hypot(
        lift.x_m - start.body_state.x_m,
        lift.y_m - start.body_state.y_m,
    ) + hypot(end_body.x_m - lift.x_m, end_body.y_m - lift.y_m)
    primitive = LeggedStepPrimitiveV2(
        kind=PrimitiveKindV2.LEG_STEP,
        start_state=start.body_state,
        end_state=end_body,
        duration_s=1.0,
        distance_m=distance,
        energy_cost=distance + foot_travel,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        start_legged_state=start,
        lift_body_state=lift,
        end_legged_state=end,
        moving_leg=moving_leg,
        target_foothold=target,
        foot_travel_m=foot_travel,
    )
    objective = request.objective_profile
    distance_cost = objective.distance_weight * primitive.distance_m
    energy_cost = objective.energy_weight * primitive.energy_cost
    time_cost = objective.time_weight * primitive.duration_s
    total = sum((distance_cost, 0.0, energy_cost, time_cost))
    return PlanningSuccessV2(
        request_id=request.request_id,
        platform_kind=PlatformKindV2.LEGGED,
        route=TypedRouteV2(PlatformKindV2.LEGGED, (primitive,), total),
        observation_projection=ObservationProjectionV2(
            source="legged_route_body_samples_gain_not_computed/v1",
            sample_states=(primitive.start_state, lift, primitive.end_state),
            expected_new_observed_cells=0.0,
            expected_information_gain=0.0,
        ),
        cost_breakdown=CostBreakdownV2(
            distance_cost,
            0.0,
            energy_cost,
            time_cost,
            total,
        ),
        validation_evidence=ValidationEvidenceV2(
            "path-planner-v2-legged-route-l2/v1",
            ValidationLevelV2.L2,
            True,
            ("legged_route_l2_valid",),
        ),
        search_telemetry=_telemetry("legged_route_l2_valid"),
        cache_evidence=CacheEvidenceV2(
            "path-planner-v2-legged-cache-disabled/v1",
            "legged-cache-disabled/v1",
            False,
        ),
    )


def _plan_legged_outcome(
    request: PlanningRequestV2,
    outcome: PlanningSuccessV2,
    *,
    monotonic_clock=None,
):
    profile = _api_legged_profile()
    kwargs = {}
    if monotonic_clock is not None:
        kwargs["monotonic_clock"] = monotonic_clock
    return plan_v2(
        request,
        registry=PlatformProfileRegistryV2((profile,)),
        providers={profile.profile_id: _ProviderSpy(profile, outcome)},
        **kwargs,
    )


@pytest.mark.parametrize("heading", [0.0, 2.0 * pi, -3.0 * pi])
def test_api_accepts_exact_typed_legged_success_with_canonical_start_and_goal(
    heading: float,
) -> None:
    request = _api_legged_request(heading=heading)
    expected = _api_legged_success(request)
    outcome = _plan_legged_outcome(request, expected)
    assert outcome is expected


def test_api_rejects_leg_kind_base_primitive_for_legged_success() -> None:
    request = _api_legged_request()
    forged = _success(
        request,
        platform_kind=PlatformKindV2.LEGGED,
        start=request.start_state,
        end=request.goal_state,
    )
    outcome = _plan_legged_outcome(request, forged)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "primitive_provider_outcome_invalid"


def test_api_rejects_legged_full_state_discontinuity_even_when_body_is_connected() -> None:
    request = _api_legged_request()
    forged = _api_legged_success(request)
    primitive = forged.route.primitives[0]
    assert type(primitive) is LeggedStepPrimitiveV2
    contacts = list(primitive.start_legged_state.foot_contacts)
    contacts[1] = LeggedFootContactV2(
        contacts[1].leg_id,
        type(contacts[1].foothold)(
            contacts[1].foothold.x + 0.25,
            contacts[1].foothold.y,
        ),
    )
    object.__setattr__(primitive.start_legged_state, "foot_contacts", tuple(contacts))
    outcome = _plan_legged_outcome(request, forged)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "primitive_provider_outcome_invalid"


@pytest.mark.parametrize(
    "declared_field",
    [
        "distance_cost",
        "risk_cost",
        "energy_cost",
        "time_cost",
        "total_cost",
        "route.total_cost",
    ],
)
def test_api_rejects_legged_cost_one_ulp_drift_despite_contract_tolerance(
    declared_field: str,
) -> None:
    request = _api_legged_request()
    forged = _api_legged_success(request)
    if declared_field == "route.total_cost":
        current = forged.route.total_cost
        object.__setattr__(
            forged.route,
            "total_cost",
            nextafter(current, float("inf")),
        )
    else:
        current = getattr(forged.cost_breakdown, declared_field)
        object.__setattr__(
            forged.cost_breakdown,
            declared_field,
            nextafter(current, float("inf")),
        )
    outcome = _plan_legged_outcome(request, forged)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "primitive_provider_outcome_invalid"


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("validator_id", "forged-legged-route-l2/v1"),
        ("level", ValidationLevelV2.L1),
        ("passed", False),
        ("checks", ("forged-legged-route-l2",)),
    ],
)
def test_api_rejects_forged_legged_final_evidence_without_replaying_validator(
    field: str,
    forged_value: object,
) -> None:
    request = _api_legged_request()
    forged = _api_legged_success(request)
    object.__setattr__(
        forged.validation_evidence,
        field,
        forged_value,
    )
    outcome = _plan_legged_outcome(request, forged)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "primitive_provider_outcome_invalid"


def _mutate_api_legged_payload(kind: str, outcome: PlanningSuccessV2) -> None:
    primitive = outcome.route.primitives[0]
    assert type(primitive) is LeggedStepPrimitiveV2
    if kind == "target":
        target = primitive.target_foothold
        object.__setattr__(
            primitive,
            "target_foothold",
            WorldPoint(target.x + 0.25, target.y),
        )
    elif kind == "full_state":
        object.__setattr__(primitive.end_legged_state, "sequence_phase", 3)
    elif kind == "resource":
        object.__setattr__(
            primitive,
            "energy_cost",
            nextafter(primitive.energy_cost, float("inf")),
        )
    elif kind == "route":
        object.__setattr__(
            outcome.route,
            "total_cost",
            nextafter(outcome.route.total_cost, float("inf")),
        )
    elif kind == "evidence":
        object.__setattr__(
            outcome.validation_evidence,
            "checks",
            ("forged-final-check",),
        )
    else:  # pragma: no cover - test helper contract
        raise AssertionError(kind)


@pytest.mark.parametrize("clock_index", [3, 4])
@pytest.mark.parametrize(
    "mutation_kind",
    ["target", "full_state", "resource", "route", "evidence"],
)
def test_api_legged_post_provider_clocks_cannot_change_final_bound_payload(
    clock_index: int,
    mutation_kind: str,
) -> None:
    request = _api_legged_request()
    forged = _api_legged_success(request)
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        if calls == clock_index:
            _mutate_api_legged_payload(mutation_kind, forged)
        return 0.0

    outcome = _plan_legged_outcome(
        request,
        forged,
        monotonic_clock=clock,
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "primitive_provider_outcome_invalid"
    assert outcome.evidence.stage == "provider_postcondition"
    assert calls == 4


def test_api_legged_final_reseal_does_not_rerun_route_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import path_planner.v2.validation as validation_module

    request = _api_legged_request()
    expected = _api_legged_success(request)

    calls: list[str] = []

    def forbidden_public(*_args):
        calls.append("public")
        raise AssertionError("API must not rerun final validator")

    def forbidden_private(*_args):
        calls.append("private")
        raise AssertionError("API must not rerun trusted final validator")

    monkeypatch.setattr(
        validation_module,
        "validate_legged_route_l2",
        forbidden_public,
    )
    monkeypatch.setattr(
        validation_module,
        "_TRUSTED_VALIDATE_LEGGED_ROUTE_L2_V2",
        forbidden_private,
    )
    outcome = _plan_legged_outcome(request, expected)
    assert outcome is expected
    assert calls == []


@pytest.mark.parametrize(
    "resource_name",
    ["distance_m", "energy_cost", "duration_s"],
)
def test_api_legged_resource_second_read_detects_calculation_boundary_drift(
    monkeypatch: pytest.MonkeyPatch,
    resource_name: str,
) -> None:
    request = _api_legged_request()
    expected = _api_legged_success(request)
    primitive = expected.route.primitives[0]
    original = getattr(primitive, resource_name)
    real_component = api_module._legged_checked_component_v2
    component_reads: list[str] = []
    mutated = False

    def drifting_component(parent, weight, resource):
        nonlocal mutated
        component_reads.append(resource.hex())
        result = real_component(parent, weight, resource)
        if not mutated and resource.hex() == original.hex():
            object.__setattr__(
                primitive,
                resource_name,
                nextafter(original, float("inf")),
            )
            mutated = True
        return result

    monkeypatch.setattr(
        api_module,
        "_legged_checked_component_v2",
        drifting_component,
    )
    outcome = _plan_legged_outcome(request, expected)
    assert mutated is True
    assert original.hex() in component_reads
    assert getattr(primitive, resource_name).hex() == nextafter(
        original,
        float("inf"),
    ).hex()
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "primitive_provider_outcome_invalid"
    assert outcome.evidence.stage == "provider_postcondition"


@pytest.mark.parametrize("resource_name", ["distance_m", "energy_cost", "duration_s"])
@pytest.mark.parametrize("bad_value", [1, True, -0.0, float("inf")])
def test_api_legged_zero_weight_still_pre_audits_every_resource(
    resource_name: str,
    bad_value,
) -> None:
    request = _api_legged_request()
    objective = request.objective_profile
    weight_name = {
        "distance_m": "distance_weight",
        "energy_cost": "energy_weight",
        "duration_s": "time_weight",
    }[resource_name]
    object.__setattr__(objective, weight_name, 0.0)
    expected = _api_legged_success(request)
    primitive = expected.route.primitives[0]
    object.__setattr__(primitive, resource_name, bad_value)
    outcome = _plan_legged_outcome(request, expected)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "primitive_provider_outcome_invalid"


def _api_hopper_fixture():
    from math import cos, sin

    from path_planner.v2.hopper_authority import (
        HOPPER_GATE5B_ALGORITHM_FIXTURE_V1,
        HopperProviderAuthorityV2,
        hopper_gate5b_algorithm_fixture_v1,
    )
    from path_planner.v2.providers.hopper import HopperPrimitiveProviderV2

    profile = hopper_gate5b_algorithm_fixture_v1()
    authority = HopperProviderAuthorityV2(
        profile,
        HOPPER_GATE5B_ALGORITHM_FIXTURE_V1.parameter_set_id,
        "hopper-provider-authority/v1",
    )
    provider = HopperPrimitiveProviderV2(authority)
    shape = (64, 64)
    snapshot = TerrainSnapshotV2(
        geometry=FineGridGeometryV2(64, 64, (-8.0, -8.0), "moon"),
        elevation_m=np.zeros(shape, dtype=np.float64),
        slope_deg=np.zeros(shape, dtype=np.float64),
        traversable_mask=np.ones(shape, dtype=bool),
        hard_obstacle_mask=np.zeros(shape, dtype=bool),
        observed_mask=np.ones(shape, dtype=bool),
        confidence=np.ones(shape, dtype=np.float64),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="gate5b-api-fixture",
            source_hash="gate5b-api-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )
    start = PoseStateV2(0.25, 0.25, 0.0)
    speed = profile.launch_speeds_mps[0]
    elevation = profile.launch_elevations_rad[0]
    flight_time = 2.0 * ((speed * sin(elevation)) / profile.gravity_mps2)
    distance = (speed * cos(elevation)) * flight_time
    goal = PoseStateV2(start.x_m, start.y_m + distance, 0.0)
    request = PlanningRequestV2(
        request_id="gate5b-api",
        platform_profile_id=profile.profile.profile_id,
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(10, 100, 0),
        timeout_s=2.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=23,
    )
    registry = PlatformProfileRegistryV2((profile.profile,))
    return request, registry, provider


def test_api_hopper_public_export_delta_is_exact_and_identity_preserving() -> None:
    import path_planner.v2.oracles as oracles
    import path_planner.v2.providers as providers
    import path_planner.v2.validation as validation
    from path_planner.v2.hopper_authority import HopperProviderAuthorityV2

    expected_oracles = (
        "HopperJumpCandidateV2",
        "HopperValidationResultV2",
        "validate_hopper_jump_l2",
    )
    expected_providers = (
        "HopperSearchStateV2",
        "HopperJumpPrimitiveV2",
        "HopperPrimitiveProviderV2",
        "hopper_state_key_v2",
        "nominal_hopper_search_state_v2",
    )
    expected_validation = (
        "HOPPER_ROUTE_VALIDATOR_ID_V2",
        "HopperRouteProbabilityDiagnosticV2",
        "validate_hopper_route_l2",
    )
    for name in expected_oracles:
        assert name in oracles.__all__
        assert getattr(v2, name) is getattr(oracles, name)
    for name in expected_providers:
        assert name in providers.__all__
        assert getattr(v2, name) is getattr(providers, name)
    for name in expected_validation:
        assert getattr(v2, name) is getattr(validation, name)
    assert v2.HopperProviderAuthorityV2 is HopperProviderAuthorityV2
    for private in (
        "HopperParameterSetRecordV2",
        "HopperResourceAuthorityV2",
        "HOPPER_PARAMETER_SET_REGISTRY_V2",
        "HOPPER_RESOURCE_AUTHORITY_V2",
    ):
        assert private not in v2.__all__
        assert not hasattr(v2, private)


def test_api_hopper_fixture_success_runs_through_public_plan_v2() -> None:
    request, registry, provider = _api_hopper_fixture()
    outcome = plan_v2(
        request,
        registry=registry,
        providers={provider.profile.profile_id: provider},
        monotonic_clock=lambda: 0.0,
    )
    assert type(outcome) is PlanningSuccessV2
    assert outcome.platform_kind is PlatformKindV2.HOPPER
    assert outcome.validation_evidence.checks == ("hopper_route_l2_valid",)
    assert type(outcome.route.primitives[0]) is v2.HopperJumpPrimitiveV2


def test_api_hopper_claimed_success_is_independently_replayed() -> None:
    request, registry, provider = _api_hopper_fixture()

    class ForgingProvider:
        profile = provider.profile
        hopper_authority = provider.hopper_authority
        hopper_resource_authority = provider.hopper_resource_authority

        def plan(self, request, anchor, deadline):
            outcome = provider.plan(request, anchor, deadline)
            assert type(outcome) is PlanningSuccessV2
            object.__setattr__(
                outcome.route.primitives[0], "selected_landing_mass", 0.999
            )
            return outcome

    outcome = plan_v2(
        request,
        registry=registry,
        providers={provider.profile.profile_id: ForgingProvider()},
        monotonic_clock=lambda: 0.0,
    )
    assert type(outcome) is PlanningFailureV2
    assert outcome.category is FailureCategoryV2.INTERNAL_ERROR
    assert outcome.reason_code == "hopper_provider_outcome_contract_mismatch"
    assert outcome.evidence.stage == "provider_postcondition"
    assert outcome.evidence.details == (
        ("actual", "hopper_primitive_contract_mismatch"),
        ("expected", "hopper_route_l2_valid"),
        ("phase", "provider_postcondition"),
    )
