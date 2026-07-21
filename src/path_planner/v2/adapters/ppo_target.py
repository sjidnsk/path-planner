from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real

from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningFailureV2,
    PlanningOutcomeV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.observation import (
    ObservedTerrainInputV2,
    project_route_observation_v2,
)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


@dataclass(frozen=True, slots=True)
class PpoTargetV2:
    x_m: float
    y_m: float
    theta_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _finite_real(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _finite_real(self.y_m, "y_m"))
        object.__setattr__(self, "theta_rad", _finite_real(self.theta_rad, "theta_rad"))


BuildPpoRequestFn = Callable[
    [
        PpoTargetV2,
        ObservedTerrainInputV2,
        str,
        ObjectiveProfileV2,
        ResourceBudgetV2,
        float,
        AcceleratorPolicyV2,
        int,
    ],
    PlanningRequestV2,
]
PlanPpoTargetFn = Callable[[PlanningRequestV2], PlanningOutcomeV2]


def build_ppo_request_v2(
    target: PpoTargetV2,
    observed_terrain: ObservedTerrainInputV2,
    platform_profile_id: str,
    objective_profile: ObjectiveProfileV2,
    resource_budget: ResourceBudgetV2,
    timeout_s: float,
    accelerator_policy: AcceleratorPolicyV2,
    determinism_seed: int,
) -> PlanningRequestV2:
    if type(target) is not PpoTargetV2:
        raise TypeError("target must be exact PpoTargetV2")
    if type(observed_terrain) is not ObservedTerrainInputV2:
        raise TypeError("observed_terrain must be exact ObservedTerrainInputV2")
    return PlanningRequestV2(
        request_id=observed_terrain.request_id,
        platform_profile_id=platform_profile_id,
        start_state=observed_terrain.start_state,
        goal_state=PoseStateV2(target.x_m, target.y_m, target.theta_rad),
        terrain_snapshot=observed_terrain.terrain_snapshot,
        objective_profile=objective_profile,
        resource_budget=resource_budget,
        timeout_s=timeout_s,
        accelerator_policy=accelerator_policy,
        determinism_seed=determinism_seed,
    )


def plan_ppo_target_v2(
    target: PpoTargetV2,
    observed_terrain: ObservedTerrainInputV2,
    platform_profile_id: str,
    objective_profile: ObjectiveProfileV2,
    resource_budget: ResourceBudgetV2,
    timeout_s: float,
    accelerator_policy: AcceleratorPolicyV2,
    determinism_seed: int,
    *,
    plan_request: PlanPpoTargetFn,
) -> PlanningOutcomeV2:
    if not callable(plan_request):
        raise TypeError("plan_request must be callable")
    request = build_ppo_request_v2(
        target,
        observed_terrain,
        platform_profile_id,
        objective_profile,
        resource_budget,
        timeout_s,
        accelerator_policy,
        determinism_seed,
    )
    outcome = plan_request(request)
    if type(outcome) is PlanningFailureV2:
        return outcome
    if type(outcome) is not PlanningSuccessV2:
        raise TypeError("plan_request must return PlanningOutcomeV2")
    projection = project_route_observation_v2(
        outcome.route,
        observed_terrain,
        endpoint_theta_rad=target.theta_rad,
    )
    return replace(outcome, observation_projection=projection)


__all__ = (
    "BuildPpoRequestFn",
    "PlanPpoTargetFn",
    "PpoTargetV2",
    "build_ppo_request_v2",
    "plan_ppo_target_v2",
)
