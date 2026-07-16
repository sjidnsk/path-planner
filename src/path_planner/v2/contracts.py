from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from path_planner.v2.terrain import TerrainSnapshotV2


PLANNING_SCHEMA_VERSION_V2 = "path-planner-v2-planning/v1"


class _OrderedValueEnum(str, Enum):
    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(item.value for item in cls)


class PlatformKindV2(_OrderedValueEnum):
    WHEEL = "wheel"
    LEGGED = "legged"
    HOPPER = "hopper"


class AcceleratorPolicyV2(_OrderedValueEnum):
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


class FailureCategoryV2(_OrderedValueEnum):
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNSAFE_START = "unsafe_start"
    UNSAFE_GOAL = "unsafe_goal"
    GOAL_POSE_UNREACHABLE = "goal_pose_unreachable"
    NO_COMPLETE_ROUTE = "no_complete_route"
    VALIDATION_FAILED = "validation_failed"
    RESOURCE_LIMIT = "resource_limit"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative_float(value: object, name: str) -> float:
    normalized = _finite_float(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and must not be bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class PoseStateV2:
    x_m: float
    y_m: float
    heading_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _finite_float(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _finite_float(self.y_m, "y_m"))
        object.__setattr__(self, "heading_rad", _finite_float(self.heading_rad, "heading_rad"))


@dataclass(frozen=True, slots=True)
class ObjectiveProfileV2:
    distance_weight: float = 1.0
    risk_weight: float = 0.0
    energy_weight: float = 0.0
    time_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("distance_weight", "risk_weight", "energy_weight", "time_weight"):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class ResourceBudgetV2:
    max_expanded_states: int = 100_000
    max_route_states: int = 10_000
    max_memory_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("max_expanded_states", "max_route_states", "max_memory_bytes"):
            _nonnegative_integer(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class TypedRouteV2:
    platform_kind: PlatformKindV2
    states: tuple[PoseStateV2, ...]
    total_cost: float
    is_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.platform_kind, PlatformKindV2):
            raise TypeError("platform_kind must be PlatformKindV2")
        if not isinstance(self.states, tuple):
            raise TypeError("states must be a tuple")
        if not self.states:
            raise ValueError("states must be nonempty")
        if any(not isinstance(state, PoseStateV2) for state in self.states):
            raise TypeError("states must contain only PoseStateV2 values")
        object.__setattr__(self, "total_cost", _nonnegative_float(self.total_cost, "total_cost"))
        if not isinstance(self.is_complete, bool):
            raise TypeError("is_complete must be bool")


@dataclass(frozen=True, slots=True)
class ValidationEvidenceV2:
    validator_id: str
    passed: bool
    checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_string(self.validator_id, "validator_id")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
        if not isinstance(self.checks, tuple):
            raise TypeError("checks must be a tuple")
        for check in self.checks:
            _nonempty_string(check, "check")


@dataclass(frozen=True, slots=True)
class PlanningRequestV2:
    request_id: str
    platform_profile_id: str
    start_state: PoseStateV2
    goal_state: PoseStateV2
    terrain_snapshot: TerrainSnapshotV2
    objective_profile: ObjectiveProfileV2
    resource_budget: ResourceBudgetV2
    timeout_s: float
    accelerator_policy: AcceleratorPolicyV2
    determinism_seed: int

    def __post_init__(self) -> None:
        _nonempty_string(self.request_id, "request_id")
        _nonempty_string(self.platform_profile_id, "platform_profile_id")
        if not isinstance(self.start_state, PoseStateV2):
            raise TypeError("start_state must be PoseStateV2")
        if not isinstance(self.goal_state, PoseStateV2):
            raise TypeError("goal_state must be PoseStateV2")
        if self.terrain_snapshot is None:
            raise TypeError("terrain_snapshot must not be None")
        if not isinstance(self.objective_profile, ObjectiveProfileV2):
            raise TypeError("objective_profile must be ObjectiveProfileV2")
        if not isinstance(self.resource_budget, ResourceBudgetV2):
            raise TypeError("resource_budget must be ResourceBudgetV2")
        object.__setattr__(self, "timeout_s", _nonnegative_float(self.timeout_s, "timeout_s"))
        if not isinstance(self.accelerator_policy, AcceleratorPolicyV2):
            raise TypeError("accelerator_policy must be AcceleratorPolicyV2")
        if isinstance(self.determinism_seed, bool) or not isinstance(self.determinism_seed, int):
            raise TypeError("determinism_seed must be an integer and must not be bool")


@dataclass(frozen=True, slots=True)
class PlanningSuccessV2:
    request_id: str
    route: TypedRouteV2
    validation_evidence: ValidationEvidenceV2
    schema_version: str = PLANNING_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        _nonempty_string(self.request_id, "request_id")
        if not isinstance(self.route, TypedRouteV2):
            raise TypeError("route must be TypedRouteV2")
        if not self.route.is_complete:
            raise ValueError("successful route must be complete")
        if not isinstance(self.validation_evidence, ValidationEvidenceV2):
            raise TypeError("validation_evidence must be ValidationEvidenceV2")
        if not self.validation_evidence.passed:
            raise ValueError("successful route validation must have passed")
        if self.schema_version != PLANNING_SCHEMA_VERSION_V2:
            raise ValueError(f"schema_version must be {PLANNING_SCHEMA_VERSION_V2}")


@dataclass(frozen=True, slots=True)
class PlanningFailureV2:
    request_id: str
    category: FailureCategoryV2
    reason_code: str
    schema_version: str = PLANNING_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        _nonempty_string(self.request_id, "request_id")
        if not isinstance(self.category, FailureCategoryV2):
            raise TypeError("category must be FailureCategoryV2")
        _nonempty_string(self.reason_code, "reason_code")
        if self.schema_version != PLANNING_SCHEMA_VERSION_V2:
            raise ValueError(f"schema_version must be {PLANNING_SCHEMA_VERSION_V2}")


PlanningOutcomeV2 = PlanningSuccessV2 | PlanningFailureV2
