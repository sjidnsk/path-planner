from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
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


class PrimitiveKindV2(_OrderedValueEnum):
    WHEEL_MOTION = "wheel_motion"
    LEG_STEP = "leg_step"
    BALLISTIC_JUMP = "ballistic_jump"


class ValidationLevelV2(_OrderedValueEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


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


def _strict_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")


def _costs_match(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


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
    distance_weight: float = 0.0
    risk_weight: float = 0.0
    energy_weight: float = 0.5
    time_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in ("distance_weight", "risk_weight", "energy_weight", "time_weight"):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))
        if not any(
            getattr(self, name) > 0.0
            for name in ("distance_weight", "risk_weight", "energy_weight", "time_weight")
        ):
            raise ValueError("objective weights must not be all-zero")


@dataclass(frozen=True, slots=True)
class ResourceBudgetV2:
    max_expanded_states: int = 100_000
    max_route_states: int = 10_000
    max_memory_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("max_expanded_states", "max_route_states", "max_memory_bytes"):
            _nonnegative_integer(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class RoutePrimitiveV2:
    kind: PrimitiveKindV2
    start_state: PoseStateV2
    end_state: PoseStateV2
    duration_s: float
    distance_m: float
    energy_cost: float
    observation_contribution: float
    validation_level: ValidationLevelV2

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PrimitiveKindV2):
            raise TypeError("kind must be PrimitiveKindV2")
        if not isinstance(self.start_state, PoseStateV2):
            raise TypeError("start_state must be PoseStateV2")
        if not isinstance(self.end_state, PoseStateV2):
            raise TypeError("end_state must be PoseStateV2")
        for name in ("duration_s", "distance_m", "energy_cost", "observation_contribution"):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))
        if not isinstance(self.validation_level, ValidationLevelV2):
            raise TypeError("validation_level must be ValidationLevelV2")


_PRIMITIVE_KIND_BY_PLATFORM = {
    PlatformKindV2.WHEEL: PrimitiveKindV2.WHEEL_MOTION,
    PlatformKindV2.LEGGED: PrimitiveKindV2.LEG_STEP,
    PlatformKindV2.HOPPER: PrimitiveKindV2.BALLISTIC_JUMP,
}


@dataclass(frozen=True, slots=True)
class TypedRouteV2:
    platform_kind: PlatformKindV2
    primitives: tuple[RoutePrimitiveV2, ...]
    total_cost: float
    is_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.platform_kind, PlatformKindV2):
            raise TypeError("platform_kind must be PlatformKindV2")
        if not isinstance(self.primitives, tuple):
            raise TypeError("primitives must be a tuple")
        if not self.primitives:
            raise ValueError("primitives must be nonempty")
        if any(not isinstance(primitive, RoutePrimitiveV2) for primitive in self.primitives):
            raise TypeError("primitives must contain only RoutePrimitiveV2 values")
        expected_kind = _PRIMITIVE_KIND_BY_PLATFORM[self.platform_kind]
        if any(primitive.kind is not expected_kind for primitive in self.primitives):
            raise ValueError("all primitive kinds must match the route platform")
        for previous, current in zip(self.primitives, self.primitives[1:], strict=False):
            if previous.end_state != current.start_state:
                raise ValueError("route primitive endpoints must be connected")
        object.__setattr__(self, "total_cost", _nonnegative_float(self.total_cost, "total_cost"))
        _strict_bool(self.is_complete, "is_complete")


@dataclass(frozen=True, slots=True)
class ObservationProjectionV2:
    source: str
    sample_states: tuple[PoseStateV2, ...]
    expected_new_observed_cells: float
    expected_information_gain: float

    def __post_init__(self) -> None:
        _nonempty_string(self.source, "source")
        if not isinstance(self.sample_states, tuple):
            raise TypeError("sample_states must be a tuple")
        if any(not isinstance(state, PoseStateV2) for state in self.sample_states):
            raise TypeError("sample_states must contain only PoseStateV2 values")
        for name in ("expected_new_observed_cells", "expected_information_gain"):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class CostBreakdownV2:
    distance_cost: float
    risk_cost: float
    energy_cost: float
    time_cost: float
    total_cost: float

    def __post_init__(self) -> None:
        component_names = ("distance_cost", "risk_cost", "energy_cost", "time_cost")
        for name in (*component_names, "total_cost"):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))
        component_total = sum(getattr(self, name) for name in component_names)
        if not _costs_match(self.total_cost, component_total):
            raise ValueError("total_cost must match the cost components")


@dataclass(frozen=True, slots=True)
class SearchTelemetryV2:
    expanded_states: int
    generated_primitives: int
    rejected_l0: int
    rejected_l1: int
    rejected_l2: int
    elapsed_s: float
    timed_out: bool
    accelerator_used: bool
    ackermann_feasible_claimed: bool
    termination_reason: str

    def __post_init__(self) -> None:
        for name in (
            "expanded_states",
            "generated_primitives",
            "rejected_l0",
            "rejected_l1",
            "rejected_l2",
        ):
            _nonnegative_integer(getattr(self, name), name)
        object.__setattr__(self, "elapsed_s", _nonnegative_float(self.elapsed_s, "elapsed_s"))
        for name in ("timed_out", "accelerator_used", "ackermann_feasible_claimed"):
            _strict_bool(getattr(self, name), name)
        _nonempty_string(self.termination_reason, "termination_reason")


@dataclass(frozen=True, slots=True)
class CacheEvidenceV2:
    cache_namespace: str
    cache_key: str
    hit: bool

    def __post_init__(self) -> None:
        _nonempty_string(self.cache_namespace, "cache_namespace")
        _nonempty_string(self.cache_key, "cache_key")
        _strict_bool(self.hit, "hit")


def _validate_failure_detail_scalar(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("failure evidence detail floats must be finite")
        return
    raise TypeError("failure evidence detail values must be JSON scalar values")


@dataclass(frozen=True, slots=True)
class FailureEvidenceV2:
    stage: str
    checks: tuple[str, ...]
    details: tuple[tuple[str, str | int | float | bool | None], ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.stage, "stage")
        if not isinstance(self.checks, tuple):
            raise TypeError("checks must be a tuple")
        for check in self.checks:
            _nonempty_string(check, "check")
        if not isinstance(self.details, tuple):
            raise TypeError("details must be a tuple")

        keys: list[str] = []
        for detail in self.details:
            if not isinstance(detail, tuple) or len(detail) != 2:
                raise TypeError("details must contain immutable (key, scalar) tuples")
            key, value = detail
            _nonempty_string(key, "detail key")
            _validate_failure_detail_scalar(value)
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError("failure evidence detail keys must be unique")
        if keys != sorted(keys):
            raise ValueError("failure evidence detail keys must be sorted")


@dataclass(frozen=True, slots=True)
class ValidationEvidenceV2:
    validator_id: str
    level: ValidationLevelV2
    passed: bool
    checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_string(self.validator_id, "validator_id")
        if not isinstance(self.level, ValidationLevelV2):
            raise TypeError("level must be ValidationLevelV2")
        _strict_bool(self.passed, "passed")
        if not isinstance(self.checks, tuple):
            raise TypeError("checks must be a tuple")
        if not self.checks:
            raise ValueError("checks must be nonempty")
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
    platform_kind: PlatformKindV2
    route: TypedRouteV2
    observation_projection: ObservationProjectionV2
    cost_breakdown: CostBreakdownV2
    validation_evidence: ValidationEvidenceV2
    search_telemetry: SearchTelemetryV2
    cache_evidence: CacheEvidenceV2
    schema_version: str = PLANNING_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        _nonempty_string(self.request_id, "request_id")
        if not isinstance(self.platform_kind, PlatformKindV2):
            raise TypeError("platform_kind must be PlatformKindV2")
        if not isinstance(self.route, TypedRouteV2):
            raise TypeError("route must be TypedRouteV2")
        if self.route.platform_kind is not self.platform_kind:
            raise ValueError("route platform must match result platform")
        if not self.route.is_complete:
            raise ValueError("successful route must be complete")
        if any(
            primitive.validation_level is not ValidationLevelV2.L2
            for primitive in self.route.primitives
        ):
            raise ValueError("successful route primitives must all be L2")
        if not isinstance(self.observation_projection, ObservationProjectionV2):
            raise TypeError("observation_projection must be ObservationProjectionV2")
        if not isinstance(self.cost_breakdown, CostBreakdownV2):
            raise TypeError("cost_breakdown must be CostBreakdownV2")
        if not isinstance(self.validation_evidence, ValidationEvidenceV2):
            raise TypeError("validation_evidence must be ValidationEvidenceV2")
        if not self.validation_evidence.passed:
            raise ValueError("successful route validation must have passed")
        if self.validation_evidence.level is not ValidationLevelV2.L2:
            raise ValueError("successful route validation must be L2")
        if not isinstance(self.search_telemetry, SearchTelemetryV2):
            raise TypeError("search_telemetry must be SearchTelemetryV2")
        if not isinstance(self.cache_evidence, CacheEvidenceV2):
            raise TypeError("cache_evidence must be CacheEvidenceV2")
        if not _costs_match(self.route.total_cost, self.cost_breakdown.total_cost):
            raise ValueError("route total_cost must match cost_breakdown total_cost")
        if self.schema_version != PLANNING_SCHEMA_VERSION_V2:
            raise ValueError(f"schema_version must be {PLANNING_SCHEMA_VERSION_V2}")


@dataclass(frozen=True, slots=True)
class PlanningFailureV2:
    request_id: str
    platform_kind: PlatformKindV2 | None
    category: FailureCategoryV2
    reason_code: str
    evidence: FailureEvidenceV2
    search_telemetry: SearchTelemetryV2
    schema_version: str = PLANNING_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        _nonempty_string(self.request_id, "request_id")
        if self.platform_kind is not None and not isinstance(self.platform_kind, PlatformKindV2):
            raise TypeError("platform_kind must be PlatformKindV2 or None")
        if not isinstance(self.category, FailureCategoryV2):
            raise TypeError("category must be FailureCategoryV2")
        _nonempty_string(self.reason_code, "reason_code")
        if not isinstance(self.evidence, FailureEvidenceV2):
            raise TypeError("evidence must be FailureEvidenceV2")
        if not isinstance(self.search_telemetry, SearchTelemetryV2):
            raise TypeError("search_telemetry must be SearchTelemetryV2")
        if self.platform_kind is None and (
            self.category is not FailureCategoryV2.UNSUPPORTED_CAPABILITY
            or self.reason_code != "platform_profile_unresolved"
            or self.evidence.stage != "profile_resolution"
        ):
            raise ValueError(
                "unresolved platform failure must use unsupported_capability, "
                "platform_profile_unresolved, and profile_resolution"
            )
        if self.schema_version != PLANNING_SCHEMA_VERSION_V2:
            raise ValueError(f"schema_version must be {PLANNING_SCHEMA_VERSION_V2}")


PlanningOutcomeV2 = PlanningSuccessV2 | PlanningFailureV2
