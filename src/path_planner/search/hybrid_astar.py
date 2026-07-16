from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Iterable

from path_planner.core import Cell, CostGrid, FailureReason, GridSpec, WorldPoint
from path_planner.search.planning_grid import PlanningGrid


HYBRID_ASTAR_POSE_PATH = "hybrid_astar_pose_path"
MAX_REPLAY_STEPS = 100_000


class _ReplayDeadlineExceeded(TimeoutError):
    pass


class _ValidatorContractError(RuntimeError):
    pass


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _call_validator(callback: Callable[[Any], bool], value: Any, name: str) -> bool:
    try:
        result = callback(value)
    except Exception as exc:
        raise _ValidatorContractError(f"{name} raised an exception") from exc
    if type(result) is not bool:
        raise _ValidatorContractError(f"{name} must return exact bool")
    return result


def _deadline_checker_expired(deadline_checker: Callable[[], bool]) -> bool:
    result = deadline_checker()
    if type(result) is not bool:
        raise TypeError("deadline_checker must return exact bool")
    return result


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    theta_rad: float

    def normalized(self) -> "Pose2D":
        return Pose2D(self.x_m, self.y_m, _normalize_angle(self.theta_rad))

    def to_dict(self) -> dict[str, float]:
        return {
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "theta_rad": float(_normalize_angle(self.theta_rad)),
            "theta_deg": float(math.degrees(_normalize_angle(self.theta_rad))),
        }


@dataclass(frozen=True)
class MotionPrimitive:
    name: str
    v_mps: float
    omega_radps: float
    duration_s: float
    reverse: bool = False
    turn_in_place: bool = False

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "v_mps": float(self.v_mps),
            "omega_radps": float(self.omega_radps),
            "duration_s": float(self.duration_s),
            "reverse": bool(self.reverse),
            "turn_in_place": bool(self.turn_in_place),
        }


@dataclass(frozen=True)
class PoseTransition:
    start: Pose2D
    primitive: MotionPrimitive
    samples: tuple[Pose2D, ...]
    end: Pose2D
    distance_m: float
    absolute_heading_change_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.start, Pose2D):
            raise TypeError("start must be Pose2D")
        if not isinstance(self.primitive, MotionPrimitive):
            raise TypeError("primitive must be MotionPrimitive")
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple")
        if len(self.samples) < 2 or any(not isinstance(sample, Pose2D) for sample in self.samples):
            raise ValueError("samples must contain at least exact start and end Pose2D values")
        if not isinstance(self.end, Pose2D):
            raise TypeError("end must be Pose2D")
        if self.samples[0] != self.start:
            raise ValueError("samples must begin with start")
        if self.samples[-1] != self.end:
            raise ValueError("samples must end with end")
        for pose in self.samples:
            _finite_real(pose.x_m, "sample x_m")
            _finite_real(pose.y_m, "sample y_m")
            _finite_real(pose.theta_rad, "sample theta_rad")
        distance_m = _finite_real(self.distance_m, "distance_m")
        heading_change = _finite_real(
            self.absolute_heading_change_rad,
            "absolute_heading_change_rad",
        )
        primitive_omega = _finite_real(
            self.primitive.omega_radps,
            "primitive omega_radps",
        )
        primitive_duration = _finite_real(
            self.primitive.duration_s,
            "primitive duration_s",
        )
        if distance_m < 0.0:
            raise ValueError("distance_m must be nonnegative")
        if heading_change < 0.0:
            raise ValueError("absolute heading change must be nonnegative")
        expected_heading_change = abs(primitive_omega) * primitive_duration
        if not math.isclose(
            heading_change,
            expected_heading_change,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("absolute heading change must match primitive motion")
        sampled_distance = sum(
            math.hypot(right.x_m - left.x_m, right.y_m - left.y_m)
            for left, right in zip(self.samples, self.samples[1:], strict=False)
        )
        if not math.isclose(distance_m, sampled_distance, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("distance_m must match sample distance")


@dataclass(frozen=True)
class PoseSearchAudit:
    generated_primitives: int = 0
    rejected_poses: int = 0
    rejected_transitions: int = 0
    timed_out: bool = False
    termination_reason: str = "not_started"

    def __post_init__(self) -> None:
        for name in ("generated_primitives", "rejected_poses", "rejected_transitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.rejected_transitions > self.generated_primitives:
            raise ValueError("rejected_transitions cannot exceed generated_primitives")
        if type(self.timed_out) is not bool:
            raise TypeError("timed_out must be bool")
        if not isinstance(self.termination_reason, str) or not self.termination_reason:
            raise ValueError("termination_reason must be a nonempty string")
        if self.timed_out != (self.termination_reason == FailureReason.TIMEOUT.value):
            raise ValueError("timeout audit fields must agree")


def replay_motion_primitive(
    start: Pose2D,
    primitive: MotionPrimitive,
    integration_dt_s: float,
    *,
    deadline_checker: Callable[[], bool] | None = None,
) -> PoseTransition:
    """Replay one primitive with the same exact integration used by Hybrid A*."""

    if not isinstance(start, Pose2D):
        raise TypeError("start must be Pose2D")
    if not isinstance(primitive, MotionPrimitive):
        raise TypeError("primitive must be MotionPrimitive")
    _finite_real(start.x_m, "start x_m")
    _finite_real(start.y_m, "start y_m")
    _finite_real(start.theta_rad, "start theta_rad")
    _finite_real(primitive.v_mps, "primitive v_mps")
    _finite_real(primitive.omega_radps, "primitive omega_radps")
    duration_s = _finite_real(primitive.duration_s, "primitive duration_s")
    integration_dt_s = _finite_real(integration_dt_s, "integration_dt_s")
    if integration_dt_s <= 0.0:
        raise ValueError("integration_dt_s must be a finite positive number")
    if deadline_checker is not None and not callable(deadline_checker):
        raise TypeError("deadline_checker must be callable")

    step_ratio = duration_s / integration_dt_s
    if not math.isfinite(step_ratio) or step_ratio > MAX_REPLAY_STEPS:
        raise ValueError(f"replay steps must not exceed {MAX_REPLAY_STEPS}")
    steps = max(1, int(math.ceil(step_ratio)))
    dt = duration_s / float(steps)
    pose = start.normalized()
    samples = [start]
    distance_m = 0.0
    absolute_heading_change_rad = 0.0
    for _ in range(steps):
        if deadline_checker is not None and _deadline_checker_expired(deadline_checker):
            raise _ReplayDeadlineExceeded("motion primitive replay deadline expired")
        if abs(primitive.omega_radps) < 1.0e-12:
            next_pose = Pose2D(
                pose.x_m + primitive.v_mps * math.cos(pose.theta_rad) * dt,
                pose.y_m + primitive.v_mps * math.sin(pose.theta_rad) * dt,
                pose.theta_rad,
            ).normalized()
        else:
            theta_next = pose.theta_rad + primitive.omega_radps * dt
            if abs(primitive.v_mps) < 1.0e-12:
                next_pose = Pose2D(pose.x_m, pose.y_m, theta_next).normalized()
            else:
                radius = primitive.v_mps / primitive.omega_radps
                next_pose = Pose2D(
                    pose.x_m + radius * (math.sin(theta_next) - math.sin(pose.theta_rad)),
                    pose.y_m - radius * (math.cos(theta_next) - math.cos(pose.theta_rad)),
                    theta_next,
                ).normalized()
        distance_m += math.hypot(next_pose.x_m - pose.x_m, next_pose.y_m - pose.y_m)
        absolute_heading_change_rad += abs(primitive.omega_radps * dt)
        samples.append(next_pose)
        pose = next_pose

    return PoseTransition(
        start=start,
        primitive=primitive,
        samples=tuple(samples),
        end=samples[-1],
        distance_m=distance_m,
        absolute_heading_change_rad=absolute_heading_change_rad,
    )


@dataclass(frozen=True)
class PoseCostBreakdown:
    translation_cost: float = 0.0
    rotation_cost: float = 0.0
    reverse_penalty: float = 0.0
    turn_penalty: float = 0.0
    slope_cost: float = 0.0
    clearance_cost: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.translation_cost
            + self.rotation_cost
            + self.reverse_penalty
            + self.turn_penalty
            + self.slope_cost
            + self.clearance_cost
        )

    def plus(self, other: "PoseCostBreakdown") -> "PoseCostBreakdown":
        return PoseCostBreakdown(
            translation_cost=self.translation_cost + other.translation_cost,
            rotation_cost=self.rotation_cost + other.rotation_cost,
            reverse_penalty=self.reverse_penalty + other.reverse_penalty,
            turn_penalty=self.turn_penalty + other.turn_penalty,
            slope_cost=self.slope_cost + other.slope_cost,
            clearance_cost=self.clearance_cost + other.clearance_cost,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "translation_cost": float(self.translation_cost),
            "rotation_cost": float(self.rotation_cost),
            "reverse_penalty": float(self.reverse_penalty),
            "turn_penalty": float(self.turn_penalty),
            "slope_cost": float(self.slope_cost),
            "clearance_cost": float(self.clearance_cost),
            "total": float(self.total),
        }


@dataclass(frozen=True)
class PosePlanRequest:
    start: Pose2D
    goal: Pose2D
    theta_bin_count: int = 72
    position_tolerance_m: float = 0.5
    theta_tolerance_rad: float = math.radians(5.0)
    max_iterations: int = 100_000
    primitive_duration_s: float = 1.0
    integration_dt_s: float = 0.25
    max_speed_mps: float = 1.0
    max_angular_speed_radps: float = math.radians(45.0)
    footprint_length_m: float = 0.612
    footprint_width_m: float = 0.580
    footprint_safety_margin_m: float = 0.0
    translation_cost_weight: float = 1.0
    rotation_cost_weight: float = 0.2
    reverse_penalty_weight: float = 0.5
    turn_penalty_weight: float = 0.05
    terrain_cost_weight: float = 1.0
    closed_key_xy_resolution_m: float | None = None

    def __post_init__(self) -> None:
        if self.theta_bin_count <= 0:
            raise ValueError("theta_bin_count must be positive")
        if self.position_tolerance_m < 0.0:
            raise ValueError("position_tolerance_m must be nonnegative")
        if self.theta_tolerance_rad < 0.0:
            raise ValueError("theta_tolerance_rad must be nonnegative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.integration_dt_s <= 0.0:
            raise ValueError("integration_dt_s must be positive")
        if self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive")
        if self.max_angular_speed_radps <= 0.0:
            raise ValueError("max_angular_speed_radps must be positive")
        if self.footprint_length_m <= 0.0 or self.footprint_width_m <= 0.0:
            raise ValueError("footprint dimensions must be positive")
        if self.closed_key_xy_resolution_m is not None and self.closed_key_xy_resolution_m <= 0.0:
            raise ValueError("closed_key_xy_resolution_m must be positive when provided")


@dataclass(frozen=True)
class PosePathDiagnostics:
    runtime_ms: float = 0.0
    expanded_pose_count: int = 0
    max_frontier_size: int = 0
    theta_bin_count: int = 72
    heuristic_policy: str = "max_2d_grid_cost_euclidean_heading"
    search_mode: str = HYBRID_ASTAR_POSE_PATH
    platform_model: str = "differential_skid_steer"
    ackermann_feasible_claimed: bool = False
    footprint_length_m: float = 0.612
    footprint_width_m: float = 0.580
    footprint_safety_margin_m: float = 0.0
    primitives: tuple[str, ...] = ()
    dominance_key_policy: str = "single_best_pose_per_cell_theta_bin"
    hard_obstacle_sources: tuple[str, ...] = (
        "physical_obstacle_cells",
        "slope_blocked_cells",
        "blocked_cells",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_ms": float(self.runtime_ms),
            "expanded_pose_count": int(self.expanded_pose_count),
            "max_frontier_size": int(self.max_frontier_size),
            "theta_bin_count": int(self.theta_bin_count),
            "heuristic_policy": self.heuristic_policy,
            "search_mode": self.search_mode,
            "platform_model": self.platform_model,
            "ackermann_feasible_claimed": bool(self.ackermann_feasible_claimed),
            "footprint_length_m": float(self.footprint_length_m),
            "footprint_width_m": float(self.footprint_width_m),
            "footprint_safety_margin_m": float(self.footprint_safety_margin_m),
            "primitives": list(self.primitives),
            "dominance_key_policy": self.dominance_key_policy,
            "hard_obstacle_sources": list(self.hard_obstacle_sources),
        }


@dataclass(frozen=True)
class PosePlanResult:
    success: bool
    pose_path: tuple[Pose2D, ...]
    control_sequence: tuple[MotionPrimitive, ...]
    legacy_cell_path: tuple[Cell, ...]
    total_cost: float
    cost_breakdown: PoseCostBreakdown
    expanded_count: int
    failure_reason: FailureReason | None
    diagnostics: PosePathDiagnostics = field(default_factory=PosePathDiagnostics)
    audit: PoseSearchAudit = field(default_factory=PoseSearchAudit)

    def __post_init__(self) -> None:
        if self.success and self.failure_reason is not None:
            raise ValueError("successful result cannot have failure_reason")
        if not self.success and self.failure_reason is None:
            raise ValueError("failed result must have failure_reason")

    def to_route_dict(self, spec: GridSpec) -> dict[str, Any]:
        return {
            "schema_version": "path-planner-hybrid-pose-route/v1",
            "trajectory_kind": HYBRID_ASTAR_POSE_PATH,
            "reachable": bool(self.success),
            "path_cost": float(self.total_cost) if self.success else None,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "grid": spec.to_dict(),
            "pose_path": [pose.to_dict() for pose in self.pose_path],
            "control_sequence": [primitive.to_dict() for primitive in self.control_sequence],
            "legacy_cell_path": [cell.to_list() for cell in self.legacy_cell_path],
            "cost_breakdown": self.cost_breakdown.to_dict(),
            "expanded_count": int(self.expanded_count),
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(order=True)
class _QueueItem:
    priority: float
    order: int
    key: tuple[int, int, int] = field(compare=False)


@dataclass(frozen=True)
class _Node:
    pose: Pose2D
    samples_from_parent: tuple[Pose2D, ...]
    parent_key: tuple[int, int, int] | None
    primitive: MotionPrimitive | None
    cost: float
    breakdown: PoseCostBreakdown


@dataclass(frozen=True)
class _StepOutcome:
    transition: PoseTransition
    samples: tuple[Pose2D, ...]
    breakdown: PoseCostBreakdown


class HybridAStarPlanner:
    """Hybrid A* pose planner for differential/skid-steer platforms.

    The continuous state is (x, y, theta). The closed-list key discretizes pose
    into (x_cell, y_cell, theta_bin); this is not the old point-only grid A*.
    """

    def __init__(self, primitives: Iterable[MotionPrimitive] | None = None) -> None:
        self._custom_primitives = tuple(primitives) if primitives is not None else None

    def plan(
        self,
        grid: CostGrid | PlanningGrid,
        request: PosePlanRequest,
        *,
        pose_validator: Callable[[Pose2D], bool] | None = None,
        transition_validator: Callable[[PoseTransition], bool] | None = None,
        deadline_monotonic_s: float | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> PosePlanResult:
        if pose_validator is not None and not callable(pose_validator):
            raise TypeError("pose_validator must be callable")
        if transition_validator is not None and not callable(transition_validator):
            raise TypeError("transition_validator must be callable")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        normalized_deadline = (
            None
            if deadline_monotonic_s is None
            else _finite_real(deadline_monotonic_s, "deadline_monotonic_s")
        )
        started = time.perf_counter()
        primitives = self._custom_primitives or default_scout_mini_primitives(request)
        generated_primitives = 0
        rejected_poses = 0
        rejected_transitions = 0

        def deadline_expired() -> bool:
            if normalized_deadline is None:
                return False
            now = _finite_real(monotonic_clock(), "monotonic_clock result")
            return now >= normalized_deadline

        def audit(reason: str, *, timed_out: bool = False) -> PoseSearchAudit:
            return PoseSearchAudit(
                generated_primitives=generated_primitives,
                rejected_poses=rejected_poses,
                rejected_transitions=rejected_transitions,
                timed_out=timed_out,
                termination_reason=reason,
            )

        if deadline_expired():
            return self._failure(
                grid, request, FailureReason.TIMEOUT, 0, 0, started, primitives,
                audit(FailureReason.TIMEOUT.value, timed_out=True),
            )

        try:
            failure, endpoint_rejections = self._validate_request(
                grid,
                request,
                started,
                pose_validator,
            )
        except _ValidatorContractError:
            return self._failure(
                grid,
                request,
                FailureReason.VALIDATOR_ERROR,
                0,
                0,
                started,
                primitives,
                audit(FailureReason.VALIDATOR_ERROR.value),
            )
        rejected_poses += endpoint_rejections
        if deadline_expired():
            return self._failure(
                grid, request, FailureReason.TIMEOUT, 0, 0, started, primitives,
                audit(FailureReason.TIMEOUT.value, timed_out=True),
            )
        if failure is not None:
            return self._failure(
                grid,
                request,
                failure.failure_reason or FailureReason.INVALID_INPUT,
                0,
                0,
                started,
                primitives,
                audit((failure.failure_reason or FailureReason.INVALID_INPUT).value),
            )

        if deadline_expired():
            return self._failure(
                grid, request, FailureReason.TIMEOUT, 0, 0, started, primitives,
                audit(FailureReason.TIMEOUT.value, timed_out=True),
            )
        grid_cost_to_goal = _grid_cost_to_goal(grid, request.goal, deadline_expired)
        if grid_cost_to_goal is None or deadline_expired():
            return self._failure(
                grid, request, FailureReason.TIMEOUT, 0, 0, started, primitives,
                audit(FailureReason.TIMEOUT.value, timed_out=True),
            )
        start_key = self._key(
            grid.spec,
            request.start,
            request.theta_bin_count,
            request.closed_key_xy_resolution_m,
        )
        start_node = _Node(request.start.normalized(), (), None, None, 0.0, PoseCostBreakdown())
        nodes: dict[tuple[int, int, int], _Node] = {start_key: start_node}
        best_cost: dict[tuple[int, int, int], float] = {start_key: 0.0}
        frontier: list[_QueueItem] = []
        heapq.heappush(frontier, _QueueItem(self._heuristic(grid, request.start, request, grid_cost_to_goal), 0, start_key))
        expanded_count = 0
        max_frontier_size = 1
        order = 0

        while frontier:
            if deadline_expired():
                return self._failure(
                    grid, request, FailureReason.TIMEOUT, expanded_count, max_frontier_size,
                    started, primitives, audit(FailureReason.TIMEOUT.value, timed_out=True),
                )
            if expanded_count >= request.max_iterations:
                return self._failure(
                    grid,
                    request,
                    FailureReason.MAX_ITERATIONS,
                    expanded_count,
                    max_frontier_size,
                    started,
                    primitives,
                    audit(FailureReason.MAX_ITERATIONS.value),
                )

            current_item = heapq.heappop(frontier)
            if deadline_expired():
                return self._failure(
                    grid, request, FailureReason.TIMEOUT, expanded_count, max_frontier_size,
                    started, primitives, audit(FailureReason.TIMEOUT.value, timed_out=True),
                )
            current = nodes[current_item.key]
            if current.cost > best_cost.get(current_item.key, math.inf) + 1.0e-12:
                continue
            expanded_count += 1

            if self._goal_reached(current.pose, request):
                if deadline_expired():
                    return self._failure(
                        grid, request, FailureReason.TIMEOUT, expanded_count, max_frontier_size,
                        started, primitives, audit(FailureReason.TIMEOUT.value, timed_out=True),
                    )
                success = self._success(
                    grid, request, current_item.key, nodes, expanded_count,
                    max_frontier_size, started, primitives, audit("success"),
                )
                if deadline_expired():
                    return self._failure(
                        grid, request, FailureReason.TIMEOUT, expanded_count, max_frontier_size,
                        started, primitives, audit(FailureReason.TIMEOUT.value, timed_out=True),
                    )
                return success

            for primitive in primitives:
                if deadline_expired():
                    return self._failure(
                        grid, request, FailureReason.TIMEOUT, expanded_count, max_frontier_size,
                        started, primitives, audit(FailureReason.TIMEOUT.value, timed_out=True),
                    )
                generated_primitives += 1
                try:
                    outcome = self._apply_primitive(
                        grid,
                        request,
                        current.pose,
                        primitive,
                        pose_validator,
                        deadline_expired,
                    )
                except _ReplayDeadlineExceeded:
                    return self._failure(
                        grid, request, FailureReason.TIMEOUT, expanded_count, max_frontier_size,
                        started, primitives, audit(FailureReason.TIMEOUT.value, timed_out=True),
                    )
                except _ValidatorContractError:
                    return self._failure(
                        grid, request, FailureReason.VALIDATOR_ERROR, expanded_count,
                        max_frontier_size, started, primitives,
                        audit(FailureReason.VALIDATOR_ERROR.value),
                    )
                if deadline_expired():
                    return self._failure(
                        grid, request, FailureReason.TIMEOUT, expanded_count, max_frontier_size,
                        started, primitives, audit(FailureReason.TIMEOUT.value, timed_out=True),
                    )
                if outcome is None:
                    rejected_poses += 1
                    continue
                if transition_validator is not None:
                    try:
                        transition_accepted = _call_validator(
                            transition_validator,
                            outcome.transition,
                            "transition_validator",
                        )
                    except _ValidatorContractError:
                        return self._failure(
                            grid, request, FailureReason.VALIDATOR_ERROR, expanded_count,
                            max_frontier_size, started, primitives,
                            audit(FailureReason.VALIDATOR_ERROR.value),
                        )
                    if not transition_accepted:
                        rejected_transitions += 1
                        continue
                if deadline_expired():
                    return self._failure(
                        grid, request, FailureReason.TIMEOUT, expanded_count, max_frontier_size,
                        started, primitives, audit(FailureReason.TIMEOUT.value, timed_out=True),
                    )
                key = self._key(
                    grid.spec,
                    outcome.transition.end,
                    request.theta_bin_count,
                    request.closed_key_xy_resolution_m,
                )
                new_cost = current.cost + outcome.breakdown.total
                if new_cost >= best_cost.get(key, math.inf):
                    continue
                best_cost[key] = new_cost
                nodes[key] = _Node(
                    pose=outcome.transition.end,
                    samples_from_parent=outcome.samples,
                    parent_key=current_item.key,
                    primitive=primitive,
                    cost=new_cost,
                    breakdown=current.breakdown.plus(outcome.breakdown),
                )
                order += 1
                heapq.heappush(
                    frontier,
                    _QueueItem(new_cost + self._heuristic(grid, outcome.transition.end, request, grid_cost_to_goal), order, key),
                )
            max_frontier_size = max(max_frontier_size, len(frontier))

        return self._failure(
            grid,
            request,
            FailureReason.UNREACHABLE,
            expanded_count,
            max_frontier_size,
            started,
            primitives,
            audit(FailureReason.UNREACHABLE.value),
        )

    def _validate_request(
        self,
        grid: CostGrid | PlanningGrid,
        request: PosePlanRequest,
        started: float,
        pose_validator: Callable[[Pose2D], bool] | None,
    ) -> tuple[PosePlanResult | None, int]:
        start_cell = grid.spec.world_to_cell(WorldPoint(request.start.x_m, request.start.y_m))
        goal_cell = grid.spec.world_to_cell(WorldPoint(request.goal.x_m, request.goal.y_m))
        primitives = self._custom_primitives or default_scout_mini_primitives(request)
        if not grid.spec.in_bounds(start_cell):
            return self._failure(grid, request, FailureReason.START_OUT_OF_BOUNDS, 0, 0, started, primitives), 0
        if not grid.spec.in_bounds(goal_cell):
            return self._failure(grid, request, FailureReason.GOAL_OUT_OF_BOUNDS, 0, 0, started, primitives), 0
        start_feasible, start_rejected = self._pose_is_feasible(
            grid, request, request.start, pose_validator,
        )
        if not start_feasible:
            return (
                self._failure(grid, request, FailureReason.START_BLOCKED, 0, 0, started, primitives),
                int(start_rejected),
            )
        goal_feasible, goal_rejected = self._pose_is_feasible(
            grid, request, request.goal, pose_validator,
        )
        if not goal_feasible:
            return (
                self._failure(grid, request, FailureReason.GOAL_BLOCKED, 0, 0, started, primitives),
                int(goal_rejected),
            )
        return None, 0

    def _apply_primitive(
        self,
        grid: CostGrid | PlanningGrid,
        request: PosePlanRequest,
        start: Pose2D,
        primitive: MotionPrimitive,
        pose_validator: Callable[[Pose2D], bool] | None,
        deadline_expired: Callable[[], bool],
    ) -> _StepOutcome | None:
        if deadline_expired():
            return None
        transition = replay_motion_primitive(
            start,
            primitive,
            request.integration_dt_s,
            deadline_checker=deadline_expired,
        )
        if deadline_expired():
            return None
        terrain_cost = 0.0
        dt = primitive.duration_s / float(len(transition.samples) - 1)
        for next_pose in transition.samples[1:]:
            if deadline_expired():
                return None
            feasible, _ = self._pose_is_feasible(grid, request, next_pose, pose_validator)
            if not feasible:
                return None
            cell = grid.spec.world_to_cell(WorldPoint(next_pose.x_m, next_pose.y_m))
            if not grid.spec.in_bounds(cell):
                return None
            terrain_cost += max(0.0, grid.cost_at(cell) - grid.min_passable_cost()) * request.terrain_cost_weight * dt

        breakdown = PoseCostBreakdown(
            translation_cost=transition.distance_m * request.translation_cost_weight,
            rotation_cost=transition.absolute_heading_change_rad * request.rotation_cost_weight,
            reverse_penalty=(transition.distance_m * request.reverse_penalty_weight if primitive.reverse else 0.0),
            turn_penalty=(transition.absolute_heading_change_rad * request.turn_penalty_weight if abs(primitive.omega_radps) > 1.0e-12 else 0.0),
            slope_cost=terrain_cost,
            clearance_cost=0.0,
        )
        return _StepOutcome(
            transition=transition,
            samples=transition.samples[1:],
            breakdown=breakdown,
        )

    def _pose_is_feasible(
        self,
        grid: CostGrid | PlanningGrid,
        request: PosePlanRequest,
        pose: Pose2D,
        pose_validator: Callable[[Pose2D], bool] | None,
    ) -> tuple[bool, bool]:
        if pose_validator is None:
            return self._pose_collision_free(grid, request, pose), False
        center_cell = grid.spec.world_to_cell(WorldPoint(pose.x_m, pose.y_m))
        if not grid.spec.in_bounds(center_cell):
            return False, False
        try:
            center_cost = grid.cost_at(center_cell)
        except Exception:
            return False, False
        if not math.isfinite(center_cost) or center_cost < 0.0:
            return False, False
        accepted = _call_validator(pose_validator, pose, "pose_validator")
        return accepted, not accepted

    def _pose_collision_free(self, grid: CostGrid | PlanningGrid, request: PosePlanRequest, pose: Pose2D) -> bool:
        center_cell = grid.spec.world_to_cell(WorldPoint(pose.x_m, pose.y_m))
        if not grid.is_passable(center_cell):
            return False
        for cell in _footprint_cells(grid.spec, pose, request):
            if not grid.is_passable(cell):
                return False
        return True

    def _goal_reached(self, pose: Pose2D, request: PosePlanRequest) -> bool:
        distance = math.hypot(pose.x_m - request.goal.x_m, pose.y_m - request.goal.y_m)
        heading = abs(_angle_diff(pose.theta_rad, request.goal.theta_rad))
        return distance <= request.position_tolerance_m and heading <= request.theta_tolerance_rad

    def _key(
        self,
        spec: GridSpec,
        pose: Pose2D,
        theta_bin_count: int,
        xy_resolution_m: float | None = None,
    ) -> tuple[int, int, int]:
        if xy_resolution_m is None:
            cell = spec.world_to_cell(WorldPoint(pose.x_m, pose.y_m))
            x_key = cell.x
            y_key = cell.y
        else:
            x_key = int(math.floor((pose.x_m - spec.origin[0]) / float(xy_resolution_m)))
            y_key = int(math.floor((pose.y_m - spec.origin[1]) / float(xy_resolution_m)))
        theta = _normalize_angle(pose.theta_rad)
        theta_bin = int(round(theta / (2.0 * math.pi / float(theta_bin_count)))) % theta_bin_count
        return (x_key, y_key, theta_bin)

    def _heuristic(
        self,
        grid: CostGrid | PlanningGrid,
        pose: Pose2D,
        request: PosePlanRequest,
        grid_cost_to_goal: dict[Cell, float],
    ) -> float:
        distance = math.hypot(pose.x_m - request.goal.x_m, pose.y_m - request.goal.y_m)
        heading = abs(_angle_diff(pose.theta_rad, request.goal.theta_rad))
        cell = grid.spec.world_to_cell(WorldPoint(pose.x_m, pose.y_m))
        grid_cost = grid_cost_to_goal.get(cell, 0.0) * grid.spec.resolution * request.translation_cost_weight
        return max(
            grid_cost,
            distance * request.translation_cost_weight,
            heading * request.rotation_cost_weight,
        )

    def _success(
        self,
        grid: CostGrid | PlanningGrid,
        request: PosePlanRequest,
        goal_key: tuple[int, int, int],
        nodes: dict[tuple[int, int, int], _Node],
        expanded_count: int,
        max_frontier_size: int,
        started: float,
        primitives: tuple[MotionPrimitive, ...],
        audit: PoseSearchAudit,
    ) -> PosePlanResult:
        pose_path, controls = _reconstruct(nodes, goal_key)
        legacy_cells = _dedupe_cells(
            grid.spec.world_to_cell(WorldPoint(pose.x_m, pose.y_m))
            for pose in pose_path
        )
        goal = nodes[goal_key]
        return PosePlanResult(
            success=True,
            pose_path=pose_path,
            control_sequence=controls,
            legacy_cell_path=legacy_cells,
            total_cost=goal.breakdown.total,
            cost_breakdown=goal.breakdown,
            expanded_count=expanded_count,
            failure_reason=None,
            diagnostics=self._diagnostics(request, expanded_count, max_frontier_size, started, primitives),
            audit=audit,
        )

    def _failure(
        self,
        grid: CostGrid | PlanningGrid,
        request: PosePlanRequest,
        reason: FailureReason,
        expanded_count: int,
        max_frontier_size: int,
        started: float,
        primitives: tuple[MotionPrimitive, ...],
        audit: PoseSearchAudit | None = None,
    ) -> PosePlanResult:
        return PosePlanResult(
            success=False,
            pose_path=(),
            control_sequence=(),
            legacy_cell_path=(),
            total_cost=math.inf,
            cost_breakdown=PoseCostBreakdown(),
            expanded_count=expanded_count,
            failure_reason=reason,
            diagnostics=self._diagnostics(request, expanded_count, max_frontier_size, started, primitives),
            audit=(
                audit
                if audit is not None
                else PoseSearchAudit(termination_reason=reason.value)
            ),
        )

    def _diagnostics(
        self,
        request: PosePlanRequest,
        expanded_count: int,
        max_frontier_size: int,
        started: float,
        primitives: tuple[MotionPrimitive, ...],
    ) -> PosePathDiagnostics:
        return PosePathDiagnostics(
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            expanded_pose_count=expanded_count,
            max_frontier_size=max_frontier_size,
            theta_bin_count=request.theta_bin_count,
            footprint_length_m=request.footprint_length_m,
            footprint_width_m=request.footprint_width_m,
            footprint_safety_margin_m=request.footprint_safety_margin_m,
            primitives=tuple(primitive.name for primitive in primitives),
            dominance_key_policy=(
                "xy_resolution_m_theta_bin/v1"
                if request.closed_key_xy_resolution_m is not None
                else "single_best_pose_per_cell_theta_bin"
            ),
        )


def default_scout_mini_primitives(request: PosePlanRequest) -> tuple[MotionPrimitive, ...]:
    v = float(request.max_speed_mps)
    w = float(request.max_angular_speed_radps)
    duration = float(request.primitive_duration_s)
    return (
        MotionPrimitive("forward", v, 0.0, duration),
        MotionPrimitive("forward_left", v, w, duration),
        MotionPrimitive("forward_right", v, -w, duration),
        MotionPrimitive("reverse", -v, 0.0, duration, reverse=True),
        MotionPrimitive("reverse_left", -v, w, duration, reverse=True),
        MotionPrimitive("reverse_right", -v, -w, duration, reverse=True),
        MotionPrimitive("turn_in_place_left", 0.0, w, duration, turn_in_place=True),
        MotionPrimitive("turn_in_place_right", 0.0, -w, duration, turn_in_place=True),
    )


def _footprint_cells(spec: GridSpec, pose: Pose2D, request: PosePlanRequest) -> tuple[Cell, ...]:
    half_length = request.footprint_length_m / 2.0 + request.footprint_safety_margin_m
    half_width = request.footprint_width_m / 2.0 + request.footprint_safety_margin_m
    radius = math.hypot(half_length, half_width)
    min_cell = spec.world_to_cell(WorldPoint(pose.x_m - radius, pose.y_m - radius))
    max_cell = spec.world_to_cell(WorldPoint(pose.x_m + radius, pose.y_m + radius))
    cos_t = math.cos(pose.theta_rad)
    sin_t = math.sin(pose.theta_rad)
    cells: list[Cell] = []
    for y in range(min_cell.y, max_cell.y + 1):
        for x in range(min_cell.x, max_cell.x + 1):
            cell = Cell(x, y)
            if not spec.in_bounds(cell):
                cells.append(cell)
                continue
            center = spec.cell_to_world(cell)
            dx = center.x - pose.x_m
            dy = center.y - pose.y_m
            local_x = dx * cos_t + dy * sin_t
            local_y = -dx * sin_t + dy * cos_t
            if abs(local_x) <= half_length + spec.resolution * 0.5 and abs(local_y) <= half_width + spec.resolution * 0.5:
                cells.append(cell)
    return tuple(cells)


def _reconstruct(
    nodes: dict[tuple[int, int, int], _Node],
    key: tuple[int, int, int],
) -> tuple[tuple[Pose2D, ...], tuple[MotionPrimitive, ...]]:
    segments: list[tuple[Pose2D, ...]] = []
    controls: list[MotionPrimitive] = []
    current_key: tuple[int, int, int] | None = key
    while current_key is not None:
        node = nodes[current_key]
        if node.parent_key is None:
            segments.append((node.pose,))
        else:
            segments.append(node.samples_from_parent or (node.pose,))
        if node.primitive is not None:
            controls.append(node.primitive)
        current_key = node.parent_key
    segments.reverse()
    controls.reverse()
    poses: list[Pose2D] = []
    for segment in segments:
        for pose in segment:
            if not poses or pose != poses[-1]:
                poses.append(pose)
    return tuple(poses), tuple(controls)


def _dedupe_cells(cells: Iterable[Cell]) -> tuple[Cell, ...]:
    result: list[Cell] = []
    for cell in cells:
        if not result or result[-1] != cell:
            result.append(cell)
    return tuple(result)


def _grid_cost_to_goal(
    grid: CostGrid | PlanningGrid,
    goal_pose: Pose2D,
    deadline_expired: Callable[[], bool] | None = None,
) -> dict[Cell, float] | None:
    if deadline_expired is not None and deadline_expired():
        return None
    goal = grid.spec.world_to_cell(WorldPoint(goal_pose.x_m, goal_pose.y_m))
    if not grid.is_passable(goal):
        return {}
    frontier: list[tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(frontier, (0.0, counter, goal))
    cost_so_far: dict[Cell, float] = {goal: 0.0}
    directions = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while frontier:
        if deadline_expired is not None and deadline_expired():
            return None
        current_cost, _, current = heapq.heappop(frontier)
        if current_cost > cost_so_far.get(current, math.inf) + 1.0e-12:
            continue
        for dx, dy, step in directions:
            if deadline_expired is not None and deadline_expired():
                return None
            neighbor = Cell(current.x + dx, current.y + dy)
            if not grid.is_passable(neighbor):
                continue
            new_cost = current_cost + step * grid.cost_at(neighbor)
            if new_cost >= cost_so_far.get(neighbor, math.inf):
                continue
            cost_so_far[neighbor] = new_cost
            counter += 1
            heapq.heappush(frontier, (new_cost, counter, neighbor))
    return cost_so_far


def _normalize_angle(theta: float) -> float:
    return theta % (2.0 * math.pi)


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi
