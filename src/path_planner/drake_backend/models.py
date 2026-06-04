from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from path_planner.core import Cell, WorldPoint

IRIS_BACKENDS = frozenset({"workspace_iris"})
IRIS_REPORT_STATUSES = frozenset({"ok", "fallback", "failed"})
IRIS_REGION_SOURCES = frozenset({"iris", "grid_box", "fallback"})
IRIS_FAILURE_STATUSES = frozenset(
    {
        "backend_unavailable",
        "invalid_region_input",
        "infeasible",
        "solver_error",
        "fallback_used",
    }
)
CONVEX_REGION_SEQUENCE_SCHEMA_VERSION = "convex_region_sequence_report/v1"
CONVEX_REGION_BACKENDS = frozenset({"workspace_iris", "fallback_box"})
CONVEX_REGION_SOURCES = frozenset({"iris", "fallback_box"})
GCS_TRAJECTORY_REPORT_SCHEMA_VERSION = "gcs_trajectory_report/v1"
GCS_TRAJECTORY_BACKENDS = frozenset({"pydrake_gcs"})
GCS_GEOMETRIC_CANDIDATE_REPORT_SCHEMA_VERSION = "gcs_geometric_candidate_report/v1"
GCS_GEOMETRIC_CANDIDATE_SELECTION_REASONS = frozenset({"gcs_candidate_quality_improved"})
GCS_GEOMETRIC_CANDIDATE_FALLBACK_REASONS = frozenset(
    {
        "cost_dominated",
        "path_duplicate_with_baseline",
        "no_quality_gain",
        "sampled_trajectory_collision",
        "gcs_report_missing",
        "gcs_trajectory_failed",
        "unsupported_route_replacement",
    }
)
GCS_MOTION_FEASIBILITY_REPORT_SCHEMA_VERSION = "gcs_motion_feasibility_report/v1"
GCS_MOTION_FEASIBILITY_MODELS = frozenset({"curvature_bounded"})
GCS_MOTION_FEASIBILITY_STATUSES = frozenset({"feasible", "infeasible", "diagnostic_only", "not_evaluated"})
GCS_MOTION_FEASIBILITY_FALLBACK_REASONS = frozenset(
    {
        "gcs_report_missing",
        "gcs_trajectory_failed",
        "insufficient_samples",
        "motion_constraint_violation",
        "curvature_constraint_violation",
        "heading_constraint_violation",
    }
)
GCS_CURVATURE_CONSTRAINED_CANDIDATE_REPORT_SCHEMA_VERSION = (
    "gcs_curvature_constrained_candidate_report/v1"
)
GCS_CURVATURE_CONSTRAINED_REPAIR_STRATEGIES = frozenset(
    {
        "not_attempted",
        "none_required",
        "moving_average_smoothing",
    }
)
GCS_CURVATURE_CONSTRAINED_FALLBACK_REASONS = frozenset(
    {
        "gcs_report_missing",
        "gcs_trajectory_failed",
        "insufficient_samples",
        "motion_report_missing",
        "motion_feasibility_not_evaluated",
        "corridor_too_narrow_for_turning_radius",
        "portal_overlap_insufficient",
        "fillet_outside_convex_region",
        "heading_transition_infeasible",
        "curvature_repair_not_better",
        "candidate_collision",
        "region_containment_failed",
        "unsupported_nonconvex_constraint",
    }
)


def _validate_choice(value: str | None, allowed: frozenset[str], name: str) -> None:
    if value is None:
        return
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")


def _cell_bounds_to_dict(min_cell: Cell, max_cell: Cell) -> dict[str, list[int]]:
    return {"min": min_cell.to_list(), "max": max_cell.to_list()}


def _world_bounds_to_dict(min_world: WorldPoint, max_world: WorldPoint) -> dict[str, list[float]]:
    return {"min": min_world.to_list(), "max": max_world.to_list()}


@dataclass(frozen=True)
class IrisRegion:
    region_id: int
    source: str
    seed_cell: Cell
    seed_world: WorldPoint
    min_cell: Cell
    max_cell: Cell
    min_world: WorldPoint
    max_world: WorldPoint
    domain_min_cell: Cell
    domain_max_cell: Cell
    domain_min_world: WorldPoint
    domain_max_world: WorldPoint
    hpolyhedron_a: tuple[tuple[float, ...], ...]
    hpolyhedron_b: tuple[float, ...]
    validation_status: str
    fallback_reason: str | None = None
    failure_status: str | None = None

    def __post_init__(self) -> None:
        if self.region_id < 0:
            raise ValueError("region_id must be nonnegative")
        _validate_choice(self.source, IRIS_REGION_SOURCES, "source")
        _validate_choice(self.failure_status, IRIS_FAILURE_STATUSES, "failure_status")
        if self.min_cell.x > self.max_cell.x or self.min_cell.y > self.max_cell.y:
            raise ValueError("min_cell must not exceed max_cell")
        if self.domain_min_cell.x > self.domain_max_cell.x or self.domain_min_cell.y > self.domain_max_cell.y:
            raise ValueError("domain_min_cell must not exceed domain_max_cell")
        if self.min_world.x > self.max_world.x or self.min_world.y > self.max_world.y:
            raise ValueError("min_world must not exceed max_world")
        if self.domain_min_world.x > self.domain_max_world.x or self.domain_min_world.y > self.domain_max_world.y:
            raise ValueError("domain_min_world must not exceed domain_max_world")
        if not self.hpolyhedron_a:
            raise ValueError("hpolyhedron_a must be nonempty")
        if len(self.hpolyhedron_a) != len(self.hpolyhedron_b):
            raise ValueError("hpolyhedron_a and hpolyhedron_b length mismatch")
        for row in self.hpolyhedron_a:
            if len(row) != 2:
                raise ValueError("hpolyhedron_a rows must be 2D")
        if self.validation_status == "":
            raise ValueError("validation_status must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "source": self.source,
            "seed_cell": self.seed_cell.to_list(),
            "seed_world": self.seed_world.to_list(),
            "cell_bounds": _cell_bounds_to_dict(self.min_cell, self.max_cell),
            "world_bounds": _world_bounds_to_dict(self.min_world, self.max_world),
            "domain_cell_bounds": _cell_bounds_to_dict(self.domain_min_cell, self.domain_max_cell),
            "domain_world_bounds": _world_bounds_to_dict(self.domain_min_world, self.domain_max_world),
            "hpolyhedron": {
                "A": [list(row) for row in self.hpolyhedron_a],
                "b": list(self.hpolyhedron_b),
            },
            "validation_status": self.validation_status,
            "failure_status": self.failure_status,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class IrisRegionReport:
    backend: str
    status: str
    seed_source: str
    domain_source: str
    obstacle_source: str
    regions: tuple[IrisRegion, ...] = ()
    obstacle_count: int = 0
    validation_status: str = "not_evaluated"
    failure_reason: str | None = None
    failure_status: str | None = None
    fallback_used: bool = False

    def __post_init__(self) -> None:
        _validate_choice(self.backend, IRIS_BACKENDS, "backend")
        _validate_choice(self.status, IRIS_REPORT_STATUSES, "status")
        _validate_choice(self.failure_status, IRIS_FAILURE_STATUSES, "failure_status")
        if self.obstacle_count < 0:
            raise ValueError("obstacle_count must be nonnegative")
        if self.status == "ok" and (self.failure_reason is not None or self.failure_status is not None):
            raise ValueError("ok report cannot include failure reason/status")
        if self.status != "ok" and self.failure_reason is None:
            raise ValueError("non-ok report must include failure_reason")
        if self.status == "ok" and not self.regions:
            raise ValueError("ok report must include regions")
        if not self.seed_source:
            raise ValueError("seed_source must be nonempty")
        if not self.domain_source:
            raise ValueError("domain_source must be nonempty")
        if not self.obstacle_source:
            raise ValueError("obstacle_source must be nonempty")
        if not self.validation_status:
            raise ValueError("validation_status must be nonempty")

    @property
    def region_count(self) -> int:
        return len(self.regions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "region_count": self.region_count,
            "seed_source": self.seed_source,
            "domain_source": self.domain_source,
            "obstacle_source": self.obstacle_source,
            "obstacle_count": self.obstacle_count,
            "validation_status": self.validation_status,
            "failure_status": self.failure_status,
            "failure_reason": self.failure_reason,
            "fallback_used": self.fallback_used,
            "regions": [region.to_dict() for region in self.regions],
        }


@dataclass(frozen=True)
class ConvexRegionSequenceItem:
    region_id: int
    backend: str
    source: str
    seed_cell: Cell
    seed_world: WorldPoint
    min_cell: Cell
    max_cell: Cell
    min_world: WorldPoint
    max_world: WorldPoint
    hpolyhedron_a: tuple[tuple[float, ...], ...]
    hpolyhedron_b: tuple[float, ...]
    covered_path_indices: tuple[int, ...]
    validation_status: str = "valid"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.region_id < 0:
            raise ValueError("region_id must be nonnegative")
        _validate_choice(self.backend, CONVEX_REGION_BACKENDS, "backend")
        _validate_choice(self.source, CONVEX_REGION_SOURCES, "source")
        if self.min_cell.x > self.max_cell.x or self.min_cell.y > self.max_cell.y:
            raise ValueError("min_cell must not exceed max_cell")
        if self.min_world.x > self.max_world.x or self.min_world.y > self.max_world.y:
            raise ValueError("min_world must not exceed max_world")
        if not self.hpolyhedron_a:
            raise ValueError("hpolyhedron_a must be nonempty")
        if len(self.hpolyhedron_a) != len(self.hpolyhedron_b):
            raise ValueError("hpolyhedron_a and hpolyhedron_b length mismatch")
        for row in self.hpolyhedron_a:
            if len(row) != 2:
                raise ValueError("hpolyhedron_a rows must be 2D")
        if not self.validation_status:
            raise ValueError("validation_status must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "backend": self.backend,
            "source": self.source,
            "seed_cell": self.seed_cell.to_list(),
            "seed_world": self.seed_world.to_list(),
            "bounds": _cell_bounds_to_dict(self.min_cell, self.max_cell),
            "world_bounds": _world_bounds_to_dict(self.min_world, self.max_world),
            "hpolyhedron": {
                "A": [list(row) for row in self.hpolyhedron_a],
                "b": list(self.hpolyhedron_b),
            },
            "covered_path_indices": list(self.covered_path_indices),
            "validation_status": self.validation_status,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ConvexRegionSequenceReport:
    backend: str
    fallback_used: bool
    coverage_status: str
    start_contained: bool
    goal_contained: bool
    adjacent_overlap_count: int
    portal_count: int
    blocked_cell_violation_count: int
    gcs_ready: bool
    gcs_ready_reason: str
    regions: tuple[ConvexRegionSequenceItem, ...] = ()
    pydrake_available: bool | None = None
    schema_version: str = CONVEX_REGION_SEQUENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_choice(self.backend, CONVEX_REGION_BACKENDS, "backend")
        if self.adjacent_overlap_count < 0:
            raise ValueError("adjacent_overlap_count must be nonnegative")
        if self.portal_count < 0:
            raise ValueError("portal_count must be nonnegative")
        if self.blocked_cell_violation_count < 0:
            raise ValueError("blocked_cell_violation_count must be nonnegative")
        if not self.coverage_status:
            raise ValueError("coverage_status must be nonempty")
        if not self.gcs_ready_reason:
            raise ValueError("gcs_ready_reason must be nonempty")

    @property
    def region_count(self) -> int:
        return len(self.regions)

    def to_route_fields(self) -> dict[str, Any]:
        return {
            "convex_region_sequence_schema_version": self.schema_version,
            "convex_region_sequence": [region.to_dict() for region in self.regions],
            "convex_region_count": self.region_count,
            "convex_region_backend": self.backend,
            "convex_region_fallback_used": self.fallback_used,
            "convex_region_coverage_status": self.coverage_status,
            "convex_region_start_contained": self.start_contained,
            "convex_region_goal_contained": self.goal_contained,
            "convex_region_adjacent_overlap_count": self.adjacent_overlap_count,
            "convex_region_portal_count": self.portal_count,
            "convex_region_blocked_cell_violation_count": self.blocked_cell_violation_count,
            "convex_region_pydrake_available": self.pydrake_available,
            "gcs_ready": self.gcs_ready,
            "gcs_ready_reason": self.gcs_ready_reason,
        }


@dataclass(frozen=True)
class GcsTrajectoryReport:
    attempted: bool
    success: bool
    backend: str
    result_status: str
    reason: str
    sample_count: int
    collision_count: int
    path_length: float
    region_count: int
    sampled_points: tuple[WorldPoint, ...] = ()
    schema_version: str = GCS_TRAJECTORY_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_choice(self.backend, GCS_TRAJECTORY_BACKENDS, "backend")
        if not self.result_status:
            raise ValueError("result_status must be nonempty")
        if not self.reason:
            raise ValueError("reason must be nonempty")
        if self.sample_count < 0:
            raise ValueError("sample_count must be nonnegative")
        if self.collision_count < 0:
            raise ValueError("collision_count must be nonnegative")
        if self.region_count < 0:
            raise ValueError("region_count must be nonnegative")
        if self.path_length < 0.0:
            raise ValueError("path_length must be nonnegative")

    def to_route_fields(self) -> dict[str, Any]:
        return {
            "gcs_trajectory_report_schema_version": self.schema_version,
            "gcs_trajectory_attempted": self.attempted,
            "gcs_trajectory_success": self.success,
            "gcs_trajectory_backend": self.backend,
            "gcs_trajectory_result_status": self.result_status,
            "gcs_trajectory_reason": self.reason,
            "gcs_trajectory_sample_count": self.sample_count,
            "gcs_trajectory_collision_count": self.collision_count,
            "gcs_trajectory_path_length": self.path_length,
            "gcs_trajectory_region_count": self.region_count,
            "gcs_trajectory_sampled_points": [point.to_list() for point in self.sampled_points],
        }


@dataclass(frozen=True)
class GcsGeometricCandidateReport:
    attempted: bool
    available: bool
    selected: bool
    selection_reason: str | None
    fallback_reason: str | None
    path_length: float | None
    path_cost: float | None
    collision_count: int
    high_cost_exposure: float | None
    baseline_overlap_ratio: float | None
    cost_delta_vs_baseline: float | None
    cost_delta_vs_postprocess: float | None
    schema_version: str = GCS_GEOMETRIC_CANDIDATE_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_choice(
            self.selection_reason,
            GCS_GEOMETRIC_CANDIDATE_SELECTION_REASONS,
            "selection_reason",
        )
        _validate_choice(
            self.fallback_reason,
            GCS_GEOMETRIC_CANDIDATE_FALLBACK_REASONS,
            "fallback_reason",
        )
        if self.collision_count < 0:
            raise ValueError("collision_count must be nonnegative")
        if self.path_length is not None and self.path_length < 0.0:
            raise ValueError("path_length must be nonnegative")
        if self.path_cost is not None and self.path_cost < 0.0:
            raise ValueError("path_cost must be nonnegative")
        if self.high_cost_exposure is not None and self.high_cost_exposure < 0.0:
            raise ValueError("high_cost_exposure must be nonnegative")
        if self.baseline_overlap_ratio is not None and not 0.0 <= self.baseline_overlap_ratio <= 1.0:
            raise ValueError("baseline_overlap_ratio must be between 0 and 1")
        if self.selected and not self.available:
            raise ValueError("selected candidate must be available")
        if self.selected and self.selection_reason is None:
            raise ValueError("selected candidate must include selection_reason")
        if self.selected and self.fallback_reason is not None:
            raise ValueError("selected candidate cannot include fallback_reason")
        if not self.selected and self.fallback_reason is None:
            raise ValueError("unselected candidate must include fallback_reason")

    def to_route_fields(self) -> dict[str, Any]:
        return {
            "gcs_candidate_report_schema_version": self.schema_version,
            "gcs_candidate_attempted": self.attempted,
            "gcs_candidate_available": self.available,
            "gcs_candidate_selected": self.selected,
            "gcs_candidate_selection_reason": self.selection_reason,
            "gcs_candidate_fallback_reason": self.fallback_reason,
            "gcs_candidate_path_length": self.path_length,
            "gcs_candidate_path_cost": self.path_cost,
            "gcs_candidate_collision_count": self.collision_count,
            "gcs_candidate_high_cost_exposure": self.high_cost_exposure,
            "gcs_candidate_baseline_overlap_ratio": self.baseline_overlap_ratio,
            "gcs_candidate_cost_delta_vs_baseline": self.cost_delta_vs_baseline,
            "gcs_candidate_cost_delta_vs_postprocess": self.cost_delta_vs_postprocess,
        }


@dataclass(frozen=True)
class GcsMotionFeasibilityReport:
    evaluated: bool
    trajectory_source: str
    motion_model: str
    feasibility_status: str
    fallback_reason: str | None
    min_turning_radius_m: float | None
    max_heading_change_deg: float
    curvature_violation_count: int
    heading_violation_count: int
    violation_indices: tuple[int, ...]
    sample_count: int
    path_length: float
    constraint_summary: dict[str, Any]
    schema_version: str = GCS_MOTION_FEASIBILITY_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_choice(self.motion_model, GCS_MOTION_FEASIBILITY_MODELS, "motion_model")
        _validate_choice(
            self.feasibility_status,
            GCS_MOTION_FEASIBILITY_STATUSES,
            "feasibility_status",
        )
        _validate_choice(
            self.fallback_reason,
            GCS_MOTION_FEASIBILITY_FALLBACK_REASONS,
            "fallback_reason",
        )
        if not self.trajectory_source:
            raise ValueError("trajectory_source must be nonempty")
        if self.min_turning_radius_m is not None and self.min_turning_radius_m < 0.0:
            raise ValueError("min_turning_radius_m must be nonnegative")
        if self.max_heading_change_deg < 0.0:
            raise ValueError("max_heading_change_deg must be nonnegative")
        if self.curvature_violation_count < 0:
            raise ValueError("curvature_violation_count must be nonnegative")
        if self.heading_violation_count < 0:
            raise ValueError("heading_violation_count must be nonnegative")
        if self.sample_count < 0:
            raise ValueError("sample_count must be nonnegative")
        if self.path_length < 0.0:
            raise ValueError("path_length must be nonnegative")

    def to_route_fields(self) -> dict[str, Any]:
        return {
            "gcs_motion_feasibility_report_schema_version": self.schema_version,
            "gcs_motion_feasibility_evaluated": self.evaluated,
            "gcs_motion_feasibility_trajectory_source": self.trajectory_source,
            "gcs_motion_feasibility_motion_model": self.motion_model,
            "gcs_motion_feasibility_feasibility_status": self.feasibility_status,
            "gcs_motion_feasibility_fallback_reason": self.fallback_reason,
            "gcs_motion_feasibility_min_turning_radius_m": self.min_turning_radius_m,
            "gcs_motion_feasibility_max_heading_change_deg": self.max_heading_change_deg,
            "gcs_motion_feasibility_curvature_violation_count": self.curvature_violation_count,
            "gcs_motion_feasibility_heading_violation_count": self.heading_violation_count,
            "gcs_motion_feasibility_violation_indices": list(self.violation_indices),
            "gcs_motion_feasibility_sample_count": self.sample_count,
            "gcs_motion_feasibility_path_length": self.path_length,
            "gcs_motion_feasibility_constraint_summary": dict(self.constraint_summary),
        }


@dataclass(frozen=True)
class GcsCurvatureConstrainedCandidateReport:
    attempted: bool
    available: bool
    selected: bool
    repair_success: bool
    source: str
    repair_strategy: str
    status_before: str
    status_after: str
    fallback_reason: str | None
    curvature_violation_count_before: int
    curvature_violation_count_after: int
    heading_violation_count_before: int
    heading_violation_count_after: int
    violation_indices_before: tuple[int, ...]
    violation_indices_after: tuple[int, ...]
    region_containment_violation_count: int
    collision_count: int
    path_length: float | None
    path_cost: float | None
    cost_delta_vs_baseline: float | None
    constraint_summary: dict[str, Any]
    schema_version: str = GCS_CURVATURE_CONSTRAINED_CANDIDATE_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be nonempty")
        _validate_choice(
            self.repair_strategy,
            GCS_CURVATURE_CONSTRAINED_REPAIR_STRATEGIES,
            "repair_strategy",
        )
        _validate_choice(
            self.status_before,
            GCS_MOTION_FEASIBILITY_STATUSES,
            "status_before",
        )
        _validate_choice(
            self.status_after,
            GCS_MOTION_FEASIBILITY_STATUSES,
            "status_after",
        )
        _validate_choice(
            self.fallback_reason,
            GCS_CURVATURE_CONSTRAINED_FALLBACK_REASONS,
            "fallback_reason",
        )
        if self.selected and not self.available:
            raise ValueError("selected candidate must be available")
        if self.selected and self.status_after != "feasible":
            raise ValueError("selected candidate must be feasible after repair")
        if self.selected and self.fallback_reason is not None:
            raise ValueError("selected candidate cannot include fallback_reason")
        if not self.selected and self.fallback_reason is None:
            raise ValueError("unselected candidate must include fallback_reason")
        if self.repair_success and self.repair_strategy in {"not_attempted", "none_required"}:
            raise ValueError("repair_success requires an active repair strategy")
        for name, value in (
            ("curvature_violation_count_before", self.curvature_violation_count_before),
            ("curvature_violation_count_after", self.curvature_violation_count_after),
            ("heading_violation_count_before", self.heading_violation_count_before),
            ("heading_violation_count_after", self.heading_violation_count_after),
            ("region_containment_violation_count", self.region_containment_violation_count),
            ("collision_count", self.collision_count),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.path_length is not None and self.path_length < 0.0:
            raise ValueError("path_length must be nonnegative")
        if self.path_cost is not None and self.path_cost < 0.0:
            raise ValueError("path_cost must be nonnegative")

    def to_route_fields(self) -> dict[str, Any]:
        return {
            "gcs_curvature_constrained_report_schema_version": self.schema_version,
            "gcs_curvature_constrained_attempted": self.attempted,
            "gcs_curvature_constrained_available": self.available,
            "gcs_curvature_constrained_selected": self.selected,
            "gcs_curvature_constrained_repair_success": self.repair_success,
            "gcs_curvature_constrained_source": self.source,
            "gcs_curvature_constrained_repair_strategy": self.repair_strategy,
            "gcs_curvature_constrained_status_before": self.status_before,
            "gcs_curvature_constrained_status_after": self.status_after,
            "gcs_curvature_constrained_fallback_reason": self.fallback_reason,
            "gcs_curvature_constrained_curvature_violation_count_before": (
                self.curvature_violation_count_before
            ),
            "gcs_curvature_constrained_curvature_violation_count_after": (
                self.curvature_violation_count_after
            ),
            "gcs_curvature_constrained_heading_violation_count_before": (
                self.heading_violation_count_before
            ),
            "gcs_curvature_constrained_heading_violation_count_after": (
                self.heading_violation_count_after
            ),
            "gcs_curvature_constrained_violation_indices_before": list(self.violation_indices_before),
            "gcs_curvature_constrained_violation_indices_after": list(self.violation_indices_after),
            "gcs_curvature_constrained_region_containment_violation_count": (
                self.region_containment_violation_count
            ),
            "gcs_curvature_constrained_collision_count": self.collision_count,
            "gcs_curvature_constrained_path_length": self.path_length,
            "gcs_curvature_constrained_path_cost": self.path_cost,
            "gcs_curvature_constrained_cost_delta_vs_baseline": self.cost_delta_vs_baseline,
            "gcs_curvature_constrained_constraint_summary": dict(self.constraint_summary),
        }
