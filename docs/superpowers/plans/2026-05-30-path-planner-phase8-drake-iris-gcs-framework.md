# Path Planner Phase 8 Drake IRIS/GCS Framework Plan

> **For agentic workers:** Implement this plan task-by-task. Phase 8 is a
> Framework First phase: do not replace the existing planner chain with Drake
> until the optional backend and fallback contracts are stable.

**Goal:** Add Drake IRIS/GCS knowledge documentation, define the optional backend
architecture, choose the IRIS variant appropriate for lunar 2D workspace planning,
and prepare tests and contracts for future implementation without making
`pydrake` a default dependency.

**Architecture:** `path_planner.core`, `search`, `postprocess`, `tracking`, and
the current `optimization` package remain Drake-free. Future Drake code should
live behind a backend boundary so missing `pydrake` never breaks default CLI or
default tests.

**Verification environment:** Use the shared Conda environment for Drake checks:

```bash
conda run -n lunar-explorer env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m drake
```

Default project verification remains:

```bash
PYTHONPATH=src python3 -m pytest
```

---

## Task 1: pydrake Knowledge Structure

- [x] Create `docs/drake/pydrake-knowledge-structure.md`.
- [x] Cover official pydrake index, geometry optimization, planning, solvers,
  trajectories, and pip installation pages.
- [x] Record the mapping from Drake concepts to `path-planner` concepts.
- [x] Explicitly mark `GraphOfConvexSets` and `GcsTrajectoryOptimization` as
  experimental Drake capabilities requiring optional backend and fallback.
- [x] Document the IRIS family split: original `Iris`, `IrisNp`, `IrisNp2`,
  `IrisZo`, and clique-cover C-space variants.
- [x] Record that Phase 8 should prioritize original 2D workspace `Iris`, not
  C-space IRIS variants.
- [x] Record why GCS output is not automatically Ackermann-feasible.

## Task 2: Phase 8 Design Spec

- [x] Create `docs/superpowers/specs/2026-05-30-path-planner-phase8-drake-iris-gcs-design.md`.
- [x] Define Phase 8 as an IRIS/GCS framework prototype.
- [x] Preserve current route semantics: `trajectory_kind` remains
  `geometric_path`.
- [x] Define candidate optional reports:
  `iris_region_report`, `region_graph_report`, and `gcs_trajectory_report`.
- [x] Define nonconvex lunar obstacle handling by convex obstacle primitives
  derived from `inflated_passable_mask`.
- [x] Define rover motion feasibility as a follow-on layer, not a Phase 8
  Ackermann hard guarantee.
- [x] Require fallback chain:

```text
Drake unavailable
  -> Drake backend failed or infeasible
  -> current fixed-corridor optimizer
  -> postprocess smoothed_path
  -> raw A* geometric_path
```

## Task 3: Optional Drake API Probe

- [x] Add a pytest marker named `drake`.
- [x] Add a Drake optional import test that skips when `pydrake` is unavailable.
- [x] Verify these APIs when Drake is available:
  `HPolyhedron`, `Point`, `Iris`, `IrisOptions`, `GraphOfConvexSets`,
  `GcsTrajectoryOptimization`, `MathematicalProgram`, `Solve`,
  `PiecewisePolynomial`, and `BsplineTrajectory`.
- [x] Run the test in `lunar-explorer`.

## Task 4: Backend Boundary Design

- [x] Introduce design docs for a future backend package, for example
  `path_planner.drake_backend`.
- [x] Keep all `pydrake` imports inside optional probe tests or the future backend boundary.
- [x] Define failure statuses for unavailable backend, invalid region input,
  infeasible GCS solve, and solver exception.
- [x] Keep the existing fixed-corridor optimizer as baseline and fallback.
- [x] Require backend reports to distinguish `workspace_iris`, `cspace_iris_np`,
  `cspace_iris_np2`, and `cspace_iris_zo`; only `workspace_iris` is in scope
  for the first implementation.

## Task 5: Obstacle, Region, And Graph Model Design

- [x] Define internal model concepts for `ConvexRegion`, `RegionGraph`,
  `RegionEdge`, `ObstaclePrimitive`, and sampled trajectory output.
- [x] Leave `SampledTrajectory` reserved for a later Drake/GCS output phase.
- [x] Record source provenance: `grid_box`, `iris`, `manual`, or `fallback`.
- [x] Define obstacle provenance: `blocked_cell_box`, `merged_blocked_rectangle`,
  `manual_convex_obstacle`, or `scene_graph_obstacle`.
- [x] Add validation status for regions; future IRIS regions must be sampled against
  `inflated_passable_mask`.
- [x] Avoid connected-component convex hulls as the default obstacle model
  because they can remove real narrow passages.
- [x] Specify that route JSON serializes numeric arrays and sampled points, not
  Drake object instances.

## Task 6: CLI, JSON, Diagnostics, And Motion Feasibility Design

- [x] Keep `--optimize-trajectory` mapped to the current fixed-corridor optimizer.
- [x] Reserve a separate future switch such as `--trajectory-backend drake-gcs`
  or `--drake-gcs`.
- [x] Add diagnostics design for drawing IRIS regions, GCS graph edges, and
  sampled GCS trajectory when present.
- [x] Add diagnostic wording that GCS geometric trajectories are not certified
  Ackermann trajectories.
- [x] Reserve a future `motion_feasibility_report` for `curvature_bounded`,
  `ackermann`, `skid_steer`, `differential`, or `custom` models.
- [x] Ensure old JSON consumers can ignore all Phase 8 reports safely.

## Task 6.1: Phase 8.1 Drake-Free Region Graph Baseline

- [x] Add Drake-free models: `ObstaclePrimitive`, `ConvexRegion`, `RegionEdge`,
  `RegionGraph`, and `RegionGraphReport`.
- [x] Build `blocked_cell_box` obstacle primitives from the footprint-safe mask.
- [x] Build `grid_box` regions from postprocess corridor sections.
- [x] Build `grid_adjacency` edges for overlapping or touching regions.
- [x] Add optional top-level `region_graph_report` to route JSON.
- [x] Keep `trajectory_kind = geometric_path` and `reachable` unchanged.
- [x] Add default tests for serialization, graph construction, and JSON integration.

## Task 7: Acceptance Tests

- [x] Default tests pass without `pydrake`.
- [x] Drake marked tests pass in `lunar-explorer`.
- [x] README and docs mention Phase 8 planned framework and full Drake backend
  not yet implemented.
- [x] CLI demo output remains backward compatible.

## Self-Review

- This plan intentionally does not implement full IRIS/GCS trajectory planning.
- This plan intentionally does not add Drake to default dependencies.
- This plan intentionally keeps A*, postprocess, tracking simulation, and
  fixed-corridor optimization as the stable fallback chain.
