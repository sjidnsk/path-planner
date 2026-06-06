# path-planner

Lunar path planner rebuilt from scratch. This package does not copy or depend on `a_gcs_ws-2.0.1`; that project is only a reference for problem framing.

## Scope

Phase 1 provides:

- core route planning models;
- semantic costmap to `cost/passable_mask` conversion;
- adapters for JSON and `dev-platform-constraints`-style arrays;
- 2D A* with 8-neighbor search and corner-cut protection;
- CLI JSON output;
- PNG/HTML diagnostics.

Phase 2 adds a lightweight postprocess layer:

- conservative `corridor` sections around the raw A* path;
- line-of-sight shortcut smoothing that outputs `smoothed_path`;
- discrete curvature post-check that outputs `curvature_report`;
- `fallback_status` so failed postprocess steps keep the raw A* path available;
- diagnostics that overlay raw and smoothed paths.

Phase 2.5 connects platform constraints from `dev-platform-constraints`:

- default platform loading uses `yutu2`;
- `PlannerPlatformProfile` is a planning-side view derived from external platform parameters;
- vehicle footprint radius is derived from `body_length`, `body_width`, and optional `--safety-margin-m`;
- semantic costmap slope and obstacle hard limits are derived from the platform profile instead of internal defaults;
- corridor and smoothing checks use the vehicle-inflated passable mask;
- curvature checks use a valid platform `min_turning_radius` or an explicit override;
- diagnostics expose platform parameters, constraint warnings, inflated obstacles, and curvature violations.

Phase 3 moves platform constraints into the search stage:

- `PlanningGrid` wraps the original `CostGrid` with search-side constraints;
- `PlanningConstraints` exposes the platform-derived footprint, slope, obstacle, clearance, and turning-radius limits used by search;
- default CLI planning runs `platform-aware A*`;
- A* expands cells from `inflated_passable_mask`, so vehicle footprint clearance is enforced before postprocess;
- diagnostics record `search_mode`, `passable_source`, platform key, footprint radius, and original versus inflated blocked counts;
- structured search terrain layers are reserved for slope, roughness, illumination, and confidence inputs from future lunar maps.

Phase 4 adds a trackable-path interface and execution feasibility diagnostics:

- `trackable_path` converts the selected raw or smoothed path into waypoint records for a path tracker;
- each waypoint includes heading, segment length, turn angle, curvature, turning radius, and recommended speed;
- `speed_profile` is a conservative recommendation derived from platform speed, local cost, and curvature;
- `TrackingSafetyReport` checks whether a configured tracking-error tube remains inside the platform-aware safe region;
- diagnostics visualize trackable waypoints, heading arrows, and tracking-safety violations.

Phase 5 adds a lightweight tracking simulation baseline:

- `tracking_simulation_report` is generated when the CLI is run with `--simulate-tracking`;
- the simulator uses a low-speed pure-pursuit geometric approximation, not MPC or vehicle dynamics;
- simulated states include position, heading, target waypoint, speed, and cross-track error;
- experiment metrics include `max_cross_track_error_m`, simulated length, minimum clearance, safety violations, mean speed, and high-cost exposure;
- diagnostics overlay the orange `Simulated Tracking Path` and show a Tracking Simulation Summary.

Phase 6 adds a fixed-corridor continuous trajectory optimization prototype:

- `trajectory_optimization_report` is generated when the CLI is run with `--optimize-trajectory`;
- the optimizer projects continuous waypoint updates back into existing safety-corridor boxes;
- the objective includes path length, second-difference smoothness, reference deviation, high-cost exposure, and a curvature proxy;
- `optimized_tracking_simulation_report` is generated when `--simulate-tracking` and `--optimize-trajectory` are used together;
- diagnostics overlay the green `Optimized Path` and show a Trajectory Optimization Summary with `baseline_vs_optimized` metrics.

Phase 7 adds execution-aware trajectory optimization v1:

- `resampled_optimized_path` is generated when `--resample-spacing-m` is set;
- execution-aware metrics include `waypoint_spacing_mean_m`, `waypoint_spacing_max_m`, `heading_change_max_deg`, `tracking_error_proxy`, and `speed_smoothness_cost`;
- optimization weights include `--optimization-weight-tracking`, `--optimization-weight-spacing`, and `--optimization-weight-speed-smoothness`;
- diagnostics distinguish the optimized path from Resampled Optimized Waypoints and show an Execution-Aware Optimization Summary;
- warnings explicitly report when cross-track error or tracking-error proxy does not improve.

Phase 8 provides a Drake IRIS/GCS framework prototype:

- Repo Docs now include a pydrake knowledge structure and Phase 8 design/plan;
- Drake integration remains an optional backend boundary, not a default dependency;
- Phase 8 prioritizes original 2D workspace `Iris` over convex obstacle primitives, not C-space IRIS or Ackermann guarantees;
- Phase 8.1 adds a Drake-free `region_graph_report` built from `grid_box` regions and `blocked_cell_box` obstacle primitives;
- Phase 8.2 adds an optional `workspace_iris` backend that can emit `iris_region_report` when `--drake-iris-regions` is enabled;
- `iris_region_report` serializes numeric `HPolyhedron` half-space arrays and falls back to `grid_box` regions when Drake is unavailable or validation fails;
- Workspace IRIS diagnostics use `merged_blocked_rectangle` obstacles when that safely reduces obstacle fragmentation, and successful IRIS regions use `postprocess_corridor_safe_component_box` domains while keeping unsafe cells out of serialized region boxes;
- Phase 8.3 lets `region_graph_report` consume valid IRIS regions as an `iris` graph source and records graph quality metrics and fallback decisions;
- diagnostics include an IRIS / Region Graph Summary that labels this as a 2D workspace safe-region diagnostic, not a GCS trajectory or vehicle feasibility proof;
- Phase 8.4 adds an opt-in fixed-sequence `pydrake_direction_cone_program` that emits `gcs_trajectory_report`, `gcs_candidate_report`, `gcs_motion_feasibility_report`, and direction-cone portal/cost diagnostics;
- Phase 8.5 adds an opt-in `pydrake_control_point_direction_cone_program` for control-point derivative-proxy direction-cone diagnostics;
- GCS direction-cone reports include `rho_source_counts`, `portal_width_min_m`, `support_width_min_m`, `constraint_tightness_min`, `trajectory_parameterization`, `control_point_count`, `derivative_constraint_count`, `candidate_decision`, `decision_reason`, and `quality_gate` so selected/blocked candidate decisions are machine-readable;
- `path_planner.drake_backend.gcs_cli_batch` runs repeatable CLI scenario batches and writes `gcs_direction_cone_cli_scenario_batch/v1` and `gcs_motion_feasibility_cli_batch/v1` summaries from route JSON evidence;
- `--optimize-trajectory` continues to mean the current fixed-corridor optimizer until a separate Drake backend switch is implemented;
- the required fallback chain is Drake unavailable or infeasible -> current optimizer -> postprocess smoothed path -> raw A* path.

Core Algorithm Stage 1 adds an opt-in `region_graph_guided` planning backend:

- default CLI planning remains `astar`, which runs the existing platform-aware A* path;
- `--planning-backend region_graph_guided` first builds the baseline A* route and `region_graph_report`, then derives a region-center waypoint skeleton when the graph is connected;
- segment-level A* plans between skeleton waypoints and stitches a candidate `geometric_path`;
- the backend falls back to baseline A* with machine-readable reasons such as `region_graph_disconnected`, `segment_astar_failed`, `region_graph_invalid`, or `region_graph_candidate_not_better`;
- successful opt-in candidates and fallbacks are recorded in additive `planning_backend_report` diagnostics.

This project does not yet implement full GCS graph search, production Bezier/B-spline GCS trajectory optimization, Ackermann trajectory optimization, a production Drake backend, exploration target selection, observation updates, or an online planning service. The current `pydrake_direction_cone_program` and `pydrake_control_point_direction_cone_program` are fixed `convex_region_sequence` MathematicalProgram prototypes for 2D geometric candidates and sampled diagnostics.

The planner returns a platform-filtered `geometric_path` plus a trackable-path interface and feasibility diagnostics, not a closed-loop controller command stream.

At the system level, the next-stage validation uses `path-planner` as the
execution evaluator inside the `dev-platform-constraints -> model-explorer ->
path-planner` semi-real JSON loop. The acceptance path is the parent repository
command `scripts/run_path_feedback_validation.sh --scenario-set all
--diagnostic-profile all --top-k 3`, which forwards tracking simulation,
fixed-corridor optimization, and optional workspace IRIS diagnostics. This does
not change `path-planner-route/v1`: `trajectory_kind` remains `geometric_path`,
`region_graph_report` and `iris_region_report` remain diagnostics, and
`--optimize-trajectory` remains the fixed-corridor optimizer rather than a Drake
GCS backend.

## Development Environment

```powershell
conda activate D:\conda_envs\lunar-explorer
$env:PYTHONPATH='src'
```

The shared `lunar-explorer` Conda environment provides Python 3.12, NumPy,
Matplotlib, and pytest. This project does not need to be installed into that
environment for local development.

## Run Tests

```powershell
python -m pytest
```

Linux shell equivalent:

```bash
PYTHONPATH=src python -m pytest
```

Optional Drake API probe in the shared Conda environment:

```bash
conda run -n lunar-explorer env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m drake
```

The default test suite does not require `pydrake`; Drake tests are marked and
skip automatically when the optional backend is unavailable. The explicit
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` keeps unrelated environment plugins from
affecting the project probe.

## Run Demo

```powershell
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo
```

Optional platform arguments:

```powershell
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo --platform yutu2 --safety-margin-m 0.1
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo --platform-config D:\codex\project\lunar-path-planning\dev-platform-constraints\configs\platforms\yutu2.json
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo --tracking-error-bound-m 0.1 --min-speed-mps 0.01
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo --simulate-tracking --lookahead-m 0.75 --time-step-s 0.2
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo --simulate-tracking --optimize-trajectory
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo --simulate-tracking --optimize-trajectory --resample-spacing-m 0.4
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo --drake-iris-regions
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/route.json --output-dir outputs/demo --drake-iris-regions --gcs-geometric-candidate
python -m path_planner.cli --input examples/demo_map_corridor.json --output-json outputs/demo/control-point-route.json --output-dir outputs/demo-control-point --drake-iris-regions --gcs-control-point-candidate --gcs-motion-feasibility
```

GCS direction-cone CLI batch evidence:

```bash
python -m path_planner.drake_backend.gcs_cli_batch --output-dir outputs/gcs-cli-batch --summary-json outputs/gcs-cli-batch/summary.json
python -m path_planner.drake_backend.gcs_cli_batch --batch-kind motion-feasibility --output-dir outputs/gcs-motion-batch --summary-json outputs/gcs-motion-batch/summary.json
```

Expected outputs:

- `outputs/demo/route.json`
- `outputs/demo/diagnostics.png`
- `outputs/demo/diagnostics.html`
- `outputs/gcs-cli-batch/summary.json` when the batch runner is used
- `outputs/gcs-motion-batch/summary.json` when the motion-feasibility batch runner is used

The route JSON preserves Phase 1 fields and adds a `postprocess` object with
`platform_profile`, `constraint_warnings`, `corridor_report`, `raw_path`,
`corridor`, `smoothed_path`, `curvature_report`, `trackable_path`,
`tracking_safety_report`, and `fallback_status`.
The top-level `diagnostics` object also records whether A* used
`platform_aware_astar` and `inflated_passable_mask`.
Phase 8.1 adds an optional top-level `region_graph_report` object with
`status`, `vertex_count`, `edge_count`, `obstacle_count`, `region_source`,
`obstacle_source`, `motion_feasibility_status`, and serialized region graph
details. This report is a geometric region-graph diagnostic, not an IRIS, GCS,
Ackermann, or skid-steer executable trajectory.
Phase 8.2 adds an optional top-level `iris_region_report` object when
`--drake-iris-regions` is enabled. The report includes `backend`, `status`,
`region_count`, `seed_source`, `domain_source`, `obstacle_source`,
`obstacle_count`, `validation_status`, `failure_status`, `fallback_used`, and
serialized numeric region details. This report is a 2D workspace safe-region
prototype, not a GCS trajectory or Ackermann/skid-steer feasibility guarantee.
Phase 8.3 extends `region_graph_report` with `quality_metrics`, including
`requested_region_source`, `graph_source`, `iris_region_count`,
`grid_fallback_region_count`, `invalid_region_count`, `fallback_ratio`,
`connected_component_count`, and `start_goal_connected`. When
`--drake-iris-regions` is enabled, valid connected IRIS regions can be used as
the graph source; otherwise the report falls back to the existing `grid_box`
graph and records why.
Core Algorithm Stage 1 can add a top-level `planning_backend_report` object when
`--planning-backend region_graph_guided` is enabled. It records the requested
and selected backend, fallback reason, segment count, skeleton cells, candidate
comparison against baseline A*, and region-graph candidate status. This is an
additive diagnostic; `path-planner-route/v1` and top-level
`trajectory_kind=geometric_path` are unchanged.
When `--gcs-trajectory-smoke`, `--gcs-geometric-candidate`,
`--gcs-control-point-candidate`, or `--gcs-motion-feasibility` is enabled, the route JSON can include
`gcs_trajectory_report`, `gcs_candidate_report`, and
`gcs_motion_feasibility_report` additive fields. The GCS trajectory backend is
either fixed-sequence `pydrake_direction_cone_program` or the opt-in
`pydrake_control_point_direction_cone_program`, not full GCS graph search.
The control-point backend constrains successive control-point differences as a
derivative proxy and records `trajectory_parameterization`,
`control_point_count`, `derivative_proxy`, `derivative_constraint_count`,
`control_point_region_containment_count`, and `objective_terms` such as
`segment_length_quadratic`, `low_cost_anchor_quadratic`, and
`control_point_second_difference_quadratic`. Its
direction-cone summary also reports portal/support/rho fields such as
`rho_source_counts`, `portal_width_min_m`, `support_width_min_m`, and
`constraint_tightness_min`. Candidate cost summaries report
`candidate_decision`, `decision_reason`, and `quality_gate` so each replacement
or blocking decision can be audited from JSON. These reports remain 2D
geometric candidate diagnostics and do not prove Ackermann feasibility.
The CLI batch runner generates small request JSON files, invokes
`path_planner.cli` for each case, and summarizes selected/blocked decisions from
the resulting route JSON. Its summary schema is
`gcs_direction_cone_cli_scenario_batch/v1` and covers selected open-corridor,
cost-dominated, duplicate-baseline, sampled-collision, motion-infeasible,
degenerate-portal, and pydrake-unavailable cases.
With `--batch-kind motion-feasibility`, the same runner emits
`gcs_motion_feasibility_cli_batch/v1` summaries for straight feasible,
gentle-turn feasible, sharp-turn blocked, tight-radius blocked,
direction-cone-success-but-motion-blocked, and pydrake-unavailable route JSON
cases. The motion batch records heading and turning-radius diagnostics,
candidate selected/blocked state, fallback reasons, and expectation failures.
It is a regression gate for sampled geometric candidates, not an Ackermann trajectory optimizer.
When `--simulate-tracking` is enabled, the route JSON also includes a top-level
`tracking_simulation_report` object with `simulated_path`, `config`, `metrics`,
and `safety_report`.
When `--optimize-trajectory` is enabled, the route JSON also includes a top-level
`trajectory_optimization_report` object with `optimized_path`, `corridor_boxes`,
`solver_status`, `fallback_status`, and `metrics`. When optimization and
tracking simulation are both enabled, `optimized_tracking_simulation_report`
and `baseline_vs_optimized` comparison metrics are also emitted.
When `--resample-spacing-m` is enabled, `trajectory_optimization_report`
also includes `resampled_optimized_path`, `resampled_trackable_path`,
`resampled_corridor_boxes`, execution-aware metrics, and warnings.

By default, shortcut smoothing only accepts cells with `cost <= 3.0`; adjust
this with `--max-shortcut-cost` when a map uses a different cost scale. The
active demo uses `examples/demo_map_corridor.json`, a non-straight corridor case
that keeps the platform-aware safety corridor feasible while showing irregular
obstacles and vehicle-inflated blocked cells.

## External Interface Direction

`dev-platform-constraints` provides platform and vehicle parameters through its
platform config loader. `path-planner` does not maintain an independent vehicle
fact model; it derives `PlannerPlatformProfile` from the external
`PlatformParameters` object. `model-explorer` can consume the route JSON fields
`reachable`, `geometric_path`, `path_cost`, `diagnostics`, `failure_reason`, and
the optional `postprocess` object.
