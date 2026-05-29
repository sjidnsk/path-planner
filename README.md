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

This project does not implement GCS, IRIS, Ackermann trajectory optimization, Drake integration, exploration target selection, observation updates, or an online planning service.

The planner returns a platform-filtered `geometric_path` plus a trackable-path interface and feasibility diagnostics, not a closed-loop controller command stream.

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
```

Expected outputs:

- `outputs/demo/route.json`
- `outputs/demo/diagnostics.png`
- `outputs/demo/diagnostics.html`

The route JSON preserves Phase 1 fields and adds a `postprocess` object with
`platform_profile`, `constraint_warnings`, `corridor_report`, `raw_path`,
`corridor`, `smoothed_path`, `curvature_report`, `trackable_path`,
`tracking_safety_report`, and `fallback_status`.
The top-level `diagnostics` object also records whether A* used
`platform_aware_astar` and `inflated_passable_mask`.
When `--simulate-tracking` is enabled, the route JSON also includes a top-level
`tracking_simulation_report` object with `simulated_path`, `config`, `metrics`,
and `safety_report`.

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
