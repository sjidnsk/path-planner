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

This project does not implement GCS, IRIS, Ackermann trajectory optimization, Drake integration, exploration target selection, observation updates, or an online planning service.

The planner returns a `geometric_path` plus Phase 2 feasibility diagnostics, not a vehicle-executable trajectory.

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
```

Expected outputs:

- `outputs/demo/route.json`
- `outputs/demo/diagnostics.png`
- `outputs/demo/diagnostics.html`

The route JSON preserves Phase 1 fields and adds a `postprocess` object with
`platform_profile`, `constraint_warnings`, `corridor_report`, `raw_path`,
`corridor`, `smoothed_path`, `curvature_report`, and `fallback_status`.

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
