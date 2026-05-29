# path-planner

Phase 1 lunar path planner rebuilt from scratch. This package does not copy or depend on `a_gcs_ws-2.0.1`; that project is only a reference for problem framing.

## Scope

Phase 1 provides:

- core route planning models;
- semantic costmap to `cost/passable_mask` conversion;
- adapters for JSON and `dev-platform-constraints`-style arrays;
- 2D A* with 8-neighbor search and corner-cut protection;
- CLI JSON output;
- PNG/HTML diagnostics.

Phase 1 does not implement GCS, IRIS, Ackermann trajectory optimization, Drake integration, exploration target selection, observation updates, or an online planning service.

The planner returns a `geometric_path`, not a vehicle-executable trajectory.

## Install For Development

```powershell
python -m pip install -e ".[dev]"
```

## Run Tests

```powershell
python -m pytest
```

## Run Demo

```powershell
python -m path_planner.cli --input examples/demo_map.json --output-json outputs/demo/route.json --output-dir outputs/demo
```

Expected outputs:

- `outputs/demo/route.json`
- `outputs/demo/diagnostics.png`
- `outputs/demo/diagnostics.html`

## External Interface Direction

`dev-platform-constraints` can provide `cost` and `passable_mask` through `DevPlatformAdapter`. `model-explorer` can consume the route JSON fields `reachable`, `geometric_path`, `path_cost`, `diagnostics`, and `failure_reason`.
