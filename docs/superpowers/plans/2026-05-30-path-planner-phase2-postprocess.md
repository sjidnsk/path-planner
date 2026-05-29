# Path Planner Phase 2 Postprocess Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add corridor generation, line-of-sight smoothing, curvature post-check, and postprocess JSON/diagnostic output after A* without implementing GCS, IRIS, Ackermann optimization, Drake, or service APIs.

**Architecture:** Phase 2 adds a focused `path_planner.postprocess` package. `search` remains unchanged and returns A* results; `postprocess` consumes `CostGrid` plus `PlanResult` and returns a serializable `PostprocessResult`; adapters, CLI, and diagnostics expose the result while preserving Phase 1 route fields.

**Tech Stack:** Python 3.12, dataclasses, enum, math, NumPy, Matplotlib Agg, pytest, standard-library JSON/argparse/pathlib.

---

## File Structure

- Create `src/path_planner/postprocess/__init__.py`: public exports.
- Create `src/path_planner/postprocess/models.py`: corridor, smoothing, curvature, fallback, and aggregate result dataclasses.
- Create `src/path_planner/postprocess/corridor.py`: conservative passable-mask corridor generation.
- Create `src/path_planner/postprocess/smoothing.py`: line-of-sight and shortcut smoothing.
- Create `src/path_planner/postprocess/curvature.py`: discrete curvature post-check.
- Create `src/path_planner/postprocess/pipeline.py`: one-call postprocess runner.
- Modify `src/path_planner/adapters/json_io.py`: optional route serialization with postprocess payload.
- Modify `src/path_planner/diagnostics/render.py`: raw/smoothed path overlay and postprocess summary.
- Modify `src/path_planner/cli.py`: run postprocess after successful A*.
- Modify `README.md`: describe Phase 2 behavior and non-goals.
- Add tests under `tests/`: `test_postprocess_corridor.py`, `test_postprocess_smoothing.py`, `test_postprocess_curvature.py`, `test_postprocess_pipeline.py`, and update CLI diagnostics tests.

---

### Task 1: Phase 2 Documents

- [x] Write `docs/superpowers/specs/2026-05-30-path-planner-phase2-design.md`.
- [x] Write this implementation plan.
- [x] Run `python -m pytest`.
- [x] Commit docs with `docs: add phase 2 postprocess plan`.

### Task 2: Postprocess Model Contract

- [x] Write failing tests for serializable result models.
- [x] Implement `models.py` and package exports.
- [x] Verify model tests pass.
- [x] Commit with `feat: add postprocess result models`.

### Task 3: Corridor Generation

- [ ] Write failing tests for normal corridor, edge corridor, and blocked path failure.
- [ ] Implement conservative corridor generation from `CostGrid` and path cells.
- [ ] Verify corridor tests pass.
- [ ] Commit with `feat: add path corridor generation`.

### Task 4: Line-of-sight Smoothing

- [ ] Write failing tests for shortcut success, obstacle-blocked shortcut, and invalid path fallback.
- [ ] Implement grid line-of-sight and shortcut smoothing.
- [ ] Verify smoothing tests pass.
- [ ] Commit with `feat: add path smoothing shortcut`.

### Task 5: Curvature Post-check

- [ ] Write failing tests for straight path, gentle path, and sharp-turn violation.
- [ ] Implement discrete curvature report.
- [ ] Verify curvature tests pass.
- [ ] Commit with `feat: add curvature post-check`.

### Task 6: Pipeline, JSON, CLI, Diagnostics

- [ ] Write failing pipeline and CLI tests for the `postprocess` JSON payload.
- [ ] Implement postprocess pipeline and route serialization extension.
- [ ] Update CLI to run postprocess after A* success.
- [ ] Update diagnostics to overlay raw and smoothed paths and include curvature/fallback summary.
- [ ] Verify updated tests pass.
- [ ] Commit with `feat: expose postprocess outputs`.

### Task 7: Final Verification And README

- [ ] Update README with Phase 2 capabilities, environment, commands, and non-goals.
- [ ] Run `python -m pytest`.
- [ ] Run CLI demo and inspect generated route JSON fields.
- [ ] Check `git status --short --ignored`.
- [ ] Commit docs with `docs: describe phase 2 postprocess workflow`.

## Self-Review

- Spec coverage: corridor, smoothing, curvature post-check, fallback preservation, JSON/CLI/diagnostics, README, and verification are covered.
- Explicit exclusions: no GCS, IRIS, Ackermann optimizer, Drake, or service API tasks are present.
- Compatibility: Phase 1 route fields remain top-level and unchanged; Phase 2 additions live under `postprocess`.
