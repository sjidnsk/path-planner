from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from math import inf, nextafter, sqrt
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import path_planner.v2.wheel_corridors as wheel_corridors
from path_planner.core import Cell
from path_planner.v2.contracts import ResourceBudgetV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
    snapshot_hash,
)
from path_planner.v2.wheel_corridors import (
    WHEEL_CORRIDOR_COMPONENT_SCHEMA_V1,
    WheelCorridorGraphV1,
    build_blocked_components_v1,
    corridor_order_key_v2,
    generate_wheel_corridors_v2,
    stable_astar_v1,
    topology_signature_v1,
)
from path_planner.v2.wheel_sqp_contracts import (
    WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2,
    WheelTopologySignatureV1,
)


def _snapshot(
    width: int,
    height: int,
    *,
    unknown: set[Cell] = frozenset(),
    hard: set[Cell] = frozenset(),
    not_traversable: set[Cell] = frozenset(),
    slope: dict[Cell, float] | None = None,
    confidence: dict[Cell, float] | None = None,
) -> TerrainSnapshotV2:
    geometry = FineGridGeometryV2(width=width, height=height)
    observed = np.ones(geometry.shape, dtype=np.bool_)
    hard_mask = np.zeros(geometry.shape, dtype=np.bool_)
    traversable = np.ones(geometry.shape, dtype=np.bool_)
    slope_deg = np.zeros(geometry.shape, dtype=np.float64)
    confidence_layer = np.ones(geometry.shape, dtype=np.float64)
    for cell in unknown:
        observed[cell.y, cell.x] = False
    for cell in hard:
        hard_mask[cell.y, cell.x] = True
        traversable[cell.y, cell.x] = False
    for cell in not_traversable:
        traversable[cell.y, cell.x] = False
    for cell, value in (slope or {}).items():
        slope_deg[cell.y, cell.x] = value
    for cell, value in (confidence or {}).items():
        confidence_layer[cell.y, cell.x] = value
    return TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.zeros(geometry.shape, dtype=np.float64),
        slope_deg=slope_deg,
        traversable_mask=traversable,
        hard_obstacle_mask=hard_mask,
        observed_mask=observed,
        confidence=confidence_layer,
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="wheel-corridor-test-map",
            source_hash="input-hash",
            physical_obstacle_cells_written=True,
        ),
    )


def _deadline(*, expired: bool = False) -> PlanningDeadlineV2:
    now = 1.0 if expired else 0.0
    return PlanningDeadlineV2(0.0, 1.0, lambda: now)


def _budget(
    *, max_expanded_states: int = 100_000, max_memory_bytes: int = 0
) -> ResourceBudgetV2:
    return ResourceBudgetV2(
        max_expanded_states=max_expanded_states,
        max_route_states=10_000,
        max_memory_bytes=max_memory_bytes,
    )


def _path_hash(snapshot: TerrainSnapshotV2, cells: tuple[Cell, ...]) -> str:
    payload = {
        "cells": [[cell.x, cell.y] for cell in cells],
        "snapshot_hash": snapshot_hash(snapshot),
        "source_id": WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _component_hash(cells: tuple[Cell, ...]) -> str:
    payload = {
        "cells": [[cell.x, cell.y] for cell in cells],
        "schema_id": WHEEL_CORRIDOR_COMPONENT_SCHEMA_V1,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ordered_identity(result) -> tuple[tuple[tuple[tuple[str, int], ...], str], ...]:
    return tuple(
        (corridor.topology_signature.entries, corridor.path_hash)
        for corridor in result.corridors
    )


class _StageClock:
    def __init__(self, expires_in_stage: str) -> None:
        self.expires_in_stage = expires_in_stage
        self.stage = "preflight"

    def __call__(self) -> float:
        return 1.0 if self.stage == self.expires_in_stage else 0.0


@pytest.mark.parametrize(
    "threshold",
    [nextafter(30.0, inf), 31.0, 30, np.float64(30.0)],
)
def test_corridor_graph_rejects_any_nonexact_frozen_slope_threshold(
    threshold: object,
) -> None:
    snapshot = _snapshot(3, 3, slope={Cell(1, 1): nextafter(30.0, inf)})

    with pytest.raises((TypeError, ValueError), match="max_slope_deg"):
        WheelCorridorGraphV1.from_snapshot(snapshot, threshold)


@pytest.mark.parametrize(
    ("expired", "reason"),
    [
        (False, "wheel_sqp_resource_budget_exceeded"),
        (True, "planning_deadline_expired"),
    ],
)
def test_resource_preflight_stops_before_graph_factory(
    monkeypatch: pytest.MonkeyPatch,
    expired: bool,
    reason: str,
) -> None:
    def forbidden_factory(*args: object, **kwargs: object) -> None:
        raise AssertionError("graph factory ran before resource preflight")

    monkeypatch.setattr(
        WheelCorridorGraphV1,
        "from_snapshot",
        classmethod(forbidden_factory),
    )

    result = generate_wheel_corridors_v2(
        _snapshot(5, 3),
        Cell(0, 1),
        Cell(4, 1),
        _budget(max_memory_bytes=1),
        _deadline(expired=expired),
    )

    assert result.reason_code == reason
    assert result.corridors == ()


def test_deadline_flip_during_passable_rows_stops_before_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RowClock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 1.0 if self.calls >= 3 else 0.0

    def forbidden_clearance(*args: object, **kwargs: object) -> None:
        raise AssertionError("clearance ran after passable-row deadline")

    monkeypatch.setattr(
        WheelCorridorGraphV1,
        "_clearance_pass",
        staticmethod(forbidden_clearance),
    )
    result = generate_wheel_corridors_v2(
        _snapshot(20, 20),
        Cell(0, 0),
        Cell(19, 19),
        _budget(),
        PlanningDeadlineV2(0.0, 1.0, _RowClock()),
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.corridors == ()
    assert result.expanded_states == 0


def test_deadline_flip_during_clearance_stops_before_snapshot_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _StageClock("clearance")
    original_clearance = WheelCorridorGraphV1._clearance_pass

    def staged_clearance(*args: object, **kwargs: object):
        clock.stage = "clearance"
        return original_clearance(*args, **kwargs)

    def forbidden_snapshot_hash(*args: object, **kwargs: object) -> None:
        raise AssertionError("snapshot hash ran after clearance deadline")

    monkeypatch.setattr(
        WheelCorridorGraphV1,
        "_clearance_pass",
        staticmethod(staged_clearance),
    )
    monkeypatch.setattr(wheel_corridors, "snapshot_hash", forbidden_snapshot_hash)
    result = generate_wheel_corridors_v2(
        _snapshot(20, 20, not_traversable={Cell(10, 10)}),
        Cell(0, 0),
        Cell(19, 19),
        _budget(),
        PlanningDeadlineV2(0.0, 1.0, clock),
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.corridors == ()
    assert result.expanded_states == 0


def test_deadline_flip_during_components_stops_before_yen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _StageClock("components")
    original_components = wheel_corridors.build_blocked_components_v1

    def staged_components(*args: object, **kwargs: object):
        clock.stage = "components"
        return original_components(*args, **kwargs)

    def forbidden_yen(*args: object, **kwargs: object) -> None:
        raise AssertionError("Yen ran after component deadline")

    monkeypatch.setattr(
        wheel_corridors,
        "build_blocked_components_v1",
        staged_components,
    )
    monkeypatch.setattr(wheel_corridors, "yen_k_shortest_v1", forbidden_yen)
    result = generate_wheel_corridors_v2(
        _snapshot(9, 7, not_traversable={Cell(4, 3)}),
        Cell(0, 3),
        Cell(8, 3),
        _budget(),
        PlanningDeadlineV2(0.0, 1.0, clock),
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.corridors == ()
    assert result.expanded_states == 0


def test_deadline_flip_during_topology_stops_before_path_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _StageClock("topology")
    original_topology = wheel_corridors.topology_signature_v1

    def staged_topology(*args: object, **kwargs: object):
        clock.stage = "topology"
        return original_topology(*args, **kwargs)

    def forbidden_path_hash(*args: object, **kwargs: object) -> None:
        raise AssertionError("path hash ran after topology deadline")

    monkeypatch.setattr(wheel_corridors, "topology_signature_v1", staged_topology)
    monkeypatch.setattr(wheel_corridors, "_path_hash", forbidden_path_hash)
    result = generate_wheel_corridors_v2(
        _snapshot(5, 3),
        Cell(0, 1),
        Cell(4, 1),
        _budget(),
        PlanningDeadlineV2(0.0, 1.0, clock),
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.corridors == ()


def test_deadline_flip_before_path_hash_payload_returns_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _StageClock("path_hash")
    original_path_hash = wheel_corridors._path_hash
    original_canonical_sha256 = wheel_corridors._canonical_sha256

    def staged_path_hash(*args: object, **kwargs: object):
        clock.stage = "path_hash"
        return original_path_hash(*args, **kwargs)

    def guarded_canonical_sha256(payload: object) -> str:
        if clock.stage == "path_hash":
            raise AssertionError("path hash payload allocated after deadline")
        return original_canonical_sha256(payload)

    monkeypatch.setattr(wheel_corridors, "_path_hash", staged_path_hash)
    monkeypatch.setattr(
        wheel_corridors,
        "_canonical_sha256",
        guarded_canonical_sha256,
    )
    result = generate_wheel_corridors_v2(
        _snapshot(5, 3),
        Cell(0, 1),
        Cell(4, 1),
        _budget(),
        PlanningDeadlineV2(0.0, 1.0, clock),
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.corridors == ()


def test_corridor_graph_uses_only_current_observed_center_cell_safety() -> None:
    rejected = (
        Cell(2, 1),
        Cell(2, 2),
        Cell(2, 3),
        Cell(2, 4),
    )
    snapshot = _snapshot(
        5,
        6,
        unknown={rejected[0]},
        hard={rejected[1]},
        not_traversable={rejected[2]},
        slope={rejected[3]: nextafter(30.0, inf)},
        confidence={Cell(1, 1): 0.0},
    )

    graph = WheelCorridorGraphV1.from_snapshot(snapshot, max_slope_deg=30.0)

    assert all(not graph.passable(cell) for cell in rejected)
    assert graph.passable(Cell(1, 1))
    assert not graph.passable(Cell(-1, 1))
    assert not graph.passable(Cell(5, 1))
    assert "footprint" not in graph.safety_scope_id
    assert "center_cell" in graph.safety_scope_id


def test_neighbors_have_frozen_order_and_cell_center_axis_semantics() -> None:
    open_graph = WheelCorridorGraphV1.from_snapshot(_snapshot(3, 3), 30.0)
    assert open_graph.neighbors(Cell(1, 1)) == (
        Cell(1, 0),
        Cell(2, 1),
        Cell(1, 2),
        Cell(0, 1),
        Cell(2, 0),
        Cell(2, 2),
        Cell(0, 2),
        Cell(0, 0),
    )
    center = open_graph.snapshot.geometry.cell_center(Cell(1, 1))
    north = open_graph.snapshot.geometry.cell_center(Cell(1, 0))
    east = open_graph.snapshot.geometry.cell_center(Cell(2, 1))
    assert (center.x, center.y) == (0.75, 0.75)
    assert (north.x, north.y) == (0.75, 0.25)
    assert (east.x, east.y) == (1.25, 0.75)
    assert open_graph.clearance_cells(Cell(1, 1)) == 3


@pytest.mark.parametrize("blocked", [{Cell(1, 0)}, {Cell(0, 1)}])
def test_diagonal_corner_cut_is_forbidden_when_either_cardinal_is_blocked(
    blocked: set[Cell],
) -> None:
    blocked_graph = WheelCorridorGraphV1.from_snapshot(
        _snapshot(3, 3, not_traversable=blocked),
        30.0,
    )

    assert Cell(1, 1) not in blocked_graph.neighbors(Cell(0, 0))


def test_clearance_and_edge_cost_use_the_frozen_invalid_mask_pass() -> None:
    graph = WheelCorridorGraphV1.from_snapshot(
        _snapshot(4, 4, not_traversable={Cell(2, 2)}),
        30.0,
    )

    assert graph.clearance_cells(Cell(1, 1)) == 1
    assert graph.clearance_cells(Cell(0, 0)) == 2
    assert graph.edge_length_m(Cell(0, 0), Cell(1, 0)) == 0.5
    assert graph.edge_length_m(Cell(0, 0), Cell(1, 1)) == 0.5 * sqrt(2.0)
    assert graph.edge_cost(Cell(0, 0), Cell(1, 0)) == 0.5 * (1.0 + 0.10 / 3.0)


def test_stable_astar_uses_lexicographic_tie_break_and_shared_expansion_count() -> None:
    graph = WheelCorridorGraphV1.from_snapshot(
        _snapshot(3, 3, not_traversable={Cell(1, 1)}),
        30.0,
    )

    path, expanded = stable_astar_v1(
        graph,
        Cell(0, 1),
        Cell(2, 1),
        budget=_budget(),
        deadline=_deadline(),
    )

    assert path == (
        Cell(0, 1),
        Cell(0, 0),
        Cell(1, 0),
        Cell(2, 0),
        Cell(2, 1),
    )
    assert expanded == 8


def test_blocked_components_are_four_connected_ordered_and_hashed() -> None:
    snapshot = _snapshot(
        5,
        5,
        not_traversable={Cell(1, 1), Cell(2, 2), Cell(2, 3)},
    )
    graph = WheelCorridorGraphV1.from_snapshot(snapshot, 30.0)

    components = build_blocked_components_v1(graph)

    assert [component.cells for component in components] == [
        (Cell(1, 1),),
        (Cell(2, 2), Cell(2, 3)),
    ]
    assert all(len(component.component_hash) == 64 for component in components)
    assert len({component.component_hash for component in components}) == 2
    assert [component.component_hash for component in components] == [
        _component_hash(component.cells) for component in components
    ]
    assert components == tuple(sorted(components, key=lambda item: item.cell_order_key))
    assert [component.cut_q4 for component in components] == [
        ((5, 5), (0, 5)),
        ((9, 15), (9, 20)),
    ]


def test_quarter_cell_cut_freezes_all_boundary_offsets_and_orientations() -> None:
    graph = WheelCorridorGraphV1.from_snapshot(
        _snapshot(
            7,
            7,
            not_traversable={Cell(1, 3), Cell(3, 1), Cell(5, 3), Cell(3, 5)},
        ),
        30.0,
    )

    components = build_blocked_components_v1(graph)

    assert [(component.boundary_name, component.cut_q4) for component in components] == [
        ("top", ((15, 5), (15, 0))),
        ("left", ((5, 13), (0, 13))),
        ("right", ((23, 15), (28, 15))),
        ("bottom", ((13, 23), (13, 28))),
    ]


def test_topology_signature_uses_exact_signed_cut_crossings() -> None:
    graph = WheelCorridorGraphV1.from_snapshot(
        _snapshot(7, 5, not_traversable={Cell(3, 2)}),
        30.0,
    )
    components = build_blocked_components_v1(graph)
    above = (
        Cell(0, 2), Cell(0, 1), Cell(1, 1), Cell(2, 1),
        Cell(3, 1), Cell(4, 1), Cell(5, 1), Cell(6, 1), Cell(6, 2),
    )
    below = (
        Cell(0, 2), Cell(0, 3), Cell(1, 3), Cell(2, 3),
        Cell(3, 3), Cell(4, 3), Cell(5, 3), Cell(6, 3), Cell(6, 2),
    )

    above_signature = topology_signature_v1(above, components)
    below_signature = topology_signature_v1(below, components)
    reversed_above_signature = topology_signature_v1(tuple(reversed(above)), components)
    cancelling = (
        Cell(2, 1),
        Cell(3, 1),
        Cell(4, 1),
        Cell(4, 0),
        Cell(3, 0),
        Cell(2, 0),
    )
    cancelling_signature = topology_signature_v1(cancelling, components)

    assert type(above_signature) is WheelTopologySignatureV1
    assert above_signature != below_signature
    assert {abs(count) for _, count in above_signature.entries} == {1}
    assert {count for _, count in below_signature.entries} == {0}
    assert [count for _, count in reversed_above_signature.entries] == [
        -count for _, count in above_signature.entries
    ]
    assert {count for _, count in cancelling_signature.entries} == {0}


def _three_route_snapshot() -> TerrainSnapshotV2:
    return _snapshot(
        9,
        9,
        not_traversable=(
            {Cell(x, 3) for x in range(2, 7)}
            | {Cell(x, 5) for x in range(2, 7)}
        ),
    )


def test_three_homotopy_classes_are_stably_sorted_and_hash_bound() -> None:
    snapshot = _three_route_snapshot()

    result = generate_wheel_corridors_v2(
        snapshot,
        Cell(0, 4),
        Cell(8, 4),
        _budget(),
        _deadline(),
    )

    assert result.reason_code == "wheel_sqp_corridors_ready"
    assert len(result.corridors) == 3
    assert len({corridor.topology_signature for corridor in result.corridors}) == 3
    assert result.corridors == tuple(sorted(result.corridors, key=corridor_order_key_v2))
    assert [corridor.path_hash for corridor in result.corridors] == [
        _path_hash(snapshot, corridor.cells) for corridor in result.corridors
    ]
    assert all(corridor.corridor_hash == corridor.path_hash for corridor in result.corridors)
    assert all(
        corridor.source_id == WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2
        for corridor in result.corridors
    )


def test_near_duplicate_paths_do_not_fill_the_three_corridor_quota() -> None:
    snapshot = _snapshot(9, 7, not_traversable={Cell(4, 3)})

    result = generate_wheel_corridors_v2(
        snapshot,
        Cell(0, 3),
        Cell(8, 3),
        _budget(),
        _deadline(),
    )

    assert result.reason_code == "wheel_sqp_corridors_ready"
    assert len(result.corridors) == 2
    assert len({corridor.topology_signature for corridor in result.corridors}) == 2


@pytest.mark.parametrize(
    ("snapshot", "budget", "deadline", "reason"),
    [
        (
            _snapshot(5, 3, not_traversable={Cell(2, 0), Cell(2, 1), Cell(2, 2)}),
            _budget(),
            _deadline(),
            "wheel_sqp_no_2d_corridor",
        ),
        (
            _snapshot(5, 3),
            _budget(max_expanded_states=0),
            _deadline(),
            "wheel_sqp_corridor_budget_exceeded",
        ),
        (
            _snapshot(5, 3),
            _budget(max_memory_bytes=1),
            _deadline(),
            "wheel_sqp_resource_budget_exceeded",
        ),
        (
            _snapshot(5, 3),
            _budget(),
            _deadline(expired=True),
            "planning_deadline_expired",
        ),
    ],
)
def test_corridor_generation_has_stable_failure_taxonomy(
    snapshot: TerrainSnapshotV2,
    budget: ResourceBudgetV2,
    deadline: PlanningDeadlineV2,
    reason: str,
) -> None:
    result = generate_wheel_corridors_v2(
        snapshot,
        Cell(0, 1),
        Cell(4, 1),
        budget,
        deadline,
    )

    assert result.reason_code == reason
    assert result.corridors == ()


def test_first_astar_and_all_yen_spurs_share_one_exact_expansion_ledger() -> None:
    snapshot = _snapshot(5, 3)
    graph = WheelCorridorGraphV1.from_snapshot(snapshot, 30.0)
    first_path, first_expansions = stable_astar_v1(
        graph,
        Cell(0, 1),
        Cell(4, 1),
        budget=_budget(),
        deadline=_deadline(),
    )
    assert first_path is not None
    assert first_expansions == 5

    result = generate_wheel_corridors_v2(
        snapshot,
        Cell(0, 1),
        Cell(4, 1),
        _budget(max_expanded_states=first_expansions),
        _deadline(),
    )

    assert result.reason_code == "wheel_sqp_corridor_budget_exceeded"
    assert result.expanded_states == first_expansions
    assert result.corridors == ()


def test_memory_accounting_and_deadline_precedence_are_fail_closed() -> None:
    snapshot = _snapshot(5, 3)
    graph_only_bytes = 5 * 3 * 48
    initial_search_record_bytes = 128

    before_record = generate_wheel_corridors_v2(
        snapshot,
        Cell(0, 1),
        Cell(4, 1),
        _budget(max_memory_bytes=graph_only_bytes),
        _deadline(),
    )
    after_first_pop = generate_wheel_corridors_v2(
        snapshot,
        Cell(0, 1),
        Cell(4, 1),
        _budget(max_memory_bytes=graph_only_bytes + initial_search_record_bytes),
        _deadline(),
    )
    expired_and_tiny = generate_wheel_corridors_v2(
        snapshot,
        Cell(0, 1),
        Cell(4, 1),
        _budget(max_memory_bytes=1),
        _deadline(expired=True),
    )

    assert before_record.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert before_record.expanded_states == 0
    assert after_first_pop.reason_code == "wheel_sqp_resource_budget_exceeded"
    assert after_first_pop.expanded_states == 1
    assert expired_and_tiny.reason_code == "planning_deadline_expired"
    assert all(
        result.corridors == ()
        for result in (before_record, after_first_pop, expired_and_tiny)
    )


def test_deadline_flip_before_neighbor_batch_returns_no_partial_corridor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.calls = 0
            self.stage = "preflight"

        def __call__(self) -> float:
            if self.stage != "astar":
                return 0.0
            self.calls += 1
            return 0.0 if self.calls < 4 else 1.0

    clock = _Clock()
    original_astar = wheel_corridors._stable_astar

    def staged_astar(*args: object, **kwargs: object):
        clock.stage = "astar"
        return original_astar(*args, **kwargs)

    monkeypatch.setattr(wheel_corridors, "_stable_astar", staged_astar)
    result = generate_wheel_corridors_v2(
        _snapshot(5, 3),
        Cell(0, 1),
        Cell(4, 1),
        _budget(),
        PlanningDeadlineV2(0.0, 1.0, clock),
    )

    assert result.reason_code == "planning_deadline_expired"
    assert result.expanded_states == 1
    assert result.corridors == ()


def test_repeated_and_copied_equivalent_snapshots_have_identical_corridors() -> None:
    snapshot = _three_route_snapshot()
    copied = replace(
        snapshot,
        elevation_m=np.array(snapshot.elevation_m, copy=True),
        slope_deg=np.array(snapshot.slope_deg, copy=True),
        traversable_mask=np.array(snapshot.traversable_mask, copy=True),
        hard_obstacle_mask=np.array(snapshot.hard_obstacle_mask, copy=True),
        observed_mask=np.array(snapshot.observed_mask, copy=True),
        confidence=np.array(snapshot.confidence, copy=True),
    )

    calls = [
        generate_wheel_corridors_v2(
            candidate, Cell(0, 4), Cell(8, 4), _budget(), _deadline()
        )
        for candidate in (snapshot, snapshot, copied)
    ]

    assert snapshot_hash(snapshot) == snapshot_hash(copied)
    assert _ordered_identity(calls[0]) == _ordered_identity(calls[1])
    assert _ordered_identity(calls[0]) == _ordered_identity(calls[2])


def test_confidence_is_not_guide_safety_but_safety_layers_change_snapshot_identity() -> None:
    base = _three_route_snapshot()
    confidence_only = _snapshot(
        9,
        9,
        not_traversable=(
            {Cell(x, 3) for x in range(2, 7)}
            | {Cell(x, 5) for x in range(2, 7)}
        ),
        confidence={Cell(1, 1): 0.125},
    )
    safety_changed = _snapshot(
        9,
        9,
        not_traversable=(
            {Cell(x, 3) for x in range(2, 7)}
            | {Cell(x, 5) for x in range(2, 7)}
            | {Cell(4, 4)}
        ),
    )

    base_result = generate_wheel_corridors_v2(
        base, Cell(0, 4), Cell(8, 4), _budget(), _deadline()
    )
    confidence_result = generate_wheel_corridors_v2(
        confidence_only, Cell(0, 4), Cell(8, 4), _budget(), _deadline()
    )
    safety_result = generate_wheel_corridors_v2(
        safety_changed, Cell(0, 4), Cell(8, 4), _budget(), _deadline()
    )

    assert [item.cells for item in base_result.corridors] == [
        item.cells for item in confidence_result.corridors
    ]
    assert [item.topology_signature for item in base_result.corridors] == [
        item.topology_signature for item in confidence_result.corridors
    ]
    assert snapshot_hash(base) != snapshot_hash(confidence_only)
    assert [item.path_hash for item in base_result.corridors] != [
        item.path_hash for item in confidence_result.corridors
    ]
    assert snapshot_hash(base) != snapshot_hash(safety_changed)
    assert [item.cells for item in base_result.corridors] != [
        item.cells for item in safety_result.corridors
    ]


@pytest.mark.parametrize("safety_layer", ["observed", "hard", "traversable", "slope"])
def test_each_safety_layer_change_can_change_corridor_and_snapshot_identity(
    safety_layer: str,
) -> None:
    blocked = Cell(2, 1)
    kwargs: dict[str, object] = {}
    if safety_layer == "observed":
        kwargs["unknown"] = {blocked}
    elif safety_layer == "hard":
        kwargs["hard"] = {blocked}
    elif safety_layer == "traversable":
        kwargs["not_traversable"] = {blocked}
    else:
        kwargs["slope"] = {blocked: nextafter(30.0, inf)}
    base = _snapshot(5, 3)
    changed = _snapshot(5, 3, **kwargs)

    base_result = generate_wheel_corridors_v2(
        base, Cell(0, 1), Cell(4, 1), _budget(), _deadline()
    )
    changed_result = generate_wheel_corridors_v2(
        changed, Cell(0, 1), Cell(4, 1), _budget(), _deadline()
    )

    assert snapshot_hash(base) != snapshot_hash(changed)
    assert base_result.corridors[0].cells != changed_result.corridors[0].cells


def test_hash_seed_subprocesses_have_identical_ordered_signature_and_path_hash() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = r'''
import json
import numpy as np
from path_planner.core import Cell
from path_planner.v2.contracts import ResourceBudgetV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import FineGridGeometryV2, TerrainProvenanceV2, TerrainSnapshotV2
from path_planner.v2.wheel_corridors import generate_wheel_corridors_v2

geometry = FineGridGeometryV2(9, 9)
traversable = np.ones(geometry.shape, dtype=np.bool_)
traversable[3, 2:7] = False
traversable[5, 2:7] = False
snapshot = TerrainSnapshotV2(
    geometry=geometry,
    elevation_m=np.zeros(geometry.shape, dtype=np.float64),
    slope_deg=np.zeros(geometry.shape, dtype=np.float64),
    traversable_mask=traversable,
    hard_obstacle_mask=np.zeros(geometry.shape, dtype=np.bool_),
    observed_mask=np.ones(geometry.shape, dtype=np.bool_),
    confidence=np.ones(geometry.shape, dtype=np.float64),
    provenance=TerrainProvenanceV2(
        "measured_terrain/v1",
        "wheel-corridor-test-map",
        "input-hash",
        True,
    ),
)
result = generate_wheel_corridors_v2(
    snapshot,
    Cell(0, 4),
    Cell(8, 4),
    ResourceBudgetV2(100000, 10000, 0),
    PlanningDeadlineV2(0.0, 1.0, lambda: 0.0),
)
print(json.dumps([[
    list(corridor.topology_signature.entries), corridor.path_hash
] for corridor in result.corridors], separators=(",", ":")))
'''
    outputs = []
    for seed in ("1", "777"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(repo_root / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())

    assert outputs[0] == outputs[1]
