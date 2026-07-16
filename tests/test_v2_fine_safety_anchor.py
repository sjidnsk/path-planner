from dataclasses import FrozenInstanceError
from importlib import import_module

import numpy as np
import pytest

from path_planner.core import Cell
from path_planner.v2.contracts import ValidationEvidenceV2, ValidationLevelV2


def _terrain():
    return import_module("path_planner.v2.terrain")


def _snapshot(terrain):
    shape = (4, 4)
    slope = np.zeros(shape)
    slope[0, 0] = 30.0
    slope[1, 1] = 89.0
    slope[3, 3] = np.nextafter(30.0, np.inf)
    traversable = np.ones(shape, dtype=bool)
    hard = np.zeros(shape, dtype=bool)
    observed = np.ones(shape, dtype=bool)
    confidence = np.full(shape, 0.8)

    observed[1, 1] = False
    hard[1, 1] = True
    traversable[1, 1] = False
    hard[2, 2] = True
    traversable[2, 2] = False
    traversable[0, 2] = False

    return terrain.TerrainSnapshotV2(
        geometry=terrain.FineGridGeometryV2(
            width=4,
            height=4,
            origin=(0.0, 0.0),
            frame_id="moon",
        ),
        elevation_m=np.zeros(shape),
        slope_deg=slope,
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=observed,
        confidence=confidence,
        provenance=terrain.TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="synthetic-fixture",
            source_hash="synthetic-fixture-hash",
            physical_obstacle_cells_written=False,
            details=(("scenario", "gate-1b"),),
        ),
    )


def _anchor():
    terrain = _terrain()
    snapshot = _snapshot(terrain)
    return terrain, snapshot, terrain.FineSafetyAnchorV2(snapshot)


@pytest.mark.parametrize(
    ("cell", "reason_code", "passed"),
    [
        (Cell(-1, 0), "terrain_out_of_bounds", False),
        (Cell(1, 1), "terrain_unknown", False),
        (Cell(2, 2), "terrain_hard_obstacle", False),
        (Cell(2, 0), "terrain_not_traversable", False),
        (Cell(3, 3), "terrain_slope_exceeded", False),
        (Cell(0, 0), "terrain_safe", True),
    ],
)
def test_query_uses_stable_fail_closed_reason_precedence(cell, reason_code, passed):
    terrain, snapshot, anchor = _anchor()

    query = anchor.query(cell, max_slope_deg=30.0)

    assert isinstance(query, terrain.SafetyQueryV2)
    assert query.cell == cell
    assert query.reason_code == reason_code
    assert query.passed is passed
    assert query.snapshot_hash == terrain.snapshot_hash(snapshot)
    assert query.validation_level is ValidationLevelV2.L2


def test_unknown_precedes_and_does_not_leak_hidden_obstacle_or_slope_values():
    _, _, anchor = _anchor()

    query = anchor.query(Cell(1, 1), max_slope_deg=30.0)

    assert query.reason_code == "terrain_unknown"
    assert query.slope_deg is None
    assert query.confidence is None


def test_slope_equal_to_threshold_passes_but_nextafter_above_fails():
    _, _, anchor = _anchor()

    equal = anchor.query(Cell(0, 0), max_slope_deg=30.0)
    above = anchor.query(Cell(3, 3), max_slope_deg=30.0)

    assert equal.reason_code == "terrain_safe"
    assert equal.passed is True
    assert equal.slope_deg == 30.0
    assert equal.confidence == 0.8
    assert above.reason_code == "terrain_slope_exceeded"
    assert above.passed is False
    assert above.slope_deg == np.nextafter(30.0, np.inf)


def test_out_of_bounds_query_does_not_expose_layer_values():
    _, _, anchor = _anchor()

    query = anchor.query(Cell(4, 0))

    assert query.reason_code == "terrain_out_of_bounds"
    assert query.slope_deg is None
    assert query.confidence is None


@pytest.mark.parametrize(
    "threshold",
    [True, -0.1, float("nan"), float("inf"), float("-inf"), "30"],
)
def test_query_rejects_invalid_max_slope_thresholds(threshold):
    _, _, anchor = _anchor()

    with pytest.raises((TypeError, ValueError), match="max_slope_deg"):
        anchor.query(Cell(0, 0), max_slope_deg=threshold)


def test_query_requires_cell_and_safety_query_is_frozen_and_slotted():
    _, _, anchor = _anchor()

    with pytest.raises(TypeError, match="Cell"):
        anchor.query((0, 0))
    query = anchor.query(Cell(0, 0))
    assert not hasattr(query, "__dict__")
    with pytest.raises(FrozenInstanceError):
        query.passed = False


def test_anchor_query_and_cached_identity_survive_all_layer_mutation_attempts():
    terrain, snapshot, anchor = _anchor()
    cell = Cell(0, 0)
    query_before = anchor.query(cell)
    hash_before = terrain.snapshot_hash(snapshot)

    for name in (
        "elevation_m",
        "slope_deg",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence",
    ):
        layer = getattr(snapshot, name)
        with pytest.raises(ValueError, match="read-only"):
            layer.flat[0] = layer.flat[0]
        with pytest.raises(ValueError, match="WRITEABLE"):
            layer.setflags(write=True)

    assert anchor.query(cell) == query_before
    assert terrain.snapshot_hash(snapshot) == hash_before
    assert anchor.query(cell).snapshot_hash == terrain.snapshot_hash(snapshot)


def test_validate_cells_supports_generator_and_preserves_duplicate_order():
    _, _, anchor = _anchor()
    cells = (Cell(0, 0), Cell(0, 0), Cell(1, 0))

    evidence = anchor.validate_cells((cell for cell in cells), max_slope_deg=30.0)

    assert isinstance(evidence, ValidationEvidenceV2)
    assert evidence.level is ValidationLevelV2.L2
    assert evidence.passed is True
    assert evidence.checks == ("terrain_safe", "terrain_safe", "terrain_safe")


def test_validate_cells_fails_closed_on_empty_generator():
    _, _, anchor = _anchor()

    evidence = anchor.validate_cells((cell for cell in ()), max_slope_deg=30.0)

    assert evidence.level is ValidationLevelV2.L2
    assert evidence.passed is False
    assert evidence.checks == ("terrain_empty_input",)


def test_validate_cells_stops_at_first_failure_with_stable_ordered_checks():
    _, _, anchor = _anchor()

    evidence = anchor.validate_cells(
        (cell for cell in (Cell(0, 0), Cell(2, 0), Cell(2, 2))),
        max_slope_deg=30.0,
    )

    assert evidence.passed is False
    assert evidence.checks == ("terrain_safe", "terrain_not_traversable")


def test_validate_cells_does_not_consume_generator_after_first_failure():
    _, _, anchor = _anchor()

    def cells():
        yield Cell(0, 0)
        yield Cell(1, 1)
        raise AssertionError("generator consumed after first failure")

    evidence = anchor.validate_cells(cells(), max_slope_deg=30.0)

    assert evidence.passed is False
    assert evidence.checks == ("terrain_safe", "terrain_unknown")


def test_validate_cells_validates_threshold_even_for_empty_input():
    _, _, anchor = _anchor()

    with pytest.raises(ValueError, match="max_slope_deg"):
        anchor.validate_cells((), max_slope_deg=float("nan"))
