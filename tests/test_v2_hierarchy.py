from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from importlib import import_module
import inspect
import json
from numbers import Real

import numpy as np
import pytest

from path_planner.core import Cell
from path_planner.v2.contracts import ValidationLevelV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


def _anchor(
    *,
    width: int = 4,
    height: int = 4,
    slope_deg: np.ndarray | None = None,
    traversable_mask: np.ndarray | None = None,
    hard_obstacle_mask: np.ndarray | None = None,
    observed_mask: np.ndarray | None = None,
) -> FineSafetyAnchorV2:
    shape = (height, width)
    snapshot = TerrainSnapshotV2(
        geometry=FineGridGeometryV2(width=width, height=height),
        elevation_m=np.zeros(shape),
        slope_deg=np.zeros(shape) if slope_deg is None else slope_deg,
        traversable_mask=(
            np.ones(shape, dtype=bool)
            if traversable_mask is None
            else traversable_mask
        ),
        hard_obstacle_mask=(
            np.zeros(shape, dtype=bool)
            if hard_obstacle_mask is None
            else hard_obstacle_mask
        ),
        observed_mask=(
            np.ones(shape, dtype=bool) if observed_mask is None else observed_mask
        ),
        confidence=np.ones(shape),
        provenance=TerrainProvenanceV2(
            source_kind="measured_terrain/v1",
            source_id="hierarchy-test",
            source_hash="hierarchy-test-source",
            physical_obstacle_cells_written=True,
        ),
    )
    return FineSafetyAnchorV2(snapshot)


def test_safe_hint_at_2r_requires_all_four_covered_fine_cells_safe() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")

    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(),
        max_slope_deg=30.0,
    )

    hint = built.hint(2, Cell(0, 0))
    assert hint.status is hierarchy.HierarchyHintStatusV2.SAFE_HINT
    assert hint.fine_cells == (
        Cell(0, 0),
        Cell(1, 0),
        Cell(0, 1),
        Cell(1, 1),
    )
    assert hint.reason_codes == ("terrain_safe",)


def test_public_contracts_are_exact_frozen_and_slotted() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(), max_slope_deg=30.0
    )
    hint = built.hint(1, Cell(0, 0))

    assert tuple(member.value for member in hierarchy.HierarchyHintStatusV2) == (
        "safe_hint",
        "unknown",
        "blocked_hint",
    )
    assert not hasattr(hint, "__dict__")
    assert not hasattr(built, "__dict__")
    with pytest.raises(FrozenInstanceError):
        hint.scale = 2
    with pytest.raises(FrozenInstanceError):
        built.max_slope_deg = 1.0


def test_direct_hierarchy_construction_cannot_bypass_anchor_build() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(), max_slope_deg=30.0
    )

    with pytest.raises(TypeError, match="build"):
        hierarchy.ConservativeHierarchyV2(
            geometry=built.geometry,
            snapshot_hash=built.snapshot_hash,
            max_slope_deg=built.max_slope_deg,
            hints=built.hints,
        )


@pytest.mark.parametrize(
    "reason_code",
    [
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_slope_exceeded",
    ],
)
def test_complete_block_is_blocked_hint_only_for_known_nontraversable_cells(
    reason_code: str,
) -> None:
    shape = (2, 2)
    kwargs: dict[str, np.ndarray] = {}
    if reason_code == "terrain_hard_obstacle":
        kwargs["traversable_mask"] = np.zeros(shape, dtype=bool)
        kwargs["hard_obstacle_mask"] = np.ones(shape, dtype=bool)
    elif reason_code == "terrain_not_traversable":
        kwargs["traversable_mask"] = np.zeros(shape, dtype=bool)
    else:
        kwargs["slope_deg"] = np.full(shape, 30.000001)
    hierarchy = import_module("path_planner.v2.hierarchy")

    hint = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=2, height=2, **kwargs),
        max_slope_deg=30.0,
    ).hint(2, Cell(0, 0))

    assert hint.status is hierarchy.HierarchyHintStatusV2.BLOCKED_HINT
    assert hint.reason_codes == (reason_code,)


def test_safe_block_with_one_blocked_cell_remains_unknown() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    traversable = np.ones((2, 2), dtype=bool)
    traversable[1, 1] = False

    hint = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=2, height=2, traversable_mask=traversable),
        max_slope_deg=30.0,
    ).hint(2, Cell(0, 0))

    assert hint.status is hierarchy.HierarchyHintStatusV2.UNKNOWN
    assert hint.reason_codes == ("terrain_not_traversable", "terrain_safe")


def test_4r_safe_hint_requires_all_sixteen_fine_cells_safe() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    traversable = np.ones((4, 4), dtype=bool)
    traversable[3, 3] = False

    hint = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=4, height=4, traversable_mask=traversable),
        max_slope_deg=30.0,
    ).hint(4, Cell(0, 0))

    assert len(hint.fine_cells) == 16
    assert hint.status is hierarchy.HierarchyHintStatusV2.UNKNOWN
    assert hint.reason_codes == ("terrain_not_traversable", "terrain_safe")


def test_unknown_plus_hidden_obstacle_queries_every_cell_and_remains_unknown() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    observed = np.ones((2, 2), dtype=bool)
    observed[0, 0] = False
    traversable = np.ones((2, 2), dtype=bool)
    traversable[1, 1] = False
    hard = np.zeros((2, 2), dtype=bool)
    hard[1, 1] = True

    hint = hierarchy.ConservativeHierarchyV2.build(
        _anchor(
            width=2,
            height=2,
            observed_mask=observed,
            traversable_mask=traversable,
            hard_obstacle_mask=hard,
        ),
        max_slope_deg=30.0,
    ).hint(2, Cell(0, 0))

    assert hint.status is hierarchy.HierarchyHintStatusV2.UNKNOWN
    assert hint.reason_codes == (
        "terrain_hard_obstacle",
        "terrain_safe",
        "terrain_unknown",
    )


def test_unknown_cell_does_not_leak_hidden_hard_obstacle_truth() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    observed = np.zeros((1, 1), dtype=bool)
    traversable = np.zeros((1, 1), dtype=bool)
    hard = np.ones((1, 1), dtype=bool)

    hint = hierarchy.ConservativeHierarchyV2.build(
        _anchor(
            width=1,
            height=1,
            observed_mask=observed,
            traversable_mask=traversable,
            hard_obstacle_mask=hard,
        ),
        max_slope_deg=30.0,
    ).hint(1, Cell(0, 0))

    assert hint.status is hierarchy.HierarchyHintStatusV2.UNKNOWN
    assert hint.reason_codes == ("terrain_unknown",)


def test_nondivisible_boundaries_are_unknown_with_exact_in_bounds_mapping() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=5, height=3),
        max_slope_deg=30.0,
    )

    boundary = built.hint(4, Cell(1, 0))
    assert boundary.status is hierarchy.HierarchyHintStatusV2.UNKNOWN
    assert boundary.fine_cells == (Cell(4, 0), Cell(4, 1), Cell(4, 2))
    assert boundary.reason_codes == ("hierarchy_incomplete_block", "terrain_safe")
    assert built.fine_cells(4, Cell(1, 0)) == boundary.fine_cells


def test_exact_threshold_is_safe_and_just_above_is_slope_blocked() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    exact = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=2, height=2, slope_deg=np.full((2, 2), 30.0)),
        max_slope_deg=30.0,
    )
    above = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=2, height=2, slope_deg=np.full((2, 2), 30.000001)),
        max_slope_deg=30.0,
    )

    assert exact.hint(2, Cell(0, 0)).status is hierarchy.HierarchyHintStatusV2.SAFE_HINT
    assert above.hint(2, Cell(0, 0)).status is hierarchy.HierarchyHintStatusV2.BLOCKED_HINT


def test_iter_hints_is_scale_then_row_major_stable() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=5, height=3), max_slope_deg=30.0
    )

    assert tuple((hint.cell.x, hint.cell.y) for hint in built.iter_hints(2)) == (
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
    )
    assert tuple((hint.scale, hint.cell.y, hint.cell.x) for hint in built.hints) == tuple(
        sorted((hint.scale, hint.cell.y, hint.cell.x) for hint in built.hints)
    )


@pytest.mark.parametrize("max_slope_deg", [True, "30", float("nan"), float("inf"), -0.1])
def test_build_rejects_invalid_thresholds(max_slope_deg: object) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    with pytest.raises((TypeError, ValueError), match="max_slope_deg"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(), max_slope_deg=max_slope_deg
        )


def test_build_normalizes_unrepresentably_large_threshold_to_value_error() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    with pytest.raises(ValueError, match="max_slope_deg.*finite"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(), max_slope_deg=10**400
        )


def test_build_rejects_invalid_anchor_and_deadline_types() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    with pytest.raises(TypeError, match="FineSafetyAnchorV2"):
        hierarchy.ConservativeHierarchyV2.build(object(), max_slope_deg=30.0)
    with pytest.raises(TypeError, match="PlanningDeadlineV2"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(), max_slope_deg=30.0, deadline=object()
        )


@pytest.mark.parametrize("scale", [0, 3, True, 1.0, "2"])
def test_accessors_reject_invalid_scales(scale: object) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(), max_slope_deg=30.0
    )
    with pytest.raises((TypeError, ValueError), match="scale"):
        built.hint(scale, Cell(0, 0))
    with pytest.raises((TypeError, ValueError), match="scale"):
        built.iter_hints(scale)
    with pytest.raises((TypeError, ValueError), match="scale"):
        built.fine_cells(scale, Cell(0, 0))


@pytest.mark.parametrize(
    "cell",
    [object(), Cell(-1, 0), Cell(0, -1), Cell(True, 0), Cell(0.0, 0), Cell(9, 9)],
)
def test_accessors_reject_invalid_coarse_cells(cell: object) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(), max_slope_deg=30.0
    )
    with pytest.raises((TypeError, ValueError), match="cell"):
        built.hint(2, cell)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_type",
        "cell",
        "validation_level",
        "passed_reason",
        "safe_slope",
        "slope_reason",
        "out_of_bounds",
    ],
)
def test_malformed_anchor_queries_raise_stable_contract_error(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    original = FineSafetyAnchorV2.query

    def malformed(self, cell, max_slope_deg=30.0):
        query = original(self, cell, max_slope_deg)
        if mutation == "wrong_type":
            return object()
        if mutation == "cell":
            return replace(query, cell=Cell(cell.x + 1, cell.y))
        if mutation == "validation_level":
            object.__setattr__(query, "validation_level", ValidationLevelV2.L1)
        elif mutation == "passed_reason":
            object.__setattr__(query, "passed", False)
        elif mutation == "safe_slope":
            object.__setattr__(query, "slope_deg", max_slope_deg + 1.0)
        elif mutation == "slope_reason":
            object.__setattr__(query, "reason_code", "terrain_slope_exceeded")
            object.__setattr__(query, "passed", False)
        elif mutation == "out_of_bounds":
            object.__setattr__(query, "reason_code", "terrain_out_of_bounds")
            object.__setattr__(query, "passed", False)
            object.__setattr__(query, "slope_deg", None)
            object.__setattr__(query, "confidence", None)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", malformed)

    with pytest.raises(hierarchy.HierarchyContractErrorV2, match="query contract"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1), max_slope_deg=30.0
        )


def test_nonexact_query_getter_exception_is_contained_without_property_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    getter_accessed = False

    class ExplosiveQuery:
        @property
        def cell(self):
            nonlocal getter_accessed
            getter_accessed = True
            raise RuntimeError("malformed query getter")

    monkeypatch.setattr(
        FineSafetyAnchorV2,
        "query",
        lambda self, cell, max_slope_deg=30.0: ExplosiveQuery(),
    )
    with pytest.raises(
        hierarchy.HierarchyContractErrorV2,
        match="fine safety anchor query contract mismatch",
    ):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1), max_slope_deg=30.0
        )
    assert getter_accessed is False


def test_exact_query_malicious_cell_equality_is_contained_before_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    original = FineSafetyAnchorV2.query
    equality_called = False

    class ExplosiveEquality:
        def __eq__(self, other):
            nonlocal equality_called
            equality_called = True
            raise RuntimeError("malformed cell equality")

    def malformed(self, cell, max_slope_deg=30.0):
        query = replace(
            original(self, cell, max_slope_deg),
            cell=Cell(cell.x, cell.y),
        )
        object.__setattr__(query.cell, "x", ExplosiveEquality())
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", malformed)
    with pytest.raises(
        hierarchy.HierarchyContractErrorV2,
        match="fine safety anchor query contract mismatch",
    ):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1), max_slope_deg=30.0
        )
    assert equality_called is False


def test_anchor_query_timeout_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")

    def timed_out(self, cell, max_slope_deg=30.0):
        raise TimeoutError("anchor timeout")

    monkeypatch.setattr(FineSafetyAnchorV2, "query", timed_out)
    with pytest.raises(TimeoutError, match="anchor timeout"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1), max_slope_deg=30.0
        )


@pytest.mark.parametrize("field_name", ["slope_deg", "confidence"])
def test_malformed_query_unrepresentable_numeric_is_contained(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    original = FineSafetyAnchorV2.query

    def malformed(self, cell, max_slope_deg=30.0):
        query = original(self, cell, max_slope_deg)
        object.__setattr__(query, field_name, 10**10000)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", malformed)
    with pytest.raises(
        hierarchy.HierarchyContractErrorV2,
        match="fine safety anchor query contract mismatch",
    ):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1), max_slope_deg=30.0
        )


@pytest.mark.parametrize("field_name", ["slope_deg", "confidence"])
@pytest.mark.parametrize("error_type", [TimeoutError, RuntimeError])
def test_malformed_query_numeric_conversion_exception_is_contained(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    error_type: type[Exception],
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    original = FineSafetyAnchorV2.query

    class ForgedReal:
        def __float__(self):
            raise error_type("forged numeric conversion")

    Real.register(ForgedReal)

    def malformed(self, cell, max_slope_deg=30.0):
        query = original(self, cell, max_slope_deg)
        object.__setattr__(query, field_name, ForgedReal())
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", malformed)
    with pytest.raises(
        hierarchy.HierarchyContractErrorV2,
        match="fine safety anchor query contract mismatch",
    ):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1), max_slope_deg=30.0
        )


def test_snapshot_identity_drift_raises_stable_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    original = FineSafetyAnchorV2.query
    call_count = 0

    def drifting(self, cell, max_slope_deg=30.0):
        nonlocal call_count
        call_count += 1
        query = original(self, cell, max_slope_deg)
        if call_count > 1:
            return replace(query, snapshot_hash="1" * 64)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", drifting)

    with pytest.raises(hierarchy.HierarchyContractErrorV2, match="snapshot identity"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=2, height=1), max_slope_deg=30.0
        )


def test_stable_forged_query_hash_is_rejected_against_snapshot_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    original = FineSafetyAnchorV2.query

    def forged(self, cell, max_slope_deg=30.0):
        return replace(
            original(self, cell, max_slope_deg),
            snapshot_hash="1" * 64,
        )

    monkeypatch.setattr(FineSafetyAnchorV2, "query", forged)
    with pytest.raises(hierarchy.HierarchyContractErrorV2, match="snapshot identity"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=2, height=1), max_slope_deg=30.0
        )


def test_cached_anchor_hash_must_match_snapshot_identity() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    anchor = _anchor(width=1, height=1)
    object.__setattr__(anchor, "_snapshot_hash", "1" * 64)

    with pytest.raises(hierarchy.HierarchyContractErrorV2, match="snapshot identity"):
        hierarchy.ConservativeHierarchyV2.build(anchor, max_slope_deg=30.0)


def test_snapshot_identity_is_rechecked_after_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    original = FineSafetyAnchorV2.query
    replacement = _anchor(width=1, height=1, slope_deg=np.ones((1, 1)))

    def drifting(self, cell, max_slope_deg=30.0):
        query = original(self, cell, max_slope_deg)
        object.__setattr__(self, "snapshot", replacement.snapshot)
        return query

    monkeypatch.setattr(FineSafetyAnchorV2, "query", drifting)
    with pytest.raises(hierarchy.HierarchyContractErrorV2, match="snapshot identity"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1), max_slope_deg=30.0
        )


def test_anchor_query_exception_is_wrapped_as_stable_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")

    def broken_query(self, cell, max_slope_deg=30.0):
        raise RuntimeError("raw provider detail")

    monkeypatch.setattr(FineSafetyAnchorV2, "query", broken_query)
    with pytest.raises(hierarchy.HierarchyContractErrorV2, match="query failed"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1), max_slope_deg=30.0
        )


def test_expired_deadline_propagates_timeout_without_querying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    queried = False

    def forbidden_query(self, cell, max_slope_deg=30.0):
        nonlocal queried
        queried = True
        raise AssertionError("expired build queried anchor")

    monkeypatch.setattr(FineSafetyAnchorV2, "query", forbidden_query)
    deadline = PlanningDeadlineV2(0.0, 1.0, lambda: 1.0)

    with pytest.raises(TimeoutError, match="deadline"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1),
            max_slope_deg=30.0,
            deadline=deadline,
        )
    assert queried is False


def test_deadline_expiry_during_construction_propagates_timeout() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    values = iter((0.0, 0.0, 2.0))
    deadline = PlanningDeadlineV2(0.0, 1.0, lambda: next(values))

    with pytest.raises(TimeoutError, match="deadline"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1),
            max_slope_deg=30.0,
            deadline=deadline,
        )


def test_deadline_is_checked_after_final_contract_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    materialized = False
    original_post_init = hierarchy.ConservativeHierarchyV2.__post_init__

    def materializing_post_init(self):
        nonlocal materialized
        original_post_init(self)
        materialized = True

    monkeypatch.setattr(
        hierarchy.ConservativeHierarchyV2,
        "__post_init__",
        materializing_post_init,
    )
    deadline = PlanningDeadlineV2(
        0.0,
        1.0,
        lambda: 2.0 if materialized else 0.0,
    )

    with pytest.raises(TimeoutError, match="deadline"):
        hierarchy.ConservativeHierarchyV2.build(
            _anchor(width=1, height=1),
            max_slope_deg=30.0,
            deadline=deadline,
        )


def test_repeated_builds_and_parallel_read_orders_have_identical_bytes() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    anchor = _anchor(width=5, height=3)
    first = hierarchy.ConservativeHierarchyV2.build(anchor, max_slope_deg=30.0)
    second = hierarchy.ConservativeHierarchyV2.build(anchor, max_slope_deg=30.0)
    requests = [(hint.scale, hint.cell) for hint in reversed(first.hints)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = tuple(executor.map(lambda item: first.hint(*item), requests))
    serial = tuple(first.hint(*item) for item in requests)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(parallel) == canonical_json_bytes(serial)


def test_hint_accessors_return_defensive_deep_copies() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=2, height=2), max_slope_deg=30.0
    )
    leaked_hint = built.hint(2, Cell(0, 0))
    leaked_fine_cell = leaked_hint.fine_cells[0]
    object.__setattr__(leaked_hint, "status", hierarchy.HierarchyHintStatusV2.UNKNOWN)
    object.__setattr__(leaked_fine_cell, "x", 99)
    iterated = built.iter_hints(2)[0]
    object.__setattr__(iterated, "reason_codes", ("terrain_unknown",))

    fresh = built.hint(2, Cell(0, 0))
    assert fresh.status is hierarchy.HierarchyHintStatusV2.SAFE_HINT
    assert fresh.fine_cells[0] == Cell(0, 0)
    assert built.iter_hints(2)[0].reason_codes == ("terrain_safe",)


def test_public_hints_view_cannot_pollute_internal_or_canonical_decisions() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=2, height=2), max_slope_deg=30.0
    )
    before = canonical_json_bytes(built)
    leaked = built.hints
    target = next(
        hint for hint in leaked if hint.scale == 2 and hint.cell == Cell(0, 0)
    )
    object.__setattr__(
        target,
        "status",
        hierarchy.HierarchyHintStatusV2.BLOCKED_HINT,
    )
    object.__setattr__(target, "reason_codes", ("terrain_hard_obstacle",))

    fresh = built.hint(2, Cell(0, 0))
    assert fresh.status is hierarchy.HierarchyHintStatusV2.SAFE_HINT
    assert fresh.reason_codes == ("terrain_safe",)
    assert built.iter_hints(2)[0] == fresh
    assert built.fine_cells(2, Cell(0, 0)) == fresh.fine_cells
    assert canonical_json_bytes(built) == before
    assert built.hints is not leaked
    assert built.hints[0] is not leaked[0]
    payload = json.loads(before)
    assert "hints" in payload
    assert "_hints" not in payload


def test_public_geometry_view_cannot_pollute_internal_or_canonical_decisions() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=2, height=2), max_slope_deg=30.0
    )
    before = canonical_json_bytes(built)
    leaked = built.geometry
    object.__setattr__(leaked, "width", 99)

    assert built.geometry.width == 2
    assert built.hint(2, Cell(0, 0)).status is hierarchy.HierarchyHintStatusV2.SAFE_HINT
    assert len(built.iter_hints(2)) == 1
    assert built.fine_cells(2, Cell(0, 0)) == (
        Cell(0, 0),
        Cell(1, 0),
        Cell(0, 1),
        Cell(1, 1),
    )
    assert canonical_json_bytes(built) == before
    assert built["geometry"] is not leaked


def test_exact_hint_lookup_does_not_iterate_all_internal_hints() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=50, height=50), max_slope_deg=30.0
    )
    expected = built.hint(4, Cell(12, 12))

    class NonIterableTuple(tuple):
        def __iter__(self):
            raise AssertionError("exact lookup iterated the full hint tuple")

    object.__setattr__(built, "_hints", NonIterableTuple(built._hints))

    assert built.hint(4, Cell(12, 12)) == expected
    assert built.fine_cells(4, Cell(12, 12)) == expected.fine_cells


def test_private_hint_index_never_controls_order_or_leaks_to_canonical_output() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    built = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=5, height=3), max_slope_deg=30.0
    )
    peer = hierarchy.ConservativeHierarchyV2.build(
        _anchor(width=5, height=3), max_slope_deg=30.0
    )
    expected_hints = built.hints
    expected_scale_two = built.iter_hints(2)
    before = canonical_json_bytes(built)
    expected_keys = tuple(
        (hint.scale, hint.cell.y, hint.cell.x) for hint in built._hints
    )

    assert tuple(built._hint_index) == expected_keys
    assert all(
        built._hint_index[key] is hint
        for key, hint in zip(expected_keys, built._hints, strict=True)
    )
    assert not hasattr(built, "hint_index")
    with pytest.raises(KeyError):
        _ = built["_hint_index"]
    payload = json.loads(before)
    assert set(payload) == {"geometry", "hints", "max_slope_deg", "snapshot_hash"}
    assert "_hint_index" not in payload

    reversed_index = dict(reversed(tuple(built._hint_index.items())))
    object.__setattr__(built, "_hint_index", reversed_index)
    assert built.hints == expected_hints
    assert built.iter_hints(2) == expected_scale_two
    assert canonical_json_bytes(built) == before
    assert built == peer
    assert hash(built) == hash(peer)

    object.__setattr__(built, "_hint_index", {})
    with pytest.raises(
        hierarchy.HierarchyContractErrorV2,
        match="hierarchy hint coverage contract mismatch",
    ):
        built.hint(2, Cell(0, 0))


def test_hierarchy_implementation_uses_query_not_raw_truth_layers() -> None:
    hierarchy = import_module("path_planner.v2.hierarchy")
    source = inspect.getsource(hierarchy)

    assert ".query(" in source
    for forbidden in (
        "elevation_m",
        "slope_deg[",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence[",
    ):
        assert forbidden not in source
