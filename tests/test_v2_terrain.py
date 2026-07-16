from dataclasses import FrozenInstanceError
from importlib import import_module

import numpy as np
import pytest

from path_planner.core import Cell, GridSpec, WorldPoint


def _terrain():
    return import_module("path_planner.v2.terrain")


def _provenance(terrain, **overrides):
    values = {
        "source_kind": "measured_terrain/v1",
        "source_id": "terrain-source-001",
        "source_hash": "source-hash-001",
        "physical_obstacle_cells_written": True,
        "details": (("mission", "south-pole"), ("tile", 7)),
    }
    values.update(overrides)
    return terrain.TerrainProvenanceV2(**values)


def _geometry(terrain, **overrides):
    values = {
        "width": 3,
        "height": 2,
        "origin": (10.0, -2.0),
        "frame_id": "moon",
    }
    values.update(overrides)
    return terrain.FineGridGeometryV2(**values)


def _layer_values(shape=(2, 3)):
    return {
        "elevation_m": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        "slope_deg": np.full(shape, 12.5, dtype=np.float32),
        "traversable_mask": np.ones(shape, dtype=bool),
        "hard_obstacle_mask": np.zeros(shape, dtype=bool),
        "observed_mask": np.ones(shape, dtype=bool),
        "confidence": np.full(shape, 0.75, dtype=np.float32),
    }


def _snapshot(terrain, *, geometry=None, provenance=None, **overrides):
    geometry = geometry or _geometry(terrain)
    values = _layer_values(geometry.shape)
    values.update(overrides)
    return terrain.TerrainSnapshotV2(
        geometry=geometry,
        provenance=provenance or _provenance(terrain),
        **values,
    )


def test_fine_geometry_uses_cell_centers_for_nonzero_origin_and_nonsquare_grid():
    terrain = _terrain()
    geometry = _geometry(terrain)

    assert geometry.resolution_m == 0.5
    assert geometry.shape == (2, 3)
    assert geometry.cell_center(Cell(0, 0)) == WorldPoint(10.25, -1.75)
    assert geometry.cell_center(Cell(2, 1)) == WorldPoint(11.25, -1.25)
    assert geometry.world_to_cell(WorldPoint(10.25, -1.75)) == Cell(0, 0)
    assert geometry.world_to_cell(WorldPoint(11.25, -1.25)) == Cell(2, 1)
    assert geometry.in_bounds(Cell(2, 1))
    assert not geometry.in_bounds(Cell(3, 1))
    assert not hasattr(geometry, "__dict__")
    with pytest.raises(FrozenInstanceError):
        geometry.width = 9


def test_v1_grid_spec_keeps_corner_semantics_while_v2_uses_cell_centers():
    terrain = _terrain()
    legacy = GridSpec(width=2, height=2, resolution=0.5, origin=(10.0, -2.0))
    geometry = terrain.FineGridGeometryV2(
        width=2,
        height=2,
        origin=(10.0, -2.0),
        frame_id="moon",
    )

    assert legacy.cell_to_world(Cell(0, 0)) == WorldPoint(10.0, -2.0)
    assert geometry.cell_center(Cell(0, 0)) == WorldPoint(10.25, -1.75)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"width": 0}, ValueError),
        ({"width": -1}, ValueError),
        ({"width": True}, TypeError),
        ({"width": 1.0}, TypeError),
        ({"height": 0}, ValueError),
        ({"height": False}, TypeError),
        ({"origin": (0.0,)}, ValueError),
        ({"origin": (0.0, float("nan"))}, ValueError),
        ({"origin": (float("inf"), 0.0)}, ValueError),
        ({"origin": (True, 0.0)}, TypeError),
        ({"frame_id": ""}, ValueError),
        ({"frame_id": "   "}, ValueError),
        ({"frame_id": 1}, ValueError),
        ({"resolution_m": 1.0}, ValueError),
        ({"resolution_m": True}, TypeError),
    ],
)
def test_fine_geometry_rejects_invalid_contract_values(overrides, error):
    terrain = _terrain()

    with pytest.raises(error):
        _geometry(terrain, **overrides)


def test_world_to_cell_assigns_internal_grid_lines_to_right_and_upper_cells():
    terrain = _terrain()
    geometry = _geometry(terrain)
    x_line = 10.5
    y_line = -1.5

    assert geometry.world_to_cell(WorldPoint(x_line, -1.75)) == Cell(1, 0)
    assert geometry.world_to_cell(WorldPoint(10.25, y_line)) == Cell(0, 1)
    assert geometry.world_to_cell(
        WorldPoint(np.nextafter(x_line, -np.inf), -1.75)
    ) == Cell(0, 0)
    assert geometry.world_to_cell(
        WorldPoint(np.nextafter(x_line, np.inf), -1.75)
    ) == Cell(1, 0)
    assert geometry.world_to_cell(
        WorldPoint(10.25, np.nextafter(y_line, -np.inf))
    ) == Cell(0, 0)
    assert geometry.world_to_cell(
        WorldPoint(10.25, np.nextafter(y_line, np.inf))
    ) == Cell(0, 1)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (WorldPoint(10.0, -2.0), Cell(0, 0)),
        (WorldPoint(np.nextafter(10.0, np.inf), -1.75), Cell(0, 0)),
        (WorldPoint(10.25, np.nextafter(-2.0, np.inf)), Cell(0, 0)),
        (WorldPoint(np.nextafter(11.5, -np.inf), -1.75), Cell(2, 0)),
        (WorldPoint(10.25, np.nextafter(-1.0, -np.inf)), Cell(0, 1)),
    ],
)
def test_world_to_cell_accepts_exact_lower_and_nextafter_inside_boundaries(point, expected):
    terrain = _terrain()

    assert _geometry(terrain).world_to_cell(point) == expected


@pytest.mark.parametrize(
    "point",
    [
        WorldPoint(np.nextafter(10.0, -np.inf), -1.75),
        WorldPoint(10.25, np.nextafter(-2.0, -np.inf)),
        WorldPoint(11.5, -1.75),
        WorldPoint(np.nextafter(11.5, np.inf), -1.75),
        WorldPoint(10.25, -1.0),
        WorldPoint(10.25, np.nextafter(-1.0, np.inf)),
    ],
)
def test_world_to_cell_rejects_nextafter_outside_and_exact_upper_boundaries(point):
    terrain = _terrain()

    with pytest.raises(ValueError, match="bounds"):
        _geometry(terrain).world_to_cell(point)


@pytest.mark.parametrize(
    "point",
    [
        WorldPoint(float("nan"), 0.0),
        WorldPoint(0.0, float("inf")),
        WorldPoint(True, 0.0),
    ],
)
def test_world_to_cell_rejects_nonfinite_or_boolean_coordinates(point):
    terrain = _terrain()

    with pytest.raises((TypeError, ValueError), match="finite"):
        _geometry(terrain).world_to_cell(point)


def test_cell_center_rejects_out_of_bounds_and_non_cell_values():
    terrain = _terrain()
    geometry = _geometry(terrain)

    with pytest.raises(ValueError, match="bounds"):
        geometry.cell_center(Cell(3, 0))
    with pytest.raises(TypeError, match="Cell"):
        geometry.cell_center((0, 0))


def test_provenance_is_frozen_slotted_and_preserves_sorted_scalar_details():
    terrain = _terrain()
    provenance = _provenance(
        terrain,
        details=(("a", None), ("b", False), ("c", 3), ("d", 1.25), ("e", "x")),
    )

    assert provenance.details == (
        ("a", None),
        ("b", False),
        ("c", 3),
        ("d", 1.25),
        ("e", "x"),
    )
    assert not hasattr(provenance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        provenance.source_id = "other"


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"source_kind": ""}, ValueError, "source_kind"),
        ({"source_id": " "}, ValueError, "source_id"),
        ({"source_hash": 1}, ValueError, "source_hash"),
        ({"physical_obstacle_cells_written": 0}, TypeError, "bool"),
        ({"details": {"a": 1}}, TypeError, "tuple"),
        ({"details": (("z", 1), ("a", 2))}, ValueError, "sorted"),
        ({"details": (("a", 1), ("a", 2))}, ValueError, "unique"),
        ({"details": (("a", []),)}, TypeError, "scalar"),
        ({"details": (("a", float("nan")),)}, ValueError, "finite"),
    ],
)
def test_provenance_rejects_invalid_or_mutable_values(overrides, error, message):
    terrain = _terrain()

    with pytest.raises(error, match=message):
        _provenance(terrain, **overrides)


def test_synthetic_provenance_accepts_only_explicit_false_physical_truth_flag():
    terrain = _terrain()
    source_kind = "synthetic_terrain_obstacle_proxy/v1"

    provenance = _provenance(
        terrain,
        source_kind=source_kind,
        physical_obstacle_cells_written=False,
    )

    assert provenance.source_kind == source_kind
    assert provenance.physical_obstacle_cells_written is False
    with pytest.raises(ValueError, match="synthetic.*False"):
        _provenance(
            terrain,
            source_kind=source_kind,
            physical_obstacle_cells_written=True,
        )


def test_snapshot_deep_copies_canonicalizes_and_freezes_every_layer():
    terrain = _terrain()
    sources = _layer_values()
    expected = {name: np.asarray(value).copy() for name, value in sources.items()}

    snapshot = _snapshot(terrain, **sources)

    for source in sources.values():
        source[...] = 0
    for name, before in expected.items():
        layer = getattr(snapshot, name)
        assert np.array_equal(layer, before.astype(layer.dtype))
        assert layer.flags.c_contiguous
        assert not layer.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            layer.flat[0] = layer.flat[0]
    for name in ("elevation_m", "slope_deg", "confidence"):
        assert getattr(snapshot, name).dtype.str == "<f8"
    for name in ("traversable_mask", "hard_obstacle_mask", "observed_mask"):
        assert getattr(snapshot, name).dtype == np.dtype(np.bool_)
    assert not hasattr(snapshot, "__dict__")
    with pytest.raises(FrozenInstanceError):
        snapshot.geometry = snapshot.geometry


@pytest.mark.parametrize(
    "layer_name",
    [
        "elevation_m",
        "slope_deg",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence",
    ],
)
def test_snapshot_layer_write_flags_cannot_be_reenabled(layer_name):
    terrain = _terrain()
    snapshot = _snapshot(terrain)
    layer = getattr(snapshot, layer_name)

    with pytest.raises(ValueError, match="WRITEABLE"):
        layer.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        layer.flat[0] = layer.flat[0]


@pytest.mark.parametrize(
    "layer_name",
    [
        "elevation_m",
        "slope_deg",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence",
    ],
)
def test_snapshot_rejects_non_2d_layers(layer_name):
    terrain = _terrain()
    dtype = bool if layer_name.endswith("_mask") else float
    bad = np.zeros(6, dtype=dtype)

    with pytest.raises(ValueError, match="2-D"):
        _snapshot(terrain, **{layer_name: bad})


@pytest.mark.parametrize(
    "layer_name",
    [
        "elevation_m",
        "slope_deg",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence",
    ],
)
def test_snapshot_rejects_mismatched_layer_shapes(layer_name):
    terrain = _terrain()
    dtype = bool if layer_name.endswith("_mask") else float
    bad = np.zeros((2, 2), dtype=dtype)

    with pytest.raises(ValueError, match="shape"):
        _snapshot(terrain, **{layer_name: bad})


@pytest.mark.parametrize("layer_name", ["elevation_m", "slope_deg", "confidence"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_snapshot_rejects_nonfinite_numeric_layers(layer_name, value):
    terrain = _terrain()
    bad = np.zeros((2, 3))
    bad[0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        _snapshot(terrain, **{layer_name: bad})


@pytest.mark.parametrize(
    "layer_name",
    ["traversable_mask", "hard_obstacle_mask", "observed_mask"],
)
@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (np.uint8, 0),
        (np.uint8, 1),
        (np.int64, 2),
        (np.int64, -1),
        (np.float64, 0.0),
        (np.float64, 1.0),
        (np.float64, 0.5),
        (np.float64, float("nan")),
        (np.float64, float("inf")),
        (np.float64, float("-inf")),
        (object, True),
    ],
)
def test_snapshot_rejects_every_nonbool_mask_dtype(
    layer_name,
    dtype,
    value,
):
    terrain = _terrain()
    bad = np.zeros((2, 3), dtype=dtype)
    bad[0, 0] = value

    with pytest.raises(TypeError, match="bool dtype"):
        _snapshot(terrain, **{layer_name: bad})


def test_snapshot_accepts_python_nested_bool_lists_as_exact_bool_masks():
    terrain = _terrain()
    layers = _layer_values()
    for name in ("traversable_mask", "hard_obstacle_mask", "observed_mask"):
        layers[name] = layers[name].tolist()

    snapshot = _snapshot(terrain, **layers)

    for name in ("traversable_mask", "hard_obstacle_mask", "observed_mask"):
        assert getattr(snapshot, name).dtype == np.dtype(np.bool_)


@pytest.mark.parametrize("value", [-0.1, -np.nextafter(0.0, 1.0)])
def test_snapshot_rejects_negative_slope(value):
    terrain = _terrain()
    slope = np.zeros((2, 3))
    slope[0, 0] = value

    with pytest.raises(ValueError, match="slope.*nonnegative"):
        _snapshot(terrain, slope_deg=slope)


@pytest.mark.parametrize("value", [-0.1, np.nextafter(1.0, np.inf)])
def test_snapshot_rejects_confidence_outside_closed_unit_interval(value):
    terrain = _terrain()
    confidence = np.full((2, 3), 0.5)
    confidence[0, 0] = value

    with pytest.raises(ValueError, match=r"confidence.*\[0, 1\]"):
        _snapshot(terrain, confidence=confidence)


def test_snapshot_rejects_overlapping_traversable_and_hard_obstacle_masks():
    terrain = _terrain()
    traversable = np.ones((2, 3), dtype=bool)
    hard = np.zeros((2, 3), dtype=bool)
    hard[0, 0] = True

    with pytest.raises(ValueError, match="overlap"):
        _snapshot(terrain, traversable_mask=traversable, hard_obstacle_mask=hard)


def test_snapshot_requires_exact_geometry_shape_and_typed_contracts():
    terrain = _terrain()
    wrong_shape = _layer_values((1, 1))

    with pytest.raises(ValueError, match="geometry"):
        terrain.TerrainSnapshotV2(
            geometry=_geometry(terrain),
            provenance=_provenance(terrain),
            **wrong_shape,
        )
    with pytest.raises(TypeError, match="FineGridGeometryV2"):
        terrain.TerrainSnapshotV2(
            geometry=object(),
            provenance=_provenance(terrain),
            **_layer_values(),
        )
    with pytest.raises(TypeError, match="TerrainProvenanceV2"):
        terrain.TerrainSnapshotV2(
            geometry=_geometry(terrain),
            provenance=object(),
            **_layer_values(),
        )


def test_synthetic_proxy_hard_mask_remains_nonphysical_provenance():
    terrain = _terrain()
    traversable = np.ones((2, 3), dtype=bool)
    hard = np.zeros((2, 3), dtype=bool)
    traversable[1, 2] = False
    hard[1, 2] = True
    provenance = _provenance(
        terrain,
        source_kind="synthetic_terrain_obstacle_proxy/v1",
        physical_obstacle_cells_written=False,
    )

    snapshot = _snapshot(
        terrain,
        provenance=provenance,
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
    )

    assert snapshot.hard_obstacle_mask[1, 2]
    assert snapshot.provenance.physical_obstacle_cells_written is False


def _reordered_layer(value, mode):
    if mode == "c":
        return np.array(value, order="C", copy=True)
    if mode == "f":
        return np.array(value, order="F", copy=True)
    base = np.empty((value.shape[0], value.shape[1] * 2), dtype=value.dtype)
    base[:, ::2] = value
    return base[:, ::2]


def test_snapshot_hash_is_stable_across_c_f_and_noncontiguous_view_inputs():
    terrain = _terrain()
    layers = _layer_values()
    snapshots = []
    for mode in ("c", "f", "view"):
        snapshots.append(
            _snapshot(
                terrain,
                **{name: _reordered_layer(value, mode) for name, value in layers.items()},
            )
        )

    hashes = [terrain.snapshot_hash(snapshot) for snapshot in snapshots]

    assert hashes[0] == hashes[1] == hashes[2]
    assert len(hashes[0]) == 64
    assert set(hashes[0]) <= set("0123456789abcdef")


def test_snapshot_hash_normalizes_signed_zero_in_layers_geometry_and_details():
    terrain = _terrain()
    positive_layers = _layer_values()
    negative_layers = _layer_values()
    positive_layers["elevation_m"][0, 0] = 0.0
    negative_layers["elevation_m"][0, 0] = -0.0
    positive = _snapshot(
        terrain,
        geometry=_geometry(terrain, origin=(0.0, 0.0)),
        provenance=_provenance(terrain, details=(("offset", 0.0),)),
        **positive_layers,
    )
    negative = _snapshot(
        terrain,
        geometry=_geometry(terrain, origin=(-0.0, -0.0)),
        provenance=_provenance(terrain, details=(("offset", -0.0),)),
        **negative_layers,
    )

    assert terrain.snapshot_hash(positive) == terrain.snapshot_hash(negative)


@pytest.mark.parametrize(
    "layer_name",
    [
        "elevation_m",
        "slope_deg",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence",
    ],
)
def test_snapshot_hash_changes_when_any_single_layer_changes(layer_name):
    terrain = _terrain()
    baseline_layers = _layer_values()
    changed_layers = {name: value.copy() for name, value in baseline_layers.items()}
    if layer_name == "hard_obstacle_mask":
        changed_layers["traversable_mask"][0, 0] = False
        changed_layers[layer_name][0, 0] = True
    elif layer_name == "traversable_mask":
        changed_layers[layer_name][0, 0] = False
    elif layer_name == "observed_mask":
        changed_layers[layer_name][0, 0] = False
    elif layer_name == "confidence":
        changed_layers[layer_name][0, 0] = 0.25
    else:
        changed_layers[layer_name][0, 0] += 1.0

    baseline = _snapshot(terrain, **baseline_layers)
    changed = _snapshot(terrain, **changed_layers)

    assert terrain.snapshot_hash(baseline) != terrain.snapshot_hash(changed)


def test_snapshot_hash_changes_with_geometry_or_provenance():
    terrain = _terrain()
    baseline = _snapshot(terrain)
    geometry_changed = _snapshot(terrain, geometry=_geometry(terrain, frame_id="other"))
    provenance_changed = _snapshot(
        terrain,
        provenance=_provenance(terrain, source_id="terrain-source-002"),
    )

    baseline_hash = terrain.snapshot_hash(baseline)

    assert baseline_hash != terrain.snapshot_hash(geometry_changed)
    assert baseline_hash != terrain.snapshot_hash(provenance_changed)


def test_snapshot_hash_requires_a_terrain_snapshot():
    terrain = _terrain()

    with pytest.raises(TypeError, match="TerrainSnapshotV2"):
        terrain.snapshot_hash(object())
