from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from path_planner.v2 import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.formal_request_codec import (
    FormalRequestArtifactV2,
    decode_formal_request_v2,
    encode_formal_request_v2,
)
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


def _request(profile_id: str) -> PlanningRequestV2:
    geometry = FineGridGeometryV2(width=2, height=2)
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.array([[0.0, 0.1], [0.2, 0.3]], dtype="<f8"),
        slope_deg=np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f8"),
        traversable_mask=np.array([[True, True], [True, True]], dtype=bool),
        hard_obstacle_mask=np.array([[False, False], [False, False]], dtype=bool),
        observed_mask=np.array([[True, True], [True, True]], dtype=bool),
        confidence=np.array([[1.0, 1.0], [1.0, 1.0]], dtype="<f8"),
        provenance=TerrainProvenanceV2(
            source_kind="fixture/v1",
            source_id="fixture-source",
            source_hash="f" * 64,
            physical_obstacle_cells_written=False,
            details=(("revision", 1),),
        ),
    )
    return PlanningRequestV2(
        request_id=f"request-{profile_id}",
        platform_profile_id=profile_id,
        start_state=PoseStateV2(0.25, 0.25, 0.0),
        goal_state=PoseStateV2(0.75, 0.75, 0.5),
        terrain_snapshot=snapshot,
        objective_profile=ObjectiveProfileV2(distance_weight=1.0),
        resource_budget=ResourceBudgetV2(10, 20, 30),
        timeout_s=1.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=7,
    )


def test_request_codec_round_trips_all_three_platform_requests() -> None:
    for profile_id in ("wheel-formal/v1", "legged-formal/v1", "hopper-formal/v1"):
        request = _request(profile_id)

        artifact = encode_formal_request_v2(request)
        decoded = decode_formal_request_v2(artifact)

        assert decoded.request_id == request.request_id
        assert decoded.platform_profile_id == request.platform_profile_id
        assert decoded.start_state == request.start_state
        assert decoded.goal_state == request.goal_state
        assert decoded.objective_profile == request.objective_profile
        assert decoded.resource_budget == request.resource_budget
        assert decoded.timeout_s == request.timeout_s
        assert decoded.accelerator_policy is request.accelerator_policy
        assert decoded.determinism_seed == request.determinism_seed
        assert decoded.terrain_snapshot.geometry == request.terrain_snapshot.geometry
        assert decoded.terrain_snapshot.provenance == request.terrain_snapshot.provenance
        assert np.array_equal(decoded.terrain_snapshot.elevation_m, request.terrain_snapshot.elevation_m)


def test_request_codec_rejects_pickle_object_dtype_and_noncanonical_json() -> None:
    artifact = encode_formal_request_v2(_request("wheel-formal/v1"))
    metadata = json.loads(artifact.metadata_json)
    noncanonical = json.dumps(metadata, indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="canonical"):
        decode_formal_request_v2(
            FormalRequestArtifactV2(noncanonical, artifact.terrain_npz, artifact.request_sha256)
        )

    stream = io.BytesIO()
    np.savez(
        stream,
        elevation_m=np.array([[object()]], dtype=object),
        slope_deg=np.zeros((1, 1), dtype="<f8"),
        traversable_mask=np.ones((1, 1), dtype=bool),
        hard_obstacle_mask=np.zeros((1, 1), dtype=bool),
        observed_mask=np.ones((1, 1), dtype=bool),
        confidence=np.ones((1, 1), dtype="<f8"),
    )
    with pytest.raises((TypeError, ValueError), match="pickle|dtype"):
        decode_formal_request_v2(
            FormalRequestArtifactV2(artifact.metadata_json, stream.getvalue(), artifact.request_sha256)
        )

    compressed = io.BytesIO()
    np.savez_compressed(
        compressed,
        elevation_m=np.zeros((2, 2), dtype="<f8"),
        slope_deg=np.zeros((2, 2), dtype="<f8"),
        traversable_mask=np.ones((2, 2), dtype=bool),
        hard_obstacle_mask=np.zeros((2, 2), dtype=bool),
        observed_mask=np.ones((2, 2), dtype=bool),
        confidence=np.ones((2, 2), dtype="<f8"),
    )
    assert any(
        entry.compress_type != zipfile.ZIP_STORED
        for entry in zipfile.ZipFile(io.BytesIO(compressed.getvalue())).infolist()
    )
    with pytest.raises(ValueError, match="uncompressed"):
        decode_formal_request_v2(
            FormalRequestArtifactV2(artifact.metadata_json, compressed.getvalue(), artifact.request_sha256)
        )


def test_request_codec_binds_every_terrain_layer_hash_and_request_sha256() -> None:
    artifact = encode_formal_request_v2(_request("legged-formal/v1"))
    metadata = json.loads(artifact.metadata_json)

    assert tuple(layer["name"] for layer in metadata["terrain"]["layers"]) == (
        "elevation_m",
        "slope_deg",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence",
    )
    assert metadata["request_sha256"] == artifact.request_sha256

    with np.load(io.BytesIO(artifact.terrain_npz), allow_pickle=False) as archive:
        layers = {name: archive[name].copy() for name in archive.files}
    layers["slope_deg"][0, 0] = 99.0
    tampered = io.BytesIO()
    np.savez(tampered, **layers)
    with pytest.raises(ValueError, match="hash|NPZ"):
        decode_formal_request_v2(
            FormalRequestArtifactV2(artifact.metadata_json, tampered.getvalue(), artifact.request_sha256)
        )
