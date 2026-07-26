from __future__ import annotations

import io
import json
from pathlib import PurePosixPath
import warnings
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


def _request(
    profile_id: str,
    *,
    source_hash: str = "f" * 64,
    first_slope_deg: float = 1.0,
) -> PlanningRequestV2:
    geometry = FineGridGeometryV2(width=2, height=2)
    snapshot = TerrainSnapshotV2(
        geometry=geometry,
        elevation_m=np.array([[0.0, 0.1], [0.2, 0.3]], dtype="<f8"),
        slope_deg=np.array(
            [[first_slope_deg, 2.0], [3.0, 4.0]],
            dtype="<f8",
        ),
        traversable_mask=np.array([[True, True], [True, True]], dtype=bool),
        hard_obstacle_mask=np.array([[False, False], [False, False]], dtype=bool),
        observed_mask=np.array([[True, True], [True, True]], dtype=bool),
        confidence=np.array([[1.0, 1.0], [1.0, 1.0]], dtype="<f8"),
        provenance=TerrainProvenanceV2(
            source_kind="fixture/v1",
            source_id="fixture-source",
            source_hash=source_hash,
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
        reencoded = encode_formal_request_v2(decoded)
        assert reencoded.metadata_json == artifact.metadata_json
        assert reencoded.terrain_npz == artifact.terrain_npz
        assert reencoded.request_sha256 == artifact.request_sha256


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


def _rewrite_npz(blob: bytes, mutation: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob), "r") as source:
        members = [(entry, source.read(entry)) for entry in source.infolist()]

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as target:
        selected = members[:-1] if mutation == "missing_member" else members
        if mutation == "member_order":
            selected = members[1:] + members[:1]
        for index, (entry, payload) in enumerate(selected):
            filename = entry.filename
            if mutation == "path_trick" and index == 0:
                filename = f"../{PurePosixPath(filename).name}"
            info = zipfile.ZipInfo(filename, date_time=entry.date_time)
            info.compress_type = (
                zipfile.ZIP_DEFLATED
                if mutation == "compressed_member" and index == 0
                else zipfile.ZIP_STORED
            )
            info.create_system = entry.create_system
            info.external_attr = entry.external_attr
            if mutation == "member_comment" and index == 0:
                info.comment = b"not-canonical"
            if mutation == "member_extra" and index == 0:
                info.extra = b"\x01\x00\x00\x00"
            if mutation == "timestamp" and index == 0:
                info.date_time = (2026, 7, 27, 12, 0, 0)
            if mutation == "external_attributes" and index == 0:
                info.external_attr ^= 0x00010000
            target.writestr(info, payload)
            if mutation == "duplicate_member" and index == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    target.writestr(info, payload)
        if mutation == "extra_member":
            target.writestr("extra.npy", b"not-an-array")
        if mutation == "archive_comment":
            target.comment = b"not-canonical"

    rewritten = output.getvalue()
    if mutation == "flag_bits":
        rewritten_bytes = bytearray(rewritten)
        local = rewritten_bytes.find(b"PK\x03\x04")
        central = rewritten_bytes.find(b"PK\x01\x02")
        assert local >= 0 and central >= 0
        rewritten_bytes[local + 6 : local + 8] = (0x0001).to_bytes(2, "little")
        rewritten_bytes[central + 8 : central + 10] = (0x0001).to_bytes(2, "little")
        rewritten = bytes(rewritten_bytes)
    return rewritten


@pytest.mark.parametrize(
    "mutation",
    (
        "archive_comment",
        "member_comment",
        "member_extra",
        "timestamp",
        "flag_bits",
        "external_attributes",
        "path_trick",
        "duplicate_member",
        "extra_member",
        "missing_member",
        "member_order",
        "compressed_member",
    ),
)
def test_request_codec_rejects_noncanonical_zip_representation(mutation: str) -> None:
    artifact = encode_formal_request_v2(_request("wheel-formal/v1"))
    mutated = _rewrite_npz(artifact.terrain_npz, mutation)
    assert mutated != artifact.terrain_npz

    with pytest.raises(ValueError, match="canonical|fixed|uncompressed|NPZ"):
        decode_formal_request_v2(
            FormalRequestArtifactV2(
                artifact.metadata_json,
                mutated,
                artifact.request_sha256,
            )
        )


@pytest.mark.parametrize(
    ("layer_name", "replacement"),
    (
        ("elevation_m", np.zeros((2, 2), dtype="<f4")),
        ("slope_deg", np.zeros((1, 4), dtype="<f8")),
    ),
)
def test_request_codec_rejects_wrong_layer_dtype_or_shape(
    layer_name: str,
    replacement: np.ndarray,
) -> None:
    artifact = encode_formal_request_v2(_request("legged-formal/v1"))
    with zipfile.ZipFile(io.BytesIO(artifact.terrain_npz), "r") as archive:
        members = [
            (entry, archive.read(entry))
            for entry in archive.infolist()
        ]
    replacement_stream = io.BytesIO()
    np.lib.format.write_array(
        replacement_stream,
        replacement,
        version=(1, 0),
        allow_pickle=False,
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        for entry, payload in members:
            if entry.filename == f"{layer_name}.npy":
                payload = replacement_stream.getvalue()
            archive.writestr(entry, payload, compress_type=zipfile.ZIP_STORED)

    with pytest.raises(ValueError, match="dtype|hash|reconstruct|canonical"):
        decode_formal_request_v2(
            FormalRequestArtifactV2(
                artifact.metadata_json,
                stream.getvalue(),
                artifact.request_sha256,
            )
        )


def test_request_codec_rejects_non_sha256_provenance_source_hash() -> None:
    with pytest.raises(ValueError, match="source_hash"):
        encode_formal_request_v2(
            _request("hopper-formal/v1", source_hash="not-a-lowercase-sha256")
        )


def test_request_codec_rejects_canonical_npz_layer_hash_drift() -> None:
    artifact = encode_formal_request_v2(_request("wheel-formal/v1"))
    changed = encode_formal_request_v2(
        _request("wheel-formal/v1", first_slope_deg=9.0)
    )

    with pytest.raises(ValueError, match="hash"):
        decode_formal_request_v2(
            FormalRequestArtifactV2(
                artifact.metadata_json,
                changed.terrain_npz,
                artifact.request_sha256,
            )
        )
