from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import zipfile

import numpy as np

from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


FORMAL_REQUEST_CODEC_SCHEMA_V2 = "path-planner-v2-formal-request-codec/v1"
_HASH_DOMAIN = b"path-planner-v2-formal-request/v1"
_CANONICAL_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_CANONICAL_ZIP_CREATE_SYSTEM = 3
_CANONICAL_ZIP_EXTERNAL_ATTR = 0o600 << 16
_LAYER_SPECS = (
    ("elevation_m", np.dtype("<f8")),
    ("slope_deg", np.dtype("<f8")),
    ("traversable_mask", np.dtype(bool)),
    ("hard_obstacle_mask", np.dtype(bool)),
    ("observed_mask", np.dtype(bool)),
    ("confidence", np.dtype("<f8")),
)


@dataclass(frozen=True, slots=True)
class FormalRequestArtifactV2:
    metadata_json: bytes
    terrain_npz: bytes
    request_sha256: str

    def __post_init__(self) -> None:
        for name in ("metadata_json", "terrain_npz"):
            if type(getattr(self, name)) is not bytes:
                raise TypeError(f"{name} must be bytes")
        if not _is_digest(self.request_sha256):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _chunk(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big", signed=False))
    hasher.update(value)


def _canonical_layer(name: str, value: object, dtype: np.dtype) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise TypeError(f"{name} must have exact {dtype.str} dtype")
    if value.ndim != 2 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous 2-D array")
    return value


def _layer_descriptors(snapshot: TerrainSnapshotV2) -> tuple[list[dict[str, object]], tuple[bytes, ...]]:
    descriptors: list[dict[str, object]] = []
    payloads: list[bytes] = []
    for name, dtype in _LAYER_SPECS:
        array = _canonical_layer(name, getattr(snapshot, name), dtype)
        payload = array.tobytes(order="C")
        descriptors.append(
            {
                "name": name,
                "dtype": dtype.str,
                "shape": list(array.shape),
                "sha256": sha256(payload).hexdigest(),
            }
        )
        payloads.append(payload)
    return descriptors, tuple(payloads)


def _metadata_without_digest(request: PlanningRequestV2) -> tuple[dict[str, object], tuple[bytes, ...]]:
    if type(request) is not PlanningRequestV2:
        raise TypeError("request must be exact PlanningRequestV2")
    snapshot = request.terrain_snapshot
    if type(snapshot) is not TerrainSnapshotV2:
        raise TypeError("request terrain_snapshot must be exact TerrainSnapshotV2")
    descriptors, payloads = _layer_descriptors(snapshot)
    geometry = snapshot.geometry
    provenance = snapshot.provenance
    if not _is_digest(provenance.source_hash):
        raise ValueError("provenance source_hash must be a lowercase SHA-256 digest")
    return (
        {
            "codec_schema_version": FORMAL_REQUEST_CODEC_SCHEMA_V2,
            "request": {
                "accelerator_policy": request.accelerator_policy.value,
                "determinism_seed": request.determinism_seed,
                "goal_state": _pose_metadata(request.goal_state),
                "objective_profile": {
                    "distance_weight": request.objective_profile.distance_weight,
                    "energy_weight": request.objective_profile.energy_weight,
                    "risk_weight": request.objective_profile.risk_weight,
                    "time_weight": request.objective_profile.time_weight,
                },
                "platform_profile_id": request.platform_profile_id,
                "request_id": request.request_id,
                "resource_budget": {
                    "max_expanded_states": request.resource_budget.max_expanded_states,
                    "max_memory_bytes": request.resource_budget.max_memory_bytes,
                    "max_route_states": request.resource_budget.max_route_states,
                },
                "start_state": _pose_metadata(request.start_state),
                "timeout_s": request.timeout_s,
            },
            "terrain": {
                "geometry": {
                    "frame_id": geometry.frame_id,
                    "height": geometry.height,
                    "origin": list(geometry.origin),
                    "resolution_m": geometry.resolution_m,
                    "width": geometry.width,
                },
                "layers": descriptors,
                "provenance": {
                    "details": [list(item) for item in provenance.details],
                    "physical_obstacle_cells_written": provenance.physical_obstacle_cells_written,
                    "source_hash": provenance.source_hash,
                    "source_id": provenance.source_id,
                    "source_kind": provenance.source_kind,
                },
            },
        },
        payloads,
    )


def _pose_metadata(pose: PoseStateV2) -> dict[str, float]:
    return {"heading_rad": pose.heading_rad, "x_m": pose.x_m, "y_m": pose.y_m}


def _request_digest(metadata_without_digest: dict[str, object], payloads: tuple[bytes, ...]) -> str:
    hasher = sha256()
    _chunk(hasher, _HASH_DOMAIN)
    _chunk(hasher, canonical_json_bytes(metadata_without_digest))
    for (name, dtype), payload in zip(_LAYER_SPECS, payloads, strict=True):
        _chunk(hasher, name.encode("ascii"))
        _chunk(hasher, dtype.str.encode("ascii"))
        _chunk(hasher, payload)
    return hasher.hexdigest()


def encode_formal_request_v2(request: PlanningRequestV2) -> FormalRequestArtifactV2:
    metadata, payloads = _metadata_without_digest(request)
    request_sha256 = _request_digest(metadata, payloads)
    complete_metadata = {**metadata, "request_sha256": request_sha256}
    return FormalRequestArtifactV2(
        metadata_json=canonical_json_bytes(complete_metadata),
        terrain_npz=_canonical_npz_bytes(
            {
                name: getattr(request.terrain_snapshot, name)
                for name, _ in _LAYER_SPECS
            }
        ),
        request_sha256=request_sha256,
    )


def _decode_metadata(raw: bytes) -> dict[str, object]:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json must be UTF-8 JSON") from exc
    if type(parsed) is not dict or canonical_json_bytes(parsed) != raw:
        raise ValueError("metadata_json must be canonical JSON")
    expected = {"codec_schema_version", "request", "request_sha256", "terrain"}
    if set(parsed) != expected or parsed["codec_schema_version"] != FORMAL_REQUEST_CODEC_SCHEMA_V2:
        raise ValueError("metadata_json has an unsupported codec schema")
    if not _is_digest(parsed["request_sha256"]):
        raise ValueError("metadata_json request_sha256 is invalid")
    return parsed


def _canonical_npy_bytes(array: np.ndarray) -> bytes:
    output = BytesIO()
    np.lib.format.write_array(
        output,
        array,
        version=(1, 0),
        allow_pickle=False,
    )
    return output.getvalue()


def _canonical_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        for name, dtype in _LAYER_SPECS:
            array = _canonical_layer(name, arrays[name], dtype)
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=_CANONICAL_ZIP_TIMESTAMP,
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = _CANONICAL_ZIP_CREATE_SYSTEM
            info.external_attr = _CANONICAL_ZIP_EXTERNAL_ATTR
            info.internal_attr = 0
            info.flag_bits = 0
            info.comment = b""
            info.extra = b""
            archive.writestr(
                info,
                _canonical_npy_bytes(array),
                compress_type=zipfile.ZIP_STORED,
            )
    return output.getvalue()


def _decode_arrays(blob: bytes) -> dict[str, np.ndarray]:
    try:
        archive = zipfile.ZipFile(BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise ValueError("terrain_npz is not a valid NPZ archive") from exc
    with archive:
        expected_names = tuple(f"{name}.npy" for name, _ in _LAYER_SPECS)
        entries = archive.infolist()
        if tuple(entry.filename for entry in entries) != expected_names:
            raise ValueError("terrain_npz layers must use the fixed canonical order")
        if any(entry.compress_type != zipfile.ZIP_STORED for entry in entries):
            raise ValueError("terrain_npz layers must be uncompressed")
    try:
        with np.load(BytesIO(blob), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name, _ in _LAYER_SPECS}
        canonical_blob = _canonical_npz_bytes(arrays)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            "terrain_npz rejects pickle data, invalid dtype, or noncanonical arrays"
        ) from exc
    if canonical_blob != blob:
        raise ValueError("terrain_npz must use the canonical NPZ representation")
    return arrays


def _request_from_metadata(metadata: dict[str, object], arrays: dict[str, np.ndarray]) -> PlanningRequestV2:
    try:
        terrain = metadata["terrain"]
        request = metadata["request"]
        if type(terrain) is not dict or type(request) is not dict:
            raise TypeError("request and terrain metadata must be objects")
        geometry_data = terrain["geometry"]
        provenance_data = terrain["provenance"]
        descriptors = terrain["layers"]
        if type(geometry_data) is not dict or type(provenance_data) is not dict or type(descriptors) is not list:
            raise TypeError("terrain metadata is malformed")
        expected_names = tuple(name for name, _ in _LAYER_SPECS)
        if tuple(item.get("name") if type(item) is dict else None for item in descriptors) != expected_names:
            raise ValueError("terrain metadata must describe all layers in fixed order")
        geometry = FineGridGeometryV2(**geometry_data)
        provenance = TerrainProvenanceV2(
            source_kind=provenance_data["source_kind"],
            source_id=provenance_data["source_id"],
            source_hash=provenance_data["source_hash"],
            physical_obstacle_cells_written=provenance_data["physical_obstacle_cells_written"],
            details=tuple(tuple(item) for item in provenance_data["details"]),
        )
        if not _is_digest(provenance.source_hash):
            raise ValueError(
                "provenance source_hash must be a lowercase SHA-256 digest"
            )
        for descriptor, (name, dtype) in zip(descriptors, _LAYER_SPECS, strict=True):
            array = _canonical_layer(name, arrays[name], dtype)
            if descriptor != {
                "name": name,
                "dtype": dtype.str,
                "shape": list(array.shape),
                "sha256": sha256(array.tobytes(order="C")).hexdigest(),
            }:
                raise ValueError(f"terrain layer {name} hash or metadata mismatch")
        snapshot = TerrainSnapshotV2(geometry=geometry, provenance=provenance, **arrays)
        objective = request["objective_profile"]
        budget = request["resource_budget"]
        return PlanningRequestV2(
            request_id=request["request_id"],
            platform_profile_id=request["platform_profile_id"],
            start_state=PoseStateV2(**request["start_state"]),
            goal_state=PoseStateV2(**request["goal_state"]),
            terrain_snapshot=snapshot,
            objective_profile=ObjectiveProfileV2(**objective),
            resource_budget=ResourceBudgetV2(**budget),
            timeout_s=request["timeout_s"],
            accelerator_policy=AcceleratorPolicyV2(request["accelerator_policy"]),
            determinism_seed=request["determinism_seed"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "metadata_json cannot reconstruct a formal request or layer hash"
        ) from exc


def decode_formal_request_v2(artifact: FormalRequestArtifactV2) -> PlanningRequestV2:
    if type(artifact) is not FormalRequestArtifactV2:
        raise TypeError("artifact must be exact FormalRequestArtifactV2")
    metadata = _decode_metadata(artifact.metadata_json)
    if metadata["request_sha256"] != artifact.request_sha256:
        raise ValueError("artifact request_sha256 does not match metadata")
    arrays = _decode_arrays(artifact.terrain_npz)
    request = _request_from_metadata(metadata, arrays)
    reconstructed, payloads = _metadata_without_digest(request)
    actual_digest = _request_digest(reconstructed, payloads)
    if actual_digest != artifact.request_sha256:
        raise ValueError("request_sha256 does not bind the decoded request")
    expected_metadata = canonical_json_bytes({**reconstructed, "request_sha256": actual_digest})
    if expected_metadata != artifact.metadata_json:
        raise ValueError("metadata_json does not bind the decoded request")
    return request
