from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = PROJECT_ROOT / "schemas" / "v3"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "v3" / "schema"
DECLARED_BENCHMARK_PROFILE = (
    PROJECT_ROOT
    / "cpp"
    / "benchmarks"
    / "fixtures"
    / "declared_benchmark_profile.json"
)

SCHEMA_PATHS = (
    "common.schema.json",
    "planning-request.schema.json",
    "planning-response.schema.json",
    "reference-bundle.schema.json",
    "safety-capability-profile.schema.json",
    "planner-algorithm-config.schema.json",
    "benchmark-profile.schema.json",
    "benchmark-report.schema.json",
    "references/wheeled.schema.json",
    "references/legged.schema.json",
    "references/hopper.schema.json",
)

VALID_FIXTURES = (
    ("valid/wheel-request.json", "planning-request.schema.json"),
    ("valid/legged-request.json", "planning-request.schema.json"),
    ("valid/hopper-request.json", "planning-request.schema.json"),
    ("valid/wheel-response.json", "planning-response.schema.json"),
    ("valid/legged-response.json", "planning-response.schema.json"),
    ("valid/hopper-response.json", "planning-response.schema.json"),
)

INVALID_FIXTURES = (
    ("invalid/runtime-deadline.json", "planning-request.schema.json"),
    ("invalid/platform-state-mismatch.json", "planning-request.schema.json"),
    ("invalid/hold-with-bundle.json", "planning-response.schema.json"),
    ("invalid/component-hash-conflict.json", "planning-response.schema.json"),
    ("invalid/zero-ballistic-flight-time.json", "planning-response.schema.json"),
    (
        "invalid/activation-provenance-mismatch.json",
        "planning-response.schema.json",
    ),
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def schema_store() -> dict[str, dict[str, Any]]:
    return {
        relative_path: _load_json(SCHEMA_ROOT / relative_path)
        for relative_path in SCHEMA_PATHS
    }


@pytest.fixture(scope="module")
def schema_registry(
    schema_store: dict[str, dict[str, Any]],
) -> Registry:
    return Registry().with_resources(
        (
            schema["$id"],
            Resource.from_contents(schema),
        )
        for schema in schema_store.values()
    )


def test_all_v3_schema_documents_are_valid(
    schema_store: dict[str, dict[str, Any]],
) -> None:
    for relative_path, schema in schema_store.items():
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as error:
            pytest.fail(f"{relative_path}: {error}")


def test_declared_benchmark_profile_matches_draft_2020_12_schema(
    schema_store: dict[str, dict[str, Any]],
    schema_registry: Registry,
) -> None:
    document = _load_json(DECLARED_BENCHMARK_PROFILE)
    validator = Draft202012Validator(
        schema_store["benchmark-profile.schema.json"],
        registry=schema_registry,
    )

    errors = tuple(validator.iter_errors(document))

    assert errors == (), "\n".join(
        f"{list(error.absolute_path)}: {error.message}" for error in errors
    )


@pytest.mark.parametrize(
    ("relative_fixture", "relative_schema"),
    VALID_FIXTURES,
    ids=[fixture for fixture, _ in VALID_FIXTURES],
)
def test_valid_v3_fixture(
    relative_fixture: str,
    relative_schema: str,
    schema_store: dict[str, dict[str, Any]],
    schema_registry: Registry,
) -> None:
    document = _load_json(FIXTURE_ROOT / relative_fixture)
    validator = Draft202012Validator(
        schema_store[relative_schema],
        registry=schema_registry,
    )

    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: list(error.absolute_path),
    )

    assert errors == [], "\n".join(
        f"{relative_fixture}:{list(error.absolute_path)}: {error.message}"
        for error in errors
    )


@pytest.mark.parametrize(
    ("relative_fixture", "relative_schema"),
    INVALID_FIXTURES,
    ids=[fixture for fixture, _ in INVALID_FIXTURES],
)
def test_invalid_v3_fixture_has_at_least_one_schema_error(
    relative_fixture: str,
    relative_schema: str,
    schema_store: dict[str, dict[str, Any]],
    schema_registry: Registry,
) -> None:
    document = _load_json(FIXTURE_ROOT / relative_fixture)
    validator = Draft202012Validator(
        schema_store[relative_schema],
        registry=schema_registry,
    )

    errors = tuple(validator.iter_errors(document))

    assert errors, f"{relative_fixture} unexpectedly passed {relative_schema}"
