from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any


def canonical_json_bytes(value: object) -> bytes:
    normalized = _canonical_json_value(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_json_value(value: object) -> Any:
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_json_value(getattr(value, field.name))
            for field in fields(value)
        }

    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("mapping keys must be strings")
        return {key: _canonical_json_value(value[key]) for key in sorted(keys)}

    if isinstance(value, (tuple, list)):
        return [_canonical_json_value(item) for item in value]

    if value is None or isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("canonical JSON values must be finite")
        return value

    raise TypeError(f"unsupported type for canonical JSON: {type(value).__name__}")
