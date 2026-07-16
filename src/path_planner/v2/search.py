from __future__ import annotations

from dataclasses import dataclass, field
from math import copysign, isfinite
from threading import Lock


def _require_exact_nonempty_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be an exact nonempty string")
    return value


def _normalize_nonnegative_real(value: object, *, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be an exact finite nonnegative real")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(
            f"{field_name} must be an exact finite nonnegative real"
        ) from None
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be an exact finite nonnegative real")
    if normalized == 0.0:
        return 0.0
    return normalized


def _checked_score(
    path_cost: float,
    heuristic: float,
    *,
    key_name: str,
) -> float:
    score = path_cost + heuristic
    if not isfinite(score):
        raise ValueError(f"{key_name} must be finite")
    if score == 0.0:
        return 0.0
    return score


@dataclass(frozen=True, slots=True, init=False)
class SearchQueueEntryV2:
    candidate_id: str
    state_key: tuple[int, ...]
    primitive_key: str
    path_cost: float
    anchor_heuristic: float
    auxiliary_heuristics: tuple[tuple[str, float], ...] = ()

    def __init__(
        self,
        candidate_id: object,
        state_key: object,
        primitive_key: object,
        path_cost: object,
        anchor_heuristic: object,
        auxiliary_heuristics: object = (),
    ) -> None:
        normalized_candidate_id = _require_exact_nonempty_string(
            candidate_id,
            field_name="candidate_id",
        )
        if (
            type(state_key) is not tuple
            or not state_key
            or any(type(component) is not int for component in state_key)
        ):
            raise ValueError("state_key must be a nonempty exact tuple of exact ints")
        normalized_primitive_key = _require_exact_nonempty_string(
            primitive_key,
            field_name="primitive_key",
        )
        normalized_path_cost = _normalize_nonnegative_real(
            path_cost,
            field_name="path_cost",
        )
        normalized_anchor_heuristic = _normalize_nonnegative_real(
            anchor_heuristic,
            field_name="anchor_heuristic",
        )
        _checked_score(
            normalized_path_cost,
            normalized_anchor_heuristic,
            key_name="authoritative search key",
        )
        normalized_auxiliary = _normalize_auxiliary_heuristics(
            auxiliary_heuristics,
            path_cost=normalized_path_cost,
        )

        object.__setattr__(self, "candidate_id", normalized_candidate_id)
        object.__setattr__(self, "state_key", state_key)
        object.__setattr__(self, "primitive_key", normalized_primitive_key)
        object.__setattr__(self, "path_cost", normalized_path_cost)
        object.__setattr__(self, "anchor_heuristic", normalized_anchor_heuristic)
        object.__setattr__(self, "auxiliary_heuristics", normalized_auxiliary)


def _normalize_auxiliary_heuristics(
    auxiliary_heuristics: object,
    *,
    path_cost: float,
) -> tuple[tuple[str, float], ...]:
    if type(auxiliary_heuristics) is not tuple:
        raise ValueError(
            "auxiliary_heuristics must be an exact tuple sorted by unique identifiers"
        )

    normalized: list[tuple[str, float]] = []
    previous_identifier: str | None = None
    for pair in auxiliary_heuristics:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError(
                "auxiliary_heuristics must contain exact identifier/value tuples"
            )
        identifier = pair[0]
        value = pair[1]
        if type(identifier) is not str or not identifier:
            raise ValueError(
                "auxiliary_heuristics identifiers must be exact nonempty strings"
            )
        if previous_identifier is not None and identifier <= previous_identifier:
            raise ValueError(
                "auxiliary_heuristics must be sorted with unique identifiers"
            )
        normalized_value = _normalize_nonnegative_real(
            value,
            field_name="auxiliary_heuristics value",
        )
        _checked_score(
            path_cost,
            normalized_value,
            key_name="auxiliary search key",
        )
        normalized.append((identifier, normalized_value))
        previous_identifier = identifier
    return tuple(normalized)


def _authoritative_key(
    entry: SearchQueueEntryV2,
) -> tuple[float, float, tuple[int, ...], str, str]:
    return (
        entry.path_cost + entry.anchor_heuristic,
        entry.path_cost,
        entry.state_key,
        entry.primitive_key,
        entry.candidate_id,
    )


def _auxiliary_value(
    entry: SearchQueueEntryV2,
    heuristic_id: str,
) -> float | None:
    for identifier, value in entry.auxiliary_heuristics:
        if identifier == heuristic_id:
            return value
        if identifier > heuristic_id:
            break
    return None


def _validated_entry_copy(entry: object) -> SearchQueueEntryV2:
    if type(entry) is not SearchQueueEntryV2:
        raise ValueError("entry must be a valid exact SearchQueueEntryV2")
    try:
        candidate_id = entry.candidate_id
        state_key = entry.state_key
        primitive_key = entry.primitive_key
        path_cost = entry.path_cost
        anchor_heuristic = entry.anchor_heuristic
        auxiliary_heuristics = entry.auxiliary_heuristics
    except (AttributeError, TypeError, ValueError):
        raise ValueError("entry must be a valid exact SearchQueueEntryV2") from None
    if (
        type(path_cost) is not float
        or type(anchor_heuristic) is not float
        or (path_cost == 0.0 and copysign(1.0, path_cost) < 0.0)
        or (
            anchor_heuristic == 0.0
            and copysign(1.0, anchor_heuristic) < 0.0
        )
        or type(auxiliary_heuristics) is not tuple
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[1]) is not float
            or (pair[1] == 0.0 and copysign(1.0, pair[1]) < 0.0)
            for pair in auxiliary_heuristics
        )
    ):
        raise ValueError("entry must be a valid exact SearchQueueEntryV2")
    try:
        return SearchQueueEntryV2(
            candidate_id=candidate_id,
            state_key=state_key,
            primitive_key=primitive_key,
            path_cost=path_cost,
            anchor_heuristic=anchor_heuristic,
            auxiliary_heuristics=auxiliary_heuristics,
        )
    except (AttributeError, OverflowError, TypeError, ValueError):
        raise ValueError("entry must be a valid exact SearchQueueEntryV2") from None


@dataclass(slots=True, eq=False)
class StableSearchQueueV2:
    _entries: dict[str, SearchQueueEntryV2] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _retired_entries: dict[str, SearchQueueEntryV2] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _lock: object = field(default_factory=Lock, init=False, repr=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def extend(self, entries: object) -> None:
        with self._lock:
            if type(entries) is not tuple or any(
                type(entry) is not SearchQueueEntryV2 for entry in entries
            ):
                raise ValueError(
                    "entries must be an exact tuple of exact SearchQueueEntryV2 values"
                )

            validated_entries = tuple(
                _validated_entry_copy(entry) for entry in entries
            )

            batch: dict[str, SearchQueueEntryV2] = {}
            for entry in validated_entries:
                existing = batch.get(entry.candidate_id)
                if existing is not None and existing != entry:
                    raise ValueError("conflicting candidate_id in search queue batch")
                batch[entry.candidate_id] = entry

            for candidate_id, entry in batch.items():
                existing = self._entries.get(candidate_id)
                if existing is None:
                    existing = self._retired_entries.get(candidate_id)
                if existing is not None and existing != entry:
                    raise ValueError("conflicting candidate_id in stable search queue")

            merged = dict(self._entries)
            for candidate_id, entry in batch.items():
                if candidate_id not in self._retired_entries:
                    merged[candidate_id] = entry
            self._entries = {
                entry.candidate_id: entry
                for entry in sorted(merged.values(), key=_authoritative_key)
            }

    def peek_anchor(self) -> SearchQueueEntryV2 | None:
        with self._lock:
            if not self._entries:
                return None
            return _validated_entry_copy(next(iter(self._entries.values())))

    def pop_anchor(self) -> SearchQueueEntryV2:
        with self._lock:
            if not self._entries:
                raise IndexError("stable search queue is empty")
            candidate_id = next(iter(self._entries))
            entry = self._entries[candidate_id]
            copied = _validated_entry_copy(entry)
            self._entries.pop(candidate_id)
            self._retired_entries[candidate_id] = entry
            return copied

    def suggest(self, heuristic_id: object) -> SearchQueueEntryV2 | None:
        normalized_id = _require_exact_nonempty_string(
            heuristic_id,
            field_name="heuristic_id",
        )
        with self._lock:
            candidates: list[
                tuple[
                    tuple[
                        float,
                        float,
                        float,
                        tuple[int, ...],
                        str,
                        str,
                    ],
                    SearchQueueEntryV2,
                ]
            ] = []
            for entry in self._entries.values():
                auxiliary = _auxiliary_value(entry, normalized_id)
                if auxiliary is None:
                    continue
                candidates.append(
                    (
                        (
                            entry.path_cost + auxiliary,
                            *_authoritative_key(entry),
                        ),
                        entry,
                    )
                )
            if not candidates:
                return None
            return _validated_entry_copy(
                min(candidates, key=lambda candidate: candidate[0])[1]
            )

    def discard(self, candidate_id: object) -> bool:
        normalized_id = _require_exact_nonempty_string(
            candidate_id,
            field_name="candidate_id",
        )
        with self._lock:
            entry = self._entries.pop(normalized_id, None)
            if entry is not None:
                self._retired_entries[normalized_id] = entry
                return True
            return False
