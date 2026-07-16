from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from itertools import permutations
from math import inf, nan
from threading import Barrier

import pytest

from path_planner.v2.search import SearchQueueEntryV2, StableSearchQueueV2
from path_planner.v2.serialization import canonical_json_bytes


def _entry(
    candidate_id: str,
    *,
    state_key: tuple[int, ...] = (0,),
    primitive_key: str = "primitive",
    path_cost: int | float = 1.0,
    anchor_heuristic: int | float = 1.0,
    auxiliary_heuristics: tuple[tuple[str, int | float], ...] = (),
) -> SearchQueueEntryV2:
    return SearchQueueEntryV2(
        candidate_id=candidate_id,
        state_key=state_key,
        primitive_key=primitive_key,
        path_cost=path_cost,
        anchor_heuristic=anchor_heuristic,
        auxiliary_heuristics=auxiliary_heuristics,
    )


def _drain(queue: StableSearchQueueV2) -> tuple[SearchQueueEntryV2, ...]:
    result: list[SearchQueueEntryV2] = []
    while len(queue):
        result.append(queue.pop_anchor())
    return tuple(result)


def _decision_projection(
    entries: tuple[SearchQueueEntryV2, ...],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (entry.candidate_id, entry.state_key, entry.primitive_key)
        for entry in entries
    )


def test_entry_is_frozen_slotted_and_normalizes_exact_real_inputs() -> None:
    entry = _entry(
        "candidate",
        state_key=(1, 2),
        path_cost=1,
        anchor_heuristic=-0.0,
        auxiliary_heuristics=(("resource", -0.0),),
    )

    assert entry.path_cost == 1.0
    assert type(entry.path_cost) is float
    assert entry.anchor_heuristic == 0.0
    assert str(entry.anchor_heuristic) == "0.0"
    assert entry.auxiliary_heuristics == (("resource", 0.0),)
    assert not hasattr(entry, "__dict__")
    with pytest.raises(FrozenInstanceError):
        entry.path_cost = 2.0  # type: ignore[misc]


def test_entry_has_deterministic_canonical_serialization() -> None:
    left = _entry(
        "candidate",
        state_key=(1, 2),
        path_cost=1,
        anchor_heuristic=-0.0,
        auxiliary_heuristics=(("resource", -0.0),),
    )
    right = _entry(
        "candidate",
        state_key=(1, 2),
        path_cost=1.0,
        anchor_heuristic=0,
        auxiliary_heuristics=(("resource", 0),),
    )

    expected = (
        b'{"anchor_heuristic":0.0,'
        b'"auxiliary_heuristics":[["resource",0.0]],'
        b'"candidate_id":"candidate","path_cost":1.0,'
        b'"primitive_key":"primitive","state_key":[1,2]}'
    )
    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("candidate_id", ""),
        ("candidate_id", _StringSubclass("candidate")),
        ("primitive_key", ""),
        ("primitive_key", _StringSubclass("primitive")),
    ),
)
def test_entry_rejects_nonexact_or_empty_identifier_fields(
    field_name: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "candidate_id": "candidate",
        "state_key": (0,),
        "primitive_key": "primitive",
        "path_cost": 1.0,
        "anchor_heuristic": 1.0,
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError, match="exact nonempty string"):
        SearchQueueEntryV2(**kwargs)


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


class _TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    "state_key",
    (
        (),
        [0],
        _TupleSubclass((0,)),
        (True,),
        (_IntSubclass(0),),
        (0.0,),
    ),
)
def test_entry_rejects_invalid_state_key(state_key: object) -> None:
    with pytest.raises(ValueError, match="state_key"):
        _entry("candidate", state_key=state_key)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    (
        True,
        _IntSubclass(1),
        _FloatSubclass(1.0),
        nan,
        inf,
        -inf,
        -0.01,
        10**1000,
    ),
)
@pytest.mark.parametrize("field_name", ("path_cost", "anchor_heuristic"))
def test_entry_rejects_invalid_numeric_inputs(
    field_name: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {field_name: value}
    with pytest.raises(ValueError, match="finite nonnegative real"):
        _entry("candidate", **kwargs)  # type: ignore[arg-type]


def test_entry_rejects_nonfinite_authoritative_sum() -> None:
    with pytest.raises(ValueError, match="authoritative search key"):
        _entry(
            "candidate",
            path_cost=1.0e308,
            anchor_heuristic=1.0e308,
        )


def test_entry_rejects_nonfinite_auxiliary_sum() -> None:
    with pytest.raises(ValueError, match="auxiliary search key"):
        _entry(
            "candidate",
            path_cost=1.0e308,
            anchor_heuristic=0.0,
            auxiliary_heuristics=(("resource", 1.0e308),),
        )


@pytest.mark.parametrize(
    "auxiliary_heuristics",
    (
        [],
        _TupleSubclass(()),
        (["resource", 1.0],),
        (("", 1.0),),
        ((_StringSubclass("resource"), 1.0),),
        (("resource", True),),
        (("resource", _IntSubclass(1)),),
        (("resource", _FloatSubclass(1.0)),),
        (("resource", nan),),
        (("resource", inf),),
        (("resource", -0.1),),
        (("z", 1.0), ("a", 2.0)),
        (("resource", 1.0), ("resource", 1.0)),
    ),
)
def test_entry_rejects_invalid_unsorted_or_duplicate_auxiliary_heuristics(
    auxiliary_heuristics: object,
) -> None:
    with pytest.raises(ValueError, match="auxiliary_heuristics"):
        _entry(
            "candidate",
            auxiliary_heuristics=auxiliary_heuristics,  # type: ignore[arg-type]
        )


def test_empty_queue_behavior_is_stable() -> None:
    queue = StableSearchQueueV2()

    assert len(queue) == 0
    assert queue.peek_anchor() is None
    assert queue.suggest("resource") is None
    queue.discard("missing")
    assert len(queue) == 0
    with pytest.raises(IndexError, match="stable search queue is empty"):
        queue.pop_anchor()
    assert not hasattr(queue, "__dict__")


def test_anchor_order_uses_every_authoritative_tie_break_field() -> None:
    entries = (
        _entry("later-f", path_cost=1.0, anchor_heuristic=2.0),
        _entry("higher-g", path_cost=2.0, anchor_heuristic=0.0),
        _entry("later-state", state_key=(1,), path_cost=1.0, anchor_heuristic=1.0),
        _entry(
            "later-primitive",
            state_key=(0,),
            primitive_key="z",
            path_cost=1.0,
            anchor_heuristic=1.0,
        ),
        _entry(
            "z-candidate",
            state_key=(0,),
            primitive_key="a",
            path_cost=1.0,
            anchor_heuristic=1.0,
        ),
        _entry(
            "a-candidate",
            state_key=(0,),
            primitive_key="a",
            path_cost=1.0,
            anchor_heuristic=1.0,
        ),
    )
    queue = StableSearchQueueV2()
    queue.extend(entries)

    assert tuple(entry.candidate_id for entry in _drain(queue)) == (
        "a-candidate",
        "z-candidate",
        "later-primitive",
        "later-state",
        "higher-g",
        "later-f",
    )


def test_anchor_order_is_independent_of_input_permutation_for_exact_ties() -> None:
    entries = tuple(
        _entry(
            candidate_id,
            state_key=(0,),
            primitive_key="same",
            path_cost=1.0,
            anchor_heuristic=1.0,
        )
        for candidate_id in ("c", "a", "b")
    )

    observed = set()
    for permuted in permutations(entries):
        queue = StableSearchQueueV2()
        queue.extend(permuted)
        observed.add(tuple(entry.candidate_id for entry in _drain(queue)))

    assert observed == {("a", "b", "c")}


def test_extend_requires_an_exact_tuple_and_exact_entries() -> None:
    queue = StableSearchQueueV2()
    entry = _entry("candidate")

    for invalid in ([entry], iter((entry,)), _TupleSubclass((entry,)), (object(),)):
        with pytest.raises(ValueError, match="exact tuple"):
            queue.extend(invalid)  # type: ignore[arg-type]
        assert len(queue) == 0


def test_same_value_replay_is_idempotent() -> None:
    queue = StableSearchQueueV2()
    entry = _entry("candidate")

    queue.extend((entry, entry))
    queue.extend((entry,))

    assert len(queue) == 1
    assert queue.pop_anchor() == entry


def test_conflicting_replay_rejects_entire_batch_atomically() -> None:
    queue = StableSearchQueueV2()
    original = _entry("original", path_cost=2.0)
    queue.extend((original,))

    conflict_a = _entry("conflict", path_cost=1.0)
    conflict_b = _entry("conflict", path_cost=3.0)
    new_entry = _entry("new", path_cost=0.0)
    with pytest.raises(ValueError, match="conflicting candidate_id"):
        queue.extend((new_entry, conflict_a, conflict_b))

    assert _drain(queue) == (original,)


def test_conflict_with_existing_entry_rejects_entire_batch_atomically() -> None:
    queue = StableSearchQueueV2()
    original = _entry("original", path_cost=2.0)
    queue.extend((original,))

    with pytest.raises(ValueError, match="conflicting candidate_id"):
        queue.extend(
            (
                _entry("new", path_cost=0.0),
                _entry("original", path_cost=9.0),
            )
        )

    assert _drain(queue) == (original,)


def test_invalid_entry_rejects_entire_batch_atomically() -> None:
    queue = StableSearchQueueV2()
    original = _entry("original")
    queue.extend((original,))

    with pytest.raises(ValueError, match="exact tuple"):
        queue.extend((_entry("new"), object()))  # type: ignore[arg-type]

    assert _drain(queue) == (original,)


def test_forged_exact_entry_rejects_entire_batch_atomically() -> None:
    queue = StableSearchQueueV2()
    original = _entry("original")
    queue.extend((original,))
    forged = _entry("forged")
    object.__setattr__(forged, "path_cost", nan)

    with pytest.raises(ValueError, match="valid exact SearchQueueEntryV2"):
        queue.extend((_entry("new"), forged))

    assert _drain(queue) == (original,)


def test_queue_defensively_copies_entries_on_write_and_read() -> None:
    queue = StableSearchQueueV2()
    expected = _entry(
        "candidate",
        path_cost=1.0,
        auxiliary_heuristics=(("resource", 1.0),),
    )
    source = _entry(
        "candidate",
        path_cost=1.0,
        auxiliary_heuristics=(("resource", 1.0),),
    )
    queue.extend((source,))
    object.__setattr__(source, "path_cost", 99.0)

    peeked = queue.peek_anchor()
    assert peeked == expected
    assert peeked is not None
    object.__setattr__(peeked, "path_cost", 98.0)
    suggested = queue.suggest("resource")
    assert suggested == expected
    assert suggested is not None
    object.__setattr__(suggested, "path_cost", 97.0)
    popped = queue.pop_anchor()
    assert popped == expected
    object.__setattr__(popped, "path_cost", 96.0)

    queue.extend((expected,))
    assert len(queue) == 0


def test_auxiliary_suggestion_uses_auxiliary_score_then_authoritative_key() -> None:
    queue = StableSearchQueueV2()
    queue.extend(
        (
            _entry(
                "anchor-first",
                path_cost=0.0,
                anchor_heuristic=0.0,
                auxiliary_heuristics=(("resource", 5.0),),
            ),
            _entry(
                "aux-tie-authority-first",
                path_cost=1.0,
                anchor_heuristic=1.0,
                auxiliary_heuristics=(("resource", 0.0),),
            ),
            _entry(
                "aux-tie-authority-second",
                path_cost=0.0,
                anchor_heuristic=3.0,
                auxiliary_heuristics=(("resource", 1.0),),
            ),
            _entry("no-resource", path_cost=0.0, anchor_heuristic=0.5),
        )
    )

    suggestion = queue.suggest("resource")
    assert suggestion is not None
    assert suggestion.candidate_id == "aux-tie-authority-first"
    assert queue.suggest("missing") is None


def test_any_suggestion_sequence_is_read_only_for_authoritative_order() -> None:
    entries = tuple(
        _entry(
            candidate_id,
            path_cost=index,
            anchor_heuristic=4 - index,
            auxiliary_heuristics=(("corridor", 3 - index), ("resource", index)),
        )
        for index, candidate_id in enumerate(("d", "c", "b", "a"))
    )
    expected_queue = StableSearchQueueV2()
    expected_queue.extend(entries)
    expected = _drain(expected_queue)

    for suggestion_ids in (
        (),
        ("resource",),
        ("corridor", "resource", "corridor"),
        ("missing", "resource", "missing", "corridor"),
    ):
        queue = StableSearchQueueV2()
        queue.extend(tuple(reversed(entries)))
        for heuristic_id in suggestion_ids:
            queue.suggest(heuristic_id)
        assert _drain(queue) == expected


@pytest.mark.parametrize("heuristic_id", ("", _StringSubclass("resource")))
def test_suggest_rejects_nonexact_or_empty_heuristic_id(
    heuristic_id: object,
) -> None:
    queue = StableSearchQueueV2()
    with pytest.raises(ValueError, match="exact nonempty string"):
        queue.suggest(heuristic_id)  # type: ignore[arg-type]


def test_discard_is_deterministic_and_idempotent() -> None:
    queue = StableSearchQueueV2()
    keep = _entry("keep", path_cost=2.0)
    remove = _entry("remove", path_cost=1.0)
    queue.extend((keep, remove))

    assert queue.discard("remove") is True
    assert queue.discard("remove") is False
    assert queue.discard("missing") is False

    assert len(queue) == 1
    assert queue.peek_anchor() == keep
    assert _drain(queue) == (keep,)


@pytest.mark.parametrize("candidate_id", ("", _StringSubclass("candidate")))
def test_discard_rejects_nonexact_or_empty_candidate_id(candidate_id: object) -> None:
    queue = StableSearchQueueV2()
    with pytest.raises(ValueError, match="exact nonempty string"):
        queue.discard(candidate_id)  # type: ignore[arg-type]


def test_reverse_worker_completion_has_same_authoritative_sequence() -> None:
    batches = (
        (_entry("c", path_cost=3.0), _entry("a", path_cost=1.0)),
        (_entry("d", path_cost=4.0), _entry("b", path_cost=2.0)),
    )

    observed: list[tuple[str, ...]] = []
    for completion_order in ((0, 1), (1, 0)):
        queue = StableSearchQueueV2()
        for batch_index in completion_order:
            queue.extend(batches[batch_index])
        observed.append(tuple(entry.candidate_id for entry in _drain(queue)))

    assert observed == [("a", "b", "c", "d"), ("a", "b", "c", "d")]


def test_concurrent_extend_is_equivalent_to_single_worker() -> None:
    batches = tuple(
        tuple(
            _entry(
                f"candidate-{batch_index}-{entry_index}",
                state_key=(entry_index, batch_index),
                primitive_key=f"primitive-{entry_index % 2}",
                path_cost=(batch_index + entry_index) % 4,
                anchor_heuristic=(7 - batch_index - entry_index) % 4,
                auxiliary_heuristics=(("resource", batch_index),),
            )
            for entry_index in range(4)
        )
        for batch_index in range(4)
    )
    single_worker = StableSearchQueueV2()
    for batch in batches:
        single_worker.extend(batch)
    expected = _drain(single_worker)

    four_workers = StableSearchQueueV2()
    barrier = Barrier(4)

    def extend_after_barrier(batch: tuple[SearchQueueEntryV2, ...]) -> None:
        barrier.wait()
        four_workers.extend(tuple(reversed(batch)))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = tuple(
            executor.submit(extend_after_barrier, batch) for batch in batches
        )
        for future in reversed(futures):
            future.result()

    actual = _drain(four_workers)
    assert actual == expected
    assert _decision_projection(actual) == _decision_projection(expected)
    assert canonical_json_bytes(actual) == canonical_json_bytes(expected)
