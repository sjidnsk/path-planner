from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real


MonotonicClockV2 = Callable[[], float]


def _finite_timestamp(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class PlanningDeadlineV2:
    started_monotonic_s: float
    deadline_monotonic_s: float
    _monotonic_clock: MonotonicClockV2 = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        started = _finite_timestamp(
            self.started_monotonic_s,
            "started_monotonic_s",
        )
        deadline = _finite_timestamp(
            self.deadline_monotonic_s,
            "deadline_monotonic_s",
        )
        if deadline < started:
            raise ValueError("deadline_monotonic_s must be at least started_monotonic_s")
        if not callable(self._monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        object.__setattr__(self, "started_monotonic_s", started)
        object.__setattr__(self, "deadline_monotonic_s", deadline)

    def _now(self) -> float:
        return _finite_timestamp(self._monotonic_clock(), "monotonic clock result")

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self._now() - self.started_monotonic_s)

    @property
    def expired(self) -> bool:
        return self._now() >= self.deadline_monotonic_s

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.deadline_monotonic_s - self._now())
