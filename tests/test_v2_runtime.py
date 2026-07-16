import pytest

from path_planner.v2.runtime import PlanningDeadlineV2


@pytest.mark.parametrize(
    ("started", "deadline", "clock", "error", "message"),
    [
        (True, 2.0, lambda: 1.0, TypeError, "started_monotonic_s"),
        (float("nan"), 2.0, lambda: 1.0, ValueError, "finite"),
        (1.0, float("inf"), lambda: 1.0, ValueError, "finite"),
        (2.0, 1.0, lambda: 1.0, ValueError, "at least"),
        (1.0, 2.0, None, TypeError, "callable"),
    ],
)
def test_planning_deadline_rejects_invalid_runtime_contracts(
    started,
    deadline,
    clock,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        PlanningDeadlineV2(started, deadline, clock)


@pytest.mark.parametrize("now", [float("nan"), float("inf"), float("-inf")])
def test_planning_deadline_rejects_nonfinite_clock_results(now) -> None:
    deadline = PlanningDeadlineV2(1.0, 2.0, lambda: now)

    with pytest.raises(ValueError, match="clock result.*finite"):
        _ = deadline.expired
