from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from path_planner.v2.contracts import PlanningOutcomeV2, PlanningRequestV2
from path_planner.v2.profiles import PlatformProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import FineSafetyAnchorV2


@runtime_checkable
class PrimitiveProviderV2(Protocol):
    profile: PlatformProfileV2
    plan: Callable[
        [PlanningRequestV2, FineSafetyAnchorV2, PlanningDeadlineV2],
        PlanningOutcomeV2,
    ]
