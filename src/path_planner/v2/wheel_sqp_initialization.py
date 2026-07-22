from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import atan2, ceil, floor, hypot, isfinite, pi

from path_planner.core import Cell, WorldPoint
from path_planner.v2.contracts import (
    ObjectiveProfileV2,
    PlanningRequestV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.profiles import WheelKinematicSQPProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import TerrainSnapshotV2, snapshot_hash
from path_planner.v2.wheel_corridors import (
    WheelBlockedComponentV1,
    WheelCorridorGraphV1,
    build_blocked_components_v1,
    topology_signature_v1,
    wheel_corridor_path_hash_v1,
)
from path_planner.v2.wheel_kinematics import (
    integrate_wheel_segment_v2,
    wheel_relative_energy_v1,
    wheel_segment_center_control_slew_v1,
)
from path_planner.v2.wheel_sqp_contracts import (
    WHEEL_KINEMATIC_INITIALIZER_V2,
    WheelCorridorV2,
    WheelSQPInitialGuessV2,
    WheelSQPModeV2,
    WheelSQPWorkLedgerV1,
    WheelSQPWorkLimitError,
    _WheelSQPInitialSegmentV1,
)


WHEEL_SQP_INITIAL_GUESS_DIGEST_V1 = "wheel_sqp_initial_guess_digest/v1"

_POINT_BYTES = 32
_COVER_CELL_BYTES = 16
_SEGMENT_BYTES = 192
_DP_RECORD_BYTES = 128
_DIGEST_BASE_ADMISSION_BYTES = 32_768
_DIGEST_CELL_ADMISSION_BYTES = 512
_DIGEST_TOPOLOGY_ENTRY_ADMISSION_BYTES = 512
_DIGEST_SEGMENT_ADMISSION_BYTES = 2_048
_NOMINAL_TRANSLATION_SPEED_MPS = 0.5

_INITIALIZATION_REASONS = frozenset(
    {
        "planning_deadline_expired",
        "wheel_sqp_resource_budget_exceeded",
        "wheel_sqp_corridor_budget_exceeded",
        "wheel_sqp_identity_mismatch",
        "wheel_sqp_numeric_contract_failed",
        "wheel_sqp_objective_unsupported",
        "wheel_sqp_initialization_failed",
    }
)


class WheelSQPInitializationError(ValueError):
    __slots__ = ("reason_code",)

    def __init__(self, reason_code: str) -> None:
        if type(reason_code) is not str:
            raise TypeError("reason_code must be exact str")
        if reason_code not in _INITIALIZATION_REASONS:
            raise ValueError("reason_code must be a wheel SQP initialization reason")
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code: str) -> None:
    raise WheelSQPInitializationError(reason_code)


def _check_deadline(ledger: WheelSQPWorkLedgerV1) -> None:
    try:
        ledger.check_deadline()
    except WheelSQPWorkLimitError as exc:
        raise WheelSQPInitializationError(exc.reason_code) from exc


def _charge_memory(ledger: WheelSQPWorkLedgerV1, amount: int) -> None:
    try:
        ledger.charge_memory(amount)
    except WheelSQPWorkLimitError as exc:
        raise WheelSQPInitializationError(exc.reason_code) from exc


def _charge_route_states(ledger: WheelSQPWorkLedgerV1, amount: int) -> None:
    try:
        ledger.charge_route_states(amount)
    except WheelSQPWorkLimitError as exc:
        raise WheelSQPInitializationError(exc.reason_code) from exc


def _validate_common(
    request: object,
    profile: object,
    ledger: object,
    deadline: object,
) -> tuple[PlanningRequestV2, WheelKinematicSQPProfileV2, WheelSQPWorkLedgerV1]:
    if type(ledger) is not WheelSQPWorkLedgerV1:
        _fail("wheel_sqp_identity_mismatch")
    if type(deadline) is not PlanningDeadlineV2:
        _fail("wheel_sqp_identity_mismatch")
    if ledger.deadline is not deadline:
        _fail("wheel_sqp_identity_mismatch")
    _check_deadline(ledger)
    if type(request) is not PlanningRequestV2:
        _fail("wheel_sqp_numeric_contract_failed")
    if type(profile) is not WheelKinematicSQPProfileV2:
        _fail("wheel_sqp_numeric_contract_failed")
    if type(request.resource_budget) is not ResourceBudgetV2:
        _fail("wheel_sqp_identity_mismatch")
    if type(ledger.resource_budget) is not ResourceBudgetV2:
        _fail("wheel_sqp_identity_mismatch")
    if ledger.resource_budget != request.resource_budget:
        _fail("wheel_sqp_identity_mismatch")
    if request.platform_profile_id != profile.profile.profile_id:
        _fail("wheel_sqp_identity_mismatch")
    if type(request.terrain_snapshot) is not TerrainSnapshotV2:
        _fail("wheel_sqp_numeric_contract_failed")
    if type(request.start_state) is not PoseStateV2 or type(request.goal_state) is not PoseStateV2:
        _fail("wheel_sqp_numeric_contract_failed")
    if type(request.objective_profile) is not ObjectiveProfileV2:
        _fail("wheel_sqp_numeric_contract_failed")
    if request.objective_profile.risk_weight != 0.0:
        _fail("wheel_sqp_objective_unsupported")
    if profile.reverse_enabled is not True or profile.turn_in_place_enabled is not True:
        _fail("wheel_sqp_identity_mismatch")
    return request, profile, ledger


def _rebuild_corridor_authorities(
    corridor: object,
    request: PlanningRequestV2,
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[WheelCorridorGraphV1, tuple[WheelBlockedComponentV1, ...]]:
    if type(corridor) is not WheelCorridorV2:
        _fail("wheel_sqp_identity_mismatch")
    try:
        graph = WheelCorridorGraphV1.from_snapshot(
            request.terrain_snapshot,
            30.0,
            ledger=ledger,
        )
        components = build_blocked_components_v1(graph, ledger=ledger)
        if any(not graph.passable(cell) for cell in corridor.cells):
            _fail("wheel_sqp_identity_mismatch")
        if any(
            right not in graph.neighbors(left)
            for left, right in zip(corridor.cells, corridor.cells[1:])
        ):
            _fail("wheel_sqp_identity_mismatch")
        expected_signature = topology_signature_v1(
            corridor.cells,
            components,
            ledger=ledger,
        )
        expected_hash = wheel_corridor_path_hash_v1(
            graph.snapshot_identity,
            corridor.cells,
            ledger=ledger,
        )
    except WheelSQPInitializationError:
        raise
    except WheelSQPWorkLimitError as exc:
        raise WheelSQPInitializationError(exc.reason_code) from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise WheelSQPInitializationError("wheel_sqp_identity_mismatch") from exc
    if expected_signature != corridor.topology_signature:
        _fail("wheel_sqp_identity_mismatch")
    if expected_hash != corridor.corridor_hash:
        _fail("wheel_sqp_identity_mismatch")
    if len(set(corridor.cells)) != len(corridor.cells):
        _fail("wheel_sqp_identity_mismatch")
    expected_guide_cost = sum(
        graph.edge_cost(left, right)
        for left, right in zip(corridor.cells, corridor.cells[1:])
    )
    expected_path_length = sum(
        graph.edge_length_m(left, right)
        for left, right in zip(corridor.cells, corridor.cells[1:])
    )
    if corridor.guide_cost != expected_guide_cost:
        _fail("wheel_sqp_identity_mismatch")
    if corridor.path_length_m != expected_path_length:
        _fail("wheel_sqp_identity_mismatch")
    geometry = request.terrain_snapshot.geometry
    try:
        start_cell = geometry.world_to_cell(
            WorldPoint(request.start_state.x_m, request.start_state.y_m)
        )
        goal_cell = geometry.world_to_cell(
            WorldPoint(request.goal_state.x_m, request.goal_state.y_m)
        )
    except (TypeError, ValueError) as exc:
        raise WheelSQPInitializationError("wheel_sqp_identity_mismatch") from exc
    if start_cell != corridor.cells[0] or goal_cell != corridor.cells[-1]:
        _fail("wheel_sqp_identity_mismatch")
    _check_deadline(ledger)
    return graph, components


def _raw_corridor_points(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
) -> tuple[tuple[float, float], ...]:
    geometry = request.terrain_snapshot.geometry
    start = (request.start_state.x_m, request.start_state.y_m)
    goal = (request.goal_state.x_m, request.goal_state.y_m)
    if len(corridor.cells) == 1:
        return (start,) if start == goal else (start, goal)
    internal = tuple(
        (point.x, point.y)
        for point in (
            geometry.cell_center(cell) for cell in corridor.cells[1:-1]
        )
    )
    return (start, *internal, goal)


def _raw_corridor_point_count(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
) -> int:
    if len(corridor.cells) != 1:
        return len(corridor.cells)
    same_position = (
        request.start_state.x_m == request.goal_state.x_m
        and request.start_state.y_m == request.goal_state.y_m
    )
    return 1 if same_position else 2


def _closed_segment_cell_entry(
    start: tuple[float, float],
    end: tuple[float, float],
    cell_x: int,
    cell_y: int,
) -> float | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    lower = 0.0
    upper = 1.0
    for coordinate, delta, minimum, maximum in (
        (start[0], dx, float(cell_x), float(cell_x + 1)),
        (start[1], dy, float(cell_y), float(cell_y + 1)),
    ):
        if delta == 0.0:
            if coordinate < minimum or coordinate > maximum:
                return None
            continue
        first = (minimum - coordinate) / delta
        second = (maximum - coordinate) / delta
        entry = min(first, second)
        exit_ = max(first, second)
        lower = max(lower, entry)
        upper = min(upper, exit_)
        if lower > upper:
            return None
    return lower


def _segment_has_grid_ambiguity(
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    if any(coordinate.is_integer() for coordinate in (*start, *end)):
        return True
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0.0 or dy == 0.0:
        return (dx == 0.0 and start[0].is_integer()) or (
            dy == 0.0 and start[1].is_integer()
        )
    first_boundary = floor(min(start[0], end[0])) + 1
    last_boundary = ceil(max(start[0], end[0])) - 1
    for boundary_x in range(first_boundary, last_boundary + 1):
        fraction = (boundary_x - start[0]) / dx
        if not 0.0 < fraction < 1.0:
            continue
        crossing_y = start[1] + fraction * dy
        if crossing_y.is_integer():
            return True
    return False


def _conservative_supercover(
    graph: WheelCorridorGraphV1,
    start: tuple[float, float],
    end: tuple[float, float],
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[tuple[Cell, ...], bool]:
    geometry = graph.snapshot.geometry
    normalized_start = (
        (start[0] - geometry.origin[0]) / geometry.resolution_m,
        (start[1] - geometry.origin[1]) / geometry.resolution_m,
    )
    normalized_end = (
        (end[0] - geometry.origin[0]) / geometry.resolution_m,
        (end[1] - geometry.origin[1]) / geometry.resolution_m,
    )
    if not all(isfinite(value) for value in (*normalized_start, *normalized_end)):
        _fail("wheel_sqp_numeric_contract_failed")
    minimum_x = floor(min(normalized_start[0], normalized_end[0]))
    maximum_x = floor(max(normalized_start[0], normalized_end[0]))
    minimum_y = floor(min(normalized_start[1], normalized_end[1]))
    maximum_y = floor(max(normalized_start[1], normalized_end[1]))
    if min(normalized_start[0], normalized_end[0]).is_integer():
        minimum_x -= 1
    if min(normalized_start[1], normalized_end[1]).is_integer():
        minimum_y -= 1
    entries: list[tuple[float, int, int, Cell]] = []
    for y in range(minimum_y, maximum_y + 1):
        _check_deadline(ledger)
        for x in range(minimum_x, maximum_x + 1):
            entry = _closed_segment_cell_entry(
                normalized_start,
                normalized_end,
                x,
                y,
            )
            if entry is None:
                continue
            cell = Cell(x, y)
            if not graph.snapshot.geometry.in_bounds(cell):
                return (), True
            _charge_memory(ledger, _COVER_CELL_BYTES)
            entries.append((entry, y, x, cell))
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    cells = tuple(item[3] for item in entries)
    return cells, _segment_has_grid_ambiguity(normalized_start, normalized_end)


def _subpath_signature_matches(
    candidate_cells: tuple[Cell, ...],
    original_cells: tuple[Cell, ...],
    components: tuple[WheelBlockedComponentV1, ...],
    ledger: WheelSQPWorkLedgerV1,
) -> bool:
    try:
        candidate_signature = topology_signature_v1(
            candidate_cells,
            components,
            ledger=ledger,
        )
        original_signature = topology_signature_v1(
            original_cells,
            components,
            ledger=ledger,
        )
    except WheelSQPWorkLimitError as exc:
        raise WheelSQPInitializationError(exc.reason_code) from exc
    except (TypeError, ValueError, OverflowError):
        return False
    return candidate_signature == original_signature


def _forced_cut_indices(
    corridor: WheelCorridorV2,
    components: tuple[WheelBlockedComponentV1, ...],
    ledger: WheelSQPWorkLedgerV1,
) -> frozenset[int]:
    forced: set[int] = {0, len(corridor.cells) - 1}
    for index, (left, right) in enumerate(zip(corridor.cells, corridor.cells[1:])):
        _check_deadline(ledger)
        try:
            signature = topology_signature_v1(
                (left, right),
                components,
                ledger=ledger,
            )
        except WheelSQPWorkLimitError as exc:
            raise WheelSQPInitializationError(exc.reason_code) from exc
        if any(count != 0 for _, count in signature.entries):
            forced.add(index)
            forced.add(index + 1)
    return frozenset(forced)


def _simplify_points(
    corridor: WheelCorridorV2,
    raw_points: tuple[tuple[float, float], ...],
    graph: WheelCorridorGraphV1,
    components: tuple[WheelBlockedComponentV1, ...],
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[tuple[float, float], ...]:
    if len(raw_points) <= 1:
        return raw_points
    if len(corridor.cells) == 1:
        forced = frozenset((0, len(raw_points) - 1))
    else:
        forced = _forced_cut_indices(corridor, components, ledger)
    kept_indices = [0]
    kept_leg_cells: list[tuple[Cell, ...]] = []
    current = 0
    while current < len(raw_points) - 1:
        _check_deadline(ledger)
        accepted: tuple[int, tuple[Cell, ...]] | None = None
        for candidate_index in range(len(raw_points) - 1, current, -1):
            if any(current < index < candidate_index for index in forced):
                continue
            if len(corridor.cells) == 1:
                original_cells = (corridor.cells[0],)
            else:
                original_cells = corridor.cells[current : candidate_index + 1]
            cover, ambiguous = _conservative_supercover(
                graph,
                raw_points[current],
                raw_points[candidate_index],
                ledger,
            )
            if candidate_index > current + 1 and ambiguous:
                continue
            if not cover or any(not graph.passable(cell) for cell in cover):
                continue
            if not _subpath_signature_matches(
                cover,
                original_cells,
                components,
                ledger,
            ):
                continue
            accepted = (candidate_index, cover)
            break
        if accepted is None:
            _check_deadline(ledger)
            _fail("wheel_sqp_initialization_failed")
        selected_index, selected_cells = accepted
        kept_indices.append(selected_index)
        kept_leg_cells.append(selected_cells)
        current = selected_index
    full_cells: list[Cell] = []
    for cells in kept_leg_cells:
        if full_cells and cells and full_cells[-1] == cells[0]:
            full_cells.extend(cells[1:])
        else:
            full_cells.extend(cells)
    if not full_cells or not _subpath_signature_matches(
        tuple(full_cells),
        corridor.cells,
        components,
        ledger,
    ):
        _fail("wheel_sqp_initialization_failed")
    return tuple(raw_points[index] for index in kept_indices)


def _split_long_legs(
    points: tuple[tuple[float, float], ...],
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[tuple[float, float], ...]:
    if len(points) <= 1:
        return points
    maximum_length = _NOMINAL_TRANSLATION_SPEED_MPS * profile.max_segment_duration_s
    result: list[tuple[float, float]] = [points[0]]
    for left, right in zip(points, points[1:]):
        _check_deadline(ledger)
        length = hypot(right[0] - left[0], right[1] - left[1])
        piece_count = max(1, ceil(length / maximum_length))
        _charge_memory(ledger, piece_count * _POINT_BYTES)
        for index in range(1, piece_count + 1):
            if index == piece_count:
                result.append(right)
                continue
            fraction = index / piece_count
            result.append(
                (
                    left[0] + (right[0] - left[0]) * fraction,
                    left[1] + (right[1] - left[1]) * fraction,
                )
            )
    _check_deadline(ledger)
    return tuple(result)


def _point_tuple(points: object) -> tuple[tuple[float, float], ...]:
    if type(points) is not tuple or not points:
        _fail("wheel_sqp_numeric_contract_failed")
    normalized: list[tuple[float, float]] = []
    for point in points:
        if type(point) is not tuple or len(point) != 2:
            _fail("wheel_sqp_numeric_contract_failed")
        x, y = point
        if type(x) is not float or type(y) is not float or not isfinite(x) or not isfinite(y):
            _fail("wheel_sqp_numeric_contract_failed")
        normalized.append((x, y))
    return tuple(normalized)


def _shortest_angle(target: float, current: float) -> float:
    raw_difference = target - current
    if -pi <= raw_difference <= pi:
        return 0.0 if raw_difference == 0.0 else raw_difference
    difference = (raw_difference + pi) % (2.0 * pi) - pi
    if difference == -pi:
        return pi
    return 0.0 if difference == 0.0 else difference


def _translation_values(
    length: float,
    mode: WheelSQPModeV2,
    profile: WheelKinematicSQPProfileV2,
) -> tuple[float, float]:
    duration = length / _NOMINAL_TRANSLATION_SPEED_MPS
    if duration < profile.min_segment_duration_s:
        duration = profile.min_segment_duration_s
    if duration > profile.max_segment_duration_s:
        _fail("wheel_sqp_initialization_failed")
    speed = length / duration
    if speed < profile.min_nonzero_control:
        _fail("wheel_sqp_initialization_failed")
    return (-speed if mode is WheelSQPModeV2.REVERSE else speed), duration


def _turn_values(
    angle: float,
    profile: WheelKinematicSQPProfileV2,
) -> tuple[tuple[float, float, float], ...]:
    if angle == 0.0:
        return ()
    count = ceil(abs(angle) / profile.max_segment_heading_change_rad)
    piece = angle / count
    duration = abs(piece) / profile.max_angular_speed_radps
    if duration < profile.min_segment_duration_s:
        duration = profile.min_segment_duration_s
    omega = piece / duration
    if abs(omega) < profile.min_nonzero_control:
        _fail("wheel_sqp_initialization_failed")
    return tuple((0.0, omega, duration) for _ in range(count))


def _control_mode(v_mps: float, omega_radps: float) -> WheelSQPModeV2:
    if v_mps > 0.0:
        return WheelSQPModeV2.FORWARD
    if v_mps < 0.0:
        return WheelSQPModeV2.REVERSE
    if omega_radps > 0.0:
        return WheelSQPModeV2.TURN_LEFT
    if omega_radps < 0.0:
        return WheelSQPModeV2.TURN_RIGHT
    return WheelSQPModeV2.STOP


def _control_cost(
    v_mps: float,
    omega_radps: float,
    duration_s: float,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
) -> float:
    objective = request.objective_profile
    return (
        objective.distance_weight * abs(v_mps) * duration_s
        + objective.energy_weight
        * wheel_relative_energy_v1(v_mps, omega_radps, duration_s, profile)
        + objective.time_weight * duration_s / profile.time_normalization_s
    )


@dataclass(frozen=True, slots=True)
class _DPRecord:
    weighted_cost: float
    reverse_count: int
    switch_count: int
    rank_sequence: tuple[int, ...]
    modes: tuple[WheelSQPModeV2, ...]
    body_heading: float

    @property
    def key(self) -> tuple[float, int, int, tuple[int, ...]]:
        return (
            self.weighted_cost,
            self.reverse_count,
            self.switch_count,
            self.rank_sequence,
        )


def _select_modes_dp(
    points: tuple[tuple[float, float], ...],
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[WheelSQPModeV2, ...]:
    legs = tuple(
        (left, right)
        for left, right in zip(points, points[1:])
        if left != right
    )
    if not legs:
        return ()
    records: tuple[_DPRecord, ...] = (
        _DPRecord(
            weighted_cost=0.0,
            reverse_count=0,
            switch_count=0,
            rank_sequence=(),
            modes=(),
            body_heading=request.start_state.heading_rad,
        ),
    )
    for left, right in legs:
        _check_deadline(ledger)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        length = hypot(dx, dy)
        line_heading = atan2(dy, dx)
        best_by_mode: dict[WheelSQPModeV2, _DPRecord] = {}
        for prior in records:
            previous_mode = prior.modes[-1] if prior.modes else None
            for rank, mode in enumerate(
                (WheelSQPModeV2.FORWARD, WheelSQPModeV2.REVERSE)
            ):
                _check_deadline(ledger)
                _charge_memory(ledger, _DP_RECORD_BYTES)
                body_heading = line_heading + (
                    pi if mode is WheelSQPModeV2.REVERSE else 0.0
                )
                turn = _shortest_angle(body_heading, prior.body_heading)
                transition_cost = sum(
                    _control_cost(v, omega, duration, request, profile)
                    for v, omega, duration in _turn_values(turn, profile)
                )
                switched = previous_mode is not None and previous_mode is not mode
                if switched:
                    transition_cost += _control_cost(
                        0.0,
                        0.0,
                        profile.min_segment_duration_s,
                        request,
                        profile,
                    )
                v_mps, duration = _translation_values(length, mode, profile)
                transition_cost += _control_cost(
                    v_mps,
                    0.0,
                    duration,
                    request,
                    profile,
                )
                candidate = _DPRecord(
                    weighted_cost=prior.weighted_cost + transition_cost,
                    reverse_count=prior.reverse_count
                    + (1 if mode is WheelSQPModeV2.REVERSE else 0),
                    switch_count=prior.switch_count + (1 if switched else 0),
                    rank_sequence=prior.rank_sequence + (rank,),
                    modes=prior.modes + (mode,),
                    body_heading=prior.body_heading + turn,
                )
                incumbent = best_by_mode.get(mode)
                if incumbent is None or candidate.key < incumbent.key:
                    best_by_mode[mode] = candidate
        records = tuple(
            best_by_mode[mode]
            for mode in (WheelSQPModeV2.FORWARD, WheelSQPModeV2.REVERSE)
        )
    terminal_records: list[tuple[tuple[float, int, int, tuple[int, ...]], _DPRecord]] = []
    for record in records:
        _check_deadline(ledger)
        final_turn = _shortest_angle(
            request.goal_state.heading_rad,
            record.body_heading,
        )
        final_cost = sum(
            _control_cost(v, omega, duration, request, profile)
            for v, omega, duration in _turn_values(final_turn, profile)
        )
        _check_deadline(ledger)
        terminal_key = (
            record.weighted_cost + final_cost,
            record.reverse_count,
            record.switch_count,
            record.rank_sequence,
        )
        terminal_records.append((terminal_key, record))
    result = min(terminal_records, key=lambda item: item[0])[1].modes
    _check_deadline(ledger)
    return result


def simplify_wheel_corridor_v2(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
    deadline: PlanningDeadlineV2,
) -> tuple[tuple[float, float], ...]:
    request, profile, ledger = _validate_common(request, profile, ledger, deadline)
    graph, components = _rebuild_corridor_authorities(corridor, request, ledger)
    raw_point_count = _raw_corridor_point_count(corridor, request)
    _charge_memory(ledger, raw_point_count * _POINT_BYTES)
    raw_points = _raw_corridor_points(corridor, request)
    if len(raw_points) != raw_point_count:
        _fail("wheel_sqp_initialization_failed")
    points = _simplify_points(
        corridor,
        raw_points,
        graph,
        components,
        ledger,
    )
    result = _split_long_legs(points, profile, ledger)
    _check_deadline(ledger)
    return result


def select_wheel_modes_v2(
    points: tuple[tuple[float, float], ...],
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
    deadline: PlanningDeadlineV2,
) -> tuple[WheelSQPModeV2, ...]:
    request, profile, ledger = _validate_common(request, profile, ledger, deadline)
    points = _point_tuple(points)
    if points[0] != (request.start_state.x_m, request.start_state.y_m):
        _fail("wheel_sqp_identity_mismatch")
    if points[-1] != (request.goal_state.x_m, request.goal_state.y_m):
        _fail("wheel_sqp_identity_mismatch")
    result = _select_modes_dp(points, request, profile, ledger)
    _check_deadline(ledger)
    return result


@dataclass(slots=True)
class _Control:
    v_mps: float
    omega_radps: float
    duration_s: float
    signed_distance_m: float
    signed_turn_angle_rad: float


def _stop_control(profile: WheelKinematicSQPProfileV2) -> _Control:
    return _Control(0.0, 0.0, profile.min_segment_duration_s, 0.0, 0.0)


def _turn_controls(
    angle: float,
    profile: WheelKinematicSQPProfileV2,
) -> tuple[_Control, ...]:
    return tuple(
        _Control(v_mps, omega, duration, 0.0, omega * duration)
        for v_mps, omega, duration in _turn_values(angle, profile)
    )


def _turn_segment_count(
    angle: float,
    profile: WheelKinematicSQPProfileV2,
) -> int:
    if angle == 0.0:
        return 0
    return ceil(abs(angle) / profile.max_segment_heading_change_rad)


def _translation_control(
    length: float,
    mode: WheelSQPModeV2,
    profile: WheelKinematicSQPProfileV2,
) -> _Control:
    v_mps, duration = _translation_values(length, mode, profile)
    return _Control(v_mps, 0.0, duration, v_mps * duration, 0.0)


def _audit_control(
    control: _Control,
    profile: WheelKinematicSQPProfileV2,
) -> None:
    if not profile.min_segment_duration_s <= control.duration_s <= profile.max_segment_duration_s:
        _fail("wheel_sqp_initialization_failed")
    mode = _control_mode(control.v_mps, control.omega_radps)
    if mode is WheelSQPModeV2.STOP:
        if (
            control.duration_s != profile.min_segment_duration_s
            or control.signed_distance_m != 0.0
            or control.signed_turn_angle_rad != 0.0
        ):
            _fail("wheel_sqp_initialization_failed")
        return
    if mode in (WheelSQPModeV2.FORWARD, WheelSQPModeV2.REVERSE):
        if (
            control.omega_radps != 0.0
            or abs(control.v_mps) < profile.min_nonzero_control
            or abs(control.v_mps) > profile.max_speed_mps
            or control.signed_distance_m != control.v_mps * control.duration_s
            or control.signed_turn_angle_rad != 0.0
        ):
            _fail("wheel_sqp_initialization_failed")
        return
    if (
        control.v_mps != 0.0
        or abs(control.omega_radps) < profile.min_nonzero_control
        or abs(control.omega_radps) > profile.max_angular_speed_radps
        or abs(control.omega_radps) * control.duration_s
        > profile.max_segment_heading_change_rad
        or control.signed_distance_m != 0.0
        or control.signed_turn_angle_rad != control.omega_radps * control.duration_s
    ):
        _fail("wheel_sqp_initialization_failed")


def _control_with_duration(control: _Control, duration_s: float) -> _Control:
    if control.signed_distance_m != 0.0:
        return _Control(
            control.signed_distance_m / duration_s,
            0.0,
            duration_s,
            control.signed_distance_m,
            0.0,
        )
    if control.signed_turn_angle_rad != 0.0:
        return _Control(
            0.0,
            control.signed_turn_angle_rad / duration_s,
            duration_s,
            0.0,
            control.signed_turn_angle_rad,
        )
    return control


def _maximum_control_duration(
    control: _Control,
    profile: WheelKinematicSQPProfileV2,
) -> float:
    if control.signed_distance_m != 0.0:
        lower_bound_limit = abs(control.signed_distance_m) / profile.min_nonzero_control
    elif control.signed_turn_angle_rad != 0.0:
        lower_bound_limit = abs(control.signed_turn_angle_rad) / profile.min_nonzero_control
    else:
        return control.duration_s
    return min(profile.max_segment_duration_s, lower_bound_limit)


def _slew_rates(
    left: _Control,
    right: _Control,
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[float, float, float]:
    _check_deadline(ledger)
    result = wheel_segment_center_control_slew_v1(
        left.v_mps,
        left.omega_radps,
        left.duration_s,
        right.v_mps,
        right.omega_radps,
        right.duration_s,
    )
    _check_deadline(ledger)
    return result


def _slew_passes(
    left: _Control,
    right: _Control,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> bool:
    speed_up, slow_down, angular = _slew_rates(left, right, ledger)
    return (
        speed_up <= profile.max_linear_accel_mps2
        and slow_down <= profile.max_linear_decel_mps2
        and angular <= profile.max_angular_accel_radps2
    )


def _slew_violation_ratio(
    left: _Control,
    right: _Control,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> float:
    speed_up, slow_down, angular = _slew_rates(left, right, ledger)
    return max(
        speed_up / profile.max_linear_accel_mps2,
        slow_down / profile.max_linear_decel_mps2,
        angular / profile.max_angular_accel_radps2,
    )


def _repair_slew_pair(
    controls: list[_Control],
    pair_index: int,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> bool:
    left = controls[pair_index]
    right = controls[pair_index + 1]
    speed_up, slow_down, angular = _slew_rates(left, right, ledger)
    candidate_indices: list[int] = []
    if speed_up > profile.max_linear_accel_mps2:
        candidate_indices.extend((pair_index + 1, pair_index))
    if slow_down > profile.max_linear_decel_mps2:
        candidate_indices.extend((pair_index, pair_index + 1))
    if angular > profile.max_angular_accel_radps2:
        if abs(right.omega_radps) > abs(left.omega_radps):
            candidate_indices.extend((pair_index + 1, pair_index))
        else:
            candidate_indices.extend((pair_index, pair_index + 1))
    ordered_indices = tuple(dict.fromkeys(candidate_indices))
    fallback: tuple[float, int, _Control] | None = None
    prior_ratio = _slew_violation_ratio(left, right, profile, ledger)
    for control_index in ordered_indices:
        control = controls[control_index]
        if _control_mode(control.v_mps, control.omega_radps) is WheelSQPModeV2.STOP:
            continue
        lower = control.duration_s
        upper = _maximum_control_duration(control, profile)
        if upper <= lower:
            continue
        upper_control = _control_with_duration(control, upper)
        candidate_left = upper_control if control_index == pair_index else left
        candidate_right = upper_control if control_index == pair_index + 1 else right
        upper_ratio = _slew_violation_ratio(
            candidate_left,
            candidate_right,
            profile,
            ledger,
        )
        if upper_ratio < prior_ratio and (
            fallback is None or (upper_ratio, control_index) < (fallback[0], fallback[1])
        ):
            fallback = (upper_ratio, control_index, upper_control)
        if not _slew_passes(candidate_left, candidate_right, profile, ledger):
            continue
        for _ in range(80):
            _check_deadline(ledger)
            midpoint = 0.5 * (lower + upper)
            midpoint_control = _control_with_duration(control, midpoint)
            candidate_left = midpoint_control if control_index == pair_index else left
            candidate_right = midpoint_control if control_index == pair_index + 1 else right
            if _slew_passes(candidate_left, candidate_right, profile, ledger):
                upper = midpoint
                upper_control = midpoint_control
            else:
                lower = midpoint
        controls[control_index] = upper_control
        _check_deadline(ledger)
        _audit_control(upper_control, profile)
        _check_deadline(ledger)
        return True
    if fallback is not None:
        controls[fallback[1]] = fallback[2]
        _check_deadline(ledger)
        _audit_control(fallback[2], profile)
        _check_deadline(ledger)
        return True
    _check_deadline(ledger)
    return False


def _repair_control_slew(
    controls: tuple[_Control, ...],
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[_Control, ...]:
    _check_deadline(ledger)
    repaired = list(controls)
    _check_deadline(ledger)
    for control in repaired:
        _check_deadline(ledger)
        _audit_control(control, profile)
    maximum_passes = profile.max_segments * 16
    for _ in range(maximum_passes):
        changed = False
        for pair_index in range(len(repaired) - 1):
            _check_deadline(ledger)
            if _slew_passes(
                repaired[pair_index],
                repaired[pair_index + 1],
                profile,
                ledger,
            ):
                continue
            if not _repair_slew_pair(repaired, pair_index, profile, ledger):
                _check_deadline(ledger)
                _fail("wheel_sqp_initialization_failed")
            changed = True
        if all(
            _slew_passes(left, right, profile, ledger)
            for left, right in zip(repaired, repaired[1:])
        ):
            result = tuple(repaired)
            _check_deadline(ledger)
            return result
        if not changed:
            break
    _check_deadline(ledger)
    _fail("wheel_sqp_initialization_failed")


def _schedule_segment_count(
    points: tuple[tuple[float, float], ...],
    modes: tuple[WheelSQPModeV2, ...],
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> int:
    count = 0
    mode_index = 0
    current_heading = request.start_state.heading_rad
    previous_mode: WheelSQPModeV2 | None = None
    for left, right in zip(points, points[1:]):
        _check_deadline(ledger)
        if left == right:
            continue
        if mode_index >= len(modes):
            _fail("wheel_sqp_numeric_contract_failed")
        mode = modes[mode_index]
        if mode not in (WheelSQPModeV2.FORWARD, WheelSQPModeV2.REVERSE):
            _fail("wheel_sqp_numeric_contract_failed")
        if previous_mode is not None and previous_mode is not mode:
            count += 1
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        line_heading = atan2(dy, dx)
        body_heading = line_heading + (pi if mode is WheelSQPModeV2.REVERSE else 0.0)
        turn = _shortest_angle(body_heading, current_heading)
        count += _turn_segment_count(turn, profile) + 1
        current_heading += turn
        previous_mode = mode
        mode_index += 1
    if mode_index != len(modes):
        _fail("wheel_sqp_numeric_contract_failed")
    final_turn = _shortest_angle(request.goal_state.heading_rad, current_heading)
    count += _turn_segment_count(final_turn, profile)
    if count == 0:
        count = 1
    _check_deadline(ledger)
    return count


def _schedule(
    points: tuple[tuple[float, float], ...],
    modes: tuple[WheelSQPModeV2, ...],
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[_Control, ...]:
    segment_count = _schedule_segment_count(
        points,
        modes,
        request,
        profile,
        ledger,
    )
    _charge_memory(ledger, segment_count * _SEGMENT_BYTES)
    _check_deadline(ledger)
    if segment_count > profile.max_segments:
        _fail("wheel_sqp_initialization_failed")
    controls: list[_Control] = []
    mode_index = 0
    current_heading = request.start_state.heading_rad
    previous_mode: WheelSQPModeV2 | None = None
    for left, right in zip(points, points[1:]):
        _check_deadline(ledger)
        if left == right:
            continue
        mode = modes[mode_index]
        mode_index += 1
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        length = hypot(dx, dy)
        line_heading = atan2(dy, dx)
        body_heading = line_heading + (pi if mode is WheelSQPModeV2.REVERSE else 0.0)
        if previous_mode is not None and previous_mode is not mode:
            controls.append(_stop_control(profile))
        turn = _shortest_angle(body_heading, current_heading)
        controls.extend(_turn_controls(turn, profile))
        current_heading += turn
        controls.append(_translation_control(length, mode, profile))
        previous_mode = mode
    final_turn = _shortest_angle(request.goal_state.heading_rad, current_heading)
    _check_deadline(ledger)
    controls.extend(_turn_controls(final_turn, profile))
    if not controls:
        controls.append(_stop_control(profile))
    _check_deadline(ledger)
    if len(controls) != segment_count:
        _fail("wheel_sqp_initialization_failed")
    result = _repair_control_slew(tuple(controls), profile, ledger)
    _check_deadline(ledger)
    return result


def _replay(
    controls: tuple[_Control, ...],
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
) -> tuple[_WheelSQPInitialSegmentV1, ...]:
    current = request.start_state
    segments: list[_WheelSQPInitialSegmentV1] = []
    for control in controls:
        _check_deadline(ledger)
        end = integrate_wheel_segment_v2(
            current,
            control.v_mps,
            control.omega_radps,
            control.duration_s,
        )
        segments.append(
            _WheelSQPInitialSegmentV1(
                start_state=current,
                end_state=end,
                v_mps=control.v_mps,
                omega_radps=control.omega_radps,
                duration_s=control.duration_s,
                mode=_control_mode(control.v_mps, control.omega_radps),
                distance_m=abs(control.v_mps) * control.duration_s,
                relative_energy=wheel_relative_energy_v1(
                    control.v_mps,
                    control.omega_radps,
                    control.duration_s,
                    profile,
                ),
            )
        )
        current = end
    result = tuple(segments)
    _check_deadline(ledger)
    return result


def _integer_admission_width(value: int) -> int:
    return max(1, value.bit_length()) + (1 if value < 0 else 0)


def _string_admission_width(value: str) -> int:
    # A JSON string uses at most six UTF-8 bytes per Python code point
    # (the longest form is a control-character ``\uXXXX`` escape).
    return 2 + 6 * len(value)


def _initial_guess_digest_admission_bytes(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    segments: tuple[_WheelSQPInitialSegmentV1, ...],
) -> int:
    amount = _DIGEST_BASE_ADMISSION_BYTES
    amount += _string_admission_width(corridor.corridor_hash)
    amount += _string_admission_width(corridor.source_id)
    amount += _string_admission_width(request.request_id)
    amount += _string_admission_width(request.platform_profile_id)
    amount += _string_admission_width(request.accelerator_policy.value)
    amount += _string_admission_width(profile.profile.profile_id)
    amount += _string_admission_width(profile.profile.capability_revision)
    amount += _string_admission_width(profile.profile.schema_version)
    amount += _string_admission_width(profile.steering_model)
    amount += _string_admission_width(profile.relative_energy_proxy_id)
    amount += _integer_admission_width(request.determinism_seed)
    amount += _integer_admission_width(request.resource_budget.max_expanded_states)
    amount += _integer_admission_width(request.resource_budget.max_route_states)
    amount += _integer_admission_width(request.resource_budget.max_memory_bytes)
    for cell in corridor.cells:
        amount += _DIGEST_CELL_ADMISSION_BYTES
        amount += _integer_admission_width(cell.x)
        amount += _integer_admission_width(cell.y)
    for component_hash, crossing_count in corridor.topology_signature.entries:
        amount += _DIGEST_TOPOLOGY_ENTRY_ADMISSION_BYTES
        amount += _string_admission_width(component_hash)
        amount += _integer_admission_width(crossing_count)
    amount += len(segments) * _DIGEST_SEGMENT_ADMISSION_BYTES
    return amount


def _initial_guess_payload(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    segments: tuple[_WheelSQPInitialSegmentV1, ...],
    terrain_identity: str,
) -> dict[str, object]:
    return {
        "domain": WHEEL_SQP_INITIAL_GUESS_DIGEST_V1,
        "initializer_id": WHEEL_KINEMATIC_INITIALIZER_V2,
        "corridor": {
            "cells": corridor.cells,
            "corridor_hash": corridor.corridor_hash,
            "source_id": corridor.source_id,
            "topology_signature": corridor.topology_signature,
        },
        "request": {
            "accelerator_policy": request.accelerator_policy,
            "determinism_seed": request.determinism_seed,
            "goal_state": request.goal_state,
            "objective_profile": request.objective_profile,
            "platform_profile_id": request.platform_profile_id,
            "request_id": request.request_id,
            "resource_budget": request.resource_budget,
            "start_state": request.start_state,
            "terrain_snapshot_hash": terrain_identity,
            "timeout_s": request.timeout_s,
        },
        "profile": profile,
        "segments": segments,
    }


def wheel_sqp_initial_guess_hash_v1(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    segments: tuple[_WheelSQPInitialSegmentV1, ...],
    *,
    ledger: WheelSQPWorkLedgerV1 | None = None,
) -> str:
    if type(corridor) is not WheelCorridorV2:
        raise TypeError("corridor must be exact WheelCorridorV2")
    if type(request) is not PlanningRequestV2:
        raise TypeError("request must be exact PlanningRequestV2")
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
    if type(segments) is not tuple or not segments:
        raise TypeError("segments must be a nonempty exact tuple")
    if any(type(segment) is not _WheelSQPInitialSegmentV1 for segment in segments):
        raise TypeError("segments must contain exact internal initial segments")
    if ledger is not None and type(ledger) is not WheelSQPWorkLedgerV1:
        raise TypeError("ledger must be exact WheelSQPWorkLedgerV1 or None")
    admission_bytes: int | None = None
    if ledger is not None:
        _check_deadline(ledger)
        admission_bytes = _initial_guess_digest_admission_bytes(
            corridor,
            request,
            profile,
            segments,
        )
        _charge_memory(ledger, admission_bytes)
    terrain_identity = snapshot_hash(request.terrain_snapshot)
    if ledger is not None:
        _check_deadline(ledger)
    payload = _initial_guess_payload(
        corridor,
        request,
        profile,
        segments,
        terrain_identity,
    )
    if ledger is not None:
        _check_deadline(ledger)
    encoded = canonical_json_bytes(payload)
    if ledger is not None:
        _check_deadline(ledger)
        assert admission_bytes is not None
        if len(encoded) > admission_bytes:
            _fail("wheel_sqp_initialization_failed")
    digest = sha256(encoded).hexdigest()
    if ledger is not None:
        _check_deadline(ledger)
    return digest


def _initial_guess_digest(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    segments: tuple[_WheelSQPInitialSegmentV1, ...],
    ledger: WheelSQPWorkLedgerV1,
) -> str:
    return wheel_sqp_initial_guess_hash_v1(
        corridor,
        request,
        profile,
        segments,
        ledger=ledger,
    )


def initialize_wheel_trajectory_v2(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    ledger: WheelSQPWorkLedgerV1,
    deadline: PlanningDeadlineV2,
) -> WheelSQPInitialGuessV2:
    request, profile, ledger = _validate_common(request, profile, ledger, deadline)
    points = simplify_wheel_corridor_v2(corridor, request, profile, ledger, deadline)
    modes = select_wheel_modes_v2(points, request, profile, ledger, deadline)
    controls = _schedule(points, modes, request, profile, ledger)
    _charge_route_states(ledger, len(controls))
    segments = _replay(controls, request, profile, ledger)
    digest = _initial_guess_digest(corridor, request, profile, segments, ledger)
    result = WheelSQPInitialGuessV2(
        corridor_hash=corridor.corridor_hash,
        start_state=request.start_state,
        requested_goal=request.goal_state,
        segments=segments,
        actual_endpoint=segments[-1].end_state,
        initial_guess_hash=digest,
    )
    _check_deadline(ledger)
    return result
