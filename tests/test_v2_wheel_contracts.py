from dataclasses import FrozenInstanceError, replace
from math import isclose, pi

import pytest

import path_planner
import path_planner.v2 as v2
from path_planner.search import MotionPrimitive, Pose2D, replay_motion_primitive
from path_planner.v2.contracts import (
    PoseStateV2,
    PrimitiveKindV2,
    ValidationLevelV2,
)
from path_planner.v2.providers.wheel import WheelMotionPrimitiveV2


def _primitive(
    control: MotionPrimitive | None = None,
    *,
    start: Pose2D | None = None,
    validation_level: ValidationLevelV2 = ValidationLevelV2.L0,
) -> WheelMotionPrimitiveV2:
    start = start or Pose2D(1.25, 1.25, 0.2)
    control = control or MotionPrimitive("curve", 0.5, 0.25, 1.0)
    replay = replay_motion_primitive(start, control, 0.25)
    samples = tuple(
        PoseStateV2(sample.x_m, sample.y_m, sample.theta_rad)
        for sample in replay.samples
    )
    return WheelMotionPrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=samples[0],
        end_state=samples[-1],
        duration_s=control.duration_s,
        distance_m=replay.distance_m,
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=validation_level,
        control_name=control.name,
        samples=samples,
        v_mps=control.v_mps,
        omega_radps=control.omega_radps,
        reverse=control.reverse,
        turn_in_place=control.turn_in_place,
    )


def _hold(state: PoseStateV2 | None = None, **overrides) -> WheelMotionPrimitiveV2:
    state = state or PoseStateV2(1.25, 1.25, 4.0 * pi + 0.2)
    values = {
        "kind": PrimitiveKindV2.WHEEL_MOTION,
        "start_state": state,
        "end_state": state,
        "duration_s": 0.0,
        "distance_m": 0.0,
        "energy_cost": 0.0,
        "observation_contribution": 0.0,
        "validation_level": ValidationLevelV2.L2,
        "control_name": "hold",
        "samples": (state,),
        "v_mps": 0.0,
        "omega_radps": 0.0,
        "reverse": False,
        "turn_in_place": False,
    }
    values.update(overrides)
    return WheelMotionPrimitiveV2(**values)


def test_wheel_motion_primitive_preserves_raw_start_and_public_replay() -> None:
    start = Pose2D(1.25, 1.25, 4.0 * pi + 0.2)
    primitive = _primitive(start=start)

    assert primitive.samples[0].heading_rad == start.theta_rad
    assert primitive.start_state is primitive.samples[0]
    assert not hasattr(primitive, "__dict__")
    with pytest.raises(FrozenInstanceError):
        primitive.control_name = "other"


def test_curve_distance_is_public_replay_chord_sum_not_arc_length() -> None:
    control = MotionPrimitive("curve", 1.0, pi / 2.0, 1.0)
    primitive = _primitive(control)

    assert primitive.distance_m < abs(control.v_mps) * control.duration_s
    assert isclose(
        primitive.distance_m,
        replay_motion_primitive(Pose2D(1.25, 1.25, 0.2), control, 0.25).distance_m,
    )


def test_hold_is_the_only_single_sample_zero_motion_contract() -> None:
    hold = _hold()

    assert hold.control_name == "hold"
    assert hold.samples == (hold.start_state,)
    assert hold.end_state is hold.start_state
    assert (hold.duration_s, hold.distance_m, hold.energy_cost) == (0.0, 0.0, 0.0)
    assert hold.validation_level is ValidationLevelV2.L2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": PrimitiveKindV2.LEG_STEP}, "WHEEL_MOTION"),
        ({"control_name": " "}, "control_name"),
        ({"samples": []}, "tuple"),
        ({"start_state": PoseStateV2(9.0, 9.0, 0.0)}, "begin"),
        ({"distance_m": 99.0}, "distance"),
        ({"duration_s": 0.5}, "replay"),
        ({"reverse": True}, "reverse"),
        ({"turn_in_place": True}, "turn_in_place"),
        ({"v_mps": float("nan")}, "finite"),
    ],
)
def test_wheel_motion_primitive_rejects_replay_or_flag_mismatch(overrides, message) -> None:
    primitive = _primitive()

    with pytest.raises((TypeError, ValueError), match=message):
        replace(primitive, **overrides)


def test_wheel_motion_converts_huge_real_overflow_to_stable_finite_error() -> None:
    with pytest.raises(ValueError, match="v_mps.*finite"):
        replace(_primitive(), v_mps=10**10_000)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"control_name": "idle"}, "hold"),
        ({"duration_s": 0.1}, "zero"),
        ({"distance_m": 0.1}, "zero"),
        ({"energy_cost": 0.1}, "zero"),
        ({"validation_level": ValidationLevelV2.L1}, "L2"),
        ({"reverse": True}, "flags"),
        ({"turn_in_place": True}, "flags"),
        ({"samples": (_hold().start_state, _hold().start_state)}, "hold"),
    ],
)
def test_hold_rejects_nonzero_unvalidated_or_multi_sample_values(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _hold(**overrides)


def test_reverse_and_turn_in_place_flags_must_match_public_control() -> None:
    reverse = _primitive(MotionPrimitive("reverse", -0.5, 0.0, 1.0, reverse=True))
    turn = _primitive(
        MotionPrimitive("turn", 0.0, 0.5, 1.0, turn_in_place=True)
    )

    assert reverse.reverse is True
    assert reverse.turn_in_place is False
    assert turn.reverse is False
    assert turn.turn_in_place is True


def test_wheel_contracts_are_exported_only_through_opt_in_v2_package() -> None:
    assert v2.WheelMotionPrimitiveV2 is WheelMotionPrimitiveV2
    assert v2.WheelProfileV2.__name__ == "WheelProfileV2"
    assert callable(v2.conservative_wheel_sweep_cells)
    assert "plan_v2" not in path_planner.__dict__


def test_two_sample_max_duration_replay_keeps_finite_inferred_dt() -> None:
    duration = float.fromhex("0x1.fffffffffffffp+1023")
    min_subnormal = float.fromhex("0x0.0000000000001p-1022")
    start = Pose2D(1.25, 1.25, 0.2)
    control = MotionPrimitive("max_duration_creep", min_subnormal, 0.0, duration)
    replay = replay_motion_primitive(start, control, duration)
    samples = tuple(
        PoseStateV2(sample.x_m, sample.y_m, sample.theta_rad)
        for sample in replay.samples
    )

    primitive = WheelMotionPrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=samples[0],
        end_state=samples[-1],
        duration_s=duration,
        distance_m=replay.distance_m,
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L0,
        control_name=control.name,
        samples=samples,
        v_mps=control.v_mps,
        omega_radps=control.omega_radps,
        reverse=False,
        turn_in_place=False,
    )

    assert primitive.samples == samples
