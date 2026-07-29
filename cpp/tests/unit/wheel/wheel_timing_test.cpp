#include <chrono>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/wheel/wheel_timing.hpp"

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] ContentRef Ref(std::string id, char digit) {
  return {std::move(id), 1U, std::string(64U, digit)};
}

[[nodiscard]] GeometricPath StraightPath() {
  return ValidatedPrimitiveChain{
      .primitives =
          {
              {
                  .primitive_id = "straight-instance",
                  .capability_primitive_id = "straight",
                  .primitive_kind = PrimitiveKind::kDriveForward,
                  .start_pose = {{0.0, 0.0, 0.0}, 0.0},
                  .end_pose = {{2.0, 0.0, 0.0}, 0.0},
                  .nominal_duration = DurationNanoseconds{3s},
                  .validation_ref = Ref("sweep", 'a'),
              },
          },
  };
}

[[nodiscard]] WheelTimingLimits TimingLimits() {
  return {
      .maximum_forward_speed_mps = 1.0,
      .maximum_reverse_speed_mps = 0.7,
      .maximum_linear_acceleration_mps2 = 0.8,
      .maximum_braking_deceleration_mps2 = 1.0,
      .maximum_yaw_rate_radps = 1.0,
      .maximum_lateral_acceleration_mps2 = 0.8,
      .terrain_speed_limit_mps = 1.0,
      .clearance_speed_limit_mps = 1.0,
      .traction_speed_limit_mps = 1.0,
  };
}

[[nodiscard]] WheelTimingConfig TimingConfig() {
  return {
      .maximum_adaptive_samples = 64U,
      .minimum_parameter_step = 0.1,
      .curvature_refinement_threshold_per_m = 0.2,
      .maximum_forward_passes = 1U,
      .maximum_backward_passes = 1U,
  };
}

TEST(WheelTimingTest,
     ReverseKeepsSIncreasingAndBodySpeedNegative) {
  const auto result = ParameterizeWheelDrive(
      StraightPath(), DriveDirection::kReverse,
      TimingLimits(), TimingConfig());

  ASSERT_TRUE(IsOk(result));
  const auto& timing = std::get<WheelTimingResult>(result);
  double previous = -1.0;
  for (std::size_t index = 0U; index <= 100U; ++index) {
    const auto offset = DurationNanoseconds{
        timing.duration.value *
        static_cast<std::int64_t>(index) / 100};
    const auto value =
        EvaluateMonotoneTimeScaling(timing.time_scaling, offset);
    ASSERT_TRUE(value.has_value());
    EXPECT_GE(*value + 1.0e-10, previous);
    previous = *value;
  }
  const auto midpoint = DurationNanoseconds{
      timing.duration.value / 2};
  const auto path_rate = EvaluateMonotoneTimeScalingDerivative(
      timing.time_scaling, midpoint);
  ASSERT_TRUE(path_rate.has_value());
  EXPECT_LT(SignedBodyForwardSpeed(
                *path_rate, timing.path_length_m,
                DriveDirection::kReverse),
            0.0);
}

TEST(WheelTimingTest, EndsAtZeroSpeedAtSafeStopAnchor) {
  const auto result = ParameterizeWheelDrive(
      StraightPath(), DriveDirection::kForward,
      TimingLimits(), TimingConfig());

  ASSERT_TRUE(IsOk(result));
  const auto& timing = std::get<WheelTimingResult>(result);
  const auto derivative =
      EvaluateMonotoneTimeScalingDerivative(
          timing.time_scaling, timing.duration);
  ASSERT_TRUE(derivative.has_value());
  EXPECT_NEAR(*derivative, 0.0, 1.0e-10);
}

TEST(WheelTimingTest,
     SpinUsesZeroRateAtBothModeBoundaries) {
  const auto spin = ParameterizeWheelSpin(
      0.0, 0.5,
      {
          .maximum_yaw_rate_radps = 1.0,
          .maximum_yaw_acceleration_radps2 = 2.0,
      });
  ASSERT_TRUE(IsOk(spin));
  const auto& trajectory =
      std::get<PiecewiseCubicScalarTrajectory>(spin);
  const auto start =
      trajectory.segments.front().start_offset;
  const auto finish =
      trajectory.segments.back().end_offset;
  const auto start_rate =
      EvaluateScalarTrajectoryDerivative(trajectory, start);
  const auto finish_rate =
      EvaluateScalarTrajectoryDerivative(trajectory, finish);
  const auto finish_yaw =
      EvaluateScalarTrajectory(trajectory, finish);

  ASSERT_TRUE(start_rate.has_value());
  ASSERT_TRUE(finish_rate.has_value());
  ASSERT_TRUE(finish_yaw.has_value());
  EXPECT_NEAR(*start_rate, 0.0, 1.0e-10);
  EXPECT_NEAR(*finish_rate, 0.0, 1.0e-10);
  EXPECT_NEAR(*finish_yaw, 0.5, 1.0e-10);
}

TEST(WheelTimingTest, AdaptiveSampleLimitFailsClosed) {
  auto config = TimingConfig();
  config.maximum_adaptive_samples = 2U;
  config.minimum_parameter_step = 0.1;

  const auto result = ParameterizeWheelDrive(
      StraightPath(), DriveDirection::kForward,
      TimingLimits(), config);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kResourceLimit);
}

}  // namespace
}  // namespace lunar::planning::v3
