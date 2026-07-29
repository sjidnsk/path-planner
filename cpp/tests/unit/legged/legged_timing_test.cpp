#include <chrono>
#include <cmath>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/legged/legged_timing.hpp"

namespace lunar::planning::v3 {
namespace {

ClampedCubicBSplinePath Spline(
    const PoseXyzYaw& start, const PoseXyzYaw& finish) {
  std::vector<PoseXyzYaw> controls;
  for (std::size_t index = 0U; index < 4U; ++index) {
    const double alpha = static_cast<double>(index) / 3.0;
    controls.push_back(
        PoseXyzYaw{
            .position_m =
                {
                    (1.0 - alpha) * start.position_m.x +
                        alpha * finish.position_m.x,
                    (1.0 - alpha) * start.position_m.y +
                        alpha * finish.position_m.y,
                    (1.0 - alpha) * start.position_m.z +
                        alpha * finish.position_m.z,
                },
            .yaw_rad =
                (1.0 - alpha) * start.yaw_rad +
                alpha * finish.yaw_rad,
        });
  }
  return ClampedCubicBSplinePath{
      .knots = {0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0},
      .control_points = std::move(controls),
  };
}

BodyFrameVelocityEnvelope Envelope() {
  return BodyFrameVelocityEnvelope{
      .forward_mps = {-0.5, 1.0},
      .lateral_mps = {-0.2, 0.2},
      .vertical_mps = {-0.1, 0.1},
      .yaw_rate_radps = {-0.25, 0.25},
  };
}

LeggedTimingConfig Config() {
  return LeggedTimingConfig{
      .sampling =
          TimeScalingConfig{
              .maximum_adaptive_samples = 33U,
              .minimum_parameter_step = 1.0e-4,
              .maximum_forward_passes = 1U,
              .maximum_backward_passes = 1U,
              .enable_jerk_smoothing = false,
              .maximum_jerk_smoothing_iterations = 0U,
          },
      .maximum_linear_acceleration_mps2 = 0.5,
      .maximum_yaw_acceleration_radps2 = 0.5,
  };
}

TEST(LeggedTimingTest, LateralPathUsesLateralVelocityLimit) {
  const GeometricPath path = Spline(
      PoseXyzYaw{.position_m = {0.0, 0.0, 0.5}, .yaw_rad = 0.0},
      PoseXyzYaw{.position_m = {0.0, 1.0, 0.5}, .yaw_rad = 0.0});

  const auto result =
      ParameterizeLeggedBodyPath(path, Envelope(), Config());

  ASSERT_TRUE(IsOk(result));
  EXPECT_GE(std::get<LeggedTimingResult>(result).duration.value,
            std::chrono::seconds{5});
}

TEST(LeggedTimingTest, UsesVerticalAndYawBounds) {
  const GeometricPath path = Spline(
      PoseXyzYaw{.position_m = {0.0, 0.0, 0.5}, .yaw_rad = 0.0},
      PoseXyzYaw{.position_m = {1.0, 0.0, 0.7}, .yaw_rad = 0.5});

  const auto result =
      ParameterizeLeggedBodyPath(path, Envelope(), Config());

  ASSERT_TRUE(IsOk(result));
  const auto& timing = std::get<LeggedTimingResult>(result);
  EXPECT_LE(timing.diagnostics.maximum_body_rate_ratio,
            1.0 + 1.0e-8);
  EXPECT_LE(timing.diagnostics.maximum_yaw_rate_ratio,
            1.0 + 1.0e-8);
}

TEST(LeggedTimingTest, EndsWithZeroPathRateAtSafeStop) {
  const GeometricPath path = Spline(
      PoseXyzYaw{.position_m = {0.0, 0.0, 0.5}, .yaw_rad = 0.0},
      PoseXyzYaw{.position_m = {1.0, 0.0, 0.5}, .yaw_rad = 0.0});

  const auto result =
      ParameterizeLeggedBodyPath(path, Envelope(), Config());

  ASSERT_TRUE(IsOk(result));
  const auto& scaling =
      std::get<LeggedTimingResult>(result).time_scaling;
  ASSERT_FALSE(scaling.segments.empty());
  const auto& last = scaling.segments.back();
  const double seconds =
      std::chrono::duration<double>(
          last.end_offset.value - last.start_offset.value)
          .count();
  const double end_rate =
      last.coefficients[1] +
      2.0 * last.coefficients[2] * seconds +
      3.0 * last.coefficients[3] * seconds * seconds;
  EXPECT_NEAR(end_rate, 0.0, 1.0e-8);
  EXPECT_TRUE(
      std::get<LeggedTimingResult>(result).diagnostics.ends_stopped);
}

}  // namespace
}  // namespace lunar::planning::v3
