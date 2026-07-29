#include <array>
#include <chrono>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/wheel/wheel_sweep_validator.hpp"

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] ContentRef Ref(std::string id, char digit) {
  return {std::move(id), 1U, std::string(64U, digit)};
}

[[nodiscard]] SafeProjection Projection(
    std::size_t width,
    std::size_t height,
    double resolution_m) {
  SafeProjection projection;
  projection.geometry = {
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
      .origin_m = {0.0, 0.0},
      .frame_id = "map",
  };
  const std::size_t count = projection.geometry.CellCount();
  projection.known_mask.assign(count, 1U);
  projection.hard_feasible_mask.assign(count, 1U);
  projection.esdf_clearance_m.assign(count, 10.0F);
  return projection;
}

[[nodiscard]] WheelCollisionEnvelope RectangleEnvelope() {
  return {
      .footprint =
          {
              .vertices_xy_m =
                  {{-0.75, -0.15}, {0.75, -0.15},
                   {0.75, 0.15}, {-0.75, 0.15}},
              .minimum_z_m = -0.1,
              .maximum_z_m = 0.4,
          },
      .horizontal_tracking_error_m = 0.0,
      .minimum_clearance_m = 0.0,
  };
}

[[nodiscard]] PiecewiseCubicScalarTrajectory YawLaw(
    double start,
    double finish) {
  const double delta = finish - start;
  return {
      .value_semantics = "unwrapped_yaw_rad",
      .segments =
          {
              {
                  .start_offset = DurationNanoseconds{0ns},
                  .end_offset = DurationNanoseconds{1s},
                  .coefficients =
                      {start, 0.0, 3.0 * delta,
                       -2.0 * delta},
              },
          },
  };
}

TEST(WheelSweepValidatorTest,
     RejectsCornerCollisionDuringNonCircularSpin) {
  auto projection = Projection(20U, 20U, 0.25);
  const std::size_t obstacle = 11U * 20U + 8U;
  projection.hard_feasible_mask[obstacle] = 0U;
  projection.esdf_clearance_m[obstacle] = 0.0F;
  WheelSweepValidator validator{
      projection,
      RectangleEnvelope(),
      {
          .maximum_subdivisions = 8U,
          .maximum_footprint_cells_per_sample = 128U,
          .additional_margin_m = 0.0,
      }};

  const auto result = validator.ValidateSpin(
      {2.125, 2.125, 0.0},
      YawLaw(0.0, std::numbers::pi / 2.0));

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.Contains("/segments", "SWEPT_COLLISION"));
}

TEST(WheelSweepValidatorTest, UnknownCellIsNeverClearance) {
  auto projection = Projection(8U, 6U, 1.0);
  const std::size_t unknown = 2U * 8U + 3U;
  projection.known_mask[unknown] = 0U;
  projection.hard_feasible_mask[unknown] = 0U;
  WheelCollisionEnvelope envelope{
      .footprint =
          {
              .vertices_xy_m =
                  {{-0.1, -0.1}, {0.1, -0.1},
                   {0.1, 0.1}, {-0.1, 0.1}},
              .minimum_z_m = -0.1,
              .maximum_z_m = 0.4,
          }};
  WheelSweepValidator validator{
      projection,
      envelope,
      {
          .maximum_subdivisions = 7U,
          .maximum_footprint_cells_per_sample = 64U,
          .additional_margin_m = 0.0,
      }};
  const ValidatedPrimitiveChain chain{
      .primitives =
          {
              {
                  .primitive_id = "drive-instance",
                  .capability_primitive_id = "drive",
                  .primitive_kind = PrimitiveKind::kDriveForward,
                  .start_pose = {{2.5, 2.5, 0.0}, 0.0},
                  .end_pose = {{4.5, 2.5, 0.0}, 0.0},
                  .nominal_duration = DurationNanoseconds{1s},
                  .validation_ref = Ref("sweep", 'a'),
              },
          },
  };

  const auto result = validator.ValidatePrimitiveChain(chain);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.Contains("/segments", "UNKNOWN_OR_UNSAFE_CELL"));
}

TEST(WheelSweepValidatorTest,
     InconclusiveSubdivisionLimitFailsClosed) {
  auto projection = Projection(8U, 6U, 1.0);
  WheelSweepValidator validator{
      projection,
      RectangleEnvelope(),
      {
          .maximum_subdivisions = 0U,
          .maximum_footprint_cells_per_sample = 64U,
          .additional_margin_m = 0.0,
      }};
  const ClampedCubicBSplinePath spline{
      .knots = {0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0},
      .control_points =
          {
              {{1.0, 1.0, 0.0}, 0.0},
              {{2.0, 1.0, 0.0}, 0.0},
              {{3.0, 2.0, 0.0}, 0.5},
              {{4.0, 2.0, 0.0}, 0.5},
          },
  };
  const MonotoneTimeScaling scaling{
      .segments =
          {
              {
                  .start_offset = DurationNanoseconds{0ns},
                  .end_offset = DurationNanoseconds{1s},
                  .coefficients = {0.0, 0.0, 3.0, -2.0},
              },
          },
  };

  const auto result = validator.ValidateSpline(spline, scaling);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.Contains(
      "/segments", "CONTINUOUS_VALIDATION_INCONCLUSIVE"));
}

}  // namespace
}  // namespace lunar::planning::v3
