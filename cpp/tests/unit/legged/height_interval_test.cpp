#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/legged/height_interval.hpp"
#include "lunar_path_planner/v3/legged/legged_terrain.hpp"

namespace lunar::planning::v3 {
namespace {

BodyMotionPrimitive VerticalLimitedPrimitive(double maximum_delta_m) {
  return BodyMotionPrimitive{
      .id = "vertical",
      .kind = BodyMotionKind::kForward,
      .relative_end = PoseXyzYaw{},
      .nominal_duration =
          DurationNanoseconds{std::chrono::seconds{1}},
      .nominal_envelope = {},
      .normalized_samples = {0.0, 0.5, 1.0},
      .secondary_costs = {},
      .sampled_body_sweep_ref =
          ContentRef{"sweep", 1U, std::string(64U, 'a')},
      .maximum_up_delta_m = maximum_delta_m,
      .maximum_down_delta_m = maximum_delta_m,
  };
}

LeggedTerrainEvaluation FeasibleAt(double minimum_m,
                                   double maximum_m) {
  return LeggedTerrainEvaluation{
      .hard_feasible = true,
      .body_height_interval = {minimum_m, maximum_m},
      .fitted_normal = {0.0, 0.0, 1.0},
      .plane_residual_m = 0.0,
      .terrain_scaled_duration =
          DurationNanoseconds{std::chrono::seconds{1}},
      .secondary_costs = {},
      .rejection_reasons = {},
  };
}

TEST(HeightIntervalTest, RejectsDisconnectedEdgeIntervals) {
  const HeightInterval source{0.40, 0.50};
  const std::vector samples{
      FeasibleAt(0.40, 0.50),
      FeasibleAt(0.70, 0.80),
      FeasibleAt(0.70, 0.80),
  };

  const auto result = PropagateLeggedEdgeHeightInterval(
      source, VerticalLimitedPrimitive(0.10), samples);

  EXPECT_FALSE(result.has_value());
}

TEST(HeightIntervalTest, PropagatesContinuousIntersection) {
  const HeightInterval source{0.40, 0.50};
  const std::vector samples{
      FeasibleAt(0.40, 0.50),
      FeasibleAt(0.45, 0.55),
      FeasibleAt(0.50, 0.60),
  };

  const auto result = PropagateLeggedEdgeHeightInterval(
      source, VerticalLimitedPrimitive(0.10), samples);

  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(result->min_m, 0.50);
  EXPECT_DOUBLE_EQ(result->max_m, 0.60);
}

TEST(HeightIntervalTest, RejectsNonFiniteOrReversedIntervals) {
  EXPECT_FALSE(IsValidHeightInterval(
      {std::numeric_limits<double>::quiet_NaN(), 1.0}));
  EXPECT_FALSE(IsValidHeightInterval({1.0, 0.0}));
  EXPECT_TRUE(IsValidHeightInterval({0.0, 0.0}));
}

}  // namespace
}  // namespace lunar::planning::v3
