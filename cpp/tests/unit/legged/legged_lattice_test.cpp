#include <array>
#include <chrono>
#include <numbers>
#include <string>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/legged/legged_lattice.hpp"

namespace lunar::planning::v3 {
namespace {

BodyMotionPrimitive Primitive(BodyMotionKind kind, Vec3 displacement,
                              double yaw_change_rad) {
  return BodyMotionPrimitive{
      .id = "primitive",
      .kind = kind,
      .relative_end =
          PoseXyzYaw{
              .position_m = displacement,
              .yaw_rad = yaw_change_rad,
          },
      .nominal_duration =
          DurationNanoseconds{std::chrono::seconds{1}},
      .nominal_envelope = {},
      .normalized_samples = {0.0, 0.5, 1.0},
      .secondary_costs = {},
      .sampled_body_sweep_ref =
          ContentRef{"sweep", 1U, std::string(64U, 'a')},
      .maximum_up_delta_m = 0.0,
      .maximum_down_delta_m = 0.0,
  };
}

TEST(LeggedLatticeTest, LateralMoveKeepsBodyYaw) {
  const LeggedLatticeState source{
      .ix = 2,
      .iy = 2,
      .iyaw = 0,
      .reachable_z = {0.4, 0.6},
  };

  const auto target = ApplyLeggedPrimitiveKinematics(
      source,
      Primitive(BodyMotionKind::kLateralLeft, {0.0, 1.0, 0.0},
                0.0),
      GridConfig{.xy_resolution_m = 1.0, .yaw_bin_count = 8U});

  ASSERT_TRUE(target.has_value());
  EXPECT_EQ(target->ix, source.ix);
  EXPECT_EQ(target->iy, source.iy + 1);
  EXPECT_EQ(target->iyaw, source.iyaw);
}

TEST(LeggedLatticeTest, SpinChangesYawWithoutTranslation) {
  const LeggedLatticeState source{
      .ix = 2,
      .iy = 2,
      .iyaw = 0,
      .reachable_z = {0.4, 0.6},
  };

  const auto target = ApplyLeggedPrimitiveKinematics(
      source,
      Primitive(BodyMotionKind::kSpin, {0.0, 0.0, 0.0},
                std::numbers::pi / 2.0),
      GridConfig{.xy_resolution_m = 1.0, .yaw_bin_count = 8U});

  ASSERT_TRUE(target.has_value());
  EXPECT_EQ(target->ix, source.ix);
  EXPECT_EQ(target->iy, source.iy);
  EXPECT_EQ(target->iyaw, 2);
}

TEST(LeggedLatticeTest, IntervalDominanceRequiresTimeAndContainment) {
  const HeightInterval wide{0.4, 0.7};
  const HeightInterval narrow{0.5, 0.6};
  EXPECT_TRUE(HeightLabelDominates(
      DurationNanoseconds{std::chrono::seconds{1}}, wide,
      DurationNanoseconds{std::chrono::seconds{2}}, narrow));
  EXPECT_FALSE(HeightLabelDominates(
      DurationNanoseconds{std::chrono::seconds{1}}, narrow,
      DurationNanoseconds{std::chrono::seconds{2}}, wide));
  EXPECT_FALSE(HeightLabelDominates(
      DurationNanoseconds{std::chrono::seconds{3}}, wide,
      DurationNanoseconds{std::chrono::seconds{2}}, narrow));
}

TEST(LeggedLatticeTest,
     FasterNarrowLabelDoesNotRemoveSlowerWideExtendableLabel) {
  const HeightLabelRecord narrow{
      .stable_label_id = "fast-narrow",
      .pose_key = 17U,
      .arrival_time =
          DurationNanoseconds{std::chrono::seconds{1}},
      .reachable_z = {0.4, 0.5},
  };
  const HeightLabelRecord wide{
      .stable_label_id = "slow-wide",
      .pose_key = 17U,
      .arrival_time =
          DurationNanoseconds{std::chrono::seconds{2}},
      .reachable_z = {0.4, 0.7},
  };

  const std::array candidates{narrow, wide};
  const auto labels =
      FilterNonDominatedHeightLabels(candidates);

  ASSERT_EQ(labels.size(), 2U);
  const HeightInterval required_next{0.6, 0.7};
  EXPECT_FALSE(IntersectHeightIntervals(
                   labels[0].reachable_z, required_next)
                   .has_value());
  EXPECT_TRUE(IntersectHeightIntervals(
                  labels[1].reachable_z, required_next)
                  .has_value());
}

}  // namespace
}  // namespace lunar::planning::v3
