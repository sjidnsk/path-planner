#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/aim_point_generator.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(AimPointGeneratorTest,
     ProducesStableFiniteInteriorCandidates) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const auto limits = std::get<HopperPlannerLimits>(
      bind_hopper_limits(hopper_test::ValidAlgorithm()));
  const TerrainCertifiedLandingRegion region{
      .region_id = "region-a",
      .landing_plane =
          {
              .origin_m = {1.0, 1.0, 0.0},
              .normal = {0.0, 0.0, 1.0},
              .basis_u = {1.0, 0.0, 0.0},
              .basis_v = {0.0, 1.0, 0.0},
              .residual_bound_m = 0.01,
          },
      .vertices_uv_ccw =
          ConvexPolygon2d{
              {{-1.0, -1.0}, {1.0, -1.0},
               {1.0, 1.0}, {-1.0, 1.0}}},
      .allowed_yaw_interval =
          {.start_rad = -0.5, .span_rad = 1.0},
      .inward_safety_margin_m = 0.1,
  };
  const GoalRegion goal{
      .goal_id = "goal",
      .target = PointGoal{
          .position_m = {1.9, 1.9, 0.0},
          .position_tolerance_m = 0.2},
  };
  const AimPointGenerator generator{capability, limits};
  const auto first =
      generator.generate(region, goal, Eigen::Vector3d::UnitX());
  const auto second =
      generator.generate(region, goal, Eigen::Vector3d::UnitX());
  ASSERT_FALSE(first.empty());
  ASSERT_EQ(first.size(), second.size());
  for (std::size_t index = 0U; index < first.size(); ++index) {
    EXPECT_EQ(first[index].aim_point_id,
              second[index].aim_point_id);
    EXPECT_TRUE(first[index].position_m.allFinite());
    EXPECT_TRUE(point_strictly_inside_region(
        first[index], region));
  }
}

}  // namespace
}  // namespace lunar::planning::v3
