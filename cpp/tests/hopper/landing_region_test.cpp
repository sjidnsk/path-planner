#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/landing_region.hpp"
#include "lunar_path_planner/v3/hopper/safe_pose_mask.hpp"
#include "lunar_path_planner/v3/hopper/swept_footprint.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(LandingRegionTest,
     UnknownCellRemainsUnsafeAndRegionAvoidsIt) {
  const auto map = hopper_test::FlatMap();
  ASSERT_NE(map, nullptr);
  const auto capability_result = bind_hopper_capability(
      hopper_test::ValidCapability(),
      hopper_test::ValidBindings());
  ASSERT_TRUE(IsOk(capability_result));
  const auto capability =
      std::get<HopperCapabilityView>(capability_result);
  const auto footprint =
      outer_approximate_rotated_footprint(
          capability.landing_footprint_body_xy,
          CircularYawInterval{
              .start_rad = 0.0, .span_rad = 0.0},
          make_uniform_unit_directions(16U), 0.0);
  ASSERT_TRUE(IsOk(footprint));
  const auto mask = build_safe_pose_mask(
      *map, capability,
      std::get<SweptFootprintEnvelope>(footprint));
  ASSERT_TRUE(IsOk(mask));

  const auto seeds = select_landing_seeds(
      std::get<SafePoseMask>(mask), 4U);
  ASSERT_FALSE(seeds.empty());
  const auto regions = LandingRegionGenerator{
      capability,
      std::get<HopperPlannerLimits>(
          bind_hopper_limits(hopper_test::ValidAlgorithm()))}
                           .generate(
                               *map,
                               std::get<SafePoseMask>(mask),
                               seeds);
  ASSERT_FALSE(regions.empty());
  for (const auto& region : regions) {
    EXPECT_TRUE(validate_terrain_certified_region(
        region, *map, std::get<SafePoseMask>(mask),
        capability).ok());
    EXPECT_GE(std::abs(polygon_signed_area(
                  region.vertices_uv_ccw)),
              capability.landing_terrain_thresholds
                  .minimum_landing_region_area_m2);
  }
}

}  // namespace
}  // namespace lunar::planning::v3
