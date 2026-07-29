#include <cmath>

#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/landing_geometry.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(LandingGeometryTest,
     CanonicalCcwIntervalCrossesPiWithoutWrapFlag) {
  const CircularYawInterval interval{
      .start_rad = 3.0,
      .span_rad = 0.4,
  };
  EXPECT_TRUE(yaw_interval_contains(interval, 3.1));
  EXPECT_TRUE(yaw_interval_contains(interval, -3.1));
  EXPECT_FALSE(yaw_interval_contains(interval, 0.0));
}

TEST(LandingGeometryTest, RejectsNonRightHandedPlaneBasis) {
  LandingPlane plane{
      .origin_m = {},
      .normal = {0.0, 0.0, 1.0},
      .basis_u = {1.0, 0.0, 0.0},
      .basis_v = {0.0, -1.0, 0.0},
      .residual_bound_m = 0.01,
  };
  EXPECT_FALSE(validate_landing_plane(plane).ok());
}

TEST(HopperConfigTest,
     RejectsMismatchedResolvedObjectIdentityAndWideningTable) {
  auto bindings = hopper_test::ValidBindings();
  bindings.gravity_model->content_ref.revision = 2U;
  EXPECT_FALSE(IsOk(bind_hopper_capability(
      hopper_test::ValidCapability(), bindings)));

  bindings = hopper_test::ValidBindings();
  auto widened = std::make_shared<AttitudeTighteningTable>(
      *bindings.attitude_tightening_table->object);
  widened->entries.front().maximum_angular_speed_radps = 2.1;
  bindings.attitude_tightening_table->object = widened;
  EXPECT_FALSE(IsOk(bind_hopper_capability(
      hopper_test::ValidCapability(), bindings)));
}

TEST(HopperConfigTest, RejectsZeroResourceCaps) {
  auto algorithm = hopper_test::ValidAlgorithm();
  algorithm.hopper.maximum_root_iterations = 0U;
  EXPECT_FALSE(IsOk(bind_hopper_limits(algorithm)));
}

}  // namespace
}  // namespace lunar::planning::v3
