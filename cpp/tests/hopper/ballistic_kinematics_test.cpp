#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/ballistic_kinematics.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(BallisticKinematicsTest,
     ArcHitsRequestedEndpointAndObeysConstantGravity) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const AimPointCandidate aim{
      .aim_point_id = "aim",
      .region_id = "target",
      .frame_id = "map",
      .position_uv = {4.0, -2.0},
      .position_m = {4.0, -2.0, 0.5},
  };

  const auto result = make_nominal_ballistic_arc(
      Eigen::Vector3d{0.0, 0.0, 1.0}, "map", aim,
      gravity_model_view(capability.gravity_model), 3.0);
  ASSERT_TRUE(IsOk(result));
  const auto& arc = std::get<NominalBallisticArc>(result);
  const auto start = evaluate_ballistic_state(arc, 0.0);
  const auto finish = evaluate_ballistic_state(arc, 3.0);

  EXPECT_TRUE(start.position_m.isApprox(
      Eigen::Vector3d{0.0, 0.0, 1.0}, 1.0e-12));
  EXPECT_TRUE(finish.position_m.isApprox(aim.position_m, 1.0e-12));
  EXPECT_TRUE(finish.velocity_mps.isApprox(
      arc.launch_velocity_mps + arc.gravity_mps2 * 3.0,
      1.0e-12));
}

TEST(BallisticKinematicsTest,
     RejectsFrameMismatchAndNonPositiveFlightTime) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const AimPointCandidate aim{
      .aim_point_id = "aim",
      .region_id = "target",
      .frame_id = "map",
      .position_uv = {0.0, 0.0},
      .position_m = {1.0, 0.0, 0.5},
  };
  const auto gravity = gravity_model_view(capability.gravity_model);

  EXPECT_FALSE(IsOk(make_nominal_ballistic_arc(
      Eigen::Vector3d::Zero(), "other", aim, gravity, 1.0)));
  EXPECT_FALSE(IsOk(make_nominal_ballistic_arc(
      Eigen::Vector3d::Zero(), "map", aim, gravity, 0.0)));
}

}  // namespace
}  // namespace lunar::planning::v3
