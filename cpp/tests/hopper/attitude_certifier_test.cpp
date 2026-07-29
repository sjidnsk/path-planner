#include <cmath>

#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/attitude_certifier.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(AttitudeCertifierTest,
     QuaternionSignDoesNotChangeShortestAngle) {
  const Eigen::Quaterniond identity =
      Eigen::Quaterniond::Identity();
  const Eigen::Quaterniond target{
      Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitY())};
  const Eigen::Quaterniond negated{
      -target.w(), -target.x(), -target.y(), -target.z()};
  EXPECT_NEAR(
      shortest_quaternion_angle_rad(identity, target),
      shortest_quaternion_angle_rad(identity, negated), 1.0e-12);
}

TEST(AttitudeCertifierTest,
     BangBangTimeUsesTriangularAndTrapezoidalCases) {
  EXPECT_NEAR(
      bang_bang_rotation_time_s(0.25, 1.0, 2.0),
      2.0 * std::sqrt(0.25 / 2.0), 1.0e-12);
  EXPECT_NEAR(
      bang_bang_rotation_time_s(2.0, 1.0, 2.0),
      2.5, 1.0e-12);
}

TEST(AttitudeCertifierTest,
     RequiresRotationAndSettleGuardBeforeLanding) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  AttitudeCertificationInput input{
      .initial_orientation_body_to_frame =
          Eigen::Quaterniond{
              Eigen::AngleAxisd(
                  0.7, Eigen::Vector3d::UnitY())},
      .initial_angular_velocity_radps =
          Eigen::Vector3d::Zero(),
      .initial_orientation_error_set = {0.0},
      .initial_angular_velocity_error_set_radps =
          hopper_test::ZeroVectorSet(),
      .landing_plane =
          {
              .origin_m = {0.0, 0.0, 0.0},
              .normal = {0.0, 0.0, 1.0},
              .basis_u = {1.0, 0.0, 0.0},
              .basis_v = {0.0, 1.0, 0.0},
          },
      .allowed_yaw_interval =
          {.start_rad = 0.0, .span_rad = 0.0},
      .flight_time_s = 0.05,
  };

  const auto result =
      AttitudeCertifier{
          attitude_capability_view(capability)}
          .certify(input);
  EXPECT_FALSE(result.boundary.has_value());
  EXPECT_EQ(
      result.rejection_reason,
      "insufficient_attitude_settle_time");
}

}  // namespace
}  // namespace lunar::planning::v3
