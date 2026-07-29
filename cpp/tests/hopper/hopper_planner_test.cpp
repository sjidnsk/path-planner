#include <memory>

#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/hopper_planner.hpp"

namespace lunar::planning::v3 {
namespace {

PlanningRequest MakeRequest() {
  auto capability =
      std::make_shared<SafetyCapabilityProfile>(
          hopper_test::ValidCapability());
  auto algorithm =
      std::make_shared<PlannerAlgorithmConfig>(
          hopper_test::ValidAlgorithm());
  return {
      .request_id = "hopper-request",
      .request_time =
          ClockStamp{"mission", std::chrono::nanoseconds{100}},
      .state_time =
          ClockStamp{"mission", std::chrono::nanoseconds{100}},
      .frame_id = "map",
      .platform_type = PlatformType::kHopper,
      .current_state =
          HopperState{
              .position_m = {3.0, 3.0, 0.5},
              .orientation_body_to_frame =
                  {1.0, 0.0, 0.0, 0.0},
              .linear_velocity_mps = {0.0, 0.0, 0.0},
              .angular_velocity_radps = {0.0, 0.0, 0.0},
              .error_bounds = hopper_test::ZeroHopperError(),
          },
      .goal =
          {
              .goal_id = "goal",
              .target =
                  PointGoal{
                      .position_m = {4.0, 3.0, 0.0},
                      .position_tolerance_m = 0.5,
                  },
              .optional_yaw_interval =
                  CircularYawInterval{
                      .start_rad = 0.0,
                      .span_rad = 0.0,
                  },
          },
      .map_snapshot = hopper_test::FlatMap(),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(algorithm),
      .capability_bindings = hopper_test::ValidBindings(),
  };
}

ResolvedTerminalSet MakeTerminal() {
  return {
      .kind = TerminalKind::kGoal,
      .candidates =
          {
              {
                  .stable_id = "terminal",
                  .cell = {8, 6},
                  .position_m = {4.25, 3.25, 0.0},
                  .yaw_interval =
                      CircularYawInterval{
                          .start_rad = 0.0,
                          .span_rad = 0.0,
                      },
                  .requires_zero_speed = true,
                  .clearance_m = 1.0,
              },
          },
      .reason_code = "GOAL",
  };
}

TEST(HopperPlannerTest,
     ReturnsFullyCertifiedSingleNextHop) {
  const auto result =
      HopperPlanner{}.Plan(MakeRequest(), MakeTerminal());
  ASSERT_TRUE(IsOk(result))
      << std::get<Error>(result).field_path << ": "
      << std::get<Error>(result).message;
  const auto& reference = std::get<HopperReference>(result);
  EXPECT_EQ(
      reference.translation_model,
      HopperReference::TranslationModel::
          kPureBallisticNoInflightTranslationControl);
  EXPECT_TRUE(
      validate_landing_containment(
          reference.predicted_landing_footprint,
          reference.next_landing_region,
          reference.attitude_boundary)
          .ok());
}

TEST(HopperPlannerTest,
     MissingMapReturnsFailureWithoutBoundary) {
  auto request = MakeRequest();
  request.map_snapshot.reset();
  const auto result =
      HopperPlanner{}.Plan(request, MakeTerminal());
  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(
      std::get<Error>(result).code,
      ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace lunar::planning::v3
