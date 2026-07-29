#include <algorithm>

#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/ballistic_time_solver.hpp"

namespace lunar::planning::v3 {
namespace {

FlightTimeSearchInput MakeSearchInput(
    const HopperCapabilityView& capability) {
  return {
      .launch_position_m = {3.0, 3.0, 0.5},
      .frame_id = "map",
      .aim_points =
          {
              {
                  .aim_point_id = "aim",
                  .region_id = "target",
                  .frame_id = "map",
                  .position_uv = {4.0, 3.0},
                  .position_m = {4.0, 3.0, 0.5},
                  .minimum_boundary_distance_m = 1.0,
              },
          },
      .gravity =
          gravity_model_view(capability.gravity_model),
      .minimum_attitude_time_s = 0.0,
      .launch_preparation_time_s = 0.1,
      .landing_settle_time_s = 0.2,
  };
}

TEST(BallisticTimeSolverTest,
     ReturnsOnlyPointwiseVerifiedCandidatesInStableOrder) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const auto limits = std::get<HopperPlannerLimits>(
      bind_hopper_limits(hopper_test::ValidAlgorithm()));
  const auto result =
      BallisticTimeSolver{capability, limits}.solve(
          MakeSearchInput(capability));
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_TRUE(std::is_sorted(
      result.candidates.begin(), result.candidates.end(),
      ballistic_candidate_order));
  for (const auto& candidate : result.candidates) {
    EXPECT_GT(candidate.arc.flight_time_s, 0.0);
    EXPECT_LE(
        candidate.arc.launch_velocity_mps.norm(),
        capability.launch_limits.maximum_launch_speed_mps +
            1.0e-12);
    EXPECT_LE(
        candidate.arc.landing_velocity_mps.norm(),
        capability.launch_limits.maximum_landing_speed_mps +
            1.0e-12);
  }
}

TEST(BallisticTimeSolverTest,
     ImpossibleLandingSpeedReturnsNoCandidate) {
  auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const auto limits = std::get<HopperPlannerLimits>(
      bind_hopper_limits(hopper_test::ValidAlgorithm()));
  const auto input = MakeSearchInput(capability);
  capability.launch_limits.maximum_landing_speed_mps = 0.01;
  const auto result =
      BallisticTimeSolver{capability, limits}.solve(input);
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_EQ(
      result.diagnostics.termination_reason,
      "no_feasible_flight_time");
}

}  // namespace
}  // namespace lunar::planning::v3
