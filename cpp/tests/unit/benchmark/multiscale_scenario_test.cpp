#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include "fixtures/system/planner_v3_system_fixture.hpp"

namespace lunar::planning::v3 {
namespace {

using system_test::ExpectedExperimentOutcome;
using system_test::MapScale;
using system_test::MapScenario;

struct MultiscaleScenarioCase final {
  PlatformType platform_type;
  MapScale scale;
  MapScenario scene;
  double width_m;
  double height_m;
  double resolution_m;
  std::size_t width;
  std::size_t height;
  std::size_t cell_count;
  double nominal_plan_distance_m;
  ExpectedExperimentOutcome expected_experiment_outcome;
  PlanningOutcome expected_planning_outcome;
  const char* name;
};

constexpr std::array<MultiscaleScenarioCase, 27U> kCases{{
    {PlatformType::kWheeled, MapScale::kTenMeter,
     MapScenario::kOpenKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_ten_open"},
    {PlatformType::kWheeled, MapScale::kTenMeter,
     MapScenario::kDetour, 12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_ten_detour"},
    {PlatformType::kWheeled, MapScale::kTenMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady, "wheel_ten_frontier"},
    {PlatformType::kWheeled, MapScale::kHundredMeter,
     MapScenario::kOpenKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_hundred_open"},
    {PlatformType::kWheeled, MapScale::kHundredMeter,
     MapScenario::kDetour, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_hundred_detour"},
    {PlatformType::kWheeled, MapScale::kHundredMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady,
     "wheel_hundred_frontier"},
    {PlatformType::kWheeled, MapScale::kThousandMeter,
     MapScenario::kOpenKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_thousand_open"},
    {PlatformType::kWheeled, MapScale::kThousandMeter,
     MapScenario::kDetour,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_thousand_detour"},
    {PlatformType::kWheeled, MapScale::kThousandMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady,
     "wheel_thousand_frontier"},
    {PlatformType::kLegged, MapScale::kTenMeter,
     MapScenario::kOpenKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_ten_open"},
    {PlatformType::kLegged, MapScale::kTenMeter,
     MapScenario::kDetour, 12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_ten_detour"},
    {PlatformType::kLegged, MapScale::kTenMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady, "legged_ten_frontier"},
    {PlatformType::kLegged, MapScale::kHundredMeter,
     MapScenario::kOpenKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_hundred_open"},
    {PlatformType::kLegged, MapScale::kHundredMeter,
     MapScenario::kDetour, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_hundred_detour"},
    {PlatformType::kLegged, MapScale::kHundredMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady,
     "legged_hundred_frontier"},
    {PlatformType::kLegged, MapScale::kThousandMeter,
     MapScenario::kOpenKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_thousand_open"},
    {PlatformType::kLegged, MapScale::kThousandMeter,
     MapScenario::kDetour,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_thousand_detour"},
    {PlatformType::kLegged, MapScale::kThousandMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady,
     "legged_thousand_frontier"},
    {PlatformType::kHopper, MapScale::kTenMeter,
     MapScenario::kOpenKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 4.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_ten_open"},
    {PlatformType::kHopper, MapScale::kTenMeter,
     MapScenario::kDetour, 12.0, 12.0, 0.5, 24U, 24U, 576U, 4.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_ten_detour"},
    {PlatformType::kHopper, MapScale::kTenMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     12.0, 12.0, 0.5, 24U, 24U, 576U, 4.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_ten_frontier"},
    {PlatformType::kHopper, MapScale::kHundredMeter,
     MapScenario::kOpenKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_hundred_open"},
    {PlatformType::kHopper, MapScale::kHundredMeter,
     MapScenario::kDetour, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_hundred_detour"},
    {PlatformType::kHopper, MapScale::kHundredMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     120.0, 80.0, 2.0, 60U, 40U, 2400U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_hundred_frontier"},
    {PlatformType::kHopper, MapScale::kThousandMeter,
     MapScenario::kOpenKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 10.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_thousand_open"},
    {PlatformType::kHopper, MapScale::kThousandMeter,
     MapScenario::kDetour,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 10.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_thousand_detour"},
    {PlatformType::kHopper, MapScale::kThousandMeter,
     MapScenario::kUnknownGoalWithSafeFrontier,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 10.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_thousand_frontier"},
}};

class MultiscaleScenarioTest
    : public testing::TestWithParam<MultiscaleScenarioCase> {};

TEST_P(MultiscaleScenarioTest,
       PreservesDenseMapContractAndProducesExpectedReference) {
  const MultiscaleScenarioCase test_case = GetParam();
  auto scenario = system_test::MakeMultiscaleSystemScenario(
      test_case.platform_type, test_case.scale, test_case.scene);
  const auto& description = scenario.description;
  const auto& map = *scenario.request.map_snapshot;
  const GridGeometry& geometry = map.geometry();

  EXPECT_EQ(description.scale, test_case.scale);
  EXPECT_EQ(description.scene, test_case.scene);
  EXPECT_DOUBLE_EQ(description.width_m, test_case.width_m);
  EXPECT_DOUBLE_EQ(description.height_m, test_case.height_m);
  EXPECT_DOUBLE_EQ(
      description.nominal_plan_distance_m,
      test_case.nominal_plan_distance_m);
  EXPECT_EQ(
      description.expected_outcome,
      test_case.expected_experiment_outcome);
  EXPECT_EQ(geometry.width, test_case.width);
  EXPECT_EQ(geometry.height, test_case.height);
  EXPECT_DOUBLE_EQ(geometry.resolution_m, test_case.resolution_m);
  EXPECT_DOUBLE_EQ(map.bounds().minimum_m.x, 0.0);
  EXPECT_DOUBLE_EQ(map.bounds().minimum_m.y, 0.0);
  EXPECT_DOUBLE_EQ(
      map.bounds().maximum_m.x, test_case.width_m);
  EXPECT_DOUBLE_EQ(
      map.bounds().maximum_m.y, test_case.height_m);
  EXPECT_EQ(map.KnownMask().size(), test_case.cell_count);
  EXPECT_EQ(map.ElevationMeters().size(), test_case.cell_count);
  EXPECT_EQ(map.SurfaceNormals().x.size(), test_case.cell_count);
  EXPECT_EQ(map.SurfaceNormals().y.size(), test_case.cell_count);
  EXPECT_EQ(map.SurfaceNormals().z.size(), test_case.cell_count);
  EXPECT_EQ(map.RoughnessMeters().size(), test_case.cell_count);
  EXPECT_EQ(map.HardObstacleMask().size(), test_case.cell_count);
  EXPECT_EQ(map.Confidence().size(), test_case.cell_count);

  EXPECT_EQ(
      scenario.request.algorithm_config->deterministic_execution
          .fixed_thread_count,
      1U);
  EXPECT_EQ(
      scenario.request.algorithm_config->learned_cost_policy.mode,
      LearnedCostPolicy::Mode::kDisabled);
  EXPECT_FALSE(scenario.request.learned_cost_snapshot.has_value());
  const double expected_ground_primitive_m =
      test_case.scale == MapScale::kTenMeter
          ? 1.0
          : (test_case.scale == MapScale::kHundredMeter ? 4.0 : 10.0);
  if (test_case.platform_type == PlatformType::kWheeled) {
    EXPECT_DOUBLE_EQ(
        scenario.request.algorithm_config->wheeled.state_lattice
            .xy_resolution_m,
        test_case.resolution_m);
    const auto& wheel = std::get<WheeledCapability>(
        scenario.request.safety_capability->content);
    ASSERT_FALSE(wheel.motion_primitives.empty());
    EXPECT_DOUBLE_EQ(
        wheel.motion_primitives.front().relative_end_pose.position_m.x,
        expected_ground_primitive_m);
  } else if (test_case.platform_type == PlatformType::kLegged) {
    EXPECT_DOUBLE_EQ(
        scenario.request.algorithm_config->legged.pose_lattice
            .xy_resolution_m,
        test_case.resolution_m);
    const auto& legged = std::get<LeggedCapability>(
        scenario.request.safety_capability->content);
    ASSERT_FALSE(legged.motion_primitives.empty());
    EXPECT_DOUBLE_EQ(
        legged.motion_primitives.front().body_frame_displacement_m.x,
        expected_ground_primitive_m);
  }

  const std::size_t obstacle_count = static_cast<std::size_t>(
      std::count(
          map.HardObstacleMask().begin(),
          map.HardObstacleMask().end(), std::uint8_t{1U}));
  const std::size_t unknown_count = static_cast<std::size_t>(
      std::count(
          map.KnownMask().begin(), map.KnownMask().end(),
          std::uint8_t{0U}));
  if (test_case.scene == MapScenario::kOpenKnown) {
    EXPECT_EQ(obstacle_count, 0U);
    EXPECT_EQ(unknown_count, 0U);
    EXPECT_TRUE(description.regions.empty());
  } else if (test_case.scene == MapScenario::kDetour) {
    EXPECT_GT(obstacle_count, 0U);
    EXPECT_EQ(unknown_count, 0U);
    ASSERT_FALSE(description.regions.empty());
    EXPECT_TRUE(std::all_of(
        description.regions.begin(), description.regions.end(),
        [](const system_test::ScenarioRegion& region) {
          return region.kind ==
                 system_test::ScenarioRegion::Kind::kHardObstacle;
        }));
  } else {
    EXPECT_GT(unknown_count, 0U);
    ASSERT_EQ(description.regions.size(), 1U);
    EXPECT_EQ(
        description.regions.front().kind,
        system_test::ScenarioRegion::Kind::kUnknown);
    const PointGoal& goal =
        std::get<PointGoal>(scenario.request.goal.target);
    const std::size_t goal_x =
        static_cast<std::size_t>(std::floor(
            (goal.position_m.x - geometry.origin_m.x) /
            geometry.resolution_m));
    const std::size_t goal_y =
        static_cast<std::size_t>(std::floor(
            (goal.position_m.y - geometry.origin_m.y) /
            geometry.resolution_m));
    const std::size_t goal_index = goal_y * geometry.width + goal_x;
    ASSERT_LT(goal_index, map.KnownMask().size());
    EXPECT_EQ(
        map.KnownMask()[goal_index],
        test_case.platform_type == PlatformType::kHopper ? 1U : 0U);
  }

  const ValidationReport request_report =
      SemanticValidator{}.Validate(scenario.request);
  ASSERT_TRUE(request_report.ok())
      << system_test::DescribeIssues(request_report);
  auto planner = system_test::MakePlanner(scenario);
  const PlanningResponse response = planner->Plan(scenario.request);
  EXPECT_EQ(
      response.planning_outcome, test_case.expected_planning_outcome)
      << JsonCodec::EncodePlanningResponse(response);
}

INSTANTIATE_TEST_SUITE_P(
    AllPlatformsScalesAndScenes,
    MultiscaleScenarioTest,
    testing::ValuesIn(kCases),
    [](const testing::TestParamInfo<MultiscaleScenarioCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace lunar::planning::v3
