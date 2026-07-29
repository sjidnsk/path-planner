#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

#include <gtest/gtest.h>

#include "fixtures/system/planner_v3_system_fixture.hpp"

namespace lunar::planning::v3 {
namespace {

using system_test::ExpectedExperimentOutcome;
using system_test::G1ReferenceScenario;
using system_test::MapScale;

struct MultiscaleScenarioCase final {
  PlatformType platform_type;
  MapScale scale;
  G1ReferenceScenario scene;
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

struct ExpectedG1Reference final {
  std::string_view scenario_id;
  std::string_view scenario_hash;
  std::string_view proxy_seed_hex;
  double hard_obstacle_fraction;
};

[[nodiscard]] constexpr ExpectedG1Reference ExpectedReference(
    const G1ReferenceScenario scene) {
  switch (scene) {
    case G1ReferenceScenario::kLowKnown:
      return {
          "validation/scenario-0027/standard-proxy/v1",
          "564835143c269d1ef4fb8d01cb7517cf4efb674bb1554c36467a1d59eb39f16c",
          "0647281ae562cc9fac85a745a0c183c2",
          0.006683349609375,
      };
    case G1ReferenceScenario::kMediumKnown:
      return {
          "validation/scenario-0007/standard-proxy/v1",
          "6f754169bb3c4837978e3e852258493f4b1c971e8a59148a9a8269fb306a95bc",
          "6b6d97f83a6183e4119cbd01f8e0e25e",
          0.01348876953125,
      };
    case G1ReferenceScenario::kHighFrontier:
      return {
          "validation/scenario-0116/standard-proxy/v1",
          "49e4254dc8e5602be67cc2e9eb68af65093ec049b463e3581edfc9b5f352b554",
          "9ded9466c337bbf7037cd239265150d5",
          0.03814697265625,
      };
  }
  return {};
}

constexpr std::array<MultiscaleScenarioCase, 27U> kCases{{
    {PlatformType::kWheeled, MapScale::kTenMeter,
     G1ReferenceScenario::kLowKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_ten_low_known"},
    {PlatformType::kWheeled, MapScale::kTenMeter,
     G1ReferenceScenario::kMediumKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_ten_medium_known"},
    {PlatformType::kWheeled, MapScale::kTenMeter,
     G1ReferenceScenario::kHighFrontier,
     12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady, "wheel_ten_high_frontier"},
    {PlatformType::kWheeled, MapScale::kHundredMeter,
     G1ReferenceScenario::kLowKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_hundred_low_known"},
    {PlatformType::kWheeled, MapScale::kHundredMeter,
     G1ReferenceScenario::kMediumKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_hundred_medium_known"},
    {PlatformType::kWheeled, MapScale::kHundredMeter,
     G1ReferenceScenario::kHighFrontier,
     120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady,
     "wheel_hundred_high_frontier"},
    {PlatformType::kWheeled, MapScale::kThousandMeter,
     G1ReferenceScenario::kLowKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_thousand_low_known"},
    {PlatformType::kWheeled, MapScale::kThousandMeter,
     G1ReferenceScenario::kMediumKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "wheel_thousand_medium_known"},
    {PlatformType::kWheeled, MapScale::kThousandMeter,
     G1ReferenceScenario::kHighFrontier,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady,
     "wheel_thousand_high_frontier"},
    {PlatformType::kLegged, MapScale::kTenMeter,
     G1ReferenceScenario::kLowKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_ten_low_known"},
    {PlatformType::kLegged, MapScale::kTenMeter,
     G1ReferenceScenario::kMediumKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_ten_medium_known"},
    {PlatformType::kLegged, MapScale::kTenMeter,
     G1ReferenceScenario::kHighFrontier,
     12.0, 12.0, 0.5, 24U, 24U, 576U, 8.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady, "legged_ten_high_frontier"},
    {PlatformType::kLegged, MapScale::kHundredMeter,
     G1ReferenceScenario::kLowKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_hundred_low_known"},
    {PlatformType::kLegged, MapScale::kHundredMeter,
     G1ReferenceScenario::kMediumKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_hundred_medium_known"},
    {PlatformType::kLegged, MapScale::kHundredMeter,
     G1ReferenceScenario::kHighFrontier,
     120.0, 80.0, 2.0, 60U, 40U, 2400U, 80.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady,
     "legged_hundred_high_frontier"},
    {PlatformType::kLegged, MapScale::kThousandMeter,
     G1ReferenceScenario::kLowKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_thousand_low_known"},
    {PlatformType::kLegged, MapScale::kThousandMeter,
     G1ReferenceScenario::kMediumKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "legged_thousand_medium_known"},
    {PlatformType::kLegged, MapScale::kThousandMeter,
     G1ReferenceScenario::kHighFrontier,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 500.0,
     ExpectedExperimentOutcome::kSafeFrontierReferenceReady,
     PlanningOutcome::kSafeFrontierReferenceReady,
     "legged_thousand_high_frontier"},
    {PlatformType::kHopper, MapScale::kTenMeter,
     G1ReferenceScenario::kLowKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 4.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_ten_low_known"},
    {PlatformType::kHopper, MapScale::kTenMeter,
     G1ReferenceScenario::kMediumKnown, 12.0, 12.0, 0.5, 24U, 24U, 576U, 4.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_ten_medium_known"},
    {PlatformType::kHopper, MapScale::kTenMeter,
     G1ReferenceScenario::kHighFrontier,
     12.0, 12.0, 0.5, 24U, 24U, 576U, 4.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_ten_high_frontier"},
    {PlatformType::kHopper, MapScale::kHundredMeter,
     G1ReferenceScenario::kLowKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_hundred_low_known"},
    {PlatformType::kHopper, MapScale::kHundredMeter,
     G1ReferenceScenario::kMediumKnown, 120.0, 80.0, 2.0, 60U, 40U, 2400U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_hundred_medium_known"},
    {PlatformType::kHopper, MapScale::kHundredMeter,
     G1ReferenceScenario::kHighFrontier,
     120.0, 80.0, 2.0, 60U, 40U, 2400U, 8.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_hundred_high_frontier"},
    {PlatformType::kHopper, MapScale::kThousandMeter,
     G1ReferenceScenario::kLowKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 10.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_thousand_low_known"},
    {PlatformType::kHopper, MapScale::kThousandMeter,
     G1ReferenceScenario::kMediumKnown,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 10.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_thousand_medium_known"},
    {PlatformType::kHopper, MapScale::kThousandMeter,
     G1ReferenceScenario::kHighFrontier,
     1200.0, 200.0, 5.0, 240U, 40U, 9600U, 10.0,
     ExpectedExperimentOutcome::kNewReferenceReady,
     PlanningOutcome::kNewReferenceReady, "hopper_thousand_high_frontier"},
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
  ASSERT_TRUE(description.g1_reference.has_value());
  EXPECT_EQ(description.g1_reference->reference, test_case.scene);
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

  const ExpectedG1Reference expected_reference =
      ExpectedReference(test_case.scene);
  EXPECT_EQ(
      description.g1_reference->source_scenario_id,
      expected_reference.scenario_id);
  EXPECT_EQ(
      description.g1_reference->source_scenario_hash,
      expected_reference.scenario_hash);
  EXPECT_EQ(
      description.g1_reference->proxy_seed_hex,
      expected_reference.proxy_seed_hex);
  EXPECT_DOUBLE_EQ(
      description.g1_reference->source_hard_obstacle_fraction,
      expected_reference.hard_obstacle_fraction);
  EXPECT_EQ(
      description.g1_reference->source_kind,
      "synthetic_terrain_obstacle_proxy/v1");
  EXPECT_FALSE(
      description.g1_reference->physical_obstacle_cells_written);
  EXPECT_EQ(
      description.g1_reference->derivation,
      "g1_validation_reference_scaled/v1");

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
  EXPECT_EQ(
      obstacle_count,
      static_cast<std::size_t>(std::llround(
          expected_reference.hard_obstacle_fraction *
          static_cast<double>(test_case.cell_count))));
  const std::size_t proxy_region_count =
      static_cast<std::size_t>(std::count_if(
          description.regions.begin(), description.regions.end(),
          [](const system_test::ScenarioRegion& region) {
            return region.kind ==
                   system_test::ScenarioRegion::Kind::
                       kSyntheticTerrainObstacleProxy;
          }));
  EXPECT_EQ(proxy_region_count, obstacle_count);
  if (test_case.scene == G1ReferenceScenario::kLowKnown ||
      test_case.scene == G1ReferenceScenario::kMediumKnown) {
    EXPECT_GT(obstacle_count, 0U);
    EXPECT_EQ(unknown_count, 0U);
    EXPECT_EQ(description.regions.size(), obstacle_count);
  } else {
    EXPECT_GT(unknown_count, 0U);
    EXPECT_EQ(description.regions.size(), obstacle_count + 1U);
    EXPECT_EQ(
        static_cast<std::size_t>(std::count_if(
            description.regions.begin(), description.regions.end(),
            [](const system_test::ScenarioRegion& region) {
              return region.kind ==
                     system_test::ScenarioRegion::Kind::kUnknown;
            })),
        1U);
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

  const Vec3 start_position = std::visit(
      [](const auto& state) {
        return state.position_m;
      },
      scenario.request.current_state);
  const PointGoal& point_goal =
      std::get<PointGoal>(scenario.request.goal.target);
  const auto cell_index = [&geometry](const Vec3 position) {
    const std::size_t x = static_cast<std::size_t>(std::floor(
        (position.x - geometry.origin_m.x) /
        geometry.resolution_m));
    const std::size_t y = static_cast<std::size_t>(std::floor(
        (position.y - geometry.origin_m.y) /
        geometry.resolution_m));
    return y * geometry.width + x;
  };
  ASSERT_LT(cell_index(start_position), map.HardObstacleMask().size());
  ASSERT_LT(
      cell_index(point_goal.position_m),
      map.HardObstacleMask().size());
  EXPECT_EQ(map.HardObstacleMask()[cell_index(start_position)], 0U);
  EXPECT_EQ(
      map.HardObstacleMask()[cell_index(point_goal.position_m)], 0U);

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

TEST(MultiscaleG1ReferenceTest,
     SharesDerivedEnvironmentAcrossAllThreePlatforms) {
  constexpr std::array<PlatformType, 3U> platforms{
      PlatformType::kWheeled,
      PlatformType::kLegged,
      PlatformType::kHopper,
  };
  constexpr std::array<MapScale, 3U> scales{
      MapScale::kTenMeter,
      MapScale::kHundredMeter,
      MapScale::kThousandMeter,
  };
  constexpr std::array<G1ReferenceScenario, 3U> references{
      G1ReferenceScenario::kLowKnown,
      G1ReferenceScenario::kMediumKnown,
      G1ReferenceScenario::kHighFrontier,
  };
  for (const MapScale scale : scales) {
    for (const G1ReferenceScenario reference : references) {
      auto baseline = system_test::MakeMultiscaleSystemScenario(
          platforms.front(), scale, reference);
      for (const PlatformType platform : platforms) {
        auto scenario = system_test::MakeMultiscaleSystemScenario(
            platform, scale, reference);
        EXPECT_TRUE(std::equal(
            scenario.request.map_snapshot->KnownMask().begin(),
            scenario.request.map_snapshot->KnownMask().end(),
            baseline.request.map_snapshot->KnownMask().begin(),
            baseline.request.map_snapshot->KnownMask().end()));
        EXPECT_TRUE(std::equal(
            scenario.request.map_snapshot->HardObstacleMask().begin(),
            scenario.request.map_snapshot->HardObstacleMask().end(),
            baseline.request.map_snapshot->HardObstacleMask().begin(),
            baseline.request.map_snapshot->HardObstacleMask().end()));
        auto repeated = system_test::MakeMultiscaleSystemScenario(
            platform, scale, reference);
        EXPECT_TRUE(std::equal(
            scenario.request.map_snapshot->HardObstacleMask().begin(),
            scenario.request.map_snapshot->HardObstacleMask().end(),
            repeated.request.map_snapshot->HardObstacleMask().begin(),
            repeated.request.map_snapshot->HardObstacleMask().end()));
      }
    }
  }
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
