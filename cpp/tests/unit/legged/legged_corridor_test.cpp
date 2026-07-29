#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/legged/legged_corridor.hpp"

namespace lunar::planning::v3 {
namespace {

ContentRef Ref(std::string id, char digit) {
  return ContentRef{
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

MapSnapshotInput MapInput() {
  constexpr std::size_t kWidth = 7U;
  constexpr std::size_t kHeight = 5U;
  constexpr std::size_t kCount = kWidth * kHeight;
  return MapSnapshotInput{
      .snapshot_ref = Ref("corridor-map", 'a'),
      .map_revision = 1U,
      .immutable_data_handle = "corridor-map-data",
      .source_time = ClockStamp{"mission", std::chrono::nanoseconds{1}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {7.0, 5.0, 2.0},
          },
      .geometry =
          {
              .width = kWidth,
              .height = kHeight,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {LayerKind::kKnownMask, Ref("known", '1')},
              {LayerKind::kElevation, Ref("elevation", '2')},
              {LayerKind::kTerrainNormal, Ref("normal", '3')},
              {LayerKind::kRoughness, Ref("roughness", '4')},
              {LayerKind::kHardObstacle, Ref("obstacle", '5')},
              {LayerKind::kConfidence, Ref("confidence", '6')},
          },
      .known_mask = std::vector<std::uint8_t>(kCount, 1U),
      .elevation_m = std::vector<float>(kCount, 0.0F),
      .normal_x = std::vector<float>(kCount, 0.0F),
      .normal_y = std::vector<float>(kCount, 0.0F),
      .normal_z = std::vector<float>(kCount, 1.0F),
      .roughness_m = std::vector<float>(kCount, 0.0F),
      .hard_obstacle_mask = std::vector<std::uint8_t>(kCount, 0U),
      .confidence = std::vector<float>(kCount, 1.0F),
  };
}

BodyConvexPolytope BodyEnvelope() {
  return BodyConvexPolytope{
      .body_frame_halfspaces =
          ConvexPolytope3{
              .halfspaces =
                  {
                      {{1.0, 0.0, 0.0}, 0.2},
                      {{-1.0, 0.0, 0.0}, 0.2},
                      {{0.0, 1.0, 0.0}, 0.2},
                      {{0.0, -1.0, 0.0}, 0.2},
                      {{0.0, 0.0, 1.0}, 0.3},
                      {{0.0, 0.0, -1.0}, 0.3},
                  },
          },
  };
}

SafetyCapabilityProfile CapabilityProfile() {
  LeggedCapability capability{
      .frame_id = "map",
      .reference_point_id = "base_link",
      .collision_envelope = BodyEnvelope(),
      .motion_model_ref = Ref("motion", 'b'),
      .analytic_cost_model_ref = Ref("cost", 'c'),
      .terrain_thresholds =
          {
              .maximum_slope_rad = 0.6,
              .maximum_roughness_m = 0.2,
              .maximum_step_height_m = 0.3,
              .maximum_gap_width_m = 0.4,
              .minimum_confidence = 0.8,
              .minimum_body_clearance_m = 0.0,
              .minimum_body_height_m = 0.4,
              .maximum_body_height_m = 0.6,
          },
      .body_velocity_limits =
          {
              .forward_mps = {-0.5, 0.5},
              .lateral_mps = {-0.5, 0.5},
              .vertical_mps = {-0.1, 0.1},
              .yaw_rate_radps = {-1.0, 1.0},
              .linear_acceleration_mps2 = 0.5,
              .yaw_acceleration_radps2 = 1.0,
          },
  };
  return SafetyCapabilityProfile{
      .content_ref = Ref("capability", 'd'),
      .content = std::move(capability),
  };
}

struct CorridorScenario final {
  std::shared_ptr<const ImmutableMapSnapshot> map;
  SafetyCapabilityProfile profile;
  LeggedCapabilityView capability;
  PlannerAlgorithmConfig algorithm;
  SafeProjection projection;
  LeggedDiscretePlan plan;
};

CorridorScenario MakeScenario() {
  auto map_result = ImmutableMapSnapshot::Create(MapInput());
  EXPECT_TRUE(IsOk(map_result));
  auto map =
      std::get<std::shared_ptr<const ImmutableMapSnapshot>>(
          std::move(map_result));
  auto profile = CapabilityProfile();
  auto capability_result = LeggedCapabilityView::Create(profile);
  EXPECT_TRUE(IsOk(capability_result));
  PlannerAlgorithmConfig algorithm{};
  algorithm.content_ref = Ref("algorithm", 'e');
  algorithm.legged.corridor =
      CorridorConfig{
          .maximum_regions = 8U,
          .maximum_inflation_iterations = 256U,
          .maximum_halfplanes_per_region = 32U,
          .maximum_split_depth = 2U,
          .minimum_overlap_m = 0.05,
          .sampling_spacing_m = 0.5,
      };
  auto projection_result = BuildSafeProjection(
      SafeProjectionRequest{
          .map = map,
          .capability =
              std::make_shared<const SafetyCapabilityProfile>(profile),
          .algorithm_config =
              std::make_shared<const PlannerAlgorithmConfig>(algorithm),
          .learned_cost = nullptr,
      });
  EXPECT_TRUE(IsOk(projection_result));
  LeggedDiscretePlan plan;
  plan.stable_candidate_id = "candidate";
  plan.grid =
      GridConfig{
          .xy_resolution_m = 1.0,
          .yaw_bin_count = 4U,
          .maximum_terminal_candidates = 8U,
      };
  plan.states = {
      {1, 2, 0, {0.4, 0.6}},
      {2, 2, 0, {0.4, 0.6}},
      {3, 2, 1, {0.4, 0.6}},
  };
  return CorridorScenario{
      .map = std::move(map),
      .profile = std::move(profile),
      .capability =
          std::get<LeggedCapabilityView>(
              std::move(capability_result)),
      .algorithm = std::move(algorithm),
      .projection =
          std::get<SafeProjection>(std::move(projection_result)),
      .plan = std::move(plan),
  };
}

class RejectEveryProduct final
    : public LeggedCartesianProductCertifier {
 public:
  [[nodiscard]] bool Certify(
      const LeggedProductBox&) const noexcept override {
    return false;
  }
};

class RequireNarrowYaw final
    : public LeggedCartesianProductCertifier {
 public:
  [[nodiscard]] bool Certify(
      const LeggedProductBox& box) const noexcept override {
    return box.yaw.max_unwrapped_rad - box.yaw.min_unwrapped_rad <=
           0.8;
  }
};

TEST(LeggedCorridorTest, RejectsUnsafeCartesianProductCorner) {
  auto scenario = MakeScenario();
  scenario.algorithm.legged.corridor.maximum_split_depth = 0U;
  const RejectEveryProduct certifier;

  const auto result = BuildLeggedCorridor(
      LeggedCorridorRequest{
          .projection = scenario.projection,
          .discrete_plan = scenario.plan,
          .capability = scenario.capability,
          .config = scenario.algorithm.legged.corridor,
          .product_certifier = &certifier,
      });

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kInvalidArgument);
}

TEST(LeggedCorridorTest, SplitsYawIntervalWhenWholeProductIsUnsafe) {
  const auto scenario = MakeScenario();
  const RequireNarrowYaw certifier;

  const auto result = BuildLeggedCorridor(
      LeggedCorridorRequest{
          .projection = scenario.projection,
          .discrete_plan = scenario.plan,
          .capability = scenario.capability,
          .config = scenario.algorithm.legged.corridor,
          .product_certifier = &certifier,
      });

  ASSERT_TRUE(IsOk(result));
  const auto& corridor = std::get<LeggedCorridor>(result);
  EXPECT_GT(corridor.sections.size(), 1U);
  for (const auto& section : corridor.sections) {
    EXPECT_TRUE(section.cartesian_product_certified);
    EXPECT_TRUE(certifier.Certify(LeggedProductBox{
        .xy = section.xy,
        .z = section.z,
        .yaw = section.yaw,
    }));
  }
}

}  // namespace
}  // namespace lunar::planning::v3
