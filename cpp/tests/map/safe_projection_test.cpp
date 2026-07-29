#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/map/safe_projection.hpp"

namespace lpp = lunar::planning::v3;

namespace {

constexpr std::size_t kDefaultWidth = 7U;
constexpr std::size_t kDefaultHeight = 7U;

lpp::ContentRef Ref(std::string id, std::uint32_t revision,
                    char digest_digit) {
  return {
      .id = std::move(id),
      .revision = revision,
      .content_hash = std::string(64U, digest_digit),
  };
}

lpp::MapSnapshotInput MakeMapInput(
    std::size_t width = kDefaultWidth,
    std::size_t height = kDefaultHeight) {
  const std::size_t count = width * height;
  return {
      .snapshot_ref = Ref("map-snapshot", 8U, 'a'),
      .map_revision = 8U,
      .immutable_data_handle = "safe-projection-map",
      .source_time =
          {.clock_id = "mission", .tick = std::chrono::nanoseconds{20}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {static_cast<double>(width),
                            static_cast<double>(height), 1.0},
          },
      .geometry =
          {
              .width = width,
              .height = height,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask, Ref("known", 1U, '1')},
              {lpp::LayerKind::kElevation,
               Ref("elevation", 1U, '2')},
              {lpp::LayerKind::kTerrainNormal,
               Ref("normal", 1U, '3')},
              {lpp::LayerKind::kRoughness,
               Ref("roughness", 1U, '4')},
              {lpp::LayerKind::kHardObstacle,
               Ref("obstacle", 1U, '5')},
              {lpp::LayerKind::kConfidence,
               Ref("confidence", 1U, '6')},
          },
      .known_mask = std::vector<std::uint8_t>(count, 1U),
      .elevation_m = std::vector<float>(count, 0.0F),
      .normal_x = std::vector<float>(count, 0.0F),
      .normal_y = std::vector<float>(count, 0.0F),
      .normal_z = std::vector<float>(count, 1.0F),
      .roughness_m = std::vector<float>(count, 0.0F),
      .hard_obstacle_mask = std::vector<std::uint8_t>(count, 0U),
      .confidence = std::vector<float>(count, 1.0F),
  };
}

std::shared_ptr<const lpp::ImmutableMapSnapshot> MakeMap(
    const lpp::MapSnapshotInput& input) {
  auto result = lpp::ImmutableMapSnapshot::Create(input);
  EXPECT_TRUE(lpp::IsOk(result));
  if (!lpp::IsOk(result)) {
    return nullptr;
  }
  return std::get<std::shared_ptr<const lpp::ImmutableMapSnapshot>>(
      std::move(result));
}

lpp::BodyConvexPolytope BodyEnvelope() {
  return {
      .body_frame_halfspaces =
          {
              .halfspaces =
                  {
                      {.normal = {1.0, 0.0, 0.0},
                       .offset_m = 0.5},
                      {.normal = {-1.0, 0.0, 0.0},
                       .offset_m = 0.5},
                      {.normal = {0.0, 1.0, 0.0},
                       .offset_m = 0.5},
                      {.normal = {0.0, -1.0, 0.0},
                       .offset_m = 0.5},
                      {.normal = {0.0, 0.0, 1.0},
                       .offset_m = 0.5},
                      {.normal = {0.0, 0.0, -1.0},
                       .offset_m = 0.5},
                  },
          },
  };
}

lpp::SafetyCapabilityProfile WheelProfile(
    double maximum_slope_rad = 0.6,
    double minimum_clearance_m = 0.0) {
  lpp::WheeledCapability wheel{
      .frame_id = "map",
      .collision_envelope =
          {
              .vertices_xy_m =
                  {{-0.25, -0.2}, {0.25, -0.2},
                   {0.25, 0.2}, {-0.25, 0.2}},
              .minimum_z_m = -0.1,
              .maximum_z_m = 0.5,
          },
      .motion_model_ref = Ref("wheel-motion", 1U, 'b'),
      .analytic_cost_model_ref = Ref("wheel-cost", 1U, 'c'),
      .hard_limits =
          {
              .maximum_forward_speed_mps = 2.0,
              .maximum_reverse_speed_mps = 1.0,
              .maximum_spin_rate_radps = 1.5,
              .maximum_forward_acceleration_mps2 = 1.0,
              .maximum_braking_deceleration_mps2 = 1.0,
              .maximum_yaw_acceleration_radps2 = 2.0,
              .maximum_lateral_acceleration_mps2 = 1.0,
              .maximum_drive_curvature_per_m = 2.0,
              .maximum_slope_rad = maximum_slope_rad,
              .minimum_clearance_m = minimum_clearance_m,
          },
  };
  return {
      .content_ref = Ref("wheel-capability", 3U, 'd'),
      .content = std::move(wheel),
  };
}

lpp::SafetyCapabilityProfile LeggedProfile(
    double maximum_slope_rad = 0.6,
    double maximum_roughness_m = 0.2,
    double minimum_body_clearance_m = 0.0) {
  lpp::LeggedCapability legged{
      .frame_id = "map",
      .reference_point_id = "body-reference",
      .collision_envelope = BodyEnvelope(),
      .motion_model_ref = Ref("legged-motion", 1U, 'b'),
      .analytic_cost_model_ref = Ref("legged-cost", 1U, 'c'),
      .terrain_thresholds =
          {
              .maximum_slope_rad = maximum_slope_rad,
              .maximum_roughness_m = maximum_roughness_m,
              .maximum_step_height_m = 0.5,
              .maximum_gap_width_m = 0.5,
              .minimum_confidence = 0.5,
              .minimum_body_clearance_m =
                  minimum_body_clearance_m,
              .minimum_body_height_m = 0.2,
              .maximum_body_height_m = 1.0,
          },
      .body_velocity_limits =
          {
              .forward_mps = {-1.0, 1.5},
              .lateral_mps = {-0.75, 0.75},
              .vertical_mps = {-0.25, 0.25},
              .yaw_rate_radps = {-1.0, 1.0},
              .linear_acceleration_mps2 = 1.0,
              .yaw_acceleration_radps2 = 2.0,
          },
  };
  return {
      .content_ref = Ref("legged-capability", 4U, 'e'),
      .content = std::move(legged),
  };
}

lpp::SafetyCapabilityProfile HopperProfile(
    double maximum_slope_rad = 0.6,
    double maximum_roughness_m = 0.2,
    double minimum_clearance_m = 0.0) {
  lpp::HopperCapability hopper{
      .frame_id = "map",
      .collision_envelope = BodyEnvelope(),
      .motion_model_ref = Ref("hopper-motion", 1U, 'b'),
      .analytic_cost_model_ref = Ref("hopper-cost", 1U, 'c'),
      .gravity_model_ref = Ref("gravity", 1U, 'd'),
      .landing_terrain_thresholds =
          {
              .maximum_slope_rad = maximum_slope_rad,
              .maximum_roughness_m = maximum_roughness_m,
              .maximum_plane_residual_m = 0.3,
              .minimum_overhead_clearance_m =
                  minimum_clearance_m,
              .minimum_lateral_clearance_m =
                  minimum_clearance_m,
              .minimum_landing_region_area_m2 = 0.25,
          },
      .launch_limits =
          {
              .maximum_launch_speed_mps = 3.0,
              .maximum_launch_impulse_newton_seconds = 10.0,
              .minimum_flight_time =
                  {.value = std::chrono::milliseconds{100}},
              .maximum_flight_time =
                  {.value = std::chrono::seconds{2}},
              .maximum_landing_speed_mps = 2.0,
              .minimum_downward_impact_speed_mps = 0.0,
              .minimum_landing_clearance_m =
                  minimum_clearance_m,
          },
      .attitude_envelope =
          {
              .maximum_angular_speed_radps = 1.0,
              .maximum_angular_acceleration_radps2 = 2.0,
              .maximum_initial_angular_speed_radps = 0.5,
          },
  };
  return {
      .content_ref = Ref("hopper-capability", 5U, 'f'),
      .content = std::move(hopper),
  };
}

std::shared_ptr<const lpp::SafetyCapabilityProfile> SharedProfile(
    lpp::SafetyCapabilityProfile profile) {
  return std::make_shared<const lpp::SafetyCapabilityProfile>(
      std::move(profile));
}

std::shared_ptr<const lpp::PlannerAlgorithmConfig> MakeConfig(
    bool enable_learned = false) {
  lpp::PlannerAlgorithmConfig config{};
  config.learned_cost_policy.mode =
      enable_learned
          ? lpp::LearnedCostPolicy::Mode::kOptionalBoundedSoftCost
          : lpp::LearnedCostPolicy::Mode::kDisabled;
  config.learned_cost_policy.maximum_absolute_energy_correction = 1.0;
  config.learned_cost_policy
      .maximum_absolute_nonfatal_risk_correction = 1.0;
  if (enable_learned) {
    config.learned_cost_model_ref = Ref("learned-model", 2U, 'a');
  }
  return std::make_shared<const lpp::PlannerAlgorithmConfig>(
      std::move(config));
}

lpp::SafeProjectionRequest ProjectionRequest(
    std::shared_ptr<const lpp::ImmutableMapSnapshot> map,
    lpp::SafetyCapabilityProfile profile,
    bool enable_learned = false) {
  return {
      .map = std::move(map),
      .capability = SharedProfile(std::move(profile)),
      .algorithm_config = MakeConfig(enable_learned),
      .learned_cost = nullptr,
  };
}

lpp::SafeProjection RequireProjection(
    lpp::SafeProjectionRequest request) {
  auto result = lpp::BuildSafeProjection(request);
  EXPECT_TRUE(lpp::IsOk(result));
  if (!lpp::IsOk(result)) {
    return {};
  }
  return std::get<lpp::SafeProjection>(std::move(result));
}

std::size_t Index(std::size_t x, std::size_t y,
                  std::size_t width = kDefaultWidth) {
  return y * width + x;
}

std::shared_ptr<const lpp::LearnedCostSnapshot> MakeLearned(
    const lpp::ImmutableMapSnapshot& map) {
  const auto count = map.geometry().CellCount();
  lpp::LearnedCostSnapshotInput input{
      .snapshot_ref = Ref("learned-snapshot", 1U, 'b'),
      .model_ref = Ref("learned-model", 2U, 'a'),
      .source_map_snapshot_ref = map.snapshot_ref(),
      .source_map_revision = map.map_revision(),
      .frame_id = std::string(map.frame_id()),
      .feature_contract_id = "safe-projection-features",
      .output_contract_id = "safe-projection-costs",
      .generated_at =
          {.clock_id = "mission", .tick = std::chrono::nanoseconds{20}},
      .certified_energy_correction_bounds = {-1.0, 1.0},
      .certified_nonfatal_risk_correction_bounds = {-1.0, 1.0},
      .energy_correction = std::vector<float>(count, 0.75F),
      .nonfatal_risk_correction = std::vector<float>(count, 0.5F),
  };
  auto result = lpp::LearnedCostSnapshot::Create(input, count);
  EXPECT_TRUE(lpp::IsOk(result));
  if (!lpp::IsOk(result)) {
    return nullptr;
  }
  return std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(
      std::move(result));
}

lpp::WheeledOrLeggedState MovingState(double x, double speed_mps,
                                      double yaw_rate_radps = 0.0) {
  return {
      .position_m = {x, 0.5, 0.0},
      .yaw_rad = 0.25,
      .linear_velocity_mps = {speed_mps, 0.0, 0.0},
      .yaw_rate_radps = yaw_rate_radps,
  };
}

}  // namespace

TEST(SafeProjectionLimits, ResolvesAllClosedCapabilityVariants) {
  const auto wheel = lpp::ResolveProjectionLimits(WheelProfile());
  const auto legged = lpp::ResolveProjectionLimits(LeggedProfile());
  const auto hopper = lpp::ResolveProjectionLimits(HopperProfile());

  ASSERT_TRUE(lpp::IsOk(wheel));
  ASSERT_TRUE(lpp::IsOk(legged));
  ASSERT_TRUE(lpp::IsOk(hopper));
  EXPECT_EQ(std::get<lpp::SafetyProjectionLimits>(wheel).platform_type,
            lpp::PlatformType::kWheeled);
  EXPECT_EQ(std::get<lpp::SafetyProjectionLimits>(legged).platform_type,
            lpp::PlatformType::kLegged);
  EXPECT_EQ(std::get<lpp::SafetyProjectionLimits>(hopper).platform_type,
            lpp::PlatformType::kHopper);
  EXPECT_FALSE(
      std::get<lpp::SafetyProjectionLimits>(wheel)
          .maximum_roughness_m.has_value());
  EXPECT_TRUE(
      std::get<lpp::SafetyProjectionLimits>(legged)
          .supports_safe_stop_anchor);
  EXPECT_FALSE(
      std::get<lpp::SafetyProjectionLimits>(hopper)
          .supports_safe_stop_anchor);
}

TEST(SafeProjectionLimits, RejectsMissingMandatoryHardLimits) {
  auto wheel = WheelProfile();
  std::get<lpp::WheeledCapability>(wheel.content)
      .hard_limits.maximum_braking_deceleration_mps2 = 0.0;
  EXPECT_FALSE(lpp::IsOk(lpp::ResolveProjectionLimits(wheel)));

  auto legged = LeggedProfile();
  std::get<lpp::LeggedCapability>(legged.content)
      .body_velocity_limits.linear_acceleration_mps2 = 0.0;
  EXPECT_FALSE(lpp::IsOk(lpp::ResolveProjectionLimits(legged)));

  auto hopper = HopperProfile();
  std::get<lpp::HopperCapability>(hopper.content)
      .launch_limits.maximum_launch_speed_mps = 0.0;
  EXPECT_FALSE(lpp::IsOk(lpp::ResolveProjectionLimits(hopper)));
}

TEST(SafeProjection, UnknownObstacleAndExcessSlopeAreNeverHardFeasible) {
  auto input = MakeMapInput();
  input.known_mask[Index(1U, 3U)] = 0U;
  input.confidence[Index(1U, 3U)] = 0.0F;
  input.hard_obstacle_mask[Index(2U, 3U)] = 1U;
  const double slope = 0.8;
  input.normal_x[Index(3U, 3U)] =
      static_cast<float>(std::sin(slope));
  input.normal_z[Index(3U, 3U)] =
      static_cast<float>(std::cos(slope));
  const auto map = MakeMap(input);
  ASSERT_NE(map, nullptr);

  for (auto profile :
       {WheelProfile(), LeggedProfile(), HopperProfile()}) {
    const auto projection =
        RequireProjection(ProjectionRequest(map, std::move(profile)));
    EXPECT_FALSE(projection.HardFeasible({1, 3}));
    EXPECT_FALSE(projection.HardFeasible({2, 3}));
    EXPECT_FALSE(projection.HardFeasible({3, 3}));
  }
}

TEST(SafeProjection, AppliesOnlyDeclaredPlatformRoughnessLimits) {
  auto input = MakeMapInput();
  input.roughness_m[Index(3U, 3U)] = 0.3F;
  const auto map = MakeMap(input);
  ASSERT_NE(map, nullptr);

  const auto wheel =
      RequireProjection(ProjectionRequest(map, WheelProfile()));
  const auto legged =
      RequireProjection(ProjectionRequest(map, LeggedProfile()));
  const auto hopper =
      RequireProjection(ProjectionRequest(map, HopperProfile()));
  EXPECT_TRUE(wheel.HardFeasible({3, 3}));
  EXPECT_FALSE(legged.HardFeasible({3, 3}));
  EXPECT_FALSE(hopper.HardFeasible({3, 3}));
}

TEST(SafeProjection, RejectsInsufficientPlatformClearance) {
  const auto map = MakeMap(MakeMapInput());
  ASSERT_NE(map, nullptr);

  for (auto profile :
       {WheelProfile(0.6, 4.1), LeggedProfile(0.6, 0.2, 4.1),
        HopperProfile(0.6, 0.2, 4.1)}) {
    const auto projection =
        RequireProjection(ProjectionRequest(map, std::move(profile)));
    EXPECT_FALSE(projection.HardFeasible({3, 3}));
  }
}

TEST(SafeProjection, ComputesDeterministicEuclideanEsdfAndComponents) {
  auto input = MakeMapInput(7U, 7U);
  input.hard_obstacle_mask[Index(3U, 3U)] = 1U;
  const auto map = MakeMap(input);
  ASSERT_NE(map, nullptr);
  const auto first =
      RequireProjection(ProjectionRequest(map, WheelProfile()));
  const auto second =
      RequireProjection(ProjectionRequest(map, WheelProfile()));

  EXPECT_FLOAT_EQ(first.ClearanceMeters({3, 3}), 0.0F);
  EXPECT_NEAR(first.ClearanceMeters({2, 2}), std::sqrt(2.0), 1.0e-6);
  EXPECT_FLOAT_EQ(first.ClearanceMeters({0, 3}), 1.0F);
  EXPECT_EQ(first.esdf_clearance_m, second.esdf_clearance_m);
  EXPECT_EQ(first.connected_component, second.connected_component);

  auto split_input = MakeMapInput(5U, 3U);
  for (std::size_t y = 0U; y < 3U; ++y) {
    split_input.hard_obstacle_mask[Index(2U, y, 5U)] = 1U;
  }
  const auto split_map = MakeMap(split_input);
  ASSERT_NE(split_map, nullptr);
  const auto split =
      RequireProjection(ProjectionRequest(split_map, WheelProfile()));
  EXPECT_EQ(split.connected_component[Index(0U, 0U, 5U)], 0);
  EXPECT_EQ(split.connected_component[Index(1U, 2U, 5U)], 0);
  EXPECT_EQ(split.connected_component[Index(3U, 0U, 5U)], 1);
  EXPECT_EQ(split.connected_component[Index(4U, 2U, 5U)], 1);
  EXPECT_EQ(split.connected_component[Index(2U, 1U, 5U)], -1);
}

TEST(SafeProjection, RetainsExactImmutableTerrainViews) {
  auto input = MakeMapInput();
  input.elevation_m[0] = 0.25F;
  input.roughness_m[0] = 0.05F;
  input.normal_x[0] = 0.6F;
  input.normal_z[0] = 0.8F;
  const auto map = MakeMap(input);
  ASSERT_NE(map, nullptr);
  const auto projection =
      RequireProjection(ProjectionRequest(map, HopperProfile()));

  EXPECT_EQ(projection.source_map().get(), map.get());
  EXPECT_EQ(projection.ElevationMeters().data(),
            map->ElevationMeters().data());
  EXPECT_EQ(projection.RoughnessMeters().data(),
            map->RoughnessMeters().data());
  EXPECT_EQ(projection.SurfaceNormals().x.data(),
            map->SurfaceNormals().x.data());
  EXPECT_FLOAT_EQ(projection.ElevationMeters()[0], 0.25F);
  EXPECT_FLOAT_EQ(projection.RoughnessMeters()[0], 0.05F);
  EXPECT_FLOAT_EQ(projection.SurfaceNormals().z[0], 0.8F);
}

TEST(SafeProjection, LearnedCostChangesOnlyResolvedSecondaryCosts) {
  const auto map = MakeMap(MakeMapInput());
  ASSERT_NE(map, nullptr);
  auto baseline_request = ProjectionRequest(map, WheelProfile(), true);
  auto learned_request = baseline_request;
  learned_request.learned_cost = MakeLearned(*map);
  ASSERT_NE(learned_request.learned_cost, nullptr);

  const auto baseline = RequireProjection(std::move(baseline_request));
  const auto learned = RequireProjection(std::move(learned_request));

  EXPECT_EQ(baseline.known_mask, learned.known_mask);
  EXPECT_EQ(baseline.hard_feasible_mask, learned.hard_feasible_mask);
  EXPECT_EQ(baseline.esdf_clearance_m, learned.esdf_clearance_m);
  EXPECT_EQ(baseline.analytic_time_cost_s,
            learned.analytic_time_cost_s);
  EXPECT_EQ(baseline.analytic_energy, learned.analytic_energy);
  EXPECT_EQ(baseline.analytic_nonfatal_risk,
            learned.analytic_nonfatal_risk);
  EXPECT_EQ(baseline.conservative_speed_limit_mps,
            learned.conservative_speed_limit_mps);
  EXPECT_EQ(baseline.connected_component,
            learned.connected_component);
  EXPECT_EQ(baseline.safe_stop_candidate_mask,
            learned.safe_stop_candidate_mask);
  EXPECT_NE(baseline.resolved_energy, learned.resolved_energy);
  EXPECT_NE(baseline.resolved_nonfatal_risk,
            learned.resolved_nonfatal_risk);
  EXPECT_EQ(learned.soft_cost_source,
            lpp::SoftCostSource::kAnalyticPlusPinnedLearned);
}

TEST(SafeProjection, PropagatesBoundedCostCompositionError) {
  auto input = MakeMapInput(1U, 1U);
  input.geometry.resolution_m = std::numeric_limits<double>::max();
  const auto map = MakeMap(input);
  ASSERT_NE(map, nullptr);
  const auto result =
      lpp::BuildSafeProjection(ProjectionRequest(map, WheelProfile()));

  ASSERT_FALSE(lpp::IsOk(result));
  const auto& error = std::get<lpp::Error>(result);
  EXPECT_EQ(error.code, lpp::ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "analytic_cost_invalid");
}

TEST(SafeProjection, DerivesConservativeStaticSpeedAndTraversalTime) {
  auto input = MakeMapInput();
  input.layer_manifest.push_back(
      {lpp::LayerKind::kStaticSpeedLimit,
       Ref("static-speed", 1U, '7')});
  input.static_speed_limit_mps =
      std::vector<float>(input.geometry.CellCount(), 0.4F);
  const auto map = MakeMap(input);
  ASSERT_NE(map, nullptr);
  const auto projection =
      RequireProjection(ProjectionRequest(map, WheelProfile()));
  const auto center = Index(3U, 3U);

  EXPECT_FLOAT_EQ(
      projection.conservative_speed_limit_mps[center], 0.4F);
  EXPECT_FLOAT_EQ(projection.analytic_time_cost_s[center], 2.5F);
}

TEST(SafeStopAnchor, WheelUsesCertifiedBrakingAndReturnsCanonicalZero) {
  const auto map = MakeMap(MakeMapInput(7U, 1U));
  ASSERT_NE(map, nullptr);
  const auto profile = WheelProfile();
  const auto projection =
      RequireProjection(ProjectionRequest(map, profile));
  const auto result = lpp::ResolveSafeStopAnchor(
      projection, MovingState(0.5, 2.0), profile);

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& anchor = std::get<lpp::SafeStopAnchor>(result);
  EXPECT_DOUBLE_EQ(anchor.target_linear_velocity_mps, 0.0);
  EXPECT_DOUBLE_EQ(anchor.target_yaw_rate_radps, 0.0);
  EXPECT_FALSE(std::signbit(anchor.target_linear_velocity_mps));
  EXPECT_FALSE(std::signbit(anchor.target_yaw_rate_radps));
  EXPECT_DOUBLE_EQ(anchor.pose.position_m.x, 2.5);
  EXPECT_DOUBLE_EQ(anchor.pose.position_m.y, 0.5);
  EXPECT_DOUBLE_EQ(anchor.pose.yaw_rad, 0.25);
  EXPECT_EQ(anchor.terrain_certification_ref, map->snapshot_ref());
}

TEST(SafeStopAnchor, LeggedUsesBodyAccelerationAsConservativeBrakeBound) {
  const auto map = MakeMap(MakeMapInput(5U, 1U));
  ASSERT_NE(map, nullptr);
  const auto profile = LeggedProfile();
  const auto projection =
      RequireProjection(ProjectionRequest(map, profile));
  const auto result = lpp::ResolveSafeStopAnchor(
      projection, MovingState(0.5, 1.0, 0.5), profile);

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& anchor = std::get<lpp::SafeStopAnchor>(result);
  EXPECT_DOUBLE_EQ(anchor.pose.position_m.x, 1.5);
  EXPECT_EQ(anchor.terrain_certification_ref, map->snapshot_ref());
}

TEST(SafeStopAnchor, RejectsInsufficientDistanceAndCapabilityMismatch) {
  const auto short_map = MakeMap(MakeMapInput(2U, 1U));
  ASSERT_NE(short_map, nullptr);
  const auto wheel = WheelProfile();
  const auto projection =
      RequireProjection(ProjectionRequest(short_map, wheel));

  EXPECT_FALSE(lpp::IsOk(lpp::ResolveSafeStopAnchor(
      projection, MovingState(0.5, 2.0), wheel)));
  EXPECT_FALSE(lpp::IsOk(lpp::ResolveSafeStopAnchor(
      projection, MovingState(0.5, 0.5), LeggedProfile())));
  EXPECT_FALSE(lpp::IsOk(lpp::ResolveSafeStopAnchor(
      projection, MovingState(0.5, 0.5, 2.0), wheel)));
}

TEST(SafeStopAnchor, HopperProjectionHasNoStaticStopCandidates) {
  const auto map = MakeMap(MakeMapInput());
  ASSERT_NE(map, nullptr);
  const auto hopper = HopperProfile();
  const auto projection =
      RequireProjection(ProjectionRequest(map, hopper));
  EXPECT_TRUE(std::ranges::all_of(
      projection.safe_stop_candidate_mask,
      [](std::uint8_t value) { return value == 0U; }));
  EXPECT_FALSE(lpp::IsOk(lpp::ResolveSafeStopAnchor(
      projection, MovingState(0.5, 0.0), hopper)));
}
