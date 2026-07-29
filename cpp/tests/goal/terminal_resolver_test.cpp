#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"

namespace lpp = lunar::planning::v3;

namespace {

lpp::ContentRef Ref(std::string id, std::uint32_t revision,
                    char digest_digit) {
  return {
      .id = std::move(id),
      .revision = revision,
      .content_hash = std::string(64U, digest_digit),
  };
}

std::size_t Index(const lpp::MapSnapshotInput& input, std::size_t x,
                  std::size_t y) {
  return y * input.geometry.width + x;
}

lpp::MapSnapshotInput MakeMapInput(std::size_t width = 5U,
                                   std::size_t height = 3U,
                                   char snapshot_digest = 'a') {
  const std::size_t count = width * height;
  return {
      .snapshot_ref = Ref("terminal-map", 17U, snapshot_digest),
      .map_revision = 17U,
      .immutable_data_handle = "terminal-map-data",
      .source_time =
          {.clock_id = "mission", .tick = std::chrono::nanoseconds{42}},
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
              {lpp::LayerKind::kElevation, Ref("elevation", 1U, '2')},
              {lpp::LayerKind::kTerrainNormal, Ref("normal", 1U, '3')},
              {lpp::LayerKind::kRoughness, Ref("roughness", 1U, '4')},
              {lpp::LayerKind::kHardObstacle, Ref("obstacle", 1U, '5')},
              {lpp::LayerKind::kConfidence, Ref("confidence", 1U, '6')},
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

lpp::DeterministicVectorSet3 ZeroAabb() {
  return lpp::AxisAlignedBox3{
      .center = {0.0, 0.0, 0.0},
      .half_extent = {0.0, 0.0, 0.0},
  };
}

lpp::SafetyCapabilityProfile WheelProfile(
    double minimum_clearance_m = 0.0,
    lpp::DeterministicVectorSet3 position_error = ZeroAabb()) {
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
              .maximum_slope_rad = 0.6,
              .minimum_clearance_m = minimum_clearance_m,
          },
      .certified_state_error_bounds =
          {
              .position_bound_m = std::move(position_error),
              .yaw_bound_rad = {0.0, 0.0},
              .linear_velocity_bound_mps = ZeroAabb(),
              .yaw_rate_bound_radps = {0.0, 0.0},
          },
  };
  return {
      .content_ref = Ref("wheel-capability", 3U, 'd'),
      .content = std::move(wheel),
  };
}

lpp::PlannerAlgorithmConfig AlgorithmConfig(
    std::size_t maximum_generated_candidates = 32U) {
  lpp::PlannerAlgorithmConfig config{};
  config.content_ref = Ref("terminal-algorithm", 2U, 'e');
  config.ara_star.resource_caps.maximum_generated_candidates =
      maximum_generated_candidates;
  config.learned_cost_policy.mode =
      lpp::LearnedCostPolicy::Mode::kDisabled;
  config.learned_cost_policy.maximum_absolute_energy_correction = 1.0;
  config.learned_cost_policy
      .maximum_absolute_nonfatal_risk_correction = 1.0;
  return config;
}

struct Scenario final {
  std::shared_ptr<const lpp::ImmutableMapSnapshot> map;
  lpp::SafetyCapabilityProfile capability;
  lpp::PlannerAlgorithmConfig algorithm_config;
  lpp::SafeProjection projection;
};

Scenario MakeScenario(
    const lpp::MapSnapshotInput& input,
    lpp::SafetyCapabilityProfile capability = WheelProfile(),
    std::size_t maximum_generated_candidates = 32U) {
  auto map_result = lpp::ImmutableMapSnapshot::Create(input);
  EXPECT_TRUE(lpp::IsOk(map_result));
  if (!lpp::IsOk(map_result)) {
    return {};
  }
  auto map =
      std::get<std::shared_ptr<const lpp::ImmutableMapSnapshot>>(
          std::move(map_result));
  auto config = AlgorithmConfig(maximum_generated_candidates);
  auto capability_for_projection =
      std::make_shared<const lpp::SafetyCapabilityProfile>(capability);
  auto config_for_projection =
      std::make_shared<const lpp::PlannerAlgorithmConfig>(config);
  auto projection_result = lpp::BuildSafeProjection(
      {
          .map = map,
          .capability = std::move(capability_for_projection),
          .algorithm_config = std::move(config_for_projection),
          .learned_cost = nullptr,
      });
  EXPECT_TRUE(lpp::IsOk(projection_result));
  if (!lpp::IsOk(projection_result)) {
    return {};
  }
  return {
      .map = std::move(map),
      .capability = std::move(capability),
      .algorithm_config = std::move(config),
      .projection =
          std::get<lpp::SafeProjection>(std::move(projection_result)),
  };
}

lpp::Vec3 CellCenter(lpp::Cell cell, double z = 0.0) {
  return {
      .x = static_cast<double>(cell.x) + 0.5,
      .y = static_cast<double>(cell.y) + 0.5,
      .z = z,
  };
}

lpp::GoalRegion PointGoalAt(lpp::Cell cell,
                            double tolerance_m = 0.01) {
  return {
      .goal_id = "point-goal",
      .target =
          lpp::PointGoal{
              .position_m = CellCenter(cell),
              .position_tolerance_m = tolerance_m,
          },
  };
}

lpp::GoalRegion PlanarGoal(
    std::vector<lpp::Vec2> vertices_uv) {
  return {
      .goal_id = "planar-goal",
      .target =
          lpp::PlanarRegionGoal{
              .plane =
                  {
                      .origin_m = {0.0, 0.0, 0.0},
                      .normal = {0.0, 0.0, 1.0},
                      .basis_u = {1.0, 0.0, 0.0},
                      .basis_v = {0.0, 1.0, 0.0},
                      .residual_bound_m = 0.01,
                  },
              .polygon = {.vertices_uv = std::move(vertices_uv)},
              .normal_tolerance_m = 0.01,
          },
  };
}

lpp::Result<lpp::ResolvedTerminalSet> Resolve(
    const Scenario& scenario, const lpp::GoalRegion& goal,
    lpp::Cell start_cell) {
  return lpp::ResolveTerminal(
      {
          .projection = scenario.projection,
          .goal_region = goal,
          .start_cell = start_cell,
          .capability = scenario.capability,
          .algorithm_config = scenario.algorithm_config,
      });
}

std::vector<lpp::Cell> CandidateCells(
    const lpp::ResolvedTerminalSet& terminals) {
  std::vector<lpp::Cell> cells;
  cells.reserve(terminals.candidates.size());
  for (const auto& candidate : terminals.candidates) {
    cells.push_back(candidate.cell);
  }
  return cells;
}

TEST(TerminalResolver, UsesReachablePointGoalCellAndYaw) {
  const Scenario scenario = MakeScenario(MakeMapInput());
  auto goal = PointGoalAt({4, 1});
  goal.optional_yaw_interval =
      lpp::CircularYawInterval{
          .start_rad = -0.4,
          .span_rad = 0.8,
          .closed = true,
      };

  const auto result = Resolve(scenario, goal, {0, 1});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  EXPECT_EQ(terminals.kind, lpp::TerminalKind::kGoal);
  EXPECT_EQ(terminals.reason_code, "known_safe_goal_reachable");
  ASSERT_EQ(terminals.candidates.size(), 1U);
  EXPECT_EQ(terminals.candidates.front().cell, (lpp::Cell{4, 1}));
  EXPECT_EQ(terminals.candidates.front().position_m.x, 4.5);
  EXPECT_TRUE(terminals.candidates.front().requires_zero_speed);
  ASSERT_TRUE(terminals.candidates.front().yaw_interval.has_value());
  EXPECT_DOUBLE_EQ(
      terminals.candidates.front().yaw_interval->start_rad, -0.4);
  EXPECT_FALSE(terminals.unresolved_tail.has_value());
}

TEST(TerminalResolver, RasterizesPlanarGoalInStableRowMajorOrder) {
  const Scenario scenario = MakeScenario(MakeMapInput(6U, 4U));
  const auto goal = PlanarGoal(
      {{3.0, 1.0}, {5.0, 1.0}, {5.0, 3.0}, {3.0, 3.0}});

  const auto result = Resolve(scenario, goal, {0, 1});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  ASSERT_EQ(terminals.kind, lpp::TerminalKind::kGoal);
  const std::vector<lpp::Cell> expected{
      {3, 1}, {4, 1}, {3, 2}, {4, 2}};
  EXPECT_EQ(CandidateCells(terminals), expected);
}

TEST(TerminalResolver, ReportsFullyKnownBlockedGoalAsInfeasible) {
  auto input = MakeMapInput();
  input.hard_obstacle_mask[Index(input, 4U, 1U)] = 1U;
  const Scenario scenario = MakeScenario(input);

  const auto result =
      Resolve(scenario, PointGoalAt({4, 1}), {0, 1});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  EXPECT_EQ(terminals.kind, lpp::TerminalKind::kGoalInfeasible);
  EXPECT_EQ(terminals.reason_code,
            "goal_fully_known_and_infeasible");
  EXPECT_TRUE(terminals.candidates.empty());
  EXPECT_FALSE(terminals.unresolved_tail.has_value());
}

TEST(TerminalResolver, DoesNotConvertDisconnectedKnownGoalToFrontier) {
  auto input = MakeMapInput(5U, 3U);
  for (std::size_t y = 0U; y < input.geometry.height; ++y) {
    input.hard_obstacle_mask[Index(input, 2U, y)] = 1U;
  }
  const Scenario scenario = MakeScenario(input);

  const auto result =
      Resolve(scenario, PointGoalAt({4, 1}), {0, 1});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  EXPECT_EQ(terminals.kind, lpp::TerminalKind::kNoKnownSafeRoute);
  EXPECT_EQ(terminals.reason_code,
            "goal_in_disconnected_known_component");
  EXPECT_TRUE(terminals.candidates.empty());
  EXPECT_FALSE(terminals.unresolved_tail.has_value());
}

TEST(TerminalResolver, InvalidStartUsesFirstDecisionTableBranch) {
  const Scenario scenario = MakeScenario(MakeMapInput());

  const auto result =
      Resolve(scenario, PointGoalAt({4, 1}), {-1, 1});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  EXPECT_EQ(terminals.kind, lpp::TerminalKind::kNoKnownSafeRoute);
  EXPECT_EQ(terminals.reason_code, "start_not_hard_feasible");
}

TEST(TerminalResolver,
     StopsAtKnownSafeFrontierAndClipsNonExecutablePreview) {
  auto input = MakeMapInput(5U, 3U);
  for (std::size_t y = 0U; y < input.geometry.height; ++y) {
    for (std::size_t x = 2U; x < input.geometry.width; ++x) {
      input.known_mask[Index(input, x, y)] = 0U;
    }
  }
  const Scenario scenario = MakeScenario(input);

  const auto result =
      Resolve(scenario, PointGoalAt({4, 1}), {0, 1});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  ASSERT_EQ(terminals.kind, lpp::TerminalKind::kSafeFrontier);
  EXPECT_EQ(terminals.reason_code,
            "known_safe_frontier_before_unknown_space");
  ASSERT_EQ(terminals.candidates.size(), 1U);
  EXPECT_EQ(terminals.candidates.front().cell, (lpp::Cell{1, 1}));
  EXPECT_TRUE(terminals.candidates.front().requires_zero_speed);
  ASSERT_TRUE(terminals.unresolved_tail.has_value());
  EXPECT_FALSE(terminals.unresolved_tail->executable);
  EXPECT_EQ(terminals.unresolved_tail->reason_code,
            "mission_intent_clipped_at_known_boundary");
  ASSERT_EQ(terminals.unresolved_tail->intent_polyline_m.size(), 2U);
  EXPECT_DOUBLE_EQ(
      terminals.unresolved_tail->intent_polyline_m.front().x, 1.5);
  EXPECT_DOUBLE_EQ(
      terminals.unresolved_tail->intent_polyline_m.front().y, 1.5);
  EXPECT_DOUBLE_EQ(
      terminals.unresolved_tail->intent_polyline_m.back().x, 2.0);
  EXPECT_DOUBLE_EQ(
      terminals.unresolved_tail->intent_polyline_m.back().y, 1.5);
  EXPECT_LT(
      terminals.unresolved_tail->intent_polyline_m.back().x, 4.5);
}

TEST(TerminalResolver,
     ClustersWithStableOrderAndCapsGeneratedRepresentatives) {
  auto input = MakeMapInput(9U, 5U);
  input.known_mask[Index(input, 6U, 1U)] = 0U;
  input.known_mask[Index(input, 6U, 2U)] = 0U;
  const Scenario scenario =
      MakeScenario(input, WheelProfile(), 2U);
  const auto goal = PointGoalAt({6, 1});

  const auto first = Resolve(scenario, goal, {4, 4});
  const auto second = Resolve(scenario, goal, {4, 4});

  ASSERT_TRUE(lpp::IsOk(first));
  ASSERT_TRUE(lpp::IsOk(second));
  const auto& first_terminals =
      std::get<lpp::ResolvedTerminalSet>(first);
  const auto& second_terminals =
      std::get<lpp::ResolvedTerminalSet>(second);
  ASSERT_EQ(first_terminals.kind, lpp::TerminalKind::kSafeFrontier);
  ASSERT_EQ(first_terminals.candidates.size(), 2U);
  const std::vector<lpp::Cell> expected{{6, 0}, {5, 1}};
  EXPECT_EQ(CandidateCells(first_terminals), expected);
  EXPECT_EQ(CandidateCells(first_terminals),
            CandidateCells(second_terminals));
  EXPECT_EQ(first_terminals.candidates[0].stable_id,
            second_terminals.candidates[0].stable_id);
  EXPECT_EQ(first_terminals.candidates[1].stable_id,
            second_terminals.candidates[1].stable_id);
}

TEST(TerminalResolver,
     AppliesAabbFootprintAndErrorClearanceAtExactBoundary) {
  auto input = MakeMapInput(5U, 3U);
  for (std::size_t y = 0U; y < input.geometry.height; ++y) {
    for (std::size_t x = 3U; x < input.geometry.width; ++x) {
      input.known_mask[Index(input, x, y)] = 0U;
    }
  }
  const auto exact_error = lpp::AxisAlignedBox3{
      .center = {0.0, 0.0, 0.0},
      .half_extent = {0.6, 0.0, 0.0},
  };
  const auto excessive_error = lpp::AxisAlignedBox3{
      .center = {0.0, 0.0, 0.0},
      .half_extent = {0.6001, 0.0, 0.0},
  };
  const Scenario exact = MakeScenario(
      input, WheelProfile(0.4, exact_error));
  const Scenario excessive = MakeScenario(
      input, WheelProfile(0.4, excessive_error));
  const auto goal = PointGoalAt({4, 1});

  const auto exact_result = Resolve(exact, goal, {1, 1});
  const auto excessive_result =
      Resolve(excessive, goal, {1, 1});

  ASSERT_TRUE(lpp::IsOk(exact_result));
  ASSERT_TRUE(lpp::IsOk(excessive_result));
  const auto& exact_terminals =
      std::get<lpp::ResolvedTerminalSet>(exact_result);
  const auto& excessive_terminals =
      std::get<lpp::ResolvedTerminalSet>(excessive_result);
  ASSERT_EQ(exact_terminals.kind, lpp::TerminalKind::kSafeFrontier);
  ASSERT_EQ(exact_terminals.candidates.size(), 1U);
  EXPECT_DOUBLE_EQ(exact_terminals.candidates.front().clearance_m,
                   1.0);
  EXPECT_EQ(excessive_terminals.kind,
            lpp::TerminalKind::kNoKnownSafeRoute);
  EXPECT_EQ(excessive_terminals.reason_code,
            "no_certified_safe_frontier");
}

TEST(TerminalResolver,
     AppliesBallOffsetAndRadiusAtExactClearanceBoundary) {
  auto input = MakeMapInput(5U, 3U);
  for (std::size_t y = 0U; y < input.geometry.height; ++y) {
    for (std::size_t x = 3U; x < input.geometry.width; ++x) {
      input.known_mask[Index(input, x, y)] = 0U;
    }
  }
  const auto exact_error = lpp::EuclideanBall3{
      .center = {0.3, 0.4, 0.0},
      .radius = 0.5,
  };
  const auto excessive_error = lpp::EuclideanBall3{
      .center = {0.3, 0.4, 0.0},
      .radius = 0.5001,
  };
  const Scenario exact =
      MakeScenario(input, WheelProfile(0.0, exact_error));
  const Scenario excessive =
      MakeScenario(input, WheelProfile(0.0, excessive_error));
  const auto goal = PointGoalAt({4, 1});

  const auto exact_result = Resolve(exact, goal, {1, 1});
  const auto excessive_result =
      Resolve(excessive, goal, {1, 1});

  ASSERT_TRUE(lpp::IsOk(exact_result));
  ASSERT_TRUE(lpp::IsOk(excessive_result));
  EXPECT_EQ(std::get<lpp::ResolvedTerminalSet>(exact_result).kind,
            lpp::TerminalKind::kSafeFrontier);
  EXPECT_EQ(
      std::get<lpp::ResolvedTerminalSet>(excessive_result).kind,
      lpp::TerminalKind::kNoKnownSafeRoute);
}

TEST(TerminalResolver, ReportsNoFrontierWhenStaticStopMaskRejectsAll) {
  auto input = MakeMapInput(5U, 3U);
  for (std::size_t y = 0U; y < input.geometry.height; ++y) {
    for (std::size_t x = 2U; x < input.geometry.width; ++x) {
      input.known_mask[Index(input, x, y)] = 0U;
    }
  }
  Scenario scenario = MakeScenario(input);
  std::fill(scenario.projection.safe_stop_candidate_mask.begin(),
            scenario.projection.safe_stop_candidate_mask.end(),
            std::uint8_t{0});

  const auto result =
      Resolve(scenario, PointGoalAt({4, 1}), {0, 1});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  EXPECT_EQ(terminals.kind, lpp::TerminalKind::kNoKnownSafeRoute);
  EXPECT_EQ(terminals.reason_code,
            "no_certified_safe_frontier");
  EXPECT_TRUE(terminals.candidates.empty());
  EXPECT_FALSE(terminals.unresolved_tail.has_value());
}

TEST(TerminalResolver,
     IsRepeatableAcrossCyclicPlanarVertexPermutation) {
  const Scenario scenario = MakeScenario(MakeMapInput(6U, 4U));
  const std::vector<lpp::Vec2> vertices{
      {3.0, 1.0}, {5.0, 1.0}, {5.0, 3.0}, {3.0, 3.0}};
  std::vector<lpp::Vec2> rotated = vertices;
  std::rotate(rotated.begin(), rotated.begin() + 2, rotated.end());

  const auto baseline =
      Resolve(scenario, PlanarGoal(vertices), {0, 1});
  const auto permuted =
      Resolve(scenario, PlanarGoal(std::move(rotated)), {0, 1});
  const auto repeated =
      Resolve(scenario, PlanarGoal(vertices), {0, 1});

  ASSERT_TRUE(lpp::IsOk(baseline));
  ASSERT_TRUE(lpp::IsOk(permuted));
  ASSERT_TRUE(lpp::IsOk(repeated));
  const auto& baseline_terminals =
      std::get<lpp::ResolvedTerminalSet>(baseline);
  const auto& permuted_terminals =
      std::get<lpp::ResolvedTerminalSet>(permuted);
  const auto& repeated_terminals =
      std::get<lpp::ResolvedTerminalSet>(repeated);
  EXPECT_EQ(CandidateCells(baseline_terminals),
            CandidateCells(permuted_terminals));
  EXPECT_EQ(CandidateCells(baseline_terminals),
            CandidateCells(repeated_terminals));
  ASSERT_EQ(baseline_terminals.candidates.size(),
            permuted_terminals.candidates.size());
  for (std::size_t index = 0U;
       index < baseline_terminals.candidates.size(); ++index) {
    EXPECT_EQ(baseline_terminals.candidates[index].stable_id,
              permuted_terminals.candidates[index].stable_id);
    EXPECT_EQ(baseline_terminals.candidates[index].stable_id,
              repeated_terminals.candidates[index].stable_id);
  }
}

TEST(TerminalResolver, StableIdIncludesFullMapSnapshotIdentity) {
  const Scenario first = MakeScenario(MakeMapInput(5U, 3U, 'a'));
  const Scenario second = MakeScenario(MakeMapInput(5U, 3U, 'f'));
  const auto goal = PointGoalAt({4, 1});

  const auto first_result = Resolve(first, goal, {0, 1});
  const auto second_result = Resolve(second, goal, {0, 1});

  ASSERT_TRUE(lpp::IsOk(first_result));
  ASSERT_TRUE(lpp::IsOk(second_result));
  const auto& first_terminal =
      std::get<lpp::ResolvedTerminalSet>(first_result);
  const auto& second_terminal =
      std::get<lpp::ResolvedTerminalSet>(second_result);
  ASSERT_EQ(first_terminal.candidates.size(), 1U);
  ASSERT_EQ(second_terminal.candidates.size(), 1U);
  EXPECT_NE(first_terminal.candidates.front().stable_id,
            second_terminal.candidates.front().stable_id);
}

TEST(TerminalResolver, RejectsMalformedProjectionFailClosed) {
  Scenario scenario = MakeScenario(MakeMapInput());
  scenario.projection.known_mask.pop_back();

  const auto result =
      Resolve(scenario, PointGoalAt({4, 1}), {0, 1});

  ASSERT_FALSE(lpp::IsOk(result));
  const auto& error = std::get<lpp::Error>(result);
  EXPECT_EQ(error.code, lpp::ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.field_path,
            "terminal_resolution.projection.known_mask");
}

}  // namespace
