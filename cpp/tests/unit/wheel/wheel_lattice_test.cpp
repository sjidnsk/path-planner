#include <algorithm>
#include <chrono>
#include <cmath>
#include <ranges>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/wheel/wheel_lattice.hpp"

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] ContentRef Ref(std::string id, char digit) {
  return {std::move(id), 1U, std::string(64U, digit)};
}

[[nodiscard]] WheelMotionPrimitive Primitive(
    std::string id,
    WheelMotionPrimitive::Kind kind,
    PoseXyzYaw relative_end) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .relative_end_pose = relative_end,
      .nominal_duration = DurationNanoseconds{500ms},
      .swept_geometry_ref = Ref("sweep", 'b'),
  };
}

[[nodiscard]] SafetyCapabilityProfile Capability() {
  return {
      .content_ref = Ref("wheel-capability", 'a'),
      .content =
          WheeledCapability{
              .frame_id = "map",
              .collision_envelope =
                  {
                      .vertices_xy_m =
                          {{-0.2, -0.2}, {0.2, -0.2},
                           {0.2, 0.2}, {-0.2, 0.2}},
                      .minimum_z_m = -0.1,
                      .maximum_z_m = 0.5,
                  },
              .motion_model_ref = Ref("motion", 'c'),
              .analytic_cost_model_ref = Ref("cost", 'd'),
              .hard_limits =
                  {
                      .maximum_forward_speed_mps = 1.0,
                      .maximum_reverse_speed_mps = 0.8,
                      .maximum_spin_rate_radps = 1.0,
                      .maximum_forward_acceleration_mps2 = 1.0,
                      .maximum_braking_deceleration_mps2 = 1.0,
                      .maximum_yaw_acceleration_radps2 = 1.0,
                      .maximum_lateral_acceleration_mps2 = 1.0,
                      .maximum_drive_curvature_per_m = 1.0,
                      .maximum_slope_rad = 0.5,
                      .minimum_clearance_m = 0.05,
                  },
              .motion_primitives =
                  {
                      Primitive(
                          "forward",
                          WheelMotionPrimitive::Kind::kDriveForwardLine,
                          {{1.0, 0.0, 0.0}, 0.0}),
                      Primitive(
                          "reverse",
                          WheelMotionPrimitive::Kind::kDriveReverseLine,
                          {{-1.0, 0.0, 0.0}, 0.0}),
                      Primitive("spin-cw",
                                WheelMotionPrimitive::Kind::kSpinCw,
                                {{0.0, 0.0, 0.0}, -0.5}),
                      Primitive("spin-ccw",
                                WheelMotionPrimitive::Kind::kSpinCcw,
                                {{0.0, 0.0, 0.0}, 0.5}),
                      Primitive(
                          "switch",
                          WheelMotionPrimitive::Kind::kStopAndSwitch,
                          {{0.0, 0.0, 0.0}, 0.0}),
                  },
          },
  };
}

[[nodiscard]] SafeProjection Projection() {
  SafeProjection projection;
  projection.geometry = {
      .width = 24U,
      .height = 24U,
      .resolution_m = 1.0,
      .origin_m = {0.0, 0.0},
      .frame_id = "map",
  };
  const std::size_t count = projection.geometry.CellCount();
  projection.known_mask.assign(count, 1U);
  projection.hard_feasible_mask.assign(count, 1U);
  projection.esdf_clearance_m.assign(count, 5.0F);
  projection.resolved_energy.assign(count, 2.0F);
  projection.resolved_nonfatal_risk.assign(count, 0.25F);
  projection.conservative_speed_limit_mps.assign(count, 1.0F);
  return projection;
}

struct AdapterFixture final {
  SafetyCapabilityProfile profile{Capability()};
  WheelCapabilityView capability{
      std::get<WheelCapabilityView>(
          WheelCapabilityView::Create(profile))};
  WheelPrimitiveCatalog catalog{
      std::get<WheelPrimitiveCatalog>(
          WheelPrimitiveCatalog::Create(profile))};
  SafeProjection projection{Projection()};
  std::vector<TerminalCandidate> terminals{
      TerminalCandidate{
          .stable_id = "goal",
          .cell = {12, 10},
          .position_m = {12.5, 10.5, 0.0},
          .requires_zero_speed = true,
          .clearance_m = 5.0,
      }};
  WheelLatticeAdapter adapter{
      catalog,
      projection,
      capability,
      terminals,
      GridConfig{
          .xy_resolution_m = 1.0,
          .yaw_bin_count = 16U,
          .maximum_terminal_candidates = 8U,
      }};
};

TEST(WheelLatticeTest, StateKeyContainsPoseAndModeOnly) {
  AdapterFixture fixture;
  const WheelLatticeState state{
      3, 7, 5, WheelMotionMode::kReverse};

  const auto key = fixture.adapter.Key(state);

  EXPECT_EQ(
      key,
      EncodeWheelStateKey(3, 7, 5, WheelMotionMode::kReverse));
  EXPECT_NE(
      key,
      EncodeWheelStateKey(3, 7, 5, WheelMotionMode::kForward));
}

TEST(WheelLatticeTest, ExpandsReverseAndBothSpinHubEdges) {
  AdapterFixture fixture;
  const auto edges = fixture.adapter.Expand(
      WheelLatticeState{10, 10, 0, WheelMotionMode::kStart});

  const auto contains_target = [&edges](WheelMotionMode mode) {
    return std::ranges::any_of(
        edges, [mode](const auto& transition) {
          return transition.successor.motion_mode == mode;
        });
  };
  EXPECT_TRUE(contains_target(WheelMotionMode::kReverse));
  EXPECT_TRUE(
      contains_target(WheelMotionMode::kSpinClockwise));
  EXPECT_TRUE(
      contains_target(WheelMotionMode::kSpinCounterClockwise));
}

TEST(WheelLatticeTest,
     ForwardToReverseUsesTwoStationaryHubTransitions) {
  AdapterFixture fixture;
  const WheelLatticeState forward{
      10, 10, 0, WheelMotionMode::kForward};
  const auto stop_edges = fixture.adapter.Expand(forward);
  const auto stop = std::ranges::find_if(
      stop_edges, [](const auto& transition) {
        return transition.successor.motion_mode ==
               WheelMotionMode::kStart;
      });
  ASSERT_NE(stop, stop_edges.end());
  EXPECT_EQ(stop->successor.ix, forward.ix);
  EXPECT_EQ(stop->successor.iy, forward.iy);
  EXPECT_EQ(stop->edge.transition_time.value, 500ms);

  const auto restart_edges =
      fixture.adapter.Expand(stop->successor);
  const auto restart = std::ranges::find_if(
      restart_edges, [](const auto& transition) {
        return transition.successor.motion_mode ==
               WheelMotionMode::kReverse;
      });
  ASSERT_NE(restart, restart_edges.end());
  EXPECT_EQ(restart->successor.ix, forward.ix);
  EXPECT_EQ(restart->successor.iy, forward.iy);
  EXPECT_EQ(restart->edge.transition_time.value, 500ms);
}

TEST(WheelLatticeTest, UnknownIntermediateCellRejectsDrive) {
  AdapterFixture fixture;
  const std::size_t unknown_index =
      10U * fixture.projection.geometry.width + 11U;
  fixture.projection.known_mask[unknown_index] = 0U;
  fixture.projection.hard_feasible_mask[unknown_index] = 0U;
  const auto edges = fixture.adapter.Expand(
      WheelLatticeState{10, 10, 0, WheelMotionMode::kForward});
  const auto drive = std::ranges::find_if(
      edges, [](const auto& transition) {
        return transition.edge.primitive_kind ==
               WheelPrimitiveKind::kDriveLine;
      });
  ASSERT_NE(drive, edges.end());

  EXPECT_FALSE(fixture.adapter.HardFeasible(*drive));
}

TEST(WheelLatticeTest,
     TransitionSecondaryCostUsesResolvedProjectionCost) {
  AdapterFixture fixture;
  const auto edges = fixture.adapter.Expand(
      WheelLatticeState{10, 10, 0, WheelMotionMode::kForward});
  const auto drive = std::ranges::find_if(
      edges, [](const auto& transition) {
        return transition.edge.primitive_kind ==
               WheelPrimitiveKind::kDriveLine;
      });
  ASSERT_NE(drive, edges.end());

  const auto costs = fixture.adapter.SecondaryCosts(*drive);
  EXPECT_DOUBLE_EQ(costs.energy, 2.0);
  EXPECT_DOUBLE_EQ(costs.risk, 0.25);
  EXPECT_DOUBLE_EQ(costs.smoothness, 0.0);
}

TEST(WheelLatticeTest, TimeHeuristicIsFiniteTranslationLowerBound) {
  AdapterFixture fixture;
  const SearchProblem<WheelLatticeState> problem{
      .start =
          WheelLatticeState{10, 10, 0, WheelMotionMode::kStart},
      .limits = {},
  };

  const auto heuristic =
      fixture.adapter.AdmissibleTimeHeuristic(problem.start, problem);

  EXPECT_GE(heuristic.value, 1s);
  EXPECT_LE(heuristic.value, 3s);
}

}  // namespace
}  // namespace lunar::planning::v3
