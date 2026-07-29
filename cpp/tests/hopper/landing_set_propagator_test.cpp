#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/landing_set_propagator.hpp"

namespace lunar::planning::v3 {
namespace {

LandingSetPropagationInput MakeInput(
    const HopperCapabilityView& capability,
    const double error_scale) {
  const AimPointCandidate aim{
      .aim_point_id = "aim",
      .region_id = "target",
      .frame_id = "map",
      .position_uv = {0.0, 0.0},
      .position_m = {2.0, 2.0, 0.5},
  };
  const auto arc = std::get<NominalBallisticArc>(
      make_nominal_ballistic_arc(
          Eigen::Vector3d{1.0, 1.0, 0.5}, "map", aim,
          gravity_model_view(capability.gravity_model), 2.0));
  const AxisAlignedBox3 error = hopper_test::Box(
      {}, {0.01 * error_scale, 0.01 * error_scale,
           0.01 * error_scale});
  return {
      .arc = arc,
      .landing_plane =
          {
              .origin_m = {0.0, 0.0, 0.0},
              .normal = {0.0, 0.0, 1.0},
              .basis_u = {1.0, 0.0, 0.0},
              .basis_v = {0.0, 1.0, 0.0},
          },
      .initial_position_error_m = error,
      .initial_velocity_error_mps = error,
      .gravity_error_mps2 = error,
      .launch_execution_velocity_error_mps = error,
      .landing_plane_error =
          {
              .origin_error_m = error,
              .normal_error = {0.0},
              .residual_error_m = {0.0, 0.0},
          },
      .certified_landing_yaw_interval =
          {.start_rad = -0.2, .span_rad = 0.4},
      .source_error_model_ref =
          capability.error_model.content_ref,
  };
}

double PolygonWidth(const ConvexPolygonUv& polygon) {
  double minimum = polygon.vertices_uv.front().x;
  double maximum = minimum;
  for (const Vec2 vertex : polygon.vertices_uv) {
    minimum = std::min(minimum, vertex.x);
    maximum = std::max(maximum, vertex.x);
  }
  return maximum - minimum;
}

TEST(LandingSetPropagatorTest,
     LargerDeterministicErrorsCannotShrinkOuterSet) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const auto limits = std::get<HopperPlannerLimits>(
      bind_hopper_limits(hopper_test::ValidAlgorithm()));
  const LandingSetPropagator propagator{capability, limits};
  const auto small = propagator.propagate(
      MakeInput(capability, 0.5));
  const auto large = propagator.propagate(
      MakeInput(capability, 1.0));

  ASSERT_TRUE(small.footprint.has_value());
  ASSERT_TRUE(large.footprint.has_value());
  EXPECT_LE(
      PolygonWidth(
          small.footprint->convex_center_landing_polygon),
      PolygonWidth(
          large.footprint->convex_center_landing_polygon));
}

TEST(LandingSetPropagatorTest,
     RejectsImpactWithoutDownwardTransversality) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const auto limits = std::get<HopperPlannerLimits>(
      bind_hopper_limits(hopper_test::ValidAlgorithm()));
  auto input = MakeInput(capability, 0.0);
  input.arc.landing_velocity_mps = Eigen::Vector3d::Zero();
  const auto result =
      LandingSetPropagator{capability, limits}.propagate(input);
  EXPECT_FALSE(result.footprint.has_value());
  EXPECT_EQ(
      result.diagnostics.rejection_reason,
      "downward_transversality_not_proven");
}

}  // namespace
}  // namespace lunar::planning::v3
