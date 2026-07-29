#include <cmath>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/legged/legged_terrain.hpp"

namespace lunar::planning::v3 {
namespace {

LeggedCapabilityView Capability() {
  return LeggedCapabilityView{
      .content_ref =
          ContentRef{"legged", 1U, std::string(64U, 'a')},
      .frame_id = "map",
      .reference_point_id = "base_link",
      .collision_envelope = {},
      .terrain_thresholds =
          LeggedTerrainThresholds{
              .maximum_slope_rad = 0.5,
              .maximum_roughness_m = 0.1,
              .maximum_step_height_m = 0.2,
              .maximum_gap_width_m = 0.25,
              .minimum_confidence = 0.8,
              .minimum_body_clearance_m = 0.15,
              .minimum_body_height_m = 0.35,
              .maximum_body_height_m = 0.65,
          },
      .velocity_envelope = {},
      .certified_state_error_bounds = {},
      .preferred_body_height_m = 0.5,
      .maximum_linear_acceleration_mps2 = 0.4,
      .maximum_yaw_acceleration_radps2 = 0.6,
  };
}

LeggedTerrainCellSample SafeSample() {
  return LeggedTerrainCellSample{
      .known = true,
      .hard_obstacle = false,
      .confidence = 0.9,
      .elevation_m = 1.0,
      .normal = {0.0, 0.0, 1.0},
      .roughness_m = 0.05,
      .maximum_neighbor_step_m = 0.1,
      .unsupported_gap_width_m = 0.0,
      .body_clearance_m = 0.2,
      .analytic_time_cost_s = 1.0,
      .resolved_energy = 2.0,
      .resolved_nonfatal_risk = 0.1,
  };
}

TEST(LeggedTerrainTest, UnknownOrLowConfidenceIsHardInfeasible) {
  auto unknown = SafeSample();
  unknown.known = false;
  auto low_confidence = SafeSample();
  low_confidence.confidence = 0.79;

  EXPECT_FALSE(
      EvaluateLeggedTerrainSample(unknown, Capability()).hard_feasible);
  EXPECT_FALSE(EvaluateLeggedTerrainSample(low_confidence, Capability())
                   .hard_feasible);
}

TEST(LeggedTerrainTest, ExactHardThresholdsPass) {
  auto sample = SafeSample();
  const auto capability = Capability();
  sample.normal = {std::sin(0.5), 0.0, std::cos(0.5)};
  sample.roughness_m = 0.1;
  sample.maximum_neighbor_step_m = 0.2;
  sample.unsupported_gap_width_m = 0.25;
  sample.body_clearance_m = 0.15;

  const auto evaluation =
      EvaluateLeggedTerrainSample(sample, capability);

  EXPECT_TRUE(evaluation.hard_feasible);
  EXPECT_DOUBLE_EQ(evaluation.body_height_interval.min_m, 1.35);
  EXPECT_DOUBLE_EQ(evaluation.body_height_interval.max_m, 1.65);
}

TEST(LeggedTerrainTest, OneUlpBeyondEachHardThresholdFails) {
  const auto capability = Capability();
  const auto above = [](double value) {
    return std::nextafter(value,
                          std::numeric_limits<double>::infinity());
  };
  const auto below = [](double value) {
    return std::nextafter(
        value, -std::numeric_limits<double>::infinity());
  };

  auto rough = SafeSample();
  rough.roughness_m = above(0.1);
  EXPECT_FALSE(
      EvaluateLeggedTerrainSample(rough, capability).hard_feasible);

  auto step = SafeSample();
  step.maximum_neighbor_step_m = above(0.2);
  EXPECT_FALSE(
      EvaluateLeggedTerrainSample(step, capability).hard_feasible);

  auto gap = SafeSample();
  gap.unsupported_gap_width_m = above(0.25);
  EXPECT_FALSE(
      EvaluateLeggedTerrainSample(gap, capability).hard_feasible);

  auto clearance = SafeSample();
  clearance.body_clearance_m = below(0.15);
  EXPECT_FALSE(
      EvaluateLeggedTerrainSample(clearance, capability).hard_feasible);
}

TEST(LeggedTerrainTest, HardFailureDoesNotPublishSoftCosts) {
  auto sample = SafeSample();
  sample.hard_obstacle = true;
  sample.resolved_energy = 9.0;
  sample.resolved_nonfatal_risk = 8.0;

  const auto evaluation =
      EvaluateLeggedTerrainSample(sample, Capability());

  EXPECT_FALSE(evaluation.hard_feasible);
  EXPECT_DOUBLE_EQ(evaluation.secondary_costs.energy, 0.0);
  EXPECT_DOUBLE_EQ(evaluation.secondary_costs.risk, 0.0);
}

}  // namespace
}  // namespace lunar::planning::v3
