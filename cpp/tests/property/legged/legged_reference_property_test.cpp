#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "../../integration/legged/legged_test_fixture.hpp"

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] PoseXyzYaw PathEnd(
    const GeometricPath& path) {
  return std::visit(
      [](const auto& concrete) {
        using Path = std::decay_t<decltype(concrete)>;
        if constexpr (
            std::is_same_v<Path, ClampedCubicBSplinePath>) {
          return EvaluateLeggedSpline(concrete, 1.0);
        } else {
          return concrete.primitives.back().end_pose;
        }
      },
      path);
}

[[nodiscard]] bool SamePose(const PoseXyzYaw& lhs,
                            const PoseXyzYaw& rhs) {
  return std::hypot(
             std::hypot(
                 lhs.position_m.x - rhs.position_m.x,
                 lhs.position_m.y - rhs.position_m.y),
             lhs.position_m.z - rhs.position_m.z) <=
             1.0e-9 &&
         std::abs(lhs.yaw_rad - rhs.yaw_rad) <= 1.0e-9;
}

[[nodiscard]] double PolynomialValue(
    const CubicPolynomialSegment& segment,
    double seconds) {
  return ((segment.coefficients[3] * seconds +
           segment.coefficients[2]) *
              seconds +
          segment.coefficients[1]) *
             seconds +
         segment.coefficients[0];
}

[[nodiscard]] double PolynomialRate(
    const CubicPolynomialSegment& segment,
    double seconds) {
  return segment.coefficients[1] +
         2.0 * segment.coefficients[2] * seconds +
         3.0 * segment.coefficients[3] * seconds * seconds;
}

void SetGeneratedTarget(legged_test::Fixture& fixture,
                        std::uint64_t seed) {
  constexpr std::array<Cell, 7U> kCells{
      Cell{2, 2}, Cell{3, 2}, Cell{4, 2}, Cell{1, 1},
      Cell{1, 3}, Cell{2, 1}, Cell{2, 3},
  };
  const Cell cell =
      kCells[seed % kCells.size()];
  const double yaw =
      static_cast<double>((seed / kCells.size()) % 4U) *
      (std::numbers::pi / 2.0);
  fixture.request.request_id =
      "generated-legged-" + std::to_string(seed);
  fixture.request.goal =
      GoalRegion{
          .goal_id = "generated-goal",
          .target =
              PointGoal{
                  .position_m =
                      {
                          static_cast<double>(cell.x) + 0.5,
                          static_cast<double>(cell.y) + 0.5,
                          0.0,
                      },
                  .position_tolerance_m = 0.25,
              },
          .optional_yaw_interval =
              CircularYawInterval{
                  .start_rad = yaw,
                  .span_rad = 0.01,
                  .closed = true,
              },
      };
  fixture.terminal.candidates.front() =
      TerminalCandidate{
          .stable_id =
              "generated-terminal-" + std::to_string(seed),
          .cell = cell,
          .position_m =
              {
                  static_cast<double>(cell.x) + 0.5,
                  static_cast<double>(cell.y) + 0.5,
                  0.0,
              },
          .yaw_interval =
              CircularYawInterval{
                  .start_rad = yaw,
                  .span_rad = 0.01,
                  .closed = true,
              },
          .requires_zero_speed = true,
          .clearance_m = 1.0,
      };
}

TEST(LeggedReferencePropertyTest,
     GeneratedReferencesStayWithinBodyContract) {
  auto fixture = legged_test::MakeFixture();
  const LeggedPlanner planner;
  std::size_t successful = 0U;
  for (std::uint64_t seed = 0U; seed < 1000U; ++seed) {
    SetGeneratedTarget(fixture, seed);
    const auto result =
        planner.Plan(fixture.request, fixture.terminal);
    ASSERT_TRUE(IsOk(result)) << "seed=" << seed;
    const auto& reference =
        std::get<LeggedBodyReference>(result);
    ++successful;

    EXPECT_EQ(reference.reference_point_id, "base_link");
    EXPECT_EQ(reference.feasibility_scope,
              "body_geometry_and_terrain_thresholds_only");
    EXPECT_FALSE(reference.footstep_feasibility_guaranteed);
    EXPECT_TRUE(std::holds_alternative<ValidatedPrimitiveChain>(
        reference.geometric_path));
    const auto& chain =
        std::get<ValidatedPrimitiveChain>(
            reference.geometric_path);
    ASSERT_FALSE(chain.primitives.empty());
    for (std::size_t index = 0U;
         index < chain.primitives.size(); ++index) {
      EXPECT_FALSE(
          chain.primitives[index].capability_primitive_id.empty());
      EXPECT_EQ(
          chain.primitives[index].validation_ref.content_hash.size(),
          64U);
      if (index > 0U) {
        EXPECT_TRUE(SamePose(
            chain.primitives[index - 1U].end_pose,
            chain.primitives[index].start_pose));
      }
    }
    EXPECT_TRUE(SamePose(
        PathEnd(reference.geometric_path),
        reference.safe_stop_anchor.pose));
    EXPECT_DOUBLE_EQ(
        reference.safe_stop_anchor.target_linear_velocity_mps,
        0.0);
    EXPECT_DOUBLE_EQ(
        reference.safe_stop_anchor.target_yaw_rate_radps,
        0.0);

    ASSERT_FALSE(reference.time_scaling.segments.empty());
    const auto& last =
        reference.time_scaling.segments.back();
    const double seconds =
        std::chrono::duration<double>(
            last.end_offset.value -
            last.start_offset.value)
            .count();
    EXPECT_NEAR(PolynomialValue(last, seconds), 1.0, 1.0e-8);
    EXPECT_NEAR(PolynomialRate(last, seconds), 0.0, 1.0e-8);

    const auto hash =
        CanonicalReferenceHash(
            PlatformReference{reference});
    ASSERT_TRUE(IsOk(hash));
    EXPECT_EQ(reference.reference_hash,
              std::get<Sha256Digest>(hash));
  }
  EXPECT_EQ(successful, 1000U);
}

TEST(LeggedReferencePropertyTest,
     RepeatedPlanningHasByteStableReferenceHash) {
  const auto fixture = legged_test::MakeFixture();
  const LeggedPlanner planner;
  std::string expected_id;
  Sha256Digest expected_hash;
  for (std::size_t repeat = 0U; repeat < 100U; ++repeat) {
    const auto result =
        planner.Plan(fixture.request, fixture.terminal);
    ASSERT_TRUE(IsOk(result));
    const auto& reference =
        std::get<LeggedBodyReference>(result);
    if (repeat == 0U) {
      expected_id = reference.reference_id;
      expected_hash = reference.reference_hash;
    }
    EXPECT_EQ(reference.reference_id, expected_id);
    EXPECT_EQ(reference.reference_hash, expected_hash);
  }
}

}  // namespace
}  // namespace lunar::planning::v3
