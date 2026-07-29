#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/wheel/wheel_spline_optimizer.hpp"

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] ContentRef Ref(std::string id, char digit) {
  return {std::move(id), 1U, std::string(64U, digit)};
}

[[nodiscard]] SafetyCapabilityProfile Profile() {
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
              .motion_model_ref = Ref("motion", 'b'),
              .analytic_cost_model_ref = Ref("cost", 'c'),
              .hard_limits =
                  {
                      .maximum_forward_speed_mps = 1.0,
                      .maximum_reverse_speed_mps = 0.8,
                      .maximum_spin_rate_radps = 1.0,
                      .maximum_forward_acceleration_mps2 = 1.0,
                      .maximum_braking_deceleration_mps2 = 1.0,
                      .maximum_yaw_acceleration_radps2 = 1.0,
                      .maximum_lateral_acceleration_mps2 = 1.0,
                      .maximum_drive_curvature_per_m = 2.0,
                      .maximum_slope_rad = 0.5,
                      .minimum_clearance_m = 0.0,
                  },
          },
  };
}

[[nodiscard]] WheelDiscreteSegment Segment() {
  const WheelLatticeState source{
      1, 1, 0, WheelMotionMode::kForward};
  const WheelLatticeState target{
      4, 1, 0, WheelMotionMode::kForward};
  return {
      .mode = WheelMotionMode::kForward,
      .edges =
          {
              {
                  .source = source,
                  .target = target,
                  .primitive_id = "drive",
                  .capability_primitive_id = "drive",
                  .primitive_kind =
                      WheelPrimitiveKind::kDriveLine,
                  .source_pose = {{1.0, 1.0, 0.0}, 0.0},
                  .target_pose = {{4.0, 1.0, 0.0}, 0.0},
                  .transition_time = DurationNanoseconds{5s},
                  .validation_ref = Ref("sweep", 'd'),
              },
          },
      .expected_time = DurationNanoseconds{5s},
  };
}

[[nodiscard]] CorridorResult Corridor() {
  return {
      .status = CorridorStatus::kCertified,
      .fallback = CorridorFallback::kNone,
      .cells =
          {
              {
                  .stable_cell_id = "corridor",
                  .centerline_s_begin = 0.0,
                  .centerline_s_end = 3.0,
                  .half_planes =
                      {
                          {{-1.0, 0.0}, 10.0},
                          {{0.0, -1.0}, 10.0},
                          {{0.0, 1.0}, 10.0},
                          {{1.0, 0.0}, 10.0},
                      },
              },
          },
      .reason_code = "certified",
  };
}

[[nodiscard]] WheelSplineConfig Config() {
  return {
      .maximum_control_points = 16U,
      .maximum_scp_iterations = 3U,
      .maximum_qp_iterations = 50U,
      .path_deviation_weight = 1.0,
      .second_difference_weight = 0.1,
      .initial_trust_region_m = 2.0,
      .minimum_trust_region_m = 0.01,
      .constraint_tolerance = 1.0e-7,
      .absolute_qp_tolerance = 1.0e-7,
      .relative_qp_tolerance = 1.0e-7,
      .maximum_curvature_per_m = 2.0,
      .time_equivalence_tolerance = DurationNanoseconds{1s},
  };
}

class GradientMinimumSolver final : public BoundedQpSolver {
 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem& problem,
      const BoundedQpSettings&) const override {
    Eigen::VectorXd primal(problem.gradient.size());
    for (Eigen::Index index = 0; index < primal.size(); ++index) {
      const double diagonal =
          problem.hessian_upper_triangle.coeff(index, index);
      primal[index] =
          diagonal > 0.0 ? -problem.gradient[index] / diagonal
                         : 0.0;
    }
    return QpSolution{
        .termination = QpTermination::kSolved,
        .primal = std::move(primal),
        .objective = 0.0,
        .primal_residual = 0.0,
        .dual_residual = 0.0,
        .iterations = 1U,
    };
  }
};

class LongDetourSolver final : public BoundedQpSolver {
 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem& problem,
      const BoundedQpSettings&) const override {
    Eigen::VectorXd primal =
        Eigen::VectorXd::Zero(problem.gradient.size());
    const Eigen::Index point_count = primal.size() / 2;
    for (Eigen::Index index = 0; index < point_count; ++index) {
      const double fraction =
          static_cast<double>(index) /
          static_cast<double>(point_count - 1);
      primal[2 * index] = 1.0 + 3.0 * fraction;
      primal[2 * index + 1] =
          (index == 0 || index + 1 == point_count) ? 1.0 : 2.9;
    }
    return QpSolution{
        .termination = QpTermination::kSolved,
        .primal = std::move(primal),
        .objective = 0.0,
        .primal_residual = 0.0,
        .dual_residual = 0.0,
        .iterations = 1U,
    };
  }
};

class InfeasibleSolver final : public BoundedQpSolver {
 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem& problem,
      const BoundedQpSettings&) const override {
    return QpSolution{
        .termination = QpTermination::kPrimalInfeasible,
        .primal = Eigen::VectorXd::Zero(problem.gradient.size()),
    };
  }
};

TEST(WheelSplineOptimizerTest, NeverMovesCommittedControlPoints) {
  const auto segment = Segment();
  const auto corridor = Corridor();
  const auto capability = std::get<WheelCapabilityView>(
      WheelCapabilityView::Create(Profile()));
  const std::vector frozen{
      FrozenControlPoint{
          .index = 0U,
          .value = {{1.2, 1.1, 0.0}, 0.0},
      },
      FrozenControlPoint{
          .index = 1U,
          .value = {{2.0, 1.0, 0.0}, 0.0},
      },
  };
  GradientMinimumSolver solver;

  const auto result = OptimizeWheelSpline(
      {
          .discrete_segment = segment,
          .corridor = corridor,
          .capability = capability,
          .config = Config(),
          .committed_points = frozen,
      },
      solver);

  ASSERT_TRUE(result.spline.has_value());
  EXPECT_DOUBLE_EQ(
      result.spline->control_points[0].position_m.x,
      frozen[0].value.position_m.x);
  EXPECT_DOUBLE_EQ(
      result.spline->control_points[0].position_m.y,
      frozen[0].value.position_m.y);
  EXPECT_DOUBLE_EQ(
      result.spline->control_points[1].position_m.x,
      frozen[1].value.position_m.x);
  EXPECT_EQ(result.scp_iterations, 3U);
}

TEST(WheelSplineOptimizerTest,
     RejectsSlowerThanDiscreteTolerance) {
  const auto segment = Segment();
  const auto corridor = Corridor();
  const auto capability = std::get<WheelCapabilityView>(
      WheelCapabilityView::Create(Profile()));
  auto config = Config();
  config.time_equivalence_tolerance =
      DurationNanoseconds{100ms};
  LongDetourSolver solver;

  const auto result = OptimizeWheelSpline(
      {
          .discrete_segment = segment,
          .corridor = corridor,
          .capability = capability,
          .config = config,
          .committed_points = {},
      },
      solver);

  EXPECT_FALSE(result.spline.has_value());
  EXPECT_EQ(result.termination,
            OptimizationTermination::kTimeToleranceExceeded);
}

TEST(WheelSplineOptimizerTest,
     QpInfeasibilityRejectsWholeSpline) {
  const auto segment = Segment();
  const auto corridor = Corridor();
  const auto capability = std::get<WheelCapabilityView>(
      WheelCapabilityView::Create(Profile()));
  InfeasibleSolver solver;

  const auto result = OptimizeWheelSpline(
      {
          .discrete_segment = segment,
          .corridor = corridor,
          .capability = capability,
          .config = Config(),
          .committed_points = {},
      },
      solver);

  EXPECT_FALSE(result.spline.has_value());
  EXPECT_EQ(result.termination,
            OptimizationTermination::kQpInfeasible);
}

}  // namespace
}  // namespace lunar::planning::v3
