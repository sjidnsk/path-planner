#include <chrono>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <gtest/gtest.h>

#include "lunar_path_planner/v3/legged/legged_spline_optimizer.hpp"

namespace lunar::planning::v3 {
namespace {

class DenseUnconstrainedQpSolver final : public BoundedQpSolver {
 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem& problem,
      const BoundedQpSettings&) const override {
    Eigen::MatrixXd hessian =
        Eigen::MatrixXd(problem.hessian_upper_triangle);
    hessian =
        hessian.selfadjointView<Eigen::Upper>();
    const Eigen::VectorXd primal =
        hessian.ldlt().solve(-problem.gradient);
    return QpSolution{
        .termination = QpTermination::kSolved,
        .primal = primal,
        .objective = 0.0,
        .primal_residual = 0.0,
        .dual_residual = 0.0,
        .iterations = 1U,
    };
  }
};

class FailingQpSolver final : public BoundedQpSolver {
 private:
  [[nodiscard]] Result<QpSolution> DoSolve(
      const SparseQpProblem& problem,
      const BoundedQpSettings&) const override {
    return QpSolution{
        .termination = QpTermination::kPrimalInfeasible,
        .primal = Eigen::VectorXd::Zero(problem.gradient.size()),
        .objective = 0.0,
        .primal_residual = 1.0,
        .dual_residual = 0.0,
        .iterations = 1U,
    };
  }
};

ConvexCorridorCell BoxCell() {
  return ConvexCorridorCell{
      .stable_cell_id = "box",
      .centerline_s_begin = 0.0,
      .centerline_s_end = 1.0,
      .half_planes =
          {
              {{-1.0, 0.0}, 0.5},
              {{0.0, -1.0}, 0.5},
              {{0.0, 1.0}, 2.0},
              {{1.0, 0.0}, 1.5},
          },
  };
}

LeggedCorridor Corridor() {
  return LeggedCorridor{
      .sections =
          {
              LeggedCorridorSection{
                  .xy = BoxCell(),
                  .z = {0.4, 0.6},
                  .yaw = {-0.5, 0.5},
                  .terrain_normal = {},
                  .cartesian_product_certified = true,
              },
          },
  };
}

LeggedDiscretePlan LateralPlan() {
  LeggedDiscretePlan plan;
  plan.stable_candidate_id = "lateral";
  plan.grid =
      GridConfig{
          .xy_resolution_m = 1.0,
          .yaw_bin_count = 8U,
      };
  plan.grid_origin_m = {0.0, 0.0};
  plan.states = {
      {0, 0, 0, {0.4, 0.6}},
      {0, 1, 0, {0.4, 0.6}},
  };
  plan.estimated_execution_time =
      DurationNanoseconds{std::chrono::seconds{5}};
  return plan;
}

LeggedSplineConfig Config() {
  return LeggedSplineConfig{
      .smoothing =
          SmoothingConfig{
              .maximum_scp_iterations = 2U,
              .maximum_trust_region_reductions = 1U,
              .initial_trust_region_m = 0.5,
              .minimum_trust_region_m = 0.01,
              .constraint_tolerance = 1.0e-7,
              .maximum_time_increase =
                  DurationNanoseconds{
                      std::chrono::milliseconds{100}},
          },
      .qp_settings =
          BoundedQpSettings{
              .max_iterations = 50U,
              .absolute_tolerance = 1.0e-7,
              .relative_tolerance = 1.0e-7,
              .polish = false,
          },
      .validation_sample_count = 17U,
  };
}

TEST(LeggedSplineOptimizerTest,
     PreservesIndependentYawForLateralMotion) {
  const auto plan = LateralPlan();
  const auto corridor = Corridor();
  DenseUnconstrainedQpSolver solver;

  const auto result = OptimizeLeggedBodySpline(
      LeggedSplineRequest{
          .discrete_plan = plan,
          .corridor = corridor,
          .config = Config(),
          .preferred_body_height_m = 0.5,
          .committed_points = {},
      },
      solver);

  ASSERT_TRUE(result.body_spline.has_value());
  EXPECT_NEAR(EvaluateLeggedSpline(*result.body_spline, 0.5).yaw_rad,
              0.0, 1.0e-8);
  EXPECT_NEAR(
      EvaluateLeggedSplineTangentYaw(*result.body_spline, 0.5),
      std::numbers::pi / 2.0, 1.0e-3);
}

TEST(LeggedSplineOptimizerTest, PreservesCommittedControlPoint) {
  const auto plan = LateralPlan();
  const auto corridor = Corridor();
  const LeggedFrozenControlPoint committed{
      .control_point_index = 0U,
      .pose =
          PoseXyzYaw{
              .position_m = {0.5, 0.5, 0.5},
              .yaw_rad = 0.0,
          },
  };
  DenseUnconstrainedQpSolver solver;

  const auto result = OptimizeLeggedBodySpline(
      LeggedSplineRequest{
          .discrete_plan = plan,
          .corridor = corridor,
          .config = Config(),
          .preferred_body_height_m = 0.5,
          .committed_points =
              std::span<const LeggedFrozenControlPoint>{
                  &committed, 1U},
      },
      solver);

  ASSERT_TRUE(result.body_spline.has_value());
  const auto& actual = result.body_spline->control_points.front();
  EXPECT_DOUBLE_EQ(actual.position_m.x, committed.pose.position_m.x);
  EXPECT_DOUBLE_EQ(actual.position_m.y, committed.pose.position_m.y);
  EXPECT_DOUBLE_EQ(actual.position_m.z, committed.pose.position_m.z);
  EXPECT_DOUBLE_EQ(actual.yaw_rad, committed.pose.yaw_rad);
}

TEST(LeggedSplineOptimizerTest, RejectsWholeSplineWhenQpFails) {
  const auto plan = LateralPlan();
  const auto corridor = Corridor();
  FailingQpSolver solver;

  const auto result = OptimizeLeggedBodySpline(
      LeggedSplineRequest{
          .discrete_plan = plan,
          .corridor = corridor,
          .config = Config(),
          .preferred_body_height_m = 0.5,
          .committed_points = {},
      },
      solver);

  EXPECT_FALSE(result.body_spline.has_value());
  EXPECT_EQ(result.termination,
            LeggedOptimizationTermination::kQpFailure);
}

}  // namespace
}  // namespace lunar::planning::v3
