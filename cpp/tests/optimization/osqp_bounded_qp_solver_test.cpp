#include <cmath>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/optimization/bounded_qp_solver.hpp"

#if defined(LPP_V3_HAS_OSQP)

namespace lunar::planning::v3 {
namespace {

SparseQpProblem MakeExactConvexUnitBoxQp() {
  SparseQpProblem problem;
  problem.hessian_upper_triangle.resize(1, 1);
  problem.hessian_upper_triangle.insert(0, 0) = 2.0;
  problem.hessian_upper_triangle.makeCompressed();
  problem.gradient = Eigen::VectorXd::Constant(1, -6.0);
  problem.constraints.resize(1, 1);
  problem.constraints.insert(0, 0) = 1.0;
  problem.constraints.makeCompressed();
  problem.lower_bounds = Eigen::VectorXd::Constant(1, 0.0);
  problem.upper_bounds = Eigen::VectorXd::Constant(1, 2.0);
  return problem;
}

BoundedQpSettings MakeOsqpSettings() {
  return BoundedQpSettings{
      .max_iterations = 10000U,
      .absolute_tolerance = 1.0e-9,
      .relative_tolerance = 1.0e-9,
      .polish = true,
  };
}

TEST(OsqpBoundedQpSolver, SolvesExactConvexUnitBoxProblem) {
  OsqpBoundedQpSolver solver;

  const auto result = solver.Solve(
      MakeExactConvexUnitBoxQp(), MakeOsqpSettings());

  ASSERT_TRUE(IsOk(result));
  const auto& solution = std::get<QpSolution>(result);
  EXPECT_EQ(solution.termination, QpTermination::kSolved);
  ASSERT_EQ(solution.primal.size(), 1);
  EXPECT_NEAR(solution.primal[0], 2.0, 1.0e-7);
  EXPECT_NEAR(solution.objective, -8.0, 1.0e-7);
  EXPECT_LE(solution.primal_residual, 1.0e-7);
  EXPECT_LE(solution.dual_residual, 1.0e-7);
  EXPECT_GT(solution.iterations, 0U);
}

TEST(OsqpBoundedQpSolver, MapsPrimalInfeasibility) {
  auto problem = MakeExactConvexUnitBoxQp();
  problem.constraints.resize(2, 1);
  problem.constraints.setZero();
  problem.constraints.insert(0, 0) = 1.0;
  problem.constraints.insert(1, 0) = 1.0;
  problem.constraints.makeCompressed();
  problem.lower_bounds.resize(2);
  problem.lower_bounds << 1.0, -10.0;
  problem.upper_bounds.resize(2);
  problem.upper_bounds << 10.0, 0.0;
  OsqpBoundedQpSolver solver;

  const auto result = solver.Solve(problem, MakeOsqpSettings());

  ASSERT_TRUE(IsOk(result));
  const auto& solution = std::get<QpSolution>(result);
  EXPECT_EQ(solution.termination, QpTermination::kPrimalInfeasible);
  EXPECT_GT(solution.iterations, 0U);
}

TEST(OsqpBoundedQpSolver, PreservesForcedOneIterationTermination) {
  auto settings = MakeOsqpSettings();
  settings.max_iterations = 1U;
  settings.absolute_tolerance = 1.0e-12;
  settings.relative_tolerance = 1.0e-12;
  settings.polish = false;
  OsqpBoundedQpSolver solver;

  const auto result = solver.Solve(
      MakeExactConvexUnitBoxQp(), settings);

  ASSERT_TRUE(IsOk(result));
  const auto& solution = std::get<QpSolution>(result);
  EXPECT_EQ(solution.termination, QpTermination::kMaxIterations);
  EXPECT_EQ(solution.iterations, 1U);
  ASSERT_EQ(solution.primal.size(), 1);
  EXPECT_TRUE(std::isfinite(solution.primal[0]));
}

}  // namespace
}  // namespace lunar::planning::v3

#else

TEST(OsqpBoundedQpSolver, BackendIsCompileTimeGuardedWhenDisabled) {
  SUCCEED();
}

#endif
