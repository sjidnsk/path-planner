#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/optimization/bounded_qp_solver.hpp"

namespace lunar::planning::v3 {
namespace {

SparseQpProblem MakeUnitBoxQp() {
  SparseQpProblem problem;
  problem.hessian_upper_triangle.resize(1, 1);
  problem.hessian_upper_triangle.insert(0, 0) = 2.0;
  problem.hessian_upper_triangle.makeCompressed();
  problem.gradient = Eigen::VectorXd::Constant(1, -2.0);
  problem.constraints.resize(1, 1);
  problem.constraints.insert(0, 0) = 1.0;
  problem.constraints.makeCompressed();
  problem.lower_bounds = Eigen::VectorXd::Constant(1, 0.0);
  problem.upper_bounds = Eigen::VectorXd::Constant(1, 2.0);
  return problem;
}

BoundedQpSettings MakeQpSettings() {
  return BoundedQpSettings{
      .max_iterations = 200U,
      .absolute_tolerance = 1.0e-8,
      .relative_tolerance = 1.0e-8,
      .polish = true,
  };
}

class DeterministicFakeQpSolver final : public BoundedQpSolver {
 public:
  explicit DeterministicFakeQpSolver(
      Result<QpSolution> result = QpSolution{
          .termination = QpTermination::kSolved,
          .primal = Eigen::VectorXd::Constant(1, 1.0),
          .objective = -1.0,
          .primal_residual = 0.0,
          .dual_residual = 0.0,
          .iterations = 3U,
      })
      : result_(std::move(result)) {}

  [[nodiscard]] std::size_t dispatch_count() const noexcept {
    return dispatch_count_;
  }

 private:
  Result<QpSolution> DoSolve(
      const SparseQpProblem&,
      const BoundedQpSettings&) const override {
    ++dispatch_count_;
    return result_;
  }

  Result<QpSolution> result_;
  mutable std::size_t dispatch_count_{};
};

void ExpectInvalidArgument(
    const Result<QpSolution>& result,
    const std::string& expected_field_path) {
  ASSERT_FALSE(IsOk(result));
  const auto& error = std::get<Error>(result);
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.field_path, expected_field_path);
  EXPECT_FALSE(error.message.empty());
}

TEST(BoundedQpSolver, DispatchesValidProblemAndPreservesBackendEvidence) {
  DeterministicFakeQpSolver solver;

  const auto first = solver.Solve(MakeUnitBoxQp(), MakeQpSettings());
  const auto second = solver.Solve(MakeUnitBoxQp(), MakeQpSettings());

  ASSERT_TRUE(IsOk(first));
  ASSERT_TRUE(IsOk(second));
  EXPECT_EQ(solver.dispatch_count(), 2U);
  const auto& first_solution = std::get<QpSolution>(first);
  const auto& second_solution = std::get<QpSolution>(second);
  EXPECT_EQ(first_solution.termination, QpTermination::kSolved);
  ASSERT_EQ(first_solution.primal.size(), 1);
  EXPECT_DOUBLE_EQ(first_solution.primal[0], 1.0);
  EXPECT_DOUBLE_EQ(first_solution.objective, -1.0);
  EXPECT_DOUBLE_EQ(first_solution.primal_residual, 0.0);
  EXPECT_DOUBLE_EQ(first_solution.dual_residual, 0.0);
  EXPECT_EQ(first_solution.iterations, 3U);
  EXPECT_EQ(first_solution.primal, second_solution.primal);
  EXPECT_DOUBLE_EQ(first_solution.objective, second_solution.objective);
  EXPECT_EQ(first_solution.iterations, second_solution.iterations);
}

TEST(BoundedQpSolver, RejectsEmptyDecisionVectorBeforeBackendDispatch) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.hessian_upper_triangle.resize(0, 0);
  problem.gradient.resize(0);
  problem.constraints.resize(0, 0);
  problem.lower_bounds.resize(0);
  problem.upper_bounds.resize(0);

  const auto result = solver.Solve(problem, MakeQpSettings());

  ExpectInvalidArgument(result, "problem.gradient");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsNonSquareHessianBeforeBackendDispatch) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.hessian_upper_triangle.resize(1, 2);

  const auto result = solver.Solve(problem, MakeQpSettings());

  ExpectInvalidArgument(result, "problem.hessian_upper_triangle");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsHessianGradientDimensionMismatch) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.gradient.conservativeResize(2);

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.gradient");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsConstraintColumnDimensionMismatch) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.constraints.conservativeResize(1, 2);

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.constraints");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsLowerBoundDimensionMismatch) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.lower_bounds.conservativeResize(2);

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.lower_bounds");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsUpperBoundDimensionMismatch) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.upper_bounds.conservativeResize(2);

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.upper_bounds");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsNonFiniteHessianCoefficient) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.hessian_upper_triangle.coeffRef(0, 0) =
      std::numeric_limits<double>::quiet_NaN();

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()),
      "problem.hessian_upper_triangle");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsNonFiniteGradientCoefficient) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.gradient[0] = std::numeric_limits<double>::infinity();

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.gradient");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsNonFiniteConstraintCoefficient) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.constraints.coeffRef(0, 0) =
      -std::numeric_limits<double>::infinity();

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.constraints");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsNonFiniteLowerBound) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.lower_bounds[0] = -std::numeric_limits<double>::infinity();

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.lower_bounds");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsNonFiniteUpperBound) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.upper_bounds[0] = std::numeric_limits<double>::infinity();

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.upper_bounds");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsReversedBoundsBeforeBackendDispatch) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.lower_bounds[0] = 2.0;
  problem.upper_bounds[0] = 1.0;

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()), "problem.bounds[0]");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsStoredLowerTriangularHessianCoefficient) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.hessian_upper_triangle.resize(2, 2);
  problem.hessian_upper_triangle.setZero();
  problem.hessian_upper_triangle.insert(0, 0) = 1.0;
  problem.hessian_upper_triangle.insert(1, 0) = 0.25;
  problem.hessian_upper_triangle.insert(1, 1) = 1.0;
  problem.hessian_upper_triangle.makeCompressed();
  problem.gradient = Eigen::VectorXd::Zero(2);
  problem.constraints.resize(1, 2);
  problem.constraints.setZero();

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()),
      "problem.hessian_upper_triangle");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsZeroMaximumIterations) {
  DeterministicFakeQpSolver solver;
  auto settings = MakeQpSettings();
  settings.max_iterations = 0U;

  ExpectInvalidArgument(
      solver.Solve(MakeUnitBoxQp(), settings), "settings.max_iterations");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, RejectsMaximumIterationsOutsideSignedRange) {
  if constexpr (
      std::numeric_limits<std::size_t>::max() >
      static_cast<std::size_t>(
          std::numeric_limits<std::int64_t>::max())) {
    DeterministicFakeQpSolver solver;
    auto settings = MakeQpSettings();
    settings.max_iterations =
        std::numeric_limits<std::size_t>::max();

    ExpectInvalidArgument(
        solver.Solve(MakeUnitBoxQp(), settings),
        "settings.max_iterations");
    EXPECT_EQ(solver.dispatch_count(), 0U);
  }
}

TEST(BoundedQpSolver, RejectsNonPositiveOrNonFiniteAbsoluteTolerance) {
  for (const double value :
       {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()}) {
    DeterministicFakeQpSolver solver;
    auto settings = MakeQpSettings();
    settings.absolute_tolerance = value;

    ExpectInvalidArgument(
        solver.Solve(MakeUnitBoxQp(), settings),
        "settings.absolute_tolerance");
    EXPECT_EQ(solver.dispatch_count(), 0U);
  }
}

TEST(BoundedQpSolver, RejectsNonPositiveOrNonFiniteRelativeTolerance) {
  for (const double value :
       {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()}) {
    DeterministicFakeQpSolver solver;
    auto settings = MakeQpSettings();
    settings.relative_tolerance = value;

    ExpectInvalidArgument(
        solver.Solve(MakeUnitBoxQp(), settings),
        "settings.relative_tolerance");
    EXPECT_EQ(solver.dispatch_count(), 0U);
  }
}

TEST(BoundedQpSolver, AcceptsPositiveSemidefiniteSingularHessian) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.hessian_upper_triangle.resize(2, 2);
  problem.hessian_upper_triangle.setZero();
  problem.hessian_upper_triangle.insert(0, 0) = 1.0;
  problem.hessian_upper_triangle.insert(0, 1) = 1.0;
  problem.hessian_upper_triangle.insert(1, 1) = 1.0;
  problem.hessian_upper_triangle.makeCompressed();
  problem.gradient = Eigen::VectorXd::Zero(2);
  problem.constraints.resize(1, 2);
  problem.constraints.setZero();

  EXPECT_TRUE(IsOk(solver.Solve(problem, MakeQpSettings())));
  EXPECT_EQ(solver.dispatch_count(), 1U);
}

TEST(BoundedQpSolver, AcceptsZeroHessian) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.hessian_upper_triangle.setZero();

  EXPECT_TRUE(IsOk(solver.Solve(problem, MakeQpSettings())));
  EXPECT_EQ(solver.dispatch_count(), 1U);
}

TEST(BoundedQpSolver, RejectsIndefiniteHessianBeforeBackendDispatch) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.hessian_upper_triangle.resize(2, 2);
  problem.hessian_upper_triangle.setZero();
  problem.hessian_upper_triangle.insert(0, 1) = 1.0;
  problem.hessian_upper_triangle.makeCompressed();
  problem.gradient = Eigen::VectorXd::Zero(2);
  problem.constraints.resize(1, 2);
  problem.constraints.setZero();

  ExpectInvalidArgument(
      solver.Solve(problem, MakeQpSettings()),
      "problem.hessian_upper_triangle");
  EXPECT_EQ(solver.dispatch_count(), 0U);
}

TEST(BoundedQpSolver, DoesNotPromoteMaximumIterationsToSolved) {
  QpSolution maximum_iterations{
      .termination = QpTermination::kMaxIterations,
      .primal = Eigen::VectorXd::Constant(1, 0.25),
      .objective = -0.4375,
      .primal_residual = 0.1,
      .dual_residual = 0.2,
      .iterations = 1U,
  };
  DeterministicFakeQpSolver solver(maximum_iterations);

  const auto result = solver.Solve(MakeUnitBoxQp(), MakeQpSettings());

  ASSERT_TRUE(IsOk(result));
  EXPECT_EQ(
      std::get<QpSolution>(result).termination,
      QpTermination::kMaxIterations);
}

}  // namespace
}  // namespace lunar::planning::v3
