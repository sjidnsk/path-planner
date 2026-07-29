#include <chrono>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/api/execution_arbitrator.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

namespace lpp = lunar::planning::v3;
namespace {

lpp::ContentRef BundleRef(std::string id = "bundle-1") {
  return {
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, 'a'),
  };
}

lpp::PlanningAttempt FailedAttempt(
    lpp::PlanningOutcome outcome =
        lpp::PlanningOutcome::kNoKnownSafeRoute) {
  return {
      .request_id = "request-1",
      .response_time =
          {
              .clock_id = "mission",
              .tick = std::chrono::nanoseconds{10},
          },
      .failure =
          lpp::Error{
              lpp::ErrorCode::kNoKnownSafeRoute,
              "planner",
              "no route",
          },
      .planning_outcome = outcome,
      .reason_code = "NO_KNOWN_SAFE_ROUTE",
      .call_diagnostics =
          {
              .api_latency =
                  lpp::DurationNanoseconds{
                      std::chrono::nanoseconds{0}},
              .termination_reason = "GRAPH_EXHAUSTED",
          },
  };
}

lpp::PreviousExecutionContext GroundExecution() {
  return {
      .active_bundle_ref = BundleRef(),
      .active_bundle_handle = "bundle-handle",
      .commit_boundary =
          lpp::TimeCommitBoundary{
              lpp::DurationNanoseconds{
                  std::chrono::milliseconds{100}}},
      .execution_cursor =
          lpp::TimeExecutionCursor{
              .offset =
                  lpp::DurationNanoseconds{
                      std::chrono::milliseconds{10}},
          },
      .controller_status = lpp::ControllerStatus::kExecuting,
      .source_map_snapshot_ref =
          {
              .id = "map",
              .revision = 1U,
              .content_hash = std::string(64U, 'b'),
          },
      .source_capability_ref =
          {
              .id = "capability",
              .revision = 1U,
              .content_hash = std::string(64U, 'c'),
          },
  };
}

lpp::PreviousExecutionContext CommittedJumpExecution() {
  return {
      .active_bundle_ref = BundleRef(),
      .active_bundle_handle = "bundle-handle",
      .commit_boundary =
          lpp::JumpCommitBoundary{
              .boundary_id = "boundary-1",
              .locked = true,
          },
      .execution_cursor =
          lpp::JumpExecutionCursor{
              .jump_state =
                  lpp::JumpExecutionState::kJumpCommitted,
              .boundary_id = "boundary-1",
          },
      .controller_status = lpp::ControllerStatus::kCommitted,
      .source_map_snapshot_ref =
          {
              .id = "map",
              .revision = 1U,
              .content_hash = std::string(64U, 'b'),
          },
      .source_capability_ref =
          {
              .id = "capability",
              .revision = 1U,
              .content_hash = std::string(64U, 'c'),
          },
  };
}

lpp::PlatformState StationaryGround() {
  return lpp::WheeledOrLeggedState{};
}

lpp::PlatformState MovingGround() {
  lpp::WheeledOrLeggedState state{};
  state.linear_velocity_mps.x = 0.1;
  return state;
}

lpp::PlatformState StationaryHopper() {
  return lpp::HopperState{};
}

}  // namespace

TEST(ExecutionArbitratorTest,
     StationaryHoldNeverCreatesOrReferencesBundle) {
  const lpp::PlanningResponse response =
      lpp::ArbitratePlanningResponse(
          FailedAttempt(), std::nullopt, StationaryGround());

  EXPECT_EQ(response.execution_directive,
            lpp::ExecutionDirective::kHoldStationary);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
  EXPECT_FALSE(response.active_bundle_ref.has_value());
  EXPECT_TRUE(
      lpp::SemanticValidator{}.Validate(response).ok());
}

TEST(ExecutionArbitratorTest,
     MovingStateWithoutReferenceFailsClosed) {
  const lpp::PlanningResponse response =
      lpp::ArbitratePlanningResponse(
          FailedAttempt(), std::nullopt, MovingGround());

  EXPECT_EQ(response.execution_directive,
            lpp::ExecutionDirective::kNoSafePlannerReference);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
  EXPECT_FALSE(response.active_bundle_ref.has_value());
  EXPECT_TRUE(
      lpp::SemanticValidator{}.Validate(response).ok());
}

TEST(ExecutionArbitratorTest,
     ValidGroundExecutionContinuesPinnedBundle) {
  const lpp::PreviousExecutionContext previous =
      GroundExecution();
  const lpp::PlanningResponse response =
      lpp::ArbitratePlanningResponse(
          FailedAttempt(), previous, StationaryGround());

  EXPECT_EQ(response.execution_directive,
            lpp::ExecutionDirective::kContinueActiveBundle);
  ASSERT_TRUE(response.active_bundle_ref.has_value());
  EXPECT_EQ(*response.active_bundle_ref,
            previous.active_bundle_ref);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
  EXPECT_TRUE(
      lpp::SemanticValidator{}.Validate(response).ok());
}

TEST(ExecutionArbitratorTest,
     InvalidatedGroundReferenceFallsBackToHold) {
  lpp::PlanningAttempt attempt = FailedAttempt(
      lpp::PlanningOutcome::kActiveReferenceInvalidated);
  attempt.previous_execution_context_valid = false;
  const lpp::PlanningResponse response =
      lpp::ArbitratePlanningResponse(
          attempt, GroundExecution(), StationaryGround());

  EXPECT_EQ(response.planning_outcome,
            lpp::PlanningOutcome::kActiveReferenceInvalidated);
  EXPECT_EQ(response.execution_directive,
            lpp::ExecutionDirective::kHoldStationary);
  EXPECT_FALSE(response.active_bundle_ref.has_value());
  EXPECT_TRUE(
      lpp::SemanticValidator{}.Validate(response).ok());
}

TEST(ExecutionArbitratorTest,
     CommittedJumpCannotBeReplacedByNewCandidate) {
  lpp::PlanningAttempt attempt = FailedAttempt();
  attempt.failure.reset();
  attempt.validated_new_bundle = lpp::ReferenceBundle{};
  attempt.planning_outcome =
      lpp::PlanningOutcome::kNewReferenceReady;
  attempt.reason_code = "NEW_REFERENCE_READY";
  const lpp::PreviousExecutionContext previous =
      CommittedJumpExecution();

  const lpp::PlanningResponse response =
      lpp::ArbitratePlanningResponse(
          attempt, previous, StationaryHopper());

  EXPECT_EQ(response.execution_directive,
            lpp::ExecutionDirective::kContinueCommittedJump);
  ASSERT_TRUE(response.active_bundle_ref.has_value());
  EXPECT_EQ(*response.active_bundle_ref,
            previous.active_bundle_ref);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
}

TEST(ExecutionArbitratorTest,
     ReadyAttemptActivatesOnlyItsValidatedBundle) {
  lpp::PlanningAttempt attempt = FailedAttempt();
  attempt.failure.reset();
  attempt.validated_new_bundle = lpp::ReferenceBundle{};
  attempt.planning_outcome =
      lpp::PlanningOutcome::kSafeFrontierReferenceReady;
  attempt.reason_code = "SAFE_FRONTIER_REACHED";

  const lpp::PlanningResponse response =
      lpp::ArbitratePlanningResponse(
          attempt, std::nullopt, StationaryGround());

  EXPECT_EQ(
      response.planning_outcome,
      lpp::PlanningOutcome::kSafeFrontierReferenceReady);
  EXPECT_EQ(response.execution_directive,
            lpp::ExecutionDirective::kActivateNewBundle);
  EXPECT_TRUE(response.new_reference_bundle.has_value());
  EXPECT_FALSE(response.active_bundle_ref.has_value());
}

TEST(ExecutionArbitratorTest,
     MalformedAttemptFailsClosedWithoutPublishingBundle) {
  lpp::PlanningAttempt attempt = FailedAttempt();
  attempt.validated_new_bundle = lpp::ReferenceBundle{};

  const lpp::PlanningResponse response =
      lpp::ArbitratePlanningResponse(
          attempt, std::nullopt, StationaryGround());

  EXPECT_EQ(response.planning_outcome,
            lpp::PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(response.execution_directive,
            lpp::ExecutionDirective::kHoldStationary);
  EXPECT_EQ(response.reason_code, "INVALID_PLANNING_ATTEMPT");
  EXPECT_FALSE(response.new_reference_bundle.has_value());
  EXPECT_FALSE(response.active_bundle_ref.has_value());
  EXPECT_TRUE(
      lpp::SemanticValidator{}.Validate(response).ok());
}

TEST(ExecutionArbitratorTest,
     NegativeZeroVelocityDoesNotAuthorizeStationaryHold) {
  lpp::WheeledOrLeggedState state{};
  state.yaw_rate_radps = -0.0;

  const lpp::PlanningResponse response =
      lpp::ArbitratePlanningResponse(
          FailedAttempt(), std::nullopt, state);

  EXPECT_EQ(response.execution_directive,
            lpp::ExecutionDirective::kNoSafePlannerReference);
}
