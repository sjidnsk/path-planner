#include <gtest/gtest.h>

#include "lunar_path_planner/v3/hopper/hopper_commitment_state_machine.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(HopperCommitmentStateMachineTest,
     FollowsCertifiedJumpLifecycle) {
  const HopperCommitmentStateMachine machine;
  HopperCommitmentSnapshot state{
      .state = HopperCommitmentState::kGroundHold,
      .commitment_scope = HopperCommitmentScope::kGroundHold,
  };
  auto transition = machine.transition(
      state,
      {
          .type = HopperCommitmentEventType::
              kPublishCertifiedCandidate,
          .bundle_id = "bundle",
          .boundary_id = "boundary",
      });
  ASSERT_TRUE(transition.accepted);
  state = transition.next;
  transition = machine.transition(
      state,
      {
          .type =
              HopperCommitmentEventType::kLockJumpBoundary,
          .bundle_id = "bundle",
          .boundary_id = "boundary",
      });
  ASSERT_TRUE(transition.accepted);
  state = transition.next;
  transition = machine.transition(
      state,
      {
          .type = HopperCommitmentEventType::kDetectLaunch,
          .boundary_id = "boundary",
      });
  ASSERT_TRUE(transition.accepted);
  state = transition.next;
  transition = machine.transition(
      state,
      {
          .type =
              HopperCommitmentEventType::kDetectStableLanding,
      });
  ASSERT_TRUE(transition.accepted);
  EXPECT_EQ(
      transition.next.state,
      HopperCommitmentState::kLandedHold);
  EXPECT_TRUE(machine.may_publish_new_jump(transition.next));
}

TEST(HopperCommitmentStateMachineTest,
     RejectionReturnsByteForByteEquivalentSnapshot) {
  const HopperCommitmentSnapshot committed{
      .state = HopperCommitmentState::kJumpCommitted,
      .active_bundle_id = "bundle",
      .active_boundary_id = "boundary",
      .commitment_scope =
          HopperCommitmentScope::kJumpBoundary,
  };
  const auto result =
      HopperCommitmentStateMachine{}.transition(
          committed,
          {
              .type = HopperCommitmentEventType::
                  kPublishCertifiedCandidate,
              .bundle_id = "other",
              .boundary_id = "other",
          });
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.next.state, committed.state);
  EXPECT_EQ(result.next.active_bundle_id, committed.active_bundle_id);
  EXPECT_EQ(
      result.next.active_boundary_id,
      committed.active_boundary_id);
  EXPECT_EQ(
      result.reason_code,
      "COMMITTED_JUMP_CANNOT_BE_REPLACED");
}

}  // namespace
}  // namespace lunar::planning::v3
