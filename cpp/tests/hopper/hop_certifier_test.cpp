#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/hop_certifier.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(HopCertifierTest,
     MissingMapNeverLeaksPartiallyCertifiedReference) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const auto limits = std::get<HopperPlannerLimits>(
      bind_hopper_limits(hopper_test::ValidAlgorithm()));
  HopCertificationContext context{
      .reference_time_origin =
          ClockStamp{"mission", std::chrono::nanoseconds{0}},
      .launch_state =
          {
              .orientation_body_to_frame =
                  Quaternion{1.0, 0.0, 0.0, 0.0},
              .error_bounds = hopper_test::ZeroHopperError(),
          },
      .map = nullptr,
      .capability = capability,
      .limits = limits,
      .source_request_id = "request",
      .source_error_model_ref =
          capability.error_model.content_ref,
  };
  HopCertifier certifier{std::move(context)};
  const auto result = certifier.certify_first_edge(
      LandingGraphNode{.node_id = "from"},
      LandingGraphNode{.node_id = "to"});
  EXPECT_FALSE(result.certified_candidate.has_value());
  EXPECT_EQ(result.rejection_reason, "missing_map");
}

}  // namespace
}  // namespace lunar::planning::v3
