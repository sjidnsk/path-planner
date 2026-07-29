#include <gtest/gtest.h>

#include "lunar_path_planner/v3/hopper/landing_graph.hpp"

namespace lunar::planning::v3 {
namespace {

class RejectThenCertifyOracle final
    : public NextHopCertificationOracle {
 public:
  std::vector<std::string> attempted;

  FirstEdgeCertificationResult certify_first_edge(
      const LandingGraphNode& from,
      const LandingGraphNode& to) override {
    attempted.push_back(from.node_id + "->" + to.node_id);
    if (to.node_id == "reject") {
      return {
          .certified_candidate = std::nullopt,
          .rejection_reason = "fixture_rejected",
      };
    }
    HopperReference reference{};
    reference.reference_id = "certified";
    return {
        .certified_candidate =
            FirstEdgeCertificationResult::CertifiedCandidate{
                .reference = std::move(reference),
                .expected_execution_time =
                    DurationNanoseconds{
                        std::chrono::seconds{3}},
                .secondary_costs =
                    SecondaryCostVector{
                        .energy = 2.0,
                    },
            },
    };
  }
};

TEST(LandingGraphTest,
     RejectsFailedFirstEdgeAndResumesDeterministically) {
  std::vector<LandingGraphNode> nodes{
      {.node_id = "start"},
      {.node_id = "reject"},
      {.node_id = "safe"},
      {.node_id = "goal"},
  };
  std::vector<LandingGraphEdge> edges{
      {
          .edge_id = "start-reject",
          .from_node = 0U,
          .to_node = 1U,
          .state = LandingEdgeState::kCheapPossible,
          .lower_bound_execution_time =
              DurationNanoseconds{std::chrono::seconds{1}},
      },
      {
          .edge_id = "reject-goal",
          .from_node = 1U,
          .to_node = 3U,
          .state = LandingEdgeState::kCheapPossible,
          .lower_bound_execution_time =
              DurationNanoseconds{std::chrono::seconds{1}},
      },
      {
          .edge_id = "start-safe",
          .from_node = 0U,
          .to_node = 2U,
          .state = LandingEdgeState::kCheapPossible,
          .lower_bound_execution_time =
              DurationNanoseconds{std::chrono::seconds{2}},
      },
      {
          .edge_id = "safe-goal",
          .from_node = 2U,
          .to_node = 3U,
          .state = LandingEdgeState::kCheapPossible,
          .lower_bound_execution_time =
              DurationNanoseconds{std::chrono::seconds{1}},
      },
  };
  HopperPlannerLimits limits{};
  limits.maximum_full_certification_attempts = 4U;
  limits.time_equivalence_tolerance =
      DurationNanoseconds{std::chrono::milliseconds{50}};
  limits.ara_star = {
      .initial_epsilon = 1.0,
      .epsilon_decrement = 0.5,
      .target_epsilon = 1.0,
      .resource_caps =
          {
              .maximum_expanded_states = 32U,
              .maximum_reopened_states = 32U,
              .maximum_generated_candidates = 8U,
              .maximum_open_states = 32U,
              .maximum_memory_bytes = 1U << 16U,
          },
  };
  RejectThenCertifyOracle oracle;
  const std::array<std::size_t, 1> terminal{3U};
  const auto result = LazyLandingGraphPlanner{}.select(
      nodes, edges, 0U, terminal, oracle, limits);
  ASSERT_TRUE(IsOk(result));
  EXPECT_EQ(
      oracle.attempted,
      (std::vector<std::string>{
          "start->reject", "start->safe"}));
  EXPECT_EQ(
      std::get<LazyNextHopSelection>(result)
          .future_preview.authority,
      FutureRoutePreview::Authority::
          kNonAuthoritativeMissionPreview);
}

}  // namespace
}  // namespace lunar::planning::v3
