#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "lunar_path_planner/v3/search/ara_star.hpp"

namespace lunar::planning::v3 {
namespace {

struct TinyState final {
  StateKey id{};
};

struct TinyEdge final {
  StateKey successor{};
  std::string id;
  std::int64_t time_ns{};
  SecondaryCostVector costs{};
  bool hard_feasible{true};
};

class TinyGraphAdapter final {
 public:
  std::map<StateKey, std::vector<TinyEdge>> outgoing;
  std::map<StateKey, std::int64_t> heuristic_ns;
  std::set<StateKey> terminals;

  mutable std::vector<std::string> call_log;
  mutable std::vector<StateKey> expansion_order;
  mutable bool invalid_edge_cost_was_queried{false};

  [[nodiscard]] StateKey Key(const TinyState& state) const {
    return state.id;
  }

  [[nodiscard]] std::vector<SearchTransition<TinyState, TinyEdge>> Expand(
      const TinyState& state) const {
    expansion_order.push_back(state.id);
    std::vector<SearchTransition<TinyState, TinyEdge>> transitions;
    if (const auto found = outgoing.find(state.id); found != outgoing.end()) {
      transitions.reserve(found->second.size());
      for (const auto& edge : found->second) {
        transitions.push_back(SearchTransition<TinyState, TinyEdge>{
            .stable_edge_id = edge.id,
            .successor = TinyState{edge.successor},
            .edge = edge,
        });
      }
    }
    return transitions;
  }

  [[nodiscard]] bool HardFeasible(
      const SearchTransition<TinyState, TinyEdge>& transition) const {
    call_log.push_back("hard:" + transition.stable_edge_id);
    return transition.edge.hard_feasible;
  }

  [[nodiscard]] DurationNanoseconds TransitionTime(
      const SearchTransition<TinyState, TinyEdge>& transition) const {
    call_log.push_back("time:" + transition.stable_edge_id);
    if (!transition.edge.hard_feasible) {
      invalid_edge_cost_was_queried = true;
    }
    return DurationNanoseconds{
        std::chrono::nanoseconds{transition.edge.time_ns}};
  }

  [[nodiscard]] DurationNanoseconds AdmissibleTimeHeuristic(
      const TinyState& state,
      const SearchProblem<TinyState>& /*problem*/) const {
    const auto found = heuristic_ns.find(state.id);
    const std::int64_t value =
        found == heuristic_ns.end() ? 0 : found->second;
    return DurationNanoseconds{std::chrono::nanoseconds{value}};
  }

  [[nodiscard]] SecondaryCostVector SecondaryCosts(
      const SearchTransition<TinyState, TinyEdge>& transition) const {
    call_log.push_back("secondary:" + transition.stable_edge_id);
    if (!transition.edge.hard_feasible) {
      invalid_edge_cost_was_queried = true;
    }
    return transition.edge.costs;
  }

  [[nodiscard]] bool IsTerminal(
      const TinyState& state,
      const SearchProblem<TinyState>& /*problem*/) const {
    return terminals.contains(state.id);
  }
};

static_assert(PlatformSearchAdapter<TinyGraphAdapter, TinyState, TinyEdge>);

[[nodiscard]] ResourceCaps GenerousLimits() {
  return ResourceCaps{
      .maximum_expanded_states = 100U,
      .maximum_reopened_states = 100U,
      .maximum_generated_candidates = 100U,
      .maximum_open_states = 100U,
      .maximum_memory_bytes = 1024U * 1024U,
  };
}

[[nodiscard]] AraStarConfig MakeConfig(const double initial_epsilon = 2.5,
                                       const double epsilon_decrement = 0.5,
                                       const double target_epsilon = 1.0) {
  return AraStarConfig{
      .initial_epsilon = initial_epsilon,
      .epsilon_decrement = epsilon_decrement,
      .target_epsilon = target_epsilon,
  };
}

[[nodiscard]] SearchProblem<TinyState> MakeProblem(
    const ResourceCaps& limits = GenerousLimits()) {
  return SearchProblem<TinyState>{
      .start = TinyState{0U},
      .limits = limits,
  };
}

[[nodiscard]] TinyGraphAdapter MakeAnytimeGraph() {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 1U, .id = "fast-a", .time_ns = 1'000'000'000LL},
      TinyEdge{
          .successor = 2U, .id = "tempt-a", .time_ns = 2'000'000'000LL},
  };
  adapter.outgoing[1U] = {
      TinyEdge{.successor = 3U, .id = "fast-b", .time_ns = 2'000'000'000LL},
  };
  adapter.outgoing[2U] = {
      TinyEdge{
          .successor = 3U, .id = "tempt-b", .time_ns = 3'000'000'000LL},
  };
  adapter.heuristic_ns = {
      {0U, 3'000'000'000LL},
      {1U, 2'000'000'000LL},
      {2U, 0LL},
      {3U, 0LL},
  };
  adapter.terminals.insert(3U);
  return adapter;
}

[[nodiscard]] std::string SerializeResult(
    const SearchResult<TinyState, TinyEdge>& result) {
  std::ostringstream stream;
  stream << static_cast<int>(result.status) << '|'
         << std::setprecision(17) << result.achieved_epsilon << '|'
         << result.expansions << '|' << result.generated_states << '|'
         << result.reason_code;
  for (const auto& candidate : result.candidates) {
    stream << "\n" << candidate.stable_path_id << '|'
           << candidate.total_time.value.count() << '|'
           << candidate.secondary_costs.energy << ','
           << candidate.secondary_costs.risk << ','
           << candidate.secondary_costs.smoothness << '|'
           << candidate.fully_hard_validated;
    for (const auto& state : candidate.states) {
      stream << "|s" << state.id;
    }
    for (const auto& edge_id : candidate.edge_ids) {
      stream << "|e" << edge_id;
    }
  }
  return stream.str();
}

TEST(AraStar, ImprovesTimeBoundWithoutUsingWallClock) {
  const auto adapter = MakeAnytimeGraph();

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig());

  ASSERT_EQ(result.status, SearchStatus::kSolved);
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.candidates.front().total_time.value,
            std::chrono::seconds{3});
  EXPECT_EQ(result.candidates.front().edge_ids,
            (std::vector<std::string>{"fast-a", "fast-b"}));
  EXPECT_DOUBLE_EQ(result.achieved_epsilon, 1.0);
}

TEST(AraStar, FiltersHardInvalidTransitionBeforeQueryingAnyCost) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 3U,
               .id = "invalid-cheap",
               .time_ns = -1LL,
               .hard_feasible = false},
      TinyEdge{.successor = 1U, .id = "safe-a", .time_ns = 1LL},
  };
  adapter.outgoing[1U] = {
      TinyEdge{.successor = 3U, .id = "safe-b", .time_ns = 2LL},
  };
  adapter.terminals.insert(3U);

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  ASSERT_EQ(result.status, SearchStatus::kSolved);
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().edge_ids,
            (std::vector<std::string>{"safe-a", "safe-b"}));
  EXPECT_FALSE(adapter.invalid_edge_cost_was_queried);
  EXPECT_EQ(std::count(adapter.call_log.begin(), adapter.call_log.end(),
                       "hard:invalid-cheap"),
            1);
  EXPECT_EQ(std::count(adapter.call_log.begin(), adapter.call_log.end(),
                       "time:invalid-cheap"),
            0);
  EXPECT_EQ(std::count(adapter.call_log.begin(), adapter.call_log.end(),
                       "secondary:invalid-cheap"),
            0);
}

TEST(AraStar, NeighborPermutationsProduceIdenticalSerializedResults) {
  const std::array<TinyEdge, 3U> root_edges{
      TinyEdge{.successor = 3U, .id = "branch-c", .time_ns = 3LL},
      TinyEdge{.successor = 1U, .id = "branch-a", .time_ns = 1LL},
      TinyEdge{.successor = 2U, .id = "branch-b", .time_ns = 2LL},
  };
  std::array<std::size_t, 3U> order{0U, 1U, 2U};
  std::string baseline;
  bool first = true;
  do {
    TinyGraphAdapter adapter;
    for (const auto index : order) {
      adapter.outgoing[0U].push_back(root_edges[index]);
    }
    adapter.terminals = {1U, 2U, 3U};

    const auto result = RunAraStar<TinyState, TinyEdge>(
        adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));
    const std::string serialized = SerializeResult(result);
    if (first) {
      baseline = serialized;
      first = false;
    } else {
      EXPECT_EQ(serialized, baseline);
    }
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(AraStar, EqualPriorityStatesExpandByStateKey) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 2U, .id = "to-two", .time_ns = 1LL},
      TinyEdge{.successor = 1U, .id = "to-one", .time_ns = 1LL},
  };
  adapter.outgoing[1U] = {
      TinyEdge{.successor = 3U, .id = "one-goal", .time_ns = 10LL},
  };
  adapter.outgoing[2U] = {
      TinyEdge{.successor = 4U, .id = "two-goal", .time_ns = 10LL},
  };
  adapter.terminals = {3U, 4U};

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  ASSERT_EQ(result.status, SearchStatus::kSolved);
  ASSERT_GE(adapter.expansion_order.size(), 3U);
  EXPECT_EQ(adapter.expansion_order[0], 0U);
  EXPECT_EQ(adapter.expansion_order[1], 1U);
  EXPECT_EQ(adapter.expansion_order[2], 2U);
}

TEST(AraStar, AcceptsZeroDurationTransitionsWithoutRounding) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 1U, .id = "zero-a", .time_ns = 0LL},
  };
  adapter.outgoing[1U] = {
      TinyEdge{.successor = 2U, .id = "zero-b", .time_ns = 0LL},
  };
  adapter.terminals.insert(2U);

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  ASSERT_EQ(result.status, SearchStatus::kSolved);
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().total_time.value.count(), 0LL);
}

TEST(AraStar, RejectsNegativeTransitionDuration) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 1U, .id = "negative", .time_ns = -1LL},
  };

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  EXPECT_EQ(result.status, SearchStatus::kInvalidAdapterCost);
  EXPECT_EQ(result.reason_code,
            "search.invalid_adapter_cost.negative_transition_time");
}

TEST(AraStar, RejectsSignedNanosecondOverflow) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 1U,
               .id = "maximum",
               .time_ns = std::numeric_limits<std::int64_t>::max()},
  };
  adapter.outgoing[1U] = {
      TinyEdge{.successor = 2U, .id = "overflow", .time_ns = 1LL},
  };
  adapter.terminals.insert(2U);

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  EXPECT_EQ(result.status, SearchStatus::kInvalidAdapterCost);
  EXPECT_EQ(result.reason_code,
            "search.invalid_adapter_cost.transition_time_overflow");
}

TEST(AraStar, RejectsNegativeHeuristic) {
  TinyGraphAdapter adapter;
  adapter.heuristic_ns[0U] = -1LL;

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  EXPECT_EQ(result.status, SearchStatus::kInvalidAdapterCost);
  EXPECT_EQ(result.reason_code,
            "search.invalid_adapter_cost.invalid_time_heuristic");
  EXPECT_EQ(result.expansions, 0U);
}

TEST(AraStar, RejectsNonFiniteSecondaryCost) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 1U,
               .id = "nan-risk",
               .time_ns = 1LL,
               .costs = SecondaryCostVector{
                   .energy = 0.0,
                   .risk = std::numeric_limits<double>::quiet_NaN(),
                   .smoothness = 0.0}},
  };

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  EXPECT_EQ(result.status, SearchStatus::kInvalidAdapterCost);
  EXPECT_EQ(result.reason_code,
            "search.invalid_adapter_cost.invalid_secondary_cost");
}

TEST(AraStar, RejectsNegativeSecondaryCost) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 1U,
               .id = "negative-energy",
               .time_ns = 1LL,
               .costs = SecondaryCostVector{
                   .energy = -0.1, .risk = 0.0, .smoothness = 0.0}},
  };

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  EXPECT_EQ(result.status, SearchStatus::kInvalidAdapterCost);
  EXPECT_EQ(result.reason_code,
            "search.invalid_adapter_cost.invalid_secondary_cost");
}

TEST(AraStar, StopsBeforeExpansionPastExpandedStateLimit) {
  auto limits = GenerousLimits();
  limits.maximum_expanded_states = 1U;
  const auto adapter = MakeAnytimeGraph();

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(limits), MakeConfig());

  EXPECT_EQ(result.status, SearchStatus::kResourceLimit);
  EXPECT_EQ(result.reason_code,
            "search.resource_limit.maximum_expanded_states");
  EXPECT_EQ(result.expansions, 1U);
  EXPECT_TRUE(result.candidates.empty());
}

TEST(AraStar, StopsWhenOpenStateLimitWouldBeExceeded) {
  auto limits = GenerousLimits();
  limits.maximum_open_states = 1U;
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 2U, .id = "two", .time_ns = 1LL},
      TinyEdge{.successor = 1U, .id = "one", .time_ns = 1LL},
  };

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(limits), MakeConfig(1.0, 0.5, 1.0));

  EXPECT_EQ(result.status, SearchStatus::kResourceLimit);
  EXPECT_EQ(result.reason_code, "search.resource_limit.maximum_open_states");
  EXPECT_EQ(result.expansions, 1U);
}

TEST(AraStar, StopsAtGeneratedCandidateLimit) {
  auto limits = GenerousLimits();
  limits.maximum_generated_candidates = 1U;
  const auto adapter = MakeAnytimeGraph();

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(limits), MakeConfig());

  ASSERT_EQ(result.status, SearchStatus::kResourceLimit);
  EXPECT_EQ(result.reason_code,
            "search.resource_limit.maximum_generated_candidates");
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().total_time.value,
            std::chrono::seconds{5});
}

TEST(AraStar, RetainsBestSoFarWhenLaterPassHitsExpansionLimit) {
  auto limits = GenerousLimits();
  limits.maximum_expanded_states = 2U;
  const auto adapter = MakeAnytimeGraph();

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(limits), MakeConfig());

  ASSERT_EQ(result.status, SearchStatus::kResourceLimit);
  EXPECT_EQ(result.reason_code,
            "search.resource_limit.maximum_expanded_states");
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().total_time.value,
            std::chrono::seconds{5});
  EXPECT_EQ(result.candidates.front().edge_ids,
            (std::vector<std::string>{"tempt-a", "tempt-b"}));
}

TEST(AraStar, SuppressesDuplicateStableEdgeIdPaths) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 1U, .id = "same-edge", .time_ns = 1LL},
      TinyEdge{.successor = 2U, .id = "same-edge", .time_ns = 2LL},
  };
  adapter.terminals = {1U, 2U};

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig(1.0, 0.5, 1.0));

  ASSERT_EQ(result.status, SearchStatus::kSolved);
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().edge_ids,
            (std::vector<std::string>{"same-edge"}));
  EXPECT_EQ(result.candidates.front().total_time.value.count(), 1LL);
}

TEST(AraStar, RepeatedRunsSerializeIdentically) {
  const auto first = RunAraStar<TinyState, TinyEdge>(
      MakeAnytimeGraph(), MakeProblem(), MakeConfig());
  const auto second = RunAraStar<TinyState, TinyEdge>(
      MakeAnytimeGraph(), MakeProblem(), MakeConfig());

  EXPECT_EQ(SerializeResult(first), SerializeResult(second));
}

TEST(AraStar, ReturnsStableNoPathResultForExhaustedGraph) {
  TinyGraphAdapter adapter;
  adapter.outgoing[0U] = {
      TinyEdge{.successor = 1U, .id = "dead-end", .time_ns = 1LL},
  };

  const auto result = RunAraStar<TinyState, TinyEdge>(
      adapter, MakeProblem(), MakeConfig());

  EXPECT_EQ(result.status, SearchStatus::kNoPath);
  EXPECT_EQ(result.reason_code, "search.no_path");
  EXPECT_TRUE(result.candidates.empty());
}

}  // namespace
}  // namespace lunar::planning::v3
