#include "lunar_path_planner/v3/search/candidate_ranker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

namespace lpp = lunar::planning::v3;

namespace {

using namespace std::chrono_literals;

[[nodiscard]] lpp::CandidateScore MakeCandidate(
    std::string stable_candidate_id,
    std::chrono::nanoseconds total_time,
    double energy,
    double risk,
    double smoothness,
    bool fully_hard_validated = true) {
  return {
      .stable_candidate_id = std::move(stable_candidate_id),
      .total_time = lpp::DurationNanoseconds{total_time},
      .energy = energy,
      .risk = risk,
      .smoothness = smoothness,
      .fully_hard_validated = fully_hard_validated,
  };
}

[[nodiscard]] const lpp::Error& RequireError(
    const lpp::Result<lpp::TimeEquivalentPool>& result) {
  EXPECT_FALSE(lpp::IsOk(result));
  return std::get<lpp::Error>(result);
}

[[nodiscard]] std::vector<std::string> CandidateIds(
    const lpp::TimeEquivalentPool& pool) {
  std::vector<std::string> ids;
  ids.reserve(pool.ordered_candidates.size());
  for (const auto& candidate : pool.ordered_candidates) {
    ids.push_back(candidate.stable_candidate_id);
  }
  return ids;
}

[[nodiscard]] std::string SerializeOrder(
    const lpp::TimeEquivalentPool& pool) {
  std::string serialized;
  for (const auto& candidate : pool.ordered_candidates) {
    if (!serialized.empty()) {
      serialized.push_back('|');
    }
    serialized.append(candidate.stable_candidate_id);
  }
  return serialized;
}

TEST(CandidateRanker, SecondaryCostsOnlyRankInsideTimeEquivalentPool) {
  const std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("fast-risky", 10'000ms, 8.0, 5.0, 2.0),
      MakeCandidate("near-efficient", 10'030ms, 2.0, 1.0, 1.0),
      MakeCandidate("too-slow", 10'070ms, 0.1, 0.1, 0.1),
      MakeCandidate("invalid", 9'000ms, 0.0, 0.0, 0.0, false),
  };

  const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
      candidates, lpp::DurationNanoseconds{50ms});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& pool = std::get<lpp::TimeEquivalentPool>(result);
  EXPECT_EQ(pool.minimum_validated_time.value, 10s);
  EXPECT_EQ(pool.tolerance.value, 50ms);
  EXPECT_EQ(CandidateIds(pool),
            (std::vector<std::string>{"near-efficient", "fast-risky"}));
}

TEST(CandidateRanker, IncludesExactToleranceBoundaryOnly) {
  const std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("minimum", 100ns, 3.0, 3.0, 3.0),
      MakeCandidate("boundary", 150ns, 2.0, 2.0, 2.0),
      MakeCandidate("outside", 151ns, 0.0, 0.0, 0.0),
  };

  const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
      candidates, lpp::DurationNanoseconds{50ns});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& pool = std::get<lpp::TimeEquivalentPool>(result);
  EXPECT_EQ(pool.minimum_validated_time.value, 100ns);
  EXPECT_EQ(CandidateIds(pool),
            (std::vector<std::string>{"boundary", "minimum"}));
}

TEST(CandidateRanker, UsesAllLexicographicTieBreakersInOrder) {
  const std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("z-id", 102ns, 1.0, 1.0, 1.0),
      MakeCandidate("a-id", 102ns, 1.0, 1.0, 1.0),
      MakeCandidate("earlier", 101ns, 1.0, 1.0, 1.0),
      MakeCandidate("smooth", 100ns, 1.0, 1.0, 0.5),
      MakeCandidate("risk", 100ns, 1.0, 0.5, 9.0),
      MakeCandidate("energy", 100ns, 0.5, 9.0, 9.0),
  };

  const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
      candidates, lpp::DurationNanoseconds{2ns});

  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(
      CandidateIds(std::get<lpp::TimeEquivalentPool>(result)),
      (std::vector<std::string>{"energy", "risk", "smooth", "earlier",
                                "a-id", "z-id"}));
}

TEST(CandidateRanker, RejectsEmptyOrEmptyValidatedInput) {
  const std::array<lpp::CandidateScore, 0> empty{};
  const auto empty_result =
      lpp::CandidateRanker::BuildTimeEquivalentPool(
          std::span<const lpp::CandidateScore>{empty},
          lpp::DurationNanoseconds{0ns});
  EXPECT_EQ(RequireError(empty_result).code,
            lpp::ErrorCode::kInvalidArgument);

  const std::vector<lpp::CandidateScore> unvalidated{
      MakeCandidate("first", 1ns, 1.0, 1.0, 1.0, false),
      MakeCandidate("second", 2ns, 2.0, 2.0, 2.0, false),
  };
  const auto unvalidated_result =
      lpp::CandidateRanker::BuildTimeEquivalentPool(
          unvalidated, lpp::DurationNanoseconds{0ns});
  EXPECT_EQ(RequireError(unvalidated_result).code,
            lpp::ErrorCode::kInvalidArgument);
}

TEST(CandidateRanker, RejectsNegativeToleranceAndValidatedTime) {
  const std::vector<lpp::CandidateScore> valid{
      MakeCandidate("valid", 1ns, 0.0, 0.0, 0.0),
  };
  const auto tolerance_result =
      lpp::CandidateRanker::BuildTimeEquivalentPool(
          valid, lpp::DurationNanoseconds{-1ns});
  EXPECT_EQ(RequireError(tolerance_result).code,
            lpp::ErrorCode::kInvalidArgument);

  const std::vector<lpp::CandidateScore> negative_time{
      MakeCandidate("negative", -1ns, 0.0, 0.0, 0.0),
  };
  const auto time_result =
      lpp::CandidateRanker::BuildTimeEquivalentPool(
          negative_time, lpp::DurationNanoseconds{0ns});
  EXPECT_EQ(RequireError(time_result).code,
            lpp::ErrorCode::kInvalidArgument);
}

TEST(CandidateRanker, RejectsTimeBandAdditionOverflow) {
  constexpr auto kMaximum =
      std::chrono::nanoseconds::rep{
          std::numeric_limits<std::chrono::nanoseconds::rep>::max()};
  const std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("near-maximum", std::chrono::nanoseconds{kMaximum - 1},
                    0.0, 0.0, 0.0),
  };

  const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
      candidates, lpp::DurationNanoseconds{2ns});

  EXPECT_EQ(RequireError(result).code, lpp::ErrorCode::kInvalidArgument);
}

TEST(CandidateRanker, RejectsDuplicateValidatedStableIds) {
  const std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("duplicate", 1ns, 0.0, 0.0, 0.0),
      MakeCandidate("duplicate", 2ns, 1.0, 1.0, 1.0),
  };

  const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
      candidates, lpp::DurationNanoseconds{1ns});

  EXPECT_EQ(RequireError(result).code, lpp::ErrorCode::kInvalidArgument);
}

TEST(CandidateRanker, UnvalidatedCandidatesCannotPoisonValidatedPool) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("selected", 10ns, 1.0, 1.0, 1.0),
      MakeCandidate("", -1ns, nan, -1.0,
                    std::numeric_limits<double>::infinity(), false),
      MakeCandidate("selected", 0ns, nan, nan, nan, false),
  };

  const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
      candidates, lpp::DurationNanoseconds{0ns});

  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(CandidateIds(std::get<lpp::TimeEquivalentPool>(result)),
            (std::vector<std::string>{"selected"}));
}

TEST(CandidateRanker, RejectsInvalidValidatedStableIds) {
  const std::array<std::string, 5> invalid_ids{
      "",
      "-starts-with-punctuation",
      "has space",
      "has@symbol",
      std::string(129U, 'a'),
  };

  for (const auto& invalid_id : invalid_ids) {
    SCOPED_TRACE(invalid_id);
    const std::vector<lpp::CandidateScore> candidates{
        MakeCandidate(invalid_id, 1ns, 0.0, 0.0, 0.0),
    };
    const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
        candidates, lpp::DurationNanoseconds{0ns});
    EXPECT_EQ(RequireError(result).code,
              lpp::ErrorCode::kInvalidArgument);
  }
}

TEST(CandidateRanker, AcceptsFullStableIdentifierAlphabet) {
  const std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("A.z_9:/-ok", 1ns, 0.0, 0.0, 0.0),
  };

  const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
      candidates, lpp::DurationNanoseconds{0ns});

  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(CandidateIds(std::get<lpp::TimeEquivalentPool>(result)),
            (std::vector<std::string>{"A.z_9:/-ok"}));
}

TEST(CandidateRanker, RejectsEveryNonFiniteOrNegativeSecondaryMetric) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::array<std::array<double, 3>, 9> invalid_metrics{{
      {-1.0, 0.0, 0.0},
      {nan, 0.0, 0.0},
      {infinity, 0.0, 0.0},
      {0.0, -1.0, 0.0},
      {0.0, nan, 0.0},
      {0.0, infinity, 0.0},
      {0.0, 0.0, -1.0},
      {0.0, 0.0, nan},
      {0.0, 0.0, infinity},
  }};

  for (const auto& metrics : invalid_metrics) {
    SCOPED_TRACE(testing::Message()
                 << "energy=" << metrics[0] << " risk=" << metrics[1]
                 << " smoothness=" << metrics[2]);
    const std::vector<lpp::CandidateScore> candidates{
        MakeCandidate("candidate", 1ns, metrics[0], metrics[1],
                      metrics[2]),
    };
    const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
        candidates, lpp::DurationNanoseconds{0ns});
    EXPECT_EQ(RequireError(result).code,
              lpp::ErrorCode::kInvalidArgument);
  }
}

TEST(CandidateRanker, EveryInputPermutationHasIdenticalOrder) {
  std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("delta", 100ns, 2.0, 0.0, 0.0),
      MakeCandidate("charlie", 101ns, 1.0, 0.0, 0.0),
      MakeCandidate("bravo", 102ns, 1.0, 0.0, 0.0),
      MakeCandidate("alpha", 102ns, 1.0, 0.0, 0.0),
      MakeCandidate("outside", 104ns, 0.0, 0.0, 0.0),
  };
  std::ranges::sort(candidates, {}, &lpp::CandidateScore::stable_candidate_id);

  std::size_t permutation_count = 0U;
  do {
    const auto result = lpp::CandidateRanker::BuildTimeEquivalentPool(
        candidates, lpp::DurationNanoseconds{2ns});
    ASSERT_TRUE(lpp::IsOk(result));
    const auto& pool = std::get<lpp::TimeEquivalentPool>(result);
    EXPECT_EQ(SerializeOrder(pool), "charlie|alpha|bravo|delta");

    const auto repeated = lpp::CandidateRanker::BuildTimeEquivalentPool(
        candidates, lpp::DurationNanoseconds{2ns});
    ASSERT_TRUE(lpp::IsOk(repeated));
    EXPECT_EQ(SerializeOrder(std::get<lpp::TimeEquivalentPool>(repeated)),
              SerializeOrder(pool));
    ++permutation_count;
  } while (std::next_permutation(
      candidates.begin(), candidates.end(),
      [](const lpp::CandidateScore& lhs,
         const lpp::CandidateScore& rhs) {
        return lhs.stable_candidate_id < rhs.stable_candidate_id;
      }));

  EXPECT_EQ(permutation_count, 120U);
}

}  // namespace
