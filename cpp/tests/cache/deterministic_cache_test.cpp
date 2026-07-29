#include "lunar_path_planner/v3/cache/deterministic_cache.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace lpp = lunar::planning::v3;

namespace {

using StringCache = lpp::DeterministicCache<int, std::string>;

[[nodiscard]] lpp::ContentRef Ref(std::string id,
                                  std::uint32_t revision,
                                  std::string hash) {
  return {
      .id = std::move(id),
      .revision = revision,
      .content_hash = std::move(hash),
  };
}

[[nodiscard]] lpp::SafeProjectionCacheKey MakeProjectionKey(
    double resolution_m = 0.25,
    std::optional<lpp::ContentRef> learned_cost =
        Ref("learned-cost", 3U, "learned-hash")) {
  return lpp::MakeSafeProjectionCacheKey(
      "map-snapshot",
      "map-revision",
      "layer-manifest-hash",
      Ref("safety-capability", 7U, "capability-hash"),
      Ref("algorithm-config", 11U, "config-hash"),
      std::move(learned_cost),
      "map",
      resolution_m,
      "certified-error-v1");
}

[[nodiscard]] std::vector<int> PublishPermutation(
    const std::array<int, 3U>& order) {
  StringCache cache{2U};
  for (const int key : order) {
    cache.Publish(
        key, std::make_shared<const std::string>(std::to_string(key)));
  }
  return cache.Keys();
}

enum class OrchestrationEvent {
  kSemanticValidation,
  kCacheLookup,
  kProjectionBuild,
  kCachePublication,
  kFinalValidation,
};

struct TestOnlyOrchestrationSpy final {
  void SemanticValidate() {
    events.push_back(OrchestrationEvent::kSemanticValidation);
    semantic_validation_complete = true;
  }

  void BeforeCacheLookup() {
    EXPECT_TRUE(semantic_validation_complete);
    events.push_back(OrchestrationEvent::kCacheLookup);
    cache_lookup_complete = true;
  }

  void ProjectionBuild() {
    EXPECT_TRUE(cache_lookup_complete);
    events.push_back(OrchestrationEvent::kProjectionBuild);
  }

  void CachePublication() {
    events.push_back(OrchestrationEvent::kCachePublication);
  }

  void FinalValidate() {
    EXPECT_TRUE(cache_lookup_complete);
    events.push_back(OrchestrationEvent::kFinalValidation);
  }

  bool semantic_validation_complete{};
  bool cache_lookup_complete{};
  std::vector<OrchestrationEvent> events;
};

[[nodiscard]] std::shared_ptr<const std::string>
TestOnlyValidatedLookupOrBuild(StringCache& cache,
                               int key,
                               TestOnlyOrchestrationSpy& spy) {
  spy.SemanticValidate();
  spy.BeforeCacheLookup();
  auto value = cache.Get(key);
  if (value == nullptr) {
    spy.ProjectionBuild();
    value =
        std::make_shared<const std::string>("projection-" + std::to_string(key));
    spy.CachePublication();
    cache.Publish(key, value);
  }
  spy.FinalValidate();
  return value;
}

static_assert(std::totally_ordered<lpp::SafeProjectionCacheKey>);
static_assert(std::same_as<
              decltype(std::declval<const StringCache&>().Get(
                  std::declval<const int&>())),
              std::shared_ptr<const std::string>>);

TEST(DeterministicCache, FinalKeySetDoesNotDependOnInsertionOrder) {
  constexpr std::array<std::array<int, 3U>, 6U> permutations{{
      {{1, 2, 3}},
      {{1, 3, 2}},
      {{2, 1, 3}},
      {{2, 3, 1}},
      {{3, 1, 2}},
      {{3, 2, 1}},
  }};

  for (const auto& order : permutations) {
    EXPECT_EQ(PublishPermutation(order), (std::vector<int>{1, 2}));
  }
}

TEST(DeterministicCache, ReplacementUpdatesValueWithoutChangingKeySet) {
  StringCache cache{2U};
  cache.Publish(1, std::make_shared<const std::string>("old"));
  cache.Publish(2, std::make_shared<const std::string>("second"));

  cache.Publish(1, std::make_shared<const std::string>("replacement"));

  EXPECT_EQ(cache.Keys(), (std::vector<int>{1, 2}));
  ASSERT_NE(cache.Get(1), nullptr);
  EXPECT_EQ(*cache.Get(1), "replacement");
}

TEST(DeterministicCache, ReadDoesNotChangeEvictionOrder) {
  StringCache cache{2U};
  cache.Publish(1, std::make_shared<const std::string>("one"));
  cache.Publish(2, std::make_shared<const std::string>("two"));

  ASSERT_NE(cache.Get(2), nullptr);
  cache.Publish(0, std::make_shared<const std::string>("zero"));

  EXPECT_EQ(cache.Keys(), (std::vector<int>{0, 1}));
  EXPECT_EQ(cache.Get(2), nullptr);
}

TEST(DeterministicCache, RejectsZeroCapacityExplicitly) {
  EXPECT_THROW(
      {
        const StringCache invalid_cache{0U};
        static_cast<void>(invalid_cache);
      },
      std::invalid_argument);
}

TEST(DeterministicCache, NullPublicationDoesNotMutateState) {
  StringCache cache{2U};
  const auto original = std::make_shared<const std::string>("original");
  cache.Publish(1, original);
  const auto keys_before = cache.Keys();
  const auto value_before = cache.Get(1);

  EXPECT_THROW(
      cache.Publish(1, std::shared_ptr<const std::string>{}),
      std::invalid_argument);

  EXPECT_EQ(cache.Keys(), keys_before);
  EXPECT_EQ(cache.Get(1), value_before);
}

TEST(DeterministicCache, ConcurrentPublicationKeepsSameLowestCapacityKeys) {
  constexpr std::array<std::array<int, 8U>, 4U> launch_orders{{
      {{0, 1, 2, 3, 4, 5, 6, 7}},
      {{7, 6, 5, 4, 3, 2, 1, 0}},
      {{3, 7, 1, 5, 0, 6, 2, 4}},
      {{4, 2, 6, 0, 5, 1, 7, 3}},
  }};

  for (std::size_t schedule = 0U; schedule < launch_orders.size();
       ++schedule) {
    StringCache cache{3U};
    std::vector<std::thread> workers;
    workers.reserve(launch_orders[schedule].size());
    for (const int key : launch_orders[schedule]) {
      workers.emplace_back([&cache, key, schedule] {
        const auto yields =
            (static_cast<std::size_t>(key) + schedule) % 4U;
        for (std::size_t index = 0U; index < yields; ++index) {
          std::this_thread::yield();
        }
        cache.Publish(
            key,
            std::make_shared<const std::string>(std::to_string(key)));
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }

    EXPECT_EQ(cache.Keys(), (std::vector<int>{0, 1, 2}));
    for (const int key : {0, 1, 2}) {
      const auto value = cache.Get(key);
      ASSERT_NE(value, nullptr);
      EXPECT_EQ(*value, std::to_string(key));
    }
  }
}

TEST(DeterministicCache, ImmutablePublicationOutlivesPublisherAndCache) {
  std::weak_ptr<const std::string> lifetime;
  std::shared_ptr<const std::string> retrieved;
  {
    StringCache cache{1U};
    auto publication = std::make_shared<const std::string>("immutable");
    lifetime = publication;
    cache.Publish(1, publication);
    publication.reset();
    EXPECT_FALSE(lifetime.expired());
    retrieved = cache.Get(1);
  }

  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(*retrieved, "immutable");
  EXPECT_FALSE(lifetime.expired());
  retrieved.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST(SafeProjectionCacheKey, EveryOutputRelevantFieldChangesIdentity) {
  const auto base = MakeProjectionKey();
  lpp::DeterministicCache<lpp::SafeProjectionCacheKey, std::string> cache{32U};
  cache.Publish(base, std::make_shared<const std::string>("base"));

  std::vector<lpp::SafeProjectionCacheKey> mutations;
  auto add_mutation = [&mutations, &base](auto mutate) {
    auto changed = base;
    mutate(changed);
    mutations.push_back(std::move(changed));
  };

  add_mutation(
      [](auto& key) { key.map_snapshot_id = "other-map-snapshot"; });
  add_mutation([](auto& key) { key.map_revision = "other-map-revision"; });
  add_mutation(
      [](auto& key) { key.layer_manifest_hash = "other-layer-hash"; });
  add_mutation([](auto& key) {
    key.safety_capability_ref.id = "other-safety-capability";
  });
  add_mutation(
      [](auto& key) { ++key.safety_capability_ref.revision; });
  add_mutation([](auto& key) {
    key.safety_capability_ref.content_hash = "other-capability-hash";
  });
  add_mutation(
      [](auto& key) { key.algorithm_config_ref.id = "other-config"; });
  add_mutation([](auto& key) { ++key.algorithm_config_ref.revision; });
  add_mutation([](auto& key) {
    key.algorithm_config_ref.content_hash = "other-config-hash";
  });
  add_mutation([](auto& key) { key.learned_cost_snapshot_ref.reset(); });
  add_mutation([](auto& key) {
    key.learned_cost_snapshot_ref->id = "other-learned-cost";
  });
  add_mutation([](auto& key) {
    ++key.learned_cost_snapshot_ref->revision;
  });
  add_mutation([](auto& key) {
    key.learned_cost_snapshot_ref->content_hash = "other-learned-hash";
  });
  add_mutation([](auto& key) { key.frame_id = "other-frame"; });
  add_mutation([](auto& key) {
    key.resolution_ieee754_bits =
        MakeProjectionKey(0.5).resolution_ieee754_bits;
  });
  add_mutation(
      [](auto& key) { key.error_bound_model_id = "other-error-model"; });

  for (const auto& changed : mutations) {
    EXPECT_NE(changed, base);
    EXPECT_EQ(cache.Get(changed), nullptr);
  }
}

TEST(SafeProjectionCacheKey, LearnedCostUsesFullContentIdentity) {
  const auto base = MakeProjectionKey();
  auto new_revision = base;
  ++new_revision.learned_cost_snapshot_ref->revision;
  auto new_hash = base;
  new_hash.learned_cost_snapshot_ref->content_hash = "other-content";

  EXPECT_EQ(new_revision.learned_cost_snapshot_ref->id,
            base.learned_cost_snapshot_ref->id);
  EXPECT_EQ(new_hash.learned_cost_snapshot_ref->id,
            base.learned_cost_snapshot_ref->id);
  EXPECT_NE(base, new_revision);
  EXPECT_NE(base, new_hash);

  lpp::DeterministicCache<lpp::SafeProjectionCacheKey, std::string> cache{3U};
  cache.Publish(base, std::make_shared<const std::string>("base"));
  cache.Publish(
      new_revision, std::make_shared<const std::string>("new-revision"));
  cache.Publish(new_hash, std::make_shared<const std::string>("new-hash"));
  EXPECT_EQ(cache.Keys().size(), 3U);
}

TEST(SafeProjectionCacheKey, NullLearnedReferenceIsAnalyticBaselineKey) {
  const auto analytic = MakeProjectionKey(0.25, std::nullopt);
  const auto repeated = MakeProjectionKey(0.25, std::nullopt);
  const auto learned = MakeProjectionKey();

  EXPECT_EQ(analytic, repeated);
  EXPECT_NE(analytic, learned);
  EXPECT_FALSE(analytic.learned_cost_snapshot_ref.has_value());
}

TEST(SafeProjectionCacheKey, NormalizesNegativeZeroBeforeBitCast) {
  const auto positive_zero = MakeProjectionKey(0.0);
  const auto negative_zero = MakeProjectionKey(-0.0);

  EXPECT_EQ(positive_zero.resolution_ieee754_bits,
            negative_zero.resolution_ieee754_bits);
  EXPECT_EQ(positive_zero, negative_zero);
}

TEST(SafeProjectionCacheKey, RejectsEveryNonFiniteResolution) {
  EXPECT_THROW(
      static_cast<void>(
          MakeProjectionKey(std::numeric_limits<double>::quiet_NaN())),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(MakeProjectionKey(
          std::numeric_limits<double>::infinity())),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(MakeProjectionKey(
          -std::numeric_limits<double>::infinity())),
      std::invalid_argument);
}

TEST(CacheOrchestration, ValidatesBeforeLookupAndAgainAfterLookup) {
  StringCache cache{1U};
  TestOnlyOrchestrationSpy miss_spy;
  const auto built = TestOnlyValidatedLookupOrBuild(cache, 4, miss_spy);
  ASSERT_NE(built, nullptr);
  EXPECT_EQ(
      miss_spy.events,
      (std::vector<OrchestrationEvent>{
          OrchestrationEvent::kSemanticValidation,
          OrchestrationEvent::kCacheLookup,
          OrchestrationEvent::kProjectionBuild,
          OrchestrationEvent::kCachePublication,
          OrchestrationEvent::kFinalValidation,
      }));

  TestOnlyOrchestrationSpy hit_spy;
  const auto reused = TestOnlyValidatedLookupOrBuild(cache, 4, hit_spy);
  EXPECT_EQ(reused, built);
  EXPECT_EQ(
      hit_spy.events,
      (std::vector<OrchestrationEvent>{
          OrchestrationEvent::kSemanticValidation,
          OrchestrationEvent::kCacheLookup,
          OrchestrationEvent::kFinalValidation,
      }));
}

}  // namespace

#if defined(LPP_V3_STANDALONE_TEST_MAIN)
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
