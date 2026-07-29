#include "multiscale_experiment.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;
namespace experiment = multiscale_experiment_internal;

[[nodiscard]] std::filesystem::path TemporaryOutputRoot() {
  const auto unique =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::path{"D:/xunce/out/path-planner-v3-test"} /
         ("multiscale-" + std::to_string(unique));
}

[[nodiscard]] std::vector<nlohmann::json> ReadJsonLines(
    const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  std::vector<nlohmann::json> records;
  for (std::string line; std::getline(input, line);) {
    records.push_back(nlohmann::json::parse(line));
  }
  return records;
}

void RemoveRunArtifacts(const ExperimentRunPaths& paths) {
  std::filesystem::remove(paths.manifest_json);
  std::filesystem::remove(paths.latency_samples_jsonl);
  std::filesystem::remove(paths.summary_json);
  std::filesystem::remove(paths.responses_jsonl);
  std::filesystem::remove(paths.manifest_json.parent_path());
}

TEST(MultiscaleExperimentStatisticsTest,
     UsesNearestRankDirectlyOnRawSamples) {
  const std::vector<DurationNanoseconds> samples{
      DurationNanoseconds{10ns},
      DurationNanoseconds{20ns},
      DurationNanoseconds{30ns},
      DurationNanoseconds{40ns},
      DurationNanoseconds{50ns},
      DurationNanoseconds{60ns},
      DurationNanoseconds{70ns},
      DurationNanoseconds{80ns},
      DurationNanoseconds{90ns},
      DurationNanoseconds{100ns},
      DurationNanoseconds{110ns},
      DurationNanoseconds{120ns},
      DurationNanoseconds{130ns},
      DurationNanoseconds{140ns},
      DurationNanoseconds{150ns},
      DurationNanoseconds{160ns},
      DurationNanoseconds{170ns},
      DurationNanoseconds{180ns},
      DurationNanoseconds{190ns},
      DurationNanoseconds{200ns},
  };

  const auto statistics =
      experiment::CalculateStatistics(
          samples, 19U, DurationNanoseconds{195ns});

  ASSERT_TRUE(IsOk(statistics));
  const auto& value = std::get<experiment::LatencyStatistics>(statistics);
  EXPECT_EQ(value.sample_count, 20U);
  EXPECT_EQ(value.correct_count, 19U);
  EXPECT_DOUBLE_EQ(value.success_rate, 0.95);
  EXPECT_EQ(value.p50.value, 100ns);
  EXPECT_EQ(value.p95.value, 190ns);
  EXPECT_EQ(value.maximum.value, 200ns);
  EXPECT_TRUE(value.p95_target_met);
}

TEST(MultiscaleExperimentInvocationTest,
     ExecutesEverySampleBeforeEvaluatingThreshold) {
  const std::vector<DurationNanoseconds> latencies{
      DurationNanoseconds{100ms},
      DurationNanoseconds{1100ms},
      DurationNanoseconds{200ms},
  };
  std::size_t invocation_count = 0U;

  const auto statistics = experiment::RunInvocations(
      latencies.size(), DurationNanoseconds{1s},
      [&latencies, &invocation_count](const std::size_t index) {
        ++invocation_count;
        return experiment::InvocationMeasurement{
            .latency = latencies.at(index),
            .outcome_correct = true,
        };
      });

  ASSERT_TRUE(IsOk(statistics));
  EXPECT_EQ(invocation_count, latencies.size());
  const auto& value = std::get<experiment::LatencyStatistics>(statistics);
  EXPECT_EQ(value.sample_count, 3U);
  EXPECT_EQ(value.p95.value, 1100ms);
  EXPECT_FALSE(value.p95_target_met);
}

TEST(MultiscaleExperimentConfigTest,
     DefaultsProduceExactMatrixCallCounts) {
  const MultiscaleExperimentConfig config;

  const auto counts = experiment::ExperimentCallCountsFor(config);

  EXPECT_EQ(counts.combination_count, 27U);
  EXPECT_EQ(counts.cold_start_call_count, 27U);
  EXPECT_EQ(counts.measured_call_count, 810U);
}

TEST(MultiscaleExperimentArtifactTest,
     TinyRealRunWritesParseableG1ReferenceArtifacts) {
  const std::filesystem::path output_root = TemporaryOutputRoot();
  const MultiscaleExperimentConfig config{
      .warmup_count = 0U,
      .measured_count_per_combination = 1U,
      .p95_target = DurationNanoseconds{1s},
  };

  const auto run = RunMultiscaleExperiment(config, output_root);

  ASSERT_TRUE(IsOk(run))
      << (IsOk(run) ? std::string{}
                    : std::get<Error>(run).field_path + ": " +
                          std::get<Error>(run).message);
  const auto& paths = std::get<ExperimentRunPaths>(run);
  const auto manifest = nlohmann::json::parse(
      std::ifstream{paths.manifest_json, std::ios::binary});
  const auto summary = nlohmann::json::parse(
      std::ifstream{paths.summary_json, std::ios::binary});
  const auto latency_samples =
      ReadJsonLines(paths.latency_samples_jsonl);
  const auto responses = ReadJsonLines(paths.responses_jsonl);

  EXPECT_EQ(
      manifest.at("schema_version"),
      "path-planner-v3-multiscale-experiment-manifest/v1");
  EXPECT_EQ(manifest.at("matrix").size(), 27U);
  EXPECT_EQ(summary.at("measured_call_count"), 27U);
  EXPECT_EQ(summary.at("cold_start_call_count"), 27U);
  EXPECT_TRUE(summary.at("all_outcomes_correct"));
  EXPECT_EQ(summary.at("scenario_results").size(), 27U);
  EXPECT_EQ(summary.at("scale_platform_results").size(), 9U);
  EXPECT_EQ(summary.at("platform_results").size(), 3U);
  EXPECT_EQ(latency_samples.size(), 54U);
  EXPECT_EQ(responses.size(), 27U);
  const std::map<std::string, nlohmann::json> expected_references{
      {"G1_LOW_KNOWN",
       {
           {"source_scenario_id",
            "validation/scenario-0027/standard-proxy/v1"},
           {"source_scenario_hash",
            "564835143c269d1ef4fb8d01cb7517cf4efb674bb1554c36467a1d59eb39f16c"},
           {"density_profile", "low"},
           {"derivation", "g1_validation_reference_scaled/v1"},
           {"synthetic_source_kind",
            "synthetic_terrain_obstacle_proxy/v1"},
           {"physical_obstacle_cells_written", false},
       }},
      {"G1_MEDIUM_KNOWN",
       {
           {"source_scenario_id",
            "validation/scenario-0007/standard-proxy/v1"},
           {"source_scenario_hash",
            "6f754169bb3c4837978e3e852258493f4b1c971e8a59148a9a8269fb306a95bc"},
           {"density_profile", "medium"},
           {"derivation", "g1_validation_reference_scaled/v1"},
           {"synthetic_source_kind",
            "synthetic_terrain_obstacle_proxy/v1"},
           {"physical_obstacle_cells_written", false},
       }},
      {"G1_HIGH_FRONTIER",
       {
           {"source_scenario_id",
            "validation/scenario-0116/standard-proxy/v1"},
           {"source_scenario_hash",
            "49e4254dc8e5602be67cc2e9eb68af65093ec049b463e3581edfc9b5f352b554"},
           {"density_profile", "high"},
           {"derivation", "g1_validation_reference_scaled/v1"},
           {"synthetic_source_kind",
            "synthetic_terrain_obstacle_proxy/v1"},
           {"physical_obstacle_cells_written", false},
       }},
  };
  std::map<std::string, std::size_t> manifest_scene_counts;
  for (const auto& entry : manifest.at("matrix")) {
    const std::string scene = entry.at("scene").get<std::string>();
    const auto expected = expected_references.find(scene);
    EXPECT_NE(expected, expected_references.end());
    if (expected != expected_references.end()) {
      EXPECT_EQ(entry.at("g1_reference"), expected->second);
      ++manifest_scene_counts[scene];
    }
  }
  for (const auto& [scene, provenance] : expected_references) {
    static_cast<void>(provenance);
    EXPECT_EQ(manifest_scene_counts[scene], 9U);
  }
  std::size_t proxy_region_count = 0U;
  for (const auto& record : responses) {
    EXPECT_TRUE(record.contains("map_bounds"));
    EXPECT_TRUE(record.contains("reference_polyline_xy_m"));
    EXPECT_TRUE(record.contains("response_hash"));
    const std::string scene = record.at("scene").get<std::string>();
    const auto expected = expected_references.find(scene);
    EXPECT_NE(expected, expected_references.end());
    if (expected != expected_references.end()) {
      EXPECT_EQ(record.at("g1_reference"), expected->second);
    }
    for (const auto& region : record.at("regions")) {
      if (region.at("kind") ==
          "SYNTHETIC_TERRAIN_OBSTACLE_PROXY") {
        ++proxy_region_count;
        EXPECT_EQ(
            region.at("source_kind"),
            "synthetic_terrain_obstacle_proxy/v1");
        EXPECT_FALSE(
            region.at("physical_obstacle_cells_written"));
      }
    }
    if (record.at("platform") == "HOPPER") {
      EXPECT_TRUE(record.contains("ballistic_arc_xz_m"));
      EXPECT_TRUE(record.contains("landing_polygon_xy_m"));
    } else {
      EXPECT_FALSE(record.contains("ballistic_arc_xz_m"));
      EXPECT_FALSE(record.contains("landing_polygon_xy_m"));
    }
  }
  EXPECT_GT(proxy_region_count, 0U);

  RemoveRunArtifacts(paths);
}

}  // namespace
}  // namespace lunar::planning::v3
