#include "benchmark_report.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] ContentRef MakeRef(std::string id, char digest_digit) {
  return ContentRef{
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digest_digit),
  };
}

[[nodiscard]] BenchmarkTimingBoundary MakeTimingBoundary() {
  return BenchmarkTimingBoundary{
      .starts_before =
          BenchmarkTimingStart::kRequestSchemaAndSnapshotValidation,
      .ends_after = BenchmarkTimingEnd::kPlanningResponseSerialization,
      .included = {
          BenchmarkModuleName::kSchemaAndContractValidation,
          BenchmarkModuleName::kSafeProjectionAndTerminalResolution,
          BenchmarkModuleName::kPlatformSearch,
          BenchmarkModuleName::kContinuousReferenceGeneration,
          BenchmarkModuleName::kContinuousValidation,
          BenchmarkModuleName::kBundlePackagingAndSerialization,
      },
      .excluded = {
          BenchmarkExcludedActivity::kRawSensorMapping,
          BenchmarkExcludedActivity::kProcessStartup,
          BenchmarkExcludedActivity::kRemoteRpcOrDiskIo,
      },
      .cold_start_reported_separately = true,
  };
}

[[nodiscard]] BenchmarkProfile MakeProfile(
    std::size_t measured_sample_count = 20U,
    bool record_module_timings = false) {
  BenchmarkProfile profile{
      .content_ref = MakeRef("benchmark-profile.v3", 'a'),
      .content =
          BenchmarkProfileContent{
              .hardware =
                  BenchmarkHardwareProfile{
                      .cpu_model = "declared-test-cpu",
                      .physical_core_count = 8U,
                      .logical_core_count = 16U,
                      .memory_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL,
                      .os_name = "Windows",
                      .os_version = "11",
                      .compiler_id = "MSVC",
                      .compiler_version = "19.44",
                      .build_type = BenchmarkBuildType::kRelease,
                  },
              .algorithm_config_ref = MakeRef("algorithm-config.v3", 'b'),
              .planner_thread_count = 4U,
              .warmup_sample_count = 10U,
              .measured_sample_count_per_platform = measured_sample_count,
              .p95_latency_target = DurationNanoseconds{1s},
              .platform_suites = {},
              .timing_boundary = MakeTimingBoundary(),
              .record_module_timings = record_module_timings,
              .planner_result_must_be_independent_of_latency_target = true,
          },
  };
  profile.content.platform_suites = {
      BenchmarkPlatformSuite{
          .platform_type = PlatformType::kWheeled,
          .safety_capability_ref = MakeRef("wheel-capability.v3", 'c'),
          .scenario_mix =
              {BenchmarkScenarioEntry{
                  .scenario_ref = MakeRef("wheel-scenario.v3", 'd'),
                  .weight = 1.0,
              }},
      },
      BenchmarkPlatformSuite{
          .platform_type = PlatformType::kLegged,
          .safety_capability_ref = MakeRef("legged-capability.v3", 'e'),
          .scenario_mix =
              {BenchmarkScenarioEntry{
                  .scenario_ref = MakeRef("legged-scenario.v3", 'f'),
                  .weight = 1.0,
              }},
      },
      BenchmarkPlatformSuite{
          .platform_type = PlatformType::kHopper,
          .safety_capability_ref = MakeRef("hopper-capability.v3", '1'),
          .scenario_mix =
              {BenchmarkScenarioEntry{
                  .scenario_ref = MakeRef("hopper-scenario.v3", '2'),
                  .weight = 1.0,
              }},
      },
  };
  return profile;
}

[[nodiscard]] const ContentRef& CapabilityFor(
    const BenchmarkProfile& profile, PlatformType platform_type) {
  for (const auto& suite : profile.content.platform_suites) {
    if (suite.platform_type == platform_type) {
      return suite.safety_capability_ref;
    }
  }
  throw std::logic_error("test profile is missing a platform suite");
}

[[nodiscard]] std::vector<BenchmarkModuleSample> MakeModuleSamples(
    DurationNanoseconds duration) {
  return {
      {BenchmarkModuleName::kSchemaAndContractValidation, duration},
      {BenchmarkModuleName::kSafeProjectionAndTerminalResolution, duration},
      {BenchmarkModuleName::kPlatformSearch, duration},
      {BenchmarkModuleName::kContinuousReferenceGeneration, duration},
      {BenchmarkModuleName::kContinuousValidation, duration},
      {BenchmarkModuleName::kBundlePackagingAndSerialization, duration},
  };
}

void AppendPlatformSamples(BenchmarkSampleCollection& samples,
                           const BenchmarkProfile& profile,
                           PlatformType platform_type,
                           const std::vector<std::chrono::nanoseconds>& values,
                           bool with_modules = false) {
  const ContentRef capability = CapabilityFor(profile, platform_type);
  samples.push_back(BenchmarkSample{
      .platform_type = platform_type,
      .safety_capability_ref = capability,
      .api_latency = DurationNanoseconds{321ns},
      .modules = {},
      .outcome = PlanningOutcome::kNewReferenceReady,
      .termination_reason = "COLD_START_COMPLETED",
      .sample_kind = BenchmarkSampleKind::kColdStart,
      .result_independence_check_passed = true,
  });
  for (const auto value : values) {
    const auto module_value =
        DurationNanoseconds{value / static_cast<std::int64_t>(6)};
    samples.push_back(BenchmarkSample{
        .platform_type = platform_type,
        .safety_capability_ref = capability,
        .api_latency = DurationNanoseconds{value},
        .modules =
            with_modules ? MakeModuleSamples(module_value)
                         : std::vector<BenchmarkModuleSample>{},
        .outcome = PlanningOutcome::kNewReferenceReady,
        .termination_reason = "TARGET_EPSILON_REACHED",
        .sample_kind = BenchmarkSampleKind::kMeasured,
        .result_independence_check_passed = true,
    });
  }
}

[[nodiscard]] BenchmarkSampleCollection MakeThreePlatformSamples(
    const BenchmarkProfile& profile,
    const std::vector<std::chrono::nanoseconds>& values,
    bool with_modules = false) {
  BenchmarkSampleCollection samples;
  for (const auto platform_type :
       {PlatformType::kWheeled, PlatformType::kLegged,
        PlatformType::kHopper}) {
    AppendPlatformSamples(samples, profile, platform_type, values,
                          with_modules);
  }
  return samples;
}

[[nodiscard]] BenchmarkReportHeader MakeHeader() {
  return BenchmarkReportHeader{
      .report_id = "benchmark-report.v3",
      .report_revision = 1U,
      .generated_at =
          ClockStamp{
              .clock_id = "benchmark-clock",
              .tick = 42ns,
          },
  };
}

[[nodiscard]] const PlatformBenchmarkResult& FindPlatformResult(
    const BenchmarkReport& report, PlatformType platform_type) {
  for (const auto& result : report.platform_results) {
    if (result.platform_type == platform_type) {
      return result;
    }
  }
  throw std::logic_error("report is missing a platform result");
}

[[nodiscard]] std::string ReadDeclaredProfileFixture() {
  std::filesystem::path source{__FILE__};
  if (source.is_relative()) {
    source = std::filesystem::absolute(source);
  }
  const auto path =
      source.parent_path().parent_path().parent_path().parent_path() /
      "benchmarks" / "fixtures" /
      "declared_benchmark_profile.json";
  std::ifstream input{path, std::ios::binary};
  if (!input.is_open()) {
    throw std::runtime_error{
        "cannot open benchmark profile fixture"};
  }
  return {
      std::istreambuf_iterator<char>{input},
      std::istreambuf_iterator<char>{},
  };
}

TEST(BenchmarkReportTest,
     DecodesDeclaredProfileAndVerifiesItsJcsContentHash) {
  const auto result =
      DecodeBenchmarkProfile(ReadDeclaredProfileFixture());

  ASSERT_TRUE(IsOk(result))
      << std::get<Error>(result).field_path << ": "
      << std::get<Error>(result).message;
  const auto& profile = std::get<BenchmarkProfile>(result);
  EXPECT_EQ(profile.content.platform_suites.size(), 3U);
  EXPECT_EQ(profile.content.p95_latency_target.value, 1s);
  EXPECT_EQ(
      profile.content.measured_sample_count_per_platform, 1000U);
}

TEST(BenchmarkReportTest,
     ProfileDecoderRejectsUnknownFieldsAndHashMismatch) {
  auto document =
      nlohmann::json::parse(ReadDeclaredProfileFixture());
  document["unexpected"] = true;

  auto result = DecodeBenchmarkProfile(document.dump());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "benchmark_profile");

  document =
      nlohmann::json::parse(ReadDeclaredProfileFixture());
  document["content"]["warmup_sample_count"] = 11;
  result = DecodeBenchmarkProfile(document.dump());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(
      std::get<Error>(result).field_path,
      "benchmark_profile.content_ref.content_hash");
}

TEST(BenchmarkReportTest, UsesNearestRankPercentiles) {
  const auto profile = MakeProfile();
  std::vector<std::chrono::nanoseconds> latencies;
  for (std::int64_t milliseconds = 1; milliseconds <= 20; ++milliseconds) {
    latencies.push_back(std::chrono::milliseconds{milliseconds});
  }
  const auto samples = MakeThreePlatformSamples(profile, latencies);

  const auto result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_TRUE(IsOk(result));
  const auto& report = std::get<BenchmarkReport>(result);
  const auto& wheeled =
      FindPlatformResult(report, PlatformType::kWheeled);
  EXPECT_EQ(wheeled.latency.median.value, 10ms);
  EXPECT_EQ(wheeled.latency.p95.value, 19ms);
  EXPECT_EQ(wheeled.latency.p99.value, 20ms);
  EXPECT_EQ(wheeled.latency.maximum.value, 20ms);
  EXPECT_EQ(wheeled.latency.sample_count, 20U);
}

TEST(BenchmarkReportTest, RoundsMeanNanosecondsToNearestTiesToEven) {
  auto profile = MakeProfile(2U);
  auto samples =
      MakeThreePlatformSamples(profile, {1ns, 2ns});

  auto result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_TRUE(IsOk(result));
  EXPECT_EQ(FindPlatformResult(std::get<BenchmarkReport>(result),
                               PlatformType::kWheeled)
                .latency.mean.value,
            2ns);

  samples = MakeThreePlatformSamples(profile, {2ns, 3ns});
  result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_TRUE(IsOk(result));
  EXPECT_EQ(FindPlatformResult(std::get<BenchmarkReport>(result),
                               PlatformType::kWheeled)
                .latency.mean.value,
            2ns);
}

TEST(BenchmarkReportTest, CopiesAllStrongReferencesAndExplicitHeader) {
  const auto profile = MakeProfile(1U);
  const auto samples = MakeThreePlatformSamples(profile, {10ms});
  const auto header = MakeHeader();

  const auto result = BuildBenchmarkReport(profile, samples, header);

  ASSERT_TRUE(IsOk(result));
  const auto& report = std::get<BenchmarkReport>(result);
  EXPECT_EQ(report.report_id, header.report_id);
  EXPECT_EQ(report.report_revision, header.report_revision);
  EXPECT_EQ(report.generated_at.clock_id, header.generated_at.clock_id);
  EXPECT_EQ(report.generated_at.tick, header.generated_at.tick);
  EXPECT_EQ(report.algorithm_config_ref,
            profile.content.algorithm_config_ref);
  EXPECT_EQ(report.benchmark_profile_ref, profile.content_ref);
  ASSERT_EQ(report.platform_results.size(), 3U);
  for (const auto& platform_result : report.platform_results) {
    EXPECT_EQ(platform_result.safety_capability_ref,
              CapabilityFor(profile, platform_result.platform_type));
    EXPECT_EQ(platform_result.p95_latency_target.value, 1s);
  }
}

TEST(BenchmarkReportTest, AppliesOneSecondAsStrictPostRunMetricOnly) {
  auto profile = MakeProfile(1U);
  auto samples =
      MakeThreePlatformSamples(profile, {999'999'999ns});

  auto result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_TRUE(IsOk(result));
  EXPECT_TRUE(FindPlatformResult(std::get<BenchmarkReport>(result),
                                 PlatformType::kWheeled)
                  .p95_latency_target_met);

  samples = MakeThreePlatformSamples(profile, {1s});
  result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_TRUE(IsOk(result));
  EXPECT_FALSE(FindPlatformResult(std::get<BenchmarkReport>(result),
                                  PlatformType::kWheeled)
                   .p95_latency_target_met);
}

TEST(BenchmarkReportTest, RejectsDuplicateOrMissingPlatformSuite) {
  auto profile = MakeProfile(1U);
  profile.content.platform_suites[2].platform_type =
      PlatformType::kLegged;
  const auto samples =
      MakeThreePlatformSamples(MakeProfile(1U), {10ms});

  const auto result =
      BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "benchmark_profile.content.platform_suites");
}

TEST(BenchmarkReportTest, RejectsSampleCapabilityReferenceMismatch) {
  const auto profile = MakeProfile(1U);
  auto samples = MakeThreePlatformSamples(profile, {10ms});
  samples.front().safety_capability_ref.content_hash =
      std::string(64U, '9');

  const auto result =
      BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "samples[0].safety_capability_ref");
}

TEST(BenchmarkReportTest, RejectsWrongMeasuredOrColdStartCardinality) {
  const auto profile = MakeProfile(2U);
  auto samples =
      MakeThreePlatformSamples(profile, {1ms, 2ms});
  samples.erase(samples.begin());

  auto result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "samples.WHEELED.cold_start");

  samples = MakeThreePlatformSamples(profile, {1ms, 2ms});
  for (auto iterator = samples.begin(); iterator != samples.end();
       ++iterator) {
    if (iterator->platform_type == PlatformType::kLegged &&
        iterator->sample_kind == BenchmarkSampleKind::kMeasured) {
      samples.erase(iterator);
      break;
    }
  }
  result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "samples.LEGGED.measured");
}

TEST(BenchmarkReportTest, AggregatesCompleteModuleTimingsWhenDeclared) {
  const auto profile = MakeProfile(2U, true);
  const auto samples =
      MakeThreePlatformSamples(profile, {60ns, 120ns}, true);

  const auto result =
      BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_TRUE(IsOk(result));
  const auto& wheeled =
      FindPlatformResult(std::get<BenchmarkReport>(result),
                         PlatformType::kWheeled);
  ASSERT_EQ(wheeled.module_timings.size(), 6U);
  EXPECT_EQ(wheeled.module_timings.front().statistics.mean.value, 15ns);
  EXPECT_EQ(wheeled.module_timings.front().statistics.p95.value, 20ns);
  EXPECT_EQ(wheeled.module_timings.front().statistics.sample_count, 2U);
}

TEST(BenchmarkReportTest, RejectsIncompleteOrUnexpectedModuleTimings) {
  auto profile = MakeProfile(1U, true);
  auto samples = MakeThreePlatformSamples(profile, {60ns}, true);
  samples[1].modules.pop_back();

  auto result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path, "samples[1].modules");

  profile = MakeProfile(1U, false);
  samples = MakeThreePlatformSamples(profile, {60ns}, false);
  samples[1].modules = MakeModuleSamples(DurationNanoseconds{1ns});
  result = BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path, "samples[1].modules");
}

TEST(BenchmarkReportTest, ReportsFailedResultIndependenceCheck) {
  const auto profile = MakeProfile(1U);
  auto samples = MakeThreePlatformSamples(profile, {10ms});
  for (auto& sample : samples) {
    if (sample.platform_type == PlatformType::kHopper &&
        sample.sample_kind == BenchmarkSampleKind::kMeasured) {
      sample.result_independence_check_passed = false;
    }
  }

  const auto result =
      BuildBenchmarkReport(profile, samples, MakeHeader());

  ASSERT_TRUE(IsOk(result));
  EXPECT_FALSE(FindPlatformResult(std::get<BenchmarkReport>(result),
                                  PlatformType::kHopper)
                   .result_independence_check_passed);
  EXPECT_TRUE(FindPlatformResult(std::get<BenchmarkReport>(result),
                                 PlatformType::kWheeled)
                  .result_independence_check_passed);
}

TEST(BenchmarkReportTest, EncodesSchemaConformantCanonicalReportFields) {
  const auto profile = MakeProfile(1U);
  const auto samples = MakeThreePlatformSamples(profile, {10ms});
  const auto built =
      BuildBenchmarkReport(profile, samples, MakeHeader());
  ASSERT_TRUE(IsOk(built));

  const auto encoded = EncodeBenchmarkReport(
      std::get<BenchmarkReport>(built));

  ASSERT_TRUE(IsOk(encoded));
  const auto document =
      nlohmann::json::parse(std::get<std::string>(encoded));
  EXPECT_EQ(
      document.at("schema_version").get<std::string>(),
      std::string{BenchmarkReport::kSchemaVersion});
  ASSERT_EQ(document.at("platform_results").size(), 3U);
  EXPECT_EQ(
      document.at("platform_results")[0]
          .at("p95_latency_target_ns"),
      "1000000000");
  EXPECT_EQ(
      document.at("benchmark_profile_ref").at("content_hash"),
      profile.content_ref.content_hash);
}

}  // namespace
}  // namespace lunar::planning::v3

#if defined(LPP_V3_BENCHMARK_REPORT_STANDALONE_TEST_MAIN)
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
