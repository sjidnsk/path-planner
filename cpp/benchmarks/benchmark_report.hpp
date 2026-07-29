#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lunar_path_planner/v3/contracts/planning_response.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

enum class BenchmarkBuildType {
  kRelease,
  kRelWithDebInfo,
};

struct BenchmarkHardwareProfile final {
  std::string cpu_model;
  std::size_t physical_core_count{};
  std::size_t logical_core_count{};
  std::uint64_t memory_bytes{};
  std::string os_name;
  std::string os_version;
  std::string compiler_id;
  std::string compiler_version;
  BenchmarkBuildType build_type{BenchmarkBuildType::kRelease};
};

enum class BenchmarkTimingStart {
  kRequestSchemaAndSnapshotValidation,
};

enum class BenchmarkTimingEnd {
  kPlanningResponseSerialization,
};

enum class BenchmarkModuleName {
  kSchemaAndContractValidation,
  kSafeProjectionAndTerminalResolution,
  kPlatformSearch,
  kContinuousReferenceGeneration,
  kContinuousValidation,
  kBundlePackagingAndSerialization,
};

enum class BenchmarkExcludedActivity {
  kRawSensorMapping,
  kProcessStartup,
  kRemoteRpcOrDiskIo,
};

inline constexpr std::array<BenchmarkModuleName, 6U>
    kCanonicalBenchmarkModules{
        BenchmarkModuleName::kSchemaAndContractValidation,
        BenchmarkModuleName::kSafeProjectionAndTerminalResolution,
        BenchmarkModuleName::kPlatformSearch,
        BenchmarkModuleName::kContinuousReferenceGeneration,
        BenchmarkModuleName::kContinuousValidation,
        BenchmarkModuleName::kBundlePackagingAndSerialization,
    };

inline constexpr std::array<PlatformType, 3U>
    kCanonicalBenchmarkPlatforms{
        PlatformType::kWheeled,
        PlatformType::kLegged,
        PlatformType::kHopper,
    };

struct BenchmarkTimingBoundary final {
  BenchmarkTimingStart starts_before{
      BenchmarkTimingStart::kRequestSchemaAndSnapshotValidation};
  BenchmarkTimingEnd ends_after{
      BenchmarkTimingEnd::kPlanningResponseSerialization};
  std::array<BenchmarkModuleName, 6U> included{
      kCanonicalBenchmarkModules};
  std::array<BenchmarkExcludedActivity, 3U> excluded{
      BenchmarkExcludedActivity::kRawSensorMapping,
      BenchmarkExcludedActivity::kProcessStartup,
      BenchmarkExcludedActivity::kRemoteRpcOrDiskIo,
  };
  bool cold_start_reported_separately{true};
};

struct BenchmarkScenarioEntry final {
  ContentRef scenario_ref;
  double weight{};
};

struct BenchmarkPlatformSuite final {
  PlatformType platform_type{PlatformType::kWheeled};
  ContentRef safety_capability_ref;
  std::vector<BenchmarkScenarioEntry> scenario_mix;
};

struct BenchmarkProfileContent final {
  BenchmarkHardwareProfile hardware;
  ContentRef algorithm_config_ref;
  std::size_t planner_thread_count{};
  std::size_t warmup_sample_count{};
  std::size_t measured_sample_count_per_platform{};
  DurationNanoseconds p95_latency_target;
  std::vector<BenchmarkPlatformSuite> platform_suites;
  BenchmarkTimingBoundary timing_boundary;
  bool record_module_timings{};
  bool planner_result_must_be_independent_of_latency_target{true};
};

struct BenchmarkProfile final {
  inline static constexpr std::string_view kSchemaVersion{
      "path-planner-v3-benchmark-profile/v1"};

  ContentRef content_ref;
  BenchmarkProfileContent content;
};

struct LatencyStatistics final {
  DurationNanoseconds mean;
  DurationNanoseconds median;
  DurationNanoseconds p95;
  DurationNanoseconds p99;
  DurationNanoseconds maximum;
  std::size_t sample_count{};
};

struct ColdStartStatistics final {
  DurationNanoseconds first_call;
  std::size_t sample_count{};
};

struct BenchmarkModuleTiming final {
  BenchmarkModuleName module_name{
      BenchmarkModuleName::kSchemaAndContractValidation};
  LatencyStatistics statistics;
};

struct PlatformBenchmarkResult final {
  PlatformType platform_type{PlatformType::kWheeled};
  ContentRef safety_capability_ref;
  LatencyStatistics latency;
  DurationNanoseconds p95_latency_target;
  bool p95_latency_target_met{};
  ColdStartStatistics cold_start;
  std::vector<BenchmarkModuleTiming> module_timings;
  bool result_independence_check_passed{};
};

struct BenchmarkReportHeader final {
  Identifier report_id;
  std::uint32_t report_revision{};
  ClockStamp generated_at;
};

struct BenchmarkReport final {
  inline static constexpr std::string_view kSchemaVersion{
      "path-planner-v3-benchmark-report/v1"};

  Identifier report_id;
  std::uint32_t report_revision{};
  ClockStamp generated_at;
  ContentRef algorithm_config_ref;
  ContentRef benchmark_profile_ref;
  std::vector<PlatformBenchmarkResult> platform_results;
};

enum class BenchmarkSampleKind {
  kColdStart,
  kMeasured,
};

struct BenchmarkModuleSample final {
  BenchmarkModuleName module_name{
      BenchmarkModuleName::kSchemaAndContractValidation};
  DurationNanoseconds latency;
};

using ModuleTimingBreakdown = std::vector<BenchmarkModuleSample>;

struct BenchmarkSample final {
  PlatformType platform_type{PlatformType::kWheeled};
  ContentRef safety_capability_ref;
  DurationNanoseconds api_latency;
  ModuleTimingBreakdown modules;
  PlanningOutcome outcome{PlanningOutcome::kInvalidRequest};
  std::string termination_reason;
  BenchmarkSampleKind sample_kind{BenchmarkSampleKind::kMeasured};
  bool result_independence_check_passed{};
};

using BenchmarkSampleCollection = std::vector<BenchmarkSample>;

[[nodiscard]] Result<bool> ValidateBenchmarkProfile(
    const BenchmarkProfile& profile) noexcept;

[[nodiscard]] Result<bool> ValidateBenchmarkReportHeader(
    const BenchmarkReportHeader& header) noexcept;

[[nodiscard]] Result<BenchmarkProfile> DecodeBenchmarkProfile(
    std::string_view payload) noexcept;

[[nodiscard]] Result<std::string> EncodeBenchmarkReport(
    const BenchmarkReport& report) noexcept;

[[nodiscard]] Result<BenchmarkReport> BuildBenchmarkReport(
    const BenchmarkProfile& profile,
    std::span<const BenchmarkSample> samples,
    const BenchmarkReportHeader& header) noexcept;

}  // namespace lunar::planning::v3
