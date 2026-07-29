#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <vector>

#include "lunar_path_planner/v3/contracts/base_types.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

struct MultiscaleExperimentConfig final {
  std::size_t warmup_count{3U};
  std::size_t measured_count_per_combination{30U};
  DurationNanoseconds p95_target{std::chrono::seconds{1}};
};

struct ExperimentRunPaths final {
  std::filesystem::path manifest_json;
  std::filesystem::path latency_samples_jsonl;
  std::filesystem::path summary_json;
  std::filesystem::path responses_jsonl;
};

[[nodiscard]] Result<ExperimentRunPaths> RunMultiscaleExperiment(
    const MultiscaleExperimentConfig& config,
    const std::filesystem::path& output_root) noexcept;

namespace multiscale_experiment_internal {

struct LatencyStatistics final {
  std::size_t sample_count{};
  std::size_t correct_count{};
  double success_rate{};
  DurationNanoseconds p50;
  DurationNanoseconds p95;
  DurationNanoseconds maximum;
  bool p95_target_met{};
};

struct InvocationMeasurement final {
  DurationNanoseconds latency;
  bool outcome_correct{};
};

using Invocation =
    std::function<InvocationMeasurement(std::size_t sample_index)>;

struct ExperimentCallCounts final {
  std::size_t combination_count{};
  std::size_t cold_start_call_count{};
  std::size_t measured_call_count{};
};

[[nodiscard]] Result<LatencyStatistics> CalculateStatistics(
    const std::vector<DurationNanoseconds>& samples,
    std::size_t correct_count,
    DurationNanoseconds p95_target) noexcept;

[[nodiscard]] Result<LatencyStatistics> RunInvocations(
    std::size_t sample_count,
    DurationNanoseconds p95_target,
    const Invocation& invocation) noexcept;

[[nodiscard]] ExperimentCallCounts ExperimentCallCountsFor(
    const MultiscaleExperimentConfig& config) noexcept;

}  // namespace multiscale_experiment_internal
}  // namespace lunar::planning::v3
