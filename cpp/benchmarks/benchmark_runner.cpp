#include "benchmark_runner.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/codec/json_codec.hpp"

namespace lunar::planning::v3 {
namespace {

struct InvocationObservation final {
  PlanningResponse response;
  DurationNanoseconds latency;
};

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] std::size_t PlatformIndex(
    PlatformType platform_type) noexcept {
  switch (platform_type) {
    case PlatformType::kWheeled:
      return 0U;
    case PlatformType::kLegged:
      return 1U;
    case PlatformType::kHopper:
      return 2U;
  }
  return kCanonicalBenchmarkPlatforms.size();
}

[[nodiscard]] std::string_view PlatformName(
    PlatformType platform_type) noexcept {
  switch (platform_type) {
    case PlatformType::kWheeled:
      return "WHEELED";
    case PlatformType::kLegged:
      return "LEGGED";
    case PlatformType::kHopper:
      return "HOPPER";
  }
  return "UNKNOWN";
}

[[nodiscard]] const BenchmarkPlatformSuite* FindSuite(
    const BenchmarkProfile& profile,
    PlatformType platform_type) noexcept {
  for (const auto& suite : profile.content.platform_suites) {
    if (suite.platform_type == platform_type) {
      return &suite;
    }
  }
  return nullptr;
}

[[nodiscard]] Result<
    std::array<const BenchmarkPlatformWorkload*, 3U>>
ValidateWorkloads(
    const BenchmarkProfile& profile,
    std::span<const BenchmarkPlatformWorkload> workloads) {
  if (profile.content.record_module_timings) {
    return Invalid(
        "benchmark_profile.content.record_module_timings",
        "PlannerV3 exposes only complete API timing; module timing "
        "instrumentation is not declared by this runner");
  }
  if (workloads.size() != kCanonicalBenchmarkPlatforms.size()) {
    return Invalid("workloads",
                   "exactly three platform workloads are required");
  }

  std::array<const BenchmarkPlatformWorkload*, 3U> ordered{};
  for (const auto& workload : workloads) {
    const std::size_t index = PlatformIndex(workload.platform_type);
    if (index >= ordered.size() || ordered[index] != nullptr) {
      return Invalid("workloads",
                     "exactly one workload per platform is required");
    }
    ordered[index] = &workload;
  }
  for (const auto platform_type : kCanonicalBenchmarkPlatforms) {
    const std::size_t index = PlatformIndex(platform_type);
    if (ordered[index] == nullptr) {
      return Invalid("workloads",
                     "exactly one workload per platform is required");
    }
    const auto* suite = FindSuite(profile, platform_type);
    if (suite == nullptr) {
      return Invalid("benchmark_profile.content.platform_suites",
                     "profile is missing a platform suite");
    }
    if (suite->scenario_mix.size() != 1U) {
      return Invalid(
          "benchmark_profile.content.platform_suites." +
              std::string(PlatformName(platform_type)) +
              ".scenario_mix",
          "the fixed API runner requires exactly one scenario per platform");
    }

    const BenchmarkPlatformWorkload& workload = *ordered[index];
    const std::string prefix =
        "workloads." + std::string(PlatformName(platform_type));
    if (workload.request == nullptr || workload.planner == nullptr) {
      return Invalid(prefix,
                     "workload request and planner must be resolved");
    }
    const PlanningRequest& request = *workload.request;
    if (request.platform_type != platform_type) {
      return Invalid(prefix + ".request.platform_type",
                     "request platform does not match workload");
    }
    if (!request.safety_capability ||
        request.safety_capability->content_ref !=
            suite->safety_capability_ref) {
      return Invalid(prefix + ".request.safety_capability",
                     "request capability must match the benchmark suite");
    }
    if (!request.algorithm_config ||
        request.algorithm_config->content_ref !=
            profile.content.algorithm_config_ref) {
      return Invalid(prefix + ".request.algorithm_config",
                     "request algorithm config must match the profile");
    }
    if (request.algorithm_config->deterministic_execution
            .fixed_thread_count !=
        profile.content.planner_thread_count) {
      return Invalid(
          prefix +
              ".request.algorithm_config.deterministic_execution."
              "fixed_thread_count",
          "planner thread count must match the benchmark profile");
    }
  }
  return ordered;
}

[[nodiscard]] Result<InvocationObservation> Invoke(
    const BenchmarkPlatformWorkload& workload,
    bool measured,
    BenchmarkClock& clock,
    BenchmarkResponseEncoder& encoder,
    std::uint64_t& encoded_response_bytes) {
  std::chrono::nanoseconds start{};
  if (measured) {
    start = clock.Now();
  }
  PlanningResponse response = workload.planner->Plan(*workload.request);
  auto encoded = encoder.Encode(response);
  if (!IsOk(encoded)) {
    return std::get<Error>(encoded);
  }
  const std::size_t encoded_size =
      std::get<std::string>(encoded).size();
  if (encoded_size >
      std::numeric_limits<std::uint64_t>::max() -
          encoded_response_bytes) {
    return Invalid("encoded_response_bytes",
                   "encoded response byte counter overflowed");
  }
  encoded_response_bytes +=
      static_cast<std::uint64_t>(encoded_size);

  std::chrono::nanoseconds latency{};
  if (measured) {
    const std::chrono::nanoseconds end = clock.Now();
    if (end < start) {
      return Invalid("benchmark_clock",
                     "benchmark clock must be monotonic");
    }
    latency = end - start;
  }
  return InvocationObservation{
      .response = std::move(response),
      .latency = DurationNanoseconds{latency},
  };
}

[[nodiscard]] BenchmarkSample MakeSample(
    PlatformType platform_type,
    const ContentRef& capability_ref,
    InvocationObservation observation,
    BenchmarkSampleKind sample_kind) {
  std::string termination_reason =
      observation.response.call_diagnostics.termination_reason;
  if (termination_reason.empty()) {
    termination_reason = observation.response.reason_code.empty()
                             ? "NATURAL_COMPLETION"
                             : observation.response.reason_code;
  }
  return BenchmarkSample{
      .platform_type = platform_type,
      .safety_capability_ref = capability_ref,
      .api_latency = observation.latency,
      .modules = {},
      .outcome = observation.response.planning_outcome,
      .termination_reason = std::move(termination_reason),
      .sample_kind = sample_kind,
      .result_independence_check_passed = true,
  };
}

}  // namespace

std::chrono::nanoseconds SteadyBenchmarkClock::Now() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch());
}

Result<std::string> JsonBenchmarkResponseEncoder::Encode(
    const PlanningResponse& response) {
  try {
    return JsonCodec::EncodePlanningResponse(response);
  } catch (const std::exception& error) {
    return Error{
        .code = ErrorCode::kNumericalFailure,
        .field_path = "planning_response",
        .message = std::string{"response encoding failed: "} +
                   error.what(),
    };
  } catch (...) {
    return Error{
        .code = ErrorCode::kNumericalFailure,
        .field_path = "planning_response",
        .message = "response encoding failed",
    };
  }
}

Result<BenchmarkRunResult> RunPlannerApiBenchmark(
    const BenchmarkProfile& profile,
    std::span<const BenchmarkPlatformWorkload> workloads,
    const BenchmarkReportHeader& header,
    BenchmarkClock& clock,
    BenchmarkResponseEncoder& encoder) noexcept {
  try {
    const auto profile_validation =
        ValidateBenchmarkProfile(profile);
    if (!IsOk(profile_validation)) {
      return std::get<Error>(profile_validation);
    }
    const auto header_validation =
        ValidateBenchmarkReportHeader(header);
    if (!IsOk(header_validation)) {
      return std::get<Error>(header_validation);
    }
    const auto workload_validation =
        ValidateWorkloads(profile, workloads);
    if (!IsOk(workload_validation)) {
      return std::get<Error>(workload_validation);
    }
    const auto& ordered = std::get<
        std::array<const BenchmarkPlatformWorkload*, 3U>>(
        workload_validation);

    BenchmarkSampleCollection samples;
    const std::size_t sample_count_per_platform =
        profile.content.measured_sample_count_per_platform + 1U;
    if (sample_count_per_platform <
        profile.content.measured_sample_count_per_platform ||
        sample_count_per_platform >
            std::numeric_limits<std::size_t>::max() /
                kCanonicalBenchmarkPlatforms.size()) {
      return Invalid("benchmark_profile.content."
                         "measured_sample_count_per_platform",
                     "sample count overflows runner capacity");
    }
    samples.reserve(
        sample_count_per_platform *
        kCanonicalBenchmarkPlatforms.size());
    std::uint64_t encoded_response_bytes = 0U;

    for (const auto platform_type : kCanonicalBenchmarkPlatforms) {
      const std::size_t index = PlatformIndex(platform_type);
      const auto& workload = *ordered[index];
      const auto* suite = FindSuite(profile, platform_type);

      auto cold = Invoke(
          workload, true, clock, encoder, encoded_response_bytes);
      if (!IsOk(cold)) {
        return std::get<Error>(cold);
      }
      samples.push_back(MakeSample(
          platform_type, suite->safety_capability_ref,
          std::get<InvocationObservation>(std::move(cold)),
          BenchmarkSampleKind::kColdStart));

      for (std::size_t warmup = 0U;
           warmup < profile.content.warmup_sample_count;
           ++warmup) {
        auto observation = Invoke(
            workload, false, clock, encoder,
            encoded_response_bytes);
        if (!IsOk(observation)) {
          return std::get<Error>(observation);
        }
      }

      for (std::size_t sample_index = 0U;
           sample_index <
           profile.content.measured_sample_count_per_platform;
           ++sample_index) {
        auto observation = Invoke(
            workload, true, clock, encoder,
            encoded_response_bytes);
        if (!IsOk(observation)) {
          return std::get<Error>(observation);
        }
        samples.push_back(MakeSample(
            platform_type, suite->safety_capability_ref,
            std::get<InvocationObservation>(
                std::move(observation)),
            BenchmarkSampleKind::kMeasured));
      }
    }

    auto report =
        BuildBenchmarkReport(profile, samples, header);
    if (!IsOk(report)) {
      return std::get<Error>(report);
    }
    return BenchmarkRunResult{
        .report =
            std::get<BenchmarkReport>(std::move(report)),
        .encoded_response_bytes = encoded_response_bytes,
    };
  } catch (const std::exception& error) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_runner",
        .message = std::string{"benchmark run failed: "} +
                   error.what(),
    };
  } catch (...) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path = "benchmark_runner",
        .message = "benchmark run failed",
    };
  }
}

}  // namespace lunar::planning::v3
