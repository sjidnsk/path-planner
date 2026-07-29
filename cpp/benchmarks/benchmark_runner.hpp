#pragma once

#include <chrono>
#include <cstdint>
#include <span>
#include <string>

#include "benchmark_report.hpp"
#include "lunar_path_planner/v3/api/planner_v3.hpp"

namespace lunar::planning::v3 {

class BenchmarkClock {
 public:
  virtual ~BenchmarkClock() = default;

  [[nodiscard]] virtual std::chrono::nanoseconds Now() noexcept = 0;
};

class SteadyBenchmarkClock final : public BenchmarkClock {
 public:
  [[nodiscard]] std::chrono::nanoseconds Now() noexcept override;
};

class BenchmarkResponseEncoder {
 public:
  virtual ~BenchmarkResponseEncoder() = default;

  [[nodiscard]] virtual Result<std::string> Encode(
      const PlanningResponse& response) = 0;
};

class JsonBenchmarkResponseEncoder final
    : public BenchmarkResponseEncoder {
 public:
  [[nodiscard]] Result<std::string> Encode(
      const PlanningResponse& response) override;
};

struct BenchmarkPlatformWorkload final {
  PlatformType platform_type{PlatformType::kWheeled};
  const PlanningRequest* request{};
  PlannerV3* planner{};
};

struct BenchmarkRunResult final {
  BenchmarkReport report;
  std::uint64_t encoded_response_bytes{};
};

[[nodiscard]] Result<BenchmarkRunResult> RunPlannerApiBenchmark(
    const BenchmarkProfile& profile,
    std::span<const BenchmarkPlatformWorkload> workloads,
    const BenchmarkReportHeader& header,
    BenchmarkClock& clock,
    BenchmarkResponseEncoder& encoder) noexcept;

}  // namespace lunar::planning::v3
